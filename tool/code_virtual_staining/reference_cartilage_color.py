"""参考图驱动的软骨颜色校正，适用于本例沿纵向延伸的带状软骨掩膜。

用参考图的“背景侧→骨侧”相对深度建立平滑 Lab 中位数曲线，校正目标图的
相应颜色曲线。不复制参考图纹理，也不对目标图进行几何变形或模糊。
接缝宽度来自参考图边缘的 10%–90% 颜色变化宽度；窄接缝跨越标注边界，
避免在组织内强行退回白色形成光晕。骨侧可在限宽范围内补足标注与深染骨之间
的小间隙。远离边界的掩膜外区域及相邻深染骨像素保留原值。
"""

import cv2
import numpy as np
from scipy import ndimage


def _bounds(mask):
    valid = mask.any(axis=1)
    left = np.argmax(mask, axis=1)
    right = mask.shape[1]-1-np.argmax(mask[:, ::-1], axis=1)
    return left, right, valid


def _depth(mask):
    left, right, valid = _bounds(mask)
    x = np.arange(mask.shape[1])[None, :]
    return np.clip((x-left[:, None])/np.maximum(1, right-left)[:, None], 0, 1).astype(np.float32), valid


def _profile(lab, mask, bins=48):
    depth, valid = _depth(mask)
    rows = np.flatnonzero(valid)
    selected_rows = ((np.arange(mask.shape[0]) >= np.quantile(rows, .08))
                     & (np.arange(mask.shape[0]) <= np.quantile(rows, .92)))
    selected = mask & selected_rows[:, None]
    index = np.minimum((depth*(bins-1)).astype(np.int32), bins-1)
    profile = np.full((bins, 3), np.nan, np.float32)
    for k in range(bins):
        pixels = lab[selected & (index == k)]
        if len(pixels):
            profile[k] = np.median(pixels, axis=0)
    for c in range(3):
        finite = np.isfinite(profile[:, c])
        if not finite.any():
            raise ValueError("软骨掩膜过窄，无法估计颜色曲线")
        profile[:, c] = np.interp(np.arange(bins), np.flatnonzero(finite), profile[finite, c])
    return ndimage.gaussian_filter1d(profile, 1.4, axis=0, mode="nearest")


def _outside_color(lab, mask, side):
    left, right, valid = _bounds(mask)
    rows = np.flatnonzero(valid)
    rows = rows[(rows >= np.quantile(rows, .12)) & (rows <= np.quantile(rows, .88))]
    offsets = np.arange(20, 51)
    positions = (left[rows, None]-offsets if side == "left" else right[rows, None]+offsets)
    in_image = (positions >= 0) & (positions < lab.shape[1])
    values = lab[rows[:, None], positions.clip(0, lab.shape[1]-1)]
    return np.median(values[in_image], axis=0) if in_image.any() else None


def _orient(lab, mask):
    left, right = _outside_color(lab, mask, "left"), _outside_color(lab, mask, "right")
    if left is None or right is None:
        raise ValueError("掩膜左右均须有至少 20 像素的上下文，以判断背景侧和骨侧")
    flip = bool(right[0] > left[0])
    return (lab[:, ::-1], mask[:, ::-1], flip) if flip else (lab, mask, flip)


def _edge_support(lab, mask, side):
    """Measure 10%–90% transition, then convert to smoothstep support width."""
    left, right, valid = _bounds(mask)
    rows = np.flatnonzero(valid)
    rows = rows[(rows >= np.quantile(rows, .12)) & (rows <= np.quantile(rows, .88))]
    offsets = np.arange(-40, 33)
    positions = (left[rows, None]+offsets if side == "outer" else right[rows, None]-offsets)
    values = lab[rows[:, None], positions.clip(0, lab.shape[1]-1)]
    curve = np.median(values, axis=0)
    background = np.median(curve[offsets <= -26], axis=0)
    tissue = np.median(curve[(offsets >= 4) & (offsets <= 20)], axis=0)
    # Outer boundary is primarily a chromatic step; the bone interface includes luminance.
    channels = slice(1, 3) if side == "outer" else slice(0, 3)
    direction = tissue[channels]-background[channels]
    denominator = float(direction@direction)
    if denominator < 2:
        return 12.0, {"measured": False, "support_px": 12.0}
    level = ((curve[:, channels]-background[channels])@direction)/denominator
    level = ndimage.gaussian_filter1d(level, 1.0)
    level = np.maximum.accumulate(np.clip(level, 0, 1))

    def crossing(target):
        k = int(np.searchsorted(level, target))
        if k <= 0 or k >= len(level):
            return None
        fraction = (target-level[k-1])/max(level[k]-level[k-1], 1e-6)
        return float(offsets[k-1]+fraction*(offsets[k]-offsets[k-1]))

    low, high = crossing(.1), crossing(.9)
    if low is None or high is None or high-low < 1:
        return 12.0, {"measured": False, "support_px": 12.0}
    # Smoothstep crosses .1/.9 at t=.1958/.8042.
    support = float(np.clip((high-low)/.6084, 4, 24))
    return support, {"measured": True, "p10_position_px": low, "p90_position_px": high,
                     "p10_p90_width_px": high-low, "support_px": support}


def reference_profile_normalize_bgr(reference_bgr, target_bgr, reference_mask, target_mask,
                                     max_edge_shift=24, strength=1.0):
    """Return (corrected BGR, blend weights, report) for vertical cartilage bands.

    Inputs may be mask bounding boxes plus >=64 px context. All dimensions and
    pixel positions are retained. Mask-exterior changes are confined to a narrow
    support band; this deliberately differs from strict inside-only feathering.
    """
    if not 0 <= max_edge_shift <= 32 or int(max_edge_shift) != max_edge_shift:
        raise ValueError("max_edge_shift 须为 0–32 的整数")
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength 须在 0–1 之间")
    masks = []
    for image, mask in [(reference_bgr, reference_mask), (target_bgr, target_mask)]:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("图像须为 uint8 BGR 三通道")
        mask = np.asarray(mask) != 0
        if mask.shape != image.shape[:2] or np.count_nonzero(mask) < 32:
            raise ValueError("掩膜须非空并与各自图像尺寸一致")
        masks.append(mask)
    rm, tm = masks
    reference_lab = cv2.cvtColor(reference_bgr.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_bgr.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
    reference_lab, rm, ref_flip = _orient(reference_lab, rm)
    target_lab, tm, tgt_flip = _orient(target_lab, tm)
    original = target_bgr[:, ::-1] if tgt_flip else target_bgr
    ref_profile, tgt_profile = _profile(reference_lab, rm), _profile(target_lab, tm)
    outer_width, outer_report = _edge_support(reference_lab, rm, "outer")
    inner_width, inner_report = _edge_support(reference_lab, rm, "inner")
    left, right, valid = _bounds(tm)
    chroma = np.hypot(target_lab[..., 1], target_lab[..., 2])
    cartilage_l = float(np.median(tgt_profile[-8:, 0]))
    bone_l_cutoff = max(0., cartilage_l-14.)
    bone_chroma_cutoff = 16.0
    bone_pixels = (target_lab[..., 0] < bone_l_cutoff) & (chroma > bone_chroma_cutoff)
    shift = np.zeros(tm.shape[0], np.int32)
    if max_edge_shift >= 3:
        positions = np.clip(right[:, None]+np.arange(1, max_edge_shift+1), 0, tm.shape[1]-1)
        bone = bone_pixels[np.arange(tm.shape[0])[:, None], positions]
        consecutive = bone[:, :-2] & bone[:, 1:-1] & bone[:, 2:]
        shift = np.where(consecutive.any(axis=1), consecutive.argmax(axis=1), 0)
        shift = ndimage.median_filter(shift, size=31)
        shift[~valid] = 0
    extended = tm.copy()
    for y in np.flatnonzero(valid):
        extended[y, right[y]+1:right[y]+1+shift[y]] = True
    depth, _ = _depth(extended)
    positions = np.linspace(0, 1, len(ref_profile))
    correction = np.stack([np.interp(depth, positions, ref_profile[:, c]-tgt_profile[:, c])
                           for c in range(3)], axis=-1).astype(np.float32)
    # Unit gain preserves target contrast; only the slowly varying Lab offset is transferred.
    normalized = cv2.cvtColor(target_lab+correction, cv2.COLOR_LAB2BGR)*255
    signed_distance = ndimage.distance_transform_edt(extended)-ndimage.distance_transform_edt(~extended)
    support = np.where(depth < .5, outer_width, inner_width)
    t = np.clip((signed_distance+support/2)/support, 0, 1)
    blend = (t*t*(3-2*t)).astype(np.float32)
    blend[bone_pixels & ~tm] = 0
    # Explicit maximum support guard makes the exterior-edit limit reviewable.
    distance_original = ndimage.distance_transform_edt(~tm)
    max_band = int(np.ceil(max_edge_shift+max(outer_width, inner_width)/2))+1
    blend[distance_original > max_band] = 0
    blend *= strength
    output = original.copy()
    selected = blend > 0
    a = blend[selected, None]
    output[selected] = np.clip(np.rint(original[selected]*(1-a)+normalized[selected]*a), 0, 255).astype(np.uint8)
    output_lab = cv2.cvtColor(output.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
    # Compare lateral median color profiles, rather than unrelated cell positions.
    before_profile_error = np.linalg.norm(tgt_profile-ref_profile, axis=1)
    after_profile = _profile(output_lab, extended)
    after_profile_error = np.linalg.norm(after_profile-ref_profile, axis=1)
    changed = np.any(output != original, axis=2)
    report = {
        "method": "reference_cartilage_depth_color_profile",
        "reference_outer_side": "right" if ref_flip else "left",
        "target_outer_side": "right" if tgt_flip else "left",
        "reference_edge_transition_outer": outer_report, "reference_edge_transition_inner": inner_report,
        "reference_lab_profile": ref_profile.tolist(), "target_before_lab_profile": tgt_profile.tolist(),
        "target_after_lab_profile": after_profile.tolist(),
        "mean_profile_delta_e_before": float(before_profile_error.mean()),
        "mean_profile_delta_e_after": float(after_profile_error.mean()),
        "outer_profile_delta_e_before": float(before_profile_error[:5].mean()),
        "outer_profile_delta_e_after": float(after_profile_error[:5].mean()),
        "inner_profile_delta_e_before": float(before_profile_error[-5:].mean()),
        "inner_profile_delta_e_after": float(after_profile_error[-5:].mean()),
        "contrast_gain": 1.0, "strength": float(strength), "max_edge_shift_px": int(max_edge_shift),
        "inner_edge_shift_quantiles_px": np.quantile(shift[valid], [0, .5, .9, 1]).tolist(),
        "maximum_allowed_outside_band_px": max_band,
        "changed_pixels": int(changed.sum()), "outside_mask_changed_pixels": int((changed & ~tm).sum()),
        "distant_background_changed_pixels": int((changed & (distance_original > max_band)).sum()),
        "neighboring_bone_changed_pixels": int((changed & bone_pixels & ~tm).sum()),
    }
    if tgt_flip:
        output, blend = output[:, ::-1].copy(), blend[:, ::-1].copy()
    return output, blend, report
