"""tool/auto_freq_filter.py

Automatic periodic stripe removal via frequency-domain notch filtering.

What it does
------------
- Automatically estimates stripe direction from the Fourier magnitude spectrum.
- Detects periodic peak pairs caused by stripes.
- Builds a smooth notch mask around those peaks.
- Filters the image in frequency domain and reconstructs a stripe-reduced result.

Script style
------------
Follows this repo's tool scripts (e.g., tool/img_invert.py): edit paths below and run.

Typical use
-----------
1) Edit INPUT_DIR / OUTPUT_DIR
2) python tool/auto_freq_filter.py

Notes
-----
- Works best for *periodic* stripes (approximately sinusoidal or repeating pattern).
- If stripes vary spatially or are non-periodic, results may be limited.

Dependencies
-----------
- numpy
- opencv-python (cv2)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# =========================
# 配置：写死路径（img_invert 风格）
# =========================
INPUT_DIR = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_BF\patches"
OUTPUT_DIR = r"C:\Users\30927\Desktop\mouse_kidney_20x_freq"

RECURSIVE = True
OVERWRITE = True

# 滤波选项
FILTER_ON_LUMA = True   # True: 仅过滤亮度(Y), 保留颜色；False: 直接滤灰度/单通道

# 自动方向检测参数
DC_EXCLUDE_RADIUS = 20      # 排除中心 DC 半径（像素）
RING_R_MIN = 40             # 只用某个半径区间的频率做方向统计（避开低频 & 高频噪声）
RING_R_MAX_RATIO = 0.50     # RING_R_MAX = min(H,W)*ratio
N_ANGLE_BINS = 180

# 峰值检测参数
PEAK_MIN_R = 20             # 峰值搜索最低半径（像素）
PEAK_MAX_R_RATIO = 1
PEAK_REL_THRESH = 0.40      # 阈值（相对最大值）用于候选峰
PEAK_TOPK = 20               # 每侧最多取多少个峰（会对称配对）

# notch mask 参数
NOTCH_SIGMA_R = 5.0         # notch 半径方向平滑尺度（越大抑制范围越宽）
NOTCH_SIGMA_T = 3.0         # 法向方向平滑尺度
NOTCH_STRENGTH = 1.0        # 1=完全抑制 notch 中心；<1=软抑制

# Debug 输出
SAVE_DEBUG = True
DEBUG_SUFFIX = "_dbg"       # 每张图输出一个子目录保存 debug

# 输出格式：尽量保持同名同后缀
JPEG_QUALITY = 95


@dataclass
class StripeModel:
    stripe_angle_deg: float  # 条纹方向（角度，0=水平向右, 90=竖直向下）
    normal_angle_deg: float  # 频域峰值方向（条纹法向）
    peaks_uv: List[Tuple[int, int]]  # 频域中心化坐标 (u,v) in FFT-shift domain


def iter_image_paths(root: Path, recursive: bool) -> Iterable[Path]:
    if not root.exists():
        return

    if root.is_file():
        if root.suffix.lower() in SUPPORTED_EXTS:
            yield root
        return

    it = root.rglob("*") if recursive else root.glob("*")
    for p in it:
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def _to_float01(gray_u8: np.ndarray) -> np.ndarray:
    return gray_u8.astype(np.float32) / 255.0


def _to_u8(gray01: np.ndarray) -> np.ndarray:
    x = np.clip(gray01 * 255.0 + 0.5, 0, 255)
    return x.astype(np.uint8)


def fft2_shift(img01: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(img01)
    return np.fft.fftshift(f)


def ifft2_unshift(Fshift: np.ndarray) -> np.ndarray:
    f = np.fft.ifftshift(Fshift)
    img = np.fft.ifft2(f)
    return np.real(img).astype(np.float32)


def log_magnitude(Fshift: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mag = np.abs(Fshift)
    return np.log(mag + eps).astype(np.float32)


def make_uv_grid(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return U,V grids centered at (0,0) in fftshift coordinates."""
    u = np.arange(w, dtype=np.float32) - (w / 2.0)
    v = np.arange(h, dtype=np.float32) - (h / 2.0)
    U, V = np.meshgrid(u, v)
    return U, V


def estimate_stripe_normal_angle(mag_log: np.ndarray, dc_exclude: int) -> float:
    """Estimate dominant normal direction (angle) from Fourier magnitude.

    Stripes in spatial domain -> peaks along the normal direction in frequency domain.
    We build an angular energy histogram on a ring and pick the max bin.

    Returns angle in degrees in [0,180): angle of vector (u,v).
    """
    h, w = mag_log.shape
    U, V = make_uv_grid(h, w)
    R = np.sqrt(U * U + V * V)

    r_max = min(h, w) * RING_R_MAX_RATIO
    ring = (R >= float(RING_R_MIN)) & (R <= float(r_max))
    ring &= R >= float(dc_exclude)

    # remove DC and very low freq
    A = (np.degrees(np.arctan2(V, U)) + 180.0) % 180.0  # fold symmetry

    # weights: suppress background by subtracting median
    vals = mag_log[ring]
    if vals.size == 0:
        return 0.0
    vals = np.maximum(vals - np.median(vals), 0.0)

    ang = A[ring]
    bins = np.linspace(0.0, 180.0, N_ANGLE_BINS + 1, dtype=np.float32)
    hist, _ = np.histogram(ang, bins=bins, weights=vals)

    k = int(np.argmax(hist))
    angle = float((bins[k] + bins[k + 1]) / 2.0)
    return angle


def _angle_to_unitvec(angle_deg: float) -> Tuple[float, float]:
    a = np.deg2rad(angle_deg)
    return float(np.cos(a)), float(np.sin(a))


def detect_peaks_along_direction(
    mag_log: np.ndarray,
    normal_angle_deg: float,
    dc_exclude: int,
) -> List[Tuple[int, int]]:
    """Detect symmetric peak pairs along the given normal direction.

    We sample magnitude along a line through the spectrum center in direction n, and
    locate local maxima above a relative threshold.

    Returns list of peaks in (u_idx, v_idx) indices (fftshift domain).
    """
    h, w = mag_log.shape
    cx = w // 2
    cy = h // 2

    nx, ny = _angle_to_unitvec(normal_angle_deg)

    # sample radii
    r_max = int(min(h, w) * PEAK_MAX_R_RATIO)
    r_min = int(max(dc_exclude + 2, PEAK_MIN_R))

    rs = np.arange(-r_max, r_max + 1, dtype=np.int32)
    us = np.clip((cx + rs * nx).round().astype(np.int32), 0, w - 1)
    vs = np.clip((cy + rs * ny).round().astype(np.int32), 0, h - 1)

    prof = mag_log[vs, us].astype(np.float32)

    # ignore center band
    center_mask = (np.abs(rs) < r_min)
    prof2 = prof.copy()
    prof2[center_mask] = -np.inf

    # threshold relative to max
    finite = np.isfinite(prof2)
    if not np.any(finite):
        return []
    mmax = float(np.max(prof2[finite]))
    if not np.isfinite(mmax) or mmax <= 0:
        return []
    thresh = mmax * float(PEAK_REL_THRESH)

    # find local maxima
    peaks_r: List[int] = []
    for i in range(1, len(rs) - 1):
        if not np.isfinite(prof2[i]):
            continue
        if prof2[i] < thresh:
            continue
        if prof2[i] >= prof2[i - 1] and prof2[i] >= prof2[i + 1]:
            peaks_r.append(int(rs[i]))

    # sort by magnitude descending
    peaks_r.sort(key=lambda rr: float(mag_log[int(np.clip(cy + rr * ny, 0, h - 1)), int(np.clip(cx + rr * nx, 0, w - 1))]), reverse=True)

    # keep topk on the positive side (and include symmetric)
    pos = [r for r in peaks_r if r > 0]
    pos = pos[: int(PEAK_TOPK)]

    peaks: List[Tuple[int, int]] = []
    for r in pos:
        for rr in (r, -r):
            u = int(np.clip(round(cx + rr * nx), 0, w - 1))
            v = int(np.clip(round(cy + rr * ny), 0, h - 1))
            peaks.append((u, v))

    # de-dup
    uniq: List[Tuple[int, int]] = []
    s = set()
    for p in peaks:
        if p not in s:
            uniq.append(p)
            s.add(p)
    return uniq


def build_notch_mask(h: int, w: int, peaks: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Build smooth notch rejection mask in fftshift coords.

    Mask is in [0,1], where 1 means keep, 0 means reject.
    """
    if not peaks:
        return np.ones((h, w), dtype=np.float32)

    U, V = make_uv_grid(h, w)
    mask = np.ones((h, w), dtype=np.float32)

    # coordinate arrays for peak centers
    for (u_idx, v_idx) in peaks:
        # convert indices to centered coords
        u0 = float(u_idx - (w / 2.0))
        v0 = float(v_idx - (h / 2.0))

        # elliptical gaussian notch
        du = U - u0
        dv = V - v0
        d2 = (du * du) / float(NOTCH_SIGMA_R * NOTCH_SIGMA_R) + (dv * dv) / float(NOTCH_SIGMA_T * NOTCH_SIGMA_T)
        notch = np.exp(-0.5 * d2).astype(np.float32)

        # rejection: 1 - strength*notch
        mask *= (1.0 - float(NOTCH_STRENGTH) * notch)

    return np.clip(mask, 0.0, 1.0)


def apply_frequency_filter(gray01: np.ndarray) -> Tuple[np.ndarray, StripeModel, np.ndarray, np.ndarray]:
    """Return filtered image and debug matrices.

    Returns:
      out01, model, mag_log, mask
    """
    h, w = gray01.shape
    F = fft2_shift(gray01)
    mag = log_magnitude(F)

    normal_angle = estimate_stripe_normal_angle(mag, dc_exclude=DC_EXCLUDE_RADIUS)
    stripe_angle = (normal_angle + 90.0) % 180.0

    peaks = detect_peaks_along_direction(mag, normal_angle, dc_exclude=DC_EXCLUDE_RADIUS)
    mask = build_notch_mask(h, w, peaks)

    # also protect DC region: keep it
    U, V = make_uv_grid(h, w)
    R = np.sqrt(U * U + V * V)
    dc = (R <= float(DC_EXCLUDE_RADIUS)).astype(np.float32)
    mask = np.maximum(mask, dc)

    Ff = F * mask
    out = ifft2_unshift(Ff)

    # normalize to [0,1] using original mean/std (avoid contrast shift)
    out = np.clip(out, 0.0, 1.0)

    model = StripeModel(
        stripe_angle_deg=float(stripe_angle),
        normal_angle_deg=float(normal_angle),
        peaks_uv=[(int(u), int(v)) for (u, v) in peaks],
    )
    return out, model, mag, mask


def _save_debug(debug_dir: Path, mag: np.ndarray, mask: np.ndarray, model: StripeModel) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)

    # mag visualization
    mag_norm = mag - np.min(mag)
    if np.max(mag_norm) > 0:
        mag_norm = mag_norm / np.max(mag_norm)
    mag_u8 = _to_u8(mag_norm)
    mag_color = cv2.applyColorMap(mag_u8, cv2.COLORMAP_TURBO)

    # draw peaks
    for (u, v) in model.peaks_uv:
        cv2.circle(mag_color, (u, v), 4, (0, 0, 0), 2)
        cv2.circle(mag_color, (u, v), 3, (0, 255, 255), 1)

    cv2.imwrite(str(debug_dir / "fft_logmag.png"), mag_color)

    # mask visualization
    mask_u8 = _to_u8(mask)
    cv2.imwrite(str(debug_dir / "mask.png"), mask_u8)

    # text
    (debug_dir / "info.txt").write_text(
        f"stripe_angle_deg={model.stripe_angle_deg:.2f}\n"
        f"normal_angle_deg={model.normal_angle_deg:.2f}\n"
        f"peaks={model.peaks_uv}\n",
        encoding="utf-8",
    )


def filter_image_bgr(img_bgr: np.ndarray) -> Tuple[np.ndarray, StripeModel]:
    if FILTER_ON_LUMA:
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[..., 0]
        y01 = _to_float01(y)
        out01, model, _, _ = apply_frequency_filter(y01)
        ycrcb[..., 0] = _to_u8(out01)
        out_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return out_bgr, model

    # else: grayscale then replicate
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    out01, model, _, _ = apply_frequency_filter(_to_float01(gray))
    out = _to_u8(out01)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return out_bgr, model


def smoke_test_synthetic() -> None:
    """Generate a synthetic striped image and verify pipeline runs."""
    h, w = 256, 256
    base = np.zeros((h, w), np.float32) + 0.5

    # synthetic stripes: rotate a sine grating
    angle = 30.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x0 = xx - w / 2
    y0 = yy - h / 2
    a = np.deg2rad(angle)
    t = x0 * np.cos(a) + y0 * np.sin(a)
    stripes = 0.08 * np.sin(2 * np.pi * t / 12.0)

    img = np.clip(base + stripes, 0, 1)
    out01, model, mag, mask = apply_frequency_filter(img)

    # ensure output differs and mask has notches
    assert np.mean(np.abs(out01 - img)) > 1e-4
    assert float(np.mean(mask)) < 0.999

    # save to temp-like folder under OUTPUT_DIR
    dbg = Path(OUTPUT_DIR) / "_synthetic_test"
    dbg.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dbg / "in.png"), _to_u8(img))
    cv2.imwrite(str(dbg / "out.png"), _to_u8(out01))
    _save_debug(dbg, mag, mask, model)


def main() -> None:
    in_dir = Path(INPUT_DIR).expanduser().resolve()
    out_dir = Path(OUTPUT_DIR).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # quick smoke test so failures are obvious
    smoke_test_synthetic()

    it = in_dir.rglob("*") if RECURSIVE else in_dir.glob("*")

    processed = 0
    skipped = 0

    for src in it:
        if not src.is_file() or src.suffix.lower() not in SUPPORTED_EXTS:
            continue

        rel = src.relative_to(in_dir)
        dst = out_dir / rel
        if dst.exists() and (not OVERWRITE):
            skipped += 1
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            skipped += 1
            continue

        # Apply
        if FILTER_ON_LUMA:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            y = ycrcb[..., 0]
            out01, model, mag, mask = apply_frequency_filter(_to_float01(y))
            ycrcb[..., 0] = _to_u8(out01)
            out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            out01, model, mag, mask = apply_frequency_filter(_to_float01(gray))
            out = cv2.cvtColor(_to_u8(out01), cv2.COLOR_GRAY2BGR)

        # Save
        ext = dst.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            cv2.imwrite(str(dst), out, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)])
        else:
            cv2.imwrite(str(dst), out)

        if SAVE_DEBUG:
            debug_dir = (dst.parent / f"{dst.stem}{DEBUG_SUFFIX}")
            _save_debug(debug_dir, mag, mask, model)

        processed += 1

    print(f"Done. processed={processed}, skipped={skipped}, out={out_dir}")


if __name__ == "__main__":
    main()

