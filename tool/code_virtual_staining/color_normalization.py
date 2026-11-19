import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog


def main():
    # 隐藏主窗口，只显示对话框
    root = tk.Tk()
    root.withdraw()

    # 1. 选择参考图像
    ref_path = filedialog.askopenfilename(
        title="选择参考图像（颜色归一化目标）",
        filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                   ("所有文件", "*.*")],
    )
    if not ref_path:
        print("未选择参考图像，退出。")
        return

    # 2. 选择待归一化图像（可多选）
    tgt_paths = filedialog.askopenfilenames(
        title="选择待归一化图像（可多选）",
        filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                   ("所有文件", "*.*")],
    )
    if not tgt_paths:
        print("未选择待归一化图像，退出。")
        return

    # 3. 选择输出文件夹
    out_dir = filedialog.askdirectory(title="选择输出文件夹")
    if not out_dir:
        print("未选择输出文件夹，退出。")
        return

    root.destroy()

    # 读取参考图像
    ref = _read_bgr(ref_path)

    # 批量归一化并保存
    count = 0
    for tgt_path in tgt_paths:
        try:
            tgt = _read_bgr(tgt_path)
            norm = reinhard_normalize_bgr(ref, tgt)

            base_name = os.path.splitext(os.path.basename(tgt_path))[0]
            out_path = os.path.join(out_dir, f"{base_name}_normalized.png")

            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            ok = cv2.imwrite(out_path, norm)
            if not ok:
                print(f"保存失败: {out_path}")
                continue

            print(f"[{count + 1}/{len(tgt_paths)}] Saved: {out_path}")
            count += 1
        except Exception as e:
            print(f"处理失败 [{tgt_path}]: {e}")

    print(f"完成！共处理 {count}/{len(tgt_paths)} 张图像。")

def _mean_std(img_lab: np.ndarray, mask: np.ndarray | None = None):
    """计算每个通道的均值与标准差（可选 mask）。"""
    means = []
    stds = []
    for c in range(3):
        ch = img_lab[:, :, c]
        if mask is not None:
            vals = ch[mask]
        else:
            vals = ch.reshape(-1)
        vals = vals.astype(np.float32)
        mu = float(vals.mean()) if vals.size else 0.0
        sigma = float(vals.std(ddof=0)) if vals.size else 1.0
        if sigma < 1e-6:
            sigma = 1.0
        means.append(mu)
        stds.append(sigma)
    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


def reinhard_normalize_bgr(
    source_bgr: np.ndarray,
    target_bgr: np.ndarray,
    mask_source: np.ndarray | None = None,
    mask_target: np.ndarray | None = None,
) -> np.ndarray:
    """
    将 target 的颜色分布归一化到 source（参考）上（Reinhard，Lab 空间）。
    source_bgr: 参考图（BGR）
    target_bgr: 待归一化图（BGR）
    mask_source/mask_target: 可选，bool mask（True 表示参与统计的像素）
    """
    if source_bgr is None or target_bgr is None:
        raise ValueError("source_bgr/target_bgr 不能为空")

    src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    src_mean, src_std = _mean_std(src_lab, mask_source)
    tgt_mean, tgt_std = _mean_std(tgt_lab, mask_target)

    out = tgt_lab.copy()
    for c in range(3):
        out[:, :, c] = (out[:, :, c] - tgt_mean[c]) * (src_std[c] / tgt_std[c]) + src_mean[c]

    # Lab in OpenCV: L,a,b in [0,255]
    out = np.clip(out, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    return out_bgr


def reinhard_normalize_masked_bgr(
    reference_bgr: np.ndarray,
    target_bgr: np.ndarray,
    reference_mask: np.ndarray,
    target_mask: np.ndarray,
    feather_width: float = 64.0,
    strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """仅归一化目标掩膜区域，并在掩膜内侧平滑过渡到原始颜色。

    使用浮点 CIELAB（L: 0–100），避免 8 位 Lab 量化在浅色区域产生色阶。
    两张图及其掩膜不要求彼此同尺寸，只须各自图像和掩膜尺寸匹配。
    feather_width 是目标图原生像素中的内侧过渡宽度，0 表示直接应用。
    返回 (BGR 输出, 0–1 混合权重, 颜色统计报告)。掩膜外像素逐值保留。
    可只传入掩膜的外接矩形以降低大图内存占用，调用方再贴回原图。
    """
    if not np.isfinite(feather_width) or feather_width < 0:
        raise ValueError("feather_width 必须是非负有限数值")
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength 必须在 0–1 之间")
    masks = []
    for image, mask, name in [(reference_bgr, reference_mask, "参考"), (target_bgr, target_mask, "目标")]:
        if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{name}图像必须为 uint8 BGR 三通道")
        mask = np.asarray(mask)
        if mask.shape != image.shape[:2]:
            raise ValueError(f"{name}图像与掩膜尺寸不匹配")
        mask = (mask != 0).astype(np.uint8)
        if not np.any(mask):
            raise ValueError(f"{name}掩膜为空")
        masks.append(mask)
    reference_mask, target_mask = masks

    def lab_stats(lab, mask):
        mean, std = cv2.meanStdDev(lab, mask=mask)
        return mean.ravel(), std.ravel()

    reference_lab = cv2.cvtColor(reference_bgr.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_bgr.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
    ref_mean, ref_std = lab_stats(reference_lab, reference_mask)
    tgt_mean, tgt_std = lab_stats(target_lab, target_mask)
    del reference_lab
    gain = np.divide(ref_std, tgt_std, out=np.ones(3), where=tgt_std > 1e-4).astype(np.float32)
    # 恒定通道无法凭空产生参考图的变化，只做均值平移。
    normalized_lab = (target_lab-tgt_mean.astype(np.float32))*gain+ref_mean.astype(np.float32)
    normalized = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)
    normalized = np.clip(normalized*255, 0, 255)
    del normalized_lab, target_lab

    if feather_width > 0:
        # 外接矩形可能恰好触及掩膜边界，显式补零使四边距离均正确。
        distance = cv2.distanceTransform(np.pad(target_mask, 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        blend = np.clip((distance-1)/feather_width, 0, 1)
        blend = blend*blend*(3-2*blend)  # Smoothstep 两端斜率为 0。
    else:
        blend = target_mask.astype(np.float32)
    blend *= strength
    output = target_bgr.copy()
    selected = blend > 0
    if np.any(selected):
        alpha = blend[selected, None]
        output[selected] = np.clip(np.rint(target_bgr[selected]*(1-alpha)+normalized[selected]*alpha), 0, 255).astype(np.uint8)
    output_lab = cv2.cvtColor(output.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
    out_mean, out_std = lab_stats(output_lab, target_mask)
    report = {
        "method": "masked_reinhard_float_cielab",
        "feather_width_px": float(feather_width), "strength": float(strength),
        "feather_profile": "inner smoothstep; first inside pixel and all outside pixels unchanged",
        "reference_mask_pixels": int(np.count_nonzero(reference_mask)),
        "target_mask_pixels": int(np.count_nonzero(target_mask)),
        "reference_lab_mean": ref_mean.tolist(), "reference_lab_std": ref_std.tolist(),
        "target_before_lab_mean": tgt_mean.tolist(), "target_before_lab_std": tgt_std.tolist(),
        "target_after_lab_mean": out_mean.tolist(), "target_after_lab_std": out_std.tolist(),
        "channel_gain": gain.tolist(),
        "mean_color_delta_e_before": float(np.linalg.norm(tgt_mean-ref_mean)),
        "mean_color_delta_e_after": float(np.linalg.norm(out_mean-ref_mean)),
        "outside_mask_changed_pixels": int(np.count_nonzero(np.any(output[target_mask == 0] != target_bgr[target_mask == 0], axis=1))),
        "changed_pixels": int(np.count_nonzero(np.any(output != target_bgr, axis=2))),
    }
    core = (blend >= 1-1e-6).astype(np.uint8)
    if np.any(core):
        mean, std = lab_stats(output_lab, core)
        report["full_strength_lab_mean"], report["full_strength_lab_std"] = mean.tolist(), std.tolist()
        report["full_strength_pixels"] = int(np.count_nonzero(core))
    return output, blend, report


def _read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"读取失败: {path}")
    return img

if __name__ == "__main__":
    main()
