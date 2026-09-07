"""
可视化 SPIF content loss 的中间结果：
- 逐通道 Sobel 梯度幅值
- 原始分辨率下、未进行逐图 RMS 归一化的加权融合梯度
- BF/AF 通道均值的 Gaussian 低频图、8×8 归一化粗图和强度相关性

对应 spif_model.py 中 FineStructureGradientLoss 和
CoarseIntensityDistributionLoss 的计算方式。
"""

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog


# ============================================================
# 与 spif_model.py 等价的 numpy 实现
# ============================================================

SOBEL_KX = np.array(
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32
) / 8.0
SOBEL_KY = np.array(
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32
) / 8.0
EPS = 1e-6


def _sobel_channel(x: np.ndarray) -> np.ndarray:
    """使用 reflect-101 padding 的单通道 Sobel 梯度幅值。"""
    gx = cv2.filter2D(
        x, cv2.CV_32F, SOBEL_KX, borderType=cv2.BORDER_REFLECT_101
    )
    gy = cv2.filter2D(
        x, cv2.CV_32F, SOBEL_KY, borderType=cv2.BORDER_REFLECT_101
    )
    return np.sqrt(gx ** 2 + gy ** 2 + EPS ** 2) - EPS


def grad_mag(x: np.ndarray, weights=None) -> np.ndarray:
    """逐通道 Sobel 梯度幅值 → 加权融合，不做逐图归一化。

    x: (H, W, C) 或 (H, W)  [值域任意，通常 [-1,1] 或 [0,255]]
    weights: 可选，长度=C 的权重列表，默认等权
    返回: (H, W) 融合梯度图
    """
    if x.ndim == 2:
        x = x[..., None]  # (H, W) -> (H, W, 1)

    C = x.shape[2]
    grads = []
    for c in range(C):
        grads.append(_sobel_channel(x[:, :, c]))

    stacked = np.stack(grads, axis=-1)  # (H, W, C)

    # 加权算术平均
    if weights is None:
        w = np.ones(C, dtype=np.float32) / C
    else:
        w = np.array(weights, dtype=np.float32)
        w = w / w.sum()
    w = w.reshape(1, 1, C)

    return np.sum(stacked * w, axis=-1)


def coarse_intensity_map(x: np.ndarray, weights=None, pool_size=8):
    """返回通道均值 Gaussian 图和 RMS 归一化后的粗网格。"""
    if x.ndim == 2:
        intensity = x
    else:
        channels = x.shape[2]
        if weights is None:
            weights = np.ones(channels, dtype=np.float32) / channels
        else:
            weights = np.asarray(weights, dtype=np.float32)
            weights = weights / weights.sum()
        intensity = np.sum(x * weights.reshape(1, 1, channels), axis=2)
    low = cv2.GaussianBlur(
        intensity,
        (21, 21),
        sigmaX=5.0,
        sigmaY=5.0,
        borderType=cv2.BORDER_REFLECT_101,
    )
    coarse = cv2.resize(
        low, (pool_size, pool_size), interpolation=cv2.INTER_AREA
    )
    coarse = coarse - coarse.mean()
    coarse = coarse / np.sqrt(np.mean(coarse ** 2) + EPS)
    return low, coarse


def coarse_intensity_loss(reference: np.ndarray, translated: np.ndarray):
    """返回粗网格相关性、1-rho 损失和逐格相关贡献。"""
    contribution = reference * translated
    correlation = float(contribution.mean())
    loss = 1.0 - correlation
    return correlation, loss, contribution


# ============================================================
# 可视化工具
# ============================================================

def norm_uint8(x: np.ndarray) -> np.ndarray:
    """归一化到 [0,255] uint8。"""
    x = x.astype(np.float32)
    vmin, vmax = x.min(), x.max()
    if vmax - vmin < 1e-6:
        return np.zeros_like(x, dtype=np.uint8)
    return ((x - vmin) / (vmax - vmin) * 255).astype(np.uint8)


def gray_to_bgr(x: np.ndarray) -> np.ndarray:
    """灰度 (H,W) 转为 BGR 三通道显示。"""
    return cv2.cvtColor(norm_uint8(x), cv2.COLOR_GRAY2BGR)


def tile_row(images: list, labels: list, gap=4):
    """水平拼接一组 BGR 图像，顶部加标签。"""
    h = max(img.shape[0] for img in images)
    w = max(img.shape[1] for img in images)

    rows = []
    for img, label in zip(images, labels):
        # 统一尺寸
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        canvas = cv2.copyMakeBorder(img, 24, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(canvas, label, (2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        rows.append(canvas)

    # 加间隔
    spacer = np.full((h + 24, gap, 3), 255, dtype=np.uint8)
    result = rows[0]
    for r in rows[1:]:
        result = np.hstack([result, spacer, r])
    return result


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # 选择三张图像
    bf_path = filedialog.askopenfilename(
        title="选择 BF 图像（灰度）",
        filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("所有文件", "*.*")],
    )
    if not bf_path:
        print("未选择 BF，退出。")
        root.destroy()
        exit()

    af_path = filedialog.askopenfilename(
        title="选择 AF 图像（灰度）",
        filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("所有文件", "*.*")],
    )
    if not af_path:
        print("未选择 AF，退出。")
        root.destroy()
        exit()

    he_path = filedialog.askopenfilename(
        title="选择 H&E 图像（RGB）",
        filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("所有文件", "*.*")],
    )
    if not he_path:
        print("未选择 H&E，退出。")
        root.destroy()
        exit()

    # 选择输出目录
    out_dir = filedialog.askdirectory(title="选择输出目录")
    if not out_dir:
        print("未选择输出目录，退出。")
        root.destroy()
        exit()

    root.destroy()

    # 读取图像
    bf = cv2.imread(bf_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    af = cv2.imread(af_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    he = cv2.imread(he_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0  # BGR

    print(f"BF : {bf_path}  ({bf.shape[1]}x{bf.shape[0]})")
    print(f"AF : {af_path}  ({af.shape[1]}x{af.shape[0]})")
    print(f"H&E: {he_path}  ({he.shape[1]}x{he.shape[0]})")

    # ---- 统一尺寸 ----
    H = min(bf.shape[0], af.shape[0], he.shape[0])
    W = min(bf.shape[1], af.shape[1], he.shape[1])
    bf = cv2.resize(bf, (W, H))
    af = cv2.resize(af, (W, H))
    he = cv2.resize(he, (W, H))

    # ---- 计算中间结果 ----
    # BF/AF 单通道尺度 1 梯度
    bf_grad = grad_mag(bf)
    af_grad = grad_mag(af)

    # BF+AF (2 通道): grad per channel + fused
    bf_af = np.stack([bf, af], axis=-1)  # (H, W, 2)
    dual_grad = grad_mag(bf_af)

    # H&E (3 通道): grad per channel + fused
    he_grad_r = grad_mag(he[:, :, 2])   # R channel
    he_grad_g = grad_mag(he[:, :, 1])   # G channel
    he_grad_b = grad_mag(he[:, :, 0])   # B channel
    he_grad = grad_mag(he)
    dual_low, dual_coarse = coarse_intensity_map(bf_af)
    he_low, he_coarse = coarse_intensity_map(he)
    intensity_corr, intensity_loss, coarse_contribution = (
        coarse_intensity_loss(dual_coarse, he_coarse)
    )

    def save_gray(path, img_data):
        """保存灰度图。"""
        cv2.imwrite(os.path.join(out_dir, path), norm_uint8(img_data))
        print(f"  {path}")

    def save_rgb(path, img_data):
        """保存 RGB/BGR 图。"""
        if img_data.ndim == 2:
            out = cv2.cvtColor(norm_uint8(img_data), cv2.COLOR_GRAY2BGR)
        else:
            out = (np.clip(img_data, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, path), out)
        print(f"  {path}")

    print("\nSaving:")
    save_rgb("01_BF_original.png", np.tile(bf[..., None], (1, 1, 3)))
    save_rgb("02_AF_original.png", np.tile(af[..., None], (1, 1, 3)))
    save_rgb("03_HE_original.png", he[..., ::-1])

    save_gray("04_BF_grad.png", bf_grad)
    save_gray("05_AF_grad.png", af_grad)

    save_gray("06_Dual_grad_original.png", dual_grad)

    save_gray("07_HE_grad_R.png", he_grad_r)
    save_gray("08_HE_grad_G.png", he_grad_g)
    save_gray("09_HE_grad_B.png", he_grad_b)
    save_gray("10_HE_grad_original.png", he_grad)

    save_gray("11_Dual_channel_mean_Gaussian.png", dual_low)
    save_gray("12_HE_channel_mean_Gaussian.png", he_low)
    save_gray("13_Dual_coarse_normalized_8x8.png", dual_coarse)
    save_gray("14_HE_coarse_normalized_8x8.png", he_coarse)
    save_gray("15_Coarse_correlation_contribution.png", coarse_contribution)
    print(
        "  Coarse intensity distribution: "
        f"correlation={intensity_corr:.6f}, "
        f"loss={intensity_loss:.6f}"
    )

    print(f"\nDone. All images saved to: {out_dir}")
