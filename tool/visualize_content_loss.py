"""
可视化 UTOM content loss 的中间结果：
- 逐通道 Sobel 梯度幅值
- 加权融合梯度（跨通道算术平均）
- 低频分量（通道均值 + 高斯模糊）

与 utom_model.py 中 grad_mag() / low_freq() 计算完全一致。
"""

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog


# ============================================================
# 与 utom_model.py 等价的 numpy 实现
# ============================================================

SOBEL_KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_KY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)


def _sobel_channel(x: np.ndarray) -> np.ndarray:
    """单通道 Sobel 梯度幅值: sqrt(gx^2 + gy^2 + 1e-6)."""
    gx = cv2.filter2D(x, cv2.CV_32F, SOBEL_KX)
    gy = cv2.filter2D(x, cv2.CV_32F, SOBEL_KY)
    return np.sqrt(gx ** 2 + gy ** 2 + 1e-6)


def grad_mag(x: np.ndarray, weights=None) -> np.ndarray:
    """逐通道 Sobel 梯度幅值 → 加权算术平均融合。

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

    if C == 1:
        return stacked[:, :, 0]

    # 加权算术平均
    if weights is None:
        w = np.ones(C, dtype=np.float32) / C
    else:
        w = np.array(weights, dtype=np.float32)
        w = w / w.sum()
    w = w.reshape(1, 1, C)

    return np.sum(stacked * w, axis=-1)  # (H, W)


def low_freq(x: np.ndarray) -> np.ndarray:
    """通道均值 + 高斯模糊 (radius=10, sigma=5.0, kernel=21×21)。

    x: (H, W, C) 或 (H, W)
    返回: (H, W) 低频图
    """
    if x.ndim == 2:
        x_mean = x
    else:
        x_mean = np.mean(x, axis=2)  # channel mean

    radius = 10
    sigma = 5.0
    coords = np.arange(2 * radius + 1, dtype=np.float32) - radius
    g1d = np.exp(-0.5 * (coords / sigma) ** 2)
    g1d = g1d / g1d.sum()
    kernel = np.outer(g1d, g1d)  # (21, 21)

    return cv2.filter2D(x_mean.astype(np.float32), cv2.CV_32F, kernel)


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
    # BF (单通道): grad
    bf_grad = grad_mag(bf)       # (H,W)
    bf_low  = low_freq(bf)       # (H,W)

    # AF (单通道): grad
    af_grad = grad_mag(af)
    af_low  = low_freq(af)

    # BF+AF (2 通道): grad per channel + fused
    bf_af = np.stack([bf, af], axis=-1)  # (H, W, 2)
    dual_grad_fused = grad_mag(bf_af)    # fused (weighted mean, default equal)
    dual_low = low_freq(bf_af)           # low-freq of channel-mean

    # H&E (3 通道): grad per channel + fused
    he_grad_r = grad_mag(he[:, :, 2])   # R channel
    he_grad_g = grad_mag(he[:, :, 1])   # G channel
    he_grad_b = grad_mag(he[:, :, 0])   # B channel
    he_grad_fused = grad_mag(he)         # 3-ch fused
    he_low  = low_freq(he)

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
    save_gray("05_BF_lowfreq.png", bf_low)

    save_gray("06_AF_grad.png", af_grad)
    save_gray("07_AF_lowfreq.png", af_low)

    save_gray("08_Dual_grad_fused.png", dual_grad_fused)
    save_gray("09_Dual_lowfreq.png", dual_low)

    save_gray("10_HE_grad_R.png", he_grad_r)
    save_gray("11_HE_grad_G.png", he_grad_g)
    save_gray("12_HE_grad_B.png", he_grad_b)
    save_gray("13_HE_grad_fused.png", he_grad_fused)
    save_gray("14_HE_lowfreq.png", he_low)

    print(f"\nDone. All images saved to: {out_dir}")
