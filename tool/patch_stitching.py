import re
from pathlib import Path

import cv2
import numpy as np


def _hann2d(h: int, w: int) -> np.ndarray:
    """2D Hann window in [0,1]."""
    if h <= 0 or w <= 0:
        raise ValueError("invalid window size")
    wy = np.hanning(h).astype(np.float32)
    wx = np.hanning(w).astype(np.float32)
    w2d = np.outer(wy, wx)
    # hanning 在边缘为0；为了避免边缘完全为0导致某些像素weight=0，这里给一个下限
    w2d = np.clip(w2d, 1e-3, 1.0)
    return w2d


def _gaussian2d(h: int, w: int, sigma_scale: float = 0.125) -> np.ndarray:
    """2D Gaussian weight in [0,1], center high, edge low."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    sigma_y = max(1.0, h * sigma_scale)
    sigma_x = max(1.0, w * sigma_scale)
    w2d = np.exp(-(((yy - cy) ** 2) / (2 * sigma_y**2) + ((xx - cx) ** 2) / (2 * sigma_x**2)))
    w2d = w2d.astype(np.float32)
    w2d = (w2d - w2d.min()) / (w2d.max() - w2d.min() + 1e-8)
    w2d = np.clip(w2d, 1e-3, 1.0)
    return w2d


def _parse_xy_from_name(name: str):
    """从文件名解析 y/x: patch_y000000_x000000.png"""
    m = re.search(r"patch_y(\d+)_x(\d+)", name)
    if not m:
        return None
    y = int(m.group(1))
    x = int(m.group(2))
    return x, y


# ======================
# 直接写死配置：你自己改这里
# ======================
patch_dir = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_fake\Trans_kidney_20x_24_1024\mouse_kidney_dual_UTOM_256\test_latest\images_fake" # 染色后的patch文件夹（文件名需包含 patch_yxxxxx_xxxxxx）
out_path = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_fake\WSI_fake\dual_refined_kidney_20x_24_256_1024.png" # 输出大图

# 大图原始尺寸（推荐手动填原图尺寸，最稳；如果填 None，会用patch最大坐标推断）
orig_w = None  # e.g. 8000
orig_h = None  # e.g. 6000

# 融合权重窗类型：'hann' 或 'gaussian'
weight_mode = "hann"

# 如果你的patch边缘是用白色padding出来的，且不希望padding区域参与融合，可以打开：
# 会把“接近pad_color”的像素权重降低到0
ignore_padding = False
pad_color_bgr = (255, 255, 255)
pad_tol = 3  # 允许的颜色误差

# 输出保存为BGR PNG


patch_root = Path(patch_dir)
if not patch_root.exists():
    raise FileNotFoundError(f"patch_dir not found: {patch_root}")

# 收集patch
img_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
items = []  # (x, y, path)
for p in patch_root.rglob("*"):
    if not p.is_file() or p.suffix.lower() not in img_exts:
        continue
    xy = _parse_xy_from_name(p.name)
    if xy is None:
        continue
    x, y = xy
    items.append((x, y, p))

if not items:
    raise RuntimeError("No patch images found with name pattern patch_yXXXXXX_xXXXXXX")

# 按坐标排序
items.sort(key=lambda t: (t[1], t[0]))

# 读第一张确定patch大小
first = cv2.imread(str(items[0][2]), cv2.IMREAD_COLOR)
if first is None:
    raise RuntimeError(f"Failed to read: {items[0][2]}")
ph, pw = first.shape[:2]

# 推断 stride：取最常见的相邻差值（比直接填更不容易出错）
xs_sorted = sorted({x for x, _, _ in items})
ys_sorted = sorted({y for _, y, _ in items})

def _most_common_positive_diff(vals):
    diffs = [b - a for a, b in zip(vals[:-1], vals[1:]) if (b - a) > 0]
    if not diffs:
        return None
    # 简单统计众数
    uniq, cnt = np.unique(np.array(diffs, dtype=np.int32), return_counts=True)
    return int(uniq[np.argmax(cnt)])

sx = _most_common_positive_diff(xs_sorted)
sy = _most_common_positive_diff(ys_sorted)
if sx is None:
    sx = pw
if sy is None:
    sy = ph

# 推断大图尺寸
max_x = max(x for x, _, _ in items)
max_y = max(y for _, y, _ in items)
W = orig_w if orig_w is not None else (max_x + pw)
H = orig_h if orig_h is not None else (max_y + ph)

# 累积器
acc = np.zeros((H, W, 3), dtype=np.float32)
wsum = np.zeros((H, W, 1), dtype=np.float32)

# 权重窗
if weight_mode.lower() == "hann":
    w_patch = _hann2d(ph, pw)
elif weight_mode.lower() == "gaussian":
    w_patch = _gaussian2d(ph, pw)
else:
    raise ValueError("weight_mode must be 'hann' or 'gaussian'")

w_patch = w_patch[..., None]  # (ph,pw,1)

pad_color = np.array(pad_color_bgr, dtype=np.uint8).reshape(1, 1, 3)

# 拼接
for x, y, p in items:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read: {p}")
    if img.shape[0] != ph or img.shape[1] != pw:
        raise RuntimeError(f"Patch size mismatch: {p} got {img.shape[1]}x{img.shape[0]}, expect {pw}x{ph}")

    x0, y0 = x, y
    x1, y1 = min(W, x + pw), min(H, y + ph)
    if x0 >= W or y0 >= H:
        continue

    # 对于最后一行/列可能越界（如果你用pad切的，理论上不会越界；这里做个保护）
    roi_w = x1 - x0
    roi_h = y1 - y0

    img_roi = img[:roi_h, :roi_w].astype(np.float32)
    w_roi = w_patch[:roi_h, :roi_w]

    if ignore_padding:
        # 找到接近 pad_color 的区域，把权重置0
        diff = np.max(np.abs(img[:roi_h, :roi_w].astype(np.int16) - pad_color.astype(np.int16)), axis=2)
        mask = (diff <= pad_tol).astype(np.float32)  # 1=padding
        w_roi = w_roi * (1.0 - mask[..., None])

    acc[y0:y1, x0:x1] += img_roi * w_roi
    wsum[y0:y1, x0:x1] += w_roi

# 归一化
wsum_safe = np.maximum(wsum, 1e-6)
out = acc / wsum_safe
out = np.clip(out, 0, 255).astype(np.uint8)

out_path_p = Path(out_path)
out_path_p.parent.mkdir(parents=True, exist_ok=True)
ok = cv2.imwrite(str(out_path_p), out)
if not ok:
    raise RuntimeError(f"Failed to save: {out_path_p}")

print("Done.")
print(f"Patch dir: {patch_root}")
print(f"Output: {out_path_p}")
print(f"Patch size: {pw}x{ph}")
print(f"Inferred stride: sx={sx}, sy={sy}")
print(f"Canvas size: W={W}, H={H}")
print(f"Num patches: {len(items)}")
