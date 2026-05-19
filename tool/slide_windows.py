import os
import math
import cv2
import numpy as np


def _positions_1d(length: int, window: int, stride: int):
    """生成一维滑窗起点序列，保证覆盖到末端（允许越界，靠pad补齐）。"""
    if length <= 0:
        raise ValueError("length must be > 0")
    if window <= 0:
        raise ValueError("window must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    if length <= window:
        return [0]

    # 常规网格
    n = int(math.ceil((length - window) / stride)) + 1
    return [i * stride for i in range(n)]


def sliding_window_patches(
    img: np.ndarray,
    window_w: int,
    window_h: int,
    overlap_w: int,
    overlap_h: int,
    pad_value_bgr=(255, 255, 255),
):
    """对单张图生成滑窗patch。

    返回: list[ (x, y, patch_bgr) ]，其中 (x,y) 为左上角在原图坐标系下的起点(可使patch越界)。
    """
    if img is None:
        raise ValueError("img is None")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expect BGR image HxWx3, got shape={img.shape}")

    H, W = img.shape[:2]
    if window_w <= 0 or window_h <= 0:
        raise ValueError("window size must be > 0")
    if overlap_w < 0 or overlap_h < 0:
        raise ValueError("overlap must be >= 0")
    if overlap_w >= window_w or overlap_h >= window_h:
        raise ValueError("overlap must be smaller than window size")

    stride_x = window_w - overlap_w
    stride_y = window_h - overlap_h

    xs = _positions_1d(W, window_w, stride_x)
    ys = _positions_1d(H, window_h, stride_y)

    pad_value = np.array(pad_value_bgr, dtype=np.uint8).reshape(1, 1, 3)

    out = []
    for y in ys:
        for x in xs:
            # 需要从原图取的有效区域
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(W, x + window_w)
            y1 = min(H, y + window_h)

            patch = np.tile(pad_value, (window_h, window_w, 1))

            # 放回patch中的位置（只在右/下越界时pad；若x/y为负也能工作，但本脚本不会生成负起点）
            px0 = x0 - x
            py0 = y0 - y
            px1 = px0 + (x1 - x0)
            py1 = py0 + (y1 - y0)

            patch[py0:py1, px0:px1] = img[y0:y1, x0:x1]
            out.append((x, y, patch))

    return out, xs, ys, (stride_x, stride_y)


# ======================
# 直接写死配置：你自己改这里
# ======================
img_path = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_Trans\WSI\Slide 24-Region 009.jpg"
save_dir = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_Trans\mosaic\kidney_24_512_1024"   # TODO: 改成输出目录

# 图像采样比例
scale = 1

# 窗口大小
window_w = 1024
window_h = 1024

# 重叠大小（与窗口同单位：像素）
overlap_w = 512
overlap_h = 512

# 边缘补全背景色：白(255,255,255) 或 黑(0,0,0)
pad_value_bgr = (255, 255, 255)

# 输出格式
ext = ".png"


os.makedirs(save_dir, exist_ok=True)

img = cv2.imread(img_path, cv2.IMREAD_COLOR)
if img is None:
    raise ValueError(f"读取失败: {img_path}")

if scale != 1.0:
    new_w = int(img.shape[1] * scale)
    new_h = int(img.shape[0] * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

patches, xs, ys, (sx, sy) = sliding_window_patches(
    img,
    window_w=window_w,
    window_h=window_h,
    overlap_w=overlap_w,
    overlap_h=overlap_h,
    pad_value_bgr=pad_value_bgr,
)

# 命名：x/y 为左上角坐标，方便重组
for i, (x, y, patch) in enumerate(patches):
    name = f"patch_y{y:06d}_x{x:06d}{ext}"
    ok = cv2.imwrite(os.path.join(save_dir, name), patch)
    if not ok:
        raise RuntimeError(f"保存失败: {name}")

print("Done.")
print(f"Input: {img_path}")
print(f"Output dir: {save_dir}")
print(f"Image size: W={img.shape[1]}, H={img.shape[0]}")
print(f"Window: W={window_w}, H={window_h}")
print(f"Overlap: W={overlap_w}, H={overlap_h}")
print(f"Stride: sx={sx}, sy={sy}")
print(f"Grid: nx={len(xs)} (x0..={xs[-1]}), ny={len(ys)} (y0..={ys[-1]})")
print(f"Num patches: {len(patches)}")

