import cv2
import os
import numpy as np

input_dir = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_BF\patches_re\images"
output_dir = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_BF\patches_re\images_white_bg"
os.makedirs(output_dir, exist_ok=True)

threshold = 104

# 形态学参数：用于在模糊前“缩小背景/扩大前景”，减少软边对白背景的侵入
morph_open_ksize = 3   # 去小噪点；越大越强（必须奇数）
dilate_ksize = 5       # 扩大前景；越大越强（必须奇数）
dilate_iter = 1        # 扩大次数

# 软边缘强度：越大过渡越宽。要求为正奇数。
blur_ksize = 11

# 关键改动：背景变白但尽量保留纹理/对比
# 1) 先对整图做局部对比增强（CLAHE），避免“背景一白就啥也看不见”
clahe_clip_limit = 2.0
clahe_tile_grid = 8  # 常用 8 或 16

# 2) 背景提亮强度：0~1，越大越接近纯白（但纹理还会通过overlay保留一些）
#    如果你想更“纸白”，增大到 0.8~1.0；想更保留纹理，减小到 0.4~0.6
bg_whiten_strength = 0.75

# ---- 参数修正（保证为奇数/合法） ----
for name in ("morph_open_ksize", "dilate_ksize", "blur_ksize"):
    v = locals()[name]
    if v % 2 == 0:
        locals()[name] = v + 1
bg_whiten_strength = float(np.clip(bg_whiten_strength, 0.0, 1.0))
clahe_tile_grid = int(max(1, clahe_tile_grid))

# CLAHE对象（复用）
clahe = cv2.createCLAHE(clipLimit=float(clahe_clip_limit), tileGridSize=(clahe_tile_grid, clahe_tile_grid))


def apply_clahe_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """在Lab空间对L通道做CLAHE，增强局部对比；输出仍为BGR uint8。"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def overlay_towards_white(img: np.ndarray, strength: float) -> np.ndarray:
    """把图像按overlay/screen风格朝白色推，同时尽量保留对比。

    这里用的是 screen: out = 1 - (1-img)*(1-white*strength) = 1 - (1-img)*(1-strength)
    strength=0 -> 不变；strength=1 -> 纯白。
    """
    img_f = img.astype(np.float32) / 255.0
    out = 1.0 - (1.0 - img_f) * (1.0 - strength)
    return (out * 255.0).clip(0, 255).astype(np.uint8)


for filename in os.listdir(input_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
        continue

    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] failed to read: {img_path}")
        continue

    # A) 先做局部对比增强（整图），让纹理更稳
    img_enh = apply_clahe_bgr(img)

    # 1) 阈值分割：用三通道均值作为灰度近似
    img_mean = img.mean(axis=2).astype("uint8")
    # THRESH_BINARY_INV: <= threshold 为 255 (前景)，> threshold 为 0 (背景)
    _, fg_mask = cv2.threshold(img_mean, threshold, 255, cv2.THRESH_BINARY_INV)

    # 2) 模糊前先做形态学，让前景边界更“保守”
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_ksize, morph_open_ksize))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
    fg_mask = cv2.dilate(fg_mask, dilate_kernel, iterations=dilate_iter)

    # 3) 软mask
    soft_fg = cv2.GaussianBlur(fg_mask, (blur_ksize, blur_ksize), 0).astype(np.float32) / 255.0
    soft_fg = soft_fg[:, :, None]  # HxWx1
    soft_bg = 1.0 - soft_fg

    # 4) 背景“提亮到白但保留对比”：先用screen朝白推，再按soft_bg混合
    bg_whitened = overlay_towards_white(img_enh, bg_whiten_strength)

    out_f = img_enh.astype(np.float32) * soft_fg + bg_whitened.astype(np.float32) * soft_bg
    out = out_f.clip(0, 255).astype(np.uint8)

    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, out)

print("Done.")
