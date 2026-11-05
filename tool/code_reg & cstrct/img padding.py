import os
import cv2
import numpy as np
from natsort import natsorted

# =========================
# 参数（按需修改）
# =========================
input_dir = r"C:\Users\30927\Desktop\mouse_liver\liver"
output_dir = r"C:\Users\30927\Desktop\mouse_liver\liver_padded"

ext = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')

# padding 像素数：横向(左右)与纵向(上下)
pad_x = 2000  # 左右各padding多少像素
pad_y = 200  # 上下各padding多少像素

# padding颜色：白色背景（灰度=255 / 彩色=(255,255,255)）
pad_value_gray = 255
pad_value_bgr = (255, 255, 255)

# 可选：输出格式（保持原扩展名则设为 None；或强制 ".png"/".jpg"）
force_ext = None  # 例如 ".png"；None 表示保持原始后缀

# =========================
# 主程序
# =========================
os.makedirs(output_dir, exist_ok=True)

image_paths = []
for fn in os.listdir(input_dir):
    if fn.lower().endswith(ext):
        image_paths.append(os.path.join(input_dir, fn))
image_paths = natsorted(image_paths)

if len(image_paths) == 0:
    raise FileNotFoundError(f"No images found in: {input_dir}")

for i, img_path in enumerate(image_paths, start=1):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[Skip] Cannot read: {img_path}")
        continue

    # 根据通道数选择padding颜色
    if img.ndim == 2:
        value = pad_value_gray
    else:
        # BGR 或 BGRA 都能用三通道 value；BGRA 时 alpha 会默认填 0，这里手动处理一下更稳
        if img.shape[2] == 4:
            value = (pad_value_bgr[0], pad_value_bgr[1], pad_value_bgr[2], 255)
        else:
            value = pad_value_bgr

    padded = cv2.copyMakeBorder(
        img,
        top=pad_y, bottom=pad_y,
        left=pad_x, right=pad_x,
        borderType=cv2.BORDER_CONSTANT,
        value=value
    )

    base = os.path.basename(img_path)
    name, old_ext = os.path.splitext(base)
    out_ext = force_ext if force_ext is not None else old_ext
    out_path = os.path.join(output_dir, f"{name}{out_ext}")

    ok = cv2.imwrite(out_path, padded)
    if not ok:
        print(f"[Warn] Failed to write: {out_path}")
    else:
        h, w = padded.shape[:2]
        print(f"[{i}/{len(image_paths)}] {base} -> {os.path.basename(out_path)}  padded_size={w}x{h}")

print("Done.")