import os
import random
import numpy as np
from PIL import Image
import albumentations as A
import cv2 as cv

Image.MAX_IMAGE_PIXELS = None

# =========================
# 参数（按需修改）
# =========================
input_dir = r"C:\Users\30927\Desktop\mouse_liver\liver_HE"                 # 输入：包含若干大视场JPG的文件夹
output_images_dir = r"C:\Users\30927\Desktop\mouse_liver\patch_HE" # 输出：所有patch输出到同一个文件夹

patches_per_image = 100   # 每张大图生成多少个patch（通过过滤后达到该数量）
min_size = 512            # 最小patch尺寸
max_size = 512           # 最大patch尺寸

# 背景过滤：白背景判定（HSV：高亮度 + 低饱和度）
white_thr = 230           # V >= white_thr 认为够“白”
sat_thr = 30              # S <= sat_thr 认为够“灰白”
max_white_ratio = 0.6     # 白背景像素比例 > 该值则丢弃（背景太多）

# 其它
jpeg_quality = 95
seed = 42
prefix = "patch"

# 重要：超大图关闭PIL像素上限（避免 DecompressionBombError）
Image.MAX_IMAGE_PIXELS = None

# =========================
# 初始化
# =========================
random.seed(seed)
np.random.seed(seed)

os.makedirs(output_images_dir, exist_ok=True)

# 定义数据增强（仅image）
transform = A.Compose([
    # A.Rotate(limit=(-90, 90), p=0.5, interpolation=cv.INTER_LINEAR, crop_border=True),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3, interpolation=cv.INTER_LINEAR),
])

def is_background_heavy(patch_rgb: np.ndarray) -> bool:
    """判断patch是否包含大量白色背景。"""
    hsv = cv.cvtColor(patch_rgb, cv.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    white_mask = (v >= white_thr) & (s <= sat_thr)
    white_ratio = float(np.mean(white_mask))
    return white_ratio > max_white_ratio

# 收集输入图像
image_paths = []
for fn in os.listdir(input_dir):
    if fn.lower().endswith((".jpg", ".jpeg")):
        image_paths.append(os.path.join(input_dir, fn))
image_paths.sort()

if len(image_paths) == 0:
    raise FileNotFoundError(f"No JPG images found in: {input_dir}")

# =========================
# 第一步：先生成patch并保存（用临时文件名）
# =========================
attempt_mul = 20  # 最多尝试次数 = 目标patch数 * attempt_mul（防止无限循环）
tmp_paths = []    # 记录已保存patch的路径，方便后面打乱/重命名
global_tmp_idx = 0

for img_idx, img_path in enumerate(image_paths, start=1):
    original_img = Image.open(img_path).convert("RGB")
    width, height = original_img.size

    if width < min_size or height < min_size:
        print(f"[Skip] {os.path.basename(img_path)} too small: {width}x{height}")
        continue

    patch_count = 0
    attempts = 0
    max_attempts = patches_per_image * attempt_mul

    print(f"[{img_idx}/{len(image_paths)}] {os.path.basename(img_path)} ({width}x{height})")

    while patch_count < patches_per_image and attempts < max_attempts:
        attempts += 1

        # 随机选择patch尺寸（不超过图像尺寸）
        patch_size = random.randint(min_size, min(max_size, width, height))

        # 随机选择位置（确保不超出边界）
        x = random.randint(0, width - patch_size)
        y = random.randint(0, height - patch_size)

        # 提取patch
        patch_img = original_img.crop((x, y, x + patch_size, y + patch_size))
        patch_np = np.array(patch_img)

        # 过滤：背景过多则丢弃
        if is_background_heavy(patch_np):
            continue

        # 数据增强
        augmented = transform(image=patch_np)
        aug_img_np = augmented["image"]

        # 临时保存（避免先存内存）
        global_tmp_idx += 1
        tmp_name = f"__tmp_{global_tmp_idx:010d}.jpg"
        tmp_path = os.path.join(output_images_dir, tmp_name)
        Image.fromarray(aug_img_np).save(tmp_path, quality=jpeg_quality)

        tmp_paths.append(tmp_path)
        patch_count += 1

        if patch_count % 20 == 0 or patch_count == patches_per_image:
            print(f"  accepted {patch_count}/{patches_per_image}, attempts={attempts}")

    if patch_count < patches_per_image:
        print(f"  [Warn] only got {patch_count}/{patches_per_image}. "
              f"Try increase max_white_ratio or decrease min_size, or increase attempt_mul.")

print(f"Saved patches (tmp, before shuffle): {len(tmp_paths)}")

# =========================
# 第二步：打乱文件顺序并重新编号（重命名）
# =========================
random.shuffle(tmp_paths)

total = len(tmp_paths)
pad = max(6, len(str(total)))

# 为了避免重命名冲突，采用两段式重命名：
# 1) 先把 __tmp_*.jpg 全部改为 __renaming_*.jpg
# 2) 再改为最终 patch_XXXXXX.jpg
stage_paths = []
for i, old_path in enumerate(tmp_paths, start=1):
    stage_name = f"__renaming_{i:010d}.jpg"
    stage_path = os.path.join(output_images_dir, stage_name)
    os.replace(old_path, stage_path)
    stage_paths.append(stage_path)

for i, stage_path in enumerate(stage_paths, start=1):
    final_name = f"{prefix}_{i:0{pad}d}.jpg"
    final_path = os.path.join(output_images_dir, final_name)
    os.replace(stage_path, final_path)

print(f"All patches saved & renumbered to: {output_images_dir}, total={total}")