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
input_dir_ch1 = r"C:\Users\30927\Desktop\mouse_kidney\kidney_DAPI\BF"        # 输入通道1：大图文件夹
input_dir_ch2 = r"C:\Users\30927\Desktop\mouse_kidney\kidney_DAPI\DAPI"     # 输入通道2：大图文件夹（与通道1一一对应）

output_dir_ch1 = r"C:\Users\30927\Desktop\mouse_kidney\patch_BF_DAPI\BF"       # 输出通道1 patch
output_dir_ch2 = r"C:\Users\30927\Desktop\mouse_kidney\patch_BF_DAPI\DAPI"    # 输出通道2 patch

patches_per_pair = 1000   # 每对大图生成多少对patch（通过过滤后达到该数量）
min_size = 512
max_size = 1024

# 背景过滤（用通道1做过滤；如果你希望两通道都过滤，见 is_pair_background_heavy）
white_thr = 230
sat_thr = 30
max_white_ratio = 0.6

# 其它
jpeg_quality = 95
seed = 42
prefix = "patch"

# =========================
# 初始化
# =========================
random.seed(seed)
np.random.seed(seed)

os.makedirs(output_dir_ch1, exist_ok=True)
os.makedirs(output_dir_ch2, exist_ok=True)

# 同步增强：两通道必须用同一个随机参数
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3, interpolation=cv.INTER_LINEAR),
], additional_targets={"image2": "image"})

def is_background_heavy(patch_rgb: np.ndarray) -> bool:
    hsv = cv.cvtColor(patch_rgb, cv.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    white_mask = (v >= white_thr) & (s <= sat_thr)
    return float(np.mean(white_mask)) > max_white_ratio

def collect_image_paths(folder):
    paths = []
    for fn in os.listdir(folder):
        if fn.lower().endswith((".jpg", ".jpeg")):
            paths.append(os.path.join(folder, fn))
    return sorted(paths)

# =========================
# 配对（按文件名）
# =========================
ch1_paths = collect_image_paths(input_dir_ch1)
ch2_paths = collect_image_paths(input_dir_ch2)
if len(ch1_paths) == 0 or len(ch2_paths) == 0:
    raise FileNotFoundError("输入文件夹为空，或没有jpg/jpeg文件。")

ch1_map = {os.path.basename(p): p for p in ch1_paths}
ch2_map = {os.path.basename(p): p for p in ch2_paths}
common_names = sorted(set(ch1_map.keys()) & set(ch2_map.keys()))
if len(common_names) == 0:
    raise RuntimeError("两个文件夹没有同名文件，无法一一对应配对。")

paired_paths = [(ch1_map[name], ch2_map[name], name) for name in common_names]
print(f"Found paired images: {len(paired_paths)}")

# =========================
# 生成patch并保存（临时名）
# =========================
attempt_mul = 20
tmp_pairs = []
global_tmp_idx = 0

for pair_idx, (p1, p2, name) in enumerate(paired_paths, start=1):
    # 通道1作为基准尺寸
    img1 = Image.open(p1).convert("RGB")
    w1, h1 = img1.size

    img2 = Image.open(p2).convert("RGB")
    w2, h2 = img2.size

    if w1 < min_size or h1 < min_size:
        print(f"[Skip] too small: {name}  ch1={w1}x{h1}")
        continue

    # 尺寸不同：把通道2缩放到通道1的尺寸（关键修改点）
    if (w1 != w2) or (h1 != h2):
        # 用双线性插值即可；如果你更想保细节，可改成 Image.BICUBIC
        img2 = img2.resize((w1, h1), resample=Image.BILINEAR)
        print(f"[Info] resized ch2 for {name}: {w2}x{h2} -> {w1}x{h1}")

    width, height = w1, h1

    patch_count = 0
    attempts = 0
    max_attempts = patches_per_pair * attempt_mul

    print(f"[{pair_idx}/{len(paired_paths)}] {name} (base={width}x{height})")

    while patch_count < patches_per_pair and attempts < max_attempts:
        attempts += 1

        patch_size = random.randint(min_size, min(max_size, width, height))
        x = random.randint(0, width - patch_size)
        y = random.randint(0, height - patch_size)

        patch1 = np.array(img1.crop((x, y, x + patch_size, y + patch_size)))
        patch2 = np.array(img2.crop((x, y, x + patch_size, y + patch_size)))

        # 背景过滤：默认仅用通道1
        if is_background_heavy(patch1):
            continue

        augmented = transform(image=patch1, image2=patch2)
        aug1 = augmented["image"]
        aug2 = augmented["image2"]

        global_tmp_idx += 1
        tmp_name = f"__tmp_{global_tmp_idx:010d}.jpg"

        tmp_path1 = os.path.join(output_dir_ch1, tmp_name)
        tmp_path2 = os.path.join(output_dir_ch2, tmp_name)

        Image.fromarray(aug1).save(tmp_path1, quality=jpeg_quality)
        Image.fromarray(aug2).save(tmp_path2, quality=jpeg_quality)

        tmp_pairs.append((tmp_path1, tmp_path2))
        patch_count += 1

        if patch_count % 20 == 0 or patch_count == patches_per_pair:
            print(f"  accepted {patch_count}/{patches_per_pair}, attempts={attempts}")

    if patch_count < patches_per_pair:
        print(f"  [Warn] only got {patch_count}/{patches_per_pair}. "
              f"Try increase max_white_ratio or increase attempt_mul.")

print(f"Saved patch pairs (tmp, before shuffle): {len(tmp_pairs)}")

# =========================
# 打乱顺序并重新编号（两通道保持一致）
# =========================
random.shuffle(tmp_pairs)

total = len(tmp_pairs)
pad = max(6, len(str(total)))

stage_pairs = []
for i, (old1, old2) in enumerate(tmp_pairs, start=1):
    stage_name = f"__renaming_{i:010d}.jpg"
    stage1 = os.path.join(output_dir_ch1, stage_name)
    stage2 = os.path.join(output_dir_ch2, stage_name)
    os.replace(old1, stage1)
    os.replace(old2, stage2)
    stage_pairs.append((stage1, stage2))

for i, (stage1, stage2) in enumerate(stage_pairs, start=1):
    final_name = f"{prefix}_{i:0{pad}d}.jpg"
    final1 = os.path.join(output_dir_ch1, final_name)
    final2 = os.path.join(output_dir_ch2, final_name)
    os.replace(stage1, final1)
    os.replace(stage2, final2)

print(f"Done. Saved paired patches:")
print(f"  ch1: {output_dir_ch1}")
print(f"  ch2: {output_dir_ch2}")
print(f"  total pairs: {total}")