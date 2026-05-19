# tool/patch_generate.py
import os
import random
from typing import List, Tuple, Optional

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Pillow resampling compat (Pillow>=9 uses Image.Resampling)
try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC  # type: ignore[attr-defined]
except Exception:
    RESAMPLE_BICUBIC = Image.BICUBIC  # type: ignore[attr-defined]

# =========================
# 参数（按需修改）
# =========================
input_dir = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_HE"    # 输入：大图文件夹
output_dir = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_HE\patches"  # 输出：patch 文件夹

patch_w = 512
patch_h = 512
total_patches = 1000

recursive = True
allow_smaller = False
output_ext = "jpg"
jpeg_quality = 95
prefix = "patch"
seed: Optional[int] = 42

# === 背景过滤（背景过多则丢弃该 patch，重新采样）===
# 原理：把 patch 转 HSV，统计“高亮+低饱和”的像素占比（近似白/背景）
# 这些阈值可参考 tool/patch_generate_2_channel.py
enable_bg_filter = True
white_thr = 150          # V >= white_thr 认为足够亮
sat_thr = 100             # S <= sat_thr 认为饱和度低
max_white_ratio = 0.5    # 白色/背景像素比例超过该值就丢弃

# 每张大图内部最多尝试次数 = 目标数量 * per_image_attempt_mul
per_image_attempt_mul = 50

# 补齐阶段的轮询上限（避免极端情况死循环）
fill_rounds_limit = 5

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =========================
# 工具函数
# =========================
def collect_image_paths(root: str, recursive_scan: bool) -> List[str]:
    paths: List[str] = []
    root = os.path.abspath(root)
    if recursive_scan:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in IMAGE_EXTS:
                    paths.append(os.path.join(dirpath, fn))
    else:
        for fn in os.listdir(root):
            p = os.path.join(root, fn)
            if os.path.isfile(p) and os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                paths.append(p)
    paths.sort()
    return paths


def rand_crop_box(img_w: int, img_h: int, pw: int, ph: int) -> Tuple[int, int, int, int] | None:
    if img_w < pw or img_h < ph:
        return None
    x0 = random.randint(0, img_w - pw)
    y0 = random.randint(0, img_h - ph)
    return (x0, y0, x0 + pw, y0 + ph)


def distribute_counts(n_images: int, total: int) -> List[int]:
    # 平均分配：base + 余数前 r 张多 1
    base = total // n_images
    r = total % n_images
    counts = [base] * n_images
    for i in range(r):
        counts[i] += 1
    return counts


def is_background_heavy(patch_rgb: Image.Image) -> bool:
    """Return True if patch has too much bright/low-saturation pixels (likely background).

    Uses PIL HSV conversion (H,S,V in 0..255).
    """
    hsv = patch_rgb.convert("HSV")
    s = hsv.getchannel("S")
    v = hsv.getchannel("V")

    # Build boolean mask using point() to avoid numpy dependency.
    s_mask = s.point(lambda px: 1 if px <= sat_thr else 0, mode="1")
    v_mask = v.point(lambda px: 1 if px >= white_thr else 0, mode="1")

    # Combine s/v masks by bitwise-and over packed bytes.
    white_mask = Image.frombytes(
        "1",
        s_mask.size,
        bytes(a & b for a, b in zip(s_mask.tobytes(), v_mask.tobytes())),
    )

    # Mode "1" histogram has 2 bins: [count of 0, count of 255]
    hist = white_mask.histogram()
    white_count = hist[0] if len(hist) > 1 else 0
    total = white_mask.size[0] * white_mask.size[1]
    white_ratio = 1 - (white_count / float(total))
    return white_ratio > max_white_ratio


def _open_prepare_image(path: str) -> Image.Image | None:
    try:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        if allow_smaller and (w < patch_w or h < patch_h):
            scale = max(patch_w / w, patch_h / h)
            new_w = max(patch_w, int(round(w * scale)))
            new_h = max(patch_h, int(round(h * scale)))
            im = im.resize((new_w, new_h), resample=RESAMPLE_BICUBIC)
        return im
    except Exception:
        return None


def _save_patch(patch: Image.Image, out_path: str, ext: str) -> bool:
    try:
        if ext in {"jpg", "jpeg"}:
            patch.save(out_path, quality=jpeg_quality, subsampling=0, optimize=True)
        else:
            patch.save(out_path)
        return True
    except Exception:
        return False


# =========================
# 主流程
# =========================
def main() -> None:
    if seed is not None:
        random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    img_paths = collect_image_paths(input_dir, recursive)
    if not img_paths:
        raise FileNotFoundError("输入文件夹为空，或未找到支持的图片格式。")

    # 预过滤：确保至少能裁出 patch（不允许放大时）
    if not allow_smaller:
        valid_paths: List[str] = []
        for p in img_paths:
            try:
                with Image.open(p) as im:
                    w, h = im.size
                if w >= patch_w and h >= patch_h:
                    valid_paths.append(p)
            except Exception:
                continue
    else:
        valid_paths = img_paths

    if not valid_paths:
        raise RuntimeError("未找到可用的大图（可能都尺寸不足或读取失败）。")

    ext = output_ext.lstrip(".").lower()
    pad = max(6, len(str(total_patches)))

    print(f"Found images: {len(valid_paths)}")
    print(f"Output: {output_dir}")
    print(f"Target patches: {total_patches}, patch={patch_w}x{patch_h}, ext={ext}")
    if enable_bg_filter:
        print(f"BG filter: enabled (white_thr={white_thr}, sat_thr={sat_thr}, max_white_ratio={max_white_ratio})")

    # \[1] 平均分配：每张大图一次性截取 assigned 张
    assigned = distribute_counts(len(valid_paths), total_patches)
    produced = 0
    global_idx = 0

    for i, (src, need) in enumerate(zip(valid_paths, assigned), start=1):
        if need <= 0:
            continue

        im = _open_prepare_image(src)
        if im is None:
            continue

        w, h = im.size
        tries = 0
        max_tries = max(1, need * per_image_attempt_mul)
        got = 0
        rejected_bg = 0

        while got < need and tries < max_tries:
            tries += 1
            box = rand_crop_box(w, h, patch_w, patch_h)
            if box is None:
                break
            patch = im.crop(box)

            if enable_bg_filter and is_background_heavy(patch):
                rejected_bg += 1
                continue

            global_idx += 1
            out_name = f"{prefix}_{global_idx:0{pad}d}.{ext}"
            out_path = os.path.join(output_dir, out_name)

            if _save_patch(patch, out_path, ext):
                got += 1
                produced += 1

        im.close()

        if i % 10 == 0 or i == len(valid_paths):
            extra = f", bg_rejected={rejected_bg}" if enable_bg_filter else ""
            print(f"Progress: images {i}/{len(valid_paths)}, produced {produced}/{total_patches}{extra}")

        if produced >= total_patches:
            break

    # \[2] 补齐：如果前面有图片读取失败/保存失败/背景过滤导致不足，轮询补齐
    remaining = total_patches - produced
    if remaining > 0:
        print(f"Filling remaining: {remaining}")
        rounds = 0
        while remaining > 0 and rounds < fill_rounds_limit:
            rounds += 1
            random.shuffle(valid_paths)

            progress_this_round = 0
            bg_rejected_round = 0

            for src in valid_paths:
                if remaining <= 0:
                    break

                batch_need = min(remaining, max(1, total_patches // max(1, len(valid_paths) * 10)))

                im = _open_prepare_image(src)
                if im is None:
                    continue

                w, h = im.size
                tries = 0
                max_tries = max(1, batch_need * per_image_attempt_mul)
                got = 0

                while got < batch_need and tries < max_tries and remaining > 0:
                    tries += 1
                    box = rand_crop_box(w, h, patch_w, patch_h)
                    if box is None:
                        break
                    patch = im.crop(box)

                    if enable_bg_filter and is_background_heavy(patch):
                        bg_rejected_round += 1
                        continue

                    global_idx += 1
                    out_name = f"{prefix}_{global_idx:0{pad}d}.{ext}"
                    out_path = os.path.join(output_dir, out_name)

                    if _save_patch(patch, out_path, ext):
                        got += 1
                        produced += 1
                        remaining -= 1
                        progress_this_round += 1

                im.close()

            msg = f"  round {rounds}: +{progress_this_round}, remaining={remaining}"
            if enable_bg_filter:
                msg += f", bg_rejected={bg_rejected_round}"
            print(msg)

            if progress_this_round == 0:
                break

    if produced < total_patches:
        raise RuntimeError(
            f"仅生成了 {produced}/{total_patches}。过滤可能过严（max_white_ratio 太小），"
            f"或图片背景太多。可尝试调大 max_white_ratio / 调整阈值 / 或调大 per_image_attempt_mul。"
        )

    print("Done.")


if __name__ == "__main__":
    main()
