import os
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np

# =========================
# 参数（按需修改）
# =========================
# 输入：大图文件夹（images）与对应mask文件夹（masks）
# 约定：mask 与 image 文件名一致（或仅扩展名不同也行），会在 mask_dir 里按“同名优先”查找
image_dir = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_BF\WSI_re"
mask_dir = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_BF\WSI_mask"

# 输出：patch文件夹（会创建 images/ 与 masks/）
output_dir = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_BF\patches_re"

patch_w = 512
patch_h = 512
total_patches = 1000

recursive = True
output_ext = "png"  # 建议用 png 保存 mask，image 也用 png 最省事
prefix = "patch"
seed: Optional[int] = 42

# === mask 背景过滤：patch中 mask==0 的比例不高于该阈值 ===
max_bg_ratio = 0.3

# 每张大图内部最多尝试次数 = 目标数量 * per_image_attempt_mul
per_image_attempt_mul = 80

# 补齐阶段的轮询上限（避免极端情况死循环）
fill_rounds_limit = 5

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =========================
# 工具函数
# =========================
def _collect_paths(root: str, recursive_scan: bool) -> List[str]:
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


def _distribute_counts(n_items: int, total: int) -> List[int]:
    base = total // n_items
    r = total % n_items
    out = [base] * n_items
    for i in range(r):
        out[i] += 1
    return out


def _rand_crop_xy(img_w: int, img_h: int, pw: int, ph: int) -> Tuple[int, int] | None:
    if img_w < pw or img_h < ph:
        return None
    x0 = random.randint(0, img_w - pw)
    y0 = random.randint(0, img_h - ph)
    return x0, y0


def _find_mask_for_image(img_path: str) -> str | None:
    """在 mask_dir 中为 image 找对应 mask。

    优先：同文件名（含扩展名）
    其次：同 stem（不同扩展名）
    """
    base = os.path.basename(img_path)
    direct = os.path.join(mask_dir, base)
    if os.path.isfile(direct):
        return direct

    stem = os.path.splitext(base)[0]
    for ext in IMAGE_EXTS:
        cand = os.path.join(mask_dir, stem + ext)
        if os.path.isfile(cand):
            return cand
    return None


def _read_image(path: str) -> np.ndarray | None:
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    return im


def _read_mask(path: str) -> np.ndarray | None:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return m


def _bg_ratio(mask_patch: np.ndarray) -> float:
    # mask==0 认为背景。若你的mask是0/255二值或0/1都适用。
    return float(np.mean(mask_patch == 0))


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def main() -> None:
    if seed is not None:
        random.seed(seed)

    out_img_dir = os.path.join(output_dir, "images")
    out_mask_dir = os.path.join(output_dir, "masks")
    _ensure_dir(out_img_dir)
    _ensure_dir(out_mask_dir)

    img_paths = _collect_paths(image_dir, recursive)
    if not img_paths:
        raise FileNotFoundError("image_dir 为空，或未找到支持的图片格式。")

    # 找到同时有mask、且尺寸足够的样本
    pairs: List[Tuple[str, str]] = []
    for p in img_paths:
        mp = _find_mask_for_image(p)
        if mp is None:
            continue

        im = _read_image(p)
        m = _read_mask(mp)
        if im is None or m is None:
            continue

        if im.shape[0] != m.shape[0] or im.shape[1] != m.shape[1]:
            continue

        if im.shape[1] < patch_w or im.shape[0] < patch_h:
            continue

        pairs.append((p, mp))

    if not pairs:
        raise RuntimeError("未找到可用的 image/mask 对（可能mask缺失、尺寸不一致或图太小）。")

    ext = output_ext.lstrip(".").lower()
    pad = max(6, len(str(total_patches)))

    print(f"Found pairs: {len(pairs)}")
    print(f"Output: {output_dir}")
    print(f"Target patches: {total_patches}, patch={patch_w}x{patch_h}, ext={ext}")
    print(f"Mask BG filter: max_bg_ratio={max_bg_ratio} (mask==0 as background)")

    assigned = _distribute_counts(len(pairs), total_patches)

    produced = 0
    global_idx = 0

    # [1] 平均分配：每对大图截取 assigned 张
    for i, ((img_p, mask_p), need) in enumerate(zip(pairs, assigned), start=1):
        if need <= 0:
            continue

        im = _read_image(img_p)
        m = _read_mask(mask_p)
        if im is None or m is None:
            continue

        h, w = m.shape
        tries = 0
        max_tries = max(1, need * per_image_attempt_mul)
        got = 0
        rejected_bg = 0

        while got < need and tries < max_tries:
            tries += 1
            xy = _rand_crop_xy(w, h, patch_w, patch_h)
            if xy is None:
                break
            x0, y0 = xy

            img_patch = im[y0 : y0 + patch_h, x0 : x0 + patch_w]
            mask_patch = m[y0 : y0 + patch_h, x0 : x0 + patch_w]

            if _bg_ratio(mask_patch) > max_bg_ratio:
                rejected_bg += 1
                continue

            global_idx += 1
            out_name = f"{prefix}_{global_idx:0{pad}d}.{ext}"

            out_img_path = os.path.join(out_img_dir, out_name)
            out_mask_path = os.path.join(out_mask_dir, out_name)

            ok1 = cv2.imwrite(out_img_path, img_patch)
            ok2 = cv2.imwrite(out_mask_path, mask_patch)
            if ok1 and ok2:
                got += 1
                produced += 1

        if i % 10 == 0 or i == len(pairs):
            print(
                f"Progress: pairs {i}/{len(pairs)}, produced {produced}/{total_patches}, bg_rejected={rejected_bg}"
            )

        if produced >= total_patches:
            break

    # [2] 补齐：若不足则随机轮询补齐
    remaining = total_patches - produced
    if remaining > 0:
        print(f"Filling remaining: {remaining}")
        rounds = 0

        pairs_shuf = pairs[:]
        while remaining > 0 and rounds < fill_rounds_limit:
            rounds += 1
            random.shuffle(pairs_shuf)

            progress_this_round = 0
            bg_rejected_round = 0

            for img_p, mask_p in pairs_shuf:
                if remaining <= 0:
                    break

                batch_need = min(remaining, max(1, total_patches // max(1, len(pairs_shuf) * 10)))

                im = _read_image(img_p)
                m = _read_mask(mask_p)
                if im is None or m is None:
                    continue

                h, w = m.shape
                tries = 0
                max_tries = max(1, batch_need * per_image_attempt_mul)
                got = 0

                while got < batch_need and tries < max_tries and remaining > 0:
                    tries += 1
                    xy = _rand_crop_xy(w, h, patch_w, patch_h)
                    if xy is None:
                        break
                    x0, y0 = xy

                    img_patch = im[y0 : y0 + patch_h, x0 : x0 + patch_w]
                    mask_patch = m[y0 : y0 + patch_h, x0 : x0 + patch_w]

                    if _bg_ratio(mask_patch) > max_bg_ratio:
                        bg_rejected_round += 1
                        continue

                    global_idx += 1
                    out_name = f"{prefix}_{global_idx:0{pad}d}.{ext}"

                    out_img_path = os.path.join(out_img_dir, out_name)
                    out_mask_path = os.path.join(out_mask_dir, out_name)

                    ok1 = cv2.imwrite(out_img_path, img_patch)
                    ok2 = cv2.imwrite(out_mask_path, mask_patch)
                    if ok1 and ok2:
                        got += 1
                        produced += 1
                        remaining -= 1
                        progress_this_round += 1

            print(f"  round {rounds}: +{progress_this_round}, remaining={remaining}, bg_rejected={bg_rejected_round}")
            if progress_this_round == 0:
                break

    if produced < total_patches:
        raise RuntimeError(
            f"仅生成了 {produced}/{total_patches}。可能 max_bg_ratio 太小 / mask背景太多 / 尝试次数不足。"
        )

    print("Done.")


if __name__ == "__main__":
    # 不使用 argparse。直接修改顶部参数后运行此脚本即可。
    main()
