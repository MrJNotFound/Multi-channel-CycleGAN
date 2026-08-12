import os
import random
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

Image.MAX_IMAGE_PIXELS = None

try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC  # type: ignore[attr-defined]
    RESAMPLE_NEAREST = Image.Resampling.NEAREST  # type: ignore[attr-defined]
except Exception:
    RESAMPLE_BICUBIC = Image.BICUBIC  # type: ignore[attr-defined]
    RESAMPLE_NEAREST = Image.NEAREST  # type: ignore[attr-defined]

# =========================
# 默认参数（运行时会通过 GUI 覆盖）
# =========================
mode = "color"
enable_secondary_input = False
primary_dir = ""
secondary_dir = ""
mask_dir = ""
output_dir = ""
patch_w = 512
patch_h = 512
total_patches = 2000
recursive = True
allow_smaller = False
output_ext = "jpg"
mask_output_ext = "png"
jpeg_quality = 95
prefix = "patch"
seed: Optional[int] = 114514
max_bg_ratio = 0.7
mask_background_value = 0
mask_background_tolerance = 0
color_bg_mode = "white_black"
white_thr = 230
black_thr = 20

# =========================
# 尺寸不一致策略（按需修改）
# =========================
# 第二输入尺寸与主图不一致时的处理策略：
#   "resize_to_primary" : 缩放第二输入到主图尺寸后再裁剪
#   "skip"              : 直接跳过该配对样本
secondary_size_mismatch = "resize_to_primary"

# mask 尺寸与主图不一致时的处理策略：
#   "resize_to_primary" : 用最近邻插值缩放到主图尺寸
#   "skip"              : 直接跳过该样本
mask_size_mismatch = "resize_to_primary"

# 每张大图内部的最大尝试倍数
# 实际最大尝试次数 = 该图目标 patch 数 * per_image_attempt_mul
# 背景过滤较严格时，可以适当调大
per_image_attempt_mul = 50

# 当第一轮平均分配后 patch 数还不够时，最多进行多少轮"补齐采样"
# 过小可能补不够，过大则在极端情况下会更慢
fill_rounds_limit = 5

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
GRAY_MODES = {"1", "L", "I", "I;16", "F", "LA"}
SourceItem = Tuple[str, Optional[str], Optional[str]]


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

def distribute_counts(n_items: int, total: int) -> List[int]:
    base = total // n_items
    remainder = total % n_items
    counts = [base] * n_items
    for i in range(remainder):
        counts[i] += 1
    return counts


def rand_crop_box(img_w: int, img_h: int, pw: int, ph: int) -> Optional[Tuple[int, int, int, int]]:
    if img_w < pw or img_h < ph:
        return None
    x0 = random.randint(0, img_w - pw)
    y0 = random.randint(0, img_h - ph)
    return x0, y0, x0 + pw, y0 + ph


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resize_to_cover(im: Image.Image, target_w: int, target_h: int, resample: int) -> Image.Image:
    w, h = im.size
    scale = max(float(target_w) / float(w), float(target_h) / float(h))
    new_w = max(target_w, int(round(w * scale)))
    new_h = max(target_h, int(round(h * scale)))
    return im.resize((new_w, new_h), resample=resample)


def open_primary_image(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path)
    except Exception:
        return None


def open_mask_image(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("L")
    except Exception:
        return None


def close_if_needed(im: Optional[Image.Image]) -> None:
    if im is not None:
        try:
            im.close()
        except Exception:
            pass


def prepare_item_images(item: SourceItem) -> Optional[Tuple[Image.Image, Optional[Image.Image], Optional[Image.Image]]]:
    primary_path, secondary_path, mask_path = item

    primary_im = open_primary_image(primary_path)
    if primary_im is None:
        return None

    secondary_im: Optional[Image.Image] = None
    mask_im: Optional[Image.Image] = None

    try:
        if allow_smaller and (primary_im.size[0] < patch_w or primary_im.size[1] < patch_h):
            resized = resize_to_cover(primary_im, patch_w, patch_h, RESAMPLE_BICUBIC)
            primary_im.close()
            primary_im = resized
        elif primary_im.size[0] < patch_w or primary_im.size[1] < patch_h:
            primary_im.close()
            return None

        if secondary_path is not None:
            secondary_im = open_primary_image(secondary_path)
            if secondary_im is None:
                primary_im.close()
                return None

            if secondary_im.size != primary_im.size:
                if secondary_size_mismatch == "resize_to_primary":
                    resized_secondary = secondary_im.resize(primary_im.size, resample=RESAMPLE_BICUBIC)
                    secondary_im.close()
                    secondary_im = resized_secondary
                else:
                    primary_im.close()
                    secondary_im.close()
                    return None

        if mask_path is not None:
            mask_im = open_mask_image(mask_path)
            if mask_im is None:
                primary_im.close()
                close_if_needed(secondary_im)
                return None

            if mask_im.size != primary_im.size:
                if mask_size_mismatch == "resize_to_primary":
                    resized_mask = mask_im.resize(primary_im.size, resample=RESAMPLE_NEAREST)
                    mask_im.close()
                    mask_im = resized_mask
                else:
                    primary_im.close()
                    close_if_needed(secondary_im)
                    mask_im.close()
                    return None

        return primary_im, secondary_im, mask_im
    except Exception:
        close_if_needed(primary_im)
        close_if_needed(secondary_im)
        close_if_needed(mask_im)
        return None


def mask_background_ratio(mask_patch: Image.Image) -> float:
    arr = np.array(mask_patch, dtype=np.int16)
    bg = np.abs(arr - int(mask_background_value)) <= int(mask_background_tolerance)
    return float(np.mean(bg))


def color_background_ratio(image_patch: Image.Image) -> float:
    if image_patch.mode in GRAY_MODES:
        arr = np.array(image_patch)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        white_mask = arr >= white_thr
        black_mask = arr <= black_thr
    else:
        rgb = image_patch.convert("RGB")
        arr = np.array(rgb)
        white_mask = np.all(arr >= white_thr, axis=2)
        black_mask = np.all(arr <= black_thr, axis=2)

    if color_bg_mode == "white":
        bg_mask = white_mask
    elif color_bg_mode == "black":
        bg_mask = black_mask
    else:
        bg_mask = white_mask | black_mask

    return float(np.mean(bg_mask))


def patch_is_accepted(primary_patch: Image.Image, mask_patch: Optional[Image.Image],
                     secondary_patch: Optional[Image.Image] = None) -> bool:
    if mode == "mask":
        if mask_patch is None:
            return False
        return mask_background_ratio(mask_patch) <= max_bg_ratio

    if mode == "color":
        if color_background_ratio(primary_patch) > max_bg_ratio:
            return False
        # 第二输入也做背景校验，确保配对的两个通道都不是背景
        if secondary_patch is not None:
            if color_background_ratio(secondary_patch) > max_bg_ratio:
                return False
        return True

    return True


def save_patch(patch: Image.Image, out_path: str, ext: str) -> bool:
    try:
        ext = ext.lower()
        to_save = patch
        if ext in {"jpg", "jpeg"} and patch.mode not in {"RGB", "L"}:
            to_save = patch.convert("RGB")

        if ext in {"jpg", "jpeg"}:
            to_save.save(out_path, quality=jpeg_quality, subsampling=0, optimize=True)
        else:
            to_save.save(out_path)
        return True
    except Exception:
        return False


def sample_from_item(
    item: SourceItem,
    need: int,
    next_index: int,
    out_primary_dir: str,
    out_secondary_dir: Optional[str],
    out_mask_dir: Optional[str],
    primary_ext: str,
    mask_ext: str,
    pad: int,
) -> Tuple[int, int, int]:
    loaded = prepare_item_images(item)
    if loaded is None:
        return 0, next_index, 0

    primary_im, secondary_im, mask_im = loaded

    produced_here = 0
    rejected_bg = 0
    tries = 0
    max_tries = max(1, need * per_image_attempt_mul)

    try:
        w, h = primary_im.size
        while produced_here < need and tries < max_tries:
            tries += 1
            box = rand_crop_box(w, h, patch_w, patch_h)
            if box is None:
                break

            primary_patch = primary_im.crop(box)
            secondary_patch = secondary_im.crop(box) if secondary_im is not None else None
            mask_patch = mask_im.crop(box) if mask_im is not None else None

            if not patch_is_accepted(primary_patch, mask_patch, secondary_patch):
                rejected_bg += 1
                close_if_needed(primary_patch)
                close_if_needed(secondary_patch)
                close_if_needed(mask_patch)
                continue

            stem = f"{prefix}_{next_index:0{pad}d}"
            primary_out = os.path.join(out_primary_dir, f"{stem}.{primary_ext}")
            secondary_out = os.path.join(out_secondary_dir, f"{stem}.{primary_ext}") if out_secondary_dir else None
            mask_out = os.path.join(out_mask_dir, f"{stem}.{mask_ext}") if out_mask_dir else None

            ok_primary = save_patch(primary_patch, primary_out, primary_ext)
            ok_secondary = True
            ok_mask = True

            if secondary_patch is not None and secondary_out is not None:
                ok_secondary = save_patch(secondary_patch, secondary_out, primary_ext)
            if mask_patch is not None and mask_out is not None:
                ok_mask = save_patch(mask_patch, mask_out, mask_ext)

            close_if_needed(primary_patch)
            close_if_needed(secondary_patch)
            close_if_needed(mask_patch)

            if ok_primary and ok_secondary and ok_mask:
                produced_here += 1
                next_index += 1
            else:
                if ok_primary and os.path.isfile(primary_out):
                    os.remove(primary_out)
                if secondary_out is not None and ok_secondary and os.path.isfile(secondary_out):
                    os.remove(secondary_out)
                if mask_out is not None and ok_mask and os.path.isfile(mask_out):
                    os.remove(mask_out)

        return produced_here, next_index, rejected_bg
    finally:
        close_if_needed(primary_im)
        close_if_needed(secondary_im)
        close_if_needed(mask_im)


def resolve_valid_items() -> List[SourceItem]:
    primary_paths = collect_image_paths(primary_dir, recursive)
    if not primary_paths:
        raise FileNotFoundError("primary_dir 为空，或未找到支持的图片格式。")

    secondary_paths: List[str] = []
    if enable_secondary_input:
        secondary_paths = collect_image_paths(secondary_dir, recursive)
        if not secondary_paths:
            raise FileNotFoundError("enable_secondary_input=True，但 secondary_dir 为空。")

    mask_paths: List[str] = []
    if mode == "mask":
        mask_paths = collect_image_paths(mask_dir, recursive)
        if not mask_paths:
            raise FileNotFoundError("mode='mask'，但 mask_dir 为空。")

    # Match by sorted order: element i in primary pairs with element i in secondary/mask.
    # Truncate to the shortest list, with a warning.
    n = len(primary_paths)
    if enable_secondary_input and len(secondary_paths) != n:
        n_old = n
        n = min(n, len(secondary_paths))
        print(f"[Info] primary has {n_old} images, secondary has {len(secondary_paths)}. "
              f"Truncating to {n} (sorted order).")
        primary_paths = primary_paths[:n]
        secondary_paths = secondary_paths[:n]
    if mode == "mask" and len(mask_paths) != n:
        n_old = n
        n = min(n, len(mask_paths))
        print(f"[Info] primary has {n_old} images, mask has {len(mask_paths)}. "
              f"Truncating to {n} (sorted order).")
        primary_paths = primary_paths[:n]
        secondary_paths = secondary_paths[:n] if enable_secondary_input else []
        mask_paths = mask_paths[:n]

    valid_items: List[SourceItem] = []
    skipped_invalid = 0

    for i in range(n):
        primary_path = primary_paths[i]
        secondary_path = secondary_paths[i] if enable_secondary_input else None
        mask_path = mask_paths[i] if mode == "mask" else None

        item = (primary_path, secondary_path, mask_path)
        loaded = prepare_item_images(item)
        if loaded is None:
            skipped_invalid += 1
            continue

        primary_im, secondary_im, mask_im = loaded
        close_if_needed(primary_im)
        close_if_needed(secondary_im)
        close_if_needed(mask_im)
        valid_items.append(item)

    if skipped_invalid > 0:
        print(f"[Info] skipped (read/size mismatch/smaller than patch): {skipped_invalid}")

    return valid_items


class ProgressTracker:
    def __init__(self, total: int, desc: str = "Generated patches") -> None:
        self.total = max(0, int(total))
        self.desc = desc
        self.current = 0
        self._use_tqdm = tqdm is not None
        self._bar = tqdm(total=self.total, desc=self.desc, unit="patch") if self._use_tqdm else None
        if not self._use_tqdm:
            self._render()

    def _render(self) -> None:
        total = max(1, self.total)
        width = 30
        ratio = min(1.0, self.current / total)
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        msg = f"\r{self.desc}: [{bar}] {self.current}/{self.total}"
        sys.stdout.write(msg)
        sys.stdout.flush()

    def update(self, delta: int) -> None:
        if delta <= 0:
            return
        self.current = min(self.total, self.current + int(delta))
        if self._use_tqdm and self._bar is not None:
            self._bar.update(delta)
        else:
            self._render()

    def log(self, message: str) -> None:
        if self._use_tqdm and self._bar is not None:
            self._bar.write(message)
        else:
            sys.stdout.write("\n" + message + "\n")
            self._render()

    def close(self) -> None:
        if self._use_tqdm and self._bar is not None:
            self._bar.close()
        else:
            self.current = min(self.current, self.total)
            self._render()
            sys.stdout.write("\n")
            sys.stdout.flush()


# =========================
# 主流程
# =========================
def main() -> None:
    if mode not in {"random", "mask", "color"}:
        raise ValueError("mode 只支持: 'random' / 'mask' / 'color'")
    if color_bg_mode not in {"white", "black", "white_black"}:
        raise ValueError("color_bg_mode 只支持: 'white' / 'black' / 'white_black'")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    out_primary_dir = os.path.join(output_dir, "images")
    out_secondary_dir = os.path.join(output_dir, "paired") if enable_secondary_input else None
    out_mask_dir = os.path.join(output_dir, "masks") if mode == "mask" else None

    ensure_dir(out_primary_dir)
    if out_secondary_dir is not None:
        ensure_dir(out_secondary_dir)
    if out_mask_dir is not None:
        ensure_dir(out_mask_dir)

    valid_items = resolve_valid_items()
    if not valid_items:
        raise RuntimeError("未找到可用样本。请检查配对文件、尺寸与过滤设置。")

    primary_ext = output_ext.lstrip(".").lower()
    mask_ext = mask_output_ext.lstrip(".").lower()
    pad = max(6, len(str(total_patches)))

    print(f"Mode: {mode}")
    print(f"Found valid source items: {len(valid_items)}")
    print(f"Output: {output_dir}")
    print(f"Target patches: {total_patches}, patch={patch_w}x{patch_h}")
    print(f"Secondary input: {'enabled' if enable_secondary_input else 'disabled'}")
    if mode == "mask":
        print(
            f"Mask filter: enabled (bg_value={mask_background_value}, tolerance={mask_background_tolerance}, max_bg_ratio={max_bg_ratio})"
        )
    elif mode == "color":
        print(
            f"Color filter: enabled (bg_mode={color_bg_mode}, white_thr={white_thr}, black_thr={black_thr}, max_bg_ratio={max_bg_ratio})"
        )
    else:
        print("Background filter: disabled")

    assigned = distribute_counts(len(valid_items), total_patches)
    produced = 0
    next_index = 1
    progress = ProgressTracker(total_patches, desc="Generated patches")

    try:
        for idx, (item, need) in enumerate(zip(valid_items, assigned), start=1):
            if need <= 0:
                continue

            got, next_index, rejected_bg = sample_from_item(
                item,
                need,
                next_index,
                out_primary_dir,
                out_secondary_dir,
                out_mask_dir,
                primary_ext,
                mask_ext,
                pad,
            )
            produced += got
            progress.update(got)

            if idx % 10 == 0 or idx == len(valid_items):
                extra = f", bg_rejected={rejected_bg}" if mode in {"mask", "color"} else ""
                progress.log(f"Progress: items {idx}/{len(valid_items)}, produced {produced}/{total_patches}{extra}")

            if produced >= total_patches:
                break

        remaining = total_patches - produced
        if remaining > 0:
            progress.log(f"Filling remaining: {remaining}")
            rounds = 0
            items_shuffled = valid_items[:]

            while remaining > 0 and rounds < fill_rounds_limit:
                rounds += 1
                random.shuffle(items_shuffled)

                progress_this_round = 0
                rejected_bg_round = 0

                for item in items_shuffled:
                    if remaining <= 0:
                        break

                    batch_need = min(remaining, max(1, total_patches // max(1, len(items_shuffled) * 10)))
                    got, next_index, rejected_bg = sample_from_item(
                        item,
                        batch_need,
                        next_index,
                        out_primary_dir,
                        out_secondary_dir,
                        out_mask_dir,
                        primary_ext,
                        mask_ext,
                        pad,
                    )
                    produced += got
                    remaining -= got
                    progress_this_round += got
                    rejected_bg_round += rejected_bg
                    progress.update(got)

                msg = f"  round {rounds}: +{progress_this_round}, remaining={remaining}"
                if mode in {"mask", "color"}:
                    msg += f", bg_rejected={rejected_bg_round}"
                progress.log(msg)

                if progress_this_round == 0:
                    break

        if produced < total_patches:
            raise RuntimeError(
                f"仅生成了 {produced}/{total_patches}。可能过滤过严、有效样本过少，或尝试次数不足。"
            )

        progress.log("Done.")
        progress.log(f"  images: {out_primary_dir}")
        if out_secondary_dir is not None:
            progress.log(f"  paired: {out_secondary_dir}")
        if out_mask_dir is not None:
            progress.log(f"  masks : {out_mask_dir}")
    finally:
        progress.close()


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # ===== 1. 模式选择 =====
    mode_var = tk.StringVar(value="color")
    secondary_var = tk.BooleanVar(value=False)
    seed_var = tk.StringVar(value="114514")

    mode_win = tk.Toplevel(root)
    mode_win.title("Patch 生成 — 模式选择")
    mode_win.resizable(False, False)
    row = 0

    tk.Label(mode_win, text="采样模式：", font=("", 11)).grid(row=row, column=0, sticky="w", padx=15, pady=(15, 2))
    row += 1
    for text, val in [("随机 (Random)", "random"), ("Mask 过滤 (Mask)", "mask"), ("颜色过滤 (Color)", "color")]:
        tk.Radiobutton(mode_win, text=text, variable=mode_var, value=val, font=("", 11)).grid(row=row, column=0, sticky="w", padx=30, pady=1)
        row += 1

    tk.Checkbutton(mode_win, text="启用第二输入（配对裁剪）", variable=secondary_var, font=("", 11)).grid(row=row, column=0, sticky="w", padx=15, pady=(8, 2))
    row += 1

    tk.Label(mode_win, text="随机种子 (留空=随机)：", font=("", 11)).grid(row=row, column=0, sticky="w", padx=15, pady=(8, 2))
    row += 1
    tk.Entry(mode_win, textvariable=seed_var, width=20).grid(row=row, column=0, sticky="w", padx=30, pady=(0, 5))
    row += 1

    tk.Button(mode_win, text="下一步", command=mode_win.destroy, width=12).grid(row=row, column=0, pady=(10, 15), padx=15)

    mode_win.grab_set()
    root.wait_window(mode_win)

    mode = mode_var.get()
    enable_secondary_input = secondary_var.get()
    seed_str = seed_var.get().strip()
    seed = int(seed_str) if seed_str else None

    # ===== 2. 选择主输入目录 =====
    primary_dir = filedialog.askdirectory(title="选择主输入图像目录")
    if not primary_dir:
        print("未选择主输入目录，退出。")
        root.destroy()
        exit()

    # ===== 3. 选择 mask 目录（仅 mask 模式） =====
    if mode == "mask":
        mask_dir = filedialog.askdirectory(title="选择 Mask 目录")
        if not mask_dir:
            print("未选择 Mask 目录，退出。")
            root.destroy()
            exit()

    # ===== 4. 选择第二输入目录 =====
    if enable_secondary_input:
        secondary_dir = filedialog.askdirectory(title="选择第二输入图像目录（配对）")
        if not secondary_dir:
            print("未选择第二输入目录，退出。")
            root.destroy()
            exit()

    # ===== 5. 选择输出目录 =====
    output_dir = filedialog.askdirectory(title="选择输出根目录")
    if not output_dir:
        print("未选择输出目录，退出。")
        root.destroy()
        exit()

    # ===== 6. 参数设置窗口 =====
    pw_var = tk.IntVar(value=512)
    ph_var = tk.IntVar(value=512)
    total_var = tk.IntVar(value=2000)
    bg_var = tk.DoubleVar(value=0.7)
    ext_var = tk.StringVar(value="jpg")
    rec_var = tk.BooleanVar(value=True)
    small_var = tk.BooleanVar(value=False)

    param_win = tk.Toplevel(root)
    param_win.title("Patch 生成 — 参数设置")
    param_win.resizable(False, False)
    pr = 0

    tk.Label(param_win, text="Patch 宽度:", font=("", 11)).grid(row=pr, column=0, sticky="e", padx=(15, 2), pady=(15, 3))
    tk.Entry(param_win, textvariable=pw_var, width=8).grid(row=pr, column=1, sticky="w")
    pr += 1
    tk.Label(param_win, text="Patch 高度:", font=("", 11)).grid(row=pr, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=ph_var, width=8).grid(row=pr, column=1, sticky="w")
    pr += 1
    tk.Label(param_win, text="目标 Patch 总数:", font=("", 11)).grid(row=pr, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=total_var, width=8).grid(row=pr, column=1, sticky="w")
    pr += 1
    tk.Label(param_win, text="最大背景占比:", font=("", 11)).grid(row=pr, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Scale(param_win, from_=0.1, to=1.0, resolution=0.05, orient="horizontal",
             variable=bg_var, length=150).grid(row=pr, column=1, sticky="w")
    pr += 1
    tk.Label(param_win, text="输出格式:", font=("", 11)).grid(row=pr, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.OptionMenu(param_win, ext_var, "jpg", "png", "tif").grid(row=pr, column=1, sticky="w")
    pr += 1
    tk.Checkbutton(param_win, text="递归扫描子目录", variable=rec_var, font=("", 11)).grid(row=pr, column=0, columnspan=2, sticky="w", padx=15, pady=3)
    pr += 1
    tk.Checkbutton(param_win, text="允许放大过小图像", variable=small_var, font=("", 11)).grid(row=pr, column=0, columnspan=2, sticky="w", padx=15, pady=3)
    pr += 1

    tk.Button(param_win, text="开始生成", command=param_win.destroy, width=12).grid(row=pr, column=0, columnspan=2, pady=(10, 15))

    param_win.grab_set()
    root.wait_window(param_win)
    root.destroy()

    patch_w = pw_var.get()
    patch_h = ph_var.get()
    total_patches = total_var.get()
    max_bg_ratio = bg_var.get()
    output_ext = ext_var.get()
    recursive = rec_var.get()
    allow_smaller = small_var.get()

    print(f"Mode: {mode} | Secondary: {enable_secondary_input}")
    print(f"Primary: {primary_dir}")
    print(f"Output: {output_dir}")
    print(f"Patch: {patch_w}x{patch_h} | Total: {total_patches} | Max BG: {max_bg_ratio}")

    main()

