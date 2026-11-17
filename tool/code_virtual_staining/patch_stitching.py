import re
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

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
    """从文件名解析 y/x: ...patch_y000000_x000000.png"""
    m = re.search(r"patch_y(\d+)_x(\d+)", name)
    if not m:
        return None
    y = int(m.group(1))
    x = int(m.group(2))
    return x, y


def _extract_group_prefix(name: str) -> str:
    """提取 patch 文件名中 `patch_y` 之前的前缀作为分组标识。
    例如: 'ROI_01_patch_y000000_x000000.png' → 'ROI_01'
          'patch_y000000_x000000.png' → '' (默认组)
    """
    m = re.match(r"^(.*?)_?patch_y\d+_x\d+", name)
    if m and m.group(1):
        return m.group(1)
    return ""


def _most_common_positive_diff(vals):
    diffs = [b - a for a, b in zip(vals[:-1], vals[1:]) if (b - a) > 0]
    if not diffs:
        return None
    uniq, cnt = np.unique(np.array(diffs, dtype=np.int32), return_counts=True)
    return int(uniq[np.argmax(cnt)])


def stitch_group(items, orig_w, orig_h, weight_mode, ignore_padding, pad_color_bgr, pad_tol):
    """对一组 patch 进行拼接，返回拼接后的大图。"""
    # 读第一张确定patch大小
    first = cv2.imread(str(items[0][2]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read: {items[0][2]}")
    ph, pw = first.shape[:2]

    xs_sorted = sorted({x for x, _, _ in items})
    ys_sorted = sorted({y for _, y, _ in items})
    sx = _most_common_positive_diff(xs_sorted) or pw
    sy = _most_common_positive_diff(ys_sorted) or ph

    max_x = max(x for x, _, _ in items)
    max_y = max(y for _, y, _ in items)
    W = orig_w if orig_w is not None else (max_x + pw)
    H = orig_h if orig_h is not None else (max_y + ph)

    acc = np.zeros((H, W, 3), dtype=np.float32)
    wsum = np.zeros((H, W, 1), dtype=np.float32)

    if weight_mode.lower() == "hann":
        w_patch = _hann2d(ph, pw)
    elif weight_mode.lower() == "gaussian":
        w_patch = _gaussian2d(ph, pw)
    else:
        raise ValueError("weight_mode must be 'hann' or 'gaussian'")
    w_patch = w_patch[..., None]

    pad_color = np.array(pad_color_bgr, dtype=np.uint8).reshape(1, 1, 3)

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
        roi_w = x1 - x0
        roi_h = y1 - y0

        img_roi = img[:roi_h, :roi_w].astype(np.float32)
        w_roi = w_patch[:roi_h, :roi_w]

        if ignore_padding:
            diff = np.max(np.abs(img[:roi_h, :roi_w].astype(np.int16) - pad_color.astype(np.int16)), axis=2)
            mask = (diff <= pad_tol).astype(np.float32)
            w_roi = w_roi * (1.0 - mask[..., None])

        acc[y0:y1, x0:x1] += img_roi * w_roi
        wsum[y0:y1, x0:x1] += w_roi

    wsum_safe = np.maximum(wsum, 1e-6)
    out = acc / wsum_safe
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, pw, ph, sx, sy, W, H


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # 1. 选择 patch 文件夹
    patch_dir = filedialog.askdirectory(title="选择 patch 文件夹")
    if not patch_dir:
        print("未选择文件夹，退出。")
        root.destroy()
        exit()

    # 2. 选择输出文件夹
    out_dir = filedialog.askdirectory(title="选择输出文件夹")
    if not out_dir:
        print("未选择输出文件夹，退出。")
        root.destroy()
        exit()

    # 3. 参数设置
    wmode_var = tk.StringVar(value="hann")
    pad_var = tk.BooleanVar(value=False)
    ow_var = tk.StringVar(value="")
    oh_var = tk.StringVar(value="")

    param_win = tk.Toplevel(root)
    param_win.title("拼接参数")
    param_win.resizable(False, False)
    r = 0

    tk.Label(param_win, text="融合权重窗:", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=(15, 3))
    tk.OptionMenu(param_win, wmode_var, "hann", "gaussian").grid(row=r, column=1, sticky="w")
    r += 1
    tk.Checkbutton(param_win, text="忽略白色 padding 区域", variable=pad_var, font=("", 11)).grid(row=r, column=0, columnspan=2, sticky="w", padx=15, pady=3)
    r += 1
    tk.Label(param_win, text="原始宽度 (留空=自动推断):", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=ow_var, width=8).grid(row=r, column=1, sticky="w")
    r += 1
    tk.Label(param_win, text="原始高度 (留空=自动推断):", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=oh_var, width=8).grid(row=r, column=1, sticky="w")
    r += 1

    tk.Button(param_win, text="开始拼接", command=param_win.destroy, width=12).grid(row=r, column=0, columnspan=2, pady=(15, 15))

    param_win.grab_set()
    root.wait_window(param_win)
    root.destroy()

    weight_mode = wmode_var.get()
    ignore_padding = pad_var.get()
    try:
        orig_w = int(ow_var.get()) if ow_var.get().strip() else None
    except ValueError:
        orig_w = None
    try:
        orig_h = int(oh_var.get()) if oh_var.get().strip() else None
    except ValueError:
        orig_h = None

    # 收集 patch
    patch_root = Path(patch_dir)
    img_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    all_items = []
    for p in patch_root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in img_exts:
            continue
        xy = _parse_xy_from_name(p.name)
        if xy is None:
            continue
        x, y = xy
        prefix = _extract_group_prefix(p.name)
        all_items.append((prefix, x, y, p))

    if not all_items:
        raise RuntimeError("No patch images found with name pattern patch_yXXXXXX_xXXXXXX")

    # 按前缀分组
    groups = {}
    for prefix, x, y, p in all_items:
        groups.setdefault(prefix, []).append((x, y, p))

    print(f"Patch 目录: {patch_root}")
    print(f"输出目录: {out_dir}")
    print(f"识别到 {len(groups)} 组图像:")
    for prefix, items in sorted(groups.items()):
        label = prefix if prefix else "(无前缀)"
        print(f"  [{label}] {len(items)} patches")
    print()

    # 逐组拼接
    total = 0
    for prefix, items in sorted(groups.items()):
        items.sort(key=lambda t: (t[1], t[0]))  # 按 y, x 排序
        label = prefix if prefix else "stitched"

        try:
            out, pw, ph, sx, sy, W, H = stitch_group(
                items, orig_w, orig_h, weight_mode, ignore_padding,
                (255, 255, 255), 3
            )
            out_name = f"{label}.png"
            out_path = str(Path(out_dir) / out_name)
            ok = cv2.imwrite(out_path, out)
            if not ok:
                print(f"  保存失败: {out_name}")
                continue
            print(f"[{label}] Done. Size={W}x{H}, stride=({sx},{sy}), patches={len(items)} -> {out_name}")
            total += 1
        except Exception as e:
            print(f"[{label}] 拼接失败: {e}")

    print(f"\n完成！共拼接 {total}/{len(groups)} 组。")
