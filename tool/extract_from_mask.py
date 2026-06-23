"""
基于 Mask 抠图：图像居中裁剪到掩膜尺寸 → 保留前景、背景置白。

用法：
    python tool/extract_from_mask.py

尺寸不一致时：掩膜不动（基准），图像居中裁剪掉多余部分，不缩放。
"""

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from natsort import natsorted


def _crop_img_to_mask(img, mask):
    """图像比掩膜大时，从原点方向裁剪图像以匹配掩膜尺寸。

    滑动窗口在 x+ / y+ 方向扩张，多余部分在右侧和底部，
    因此保留左上角 (0,0) → (mask_w, mask_h)。
    掩膜不动（是基准）。
    """
    h_img, w_img = img.shape[:2]
    h_msk, w_msk = mask.shape[:2]
    if h_img == h_msk and w_img == w_msk:
        return img
    if h_img < h_msk or w_img < w_msk:
        raise RuntimeError(f"图像 ({w_img}x{h_img}) 小于掩膜 ({w_msk}x{h_msk})，无法裁剪")
    return img[:h_msk, :w_msk]


def apply_mask(img_path, mask_path, out_path, thresh=128):
    """图像居中裁剪到掩膜尺寸 → 应用掩膜保留前景。"""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"读取图像失败: {img_path}")

    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"读取 mask 失败: {mask_path}")

    # 图像居中裁剪到掩膜尺寸（掩膜不动）
    img = _crop_img_to_mask(img, mask)

    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask_gray = mask[:, :, 3]
        else:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask

    fg = (mask_gray.astype(np.uint8) >= thresh)
    out = np.full_like(img, 255)
    out[fg] = img[fg]

    ok = cv2.imwrite(out_path, out)
    if not ok:
        raise RuntimeError(f"保存失败: {out_path}")
    return int(fg.sum()), fg.size


# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # 0. 选择模式
    mode_var = tk.StringVar(value="多对多")
    mode_win = tk.Toplevel(root)
    mode_win.title("选择模式")
    mode_win.resizable(False, False)
    tk.Label(mode_win, text="请选择处理模式：", font=("", 12)).pack(padx=20, pady=(15, 5))
    for text, val in [
        ("一对一：一张图像 + 一张 Mask", "一对一"),
        ("多对多：多选图像与 Mask，按排序一一对应", "多对多"),
    ]:
        tk.Radiobutton(mode_win, text=text, variable=mode_var, value=val,
                       font=("", 11)).pack(anchor="w", padx=20, pady=3)
    tk.Button(mode_win, text="下一步", command=mode_win.destroy, width=12).pack(pady=(10, 15))
    mode_win.grab_set()
    root.wait_window(mode_win)

    mode = mode_var.get()

    # 1. 选择图像和 Mask
    img_paths = []
    mask_paths = []
    out_dir = ""
    out_path_single = None

    if mode == "一对一":
        ip = filedialog.askopenfilename(
            title="选择待抠图图像",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("所有文件", "*.*")],
        )
        if not ip: print("未选择图像，退出。"); root.destroy(); exit()
        mp = filedialog.askopenfilename(
            title="选择 Mask 图像",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("所有文件", "*.*")],
        )
        if not mp: print("未选择 Mask，退出。"); root.destroy(); exit()
        op = filedialog.asksaveasfilename(
            title="保存抠图结果",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
        )
        if not op: print("未选择输出路径，退出。"); root.destroy(); exit()
        img_paths = [ip]
        mask_paths = [mp]
        out_dir = os.path.dirname(op)
        out_path_single = op
    else:
        raw_imgs = filedialog.askopenfilenames(
            title="选择待抠图图像（可多选）",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("所有文件", "*.*")],
        )
        if not raw_imgs: print("未选择图像，退出。"); root.destroy(); exit()
        raw_masks = filedialog.askopenfilenames(
            title="选择 Mask 图像（可多选）",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("所有文件", "*.*")],
        )
        if not raw_masks: print("未选择 Mask，退出。"); root.destroy(); exit()
        out_dir = filedialog.askdirectory(title="选择输出文件夹")
        if not out_dir: print("未选择输出文件夹，退出。"); root.destroy(); exit()

        img_paths = natsorted(list(raw_imgs))
        mask_paths = natsorted(list(raw_masks))

        n_img = len(img_paths)
        n_mask = len(mask_paths)
        n = min(n_img, n_mask)
        if n_img != n_mask:
            messagebox.showwarning("数量不一致",
                f"图像 {n_img} 张，Mask {n_mask} 张。\n按排序取前 {n} 对处理，多余的将被忽略。")
        img_paths = img_paths[:n]
        mask_paths = mask_paths[:n]

    # 2. 阈值设置
    thresh_var = tk.IntVar(value=128)
    param_win = tk.Toplevel(root)
    param_win.title("参数设置")
    param_win.resizable(False, False)
    tk.Label(param_win, text="Mask 阈值 (>= 该值视为前景)：", font=("", 11)).pack(padx=15, pady=(15, 3))
    tk.Scale(param_win, from_=0, to=255, orient="horizontal", variable=thresh_var, length=200).pack(padx=15)
    tk.Label(param_win, textvariable=thresh_var, font=("", 10, "bold")).pack(pady=(0, 5))
    tk.Button(param_win, text="开始抠图", command=param_win.destroy, width=12).pack(pady=(5, 15))
    param_win.grab_set()
    root.wait_window(param_win)
    root.destroy()

    thresh = thresh_var.get()

    print(f"模式: {mode}")
    print(f"阈值: {thresh}")
    print(f"输出目录: {out_dir}")
    if mode == "一对一":
        print(f"图像: {img_paths[0]}")
        print(f"Mask: {mask_paths[0]}")
    else:
        print(f"配对数量: {len(img_paths)} 对")
    print()

    # 3. 处理
    count = 0
    for i, (img_p, mask_p) in enumerate(zip(img_paths, mask_paths), 1):
        try:
            base = os.path.splitext(os.path.basename(img_p))[0]
            out_p = out_path_single if mode == "一对一" else os.path.join(out_dir, f"{base}_extracted.png")

            fg_px, total = apply_mask(img_p, mask_p, out_p, thresh)
            print(f"[{i}/{len(img_paths)}] {os.path.basename(img_p)} -> {os.path.basename(out_p)}  "
                  f"(FG: {fg_px}/{total}, {100*fg_px/total:.1f}%)")
            count += 1
        except Exception as e:
            print(f"[{i}/{len(img_paths)}] 失败 [{img_p}]: {e}")

    print(f"\n完成！共处理 {count}/{len(img_paths)} 张。")
