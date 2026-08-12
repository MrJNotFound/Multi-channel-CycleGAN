"""
基于 Mask 抠图（多虚拟染色共享 Mask）
------------------------------------
输入 1 个 mask 文件夹 + 多个图像文件夹（同一内容的不同虚拟染色），
用相同的 mask 对各图像文件夹分别抠图，输出到以图像文件夹命名的子文件夹中。

用法：
    python tool/extract_from_mask.py
"""

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from natsort import natsorted


def _crop_img_to_mask(img, mask):
    """图像比掩膜大时，从左上角裁掉多余部分以匹配掩膜尺寸。

    掩膜是基准（不动），图像保留 (0,0) → (mask_w, mask_h)。
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

    # 图像裁剪对齐到掩膜尺寸
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


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # ===== Step 1: 选择 Mask 文件夹 =====
    mask_dir = filedialog.askdirectory(title="选择 Mask 文件夹")
    if not mask_dir:
        print("未选择 Mask 文件夹，退出。")
        root.destroy()
        exit()
    print(f"Mask 文件夹: {mask_dir}")

    # ===== Step 2: 选择图像父文件夹（自动递归子文件夹） =====
    img_parent = filedialog.askdirectory(
        title="选择图像父文件夹（其下所有子文件夹将作为不同虚拟染色的图像源）"
    )
    if not img_parent:
        print("未选择图像文件夹，退出。")
        root.destroy()
        exit()

    # 自动收集所有直接子文件夹
    img_parent = os.path.abspath(img_parent)
    img_dirs = []
    for name in sorted(os.listdir(img_parent)):
        full = os.path.join(img_parent, name)
        if os.path.isdir(full):
            img_dirs.append(full)
    if not img_dirs:
        print(f"父文件夹下没有子文件夹，将父文件夹本身作为唯一图像源。")
        img_dirs = [img_parent]
    for i, d in enumerate(img_dirs):
        print(f"图像文件夹 {i + 1}: {os.path.basename(d)}")

    print(f"\n共 {len(img_dirs)} 个图像文件夹。")

    # ===== Step 3: 选择输出根目录 =====
    out_root = filedialog.askdirectory(title="选择输出根目录（将为每个图像文件夹创建子文件夹）")
    if not out_root:
        print("未选择输出目录，退出。")
        root.destroy()
        exit()
    print(f"输出根目录: {out_root}")

    # ===== Step 4: 阈值设置 =====
    thresh_var = tk.IntVar(value=128)
    param_win = tk.Toplevel(root)
    param_win.title("参数设置")
    param_win.resizable(False, False)
    tk.Label(param_win, text="Mask 阈值 (>= 该值视为前景)：", font=("", 11)).pack(
        padx=15, pady=(15, 3))
    tk.Scale(param_win, from_=0, to=255, orient="horizontal",
             variable=thresh_var, length=200).pack(padx=15)
    tk.Label(param_win, textvariable=thresh_var, font=("", 10, "bold")).pack(pady=(0, 5))
    tk.Button(param_win, text="开始抠图", command=param_win.destroy, width=12).pack(pady=(5, 15))
    param_win.grab_set()
    root.wait_window(param_win)
    root.destroy()

    thresh = thresh_var.get()
    print(f"阈值: {thresh}\n")

    # ===== Step 5: 扫描 Mask 文件 =====
    mask_root = os.path.abspath(mask_dir)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    mask_files = []
    for f in os.listdir(mask_root):
        if os.path.splitext(f)[1].lower() in exts:
            mask_files.append(f)
    mask_files = natsorted(mask_files)
    if not mask_files:
        print("Mask 文件夹中没有找到图像文件，退出。")
        exit()
    print(f"Mask 文件: {len(mask_files)} 个")

    # ===== Step 6: 逐图像文件夹处理 =====
    for d_idx, img_dir in enumerate(img_dirs):
        img_root = os.path.abspath(img_dir)
        folder_name = os.path.basename(img_dir.rstrip(os.sep).rstrip("/"))
        out_dir = os.path.join(out_root, folder_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'=' * 50}")
        print(f"图像文件夹 [{d_idx + 1}/{len(img_dirs)}]: {folder_name}")
        print(f"输出到: {out_dir}")
        print(f"{'=' * 50}")

        # 扫描图像文件夹，按文件名自然排序
        img_list = []
        for f in os.listdir(img_root):
            if os.path.splitext(f)[1].lower() in exts:
                img_list.append(f)
        img_list = natsorted(img_list)

        if not img_list:
            print(f"  ⚠ 该文件夹无图像，跳过。")
            continue

        # 按排序一一配对（数量不一致时取较小值并警告）
        n_mask = len(mask_files)
        n_img = len(img_list)
        n = min(n_mask, n_img)
        if n_mask != n_img:
            print(f"  ⚠ Mask {n_mask} 张，图像 {n_img} 张，按排序取前 {n} 对。")

        print(f"  配对: {n} 对，处理中...")

        count = 0
        for i in range(n):
            try:
                img_name = img_list[i]
                mask_name = mask_files[i]
                img_p = os.path.join(img_root, img_name)
                mask_p = os.path.join(mask_root, mask_name)
                # 输出名用图像的原始文件名
                out_name = f"{os.path.splitext(img_name)[0]}_extracted.png"
                out_p = os.path.join(out_dir, out_name)

                fg_px, total = apply_mask(img_p, mask_p, out_p, thresh)
                print(f"  [{i + 1}/{n}] {img_name} ← {mask_name}  "
                      f"(FG: {fg_px}/{total}, {100 * fg_px / total:.1f}%)")
                count += 1
            except Exception as e:
                print(f"  [{i + 1}/{n}] 失败 [{img_name}]: {e}")

        print(f"  完成: {count}/{n} 张 → {out_dir}")

    print(f"\n{'=' * 50}")
    print(f"全部完成！共 {len(img_dirs)} 个图像文件夹，输出至: {out_root}")
    print(f"{'=' * 50}")
