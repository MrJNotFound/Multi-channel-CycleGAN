"""
16-bit → 8-bit HDR 色调映射转换
------------------------------
使用色调映射公式 Output = Input / (Input + Bias) 将 16-bit TIFF
非线性压缩到 8-bit，保留高光细节。Bias 越小图像越亮。

用法：
    python tool/16_8_bit_hdr.py
"""

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # ===== Step 1: 选择输入文件夹 =====
    input_dir = filedialog.askdirectory(title="选择 16-bit TIFF 输入文件夹")
    if not input_dir:
        print("未选择输入文件夹，退出。")
        root.destroy()
        exit()
    print(f"输入: {input_dir}")

    # ===== Step 2: 选择输出文件夹 =====
    output_dir = filedialog.askdirectory(title="选择输出文件夹")
    if not output_dir:
        print("未选择输出文件夹，退出。")
        root.destroy()
        exit()
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出: {output_dir}")

    # ===== Step 3: 参数设置 =====
    bias_var = tk.DoubleVar(value=5000.0)
    fmt_var = tk.StringVar(value="png")

    param_win = tk.Toplevel(root)
    param_win.title("HDR 转换参数")
    param_win.resizable(False, False)

    tk.Label(param_win, text="Bias 参数 (越小越亮)：", font=("", 11)).pack(
        padx=15, pady=(15, 3))
    tk.Scale(param_win, from_=100, to=50000, resolution=100, orient="horizontal",
             variable=bias_var, length=300).pack(padx=15)
    bias_label = tk.Label(param_win, textvariable=bias_var, font=("", 10, "bold"))
    bias_label.pack(pady=(0, 8))
    tk.Label(param_win, text="(典型值: 2000~10000，16-bit 范围 0~65535)",
             font=("", 8), fg="gray").pack()

    tk.Label(param_win, text="输出格式：", font=("", 11)).pack(pady=(12, 3))
    tk.OptionMenu(param_win, fmt_var, "png", "jpg", "bmp", "tiff", "webp").pack(pady=(0, 8))

    tk.Button(param_win, text="开始转换", command=param_win.destroy,
              width=12).pack(pady=(5, 15))
    param_win.grab_set()
    root.wait_window(param_win)
    root.destroy()

    bias = bias_var.get()
    out_fmt = fmt_var.get()
    print(f"Bias: {bias:.0f}  输出格式: {out_fmt}\n")

    # ===== Step 4: 批量转换 =====
    count = 0
    files = sorted([f for f in os.listdir(input_dir)
                    if f.lower().endswith(('.tif', '.tiff'))])

    if not files:
        print("输入文件夹中没有 .tif/.tiff 文件，退出。")
        exit()

    print(f"共 {len(files)} 个文件，处理中...\n")

    for fn in files:
        in_path = os.path.join(input_dir, fn)
        img16 = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        if img16 is None:
            print(f"  读取失败，跳过: {fn}")
            continue

        img_float = img16.astype(np.float32)
        img_hdr = img_float / (img_float + bias)
        img8 = (img_hdr * 255).clip(0, 255).astype(np.uint8)

        out_fn = os.path.splitext(fn)[0] + f".{out_fmt}"
        out_path = os.path.join(output_dir, out_fn)
        ok = cv2.imwrite(out_path, img8)
        if not ok:
            print(f"  保存失败: {out_fn}")
            continue
        count += 1
        print(f"  [{count}/{len(files)}] {fn} → {out_fn}")

    print(f"\n完成！共转换 {count}/{len(files)} 个文件。")
