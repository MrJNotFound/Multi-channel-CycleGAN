"""
图像颜色反转（HSV V 通道反转 / 灰度反转）
----------------------------------------
彩色图像在 HSV 空间反转 V（亮度）通道，避免直接 BGR 反转导致的色偏；
灰度图像直接 255 - pixel。

用法：
    python tool/img_invert.py
"""

import os
import cv2
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def invert_img(bgr_img):
    # 彩色图像: 转换到 HSV，反转 V 通道，再转换回 BGR
    if len(bgr_img.shape) == 3 and bgr_img.shape[2] == 3:
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.subtract(255, v)  # V = 255 - V
        hsv_inv = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv_inv, cv2.COLOR_HSV2BGR)
    # 灰度图像: 直接反转像素值
    else:
        return cv2.subtract(255, bgr_img)


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # ===== 选择输入文件夹 =====
    input_dir = filedialog.askdirectory(title="选择输入文件夹（将递归扫描子文件夹）")
    if not input_dir:
        print("未选择输入文件夹，退出。")
        root.destroy()
        exit()

    # ===== 选择输出文件夹 =====
    output_dir = filedialog.askdirectory(title="选择输出文件夹")
    if not output_dir:
        print("未选择输出文件夹，退出。")
        root.destroy()
        exit()

    root.destroy()

    in_dir = Path(input_dir).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"输入: {in_dir}")
    print(f"输出: {out_dir}\n")

    count = 0
    for src in in_dir.rglob("*"):
        if not src.is_file() or src.suffix.lower() not in SUPPORTED_EXTS:
            continue

        rel = src.relative_to(in_dir)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        out = invert_img(img)
        cv2.imwrite(str(dst), out)
        count += 1
        print(f"  [{count}] {rel}")

    print(f"\n完成！共处理 {count} 个文件。")
