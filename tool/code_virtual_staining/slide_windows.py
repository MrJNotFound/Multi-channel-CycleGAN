import os
import math
import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np


def _positions_1d(length: int, window: int, stride: int):
    """生成一维滑窗起点序列，保证覆盖到末端（允许越界，靠pad补齐）。"""
    if length <= 0:
        raise ValueError("length must be > 0")
    if window <= 0:
        raise ValueError("window must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    if length <= window:
        return [0]

    # 常规网格
    n = int(math.ceil((length - window) / stride)) + 1
    return [i * stride for i in range(n)]


def sliding_window_patches(
    img: np.ndarray,
    window_w: int,
    window_h: int,
    overlap_w: int,
    overlap_h: int,
    pad_value_bgr=(255, 255, 255),
):
    """对单张图生成滑窗patch。

    返回: list[ (x, y, patch_bgr) ]，其中 (x,y) 为左上角在原图坐标系下的起点(可使patch越界)。
    """
    if img is None:
        raise ValueError("img is None")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expect BGR image HxWx3, got shape={img.shape}")

    H, W = img.shape[:2]
    if window_w <= 0 or window_h <= 0:
        raise ValueError("window size must be > 0")
    if overlap_w < 0 or overlap_h < 0:
        raise ValueError("overlap must be >= 0")
    if overlap_w >= window_w or overlap_h >= window_h:
        raise ValueError("overlap must be smaller than window size")

    stride_x = window_w - overlap_w
    stride_y = window_h - overlap_h

    xs = _positions_1d(W, window_w, stride_x)
    ys = _positions_1d(H, window_h, stride_y)

    pad_value = np.array(pad_value_bgr, dtype=np.uint8).reshape(1, 1, 3)

    out = []
    for y in ys:
        for x in xs:
            # 需要从原图取的有效区域
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(W, x + window_w)
            y1 = min(H, y + window_h)

            patch = np.tile(pad_value, (window_h, window_w, 1))

            # 放回patch中的位置（只在右/下越界时pad；若x/y为负也能工作，但本脚本不会生成负起点）
            px0 = x0 - x
            py0 = y0 - y
            px1 = px0 + (x1 - x0)
            py1 = py0 + (y1 - y0)

            patch[py0:py1, px0:px1] = img[y0:y1, x0:x1]
            out.append((x, y, patch))

    return out, xs, ys, (stride_x, stride_y)


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # 1. 选择输入图像（可多选）
    img_paths = filedialog.askopenfilenames(
        title="选择待切分图像（可多选）",
        filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif *.webp"),
                   ("所有文件", "*.*")],
    )
    if not img_paths:
        print("未选择图像，退出。")
        root.destroy()
        exit()

    # 2. 选择输出根目录
    parent_dir = filedialog.askdirectory(title="选择输出根目录（每张图像将创建子文件夹）")
    if not parent_dir:
        print("未选择输出目录，退出。")
        root.destroy()
        exit()

    # 3. 参数设置窗口
    scale_var = tk.DoubleVar(value=0.5)
    ww_var = tk.IntVar(value=1024)
    wh_var = tk.IntVar(value=1024)
    ow_var = tk.IntVar(value=512)
    oh_var = tk.IntVar(value=512)
    pad_var = tk.StringVar(value="white")
    ext_var = tk.StringVar(value=".png")

    param_win = tk.Toplevel(root)
    param_win.title("滑窗切分 — 参数设置")
    param_win.resizable(False, False)
    pr = 0

    tk.Label(param_win, text="采样比例 (0.1~2.0)：", font=("", 11)).grid(row=pr, column=0, sticky="e", padx=(15, 2), pady=(15, 3))
    tk.Scale(param_win, from_=0.1, to=2.0, resolution=0.05, orient="horizontal",
             variable=scale_var, length=150).grid(row=pr, column=1, sticky="w")
    pr += 1
    tk.Label(param_win, text="窗口宽度:", font=("", 11)).grid(row=pr, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=ww_var, width=8).grid(row=pr, column=1, sticky="w")
    pr += 1
    tk.Label(param_win, text="窗口高度:", font=("", 11)).grid(row=pr, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=wh_var, width=8).grid(row=pr, column=1, sticky="w")
    pr += 1
    tk.Label(param_win, text="重叠宽度:", font=("", 11)).grid(row=pr, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=ow_var, width=8).grid(row=pr, column=1, sticky="w")
    pr += 1
    tk.Label(param_win, text="重叠高度:", font=("", 11)).grid(row=pr, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=oh_var, width=8).grid(row=pr, column=1, sticky="w")
    pr += 1
    tk.Label(param_win, text="边缘填充色:", font=("", 11)).grid(row=pr, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.OptionMenu(param_win, pad_var, "white", "black").grid(row=pr, column=1, sticky="w")
    pr += 1
    tk.Label(param_win, text="输出格式:", font=("", 11)).grid(row=pr, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.OptionMenu(param_win, ext_var, ".png", ".jpg", ".tif").grid(row=pr, column=1, sticky="w")
    pr += 1

    tk.Button(param_win, text="开始切分", command=param_win.destroy, width=12).grid(row=pr, column=0, columnspan=2, pady=(15, 15))

    param_win.grab_set()
    root.wait_window(param_win)
    root.destroy()

    scale = scale_var.get()
    window_w = ww_var.get()
    window_h = wh_var.get()
    overlap_w = ow_var.get()
    overlap_h = oh_var.get()
    pad_value_bgr = (255, 255, 255) if pad_var.get() == "white" else (0, 0, 0)
    ext = ext_var.get()

    print(f"共选择 {len(img_paths)} 张图像")
    print(f"输出根目录: {parent_dir}")

    total_patches = 0
    for idx, img_path in enumerate(img_paths, 1):
        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[{idx}/{len(img_paths)}] 读取失败，跳过: {img_path}")
                continue

            if scale != 1.0:
                new_w = int(img.shape[1] * scale)
                new_h = int(img.shape[0] * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 按图像名创建子文件夹
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            save_dir = os.path.join(parent_dir, base_name)
            os.makedirs(save_dir, exist_ok=True)

            patches, xs, ys, (sx, sy) = sliding_window_patches(
                img,
                window_w=window_w,
                window_h=window_h,
                overlap_w=overlap_w,
                overlap_h=overlap_h,
                pad_value_bgr=pad_value_bgr,
            )

            for x, y, patch in patches:
                name = f"patch_y{y:06d}_x{x:06d}{ext}"
                ok = cv2.imwrite(os.path.join(save_dir, name), patch)
                if not ok:
                    print(f"  保存失败: {name}")

            print(f"[{idx}/{len(img_paths)}] {base_name}: {len(patches)} patches -> {save_dir}")
            total_patches += len(patches)
        except Exception as e:
            print(f"[{idx}/{len(img_paths)}] 处理失败 [{img_path}]: {e}")

    print(f"\nDone. 共 {len(img_paths)} 张图像, {total_patches} 个 patches.")

