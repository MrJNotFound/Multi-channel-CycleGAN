import cv2
import tkinter as tk
from tkinter import filedialog

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # 1. 选择输入图像
    img_path = filedialog.askopenfilename(
        title="选择图像",
        filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif *.webp"),
                   ("所有文件", "*.*")],
    )
    if not img_path:
        print("未选择图像，退出。")
        root.destroy()
        exit()

    # 2. 选择输出文件
    out_path = filedialog.asksaveasfilename(
        title="保存带比例尺的图像",
        defaultextension=".png",
        filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp"), ("TIF", "*.tif")],
    )
    if not out_path:
        print("未选择输出路径，退出。")
        root.destroy()
        exit()

    # 3. 参数设置窗口
    pixel_size_var = tk.DoubleVar(value=1.0)
    unit_var = tk.StringVar(value="um")
    bar_len_var = tk.DoubleVar(value=0.0)  # 0 = auto
    pos_var = tk.StringVar(value="bottom-right")
    color_var = tk.StringVar(value="white")
    font_scale_var = tk.DoubleVar(value=2.0)
    bar_width_var = tk.IntVar(value=20)

    param_win = tk.Toplevel(root)
    param_win.title("比例尺参数")
    param_win.resizable(False, False)
    r = 0

    tk.Label(param_win, text="每像素物理尺寸:", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=(15, 3))
    tk.Entry(param_win, textvariable=pixel_size_var, width=10).grid(row=r, column=1, sticky="w")
    tk.OptionMenu(param_win, unit_var, "nm", "um", "mm", "m").grid(row=r, column=2, sticky="w", padx=(2, 15))
    r += 1

    tk.Label(param_win, text="比例尺长度 (0=自动):", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=bar_len_var, width=10).grid(row=r, column=1, sticky="w")
    tk.Label(param_win, text="(与像素单位一致)", font=("", 9)).grid(row=r, column=2, sticky="w", padx=(2, 15))
    r += 1

    tk.Label(param_win, text="比例尺位置:", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.OptionMenu(param_win, pos_var, "bottom-right", "bottom-left", "top-right", "top-left").grid(row=r, column=1, sticky="w")
    r += 1

    tk.Label(param_win, text="比例尺颜色:", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.OptionMenu(param_win, color_var, "white", "black").grid(row=r, column=1, sticky="w")
    r += 1

    tk.Label(param_win, text="字体大小:", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Scale(param_win, from_=0.5, to=5.0, resolution=0.1, orient="horizontal",
             variable=font_scale_var, length=120).grid(row=r, column=1, sticky="w")
    r += 1

    tk.Label(param_win, text="比例尺粗细 (px):", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.Entry(param_win, textvariable=bar_width_var, width=10).grid(row=r, column=1, sticky="w")
    r += 1

    tk.Button(param_win, text="生成", command=param_win.destroy, width=12).grid(row=r, column=0, columnspan=3, pady=(15, 15))

    param_win.grab_set()
    root.wait_window(param_win)
    root.destroy()

    # 获取参数
    pixel_size = pixel_size_var.get()
    unit = unit_var.get()
    bar_len_user = bar_len_var.get()
    position = pos_var.get()
    color_name = color_var.get()
    font_scale = font_scale_var.get()
    bar_width_px = bar_width_var.get()

    if pixel_size <= 0:
        raise ValueError("每像素物理尺寸必须 > 0")

    # 单位换算到米（用于自动选择合适长度）
    unit_to_m = {"nm": 1e-9, "um": 1e-6, "mm": 1e-3, "m": 1.0}
    pixel_size_m = pixel_size * unit_to_m[unit]

    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"读取失败: {img_path}")
    H, W = img.shape[:2]

    # 自动计算比例尺长度（取图像宽度 20%~40% 的合适整数）
    if bar_len_user <= 0:
        img_width_physical = W * pixel_size  # 图像物理宽度（用户单位）
        candidates = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
        # 比例尺目标：图像宽度的 25%
        target = img_width_physical * 0.25
        bar_len = min(candidates, key=lambda c: abs(c - target))
        # 对于小于1的情况
        if bar_len < 1:
            small_candidates = [0.1, 0.2, 0.5, 1]
            bar_len = min(small_candidates, key=lambda c: abs(c - target))
    else:
        bar_len = bar_len_user

    # 计算比例尺像素长度
    bar_px = int(round(bar_len / pixel_size))

    # 单位标签（自动调整前缀）
    if bar_len >= 1000 and unit in ("nm", "um", "mm"):
        display_val = bar_len / 1000
        if unit == "nm":
            display_unit = "um"
        elif unit == "um":
            display_unit = "mm"
        else:
            display_unit = "cm"
    else:
        display_val = bar_len
        display_unit = unit

    # 格式化数值（去掉多余的 0）
    if display_val == int(display_val):
        label = f"{int(display_val)} {display_unit}"
    else:
        label = f"{display_val:.1f} {display_unit}"

    print(f"图像尺寸: {W}x{H} px")
    print(f"每像素: {pixel_size} {unit}")
    print(f"比例尺: {bar_len} {unit} = {bar_px} px  |  标签: {label}")

    # 比例尺位置
    margin = 30
    if "bottom" in position:
        y0 = H - margin - bar_width_px - 10
    else:
        y0 = margin + 10

    if "right" in position:
        x0 = W - margin - bar_px
    else:
        x0 = margin

    # 颜色
    color = (255, 255, 255) if color_name == "white" else (0, 0, 0)

    # 绘制比例尺
    # 横线
    cv2.rectangle(img, (x0, y0), (x0 + bar_px, y0 + bar_width_px), color, -1)

    # 竖线端点
    end_h = bar_width_px + 8
    cv2.rectangle(img, (x0, y0 - 4), (x0 + bar_width_px // 3, y0 + end_h), color, -1)
    cv2.rectangle(img, (x0 + bar_px - bar_width_px // 3, y0 - 4), (x0 + bar_px, y0 + end_h), color, -1)

    # 文字：在比例尺上方
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(2, bar_width_px // 5))
    text_w, text_h = text_size[0]
    text_x = x0 + (bar_px - text_w) // 2
    text_y = y0 - 10
    if text_y - text_h < 0:
        text_y = y0 + bar_width_px + text_h + 10  # 如果上方不够，放下方
    cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, max(2, bar_width_px // 5))

    # 保存
    ok = cv2.imwrite(out_path, img)
    if not ok:
        raise RuntimeError(f"保存失败: {out_path}")

    print(f"Saved: {out_path}")
