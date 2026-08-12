import os
import cv2
import tkinter as tk
from tkinter import filedialog
from natsort import natsorted


def convert_and_resize_images(input_dir, output_dir, scale=1.0, save_grayscale=False, rename=False, out_fmt="png"):
    """
    将 input_dir 下所有图片（递归）转换并缩放，输出到 output_dir。
    支持 jpg, jpeg, bmp, tiff, gif, webp, png 格式。
    输出文件名按自然排序，命名为 0001、0002、0003……

    参数:
      - input_dir: 输入目录
      - output_dir: 输出目录
      - scale: 缩放比例（浮点数），例如 0.5、1.0、2.0
      - save_grayscale: 是否保存为灰度图（布尔值，默认False，即保存为彩色）
      - rename: 是否重命名为序号
      - out_fmt: 输出图像格式 ("png", "jpg", "bmp", "tiff", "webp")
    """
    img_exts = ('.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp', '.png')

    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    # 收集所有图片文件的完整路径
    img_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(img_exts):
                img_files.append(os.path.join(root, file))

    # 使用natsort进行自然排序
    img_files = natsorted(img_files)

    os.makedirs(output_dir, exist_ok=True)

    # 处理并保存
    for idx, img_path in enumerate(img_files, 1):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"无法读取文件: {img_path}")
            continue
        # 缩放
        if scale != 1.0:
            width = max(1, int(img.shape[1] * scale))
            height = max(1, int(img.shape[0] * scale))
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        # 转为灰度（如果需要）
        if save_grayscale:
            if len(img.shape) == 3:
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 如果rename=False，使用原始文件名（扩展名改为输出格式）；如果rename=True，使用0001、0002……
        if rename:
            file_name = f"{idx:04d}.{out_fmt}"
        else:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            file_name = f"{base_name}.{out_fmt}"
        output_path = os.path.join(output_dir, file_name)
        cv2.imwrite(output_path, img)
        print(f"保存: {output_path}")


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # 1. 选择待处理图像（可多选）
    img_paths = filedialog.askopenfilenames(
        title="选择待处理图像（可多选）",
        filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif *.webp"),
                   ("所有文件", "*.*")],
    )
    if not img_paths:
        print("未选择图像，退出。")
        root.destroy()
        exit()

    # 2. 选择输出文件夹
    output_dir = filedialog.askdirectory(title="选择输出文件夹")
    if not output_dir:
        print("未选择输出文件夹，退出。")
        root.destroy()
        exit()

    # 3. 参数设置窗口
    scale_var = tk.DoubleVar(value=0.5)
    gray_var = tk.BooleanVar(value=False)
    rename_var = tk.BooleanVar(value=False)
    fmt_var = tk.StringVar(value="png")

    opt_win = tk.Toplevel(root)
    opt_win.title("参数设置")
    opt_win.resizable(False, False)

    tk.Label(opt_win, text="缩放比例：", font=("", 11)).grid(row=0, column=0, sticky="w", padx=15, pady=(15, 2))
    scale_frame = tk.Frame(opt_win)
    scale_frame.grid(row=1, column=0, padx=15, pady=(0, 8), sticky="w")
    tk.Entry(scale_frame, textvariable=scale_var, width=8).pack(side="left")
    tk.Label(scale_frame, text="  (如 0.375, 1.0, 2.0)", font=("", 9), fg="gray").pack(side="left")

    tk.Checkbutton(opt_win, text="保存为灰度图", variable=gray_var, font=("", 11)).grid(row=2, column=0, sticky="w", padx=15, pady=2)
    tk.Checkbutton(opt_win, text="重命名为序号 (0001, 0002...)", variable=rename_var, font=("", 11)).grid(row=3, column=0, sticky="w", padx=15, pady=2)

    tk.Label(opt_win, text="输出格式：", font=("", 11)).grid(row=4, column=0, sticky="w", padx=15, pady=(8, 2))
    tk.OptionMenu(opt_win, fmt_var, "png", "jpg", "bmp", "tiff", "webp").grid(row=5, column=0, padx=15, pady=(0, 8))

    tk.Button(opt_win, text="开始处理", command=opt_win.destroy, width=12).grid(row=6, column=0, pady=(10, 15), padx=15)

    opt_win.grab_set()
    root.wait_window(opt_win)
    root.destroy()

    try:
        scale = float(scale_var.get())
        if scale <= 0:
            raise ValueError
    except (ValueError, tk.TclError):
        print(f"缩放比例输入无效: '{scale_var.get()}'，请输入正数（如 0.375, 1.0）")
        exit(1)
    save_grayscale = gray_var.get()
    rename = rename_var.get()
    out_fmt = fmt_var.get()

    print(f"输出目录: {output_dir}")
    print(f"缩放: {scale}x | 灰度: {save_grayscale} | 重命名: {rename} | 格式: {out_fmt}")

    # 按自然排序
    img_paths = natsorted(list(img_paths))
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for idx, img_path in enumerate(img_paths, 1):
        try:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"无法读取文件: {img_path}")
                continue

            # 缩放
            if scale != 1.0:
                width = max(1, int(img.shape[1] * scale))
                height = max(1, int(img.shape[0] * scale))
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

            # 转灰度
            if save_grayscale:
                if len(img.shape) == 3:
                    if img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 输出文件名
            if rename:
                file_name = f"{idx:04d}.{out_fmt}"
            else:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                file_name = f"{base_name}.{out_fmt}"

            output_path = os.path.join(output_dir, file_name)
            ok = cv2.imwrite(output_path, img)
            if not ok:
                print(f"保存失败: {output_path}")
                continue

            print(f"[{idx}/{len(img_paths)}] Saved: {output_path}")
            count += 1
        except Exception as e:
            print(f"处理失败 [{img_path}]: {e}")

    print(f"完成！共处理 {count}/{len(img_paths)} 张图像。")