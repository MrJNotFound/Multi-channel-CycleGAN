"""
将 BF / AF 图像整理为 DualChannelDataset 所需的目录结构。

两种输入模式：
  - 子文件夹模式：选择 BF/AF 根目录，自动按子文件夹配对
  - 图像多选模式：直接多选 BF 和 AF 图像，按排序一一配对

输出结构:
    output_root/
    ├── testA_BF/ (或 trainA_BF)
    └── testA_AF/ (或 trainA_AF)
"""

import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from natsort import natsorted

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def collect_subdir_structure(root_dir: str):
    """收集 root_dir 下子文件夹内的图像文件（按名称排序）。
    返回: [(subfolder_name, [filename, ...]), ...]
    """
    result = []
    for sub in natsorted(os.listdir(root_dir)):
        sub_path = os.path.join(root_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        files = []
        for f in natsorted(os.listdir(sub_path)):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                files.append(f)
        if files:
            result.append((sub, files))
    return result


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # ===== Step 0: 选择输入模式 =====
    mode_var = tk.StringVar(value="子文件夹")
    mode_win = tk.Toplevel(root)
    mode_win.title("选择输入模式")
    mode_win.resizable(False, False)
    tk.Label(mode_win, text="请选择输入模式：", font=("", 12)).pack(padx=20, pady=(15, 5))
    for text, val in [
        ("子文件夹模式：选择 BF/AF 根目录，自动按子文件夹配对", "子文件夹"),
        ("图像多选模式：直接多选 BF 和 AF 图像，按排序配对", "图像多选"),
    ]:
        tk.Radiobutton(mode_win, text=text, variable=mode_var, value=val,
                       font=("", 11)).pack(anchor="w", padx=20, pady=3)
    tk.Button(mode_win, text="下一步", command=mode_win.destroy,
              width=12).pack(pady=(10, 15))
    mode_win.grab_set()
    root.wait_window(mode_win)

    input_mode = mode_var.get()

    # ===== 根据模式选择输入 =====
    if input_mode == "子文件夹":
        # 1a. 选择 BF 根目录
        bf_root = filedialog.askdirectory(title="选择 BF 通道根目录（内含子文件夹）")
        if not bf_root:
            print("未选择 BF 目录，退出。"); root.destroy(); exit()

        # 1b. 选择 AF 根目录
        af_root = filedialog.askdirectory(title="选择 AF 通道根目录（内含子文件夹）")
        if not af_root:
            print("未选择 AF 目录，退出。"); root.destroy(); exit()

        # 收集子文件夹结构
        bf_list = collect_subdir_structure(bf_root)
        af_list = collect_subdir_structure(af_root)

        if not bf_list:
            raise RuntimeError(f"BF 目录下未找到包含图像的子文件夹: {bf_root}")
        if not af_list:
            raise RuntimeError(f"AF 目录下未找到包含图像的子文件夹: {af_root}")

        n_bf = len(bf_list)
        n_af = len(af_list)
        n_pair = min(n_bf, n_af)
        if n_bf != n_af:
            print(f"警告: BF 有 {n_bf} 个子文件夹，AF 有 {n_af} 个，按顺序取前 {n_pair} 个配对。")

    else:  # 图像多选
        # 1a. 多选 BF 图像
        bf_paths = filedialog.askopenfilenames(
            title="选择 BF 通道图像（可多选）",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                       ("所有文件", "*.*")],
        )
        if not bf_paths:
            print("未选择 BF 图像，退出。"); root.destroy(); exit()
        bf_paths = natsorted(list(bf_paths))

        # 1b. 多选 AF 图像
        af_paths = filedialog.askopenfilenames(
            title="选择 AF 通道图像（可多选，与 BF 按排序一一配对）",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                       ("所有文件", "*.*")],
        )
        if not af_paths:
            print("未选择 AF 图像，退出。"); root.destroy(); exit()
        af_paths = natsorted(list(af_paths))

        n_bf = len(bf_paths)
        n_af = len(af_paths)
        n_pair = min(n_bf, n_af)
        if n_bf != n_af:
            messagebox.showwarning("数量不一致",
                f"BF {n_bf} 张，AF {n_af} 张。按排序取前 {n_pair} 对，多余的将被忽略。")

    # ===== Step 2: 选择输出根目录 =====
    out_root = filedialog.askdirectory(title="选择输出根目录（将创建 testA_BF / testA_AF）")
    if not out_root:
        print("未选择输出目录，退出。")
        root.destroy()
        exit()

    # ===== Step 3: 选项设置 =====
    phase_var = tk.StringVar(value="testA")
    op_var = tk.StringVar(value="copy")
    prefix_var = tk.BooleanVar(value=True)

    opt_win = tk.Toplevel(root)
    opt_win.title("数据集构建选项")
    opt_win.resizable(False, False)
    r = 0

    tk.Label(opt_win, text="数据集阶段:", font=("", 11)).grid(
        row=r, column=0, sticky="e", padx=(15, 2), pady=(15, 3))
    tk.OptionMenu(opt_win, phase_var, "testA", "trainA").grid(row=r, column=1, sticky="w")
    r += 1

    tk.Label(opt_win, text="操作模式:", font=("", 11)).grid(
        row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.OptionMenu(opt_win, op_var, "copy", "move").grid(row=r, column=1, sticky="w")
    r += 1

    tk.Checkbutton(opt_win, text="文件名加前缀（子文件夹模式用子文件夹名，多选模式用序号）",
                   variable=prefix_var, font=("", 11)).grid(
        row=r, column=0, columnspan=2, sticky="w", padx=15, pady=3)
    r += 1

    tk.Button(opt_win, text="开始构建", command=opt_win.destroy,
              width=12).grid(row=r, column=0, columnspan=2, pady=(10, 15))

    opt_win.grab_set()
    root.wait_window(opt_win)
    root.destroy()

    phase = phase_var.get()
    op = shutil.copy2 if op_var.get() == "copy" else shutil.move
    add_prefix = prefix_var.get()

    # 创建输出目录
    out_bf = os.path.join(out_root, f"{phase}_BF")
    out_af = os.path.join(out_root, f"{phase}_AF")
    os.makedirs(out_bf, exist_ok=True)
    os.makedirs(out_af, exist_ok=True)

    print(f"输入模式: {input_mode}")
    print(f"输出目录: {out_root}")
    print(f"配对数量: {n_pair}")
    print(f"操作: {op_var.get()} | 阶段: {phase} | 前缀: {add_prefix}")
    print()

    total_pairs = 0

    # ===== 处理 =====
    if input_mode == "子文件夹":
        print(f"BF 根目录: {bf_root}")
        print(f"AF 根目录: {af_root}\n")

        for i in range(n_pair):
            bf_sub, bf_files_list = bf_list[i]
            af_sub, af_files_list = af_list[i]

            bf_files_set = set(bf_files_list)
            af_files_set = set(af_files_list)
            common_files = natsorted(bf_files_set & af_files_set)
            bf_only = bf_files_set - af_files_set
            af_only = af_files_set - bf_files_set

            if bf_only:
                print(f"  [{bf_sub} <-> {af_sub}] BF 独有 {len(bf_only)} 个文件，已跳过")
            if af_only:
                print(f"  [{bf_sub} <-> {af_sub}] AF 独有 {len(af_only)} 个文件，已跳过")

            for fname in common_files:
                src_bf = os.path.join(bf_root, bf_sub, fname)
                src_af = os.path.join(af_root, af_sub, fname)
                new_name = f"{bf_sub}_{fname}" if add_prefix else fname

                dst_bf = os.path.join(out_bf, new_name)
                dst_af = os.path.join(out_af, new_name)

                try:
                    op(src_bf, dst_bf)
                    op(src_af, dst_af)
                    total_pairs += 1
                except Exception as e:
                    print(f"  错误 [{bf_sub}/{fname}]: {e}")

            if len(common_files) > 0:
                print(f"  [{bf_sub}] <-> [{af_sub}] : {len(common_files)} 对")

    else:  # 图像多选
        for i in range(n_pair):
            src_bf = bf_paths[i]
            src_af = af_paths[i]

            bf_name = os.path.basename(src_bf)
            af_name = os.path.basename(src_af)

            if add_prefix:
                new_name = f"{i + 1:04d}_{bf_name}"
            else:
                new_name = bf_name

            dst_bf = os.path.join(out_bf, new_name)
            dst_af = os.path.join(out_af, new_name)

            try:
                op(src_bf, dst_bf)
                op(src_af, dst_af)
                total_pairs += 1
            except Exception as e:
                print(f"  错误 [{bf_name}]: {e}")

            if (i + 1) % 50 == 0 or i == n_pair - 1:
                print(f"  [{i + 1}/{n_pair}] {bf_name} <-> {af_name}")

    print(f"\n完成！共 {total_pairs} 对 BF/AF 图像。")
    print(f"  {phase}_BF: {out_bf}")
    print(f"  {phase}_AF: {out_af}")
    print(f"\n推理命令参考:")
    print(f"  --dataroot {out_root}")
    print(f"  --dataset_mode dual_channel")
