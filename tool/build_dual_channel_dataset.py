"""
将两个文件夹中一一对应的子文件夹内的 BF / AF 图块，
整理为 DualChannelDataset 推理所需的目录结构。

输入结构示例:
    BF_root/
    ├── ROI_01/
    │   ├── patch_0001.png
    │   └── ...
    ├── ROI_02/
    │   └── ...
    AF_root/
    ├── ROI_01/          (与 BF_root 子文件夹名一一对应)
    │   ├── patch_0001.png
    │   └── ...
    ├── ROI_02/
    │   └── ...

输出结构:
    output_root/
    └── testA_BF/        (或 trainA_BF)
    │   ├── ROI_01_patch_0001.png
    │   └── ...
    └── testA_AF/        (或 trainA_AF)
        ├── ROI_01_patch_0001.png
        └── ...
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

    # 1. 选择 BF 根目录
    bf_root = filedialog.askdirectory(title="选择 BF 通道根目录（内含子文件夹）")
    if not bf_root:
        print("未选择 BF 目录，退出。")
        root.destroy()
        exit()

    # 2. 选择 AF 根目录
    af_root = filedialog.askdirectory(title="选择 AF 通道根目录（内含子文件夹）")
    if not af_root:
        print("未选择 AF 目录，退出。")
        root.destroy()
        exit()

    # 3. 选择输出根目录
    out_root = filedialog.askdirectory(title="选择输出根目录（将创建 testA_BF / testA_AF）")
    if not out_root:
        print("未选择输出目录，退出。")
        root.destroy()
        exit()

    # 4. 选择模式 / 前缀
    mode_var = tk.StringVar(value="copy")
    phase_var = tk.StringVar(value="testA")
    prefix_var = tk.BooleanVar(value=True)

    opt_win = tk.Toplevel(root)
    opt_win.title("数据集构建选项")
    opt_win.resizable(False, False)
    r = 0

    tk.Label(opt_win, text="数据集阶段:", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=(15, 3))
    tk.OptionMenu(opt_win, phase_var, "testA", "trainA").grid(row=r, column=1, sticky="w")
    r += 1

    tk.Label(opt_win, text="操作模式:", font=("", 11)).grid(row=r, column=0, sticky="e", padx=(15, 2), pady=3)
    tk.OptionMenu(opt_win, mode_var, "copy", "move").grid(row=r, column=1, sticky="w")
    r += 1

    tk.Checkbutton(opt_win, text="文件名加子文件夹前缀（避免重名冲突）",
                   variable=prefix_var, font=("", 11)).grid(row=r, column=0, columnspan=2, sticky="w", padx=15, pady=3)
    r += 1

    tk.Button(opt_win, text="开始构建", command=opt_win.destroy, width=12).grid(row=r, column=0, columnspan=2, pady=(10, 15))

    opt_win.grab_set()
    root.wait_window(opt_win)
    root.destroy()

    mode = mode_var.get()
    phase = phase_var.get()
    add_prefix = prefix_var.get()

    # 收集子文件夹结构（按名称排序，不做名称匹配）
    bf_list = collect_subdir_structure(bf_root)
    af_list = collect_subdir_structure(af_root)

    if not bf_list:
        raise RuntimeError(f"BF 目录下未找到包含图像的子文件夹: {bf_root}")
    if not af_list:
        raise RuntimeError(f"AF 目录下未找到包含图像的子文件夹: {af_root}")

    n_bf = len(bf_list)
    n_af = len(af_list)
    n = min(n_bf, n_af)

    if n_bf != n_af:
        print(f"警告: BF 有 {n_bf} 个子文件夹，AF 有 {n_af} 个，按顺序取前 {n} 个配对。")

    # 创建输出目录
    out_bf = os.path.join(out_root, f"{phase}_BF")
    out_af = os.path.join(out_root, f"{phase}_AF")
    os.makedirs(out_bf, exist_ok=True)
    os.makedirs(out_af, exist_ok=True)

    print(f"BF 根目录 : {bf_root}")
    print(f"AF 根目录 : {af_root}")
    print(f"输出目录  : {out_root}")
    print(f"配对子文件夹: {n} 对 (按排序一一对应)")
    print(f"模式: {mode} | 阶段: {phase} | 前缀: {add_prefix}")
    print()

    total_pairs = 0
    op = shutil.copy2 if mode == "copy" else shutil.move

    for i in range(n):
        bf_sub, bf_files_list = bf_list[i]
        af_sub, af_files_list = af_list[i]

        # 找出共有文件（按文件名匹配）
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

            if add_prefix:
                # 用 BF 子文件夹名作为前缀（保证 BF/AF 对同名）
                new_name = f"{bf_sub}_{fname}"
            else:
                new_name = fname

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

    print(f"\n完成！共 {total_pairs} 对 BF/AF 图像。")
    print(f"  {phase}_BF: {out_bf}")
    print(f"  {phase}_AF: {out_af}")
    print(f"\n推理命令参考:")
    print(f"  --dataroot {out_root}")
    print(f"  --dataset_mode dual_channel")
