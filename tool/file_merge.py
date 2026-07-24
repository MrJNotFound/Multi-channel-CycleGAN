
from pathlib import Path
import shutil
import tkinter as tk
from tkinter import filedialog
from natsort import natsorted

# 支持的图像扩展名
exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# ======================
# 交互选择文件夹
# ======================
root = tk.Tk()
root.withdraw()

dir_a = filedialog.askdirectory(title="选择第一个文件夹（其文件排在前面）")
if not dir_a:
    print("未选择第一个文件夹，退出。")
    exit(0)

dir_b = filedialog.askdirectory(title="选择第二个文件夹（其文件排在后面）")
if not dir_b:
    print("未选择第二个文件夹，退出。")
    exit(0)

out_dir = filedialog.askdirectory(title="选择输出文件夹")
if not out_dir:
    print("未选择输出文件夹，退出。")
    exit(0)

root.destroy()

print(f"Folder A: {dir_a}")
print(f"Folder B: {dir_b}")
print(f"Output:   {out_dir}")
print()

# ======================
# 主逻辑
# ======================
src_a = Path(dir_a)
src_b = Path(dir_b)
dst_root = Path(out_dir)

dst_root.mkdir(parents=True, exist_ok=True)

# 收集两个文件夹中所有图像文件（仅文件名 → Path 映射）
files_a = {}
for p in src_a.rglob("*"):
    if p.is_file() and p.suffix.lower() in exts:
        # 同名文件只保留第一个遇到的（如需保留所有，可改为 list）
        if p.name not in files_a:
            files_a[p.name] = p

files_b = {}
for p in src_b.rglob("*"):
    if p.is_file() and p.suffix.lower() in exts:
        if p.name not in files_b:
            files_b[p.name] = p

# 找出同名文件
common_names = natsorted(set(files_a.keys()) & set(files_b.keys()))

if not common_names:
    print("No common files found between the two folders.")
    exit(0)

print(f"Found {len(common_names)} common files.")

# 编号：第一个文件夹的文件从 0001 开始，第二个文件夹的紧接其后
counter = 1

# 先处理第一个文件夹的所有同名文件
for name in common_names:
    src_path = files_a[name]
    dst_name = f"{counter:04d}{src_path.suffix}"
    dst_path = dst_root / dst_name
    shutil.copy2(str(src_path), str(dst_path))
    print(f"[{counter:04d}] A <- {src_path}")
    counter += 1

# 再处理第二个文件夹的所有同名文件
for name in common_names:
    src_path = files_b[name]
    dst_name = f"{counter:04d}{src_path.suffix}"
    dst_path = dst_root / dst_name
    shutil.copy2(str(src_path), str(dst_path))
    print(f"[{counter:04d}] B <- {src_path}")
    counter += 1

print(f"Done. {counter - 1} files saved to: {dst_root}")
