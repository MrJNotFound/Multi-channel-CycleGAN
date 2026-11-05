import os
import shutil

# 1. 设置路径
root_dir = r"C:\Users\30927\Desktop\mouse brain full res"  # 如 r"D:\your\path"
output_dir = r"C:\Users\30927\Desktop\mouse brain full res"  # 如 r"D:\your\output_path"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. 获取并排序所有文件夹
folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
folders.sort(key=lambda x: int(x.split('-')[0]))  # 按数字顺序排序

# 3. 遍历文件夹和文件，重命名复制
img_idx = 1
for folder in folders:
    folder_path = os.path.join(root_dir, folder)
    # 获取并排序图片文件
    imgs = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    imgs.sort()  # 默认Region 001.bmp, Region 002.bmp...排序
    for img in imgs:
        src = os.path.join(folder_path, img)
        dst_name = f"Region_{img_idx:03d}.jpg"
        dst = os.path.join(output_dir, dst_name)
        shutil.copy(src, dst)
        img_idx += 1

print("全部图片已按顺序重命名并复制完成！")
