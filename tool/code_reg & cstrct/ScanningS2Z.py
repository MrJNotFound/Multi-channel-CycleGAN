import os
import shutil

def s_to_raster_rename(image_paths, grid_rows, grid_cols, out_dir):
    """
    将S型扫描（Z字形）顺序的图片重新按同向(raster)顺序编号并保存到out_dir
    Args:
        image_paths: list of str, 原始S型扫描顺序图片路径
        grid_rows: int, 行数
        grid_cols: int, 列数
        out_dir: 输出文件夹
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    n_per_grid = grid_rows * grid_cols
    total = len(image_paths)
    n_groups = total // n_per_grid
    for g in range(n_groups):
        block = image_paths[g*n_per_grid : (g+1)*n_per_grid]
        # 还原为二维 S形网格
        mat = []
        for r in range(grid_rows):
            row = block[r*grid_cols : (r+1)*grid_cols]
            if r % 2 == 1:  # 奇数行需要翻转
                row = row[::-1]
            mat.append(row)
        # 顺序输出为同向扫描
        idx_flat = 0
        for r in range(grid_rows):
            for c in range(grid_cols):
                src_path = mat[r][c]
                # 新的编号
                global_idx = g*n_per_grid + r*grid_cols + c
                new_name = f"img_{global_idx+1:04d}{os.path.splitext(src_path)[1]}"
                dst_path = os.path.join(out_dir, new_name)
                shutil.copy2(src_path, dst_path)
                idx_flat += 1
    print(f"重排并复制完成，输出到 {out_dir}")

# 用法示例
if __name__ == "__main__":
    # 示范用法
    # 假设 image_paths = ["a.png", "b.png", "c.png",...], grid_rows=4, grid_cols=5
    # 你可以用glob等来批量获取图片列表
    import glob
    img_dir = r"C:\Users\30927\Desktop\img_histology\stain_bone\block_face\original\bone_2"
    out_dir = r"C:\Users\30927\Desktop\img_histology\stain_bone\block_face\original\bone_21"
    grid_rows, grid_cols = 13, 12

    # 以文件名排序为例，请按S型扫描现有顺序排列！
    image_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))  # 按需更换扩展名
    s_to_raster_rename(image_paths, grid_rows, grid_cols, out_dir)