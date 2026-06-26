import cv2
import os
import tkinter as tk
from tkinter import filedialog
from natsort import natsorted

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # 1. 选择待裁剪图像（可多选）
    img_paths = filedialog.askopenfilenames(
        title="选择待裁剪图像（可多选，第一张用于框选ROI）",
        filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif *.webp"),
                   ("所有文件", "*.*")],
    )
    if not img_paths:
        print("未选择图像，退出。")
        root.destroy()
        exit()

    # 2. 选择输出文件夹
    save_dir = filedialog.askdirectory(title="选择输出文件夹")
    if not save_dir:
        print("未选择输出文件夹，退出。")
        root.destroy()
        exit()

    root.destroy()

    img_paths = natsorted(list(img_paths))
    print(f"共选择 {len(img_paths)} 张图像")

    # 显示第一张图片，手动选择ROI
    first_img = cv2.imread(img_paths[0])
    if first_img is None:
        raise FileNotFoundError(f"读取第一张图像失败: {img_paths[0]}")
    h0, w0 = first_img.shape[:2]

    # 设定显示窗口最大宽度
    max_width = 500
    scale = 1.0
    if w0 > max_width:
        scale = max_width / w0
        disp_img = cv2.resize(first_img, (int(w0 * scale), int(h0 * scale)))
    else:
        disp_img = first_img.copy()

    # 选择ROI
    roi_disp = cv2.selectROI("Select ROI (on scaled image)", disp_img, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    # 将ROI坐标映射回原图
    x_disp, y_disp, w_disp, h_disp = map(int, roi_disp)
    x = int(x_disp / scale)
    y = int(y_disp / scale)
    w = int(w_disp / scale)
    h = int(h_disp / scale)
    print(f"ROI on original image: x={x}, y={y}, w={w}, h={h}")

    if w == 0 or h == 0:
        print("未选中有效ROI区域，退出。")
        exit()

    # 批量裁剪并保存
    os.makedirs(save_dir, exist_ok=True)
    print(f"输出目录: {save_dir}")

    count = 0
    for idx, img_path in enumerate(img_paths):
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法读取文件: {img_path}")
                continue
            crop = img[y:y+h, x:x+w]
            save_path = os.path.join(save_dir, f'ROI_{idx:04d}.png')
            ok = cv2.imwrite(save_path, crop)
            if not ok:
                print(f"保存失败: {save_path}")
                continue
            print(f"[{idx + 1}/{len(img_paths)}] Saved: {save_path}")
            count += 1
        except Exception as e:
            print(f"处理失败 [{img_path}]: {e}")

    print(f"完成！共裁剪 {count}/{len(img_paths)} 张图像。")
