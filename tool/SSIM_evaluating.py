import cv2
import tkinter as tk
from tkinter import filedialog
from skimage.metrics import structural_similarity as ssim


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # 1. 选择第一张图像
    path1 = filedialog.askopenfilename(
        title="选择第一张图像",
        filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif *.webp"),
                   ("所有文件", "*.*")],
    )
    if not path1:
        print("未选择第一张图像，退出。")
        root.destroy()
        exit()

    # 2. 选择第二张图像
    path2 = filedialog.askopenfilename(
        title="选择第二张图像",
        filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif *.webp"),
                   ("所有文件", "*.*")],
    )
    if not path2:
        print("未选择第二张图像，退出。")
        root.destroy()
        exit()

    root.destroy()

    # 读取图像
    img1 = cv2.imread(path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(path2, cv2.IMREAD_COLOR)

    if img1 is None:
        raise FileNotFoundError(f"读取失败: {path1}")
    if img2 is None:
        raise FileNotFoundError(f"读取失败: {path2}")

    print(f"图像1: {path1}  ({img1.shape[1]}x{img1.shape[0]})")
    print(f"图像2: {path2}  ({img2.shape[1]}x{img2.shape[0]})")

    # 尺寸不一致时，将img2缩放至img1尺寸
    if img1.shape[:2] != img2.shape[:2]:
        print(f"尺寸不一致，将图像2缩放至 {img1.shape[1]}x{img1.shape[0]}")
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    # 计算 SSIM
    if img1.ndim == 3 and img2.ndim == 3:
        ssim_val = ssim(img1, img2, channel_axis=2, data_range=255)
    else:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2
        ssim_val = ssim(gray1, gray2, data_range=255)

    print(f"\nSSIM: {ssim_val:.6f}")
