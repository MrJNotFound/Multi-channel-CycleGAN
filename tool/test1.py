# 读取单张图像，截掉图像左侧一部分，然后保存
import cv2
import os

img_dir = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_BF\WSI"
output_dir = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_BF\WSI"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

img = cv2.imread(os.path.join(img_dir, "kidney_20x_02.png"), cv2.IMREAD_GRAYSCALE)
if img is None:
    print("无法读取图像")
else:
    # 截掉图像左侧100像素
    cropped_img = img[:, 500:]

    # 保存截掉后的图像
    output_path = os.path.join(output_dir, "kidney_20x_02_cropped.png")
    cv2.imwrite(output_path, cropped_img)
    print(f"保存: {output_path}")

