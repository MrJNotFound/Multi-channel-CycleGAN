import cv2 as cv
import os

img_dir = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_BF\kidney_20x_2\WSI_low_res"
output_dir = r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_BF\kidney_20x_2\WSI_low_res_overlay"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(img_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        img_path = os.path.join(img_dir, filename)
        img = cv.imread(img_path)

        # rgb通道取平均值得到灰度
        img_gray = img.mean(axis=2)

        # # 使用全局自适应阈值方法（Otsu's method）进行二值化
        # threshold, img_binary = cv.threshold(img_gray.astype('uint8'), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # print(f"Otsu's threshold: {threshold}")

        # 使用给定阈值进行二值化
        threshold_value = 120  # 可以根据需要调整这个值
        _, img_binary = cv.threshold(img_gray.astype('uint8'), threshold_value, 255, cv.THRESH_BINARY)

        # 前景与背景使用不同颜色叠加到原图
        img_overlay = img.copy()
        img_overlay[img_binary == 255] = [255, 255, 255]
        img_overlay = cv.resize(img_overlay, [img.shape[1]//1, img.shape[0]//1], interpolation=cv.INTER_AREA)
        cv.imwrite(os.path.join(output_dir, filename), img_overlay)

