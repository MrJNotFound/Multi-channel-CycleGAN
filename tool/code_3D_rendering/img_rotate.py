import cv2
import os

# 读取文件夹图像，旋转图像后输出

input_path = r"C:\Users\30927\Desktop\img_histology\regist_brain_HE\ROI_elastic_2_2"
output_path = r"C:\Users\30927\Desktop\img_histology\regist_brain_HE\ROI_elastic_2_2_rotate"
if not os.path.exists(output_path):
    os.makedirs(output_path)

for img_path in os.listdir(input_path):
    img = cv2.imread(os.path.join(input_path, img_path))
    if img is None:
        print(f"Failed to read {img_path}")
        continue
    # 逆时针旋转图像
    rotated_img = cv2.rotate(img, cv2.ROTATE_180)
    output_file = os.path.join(output_path, f"flipped_{img_path}")
    cv2.imwrite(output_file, rotated_img)
    print(f"Saved flipped image to {output_file}")

