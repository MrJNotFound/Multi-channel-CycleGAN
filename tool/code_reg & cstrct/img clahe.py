import cv2
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from natsort import natsorted

# 输入和输出文件夹路径
input_path = 'C:/Users/30927/Desktop/all/rigid_registered/registered_elastic_inverse'
# 替换为你的输入文件夹路径
output_path = 'C:/Users/30927/Desktop/all/rigid_registered/registered_elastic_enhanced'
# 输出文件夹路径

# 确保输出文件夹存在
os.makedirs(output_path, exist_ok=True)

# 获取文件夹中所有图像文件
image_files = [f for f in os.listdir(input_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
image_files = natsorted(image_files)

# 处理每张图片
for index, file_name in tqdm(enumerate(image_files)):
    img_path = os.path.join(input_path, file_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图

    # 应用CLAHE
    clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(10, 10))
    img_clahe = clahe.apply(img)

    # plt.subplot(1, 2, 1)
    # plt.imshow(img, cmap='gray')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(img_clahe, cmap='gray')
    # plt.show()

    # 保存增强后的图像
    output_file_name = file_name
    enhanced_path = os.path.join(output_path, output_file_name)
    cv2.imwrite(enhanced_path, img_clahe)

print("自适应灰度直方图均衡完成，增强后的图像已导出！")
