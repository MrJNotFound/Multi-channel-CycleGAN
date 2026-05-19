import cv2 as cv
import os
import numpy as np

input_dir = r"C:\Users\30927\Desktop\CycleGAN-and-pix2pix\datasets\mouse_kidney\1"
output_dir = r"C:\Users\30927\Desktop\CycleGAN-and-pix2pix\datasets\mouse_kidney\2"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

threshold = 235

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        img_path = os.path.join(input_dir, filename)
        img = cv.imread(img_path)
        # 把img的像素值从0-255线性映射到0-1, 类似tensor的归一化处理
        img = img.astype(np.float32) / 255.0
        # 再把img的像素值从0-1线性映射到-1到1, 类似tensor的归一化处理
        img = img * 2 - 1
        img_mean = img.mean(axis=2)
        img_normal = (img_mean - (threshold/127.5-1))*100
        real_A_sigmoid = 1 / (1 + np.exp(-img_normal))
        real_A_sigmoid = (real_A_sigmoid * 255).astype('uint8')
        cv.imwrite(os.path.join(output_dir, filename), real_A_sigmoid)
