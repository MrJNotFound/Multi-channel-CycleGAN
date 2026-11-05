import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# 配准参数
N_MATCHES = 200
scale_factor = 1  # 下采样比例

fixed_img = cv2.imread(r"C:\Users\30927\Desktop\all\Region_001.bmp", cv2.IMREAD_COLOR)
moving_img = cv2.imread(r"C:\Users\30927\Desktop\all\Region_002.bmp", cv2.IMREAD_COLOR)

fixed_img_lowres = cv2.resize(fixed_img, (int(fixed_img.shape[1] / scale_factor), int(fixed_img.shape[0] / scale_factor)))
moving_img_lowres = cv2.resize(moving_img, (int(moving_img.shape[1] / scale_factor), int(moving_img.shape[0] / scale_factor)))

# 1. 特征点检测（用灰度图）
fixed_gray = cv2.cvtColor(fixed_img_lowres, cv2.COLOR_BGR2GRAY)
moving_gray = cv2.cvtColor(moving_img_lowres, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create(5000)
kp1, des1 = orb.detectAndCompute(fixed_gray, None)
kp2, des2 = orb.detectAndCompute(moving_gray, None)

# 2. 特征点匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
matches = matches[:N_MATCHES]

# 3. 获取匹配点
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# 4. 估算仿射变换（低分辨率）
M_low, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=5)

# 5. 将仿射矩阵应用到高分辨率
M_high = M_low.copy()
M_high[0, 2] *= scale_factor
M_high[1, 2] *= scale_factor

# 6. 变换高分辨率moving图像
h, w = fixed_img.shape[:2]
registered = cv2.warpAffine(moving_img, M_high, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255, 255, 255))

# 8. 可视化匹配点（用灰度图）
draw_matches = cv2.drawMatches(fixed_gray, kp1, moving_gray, kp2, matches, None, flags=2)
cv2.imwrite('matches.png', draw_matches)

# region1 = fixed_img[9500:10500, 7000:8000]
# region2 = registered[9500:10500, 7000:8000]
#
# cv2.imwrite('region1.png', region1)
# cv2.imwrite('region2.png', region2)


