import numpy as np
import cv2


def freq_filter(
        img,
        angle,
        bandwidth=20
):
    """
    频域带阻滤波，输入图像、频谱图条纹方向（由交互获得）、带宽，返回滤波后图像。
    注意：angle为频谱图中条纹方向，直接用于mask。
    优化掩膜生成部分，提升效率。
    """
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    # 生成坐标网格
    y = np.arange(rows) - crow
    x = np.arange(cols) - ccol
    X, Y = np.meshgrid(x, y)
    theta = np.degrees(np.arctan2(Y, X))
    # 计算与目标方向的最小夹角（周期性）
    diff = (theta - angle + 180) % 360 - 180
    mask = np.ones((rows, cols), np.float32)
    mask[np.abs(diff) < bandwidth] = 0
    mask[np.abs(diff) > (180 - bandwidth)] = 0
    # 傅里叶变换与滤波
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift_masked = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_masked)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back_norm = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back_uint8 = np.uint8(img_back_norm)
    return img_back_uint8
