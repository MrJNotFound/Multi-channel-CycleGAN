import cv2
import numpy as np


def enhance_image(
    img: np.ndarray,
    clahe_clip_limit: float = 8,
    clahe_tile_grid_size: tuple = (10, 10),
    mode: str = "bgr"
):
    """
    对图像应用 CLAHE 增强，兼容灰度与彩色图像。
    要求输入为 8-bit（uint8），灰度图保持单通道输出，彩色图按三个通道分别增强后再合并。

    参数:
    - img: 输入图像（np.ndarray），可为灰度 (H,W) 或 BGR 彩色 (H,W,3) 或 BGRA (H,W,4)
    - clahe_clip_limit: CLAHE 的 clipLimit 参数
    - clahe_tile_grid_size: CLAHE 的 tileGridSize 参数
    - mode:
        "bgr"  : 彩色图对 B、G、R 三个通道分别应用 CLAHE（满足每通道增强需求）
        "gray" : 强制按灰度增强（若彩色将转换为灰度后增强并返回灰度）

    返回:
    - 增强后的图像（与输入通道数保持一致；"gray"模式返回灰度图）
    """
    if img is None:
        return img

    # 仅支持 8-bit 图像进行 CLAHE；若不是 uint8，则直接返回以保持位深一致
    if img.dtype != np.uint8:
        return img

    has_alpha = (img.ndim == 3 and img.shape[2] == 4)
    alpha = None
    if has_alpha:
        alpha = img[:, :, 3].copy()
        img = img[:, :, :3]

    # 灰度图
    if img.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
        out = clahe.apply(img)
        return out

    # 彩色图 (BGR)
    if mode == "gray":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
        out = clahe.apply(gray)
        return out

    # HSV增强，只增强S和V
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # CLAHE
    clahe_v = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
    clahe_s = cv2.createCLAHE(clipLimit=clahe_clip_limit/2, tileGridSize=clahe_tile_grid_size)

    v2 = clahe_v.apply(v)
    s2 = clahe_s.apply(s)

    hsv2 = cv2.merge((h, s2, v2))
    bgr_eq = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

    if has_alpha:
        bgr_eq = cv2.merge((bgr_eq, alpha))
    return bgr_eq


def resize_image(
    img,
    resize_ratio=1/2,
    interpolation_down=cv2.INTER_AREA,
    interpolation_up=cv2.INTER_LINEAR
):
    """
    对图像进行缩放，兼容灰度与彩色；保持通道不变。
    :param img: 输入图像（np.ndarray）
    :param resize_ratio: 缩放比例
    :param interpolation_down: 缩小时插值方式（默认 INTER_AREA）
    :param interpolation_up: 放大时插值方式（默认 INTER_LINEAR）
    :return: 缩放后的图像（np.ndarray）
    """
    if img is None:
        return img
    h, w = img.shape[:2]
    new_size = (max(1, int(w * resize_ratio)), max(1, int(h * resize_ratio)))
    interp = interpolation_down if resize_ratio < 1.0 else interpolation_up
    img_resized = cv2.resize(img, new_size, interpolation=interp)
    return img_resized


def rotate_image(
    img,
    angle=0,
    interpolation=cv2.INTER_NEAREST
):
    """
    对图像进行旋转，自动调整画布避免内容丢失，兼容灰度与彩色。
    :param img: 输入图像（np.ndarray）
    :param angle: 旋转角度（正值为逆时针旋转）
    :param interpolation: 插值方式（默认最近邻，掩码友好；自然图像可用 cv2.INTER_LINEAR）
    :param border_mode: 边界填充模式（默认常数）
    :return: 旋转后的图像（np.ndarray）
    """
    if img is None:
        return img
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 计算新图像尺寸
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    # 调整仿射矩阵的平移部分
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2
    img_rotated = cv2.warpAffine(img,rotation_matrix,(new_w, new_h),flags=interpolation)
    return img_rotated

