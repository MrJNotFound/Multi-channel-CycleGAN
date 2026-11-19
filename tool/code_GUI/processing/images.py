"""Image file I/O and shared size and rotation helpers."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np


def read_image(path, flags=cv2.IMREAD_UNCHANGED):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, flags) if data.size else None
    except (OSError, cv2.error):
        return None


def write_image(path, image):
    target = Path(path)
    success, encoded = cv2.imencode(target.suffix.lower(), image)
    if not success:
        raise OSError(f"无法编码图像：{target.name}")
    temporary = None
    try:
        with NamedTemporaryFile(dir=target.parent, prefix=".image_", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(memoryview(encoded))
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

def preview_geometry(shape, max_size):
    """Return a bounded (width, height) and the nominal scale."""
    height, width = shape[:2]
    scale = min(max_size / max(height, width), 1.0)
    return (max(1, int(width * scale)), max(1, int(height * scale))), scale


def fit_preview(image, max_size=800, *, interpolation=cv2.INTER_AREA):
    if image is None:
        return None
    dimensions, scale = preview_geometry(image.shape, max_size)
    return cv2.resize(image, dimensions, interpolation=interpolation) if scale < 1 else image


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
