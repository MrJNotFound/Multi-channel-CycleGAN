import numpy as np
import cv2
from typing import Optional
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QMessageBox


def np_to_qimage(img: np.ndarray) -> QImage:
    """将numpy图像转换为QImage（支持灰度、BGR三通道）。"""
    if img is None:
        return QImage()
    if img.ndim == 2:
        h, w = img.shape
        return QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
    elif img.ndim == 3:
        h, w, c = img.shape
        if c == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        else:
            raise ValueError("仅支持单通道或3通道图像")
    else:
        raise ValueError("不支持的图像维度")


def show_info(parent, msg: str):
    QMessageBox.information(parent, "提示", msg)


def fit_preview(img: Optional[np.ndarray], max_size: int = 800) -> Optional[np.ndarray]:
    """按较长边缩放至不超过max_size便于预览。"""
    if img is None:
        return None
    if img.ndim == 2:
        h, w = img.shape
    else:
        h, w = img.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img