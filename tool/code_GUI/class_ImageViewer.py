import numpy as np
from typing import Optional
from func_tools import np_to_qimage
from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QPixmap, QPainter, QWheelEvent, QMouseEvent


class InteractiveImageLabel(QLabel):
# 支持缩放和拖拽的图像显示控件
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(600, 600)
        self.setStyleSheet("border: 1px solid #ccc; background-color: #f9f9f9;")
        self.original_pixmap: Optional[QPixmap] = None
        self.scaled_pixmap: Optional[QPixmap] = None
        self.scale_factor = 1.0
        self.min_scale = 0.1
        self.max_scale = 5.0
        self.zoom_step = 1.2
        self.dragging = False
        self.last_pan_point = QPoint()
        self.image_offset = QPoint(0, 0)
        self.setMouseTracking(True)

    def set_image(self, img: Optional[np.ndarray], reset_view: bool = True):
        if img is None:
            self.original_pixmap = None
            self.scaled_pixmap = None
            self.setText("无图像")
            self.update()
            return
        qimg = np_to_qimage(img)
        self.original_pixmap = QPixmap.fromImage(qimg)
        if reset_view:
            self._fit_to_window()
        else:
            self._update_scaled_pixmap()
        self.update()

    def _fit_to_window(self):
        if self.original_pixmap is None:
            return
        widget_size = self.size()
        pixmap_size = self.original_pixmap.size()
        margin = 20
        available_width = max(widget_size.width() - margin * 2, 1)
        available_height = max(widget_size.height() - margin * 2, 1)
        scale_x = available_width / pixmap_size.width()
        scale_y = available_height / pixmap_size.height()
        self.scale_factor = min(scale_x, scale_y, 1.0)
        self.image_offset = QPoint(0, 0)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if self.original_pixmap is None:
            return

        scaled_size = self.original_pixmap.size() * self.scale_factor
        self.scaled_pixmap = self.original_pixmap.scaled(
            scaled_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.original_pixmap is not None and self.scale_factor is None:
            self._fit_to_window()
            self.update()

    def wheelEvent(self, event: QWheelEvent):
        if self.original_pixmap is None:
            return

        angle_delta = event.angleDelta().y()
        if angle_delta > 0:
            new_scale = self.scale_factor * self.zoom_step
        else:
            new_scale = self.scale_factor / self.zoom_step
        new_scale = max(self.min_scale, min(self.max_scale, new_scale))

        if new_scale != self.scale_factor:
            mouse_pos = event.position().toPoint()
            old_image_pos = self._widget_to_image_pos(mouse_pos)
            self.scale_factor = new_scale
            self._update_scaled_pixmap()
            new_image_pos = self._widget_to_image_pos(mouse_pos)
            if old_image_pos is not None and new_image_pos is not None:
                offset_delta = (old_image_pos - new_image_pos) * self.scale_factor
                self.image_offset += offset_delta
            self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self.original_pixmap is not None:
            self.dragging = True
            self.last_pan_point = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.dragging and self.original_pixmap is not None:
            delta = event.position().toPoint() - self.last_pan_point
            self.image_offset += delta
            self.last_pan_point = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self.original_pixmap is not None:
            self._fit_to_window()
            self.update()

    def _widget_to_image_pos(self, widget_pos: QPoint) -> Optional[QPoint]:
        if self.scaled_pixmap is None:
            return None
        widget_rect = self.rect()
        pixmap_rect = self.scaled_pixmap.rect()
        center_offset = QPoint(
            (widget_rect.width() - pixmap_rect.width()) // 2,
            (widget_rect.height() - pixmap_rect.height()) // 2
        )
        image_topleft = center_offset + self.image_offset
        relative_pos = widget_pos - image_topleft
        if self.scale_factor > 0:
            image_pos = QPoint(
                int(relative_pos.x() / self.scale_factor),
                int(relative_pos.y() / self.scale_factor)
            )
            return image_pos
        return None

    def paintEvent(self, event):
        """自定义绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.palette().color(self.backgroundRole()))

        if self.scaled_pixmap is None:
            painter.setPen(self.palette().color(self.foregroundRole()))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())
            return

        widget_rect = self.rect()
        pixmap_rect = self.scaled_pixmap.rect()
        center_offset = QPoint(
            (widget_rect.width() - pixmap_rect.width()) // 2,
            (widget_rect.height() - pixmap_rect.height()) // 2
        )
        draw_pos = center_offset + self.image_offset
        painter.drawPixmap(draw_pos, self.scaled_pixmap)
        painter.setPen(Qt.GlobalColor.gray)
        painter.drawRect(draw_pos.x() - 1, draw_pos.y() - 1,
                        pixmap_rect.width() + 1, pixmap_rect.height() + 1)