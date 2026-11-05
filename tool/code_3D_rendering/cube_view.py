import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PyQt6.QtGui import QImage, QPainter, QTransform
from PyQt6.QtWidgets import QWidget

from volume_tools import normalize_to_uint8, clamp_index, upsample_along_z


@dataclass
class FaceTexture:
    name: str
    image: QImage
    p0: Tuple[float, float, float]
    p1: Tuple[float, float, float]
    p2: Tuple[float, float, float]
    depth: float


def _to_qimage(img: np.ndarray) -> QImage:
    img8 = normalize_to_uint8(img)
    img8 = np.ascontiguousarray(img8)
    if img8.ndim == 2:
        h, w = img8.shape
        return QImage(img8.tobytes(), w, h, w, QImage.Format.Format_Grayscale8).copy()
    h, w, c = img8.shape
    if c >= 3:
        rgb = np.ascontiguousarray(img8[:, :, :3])
        return QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    gray = np.ascontiguousarray(img8[:, :, 0])
    return QImage(gray.tobytes(), w, h, w, QImage.Format.Format_Grayscale8).copy()


def _to_qimage_rgba(img: np.ndarray, mask: np.ndarray) -> QImage:
    img8 = normalize_to_uint8(img)
    img8 = np.ascontiguousarray(img8)
    if img8.ndim == 2:
        rgb = np.repeat(img8[:, :, None], 3, axis=2)
    else:
        if img8.shape[-1] >= 3:
            rgb = img8[:, :, :3]
        else:
            rgb = np.repeat(img8[:, :, :1], 3, axis=2)
    alpha = np.where(mask, 255, 0).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    h, w = rgba.shape[:2]
    return QImage(rgba.tobytes(), w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()


def _affine_from_points(src_w: int, src_h: int, p0, p1, p2) -> QTransform:
    if src_w <= 0 or src_h <= 0:
        return QTransform()
    a = (p1[0] - p0[0]) / src_w
    d = (p1[1] - p0[1]) / src_w
    b = (p2[0] - p0[0]) / src_h
    e = (p2[1] - p0[1]) / src_h
    c = p0[0]
    f = p0[1]
    return QTransform(a, d, 0.0, b, e, 0.0, c, f, 1.0)


class CubeView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._volume = None
        self._is_color = False
        self._cut_z = 0
        self._cut_y = 0
        self._cut_x = 0
        self._rendering = "translucent"
        self._spacing = (1.0, 1.0, 1.0)
        self._cached = None
        self._azimuth = 45.0
        self._elevation = 35.264
        self._last_pos = None

    def set_volume(self, volume: np.ndarray, is_color: bool, spacing=None):
        self._volume = volume
        self._is_color = is_color
        if spacing is not None:
            self._spacing = spacing
        self.update_view(self._cut_z, self._cut_y, self._cut_x, self._rendering)

    def update_view(self, cut_z: int, cut_y: int, cut_x: int, rendering: str):
        if self._volume is None:
            return
        self._rendering = rendering
        self._cut_z = clamp_index(cut_z, self._volume.shape[0])
        self._cut_y = clamp_index(cut_y, self._volume.shape[1])
        self._cut_x = clamp_index(cut_x, self._volume.shape[2])
        self._cached = self._render_cube()
        self.update()

    def render_at_size(self, cut_z: int, cut_y: int, cut_x: int, rendering: str, width: int, height: int) -> QImage:
        if self._volume is None:
            return QImage()
        self._rendering = rendering
        self._cut_z = clamp_index(cut_z, self._volume.shape[0])
        self._cut_y = clamp_index(cut_y, self._volume.shape[1])
        self._cut_x = clamp_index(cut_x, self._volume.shape[2])
        return self._render_cube(width, height)

    def mousePressEvent(self, event):
        if event.button() == event.button().LeftButton:
            self._last_pos = event.position().toPoint()

    def mouseMoveEvent(self, event):
        if self._last_pos is None:
            return
        pos = event.position().toPoint()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()
        self._azimuth = (self._azimuth + dx * 0.5) % 360.0
        self._elevation = float(np.clip(self._elevation - dy * 0.5, -85.0, 85.0))
        self._last_pos = pos
        self._cached = self._render_cube()
        self.update()

    def mouseReleaseEvent(self, event):
        self._last_pos = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        if self._cached is None:
            return
        img = self._cached
        x = (self.width() - img.width()) // 2
        y = (self.height() - img.height()) // 2
        painter.drawImage(x, y, img)

    def _render_cube(self, force_width: int = None, force_height: int = None) -> QImage:
        faces = self._build_faces(force_width, force_height)
        if not faces:
            return QImage()
        width = max(10, force_width if force_width is not None else self.width())
        height = max(10, force_height if force_height is not None else self.height())
        image = QImage(width, height, QImage.Format.Format_ARGB32)
        image.fill(0)
        painter = QPainter(image)
        reverse = self._near_is_small_depth(faces)
        for face in sorted(faces, key=lambda f: f.depth, reverse=reverse):
            transform = _affine_from_points(face.image.width(), face.image.height(), face.p0, face.p1, face.p2)
            painter.save()
            painter.setTransform(transform, combine=False)
            painter.drawImage(0, 0, face.image)
            painter.restore()
        painter.end()
        return image

    def _near_is_small_depth(self, faces: List[FaceTexture]) -> bool:
        front = next((f for f in faces if f.name == "front"), None)
        back = next((f for f in faces if f.name == "back"), None)
        if front is None or back is None:
            return False
        return front.depth < back.depth

    def _build_faces(self, force_width: int = None, force_height: int = None) -> List[FaceTexture]:
        if self._volume is None:
            return []
        vol = self._volume
        z_full = vol.shape[0] - 1
        y_full = vol.shape[1] - 1
        x_full = vol.shape[2] - 1

        sz, sy, sx = self._spacing
        z_scale = sz / sx if sx > 0 else 1.0

        cut_z = min(z_full, self._cut_z)
        cut_y = min(y_full, self._cut_y)
        cut_x = min(x_full, self._cut_x)

        if self._rendering != "translucent":
            cut_z, cut_y, cut_x = z_full, y_full, x_full

        cut_z_proj = cut_z * z_scale

        mask_xy = (np.arange(y_full + 1)[:, None] <= cut_y) & (np.arange(x_full + 1)[None, :] <= cut_x)
        mask_xz = (np.arange(z_full + 1)[:, None] <= cut_z) & (np.arange(x_full + 1)[None, :] <= cut_x)
        mask_yz = (np.arange(z_full + 1)[:, None] <= cut_z) & (np.arange(y_full + 1)[None, :] <= cut_y)

        def tex_xy(z_idx: int):
            return _to_qimage_rgba(vol[z_idx, :, :], mask_xy)

        def tex_xz(y_idx: int):
            slice_2d = vol[:, y_idx, :]
            mask_2d = mask_xz
            if z_scale > 1.0:
                slice_2d = upsample_along_z(slice_2d, z_scale)
                mask_2d = upsample_along_z(mask_2d.astype(np.float32), z_scale) >= 0.5
            return _to_qimage_rgba(slice_2d, mask_2d)

        def tex_yz(x_idx: int):
            slice_2d = vol[:, :, x_idx]
            mask_2d = mask_yz
            if z_scale > 1.0:
                slice_2d = upsample_along_z(slice_2d, z_scale)
                mask_2d = upsample_along_z(mask_2d.astype(np.float32), z_scale) >= 0.5
            return _to_qimage_rgba(slice_2d, mask_2d)

        faces = []
        corners = self._projector(x_full, y_full, z_full * z_scale, force_width, force_height)

        def face_depth_xy(z_val: int, x_len: int, y_len: int) -> float:
            _, _, _, d = corners.face_xy(z=z_val, x_len=x_len, y_len=y_len)
            return d

        def face_depth_xz(y_val: int, x_len: int, z_len: int) -> float:
            _, _, _, d = corners.face_xz(y=y_val, x_len=x_len, z_len=z_len)
            return d

        def face_depth_yz(x_val: int, y_len: int, z_len: int) -> float:
            _, _, _, d = corners.face_yz(x=x_val, y_len=y_len, z_len=z_len)
            return d

        def add_face(name, img, p0, p1, p2, depth):
            if img.isNull():
                return
            faces.append(FaceTexture(name, img, p0, p1, p2, depth))

        # XY faces (front/back)
        img_front = tex_xy(cut_z)
        img_back = tex_xy(0)
        p0, p1, p2, _ = corners.face_xy(z=cut_z_proj)
        depth_front = face_depth_xy(cut_z_proj, cut_x, cut_y)
        add_face("front", img_front, p0, p1, p2, depth_front)
        p0, p1, p2, _ = corners.face_xy(z=0)
        depth_back = face_depth_xy(0, cut_x, cut_y)
        add_face("back", img_back, p0, p1, p2, depth_back)

        # XZ faces (top/bottom)
        img_top = tex_xz(0)
        img_bottom = tex_xz(cut_y)
        p0, p1, p2, _ = corners.face_xz(y=0)
        depth_top = face_depth_xz(0, cut_x, cut_z_proj)
        add_face("top", img_top, p0, p1, p2, depth_top)
        p0, p1, p2, _ = corners.face_xz(y=cut_y)
        depth_bottom = face_depth_xz(cut_y, cut_x, cut_z_proj)
        add_face("bottom", img_bottom, p0, p1, p2, depth_bottom)

        # YZ faces (left/right)
        img_left = tex_yz(0)
        img_right = tex_yz(cut_x)
        p0, p1, p2, _ = corners.face_yz(x=0)
        depth_left = face_depth_yz(0, cut_y, cut_z_proj)
        add_face("left", img_left, p0, p1, p2, depth_left)
        p0, p1, p2, _ = corners.face_yz(x=cut_x)
        depth_right = face_depth_yz(cut_x, cut_y, cut_z_proj)
        add_face("right", img_right, p0, p1, p2, depth_right)

        return faces

    def _projector(self, size_x: float, size_y: float, size_z: float, force_width: int = None, force_height: int = None):
        w = force_width if force_width is not None else self.width()
        h = force_height if force_height is not None else self.height()
        return _CubeProjector(w, h, size_x, size_y, size_z, self._azimuth, self._elevation)


class _CubeProjector:
    def __init__(self, width: int, height: int, size_x: float, size_y: float, size_z: float, azimuth: float, elevation: float):
        self.width = max(10, width)
        self.height = max(10, height)
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self._azimuth = azimuth
        self._elevation = elevation
        self._rotation = self._build_rotation()
        self._scale, self._offset = self._fit_to_view()

    def _build_rotation(self) -> np.ndarray:
        az = math.radians(self._azimuth)
        ax = math.radians(self._elevation)
        cz, sz = math.cos(az), math.sin(az)
        cx, sx = math.cos(ax), math.sin(ax)
        rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
        return rx @ rz

    def _fit_to_view(self):
        corners = np.array(
            [
                [0.0, 0.0, 0.0],
                [self.size_x, 0.0, 0.0],
                [0.0, self.size_y, 0.0],
                [0.0, 0.0, self.size_z],
                [self.size_x, self.size_y, 0.0],
                [self.size_x, 0.0, self.size_z],
                [0.0, self.size_y, self.size_z],
                [self.size_x, self.size_y, self.size_z],
            ]
        )
        proj = (self._rotation @ corners.T).T
        min_xy = proj[:, :2].min(axis=0)
        max_xy = proj[:, :2].max(axis=0)
        span = max_xy - min_xy
        margin = 20.0
        scale = min(
            (self.width - 2 * margin) / max(span[0], 1e-6),
            (self.height - 2 * margin) / max(span[1], 1e-6),
        )
        extra_x = (self.width - 2 * margin) - span[0] * scale
        extra_y = (self.height - 2 * margin) - span[1] * scale
        offset = np.array([margin + extra_x * 0.5, margin + extra_y * 0.5]) - min_xy * scale
        return scale, offset

    def _project_point(self, point: Tuple[float, float, float]):
        p = np.array(point)
        proj = self._rotation @ p
        xy = proj[:2] * self._scale + self._offset
        return xy[0], xy[1], proj[2]

    def face_xy(self, z: float, x_len: float = None, y_len: float = None):
        if x_len is None:
            x_len = self.size_x
        if y_len is None:
            y_len = self.size_y
        p0 = self._project_point((0.0, 0.0, z))
        p1 = self._project_point((x_len, 0.0, z))
        p2 = self._project_point((0.0, y_len, z))
        depth = (p0[2] + p1[2] + p2[2]) / 3.0
        return (p0[0], p0[1]), (p1[0], p1[1]), (p2[0], p2[1]), depth

    def face_xz(self, y: float, x_len: float = None, z_len: float = None):
        if x_len is None:
            x_len = self.size_x
        if z_len is None:
            z_len = self.size_z
        p0 = self._project_point((0.0, y, 0.0))
        p1 = self._project_point((x_len, y, 0.0))
        p2 = self._project_point((0.0, y, z_len))
        depth = (p0[2] + p1[2] + p2[2]) / 3.0
        return (p0[0], p0[1]), (p1[0], p1[1]), (p2[0], p2[1]), depth

    def face_yz(self, x: float, y_len: float = None, z_len: float = None):
        if y_len is None:
            y_len = self.size_y
        if z_len is None:
            z_len = self.size_z
        p0 = self._project_point((x, 0.0, 0.0))
        p1 = self._project_point((x, y_len, 0.0))
        p2 = self._project_point((x, 0.0, z_len))
        depth = (p0[2] + p1[2] + p2[2]) / 3.0
        return (p0[0], p0[1]), (p1[0], p1[1]), (p2[0], p2[1]), depth
