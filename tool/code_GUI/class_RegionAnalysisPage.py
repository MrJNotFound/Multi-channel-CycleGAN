import os
from typing import List, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QFileDialog, QSlider, QGroupBox,
    QListWidget, QProgressBar
)
from class_ImageViewer import InteractiveImageLabel
from func_cell_tracking import analyze_layer_iou_by_paths
from func_img_segmentation import kmeans_watershed_segmentation, filter_regions
from func_tools import show_info


def get_resize_shape(src: np.ndarray, max_side: int = 1000) -> Tuple[Tuple[int, int], float]:
    h, w = src.shape[:2]
    scale = max_side / max(h, w) if max(h, w) > max_side else 1.0
    return (int(w * scale), int(h * scale)), scale

def resize_if_needed(img: np.ndarray, max_side: int = 1000) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img.copy(), 1.0
    (nw, nh), scale = get_resize_shape(img, max_side)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR if img.ndim == 3 else cv2.INTER_NEAREST)
    return resized, scale

def upsample_to_shape(small: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

def parula_colormap(val, vmin=0, vmax=1):
    normed = np.clip((val - vmin) / (vmax - vmin), 0, 1)
    # noinspection PyBroadException
    try:
        cmap = plt.get_cmap("parula")
    except Exception:
        cmap = plt.get_cmap("viridis")
    colors = cmap(normed)
    return (colors[..., :3] * 255).astype(np.uint8)

class SegmentWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(list)
    interrupted = pyqtSignal()
    message = pyqtSignal(str)
    def __init__(self, image_list, save_dir):
        super().__init__()
        self.image_list = image_list
        self.save_dir = save_dir
        self._cancel = False

    def run(self):
        saved_paths = []
        for i, img_path in enumerate(self.image_list):
            if self._cancel:
                self.interrupted.emit()  # type: ignore
                return
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                self.message.emit(f"无法读取: {img_path}")  # type: ignore
                continue
            try:
                markers = np.uint8(kmeans_watershed_segmentation(img))
                name = os.path.splitext(os.path.basename(img_path))[0]
                save_path = os.path.join(self.save_dir, f"{name}_mask.png")
                cv2.imwrite(save_path, markers)
                saved_paths.append((img_path, save_path))
            except Exception as e:
                self.message.emit(f"{img_path} 分割失败: {e}")  # type: ignore
                continue
            self.progress.emit(i+1)  # type: ignore
        self.finished.emit(saved_paths)  # type: ignore

    def cancel(self):
        self._cancel = True

class FilterWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(list)
    interrupted = pyqtSignal()
    message = pyqtSignal(str)
    def __init__(self, label_paths, save_dir, low, up, circ, thumb_shape, full_shape):
        super().__init__()
        self.label_paths = label_paths
        self.save_dir = save_dir
        self.low = low
        self.up = up
        self.circ = circ
        self.thumb_shape = thumb_shape
        self.full_shape = full_shape
        self._cancel = False

    def run(self):
        saved_paths = []
        if self.thumb_shape is not None and self.full_shape is not None:
            thumb_area = self.thumb_shape[0] * self.thumb_shape[1]
            full_area = self.full_shape[0] * self.full_shape[1]
            scale = full_area / thumb_area
        else:
            scale = 1.0
        for i, label_path in enumerate(self.label_paths):
            if self._cancel:
                self.interrupted.emit()  # type: ignore
                return
            lbl = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            if lbl is None:
                self.message.emit(f"无法读取: {label_path}")  # type: ignore
                continue
            if lbl.ndim == 3:
                lbl = cv2.cvtColor(lbl, cv2.COLOR_BGR2GRAY)
            try:
                low_full = int(self.low * scale)
                up_full = int(self.up * scale)
                filtered = filter_regions(lbl, area_threshold_low=low_full, area_threshold_up=up_full, circularity_threshold=self.circ)
                save_img = filtered.astype(np.uint8)
                name = os.path.splitext(os.path.basename(label_path))[0]
                save_path = os.path.join(self.save_dir, f"{name}_filtered.png")
                cv2.imwrite(save_path, save_img)
                saved_paths.append((label_path, save_path))
            except Exception as e:
                self.message.emit(f"{label_path} 筛选失败: {e}")  # type: ignore
                continue
            self.progress.emit(i+1)  # type: ignore
        self.finished.emit(saved_paths)  # type: ignore

    def cancel(self):
        self._cancel = True

class RegionAnalysisPage(QWidget):
    def __init__(self):
        super().__init__()
        self.image_list: List[str] = []
        self.current_index: int = -1
        self.label_image_paths: List[str] = []
        self.label_current_index: int = -1

        self.src_img: Optional[np.ndarray] = None
        self.label_img: Optional[np.ndarray] = None

        self.thumb_img: Optional[np.ndarray] = None
        self.thumb_label: Optional[np.ndarray] = None
        self.thumb_filtered_label: Optional[np.ndarray] = None
        self.thumb_scale: float = 1.0

        self.iou_scores: Optional[np.ndarray] = None
        self.iou_map: Optional[np.ndarray] = None
        self.iou_threshold: float = 0.0

        self.show_overlay: bool = True
        self.btn_toggle_overlay: Optional[QPushButton] = None
        self.btn_toggle_colorbar: Optional[QPushButton] = None
        self.progress_bar: Optional[QProgressBar] = None

        self._suppress_selection_signals: bool = False
        self.preview_max_side = 1000

        self.segment_worker: Optional[SegmentWorker] = None
        self.filter_worker: Optional[FilterWorker] = None

        self._last_iou_result = None
        self._last_image_idx = -1

        self.colorbar_label: Optional[QLabel] = None
        self.colorbar_img_path = './icons/color bar.png'
        self.colorbar_fixed_width = 100
        self.colorbar_fixed_height = 500
        self._colorbar_showing = False

        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        ctrl = QVBoxLayout()
        ctrl_widget = QWidget()
        ctrl_widget.setLayout(ctrl)
        ctrl_widget.setMinimumWidth(200)
        ctrl_widget.setMaximumWidth(230)

        # 分割
        seg_group = QGroupBox("分割")
        seg_layout = QVBoxLayout()

        btn_load = QPushButton("选择原始图像")
        btn_load.clicked.connect(self.on_load)  # type: ignore
        seg_layout.addWidget(btn_load)

        btn_clear_image_list = QPushButton("清除原图列表")
        btn_clear_image_list.clicked.connect(self.on_clear_image_list)  # type: ignore
        seg_layout.addWidget(btn_clear_image_list)

        btn_segment = QPushButton("批量分割")
        btn_segment.clicked.connect(self.on_batch_segment_async)  # type: ignore
        seg_layout.addWidget(btn_segment)

        seg_group.setLayout(seg_layout)
        ctrl.addWidget(seg_group)

        # 筛选
        fil_group = QGroupBox("标签筛选")
        fil_layout = QVBoxLayout()

        btn_choose_label = QPushButton("选择标签图像")
        btn_choose_label.clicked.connect(self.on_choose_label_images)  # type: ignore
        fil_layout.addWidget(btn_choose_label)

        btn_clear_label_list = QPushButton("清除标签列表")
        btn_clear_label_list.clicked.connect(self.on_clear_label_list)  # type: ignore
        fil_layout.addWidget(btn_clear_label_list)

        self.sld_area_low = QSlider(Qt.Orientation.Horizontal)
        self.sld_area_low.setRange(0, 100)
        self.sld_area_low.setValue(10)
        self.lbl_area_low = QLabel("面积下限: 10")
        self.sld_area_low.valueChanged.connect(lambda v: self.lbl_area_low.setText(f"面积下限: {v}"))  # type: ignore
        self.sld_area_low.valueChanged.connect(self._on_filter_param_changed)  # type: ignore
        fil_layout.addWidget(self.lbl_area_low)
        fil_layout.addWidget(self.sld_area_low)

        self.sld_area_up = QSlider(Qt.Orientation.Horizontal)
        self.sld_area_up.setRange(1000, 1000000)
        self.sld_area_up.setValue(5000)
        self.lbl_area_up = QLabel("面积上限: 5000")
        self.sld_area_up.valueChanged.connect(lambda v: self.lbl_area_up.setText(f"面积上限: {v}"))  # type: ignore
        self.sld_area_up.valueChanged.connect(self._on_filter_param_changed)  # type: ignore
        fil_layout.addWidget(self.lbl_area_up)
        fil_layout.addWidget(self.sld_area_up)

        self.sld_circ = QSlider(Qt.Orientation.Horizontal)
        self.sld_circ.setRange(0, 100)
        self.sld_circ.setValue(30)
        self.lbl_circ = QLabel("圆度阈值: 0.30")
        self.sld_circ.valueChanged.connect(lambda v: self.lbl_circ.setText(f"圆度阈值: {v/100:.2f}"))  # type: ignore
        self.sld_circ.valueChanged.connect(self._on_filter_param_changed)  # type: ignore
        fil_layout.addWidget(self.lbl_circ)
        fil_layout.addWidget(self.sld_circ)

        btn_filter = QPushButton("批量筛选")
        btn_filter.clicked.connect(self.on_batch_filter_async)  # type: ignore
        fil_layout.addWidget(btn_filter)

        fil_group.setLayout(fil_layout)
        ctrl.addWidget(fil_group)

        # IOU分析
        iou_group = QGroupBox("IOU连续性分析")
        iou_layout = QVBoxLayout()

        btn_iou = QPushButton("执行连续性分析")
        btn_iou.clicked.connect(self.on_iou_analysis)  # type: ignore
        iou_layout.addWidget(btn_iou)

        self.sld_iou = QSlider(Qt.Orientation.Horizontal)
        self.sld_iou.setRange(0, 100)
        self.sld_iou.setValue(0)
        self.sld_iou.setSingleStep(1)
        self.lbl_iou = QLabel("连续性阈值: 0.00")
        self.sld_iou.valueChanged.connect(lambda v: self.lbl_iou.setText(f"连续性阈值: {v/100:.2f}"))  # type: ignore
        self.sld_iou.valueChanged.connect(self._on_iou_thresh_changed)  # type: ignore
        iou_layout.addWidget(self.lbl_iou)
        iou_layout.addWidget(self.sld_iou)

        btn_save_iou = QPushButton("批量保存结果")
        btn_save_iou.clicked.connect(self.on_batch_save_iou)  # type: ignore
        iou_layout.addWidget(btn_save_iou)

        iou_group.setLayout(iou_layout)
        ctrl.addWidget(iou_group)
        ctrl.addStretch(1)

        # 中部文件列表
        middle_layout = QVBoxLayout()
        middle_layout.addWidget(QLabel("原始图像列表"))
        self.list_widget = QListWidget()
        self.list_widget.setMaximumWidth(150)
        self.list_widget.setMinimumWidth(100)
        self.list_widget.currentRowChanged.connect(self._on_image_selected)  # type: ignore
        middle_layout.addWidget(self.list_widget)
        self.lbl_image_info = QLabel("未选择原始图像")
        self.lbl_image_info.setStyleSheet("font-size: 10px; color: #666;")
        self.lbl_image_info.setWordWrap(True)
        middle_layout.addWidget(self.lbl_image_info)
        middle_layout.addWidget(QLabel("标签图像列表"))
        self.label_list_widget = QListWidget()
        self.label_list_widget.setMaximumWidth(150)
        self.label_list_widget.setMinimumWidth(100)
        self.label_list_widget.currentRowChanged.connect(self._on_label_selected)  # type: ignore
        middle_layout.addWidget(self.label_list_widget)
        self.lbl_label_info = QLabel("未选择标签图像")
        self.lbl_label_info.setStyleSheet("font-size: 10px; color: #666;")
        self.lbl_label_info.setWordWrap(True)
        middle_layout.addWidget(self.lbl_label_info)

        # 右侧预览区
        preview_layout = QHBoxLayout()
        preview_inner_layout = QVBoxLayout()
        header_layout = QHBoxLayout()
        preview_title = QLabel("区域分析预览")
        preview_title.setStyleSheet("font-weight: bold; font-size: 14px; margin: 5px 0;")
        header_layout.addWidget(preview_title)
        header_layout.addStretch()
        self.btn_toggle_overlay = QPushButton("显示标签")
        self.btn_toggle_overlay.setCheckable(True)
        self.btn_toggle_overlay.setChecked(True)
        self.btn_toggle_overlay.setToolTip("点击切换原图与叠加图显示")
        self.btn_toggle_overlay.toggled.connect(self._on_overlay_toggled)  # type: ignore
        header_layout.addWidget(self.btn_toggle_overlay)
        self.btn_toggle_colorbar = QPushButton("color bar")
        self.btn_toggle_colorbar.setCheckable(True)
        self.btn_toggle_colorbar.setChecked(False)
        self.btn_toggle_colorbar.setToolTip("显示/隐藏色条")
        self.btn_toggle_colorbar.toggled.connect(self._on_colorbar_toggled)  # type: ignore
        header_layout.addWidget(self.btn_toggle_colorbar)
        preview_inner_layout.addLayout(header_layout)

        self.interactive_image = InteractiveImageLabel()
        preview_inner_layout.addWidget(self.interactive_image, 1)
        tips_layout = QHBoxLayout()
        tips_layout.setContentsMargins(0, 2, 0, 5)
        tips_label = QLabel("滚轮缩放 | 左键拖拽 | 双击重置")
        tips_label.setStyleSheet("color: #666; font-size: 10px;")
        tips_layout.addWidget(tips_label)
        tips_layout.addStretch()
        preview_inner_layout.addLayout(tips_layout)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        preview_inner_layout.addWidget(self.progress_bar)

        # color bar
        colorbar_outer_layout = QVBoxLayout()
        colorbar_outer_layout.addStretch(1)
        self.colorbar_label = QLabel()
        self.colorbar_label.setVisible(False)
        self.colorbar_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.colorbar_label.setFixedWidth(self.colorbar_fixed_width)
        colorbar_outer_layout.addWidget(self.colorbar_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        colorbar_outer_layout.addStretch(1)

        preview_layout.addLayout(preview_inner_layout, 1)
        preview_layout.addLayout(colorbar_outer_layout, 0)
        layout.addWidget(ctrl_widget, 0)
        layout.addLayout(middle_layout, 0)
        layout.addLayout(preview_layout, 1)

    def _on_colorbar_toggled(self, checked: bool):
        self._colorbar_showing = checked
        self._update_colorbar(self._colorbar_showing)
        self._update_preview()

    def on_batch_segment_async(self):
        if self.segment_worker and self.segment_worker.isRunning():
            self.segment_worker.cancel()
            return
        if not self.image_list:
            show_info(self, "请先选择原始图像")
            return
        save_dir = QFileDialog.getExistingDirectory(self, "选择批量保存目录")
        if not save_dir:
            return
        self.progress_bar.setRange(0, len(self.image_list))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.segment_worker = SegmentWorker(self.image_list, save_dir)
        self.segment_worker.progress.connect(self.progress_bar.setValue)  # type: ignore
        self.segment_worker.message.connect(lambda msg: show_info(self, msg))  # type: ignore
        self.segment_worker.finished.connect(self._on_segment_done)  # type: ignore
        self.segment_worker.interrupted.connect(self._on_segment_cancelled)  # type: ignore
        self.segment_worker.start()

    def _on_segment_done(self, saved_paths):
        self.progress_bar.setVisible(False)
        show_info(self, f"分割完成: {len(saved_paths)}张")
        for _, mask_path in saved_paths:
            self._add_label_to_list(mask_path)
        self.segment_worker = None

    def _on_segment_cancelled(self):
        self.progress_bar.setVisible(False)
        show_info(self, "分割已中断")
        self.segment_worker = None

    def on_batch_filter_async(self):
        if self.filter_worker and self.filter_worker.isRunning():
            self.filter_worker.cancel()
            return
        if not self.label_image_paths:
            show_info(self, "请先指定标签图像")
            return
        save_dir = QFileDialog.getExistingDirectory(self, "选择批量筛选保存目录")
        if not save_dir:
            return
        low = int(self.sld_area_low.value())
        up = int(self.sld_area_up.value())
        circ = self.sld_circ.value() / 100
        if self.label_image_paths:
            first_path = self.label_image_paths[0]
            lbl_full = cv2.imread(first_path, cv2.IMREAD_UNCHANGED)
            if lbl_full is not None and lbl_full.ndim == 3:
                lbl_full = cv2.cvtColor(lbl_full, cv2.COLOR_BGR2GRAY)
            thumb, _ = resize_if_needed(lbl_full, self.preview_max_side)
            thumb_shape = thumb.shape[:2] if thumb is not None else None
            full_shape = lbl_full.shape[:2] if lbl_full is not None else None
        else:
            thumb_shape = None
            full_shape = None
        self.progress_bar.setRange(0, len(self.label_image_paths))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.filter_worker = FilterWorker(self.label_image_paths, save_dir, low, up, circ, thumb_shape, full_shape)
        self.filter_worker.progress.connect(self.progress_bar.setValue)  # type: ignore
        self.filter_worker.message.connect(lambda msg: show_info(self, msg))  # type: ignore
        self.filter_worker.finished.connect(self._on_filter_done)  # type: ignore
        self.filter_worker.interrupted.connect(self._on_filter_cancelled)  # type: ignore
        self.filter_worker.start()

    def _on_filter_done(self, saved_paths):
        self.progress_bar.setVisible(False)
        show_info(self, f"筛选完成: {len(saved_paths)}张")
        self.on_clear_label_list()
        for _, filtered_path in saved_paths:
            self._add_label_to_list(filtered_path)
        self.filter_worker = None

    def _on_filter_cancelled(self):
        self.progress_bar.setVisible(False)
        show_info(self, "筛选已中断")
        self.filter_worker = None

    def on_load(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择原始图像",
            filter="Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not paths:
            return
        self.image_list = list(paths)
        self.list_widget.clear()
        for p in self.image_list:
            self.list_widget.addItem(os.path.basename(p))
        self.current_index = 0 if self.image_list else -1
        self.lbl_image_info.setText("未选择原始图像")
        self._clear_states()
        if self.current_index != -1:
            self._set_current_pair_index(0)

    def on_clear_image_list(self):
        self.image_list.clear()
        self.list_widget.clear()
        self.current_index = -1
        self.src_img = None
        self.thumb_img = None
        self.thumb_label = None
        self.thumb_filtered_label = None
        self.iou_scores = None
        self.lbl_image_info.setText("未选择原始图像")
        self._update_preview(force_reset=True)

    def on_clear_label_list(self):
        self.label_image_paths.clear()
        self.label_list_widget.clear()
        self.label_current_index = -1
        self.label_img = None
        self.thumb_label = None
        self.thumb_filtered_label = None
        self.iou_scores = None
        self.lbl_label_info.setText("未选择标签图像")
        self._update_preview(force_reset=True)

    def _on_image_selected(self, row: int):
        if self._suppress_selection_signals:
            return
        self._set_current_pair_index(row)

    def _on_label_selected(self, row: int):
        if self._suppress_selection_signals:
            return
        self._last_iou_result = None
        self._set_current_pair_index(row)

    def _clear_states(self):
        self.src_img = None
        self.label_img = None
        self.thumb_img = None
        self.thumb_label = None
        self.thumb_filtered_label = None
        self.iou_scores = None

    def _set_current_pair_index(self, index: int):
        self.current_index = index if 0 <= index < len(self.image_list) else -1
        self.label_current_index = index if 0 <= index < len(self.label_image_paths) else -1
        self._suppress_selection_signals = True
        try:
            if self.current_index != -1 and self.list_widget.currentRow() != self.current_index:
                self.list_widget.setCurrentRow(self.current_index)
            if self.label_current_index != -1 and self.label_list_widget.currentRow() != self.label_current_index:
                self.label_list_widget.setCurrentRow(self.label_current_index)
        finally:
            self._suppress_selection_signals = False
        self._load_pair_images(self.current_index)
        view_reset = (self.current_index != self._last_image_idx)
        self._last_image_idx = self.current_index
        self._update_preview(force_reset=view_reset)

    def _load_pair_images(self, index: int):
        self._clear_states()
        if 0 <= index < len(self.image_list):
            img_path = self.image_list[index]
            self.src_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            self.lbl_image_info.setText(f"文件: {os.path.basename(img_path)}")
            self.thumb_img, self.thumb_scale = resize_if_needed(self.src_img, self.preview_max_side)
        else:
            self.lbl_image_info.setText("未选择原始图像")
            self.thumb_img = None
            self.thumb_scale = 1.0
        if 0 <= index < len(self.label_image_paths):
            lbl_path = self.label_image_paths[index]
            lbl = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
            if lbl is not None and lbl.ndim == 3:
                lbl = cv2.cvtColor(lbl, cv2.COLOR_BGR2GRAY)
            self.label_img = lbl
            self.lbl_label_info.setText(f"标签文件: {os.path.basename(lbl_path)}")
            if self.thumb_img is not None and self.label_img is not None:
                self.thumb_label = cv2.resize(self.label_img, (self.thumb_img.shape[1], self.thumb_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                self._update_filtered_label()
        else:
            self.lbl_label_info.setText("未选择标签图像")
            self.thumb_label = None
            self.thumb_filtered_label = None

    def _add_label_to_list(self, png_path):
        if png_path not in self.label_image_paths:
            self.label_image_paths.append(png_path)
            self.label_list_widget.addItem(os.path.basename(png_path))
            if len(self.label_image_paths) == 1:
                self._set_current_pair_index(0)

    def _on_filter_param_changed(self, *_):
        if self.thumb_label is not None:
            self._update_filtered_label()

    def _update_filtered_label(self):
        if self.thumb_label is None:
            self.thumb_filtered_label = None
            self._update_preview(force_reset=False)
            return
        low = int(self.sld_area_low.value())
        up = int(self.sld_area_up.value())
        if low > up:
            low = up
            self.sld_area_low.setValue(up)
        circ = self.sld_circ.value() / 100
        try:
            self.thumb_filtered_label = filter_regions(
                self.thumb_label,
                area_threshold_low=low,
                area_threshold_up=up,
                circularity_threshold=circ
            )
        except Exception as e:
            show_info(self, f"筛选失败: {e}")
            self.thumb_filtered_label = self.thumb_label
        self._update_preview(force_reset=False)

    def _on_overlay_toggled(self, checked: bool):
        self.show_overlay = checked
        self._update_preview(force_reset=False)

    def _update_colorbar(self, show: bool):
        if self.colorbar_label is None:
            return
        if show and os.path.exists(self.colorbar_img_path):
            qpix = QPixmap(self.colorbar_img_path)
            target_width = self.colorbar_fixed_width
            target_height = self.colorbar_fixed_height
            scale_factor = min(target_width / qpix.width(), target_height / qpix.height(), 1.0)
            new_w, new_h = int(qpix.width() * scale_factor), int(qpix.height() * scale_factor)
            scaled_pix = qpix.scaled(new_w, new_h, Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
            self.colorbar_label.setPixmap(scaled_pix)
            self.colorbar_label.setFixedWidth(target_width)
            self.colorbar_label.setFixedHeight(target_height)
            self.colorbar_label.setVisible(True)
        else:
            self.colorbar_label.clear()
            self.colorbar_label.setVisible(False)

    def _update_preview(self, force_reset=False):
        img = None
        show_overlay = self.btn_toggle_overlay.isChecked() if self.btn_toggle_overlay is not None else True
        show_colorbar = self._colorbar_showing

        if self._last_iou_result is not None:
            region_iou_map, thumb_label, thumb_img, color_map = self._last_iou_result
            iou_thresh = self.sld_iou.value() / 100
            mask = (region_iou_map >= iou_thresh) & (thumb_label > 0)
            overlay = thumb_img.copy()
            overlay[mask] = color_map[mask].astype(np.uint8)
            img = overlay if show_overlay else thumb_img
        elif self.thumb_filtered_label is not None and self.thumb_img is not None:
            img = self._make_mask_overlay_preview(self.thumb_img, self.thumb_filtered_label) if show_overlay else self.thumb_img
        elif self.thumb_label is not None and self.thumb_img is not None:
            img = self._make_mask_overlay_preview(self.thumb_img, self.thumb_label) if show_overlay else self.thumb_img
        elif self.thumb_img is not None:
            img = self.thumb_img

        if img is not None:
            self.interactive_image.set_image(img, reset_view=force_reset)
        else:
            self.interactive_image.set_image(None, reset_view=True)

        self._update_colorbar(show_colorbar)

    @staticmethod
    def _make_mask_overlay_preview(src_thumb: np.ndarray, label_thumb: np.ndarray) -> np.ndarray:
        if np.count_nonzero(label_thumb) == 0:
            return src_thumb
        mask = (label_thumb > 0)
        out = src_thumb.copy()
        out[mask] = (0, 255, 0)
        return out

    def on_choose_label_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "选择标签图像(支持png)", filter="PNG (*.png)")
        if not files:
            return
        self.label_image_paths = list(files)
        self.label_list_widget.clear()
        for path in self.label_image_paths:
            self.label_list_widget.addItem(os.path.basename(path))
        self.label_current_index = 0 if self.label_image_paths else -1
        if self.label_current_index != -1:
            self._set_current_pair_index(0)

    def on_iou_analysis(self):
        if not self.label_image_paths or self.label_current_index == -1:
            show_info(self, "请先选择标签图像")
            return

        self.sld_area_low.blockSignals(True)
        self.sld_area_up.blockSignals(True)
        self.sld_circ.blockSignals(True)
        self.sld_area_low.setValue(self.sld_area_low.minimum())
        self.lbl_area_low.setText(f"面积下限: {self.sld_area_low.value()}")
        self.sld_area_up.setValue(self.sld_area_up.maximum())
        self.lbl_area_up.setText(f"面积上限: {self.sld_area_up.value()}")
        self.sld_circ.setValue(self.sld_circ.minimum())
        self.lbl_circ.setText(f"圆度阈值: {self.sld_circ.value() / 100:.2f}")
        self.sld_area_low.blockSignals(False)
        self.sld_area_up.blockSignals(False)
        self.sld_circ.blockSignals(False)

        self._update_filtered_label()

        idx = self.label_current_index
        label_paths = self.label_image_paths
        label_img = cv2.imread(label_paths[idx], cv2.IMREAD_UNCHANGED)
        if label_img is None:
            show_info(self, "无法读取标签图像")
            return
        if label_img.ndim == 3:
            label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
        try:
            iou_scores = np.array(analyze_layer_iou_by_paths(label_paths, idx))
        except Exception as e:
            show_info(self, f"IOU分析失败: {e}")
            return
        if idx < len(self.image_list):
            src_img = cv2.imread(self.image_list[idx], cv2.IMREAD_COLOR)
        else:
            src_img = np.zeros((label_img.shape[0], label_img.shape[1], 3), dtype=np.uint8)
        thumb_img, _ = resize_if_needed(src_img, self.preview_max_side)
        thumb_label = cv2.resize(label_img, (thumb_img.shape[1], thumb_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        region_ids = np.unique(thumb_label)
        region_ids = region_ids[region_ids > 0]
        region_iou_map = np.zeros_like(thumb_label, dtype=np.float32)
        for i, region_id in enumerate(region_ids):
            if i < len(iou_scores):
                region_iou_map[thumb_label == region_id] = iou_scores[i]
        color_map = parula_colormap(region_iou_map, vmin=0, vmax=1)
        self._last_iou_result = (region_iou_map, thumb_label, thumb_img, color_map)
        self._colorbar_showing = True
        self.btn_toggle_colorbar.setChecked(True)
        self._update_preview(force_reset=False)

    def _on_iou_thresh_changed(self):
        self._update_preview(force_reset=False)

    def on_batch_save_iou(self):
        save_dir = QFileDialog.getExistingDirectory(self, "选择IOU分析保存目录")
        if not save_dir:
            return
        iou_thresh = self.sld_iou.value() / 100
        self.progress_bar.setRange(0, len(self.label_image_paths))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        for idx, label_path in enumerate(self.label_image_paths):
            label_img = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            if label_img is None:
                show_info(self, f"无法读取: {label_path}")
                self.progress_bar.setValue(idx + 1)
                continue
            if label_img.ndim == 3:
                label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
            try:
                low = int(self.sld_area_low.value())
                up = int(self.sld_area_up.value())
                circ = self.sld_circ.value() / 100
                filtered_label = filter_regions(
                    label_img,
                    area_threshold_low=low,
                    area_threshold_up=up,
                    circularity_threshold=circ
                )
                unique_ids = np.unique(filtered_label)
                unique_ids = unique_ids[unique_ids > 0]
                remap = {oid: idx + 1 for idx, oid in enumerate(unique_ids)}
                relabeled = np.zeros_like(filtered_label, dtype=np.uint16)
                for oid, nid in remap.items():
                    relabeled[filtered_label == oid] = nid
                filtered_save_path = os.path.join(
                    save_dir,
                    f"{os.path.splitext(os.path.basename(label_path))[0]}_filtered_relabel.png"
                )
                cv2.imwrite(filtered_save_path, relabeled.astype(np.uint16))

                iou_scores = np.array(analyze_layer_iou_by_paths(self.label_image_paths, idx))
            except Exception as e:
                show_info(self, f"{label_path} IOU分析失败: {e}")
                self.progress_bar.setValue(idx + 1)
                continue

            if idx < len(self.image_list):
                src_img = cv2.imread(self.image_list[idx], cv2.IMREAD_COLOR)
            else:
                src_img = np.zeros((label_img.shape[0], label_img.shape[1], 3), dtype=np.uint8)
            thumb_img, _ = resize_if_needed(src_img, self.preview_max_side)
            thumb_label = cv2.resize(relabeled, (thumb_img.shape[1], thumb_img.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)

            region_ids = np.unique(thumb_label)
            region_ids = region_ids[region_ids > 0]
            region_iou_map = np.zeros_like(thumb_label, dtype=np.float32)
            for i, region_id in enumerate(region_ids):
                if i < len(iou_scores):
                    region_iou_map[thumb_label == region_id] = iou_scores[i]
            mask = (region_iou_map >= iou_thresh) & (thumb_label > 0)
            color_map = parula_colormap(region_iou_map, vmin=0, vmax=1)
            overlay = thumb_img.copy()
            overlay[mask] = color_map[mask].astype(np.uint8)
            save_path = os.path.join(
                save_dir,
                f"{os.path.splitext(os.path.basename(label_path))[0]}_iou_analysis.png"
            )
            cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            self.progress_bar.setValue(idx + 1)

        self.progress_bar.setVisible(False)
        show_info(self, "批量IOU分析保存完成")