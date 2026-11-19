import os
from functools import partial
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QFileDialog, QSlider, QGroupBox,
    QListWidget, QProgressBar
)
from ...processing.images import fit_preview, preview_geometry, read_image
from ...processing.region_analysis import (
    batch_process, filter_image, filter_regions, iou_preview, read_labels, save_iou, segment_image,
)
from .. import ICON_DIR
from ..tasks import TaskThread, stop_tasks
from ..widgets import InteractiveImageLabel, show_info


class RegionAnalysisPage(QWidget):
    def __init__(self):
        super().__init__()
        self.image_list: List[str] = []
        self.current_index: int = -1
        self.label_image_paths: List[str] = []
        self.label_current_index: int = -1


        self.thumb_img: Optional[np.ndarray] = None
        self.thumb_label: Optional[np.ndarray] = None
        self.thumb_filtered_label: Optional[np.ndarray] = None


        self.btn_toggle_overlay: Optional[QPushButton] = None
        self.btn_toggle_colorbar: Optional[QPushButton] = None
        self.progress_bar: Optional[QProgressBar] = None

        self._suppress_selection_signals: bool = False
        self.preview_max_side = 1000

        self.worker: TaskThread | None = None
        self._task_kind = None

        self._last_iou_result = None
        self._last_image_idx = -1

        self.colorbar_label: Optional[QLabel] = None
        self.colorbar_img_path = str(ICON_DIR / 'iou_colorbar.png')
        self.colorbar_fixed_width = 100
        self.colorbar_fixed_height = 500
        self._colorbar_showing = False

        self._build_ui()
        QCoreApplication.instance().aboutToQuit.connect(self._stop_worker)

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

    def _start_task(self, kind, count, work, on_success):
        self.progress_bar.setRange(0, count)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        worker = TaskThread(work, self)
        self.worker = worker
        self._task_kind = kind
        worker.progress.connect(self.progress_bar.setValue)
        worker.finished.connect(lambda: self._finish_task(kind, on_success))
        worker.start()

    def _finish_task(self, kind, on_success):
        worker = self.worker
        self.worker = self._task_kind = None
        self.progress_bar.hide()
        worker.deleteLater()
        if worker.error:
            show_info(self, f"{kind}失败：{worker.error}")
        elif worker.cancelled:
            show_info(self, f"{kind}已中断，已写入的结果已保存在所选目录。")
        else:
            on_success(worker.result)

    def _stop_worker(self):
        stop_tasks(self.worker)

    def _start_region_batch(self, kind, paths, operation, on_saved=None):
        def work(progress, check_interrupted):
            return batch_process(paths, operation, progress=progress, check_interrupted=check_interrupted)

        def done(result):
            saved, errors = result
            if on_saved is not None and saved:
                on_saved(saved)
            message = f"{kind}完成。已保存 {len(saved)} 张。"
            if errors:
                message += f"\n失败 {len(errors)} 张。\n" + "\n".join(errors)
            show_info(self, message)

        self._start_task(kind, len(paths), work, done)

    def _busy(self, kind):
        if self.worker is None:
            return False
        if self._task_kind == kind:
            self.worker.requestInterruption()
        else:
            show_info(self, f"请等待{self._task_kind}完成。")
        return True

    def on_batch_segment_async(self):
        if self._busy("分割"):
            return
        paths = tuple(self.image_list)
        if not paths:
            show_info(self, "请先选择原始图像")
            return
        destination = QFileDialog.getExistingDirectory(self, "选择批量保存目录")
        if not destination:
            return

        def add_labels(saved):
            if tuple(self.image_list) == paths:
                for path in saved:
                    self._add_label_to_list(str(path))

        self._start_region_batch(
            "分割", paths, lambda _index, path: segment_image(path, Path(destination)), add_labels,
        )

    def on_batch_filter_async(self):
        if self._busy("筛选"):
            return
        paths = tuple(self.label_image_paths)
        if not paths:
            show_info(self, "请先指定标签图像")
            return
        destination = QFileDialog.getExistingDirectory(self, "选择批量筛选保存目录")
        if not destination:
            return
        try:
            labels = read_labels(paths[0])
        except ValueError as error:
            show_info(self, str(error))
            return
        (width, height), _ = preview_geometry(labels.shape, self.preview_max_side)
        scale = labels.size / (width * height)
        low = int(self.sld_area_low.value() * scale)
        high = int(self.sld_area_up.value() * scale)
        circularity = self.sld_circ.value() / 100

        def replace_labels(saved):
            if tuple(self.label_image_paths) == paths:
                self.on_clear_label_list()
                for path in saved:
                    self._add_label_to_list(str(path))

        self._start_region_batch(
            "筛选", paths,
            lambda _index, path: filter_image(path, Path(destination), low, high, circularity),
            replace_labels,
        )

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
        self.thumb_img = None
        self.thumb_label = None
        self.thumb_filtered_label = None
        self.lbl_image_info.setText("未选择原始图像")
        self._update_preview(force_reset=True)

    def on_clear_label_list(self):
        self.label_image_paths.clear()
        self.label_list_widget.clear()
        self.label_current_index = -1
        self.thumb_label = None
        self.thumb_filtered_label = None
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
        self.thumb_img = None
        self.thumb_label = None
        self.thumb_filtered_label = None

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
            source = read_image(img_path, cv2.IMREAD_COLOR)
            self.lbl_image_info.setText(f"文件: {os.path.basename(img_path)}")
            self.thumb_img = fit_preview(source, self.preview_max_side, interpolation=cv2.INTER_LINEAR)
        else:
            self.lbl_image_info.setText("未选择原始图像")
            self.thumb_img = None
        if 0 <= index < len(self.label_image_paths):
            lbl_path = self.label_image_paths[index]
            lbl = read_image(lbl_path, cv2.IMREAD_UNCHANGED)
            if lbl is not None and lbl.ndim == 3:
                lbl = cv2.cvtColor(lbl, cv2.COLOR_BGR2GRAY)
            self.lbl_label_info.setText(f"标签文件: {os.path.basename(lbl_path)}")
            if self.thumb_img is not None and lbl is not None:
                self.thumb_label = cv2.resize(lbl, (self.thumb_img.shape[1], self.thumb_img.shape[0]), interpolation=cv2.INTER_NEAREST)
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

    def _on_overlay_toggled(self, _checked: bool):
        self._update_preview(force_reset=False)

    def _update_colorbar(self, show: bool):
        if self.colorbar_label is None:
            return
        if show and os.path.exists(self.colorbar_img_path):
            qpix = QPixmap(self.colorbar_img_path)
            if qpix.isNull():
                self.colorbar_label.hide()
                return
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
        if self._busy("IOU分析"):
            return
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

        index = self.label_current_index
        labels = tuple(self.label_image_paths)
        sources = tuple(self.image_list)
        max_side = self.preview_max_side

        def work(progress, check_interrupted):
            result = iou_preview(labels, index, sources, max_side)
            check_interrupted()
            progress(1)
            return result

        def done(result):
            if (self.label_current_index, tuple(self.label_image_paths), tuple(self.image_list)) != (index, labels, sources):
                return
            self._last_iou_result = result
            self._colorbar_showing = True
            self.btn_toggle_colorbar.setChecked(True)
            self._update_preview(force_reset=False)

        self._start_task("IOU分析", 1, work, done)

    def _on_iou_thresh_changed(self):
        self._update_preview(force_reset=False)

    def on_batch_save_iou(self):
        if self._busy("批量IOU分析"):
            return
        labels = tuple(self.label_image_paths)
        if not labels:
            show_info(self, "请先选择标签图像")
            return
        destination = QFileDialog.getExistingDirectory(self, "选择IOU分析保存目录")
        if not destination:
            return
        limits = (self.sld_area_low.value(), self.sld_area_up.value(), self.sld_circ.value() / 100)
        operation = partial(
            save_iou, output_dir=Path(destination), paths=labels, source_paths=tuple(self.image_list),
            max_side=self.preview_max_side, limits=limits, minimum=self.sld_iou.value() / 100,
        )
        self._start_region_batch("批量IOU分析", labels, operation)
