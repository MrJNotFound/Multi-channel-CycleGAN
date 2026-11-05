import os
from typing import List, Optional, Dict
import cv2
import numpy as np
from class_ImageViewer import InteractiveImageLabel
from func_img_stitching import stitch_images_mosaic
from func_img_preprocessing import rotate_image
from func_tools import show_info, fit_preview
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QFileDialog, QComboBox, QSpinBox,
    QListWidget, QGroupBox, QDoubleSpinBox, QProgressBar, QSlider
)

class BatchMosaicWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(int, str)
    message = pyqtSignal(str)

    def __init__(
        self, image_paths, grid_rows, grid_cols, overlap_w, overlap_h, black_th, angle,
        save_dir, direction_width, direction_height, compress_ratio
    ):
        super().__init__()
        self.image_paths = image_paths
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.overlap_w = overlap_w  # 原图像素
        self.overlap_h = overlap_h
        self.black_th = black_th
        self.angle = angle
        self.save_dir = save_dir
        self.direction_width = direction_width
        self.direction_height = direction_height
        self.compress_ratio = compress_ratio

    def run(self):
        total = len(self.image_paths)
        n_per_grid = self.grid_rows * self.grid_cols
        # 分组处理，每组只处理满员的grid_rows*grid_cols，最后不足的丢弃
        groups = [
            self.image_paths[i:i + n_per_grid]
            for i in range(0, total - total % n_per_grid, n_per_grid)
        ]
        if len(groups) == 0:
            self.message.emit("没有可拼接的分组")
            self.finished.emit(0, self.save_dir)
            return
        saved_count = 0
        for g_idx, group in enumerate(groups):
            rotated_group = []
            ok = True
            for img_path in group:
                src = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if src is None:
                    ok = False
                    self.message.emit(f"第{g_idx+1}组拼接失败：{os.path.basename(img_path)}读取失败")
                    break
                img = rotate_image(src, self.angle) if self.angle != 0 else src
                if self.compress_ratio < 1.0:
                    img = cv2.resize(img, (0, 0), fx=self.compress_ratio, fy=self.compress_ratio, interpolation=cv2.INTER_NEAREST)
                rotated_group.append(img)
            if not ok:
                self.progress.emit(g_idx + 1)
                continue
            shapes = [im.shape for im in rotated_group]
            if len(set(shapes)) > 1:
                self.message.emit(f"第{g_idx+1}组拼接失败：组内存在尺寸不一致的图片")
                self.progress.emit(g_idx + 1)
                continue
            overlap_w_export = int(self.overlap_w * self.compress_ratio)
            overlap_h_export = int(self.overlap_h * self.compress_ratio)
            try:
                result = stitch_images_mosaic(
                    rotated_group,
                    grid_shape=(self.grid_rows, self.grid_cols),
                    overlap_width=overlap_w_export,
                    overlap_height=overlap_h_export,
                    black_threshold=self.black_th,
                    direction_width=self.direction_width,
                    direction_height=self.direction_height
                )
            except Exception as e:
                self.message.emit(f"第{g_idx+1}组马赛克拼接函数报错: {e}")
                self.progress.emit(g_idx + 1)
                continue
            save_path = self.save_dir
            cv2.imwrite(save_path, result)
            saved_count += 1
            self.progress.emit(g_idx + 1)
        self.finished.emit(saved_count, self.save_dir)

class StitchPage(QWidget):
    def __init__(self):
        super().__init__()
        self.image_paths: List[str] = []
        self.current_index: int = -1

        self._source_cache: Dict[int, np.ndarray] = {}
        self._source_cache_order: List[int] = []
        self._max_source_cache = 8

        self._preview_cache: Dict[int, np.ndarray] = {}
        self._preview_cache_order: List[int] = []
        self._max_preview_cache = 12
        self._preview_max_size = 1000

        self.stitch_worker: Optional[BatchMosaicWorker] = None
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)

        # 左侧参数区
        ctrl = QVBoxLayout()
        ctrl_widget = QWidget()
        ctrl_widget.setLayout(ctrl)
        ctrl_widget.setMinimumWidth(180)
        ctrl_widget.setMaximumWidth(220)

        # 文件操作分组
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout()

        btn_load = QPushButton("选择图像")
        btn_load.clicked.connect(self.on_choose)
        file_layout.addWidget(btn_load)

        btn_clear = QPushButton("清空列表")
        btn_clear.clicked.connect(self.on_clear_list)
        file_layout.addWidget(btn_clear)

        btn_save_batch = QPushButton("拼接并保存")
        btn_save_batch.clicked.connect(self.on_batch_stitch_async)
        file_layout.addWidget(btn_save_batch)

        file_group.setLayout(file_layout)
        ctrl.addWidget(file_group)

        # 拼接参数分组
        param_group = QGroupBox("拼接参数")
        param_layout = QVBoxLayout()

        # 列数
        self.sp_grid_col = QSpinBox()
        self.sp_grid_col.setRange(1, 100)
        self.sp_grid_col.setValue(2)
        self.sp_grid_col.valueChanged.connect(self._on_any_param_changed)
        param_layout.addWidget(QLabel("横向数量"))
        param_layout.addWidget(self.sp_grid_col)

        # 行数
        self.sp_grid_row = QSpinBox()
        self.sp_grid_row.setRange(1, 100)
        self.sp_grid_row.setValue(2)
        self.sp_grid_row.valueChanged.connect(self._on_any_param_changed)
        param_layout.addWidget(QLabel("纵向数量"))
        param_layout.addWidget(self.sp_grid_row)

        # 横向朝向
        self.cmb_dir_width = QComboBox()
        self.cmb_dir_width.addItems(["向右", "向左"])
        self.cmb_dir_width.currentTextChanged.connect(self._on_any_param_changed)
        param_layout.addWidget(QLabel("横向拼接朝向"))
        param_layout.addWidget(self.cmb_dir_width)

        # 纵向朝向
        self.cmb_dir_height = QComboBox()
        self.cmb_dir_height.addItems(["向下", "向上"])
        self.cmb_dir_height.currentTextChanged.connect(self._on_any_param_changed)
        param_layout.addWidget(QLabel("纵向拼接朝向"))
        param_layout.addWidget(self.cmb_dir_height)

        # 压缩比例
        self.sp_compress = QDoubleSpinBox()
        self.sp_compress.setRange(0.1, 1.0)
        self.sp_compress.setSingleStep(0.05)
        self.sp_compress.setDecimals(2)
        self.sp_compress.setValue(1.0)
        self.sp_compress.valueChanged.connect(self._on_any_param_changed)
        param_layout.addWidget(QLabel("压缩比例"))
        param_layout.addWidget(self.sp_compress)

        # 旋转角度
        self.sp_angle = QDoubleSpinBox()
        self.sp_angle.setRange(-180, 180)
        self.sp_angle.setSingleStep(0.1)
        self.sp_angle.setDecimals(1)
        self.sp_angle.setValue(0)
        self.sp_angle.valueChanged.connect(self._on_any_param_changed)
        param_layout.addWidget(QLabel("旋转角度 (°)"))
        param_layout.addWidget(self.sp_angle)

        # 横向重叠像素（滑条），单位为“原图像素”
        self.sld_overlap_w = QSlider(Qt.Orientation.Horizontal)
        self.sld_overlap_w.setRange(1, 500)
        self.sld_overlap_w.setValue(60)
        self.lbl_overlap_w = QLabel("横向重叠: 60")
        self.sld_overlap_w.valueChanged.connect(lambda v: self.lbl_overlap_w.setText(f"横向重叠: {v}"))
        self.sld_overlap_w.valueChanged.connect(self._on_any_param_changed)
        param_layout.addWidget(self.lbl_overlap_w)
        param_layout.addWidget(self.sld_overlap_w)

        # 纵向重叠像素（滑条），单位为“原图像素”
        self.sld_overlap_h = QSlider(Qt.Orientation.Horizontal)
        self.sld_overlap_h.setRange(1, 500)
        self.sld_overlap_h.setValue(60)
        self.lbl_overlap_h = QLabel("纵向重叠: 60")
        self.sld_overlap_h.valueChanged.connect(lambda v: self.lbl_overlap_h.setText(f"纵向重叠: {v}"))
        self.sld_overlap_h.valueChanged.connect(self._on_any_param_changed)
        param_layout.addWidget(self.lbl_overlap_h)
        param_layout.addWidget(self.sld_overlap_h)

        # 重叠阈值（滑条）
        self.sld_black = QSlider(Qt.Orientation.Horizontal)
        self.sld_black.setRange(0, 255)
        self.sld_black.setValue(10)
        self.lbl_black = QLabel("重叠阈值: 10")
        self.sld_black.valueChanged.connect(lambda v: self.lbl_black.setText(f"重叠阈值: {v}"))
        self.sld_black.valueChanged.connect(self._on_any_param_changed)
        param_layout.addWidget(self.lbl_black)
        param_layout.addWidget(self.sld_black)

        param_group.setLayout(param_layout)
        ctrl.addWidget(param_group)
        ctrl.addStretch(1)

        # 文件列表
        middle_layout = QVBoxLayout()
        middle_layout.addWidget(QLabel("图像列表"))

        self.list_widget = QListWidget()
        self.list_widget.setMaximumWidth(150)
        self.list_widget.setMinimumWidth(100)
        self.list_widget.currentRowChanged.connect(self._on_image_selected)
        middle_layout.addWidget(self.list_widget)

        self.lbl_image_info = QLabel("未选择图像")
        self.lbl_image_info.setStyleSheet("font-size: 10px; color: #666;")
        self.lbl_image_info.setWordWrap(True)
        middle_layout.addWidget(self.lbl_image_info)

        # 右侧预览窗口
        preview_layout = QVBoxLayout()
        preview_title = QLabel("拼接预览")
        preview_title.setStyleSheet("font-weight: bold; font-size: 14px; margin: 5px 0;")
        preview_layout.addWidget(preview_title)

        self.interactive_image = InteractiveImageLabel()
        preview_layout.addWidget(self.interactive_image, 1)

        tips_layout = QHBoxLayout()
        tips_layout.setContentsMargins(0, 2, 0, 5)
        tips_label = QLabel("滚轮缩放 | 左键拖拽 | 双击重置")
        tips_label.setStyleSheet("color: #666; font-size: 10px;")
        tips_layout.addWidget(tips_label)
        tips_layout.addStretch()
        preview_layout.addLayout(tips_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        preview_layout.addWidget(self.progress_bar)

        layout.addWidget(ctrl_widget, 0)
        layout.addLayout(middle_layout, 0)
        layout.addLayout(preview_layout, 1)

    def _on_any_param_changed(self, *_):
        self._preview_cache.clear()
        self._update_preview_for_current(reset_view=False)

    def _get_source_image(self, index: int) -> Optional[np.ndarray]:
        if index < 0 or index >= len(self.image_paths):
            return None
        if index in self._source_cache:
            if index in self._source_cache_order:
                self._source_cache_order.remove(index)
            self._source_cache_order.append(index)
            return self._source_cache[index]
        path = self.image_paths[index]
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            show_info(self, f"无法读取图像: {os.path.basename(path)}")
            return None
        self._put_source_cache(index, img)
        return img

    def _put_source_cache(self, index: int, img: np.ndarray):
        self._source_cache[index] = img
        if index in self._source_cache_order:
            self._source_cache_order.remove(index)
        self._source_cache_order.append(index)
        while len(self._source_cache_order) > self._max_source_cache:
            old_idx = self._source_cache_order.pop(0)
            if old_idx in self._source_cache:
                del self._source_cache[old_idx]

    def _get_preview_image(self, index: int) -> Optional[np.ndarray]:
        if index < 0 or index >= len(self.image_paths):
            return None
        if index in self._preview_cache:
            if index in self._preview_cache_order:
                self._preview_cache_order.remove(index)
            self._preview_cache_order.append(index)
            return self._preview_cache[index]
        src = self._get_source_image(index)
        if src is None:
            return None
        preview = fit_preview(src, max_size=self._preview_max_size)
        self._put_preview_cache(index, preview)
        return preview

    def _put_preview_cache(self, index: int, img: np.ndarray):
        self._preview_cache[index] = img
        if index in self._preview_cache_order:
            self._preview_cache_order.remove(index)
        self._preview_cache_order.append(index)
        while len(self._preview_cache_order) > self._max_preview_cache:
            old_idx = self._preview_cache_order.pop(0)
            if old_idx in self._preview_cache:
                del self._preview_cache[old_idx]

    def _clear_caches(self):
        self._source_cache.clear()
        self._source_cache_order.clear()
        self._preview_cache.clear()
        self._preview_cache_order.clear()

    def on_choose(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择图像(顺序即拼接顺序)",
            filter="Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not files:
            return
        self.image_paths = list(files)
        self._clear_caches()
        self.list_widget.clear()
        for p in files:
            self.list_widget.addItem(os.path.basename(p))
        self.lbl_image_info.setText("未选择图像")
        self.interactive_image.set_image(None, reset_view=True)
        if len(files) > 0:
            self.list_widget.setCurrentRow(0)
        show_info(self, f"已载入 {len(files)} 个路径")

    def on_clear_list(self):
        self.image_paths.clear()
        self._clear_caches()
        self.list_widget.clear()
        self.current_index = -1
        self.interactive_image.set_image(None, reset_view=True)
        self.lbl_image_info.setText("未选择图像")
        self.progress_bar.setVisible(False)

    def _on_image_selected(self, row: int):
        self.current_index = row
        if row == -1 or row >= len(self.image_paths):
            self.lbl_image_info.setText("未选择图像")
            self.interactive_image.set_image(None, reset_view=True)
            return
        img_path = self.image_paths[row]
        filename = os.path.basename(img_path)
        self.lbl_image_info.setText(f"文件: {filename}")
        self._update_preview_for_current(reset_view=True)

    def _update_preview_for_current(self, reset_view: bool = True):
        idx = self.current_index
        if idx == -1 or idx >= len(self.image_paths):
            self.interactive_image.set_image(None, reset_view=True)
            return

        grid_r = self.sp_grid_row.value()
        grid_c = self.sp_grid_col.value()
        angle = self.sp_angle.value()
        overlap_w = int(self.sld_overlap_w.value())
        overlap_h = int(self.sld_overlap_h.value())
        black_th = int(self.sld_black.value())
        dir_width = self.cmb_dir_width.currentText()
        dir_height = self.cmb_dir_height.currentText()

        # 只展示2x2局部，起点为当前位置（row0, col0）
        imgs = []
        shapes = []
        preview_scale = None
        row0 = idx // grid_c
        col0 = idx % grid_c
        for r in range(2):
            for c in range(2):
                row_idx = row0 + r
                col_idx = col0 + c
                real_idx = row_idx * grid_c + col_idx
                if row_idx < grid_r and col_idx < grid_c and real_idx < len(self.image_paths):
                    src = self._get_source_image(real_idx)
                    if src is None:
                        src = np.zeros((256,256,3), dtype=np.uint8)
                    preview = fit_preview(src, max_size=self._preview_max_size)
                    if preview_scale is None:
                        preview_scale = preview.shape[0] / src.shape[0]
                    if angle != 0:
                        preview = rotate_image(preview, angle)
                    imgs.append(preview)
                    shapes.append(preview.shape)
                else:
                    if imgs:
                        imgs.append(np.zeros_like(imgs[0]))
                        shapes.append(imgs[0].shape)
                    else:
                        imgs.append(np.zeros((256,256,3), dtype=np.uint8))
                        shapes.append((256,256,3))

        if preview_scale is None:
            preview_scale = 1.0
        overlap_w_preview = max(1, int(overlap_w * preview_scale))
        overlap_h_preview = max(1, int(overlap_h * preview_scale))

        if len(set(shapes)) > 1:
            self.interactive_image.set_image(None, reset_view=True)
            show_info(self, "马赛克预览失败：图片尺寸不一致，无法拼接！")
            return
        preview_img = stitch_images_mosaic(
            imgs,
            grid_shape=(2, 2),
            overlap_width=overlap_w_preview,
            overlap_height=overlap_h_preview,
            black_threshold=black_th,
            direction_width=dir_width,
            direction_height=dir_height
        )
        self.interactive_image.set_image(preview_img, reset_view=reset_view)

    def on_batch_stitch_async(self):
        if self.stitch_worker and self.stitch_worker.isRunning():
            return
        if not self.image_paths:
            show_info(self, "请先选择图像")
            return
        grid_row = int(self.sp_grid_row.value())
        grid_col = int(self.sp_grid_col.value())
        overlap_w = int(self.sld_overlap_w.value())
        overlap_h = int(self.sld_overlap_h.value())
        black_th = int(self.sld_black.value())
        angle = self.sp_angle.value()
        ratio = float(self.sp_compress.value())
        dir_width = self.cmb_dir_width.currentText()
        dir_height = self.cmb_dir_height.currentText()
        save_dir, file_type = QFileDialog.getSaveFileName(
            self,
            "保存拼接结果",
            "",
            "JPG Files (*.jpg);;PNG Files (*.png);;TIFF Files (*.tiff);;BMP Files (*.bmp)"
        )
        if not save_dir:
            return

        n_per_grid = grid_row * grid_col
        total = len(self.image_paths)
        groups_count = total // n_per_grid
        self.progress_bar.setRange(0, groups_count)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.stitch_worker = BatchMosaicWorker(
            self.image_paths, grid_row, grid_col, overlap_w, overlap_h, black_th, angle, save_dir,
            direction_width=dir_width,
            direction_height=dir_height,
            compress_ratio=ratio
        )
        self.stitch_worker.progress.connect(self.progress_bar.setValue)
        self.stitch_worker.message.connect(lambda msg: show_info(self, msg))
        self.stitch_worker.finished.connect(self._on_batch_stitch_done)
        self.stitch_worker.start()

    def _on_batch_stitch_done(self, saved_count, save_dir):
        self.progress_bar.setFormat("处理完成")
        import threading
        def hide_progress_bar():
            import time; time.sleep(1.2)
            self.progress_bar.setVisible(False)
        threading.Thread(target=hide_progress_bar).start()
        show_info(self, f"拼接完成！共保存 {saved_count} 组到\n{save_dir}")
        self.interactive_image.set_image(None, reset_view=True)
        self.stitch_worker = None