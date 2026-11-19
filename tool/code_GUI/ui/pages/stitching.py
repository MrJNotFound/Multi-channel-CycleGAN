from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

from PyQt6.QtCore import QCoreApplication, QSignalBlocker, QTimer, Qt
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QFileDialog, QComboBox, QSpinBox,
    QListWidget, QGroupBox, QDoubleSpinBox, QProgressBar, QSlider, QCheckBox,
)

from ...processing.images import fit_preview, read_image
from ...processing.large_image_backend import estimate_stitching_parameters
from ...processing.stitching import MosaicOptions, export_mosaics, local_preview_indices, render_mosaic
from ..tasks import TaskThread, stop_tasks
from ..widgets import InteractiveImageLabel, show_info


class StitchingPage(QWidget):
    def __init__(self):
        super().__init__()
        self.image_paths: list[str] = []
        self.current_index = -1
        self._preview_cache = OrderedDict()
        self._preview_max_size = 1000
        self.stitch_worker: TaskThread | None = None
        self.estimate_worker: TaskThread | None = None
        self._estimate_signature = None
        self._build_ui()
        self._hide_progress_timer = QTimer(self)
        self._hide_progress_timer.setSingleShot(True)
        self._hide_progress_timer.timeout.connect(self.progress_bar.hide)
        QCoreApplication.instance().aboutToQuit.connect(self._stop_workers)

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

        self.btn_auto_estimate = QPushButton("自动估计参数")
        self.btn_auto_estimate.setToolTip(
            "按当前行列、列表顺序和朝向，估计第一组图像的横纵重叠与旋转角度。\n"
            "至少需要 2 行、2 列；重叠搜索范围 0%–30%，角度范围 ±8°。"
        )
        self.btn_auto_estimate.clicked.connect(self.on_auto_estimate)
        param_layout.addWidget(self.btn_auto_estimate)

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

        self.chk_fill_gaps = QCheckBox("填充黑色拼接缝隙")
        self.chk_fill_gaps.setToolTip("用周围像素估算填补旋转和拼接后未被图块覆盖的区域，包括外缘。")
        self.chk_fill_gaps.toggled.connect(self._on_any_param_changed)
        param_layout.addWidget(self.chk_fill_gaps)

        self.chk_blend_overlaps = QCheckBox("亮度色彩归一与重叠过渡")
        self.chk_blend_overlaps.setToolTip("根据对应的重叠区域校正图块亮度和色差，然后渐变融合。")
        self.chk_blend_overlaps.toggled.connect(self._on_any_param_changed)
        param_layout.addWidget(self.chk_blend_overlaps)

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

    def _auto_estimate_signature(self):
        count = self.sp_grid_row.value() * self.sp_grid_col.value()
        return (
            tuple(self.image_paths[:count]), self.sp_grid_row.value(), self.sp_grid_col.value(),
            self.cmb_dir_width.currentText(), self.cmb_dir_height.currentText(),
            self.sp_angle.value(), self.sld_overlap_w.value(), self.sld_overlap_h.value(),
        )

    def on_auto_estimate(self):
        if self.estimate_worker is not None:
            return
        if self.stitch_worker is not None:
            show_info(self, "请等待当前拼接完成后再估计参数。")
            return
        rows, cols = self.sp_grid_row.value(), self.sp_grid_col.value()
        if rows < 2 or cols < 2:
            show_info(self, "自动估计需要至少 2 行、2 列，请先设置横向数量和纵向数量。")
            return
        if len(self.image_paths) < rows * cols:
            show_info(self, f"当前设置需要 {rows * cols} 张图像，请先选择足够的图像。")
            return
        self._estimate_signature = self._auto_estimate_signature()
        self.btn_auto_estimate.setEnabled(False)
        self.btn_auto_estimate.setText("正在估计……")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("自动估计参数 %p%")
        self.progress_bar.setVisible(True)
        paths, rows, cols, direction_width, direction_height, *_ = self._estimate_signature

        def work(progress, check_interrupted):
            def report(current, total, _message):
                check_interrupted()
                progress(round(100 * current / max(total, 1)))

            return estimate_stitching_parameters(paths, rows, cols, direction_width, direction_height, progress=report)

        self._hide_progress_timer.stop()
        worker = TaskThread(work, self)
        self.estimate_worker = worker
        worker.progress.connect(self.progress_bar.setValue)
        worker.finished.connect(self._on_auto_estimate_finished)
        worker.start()

    def _on_auto_estimate_result(self, parameters):
        if self._estimate_signature != self._auto_estimate_signature():
            show_info(self, "图像或拼接参数已变化，估计结果未应用，请重新估计。")
            return
        controls = (self.sp_angle, self.sld_overlap_w, self.sld_overlap_h)
        blockers = [QSignalBlocker(control) for control in controls]
        try:
            self.sp_angle.setValue(parameters["angle"])
            for slider, key in (
                (self.sld_overlap_w, "overlap_width"), (self.sld_overlap_h, "overlap_height")
            ):
                slider.setMaximum(max(slider.maximum(), parameters[key]))
                slider.setValue(parameters[key])
            self.lbl_overlap_w.setText(f"横向重叠: {self.sld_overlap_w.value()}")
            self.lbl_overlap_h.setText(f"纵向重叠: {self.sld_overlap_h.value()}")
        finally:
            blockers.clear()
        self._on_any_param_changed()
        show_info(self, f"自动估计完成：横向重叠 {parameters['overlap_width']} 像素，"
                  f"纵向重叠 {parameters['overlap_height']} 像素，旋转 {parameters['angle']:+.1f}°。\n"
                  "参数已回填，可继续手动调整。")

    def _on_auto_estimate_finished(self):
        worker = self.estimate_worker
        self.estimate_worker = None
        self.btn_auto_estimate.setEnabled(True)
        self.btn_auto_estimate.setText("自动估计参数")
        self.progress_bar.setVisible(False)
        self.progress_bar.setFormat("%p%")
        worker.deleteLater()
        if worker.cancelled:
            return
        if worker.error:
            show_info(self, f"自动估计失败：{worker.error}")
        else:
            self._on_auto_estimate_result(worker.result)

    def _stop_workers(self):
        stop_tasks(self.estimate_worker, self.stitch_worker)

    def _on_any_param_changed(self, *_):
        self._update_preview_for_current(reset_view=False)

    def _preview_tile(self, index):
        path = self.image_paths[index]
        if path in self._preview_cache:
            self._preview_cache.move_to_end(path)
            return self._preview_cache[path]
        source = read_image(path)
        if source is None:
            raise ValueError(f"无法读取图像：{Path(path).name}")
        tile = fit_preview(source, max_size=self._preview_max_size)
        scale = tuple(out / original for out, original in zip(tile.shape[:2], source.shape[:2]))
        entry = (tile, scale, source.shape)
        self._preview_cache[path] = entry
        if len(self._preview_cache) > 12:
            self._preview_cache.popitem(last=False)
        return entry

    def on_choose(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择图像(顺序即拼接顺序)",
            filter="Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not files:
            return
        self.image_paths = list(files)
        self._preview_cache.clear()
        self.list_widget.clear()
        for p in files:
            self.list_widget.addItem(Path(p).name)
        self.lbl_image_info.setText("未选择图像")
        self.interactive_image.set_image(None, reset_view=True)
        if len(files) > 0:
            self.list_widget.setCurrentRow(0)
        show_info(self, f"已载入 {len(files)} 个路径")

    def on_clear_list(self):
        self.image_paths.clear()
        self._preview_cache.clear()
        self.list_widget.clear()
        self.current_index = -1
        self.interactive_image.set_image(None, reset_view=True)
        self.lbl_image_info.setText("未选择图像")

    def _on_image_selected(self, row: int):
        self.current_index = row
        if row == -1 or row >= len(self.image_paths):
            self.lbl_image_info.setText("未选择图像")
            self.interactive_image.set_image(None, reset_view=True)
            return
        img_path = self.image_paths[row]
        filename = Path(img_path).name
        self.lbl_image_info.setText(f"文件: {filename}")
        self._update_preview_for_current(reset_view=True)

    def _mosaic_options(self):
        return MosaicOptions(
            rows=self.sp_grid_row.value(),
            cols=self.sp_grid_col.value(),
            overlap_w=self.sld_overlap_w.value(),
            overlap_h=self.sld_overlap_h.value(),
            black_threshold=self.sld_black.value(),
            angle=self.sp_angle.value(),
            direction_width=self.cmb_dir_width.currentText(),
            direction_height=self.cmb_dir_height.currentText(),
            fill_gaps=self.chk_fill_gaps.isChecked(),
            blend_overlaps=self.chk_blend_overlaps.isChecked(),
        )

    def _update_preview_for_current(self, reset_view: bool = True):
        if not 0 <= self.current_index < len(self.image_paths):
            self.interactive_image.set_image(None, reset_view=True)
            return
        options = self._mosaic_options()
        (rows, cols), indices = local_preview_indices(
            self.current_index, len(self.image_paths), options.rows, options.cols,
        )
        try:
            entries = [self._preview_tile(index) for index in indices]
            images, scales, source_shapes = zip(*entries)
            if len(set(source_shapes)) > 1:
                raise ValueError("组内图像尺寸或通道数不一致。")
            preview = render_mosaic(images, replace(options, rows=rows, cols=cols), overlap_scale=scales[0])
        except ValueError as error:
            self.interactive_image.set_image(None, reset_view=True)
            self.interactive_image.setText(f"预览失败：{error}")
            return
        self.interactive_image.set_image(preview, reset_view=reset_view)

    def on_batch_stitch_async(self):
        if self.estimate_worker is not None:
            show_info(self, "请等待自动估计完成后再拼接。")
            return
        if self.stitch_worker is not None:
            return
        if not self.image_paths:
            show_info(self, "请先选择图像")
            return
        options = self._mosaic_options()
        paths = tuple(self.image_paths)
        groups = len(paths) // (options.rows * options.cols)
        if not groups:
            show_info(self, f"当前网格需要至少 {options.rows * options.cols} 张图像。")
            return
        destination, _ = QFileDialog.getSaveFileName(
            self, "保存拼接结果", "",
            "JPG Files (*.jpg);;PNG Files (*.png);;TIFF Files (*.tiff);;BMP Files (*.bmp)",
        )
        if not destination:
            return
        scale = self.sp_compress.value()

        def work(progress, check_interrupted):
            return export_mosaics(
                paths, destination, options, scale,
                progress=progress, check_interrupted=check_interrupted,
            )

        self._hide_progress_timer.stop()
        self.progress_bar.setRange(0, groups)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setVisible(True)
        worker = TaskThread(work, self)
        self.stitch_worker = worker
        worker.progress.connect(self.progress_bar.setValue)
        worker.finished.connect(self._on_stitch_finished)
        worker.start()

    def _on_stitch_finished(self):
        worker = self.stitch_worker
        self.stitch_worker = None
        worker.deleteLater()
        if worker.cancelled or worker.error:
            self.progress_bar.hide()
            if worker.error:
                show_info(self, f"拼接失败：{worker.error}")
            return
        saved, errors, remainder = worker.result
        self.progress_bar.setFormat("处理完成")
        self._hide_progress_timer.start(1200)
        if errors or remainder:
            details = errors[:]
            if remainder:
                details.append(f"最后 {remainder} 张图像不足一组，已跳过。")
            show_info(self, f"已保存 {len(saved)} 组。\n" + "\n".join(details))
        elif saved:
            self._on_batch_stitch_done(len(saved), str(saved[0] if len(saved) == 1 else saved[0].parent))

    def _on_batch_stitch_done(self, saved_count, destination):
        show_info(self, f"拼接完成！共保存 {saved_count} 组到\n{destination}")
        self.interactive_image.set_image(None, reset_view=True)
