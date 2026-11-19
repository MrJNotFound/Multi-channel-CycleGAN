from functools import partial
from pathlib import Path

from PyQt6.QtCore import QCoreApplication, Qt, QTimer
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QFileDialog, QSlider, QCheckBox, QSpinBox, QGroupBox, QListWidget,
    QProgressBar, QDoubleSpinBox, QFormLayout,
)

from ...processing.images import fit_preview, read_image
from ...processing.preprocessing import PreprocessingOptions, export_images, process_image
from ..tasks import TaskThread, stop_tasks
from ..widgets import InteractiveImageLabel, show_info


class PreprocessingPage(QWidget):
    def __init__(self):
        super().__init__()
        self.image_list: list[str] = []
        self.current_index: int = -1
        self.result_cache: dict = {}
        self.show_original = False
        self.batch_worker: TaskThread | None = None
        self._preview_request = None
        self._preview_worker = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._start_preview)
        self._build_ui()
        self._hide_save_progress_timer = QTimer(self)
        self._hide_save_progress_timer.setSingleShot(True)
        self._hide_save_progress_timer.timeout.connect(self.progress_bar.hide)
        QCoreApplication.instance().aboutToQuit.connect(self._stop_workers)

    def _build_ui(self):
        layout = QHBoxLayout(self)

        # 左侧控制区
        ctrl = QVBoxLayout()
        ctrl_widget = QWidget()
        ctrl_widget.setLayout(ctrl)
        ctrl_widget.setMinimumWidth(180)
        ctrl_widget.setMaximumWidth(200)

        # 文件操作
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout()

        btn_load = QPushButton("选择图像")
        btn_load.clicked.connect(self.on_load)  # type: ignore
        file_layout.addWidget(btn_load)

        btn_clear = QPushButton("清空列表")
        btn_clear.clicked.connect(self.on_clear_list)  # type: ignore
        file_layout.addWidget(btn_clear)

        btn_save_single = QPushButton("保存当前图像")
        btn_save_single.clicked.connect(self.on_save_single)  # type: ignore
        file_layout.addWidget(btn_save_single)

        btn_save_batch = QPushButton("批量保存全部")
        btn_save_batch.clicked.connect(self.on_save_batch_async)  # type: ignore
        file_layout.addWidget(btn_save_batch)

        file_group.setLayout(file_layout)
        ctrl.addWidget(file_group)

        # CLAHE增强
        clahe_group = QGroupBox("CLAHE 增强")
        cg_layout = QVBoxLayout()
        self.sld_clip = QSlider(Qt.Orientation.Horizontal)
        self.sld_clip.setRange(1, 30)
        self.sld_clip.setValue(8)
        self.lbl_clip = QLabel("限制对比度: 8")
        self.sld_clip.valueChanged.connect(lambda v: self._on_slider_label(self.lbl_clip, f"限制对比度: {v}"))  # type: ignore
        self.sld_clip.valueChanged.connect(self._on_param_changed)  # type: ignore
        cg_layout.addWidget(self.lbl_clip)
        cg_layout.addWidget(self.sld_clip)

        self.sp_tile_x = QSpinBox()
        self.sp_tile_x.setRange(2, 46)
        self.sp_tile_x.setValue(5)
        self.sp_tile_y = QSpinBox()
        self.sp_tile_y.setRange(2, 64)
        self.sp_tile_y.setValue(5)
        self.sp_tile_x.valueChanged.connect(self._on_param_changed)  # type: ignore
        self.sp_tile_y.valueChanged.connect(self._on_param_changed)  # type: ignore
        cg_layout.addWidget(QLabel("网格尺寸 X/Y"))
        tile_row = QHBoxLayout()
        tile_row.addWidget(self.sp_tile_x)
        tile_row.addWidget(self.sp_tile_y)
        cg_layout.addLayout(tile_row)
        clahe_group.setLayout(cg_layout)
        ctrl.addWidget(clahe_group)

        # 频域滤波
        freq_group = QGroupBox("频域带阻滤波")
        fg_layout = QVBoxLayout()
        self.chk_freq = QCheckBox("启用滤波")
        self.chk_freq.stateChanged.connect(self._on_param_changed)  # type: ignore
        fg_layout.addWidget(self.chk_freq)

        self.sld_angle = QSpinBox()
        self.sld_angle.setRange(-180, 180)
        self.sld_angle.setValue(0)
        self.lbl_angle = QLabel("角度: 0°")
        self.sld_angle.valueChanged.connect(lambda v: self._on_slider_label(self.lbl_angle, f"角度: {v}°"))  # type: ignore
        self.sld_angle.valueChanged.connect(self._on_param_changed)  # type: ignore
        fg_layout.addWidget(self.lbl_angle)
        fg_layout.addWidget(self.sld_angle)

        self.sld_band = QSpinBox()
        self.sld_band.setRange(1, 90)
        self.sld_band.setValue(20)
        self.lbl_band = QLabel("带宽: 20")
        self.sld_band.valueChanged.connect(lambda v: self._on_slider_label(self.lbl_band, f"带宽: {v}"))  # type: ignore
        self.sld_band.valueChanged.connect(self._on_param_changed)  # type: ignore
        fg_layout.addWidget(self.lbl_band)
        fg_layout.addWidget(self.sld_band)
        freq_group.setLayout(fg_layout)
        ctrl.addWidget(freq_group)

        self.toolmark_group = QGroupBox("方向性 Median 残差滤除")
        self.toolmark_group.setCheckable(True)
        self.toolmark_group.setChecked(False)
        toolmark_layout = QFormLayout(self.toolmark_group)

        self.sp_toolmark_strength = QDoubleSpinBox()
        self.sp_toolmark_strength.setRange(0, 1)
        self.sp_toolmark_strength.setSingleStep(0.05)
        self.sp_toolmark_strength.setValue(0.7)
        toolmark_layout.addRow("去除强度", self.sp_toolmark_strength)

        self.chk_toolmark_auto = QCheckBox("自动识别方向")
        self.chk_toolmark_auto.setChecked(True)
        toolmark_layout.addRow(self.chk_toolmark_auto)
        self.sp_toolmark_angle = QDoubleSpinBox()
        self.sp_toolmark_angle.setRange(-90, 90)
        self.sp_toolmark_angle.setDecimals(1)
        self.sp_toolmark_angle.setSuffix("°")
        self.sp_toolmark_angle.setToolTip("法向垂直于刀纹。竖直刀纹为 0°，水平刀纹为 ±90°。")
        self.sp_toolmark_angle.setEnabled(False)
        toolmark_layout.addRow("法向角", self.sp_toolmark_angle)

        self.sp_toolmark_kernel = QSpinBox()
        self.sp_toolmark_kernel.setRange(3, 101)
        self.sp_toolmark_kernel.setSingleStep(2)
        self.sp_toolmark_kernel.setValue(51)
        self.sp_toolmark_kernel.setToolTip("以原图像素计，建议大于刀纹宽度的两倍。")
        toolmark_layout.addRow("一维核长度", self.sp_toolmark_kernel)
        self.toolmark_group.toggled.connect(self._on_param_changed)
        self.chk_toolmark_auto.toggled.connect(lambda checked: self.sp_toolmark_angle.setEnabled(not checked))
        self.chk_toolmark_auto.toggled.connect(self._on_param_changed)
        for spin in (self.sp_toolmark_strength, self.sp_toolmark_angle, self.sp_toolmark_kernel):
            spin.valueChanged.connect(self._on_param_changed)
        ctrl.addWidget(self.toolmark_group)

        ctrl.addStretch(1)

        # 文件列表
        middle_layout = QVBoxLayout()
        middle_layout.addWidget(QLabel("图像列表"))

        self.list_widget = QListWidget()
        self.list_widget.setMaximumWidth(150)
        self.list_widget.setMinimumWidth(100)
        self.list_widget.currentRowChanged.connect(self._on_image_selected)  # type: ignore
        middle_layout.addWidget(self.list_widget)

        # 显示当前图像信息
        self.lbl_image_info = QLabel("未选择图像")
        self.lbl_image_info.setStyleSheet("font-size: 10px; color: #666;")
        self.lbl_image_info.setWordWrap(True)
        middle_layout.addWidget(self.lbl_image_info)

        # 右侧预览区域
        preview_layout = QVBoxLayout()

        # 切换显示按钮
        switch_layout = QHBoxLayout()
        switch_layout.setContentsMargins(0, 5, 0, 5)
        self.btn_original = QPushButton("原图")
        self.btn_original.setCheckable(True)
        self.btn_original.setChecked(False)  # 默认未按下
        self.btn_original.clicked.connect(self._toggle_original)  # type: ignore
        button_style = """
            QPushButton {
                min-height: 25px;
                min-width: 80px;
                border: 1px solid #ccc;
                border-radius: 5px;
                background-color: #f0f0f0;
                font-size: 11px;
            }
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
                border-color: #45a049;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:checked:hover {
                background-color: #45a049;
            }
        """
        self.btn_original.setStyleSheet(button_style)
        switch_layout.addWidget(self.btn_original)
        switch_layout.addStretch()

        # 当前显示状态标签
        self.lbl_current_view = QLabel("当前显示: 处理结果")
        self.lbl_current_view.setStyleSheet("font-weight: bold; color: #333; font-size: 11px;")
        switch_layout.addWidget(self.lbl_current_view)

        preview_layout.addLayout(switch_layout)

        # 交互式图像显示控件
        self.interactive_image = InteractiveImageLabel()
        preview_layout.addWidget(self.interactive_image, 1)

        # 操作提示
        tips_layout = QHBoxLayout()
        tips_layout.setContentsMargins(0, 2, 0, 5)
        tips_label = QLabel("滚轮缩放 | 左键拖拽 | 双击重置")
        tips_label.setStyleSheet("color: #666; font-size: 10px;")
        tips_layout.addWidget(tips_label)
        tips_layout.addStretch()
        preview_layout.addLayout(tips_layout)

        # 图像窗口下方进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        preview_layout.addWidget(self.progress_bar)

        layout.addWidget(ctrl_widget, 0)
        layout.addLayout(middle_layout, 0)
        layout.addLayout(preview_layout, 1)

    def _toggle_original(self):
        """切换原图/处理结果显示"""
        if self.current_index == -1:
            return

        if self.btn_original.isChecked():
            self.show_original = True
            self.lbl_current_view.setText("当前显示: 原图")
        else:
            self.show_original = False
            self.lbl_current_view.setText("当前显示: 处理结果")
        self.update_preview(reset_view=False)

    def _processing_options(self):
        toolmarks = None
        if self.toolmark_group.isChecked():
            toolmarks = dict(
                strength=self.sp_toolmark_strength.value(),
                auto_orientation=self.chk_toolmark_auto.isChecked(),
                orientation_deg=self.sp_toolmark_angle.value(),
                directional_kernel=self.sp_toolmark_kernel.value(),
            )
        frequency = (self.sld_angle.value(), self.sld_band.value()) if self.chk_freq.isChecked() else None
        return PreprocessingOptions(
            self.sld_clip.value(),
            (self.sp_tile_x.value(), self.sp_tile_y.value()),
            frequency,
            toolmarks,
        )

    @staticmethod
    def _render_preview(path, options, _progress, check_interrupted):
        image = read_image(path)
        if image is None:
            raise ValueError("无法读取图像")
        scale = None
        if options.toolmarks is not None:
            height, width = image.shape[:2]
            image = fit_preview(image, max_size=1024)
            scale = (image.shape[0] / height, image.shape[1] / width)
        check_interrupted()
        return process_image(image, options, preview_scale=scale)

    def _start_preview(self):
        if self._preview_request is None or self._preview_worker is not None:
            return
        path, _, _, options = self._preview_request
        worker = TaskThread(partial(self._render_preview, path, options), self)
        self._preview_worker = worker
        request = self._preview_request
        worker.finished.connect(lambda: self._finish_preview(request))
        worker.start()

    def _finish_preview(self, request):
        worker = self._preview_worker
        self._preview_worker = None
        worker.deleteLater()
        if self._preview_request != request:
            if self._preview_request is not None:
                self._preview_timer.start(0)
            return
        self._preview_request = None
        if worker.cancelled:
            return
        if worker.error:
            self.interactive_image.set_image(None, reset_view=True)
            self.interactive_image.setText(f"预览失败：{worker.error}")
            self.lbl_current_view.setText("预览失败")
            return
        _, index, reset_view, _ = request
        result, report = worker.result
        if len(self.result_cache) >= 5:
            self.result_cache.pop(next(iter(self.result_cache)))
        self.result_cache[index] = result
        self.interactive_image.set_image(result, reset_view=reset_view)
        if report is None:
            self.lbl_current_view.setText("当前显示: 处理结果")
        elif report.applied:
            self.lbl_current_view.setText(f"当前显示: 处理结果 · 刀纹法向角 {report.orientation_deg:+.1f}°")
        else:
            self.lbl_current_view.setText("当前显示: 处理结果 · 未应用刀纹滤除")

    def _stop_workers(self):
        self._preview_request = None
        self._preview_timer.stop()
        stop_tasks(self._preview_worker, self.batch_worker)

    @staticmethod
    def _on_slider_label(label: QLabel, text: str):
        label.setText(text)

    def _on_param_changed(self, *_args):
        self.result_cache.clear()
        self.update_preview()

    def _on_image_selected(self, row: int):
        if row == -1 or row >= len(self.image_list):
            self.current_index = -1
            self.update_preview()
            self.lbl_image_info.setText("未选择图像")
            return
        self.current_index = row
        img_path = self.image_list[row]
        filename = Path(img_path).name
        self.lbl_image_info.setText(f"文件: {filename}")
        self.show_original = False
        self.btn_original.setChecked(False)
        self.lbl_current_view.setText("当前显示: 处理结果")
        self.update_preview(reset_view=True)

    def on_load(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择图像(支持多选)",
            filter="Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not paths:
            return
        self.image_list.clear()
        self.result_cache.clear()
        self.list_widget.clear()
        for path in paths:
            self.image_list.append(path)
            filename = Path(path).name
            self.list_widget.addItem(filename)
        valid_count = len(self.image_list)
        if valid_count == 0:
            show_info(self, "没有有效的图像文件")
            return
        if valid_count > 0:
            self.list_widget.setCurrentRow(0)
        show_info(self, f"成功载入 {valid_count} 张图像")

    def on_clear_list(self):
        for key in list(self.result_cache.keys()):
            del self.result_cache[key]
        self.image_list.clear()
        self.result_cache.clear()
        self.list_widget.clear()
        self.current_index = -1
        self.show_original = False
        self.btn_original.setChecked(False)
        self.lbl_current_view.setText("当前显示: 处理结果")
        self.update_preview()
        self.lbl_image_info.setText("未选择图像")

    def update_preview(self, reset_view: bool = False):
        self._preview_request = None
        self._preview_timer.stop()
        index = self.current_index
        if not 0 <= index < len(self.image_list):
            self.interactive_image.set_image(None, reset_view=True)
            return
        path = self.image_list[index]
        if self.show_original:
            self.interactive_image.set_image(read_image(path), reset_view=reset_view)
            return
        if index in self.result_cache:
            self.lbl_current_view.setText("当前显示: 处理结果")
            self.interactive_image.set_image(self.result_cache[index], reset_view=reset_view)
            return
        self._preview_request = (path, index, reset_view, self._processing_options())
        self.lbl_current_view.setText("正在计算预览…")
        self._preview_timer.start(150)

    def on_save_single(self):
        if self.batch_worker is not None:
            return
        if self.current_index == -1:
            show_info(self, "请先选择要保存的图像")
            return
        source = self.image_list[self.current_index]
        path, _ = QFileDialog.getSaveFileName(
            self, "保存处理结果", f"processed_{Path(source).name}",
            filter="PNG (*.png);;JPG (*.jpg);;BMP (*.bmp);;TIFF (*.tif *.tiff)",
        )
        if path:
            self._start_save_worker((source,), path, single_file=True)

    def on_save_batch_async(self):
        if self.batch_worker is not None:
            return
        if not self.image_list:
            show_info(self, "没有图像需要保存")
            return
        destination = QFileDialog.getExistingDirectory(self, "选择批量保存目录")
        if destination:
            self._start_save_worker(tuple(self.image_list), destination, single_file=False)

    def _start_save_worker(self, paths, destination, *, single_file):
        self._hide_save_progress_timer.stop()
        self.progress_bar.setRange(0, len(paths))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setVisible(True)
        options = self._processing_options()

        def work(progress, check_interrupted):
            return export_images(
                paths, destination, options, single_file=single_file,
                progress=progress, check_interrupted=check_interrupted,
            )

        worker = TaskThread(work, self)
        self.batch_worker = worker
        worker.progress.connect(self.progress_bar.setValue)
        worker.finished.connect(lambda: self._on_save_finished(single_file))
        worker.start()

    def _on_save_finished(self, single_file):
        worker = self.batch_worker
        self.batch_worker = None
        worker.deleteLater()
        if single_file or worker.cancelled or worker.error:
            self.progress_bar.hide()
        else:
            self.progress_bar.setFormat("处理完成")
            self._hide_save_progress_timer.start(1200)
        if worker.cancelled:
            return
        if worker.error:
            show_info(self, f"保存失败：{worker.error}")
            return
        saved, errors = worker.result
        if single_file:
            show_info(self, "当前图像已保存" if saved else f"保存失败：{errors[0]}")
        elif errors:
            show_info(self, f"批量保存完成。成功 {saved} 张，失败 {len(errors)} 张。\n" + "\n".join(errors))
        else:
            show_info(self, f"批量保存完成！成功处理并保存 {saved} 张图像")
