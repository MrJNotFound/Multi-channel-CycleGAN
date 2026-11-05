import os
from typing import List, Optional
import cv2
from func_img_registrating import batch_register_images_anchor_subseq
from func_tools import show_info
from class_ImageViewer import InteractiveImageLabel
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QFileDialog, QCheckBox,
    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QProgressBar, QGroupBox,
    QListWidget
)

class RegisterWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    message = pyqtSignal(str)

    def __init__(self, image_paths: List[str], output_folder: str, params: dict):
        super().__init__()
        self.image_paths = image_paths
        self.output_folder = output_folder
        self.params = params
        self._cancel = False

    def run(self):
        try:
            batch_register_images_anchor_subseq(
                image_paths=self.image_paths,
                output_folder=self.output_folder,
                anchor_interval=self.params.get('anchor_interval', 8),
                mode=self.params.get('mode', 'rigid'),
                only_rigid=self.params.get('only_rigid', False),
                scale_factor=self.params.get('scale_factor', 0.2),
                shrink_factors=self.params.get('shrink_factors', [5]),
                smoothing_sigmas=self.params.get('smoothing_sigmas', [1]),
                n_matches=self.params.get('n_matches', 200),
                mesh_size=self.params.get('mesh_size', 4),
                optimizer_iterations=self.params.get('optimizer_iterations', 50),
                optimizer_tol=self.params.get('optimizer_tol', 1e-5),
                optimizer_bounds=self.params.get('optimizer_bounds', (-100, 100)),
                progress_callback=self._on_progress
            )
        except Exception as e:
            self.message.emit(f"配准出错: {str(e)}")  # type: ignore
        self.finished.emit()  # type: ignore

    def _on_progress(self, val):
        self.progress.emit(val)  # type: ignore
    def cancel(self):
        self._cancel = True

class RegistrationPage(QWidget):
    def __init__(self):
        super().__init__()
        self.lbl_output = None
        self.image_paths: List[str] = []
        self.result_paths: List[str] = []
        self.current_index: int = -1
        self.result_index: int = -1
        self.output_folder: str = ""
        self.worker: Optional[RegisterWorker] = None
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)

        ctrl = QVBoxLayout()
        ctrl_widget = QWidget()
        ctrl_widget.setLayout(ctrl)
        ctrl_widget.setMinimumWidth(180)
        ctrl_widget.setMaximumWidth(200)

        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout()

        btn_load = QPushButton("选择图像")
        btn_load.clicked.connect(self.on_load_images)  # type: ignore
        file_layout.addWidget(btn_load)

        btn_clear = QPushButton("清空列表")
        btn_clear.clicked.connect(self.on_clear_list)  # type: ignore
        file_layout.addWidget(btn_clear)

        self.btn_run = QPushButton("开始配准")
        self.btn_run.clicked.connect(self.on_run)  # type: ignore
        file_layout.addWidget(self.btn_run)

        file_group.setLayout(file_layout)
        ctrl.addWidget(file_group)

        basic_group = QGroupBox("基础参数")
        basic_layout = QVBoxLayout()

        self.sp_anchor = QSpinBox()
        self.sp_anchor.setRange(1, 999)
        self.sp_anchor.setValue(8)
        basic_layout.addWidget(QLabel("锚点间隔"))
        basic_layout.addWidget(self.sp_anchor)

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["刚性", "刚性+弹性"])
        basic_layout.addWidget(QLabel("配准模式"))
        basic_layout.addWidget(self.cmb_mode)

        self.chk_onlyrigid = QCheckBox("仅平移旋转")
        basic_layout.addWidget(self.chk_onlyrigid)

        self.dsb_scale = QDoubleSpinBox()
        self.dsb_scale.setRange(0.05, 1.0)
        self.dsb_scale.setSingleStep(0.05)
        self.dsb_scale.setValue(0.2)
        basic_layout.addWidget(QLabel("下采样系数"))
        basic_layout.addWidget(self.dsb_scale)

        basic_group.setLayout(basic_layout)
        ctrl.addWidget(basic_group)

        advanced_group = QGroupBox("高级参数")
        advanced_layout = QVBoxLayout()

        self.sp_matches = QSpinBox()
        self.sp_matches.setRange(10, 5000)
        self.sp_matches.setValue(200)
        advanced_layout.addWidget(QLabel("匹配点数"))
        advanced_layout.addWidget(self.sp_matches)

        self.sp_mesh = QSpinBox()
        self.sp_mesh.setRange(2, 64)
        self.sp_mesh.setValue(4)
        advanced_layout.addWidget(QLabel("网格大小"))
        advanced_layout.addWidget(self.sp_mesh)

        self.le_shrink = QLineEdit("5")
        advanced_layout.addWidget(QLabel("收缩因子 (逗号分隔)"))
        advanced_layout.addWidget(self.le_shrink)

        self.le_sigma = QLineEdit("1")
        advanced_layout.addWidget(QLabel("平滑系数 (逗号分隔)"))
        advanced_layout.addWidget(self.le_sigma)

        self.sp_iter = QSpinBox()
        self.sp_iter.setRange(1, 5000)
        self.sp_iter.setValue(50)
        advanced_layout.addWidget(QLabel("优化迭代次数"))
        advanced_layout.addWidget(self.sp_iter)

        self.le_tol = QLineEdit("1e-5")
        advanced_layout.addWidget(QLabel("优化容忍度"))
        advanced_layout.addWidget(self.le_tol)

        self.le_bounds = QLineEdit("-100,100")
        advanced_layout.addWidget(QLabel("优化边界"))
        advanced_layout.addWidget(self.le_bounds)

        advanced_group.setLayout(advanced_layout)
        ctrl.addWidget(advanced_group)
        ctrl.addStretch(1)

        middle_layout = QVBoxLayout()
        middle_layout.addWidget(QLabel("图像列表"))

        self.list_widget = QListWidget()
        self.list_widget.setMaximumWidth(100)
        self.list_widget.setMinimumWidth(150)
        self.list_widget.currentRowChanged.connect(self._on_image_selected)  # type: ignore
        middle_layout.addWidget(self.list_widget)

        self.lbl_image_info = QLabel("未选择图像")
        self.lbl_image_info.setStyleSheet("font-size: 10px; color: #666;")
        self.lbl_image_info.setWordWrap(True)
        middle_layout.addWidget(self.lbl_image_info)

        middle_layout.addWidget(QLabel("处理结果列表"))

        self.result_list_widget = QListWidget()
        self.result_list_widget.setMaximumWidth(100)
        self.result_list_widget.setMinimumWidth(150)
        self.result_list_widget.currentRowChanged.connect(self._on_result_selected)  # type: ignore
        middle_layout.addWidget(self.result_list_widget)

        self.lbl_result_info = QLabel("未选择结果")
        self.lbl_result_info.setStyleSheet("font-size: 10px; color: #666;")
        self.lbl_result_info.setWordWrap(True)
        middle_layout.addWidget(self.lbl_result_info)

        preview_layout = QVBoxLayout()
        preview_title = QLabel("图像预览")
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

    def on_load_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择图像(支持多选)",
            filter="Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not paths:
            return
        self.image_paths.clear()
        self.list_widget.clear()
        self.lbl_image_info.setText("未选择图像")
        self.result_paths.clear()
        self.result_list_widget.clear()
        self.lbl_result_info.setText("未选择结果")
        for path in paths:
            self.image_paths.append(path)
            filename = os.path.basename(path)
            self.list_widget.addItem(filename)
        valid_count = len(self.image_paths)
        if valid_count > 0:
            self.list_widget.setCurrentRow(0)
        show_info(self, f"成功载入 {valid_count} 张图像")

    def on_clear_list(self):
        self.image_paths.clear()
        self.list_widget.clear()
        self.current_index = -1
        self.interactive_image.set_image(None, reset_view=True)
        self.lbl_image_info.setText("未选择图像")
        self.result_paths.clear()
        self.result_list_widget.clear()
        self.result_index = -1
        self.lbl_result_info.setText("未选择结果")

    def _on_image_selected(self, row: int):
        if row == -1 or row >= len(self.image_paths):
            self.current_index = -1
            self.interactive_image.set_image(None, reset_view=True)
            self.lbl_image_info.setText("未选择图像")
            return
        self.current_index = row
        img_path = self.image_paths[row]
        filename = os.path.basename(img_path)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None:
                self.lbl_image_info.setText(f"文件: {filename}")
                self.interactive_image.set_image(img, reset_view=True)
            else:
                self.lbl_image_info.setText(f"文件: {filename}\n无法读取图像")
                self.interactive_image.set_image(None, reset_view=True)
        except Exception as e:
            self.lbl_image_info.setText(f"文件: {filename}\n读取错误: {str(e)}")
            self.interactive_image.set_image(None, reset_view=True)

    def _on_result_selected(self, row: int):
        if row == -1 or row >= len(self.result_paths):
            self.result_index = -1
            self.interactive_image.set_image(None, reset_view=True)
            self.lbl_result_info.setText("未选择结果")
            return
        self.result_index = row
        result_path = self.result_paths[row]
        filename = os.path.basename(result_path)
        try:
            img = cv2.imread(result_path, cv2.IMREAD_COLOR)
            if img is not None:
                self.lbl_result_info.setText(f"结果文件: {filename}")
                self.interactive_image.set_image(img, reset_view=True)
            else:
                self.lbl_result_info.setText(f"结果文件: {filename}\n无法读取图像")
                self.interactive_image.set_image(None, reset_view=True)
        except Exception as e:
            self.lbl_result_info.setText(f"结果文件: {filename}\n读取错误: {str(e)}")
            self.interactive_image.set_image(None, reset_view=True)

    def on_choose_output(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if not folder:
            return
        self.output_folder = folder
        self.lbl_output.setText(f"输出路径: {folder}")

    @staticmethod
    def _parse_list(text: str) -> List[float]:
        vals = []
        for t in text.split(','):
            t = t.strip()
            if not t:
                continue
            try:
                vals.append(float(t))
            except ValueError:
                pass
        return [int(v) if isinstance(v, float) and v.is_integer() else v for v in vals]

    def on_run(self):
        if not self.image_paths:
            show_info(self, "请先选择图像")
            return
        output_folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if not output_folder:
            return
        self.output_folder = output_folder
        if self._contains_chinese(self.output_folder):
            show_info(self, "错误：输出路径不能包含中文字符！\n请选择不含中文的路径。")
            return
        try:
            tol_val = float(self.le_tol.text())
        except ValueError:
            tol_val = 1e-5
        bounds_list = self._parse_list(self.le_bounds.text())
        params = dict(
            anchor_interval=int(self.sp_anchor.value()),
            mode=self.cmb_mode.currentText(),
            only_rigid=self.chk_onlyrigid.isChecked(),
            scale_factor=float(self.dsb_scale.value()),
            n_matches=int(self.sp_matches.value()),
            mesh_size=int(self.sp_mesh.value()),
            shrink_factors=self._parse_list(self.le_shrink.text()) or [5],
            smoothing_sigmas=self._parse_list(self.le_sigma.text()) or [1],
            optimizer_iterations=int(self.sp_iter.value()),
            optimizer_tol=tol_val,
            optimizer_bounds=tuple(bounds_list[:2]) if bounds_list else (-100, 100)
        )
        self.progress_bar.setRange(0, len(self.image_paths))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.btn_run.setEnabled(False)
        self.worker = RegisterWorker(self.image_paths, self.output_folder, params)
        self.worker.progress.connect(self.progress_bar.setValue)  # type: ignore
        self.worker.finished.connect(self._on_done)  # type: ignore
        self.worker.message.connect(lambda msg: show_info(self, msg))  # type: ignore
        self.worker.start()

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False

    def _on_done(self):
        self.progress_bar.setVisible(False)
        self.btn_run.setEnabled(True)
        show_info(self, f"配准处理完成！\n结果保存在: {self.output_folder}")
        self.result_paths.clear()
        self.result_list_widget.clear()
        if os.path.isdir(self.output_folder):
            for fname in os.listdir(self.output_folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    fullpath = os.path.join(self.output_folder, fname)
                    self.result_paths.append(fullpath)
                    self.result_list_widget.addItem(fname)
            if self.result_paths:
                self.result_list_widget.setCurrentRow(0)
                self.lbl_result_info.setText(f"已发现 {len(self.result_paths)} 个结果文件")
            else:
                self.lbl_result_info.setText("未发现结果文件")
        else:
            self.lbl_result_info.setText("输出目录不存在")