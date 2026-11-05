import os
from typing import List, Optional
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QFileDialog, QSlider, QCheckBox, QSpinBox, QGroupBox, QListWidget,
    QProgressBar
)
from class_ImageViewer import InteractiveImageLabel
from func_freq_filter import freq_filter
from func_img_preprocessing import enhance_image
from func_tools import show_info


class BatchSaveWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(int, list)
    message = pyqtSignal(str)

    def __init__(self, image_list, save_dir, preprocess_params, use_freq):
        super().__init__()
        self.image_list = image_list
        self.save_dir = save_dir
        self.preprocess_params = preprocess_params
        self.use_freq = use_freq

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        """将任意类型图像安全转换为 uint8；彩色逐通道归一化，灰度整体归一化。"""
        if img.dtype == np.uint8:
            return img
        img = np.nan_to_num(img)
        if img.ndim == 2:
            min_val, max_val = float(img.min()), float(img.max())
            if max_val == min_val:
                return np.zeros_like(img, dtype=np.uint8)
            norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            return norm.astype(np.uint8)
        elif img.ndim == 3:
            channels = cv2.split(img)
            out = []
            for ch in channels:
                ch = np.nan_to_num(ch)
                mn, mx = float(ch.min()), float(ch.max())
                if mx == mn:
                    out.append(np.zeros_like(ch, dtype=np.uint8))
                else:
                    out.append(cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
            return cv2.merge(out)
        return img.astype(np.uint8)

    @staticmethod
    def _ensure_save_format(proc_img: np.ndarray, src_img: np.ndarray) -> np.ndarray:
        """
        确保保存的图像符合要求：
        - 若为灰度，则保存为 8-bit（单通道 uint8）
        - 若为彩色，则保留所有通道（3 通道或 4 通道），并保证为 uint8
        """
        if src_img.ndim == 2:
            # 源为灰度：输出单通道 uint8
            if proc_img.ndim == 3:
                proc_img = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
            proc_img = BatchSaveWorker._to_uint8(proc_img)
            if proc_img.ndim != 2:
                proc_img = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
            return proc_img

        if src_img.ndim == 3 and src_img.shape[2] == 3:
            # 源为 3 通道彩色：输出 3 通道 uint8
            if proc_img.ndim == 2:
                proc_img = cv2.cvtColor(proc_img, cv2.COLOR_GRAY2BGR)
            elif proc_img.ndim == 3 and proc_img.shape[2] == 4:
                # 若不期望 alpha，去掉 alpha
                proc_img = proc_img[:, :, :3]
            proc_img = BatchSaveWorker._to_uint8(proc_img)
            if proc_img.ndim != 3 or proc_img.shape[2] != 3:
                proc_img = cv2.cvtColor(proc_img, cv2.COLOR_GRAY2BGR)
            return proc_img

        if src_img.ndim == 3 and src_img.shape[2] == 4:
            # 源为 4 通道彩色：输出 4 通道 uint8（保留 alpha）
            alpha = src_img[:, :, 3]
            if proc_img.ndim == 2:
                bgr = cv2.cvtColor(proc_img, cv2.COLOR_GRAY2BGR)
                proc_img = cv2.merge((bgr, alpha))
            elif proc_img.ndim == 3 and proc_img.shape[2] == 3:
                proc_img = cv2.merge((proc_img, alpha))
            elif proc_img.ndim == 3 and proc_img.shape[2] == 4:
                # 已有 alpha，保持
                pass
            else:
                # 兜底：构造三通道再合并 alpha
                bgr = proc_img if proc_img.ndim == 3 else cv2.cvtColor(proc_img, cv2.COLOR_GRAY2BGR)
                proc_img = cv2.merge((bgr, alpha))
            proc_img = BatchSaveWorker._to_uint8(proc_img)
            return proc_img

        # 其他情况兜底
        return BatchSaveWorker._to_uint8(proc_img)

    def run(self):
        success_count = 0
        failed_files = []
        for i, img_path in enumerate(self.image_list):
            self.progress.emit(i)  # type: ignore
            try:
                # 保留原通道（灰度/彩色/含 alpha）
                src_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if src_img is None:
                    failed_files.append(os.path.basename(img_path))
                    continue

                clip = self.preprocess_params['clip']
                tile = self.preprocess_params['tile']
                try:
                    # 灰度保持单通道；彩色按三个通道分别 CLAHE 后合并
                    proc_img = enhance_image(
                        src_img,
                        clahe_clip_limit=clip,
                        clahe_tile_grid_size=tile,
                        mode="bgr"
                    )
                except Exception as e:
                    self.message.emit(f"{img_path} enhance_image error: {e}")  # type: ignore
                    proc_img = src_img

                # 频域滤波仅针对灰度；若启用则在当前结果上转灰后滤波，并按原格式输出
                if self.use_freq:
                    try:
                        gray_for_freq = proc_img
                        if gray_for_freq.ndim == 3:
                            gray_for_freq = cv2.cvtColor(gray_for_freq, cv2.COLOR_BGR2GRAY)
                        gray_for_freq = freq_filter(gray_for_freq, self.preprocess_params['angle'], self.preprocess_params['bandwidth'])
                        proc_img = gray_for_freq  # 先保持为灰度，稍后根据原图格式转换
                    except Exception as e:
                        self.message.emit(f"{img_path} freq_filter error: {e}")  # type: ignore

                # 按源图格式，确保保存格式满足要求
                proc_img = self._ensure_save_format(proc_img, src_img)

                # 保存：与输入图像扩展名一致
                original_filename = os.path.basename(img_path)
                name, ext = os.path.splitext(original_filename)
                save_filename = f"processed_{name}{ext}"
                save_path = os.path.join(self.save_dir, save_filename)
                if self._contains_chinese(save_path):
                    failed_files.append(f"{original_filename} (文件名包含中文)")
                    continue

                cv2.imwrite(save_path, proc_img)
                success_count += 1
                del src_img, proc_img
            except Exception as e:
                self.message.emit(f"处理 {img_path} 时出错: {e}")  # type: ignore
                failed_files.append(os.path.basename(img_path))
        self.progress.emit(len(self.image_list))  # type: ignore
        self.finished.emit(success_count, failed_files)  # type: ignore

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False

class PreprocessPage(QWidget):
    def __init__(self):
        super().__init__()
        self.image_list: List[str] = []
        self.current_index: int = -1
        self.result_cache: dict = {}
        self.use_freq = False
        self.show_original = True
        self.progress_bar: Optional[QProgressBar] = None
        self.batch_worker: Optional[BatchSaveWorker] = None
        self._build_ui()

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
            src_img = self._get_current_source_image()
            self.interactive_image.set_image(src_img, reset_view=False)
        else:
            self.show_original = False
            self.lbl_current_view.setText("当前显示: 处理结果")
            result_img = self._get_current_result_image()
            self.interactive_image.set_image(result_img, reset_view=False)

    def _get_current_source_image(self) -> Optional[np.ndarray]:
        if self.current_index == -1 or self.current_index >= len(self.image_list):
            return None
        img_path = self.image_list[self.current_index]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        return img

    def _get_current_result_image(self) -> Optional[np.ndarray]:
        if self.current_index == -1:
            return None
        if self.current_index in self.result_cache:
            return self.result_cache[self.current_index]
        src_img = self._get_current_source_image()
        if src_img is None:
            return None
        result_img = self._process_image(src_img)
        self._manage_cache()
        self.result_cache[self.current_index] = result_img
        return result_img

    def _manage_cache(self):
        max_cache_size = 5
        if len(self.result_cache) >= max_cache_size:
            oldest_key = min(self.result_cache.keys())
            del self.result_cache[oldest_key]

    def _process_image(self, img: np.ndarray) -> Optional[np.ndarray]:
        if img is None:
            return None
        clip = int(self.sld_clip.value())
        tile = (int(self.sp_tile_x.value()), int(self.sp_tile_y.value()))
        try:
            # 彩色按通道分别 CLAHE；灰度直接 CLAHE
            proc = enhance_image(
                img,
                clahe_clip_limit=clip,
                clahe_tile_grid_size=tile,
                mode="bgr"
            )
        except Exception as e:
            print("enhance_image error:", e)
            proc = img
        # 频域滤波（仅对灰度执行；预览转回 BGR 方便查看）
        if self.use_freq:
            angle = int(self.sld_angle.value())
            bandwidth = int(self.sld_band.value())
            try:
                gray_for_freq = proc
                if gray_for_freq.ndim == 3:
                    gray_for_freq = cv2.cvtColor(gray_for_freq, cv2.COLOR_BGR2GRAY)
                gray_for_freq = freq_filter(gray_for_freq, angle, bandwidth)
                proc = cv2.cvtColor(gray_for_freq, cv2.COLOR_GRAY2BGR)
            except Exception as e:
                print("freq_filter error:", e)
        # 预览返回彩色，便于显示
        result_img = proc if proc.ndim == 3 else cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
        return result_img

    @staticmethod
    def _on_slider_label(label: QLabel, text: str):
        label.setText(text)

    def _on_param_changed(self, *_args):
        self.use_freq = self.chk_freq.isChecked()
        self.result_cache.clear()
        self.update_preview()

    def _on_image_selected(self, row: int):
        if row == -1 or row >= len(self.image_list):
            self.current_index = -1
            self.interactive_image.set_image(None, reset_view=True)
            self.lbl_image_info.setText("未选择图像")
            return
        self.current_index = row
        img_path = self.image_list[row]
        filename = os.path.basename(img_path)
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
            filename = os.path.basename(path)
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
        self.interactive_image.set_image(None, reset_view=True)
        self.lbl_image_info.setText("未选择图像")
        if self.progress_bar is not None:
            self.progress_bar.setVisible(False)
            self.progress_bar.setFormat("")
        import gc
        gc.collect()

    def update_preview(self, reset_view: bool = False):
        if self.current_index == -1:
            self.interactive_image.set_image(None, reset_view=True)
            return
        if self.show_original:
            img = self._get_current_source_image()
            self.interactive_image.set_image(img, reset_view=reset_view)
        else:
            img = self._get_current_result_image()
            self.interactive_image.set_image(img, reset_view=reset_view)

    def on_save_single(self):
        if self.current_index == -1:
            show_info(self, "请先选择要保存的图像")
            return
        # 使用原图进行一次处理，以保留通道与位深一致
        src_path = self.image_list[self.current_index]
        src_img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        if src_img is None:
            show_info(self, "无法读取原图")
            return

        clip = int(self.sld_clip.value())
        tile = (int(self.sp_tile_x.value()), int(self.sp_tile_y.value()))
        try:
            proc_img = enhance_image(
                src_img,
                clahe_clip_limit=clip,
                clahe_tile_grid_size=tile,
                mode="bgr"
            )
        except Exception as e:
            self.result_cache.pop(self.current_index, None)
            show_info(self, f"增强出错: {e}")
            return

        if self.use_freq:
            angle = int(self.sld_angle.value())
            bandwidth = int(self.sld_band.value())
            try:
                gray_for_freq = proc_img
                if gray_for_freq.ndim == 3:
                    gray_for_freq = cv2.cvtColor(gray_for_freq, cv2.COLOR_BGR2GRAY)
                gray_for_freq = freq_filter(gray_for_freq, angle, bandwidth)
                proc_img = gray_for_freq  # 后续按源格式转换
            except Exception as e:
                show_info(self, f"频域滤波出错: {e}")

        # 确保保存格式符合要求
        # noinspection PyProtectedMember
        proc_img = BatchSaveWorker._ensure_save_format(proc_img, src_img)
        # 默认使用原扩展名
        current_filename = os.path.basename(src_path)
        name, ext = os.path.splitext(current_filename)
        default_name = f"processed_{name}{ext}"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存处理结果",
            default_name,
            filter="PNG (*.png);;JPG (*.jpg);;BMP (*.bmp);;TIFF (*.tif *.tiff)"
        )
        if not path:
            return
        cv2.imwrite(path, proc_img)
        show_info(self, "当前图像已保存")

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False

    def on_save_batch_async(self):
        if not self.image_list:
            show_info(self, "没有图像需要保存")
            return
        save_dir = QFileDialog.getExistingDirectory(self, "选择批量保存目录")
        if not save_dir:
            return
        if self._contains_chinese(save_dir):
            show_info(self, "错误：保存路径不能包含中文字符！\n请选择不含中文的路径。")
            return
        self.progress_bar.setRange(0, len(self.image_list))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.result_cache.clear()
        preprocess_params = dict(
            clip=int(self.sld_clip.value()),
            tile=(int(self.sp_tile_x.value()), int(self.sp_tile_y.value())),
            angle=int(self.sld_angle.value()),
            bandwidth=int(self.sld_band.value())
        )
        self.batch_worker = BatchSaveWorker(self.image_list, save_dir, preprocess_params, self.chk_freq.isChecked())
        self.batch_worker.progress.connect(self.progress_bar.setValue)  # type: ignore
        self.batch_worker.message.connect(lambda msg: show_info(self, msg))  # type: ignore
        self.batch_worker.finished.connect(self._on_batch_save_done)  # type: ignore
        self.batch_worker.start()

    def _on_batch_save_done(self, success_count, failed_files):
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.progress_bar.setFormat("处理完成")
        import threading
        def hide_progress_bar():
            import time; time.sleep(1.2)
            if self.progress_bar is not None:
                self.progress_bar.setVisible(False)
        threading.Thread(target=hide_progress_bar).start()
        if failed_files:
            show_info(self,
                      f"批量保存完成！\n成功: {success_count} 张\n失败: {len(failed_files)} 张\n失败文件: {', '.join(failed_files)}")
        else:
            show_info(self, f"批量保存完成！成功处理并保存 {success_count} 张图像")
        if self.current_index != -1:
            self.update_preview()
        self.batch_worker = None