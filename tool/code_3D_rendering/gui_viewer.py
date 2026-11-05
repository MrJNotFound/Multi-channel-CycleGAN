import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QColor, QPainter
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QSlider,
    QGroupBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMainWindow,
    QSplitter,
    QSpinBox,
    QDoubleSpinBox,
    QProgressDialog,
)

from cube_view import CubeView
from volume_io import load_image_stack
from volume_tools import clamp_index


class _VideoExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Export Settings")
        layout = QFormLayout(self)

        path_row = QHBoxLayout()
        self.edit_path = QLineEdit()
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse)
        path_row.addWidget(self.edit_path, 1)
        path_row.addWidget(btn_browse)

        self.spin_w = QSpinBox()
        self.spin_w.setRange(320, 3840)
        self.spin_w.setValue(1280)
        self.spin_h = QSpinBox()
        self.spin_h.setRange(240, 2160)
        self.spin_h.setValue(720)
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 60)
        self.spin_fps.setValue(24)

        layout.addRow("Output:", path_row)
        layout.addRow("Width:", self.spin_w)
        layout.addRow("Height:", self.spin_h)
        layout.addRow("FPS:", self.spin_fps)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _browse(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save video", "", "MP4 (*.mp4)")
        if path:
            self.edit_path.setText(path)

    def settings(self):
        return {
            "path": self.edit_path.text(),
            "width": self.spin_w.value(),
            "height": self.spin_h.value(),
            "fps": self.spin_fps.value(),
        }


class VolumeViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Volume Rendering")
        self.resize(1400, 900)

        self.img_folder = ""
        self.img_stack_raw = None
        self.display_volume = None
        self.is_color = False
        self.current_spacing = None

        self._build_ui()

    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(320)
        left_panel.setMinimumWidth(240)

        param_group = QGroupBox("Parameters")
        param_layout = QFormLayout()
        self.edit_spacing_z = QLineEdit("1.0")
        self.edit_spacing_y = QLineEdit("1.0")
        self.edit_spacing_x = QLineEdit("1.0")
        for edit in [self.edit_spacing_z, self.edit_spacing_y, self.edit_spacing_x]:
            edit.setMaximumWidth(85)

        param_layout.addRow("Spacing Z", self.edit_spacing_z)
        param_layout.addRow("Spacing Y", self.edit_spacing_y)
        param_layout.addRow("Spacing X", self.edit_spacing_x)

        self.spin_downsample = QDoubleSpinBox()
        self.spin_downsample.setRange(1.0, 10.0)
        self.spin_downsample.setSingleStep(0.5)
        self.spin_downsample.setValue(1.0)
        self.spin_downsample.setMaximumWidth(85)
        param_layout.addRow("Downsample", self.spin_downsample)

        param_group.setLayout(param_layout)
        left_layout.addWidget(param_group)

        file_group = QGroupBox("Image Folder")
        file_layout = QVBoxLayout()
        self.btn_select_folder = QPushButton("Select folder")
        self.btn_select_folder.clicked.connect(self.on_choose_img_folder)  # type: ignore
        self.lbl_folder = QLabel("No folder selected")
        self.lbl_folder.setWordWrap(True)
        file_layout.addWidget(self.btn_select_folder)
        file_layout.addWidget(self.lbl_folder)
        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

        render_group = QGroupBox("Rendering")
        render_layout = QFormLayout()
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(60)
        self.slider_opacity.valueChanged.connect(self.update_render_settings)  # type: ignore
        render_layout.addRow("Opacity", self.slider_opacity)
        self.cmb_rendering = QComboBox()
        self.cmb_rendering.addItems(["mip", "minip", "translucent"])
        self.cmb_rendering.currentTextChanged.connect(self.update_render_settings)  # type: ignore
        render_layout.addRow("Rendering", self.cmb_rendering)
        render_group.setLayout(render_layout)
        left_layout.addWidget(render_group)

        self.btn_render = QPushButton("Render 3D")
        self.btn_render.clicked.connect(self.on_render_3d)  # type: ignore
        left_layout.addWidget(self.btn_render)

        self.btn_export_video = QPushButton("Export Video")
        self.btn_export_video.clicked.connect(self.on_export_video)  # type: ignore
        self.btn_export_video.setEnabled(False)
        left_layout.addWidget(self.btn_export_video)

        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        left_layout.addWidget(self.status)
        left_layout.addStretch(1)

        splitter.addWidget(left_panel)

        self.cube_view = CubeView(self)
        splitter.addWidget(self.cube_view)

        self.slice_panel = self._build_slice_panel()
        splitter.addWidget(self.slice_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        self.setCentralWidget(splitter)

    def on_choose_img_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select image folder")
        if not folder:
            return
        self.img_folder = folder
        self.lbl_folder.setText(folder)
        self.status.setText("Folder selected")

    def on_render_3d(self):
        if not self.img_folder:
            self.status.setText("Please select an image folder")
            return

        try:
            spacing = (
                float(self.edit_spacing_z.text()),
                float(self.edit_spacing_y.text()),
                float(self.edit_spacing_x.text()),
            )
        except ValueError as exc:
            self.status.setText(f"Invalid spacing: {exc}")
            return

        try:
            stack, is_color = load_image_stack(
                self.img_folder,
                ["*.bmp", "*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"],
                downsample=float(self.spin_downsample.value()),
            )
        except Exception as exc:
            self.status.setText(f"Load failed: {exc}")
            return

        self.img_stack_raw = stack
        self.display_volume = stack
        self.is_color = is_color
        self.current_spacing = spacing

        self._set_cut_controls_enabled(True)
        self._init_cut_controls(stack.shape)
        self._update_cube_view()
        self.btn_export_video.setEnabled(True)

        self.status.setText("Rendered")

    def update_render_settings(self):
        if self.display_volume is None:
            return
        self._update_cube_view()

    def _build_slice_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        group = QGroupBox("Cut Planes")
        group_layout = QVBoxLayout(group)

        self.lbl_cut_z = QLabel("Cut Z")
        self.slider_cut_z = QSlider(Qt.Orientation.Horizontal)
        self.slider_cut_z.valueChanged.connect(self.on_cut_changed)  # type: ignore

        self.lbl_cut_y = QLabel("Cut Y")
        self.slider_cut_y = QSlider(Qt.Orientation.Horizontal)
        self.slider_cut_y.valueChanged.connect(self.on_cut_changed)  # type: ignore

        self.lbl_cut_x = QLabel("Cut X")
        self.slider_cut_x = QSlider(Qt.Orientation.Horizontal)
        self.slider_cut_x.valueChanged.connect(self.on_cut_changed)  # type: ignore

        group_layout.addWidget(self.lbl_cut_z)
        group_layout.addWidget(self.slider_cut_z)
        group_layout.addWidget(self.lbl_cut_y)
        group_layout.addWidget(self.slider_cut_y)
        group_layout.addWidget(self.lbl_cut_x)
        group_layout.addWidget(self.slider_cut_x)

        group.setLayout(group_layout)

        layout.addWidget(group)
        layout.addStretch(1)
        panel.setLayout(layout)
        self._set_cut_controls_enabled(False)
        return panel

    def _set_cut_controls_enabled(self, enabled: bool):
        for w in [self.slider_cut_z, self.slider_cut_y, self.slider_cut_x]:
            w.setEnabled(enabled)

    def _init_cut_controls(self, shape):
        if len(shape) < 3:
            self._set_cut_controls_enabled(False)
            return
        self.slider_cut_z.setRange(0, max(0, shape[0] - 1))
        self.slider_cut_y.setRange(0, max(0, shape[1] - 1))
        self.slider_cut_x.setRange(0, max(0, shape[2] - 1))
        self.slider_cut_z.setValue(shape[0] - 1 if shape[0] > 0 else 0)
        self.slider_cut_y.setValue(shape[1] - 1 if shape[1] > 0 else 0)
        self.slider_cut_x.setValue(shape[2] - 1 if shape[2] > 0 else 0)
        self._set_cut_controls_enabled(True)

    def on_cut_changed(self):
        if self.display_volume is None:
            return
        self._update_cube_view()

    def _update_cube_view(self):
        if self.display_volume is None:
            return
        self.cube_view.set_volume(self.display_volume, self.is_color, self.current_spacing)
        self.cube_view.update_view(
            self.slider_cut_z.value(),
            self.slider_cut_y.value(),
            self.slider_cut_x.value(),
            self.cmb_rendering.currentText(),
        )

    def on_export_video(self):
        if self.display_volume is None:
            return
        dlg = _VideoExportDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        s = dlg.settings()
        if not s["path"]:
            self.status.setText("No output path specified")
            return
        self._export_video(s)

    def _export_video(self, settings):
        s = settings
        vol = self.display_volume
        z_max = vol.shape[0] - 1
        y_max = vol.shape[1] - 1
        x_max = vol.shape[2] - 1

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(s["path"], fourcc, s["fps"], (s["width"], s["height"]))
        if not writer.isOpened():
            self.status.setText("Failed to create video writer")
            return

        orig_z = self.slider_cut_z.value()
        orig_y = self.slider_cut_y.value()
        orig_x = self.slider_cut_x.value()

        # Block slider signals to avoid redundant renders during export
        for sl in [self.slider_cut_z, self.slider_cut_y, self.slider_cut_x]:
            sl.blockSignals(True)

        half_z = s["fps"] * 5
        half_y = s["fps"] * 8
        half_x = s["fps"] * 8
        pause_frames = s["fps"] * 1
        gap_frames = s["fps"] * 3

        axes = [
            (z_max, self.slider_cut_z, half_z),
            (y_max, self.slider_cut_y, half_y),
            (x_max, self.slider_cut_x, half_x),
        ]

        total_frames = sum(h * 2 + pause_frames + gap_frames for _, _, h in axes) + gap_frames

        progress = QProgressDialog("Exporting video...", "Cancel", 0, total_frames, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        QApplication.processEvents()

        # Preserve original rendering mode and set to translucent for export
        orig_rendering = self.cmb_rendering.currentText()
        if orig_rendering != "translucent":
            self.cmb_rendering.setCurrentText("translucent")

        cancelled = False
        frame_count = 0

        for ax_idx, (ax_max, slider, half_sweep) in enumerate(axes):
            if cancelled:
                break

            other_sliders = [(z_max, self.slider_cut_z), (y_max, self.slider_cut_y), (x_max, self.slider_cut_x)]
            other_sliders.pop(ax_idx)

            # --- Gap / transition (full volume, all axes max) ---
            for other_max, other_slider in other_sliders:
                other_slider.setValue(other_max)
            slider.setValue(ax_max)

            for _ in range(gap_frames):
                if progress.wasCanceled():
                    cancelled = True
                    break
                self._render_and_write(writer, s["width"], s["height"])
                frame_count += 1
                progress.setValue(frame_count)
                QApplication.processEvents()

            if cancelled:
                break

            # --- Sweep: max → 0 → max ---
            for other_max, other_slider in other_sliders:
                other_slider.setValue(other_max)

            # Phase 1: max → 0
            for fi in range(half_sweep):
                if progress.wasCanceled():
                    cancelled = True
                    break
                pos = 1.0 - fi / half_sweep
                slider.setValue(int(round(pos * ax_max)))
                self._render_and_write(writer, s["width"], s["height"])
                frame_count += 1
                progress.setValue(frame_count)
                QApplication.processEvents()

            if cancelled:
                break

            # Phase 2: pause at 0
            slider.setValue(0)
            for _ in range(pause_frames):
                if progress.wasCanceled():
                    cancelled = True
                    break
                self._render_and_write(writer, s["width"], s["height"])
                frame_count += 1
                progress.setValue(frame_count)
                QApplication.processEvents()

            if cancelled:
                break

            # Phase 3: 0 → max
            for fi in range(half_sweep):
                if progress.wasCanceled():
                    cancelled = True
                    break
                pos = fi / half_sweep
                slider.setValue(int(round(pos * ax_max)))
                self._render_and_write(writer, s["width"], s["height"])
                frame_count += 1
                progress.setValue(frame_count)
                QApplication.processEvents()

            if cancelled:
                break

        # --- Final gap: full volume for 3 seconds ---
        if not cancelled:
            self.slider_cut_z.setValue(z_max)
            self.slider_cut_y.setValue(y_max)
            self.slider_cut_x.setValue(x_max)
            for _ in range(gap_frames):
                if progress.wasCanceled():
                    cancelled = True
                    break
                self._render_and_write(writer, s["width"], s["height"])
                frame_count += 1
                progress.setValue(frame_count)
                QApplication.processEvents()

        writer.release()
        progress.close()

        # Restore original state
        for sl in [self.slider_cut_z, self.slider_cut_y, self.slider_cut_x]:
            sl.blockSignals(False)
        self.slider_cut_z.setValue(orig_z)
        self.slider_cut_y.setValue(orig_y)
        self.slider_cut_x.setValue(orig_x)
        self.cmb_rendering.setCurrentText(orig_rendering)

        self.status.setText(
            f"Video saved: {s['path']} ({frame_count} frames)"
            if not cancelled else "Export cancelled"
        )

    def _render_and_write(self, writer, target_w, target_h):
        img = self.cube_view.render_at_size(
            self.slider_cut_z.value(),
            self.slider_cut_y.value(),
            self.slider_cut_x.value(),
            self.cmb_rendering.currentText(),
            target_w, target_h,
        )
        if img is None or img.isNull():
            return

        # Composite onto white background for video
        canvas = QImage(target_w, target_h, QImage.Format.Format_ARGB32)
        canvas.fill(QColor(255, 255, 255))
        painter = QPainter(canvas)
        painter.drawImage(0, 0, img)
        painter.end()

        canvas = canvas.convertToFormat(QImage.Format.Format_RGB888)
        w, h = canvas.width(), canvas.height()
        ptr = canvas.bits()
        ptr.setsize(h * w * 3)
        rgb = np.frombuffer(ptr, np.uint8).reshape(h, w, 3)
        writer.write(rgb[:, :, ::-1])  # RGB → BGR for OpenCV

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_cube_view()
