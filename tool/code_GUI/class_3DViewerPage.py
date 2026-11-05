import glob
import os
import cv2
import imageio.v3 as iio
import napari
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QSlider, QGroupBox,
    QComboBox, QCheckBox, QFormLayout, QLineEdit, QMainWindow, QDockWidget, QSplitter, QSpinBox
)
from scipy.ndimage import label as ndi_label
from skimage.measure import regionprops_table


def load_stack_multi_type(folder, patterns, downsample=1, interp=cv2.INTER_AREA):
    files = []
    for pattern in patterns:
        files += glob.glob(os.path.join(folder, pattern))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No files found in {folder} with patterns {patterns}")
    stack = []
    for f in files:
        img = iio.imread(f)
        if img.ndim == 2:
            img = img[..., np.newaxis]
        if downsample > 1:
            h, w = img.shape[:2]
            img = cv2.resize(img, (w // downsample, h // downsample), interpolation=interp)
        stack.append(img)
    arr = np.stack(stack, axis=0)
    if arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)
    return arr


def random_color_dict(labels):
    import matplotlib
    unique_labels = np.unique(labels)
    color_dict = {}
    rng = np.random.default_rng(42)
    colormap = matplotlib.colormaps['tab20']
    for i, label_val in enumerate(unique_labels):
        if label_val == 0:
            continue
        if i < 20:
            rgba = colormap(i)
            color = (float(rgba[0]), float(rgba[1]), float(rgba[2]), 1.0)  # 确保是float类型
        else:
            color_vals = rng.uniform(0.1, 0.9, size=3).tolist()
            color = (float(color_vals[0]), float(color_vals[1]), float(color_vals[2]), 1.0)  # 使用元组和float
        color_dict[int(label_val)] = color  # type: ignore
    return color_dict


def repair_mask(mask, closing_size=3, dilation_size=1):
    from scipy.ndimage import generate_binary_structure, binary_closing, binary_dilation
    pad_width = max(closing_size, dilation_size)
    # 三维padding
    mask_padded = np.pad(mask, pad_width, mode='constant', constant_values=0)
    structure = generate_binary_structure(3, 2)
    closed = binary_closing(mask_padded, structure=structure, iterations=closing_size)
    if dilation_size > 0:
        closed = binary_dilation(closed, structure=structure, iterations=dilation_size)
    # 裁剪回原始大小
    slices = tuple(slice(pad_width, -pad_width) for _ in range(3))
    closed = closed[slices]
    return closed


class NapariEmbed(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.viewer = napari.Viewer(show=False)
        winn = self.viewer.window._qt_window
        winn.menuBar().setVisible(False)
        winn.statusBar().setVisible(False)
        for dock in winn.findChildren(QDockWidget):
            dock.setVisible(False)
        for child in winn.findChildren(QWidget):
            if child.objectName() == 'toolbar':
                child.setVisible(False)

        # 视角切换按钮
        self.btn_3d = QPushButton("三维视角")
        self.btn_xy = QPushButton("XY截面")
        self.btn_yz = QPushButton("YZ截面")
        self.btn_xz = QPushButton("XZ截面")
        for btn in [self.btn_3d, self.btn_xy, self.btn_yz, self.btn_xz]:
            btn.setCheckable(True)
            btn.setFixedSize(100, 36)
        self.btn_3d.clicked.connect(self.show_3d)  # type: ignore
        self.btn_xy.clicked.connect(lambda: self.show_slice('xy'))  # type: ignore
        self.btn_yz.clicked.connect(lambda: self.show_slice('yz'))  # type: ignore
        self.btn_xz.clicked.connect(lambda: self.show_slice('xz'))  # type: ignore

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_3d)
        btn_layout.addWidget(self.btn_xy)
        btn_layout.addWidget(self.btn_yz)
        btn_layout.addWidget(self.btn_xz)
        btn_layout.addStretch(1)

        main_layout = QVBoxLayout(self)
        btn_container = QWidget()
        btn_container.setLayout(btn_layout)
        main_layout.addWidget(btn_container)
        main_layout.addWidget(winn, stretch=1)
        self.setLayout(main_layout)
        self.show_3d()

    def update_button_status(self, mode):
        self.btn_3d.setChecked(mode == '3d')
        self.btn_xy.setChecked(mode == 'xy')
        self.btn_yz.setChecked(mode == 'yz')
        self.btn_xz.setChecked(mode == 'xz')

    def show_3d(self):
        self.viewer.dims.ndisplay = 3
        self.viewer.camera.angles = (0, 30, 60)
        self.update_button_status('3d')

    def show_slice(self, axis):
        self.viewer.dims.ndisplay = 2
        if axis == 'xy':
            self.viewer.dims.order = tuple([0, 1, 2])
        elif axis == 'yz':
            self.viewer.dims.order = tuple([2, 0, 1])
        elif axis == 'xz':
            self.viewer.dims.order = tuple([1, 0, 2])
        self.update_button_status(axis)


class Viewer3DLabelPage(QMainWindow):
    def __init__(self):
        super().__init__()
        self.last_single_label = None
        self.layer_label = None
        self.layer_b = None
        self.layer_g = None
        self.layer_r = None
        self.img_folder = ""
        self.label_folder = ""
        self.img_patterns = ["*.bmp", "*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]
        self.label_patterns = ["*.png", "*.tif", "*.tiff"]
        self.spacing = [1.0, 1.0, 1.0]
        self.downsample = 1
        self.img_stack = None
        self.label_stack = None
        self.labeled_cells = None
        self.num_cells = 0
        self.color_dict = {}
        self.img_loaded = False
        self.label_loaded = False
        self._build_ui()

    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        # 左侧面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(300)
        left_panel.setMinimumWidth(220)

        # 比例参数
        param_group = QGroupBox("参数设置")
        param_layout = QFormLayout()
        self.edit_spacing_z = QLineEdit("1.0")
        self.edit_spacing_y = QLineEdit("1.0")
        self.edit_spacing_x = QLineEdit("1.0")
        self.edit_spacing_z.setMaximumWidth(85)
        self.edit_spacing_y.setMaximumWidth(85)
        self.edit_spacing_x.setMaximumWidth(85)
        hz = QHBoxLayout()
        hz.addWidget(self.edit_spacing_z)
        hz.addWidget(QLabel("μm"))
        hy = QHBoxLayout()
        hy.addWidget(self.edit_spacing_y)
        hy.addWidget(QLabel("μm"))
        hx = QHBoxLayout()
        hx.addWidget(self.edit_spacing_x)
        hx.addWidget(QLabel("μm"))
        wz = QWidget()
        wz.setLayout(hz)
        wy = QWidget()
        wy.setLayout(hy)
        wx = QWidget()
        wx.setLayout(hx)
        param_layout.addRow("Z方向像素长度", wz)
        param_layout.addRow("Y方向像素长度", wy)
        param_layout.addRow("X方向像素长度", wx)
        param_group.setLayout(param_layout)
        left_layout.addWidget(param_group)

        # 文件夹读取
        file_group = QGroupBox("文件夹选择")
        file_layout = QHBoxLayout()
        btn_img = QPushButton("选择原图文件夹")
        btn_img.clicked.connect(self.on_choose_img_folder)  # type: ignore
        btn_label = QPushButton("选择Label文件夹")
        btn_label.clicked.connect(self.on_choose_label_folder)  # type: ignore
        file_layout.addWidget(btn_img)
        file_layout.addWidget(btn_label)
        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

        # 渲染参数
        opts_group = QGroupBox("显示控制")
        opts_layout = QFormLayout()

        # RGB渲染模式
        self.slider_rgb = QSlider(Qt.Orientation.Horizontal)
        self.slider_rgb.setRange(0, 100)
        self.slider_rgb.setValue(50)
        self.slider_rgb.valueChanged.connect(self.update_opacity_and_render)  # type: ignore
        opts_layout.addRow("RGB通道透明度", self.slider_rgb)
        self.cmb_rgb_render = QComboBox()
        self.cmb_rgb_render.addItems(["mip", "minip"])
        self.cmb_rgb_render.currentTextChanged.connect(self.update_opacity_and_render)  # type: ignore
        opts_layout.addRow("RGB渲染模式", self.cmb_rgb_render)

        # 标签透明度
        self.slider_label = QSlider(Qt.Orientation.Horizontal)
        self.slider_label.setRange(0, 100)
        self.slider_label.setValue(60)
        self.slider_label.valueChanged.connect(self.update_label_opacity)  # type: ignore
        opts_layout.addRow("标签透明度", self.slider_label)

        # 单标签选项
        self.cb_single_label = QCheckBox("只查看单个标签")
        self.cb_single_label.toggled.connect(self.on_toggle_single_label)  # type: ignore
        opts_layout.addRow(self.cb_single_label)
        label_spin_col_layout = QVBoxLayout()
        label_spin_col_layout.setContentsMargins(0, 0, 0, 0)
        label_spin_col_layout.setSpacing(0)
        label_text = QLabel("选择标签:")
        label_spin_layout = QHBoxLayout()
        self.spin_label = QSpinBox()
        self.spin_label.setMinimum(1)
        self.spin_label.setMaximum(1)
        self.spin_label.setEnabled(False)
        self.spin_label.valueChanged.connect(self.display_label_properties_spin)  # type: ignore
        label_spin_layout.addWidget(self.spin_label)
        label_spin_layout.addWidget(QLabel("号标签"))
        ws = QWidget()
        ws.setLayout(label_spin_layout)

        label_spin_col_layout.addWidget(label_text)
        label_spin_col_layout.addWidget(ws)

        label_spin_col_widget = QWidget()
        label_spin_col_widget.setLayout(label_spin_col_layout)

        opts_layout.addRow(label_spin_col_widget)

        self.label_props = QLabel("标签属性：")
        self.label_props.setWordWrap(True)
        opts_layout.addRow(self.label_props)

        opts_group.setLayout(opts_layout)
        left_layout.addWidget(opts_group)

        btn_show = QPushButton("三维显示")
        btn_show.clicked.connect(self.on_show_3d)  # type: ignore
        left_layout.addWidget(btn_show)

        self.status = QLabel("请先选择文件夹")
        left_layout.addWidget(self.status)
        left_layout.addStretch(1)
        splitter.addWidget(left_panel)

        self.napari_box = NapariEmbed(self)
        splitter.addWidget(self.napari_box)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

    def on_choose_img_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择原图文件夹")
        if not folder:
            return
        self.img_folder = folder
        self.img_loaded = False
        try:
            self.img_stack = load_stack_multi_type(self.img_folder, self.img_patterns, downsample=self.downsample)
            self.img_loaded = True
            self.status.setText(f"原图文件夹已读取: {folder}")
        except Exception as e:
            self.img_stack = None
            self.status.setText(f"原图文件夹读取失败: {e}")
            self.img_loaded = False

        self.update_status_bar()

    def on_choose_label_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择Label文件夹")
        if not folder:
            return
        self.label_folder = folder
        self.label_loaded = False
        try:
            self.label_stack = load_stack_multi_type(self.label_folder, self.label_patterns)
            self.label_loaded = True
            self.status.setText(f"Label文件夹已读取: {folder}")
        except Exception as e:
            self.label_stack = None
            self.status.setText(f"Label文件夹读取失败: {e}")
            self.label_loaded = False

        self.update_status_bar()

    def update_status_bar(self):
        text = []
        if self.img_folder:
            text.append(f"原图{'已读取' if self.img_loaded else '未读取'}")
        else:
            text.append("原图未选择")
        if self.label_folder:
            text.append(f"标签{'已读取' if self.label_loaded else '未读取'}")
        else:
            text.append("标签未选择")
        self.status.setText(" | ".join(text))

    def on_show_3d(self):
        try:
            self.spacing = [
                float(self.edit_spacing_z.text()),
                float(self.edit_spacing_y.text()),
                float(self.edit_spacing_x.text())
            ]
        except Exception as e:
            self.status.setText(f"像素长度参数错误: {e}")
            self.spin_label.setEnabled(False)
            self.spin_label.setMaximum(1)
            self.spin_label.setMinimum(1)
            self.label_props.setText("标签属性：")
            self.cb_single_label.setChecked(False)
            self.labeled_cells = None
            self.num_cells = 0
            return

        viewer = self.napari_box.viewer
        viewer.layers.clear()
        scale = tuple(self.spacing)

        any_image = self.img_stack is not None and self.img_loaded
        any_label = self.label_stack is not None and self.label_loaded

        if not any_image and not any_label:
            self.status.setText("请先选择并读取原图或标签文件夹")
            # 重置单标签相关控件
            self.spin_label.setEnabled(False)
            self.spin_label.setMaximum(1)
            self.spin_label.setMinimum(1)
            self.label_props.setText("标签属性：")
            self.cb_single_label.setChecked(False)
            self.labeled_cells = None
            self.num_cells = 0
            return

        show_ok = False

        # 原图显示
        if any_image:
            try:
                if self.img_stack.ndim == 4 and self.img_stack.shape[-1] >= 3:
                    R = np.ascontiguousarray(self.img_stack[..., 2], dtype=np.float32)
                    G = np.ascontiguousarray(self.img_stack[..., 1], dtype=np.float32)
                    B = np.ascontiguousarray(self.img_stack[..., 0], dtype=np.float32)
                    opacity = self.slider_rgb.value() / 100
                    rendering = self.cmb_rgb_render.currentText()
                    self.layer_r = viewer.add_image(R, name="Red", colormap="red", blending="additive", opacity=opacity, rendering=rendering, scale=scale)  # type: ignore
                    self.layer_g = viewer.add_image(G, name="Green", colormap="green", blending="additive", opacity=opacity, rendering=rendering, scale=scale)  # type: ignore
                    self.layer_b = viewer.add_image(B, name="Blue", colormap="blue", blending="additive", opacity=opacity, rendering=rendering, scale=scale)  # type: ignore
                else:
                    img_data = np.ascontiguousarray(self.img_stack, dtype=np.float32)
                    self.layer_r = viewer.add_image(img_data, name="image", opacity=1.0, scale=scale)  # type: ignore
                    self.layer_g = self.layer_b = None
                show_ok = True
            except Exception as e:
                self.status.setText(f"原图显示失败: {e}")
                return

        # 标签显示
        if any_label:
            try:
                mask = np.array(self.label_stack != 0, dtype=np.uint8)
                mask_repaired = repair_mask(mask, closing_size=3, dilation_size=1)
                self.labeled_cells, self.num_cells = ndi_label(mask_repaired)
                self.color_dict = random_color_dict(self.labeled_cells)
                self.layer_label = viewer.add_labels(
                    self.labeled_cells,
                    name="cells",
                    opacity=self.slider_label.value()/100,
                    blending='additive',
                    scale=scale
                )
                self.spin_label.setMaximum(self.num_cells if self.num_cells > 0 else 1)
                self.spin_label.setMinimum(1)
                self.spin_label.setEnabled(self.cb_single_label.isChecked())
                self.label_props.setText("标签属性：")
                self.layer_label.visible = True
                self.last_single_label = None
                show_ok = True
            except Exception as e:
                self.status.setText(f"标签图像显示失败: {e}")
                self.spin_label.setEnabled(False)
                self.spin_label.setMaximum(1)
                self.spin_label.setMinimum(1)
                self.label_props.setText("标签属性：")
                self.cb_single_label.setChecked(False)
                self.labeled_cells = None
                self.num_cells = 0
                return
        else:
            self.spin_label.setEnabled(False)
            self.spin_label.setMaximum(1)
            self.spin_label.setMinimum(1)
            self.label_props.setText("标签属性：")
            self.cb_single_label.setChecked(False)
            self.labeled_cells = None
            self.num_cells = 0

        if show_ok:
            self.napari_box.show_3d()
            self.update_status_bar()
            # 提示当前显示内容
            if any_image and any_label:
                self.status.setText(self.status.text() + " | 已显示原图和标签")
            elif any_image:
                self.status.setText(self.status.text() + " | 仅显示原图")
            elif any_label:
                self.status.setText(self.status.text() + " | 仅显示标签")

    def update_opacity_and_render(self):
        opacity = self.slider_rgb.value() / 100
        rendering = self.cmb_rgb_render.currentText()
        for l in [getattr(self, "layer_r", None), getattr(self, "layer_g", None), getattr(self, "layer_b", None)]:
            if l: l.opacity = opacity
            if l: l.rendering = rendering

    def update_label_opacity(self):
        if hasattr(self, "layer_label") and self.layer_label:
            self.layer_label.opacity = self.slider_label.value() / 100

    def on_toggle_single_label(self, checked):
        self.spin_label.setEnabled(checked and self.num_cells > 0)
        if not checked:
            if hasattr(self, "layer_label") and self.layer_label:
                self.layer_label.data = self.labeled_cells
                self.layer_label.visible = True
            self.label_props.setText("标签属性：")
            self.last_single_label = None
        else:
            self.display_label_properties_spin(self.spin_label.value())

    def display_label_properties_spin(self, val):
        if not self.cb_single_label.isChecked() or val is None or self.num_cells == 0:
            return
        label_id = int(val)
        try:
            if hasattr(self, "layer_label") and self.layer_label:
                mask = np.uint8(self.labeled_cells == label_id) * label_id
                self.layer_label.data = mask
                self.layer_label.visible = True
                self.last_single_label = label_id
            props = regionprops_table(np.uint8(self.labeled_cells==label_id), properties=[
                'area', 'equivalent_diameter', 'bbox', 'centroid', 'major_axis_length', 'minor_axis_length'
            ], spacing=tuple(self.spacing))
            area = props['area'][0] if 'area' in props and len(props['area']) else 0
            equiv_d = props['equivalent_diameter'][0] if 'equivalent_diameter' in props and len(props['equivalent_diameter']) else 0
            maj = props['major_axis_length'][0] if 'major_axis_length' in props and len(props['major_axis_length']) else 0
            minj = props['minor_axis_length'][0] if 'minor_axis_length' in props and len(props['minor_axis_length']) else 0
            msg = (
                f"标签ID: {label_id}\n"
                f"体积: {area:.2f} μm³\n"
                f"等效直径: {equiv_d:.2f} μm\n"
                f"主轴长度: {maj:.2f} μm\n"
                f"次轴长度: {minj:.2f} μm\n"
            )
            self.label_props.setText("标签属性：\n" + msg)
        except Exception as e:
            self.label_props.setText(f"属性计算失败: {e}")

    def showEvent(self, event):
        if hasattr(self, "layer_label") and self.layer_label and hasattr(self, "labeled_cells") and self.labeled_cells is not None:
            if not self.cb_single_label.isChecked() and getattr(self, "last_single_label", None) is not None:
                self.layer_label.data = self.labeled_cells
                self.layer_label.visible = True
                self.last_single_label = None