import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QDockWidget
import numpy as np
import cv2
import os
import napari

class SimpleNapariUI(QMainWindow):
    def __init__(self, volume_data):
        super().__init__()
        self.setWindowTitle("简化版Napari界面")
        self.viewer = napari.Viewer(show=False)

        # 数据 shape (Z, Y, X, C)
        volume_swap = np.transpose(volume_data, (1, 0, 2, 3))
        # 拆分通道
        red = volume_swap[..., 0]
        green = volume_swap[..., 1]
        blue = volume_swap[..., 2]
        scale_vector = [1, 10, 1]

        # 分别添加，每个layer设置颜色和混合模式
        self.viewer.add_image(red, scale=scale_vector, name='Red', colormap='red', blending='additive', rendering='minip')
        self.viewer.add_image(green, scale=scale_vector, name='Green', colormap='green', blending='additive', rendering='minip')
        self.viewer.add_image(blue, scale=scale_vector, name='Blue', colormap='blue', blending='additive', rendering='minip')

        # 隐藏napari多余控件
        win = self.viewer.window._qt_window
        win.menuBar().setVisible(False)
        win.statusBar().setVisible(False)
        # 隐藏左侧layer controls和layer list
        for dock in win.findChildren(QDockWidget):
            dock.setVisible(False)
        # 隐藏底部工具栏
        for child in win.findChildren(QWidget):
            if child.objectName() == 'toolbar':
                child.setVisible(False)

        # 4个按钮，改为checkable
        self.btn_3d = QPushButton("三维视角")
        self.btn_xy = QPushButton("XY截面")
        self.btn_yz = QPushButton("YZ截面")
        self.btn_xz = QPushButton("XZ截面")
        for btn in [self.btn_3d, self.btn_xy, self.btn_yz, self.btn_xz]:
            btn.setCheckable(True)

        self.btn_3d.clicked.connect(self.show_3d)
        self.btn_xy.clicked.connect(lambda: self.show_slice('xy'))
        self.btn_yz.clicked.connect(lambda: self.show_slice('yz'))
        self.btn_xz.clicked.connect(lambda: self.show_slice('xz'))

        layout = QVBoxLayout()
        layout.addWidget(self.btn_3d)
        layout.addWidget(self.btn_xy)
        layout.addWidget(self.btn_yz)
        layout.addWidget(self.btn_xz)
        layout.addWidget(win)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 默认三维视图
        self.show_3d()

    def update_button_status(self, mode):
        self.btn_3d.setChecked(mode == '3d')
        self.btn_xy.setChecked(mode == 'xy')
        self.btn_yz.setChecked(mode == 'yz')
        self.btn_xz.setChecked(mode == 'xz')

    def show_3d(self):
        self.viewer.dims.ndisplay = 3
        self.update_button_status('3d')

    def show_slice(self, axis):
        self.viewer.dims.ndisplay = 2
        # napari 4维数据，order必须是4个轴
        if axis == 'xy':
            self.viewer.dims.order = [0, 1, 2]  # Z, Y, X, C
        elif axis == 'yz':
            self.viewer.dims.order = [2, 1, 0]  # X, Y, Z, C
        elif axis == 'xz':
            self.viewer.dims.order = [1, 0, 2]  # Y, X, Z, C
        self.update_button_status(axis)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 1. 读取图片并按顺序堆叠
    input_path = 'C:/Users/30927/Desktop/brain regist/elastic_ROI1'
    file_list = [f for f in os.listdir(input_path) if f.endswith(('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
    file_list.sort()

    scale_factor = 2
    imgs = []
    for f in file_list:
        img = cv2.imread(os.path.join(input_path, f), cv2.IMREAD_COLOR_RGB)
        img_resized = cv2.resize(img, [int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor)])
        imgs.append(np.array(img_resized))
    # 堆叠成体数据 (Z, H, W, 3)
    volume = np.stack(imgs, axis=0)

    window = SimpleNapariUI(volume)
    window.show()
    sys.exit(app.exec())
