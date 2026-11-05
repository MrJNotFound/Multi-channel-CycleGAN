import os
import sys
import numpy as np
import cv2
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QSlider, QLabel, QPushButton, QFileDialog,
                             QSplitter, QGroupBox, QGridLayout, QSizePolicy)
from PyQt6.QtCore import Qt, QTimer
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from scipy import ndimage


class TissueViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        # 初始化变量
        self.image_folder = ""  # 请修改为您的图像文件夹路径
        self.images = None  # 存储3D图像数据
        self.image_shape = None
        self.volume = None  # VTK体数据

        # 初始化UI
        self.init_ui()

        # 设置窗口大小
        self.resize(1400, 900)

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("组织切片三维可视化")

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)

        # 左侧：三维视图
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # VTK渲染窗口
        self.vtk_widget = QVTKRenderWindowInteractor()
        left_layout.addWidget(self.vtk_widget)

        # 控制按钮
        control_group = QGroupBox("控制")
        control_layout = QGridLayout()

        self.load_btn = QPushButton("加载图像文件夹")
        self.load_btn.clicked.connect(self.load_images)
        control_layout.addWidget(self.load_btn, 0, 0)

        self.reset_view_btn = QPushButton("重置视图")
        self.reset_view_btn.clicked.connect(self.reset_view)
        control_layout.addWidget(self.reset_view_btn, 0, 1)

        # 截面显示开关
        self.show_axial_cb = QPushButton("轴向截面: 开")
        self.show_axial_cb.setCheckable(True)
        self.show_axial_cb.setChecked(True)
        self.show_axial_cb.clicked.connect(self.update_view)
        control_layout.addWidget(self.show_axial_cb, 1, 0)

        self.show_coronal_cb = QPushButton("冠状截面: 开")
        self.show_coronal_cb.setCheckable(True)
        self.show_coronal_cb.setChecked(True)
        self.show_coronal_cb.clicked.connect(self.update_view)
        control_layout.addWidget(self.show_coronal_cb, 1, 1)

        self.show_sagittal_cb = QPushButton("矢状截面: 开")
        self.show_sagittal_cb.setCheckable(True)
        self.show_sagittal_cb.setChecked(True)
        self.show_sagittal_cb.clicked.connect(self.update_view)
        control_layout.addWidget(self.show_sagittal_cb, 2, 0)

        control_layout.setColumnStretch(0, 1)
        control_layout.setColumnStretch(1, 1)
        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)

        # 右侧：截面视图和滑块
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # 右侧视图分为三部分（三个截面的2D视图）
        sections_group = QGroupBox("截面视图")
        sections_layout = QVBoxLayout()

        # 轴向截面（XY平面）
        axial_group = QGroupBox("轴向截面 (Axial)")
        axial_layout = QVBoxLayout()
        self.axial_label = QLabel()
        self.axial_label.setMinimumSize(250, 250)
        self.axial_label.setAlignment(Qt.AlignCenter)
        self.axial_label.setStyleSheet("border: 1px solid gray; background-color: black;")
        axial_layout.addWidget(self.axial_label)

        self.axial_slider = QSlider(Qt.Horizontal)
        self.axial_slider.setMinimum(0)
        self.axial_slider.setMaximum(100)
        self.axial_slider.valueChanged.connect(self.update_view)
        axial_layout.addWidget(self.axial_slider)
        self.axial_value_label = QLabel("位置: 0")
        axial_layout.addWidget(self.axial_value_label)
        axial_group.setLayout(axial_layout)
        sections_layout.addWidget(axial_group)

        # 冠状截面（XZ平面）
        coronal_group = QGroupBox("冠状截面 (Coronal)")
        coronal_layout = QVBoxLayout()
        self.coronal_label = QLabel()
        self.coronal_label.setMinimumSize(250, 250)
        self.coronal_label.setAlignment(Qt.AlignCenter)
        self.coronal_label.setStyleSheet("border: 1px solid gray; background-color: black;")
        coronal_layout.addWidget(self.coronal_label)

        self.coronal_slider = QSlider(Qt.Horizontal)
        self.coronal_slider.setMinimum(0)
        self.coronal_slider.setMaximum(100)
        self.coronal_slider.valueChanged.connect(self.update_view)
        coronal_layout.addWidget(self.coronal_slider)
        self.coronal_value_label = QLabel("位置: 0")
        coronal_layout.addWidget(self.coronal_value_label)
        coronal_group.setLayout(coronal_layout)
        sections_layout.addWidget(coronal_group)

        # 矢状截面（YZ平面）
        sagittal_group = QGroupBox("矢状截面 (Sagittal)")
        sagittal_layout = QVBoxLayout()
        self.sagittal_label = QLabel()
        self.sagittal_label.setMinimumSize(250, 250)
        self.sagittal_label.setAlignment(Qt.AlignCenter)
        self.sagittal_label.setStyleSheet("border: 1px solid gray; background-color: black;")
        sagittal_layout.addWidget(self.sagittal_label)

        self.sagittal_slider = QSlider(Qt.Horizontal)
        self.sagittal_slider.setMinimum(0)
        self.sagittal_slider.setMaximum(100)
        self.sagittal_slider.valueChanged.connect(self.update_view)
        sagittal_layout.addWidget(self.sagittal_slider)
        self.sagittal_value_label = QLabel("位置: 0")
        sagittal_layout.addWidget(self.sagittal_value_label)
        sagittal_group.setLayout(sagittal_layout)
        sections_layout.addWidget(sagittal_group)

        sections_group.setLayout(sections_layout)
        right_layout.addWidget(sections_group)

        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([800, 400])

        main_layout.addWidget(splitter)

        # 初始化VTK场景
        self.setup_vtk()

        # 如果默认路径存在，自动加载
        if self.image_folder and os.path.exists(self.image_folder):
            QTimer.singleShot(100, self.load_images)

    def setup_vtk(self):
        """设置VTK渲染器"""
        # 创建渲染器
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.1, 0.2, 0.3)

        # 获取渲染窗口
        self.render_window = self.vtk_widget.GetRenderWindow()
        self.render_window.AddRenderer(self.renderer)

        # 创建交互器
        self.interactor = self.render_window.GetInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        # 添加坐标轴
        axes = vtk.vtkAxesActor()
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(axes)
        self.axes_widget.SetInteractor(self.interactor)
        self.axes_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
        self.axes_widget.SetEnabled(1)

        # 初始化截面平面
        self.axial_plane = vtk.vtkPlane()
        self.coronal_plane = vtk.vtkPlane()
        self.sagittal_plane = vtk.vtkPlane()

        # 初始渲染
        self.render_window.Render()

    def remove_background(self, image, threshold=240):
        """
        去除白色背景
        image: 输入图像 (H, W) 或 (H, W, C)
        threshold: 白色阈值，高于此值的像素被认为是背景
        返回: 背景去除后的图像，背景设为0
        """
        if len(image.shape) == 3:
            # 彩色图像：转换为灰度来判断背景
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask = gray < threshold
        else:
            # 灰度图像
            mask = image < threshold

        # 创建RGBA图像
        if len(image.shape) == 2:
            rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            rgba[:, :, 0] = image
            rgba[:, :, 1] = image
            rgba[:, :, 2] = image
        else:
            rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            rgba[:, :, :3] = image

        # 设置alpha通道
        rgba[:, :, 3] = mask * 255

        return rgba

    def load_images(self):
        """加载图像文件夹"""
        if not self.image_folder:
            folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
            if not folder:
                return
            self.image_folder = folder

        print(f"正在加载图像从: {self.image_folder}")

        # 获取所有图像文件
        image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']
        image_files = []
        for file in sorted(os.listdir(self.image_folder)):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(self.image_folder, file))

        if not image_files:
            print("错误：未找到图像文件！")
            return

        print(f"找到 {len(image_files)} 个图像文件")

        # 加载第一个图像以获取尺寸
        first_image = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
        if first_image is None:
            first_image = cv2.imread(image_files[0])
        h, w = first_image.shape[:2]

        # 初始化3D数组
        self.images = np.zeros((len(image_files), h, w, 4), dtype=np.uint8)

        # 加载并处理每个图像
        for i, file in enumerate(image_files):
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = cv2.imread(file)
                if img is None:
                    continue

            # 去除白色背景
            img_processed = self.remove_background(img)

            # 确保尺寸一致
            if img_processed.shape[:2] != (h, w):
                img_processed = cv2.resize(img_processed, (w, h))

            self.images[i] = img_processed

            if i % 10 == 0:
                print(f"处理进度: {i + 1}/{len(image_files)}")

        self.image_shape = self.images.shape
        print(f"加载完成！图像堆叠形状: {self.image_shape}")

        # 更新滑块范围
        self.axial_slider.setMaximum(self.image_shape[0] - 1)
        self.axial_slider.setValue(self.image_shape[0] // 2)

        self.coronal_slider.setMaximum(self.image_shape[1] - 1)
        self.coronal_slider.setValue(self.image_shape[1] // 2)

        self.sagittal_slider.setMaximum(self.image_shape[2] - 1)
        self.sagittal_slider.setValue(self.image_shape[2] // 2)

        # 创建体渲染
        self.create_volume_rendering()

        # 更新显示
        self.update_view()

        # 重置相机
        self.renderer.ResetCamera()
        self.render_window.Render()

    def create_volume_rendering(self):
        """创建体渲染"""
        if self.images is None:
            return

        # 清除现有的体渲染
        self.renderer.RemoveAllViewProps()

        # 将numpy数组转换为VTK数据
        # 使用alpha通道进行体渲染
        depth, height, width, _ = self.images.shape

        # 创建VTK图像数据
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(width, height, depth)
        image_data.SetSpacing(1.0, 1.0, 2.0)  # Z方向间距可以调整

        # 使用亮度值（取RGB的平均值或使用灰度值）
        vtk_array = vtk.vtkUnsignedCharArray()
        vtk_array.SetNumberOfComponents(1)
        vtk_array.SetName("Intensity")

        # 转换数据
        intensity_data = self.images[:, :, :, 0].flatten()  # 使用R通道
        vtk_array.SetArray(intensity_data, len(intensity_data), 1)
        image_data.GetPointData().SetScalars(vtk_array)

        # 创建体渲染映射器
        volume_mapper = vtk.vtkSmartVolumeMapper()
        volume_mapper.SetInputData(image_data)

        # 创建颜色传输函数
        color_func = vtk.vtkColorTransferFunction()
        color_func.AddRGBPoint(0, 0.0, 0.0, 0.0)
        color_func.AddRGBPoint(30, 0.5, 0.3, 0.2)
        color_func.AddRGBPoint(100, 0.8, 0.6, 0.4)
        color_func.AddRGBPoint(200, 0.9, 0.8, 0.7)
        color_func.AddRGBPoint(255, 1.0, 1.0, 1.0)

        # 创建不透明度传输函数
        opacity_func = vtk.vtkPiecewiseFunction()
        opacity_func.AddPoint(0, 0.0)
        opacity_func.AddPoint(20, 0.0)
        opacity_func.AddPoint(50, 0.1)
        opacity_func.AddPoint(100, 0.3)
        opacity_func.AddPoint(200, 0.6)
        opacity_func.AddPoint(255, 1.0)

        # 创建体属性
        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_func)
        volume_property.SetScalarOpacity(opacity_func)
        volume_property.SetInterpolationTypeToLinear()
        volume_property.ShadeOn()
        volume_property.SetAmbient(0.3)
        volume_property.SetDiffuse(0.7)
        volume_property.SetSpecular(0.3)

        # 创建体
        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(volume_mapper)
        self.volume.SetProperty(volume_property)

        # 添加体到渲染器
        self.renderer.AddVolume(self.volume)

        # 设置裁剪平面（稍后在update_view中更新）
        self.setup_clipping_planes()

    def setup_clipping_planes(self):
        """设置裁剪平面"""
        if self.images is None:
            return

        depth, height, width, _ = self.images.shape

        # 轴向裁剪平面（Z方向）
        self.axial_plane.SetOrigin(0, 0, 0)
        self.axial_plane.SetNormal(0, 0, 1)

        # 冠状裁剪平面（Y方向）
        self.coronal_plane.SetOrigin(0, 0, 0)
        self.coronal_plane.SetNormal(0, -1, 0)

        # 矢状裁剪平面（X方向）
        self.sagittal_plane.SetOrigin(0, 0, 0)
        self.sagittal_plane.SetNormal(1, 0, 0)

    def update_view(self):
        """更新所有视图"""
        if self.images is None:
            return

        # 获取滑块值
        z_pos = self.axial_slider.value()
        y_pos = self.coronal_slider.value()
        x_pos = self.sagittal_slider.value()

        # 更新标签
        self.axial_value_label.setText(f"位置: {z_pos}/{self.image_shape[0] - 1}")
        self.coronal_value_label.setText(f"位置: {y_pos}/{self.image_shape[1] - 1}")
        self.sagittal_value_label.setText(f"位置: {x_pos}/{self.image_shape[2] - 1}")

        # 更新裁剪平面
        if self.volume:
            self.renderer.RemoveAllClippingPlanes()

            # 轴向裁剪（显示指定Z层之上的部分）
            if self.show_axial_cb.isChecked():
                self.axial_plane.SetOrigin(0, 0, z_pos * 2.0)  # Z间距为2
                self.renderer.AddClippingPlane(self.axial_plane)
                self.show_axial_cb.setText("轴向截面: 开")
            else:
                self.show_axial_cb.setText("轴向截面: 关")

            # 冠状裁剪（显示指定Y层之前的部分）
            if self.show_coronal_cb.isChecked():
                self.coronal_plane.SetOrigin(0, y_pos, 0)
                self.renderer.AddClippingPlane(self.coronal_plane)
                self.show_coronal_cb.setText("冠状截面: 开")
            else:
                self.show_coronal_cb.setText("冠状截面: 关")

            # 矢状裁剪（显示指定X层之前的部分）
            if self.show_sagittal_cb.isChecked():
                self.sagittal_plane.SetOrigin(x_pos, 0, 0)
                self.renderer.AddClippingPlane(self.sagittal_plane)
                self.show_sagittal_cb.setText("矢状截面: 开")
            else:
                self.show_sagittal_cb.setText("矢状截面: 关")

        # 更新右侧截面图像
        self.update_section_views(z_pos, y_pos, x_pos)

        # 渲染
        self.render_window.Render()

    def update_section_views(self, z_pos, y_pos, x_pos):
        """更新截面视图"""
        if self.images is None:
            return

        # 提取截面图像
        axial_slice = self.images[z_pos, :, :, :3]  # RGB
        coronal_slice = self.images[:, y_pos, :, :3]  # RGB
        sagittal_slice = self.images[:, :, x_pos, :3]  # RGB

        # 转换为QImage并显示
        self.display_slice(axial_slice, self.axial_label, rotate=False)
        self.display_slice(coronal_slice, self.coronal_label, rotate=False)
        self.display_slice(sagittal_slice, self.sagittal_label, rotate=False)

    def display_slice(self, slice_data, label, rotate=False):
        """在QLabel上显示切片图像"""
        from PyQt5.QtGui import QImage, QPixmap

        if slice_data is None:
            return

        # 归一化到0-255
        if slice_data.dtype != np.uint8:
            slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8) * 255
            slice_data = slice_data.astype(np.uint8)

        h, w = slice_data.shape[:2]

        # 创建QImage
        if len(slice_data.shape) == 2:
            qimg = QImage(slice_data.data, w, h, w, QImage.Format_Grayscale8)
        else:
            qimg = QImage(slice_data.data, w, h, w * 3, QImage.Format_RGB888)

        # 缩放图像以适应标签
        pixmap = QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)

    def reset_view(self):
        """重置视图"""
        if self.images is not None:
            self.axial_slider.setValue(self.image_shape[0] // 2)
            self.coronal_slider.setValue(self.image_shape[1] // 2)
            self.sagittal_slider.setValue(self.image_shape[2] // 2)

            self.renderer.ResetCamera()
            self.update_view()


def main():
    # 检查并创建QApplication
    app = QApplication(sys.argv)

    # 创建主窗口
    viewer = TissueViewer()

    # 设置图像文件夹路径（请修改为您的实际路径）
    viewer.image_folder = r"D:/tissue_slices"  # 修改这里为您的图像文件夹路径

    viewer.show()

    # 初始化VTK交互器
    viewer.vtk_widget.GetRenderWindow().GetInteractor().Initialize()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()