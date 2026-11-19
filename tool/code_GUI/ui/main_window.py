from importlib import import_module
import sys

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget,
    QHBoxLayout, QVBoxLayout, QLabel, QPushButton
)
from . import ICON_DIR


PAGES = (
    ("图像处理", "preprocessing", "PreprocessingPage", "requirements.txt"),
    ("图像配准", "registration", "RegistrationPage", "requirements.txt"),
    ("图像拼接", "stitching", "StitchingPage", "requirements.txt"),
    ("区域分析", "region_analysis", "RegionAnalysisPage", "requirements.txt"),
    ("三维查看", "volume_viewer", "VolumeViewerPage", "requirements-3d.txt"),
)


class WelcomePage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)
        layout.addStretch(1)

        # Logo
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        pix = QPixmap(str(ICON_DIR / "school_logo.png"))
        if not pix.isNull():
            self.logo_label.setPixmap(pix.scaled(180, 180, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        layout.addWidget(self.logo_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        # 提示文字
        self.tips_label = QLabel("点击顶部功能按键进入对应界面")
        self.tips_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.tips_label.setStyleSheet("font-size: 22px; color: #007bff; font-weight: bold; margin-top: 15px;")
        layout.addWidget(self.tips_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch(2)

        # 使logo和文字位置
        self.setMinimumWidth(400)
        self.setMinimumHeight(320)
        self.logo_label.setFixedSize(180, 180)
        self.tips_label.setMinimumHeight(40)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("断层图像三维重建与分析软件 V1.1")
        self.resize(1200, 800)
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # 顶部按钮栏
        top_toolbar = QWidget()
        top_toolbar.setFixedHeight(90)
        top_toolbar.setStyleSheet("background-color: #f8f9fa; border-bottom: 1px solid #dee2e6;")
        toolbar_layout = QHBoxLayout(top_toolbar)
        toolbar_layout.setContentsMargins(5, 5, 5, 5)
        toolbar_layout.setSpacing(20)
        self.btns = []
        self.lbl_descriptions = []

        for i, (name, module_name, _, _) in enumerate(PAGES):
            module_layout = QVBoxLayout()
            module_layout.setSpacing(5)
            module_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            icon_btn = QPushButton()
            icon_btn.setIcon(QIcon(str(ICON_DIR / f"{module_name}.png")))
            icon_btn.setIconSize(QSize(32, 32))
            icon_btn.setMinimumSize(40, 40)
            icon_btn.setToolTip(name)
            icon_btn.clicked.connect(lambda checked, idx=i: self._switch_page(idx))  # type: ignore
            self.btns.append(icon_btn)
            module_layout.addWidget(icon_btn, alignment=Qt.AlignmentFlag.AlignHCenter)
            desc_label = QLabel(name)
            desc_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            self.lbl_descriptions.append(desc_label)
            module_layout.addWidget(desc_label, alignment=Qt.AlignmentFlag.AlignHCenter)
            module_widget = QWidget()
            module_widget.setLayout(module_layout)
            toolbar_layout.addWidget(module_widget)

        toolbar_layout.addStretch()
        main_layout.addWidget(top_toolbar)

        # 页面区域
        self.pages = QStackedWidget()
        self.welcome_page = WelcomePage()
        self.pages.addWidget(self.welcome_page)
        self._loaded_pages = {}
        self._unavailable_page = QLabel()
        self._unavailable_page.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._unavailable_page.setWordWrap(True)
        self._unavailable_page.setMargin(24)
        self.pages.addWidget(self._unavailable_page)
        main_layout.addWidget(self.pages, 1)
        # 默认显示欢迎页
        self.pages.setCurrentIndex(0)
        self._update_button_styles(-1)

    def _switch_page(self, index):
        if index not in self._loaded_pages:
            name, module_name, class_name, requirements = PAGES[index]
            try:
                page = getattr(import_module(f".pages.{module_name}", __package__), class_name)()
            except (ImportError, RuntimeError, OSError) as error:
                self._unavailable_page.setText(
                    f"无法打开{name}\n{error}\n请检查 {requirements} 中的依赖，再点击此模块重试。"
                )
                self.pages.setCurrentWidget(self._unavailable_page)
                self._update_button_styles(index)
                return
            self.pages.addWidget(page)
            self._loaded_pages[index] = page
        self.pages.setCurrentWidget(self._loaded_pages[index])
        self._update_button_styles(index)

    def _update_button_styles(self, selected_index):
        for i, btn in enumerate(self.btns):
            desc_label = self.lbl_descriptions[i]
            if i == selected_index:
                btn.setStyleSheet(
                    "QPushButton { "
                    "background-color: #007bff; "
                    "color: white; "
                    "border: 2px solid #0056b3; "
                    "border-radius: 8px; "
                    "} "
                    "QPushButton:hover { "
                    "background-color: #0056b3; "
                    "} "
                    "QPushButton:pressed { "
                    "background-color: #004085; "
                    "}"
                )
                desc_label.setStyleSheet("color: #007bff; font-size: 12px; font-weight: bold;")
            else:
                btn.setStyleSheet(
                    "QPushButton { "
                    "background-color: #ffffff; "
                    "color: #6c757d; "
                    "border: 1px solid #dee2e6; "
                    "border-radius: 8px; "
                    "} "
                    "QPushButton:hover { "
                    "background-color: #f8f9fa; "
                    "border: 1px solid #adb5bd; "
                    "color: #495057; "
                    "} "
                    "QPushButton:pressed { "
                    "background-color: #e9ecef; "
                    "}"
                )
                desc_label.setStyleSheet("color: #6c757d; font-size: 12px; font-weight: normal;")

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()
