from __future__ import annotations

import json
import math
import os
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QPointF, QRunnable, QSettings, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .crop import (
    CropSettings,
    export_patches,
    load_crop_settings,
    patch_layout,
    save_crop_settings,
)
from .file_index import TileIndexError, discover_numeric_tiles
from .geometry import mosaic_bounds, regular_positions
from .models import (
    AutoSearchSettings,
    FastAxis,
    GridGeometry,
    MarkFilterSettings,
    ScanPattern,
    StartCorner,
    StitchProject,
    TileSet,
    ToolMarkMethod,
)
from .mosaic import render_mosaic, render_mosaic_thumbnail
from .preview import compose_local_preview, estimated_preview_memory_mb
from .registration import AutoRegistrationResult, AutoStitcher
from .raster import open_raster
from .toolmarks import (
    ToolMarkOutput,
    load_toolmark_settings,
    linked_source_folder,
    process_toolmark_input,
    preview_tool_marks,
    remove_tool_marks,
    save_toolmark_settings,
    suggested_toolmark_output,
)


APP_STYLE = """
QMainWindow, QWidget { background: #f3f5f7; color: #1c252d; font-size: 13px; }
QTabWidget::pane { border: 0; }
QTabBar::tab { min-width: 104px; padding: 10px 16px; border: 0; border-bottom: 2px solid transparent; background: #ffffff; }
QTabBar::tab:selected { color: #176ca8; border-bottom-color: #176ca8; }
QGroupBox { background: #ffffff; border: 1px solid #dbe2e8; border-radius: 7px; margin-top: 10px; padding: 12px 10px 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { min-height: 30px; background: #ffffff; border: 1px solid #cfd8df; border-radius: 5px; padding: 1px 6px; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #2a7db5; }
QPushButton, QToolButton { min-height: 31px; border: 1px solid #cfd8df; border-radius: 6px; background: #ffffff; padding: 2px 10px; }
QPushButton:hover, QToolButton:hover { background: #edf4f8; }
QPushButton[primary="true"] { color: white; background: #176ca8; border-color: #176ca8; font-weight: 600; }
QPushButton[primary="true"]:hover { background: #125d92; }
QPushButton:disabled { color: #9aa5ae; background: #edf0f2; }
QToolButton:checked { color: #176ca8; border-color: #176ca8; background: #edf6fc; }
QToolButton[pathChoice="true"] { min-width: 118px; min-height: 92px; }
QGraphicsView { background: #151d24; border: 0; }
QScrollArea { border: 0; background: #ffffff; }
QProgressBar { min-height: 18px; border: 1px solid #d0d8de; border-radius: 5px; background: #ffffff; text-align: center; }
QProgressBar::chunk { background: #3a8dbd; border-radius: 4px; }
QLabel[muted="true"] { color: #667582; }
QFrame[panel="true"] { background: #ffffff; border: 1px solid #dbe2e8; border-radius: 7px; }
"""


class WorkerSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    progress = Signal(int, int, str)
    finished = Signal()
    released = Signal(object)


class Worker(QRunnable):
    def __init__(self, function: Callable[..., Any], *args: Any, with_progress: bool = False, **kwargs: Any) -> None:
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.with_progress = with_progress
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            if self.with_progress:
                self.kwargs["progress"] = self.signals.progress.emit
            result = self.function(*self.args, **self.kwargs)
        except Exception:
            self._safe_emit(self.signals.error, traceback.format_exc())
        else:
            self._safe_emit(self.signals.result, result)
        finally:
            self._safe_emit(self.signals.finished)
            self._safe_emit(self.signals.released, self)

    @staticmethod
    def _safe_emit(signal, *args: Any) -> None:
        try:
            signal.emit(*args)
        except RuntimeError:
            # The application can be closed while a background task is
            # finishing.  Dropping that late notification is safe.
            pass


def _start_worker(owner: QWidget, worker: Worker) -> None:
    """Keep QRunnable and signal objects alive until queued slots finish."""
    active: set[Worker] = owner._active_workers  # type: ignore[attr-defined]
    worker.setAutoDelete(False)
    active.add(worker)
    worker.signals.released.connect(active.discard)
    owner.thread_pool.start(worker)  # type: ignore[attr-defined]


class ZoomView(QGraphicsView):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._auto_fit = True

    def set_pil_image(self, image: Image.Image) -> None:
        array = np.asarray(image.convert("RGB")).copy()
        qimage = QImage(array.data, array.shape[1], array.shape[0], array.strides[0], QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage)
        self.scene().clear()
        self._pixmap_item = self.scene().addPixmap(pixmap)
        self.scene().setSceneRect(self._pixmap_item.boundingRect())
        self._auto_fit = True
        self._fit_current()

    def _fit_current(self) -> None:
        if self._pixmap_item is not None and not self.sceneRect().isEmpty():
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if self._auto_fit:
            QTimer.singleShot(0, self._fit_current)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._auto_fit:
            self._fit_current()

    def set_path(self, path: Path, max_size: int = 4096) -> None:
        with Image.open(path) as image:
            display = image.convert("RGB")
            display.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            self.set_pil_image(display.copy())

    def wheelEvent(self, event) -> None:  # noqa: N802
        self._auto_fit = False
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self._auto_fit = True
        self._fit_current()
        super().mouseDoubleClickEvent(event)


class FloatControl(QWidget):
    valueChanged = Signal(float)

    def __init__(self, minimum: float, maximum: float, value: float, suffix: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(round(minimum * 10), round(maximum * 10))
        self.spin = QDoubleSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setDecimals(1)
        self.spin.setSingleStep(0.1)
        self.spin.setSuffix(suffix)
        self.spin.setFixedWidth(88)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)
        self.slider.valueChanged.connect(lambda raw: self.spin.setValue(raw / 10.0))
        self.spin.valueChanged.connect(lambda number: self.slider.setValue(round(number * 10)))
        self.spin.valueChanged.connect(self.valueChanged)
        self.setValue(value)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, value: float) -> None:  # noqa: N802
        self.spin.setValue(value)


class CornerButton(QToolButton):
    def __init__(self, corner: StartCorner, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.corner = corner
        self.label = label
        self.setCheckable(True)
        self.setFixedSize(62, 62)

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        left, top, span = 17.0, 12.0, 28.0
        painter.setPen(QPen(QColor("#a8b6c1"), 1.2))
        for row in range(3):
            for col in range(3):
                painter.drawEllipse(QPointF(left + col * span / 2, top + row * span / 2), 1.7, 1.7)
        row = 0 if self.corner in {StartCorner.TOP_LEFT, StartCorner.TOP_RIGHT} else 2
        col = 0 if self.corner in {StartCorner.TOP_LEFT, StartCorner.BOTTOM_LEFT} else 2
        painter.setBrush(QColor("#1777ad"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(left + col * span / 2, top + row * span / 2), 4.4, 4.4)
        painter.setPen(QColor("#31404b"))
        painter.drawText(self.rect().adjusted(0, 40, 0, 0), Qt.AlignmentFlag.AlignHCenter, self.label)


class ScanPathButton(QToolButton):
    def __init__(self, fast_axis: FastAxis, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.fast_axis = fast_axis
        self.corner = StartCorner.TOP_LEFT
        self.snake = True
        self.setCheckable(True)
        self.setProperty("pathChoice", True)
        self.setMinimumSize(126, 96)

    def configure(self, corner: StartCorner, snake: bool) -> None:
        self.corner = corner
        self.snake = snake
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pattern = ScanPattern(self.corner, self.fast_axis, self.snake)
        points: list[QPointF] = []
        left, top, width, height = 24.0, 12.0, self.width() - 48.0, 54.0
        for slot in range(9):
            row, col = pattern.slot_to_rc(slot, 3, 3)
            points.append(QPointF(left + col * width / 2, top + row * height / 2))
        for index in range(len(points) - 1):
            a, b = points[index], points[index + 1]
            source = pattern.slot_to_rc(index, 3, 3)
            target = pattern.slot_to_rc(index + 1, 3, 3)
            scan_move = (
                abs(source[0] - target[0]) + abs(source[1] - target[1]) == 1
            )
            pen = QPen(QColor("#2581b8" if scan_move else "#9eabb4"), 2.0)
            if not scan_move:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(a, b)
            if scan_move:
                self._draw_arrow(painter, a, b)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#a8b6c1"))
        for point in points:
            painter.drawEllipse(point, 2.2, 2.2)
        painter.setBrush(QColor("#0d6f9f"))
        painter.drawEllipse(points[0], 4.8, 4.8)
        painter.setPen(QColor("#31404b"))
        label = "横向优先" if self.fast_axis is FastAxis.HORIZONTAL else "纵向优先"
        painter.drawText(self.rect().adjusted(0, 72, 0, 0), Qt.AlignmentFlag.AlignHCenter, label)

    @staticmethod
    def _draw_arrow(painter: QPainter, start: QPointF, end: QPointF) -> None:
        dx, dy = end.x() - start.x(), end.y() - start.y()
        length = math.hypot(dx, dy)
        if length < 1:
            return
        ux, uy = dx / length, dy / length
        tip = QPointF((start.x() + end.x()) / 2, (start.y() + end.y()) / 2)
        side = 4.0
        back = QPointF(tip.x() - ux * 6, tip.y() - uy * 6)
        normal = QPointF(-uy * side, ux * side)
        painter.setBrush(QColor("#2581b8"))
        painter.drawPolygon(
            QPolygonF([tip, back + normal, back - normal])
        )


class StitchTab(QWidget):
    projectSaved = Signal(str)

    def __init__(self, thread_pool: QThreadPool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.thread_pool = thread_pool
        self.tile_set: TileSet | None = None
        self.registration_tile_set: TileSet | None = None
        self.source_folder: Path | None = None
        self.refined_positions: dict[tuple[int, int], np.ndarray] | None = None
        self.position_residuals: dict[tuple[int, int], np.ndarray] | None = None
        self._busy = False
        self._active_workers: set[Worker] = set()
        self._loading_project = False
        self._updating_geometry = False
        self._preview_generation = 0
        self._preview_running = False
        self._preview_pending = False
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(100)
        self._preview_timer.timeout.connect(self.refresh_preview)
        self._build()

    def _build(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        controls = QWidget()
        controls.setMinimumWidth(380)
        controls.setMaximumWidth(430)
        left = QVBoxLayout(controls)
        left.setContentsMargins(12, 10, 12, 12)

        arrange = QGroupBox("1. 图像排列")
        arrange_layout = QVBoxLayout(arrange)
        form = QFormLayout()
        folder_row = QWidget(); folder_layout = QHBoxLayout(folder_row); folder_layout.setContentsMargins(0, 0, 0, 0)
        self.folder_edit = QLineEdit(); self.folder_edit.setReadOnly(True)
        choose = QPushButton("选择…"); choose.clicked.connect(self.choose_folder)
        folder_layout.addWidget(self.folder_edit, 1); folder_layout.addWidget(choose)
        form.addRow("图像文件夹", folder_row)
        self.rows_spin = QSpinBox(); self.rows_spin.setRange(1, 10000); self.rows_spin.setValue(5)
        self.cols_spin = QSpinBox(); self.cols_spin.setRange(1, 10000); self.cols_spin.setValue(6)
        grid = QWidget(); grid_layout = QHBoxLayout(grid); grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.addWidget(QLabel("行")); grid_layout.addWidget(self.rows_spin)
        grid_layout.addWidget(QLabel("列")); grid_layout.addWidget(self.cols_spin)
        form.addRow("网格", grid)
        arrange_layout.addLayout(form)

        scan_row = QHBoxLayout(); scan_row.addWidget(QLabel("扫描方式"))
        self.snake_yes = QRadioButton("蛇形"); self.snake_yes.setChecked(True)
        self.snake_no = QRadioButton("逐行/列返回")
        scan_row.addWidget(self.snake_yes); scan_row.addWidget(self.snake_no); scan_row.addStretch(1)
        arrange_layout.addLayout(scan_row)
        arrange_layout.addWidget(QLabel("起始位置"))
        corner_row = QHBoxLayout(); self.corner_group = QButtonGroup(self)
        corner_specs = [
            (StartCorner.TOP_LEFT, "左上"), (StartCorner.TOP_RIGHT, "右上"),
            (StartCorner.BOTTOM_LEFT, "左下"), (StartCorner.BOTTOM_RIGHT, "右下"),
        ]
        for index, (corner, label) in enumerate(corner_specs):
            button = CornerButton(corner, label)
            if index == 0: button.setChecked(True)
            self.corner_group.addButton(button); corner_row.addWidget(button)
        corner_row.addStretch(1); arrange_layout.addLayout(corner_row)
        arrange_layout.addWidget(QLabel("扫描路径（起点为蓝点，虚线为返回移动）"))
        path_row = QHBoxLayout(); self.path_group = QButtonGroup(self)
        for axis in (FastAxis.HORIZONTAL, FastAxis.VERTICAL):
            button = ScanPathButton(axis)
            if axis is FastAxis.HORIZONTAL: button.setChecked(True)
            self.path_group.addButton(button); path_row.addWidget(button)
        path_row.addStretch(1); arrange_layout.addLayout(path_row)
        self.index_button = QPushButton("读取并检查"); self.index_button.clicked.connect(self.reindex)
        arrange_layout.addWidget(self.index_button)
        self.input_summary = QLabel("命名格式：000001.png；缺号保留空位")
        self.input_summary.setProperty("muted", True); self.input_summary.setWordWrap(True)
        arrange_layout.addWidget(self.input_summary)
        left.addWidget(arrange)

        method = QGroupBox("2. 拼接方式")
        method_layout = QVBoxLayout(method)
        mode_row = QHBoxLayout()
        self.auto_radio = QRadioButton("自动拼接"); self.auto_radio.setChecked(True)
        self.manual_radio = QRadioButton("手动拼接")
        mode_row.addWidget(self.auto_radio); mode_row.addWidget(self.manual_radio); mode_row.addStretch(1)
        method_layout.addLayout(mode_row)
        method_layout.addWidget(QLabel("排除坏图编号"))
        self.excluded_edit = QLineEdit(); self.excluded_edit.setPlaceholderText("例如 17, 31-35")
        self.excluded_edit.editingFinished.connect(self.reindex); method_layout.addWidget(self.excluded_edit)
        self.auto_search_group = QGroupBox("自动搜索范围")
        search_form = QFormLayout(self.auto_search_group)
        self.search_overlap_min = QDoubleSpinBox(); self.search_overlap_min.setRange(0, 30); self.search_overlap_min.setValue(0); self.search_overlap_min.setSuffix(" %")
        self.search_overlap_max = QDoubleSpinBox(); self.search_overlap_max.setRange(0, 30); self.search_overlap_max.setValue(30); self.search_overlap_max.setSuffix(" %")
        overlap_range = QWidget(); overlap_layout = QHBoxLayout(overlap_range); overlap_layout.setContentsMargins(0, 0, 0, 0)
        overlap_layout.addWidget(self.search_overlap_min); overlap_layout.addWidget(QLabel("至")); overlap_layout.addWidget(self.search_overlap_max)
        search_form.addRow("重叠", overlap_range)
        self.search_rotation_min = QDoubleSpinBox(); self.search_rotation_min.setRange(-15, 15); self.search_rotation_min.setValue(-8); self.search_rotation_min.setSuffix("°")
        self.search_rotation_max = QDoubleSpinBox(); self.search_rotation_max.setRange(-15, 15); self.search_rotation_max.setValue(8); self.search_rotation_max.setSuffix("°")
        rotation_range = QWidget(); rotation_layout = QHBoxLayout(rotation_range); rotation_layout.setContentsMargins(0, 0, 0, 0)
        rotation_layout.addWidget(self.search_rotation_min); rotation_layout.addWidget(QLabel("至")); rotation_layout.addWidget(self.search_rotation_max)
        search_form.addRow("图块旋转", rotation_range)
        self.sample_pairs = QSpinBox(); self.sample_pairs.setRange(2, 200); self.sample_pairs.setValue(24)
        self.registration_size = QComboBox()
        for size in (512, 768, 1024, 1536, 2048): self.registration_size.addItem(f"{size} px", size)
        self.registration_size.setCurrentIndex(2)
        search_form.addRow("每方向抽样", self.sample_pairs); search_form.addRow("配准最长边", self.registration_size)
        registration_note = QLabel("自动估算时先把每张图的最长边缩小到此像素数；数值越大，细节更多但计算更慢。")
        registration_note.setProperty("muted", True); registration_note.setWordWrap(True)
        search_form.addRow("", registration_note)
        method_layout.addWidget(self.auto_search_group)
        self.auto_button = QPushButton("自动估算"); self.auto_button.setProperty("primary", True); self.auto_button.clicked.connect(self.run_auto)
        method_layout.addWidget(self.auto_button)
        left.addWidget(method)

        params = QGroupBox("3. 拼接参数")
        params_form = QFormLayout(params)
        self.overlap_x = FloatControl(0, 30, 10, "%")
        self.overlap_y = FloatControl(0, 30, 10, "%")
        self.rotation = FloatControl(-15, 15, 0, "°")
        self.overlap_blend = FloatControl(0, 100, 50, "%")
        params_form.addRow("X 重叠", self.overlap_x)
        params_form.addRow("Y 重叠", self.overlap_y)
        params_form.addRow("图块旋转", self.rotation)
        params_form.addRow("重叠取值", self.overlap_blend)
        left.addWidget(params)
        load_scheme = QPushButton("打开已有方案…"); load_scheme.clicked.connect(self.choose_project)
        left.addWidget(load_scheme); left.addStretch(1)

        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(controls); outer.addWidget(scroll)

        center = QWidget(); center_layout = QVBoxLayout(center); center_layout.setContentsMargins(10, 10, 10, 10)
        preview_bar = QHBoxLayout(); preview_bar.addWidget(QLabel("局部预览"))
        self.preview_group = QButtonGroup(self)
        for size in (2, 3, 4):
            button = QToolButton(); button.setText(f"{size} × {size}"); button.setCheckable(True)
            if size == 3: button.setChecked(True)
            self.preview_group.addButton(button, size); preview_bar.addWidget(button)
        self.preview_group.idClicked.connect(lambda _id: self.schedule_preview())
        preview_bar.addStretch(1); preview_bar.addWidget(QLabel("中心编号"))
        self.center_spin = QSpinBox(); self.center_spin.setRange(1, 1); self.center_spin.valueChanged.connect(self.schedule_preview)
        preview_bar.addWidget(self.center_spin); center_layout.addLayout(preview_bar)
        self.preview = ZoomView(); self.preview.set_pil_image(Image.new("RGB", (900, 560), (22, 29, 36)))
        center_layout.addWidget(self.preview, 1)
        self.preview_status = QLabel("选择文件夹后开始"); self.preview_status.setProperty("muted", True)
        center_layout.addWidget(self.preview_status); outer.addWidget(center, 1)

        result = QWidget(); result.setMinimumWidth(220); result.setMaximumWidth(275)
        result_layout = QVBoxLayout(result); result_layout.setContentsMargins(8, 10, 12, 12)
        result_group = QGroupBox("自动结果")
        result_form = QFormLayout(result_group)
        self.result_overlap = QLabel("—"); self.result_rotation = QLabel("—"); self.result_edges = QLabel("—")
        result_form.addRow("X / Y 重叠", self.result_overlap)
        result_form.addRow("图块旋转", self.result_rotation)
        result_form.addRow("低可信接缝", self.result_edges)
        result_layout.addWidget(result_group)
        self.progress = QProgressBar(); result_layout.addWidget(self.progress)
        self.operation_status = QLabel("就绪"); self.operation_status.setProperty("muted", True); self.operation_status.setWordWrap(True)
        result_layout.addWidget(self.operation_status); result_layout.addStretch(1)
        handoff = QLabel("确认局部预览后保存拼接方案；最终大图在“大图合成”模块生成。")
        handoff.setProperty("muted", True); handoff.setWordWrap(True)
        result_layout.addWidget(handoff)
        save_scheme = QPushButton("保存拼接方案…")
        save_scheme.setProperty("primary", True); save_scheme.clicked.connect(self.choose_save_project)
        result_layout.addWidget(save_scheme); outer.addWidget(result)

        self.auto_radio.toggled.connect(self._mode_changed)
        self.snake_yes.toggled.connect(self._scan_choice_changed)
        self.corner_group.buttonClicked.connect(self._scan_choice_changed)
        self.path_group.buttonClicked.connect(self._mapping_changed)
        for control in (self.overlap_x, self.overlap_y, self.rotation, self.overlap_blend): control.valueChanged.connect(self._geometry_changed)
        for widget in (self.rows_spin, self.cols_spin): widget.valueChanged.connect(self._mapping_changed)
        self._update_path_options(); self._mode_changed(True)

    def pattern(self) -> ScanPattern:
        corner_button = self.corner_group.checkedButton()
        path_button = self.path_group.checkedButton()
        return ScanPattern(corner_button.corner, path_button.fast_axis, self.snake_yes.isChecked())

    def geometry(self) -> GridGeometry:
        return GridGeometry(self.overlap_x.value(), self.overlap_y.value(), self.rotation.value())

    def auto_settings(self) -> AutoSearchSettings:
        settings = AutoSearchSettings(
            self.search_overlap_min.value(), self.search_overlap_max.value(),
            self.search_rotation_min.value(), self.search_rotation_max.value(),
            self.sample_pairs.value(), self.registration_size.currentData(),
        )
        settings.validate(); return settings

    def _scan_choice_changed(self, *_args) -> None:
        self._update_path_options(); self._mapping_changed()

    def _update_path_options(self) -> None:
        corner = self.corner_group.checkedButton().corner
        snake = self.snake_yes.isChecked()
        for button in self.path_group.buttons(): button.configure(corner, snake)

    def choose_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        if selected:
            self.source_folder = Path(selected); self.folder_edit.setText(selected); self.reindex()

    def reindex(self) -> None:
        if not self.source_folder: return
        try:
            tile_set = discover_numeric_tiles(self.source_folder, self.rows_spin.value(), self.cols_spin.value(), self.pattern())
            excluded = parse_number_ranges(self.excluded_edit.text())
            if excluded:
                tile_set.records = [tile for tile in tile_set.records if tile.number not in excluded]
                tile_set.warnings.append(f"已排除 {len(excluded)} 个编号")
            self.tile_set = tile_set
            self.registration_tile_set = tile_set
            original_folder = linked_source_folder(self.source_folder)
            if original_folder is not None:
                try:
                    original_set = discover_numeric_tiles(
                        original_folder, self.rows_spin.value(), self.cols_spin.value(), self.pattern()
                    )
                    if excluded:
                        original_set.records = [tile for tile in original_set.records if tile.number not in excluded]
                    if (original_set.tile_width, original_set.tile_height) == (tile_set.tile_width, tile_set.tile_height):
                        self.registration_tile_set = original_set
                        tile_set.warnings.append("自动拼接将参考刀纹去除前的原图")
                    else:
                        tile_set.warnings.append("原图尺寸与处理后图像不一致，自动拼接改用当前图像")
                except Exception:
                    tile_set.warnings.append("未能读取关联原图，自动拼接改用当前图像")
            self.refined_positions = None; self.position_residuals = None
            self.center_spin.setRange(1, tile_set.rows * tile_set.cols)
            self.center_spin.setValue(max(1, (tile_set.rows * tile_set.cols + 1) // 2))
            summary = f"{tile_set.count} 张 · {tile_set.tile_width}×{tile_set.tile_height}"
            if tile_set.warnings: summary += "\n" + "；".join(tile_set.warnings)
            self.input_summary.setText(summary); self.schedule_preview()
        except Exception as exc:
            QMessageBox.warning(self, "无法读取图像", str(exc))

    def schedule_preview(self, *_args) -> None:
        if self.tile_set:
            self._preview_generation += 1
            self._preview_pending = True
            self.preview_status.setText("预览待更新…")
            self._preview_timer.start()

    def refresh_preview(self) -> None:
        if not self.tile_set:
            return
        if self._busy or self._preview_running:
            self._preview_pending = True
            return
        generation = self._preview_generation
        self._preview_pending = False
        self._preview_running = True
        size = self.preview_group.checkedId() if self.preview_group.checkedId() > 0 else 3
        worker = Worker(
            compose_local_preview, self.tile_set, self.geometry(), self.center_spin.value() - 1,
            size, 420, self.refined_positions, True, None, self.overlap_blend.value(),
        )
        worker.signals.result.connect(
            lambda image, token=generation: self._preview_ready(image) if token == self._preview_generation else None
        )
        worker.signals.error.connect(self._show_traceback)
        worker.signals.finished.connect(self._preview_finished)
        self.preview_status.setText("正在更新局部预览…")
        _start_worker(self, worker)

    def _preview_finished(self) -> None:
        self._preview_running = False
        if self._preview_pending and not self._busy:
            self._preview_timer.start(0)

    def _preview_ready(self, image: Image.Image) -> None:
        self.preview.set_pil_image(image)
        if self.tile_set:
            size = self.preview_group.checkedId()
            memory = estimated_preview_memory_mb(size, self.tile_set.tile_width or 1, self.tile_set.tile_height or 1)
            self.preview_status.setText(f"{size}×{size} · 原图完全解码约 {memory:.0f} MB · 当前使用缩略图")

    def run_auto(self) -> None:
        if not self.tile_set:
            QMessageBox.information(self, "尚未导入", "请先选择并检查图像文件夹。"); return
        try: settings = self.auto_settings()
        except Exception as exc:
            QMessageBox.warning(self, "搜索范围无效", str(exc)); return
        self._set_busy(True, "自动估算重叠与图块旋转…")
        stitcher = AutoStitcher(
            self.registration_tile_set or self.tile_set,
            overlap_range=(settings.overlap_min_percent, settings.overlap_max_percent),
            rotation_range=(settings.rotation_min_deg, settings.rotation_max_deg),
            max_registration_size=settings.registration_size,
            sample_pairs_per_direction=settings.sample_pairs_per_direction,
        )
        worker = Worker(stitcher.run, with_progress=True)
        worker.signals.progress.connect(self._progress); worker.signals.result.connect(self._auto_ready)
        worker.signals.error.connect(self._show_traceback); worker.signals.finished.connect(lambda: self._set_busy(False))
        _start_worker(self, worker)

    def _auto_ready(self, result: AutoRegistrationResult) -> None:
        self._updating_geometry = True
        self.overlap_x.setValue(result.geometry.overlap_x_percent)
        self.overlap_y.setValue(result.geometry.overlap_y_percent)
        self.rotation.setValue(result.geometry.tile_rotation_deg)
        self._updating_geometry = False
        base = regular_positions(self.tile_set, result.geometry) if self.tile_set else {}
        self.refined_positions = result.positions
        self.position_residuals = {
            key: value - base.get(key, value) for key, value in result.positions.items()
        }
        self.result_overlap.setText(f"{result.geometry.overlap_x_percent:.1f}% / {result.geometry.overlap_y_percent:.1f}%")
        self.result_rotation.setText(f"{result.geometry.tile_rotation_deg:+.2f}°")
        self.result_edges.setText(f"{len(result.low_confidence_edges)} / {len(result.edges)}")
        self.operation_status.setText("；".join(result.messages) or "自动估算完成，可继续微调参数")
        self.schedule_preview()

    def _geometry_changed(self, *_args) -> None:
        if self._updating_geometry or self._loading_project: return
        if self.tile_set and self.position_residuals is not None:
            base = regular_positions(self.tile_set, self.geometry())
            self.refined_positions = {
                key: value + self.position_residuals.get(key, np.zeros(2)) for key, value in base.items()
            }
        else:
            self.refined_positions = None
        self.schedule_preview()

    def choose_project(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "打开拼接方案", "", "拼接方案 (*.stitch.json *.mosaic.json *.json)")
        if selected: self.load_project(Path(selected))

    def choose_save_project(self) -> None:
        default = f"{self.source_folder.name if self.source_folder else 'project'}.stitch.json"
        selected, _ = QFileDialog.getSaveFileName(self, "保存拼接方案", default, "拼接方案 (*.stitch.json)")
        if selected:
            if not selected.lower().endswith(".json"): selected += ".stitch.json"
            path = Path(selected)
            self.save_project(path)
            self.operation_status.setText(f"方案已保存：{path.name}")

    def save_project(self, path: Path) -> None:
        source_relative = ""
        if self.source_folder:
            try:
                source_relative = os.path.relpath(self.source_folder, path.expanduser().resolve().parent)
            except ValueError:
                pass
        project = StitchProject(
            source_folder=str(self.source_folder or ""), source_folder_relative=source_relative,
            rows=self.rows_spin.value(), cols=self.cols_spin.value(),
            start_corner=self.pattern().start_corner.value, fast_axis=self.pattern().fast_axis.value,
            snake=self.pattern().snake, stitch_mode="automatic" if self.auto_radio.isChecked() else "manual",
            tile_width=self.tile_set.tile_width if self.tile_set and self.tile_set.tile_width else 0,
            tile_height=self.tile_set.tile_height if self.tile_set and self.tile_set.tile_height else 0,
            geometry=self.geometry(), overlap_blend_percent=self.overlap_blend.value(), auto_search=AutoSearchSettings(
                self.search_overlap_min.value(), self.search_overlap_max.value(),
                self.search_rotation_min.value(), self.search_rotation_max.value(),
                self.sample_pairs.value(), self.registration_size.currentData(),
            ),
            excluded_numbers=sorted(parse_number_ranges(self.excluded_edit.text())),
            refined_positions={f"{r},{c}": value.tolist() for (r, c), value in (self.refined_positions or {}).items()},
        )
        path.write_text(json.dumps(project.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        self.projectSaved.emit(str(path.expanduser().resolve()))

    def load_project(self, path: Path) -> None:
        project = StitchProject.from_dict(json.loads(path.read_text(encoding="utf-8")))
        self._loading_project = True
        try:
            source_folder = Path(project.source_folder).expanduser() if project.source_folder else None
            if (source_folder is None or not source_folder.is_dir()) and project.source_folder_relative:
                relative = (path.expanduser().resolve().parent / project.source_folder_relative).resolve()
                if relative.is_dir(): source_folder = relative
            self.source_folder = source_folder
            self.folder_edit.setText(str(source_folder or "")); self.rows_spin.setValue(project.rows); self.cols_spin.setValue(project.cols)
            for button in self.corner_group.buttons(): button.setChecked(button.corner is StartCorner(project.start_corner))
            self.snake_yes.setChecked(project.snake); self.snake_no.setChecked(not project.snake)
            for button in self.path_group.buttons(): button.setChecked(button.fast_axis is FastAxis(project.fast_axis))
            self._update_path_options()
            self.overlap_x.setValue(project.geometry.overlap_x_percent); self.overlap_y.setValue(project.geometry.overlap_y_percent)
            self.rotation.setValue(project.geometry.tile_rotation_deg)
            self.overlap_blend.setValue(project.overlap_blend_percent)
            auto = project.auto_search
            self.search_overlap_min.setValue(auto.overlap_min_percent); self.search_overlap_max.setValue(auto.overlap_max_percent)
            self.search_rotation_min.setValue(auto.rotation_min_deg); self.search_rotation_max.setValue(auto.rotation_max_deg)
            self.sample_pairs.setValue(auto.sample_pairs_per_direction)
            index = self.registration_size.findData(auto.registration_size)
            if index >= 0: self.registration_size.setCurrentIndex(index)
            self.excluded_edit.setText(", ".join(str(number) for number in project.excluded_numbers))
            self.auto_radio.setChecked(project.stitch_mode == "automatic")
            self.manual_radio.setChecked(project.stitch_mode == "manual")
        finally:
            self._loading_project = False
        self.reindex()
        saved = {
            tuple(map(int, key.split(","))): np.asarray(value, dtype=np.float64)
            for key, value in project.refined_positions.items()
        }
        self.refined_positions = saved or None
        if saved and self.tile_set:
            base = regular_positions(self.tile_set, self.geometry())
            self.position_residuals = {key: value - base.get(key, value) for key, value in saved.items()}
        self.schedule_preview()

    def _mode_changed(self, automatic: bool) -> None:
        self.auto_search_group.setVisible(automatic)
        self.auto_button.setVisible(automatic); self.auto_button.setEnabled(automatic and not self._busy)

    def _mapping_changed(self, *_args) -> None:
        if self.source_folder and not self._loading_project:
            self._preview_timer.stop(); QTimer.singleShot(120, self.reindex)

    def _progress(self, current: int, total: int, message: str) -> None:
        self.progress.setValue(round(100 * current / max(1, total)))
        suffix = f"{current}%" if total == 100 else f"{current}/{total}"
        self.operation_status.setText(f"{message} · {suffix}")

    def _set_busy(self, busy: bool, message: str | None = None) -> None:
        self._busy = busy; self.auto_button.setEnabled(not busy and self.auto_radio.isChecked())
        self.index_button.setEnabled(not busy)
        if busy:
            self.progress.setValue(0)
        if message: self.operation_status.setText(message)
        if not busy:
            self.progress.setValue(0)
            if self._preview_pending:
                self._preview_timer.start(0)

    def _show_traceback(self, details: str) -> None:
        self.operation_status.setText("操作失败")
        message = details.strip().splitlines()[-1] if details.strip() else "未知错误"
        box = QMessageBox(self); box.setWindowTitle("操作失败"); box.setText(message); box.setDetailedText(details); box.exec()


class ToolMarkTab(QWidget):
    marksRemoved = Signal(str, str)

    def __init__(self, thread_pool: QThreadPool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.thread_pool = thread_pool
        self.source_path: Path | None = None
        self.source_preview_path: Path | None = None
        self.output_path: Path | None = None
        self.source_files: list[Path] = []
        self._preview_source: np.ndarray | None = None
        self._preview_scale = (1.0, 1.0)
        self._preview_processed: np.ndarray | None = None
        self._busy = False
        self._active_workers: set[Worker] = set()
        self._loading_settings = False
        self._source_generation = 0
        self._preview_generation = 0
        self._preview_running = False
        self._preview_pending = False
        self._changing_input_mode = False
        self._preview_timer = QTimer(self); self._preview_timer.setSingleShot(True); self._preview_timer.setInterval(180)
        self._preview_timer.timeout.connect(self._run_filter_preview)
        self._build()

    def _build(self) -> None:
        outer = QHBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        controls = QWidget(); controls.setMinimumWidth(350); controls.setMaximumWidth(405)
        left = QVBoxLayout(controls); left.setContentsMargins(12, 10, 12, 12)

        source_group = QGroupBox("1. 处理对象")
        source_layout = QVBoxLayout(source_group)
        mode_row = QHBoxLayout()
        self.input_mode_group = QButtonGroup(self)
        self.single_radio = QRadioButton("单张 / 大图"); self.single_radio.setChecked(True)
        self.batch_radio = QRadioButton("批量文件夹")
        self.input_mode_group.addButton(self.single_radio)
        self.input_mode_group.addButton(self.batch_radio)
        mode_row.addWidget(self.single_radio); mode_row.addWidget(self.batch_radio); mode_row.addStretch(1)
        source_layout.addLayout(mode_row)
        self.choose_input_button = QPushButton("选择图像…"); self.choose_input_button.clicked.connect(self.choose_input)
        source_layout.addWidget(self.choose_input_button)
        self.source_edit = QLineEdit(); self.source_edit.setReadOnly(True); source_layout.addWidget(self.source_edit)
        self.source_summary = QLabel("尚未选择图像")
        self.source_summary.setProperty("muted", True); self.source_summary.setWordWrap(True)
        source_layout.addWidget(self.source_summary)
        self.preview_picker = QWidget()
        item_row = QHBoxLayout(self.preview_picker); item_row.setContentsMargins(0, 0, 0, 0); item_row.addWidget(QLabel("预览样图"))
        self.item_combo = QComboBox(); self.item_combo.currentIndexChanged.connect(self._load_selected_preview)
        item_row.addWidget(self.item_combo, 1); source_layout.addWidget(self.preview_picker)
        left.addWidget(source_group)

        algorithm = QGroupBox("2. 去除算法")
        algorithm_layout = QVBoxLayout(algorithm)
        method_form = QFormLayout()
        self.method_combo = QComboBox()
        self.method_combo.addItem("FFT + Gaussian Notch", ToolMarkMethod.FFT_GAUSSIAN_NOTCH.value)
        self.method_combo.addItem("方向校正 + 局部条纹模板相减", ToolMarkMethod.PERIODIC_TEMPLATE.value)
        self.method_combo.addItem("方向性 Median 残差滤除", ToolMarkMethod.DIRECTIONAL_1D.value)
        method_form.addRow("算法", self.method_combo)
        self.strength = FloatControl(0, 1, 0.7, "")
        method_form.addRow("去除强度", self.strength)
        algorithm_layout.addLayout(method_form)

        orientation_row = QHBoxLayout()
        self.auto_orientation = QCheckBox("自动识别刀纹方向"); self.auto_orientation.setChecked(True)
        orientation_row.addWidget(self.auto_orientation); orientation_row.addStretch(1)
        algorithm_layout.addLayout(orientation_row)
        self.orientation = FloatControl(-90, 90, 0, "°"); self.orientation.setEnabled(False)
        self.orientation.setToolTip("法向垂直于刀纹：0° 对应竖直刀纹，±90° 对应水平刀纹；图像坐标中向右下为正角。")
        orientation_form = QFormLayout(); orientation_form.addRow("法向角", self.orientation); algorithm_layout.addLayout(orientation_form)
        period_row = QHBoxLayout()
        self.auto_period = QCheckBox("自动识别周期"); self.auto_period.setChecked(True)
        period_row.addWidget(self.auto_period); period_row.addStretch(1); algorithm_layout.addLayout(period_row)
        self.period = FloatControl(3, 512, 18, " px"); self.period.setEnabled(False)
        self.period.setToolTip("以原图像素计。间距不规则时仅代表主尺度；FFT 自动模式会检测多个频率，无需强行匹配单一周期。")
        period_form = QFormLayout(); period_form.addRow("周期 / 主尺度", self.period); algorithm_layout.addLayout(period_form)

        self.method_stack = QStackedWidget()
        fft_page = QWidget(); fft_form = QFormLayout(fft_page); fft_form.setContentsMargins(0, 0, 0, 0)
        self.angular_width = FloatControl(1, 20, 5, "°")
        self.frequency_width = QDoubleSpinBox(); self.frequency_width.setRange(0.0005, 0.03); self.frequency_width.setDecimals(4); self.frequency_width.setSingleStep(0.0005); self.frequency_width.setValue(0.0035)
        self.max_peaks = QSpinBox(); self.max_peaks.setRange(1, 64); self.max_peaks.setValue(14)
        self.max_peaks.setToolTip("限制最多处理多少个 FFT 频率峰或谐波，并不是画面中的刀纹条数。")
        self.max_peaks_label = QLabel("频率峰/谐波上限")
        self.max_peaks_label.setToolTip(self.max_peaks.toolTip())
        fft_form.addRow("方向宽度", self.angular_width); fft_form.addRow("频率宽度", self.frequency_width); fft_form.addRow(self.max_peaks_label, self.max_peaks)
        template_page = QWidget(); template_form = QFormLayout(template_page); template_form.setContentsMargins(0, 0, 0, 0)
        self.template_smoothing = FloatControl(0, 5, 1.2, " px")
        template_form.addRow("模板平滑", self.template_smoothing)
        template_note = QLabel("沿刀纹分段估计稳健模板，仅扣除条纹分量。支持不等间距刀纹；周期控制背景分离尺度。")
        template_note.setProperty("muted", True); template_note.setWordWrap(True); template_form.addRow("", template_note)
        directional_page = QWidget(); directional_form = QFormLayout(directional_page); directional_form.setContentsMargins(0, 0, 0, 0)
        self.directional_kernel = QSpinBox(); self.directional_kernel.setRange(3, 101); self.directional_kernel.setSingleStep(2); self.directional_kernel.setValue(51)
        directional_form.addRow("一维核长度", self.directional_kernel)
        directional_note = QLabel("从法向 Median 残差中提取沿刀纹连续的分量。核宽以原图像素计，应大于刀纹宽度的两倍；过大会削弱同方向细节。")
        directional_note.setProperty("muted", True); directional_note.setWordWrap(True); directional_form.addRow("", directional_note)
        self.method_stack.addWidget(fft_page); self.method_stack.addWidget(template_page); self.method_stack.addWidget(directional_page)
        algorithm_layout.addWidget(self.method_stack)
        block_form = QFormLayout()
        self.block_size = QComboBox()
        for size in (512, 1024, 1536, 2048): self.block_size.addItem(f"{size} px", size)
        self.block_size.setCurrentIndex(1)
        self.block_size_label = QLabel("大图分块边长")
        block_form.addRow(self.block_size_label, self.block_size)
        self.block_note = QLabel("仅处理单张超大图时使用：每次处理一个方形区域以限制内存；数值不会改变输出分辨率。")
        self.block_note.setProperty("muted", True); self.block_note.setWordWrap(True)
        block_form.addRow("", self.block_note)
        algorithm_layout.addLayout(block_form)
        left.addWidget(algorithm)

        settings_row = QHBoxLayout()
        self.load_settings_button = QPushButton("打开参数…"); self.load_settings_button.clicked.connect(self.choose_settings)
        self.save_settings_button = QPushButton("保存参数…"); self.save_settings_button.clicked.connect(self.choose_save_settings)
        settings_row.addWidget(self.load_settings_button); settings_row.addWidget(self.save_settings_button); left.addLayout(settings_row)

        output_group = QGroupBox("3. 输出")
        output_layout = QVBoxLayout(output_group)
        output_row = QHBoxLayout()
        self.output_edit = QLineEdit(); self.output_edit.setReadOnly(True)
        self.choose_output_button = QPushButton("更改…"); self.choose_output_button.clicked.connect(self.choose_output)
        output_row.addWidget(self.output_edit, 1); output_row.addWidget(self.choose_output_button)
        output_layout.addLayout(output_row)
        self.batch_note = QLabel("整批沿用上方算法和参数，并保留原文件名与格式。自动识别开启时，每张图分别估算方向和周期。")
        self.batch_note.setProperty("muted", True); self.batch_note.setWordWrap(True)
        output_layout.addWidget(self.batch_note)
        left.addWidget(output_group)

        self.progress = QProgressBar(); left.addWidget(self.progress)
        self.status = QLabel(""); self.status.setProperty("muted", True); self.status.setWordWrap(True); left.addWidget(self.status)
        left.addStretch(1)
        self.apply_button = QPushButton("处理并保存"); self.apply_button.setProperty("primary", True); self.apply_button.setEnabled(False); self.apply_button.clicked.connect(self.process_and_save)
        left.addWidget(self.apply_button)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(controls)
        scroll.setMinimumWidth(410); scroll.setMaximumWidth(470); outer.addWidget(scroll)

        preview = QWidget(); preview_layout = QVBoxLayout(preview); preview_layout.setContentsMargins(10, 10, 12, 12)
        preview_bar = QHBoxLayout(); preview_bar.addWidget(QLabel("实时对比")); preview_bar.addStretch(1)
        self.difference_check = QCheckBox("右侧显示差异增强"); self.difference_check.toggled.connect(self._update_processed_view)
        preview_bar.addWidget(self.difference_check); preview_layout.addLayout(preview_bar)
        views = QHBoxLayout()
        original_panel = QVBoxLayout(); original_panel.addWidget(QLabel("原图"))
        self.original_view = ZoomView(); original_panel.addWidget(self.original_view, 1)
        result_panel = QVBoxLayout(); self.processed_title = QLabel("处理后"); result_panel.addWidget(self.processed_title)
        self.processed_view = ZoomView(); result_panel.addWidget(self.processed_view, 1)
        views.addLayout(original_panel, 1); views.addLayout(result_panel, 1); preview_layout.addLayout(views, 1)
        self.analysis_label = QLabel("方向、周期和置信度将在这里显示")
        self.analysis_label.setProperty("muted", True); preview_layout.addWidget(self.analysis_label)
        outer.addWidget(preview, 1)

        self.method_combo.currentIndexChanged.connect(self._method_changed)
        self.auto_orientation.toggled.connect(lambda checked: (self.orientation.setEnabled(not checked), self.schedule_preview()))
        self.auto_period.toggled.connect(lambda checked: (self.period.setEnabled(not checked), self.schedule_preview()))
        for control in (self.strength, self.orientation, self.period, self.angular_width, self.template_smoothing):
            control.valueChanged.connect(self.schedule_preview)
        for control in (self.frequency_width, self.max_peaks, self.directional_kernel):
            control.valueChanged.connect(self.schedule_preview)
        self.batch_radio.toggled.connect(self._input_mode_changed)
        self._sync_input_mode(False)
        self._method_changed(0)

    def settings(self) -> MarkFilterSettings:
        return MarkFilterSettings(
            enabled=True, method=self.method_combo.currentData(), strength=self.strength.value(),
            auto_orientation=self.auto_orientation.isChecked(), orientation_deg=self.orientation.value(),
            auto_period=self.auto_period.isChecked(), period_px=self.period.value(),
            angular_width_deg=self.angular_width.value(), frequency_width=self.frequency_width.value(),
            max_peaks=self.max_peaks.value(), template_smoothing=self.template_smoothing.value(),
            directional_kernel=self.directional_kernel.value(), block_size=self.block_size.currentData(),
            block_halo=max(64, self.block_size.currentData() // 8),
        )

    def _input_mode_changed(self, batch_mode: bool) -> None:
        if self._changing_input_mode:
            return
        if self.source_path is not None and self.source_path.is_dir() != batch_mode:
            self._clear_input()
        self._sync_input_mode(batch_mode)

    def _sync_input_mode(self, batch_mode: bool) -> None:
        self.choose_input_button.setText("选择图像文件夹…" if batch_mode else "选择图像…")
        self.preview_picker.setVisible(batch_mode)
        self.batch_note.setVisible(batch_mode)
        self.block_size_label.setVisible(not batch_mode)
        self.block_size.setVisible(not batch_mode)
        self.block_note.setVisible(not batch_mode)
        self.apply_button.setText("开始批量处理" if batch_mode else "处理并保存")
        self.choose_output_button.setText("选择输出文件夹…" if batch_mode else "选择保存位置…")
        self.choose_output_button.setEnabled(self.source_path is not None and not self._busy)

    def _clear_input(self) -> None:
        self._source_generation += 1
        self._preview_generation += 1
        self._preview_timer.stop()
        self.source_path = None
        self.source_preview_path = None
        self.output_path = None
        self.source_files = []
        self._preview_source = None
        self._preview_scale = (1.0, 1.0)
        self._preview_processed = None
        self.source_edit.clear(); self.output_edit.clear(); self.item_combo.clear()
        self.source_summary.setText("尚未选择图像文件夹" if self.batch_radio.isChecked() else "尚未选择图像")
        self.status.clear(); self.analysis_label.setText("方向、周期和置信度将在这里显示")
        self.apply_button.setEnabled(False)

    def _set_default_output(self) -> None:
        self.output_path = suggested_toolmark_output(self.source_path) if self.source_path else None
        self.output_edit.setText(str(self.output_path or ""))
        self.choose_output_button.setEnabled(self.source_path is not None and not self._busy)

    def _set_idle_status(self) -> None:
        if self.source_path is None:
            self.status.clear()
        elif self.source_path.is_dir():
            self.status.setText(f"批量任务：{len(self.source_files):,} 张图像")
        else:
            self.status.setText(f"当前图像：{self.source_path.name}")

    def choose_input(self) -> None:
        if self.batch_radio.isChecked():
            self.choose_folder()
        else:
            self.choose_file()

    def choose_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)")
        if selected: self.open_source(Path(selected))

    def choose_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        if selected: self.open_source(Path(selected))

    def choose_output(self) -> None:
        if self.source_path is None:
            QMessageBox.information(self, "尚未选择", "请先选择处理对象。")
            return
        suggested = self.output_path or suggested_toolmark_output(self.source_path)
        if self.source_path.is_dir():
            initial = suggested if suggested.is_dir() else suggested.parent
            selected = QFileDialog.getExistingDirectory(self, "选择批量输出文件夹", str(initial))
            if selected:
                self.output_path = Path(selected).resolve()
        else:
            selected, _ = QFileDialog.getSaveFileName(
                self, "保存去除刀纹后的大图", str(suggested), "OME-TIFF (*.ome.tif *.ome.tiff)"
            )
            if selected:
                if not selected.lower().endswith((".ome.tif", ".ome.tiff")):
                    selected += ".ome.tif"
                self.output_path = Path(selected).resolve()
        self.output_edit.setText(str(self.output_path or ""))

    def open_source(self, path: Path, preview_path: Path | None = None) -> None:
        try:
            source_path = path.resolve()
            if source_path.is_dir():
                source_files = [
                    candidate for candidate in source_path.iterdir()
                    if candidate.is_file() and candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
                ]
                source_files.sort(key=lambda candidate: (0, int(candidate.stem)) if candidate.stem.isdigit() else (1, candidate.name.lower()))
            elif source_path.is_file():
                source_files = [source_path]
            else:
                raise FileNotFoundError(source_path)
            if not source_files:
                raise ValueError("文件夹中没有支持的图像")
        except Exception as exc:
            self._clear_input()
            self.status.setText(f"无法读取输入：{exc}")
            return

        self.source_path = source_path
        self.source_files = source_files
        self.source_preview_path = None
        if source_path.is_file():
            candidates = [preview_path, source_path.with_suffix(".preview.png")]
            self.source_preview_path = next(
                (candidate.resolve() for candidate in candidates if candidate and candidate.is_file()),
                None,
            )
        batch_mode = source_path.is_dir()
        self._changing_input_mode = True
        try:
            self.batch_radio.setChecked(batch_mode)
            self.single_radio.setChecked(not batch_mode)
        finally:
            self._changing_input_mode = False
        self._sync_input_mode(batch_mode)
        self.source_edit.setText(str(self.source_path))
        self.source_summary.setText(
            f"已载入 {len(self.source_files):,} 张图像；选择任意样图检查参数"
            if batch_mode else f"已载入：{self.source_path.name}"
        )
        self._set_default_output()
        self.apply_button.setEnabled(True)

        self._source_generation += 1
        self._preview_generation += 1
        self._preview_source = None
        self._preview_processed = None
        self.item_combo.setUpdatesEnabled(False)
        self.item_combo.blockSignals(True)
        self.item_combo.clear()
        for file in self.source_files:
            self.item_combo.addItem(file.name, file)
        if self.source_files:
            self.item_combo.setCurrentIndex(len(self.source_files) // 2 if self.source_path.is_dir() else 0)
        self.item_combo.blockSignals(False)
        self.item_combo.setUpdatesEnabled(True)
        self.status.setText("正在载入预览…")
        self._load_selected_preview()

    def _load_selected_preview(self, *_args) -> None:
        path = self.item_combo.currentData()
        if not path: return
        preview_source = self.source_preview_path if len(self.source_files) == 1 and self.source_preview_path else Path(path)
        self._source_generation += 1
        generation = self._source_generation
        self._preview_generation += 1
        self.status.setText("正在读取预览区域…")
        worker = Worker(_read_toolmark_preview_with_scale, Path(path), preview_source)
        worker.signals.result.connect(
            lambda image, token=generation: self._preview_loaded(image) if token == self._source_generation else None
        )
        worker.signals.error.connect(self._show_error)
        _start_worker(self, worker)

    def _preview_loaded(self, frame: tuple[np.ndarray, tuple[float, float]]) -> None:
        image, self._preview_scale = frame
        self._preview_source = image; self.original_view.set_pil_image(Image.fromarray(image, mode="RGB")); self.schedule_preview()

    def schedule_preview(self, *_args) -> None:
        if self._preview_source is not None and not self._loading_settings:
            self._preview_generation += 1
            self._preview_pending = True
            self.status.setText("预览待更新…")
            self._preview_timer.start()

    def _run_filter_preview(self) -> None:
        if self._preview_source is None:
            return
        if self._busy or self._preview_running:
            self._preview_pending = True
            return
        generation = self._preview_generation
        self._preview_pending = False
        self._preview_running = True
        self.status.setText("正在更新预览…")
        worker = Worker(preview_tool_marks, self._preview_source.copy(), self.settings(), self._preview_scale)
        worker.signals.result.connect(
            lambda result, token=generation: self._filter_preview_ready(result) if token == self._preview_generation else None
        )
        worker.signals.error.connect(self._show_error)
        worker.signals.finished.connect(self._filter_preview_finished)
        _start_worker(self, worker)

    def _filter_preview_finished(self) -> None:
        self._preview_running = False
        if self._preview_pending and not self._busy:
            self._preview_timer.start(0)

    def _filter_preview_ready(self, result: tuple[np.ndarray, object]) -> None:
        self._preview_processed, report = result
        # Expose the per-image estimates in the controls.  The controls remain
        # disabled while automatic detection is on; turning it off starts the
        # user's fine-tuning from exactly the values shown for this image.
        self._loading_settings = True
        try:
            if self.auto_orientation.isChecked():
                self.orientation.setValue(report.orientation_deg)
            if self.auto_period.isChecked():
                self.period.setValue(report.period_px)
        finally:
            self._loading_settings = False
        self._update_processed_view()
        self.analysis_label.setText(
            f"法向角 {report.orientation_deg:+.1f}°（置信度 {report.orientation_confidence:.2f}） · "
            f"主周期 {report.period_px:.1f} 原图 px（置信度 {report.period_confidence:.2f}）"
        )
        scene = self.original_view.scene()
        for item in scene.items():
            if item.data(0) == "toolmark_detection":
                scene.removeItem(item)
        if report.processing_region is not None:
            x0, y0, x1, y1 = report.processing_region
            sy, sx = self._preview_scale
            pen = QPen(QColor("#e59422"), 2)
            pen.setCosmetic(True)
            outline = scene.addRect(x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy, pen)
            outline.setData(0, "toolmark_detection")
            self.analysis_label.setText(self.analysis_label.text() + " · 局部检出：仅处理橙框区域")
        if not report.applied:
            self.analysis_label.setText(self.analysis_label.text() + " · 未检出可靠刀纹或未启用滤除")
        self._set_idle_status()

    def _update_processed_view(self, *_args) -> None:
        if self._preview_processed is None or self._preview_source is None: return
        if self.difference_check.isChecked():
            display = np.clip(np.abs(self._preview_processed.astype(np.int16) - self._preview_source.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
            self.processed_title.setText("差异增强 ×4")
        else:
            display = self._preview_processed; self.processed_title.setText("处理后")
        self.processed_view.set_pil_image(Image.fromarray(display, mode="RGB"))

    def _method_changed(self, index: int) -> None:
        self.method_stack.setCurrentIndex(index); self.schedule_preview()

    def choose_settings(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "打开刀纹去除参数", "", "参数 (*.toolmarks.json *.json)")
        if selected: self.apply_settings(load_toolmark_settings(selected))

    def choose_save_settings(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(self, "保存刀纹去除参数", "toolmarks.toolmarks.json", "参数 (*.toolmarks.json)")
        if selected:
            if not selected.lower().endswith(".json"): selected += ".toolmarks.json"
            save_toolmark_settings(selected, self.settings())

    def apply_settings(self, settings: MarkFilterSettings) -> None:
        self._loading_settings = True
        try:
            index = self.method_combo.findData(settings.method)
            if index >= 0: self.method_combo.setCurrentIndex(index)
            self.strength.setValue(settings.strength); self.auto_orientation.setChecked(settings.auto_orientation)
            self.orientation.setValue(settings.orientation_deg); self.auto_period.setChecked(settings.auto_period)
            self.period.setValue(settings.period_px); self.angular_width.setValue(settings.angular_width_deg)
            self.frequency_width.setValue(settings.frequency_width); self.max_peaks.setValue(settings.max_peaks)
            self.template_smoothing.setValue(settings.template_smoothing); self.directional_kernel.setValue(settings.directional_kernel)
            block_index = self.block_size.findData(settings.block_size)
            if block_index >= 0: self.block_size.setCurrentIndex(block_index)
        finally:
            self._loading_settings = False
        self.orientation.setEnabled(not settings.auto_orientation); self.period.setEnabled(not settings.auto_period); self.schedule_preview()

    def process_and_save(self) -> None:
        if not self.source_path:
            QMessageBox.information(self, "尚未选择", "请先选择处理对象。"); return
        output = self.output_path or suggested_toolmark_output(self.source_path)
        if self.source_path.is_dir():
            if output.resolve() == self.source_path.resolve():
                QMessageBox.warning(self, "输出位置无效", "批量输出文件夹不能与输入文件夹相同。")
                return
            if output.is_dir() and any(output.iterdir()):
                answer = QMessageBox.question(
                    self,
                    "输出文件夹已有内容",
                    f"{output.name} 已存在。继续会覆盖同名处理结果，但保留其他文件。是否继续？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel,
                )
                if answer != QMessageBox.StandardButton.Yes: return
        elif output.exists():
            answer = QMessageBox.question(
                self, "输出文件已存在", f"{output.name} 已存在，是否覆盖？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes: return
        self._busy = True
        for widget in (
            self.apply_button, self.choose_input_button, self.choose_output_button,
            self.single_radio, self.batch_radio, self.load_settings_button,
            self.save_settings_button,
        ):
            widget.setEnabled(False)
        self.status.setText("正在批量去除刀纹…" if self.source_path.is_dir() else "正在处理并保存…")
        worker = Worker(process_toolmark_input, self.source_path, output, self.settings(), with_progress=True)
        worker.signals.progress.connect(lambda current, total, message: (self.progress.setValue(round(100 * current / max(1, total))), self.status.setText(f"{message} · {current}/{total}")))
        worker.signals.result.connect(self._output_ready); worker.signals.error.connect(self._show_error)
        worker.signals.finished.connect(self._processing_finished); _start_worker(self, worker)

    def _output_ready(self, result: ToolMarkOutput) -> None:
        if result.output_path.is_dir():
            self.status.setText(f"批量完成：{result.processed_count:,} 张")
            message = f"已处理：{result.processed_count:,} 张\n输出文件夹：{result.output_path}\n参数与分析：{result.settings_path}"
        else:
            self.status.setText(f"已保存：{result.output_path.name}；参数：{result.settings_path.name}")
            message = f"输出：{result.output_path}\n参数与分析：{result.settings_path}"
        self.marksRemoved.emit(str(result.output_path), str(result.preview_path or ""))
        QMessageBox.information(self, "完成", message)

    def _processing_finished(self) -> None:
        self._busy = False; self.apply_button.setEnabled(self.source_path is not None); self.progress.setValue(0)
        self.choose_input_button.setEnabled(True); self.single_radio.setEnabled(True); self.batch_radio.setEnabled(True)
        self.choose_output_button.setEnabled(self.source_path is not None)
        self.load_settings_button.setEnabled(True); self.save_settings_button.setEnabled(True)
        if self._preview_pending:
            self._preview_timer.start(0)

    def _show_error(self, details: str) -> None:
        self.status.setText("操作失败")
        box = QMessageBox(self); box.setWindowTitle("操作失败"); box.setText(details.strip().splitlines()[-1]); box.setDetailedText(details); box.exec()


class MosaicTab(QWidget):
    mosaicCreated = Signal(str, str)

    def __init__(self, thread_pool: QThreadPool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.thread_pool = thread_pool
        self.image_folder: Path | None = None
        self.project_path: Path | None = None
        self.project: StitchProject | None = None
        self.tile_set: TileSet | None = None
        self.output_path: Path | None = None
        self.path: Path | None = None
        self._busy = False
        self._active_workers: set[Worker] = set()
        self._build()

    def _build(self) -> None:
        outer = QHBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        controls = QWidget(); controls.setMinimumWidth(390); controls.setMaximumWidth(455)
        left = QVBoxLayout(controls); left.setContentsMargins(12, 10, 12, 12)

        inputs = QGroupBox("1. 合成输入")
        form = QFormLayout(inputs)
        folder_row = QWidget(); folder_layout = QHBoxLayout(folder_row); folder_layout.setContentsMargins(0, 0, 0, 0)
        self.folder_edit = QLineEdit(); self.folder_edit.setReadOnly(True)
        self.folder_button = QPushButton("选择…"); self.folder_button.clicked.connect(self.choose_folder)
        folder_layout.addWidget(self.folder_edit, 1); folder_layout.addWidget(self.folder_button)
        form.addRow("图像文件夹", folder_row)
        project_row = QWidget(); project_layout = QHBoxLayout(project_row); project_layout.setContentsMargins(0, 0, 0, 0)
        self.project_edit = QLineEdit(); self.project_edit.setReadOnly(True)
        self.project_button = QPushButton("选择…"); self.project_button.clicked.connect(self.choose_project)
        project_layout.addWidget(self.project_edit, 1); project_layout.addWidget(self.project_button)
        form.addRow("拼接方案", project_row)
        self.input_summary = QLabel("分别选择图像文件夹和 .stitch.json 拼接方案")
        self.input_summary.setProperty("muted", True); self.input_summary.setWordWrap(True)
        form.addRow("", self.input_summary)
        left.addWidget(inputs)

        output = QGroupBox("2. 大图输出")
        output_layout = QVBoxLayout(output)
        output_row = QHBoxLayout()
        self.output_edit = QLineEdit(); self.output_edit.setReadOnly(True)
        self.output_button = QPushButton("更改…"); self.output_button.clicked.connect(self.choose_output)
        output_row.addWidget(self.output_edit, 1); output_row.addWidget(self.output_button)
        output_layout.addLayout(output_row)
        thumbnail_row = QHBoxLayout()
        thumbnail_row.addWidget(QLabel("缩略图最长边"))
        self.thumbnail_edge = QSpinBox()
        self.thumbnail_edge.setRange(800, 4096)
        self.thumbnail_edge.setSingleStep(200)
        self.thumbnail_edge.setValue(2400)
        self.thumbnail_edge.setSuffix(" px")
        thumbnail_row.addWidget(self.thumbnail_edge)
        thumbnail_row.addStretch(1)
        output_layout.addLayout(thumbnail_row)
        self.canvas_summary = QLabel("载入输入后显示预计画布大小")
        self.canvas_summary.setProperty("muted", True); self.canvas_summary.setWordWrap(True)
        output_layout.addWidget(self.canvas_summary)
        left.addWidget(output)

        self.progress = QProgressBar(); left.addWidget(self.progress)
        self.status = QLabel(""); self.status.setProperty("muted", True); self.status.setWordWrap(True)
        left.addWidget(self.status); left.addStretch(1)
        self.thumbnail_button = QPushButton("生成大图拼接缩略图")
        self.thumbnail_button.setEnabled(False)
        self.thumbnail_button.clicked.connect(self.generate_thumbnail)
        left.addWidget(self.thumbnail_button)
        self.generate_button = QPushButton("合成大图")
        self.generate_button.setProperty("primary", True); self.generate_button.setEnabled(False)
        self.generate_button.clicked.connect(self.generate_mosaic); left.addWidget(self.generate_button)

        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(controls)
        scroll.setMinimumWidth(410); scroll.setMaximumWidth(470); outer.addWidget(scroll)

        viewer = QWidget(); viewer_layout = QVBoxLayout(viewer); viewer_layout.setContentsMargins(10, 10, 12, 12)
        top = QHBoxLayout(); top.addWidget(QLabel("大图预览"))
        self.path_label = QLabel("尚未生成或打开大图"); self.path_label.setProperty("muted", True)
        top.addWidget(self.path_label, 1)
        open_button = QPushButton("打开已有大图…"); open_button.clicked.connect(self.choose_existing)
        top.addWidget(open_button); viewer_layout.addLayout(top)
        self.view = ZoomView(); self.view.set_pil_image(Image.new("RGB", (900, 560), (22, 29, 36)))
        viewer_layout.addWidget(self.view, 1); outer.addWidget(viewer, 1)

    def choose_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择待合成图像文件夹")
        if selected:
            self.set_image_folder(Path(selected))

    def set_image_folder(self, path: Path) -> None:
        folder = path.expanduser().resolve()
        if not folder.is_dir():
            QMessageBox.warning(self, "无法读取", f"图像文件夹不存在：{folder}")
            return
        self.image_folder = folder
        self.folder_edit.setText(str(folder))
        self.output_path = folder.with_name(folder.name + "_stitched.ome.tif")
        self.output_edit.setText(str(self.output_path))
        self._refresh_inputs()

    def choose_project(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "选择拼接方案", "", "拼接方案 (*.stitch.json *.mosaic.json *.json)"
        )
        if selected:
            self.load_project(Path(selected))

    def load_project(self, path: Path, prefer_project_source: bool = False) -> None:
        try:
            resolved = path.expanduser().resolve()
            project = StitchProject.from_dict(json.loads(resolved.read_text(encoding="utf-8")))
        except Exception as exc:
            QMessageBox.warning(self, "无法打开拼接方案", str(exc))
            return
        self.project_path = resolved
        self.project = project
        self.project_edit.setText(str(resolved))
        source = self._project_source(project, resolved)
        if source is not None and (prefer_project_source or self.image_folder is None):
            self.image_folder = source
            self.folder_edit.setText(str(source))
            self.output_path = source.with_name(source.name + "_stitched.ome.tif")
            self.output_edit.setText(str(self.output_path))
        self._refresh_inputs()

    @staticmethod
    def _project_source(project: StitchProject, project_path: Path) -> Path | None:
        candidates: list[Path] = []
        if project.source_folder:
            candidates.append(Path(project.source_folder).expanduser())
        if project.source_folder_relative:
            candidates.append(project_path.parent / project.source_folder_relative)
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved.is_dir():
                return resolved
        return None

    def _refresh_inputs(self) -> None:
        self.tile_set = None
        self.generate_button.setEnabled(False)
        self.thumbnail_button.setEnabled(False)
        if self.image_folder is None or self.project is None:
            missing = []
            if self.image_folder is None: missing.append("图像文件夹")
            if self.project is None: missing.append("拼接方案")
            self.input_summary.setText("还需选择：" + "、".join(missing))
            return
        try:
            pattern = ScanPattern(
                StartCorner(self.project.start_corner),
                FastAxis(self.project.fast_axis),
                self.project.snake,
            )
            tile_set = discover_numeric_tiles(
                self.image_folder, self.project.rows, self.project.cols, pattern
            )
            excluded = set(self.project.excluded_numbers)
            if excluded:
                tile_set.records = [record for record in tile_set.records if record.number not in excluded]
                tile_set.warnings.append(f"已按方案排除 {len(excluded)} 个编号")
            expected_size = (self.project.tile_width, self.project.tile_height)
            actual_size = (tile_set.tile_width or 0, tile_set.tile_height or 0)
            if all(expected_size) and actual_size != expected_size:
                raise ValueError(
                    f"图像尺寸为 {actual_size[0]}×{actual_size[1]}，方案要求 {expected_size[0]}×{expected_size[1]}"
                )
            self.tile_set = tile_set
            geometry = self.project.geometry
            positions = self._positions()
            min_x, min_y, max_x, max_y = mosaic_bounds(
                positions, tile_set.tile_width or 0, tile_set.tile_height or 0,
                geometry.tile_rotation_deg,
            )
            width, height = max_x - min_x, max_y - min_y
            self.input_summary.setText(
                f"{tile_set.count:,} 张 · {self.project.rows}×{self.project.cols} 网格 · "
                f"重叠 {geometry.overlap_x_percent:.1f}% / {geometry.overlap_y_percent:.1f}% · "
                f"旋转 {geometry.tile_rotation_deg:+.2f}°"
                + (("\n" + "；".join(tile_set.warnings)) if tile_set.warnings else "")
            )
            self.canvas_summary.setText(
                f"预计 {width:,} × {height:,} 像素 · RGB 工作缓存约 {width * height * 3 / 1_000_000_000:.2f} GB"
            )
            self.status.setText("输入已就绪")
            self.generate_button.setEnabled(not self._busy)
            self.thumbnail_button.setEnabled(not self._busy)
        except Exception as exc:
            self.input_summary.setText(f"输入不匹配：{exc}")
            self.canvas_summary.setText("请检查文件夹和拼接方案")

    def _positions(self) -> dict[tuple[int, int], np.ndarray]:
        if self.tile_set is None or self.project is None:
            return {}
        positions = regular_positions(self.tile_set, self.project.geometry)
        for key, value in self.project.refined_positions.items():
            try:
                row_col = tuple(map(int, key.split(",")))
                if len(row_col) == 2:
                    positions[row_col] = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError):
                continue
        return positions

    def choose_output(self) -> None:
        default = str(self.output_path or "mosaic_stitched.ome.tif")
        selected, _ = QFileDialog.getSaveFileName(
            self, "保存合成大图", default, "OME-TIFF (*.ome.tif *.ome.tiff)"
        )
        if selected:
            if not selected.lower().endswith((".ome.tif", ".ome.tiff")):
                selected += ".ome.tif"
            self.output_path = Path(selected).expanduser().resolve()
            self.output_edit.setText(str(self.output_path))

    def _thumbnail_output_path(self) -> Path:
        if self.output_path is not None:
            output = self.output_path
        elif self.image_folder is not None:
            output = self.image_folder.with_name(
                self.image_folder.name + "_stitched.ome.tif"
            )
        else:
            output = Path("mosaic_stitched.ome.tif").resolve()
        name = output.name
        lower_name = name.lower()
        for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
            if lower_name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        else:
            name = output.stem
        return output.with_name(name + "_thumbnail.png")

    def generate_thumbnail(self) -> None:
        if self.tile_set is None or self.project is None:
            QMessageBox.information(
                self, "输入未就绪", "请先选择匹配的图像文件夹和拼接方案。"
            )
            return
        output = self._thumbnail_output_path()
        if output.exists():
            answer = QMessageBox.question(
                self,
                "缩略图已存在",
                f"{output.name} 已存在，是否覆盖？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        geometry = self.project.geometry
        self._set_busy(True, "正在生成大图拼接缩略图…")
        worker = Worker(
            render_mosaic_thumbnail,
            self.tile_set,
            self._positions(),
            output,
            geometry.tile_rotation_deg,
            self.project.overlap_blend_percent,
            self.thumbnail_edge.value(),
            with_progress=True,
        )
        worker.signals.progress.connect(self._progress)
        worker.signals.result.connect(self._thumbnail_ready)
        worker.signals.error.connect(self._show_error)
        worker.signals.finished.connect(lambda: self._set_busy(False))
        _start_worker(self, worker)

    def generate_mosaic(self) -> None:
        if self.tile_set is None or self.project is None:
            QMessageBox.information(self, "输入未就绪", "请先选择匹配的图像文件夹和拼接方案。")
            return
        if self.output_path is None:
            if self.image_folder is None:  # guarded by tile_set, kept explicit for type safety
                return
            output = self.image_folder.with_name(self.image_folder.name + "_stitched.ome.tif")
        else:
            output = self.output_path
        if output.exists():
            answer = QMessageBox.question(
                self, "输出文件已存在", f"{output.name} 已存在，是否覆盖？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        positions = self._positions()
        geometry = self.project.geometry
        min_x, min_y, max_x, max_y = mosaic_bounds(
            positions, self.tile_set.tile_width or 0, self.tile_set.tile_height or 0,
            geometry.tile_rotation_deg,
        )
        decoded_gb = (max_x - min_x) * (max_y - min_y) * 3 / 1_000_000_000
        if decoded_gb >= 2.0:
            answer = QMessageBox.question(
                self, "确认大图规模",
                f"预计画布为 {max_x - min_x:,} × {max_y - min_y:,} 像素，工作缓存约 {decoded_gb:.1f} GB。\n输出还需要额外磁盘空间，是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._set_busy(True, "正在分块合成大图…")
        worker = Worker(
            render_mosaic, self.tile_set, positions, output,
            geometry.tile_rotation_deg, self.project.overlap_blend_percent,
            with_progress=True,
        )
        worker.signals.progress.connect(self._progress)
        worker.signals.result.connect(self._mosaic_ready)
        worker.signals.error.connect(self._show_error)
        worker.signals.finished.connect(lambda: self._set_busy(False))
        _start_worker(self, worker)

    def _progress(self, current: int, total: int, message: str) -> None:
        self.progress.setValue(round(100 * current / max(1, total)))
        self.status.setText(f"{message} · {current}/{total}")

    def _mosaic_ready(self, result: tuple[Path, Path]) -> None:
        output, preview = result
        self.open_path(output, preview)
        self.status.setText(f"合成完成：{output.name}")
        self.mosaicCreated.emit(str(output), str(preview))
        QMessageBox.information(self, "完成", f"大图：{output}\n预览：{preview}")

    def _thumbnail_ready(self, output: Path) -> None:
        self.open_path(output)
        self.status.setText(f"缩略图已生成：{output.name}")
        QMessageBox.information(self, "完成", f"拼接缩略图：{output}")

    def _set_busy(self, busy: bool, message: str | None = None) -> None:
        self._busy = busy
        self.generate_button.setEnabled(not busy and self.tile_set is not None)
        self.thumbnail_button.setEnabled(not busy and self.tile_set is not None)
        self.folder_button.setEnabled(not busy); self.project_button.setEnabled(not busy)
        self.output_button.setEnabled(not busy)
        self.thumbnail_edge.setEnabled(not busy)
        if busy: self.progress.setValue(0)
        if message: self.status.setText(message)
        if not busy: self.progress.setValue(0)

    def _show_error(self, details: str) -> None:
        self.status.setText("合成失败")
        box = QMessageBox(self); box.setWindowTitle("合成失败")
        box.setText(details.strip().splitlines()[-1]); box.setDetailedText(details); box.exec()

    def choose_existing(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "打开已有大图", "", "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)"
        )
        if selected:
            path = Path(selected)
            preview = path.with_suffix(".preview.png")
            self.open_path(path, preview if preview.exists() else None)

    def open_path(self, path: Path, preview_path: Path | None = None) -> None:
        self.path = path
        self.path_label.setText(str(path))
        display = preview_path if preview_path and preview_path.exists() else path
        try:
            self.view.set_path(display)
        except Exception as exc:
            QMessageBox.warning(self, "无法打开", str(exc))


class CropTab(QWidget):
    def __init__(self, thread_pool: QThreadPool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.thread_pool = thread_pool
        self._active_workers: set[Worker] = set()
        self.source_path: Path | None = None
        self.source_size: tuple[int, int] | None = None
        outer = QHBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        controls = QWidget(); controls.setMinimumWidth(310); controls.setMaximumWidth(370)
        left = QVBoxLayout(controls); left.setContentsMargins(12, 10, 12, 12)
        group = QGroupBox("分割设置"); form = QFormLayout(group)
        source_row = QWidget(); source_layout = QHBoxLayout(source_row); source_layout.setContentsMargins(0, 0, 0, 0)
        self.source_edit = QLineEdit(); self.source_edit.setReadOnly(True)
        choose = QPushButton("选择…"); choose.clicked.connect(self.choose_source)
        source_layout.addWidget(self.source_edit, 1); source_layout.addWidget(choose)
        form.addRow("大图", source_row)
        self.mode_combo = QComboBox(); self.mode_combo.addItem("固定尺寸", "size"); self.mode_combo.addItem("指定行列", "count")
        form.addRow("方式", self.mode_combo)
        self.patch_w = QSpinBox(); self.patch_w.setRange(64, 32768); self.patch_w.setValue(1024); self.patch_w.setSingleStep(64)
        self.patch_h = QSpinBox(); self.patch_h.setRange(64, 32768); self.patch_h.setValue(1024); self.patch_h.setSingleStep(64)
        size_row = QWidget(); size_layout = QHBoxLayout(size_row); size_layout.setContentsMargins(0, 0, 0, 0); size_layout.addWidget(self.patch_w); size_layout.addWidget(QLabel("×")); size_layout.addWidget(self.patch_h)
        form.addRow("切片尺寸", size_row)
        self.rows = QSpinBox(); self.rows.setRange(1, 10000); self.rows.setValue(10)
        self.cols = QSpinBox(); self.cols.setRange(1, 10000); self.cols.setValue(10)
        count_row = QWidget(); count_layout = QHBoxLayout(count_row); count_layout.setContentsMargins(0, 0, 0, 0); count_layout.addWidget(self.rows); count_layout.addWidget(QLabel("×")); count_layout.addWidget(self.cols)
        form.addRow("行列数", count_row)
        self.overlap_x = FloatControl(0, 30, 0, "%"); self.overlap_y = FloatControl(0, 30, 0, "%"); self.angle = FloatControl(-45, 45, 0, "°")
        form.addRow("X 重叠", self.overlap_x); form.addRow("Y 重叠", self.overlap_y); form.addRow("网格角度", self.angle)
        self.edge = QComboBox(); self.edge.addItem("补黑", "pad"); self.edge.addItem("镜像填充", "reflect"); self.edge.addItem("跳过不完整块", "skip")
        self.format = QComboBox(); self.format.addItem("PNG", "png"); self.format.addItem("TIFF", "tiff"); self.format.addItem("JPEG", "jpeg")
        form.addRow("边缘", self.edge); form.addRow("格式", self.format)
        left.addWidget(group)
        settings_row = QHBoxLayout()
        load_settings = QPushButton("打开参数…"); load_settings.clicked.connect(self.choose_settings)
        save_settings = QPushButton("保存参数…"); save_settings.clicked.connect(self.choose_save_settings)
        settings_row.addWidget(load_settings); settings_row.addWidget(save_settings); left.addLayout(settings_row)
        self.estimate = QLabel("选择大图后显示预计数量"); self.estimate.setProperty("muted", True); self.estimate.setWordWrap(True)
        left.addWidget(self.estimate)
        self.progress = QProgressBar(); left.addWidget(self.progress)
        self.export_button = QPushButton("开始分割…"); self.export_button.setProperty("primary", True); self.export_button.clicked.connect(self.export)
        left.addWidget(self.export_button); left.addStretch(1)
        outer.addWidget(controls)
        self.view = ZoomView(); outer.addWidget(self.view, 1)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        for control in (self.patch_w, self.patch_h, self.rows, self.cols): control.valueChanged.connect(self.update_estimate)
        for control in (self.overlap_x, self.overlap_y, self.angle): control.valueChanged.connect(self.update_estimate)
        self._mode_changed()

    def settings(self) -> CropSettings:
        return CropSettings(
            mode=self.mode_combo.currentData(), patch_width=self.patch_w.value(), patch_height=self.patch_h.value(),
            rows=self.rows.value(), cols=self.cols.value(), overlap_x_percent=self.overlap_x.value(),
            overlap_y_percent=self.overlap_y.value(), angle_deg=self.angle.value(), edge_mode=self.edge.currentData(), output_format=self.format.currentData(),
        )

    def choose_source(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "选择大图", "", "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)")
        if selected:
            self.open_source(Path(selected))

    def open_source(self, path: Path, preview_path: Path | None = None) -> None:
        self.source_path = path; self.source_edit.setText(str(path))
        try:
            with Image.open(preview_path if preview_path and preview_path.exists() else path) as image:
                self.view.set_pil_image(image.convert("RGB").copy())
            try:
                import tifffile
                if path.suffix.lower() in {".tif", ".tiff"}:
                    with tifffile.TiffFile(path) as tif:
                        shape = tif.series[0].shape
                    self.source_size = (shape[-2], shape[-3]) if len(shape) >= 3 and shape[-1] in (3, 4) else (shape[-1], shape[-2])
                else:
                    with Image.open(path) as image:
                        self.source_size = image.size
            except Exception:
                with Image.open(path) as image:
                    self.source_size = image.size
            self.update_estimate()
        except Exception as exc:
            QMessageBox.warning(self, "无法打开", str(exc))

    def update_estimate(self, *_args) -> None:
        if not self.source_size:
            return
        try:
            specs = patch_layout(*self.source_size, self.settings())
            self.estimate.setText(f"预计生成 {len(specs):,} 张 · 命名为 patch_r0000_c0000.{self.format.currentData()}")
        except Exception as exc:
            self.estimate.setText(str(exc))

    def export(self) -> None:
        if not self.source_path:
            QMessageBox.information(self, "尚未选择", "请先选择需要分割的大图。")
            return
        folder = QFileDialog.getExistingDirectory(self, "选择分割输出文件夹")
        if not folder:
            return
        target = Path(folder)
        has_previous_output = (
            (target / "manifest.csv").exists()
            or (target / "split_settings.json").exists()
            or (target / "crop_settings.json").exists()
            or any(target.glob("patch_r*_c*.*"))
        )
        if has_previous_output:
            answer = QMessageBox.question(
                self,
                "检测到已有分割结果",
                "继续会覆盖同名切片、manifest.csv 和 split_settings.json；其他文件会保留。是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes: return
        self.export_button.setEnabled(False)
        worker = Worker(export_patches, self.source_path, folder, self.settings(), with_progress=True)
        worker.signals.progress.connect(lambda current, total, _message: self.progress.setValue(round(100 * current / max(1, total))))
        worker.signals.result.connect(lambda result: QMessageBox.information(self, "完成", f"已生成 {result[1]} 张切片\n{result[0]}"))
        worker.signals.error.connect(lambda details: QMessageBox.warning(self, "分割失败", details.strip().splitlines()[-1]))
        worker.signals.finished.connect(lambda: (self.export_button.setEnabled(True), self.progress.setValue(0)))
        _start_worker(self, worker)

    def choose_settings(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "打开分割参数", "", "分割参数 (*.split.json *.crop.json *.json)")
        if selected:
            try:
                self.apply_settings(load_crop_settings(selected))
            except Exception as exc:
                QMessageBox.warning(self, "无法打开参数", str(exc))

    def choose_save_settings(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(self, "保存分割参数", "split_settings.split.json", "分割参数 (*.split.json)")
        if selected:
            if not selected.lower().endswith(".json"): selected += ".split.json"
            save_crop_settings(selected, self.settings())

    def apply_settings(self, settings: CropSettings) -> None:
        index = self.mode_combo.findData(settings.mode)
        if index >= 0: self.mode_combo.setCurrentIndex(index)
        self.patch_w.setValue(settings.patch_width); self.patch_h.setValue(settings.patch_height)
        self.rows.setValue(settings.rows); self.cols.setValue(settings.cols)
        self.overlap_x.setValue(settings.overlap_x_percent); self.overlap_y.setValue(settings.overlap_y_percent)
        self.angle.setValue(settings.angle_deg)
        edge_index = self.edge.findData(settings.edge_mode)
        if edge_index >= 0: self.edge.setCurrentIndex(edge_index)
        format_index = self.format.findData(settings.output_format)
        if format_index >= 0: self.format.setCurrentIndex(format_index)
        self.update_estimate()

    def _mode_changed(self, *_args) -> None:
        fixed_size = self.mode_combo.currentData() == "size"
        self.patch_w.setEnabled(fixed_size); self.patch_h.setEnabled(fixed_size)
        self.rows.setEnabled(not fixed_size); self.cols.setEnabled(not fixed_size)
        self.update_estimate()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("显微图像工具")
        self.resize(1380, 850)
        self.thread_pool = QThreadPool.globalInstance()
        # Image decoding and FFT tasks have substantial per-task memory use.
        # Two workers keep the UI responsive without multiplying memory peaks.
        self.thread_pool.setMaxThreadCount(2)
        self.tabs = QTabWidget()
        self.stitch_tab = StitchTab(self.thread_pool)
        self.toolmark_tab = ToolMarkTab(self.thread_pool)
        self.mosaic_tab = MosaicTab(self.thread_pool)
        self.crop_tab = CropTab(self.thread_pool)
        self.tabs.addTab(self.stitch_tab, "大图拼接")
        self.tabs.addTab(self.toolmark_tab, "刀纹去除")
        self.tabs.addTab(self.mosaic_tab, "大图合成")
        self.tabs.addTab(self.crop_tab, "大图分割")
        self.setCentralWidget(self.tabs)
        self.stitch_tab.projectSaved.connect(
            lambda path: self.mosaic_tab.load_project(Path(path), prefer_project_source=True)
        )
        self.mosaic_tab.mosaicCreated.connect(self._mosaic_created)
        self.toolmark_tab.marksRemoved.connect(self._marks_removed)
        self._build_menu()
        self._restore_window()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        open_project = QAction("打开拼接方案…", self); open_project.setShortcut(QKeySequence.StandardKey.Open); open_project.triggered.connect(self.open_project)
        save_project = QAction("保存拼接方案…", self); save_project.setShortcut(QKeySequence.StandardKey.Save); save_project.triggered.connect(self.save_project)
        quit_action = QAction("退出", self); quit_action.setShortcut(QKeySequence.StandardKey.Quit); quit_action.triggered.connect(self.close)
        file_menu.addAction(open_project); file_menu.addAction(save_project); file_menu.addSeparator(); file_menu.addAction(quit_action)

    def open_project(self) -> None:
        self.stitch_tab.choose_project(); self.tabs.setCurrentWidget(self.stitch_tab)

    def save_project(self) -> None:
        self.stitch_tab.choose_save_project()

    def _mosaic_created(self, output: str, preview: str) -> None:
        output_path, preview_path = Path(output), Path(preview)
        self.crop_tab.open_source(output_path, preview_path)
        self.tabs.setCurrentWidget(self.mosaic_tab)

    def _marks_removed(self, output: str, preview: str) -> None:
        output_path = Path(output)
        if output_path.is_dir():
            self.mosaic_tab.set_image_folder(output_path)
            self.tabs.setCurrentWidget(self.mosaic_tab)
            return
        preview_path = Path(preview) if preview else None
        self.mosaic_tab.open_path(output_path, preview_path)
        self.crop_tab.open_source(output_path, preview_path)
        self.tabs.setCurrentWidget(self.mosaic_tab)

    def _restore_window(self) -> None:
        settings = QSettings("OpenAI", "LargeImageTool")
        geometry = settings.value("windowGeometry")
        if geometry:
            self.restoreGeometry(geometry)

    def closeEvent(self, event) -> None:  # noqa: N802
        QSettings("OpenAI", "LargeImageTool").setValue("windowGeometry", self.saveGeometry())
        super().closeEvent(event)


def _read_toolmark_preview(path: Path, maximum: int = 1024) -> np.ndarray:
    # Preview the whole frame, scaled to a bounded display size.  The previous
    # center crop hid the outer part of ordinary 3840×2160 microscope tiles.
    with Image.open(path) as image:
        image.thumbnail((maximum, maximum), Image.Resampling.LANCZOS)
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _read_toolmark_preview_with_scale(
    source_path: Path, preview_path: Path,
) -> tuple[np.ndarray, tuple[float, float]]:
    image = _read_toolmark_preview(preview_path)
    # Sidecar previews can have a different resolution from the actual raster.
    with open_raster(source_path) as raster:
        scale = (image.shape[0] / raster.height, image.shape[1] / raster.width)
    return image, scale


def parse_number_ranges(text: str) -> set[int]:
    numbers: set[int] = set()
    for token in (part.strip() for part in text.split(",")):
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                start, end = end, start
            numbers.update(range(start, end + 1))
        else:
            numbers.add(int(token))
    return numbers


def create_application(argv: list[str]) -> QApplication:
    app = QApplication(argv)
    app.setApplicationName("显微图像工具")
    app.setOrganizationName("OpenAI")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)
    return app
