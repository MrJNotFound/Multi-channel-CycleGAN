import importlib.util
import math
import os
from pathlib import Path
import threading
import time

import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
GUI_DIR = ROOT / "tool" / "code_GUI"
LARGE_IMAGE_SRC = ROOT / "tool" / "large-image-tool" / "src"
spec = importlib.util.spec_from_file_location("stitch_estimation_under_test", GUI_DIR / "func_stitching_estimation.py")
estimation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(estimation)


@pytest.fixture
def make_tiles(tmp_path, monkeypatch):
    pytest.importorskip("scipy")
    monkeypatch.syspath_prepend(str(LARGE_IMAGE_SRC))
    from large_image_tool.models import GridGeometry, ScanPattern, StartCorner
    from large_image_tool.synthetic import generate_turning_dataset

    def generate(direction_width="向左", direction_height="向下", rotation=2.0):
        corner = {
            ("向左", "向下"): StartCorner.TOP_LEFT,
            ("向右", "向下"): StartCorner.TOP_RIGHT,
            ("向左", "向上"): StartCorner.BOTTOM_LEFT,
            ("向右", "向上"): StartCorner.BOTTOM_RIGHT,
        }[(direction_width, direction_height)]
        folder = tmp_path / f"图像_{direction_width}_{direction_height}_{rotation}"
        generate_turning_dataset(
            folder, 2, 3, (320, 240), GridGeometry(20.0, 15.0, rotation),
            pattern=ScanPattern(start_corner=corner, snake=False), jitter_px=0, seed=12,
        )
        return [str(path) for path in sorted(folder.glob("*.png"))]

    return generate


@pytest.mark.parametrize("direction_width", ["向左", "向右"])
@pytest.mark.parametrize("direction_height", ["向上", "向下"])
@pytest.mark.parametrize("rotation", [-2.0, 0.0, 2.0])
def test_reuses_estimator_with_existing_directions_and_canvas_units(
    make_tiles, direction_width, direction_height, rotation
):
    paths = make_tiles(direction_width, direction_height, rotation)
    progress = []
    result = estimation.estimate_stitching_parameters(
        paths, 2, 3, direction_width, direction_height,
        progress=lambda current, total, message: progress.append((current, total)),
    )
    assert result["angle"] == pytest.approx(-rotation, abs=0.3)
    radians = math.radians(result["angle"])
    rotated_width = int(240 * abs(math.sin(radians)) + 320 * abs(math.cos(radians)))
    rotated_height = int(240 * abs(math.cos(radians)) + 320 * abs(math.sin(radians)))
    assert result["overlap_width"] == pytest.approx(rotated_width - 320 * 0.8, abs=2)
    assert result["overlap_height"] == pytest.approx(rotated_height - 240 * 0.85, abs=2)
    assert result["sample_pairs"] == 7
    assert progress[-1] == (7, 7)


def test_only_first_complete_group_is_estimated(make_tiles):
    paths = make_tiles()
    result = estimation.estimate_stitching_parameters(
        paths + ["this_missing_file_is_in_the_next_group.png"], 2, 3, "向左", "向下"
    )
    assert result["sample_pairs"] == 7


@pytest.mark.parametrize("rows,cols,count", [(1, 4, 4), (4, 1, 4), (2, 3, 5)])
def test_invalid_grid_is_rejected(rows, cols, count):
    with pytest.raises(ValueError):
        estimation.estimate_stitching_parameters(["missing.png"] * count, rows, cols, "向左", "向下")


def test_mismatched_image_sizes_are_rejected(tmp_path):
    paths = []
    for index in range(4):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (128 + index, 96)).save(path)
        paths.append(str(path))
    with pytest.raises(ValueError, match="尺寸不一致"):
        estimation.estimate_stitching_parameters(paths, 2, 2, "向左", "向下")


def test_blank_images_do_not_overwrite_parameters_with_a_false_estimate(tmp_path):
    paths = []
    for index in range(4):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (96, 96), (128, 128, 128)).save(path)
        paths.append(str(path))
    with pytest.raises(ValueError, match="可靠"):
        estimation.estimate_stitching_parameters(paths, 2, 2, "向左", "向下")


@pytest.fixture(scope="module")
def qt_app():
    previous = os.environ.get("QT_QPA_PLATFORM")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    widgets = pytest.importorskip("PyQt6.QtWidgets")
    app = widgets.QApplication.instance() or widgets.QApplication([])
    yield app
    if previous is None:
        os.environ.pop("QT_QPA_PLATFORM", None)
    else:
        os.environ["QT_QPA_PLATFORM"] = previous


@pytest.fixture
def page(qt_app, monkeypatch):
    monkeypatch.syspath_prepend(str(GUI_DIR))
    import class_StitchingPage

    messages = []
    monkeypatch.setattr(class_StitchingPage, "show_info", lambda parent, text: messages.append(text))
    widget = class_StitchingPage.StitchPage()
    yield widget, class_StitchingPage, messages
    widget._stop_auto_estimation()
    qt_app.processEvents()
    widget.deleteLater()
    qt_app.processEvents()


def wait_for_estimator(widget, qt_app):
    deadline = time.monotonic() + 15
    while widget.estimate_worker is not None and time.monotonic() < deadline:
        qt_app.processEvents()
        time.sleep(0.005)
    assert widget.estimate_worker is None, "Parameter estimation timed out"


def test_button_runs_in_background_and_fills_existing_controls(page, qt_app, make_tiles):
    widget, module, messages = page
    widget.sp_grid_col.setValue(3)
    widget.cmb_dir_width.setCurrentText("向左")
    widget.image_paths = make_tiles()
    widget.btn_auto_estimate.click()
    assert not widget.btn_auto_estimate.isEnabled()
    wait_for_estimator(widget, qt_app)
    assert widget.btn_auto_estimate.isEnabled()
    assert widget.sp_angle.value() == pytest.approx(-2.0, abs=0.3)
    assert widget.sld_overlap_w.value() == pytest.approx(72, abs=3)
    assert widget.sld_overlap_h.value() == pytest.approx(47, abs=3)
    assert widget.sp_compress.value() == 1.0
    assert widget.sld_black.value() == 10
    assert any("自动估计完成" in message for message in messages)


def test_parameter_commit_updates_preview_once_and_extends_slider_only_as_needed(page, monkeypatch):
    widget, module, messages = page
    preview_calls = []
    monkeypatch.setattr(widget, "_update_preview_for_current", lambda **kwargs: preview_calls.append(kwargs))
    widget._estimate_signature = widget._auto_estimate_signature()
    widget._on_auto_estimate_result({"overlap_width": 812, "overlap_height": 613, "angle": -1.2})
    assert len(preview_calls) == 1
    assert widget.sld_overlap_w.value() == widget.sld_overlap_w.maximum() == 812
    assert widget.sld_overlap_h.value() == widget.sld_overlap_h.maximum() == 613
    assert widget.sp_angle.value() == -1.2
    assert widget.lbl_overlap_w.text() == "横向重叠: 812"
    assert widget.lbl_overlap_h.text() == "纵向重叠: 613"


def test_changed_images_discard_stale_background_result(page, qt_app, monkeypatch):
    widget, module, messages = page
    started, release = threading.Event(), threading.Event()

    def estimate(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return {"overlap_width": 90, "overlap_height": 80, "angle": 2.0}

    monkeypatch.setattr(module, "estimate_stitching_parameters", estimate)
    widget.image_paths = [f"{index}.png" for index in range(4)]
    widget.on_auto_estimate()
    try:
        assert started.wait(5)
        widget.on_clear_list()
    finally:
        release.set()
    wait_for_estimator(widget, qt_app)
    assert widget.sp_angle.value() == 0
    assert widget.sld_overlap_w.value() == widget.sld_overlap_h.value() == 60
    assert any("未应用" in message for message in messages)


def test_estimation_failure_restores_button_and_manual_values(page, qt_app, monkeypatch):
    widget, module, messages = page

    def estimate(*args, **kwargs):
        raise ValueError("no reliable matches")

    monkeypatch.setattr(module, "estimate_stitching_parameters", estimate)
    widget.image_paths = [f"{index}.png" for index in range(4)]
    widget.on_auto_estimate()
    wait_for_estimator(widget, qt_app)
    assert widget.btn_auto_estimate.isEnabled()
    assert widget.sp_angle.value() == 0 and widget.sld_overlap_w.value() == 60
    assert any("no reliable matches" in message for message in messages)


def test_not_enough_images_does_not_start_worker(page):
    widget, module, messages = page
    widget.on_auto_estimate()
    assert widget.estimate_worker is None and widget.btn_auto_estimate.isEnabled()
    assert messages
