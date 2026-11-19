from pathlib import Path
import time

import numpy as np
import pytest


@pytest.fixture
def toolmark_module(monkeypatch):
    pytest.importorskip("scipy")
    tool = Path(__file__).resolve().parents[1] / "tool"
    monkeypatch.syspath_prepend(str(tool / "large-image-tool" / "src"))
    from tool.code_GUI.processing import large_image_backend

    return large_image_backend


def stripe_image():
    y, x = np.mgrid[:160, :256]
    clean = 125 + 8 * np.sin(y / 27) + 7 * np.cos(x / 42)
    image = np.rint(clean + 24 * np.sin(x * (2 * np.pi / 17.4))).astype(np.uint8)
    return image, clean


def test_directional_filter_reuses_backend_and_preserves_alpha(toolmark_module):
    from large_image_tool.models import MarkFilterSettings, ToolMarkMethod
    from large_image_tool.toolmarks import remove_tool_marks

    image, clean = stripe_image()
    alpha = np.tile(np.arange(image.shape[1], dtype=np.uint8), (image.shape[0], 1))
    bgra = np.stack((image + 5, image, image - 10, alpha), axis=2)
    settings = dict(strength=0.9, auto_orientation=True, directional_kernel=51)
    actual, report = toolmark_module.directional_median(bgra, settings)
    expected, _ = remove_tool_marks(
        bgra[..., [2, 1, 0]], MarkFilterSettings(
            method=ToolMarkMethod.DIRECTIONAL_1D.value, auto_period=False, **settings,
        ),
    )
    assert report.method == ToolMarkMethod.DIRECTIONAL_1D.value and report.applied
    assert abs(report.orientation_deg) < 1
    np.testing.assert_array_equal(actual[..., :3], expected[..., ::-1])
    np.testing.assert_array_equal(actual[..., 3], alpha)
    center = np.s_[:, 20:-20]
    before = np.mean((image[center] - clean[center]) ** 2)
    after = np.mean((actual[..., 1][center] - clean[center]) ** 2)
    assert after < before * 0.15


def test_directional_option_routes_preview_and_both_saves(toolmark_module, tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PyQt6.QtWidgets")
    from tool.code_GUI.processing.images import read_image, write_image
    from tool.code_GUI.ui.pages import preprocessing as page_module

    app = widgets.QApplication.instance() or widgets.QApplication([])
    page = page_module.PreprocessingPage()
    messages = []
    monkeypatch.setattr(page_module, "show_info", lambda _page, message: messages.append(message))

    source_path = tmp_path / "输入刀纹.png"
    single_path = tmp_path / "单张结果.png"
    batch_dir = tmp_path / "批量结果"
    batch_dir.mkdir()
    source, _ = stripe_image()
    write_image(source_path, source)
    monkeypatch.setattr(widgets.QFileDialog, "getOpenFileNames", lambda *_a, **_kw: ([str(source_path)], "PNG"))
    monkeypatch.setattr(widgets.QFileDialog, "getSaveFileName", lambda *_a, **_kw: (str(single_path), "PNG"))
    monkeypatch.setattr(widgets.QFileDialog, "getExistingDirectory", lambda *_a, **_kw: str(batch_dir))

    def wait_for(condition):
        deadline = time.monotonic() + 8
        while not condition() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert condition()

    try:
        assert not page.toolmark_group.isChecked()
        page.toolmark_group.setChecked(True)
        page.on_load()
        assert not page.sp_toolmark_angle.isEnabled()
        page.chk_toolmark_auto.setChecked(False)
        assert page.sp_toolmark_angle.isEnabled()
        page.sp_toolmark_strength.setValue(0.9)
        wait_for(lambda: 0 in page.result_cache)
        preview = page.result_cache[0]
        assert "刀纹法向角" in page.lbl_current_view.text()
        page.on_save_single()
        assert page.batch_worker is not None
        wait_for(lambda: page.batch_worker is None)
        page.on_save_batch_async()
        assert page.batch_worker is not None
        wait_for(lambda: page.batch_worker is None)
        np.testing.assert_array_equal(read_image(single_path), preview)
        np.testing.assert_array_equal(read_image(batch_dir / f"processed_{source_path.name}"), preview)
        assert any("当前图像已保存" in message for message in messages)
        assert any("成功处理并保存 1 张" in message for message in messages)
    finally:
        page._stop_workers()
        app.processEvents()
        page.deleteLater()
        app.processEvents()
