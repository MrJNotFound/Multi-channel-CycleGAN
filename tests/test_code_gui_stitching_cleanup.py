import time

import numpy as np
import pytest


@pytest.fixture
def stitching():
    pytest.importorskip("scipy")
    from tool.code_GUI.processing.stitching import stitch_images_mosaic

    return stitch_images_mosaic


def test_gap_fill_repairs_padding_and_internal_holes_without_touching_data(stitching):
    image = np.full((218, 980), 30000, dtype=np.uint16)
    image[110:125, 480:491] = 0  # Measured dark pixels in the source tile.
    coverage = np.ones_like(image, dtype=np.uint8)
    coverage[:90] = coverage[-15:] = 0
    coverage[100:106, 760:780] = 0  # Internal gap that crosses a processing block boundary.
    image[coverage == 0] = 0
    before = image.copy()

    filled = stitching([image], (1, 1), fill_gaps=True, coverage_masks=[coverage])

    assert filled.dtype == np.uint16
    np.testing.assert_array_equal(filled[coverage != 0], before[coverage != 0])
    assert np.all(filled[coverage == 0] == 30000)
    np.testing.assert_array_equal(image, before)


def test_color_normalization_keeps_a_single_ramp_across_tile_boundaries(stitching):
    height, width, overlap = 40, 64, 24
    step = width - overlap
    x = np.arange(width + step, dtype=np.uint16)
    reference = np.broadcast_to((x[:, None] + [70, 77, 90]) * 120, (height, width + step, 3))
    left = reference[:, :width].astype(np.uint16)
    right = (reference[:, step:] + [1200, 2400, 4800]).astype(np.uint16)
    # A valid outlier should not skew the median offset estimate.
    right[5:9, :10] = 40000

    original = stitching([left, right], (1, 2), overlap_width=overlap)
    corrected = stitching([left, right], (1, 2), overlap_width=overlap, blend_overlaps=True)
    reversed_result = stitching(
        [right, left], (1, 2), overlap_width=overlap, blend_overlaps=True, direction_width="向右"
    )

    assert corrected.dtype == original.dtype == np.uint16
    assert corrected.shape == original.shape
    assert np.diff(original[0].astype(int), axis=0).max() > 1000
    np.testing.assert_allclose(np.diff(corrected[0].astype(int), axis=0), 120, atol=2)
    np.testing.assert_array_equal(corrected, reversed_result)


def test_options_reach_preview_and_saved_image(stitching, tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PyQt6.QtWidgets")
    cv2 = pytest.importorskip("cv2")
    from tool.code_GUI.processing.stitching import prepare_tile
    from tool.code_GUI.ui.pages import stitching as page_module

    app = widgets.QApplication.instance() or widgets.QApplication([])
    page = page_module.StitchingPage()
    messages = []
    monkeypatch.setattr(page_module, "show_info", lambda _page, message: messages.append(message))
    output = tmp_path / "saved.png"
    inputs = [tmp_path / "left.png", tmp_path / "right.png"]
    for path, value in zip(inputs, (110, 160)):
        assert cv2.imwrite(str(path), np.full((96, 112), value, dtype=np.uint8))
    monkeypatch.setattr(widgets.QFileDialog, "getSaveFileName", lambda *_args: (str(output), "PNG"))
    previews, completed = [], []
    monkeypatch.setattr(page.interactive_image, "set_image", lambda image, **_kwargs: previews.append(image))
    monkeypatch.setattr(page, "_on_batch_stitch_done", lambda count, path: completed.append((count, path)))
    try:
        assert not page.chk_fill_gaps.isChecked() and not page.chk_blend_overlaps.isChecked()
        page.sp_grid_row.setValue(1)
        page.cmb_dir_width.setCurrentText("向左")
        page.sld_overlap_w.setValue(30)
        page.sp_angle.setValue(6)
        page.image_paths = [str(path) for path in inputs]
        page.current_index = 0
        page.chk_blend_overlaps.setChecked(True)
        assert previews[-1].min() == 0
        page.chk_fill_gaps.setChecked(True)
        preview = previews[-1]
        assert preview.min() > 0 and np.ptp(preview) < 12
        # A rounded resize must sample the image and its coverage at the same coordinates.
        tile, coverage = prepare_tile(np.full((96, 112), 110, np.uint8), 6, 0.53, True)
        np.testing.assert_array_equal(tile != 0, coverage != 0)
        page.on_batch_stitch_async()
        worker = page.stitch_worker
        assert worker is not None and worker.wait(5000)
        deadline = time.monotonic() + 2
        while not completed and time.monotonic() < deadline:
            app.processEvents()
        assert completed == [(1, str(output))] and not messages
        np.testing.assert_array_equal(cv2.imread(str(output), cv2.IMREAD_UNCHANGED), preview)
    finally:
        if page.stitch_worker is not None:
            page.stitch_worker.wait()
        page.deleteLater()
        app.processEvents()
