from dataclasses import replace
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2")
ndimage = pytest.importorskip("scipy.ndimage")
from tool.code_virtual_staining.tissue_segmentation import (
    SegmentationSettings, export_masks, load_analysis_image, segment_tissue,
)


def tissue_image():
    image = np.full((240, 360), 240, np.uint8)
    cv2.rectangle(image, (50, 40), (180, 180), 70, -1)
    cv2.rectangle(image, (225, 70), (285, 160), 85, -1)
    cv2.circle(image, (100, 100), 22, 240, -1)
    cv2.rectangle(image, (145, 40), (165, 100), 240, -1)  # Open notch in boundary.
    image[3:5, 3:5] = 70  # Debris must not enlarge the convex hull.
    return image


def test_default_mask_covers_tissue_is_connected_and_has_no_holes():
    source = tissue_image()
    before = source.copy()
    mask, report = segment_tissue(source)
    foreground = source < 100
    foreground[:10] = False
    assert np.all(mask[foreground] == 255)
    assert mask[100, 100] == 255  # Hole filled.
    assert mask[110, 200] == 255  # Gap between disconnected pieces bridged.
    assert mask[3, 3] == 0
    assert report["components"] == 1
    np.testing.assert_array_equal(ndimage.binary_fill_holes(mask), mask > 0)
    np.testing.assert_array_equal(source, before)
    assert set(np.unique(mask)) == {0, 255}


def test_largest_and_all_preserve_concavity_and_allow_fill_toggle():
    settings = SegmentationSettings(method="manual", threshold=150, blur_sigma=0,
        closing_radius=0, dilation_radius=0, region_mode="all", fill_holes=False)
    source = tissue_image()
    separate, report = segment_tissue(source, settings)
    assert report["components"] == 2
    assert separate[100, 100] == 0
    filled, report = segment_tissue(source, replace(settings, fill_holes=True))
    assert filled[100, 100] == 255
    assert filled[60, 155] == 0  # Open notch remains outside.
    largest, report = segment_tissue(source, replace(settings, fill_holes=True, region_mode="largest"))
    assert report["components"] == 1
    assert largest[110, 260] == 0
    assert largest[110, 70] == 255


@pytest.mark.parametrize("bright", [False, True])
def test_auto_polarity_uses_border_when_tissue_occupies_majority(bright):
    source = np.full((120, 160), 240, np.uint8)
    source[4:-4, 4:-4] = 40
    if bright:
        source = 255-source
    mask, report = segment_tissue(source, SegmentationSettings(blur_sigma=0, closing_radius=0, dilation_radius=0))
    assert report["polarity"] == ("bright" if bright else "dark")
    assert mask[60, 80] == 255 and mask[0, 0] == 0


def test_saturation_and_manual_threshold_offset():
    rgb = np.full((120, 160, 3), 235, np.uint8)
    rgb[20:100, 25:130] = [160, 70, 180]
    mask, report = segment_tissue(rgb, SegmentationSettings(channel="saturation", blur_sigma=0,
        closing_radius=0, dilation_radius=0))
    assert report["polarity"] == "bright"
    assert mask[60, 70] == 255 and mask[0, 0] == 0
    source = np.full((120, 160), 230, np.uint8)
    source[20:100, 25:130] = 150
    manual = SegmentationSettings(method="manual", threshold=140, polarity="dark", blur_sigma=0,
        closing_radius=0, dilation_radius=0)
    first, _ = segment_tissue(source, manual)
    second, report = segment_tissue(source, replace(manual, threshold_offset=20))
    assert not np.any(first)
    assert second[60, 70] == 255
    assert report["applied_threshold"] == 160


@pytest.mark.parametrize("value", [0, 128, 255])
def test_uniform_image_does_not_produce_false_tissue(value):
    mask, report = segment_tissue(np.full((120, 160), value, np.uint8))
    assert not np.any(mask)
    assert report["components"] == 0


def test_holes_in_tissue_touching_frame_are_filled_without_filling_background():
    source = np.full((160, 240), 240, np.uint8)
    source[10:140, :100] = 50
    source[40:90, 20:60] = 240
    mask, _ = segment_tissue(source, SegmentationSettings(method="manual", threshold=100,
        polarity="dark", region_mode="all", blur_sigma=0, closing_radius=0, dilation_radius=0))
    assert mask[50, 40] == 255
    assert mask[50, 0] == 255
    assert mask[50, 150] == 0


def test_export_is_binary_original_size_and_matches_preview_with_dpi_and_duplicates(tmp_path):
    paths = [tmp_path / "组织.png", tmp_path / "组织.bmp"]
    original = Image.fromarray(tissue_image()).resize((1080, 720), Image.Resampling.NEAREST)
    for path in paths:
        original.save(path, dpi=(12700,12700))
    source_bytes = [path.read_bytes() for path in paths]
    output = tmp_path / "掩膜"
    settings = SegmentationSettings(analysis_max_side=256)
    result = export_masks(paths, output, settings)
    assert not result["errors"]
    assert len(result["results"]) == 2
    assert len({row["output"] for row in result["results"]}) == 2
    for path, row in zip(paths, result["results"]):
        analysis = load_analysis_image(path, settings.analysis_max_side)
        preview, _ = segment_tissue(analysis.rgb, settings)
        expected = np.array(Image.fromarray(preview).resize((1080,720), Image.Resampling.NEAREST))
        with Image.open(row["output"]) as mask:
            assert mask.mode == "L" and mask.size == (1080,720)
            assert abs(mask.info["dpi"][0]-12700) < .1
            np.testing.assert_array_equal(np.array(mask), expected)
    assert [path.read_bytes() for path in paths] == source_bytes
    saved = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    assert saved["settings"]["analysis_max_side"] == 256
    assert len(saved["results"]) == 2
    repeat = export_masks(paths, output, settings)
    assert len(list(output.glob("*_mask*.png"))) == 4
    assert not repeat["errors"]


def test_16bit_and_transparent_input(tmp_path):
    gray = np.full((200,300), 60000, np.uint16)
    gray[40:160, 60:240] = 12000
    path = tmp_path / "sixteen.tiff"
    Image.fromarray(gray).save(path)
    analysis = load_analysis_image(path, 256)
    assert analysis.rgb.max() > 220 and analysis.rgb.min() < 60
    mask, report = segment_tissue(analysis.rgb)
    assert report["components"] == 1
    assert mask[mask.shape[0]//2,mask.shape[1]//2] == 255
    rgba = np.zeros((120,160,4),np.uint8)
    rgba[30:90,40:120] = [70,60,80,255]
    path = tmp_path / "alpha.png"
    Image.fromarray(rgba).save(path)
    analysis = load_analysis_image(path)
    np.testing.assert_array_equal(analysis.rgb[0,0], [255,255,255])
    mask, _ = segment_tissue(analysis.rgb)
    assert mask[0,0] == 0 and mask[60,80] == 255


def test_batch_continues_after_bad_file_and_honors_cancellation(tmp_path):
    valid, bad = tmp_path / "good.png", tmp_path / "bad.png"
    Image.fromarray(tissue_image()).save(valid)
    bad.write_bytes(b"not an image")
    result = export_masks([bad,valid],tmp_path / "out",SegmentationSettings())
    assert len(result["results"]) == len(result["errors"]) == 1
    cancelled = threading.Event()
    cancelled.set()
    result = export_masks([valid],tmp_path / "cancel",SegmentationSettings(),cancel=cancelled)
    assert result["cancelled"] and not result["results"]
    assert not list((tmp_path/"cancel").glob("*.png"))


def test_gui_preview_and_export_use_latest_parameters(tmp_path, monkeypatch):
    import tkinter as tk
    from tool.code_virtual_staining import threshold_seg as gui
    path = tmp_path / "sample.png"
    Image.fromarray(tissue_image()).save(path)
    root = tk.Tk()
    root.withdraw()
    app = gui.SegmentationApp(root)
    monkeypatch.setattr(gui.filedialog,"askopenfilenames",lambda **kwargs: (str(path),))
    def pump(predicate):
        deadline = time.monotonic()+8
        while time.monotonic() < deadline:
            root.update()
            if predicate():
                return
            time.sleep(.01)
        raise AssertionError(app.status.get())
    try:
        app.choose_images()
        pump(lambda: app.preview is not None)
        assert app.preview[2]["components"] == 1
        app.vars["method"].set("手动固定阈值")
        app.vars["threshold"].set("10")
        app.vars["polarity"].set("暗色是组织")
        pump(lambda: app.preview[2]["applied_threshold"] == 10)
        assert not np.any(app.preview[1])
        app.output_var.set(str(tmp_path/"gui_masks"))
        app.export()
        pump(lambda: not app.exporting)
        with Image.open(tmp_path/"gui_masks"/"sample_mask.png") as result:
            assert result.mode == "L" and result.getbbox() is None
        assert "成功 1" in app.status.get()
    finally:
        app.close()
