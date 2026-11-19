import importlib

import numpy as np
import pytest


cv2 = pytest.importorskip("cv2")


def test_optional_3d_dependency_does_not_block_other_pages(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PyQt6.QtWidgets")
    from tool.code_GUI.ui import main_window as gui

    def import_page(name, package):
        if name.endswith(".volume_viewer"):
            raise ModuleNotFoundError("missing optional 3D backend")
        return importlib.import_module(name, package)

    monkeypatch.setattr(gui, "import_module", import_page)
    app = widgets.QApplication.instance() or widgets.QApplication([])
    window = gui.MainWindow()
    try:
        assert not window._loaded_pages
        assert all(not button.icon().isNull() for button in window.btns)
        window._switch_page(4)
        assert "requirements-3d.txt" in window.pages.currentWidget().text()
        for index, expected in enumerate(("PreprocessingPage", "RegistrationPage", "StitchingPage", "RegionAnalysisPage")):
            window._switch_page(index)
            assert type(window.pages.currentWidget()).__name__ == expected
        first = window._loaded_pages[0]
        window._switch_page(0)
        assert window.pages.currentWidget() is first
    finally:
        window.deleteLater()
        app.processEvents()


def test_multiple_mosaics_keep_separate_complete_outputs(tmp_path):
    from tool.code_GUI.processing.images import read_image, write_image
    from tool.code_GUI.processing.stitching import MosaicOptions, export_mosaics, local_preview_indices

    paths = [tmp_path / f"图块{index}.png" for index in range(5)]
    for index, path in enumerate(paths, 1):
        write_image(path, np.full((15, 20), index * 30, np.uint8))
    options = MosaicOptions(1, 2, 1, 0, 10, 0, "向左", "向下", False, False)
    destination = tmp_path / "拼接.png"
    progress = []
    callbacks = dict(progress=progress.append, check_interrupted=lambda: None)

    saved, errors, remainder = export_mosaics(paths, destination, options, 0.5, **callbacks)
    assert [path.name for path in saved] == ["拼接_0001.png", "拼接_0002.png"]
    assert not errors and remainder == 1 and progress == [1, 2]
    for path, colors in zip(saved, ({30, 60}, {90, 120})):
        assert set(np.unique(read_image(path))) == colors
    assert local_preview_indices(2, len(paths), 1, 2) == ((1, 2), [2, 3])
    with pytest.raises(FileExistsError):
        export_mosaics(paths, destination, options, 0.5, **callbacks)


def test_registration_continues_with_last_good_anchor_after_a_failed_input(tmp_path, monkeypatch):
    from tool.code_GUI.processing import registration
    from tool.code_GUI.processing.images import write_image

    paths = [tmp_path / f"切片{index}.png" for index in range(5)]
    for index in (0, 1, 3, 4):
        write_image(paths[index], np.full((12, 20, 3), index, np.uint8))
    paths[2].write_bytes(b"broken image")
    matched_to = []

    def register(fixed, moving, **_kwargs):
        matched_to.append((int(fixed[0, 0, 0]), int(moving[0, 0, 0])))
        return moving

    monkeypatch.setattr(registration, "rigid_registration", register)
    progress = []
    saved, errors = registration.batch_register_images_anchor_subseq(
        paths, tmp_path / "配准结果", anchor_interval=2, progress_callback=progress.append,
    )
    assert [path.name for path in saved] == [paths[i].name for i in (0, 1, 3, 4)]
    assert len(errors) == 1 and paths[2].name in errors[0]
    assert matched_to == [(0, 1), (0, 3), (0, 4)]
    assert progress[-1] == 100


def test_label_export_keeps_ids_and_iou_scores_follow_original_labels(tmp_path):
    pytest.importorskip("skimage")
    from tool.code_GUI.processing.region_analysis import iou_preview, iou_score_map, read_labels, write_labels

    labels = np.array([[0, 2, 2], [0, 300, 300]], dtype=np.int32)
    path = tmp_path / "标签.png"
    write_labels(path, labels)
    restored = read_labels(path)
    assert restored.dtype == np.uint16
    np.testing.assert_array_equal(restored, labels)
    filtered = restored.copy()
    filtered[filtered == 2] = 0
    expected = np.array([[0, 0, 0], [0, 0.75, 0.75]], dtype=np.float32)
    np.testing.assert_array_equal(iou_score_map(labels, filtered, [0.1, 0.75]), expected)
    _, preview_labels, background, colors = iou_preview([path], 0, [], 100)
    np.testing.assert_array_equal(preview_labels, labels)
    assert background.shape == colors.shape == (2, 3, 3)
