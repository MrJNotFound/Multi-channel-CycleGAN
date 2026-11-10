from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("SimpleITK")
pytest.importorskip("cv2")
import cv2
import SimpleITK as sitk

from tool import img_registrating_pair as pair
from tool import img_registrating_series as series


def rectangle(color=True):
    y, x = np.mgrid[:48, :80]
    gray = ((3*x + 5*y) % 200 + 20).astype(np.uint8)
    return np.stack([gray, np.flipud(gray), np.fliplr(gray)], axis=-1) if color else gray


@pytest.mark.parametrize("color", [False, True])
@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_cached_elastic_matches_direct_for_rectangular_reference(color, scale):
    reference = rectangle(color)
    options = dict(mesh_size=2, shrink_factors=[1], smoothing_sigmas=[0],
                   optimizer_iterations=1, scale_factor=scale)
    transform, fixed, original_shape = series.elastic_registration_transform(
        reference, reference, **options)
    assert original_shape == (48, 80)  # Existing saved parameter format is unchanged.
    actual = pair._apply_elastic_on_reference(reference, fixed, transform, reference)
    direct = series.elastic_registration(reference, reference, **options)
    assert actual.shape == direct.shape == reference.shape
    np.testing.assert_allclose(actual, direct, atol=1)
    if scale == 1.0:
        np.testing.assert_allclose(actual, reference, atol=1)


@pytest.mark.parametrize("missing_shape", [False, True])
def test_import_legacy_parameters_uses_current_reference_size(tmp_path, missing_shape):
    reference = rectangle()
    small = cv2.resize(reference, (40, 24), interpolation=cv2.INTER_AREA)
    fixed = sitk.GetImageFromArray(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32))
    transform = sitk.BSplineTransformInitializer(fixed, [2, 2])
    pair._save_params(str(tmp_path), "弹性", {}, {1: (transform, fixed, (48, 80))})
    shape_file = tmp_path / "elastic/pair_01_shape.npz"
    with np.load(shape_file) as stored:
        np.testing.assert_array_equal(stored["shape"], [48, 80])
    if missing_shape:
        shape_file.unlink()
    _, elastic = pair._load_params(str(tmp_path), "弹性", 1)
    loaded_transform, loaded_fixed, shape = elastic[1]
    assert shape is None if missing_shape else shape == (48, 80)
    actual = pair._apply_elastic_on_reference(
        reference, loaded_fixed, loaded_transform, reference)
    expected = cv2.resize(small, (80, 48), interpolation=cv2.INTER_LANCZOS4)
    assert actual.shape == reference.shape
    np.testing.assert_allclose(actual, expected, atol=1)


def image_names(root, folder, names):
    directory = root / folder
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"discovery-only fixture")
    return directory


def test_parent_scanning_is_one_level_sorted_and_excludes_output(tmp_path):
    image_names(tmp_path, "Z_group", ["second.png", "first.png", "notes.txt"])
    image_names(tmp_path, "A_group", ["b.JPG", "a.png"])
    image_names(tmp_path, "TooFew", ["only.png"])
    image_names(tmp_path, "Nested/deeper", ["a.png", "b.png"])
    output = image_names(tmp_path, "registered", ["a.png", "b.png"])
    (tmp_path / "parent_image.png").write_bytes(b"not a group")
    groups, skipped = pair._discover_groups_from_parent(tmp_path, 2, output)
    assert [name for name, _ in groups] == ["A_group", "Z_group"]
    assert [Path(p).name for p in groups[0][1]] == ["a.png", "b.JPG"]
    assert [Path(p).name for p in groups[1][1]] == ["first.png", "second.png"]
    assert any("TooFew" in item for item in skipped)
    assert any("Nested" in item for item in skipped)
    assert any("registered" in item for item in skipped)


def test_parent_output_cannot_overwrite_source(tmp_path):
    image_names(tmp_path, "group", ["a.png"])
    with pytest.raises(ValueError, match="覆盖"):
        pair._discover_groups_from_parent(tmp_path, 1, tmp_path)


def test_parent_mode_only_needs_one_folder_selection(tmp_path):
    source = tmp_path / "input"
    image_names(source, "first", ["a.png", "b.png"])
    image_names(source, "second", ["a.png", "b.png"])
    with patch.object(pair, "_select_group_source", return_value="parent"), \
         patch.object(pair.filedialog, "askdirectory", return_value=str(source)) as pick, \
         patch.object(pair.filedialog, "askopenfilenames") as files:
        groups = pair._pick_moving_groups(None, 2, str(tmp_path / "output"))
    assert [name for name, _ in groups] == ["first", "second"]
    pick.assert_called_once()
    files.assert_not_called()


def test_manual_group_selection_remains_available():
    with patch.object(pair, "_select_group_source", return_value="files"), \
         patch.object(pair, "_pick_images", return_value=["a.png", "b.png"]), \
         patch.object(pair, "_continue_dialog", return_value=False):
        assert pair._pick_moving_groups(None, 2, "output") == [
            ("group_01", ["a.png", "b.png"])]
