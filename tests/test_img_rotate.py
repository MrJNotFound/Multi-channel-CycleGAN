import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from tool.code_3D_rendering import img_rotate


def save_image(path, image):
    ok, encoded = cv2.imencode(path.suffix, image)
    assert ok
    encoded.tofile(path)


def load_image(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


@pytest.mark.parametrize("workers", [1, 3])
def test_parallel_rotation_is_90_clockwise_with_unicode_paths(tmp_path, workers):
    source = tmp_path / "输入图像"
    destination = tmp_path / "旋转结果"
    source.mkdir()
    rng = np.random.default_rng(12)
    images = {
        "彩色.png": rng.integers(0, 256, (59, 83, 3), dtype=np.uint8),
        "gray.PNG": rng.integers(0, 256, (43, 68), dtype=np.uint8),
        "alpha.png": rng.integers(0, 256, (36, 77, 4), dtype=np.uint8),
        "sixteen.png": rng.integers(0, 65536, (29, 39), dtype=np.uint16),
    }
    for name, image in images.items():
        save_image(source / name, image)
    (source / "metadata.png").mkdir()
    (source / "notes.txt").write_text("ignored")
    thread_count = cv2.getNumThreads()

    result = img_rotate.rotate_images(source, destination, workers, progress=False)

    assert result.total == result.succeeded == len(images)
    assert not result.errors
    assert cv2.getNumThreads() == thread_count
    assert {path.name for path in destination.iterdir()} == {f"flipped_{name}" for name in images}
    for name in images:
        expected = np.rot90(load_image(source / name), k=-1)
        np.testing.assert_array_equal(load_image(destination / f"flipped_{name}"), expected)


def test_corrupt_image_is_reported_without_overwriting_existing_result(tmp_path):
    source, destination = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    destination.mkdir()
    save_image(source / "good.png", np.full((20, 25, 3), 72, dtype=np.uint8))
    (source / "broken.png").write_bytes(b"invalid PNG")
    (source / "empty.png").touch()
    existing = destination / "flipped_broken.png"
    save_image(existing, np.full((20, 25, 3), 91, dtype=np.uint8))
    before = existing.read_bytes()

    result = img_rotate.rotate_images(source, destination, 3, progress=False)

    assert result.succeeded == 1
    assert {name for name, error in result.errors} == {"broken.png", "empty.png"}
    assert existing.read_bytes() == before
    assert (destination / "flipped_good.png").exists()
    assert not (destination / "flipped_empty.png").exists()
    assert not list(destination.glob("*.tmp"))


def test_write_failure_preserves_previous_output_and_cleans_temp(tmp_path, monkeypatch):
    source, destination = tmp_path / "source.png", tmp_path / "destination.png"
    save_image(source, np.full((32, 40, 3), 111, dtype=np.uint8))
    destination.write_bytes(b"previous complete output")

    def fail_replace(*args):
        raise PermissionError("output is locked")

    monkeypatch.setattr(img_rotate.os, "replace", fail_replace)
    with pytest.raises(PermissionError, match="locked"):
        img_rotate.rotate_one(source, destination)
    assert destination.read_bytes() == b"previous complete output"
    assert not list(tmp_path.glob("*.tmp"))


def test_invalid_input_does_not_create_output(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    with pytest.raises(ValueError):
        img_rotate.rotate_images(source, destination, progress=False)
    source.mkdir()
    with pytest.raises(ValueError):
        img_rotate.rotate_images(source, destination, progress=False)
    with pytest.raises(ValueError):
        img_rotate.rotate_images(source, source, progress=False)
    with pytest.raises(ValueError):
        img_rotate.rotate_images(source, destination, 0, progress=False)
    assert not destination.exists()
