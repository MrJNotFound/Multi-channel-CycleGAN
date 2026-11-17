import importlib.util
from pathlib import Path
import sys
import threading
import time

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
module_path = Path(__file__).resolve().parents[1] / "tool" / "code_reg & cstrct" / "img_rotate.py"
spec = importlib.util.spec_from_file_location("img_rotate_under_test", module_path)
img_rotate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = img_rotate
spec.loader.exec_module(img_rotate)


def save_image(path, image):
    ok, encoded = cv2.imencode(path.suffix, image)
    assert ok
    encoded.tofile(path)


def load_image(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


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
        actual = load_image(destination / f"flipped_{name}")
        assert actual.dtype == images[name].dtype
        np.testing.assert_array_equal(actual, expected)


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


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_rotation_and_grayscale_preserve_16_bit_pixels(tmp_path, rotation):
    source, destination = tmp_path / "输入", tmp_path / "输出"
    source.mkdir()
    image = np.random.default_rng(4).integers(0, 65536, (17, 29, 4), dtype=np.uint16)
    save_image(source / "切片.PNG", image)
    options = img_rotate.RotationOptions(
        rotation=rotation, save_grayscale=True, output_format="tiff", prefix="处理_"
    )
    result = img_rotate.rotate_images(source, destination, 2, options=options, progress=False)
    assert result.succeeded == 1 and not result.errors
    expected = np.rot90(cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY), k=-(rotation // 90))
    actual = load_image(destination / "处理_切片.tif")
    assert actual.dtype == np.uint16
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("output_format", ["png", "jpg", "tif", "bmp", "webp"])
def test_format_selection_and_case_insensitive_input_filter(tmp_path, output_format):
    source, destination = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    image = np.arange(19 * 31 * 3, dtype=np.uint8).reshape(19, 31, 3)
    save_image(source / "selected.PNG", image)
    save_image(source / "ignored.jpg", image)
    options = img_rotate.RotationOptions(input_extensions=("PNG",), output_format=output_format)
    result = img_rotate.rotate_images(source, destination, options=options, progress=False)
    assert result.total == result.succeeded == 1
    target = destination / f"flipped_selected.{output_format}"
    assert [path.name for path in destination.iterdir()] == [target.name]
    actual = load_image(target)
    assert actual.shape == (31, 19, 3)
    if output_format != "jpg":
        np.testing.assert_array_equal(actual, np.rot90(image, k=-1))
    else:
        assert target.read_bytes().startswith(b"\xff\xd8")


@pytest.mark.parametrize("output_format", ["jpg", "bmp", "webp"])
def test_incompatible_bit_depth_is_reported_without_conversion(tmp_path, output_format):
    source, destination = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    save_image(source / "depth.png", np.full((12, 14), 40000, dtype=np.uint16))
    options = img_rotate.RotationOptions(output_format=output_format)
    result = img_rotate.rotate_images(source, destination, options=options, progress=False)
    assert result.succeeded == 0 and result.processed == 1
    assert len(result.errors) == 1 and "uint16" in result.errors[0][1]
    assert not list(destination.iterdir())


def test_jpeg_alpha_is_reported(tmp_path):
    source, destination = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    save_image(source / "alpha.png", np.zeros((12, 14, 4), dtype=np.uint8))
    options = img_rotate.RotationOptions(output_format="jpg")
    result = img_rotate.rotate_images(source, destination, options=options, progress=False)
    assert result.succeeded == 0 and "透明通道" in result.errors[0][1]


def test_recursion_preserves_subfolders_and_excludes_output(tmp_path):
    source = tmp_path / "输入"
    destination = source / "输出"
    for name in ("甲", "乙"):
        (source / name).mkdir(parents=True)
        save_image(source / name / "same.png", np.full((11, 15), 77, dtype=np.uint8))
    destination.mkdir()
    save_image(destination / "previous.png", np.full((11, 15), 88, dtype=np.uint8))
    options = img_rotate.RotationOptions(recursive=True, output_format="png")
    for _ in range(2):
        result = img_rotate.rotate_images(source, destination, 2, options=options, progress=False)
        assert result.total == result.succeeded == 2
    for name in ("甲", "乙"):
        assert (destination / name / "flipped_same.png").exists()


def test_format_conversion_collision_is_rejected_before_writing(tmp_path):
    source, destination = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    image = np.zeros((12, 15, 3), dtype=np.uint8)
    save_image(source / "same.png", image)
    save_image(source / "same.jpg", image)
    with pytest.raises(ValueError, match="文件名重复"):
        img_rotate.rotate_images(source, destination, options=img_rotate.RotationOptions(output_format="png"), progress=False)
    assert not destination.exists()


def test_recursive_output_cannot_replace_another_input_image(tmp_path):
    source, destination = tmp_path / "input", tmp_path
    (source / "input").mkdir(parents=True)
    original = source / "flipped_sample.png"
    save_image(original, np.full((12, 15), 70, dtype=np.uint8))
    save_image(source / "input" / "sample.png", np.full((12, 15), 90, dtype=np.uint8))
    before = original.read_bytes()
    with pytest.raises(ValueError, match="覆盖输入图像"):
        img_rotate.rotate_images(
            source, destination, options=img_rotate.RotationOptions(recursive=True), progress=False
        )
    assert original.read_bytes() == before
    assert not (destination / "flipped_flipped_sample.png").exists()


@pytest.mark.parametrize("workers", [1, 3])
def test_cancellation_does_not_process_the_entire_batch(tmp_path, workers):
    source, destination = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    image = np.full((40, 55), 23, dtype=np.uint8)
    for index in range(30):
        save_image(source / f"{index:03d}.png", image)
    stop = threading.Event()

    def report(update):
        if update.completed:
            stop.set()

    previous_threads = cv2.getNumThreads()
    result = img_rotate.rotate_images(
        source, destination, workers, progress=False, stop_event=stop,
        progress_callback=report, progress_interval=0,
    )
    assert result.cancelled and 1 <= result.processed <= workers * 2
    assert result.succeeded == result.processed and not result.errors
    assert len(list(destination.glob("*.png"))) == result.succeeded
    assert cv2.getNumThreads() == previous_threads


def test_executor_queue_is_bounded(tmp_path, monkeypatch):
    source, destination = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    for index in range(60):
        (source / f"{index}.png").write_bytes(b"placeholder")
    real_executor = img_rotate.ThreadPoolExecutor
    executors = []

    class TrackingExecutor(real_executor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.counter_lock = threading.Lock()
            self.outstanding = self.maximum = 0
            executors.append(self)

        def submit(self, *args, **kwargs):
            with self.counter_lock:
                self.outstanding += 1
                self.maximum = max(self.maximum, self.outstanding)
            future = super().submit(*args, **kwargs)
            future.add_done_callback(self.finished)
            return future

        def finished(self, future):
            with self.counter_lock:
                self.outstanding -= 1

    monkeypatch.setattr(img_rotate, "ThreadPoolExecutor", TrackingExecutor)
    monkeypatch.setattr(img_rotate, "rotate_one", lambda *args: time.sleep(0.003))
    result = img_rotate.rotate_images(source, destination, 3, progress=False)
    assert result.succeeded == 60
    assert 3 <= executors[0].maximum <= 6


@pytest.mark.parametrize("kwargs", [
    {"rotation": 45}, {"png_compression": 10}, {"jpeg_quality": 0},
    {"prefix": "../"}, {"input_extensions": ("unknown",)}, {"output_format": "unknown"},
])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        img_rotate.RotationOptions(**kwargs)


def test_default_entrypoint_opens_gui_and_cli_still_works(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(img_rotate, "launch_gui", lambda: called.append(True) or 0)
    assert img_rotate.main([]) == 0 and called == [True]
    source, destination = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    image = np.arange(12 * 15, dtype=np.uint8).reshape(12, 15)
    save_image(source / "sample.png", image)
    assert img_rotate.main([
        "--input", str(source), "--output", str(destination), "--workers", "1",
        "--rotation", "180", "--format", "png", "--prefix", "done_",
    ]) == 0
    np.testing.assert_array_equal(load_image(destination / "done_sample.png"), np.rot90(image, 2))


def test_native_gui_processes_selected_parameters(tmp_path):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"Tk display unavailable: {error}")
    root.withdraw()
    try:
        app = img_rotate.RotationApp(root)
        source, destination = tmp_path / "输入", tmp_path / "输出"
        source.mkdir()
        image = np.arange(12 * 15 * 3, dtype=np.uint8).reshape(12, 15, 3)
        save_image(source / "sample.png", image)
        save_image(source / "ignored.jpg", image)
        app.input_var.set(str(source))
        app.output_var.set(str(destination))
        app.rotation_var.set("逆时针 90°")
        app.input_format_var.set("PNG (.png)")
        app.output_format_var.set("TIFF (.tif)")
        # TIFF 不使用 PNG/JPEG 参数，已禁用的输入不应影响处理。
        app.png_compression_var.set("unused")
        app.jpeg_quality_var.set("unused")
        app.grayscale_var.set(True)
        app.prefix_var.set("旋转_")
        app.start()
        assert app.running and str(app.output_format_combo.cget("state")) == "disabled"
        deadline = time.monotonic() + 10
        while app.running and time.monotonic() < deadline:
            root.update()
            time.sleep(0.01)
        assert not app.running, "GUI processing timed out"
        expected = np.rot90(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        np.testing.assert_array_equal(load_image(destination / "旋转_sample.tif"), expected)
        assert len(list(destination.iterdir())) == 1
        assert str(app.output_format_combo.cget("state")) == "readonly"
        assert str(app.png_spin.cget("state")) == "disabled"
        assert float(app.progress['value']) == float(app.progress['maximum']) == 1
    finally:
        root.destroy()
