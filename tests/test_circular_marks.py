import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")
from tool.remove_circular_marks import RingSettings, remove_circular_marks, read_image, main


def synthetic():
    h, w = 360, 540
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    center = (286.0, 171.0)
    radius = np.hypot(xx-center[0], yy-center[1])
    rng = np.random.default_rng(53)
    texture = cv2.GaussianBlur(rng.normal(0, 3, (h, w)).astype(np.float32), (0, 0), .8)
    base = 158 + .015*xx + texture
    clean = np.stack([base-3, base, base+4], axis=2)
    structures = np.zeros((h, w), np.uint8)
    for x in range(90, 470, 90):
        cv2.line(clean, (x, 60), (x+25, 300), (235, 238, 240), 5)
        cv2.line(structures, (x, 60), (x+25, 300), 255, 5)
    cv2.circle(clean, (410, 130), 18, (85, 75, 155), -1)
    cv2.circle(structures, (410, 130), 18, 255, -1)
    rings = 8*np.sin(2*np.pi*radius/9) + 3*np.sin(2*np.pi*radius/4)
    rings[structures > 0] = 0
    noisy = np.clip(np.rint(clean+rings[..., None]), 0, 255).astype(np.uint8)
    return np.rint(clean).astype(np.uint8), noisy, center, structures, radius


def test_auto_center_removes_rings_and_preserves_protected_structures():
    clean, noisy, center, structures, radius = synthetic()
    original = noisy.copy()
    filtered, report, delta = remove_circular_marks(noisy)
    assert report["applied"]
    assert report["center_estimation"]["reliable"]
    assert np.linalg.norm(np.array(report["center_estimation"]["center_xy"])-center) < 5
    background = (cv2.dilate(structures, np.ones((11,11), np.uint8)) == 0) & (radius > 70)
    before = np.mean((noisy.astype(float)-clean)[background]**2)
    after = np.mean((filtered.astype(float)-clean)[background]**2)
    assert after < .4*before
    np.testing.assert_array_equal(filtered[structures > 0], noisy[structures > 0])
    np.testing.assert_array_equal(noisy, original)
    assert report["protected_pixels_changed"] == 0
    assert np.max(np.abs(delta)) <= 20


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_alpha_color_differences_and_manual_mask_are_preserved(dtype):
    _, noisy, center, _, _ = synthetic()
    factor = 257 if dtype == np.uint16 else 1
    source = np.dstack([noisy.astype(dtype)*factor, np.full(noisy.shape[:2], 255*factor, dtype)])
    source[:15, :, 3] = 0
    source[15:30, :, 3] = 100*factor
    mask = np.zeros(source.shape[:2], np.uint8)
    mask[220:260, 300:400] = 255
    filtered, report, _ = remove_circular_marks(source, center=center, protect_mask=mask)
    assert filtered.dtype == dtype
    assert report["applied"]
    np.testing.assert_array_equal(filtered[:30], source[:30])
    np.testing.assert_array_equal(filtered[mask > 0], source[mask > 0])
    np.testing.assert_array_equal(filtered[..., 3], source[..., 3])
    np.testing.assert_array_equal(filtered[..., 0].astype(int)-filtered[..., 1],
                                  source[..., 0].astype(int)-source[..., 1])
    np.testing.assert_array_equal(filtered[..., 2].astype(int)-filtered[..., 1],
                                  source[..., 2].astype(int)-source[..., 1])


@pytest.mark.parametrize("noise", [False, True])
def test_no_reliable_circle_is_left_unchanged(noise):
    rng = np.random.default_rng(4)
    image = (np.clip(160+rng.normal(0, 2, (300, 450)), 0, 255).astype(np.uint8)
             if noise else np.full((300, 450), 160, np.uint8))
    filtered, report, delta = remove_circular_marks(image)
    np.testing.assert_array_equal(filtered, image)
    assert not report["applied"]
    assert not np.any(delta)


def test_zero_strength_and_invalid_parameters():
    _, image, center, _, _ = synthetic()
    filtered, report, _ = remove_circular_marks(image, RingSettings(strength=0), center)
    np.testing.assert_array_equal(filtered, image)
    assert not report["applied"]
    with pytest.raises(ValueError):
        remove_circular_marks(image, RingSettings(strength=float("nan")))
    with pytest.raises(ValueError):
        remove_circular_marks(image, center=(float("nan"), 0))
    with pytest.raises(ValueError):
        remove_circular_marks(image, protect_mask=np.zeros((30, 30)))


def test_cli_writes_lossless_image_report_and_comparison_without_overwrite(tmp_path):
    import json
    _, noisy, center, _, _ = synthetic()
    source, output = tmp_path / "原图.png", tmp_path / "结果.png"
    ok, encoded = cv2.imencode(".png", noisy)
    assert ok
    encoded.tofile(source)
    args = [str(source), str(output), "--center", *map(str, center)]
    assert main(args) == 0
    report = json.loads(output.with_suffix(".report.json").read_text(encoding="utf-8"))
    assert report["processed_size_wh"] == [540, 360]
    assert not report["preview"]
    assert report["protected_pixels_changed"] == 0
    assert output.with_suffix(".comparison.png").is_file()
    assert output.with_suffix(".removed.png").is_file()
    saved = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 1
    assert output.read_bytes() == saved


def test_preview_is_explicit_and_records_original_dimensions(tmp_path, monkeypatch):
    from tool import remove_circular_marks as rings
    _, source, _, _, _ = synthetic()
    path = tmp_path / "large.jpg"
    ok, data = cv2.imencode(".jpg", source)
    assert ok
    data.tofile(path)
    monkeypatch.setattr(rings, "MAX_PIXELS", 100_000)
    with pytest.raises(ValueError, match="preview-max-side"):
        read_image(path)
    preview, info = read_image(path, preview_max_side=180)
    assert preview.shape[:2] == (120, 180)
    assert info == {"source_size_wh": [540, 360], "preview": True,
                    "processed_size_wh": [180, 120]}
