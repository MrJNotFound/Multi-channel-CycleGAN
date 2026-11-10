from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from large_image_tool.image_ops import estimate_mark_orientation, merge_filtered_region, suppress_tool_marks, to_gray
from large_image_tool.models import MarkFilterSettings, ToolMarkMethod
from large_image_tool.raster import open_raster
from large_image_tool.stripe_analysis import detect_orientation
from large_image_tool.toolmarks import (
    analyze_tool_marks, estimate_period, preview_tool_marks,
    process_toolmark_input, remove_tool_marks,
)


def stripe_image(shape=(256, 384), angle=0.0, period=17.4):
    y, x = np.mgrid[:shape[0], :shape[1]]
    normal = x * np.cos(np.radians(angle)) + y * np.sin(np.radians(angle))
    plane = 128 + 34 * np.sin(2 * np.pi * normal / period)
    return np.repeat(np.rint(plane).astype(np.uint8)[..., None], 3, axis=2)


class ToolMarkRegressionTests(unittest.TestCase):
    @staticmethod
    def corner_stripes():
        rng = np.random.default_rng(19)
        y, x = np.mgrid[:512, :384]
        clean = 128 + rng.normal(0, 9, (512, 384))
        mask = (x < 150) & (y > 432)
        marked = np.rint(clean + 26 * np.sin(2 * np.pi * (x-y) / np.sqrt(2) / 17.4) * mask)
        return clean, marked.astype(np.uint8), mask

    def test_small_corner_stripes_detect_and_filter_in_all_four_corners(self):
        clean, source, mask = self.corner_stripes()
        for axes in ((), (0,), (1,), (0, 1)):
            with self.subTest(corner=axes):
                image = np.flip(source, axis=axes).copy()
                reference = np.flip(clean, axis=axes)
                marked = np.flip(mask, axis=axes)
                filtered, report = suppress_tool_marks(image, MarkFilterSettings(strength=0.85))
                expected_angle = 45 if len(axes) == 1 else -45
                self.assertAlmostEqual(report.normal_angle_deg, expected_angle, delta=1)
                self.assertTrue(report.applied)
                self.assertIsNotNone(report.processing_region)
                x0, y0, x1, y1 = report.processing_region
                outside = np.ones(image.shape, dtype=bool)
                outside[y0:y1, x0:x1] = False
                np.testing.assert_array_equal(filtered[outside], image[outside])
                before = float(np.mean((image[marked] - reference[marked]) ** 2))
                after = float(np.mean((filtered[marked] - reference[marked]) ** 2))
                self.assertLess(after, before * 0.6)
                self.assertLess(min(abs(f-1/17.4) for f in report.peak_frequencies), 0.004)

    def test_localized_sample_all_methods_preserve_unselected_pixels(self):
        path = Path(__file__).resolve().parents[1] / 'sample_data' / 'processed_0443.png'
        if not path.exists():
            self.skipTest('Optional microscopy sample is unavailable')
        with Image.open(path) as image:
            source = np.asarray(image.convert('RGB'))
        for method in ToolMarkMethod:
            with self.subTest(method=method):
                filtered, report = remove_tool_marks(source, MarkFilterSettings(method=method.value))
                self.assertTrue(report.applied)
                self.assertAlmostEqual(report.orientation_deg, -45, delta=2)
                self.assertIsNotNone(report.processing_region)
                x0, y0, x1, y1 = report.processing_region
                self.assertGreater(y0, source.shape[0] * 0.4)
                self.assertEqual(x0, 0)
                outside = np.ones(source.shape[:2], dtype=bool)
                outside[y0:y1, x0:x1] = False
                np.testing.assert_array_equal(filtered[outside], source[outside])
                self.assertGreater(float(np.abs(filtered.astype(float)-source).max()), 5)

    def test_local_preview_region_is_reported_in_original_pixels(self):
        _, gray, _ = self.corner_stripes()
        source = np.repeat(gray[..., None], 3, axis=2)
        enlarged = np.asarray(Image.fromarray(source).resize((768, 1024), Image.Resampling.BICUBIC))
        _, expected = remove_tool_marks(source, MarkFilterSettings())
        _, report = preview_tool_marks(source, MarkFilterSettings(), (0.5, 0.5))
        self.assertEqual(report.processing_region, tuple(2 * value for value in expected.processing_region))
        full = detect_orientation(to_gray(enlarged))
        self.assertAlmostEqual(full.angle_deg, report.orientation_deg, delta=1)

    def test_local_export_does_not_expand_correction_to_entire_block(self):
        _, gray, _ = self.corner_stripes()
        source = np.repeat(gray[..., None], 3, axis=2)
        cfg = MarkFilterSettings(block_size=1024)
        expected, report = remove_tool_marks(source, cfg)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            Image.fromarray(source).save(root / 'source.png')
            output = process_toolmark_input(root / 'source.png', root / 'filtered.ome.tif', cfg)
            with open_raster(output.output_path) as raster:
                actual = raster.read_region(0, 0, source.shape[1], source.shape[0])
            np.testing.assert_array_equal(actual, expected)
            metadata = json.loads(output.settings_path.read_text(encoding='utf-8'))
            saved_region = metadata['analysis']['block_0000_0000']['processing_region']
            self.assertEqual(saved_region, list(report.processing_region))

    def test_multiblock_export_preserves_whole_image_detection_scope(self):
        path = Path(__file__).resolve().parents[1] / 'sample_data' / 'processed_0443.png'
        if not path.exists():
            self.skipTest('Optional microscopy sample is unavailable')
        with Image.open(path) as image:
            source = np.asarray(image.convert('RGB'))
        scope = detect_orientation(to_gray(source)).region
        self.assertIsNotNone(scope)
        x0, y0, x1, y1 = scope
        outside = np.ones(source.shape[:2], dtype=bool)
        outside[y0:y1, x0:x1] = False
        with tempfile.TemporaryDirectory() as raw:
            output = process_toolmark_input(path, Path(raw) / 'result.ome.tif', MarkFilterSettings(block_size=1024))
            with open_raster(output.output_path) as raster:
                filtered = raster.read_region(0, 0, source.shape[1], source.shape[0])
            np.testing.assert_array_equal(filtered[outside], source[outside])
            self.assertGreater(float(np.abs(filtered.astype(float)-source).max()), 5)
            metadata = json.loads(output.settings_path.read_text(encoding='utf-8'))
            self.assertGreater(len(metadata['analysis']), 1)
            for report in metadata['analysis'].values():
                if report['applied']:
                    ax0, ay0, ax1, ay1 = report['processing_region']
                    self.assertTrue(x0 <= ax0 < ax1 <= x1)
                    self.assertTrue(y0 <= ay0 < ay1 <= y1)

    def test_one_pixel_region_at_export_boundary_blends_safely(self):
        source = np.full((8, 8, 3), 128, dtype=np.uint8)
        filtered = merge_filtered_region(source, np.zeros((4, 1, 3), dtype=np.uint8), (3, 4, 4, 8))
        self.assertEqual(filtered.shape, source.shape)
        np.testing.assert_array_equal(filtered[:, :3], source[:, :3])
        np.testing.assert_array_equal(filtered[:, 4:], source[:, 4:])

    def test_oblique_directions_and_fractional_period(self):
        for angle in (-89, -55, -21, 0, 45, 89):
            with self.subTest(angle=angle):
                source = stripe_image(angle=angle)
                direction, confidence, period, _ = analyze_tool_marks(source, MarkFilterSettings())
                self.assertLess(abs((direction - angle + 90) % 180 - 90), 0.5)
                self.assertGreater(confidence, 0.6)
                self.assertAlmostEqual(period, 17.4, delta=0.3)

    def test_sample_direction_ignores_dominant_tissue_edges(self):
        path = Path(__file__).resolve().parents[1] / 'sample_data' / 'processed_0189.png'
        if not path.exists():
            self.skipTest('Optional microscopy sample is unavailable')
        with Image.open(path) as image:
            source = np.asarray(image.convert('RGB'))
        direction, confidence = estimate_mark_orientation(to_gray(source))
        # Visually measured long scratches slope down-right at about 32 degrees.
        self.assertGreater(direction, -62)
        self.assertLess(direction, -53)
        self.assertGreater(confidence, 0.4)

    def test_analysis_accepts_readonly_float_without_mutation(self):
        gray = to_gray(stripe_image(angle=27))
        original = gray.copy()
        gray.flags.writeable = False
        estimate_mark_orientation(gray)
        estimate_period(gray, 27)
        suppress_tool_marks(gray, MarkFilterSettings())
        np.testing.assert_array_equal(gray, original)

    def test_large_image_period_and_fft_use_original_pixels(self):
        source = stripe_image(shape=(192, 2304), period=37.25)
        period, _ = estimate_period(to_gray(source), 0)
        self.assertAlmostEqual(period, 37.25, delta=0.5)
        filtered, report = suppress_tool_marks(source, MarkFilterSettings(strength=0.9))
        self.assertTrue(report.applied)
        self.assertLess(float(np.std(filtered[24:-24, 50:-50].astype(float))), float(source.std()) * 0.35)
        self.assertLess(min(abs(f - 1 / 37.25) for f in report.peak_frequencies), 0.002)

    def test_tiny_strength_does_not_resample_the_original(self):
        rng = np.random.default_rng(7)
        source = rng.integers(20, 230, (97, 142, 3), dtype=np.uint8)
        for method in ToolMarkMethod:
            with self.subTest(method=method):
                cfg = MarkFilterSettings(method=method.value, strength=1e-7,
                    auto_orientation=False, orientation_deg=31.3,
                    auto_period=False, period_px=17.4)
                result, _ = remove_tool_marks(source, cfg)
                np.testing.assert_array_equal(result, source)

    def test_spatial_filters_remove_fractional_and_irregular_marks(self):
        y, x = np.mgrid[:256, :384]
        angle = 27.3
        normal = x * np.cos(np.radians(angle)) + y * np.sin(np.radians(angle))
        clean = 128 + 12 * np.sin(x / 49) + 8 * np.cos(y / 55)
        # Preserve a small dark feature, which direct median filtering smears.
        clean -= 55 * np.exp(-((x-175)**2 + (y-140)**2) / 12)
        scratches = sum(-48 * np.exp(-0.5 * ((normal-center)/1.7)**2) for center in (45, 91, 143, 221, 279, 342))
        for marks in (30*np.sin(2*np.pi*normal/17.4), scratches):
            source = np.repeat(np.rint(clean + marks).astype(np.uint8)[..., None], 3, axis=2)
            before = float(np.mean((to_gray(source)-clean)**2))
            for method in (ToolMarkMethod.PERIODIC_TEMPLATE, ToolMarkMethod.DIRECTIONAL_1D):
                with self.subTest(periodic=marks is not scratches, method=method):
                    result, _ = remove_tool_marks(source, MarkFilterSettings(
                        method=method.value, strength=0.9, auto_orientation=False,
                        orientation_deg=angle, auto_period=False, period_px=50 if marks is scratches else 17.4,
                    ))
                    error = float(np.mean((to_gray(result)[16:-16,16:-16]-clean[16:-16,16:-16])**2))
                    self.assertLess(error, before * 0.5)
                    # Contrast along the same stripe distinguishes preservation
                    # of the spot from mere residual stripe brightness.
                    contrast = float(to_gray(result)[151,170] - to_gray(result)[140,175])
                    self.assertLess(abs(contrast - (clean[151,170] - clean[140,175])), 15)

    def test_preview_manual_settings_and_report_use_original_pixels(self):
        source = stripe_image(shape=(256, 512), period=32)
        preview = np.asarray(Image.fromarray(source).resize((256, 128), Image.Resampling.LANCZOS))
        cfg = MarkFilterSettings(strength=0.9, auto_orientation=False, auto_period=False, period_px=32)
        result, report = preview_tool_marks(preview, cfg, (0.5, 0.5))
        self.assertEqual(report.period_px, 32)
        self.assertEqual(cfg.period_px, 32)
        self.assertLess(float(result.std()), float(preview.std()) * 0.35)
        _, auto_report = preview_tool_marks(preview, MarkFilterSettings(), (0.5, 0.5))
        self.assertAlmostEqual(auto_report.period_px, 32, delta=1)

    def test_fft_export_keeps_automatic_frequency_detection(self):
        source = stripe_image(shape=(128, 192), period=17.4)
        cfg = MarkFilterSettings(auto_orientation=False, strength=0.85, block_size=256)
        expected, _ = remove_tool_marks(source, cfg)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            Image.fromarray(source).save(root / 'source.png')
            output = process_toolmark_input(root / 'source.png', root / 'filtered.ome.tif', cfg)
            with open_raster(output.output_path) as raster:
                actual = raster.read_region(0, 0, 192, 128)
            np.testing.assert_array_equal(actual, expected)

    def test_no_direction_evidence_leaves_image_untouched(self):
        rng = np.random.default_rng(2)
        sources = [np.full((128, 192, 3), 128, dtype=np.uint8)]
        sources += [rng.integers(100,150,(256,384,3),dtype=np.uint8) for _ in range(5)]
        for source in sources:
            for method in ToolMarkMethod:
                result, report = remove_tool_marks(source, MarkFilterSettings(method=method.value))
                self.assertFalse(report.applied)
                np.testing.assert_array_equal(result, source)


if __name__ == '__main__':
    unittest.main()
