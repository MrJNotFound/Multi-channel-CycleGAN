from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from large_image_tool.crop import (
    CropSettings,
    export_patches,
    load_crop_settings,
    patch_layout,
    save_crop_settings,
)
from large_image_tool.file_index import discover_numeric_tiles
from large_image_tool.geometry import mosaic_bounds, regular_positions, step_vectors
from large_image_tool.image_ops import suppress_tool_marks, to_gray
from large_image_tool.models import (
    AutoSearchSettings,
    FastAxis,
    GridGeometry,
    MarkFilterSettings,
    ScanPattern,
    StartCorner,
    StitchProject,
    ToolMarkMethod,
)
from large_image_tool.mosaic import render_mosaic, render_mosaic_thumbnail
from large_image_tool.preview import compose_local_preview
from large_image_tool.raster import open_raster
from large_image_tool.registration import AutoStitcher, PairwiseShift, phase_correlation_candidates
from large_image_tool.synthetic import generate_turning_dataset
from large_image_tool.toolmarks import (
    linked_source_folder,
    load_toolmark_settings,
    process_toolmark_input,
    remove_tool_marks,
    save_toolmark_settings,
)


class ScanPatternTests(unittest.TestCase):
    def test_every_pattern_maps_each_slot_once(self) -> None:
        expected = {(row, col) for row in range(3) for col in range(4)}
        for corner in StartCorner:
            for axis in FastAxis:
                for snake in (False, True):
                    pattern = ScanPattern(corner, axis, snake)
                    actual = {pattern.slot_to_rc(slot, 3, 4) for slot in range(12)}
                    self.assertEqual(actual, expected)

    def test_known_snake_sequence(self) -> None:
        pattern = ScanPattern(StartCorner.TOP_LEFT, FastAxis.HORIZONTAL, True)
        actual = [pattern.slot_to_rc(slot, 2, 3) for slot in range(6)]
        self.assertEqual(actual, [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)])


class FileIndexTests(unittest.TestCase):
    def test_numeric_sort_and_missing_first_are_unambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for number in (2, 3, 5):
                Image.new("RGB", (24, 16), (number, 0, 0)).save(root / f"{number:05d}.png")
            tile_set = discover_numeric_tiles(root, 2, 3, ScanPattern())
            self.assertEqual([record.number for record in tile_set.records], [2, 3, 5])
            self.assertEqual(tile_set.missing_slots, [0, 3, 5])
            self.assertEqual([(record.number, record.slot) for record in tile_set.records], [(2, 1), (3, 2), (5, 4)])


class GeometryTests(unittest.TestCase):
    def test_step_vectors(self) -> None:
        horizontal, vertical = step_vectors(GridGeometry(10, 20, 7), 100, 50)
        np.testing.assert_allclose(horizontal, [90, 0], atol=1e-8)
        np.testing.assert_allclose(vertical, [0, 40], atol=1e-8)

    def test_missing_outer_tile_still_reserves_canvas_cell(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for number in (1, 2, 3):
                Image.new("RGB", (100, 50)).save(root / f"{number:05d}.png")
            tile_set = discover_numeric_tiles(root, 2, 2, ScanPattern())
            positions = regular_positions(tile_set, GridGeometry(0, 0, 0))
            self.assertEqual(set(positions), {(0, 0), (0, 1), (1, 0), (1, 1)})
            self.assertEqual(mosaic_bounds(positions, 100, 50), (0, 0, 200, 100))


class SettingsTests(unittest.TestCase):
    def test_v1_project_migrates_to_single_tile_rotation(self) -> None:
        project = StitchProject.from_dict(
            {
                "schema_version": 1,
                "source_folder": "/tmp/tiles",
                "rows": 2,
                "cols": 3,
                "geometry": {
                    "overlap_x_percent": 11,
                    "overlap_y_percent": 13,
                    "angle_deg": 2.25,
                    "orthogonality_error_deg": 0.4,
                },
                "mark_filter": {"enabled": True},
                "refined_positions": {"0,1": [90, 3]},
            }
        )
        self.assertEqual(project.schema_version, 4)
        self.assertEqual((project.tile_width, project.tile_height), (0, 0))
        self.assertEqual(project.overlap_blend_percent, 50.0)
        self.assertEqual(project.stitch_mode, "automatic")
        self.assertAlmostEqual(project.geometry.tile_rotation_deg, -2.25)
        self.assertFalse(hasattr(project, "mark_filter"))
        self.assertEqual(project.refined_positions, {})

    def test_settings_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            mark_settings = MarkFilterSettings(
                method=ToolMarkMethod.PERIODIC_TEMPLATE.value,
                strength=0.62,
                auto_period=False,
                period_px=21.5,
            )
            mark_path = save_toolmark_settings(root / "marks.toolmarks.json", mark_settings)
            self.assertEqual(load_toolmark_settings(mark_path), mark_settings)
            crop_settings = CropSettings(mode="count", rows=7, cols=9, angle_deg=1.2)
            crop_path = save_crop_settings(root / "crop.crop.json", crop_settings)
            self.assertEqual(load_crop_settings(crop_path), crop_settings)

    def test_auto_search_validation(self) -> None:
        AutoSearchSettings(0, 30, -8, 8, 24, 1024).validate()
        with self.assertRaises(ValueError):
            AutoSearchSettings(20, 10).validate()


class RegistrationTests(unittest.TestCase):
    def test_phase_correlation_recovers_partial_overlap(self) -> None:
        rng = np.random.default_rng(4)
        canvas = rng.normal(size=(260, 500)).astype(np.float32)
        reference = canvas[20:220, 20:260]
        moving = canvas[25:225, 190:430]
        candidates = phase_correlation_candidates(reference, moving, "horizontal", (0, 30))
        self.assertTrue(candidates)
        self.assertAlmostEqual(candidates[0].dx, 170, delta=1)
        self.assertAlmostEqual(candidates[0].dy, 5, delta=1)
        self.assertGreater(candidates[0].ncc, 0.9)

    def test_synthetic_grid_auto_registration(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            truth = GridGeometry(14.0, 12.0, 1.4)
            generate_turning_dataset(root, 3, 4, (320, 200), truth, jitter_px=0.8, seed=5)
            tile_set = discover_numeric_tiles(root, 3, 4, ScanPattern())
            progress_values: list[int] = []
            result = AutoStitcher(tile_set, max_registration_size=320).run(
                progress=lambda current, total, _message: progress_values.append(
                    round(100 * current / total)
                )
            )
            self.assertEqual(progress_values, sorted(progress_values))
            self.assertEqual(progress_values[-1], 100)
            self.assertNotIn(100, progress_values[:-1])
            self.assertAlmostEqual(result.geometry.overlap_x_percent, truth.overlap_x_percent, delta=0.8)
            self.assertAlmostEqual(result.geometry.overlap_y_percent, truth.overlap_y_percent, delta=0.8)
            self.assertAlmostEqual(result.geometry.tile_rotation_deg, truth.tile_rotation_deg, delta=0.5)
            np.testing.assert_allclose(result.positions[(0, 1)][1], result.positions[(0, 0)][1], atol=1e-8)
            np.testing.assert_allclose(result.positions[(1, 0)][0], result.positions[(0, 0)][0], atol=1e-8)

    def test_verification_pass_obeys_pair_sampling_budget(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for number in range(1, 37):
                Image.new("RGB", (40, 24), (number, 80, 120)).save(root / f"{number:06d}.png")
            tile_set = discover_numeric_tiles(root, 6, 6, ScanPattern())
            stitcher = AutoStitcher(tile_set, sample_pairs_per_direction=3)

            def fake_register(source, target, _direction, prior):
                self.assertIsNotNone(prior)
                dx = (target.col - source.col) * 36.0
                dy = (target.row - source.row) * 21.6
                return PairwiseShift(
                    (source.row, source.col),
                    (target.row, target.col),
                    dx,
                    dy,
                    1.0,
                    1.0,
                )

            with patch.object(stitcher, "_register_records", side_effect=fake_register) as register:
                result = stitcher.refine_all(GridGeometry(10.0, 10.0, 0.0))
            self.assertEqual(register.call_count, 6)
            self.assertEqual(len(result.edges), 6)

    def test_rotated_mosaic_is_readable_as_tiled_ome(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            geometry = GridGeometry(14.0, 12.0, 2.0)
            generate_turning_dataset(root / "tiles", 2, 3, (160, 100), geometry, jitter_px=0, seed=9)
            tile_set = discover_numeric_tiles(root / "tiles", 2, 3, ScanPattern())
            output, preview = render_mosaic(
                tile_set,
                regular_positions(tile_set, geometry),
                root / "mosaic.ome.tif",
                geometry.tile_rotation_deg,
                chunk_size=128,
            )
            self.assertTrue(output.is_file())
            self.assertTrue(preview.is_file())
            with open_raster(output) as raster:
                self.assertGreater(raster.width, 400)
                self.assertGreater(raster.height, 170)
                self.assertEqual(raster.read_region(10, 10, 40, 30).shape, (30, 40, 3))

    def test_mosaic_thumbnail_is_rendered_without_full_mosaic(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            geometry = GridGeometry(14.0, 12.0, 2.0)
            generate_turning_dataset(
                root / "tiles", 2, 3, (160, 100), geometry, jitter_px=0, seed=31
            )
            tile_set = discover_numeric_tiles(root / "tiles", 2, 3, ScanPattern())
            progress: list[tuple[int, int, str]] = []
            output = render_mosaic_thumbnail(
                tile_set,
                regular_positions(tile_set, geometry),
                root / "thumbnail.png",
                geometry.tile_rotation_deg,
                maximum_edge=256,
                progress=lambda current, total, message: progress.append(
                    (current, total, message)
                ),
            )
            self.assertTrue(output.is_file())
            with Image.open(output) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertLessEqual(max(image.size), 256)
                self.assertEqual(max(image.size), 256)
            self.assertEqual(progress[-1], (tile_set.count, tile_set.count, "生成拼接缩略图"))
            self.assertFalse((root / "thumbnail.ome.tif").exists())


class BlendTests(unittest.TestCase):
    def test_overlap_choice_selects_earlier_or_later_tile(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            Image.new("RGB", (100, 60), (240, 20, 20)).save(root / "000001.png")
            Image.new("RGB", (100, 60), (20, 20, 240)).save(root / "000002.png")
            tile_set = discover_numeric_tiles(root, 1, 2, ScanPattern(snake=False))
            geometry = GridGeometry(30.0, 0.0, 0.0)
            earlier = np.asarray(compose_local_preview(
                tile_set, geometry, 0, 2, 100, show_labels=False, overlap_blend_percent=0,
            ))
            later = np.asarray(compose_local_preview(
                tile_set, geometry, 0, 2, 100, show_labels=False, overlap_blend_percent=100,
            ))
            np.testing.assert_array_equal(earlier[30, 85], [240, 20, 20])
            np.testing.assert_array_equal(later[30, 85], [20, 20, 240])

    def test_overlap_choice_is_applied_to_exported_mosaic(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            Image.new("RGB", (100, 60), (240, 20, 20)).save(root / "000001.png")
            Image.new("RGB", (100, 60), (20, 20, 240)).save(root / "000002.png")
            tile_set = discover_numeric_tiles(root, 1, 2, ScanPattern(snake=False))
            geometry = GridGeometry(30.0, 0.0, 0.0)
            positions = regular_positions(tile_set, geometry)
            earlier_path, _ = render_mosaic(
                tile_set, positions, root / "earlier.ome.tif", 0.0, 0.0, chunk_size=128,
            )
            later_path, _ = render_mosaic(
                tile_set, positions, root / "later.ome.tif", 0.0, 100.0, chunk_size=128,
            )
            with open_raster(earlier_path) as earlier, open_raster(later_path) as later:
                # x=72 is close to the later tile's left edge; 100% must still
                # select it completely rather than retain an old feather blend.
                np.testing.assert_array_equal(earlier.read_region(72, 30, 1, 1)[0, 0], [240, 20, 20])
                np.testing.assert_array_equal(later.read_region(72, 30, 1, 1)[0, 0], [20, 20, 240])

    def test_rotated_empty_corner_does_not_cover_valid_tile(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first_color = np.array([238, 32, 28], dtype=np.uint8)
            Image.new("RGB", (100, 60), tuple(first_color)).save(root / "000001.png")
            Image.new("RGB", (100, 60), (25, 45, 235)).save(root / "000002.png")
            tile_set = discover_numeric_tiles(root, 1, 2, ScanPattern(snake=False))
            geometry = GridGeometry(30.0, 0.0, 10.0)
            preview = np.asarray(compose_local_preview(
                tile_set, geometry, 0, 2, 100, show_labels=False, overlap_blend_percent=100,
            ))

            rotated_mask = np.asarray(
                Image.new("L", (100, 60), 255).rotate(
                    -10.0, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=0,
                )
            ) > 0
            mask_h, mask_w = rotated_mask.shape
            first = np.zeros(preview.shape[:2], dtype=bool)
            second = np.zeros(preview.shape[:2], dtype=bool)
            first[:mask_h, :mask_w] = rotated_mask
            second[:mask_h, 70 : 70 + mask_w] = rotated_mask
            exposed_first = first & ~second
            bounding_box = np.zeros_like(exposed_first)
            bounding_box[:mask_h, 70 : 70 + mask_w] = True
            pixels = preview[exposed_first & bounding_box]
            self.assertGreater(len(pixels), 0)
            self.assertTrue(np.all(pixels == first_color))


class FilterTests(unittest.TestCase):
    def test_soft_notch_reduces_periodic_mark(self) -> None:
        height, width, period = 256, 384, 16
        _y, x = np.mgrid[:height, :width]
        rng = np.random.default_rng(2)
        base = 120 + 15 * np.sin(2 * np.pi * x / 51) + rng.normal(0, 3, (height, width))
        marked = np.clip(base + 42 * np.sin(2 * np.pi * x / period), 0, 255).astype(np.uint8)
        source = np.repeat(marked[..., None], 3, axis=2)
        filtered, report = suppress_tool_marks(source, MarkFilterSettings(enabled=True, strength=0.85))

        def amplitude(image: np.ndarray) -> float:
            profile = to_gray(image).mean(axis=0)
            return float(abs(np.fft.rfft(profile - profile.mean())[round(width / period)]))

        self.assertTrue(report.applied)
        self.assertLess(amplitude(filtered), amplitude(source) * 0.35)

    def test_all_three_toolmark_methods_reduce_known_period(self) -> None:
        height, width, period = 180, 300, 15
        _y, x = np.mgrid[:height, :width]
        texture = 118 + 8 * np.sin(2 * np.pi * x / 67) + 38 * np.sin(2 * np.pi * x / period)
        source = np.repeat(np.clip(texture, 0, 255).astype(np.uint8)[..., None], 3, axis=2)

        def amplitude(image: np.ndarray) -> float:
            profile = to_gray(image).mean(axis=0)
            return float(abs(np.fft.rfft(profile - profile.mean())[round(width / period)]))

        before = amplitude(source)
        for method in ToolMarkMethod:
            settings = MarkFilterSettings(
                method=method.value,
                strength=0.9,
                auto_orientation=False,
                orientation_deg=0.0,
                auto_period=False,
                period_px=period,
                directional_kernel=15,
            )
            filtered, report = remove_tool_marks(source, settings)
            self.assertTrue(report.applied, method.value)
            self.assertLess(amplitude(filtered), before * 0.75, method.value)

    def test_folder_processing_preserves_names_and_links_originals(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "raw_tiles"
            source.mkdir()
            for number in (1, 2, 10):
                x = np.arange(96)[None, :]
                plane = np.broadcast_to(120 + 30 * np.sin(2 * np.pi * x / 12), (64, 96))
                image = np.repeat(np.clip(plane, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
                Image.fromarray(image, mode="RGB").save(source / f"{number:06d}.png")
            output = root / "filtered_tiles"
            result = process_toolmark_input(
                source,
                output,
                MarkFilterSettings(auto_orientation=False, auto_period=False, period_px=12),
            )
            self.assertEqual(result.processed_count, 3)
            self.assertEqual(
                sorted(path.name for path in output.glob("*.png")),
                ["000001.png", "000002.png", "000010.png"],
            )
            self.assertFalse(list(output.glob(".*.writing.*")))
            self.assertEqual(linked_source_folder(output), source.resolve())

    def test_large_raster_processing_writes_ome_preview_and_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            x = np.arange(330)[None, :]
            plane = np.broadcast_to(115 + 32 * np.sin(2 * np.pi * x / 15), (210, 330))
            source = root / "large.png"
            Image.fromarray(
                np.repeat(np.clip(plane, 0, 255).astype(np.uint8)[..., None], 3, axis=2),
                mode="RGB",
            ).save(source)
            result = process_toolmark_input(
                source,
                root / "large_marks_removed.ome.tif",
                MarkFilterSettings(
                    auto_orientation=False,
                    auto_period=False,
                    period_px=15,
                    block_size=128,
                    block_halo=32,
                ),
            )
            self.assertTrue(result.output_path.is_file())
            self.assertTrue(result.preview_path and result.preview_path.is_file())
            self.assertTrue(result.settings_path.is_file())
            self.assertFalse((root / "large_marks_removed.ome.tif.working.uint8").exists())
            self.assertFalse((root / "large_marks_removed.ome.tif.working.weights.float16").exists())
            with open_raster(result.output_path) as raster:
                self.assertEqual((raster.width, raster.height), (330, 210))
                self.assertEqual(raster.read_region(5, 7, 20, 30).shape, (30, 20, 3))


class CropTests(unittest.TestCase):
    def test_count_layout_and_export_manifest(self) -> None:
        settings = CropSettings(mode="count", rows=2, cols=3, overlap_x_percent=10, overlap_y_percent=20)
        specs = patch_layout(300, 160, settings)
        self.assertEqual(len(specs), 6)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.png"
            output = root / "patches"
            Image.new("RGB", (300, 160), (40, 80, 120)).save(source)
            manifest, count = export_patches(source, output, settings)
            self.assertEqual(count, 6)
            self.assertTrue(manifest.exists())
            self.assertEqual(len(list(output.glob("patch_*.png"))), 6)
            settings_payload = json.loads((output / "split_settings.json").read_text(encoding="utf-8"))
            self.assertEqual(settings_payload["written_count"], 6)
            self.assertEqual(settings_payload["kind"], "large_image_split")


if __name__ == "__main__":
    unittest.main()
