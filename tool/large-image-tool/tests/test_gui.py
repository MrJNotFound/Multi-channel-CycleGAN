from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication

from large_image_tool.app import MainWindow, _read_toolmark_preview, create_application
from large_image_tool.models import GridGeometry, ToolMarkMethod
from large_image_tool.synthetic import generate_turning_dataset


class GuiInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or create_application([])

    def wait_for(self, predicate, timeout: float = 8.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            time.sleep(0.01)
        self.fail("GUI operation timed out")

    def test_stitch_controls_update_preview_and_auto_progress_resets(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            truth = GridGeometry(14.0, 12.0, 1.4)
            generate_turning_dataset(root, 3, 4, (240, 140), truth, jitter_px=0.4, seed=11)
            window = MainWindow()
            stitch = window.stitch_tab
            stitch.source_folder = root
            stitch.folder_edit.setText(str(root))
            stitch.rows_spin.setValue(3)
            stitch.cols_spin.setValue(4)
            stitch.reindex()
            self.wait_for(lambda: not stitch._preview_running and not stitch._preview_pending)
            initial_width = stitch.preview.sceneRect().width()

            stitch.overlap_x.setValue(20.0)
            self.wait_for(lambda: not stitch._preview_running and not stitch._preview_pending)
            self.assertNotEqual(stitch.preview.sceneRect().width(), initial_width)

            stitch.preview_group.button(2).click()
            self.wait_for(lambda: not stitch._preview_running and not stitch._preview_pending)
            self.assertEqual(stitch.preview_group.checkedId(), 2)
            self.assertTrue(stitch.preview_status.text().startswith("2×2"))

            stitch.run_auto()
            self.wait_for(
                lambda: not stitch._busy
                and not stitch._preview_running
                and not stitch._preview_pending
                and len(stitch._active_workers) == 0,
                timeout=15.0,
            )
            self.assertEqual(stitch.progress.value(), 0)
            self.assertAlmostEqual(stitch.overlap_x.value(), truth.overlap_x_percent, delta=1.0)
            self.assertEqual(len(stitch._active_workers), 0)
            window.close()

    def test_saved_scheme_drives_filtered_folder_mosaic(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            source = workspace / "raw_tiles"
            filtered = workspace / "filtered_tiles"
            truth = GridGeometry(12.0, 10.0, 1.0)
            generate_turning_dataset(source, 2, 3, (120, 80), truth, jitter_px=0, seed=23)
            shutil.copytree(source, filtered)

            window = MainWindow()
            self.assertEqual(window.tabs.tabText(2), "大图合成")
            self.assertEqual(window.tabs.tabText(3), "大图分割")
            stitch = window.stitch_tab
            self.assertFalse(hasattr(stitch, "export_button"))
            stitch.source_folder = source
            stitch.folder_edit.setText(str(source))
            stitch.rows_spin.setValue(2)
            stitch.cols_spin.setValue(3)
            stitch.reindex()
            self.wait_for(lambda: not stitch._preview_running and not stitch._preview_pending)
            stitch.overlap_x.setValue(truth.overlap_x_percent)
            stitch.overlap_y.setValue(truth.overlap_y_percent)
            stitch.rotation.setValue(truth.tile_rotation_deg)
            scheme = workspace / "scan.stitch.json"
            stitch.save_project(scheme)

            payload = json.loads(scheme.read_text(encoding="utf-8"))
            self.assertEqual((payload["tile_width"], payload["tile_height"]), (120, 80))
            mosaic = window.mosaic_tab
            self.assertEqual(mosaic.project_path, scheme.resolve())
            mosaic.set_image_folder(filtered)
            self.assertIsNotNone(mosaic.tile_set)
            self.assertTrue(mosaic.generate_button.isEnabled())
            self.assertTrue(mosaic.thumbnail_button.isEnabled())
            output = workspace / "result.ome.tif"
            mosaic.output_path = output
            mosaic.output_edit.setText(str(output))

            with patch("large_image_tool.app.QMessageBox.information") as information:
                mosaic.generate_thumbnail()
                self.wait_for(
                    lambda: not mosaic._busy and len(mosaic._active_workers) == 0,
                    timeout=20.0,
                )
                information.assert_called()

            thumbnail = workspace / "result_thumbnail.png"
            self.assertTrue(thumbnail.is_file())
            self.assertFalse(output.exists())
            self.assertEqual(mosaic.path, thumbnail)

            with patch("large_image_tool.app.QMessageBox.information") as information:
                mosaic.generate_mosaic()
                self.wait_for(
                    lambda: not mosaic._busy and len(mosaic._active_workers) == 0,
                    timeout=20.0,
                )
                information.assert_called()

            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".preview.png").is_file())
            self.assertEqual(mosaic.path, output)
            # Geometry changes also queue a stitch preview. Finish it before
            # TemporaryDirectory removes the source images used by that worker.
            self.wait_for(
                lambda: not stitch._preview_running
                and not stitch._preview_pending
                and len(stitch._active_workers) == 0,
            )
            window.close()

    def test_toolmark_folder_with_many_items_loads_one_bounded_preview(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            seed = root / "000001.png"
            Image.new("RGB", (384, 216), (90, 120, 135)).save(seed)
            for number in range(2, 871):
                shutil.copyfile(seed, root / f"{number:06d}.png")

            window = MainWindow()
            toolmarks = window.toolmark_tab
            toolmarks.open_source(root)
            self.wait_for(
                lambda: toolmarks._preview_processed is not None
                and not toolmarks._preview_running
                and not toolmarks._preview_pending,
                timeout=15.0,
            )
            self.assertEqual(len(toolmarks.source_files), 870)
            self.assertEqual(toolmarks.item_combo.count(), 870)
            self.assertEqual(toolmarks.item_combo.currentIndex(), 435)
            self.assertTrue(toolmarks.batch_radio.isChecked())
            self.assertEqual(toolmarks.apply_button.text(), "开始批量处理")
            self.assertEqual(toolmarks.output_path, root.with_name(root.name + "_marks_removed"))
            self.assertLessEqual(max(toolmarks._preview_source.shape[:2]), 1024)
            self.assertEqual(toolmarks.max_peaks.value(), 14)
            self.assertEqual(toolmarks.max_peaks_label.text(), "频率峰/谐波上限")
            self.assertTrue(toolmarks.block_size.isHidden())
            self.assertEqual(len(toolmarks._active_workers), 0)
            window.close()

    def test_toolmark_preview_scales_the_whole_frame(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "wide.png"
            image = np.zeros((400, 1600, 3), dtype=np.uint8)
            image[:, :200] = (240, 20, 20)
            image[:, 200:1400] = (20, 220, 20)
            image[:, 1400:] = (20, 20, 240)
            Image.fromarray(image, mode="RGB").save(path)

            preview = _read_toolmark_preview(path)
            self.assertEqual(preview.shape, (256, 1024, 3))
            self.assertGreater(int(preview[128, 20, 0]), 200)
            self.assertGreater(int(preview[128, -20, 2]), 200)

    def test_toolmark_batch_mode_processes_folder_with_selected_settings(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            source = workspace / "source"
            source.mkdir()
            x = np.arange(128)[None, :]
            for number in range(1, 4):
                plane = np.broadcast_to(120 + 35 * np.sin(2 * np.pi * x / 12), (72, 128))
                rgb = np.repeat(np.clip(plane, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
                Image.fromarray(rgb, mode="RGB").save(source / f"{number:06d}.png")

            window = MainWindow()
            toolmarks = window.toolmark_tab
            toolmarks.open_source(source)
            self.wait_for(
                lambda: toolmarks._preview_processed is not None
                and not toolmarks._preview_running
                and not toolmarks._preview_pending,
                timeout=15.0,
            )
            method_index = toolmarks.method_combo.findData(ToolMarkMethod.DIRECTIONAL_1D.value)
            toolmarks.method_combo.setCurrentIndex(method_index)
            toolmarks.strength.setValue(0.4)
            toolmarks.auto_orientation.setChecked(False)
            toolmarks.orientation.setValue(0.0)
            toolmarks.auto_period.setChecked(False)
            toolmarks.period.setValue(12.0)
            output = workspace / "filtered"
            toolmarks.output_path = output
            toolmarks.output_edit.setText(str(output))
            self.wait_for(
                lambda: not toolmarks._preview_running and not toolmarks._preview_pending,
                timeout=15.0,
            )

            with patch("large_image_tool.app.QMessageBox.information") as information:
                toolmarks.process_and_save()
                self.wait_for(
                    lambda: not toolmarks._busy and len(toolmarks._active_workers) == 0,
                    timeout=15.0,
                )
                information.assert_called()

            self.assertEqual(
                sorted(path.name for path in output.glob("*.png")),
                ["000001.png", "000002.png", "000003.png"],
            )
            metadata = json.loads((output / "toolmark_settings.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["settings"]["method"], ToolMarkMethod.DIRECTIONAL_1D.value)
            self.assertAlmostEqual(metadata["settings"]["strength"], 0.4)
            self.assertFalse(list(output.glob(".*.writing.*")))
            window.close()

    def test_local_detection_outline_tracks_the_selected_image(self) -> None:
        path = Path(__file__).resolve().parents[1] / 'sample_data' / 'processed_0443.png'
        if not path.exists():
            self.skipTest('Optional microscopy sample is unavailable')
        window = MainWindow()
        tab = window.toolmark_tab
        tab.open_source(path)
        self.wait_for(lambda: tab._preview_processed is not None
            and not tab._preview_running and not tab._preview_pending
            and len(tab._active_workers) == 0, timeout=15)
        outlines = [item for item in tab.original_view.scene().items()
            if item.data(0) == 'toolmark_detection']
        self.assertEqual(len(outlines), 1)
        self.assertIn('局部检出', tab.analysis_label.text())
        self.assertAlmostEqual(tab.orientation.value(), -45, delta=2)
        self.assertGreater(outlines[0].rect().top(), tab._preview_source.shape[0] * 0.4)
        with tempfile.TemporaryDirectory() as raw:
            blank = Path(raw) / 'blank.png'
            Image.new('RGB', (384, 256), (128, 128, 128)).save(blank)
            tab.open_source(blank)
            self.wait_for(lambda: tab._preview_processed is not None
                and not tab._preview_running and not tab._preview_pending
                and len(tab._active_workers) == 0, timeout=15)
            self.assertFalse(any(item.data(0) == 'toolmark_detection'
                for item in tab.original_view.scene().items()))
            self.assertNotIn('局部检出', tab.analysis_label.text())
        window.close()

    def test_toolmark_preview_controls_keep_original_pixel_units(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "wide.png"
            x = np.arange(2048)[None, :]
            plane = np.broadcast_to(128 + 40 * np.sin(2 * np.pi * x / 32), (512, 2048))
            source = np.repeat(np.rint(plane).astype(np.uint8)[..., None], 3, axis=2)
            Image.fromarray(source).save(path)
            window = MainWindow()
            tab = window.toolmark_tab
            tab.open_source(path)
            self.wait_for(lambda: tab._preview_processed is not None
                and not tab._preview_running and not tab._preview_pending
                and len(tab._active_workers) == 0, timeout=15)
            self.assertEqual(tab._preview_scale, (0.5, 0.5))
            self.assertAlmostEqual(tab.period.value(), 32, delta=1)
            tab.auto_orientation.setChecked(False)
            tab.auto_period.setChecked(False)
            self.wait_for(lambda: not tab._preview_running and not tab._preview_pending
                and len(tab._active_workers) == 0, timeout=15)
            self.assertAlmostEqual(tab.settings().period_px, 32, delta=1)
            self.assertLess(float(tab._preview_processed.std()), float(tab._preview_source.std()) * 0.5)
            window.close()

    def test_toolmark_sidecar_preview_uses_raster_dimensions(self) -> None:
        from large_image_tool.app import _read_toolmark_preview_with_scale
        from large_image_tool.toolmarks import preview_tool_marks
        from large_image_tool.models import MarkFilterSettings
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "source.png"
            sidecar = Path(raw) / "source.preview.png"
            x = np.arange(2048)[None, :]
            plane = np.broadcast_to(128 + 30*np.sin(2*np.pi*x/64), (512, 2048))
            source = Image.fromarray(np.repeat(np.rint(plane).astype(np.uint8)[..., None], 3, axis=2))
            source.save(path)
            source.resize((512, 128), Image.Resampling.LANCZOS).save(sidecar)
            image, scale = _read_toolmark_preview_with_scale(path, sidecar)
            self.assertEqual(scale, (0.25, 0.25))
            _, report = preview_tool_marks(image, MarkFilterSettings(), scale)
            self.assertAlmostEqual(report.period_px, 64, delta=1)

    def test_toolmark_auto_parameters_follow_selected_image(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            height, width = 256, 512
            x = np.arange(width)[None, :]
            for number, period in ((1, 12), (2, 28)):
                plane = np.broadcast_to(128 + 50 * np.sin(2 * np.pi * x / period), (height, width))
                rgb = np.repeat(np.clip(plane, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
                Image.fromarray(rgb, mode="RGB").save(root / f"{number:06d}.png")

            window = MainWindow()
            toolmarks = window.toolmark_tab
            toolmarks.open_source(root)
            self.wait_for(
                lambda: toolmarks._preview_processed is not None
                and not toolmarks._preview_running
                and not toolmarks._preview_pending,
                timeout=15.0,
            )
            second_period = toolmarks.period.value()
            self.assertAlmostEqual(second_period, 28, delta=3)

            toolmarks.item_combo.setCurrentIndex(0)
            self.wait_for(
                lambda: not toolmarks._preview_running
                and not toolmarks._preview_pending
                and abs(toolmarks.period.value() - second_period) > 5,
                timeout=15.0,
            )
            self.assertAlmostEqual(toolmarks.period.value(), 12, delta=2)
            toolmarks.auto_period.setChecked(False)
            self.assertTrue(toolmarks.period.isEnabled())
            self.assertAlmostEqual(toolmarks.period.value(), 12, delta=2)
            window.close()


if __name__ == "__main__":
    unittest.main()
