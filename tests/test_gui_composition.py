"""Hidden-window GUI tests for component switching and old config migration."""

import json
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

from gui.app import ControlCenter
from models.registry import COMPONENT_LABELS, MODEL_SPECS


class CompositionUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.app = ControlCenter()
        except tk.TclError as exc:
            raise unittest.SkipTest(f"Tk runtime unavailable: {exc}") from exc
        cls.app.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.app.destroy()

    def test_each_combination_updates_visible_parameters_without_stale_fields(self):
        app = self.app
        for name, spec in MODEL_SPECS.items():
            with self.subTest(name=name):
                app._select_module(name)
                app.capture_fields()
                self.assertEqual(int(app.values["input_nc"]), spec.defaults["input_nc"])
                self.assertEqual(app.module_var.get(), name)
                if spec.fusion == "fusion" and spec.extra_loss == "base":
                    self.assertEqual(float(app.values["lambda_content"]), 0)
                for field, value in spec.components.items():
                    self.assertEqual(app.component_vars[field].get(), COMPONENT_LABELS[field][value])
                    self.assertIn(app.component_vars[field].get(), app.component_combos[field]["values"])

    def test_upstream_choice_updates_only_supported_downstream_options(self):
        app = self.app
        app._select_module("cycle_gan_dual_fusion_structure")
        app.component_vars["architecture"].set(COMPONENT_LABELS["architecture"]["cut"])
        app.on_component_changed("architecture")
        self.assertEqual(app.module_var.get(), "cut_dual_stack_base")
        self.assertEqual(app.component_combos["fusion"]["values"], (COMPONENT_LABELS["fusion"]["stack"],))
        app.component_vars["modalities"].set(COMPONENT_LABELS["modalities"]["single"])
        app.on_component_changed("modalities")
        self.assertEqual(app.module_var.get(), "cut_single_none_base")
        self.assertEqual(int(app.field_vars["input_nc"].get()), 3)

    def test_schema_2_load_migrates_without_overwriting_saved_values(self):
        app = self.app
        app._select_module("cycle_gan_single_none_base")
        app.field_vars["batch_size"].set("7")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            path.write_text(json.dumps({
                "schema": 2, "task": "train", "model_module": "spif",
                "dataset_mode": "dual_channel", "initialization": "fresh",
                "values": {"input_nc": 2, "output_nc": 3, "batch_size": 5,
                           "name": "old experiment", "spif_arch": "legacy"},
            }), encoding="utf-8")
            with patch("gui.app.filedialog.askopenfilename", return_value=str(path)):
                app.load_config()
        self.assertEqual(app.module_var.get(), "cycle_gan_dual_fusion_structure")
        self.assertEqual(int(app.field_vars["batch_size"].get()), 5)
        self.assertEqual(app.values["spif_arch"], "legacy")

    def test_schema_3_saves_components_and_roundtrips(self):
        app = self.app
        selected = "pix2pix_single_none_base_colorization"
        app._select_module(selected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new.json"
            with patch("gui.app.filedialog.asksaveasfilename", return_value=str(path)):
                app.save_config()
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["schema"], 3)
            self.assertEqual(saved["components"], MODEL_SPECS[selected].components)
            app._select_module("cycle_gan_single_none_base")
            with patch("gui.app.filedialog.askopenfilename", return_value=str(path)):
                app.load_config()
            self.assertEqual(app.module_var.get(), selected)
            self.assertEqual(app.dataset_var.get(), "colorization")

    def test_task_switch_retains_checkpoint_architecture(self):
        app = self.app
        app._select_module("cycle_gan_dual_fusion_structure")
        app.values.update(spif_arch="legacy", ngf=16, netG="resnet_6blocks")
        app.task_var.set("inference")
        app.on_task_changed()
        self.assertEqual(app.values["spif_arch"], "legacy")
        self.assertEqual(int(app.values["ngf"]), 16)
        self.assertEqual(app.values["netG"], "resnet_6blocks")
        app.task_var.set("train")
        app.on_task_changed()
        self.assertEqual(app.values["spif_arch"], "legacy")


if __name__ == "__main__":
    unittest.main()
