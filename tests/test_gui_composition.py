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

    def test_cycle_gan_input_mode_switch_links_channels_dataset_and_loss(self):
        app = self.app
        for loss in ("base", "saliency"):
            with self.subTest(loss=loss):
                app._select_module(f"cycle_gan_single_none_{loss}")
                app.component_vars["modalities"].set(COMPONENT_LABELS["modalities"]["dual"])
                app.on_component_changed("modalities")
                # Moving from single input may default to Fusion; select Stack explicitly.
                app.component_vars["fusion"].set(COMPONENT_LABELS["fusion"]["stack"])
                app.on_component_changed("fusion")
                app.component_vars["extra_loss"].set(COMPONENT_LABELS["extra_loss"][loss])
                app.on_component_changed("extra_loss")
                self.assertEqual(app.module_var.get(), f"cycle_gan_dual_stack_{loss}")
                self.assertEqual(app.dataset_var.get(), "dual_channel")
                self.assertEqual(int(app.field_vars["input_nc"].get()), 2)
                self.assertEqual(float(app.values["lambda_identity"]), 0)
                app.component_vars["modalities"].set(COMPONENT_LABELS["modalities"]["single"])
                app.on_component_changed("modalities")
                self.assertEqual(app.module_var.get(), f"cycle_gan_single_none_{loss}")
                self.assertEqual(app.dataset_var.get(), "unaligned")
                self.assertIn(int(app.field_vars["input_nc"].get()), (1, 3))

    def test_old_gui_config_is_rejected_without_changing_selection(self):
        app = self.app
        app._select_module("cycle_gan_single_none_base")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            path.write_text(json.dumps({"schema": 2}), encoding="utf-8")
            with patch("gui.app.filedialog.askopenfilename", return_value=str(path)), patch("gui.app.messagebox.showerror") as error:
                app.load_config()
                error.assert_called_once()
        self.assertEqual(app.module_var.get(), "cycle_gan_single_none_base")

    def test_version_selector_changes_command_and_roundtrips(self):
        app = self.app
        app._select_module("cycle_gan_dual_fusion_structure")
        app.version_var.set("v1")
        app.on_version_changed()
        self.assertEqual(app.values["spif_version"], "v1")
        self.assertEqual(app.version_combo["values"], ("v1", "v2"))
        app._select_module("cycle_gan_dual_fusion_base")
        self.assertEqual(app.version_var.get(), "v1")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v1.json"
            with patch("gui.app.filedialog.asksaveasfilename", return_value=str(path)):
                app.save_config()
            app._select_module("cycle_gan_single_none_base")
            with patch("gui.app.filedialog.askopenfilename", return_value=str(path)):
                app.load_config()
            self.assertEqual(app.version_var.get(), "v1")
            self.assertEqual(app.values["spif_version"], "v1")

    def test_schema_4_saves_components_and_roundtrips(self):
        app = self.app
        selected = "pix2pix_single_none_base_colorization"
        app._select_module(selected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new.json"
            with patch("gui.app.filedialog.asksaveasfilename", return_value=str(path)):
                app.save_config()
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["schema"], 4)
            self.assertEqual(saved["components"], MODEL_SPECS[selected].components)
            app._select_module("cycle_gan_single_none_base")
            with patch("gui.app.filedialog.askopenfilename", return_value=str(path)):
                app.load_config()
            self.assertEqual(app.module_var.get(), selected)
            self.assertEqual(app.dataset_var.get(), "colorization")

    def test_task_switch_retains_checkpoint_architecture(self):
        app = self.app
        app._select_module("cycle_gan_dual_fusion_structure")
        app.values.update(spif_version="v1", ngf=16, netG="resnet_6blocks")
        app.task_var.set("inference")
        app.on_task_changed()
        self.assertEqual(app.values["spif_version"], "v1")
        self.assertEqual(int(app.values["ngf"]), 16)
        self.assertEqual(app.values["netG"], "resnet_6blocks")
        app.task_var.set("train")
        app.on_task_changed()
        self.assertEqual(app.values["spif_version"], "v1")


if __name__ == "__main__":
    unittest.main()
