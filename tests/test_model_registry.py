"""Organization/compatibility tests; no dataset or user checkpoint required."""

import importlib
import contextlib
import io
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import models
from models.registry import (
    COMPONENT_LABELS, IMPLEMENTATIONS, MODEL_SPECS,
    canonical_name, composition_candidates, composition_errors,
)
from gui.launcher import (
    build_command, default_values, discover_models, recommended_dataset_for,
)
from options.train_options import TrainOptions
from options.test_options import TestOptions


class ModelRegistryTests(unittest.TestCase):
    def test_old_imports_resolve_to_exact_original_classes(self):
        for name, (module_path, class_name) in IMPLEMENTATIONS.items():
            with self.subTest(name=name):
                legacy = importlib.import_module(f"models.{name}_model")
                organized = importlib.import_module(module_path)
                self.assertIs(getattr(legacy, class_name), getattr(organized, class_name))
                self.assertIs(models.find_model_using_name(name), getattr(organized, class_name))
                cls = getattr(organized, class_name)
                self.assertIs(pickle.loads(pickle.dumps(cls)), cls)

    def test_loss_imports_and_serialization_remain_compatible(self):
        from models.spif_model import FineStructureGradientLoss as OldLoss
        from models.losses.structure import FineStructureGradientLoss
        from models.patchnce import PatchNCELoss as OldNCE
        from models.losses.patchnce import PatchNCELoss
        self.assertIs(OldLoss, FineStructureGradientLoss)
        self.assertIs(OldNCE, PatchNCELoss)
        self.assertIsInstance(pickle.loads(pickle.dumps(OldLoss())), FineStructureGradientLoss)

    def test_gui_discovers_compositions_not_old_alias_duplicates(self):
        names = discover_models()
        self.assertTrue(set(MODEL_SPECS).issubset(names))
        self.assertFalse(set(IMPLEMENTATIONS).intersection(names))

    def test_composition_choices_are_unique_and_reachable(self):
        tuples = [tuple(spec.components.items()) for spec in MODEL_SPECS.values()]
        self.assertEqual(len(tuples), len(set(tuples)))
        for spec in MODEL_SPECS.values():
            filters = {}
            for field in COMPONENT_LABELS:
                self.assertIn(spec, composition_candidates(filters))
                filters[field] = getattr(spec, field)
            self.assertEqual(composition_candidates(filters), [spec])
        self.assertEqual(composition_candidates({"architecture": "cut", "fusion": "fusion"}), [])

    def test_all_compositions_roundtrip_through_real_train_and_test_parsers(self):
        for task in ("train", "inference"):
            for name, spec in MODEL_SPECS.items():
                with self.subTest(task=task, name=name):
                    dataset = recommended_dataset_for(task, name)
                    values = default_values(task, name, dataset)
                    values.update(dataroot="unused", name="roundtrip", gpu_ids="-1")
                    command = build_command(task, name, dataset, values, "python")
                    options = TrainOptions() if task == "train" else TestOptions()
                    with patch.object(sys, "argv", command[1:]):
                        opt = options.gather_options()
                    selected = opt.model if task == "train" else opt.source_model
                    self.assertEqual(selected, name)
                    self.assertEqual(composition_errors(selected, vars(opt)), [])
                    self.assertIs(models.find_model_using_name(name), models.find_model_using_name(spec.legacy_name))
                    self.assertEqual(opt.input_nc, 2 if spec.modalities == "dual" else spec.defaults["input_nc"])

    def test_migration_uses_parameters_and_preserves_ambiguous_legacy_ids(self):
        self.assertEqual(canonical_name("spif", {"input_nc": 1}), "cycle_gan_single_none_structure")
        self.assertEqual(canonical_name("spif", {"input_nc": 2, "spif_arch": "stack"}), "cycle_gan_dual_stack_structure")
        self.assertEqual(canonical_name("spif", {"input_nc": 2, "lambda_content": 0}), "cycle_gan_dual_fusion_base")
        self.assertEqual(canonical_name("cycle_gan_fusion", {"input_nc": 2}), "cycle_gan_fusion")
        self.assertEqual(canonical_name("cycle_gan", {"input_nc": 2}), "cycle_gan")
        self.assertEqual(canonical_name("cut", {"input_nc": 2}), "cut")
        self.assertEqual(canonical_name("spif", {"input_nc": 2, "netG": "unet_256"}), "spif")

    def test_incompatible_compositions_are_rejected_before_running(self):
        cases = (
            ("cycle_gan_dual_fusion_structure", {"input_nc": 3}),
            ("cycle_gan_dual_fusion_structure", {"netG": "unet_256"}),
            ("cycle_gan_dual_fusion_structure", {"spif_arch": "stack"}),
            ("cycle_gan_dual_stack_structure", {"spif_arch": "v2"}),
            ("cycle_gan_single_none_base", {"input_nc": 2}),
            ("cycle_gan_dual_stack_base", {"netG": "spif_v2_9blocks"}),
            ("cycle_gan_dual_fusion_base", {"lambda_content": 1}),
            ("pix2pix_single_none_base_colorization", {"output_nc": 3}),
            ("cut_dual_stack_base", {"nce_idt": True}),
        )
        for name, overrides in cases:
            with self.subTest(name=name, overrides=overrides):
                dataset = recommended_dataset_for("train", name)
                values = default_values("train", name, dataset)
                values.update(dataroot="unused", name="test", **overrides)
                self.assertTrue(composition_errors(name, values))
                with self.assertRaises(ValueError):
                    build_command("train", name, dataset, values)

    def test_inference_generator_dispatch_matches_legacy_name(self):
        # Verify alias dispatch for all compositions without allocating GANs.
        from models.runtime.inference import TestModel
        for name, spec in MODEL_SPECS.items():
            dataset = recommended_dataset_for("inference", name)
            values = default_values("inference", name, dataset)
            values.update(dataroot="unused", name="dispatch", gpu_ids="-1")
            command = build_command("inference", name, dataset, values)
            with patch.object(sys, "argv", command[1:]):
                opt = TestOptions().gather_options()
            opt.isTrain = False
            opt.device = torch.device("cpu")
            opt.gpu_ids = []
            with patch("models.runtime.inference.networks.define_G", return_value=torch.nn.Identity()) as factory:
                runner = TestModel(opt)
                new_args = factory.call_args
                opt.source_model = spec.legacy_name
                TestModel(opt)
                self.assertEqual(new_args, factory.call_args)
                self.assertEqual(runner.source_model, spec.legacy_name)
                if spec.fusion == "fusion":
                    self.assertEqual(new_args.args[3], "spif_v2_9blocks")
                if spec.adapter == "colorization":
                    self.assertEqual(runner.visual_names, ["real", "fake_rgb"])

    def test_fusion_compositions_train_one_batch_and_reload_generator(self):
        # A bounded real CPU forward/backward + state_dict handoff, not a GAN run.
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous_threads)
        for loss in ("base", "structure"):
            name = f"cycle_gan_dual_fusion_{loss}"
            with self.subTest(loss=loss), tempfile.TemporaryDirectory() as directory:
                values = default_values("train", name, "dual_channel")
                values.update(dataroot="unused", name="smoke", checkpoints_dir=directory,
                              gpu_ids="-1", ngf=4, ndf=4, netG="resnet_6blocks")
                command = build_command("train", name, "dual_channel", values)
                with patch.object(sys, "argv", command[1:]):
                    opt = TrainOptions().gather_options()
                opt.isTrain, opt.device, opt.gpu_ids = True, torch.device("cpu"), []
                (Path(directory) / "smoke").mkdir()
                with contextlib.redirect_stdout(io.StringIO()):
                    model = models.create_model(opt)
                    model.set_current_epoch(1)
                    data = {"A": torch.randn(1, 2, 32, 32), "B": torch.randn(1, 3, 32, 32),
                            "A_paths": ["a.png"], "B_paths": ["b.png"]}
                    model.set_input(data)
                    model.optimize_parameters()
                    self.assertTrue(torch.isfinite(model.loss_G).all())
                    if loss == "base":
                        self.assertEqual(float(model.loss_content), 0)
                    model.save_networks("smoke")
                    infer_values = default_values("inference", name, "dual_channel")
                    for field in ("name", "checkpoints_dir", "gpu_ids", "ngf", "netG", "dataroot"):
                        infer_values[field] = values[field]
                    infer_command = build_command("inference", name, "dual_channel", infer_values)
                    with patch.object(sys, "argv", infer_command[1:]):
                        infer_opt = TestOptions().gather_options()
                    infer_opt.isTrain, infer_opt.device, infer_opt.gpu_ids = False, torch.device("cpu"), []
                    runner = models.create_model(infer_opt)
                    runner.load_networks("smoke")
                    runner.set_input(data)
                    runner.test()
                    self.assertEqual(tuple(runner.fake.shape), (1, 3, 32, 32))
                    for key, weight in model.netG_A.state_dict().items():
                        self.assertTrue(torch.equal(weight.cpu(), runner.netG.state_dict()[key].cpu()))


if __name__ == "__main__":
    unittest.main()
