"""Composition and version tests; no dataset or user checkpoint required."""

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
    composition_candidates, composition_errors, generator_names,
)
from gui.launcher import (
    build_command, default_values, discover_models, recommended_dataset_for,
)
from options.train_options import TrainOptions
from options.test_options import TestOptions as InferenceOptions


class ModelRegistryTests(unittest.TestCase):
    def test_model_sources_are_flat_and_prefixed(self):
        root = Path(models.__file__).resolve().parent
        self.assertFalse(list(root.glob("*/*.py")))
        for module_path, _ in IMPLEMENTATIONS.values():
            self.assertEqual(len(module_path.split(".")), 2)
            self.assertTrue(module_path.endswith("_model"))
        self.assertFalse(list(root.glob("loss_*.py")))
        for module in ("test_model", "template_model"):
            importlib.import_module(f"models.{module}")

    def test_registered_classes_and_losses_are_importable(self):
        for spec in MODEL_SPECS.values():
            path, name = IMPLEMENTATIONS[spec.implementation]
            cls = getattr(importlib.import_module(path), name)
            self.assertIs(models.find_model_using_name(spec.key), cls)
            self.assertIs(pickle.loads(pickle.dumps(cls)), cls)
        from models.cycle_gan_structure_model import FineStructureGradientLoss
        self.assertIsInstance(pickle.loads(pickle.dumps(FineStructureGradientLoss())), FineStructureGradientLoss)

    def test_removed_aliases_are_rejected(self):
        for name in ("spif", "spif_stack", "cycle_gan", "cut_stack", "cycle_gan_fusion"):
            with self.assertRaises(ValueError):
                models.find_model_using_name(name)
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("models.spif_model")

    def test_version_parameters_are_explicit(self):
        from gui.launcher import build_option_parser
        parser, _ = build_option_parser("train", "cycle_gan_dual_fusion_structure", "dual_channel")
        for version in ("v1", "v2"):
            self.assertEqual(parser.parse_args(["--dataroot", ".", "--spif_version", version]).spif_version, version)
        for flags in (["--spif_version", "legacy"], ["--spif_arch", "v2"], ["--lambda_low_freq", "0.5"]):
            with self.assertRaises(ValueError):
                parser.parse_args(["--dataroot", ".", *flags])

    def test_checkpoint_loading_never_silently_drops_parameters(self):
        from models.base_model import BaseModel
        loader = object.__new__(models.find_model_using_name("cycle_gan_single_none_base"))
        network = torch.nn.Linear(2, 3)
        weights = network.state_dict()
        weights["unrecognized.weight"] = torch.ones(1)
        with self.assertRaises(RuntimeError):
            BaseModel._load_network_state(loader, network, weights, "synthetic.pth")

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
                    options = TrainOptions() if task == "train" else InferenceOptions()
                    with patch.object(sys, "argv", command[1:]):
                        opt = options.gather_options()
                    selected = opt.model if task == "train" else opt.source_model
                    self.assertEqual(selected, name)
                    self.assertEqual(composition_errors(selected, vars(opt)), [])
                    self.assertEqual(models.find_model_using_name(name).__name__, IMPLEMENTATIONS[spec.implementation][1])
                    self.assertEqual(opt.input_nc, 2 if spec.modalities == "dual" else spec.defaults["input_nc"])

    def test_incompatible_compositions_are_rejected_before_running(self):
        cases = (
            ("cycle_gan_dual_fusion_structure", {"input_nc": 3}),
            ("cycle_gan_dual_fusion_structure", {"netG": "unet_256"}),
            ("cycle_gan_dual_fusion_structure", {"spif_version": "legacy"}),
            ("cycle_gan_dual_stack_structure", {"spif_version": "v2"}),
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

    def test_inference_generator_dispatch_matches_composition(self):
        # Verify registered generator dispatch without allocating GANs.
        from models.test_model import TestModel
        for name, spec in MODEL_SPECS.items():
            dataset = recommended_dataset_for("inference", name)
            values = default_values("inference", name, dataset)
            values.update(dataroot="unused", name="dispatch", gpu_ids="-1")
            command = build_command("inference", name, dataset, values)
            with patch.object(sys, "argv", command[1:]):
                opt = InferenceOptions().gather_options()
            opt.isTrain = False
            opt.device = torch.device("cpu")
            opt.gpu_ids = []
            with patch("models.test_model.networks.define_G", return_value=torch.nn.Identity()) as factory:
                runner = TestModel(opt)
                new_args = factory.call_args
                self.assertEqual(new_args.args[3], generator_names(name, vars(opt))[0])
                self.assertEqual(runner.source_model, name)
                if spec.fusion == "fusion":
                    self.assertEqual(new_args.args[3], "spif_v2_9blocks")
                if spec.adapter == "colorization":
                    self.assertEqual(runner.visual_names, ["real", "fake_rgb"])

    def test_fusion_compositions_train_one_batch_and_reload_generator(self):
        # A bounded real CPU forward/backward + state_dict handoff, not a GAN run.
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous_threads)
        for version, loss in (("v1", "base"), ("v1", "structure"), ("v2", "base"), ("v2", "structure")):
            name = f"cycle_gan_dual_fusion_{loss}"
            with self.subTest(version=version, loss=loss), tempfile.TemporaryDirectory() as directory:
                values = default_values("train", name, "dual_channel")
                values.update(dataroot="unused", name="smoke", checkpoints_dir=directory,
                              gpu_ids="-1", ngf=4, ndf=4, netG="resnet_6blocks", spif_version=version)
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
                    for field in ("name", "checkpoints_dir", "gpu_ids", "ngf", "netG", "dataroot", "spif_version"):
                        infer_values[field] = values[field]
                    infer_command = build_command("inference", name, "dual_channel", infer_values)
                    with patch.object(sys, "argv", infer_command[1:]):
                        infer_opt = InferenceOptions().gather_options()
                    infer_opt.isTrain, infer_opt.device, infer_opt.gpu_ids = False, torch.device("cpu"), []
                    runner = models.create_model(infer_opt)
                    runner.load_networks("smoke")
                    runner.set_input(data)
                    runner.test()
                    self.assertEqual(tuple(runner.fake.shape), (1, 3, 32, 32))
                    for key, weight in model.netG_A.state_dict().items():
                        self.assertTrue(torch.equal(weight.cpu(), runner.netG.state_dict()[key].cpu()))

    def test_every_composition_has_matching_train_and_inference_generators(self):
        for name, spec in MODEL_SPECS.items():
            with self.subTest(name=name), contextlib.redirect_stdout(io.StringIO()):
                state_keys = []
                for task in ("train", "inference"):
                    dataset = recommended_dataset_for(task, name)
                    values = default_values(task, name, dataset)
                    values.update(dataroot="unused", name="shape_test", gpu_ids="-1", ngf=4, netG="resnet_6blocks")
                    command = build_command(task, name, dataset, values)
                    with patch.object(sys, "argv", command[1:]):
                        options = TrainOptions() if task == "train" else InferenceOptions()
                        opt = options.gather_options()
                    # Build only generators; no optimizer step or files are needed.
                    opt.isTrain, opt.device, opt.gpu_ids = False, torch.device("cpu"), []
                    model = models.create_model(opt)
                    net = model.netG_A if task == "train" and spec.architecture == "cycle_gan" else model.netG
                    state_keys.append({key: tuple(value.shape) for key, value in net.state_dict().items()})
                self.assertEqual(*state_keys)


if __name__ == "__main__":
    unittest.main()
