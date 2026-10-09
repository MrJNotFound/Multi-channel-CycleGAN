"""Single-source and Stack modes must use the same training implementation."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import models
from gui.launcher import build_command, default_values, recommended_dataset_for
from options.train_options import TrainOptions
from options.test_options import TestOptions as InferenceOptions


class CycleGANInputModeTests(unittest.TestCase):
    def options(self, task, name, directory, channels):
        dataset = recommended_dataset_for(task, name)
        values = default_values(task, name, dataset)
        values.update(dataroot="unused", name="smoke", checkpoints_dir=directory,
                      gpu_ids="-1", input_nc=channels, output_nc=3,
                      ngf=4, netG="resnet_6blocks")
        if task == "train":
            # Deliberately request identity even with incompatible channels.
            values.update(ndf=4, pool_size=0, lambda_identity=0.5)
        command = build_command(task, name, dataset, values)
        with patch.object(sys, "argv", command[1:]):
            parser = TrainOptions() if task == "train" else InferenceOptions()
            opt = parser.gather_options()
        opt.isTrain, opt.device, opt.gpu_ids = task == "train", torch.device("cpu"), []
        opt.counter, opt.data_size = 1, 10
        return opt

    def test_input_modes_share_classes_without_stack_wrappers(self):
        for loss in ("base", "saliency", "structure"):
            self.assertIs(
                models.find_model_using_name(f"cycle_gan_single_none_{loss}"),
                models.find_model_using_name(f"cycle_gan_dual_stack_{loss}"),
            )
        root = Path(__file__).resolve().parents[1] / "models"
        self.assertFalse(list(root.glob("*stack*_model.py")))

    def test_train_visualize_freeze_and_reload_each_input_mode(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous_threads)
        for loss in ("base", "saliency"):
            for channels in (1, 2, 3):
                mode = "dual_stack" if channels == 2 else "single_none"
                name = f"cycle_gan_{mode}_{loss}"
                with self.subTest(loss=loss, channels=channels), tempfile.TemporaryDirectory() as directory:
                    with contextlib.redirect_stdout(io.StringIO()):
                        opt = self.options("train", name, directory, channels)
                        model = models.create_model(opt)
                        self.assertEqual(opt.lambda_identity, 0.5 if channels == 3 else 0)
                        data = {"A": torch.randn(1, channels, 32, 32),
                                "B": torch.randn(1, 3, 32, 32),
                                "A_paths": ["a.png"], "B_paths": ["b.png"]}
                        model.set_input(data)
                        model.optimize_parameters()
                        self.assertEqual(tuple(model.fake_B.shape), (1, 3, 32, 32))
                        self.assertEqual(tuple(model.rec_A.shape), (1, channels, 32, 32))
                        self.assertTrue(all(torch.isfinite(torch.tensor(v))
                                            for v in model.get_current_losses().values()))
                        model.compute_visuals()
                        visuals = model.get_current_visuals()
                        self.assertEqual("idt_A" in visuals, channels == 3)
                        if channels == 2:
                            for prefix in ("real_A", "fake_A", "rec_A"):
                                for suffix in ("bf", "af"):
                                    self.assertEqual(tuple(visuals[f"{prefix}_{suffix}"].shape), (1, 3, 32, 32))
                            self.assertNotIn("real_A", visuals)

                        # Shared fine-tuning hook must actually freeze both Ds.
                        before = {name: {k: v.clone() for k, v in getattr(model, name).state_dict().items()}
                                  for name in ("netD_A", "netD_B")}
                        model.set_discriminator_updates(False)
                        model.optimize_parameters()
                        for net_name, weights in before.items():
                            net = getattr(model, net_name)
                            self.assertTrue(all(torch.equal(v, net.state_dict()[k]) for k, v in weights.items()))
                            self.assertTrue(all(not p.requires_grad for p in net.parameters()))
                        model.set_discriminator_updates(True)
                        model.optimize_parameters()
                        self.assertTrue(all(p.requires_grad for p in model.netD_A.parameters()))
                        self.assertTrue(any(not torch.equal(v, model.netD_A.state_dict()[k])
                                            for k, v in before["netD_A"].items()))

                        (Path(directory) / "smoke").mkdir()
                        model.save_networks("smoke")
                        infer_opt = self.options("inference", name, directory, channels)
                        runner = models.create_model(infer_opt)
                        runner.load_networks("smoke")
                        runner.set_input(data)
                        runner.eval()
                        runner.test()
                        model.netG_A.eval()
                        with torch.no_grad():
                            expected = model.netG_A(data["A"])
                        torch.testing.assert_close(runner.fake, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
