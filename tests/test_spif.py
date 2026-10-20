"""Regression checks for the SPIF model, data contract, and public entry points."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from spif.data import SPIFDataset
from spif.losses import CoarseIntensityDistributionLoss, FineStructureGradientLoss, frontloaded_cosine_decay
from spif.networks import SPIFCompetitiveFusion, SPIFDualV2Generator, forward_generator, instance_norm


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def create_images(root):
    rng = np.random.default_rng(7)
    for folder in ("trainA_BF", "trainA_AF", "trainB", "testA_BF", "testA_AF"):
        (root / folder).mkdir(parents=True)
    for index in range(3):
        array = rng.integers(0, 256, (40, 40), dtype=np.uint8)
        for channel in ("BF", "AF"):
            Image.fromarray(array).save(root / f"trainA_{channel}" / f"{index}.png")
    for index in range(2):
        Image.fromarray(rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)).save(
            root / "trainB" / f"he_{index}.png")
    for channel in ("BF", "AF"):
        for filename, shape in (("tile.png", (29, 31)), ("nested/tile.png", (32, 36))):
            path = root / f"testA_{channel}" / filename
            path.parent.mkdir(exist_ok=True)
            Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8)).save(path)


class ModelChecks(unittest.TestCase):
    def test_fusion_weights_are_complementary_and_differentiable(self):
        fusion = SPIFCompetitiveFusion(8, norm_layer=instance_norm())
        bf = torch.randn(2, 8, 16, 16, requires_grad=True)
        af = torch.randn_like(bf, requires_grad=True)
        weight_bf, weight_af = fusion.compute_modality_weights(bf, af)
        self.assertTrue(torch.all((weight_bf >= 0) & (weight_bf <= 1)))
        torch.testing.assert_close(weight_bf + weight_af, torch.ones_like(weight_bf))
        fusion(bf, af).square().mean().backward()
        self.assertIsNotNone(bf.grad)
        self.assertIsNotNone(af.grad)
        self.assertTrue(torch.isfinite(bf.grad).all())

    def test_forward_reverse_cycle_and_skip_shapes(self):
        forward = forward_generator(ngf=4, n_blocks=1)
        reverse = SPIFDualV2Generator(3, 2, ngf=4, n_blocks=1, norm_layer=instance_norm())
        source = torch.randn(2, 2, 32, 32, requires_grad=True)
        translated = forward(source)
        reconstructed = reverse(translated)
        self.assertEqual(translated.shape, (2, 3, 32, 32))
        self.assertEqual(reconstructed.shape, source.shape)
        reconstructed.abs().mean().backward()
        self.assertTrue(torch.isfinite(source.grad).all())
        self.assertEqual(sum(isinstance(m, SPIFCompetitiveFusion) for m in forward.modules()), 1)

    def test_sobel_preserves_amplitude_without_background_edges(self):
        loss = FineStructureGradientLoss()
        self.assertEqual(torch.count_nonzero(loss.gradient_map(torch.ones(1, 2, 32, 32))).item(), 0)
        image = torch.randn(1, 2, 32, 32)
        reference = loss.gradient_map(image)
        torch.testing.assert_close(reference, loss.gradient_map(-image), atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(reference * 3.5, loss.gradient_map(image * 3.5), atol=2e-5, rtol=2e-5)
        weak = loss.gradient_map(image * 1e-4)
        self.assertLess(weak.square().mean().sqrt().item(), 1e-3)

    def test_cross_channel_losses_backpropagate(self):
        source = torch.randn(2, 2, 32, 32)
        for loss in (FineStructureGradientLoss(), CoarseIntensityDistributionLoss()):
            with self.subTest(loss=type(loss).__name__):
                translated = torch.randn(2, 3, 32, 32, requires_grad=True)
                value = loss(source, translated, source_weights=[0.4, 0.6])
                value.backward()
                self.assertTrue(torch.isfinite(value))
                self.assertTrue(torch.isfinite(translated.grad).all())
                self.assertGreater(torch.count_nonzero(translated.grad).item(), 0)

    def test_intensity_correlation_rejects_brightness_inversion(self):
        loss = CoarseIntensityDistributionLoss(radius=3, sigma=1.5, pool_size=4)
        source = torch.linspace(-1, 1, 32).view(1, 1, 1, 32).expand(1, 2, 32, 32)
        same = (source[:, :1] * 0.15 + 0.7).repeat(1, 3, 1, 1)
        inverse = (-source[:, :1] * 2.5 + 0.35).repeat(1, 3, 1, 1)
        self.assertLess(loss(source, same).item(), 0.01)
        self.assertGreater(loss(source, inverse).item(), 1.99)
        translated = torch.randn(1, 3, 32, 32, requires_grad=True)
        constant_loss = loss(torch.ones_like(source), translated)
        constant_loss.backward()
        self.assertEqual(constant_loss.item(), 0)
        self.assertEqual(torch.count_nonzero(translated.grad).item(), 0)

    def test_independent_guidance_schedules(self):
        self.assertEqual(frontloaded_cosine_decay(1, 200, 0.1), 1)
        self.assertGreater(frontloaded_cosine_decay(10, 200, 0.1), 0)
        self.assertEqual(frontloaded_cosine_decay(20, 200, 0.1), 0)
        self.assertEqual(frontloaded_cosine_decay(5, 200, 0.025), 0)
        self.assertGreater(frontloaded_cosine_decay(5, 200, 0.1), 0)


class DataChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        create_images(self.root)

    def test_spatial_augmentation_keeps_bf_af_aligned(self):
        dataset = SPIFDataset(self.root, training=True, load_size=36, crop_size=32)
        self.assertEqual(len(dataset), 3)
        for index in range(6):
            sample = dataset[index]
            torch.testing.assert_close(sample["A"][0], sample["A"][1], atol=0, rtol=0)
            self.assertEqual(sample["B"].shape, (3, 32, 32))

    def test_test_data_needs_no_he_and_keeps_size(self):
        sample = SPIFDataset(self.root)[1]
        self.assertNotIn("B", sample)
        self.assertEqual(sample["A"].shape, (2, 29, 31))

    def test_missing_partner_is_rejected(self):
        (self.root / "testA_AF" / "tile.png").unlink()
        with self.assertRaisesRegex(ValueError, "identical relative filenames"):
            SPIFDataset(self.root)

    def test_mismatched_dimensions_are_rejected(self):
        Image.new("L", (17, 17)).save(self.root / "testA_AF" / "tile.png")
        with self.assertRaisesRegex(ValueError, "same size"):
            SPIFDataset(self.root)[1]

    def test_training_requires_he(self):
        for path in (self.root / "trainB").iterdir():
            path.unlink()
        with self.assertRaisesRegex(ValueError, "No supported images"):
            SPIFDataset(self.root, training=True)


class WorkflowChecks(unittest.TestCase):
    def test_training_resume_and_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            data = work / "data"
            create_images(data)
            checkpoints = work / "checkpoints"
            results = work / "results"

            def cli(script, *arguments, expected=0):
                completed = subprocess.run(
                    [sys.executable, str(ROOT / script), *map(str, arguments)],
                    cwd=work, capture_output=True, text=True, timeout=180)
                self.assertEqual(completed.returncode, expected, completed.stdout + completed.stderr)
                return completed

            runtime = ("--dataroot", data, "--checkpoints_dir", checkpoints,
                       "--device", "cpu", "--cpu_threads", 1)
            tiny = ("--ngf", 4, "--ndf", 4, "--n_blocks", 1, "--crop_size", 32,
                    "--load_size", 36, "--batch_size", 2, "--pool_size", 2,
                    "--n_epochs", 1, "--n_epochs_decay", 1, "--seed", 23)
            cli("train.py", *runtime, *tiny, "--name", "full")
            cli("train.py", *runtime, *tiny, "--name", "split", "--stop_after_epoch", 1)
            first = torch.load(checkpoints / "split/latest.pt", weights_only=True)
            self.assertEqual(first["epoch"], 1)
            self.assertEqual(first["steps"], 2)
            self.assertTrue(first["optimizer_G"]["state"])
            self.assertEqual(len(first["pool_A"]), 2)
            cli("train.py", *runtime, "--name", "split", "--resume")
            full = torch.load(checkpoints / "full/latest.pt", weights_only=True)
            resumed = torch.load(checkpoints / "split/latest.pt", weights_only=True)
            self.assertEqual(resumed["epoch"], 2)
            self.assertEqual(resumed["steps"], 4)
            for name in full["networks"]:
                for key, tensor in full["networks"][name].items():
                    torch.testing.assert_close(tensor, resumed["networks"][name][key], atol=0, rtol=0)
            with (checkpoints / "split/losses.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["epoch"] for row in rows], ["1", "2"])
            self.assertGreater(float(rows[0]["content"]), 0)
            self.assertEqual(float(rows[1]["content"]), 0)
            self.assertTrue((checkpoints / "split/samples/epoch_0002_fake_HE.png").is_file())
            conflict = cli("train.py", *runtime, "--name", "split", "--resume",
                           "--lr", 0.01, expected=2)
            self.assertIn("Cannot change --lr", conflict.stderr)
            cli("train.py", *runtime, "--name", "split", expected=2)
            cli("test.py", *runtime, "--name", "split", "--results_dir", results)
            for name, size in (("tile.png_HE.png", (31, 29)), ("nested/tile.png_HE.png", (36, 32))):
                with Image.open(results / "split" / name) as image:
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.size, size)
            report = json.loads((results / "split/inference.json").read_text())
            self.assertEqual(len(report["images"]), 2)
            # Exercise Windows/spawn-safe worker initialization as well.
            cli("train.py", *runtime, *tiny, "--name", "workers", "--workers", 1,
                "--stop_after_epoch", 1, "--save_every", 1)
            self.assertTrue((checkpoints / "workers/epoch_0001.pt").is_file())


if __name__ == "__main__":
    unittest.main()
