"""Train the SPIF cycle with adversarial, cycle, and structure-preserving losses."""

from dataclasses import asdict, dataclass
import itertools
import math
import random

import torch
import torch.nn.functional as F

from .losses import CoarseIntensityDistributionLoss, FineStructureGradientLoss, frontloaded_cosine_decay
from .networks import PatchDiscriminator, SPIFDualV2Generator, forward_generator, initialize_weights, instance_norm
from .runtime import ARCHITECTURE, CHECKPOINT_VERSION


@dataclass
class TrainConfig:
    """SPIF hyperparameters saved inside every checkpoint."""

    ngf: int = 64
    ndf: int = 64
    n_blocks: int = 9
    load_size: int = 286
    crop_size: int = 256
    batch_size: int = 1
    n_epochs: int = 100
    n_epochs_decay: int = 100
    lr: float = 0.0002
    beta1: float = 0.5
    lambda_A: float = 10.0
    lambda_B: float = 10.0
    lambda_sp: float = 1.0
    lambda_intensity: float = 0.5
    sp_grad_decay_fraction: float = 0.1
    sp_intensity_decay_fraction: float = 0.025
    intensity_pool_size: int = 8
    grad_channel_weights: list | None = None
    pool_size: int = 50
    no_flip: bool = False
    seed: int = 42

    def validate(self):
        for name in ("ngf", "ndf", "n_blocks", "batch_size", "n_epochs"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.n_epochs_decay < 0 or self.pool_size < 0:
            raise ValueError("n_epochs_decay and pool_size must be non-negative.")
        if self.crop_size < 32 or self.crop_size % 4 or self.load_size < self.crop_size:
            raise ValueError("crop_size must be >= 32 and divisible by 4; load_size must be >= crop_size.")
        if not 2 <= self.intensity_pool_size <= self.crop_size:
            raise ValueError("intensity_pool_size must be between 2 and crop_size.")
        if not math.isfinite(self.lr) or self.lr <= 0 or not 0 <= self.beta1 < 1:
            raise ValueError("lr must be positive and finite; beta1 must be in [0, 1).")
        for name in ("lambda_A", "lambda_B", "lambda_sp", "lambda_intensity"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not 0 < self.sp_intensity_decay_fraction <= self.sp_grad_decay_fraction <= 1:
            raise ValueError("Require 0 < sp_intensity_decay_fraction <= sp_grad_decay_fraction <= 1.")
        if not 0 <= self.seed < 2 ** 32:
            raise ValueError("seed must be in [0, 2**32).")
        if self.grad_channel_weights is not None:
            weights = self.grad_channel_weights
            if (len(weights) != 2 or any(not math.isfinite(w) or w < 0 for w in weights)
                    or sum(weights) <= 0):
                raise ValueError("grad_channel_weights requires two finite, non-negative values with a positive sum.")

    @property
    def total_epochs(self):
        return self.n_epochs + self.n_epochs_decay


class ImagePool:
    """Replay previously generated images for the discriminator updates."""

    def __init__(self, size):
        self.size = size
        self.images = []

    def query(self, batch):
        if self.size == 0:
            return batch.detach()
        output = []
        for image in batch.detach().split(1):
            if len(self.images) < self.size:
                self.images.append(image.clone())
                output.append(image)
            elif random.random() > 0.5:
                index = random.randrange(self.size)
                output.append(self.images[index].clone())
                self.images[index] = image.clone()
            else:
                output.append(image)
        return torch.cat(output)


class SPIFTrainer:
    """Own the four SPIF networks and the two Adam optimizers."""

    def __init__(self, config, device):
        config.validate()
        self.config = config
        self.device = device
        self.networks = {
            "G_A": forward_generator(config.ngf, config.n_blocks),
            "G_B": SPIFDualV2Generator(3, 2, ngf=config.ngf, n_blocks=config.n_blocks,
                                      norm_layer=instance_norm()),
            "D_A": PatchDiscriminator(3, config.ndf),
            "D_B": PatchDiscriminator(2, config.ndf),
        }
        for network in self.networks.values():
            initialize_weights(network)
            network.to(device).train()
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.networks["G_A"].parameters(), self.networks["G_B"].parameters()),
            lr=config.lr, betas=(config.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(
            itertools.chain(self.networks["D_A"].parameters(), self.networks["D_B"].parameters()),
            lr=config.lr, betas=(config.beta1, 0.999))
        self.gradient_loss = FineStructureGradientLoss().to(device)
        self.intensity_loss = CoarseIntensityDistributionLoss(pool_size=config.intensity_pool_size).to(device)
        self.pool_A = ImagePool(config.pool_size)
        self.pool_B = ImagePool(config.pool_size)

    def set_epoch(self, epoch):
        # The original CycleGAN/SPIF linear scheduler, evaluated at epoch start.
        factor = 1.0 - max(0, epoch - 1 - self.config.n_epochs) / (self.config.n_epochs_decay + 1)
        self.lr = self.config.lr * factor
        for optimizer in (self.optimizer_G, self.optimizer_D):
            for group in optimizer.param_groups:
                group["lr"] = self.lr
        self.gradient_weight = self.config.lambda_sp * frontloaded_cosine_decay(
            epoch, self.config.total_epochs, self.config.sp_grad_decay_fraction)
        self.intensity_weight = self.config.lambda_sp * self.config.lambda_intensity * frontloaded_cosine_decay(
            epoch, self.config.total_epochs, self.config.sp_intensity_decay_fraction)

    def _discriminator_grad(self, enabled):
        for name in ("D_A", "D_B"):
            for parameter in self.networks[name].parameters():
                parameter.requires_grad_(enabled)

    @staticmethod
    def _gan(prediction, real):
        target = torch.ones_like(prediction) if real else torch.zeros_like(prediction)
        return F.mse_loss(prediction, target)

    def step(self, batch):
        real_A, real_B = batch["A"].to(self.device), batch["B"].to(self.device)
        G_A, G_B = self.networks["G_A"], self.networks["G_B"]
        D_A, D_B = self.networks["D_A"], self.networks["D_B"]
        self._discriminator_grad(False)
        self.optimizer_G.zero_grad(set_to_none=True)
        fake_B = G_A(real_A)
        rec_A = G_B(fake_B)
        fake_A = G_B(real_B)
        rec_B = G_A(fake_A)
        losses = {
            "G_A": self._gan(D_A(fake_B), True),
            "G_B": self._gan(D_B(fake_A), True),
            "cycle_A": F.l1_loss(rec_A, real_A) * self.config.lambda_A,
            "cycle_B": F.l1_loss(rec_B, real_B) * self.config.lambda_B,
        }
        weights = self.config.grad_channel_weights
        for name, criterion, weight in (
                ("sp_grad", self.gradient_loss, self.gradient_weight),
                ("sp_int", self.intensity_loss, self.intensity_weight)):
            if weight > 0:
                losses[name + "_A"] = criterion(real_A, fake_B, source_weights=weights)
                losses[name + "_B"] = criterion(real_B, fake_A, translated_weights=weights)
            else:
                losses[name + "_A"] = fake_B.new_zeros(())
                losses[name + "_B"] = fake_B.new_zeros(())
        losses["content"] = (
            self.gradient_weight * (losses["sp_grad_A"] + losses["sp_grad_B"])
            + self.intensity_weight * (losses["sp_int_A"] + losses["sp_int_B"]))
        loss_G = losses["G_A"] + losses["G_B"] + losses["cycle_A"] + losses["cycle_B"] + losses["content"]
        if not torch.isfinite(loss_G):
            raise FloatingPointError("Non-finite generator loss. Check image values and training parameters.")
        loss_G.backward()
        self.optimizer_G.step()

        self._discriminator_grad(True)
        self.optimizer_D.zero_grad(set_to_none=True)
        losses["D_A"] = 0.5 * (self._gan(D_A(real_B), True)
                               + self._gan(D_A(self.pool_B.query(fake_B)), False))
        losses["D_B"] = 0.5 * (self._gan(D_B(real_A), True)
                               + self._gan(D_B(self.pool_A.query(fake_A)), False))
        loss_D = losses["D_A"] + losses["D_B"]
        if not torch.isfinite(loss_D):
            raise FloatingPointError("Non-finite discriminator loss.")
        loss_D.backward()
        self.optimizer_D.step()
        self.preview = {
            "BF": real_A[0, 0:1].detach(), "AF": real_A[0, 1:2].detach(),
            "fake_HE": fake_B[0].detach(), "real_HE": real_B[0].detach(),
        }
        scalars = {name: value.detach().item() for name, value in losses.items()}
        scalars.update(G=loss_G.detach().item(), D=loss_D.detach().item(),
                       sp_grad_weight=self.gradient_weight, sp_int_weight=self.intensity_weight)
        return scalars

    def checkpoint(self, epoch, steps, rng):
        return {
            "format_version": CHECKPOINT_VERSION, "architecture": ARCHITECTURE,
            "config": asdict(self.config), "epoch": epoch, "steps": steps,
            "networks": {name: {k: v.detach().cpu() for k, v in net.state_dict().items()}
                         for name, net in self.networks.items()},
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
            "pool_A": [image.cpu() for image in self.pool_A.images],
            "pool_B": [image.cpu() for image in self.pool_B.images],
            "rng": rng,
        }

    def restore(self, checkpoint):
        for name, network in self.networks.items():
            network.load_state_dict(checkpoint["networks"][name], strict=True)
        self.optimizer_G.load_state_dict(checkpoint["optimizer_G"])
        self.optimizer_D.load_state_dict(checkpoint["optimizer_D"])
        self.pool_A.images = [image.to(self.device) for image in checkpoint["pool_A"]]
        self.pool_B.images = [image.to(self.device) for image in checkpoint["pool_B"]]
