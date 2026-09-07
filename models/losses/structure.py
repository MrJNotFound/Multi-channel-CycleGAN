import math

import torch
import torch.nn.functional as F


def frontloaded_cosine_decay(epoch, total_epochs, active_fraction=0.1):
    """Return a smooth early-training weight that reaches zero at a cutoff.

    The schedule is expressed purely in epochs, so its behavior is independent
    of batch size, dataset size, the last partial batch, and DDP world size.
    Epoch 1 starts at one; ``ceil(total_epochs * active_fraction)`` is the
    first epoch whose weight is zero.
    """
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive.")
    if not math.isfinite(active_fraction) or not 0 < active_fraction <= 1:
        raise ValueError("active_fraction must be in (0, 1].")
    if epoch < 1:
        raise ValueError("epoch must be at least 1.")

    active_epochs = max(1, math.ceil(total_epochs * active_fraction))
    if active_epochs == 1:
        return 1.0 if epoch == 1 else 0.0
    if epoch >= active_epochs:
        return 0.0

    progress = (epoch - 1) / (active_epochs - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class FineStructureGradientLoss(torch.nn.Module):
    """Original-resolution gradient-magnitude loss for fine structures.

    Sobel filtering uses reflection padding so crop borders do not become
    artificial edges. Per-channel magnitudes are fused without per-image
    normalization, so nearly uniform regions cannot have weak noise amplified
    to unit energy. No image pyramid or spatial downsampling is used.
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive.")

        self.eps = float(eps)
        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[-1.0, 0.0, 1.0],
                 [-2.0, 0.0, 2.0],
                 [-1.0, 0.0, 1.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3) / 8.0,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[-1.0, -2.0, -1.0],
                 [0.0, 0.0, 0.0],
                 [1.0, 2.0, 1.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3) / 8.0,
        )

    def _channel_weights(self, x, channel_weights):
        channels = x.shape[1]
        if channel_weights is None:
            return x.new_full((channels,), 1.0 / channels)

        weights = torch.as_tensor(
            channel_weights, device=x.device, dtype=x.dtype
        ).flatten()
        if weights.numel() != channels:
            raise ValueError(
                "Gradient channel weights do not match the tensor channel "
                f"count: {weights.numel()} vs {channels}."
            )
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError(
                "Gradient channel weights must be finite and non-negative."
            )
        if weights.sum() <= 0:
            raise ValueError("Gradient channel weights must have a positive sum.")
        return weights / weights.sum()

    def gradient_map(self, x, channel_weights=None):
        """Return one unnormalized structural gradient map per image."""
        if x.ndim != 4 or x.shape[1] < 1:
            raise ValueError("Expected an NCHW tensor with at least one channel.")
        if min(x.shape[-2:]) < 2:
            raise ValueError("Sobel reflection padding requires H and W >= 2.")

        # Keep derivative filtering in float32 under AMP; casting remains
        # differentiable and avoids eps underflow in float16.
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        channels = work.shape[1]
        kernel_x = self.sobel_x.to(dtype=work.dtype).expand(
            channels, 1, 3, 3
        )
        kernel_y = self.sobel_y.to(dtype=work.dtype).expand(
            channels, 1, 3, 3
        )
        padded = F.pad(work, (1, 1, 1, 1), mode="reflect")
        grad_x = F.conv2d(padded, kernel_x, groups=channels)
        grad_y = F.conv2d(padded, kernel_y, groups=channels)

        # Subtract eps so a perfectly constant image produces an exact zero.
        magnitude = torch.sqrt(
            grad_x.square() + grad_y.square() + self.eps ** 2
        ) - self.eps
        weights = self._channel_weights(work, channel_weights).view(
            1, channels, 1, 1
        )
        return torch.sum(magnitude * weights, dim=1, keepdim=True)

    def forward(self, source, translated, source_weights=None,
                translated_weights=None):
        if source.shape[0] != translated.shape[0]:
            raise ValueError("Source and translated batches must have equal size.")
        if source.shape[-2:] != translated.shape[-2:]:
            raise ValueError(
                "Source and translated images must have equal spatial size."
            )

        source_gradient = self.gradient_map(source, source_weights)
        translated_gradient = self.gradient_map(
            translated, translated_weights
        )
        return F.l1_loss(translated_gradient, source_gradient)


class CoarseIntensityDistributionLoss(torch.nn.Module):
    """Guide the coarse spatial distribution of bright and dark tissue regions.

    This keeps the original SPIF intensity pipeline -- channel averaging and a
    21x21 Gaussian low-pass filter by default -- but does not regress absolute
    brightness. The low-pass maps are pooled to a small grid, spatially
    centered, and RMS-normalized. The loss minimizes one minus their correlation,
    providing direction-based supervision without matching absolute brightness.
    Its independently decaying epoch weight confines this guidance to the
    beginning of training.
    """

    def __init__(self, radius=10, sigma=5.0, pool_size=8, eps=1e-6,
                 min_source_rms=1e-4):
        super().__init__()
        radius = int(radius)
        sigma = float(sigma)
        pool_size = int(pool_size)
        eps = float(eps)
        min_source_rms = float(min_source_rms)
        if radius < 1:
            raise ValueError("Intensity Gaussian radius must be positive.")
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("Intensity Gaussian sigma must be positive.")
        if pool_size < 2:
            raise ValueError("Intensity pool size must be at least 2.")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("Intensity eps must be finite and positive.")
        if not math.isfinite(min_source_rms) or min_source_rms < 0:
            raise ValueError(
                "Minimum source RMS must be finite and non-negative."
            )

        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        gaussian_1d = torch.exp(-0.5 * (coords / sigma).square())
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        self.radius = radius
        self.pool_size = pool_size
        self.eps = eps
        self.min_source_rms = min_source_rms
        self.register_buffer(
            "gaussian_kernel_x", gaussian_1d.view(1, 1, 1, -1)
        )
        self.register_buffer(
            "gaussian_kernel_y", gaussian_1d.view(1, 1, -1, 1)
        )

    @staticmethod
    def _channel_weights(x, channel_weights):
        channels = x.shape[1]
        if channel_weights is None:
            return x.new_full((channels,), 1.0 / channels)

        weights = torch.as_tensor(
            channel_weights, device=x.device, dtype=x.dtype
        ).flatten()
        if weights.numel() != channels:
            raise ValueError(
                "Intensity channel weights do not match the tensor channel "
                f"count: {weights.numel()} vs {channels}."
            )
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError(
                "Intensity channel weights must be finite and non-negative."
            )
        if weights.sum() <= 0:
            raise ValueError(
                "Intensity channel weights must have a positive sum."
            )
        return weights / weights.sum()

    def coarse_intensity_map(self, x, channel_weights=None):
        """Return the original channel-mean low pass on a coarse spatial grid."""
        if x.ndim != 4 or x.shape[1] < 1:
            raise ValueError("Expected an NCHW tensor with at least one channel.")
        if min(x.shape[-2:]) <= self.radius:
            raise ValueError(
                "Reflection padding for the intensity term requires image "
                f"height and width greater than radius={self.radius}."
            )
        if min(x.shape[-2:]) < self.pool_size:
            raise ValueError(
                "Intensity pool size cannot exceed the input height or width."
            )

        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        weights = self._channel_weights(work, channel_weights).view(
            1, work.shape[1], 1, 1
        )
        intensity = torch.sum(work * weights, dim=1, keepdim=True)
        kernel_x = self.gaussian_kernel_x.to(dtype=work.dtype)
        kernel_y = self.gaussian_kernel_y.to(dtype=work.dtype)
        low = F.conv2d(
            F.pad(
                intensity, (self.radius, self.radius, 0, 0), mode="reflect"
            ),
            kernel_x,
        )
        low = F.conv2d(
            F.pad(
                low, (0, 0, self.radius, self.radius), mode="reflect"
            ),
            kernel_y,
        )
        return F.adaptive_avg_pool2d(
            low, output_size=(self.pool_size, self.pool_size)
        )

    def _correlation_and_validity(self, source, translated,
                                  source_weights=None,
                                  translated_weights=None):
        if source.shape[0] != translated.shape[0]:
            raise ValueError("Source and translated batches must have equal size.")
        if source.shape[-2:] != translated.shape[-2:]:
            raise ValueError(
                "Source and translated images must have equal spatial size."
            )

        source_low = self.coarse_intensity_map(source, source_weights)
        translated_low = self.coarse_intensity_map(
            translated, translated_weights
        )
        source_centered = source_low - source_low.mean(
            dim=(-2, -1), keepdim=True
        )
        translated_centered = translated_low - translated_low.mean(
            dim=(-2, -1), keepdim=True
        )
        source_var = source_centered.square().mean(dim=(-2, -1, -3))
        translated_var = translated_centered.square().mean(dim=(-2, -1, -3))
        covariance = (source_centered * translated_centered).mean(
            dim=(-2, -1, -3)
        )
        denominator = torch.sqrt(
            (source_var + self.eps) * (translated_var + self.eps)
        )
        correlation = (covariance / denominator).clamp(-1.0, 1.0)
        valid = source_var > self.min_source_rms ** 2
        return correlation, valid

    def coarse_correlation(self, source, translated, source_weights=None,
                           translated_weights=None):
        """Return one coarse normalized intensity correlation per image."""
        correlation, _ = self._correlation_and_validity(
            source,
            translated,
            source_weights=source_weights,
            translated_weights=translated_weights,
        )
        return correlation

    def forward(self, source, translated, source_weights=None,
                translated_weights=None):
        correlation, valid = self._correlation_and_validity(
            source,
            translated,
            source_weights=source_weights,
            translated_weights=translated_weights,
        )
        penalties = 1.0 - correlation
        valid_weights = valid.to(dtype=penalties.dtype)
        return torch.sum(penalties * valid_weights) / valid_weights.sum().clamp_min(1)
