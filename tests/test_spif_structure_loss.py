import math

import pytest
import torch

from models.spif_model import (
    CoarseIntensityDistributionLoss,
    FineStructureGradientLoss,
    frontloaded_cosine_decay,
)


def test_reflection_padding_does_not_create_edges_on_constant_images():
    loss_fn = FineStructureGradientLoss()
    constant = torch.ones(2, 3, 32, 32)

    gradient = loss_fn.gradient_map(constant)
    half_gradient = loss_fn.gradient_map(constant.half())

    assert torch.count_nonzero(gradient) == 0
    assert torch.count_nonzero(half_gradient) == 0
    assert half_gradient.dtype == torch.float32


def test_gradient_keeps_contrast_amplitude_and_is_sign_invariant():
    torch.manual_seed(13)
    loss_fn = FineStructureGradientLoss()
    image = torch.randn(2, 2, 32, 32)

    reference = loss_fn.gradient_map(image)
    scaled = loss_fn.gradient_map(image * 3.5)
    inverted = loss_fn.gradient_map(-image)

    assert torch.allclose(reference, inverted, atol=2e-5, rtol=2e-5)
    assert torch.allclose(reference * 3.5, scaled, atol=2e-5, rtol=2e-5)


def test_weak_noise_is_not_amplified_by_gradient_map():
    torch.manual_seed(15)
    loss_fn = FineStructureGradientLoss()
    weak_noise = torch.randn(2, 2, 32, 32) * 1e-4

    gradient = loss_fn.gradient_map(weak_noise)

    assert gradient.square().mean().sqrt() < 1e-3


def test_original_resolution_gradient_supports_cross_domain_channels_and_backward():
    torch.manual_seed(17)
    loss_fn = FineStructureGradientLoss()
    source = torch.randn(2, 2, 64, 64)
    translated = torch.randn(2, 3, 64, 64, requires_grad=True)

    loss = loss_fn(
        source,
        translated,
        source_weights=torch.tensor([0.4, 0.6]),
    )
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert translated.grad is not None
    assert torch.isfinite(translated.grad).all()


def test_gradient_loss_is_direct_original_resolution_l1():
    torch.manual_seed(18)
    loss_fn = FineStructureGradientLoss()
    source = torch.randn(1, 2, 32, 32)
    translated = torch.randn(1, 3, 32, 32)

    expected = (
        loss_fn.gradient_map(translated) - loss_fn.gradient_map(source)
    ).abs().mean()
    actual = loss_fn(source, translated)

    assert actual.item() == pytest.approx(expected.item())


def test_coarse_intensity_loss_ignores_brightness_scale_but_rejects_reversal():
    torch.manual_seed(19)
    loss_fn = CoarseIntensityDistributionLoss(
        radius=3, sigma=1.5, pool_size=4
    )
    structure = torch.linspace(-1, 1, 32).view(1, 1, 1, 32).expand(
        2, 1, 32, 32
    )
    source = structure.repeat(1, 2, 1, 1)
    same = (structure * 0.15 + 0.7).repeat(1, 3, 1, 1)
    inverted = (-structure * 2.5 + 0.35).repeat(1, 3, 1, 1)

    same_loss = loss_fn(source, same)
    inverted_loss = loss_fn(source, inverted)
    same_correlation = loss_fn.coarse_correlation(source, same)
    inverted_correlation = loss_fn.coarse_correlation(source, inverted)

    assert torch.all(same_correlation > 0.99)
    assert torch.all(inverted_correlation < -0.99)
    assert same_loss.item() == pytest.approx(
        (1.0 - same_correlation).mean().item()
    )
    assert inverted_loss.item() == pytest.approx(
        (1.0 - inverted_correlation).mean().item()
    )
    assert inverted_loss > 1.99


def test_coarse_intensity_loss_is_one_minus_correlation_without_hinge():
    loss_fn = CoarseIntensityDistributionLoss(
        radius=3, sigma=1.5, pool_size=4
    )
    source = torch.linspace(-1, 1, 32).view(1, 1, 1, 32).expand(
        1, 1, 32, 32
    )
    translated = (source * 1e-3 + 0.8).requires_grad_()

    correlation = loss_fn.coarse_correlation(source, translated)
    loss = loss_fn(source, translated)
    loss.backward()

    assert loss.item() == pytest.approx((1.0 - correlation).mean().item())
    assert loss.item() > 0.0
    assert translated.grad is not None
    assert torch.count_nonzero(translated.grad) > 0


def test_coarse_intensity_loss_ignores_constant_source_patches():
    loss_fn = CoarseIntensityDistributionLoss(
        radius=3, sigma=1.5, pool_size=4
    )
    source = torch.ones(2, 2, 32, 32)
    translated = torch.randn(2, 3, 32, 32, requires_grad=True)

    loss = loss_fn(source, translated)
    loss.backward()

    assert loss.item() == pytest.approx(0.0)
    assert translated.grad is not None
    assert torch.count_nonzero(translated.grad) == 0


def test_coarse_intensity_loss_supports_cross_domain_channels_and_backward():
    torch.manual_seed(23)
    loss_fn = CoarseIntensityDistributionLoss(
        radius=3, sigma=1.5, pool_size=4
    )
    source = torch.randn(2, 2, 32, 32)
    translated = torch.randn(2, 3, 32, 32, requires_grad=True)

    loss = loss_fn(
        source,
        translated,
        source_weights=torch.tensor([0.4, 0.6]),
    )
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert translated.grad is not None
    assert torch.isfinite(translated.grad).all()


def test_frontloaded_cosine_schedule_reaches_zero_at_ten_percent():
    assert frontloaded_cosine_decay(1, 200, 0.1) == pytest.approx(1.0)
    assert 0.0 < frontloaded_cosine_decay(10, 200, 0.1) < 1.0
    assert frontloaded_cosine_decay(20, 200, 0.1) == pytest.approx(0.0)
    assert frontloaded_cosine_decay(200, 200, 0.1) == pytest.approx(0.0)

    # The midpoint is close to one half for a cosine schedule.
    assert frontloaded_cosine_decay(10, 200, 0.1) == pytest.approx(
        0.5 * (1.0 + math.cos(math.pi * 9.0 / 19.0))
    )


def test_intensity_schedule_finishes_before_gradient_schedule():
    assert frontloaded_cosine_decay(1, 200, 0.025) == pytest.approx(1.0)
    assert 0.0 < frontloaded_cosine_decay(3, 200, 0.025) < 1.0
    assert frontloaded_cosine_decay(5, 200, 0.025) == pytest.approx(0.0)
    assert frontloaded_cosine_decay(5, 200, 0.1) > 0.0
