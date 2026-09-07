import functools

import pytest
import torch

from models.base_model import BaseModel
from models.networks import (
    SPIFCompetitiveFusion,
    SPIFGenerator,
    SPIFDualV2Generator,
    SPIFV2Generator,
    define_G,
)


def _instance_norm():
    return functools.partial(
        torch.nn.InstanceNorm2d, affine=False, track_running_stats=False
    )


def test_competitive_weights_are_complementary_and_differentiable():
    torch.manual_seed(7)
    fusion = SPIFCompetitiveFusion(16, norm_layer=_instance_norm())
    bf = torch.randn(2, 16, 16, 16, requires_grad=True)
    af = torch.randn(2, 16, 16, 16, requires_grad=True)

    weight_bf, weight_af = fusion.compute_modality_weights(bf, af)

    assert weight_bf.shape == bf.shape
    assert torch.all((weight_bf >= 0) & (weight_bf <= 1))
    assert torch.allclose(
        weight_bf + weight_af, torch.ones_like(weight_bf), atol=1e-6
    )

    fusion(bf, af).mean().backward()
    assert bf.grad is not None
    assert af.grad is not None


def test_v2_cycle_preserves_tensor_shapes_and_backpropagates():
    torch.manual_seed(11)
    norm_layer = _instance_norm()
    forward = SPIFV2Generator(
        2, 3, ngf=8, norm_layer=norm_layer, n_blocks=1
    )
    reverse = SPIFDualV2Generator(
        3, 2, ngf=8, norm_layer=norm_layer, n_blocks=1
    )
    source = torch.randn(2, 2, 64, 64, requires_grad=True)

    fake_he = forward(source)
    reconstructed = reverse(fake_he)

    assert fake_he.shape == (2, 3, 64, 64)
    assert reconstructed.shape == source.shape
    (fake_he.abs().mean() + reconstructed.abs().mean()).backward()
    assert source.grad is not None


def test_v2_forward_uses_one_competitive_fusion_then_shared_downsampling():
    forward = SPIFV2Generator(
        2, 3, ngf=8, norm_layer=_instance_norm(), n_blocks=1
    )

    fusion_count = sum(
        isinstance(module, SPIFCompetitiveFusion)
        for module in forward.modules()
    )
    assert fusion_count == 1
    assert hasattr(forward, "down_shared")
    assert not hasattr(forward, "down_bf")
    assert not hasattr(forward, "down_af")
    assert not hasattr(forward, "fusion_quarter")


def test_v2_architectures_are_registered_in_generator_factory():
    forward = define_G(
        2, 3, ngf=8, netG="spif_v2_6blocks", norm="instance"
    )
    reverse = define_G(
        3, 2, ngf=8, netG="spif_dual_v2_6blocks", norm="instance"
    )

    assert isinstance(forward, SPIFV2Generator)
    assert isinstance(reverse, SPIFDualV2Generator)


def test_legacy_checkpoint_cannot_be_silently_loaded_as_v2():
    norm_layer = _instance_norm()
    legacy = SPIFGenerator(
        2, 3, ngf=8, norm_layer=norm_layer, n_blocks=1
    )
    v2 = SPIFV2Generator(
        2, 3, ngf=8, norm_layer=norm_layer, n_blocks=1
    )

    with pytest.raises(RuntimeError, match="not a SPIF V2"):
        BaseModel._validate_spif_checkpoint_architecture(
            v2, legacy.state_dict(), "legacy_net_G_A.pth"
        )
    BaseModel._validate_spif_checkpoint_architecture(
        v2, v2.state_dict(), "v2_net_G_A.pth"
    )

    two_stage_v2_state = {
        "fusion_half.placeholder": torch.empty(0),
        "fusion_quarter.placeholder": torch.empty(0),
    }
    with pytest.raises(RuntimeError, match="retired two-stage SPIF V2"):
        BaseModel._validate_spif_checkpoint_architecture(
            v2, two_stage_v2_state, "two_stage_v2_net_G_A.pth"
        )
