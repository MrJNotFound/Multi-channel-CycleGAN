"""Fusion CycleGAN — alias for the current CycleGAN with dual-branch generator.

When input_nc=2, automatically switches to DualBranchResnetGenerator (G_A) and
DualOutputResnetGenerator (G_B) with CrossChannelFusion attention.
"""

from .cycle_gan_model import CycleGANModel


class CycleGANFusionModel(CycleGANModel):
    """Fusion variant: dual-branch encoder heads + CrossChannelFusion for 2-channel input."""
    pass
