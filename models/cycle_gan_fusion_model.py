"""Fusion CycleGAN — alias for the current CycleGAN with SPIF dual-branch generator.

When input_nc=2, automatically switches to SPIFGenerator (G_A) and
SPIFDualGenerator (G_B) with SPIFFusion attention.
"""

from .cycle_gan_model import CycleGANModel


class CycleGANFusionModel(CycleGANModel):
    """Fusion variant: dual-branch encoder heads + SPIFFusion for 2-channel input."""
    pass
