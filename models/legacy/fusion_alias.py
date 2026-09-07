"""Historical misleading name: this is a standard CycleGAN, NOT fusion.

Retained only for old imports/options. Use the canonical composition registry
to request real SPIF fusion (with or without the optional structure loss).
"""

from models.cycle_gan.standard import CycleGANModel


class CycleGANFusionModel(CycleGANModel):
    """Compatibility alias; preserves the original standard-generator behavior."""
    pass
