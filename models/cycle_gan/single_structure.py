"""SPIF-Single: SP loss with standard single-channel ResNet generators.

Usage: --model spif_single --input_nc 1 --dataset_mode unaligned
"""

from models.cycle_gan.structure import SPIFModel


class SPIFSingleModel(SPIFModel):
    """Single-channel CycleGAN with the SPIF structure-preserving loss."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = SPIFModel.modify_commandline_options(parser, is_train)
        parser.set_defaults(netG="resnet_9blocks")
        return parser

    def __init__(self, opt):
        # input_nc is not two, so SPIFModel naturally keeps standard ResNet
        # generators while adding the structure-preserving loss exactly once.
        SPIFModel.__init__(self, opt)
