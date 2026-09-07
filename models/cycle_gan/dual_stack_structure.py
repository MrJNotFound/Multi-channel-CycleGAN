"""SPIF-Stack: SP loss with a standard stacked dual-channel generator.

Usage: --model spif_stack --input_nc 2 --dataset_mode dual_channel
"""

from models.cycle_gan.structure import SPIFModel


class SPIFStackModel(SPIFModel):
    """Standard ResNet generators plus the SPIF structure-preserving loss."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = SPIFModel.modify_commandline_options(parser, is_train)
        parser.set_defaults(netG="resnet_9blocks", spif_arch="stack")
        return parser

    def __init__(self, opt):
        # SPIFModel can select the standard stacked ResNet directly, so no
        # networks or optimizers need to be constructed and replaced here.
        SPIFModel.__init__(self, opt)
