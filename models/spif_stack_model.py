"""SPIF-Stack: gradient-domain content loss + stacked dual-channel input.

Standard ResNet generator with stacked BF+AF (2ch) input (no dual heads,
no SPIFFusion). Keeps the full SPIF gradient-domain + low-frequency content
loss.

Usage: --model spif_stack  (--input_nc 2, --dataset_mode dual_channel)
"""

import torch
import itertools
from util.image_pool import ImagePool
from .cycle_gan_model import CycleGANModel
from .spif_model import SPIFModel
from . import networks


class SPIFStackModel(SPIFModel):
    """SPIF content loss with standard stacked dual-channel ResNet generator."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = SPIFModel.modify_commandline_options(parser, is_train)
        parser.set_defaults(netG="resnet_9blocks")
        return parser

    def __init__(self, opt):
        CycleGANModel.__init__(self, opt)
        self.loss_names.append("content")

        # Standard generators — no auto-switch to SPIF variants
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                         opt.netG, opt.norm, not opt.no_dropout,
                                         opt.init_type, opt.init_gain)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf,
                                         opt.netG, opt.norm, not opt.no_dropout,
                                         opt.init_type, opt.init_gain)

        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                             opt.n_layers_D, opt.norm,
                                             opt.init_type, opt.init_gain)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                             opt.n_layers_D, opt.norm,
                                             opt.init_type, opt.init_gain)

        if self.isTrain:
            if opt.lambda_identity > 0.0 and opt.input_nc != opt.output_nc:
                print(f"WARNING: lambda_identity={opt.lambda_identity} requires "
                      f"input_nc==output_nc, got input_nc={opt.input_nc}, "
                      f"output_nc={opt.output_nc}. Disabling identity loss.")
                opt.lambda_identity = 0.0
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.lambda_low_freq = opt.lambda_low_freq
            if opt.grad_channel_weights is not None:
                w = torch.tensor(opt.grad_channel_weights, dtype=torch.float32)
                self.grad_channel_weights = w / w.sum()
            else:
                self.grad_channel_weights = None
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
