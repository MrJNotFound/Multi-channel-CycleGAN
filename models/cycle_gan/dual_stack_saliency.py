"""UTOM-Stack — UTOM content loss with stacked dual-channel input.

Unlike the standard UTOM (which uses a single-channel input), this variant
feeds stacked BF+AF (2-channel) directly into a standard generator — no
dual-branch heads, no SPIFFusion, no per-channel output heads.

Serves as the dual-channel ablation baseline for UTOM: measures the value
of a second input modality without architectural changes.

Usage: --model utom_stack  (requires --input_nc 2, --dataset_mode dual_channel)
"""

import torch
import itertools
from util.image_pool import ImagePool
from models.cycle_gan.saliency import UTOMModel
from models import networks


class UTOMStackModel(UTOMModel):
    """UTOM with stacked 2-channel input — standard generator, no dual heads.

    G_A: standard generator (2 -> 3)     stacked BF+AF -> RGB H&E
    G_B: standard generator (3 -> 2)     RGB H&E -> stacked BF+AF

    Keeps UTOM's threshold sigmoid content loss.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = UTOMModel.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        # Bypass UTOMModel/CycleGANModel generator auto-switch:
        # call BaseModel.__init__ directly, then set up UTOM manually
        from models.base_model import BaseModel
        BaseModel.__init__(self, opt)
        self.dual_channel = (opt.input_nc == 2)
        self.loss_names = ["D_A", "G_A", "cycle_A", "idt_A", "D_B", "G_B", "cycle_B", "idt_B", "content"]
        visual_names_A = ["real_A", "fake_B", "rec_A"]
        visual_names_B = ["real_B", "fake_A", "rec_B"]
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append("idt_B")
            visual_names_B.append("idt_A")
        self.visual_names = visual_names_A + visual_names_B

        if self.isTrain:
            self.model_names = ["G_A", "G_B", "D_A", "D_B"]
        else:
            self.model_names = ["G_A", "G_B"]

        # Always use standard generators — no auto-switch to SPIF variants
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
                      f"input_nc==output_nc, but got input_nc={opt.input_nc}, "
                      f"output_nc={opt.output_nc}. Disabling identity loss.")
                opt.lambda_identity = 0.0
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.threshold_A = opt.threshold_A
            self.threshold_B = opt.threshold_B
            # Mask polarity: bright tissue (sigmoid) or dark tissue (1-sigmoid)
            self._mask_A = (lambda x: torch.sigmoid(x)) if opt.domain_A_tissue == "bright" else (lambda x: 1 - torch.sigmoid(x))
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
