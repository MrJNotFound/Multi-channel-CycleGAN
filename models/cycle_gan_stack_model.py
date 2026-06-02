"""Stacked CycleGAN — vanilla CycleGAN that treats 2-channel input as stacked images.

Unlike the fusion variant (cycle_gan / cycle_gan_fusion), this model does NOT use
DualBranchResnetGenerator or CrossChannelFusion. It feeds the stacked 2-channel
tensor directly into a standard ResnetGenerator, serving as an ablation baseline.

When input_nc=2, domain A is a (2, H, W) tensor and the generator is a standard
ResnetGenerator(2→3) — no per-channel heads, no cross-modal attention.
"""

import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CycleGANStackModel(BaseModel):
    """Vanilla CycleGAN for stacked 2-channel input — ablation baseline.

    Generators are always standard ResnetGenerator regardless of input_nc.
    No dual-branch heads, no CrossChannelFusion, no per-channel output heads.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument("--lambda_A", type=float, default=10.0,
                                help="weight for cycle loss (A -> B -> A)")
            parser.add_argument("--lambda_B", type=float, default=10.0,
                                help="weight for cycle loss (B -> A -> B)")
            parser.add_argument("--lambda_identity", type=float, default=0.5,
                                help="use identity mapping")
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["D_A", "G_A", "cycle_A", "idt_A", "D_B", "G_B", "cycle_B", "idt_B"]
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

        # Always use standard generators — no dual-branch / dual-output switch
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
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        self.loss_G = (self.loss_G_A + self.loss_G_B + self.loss_cycle_A +
                       self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B)
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
