"""UTOM (Unsupervised content-preserving Transformation for Optical Microscopy).

UTOM adds a threshold-based sigmoid saliency content loss to CycleGAN.
The content loss uses intensity thresholds to create soft tissue masks,
then matches them between source and generated images via L1 loss.

Reference: Li et al., "Unsupervised content-preserving transformation for
optical microscopy", Light: Science & Applications (2021).
https://www.nature.com/articles/s41377-021-00484-y
"""

import torch
import numpy as np
from models.cycle_gan_model import CycleGANModel


class UTOMModel(CycleGANModel):
    """UTOM model with threshold-based saliency content loss.

    Inherits standard CycleGAN losses (GAN + cycle + identity) and adds
    a content preservation term that matches intensity-thresholded tissue
    masks between source and translated images.

    Single-source and dual-source Stack inputs share this implementation.
    Select cycle_gan_single_none_saliency or cycle_gan_dual_stack_saliency;
    the composition configures the dataset and input channels.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = CycleGANModel.modify_commandline_options(parser, is_train)
        parser.add_argument("--threshold_A", type=float, default=70,
                            help="intensity threshold for domain A (source) "
                            "tissue mask. Lower values include more tissue.")
        parser.add_argument("--threshold_B", type=float, default=220,
                            help="intensity threshold for domain B (target) "
                            "tissue mask. Higher values include more tissue.")
        parser.add_argument("--domain_A_tissue", type=str, default="bright",
                            choices=["bright", "dark"],
                            help="Is domain A tissue bright or dark relative to background? "
                            "'bright' = bright tissue on dark background (original UTOM, e.g. AF). "
                            "'dark' = dark tissue on bright background (e.g. inverted AF, BF).")
        # Content loss weight decays as 50 * exp(-counter / data_size)
        # UTOM defaults identity loss to 0 (input_nc usually != output_nc)
        parser.set_defaults(lambda_identity=0)
        return parser

    def __init__(self, opt):
        CycleGANModel.__init__(self, opt)
        # Add content loss to tracked losses
        self.loss_names.append("content")
        self.threshold_A = opt.threshold_A
        self.threshold_B = opt.threshold_B
        # "bright" = sigmoid (tissue above threshold), "dark" = 1-sigmoid (tissue below threshold)
        self._mask_A = (lambda x: torch.sigmoid(x)) if opt.domain_A_tissue == "bright" else (lambda x: 1 - torch.sigmoid(x))

    def content_loss(self):
        """Threshold-based sigmoid saliency content loss.

        Computes per-channel mean intensity, normalizes relative to thresholds,
        applies sigmoid to create soft tissue masks, and matches masks between
        source and generated images via L1.

        Domain B (H&E target) always uses 1-sigmoid: stained tissue = dark.
        Domain A (source) uses sigmoid (tissue=bright) by default, or
        1-sigmoid (tissue=dark) when --domain_A_tissue dark (inverted AF/BF).

        The loss decays exponentially: rate = 50 * exp(-counter / data_size).
        """
        L1 = torch.nn.L1Loss()

        # Channel-wise mean intensity
        real_A_mean = torch.mean(self.real_A, dim=1, keepdim=True)
        real_B_mean = torch.mean(self.real_B, dim=1, keepdim=True)
        fake_A_mean = torch.mean(self.fake_A, dim=1, keepdim=True)
        fake_B_mean = torch.mean(self.fake_B, dim=1, keepdim=True)

        # Normalize relative to thresholds (mapped from [0,255] to [-1,1])
        thresh_A_norm = self.threshold_A / 127.5 - 1
        thresh_B_norm = self.threshold_B / 127.5 - 1

        real_A_normal = (real_A_mean - thresh_A_norm) * 100
        real_B_normal = (real_B_mean - thresh_B_norm) * 100
        fake_A_normal = (fake_A_mean - thresh_B_norm) * 100
        fake_B_normal = (fake_B_mean - thresh_A_norm) * 100

        # Domain A: uses configured mask (sigmoid for bright tissue, 1-sigmoid for dark)
        # Domain B: always 1-sigmoid (H&E stained tissue = dark on bright background)
        real_A_mask = self._mask_A(real_A_normal)
        real_B_mask = 1 - torch.sigmoid(real_B_normal)
        fake_A_mask = self._mask_A(fake_A_normal)
        fake_B_mask = 1 - torch.sigmoid(fake_B_normal)

        # Match source tissue mask -> generated target mask, and vice versa
        content_loss_A = L1(real_A_mask, fake_B_mask)
        content_loss_B = L1(fake_A_mask, real_B_mask)

        # Exponential decay: strong early guidance, cedes to GAN+cycle later
        rate = 50 * np.exp(-(self.opt.counter / self.opt.data_size))
        return rate * (content_loss_A + content_loss_B)

    def backward_G(self):
        """Override to include content loss in generator optimization."""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Identity loss
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # UTOM content loss
        self.loss_content = self.content_loss()
        # Combined
        self.loss_G = (self.loss_G_A + self.loss_G_B + self.loss_cycle_A +
                       self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B +
                       self.loss_content)
        self.loss_G.backward()
