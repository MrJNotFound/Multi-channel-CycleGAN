import torch
import itertools
import math
from util.image_pool import ImagePool
from models.base_model import BaseModel
from models import networks
from models.registry import generator_names
import torch.nn.functional as F


class SPIFModel(BaseModel):
    """
    SPIF (Structure-Preserving Image Fusion) model based on CycleGAN.

    Extends CycleGAN with an original-resolution fine-structure gradient loss,
    a brief coarse intensity-distribution guide, and cross-channel feature
    fusion for dual-channel input. Designed for virtual H&E staining from BF + AF.

    The model training requires '--dataset_mode dual_channel' dataset.
    By default, ``--spif_version v2`` selects the refined forward and reverse
    generators, with a basic PatchGAN discriminator and an LSGAN objective.

    Core innovations:
    1. Single-stage competitive fusion — BF and AF are jointly reweighted in
       both channel and spatial dimensions at H/2 before shared encoding.
    2. Original-resolution gradients preserve fine structures, while a
       faster-decaying coarse correlation loss guides the large-scale intensity
       distribution without regressing exact BF/AF-to-H&E brightness values.

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument("--lambda_A", type=float, default=10.0, help="weight for cycle loss (A -> B -> A)")
            parser.add_argument("--lambda_B", type=float, default=10.0, help="weight for cycle loss (B -> A -> B)")
            parser.add_argument(
                "--lambda_identity",
                type=float,
                default=0.5,
                help="use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1",
            )
            parser.add_argument("--lambda_content", type=float, default=1.0,
                                help="overall weight for SPIF content loss. Set to 0 to disable "
                                     "the entire structure-preserving loss (useful for ablation or fine-tuning)")
            parser.add_argument("--lambda_sp", type=float, default=1.0,
                                help="initial weight of the early-training SPIF loss")
            parser.add_argument(
                "--lambda_intensity",
                dest="lambda_intensity", type=float, default=0.5,
                help=(
                    "relative weight of the coarse intensity-distribution term"
                ),
            )
            parser.add_argument(
                "--sp_grad_decay_fraction",
                dest="sp_grad_decay_fraction", type=float, default=0.1,
                help=(
                    "fraction of total epochs for fine-structure gradient guidance"
                ),
            )
            parser.add_argument(
                "--sp_intensity_decay_fraction", type=float, default=0.025,
                help=(
                    "fraction of total epochs for coarse intensity-distribution "
                    "guidance; should normally be shorter than the gradient window"
                ),
            )
            parser.add_argument(
                "--intensity_pool_size", type=int, default=8,
                help="coarse grid size used by the intensity-map correlation",
            )
            parser.add_argument("--grad_channel_weights", type=float, nargs="+", default=None,
                                help="source-domain per-channel weights for gradient fusion and "
                                     "the original-style intensity channel mean. "
                                     "e.g. '--grad_channel_weights 0.5 0.5' for equal weighting. "
                                     "Weights must match input_nc and are normalized to sum=1. "
                                     "Target-domain channels always use equal weights.")

        return parser

    def __init__(self, opt):
        """Initialize the SPIF model.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = [
            "D_A", "G_A", "cycle_A", "idt_A", "D_B", "G_B",
            "cycle_B", "idt_B", "sp_grad_A", "sp_grad_B",
            "sp_int_A", "sp_int_B", "sp_grad_weight", "sp_int_weight",
            "content",
        ]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ["real_A", "fake_B", "rec_A"]
        visual_names_B = ["real_B", "fake_A", "rec_B"]
        self.dual_channel = (opt.input_nc == 2)
        if self.dual_channel:
            visual_names_A = ["real_A_bf", "real_A_af", "fake_B", "rec_A_bf", "rec_A_af"]
            visual_names_B = ["real_B", "fake_A_bf", "fake_A_af", "rec_B"]
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_B(A)
            visual_names_A.append("idt_B")
            visual_names_B.append("idt_A")

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ["G_A", "G_B", "D_A", "D_B"]
        else:  # during test time, only load Gs
            self.model_names = ["G_A", "G_B"]

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        netG_A_type, netG_B_type = generator_names(opt.model, vars(opt))
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, netG_A_type, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, netG_B_type, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)

        if self.isTrain:
            if opt.lambda_identity > 0.0 and opt.input_nc != opt.output_nc:
                print(f"WARNING: lambda_identity={opt.lambda_identity} requires input_nc==output_nc, "
                      f"but got input_nc={opt.input_nc}, output_nc={opt.output_nc}. "
                      "Disabling identity loss.")
                opt.lambda_identity = 0.0
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self._init_content_loss_options(opt)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.update_discriminators = True

    def set_discriminator_updates(self, enabled):
        """Enable or disable discriminator optimization during fine-tuning."""
        self.update_discriminators = bool(enabled)
        if not self.update_discriminators:
            self.set_requires_grad([self.netD_A, self.netD_B], False)

    def _init_content_loss_options(self, opt):
        """Initialize and validate options shared by all SPIF loss variants."""
        self.lambda_content = opt.lambda_content
        self.lambda_sp = opt.lambda_sp
        self.lambda_intensity = opt.lambda_intensity
        self.sp_grad_decay_fraction = opt.sp_grad_decay_fraction
        self.sp_intensity_decay_fraction = getattr(
            opt, "sp_intensity_decay_fraction", 0.025
        )
        self.intensity_pool_size = getattr(opt, "intensity_pool_size", 8)
        self.grad_channel_weights = None
        self.current_epoch = max(1, int(getattr(opt, "epoch_count", 1)))

        for name, value in (
            ("--lambda_content", self.lambda_content),
            ("--lambda_sp", self.lambda_sp),
            ("--lambda_intensity", self.lambda_intensity),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite, non-negative value.")
        for name, value in (
            ("--sp_grad_decay_fraction", self.sp_grad_decay_fraction),
            ("--sp_intensity_decay_fraction", self.sp_intensity_decay_fraction),
        ):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1].")
        if self.sp_intensity_decay_fraction > self.sp_grad_decay_fraction:
            raise ValueError(
                "--sp_intensity_decay_fraction must not exceed "
                "--sp_grad_decay_fraction."
            )

        self.sp_gradient_loss = FineStructureGradientLoss().to(self.device)
        self.sp_intensity_loss = CoarseIntensityDistributionLoss(
            pool_size=self.intensity_pool_size,
        ).to(self.device)

        if opt.grad_channel_weights is None:
            return

        w = torch.tensor(opt.grad_channel_weights, dtype=torch.float32)
        if w.numel() != opt.input_nc:
            raise ValueError(
                "--grad_channel_weights describes source-domain channels and must "
                f"contain input_nc={opt.input_nc} values, got {w.numel()}."
            )
        if not torch.isfinite(w).all():
            raise ValueError("--grad_channel_weights must contain only finite values.")
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError(
                "--grad_channel_weights must be non-negative and have a positive sum."
            )
        self.grad_channel_weights = w / w.sum()

    def set_current_epoch(self, epoch):
        """Update the epoch-based SP schedule (identical across DDP ranks)."""
        epoch = int(epoch)
        if epoch < 1:
            raise ValueError("Current epoch must be at least 1.")
        self.current_epoch = epoch

    def _effective_sp_weights(self):
        total_epochs = int(self.opt.n_epochs + self.opt.n_epochs_decay)
        gradient_schedule = frontloaded_cosine_decay(
            self.current_epoch, total_epochs, self.sp_grad_decay_fraction
        )
        intensity_schedule = frontloaded_cosine_decay(
            self.current_epoch,
            total_epochs,
            self.sp_intensity_decay_fraction,
        )
        common_weight = self.lambda_content * self.lambda_sp
        gradient_weight = common_weight * gradient_schedule
        intensity_weight = (
            common_weight * self.lambda_intensity * intensity_schedule
        )
        return gradient_weight, intensity_weight

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

    def compute_visuals(self):
        if getattr(self, 'dual_channel', False):
            self.real_A_bf = self.real_A[:, 0:1, :, :].repeat(1, 3, 1, 1)
            self.real_A_af = self.real_A[:, 1:2, :, :].repeat(1, 3, 1, 1)
            self.rec_A_bf = self.rec_A[:, 0:1, :, :].repeat(1, 3, 1, 1)
            self.rec_A_af = self.rec_A[:, 1:2, :, :].repeat(1, 3, 1, 1)
            self.fake_A_bf = self.fake_A[:, 0:1, :, :].repeat(1, 3, 1, 1)
            self.fake_A_af = self.fake_A[:, 1:2, :, :].repeat(1, 3, 1, 1)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_content = self.content_loss()
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_content
        # print('self.loss_idt_A -----> ',self.loss_idt_A, 'self.loss_idt_B -----> ',self.loss_idt_B, 'content_loss_value -----> ',content_loss_value)
        self.loss_G.backward()

    def content_loss(self):
        """Return independently scheduled fine-gradient and coarse-intensity terms.

        The gradient term compares unnormalized Sobel magnitudes only at the
        original image resolution to preserve fine structures. The intensity
        term follows the original channel-mean and Gaussian-low-pass path, then
        minimizes one minus the coarse normalized correlation. It guides the
        large-scale distribution of bright and dark regions without matching
        absolute brightness, and its faster epoch schedule removes the guidance
        after the initial training stage.
        """
        gradient_weight, intensity_weight = self._effective_sp_weights()
        self.loss_sp_grad_weight = gradient_weight
        self.loss_sp_int_weight = intensity_weight
        zero = self.fake_B.new_zeros(())

        if gradient_weight > 0:
            # Source-domain channel weights apply to real_A in the forward
            # direction and fake_A in the reverse direction. H&E channels use
            # equal weights.
            self.loss_sp_grad_A = self.sp_gradient_loss(
                self.real_A,
                self.fake_B,
                source_weights=self.grad_channel_weights,
            )
            self.loss_sp_grad_B = self.sp_gradient_loss(
                self.real_B,
                self.fake_A,
                translated_weights=self.grad_channel_weights,
            )
        else:
            self.loss_sp_grad_A = zero
            self.loss_sp_grad_B = zero

        if intensity_weight > 0:
            # Match the original paper's channel averaging before low-pass
            # filtering. AF is expected to have been inverted during dataset
            # preparation so its foreground/background brightness arrangement
            # is comparable to BF, as in the original preprocessing pipeline.
            self.loss_sp_int_A = self.sp_intensity_loss(
                self.real_A,
                self.fake_B,
                source_weights=self.grad_channel_weights,
            )
            self.loss_sp_int_B = self.sp_intensity_loss(
                self.real_B,
                self.fake_A,
                translated_weights=self.grad_channel_weights,
            )
        else:
            self.loss_sp_int_A = zero
            self.loss_sp_int_B = zero

        return gradient_weight * (
            self.loss_sp_grad_A + self.loss_sp_grad_B
        ) + intensity_weight * (
            self.loss_sp_int_A + self.loss_sp_int_B
        )


    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        if not self.update_discriminators:
            # Keep the logged losses explicit instead of carrying values from a
            # previous epoch. The discriminators remain usable in backward_G,
            # but their parameters receive no gradients or optimizer updates.
            self.loss_D_A = 0.0
            self.loss_D_B = 0.0
            return
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights


def frontloaded_cosine_decay(epoch, total_epochs, active_fraction=0.1):
    """Return a smooth early-training weight that reaches zero at a cutoff.

    The schedule is expressed purely in epochs, so its behavior is independent
    of batch size, dataset size, the last partial batch, and DDP world size.
    Epoch 1 starts at one; ``ceil(total_epochs * active_fraction)`` is the
    first epoch whose weight is zero.
    """
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive.")
    if not math.isfinite(active_fraction) or not 0 < active_fraction <= 1:
        raise ValueError("active_fraction must be in (0, 1].")
    if epoch < 1:
        raise ValueError("epoch must be at least 1.")

    active_epochs = max(1, math.ceil(total_epochs * active_fraction))
    if active_epochs == 1:
        return 1.0 if epoch == 1 else 0.0
    if epoch >= active_epochs:
        return 0.0

    progress = (epoch - 1) / (active_epochs - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class FineStructureGradientLoss(torch.nn.Module):
    """Original-resolution gradient-magnitude loss for fine structures.

    Sobel filtering uses reflection padding so crop borders do not become
    artificial edges. Per-channel magnitudes are fused without per-image
    normalization, so nearly uniform regions cannot have weak noise amplified
    to unit energy. No image pyramid or spatial downsampling is used.
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive.")

        self.eps = float(eps)
        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[-1.0, 0.0, 1.0],
                 [-2.0, 0.0, 2.0],
                 [-1.0, 0.0, 1.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3) / 8.0,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[-1.0, -2.0, -1.0],
                 [0.0, 0.0, 0.0],
                 [1.0, 2.0, 1.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3) / 8.0,
        )

    def _channel_weights(self, x, channel_weights):
        channels = x.shape[1]
        if channel_weights is None:
            return x.new_full((channels,), 1.0 / channels)

        weights = torch.as_tensor(
            channel_weights, device=x.device, dtype=x.dtype
        ).flatten()
        if weights.numel() != channels:
            raise ValueError(
                "Gradient channel weights do not match the tensor channel "
                f"count: {weights.numel()} vs {channels}."
            )
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError(
                "Gradient channel weights must be finite and non-negative."
            )
        if weights.sum() <= 0:
            raise ValueError("Gradient channel weights must have a positive sum.")
        return weights / weights.sum()

    def gradient_map(self, x, channel_weights=None):
        """Return one unnormalized structural gradient map per image."""
        if x.ndim != 4 or x.shape[1] < 1:
            raise ValueError("Expected an NCHW tensor with at least one channel.")
        if min(x.shape[-2:]) < 2:
            raise ValueError("Sobel reflection padding requires H and W >= 2.")

        # Keep derivative filtering in float32 under AMP; casting remains
        # differentiable and avoids eps underflow in float16.
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        channels = work.shape[1]
        kernel_x = self.sobel_x.to(dtype=work.dtype).expand(
            channels, 1, 3, 3
        )
        kernel_y = self.sobel_y.to(dtype=work.dtype).expand(
            channels, 1, 3, 3
        )
        padded = F.pad(work, (1, 1, 1, 1), mode="reflect")
        grad_x = F.conv2d(padded, kernel_x, groups=channels)
        grad_y = F.conv2d(padded, kernel_y, groups=channels)

        # Subtract eps so a perfectly constant image produces an exact zero.
        magnitude = torch.sqrt(
            grad_x.square() + grad_y.square() + self.eps ** 2
        ) - self.eps
        weights = self._channel_weights(work, channel_weights).view(
            1, channels, 1, 1
        )
        return torch.sum(magnitude * weights, dim=1, keepdim=True)

    def forward(self, source, translated, source_weights=None,
                translated_weights=None):
        if source.shape[0] != translated.shape[0]:
            raise ValueError("Source and translated batches must have equal size.")
        if source.shape[-2:] != translated.shape[-2:]:
            raise ValueError(
                "Source and translated images must have equal spatial size."
            )

        source_gradient = self.gradient_map(source, source_weights)
        translated_gradient = self.gradient_map(
            translated, translated_weights
        )
        return F.l1_loss(translated_gradient, source_gradient)


class CoarseIntensityDistributionLoss(torch.nn.Module):
    """Guide the coarse spatial distribution of bright and dark tissue regions.

    This keeps the original SPIF intensity pipeline -- channel averaging and a
    21x21 Gaussian low-pass filter by default -- but does not regress absolute
    brightness. The low-pass maps are pooled to a small grid, spatially
    centered, and RMS-normalized. The loss minimizes one minus their correlation,
    providing direction-based supervision without matching absolute brightness.
    Its independently decaying epoch weight confines this guidance to the
    beginning of training.
    """

    def __init__(self, radius=10, sigma=5.0, pool_size=8, eps=1e-6,
                 min_source_rms=1e-4):
        super().__init__()
        radius = int(radius)
        sigma = float(sigma)
        pool_size = int(pool_size)
        eps = float(eps)
        min_source_rms = float(min_source_rms)
        if radius < 1:
            raise ValueError("Intensity Gaussian radius must be positive.")
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("Intensity Gaussian sigma must be positive.")
        if pool_size < 2:
            raise ValueError("Intensity pool size must be at least 2.")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("Intensity eps must be finite and positive.")
        if not math.isfinite(min_source_rms) or min_source_rms < 0:
            raise ValueError(
                "Minimum source RMS must be finite and non-negative."
            )

        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        gaussian_1d = torch.exp(-0.5 * (coords / sigma).square())
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        self.radius = radius
        self.pool_size = pool_size
        self.eps = eps
        self.min_source_rms = min_source_rms
        self.register_buffer(
            "gaussian_kernel_x", gaussian_1d.view(1, 1, 1, -1)
        )
        self.register_buffer(
            "gaussian_kernel_y", gaussian_1d.view(1, 1, -1, 1)
        )

    @staticmethod
    def _channel_weights(x, channel_weights):
        channels = x.shape[1]
        if channel_weights is None:
            return x.new_full((channels,), 1.0 / channels)

        weights = torch.as_tensor(
            channel_weights, device=x.device, dtype=x.dtype
        ).flatten()
        if weights.numel() != channels:
            raise ValueError(
                "Intensity channel weights do not match the tensor channel "
                f"count: {weights.numel()} vs {channels}."
            )
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError(
                "Intensity channel weights must be finite and non-negative."
            )
        if weights.sum() <= 0:
            raise ValueError(
                "Intensity channel weights must have a positive sum."
            )
        return weights / weights.sum()

    def coarse_intensity_map(self, x, channel_weights=None):
        """Return the original channel-mean low pass on a coarse spatial grid."""
        if x.ndim != 4 or x.shape[1] < 1:
            raise ValueError("Expected an NCHW tensor with at least one channel.")
        if min(x.shape[-2:]) <= self.radius:
            raise ValueError(
                "Reflection padding for the intensity term requires image "
                f"height and width greater than radius={self.radius}."
            )
        if min(x.shape[-2:]) < self.pool_size:
            raise ValueError(
                "Intensity pool size cannot exceed the input height or width."
            )

        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        weights = self._channel_weights(work, channel_weights).view(
            1, work.shape[1], 1, 1
        )
        intensity = torch.sum(work * weights, dim=1, keepdim=True)
        kernel_x = self.gaussian_kernel_x.to(dtype=work.dtype)
        kernel_y = self.gaussian_kernel_y.to(dtype=work.dtype)
        low = F.conv2d(
            F.pad(
                intensity, (self.radius, self.radius, 0, 0), mode="reflect"
            ),
            kernel_x,
        )
        low = F.conv2d(
            F.pad(
                low, (0, 0, self.radius, self.radius), mode="reflect"
            ),
            kernel_y,
        )
        return F.adaptive_avg_pool2d(
            low, output_size=(self.pool_size, self.pool_size)
        )

    def _correlation_and_validity(self, source, translated,
                                  source_weights=None,
                                  translated_weights=None):
        if source.shape[0] != translated.shape[0]:
            raise ValueError("Source and translated batches must have equal size.")
        if source.shape[-2:] != translated.shape[-2:]:
            raise ValueError(
                "Source and translated images must have equal spatial size."
            )

        source_low = self.coarse_intensity_map(source, source_weights)
        translated_low = self.coarse_intensity_map(
            translated, translated_weights
        )
        source_centered = source_low - source_low.mean(
            dim=(-2, -1), keepdim=True
        )
        translated_centered = translated_low - translated_low.mean(
            dim=(-2, -1), keepdim=True
        )
        source_var = source_centered.square().mean(dim=(-2, -1, -3))
        translated_var = translated_centered.square().mean(dim=(-2, -1, -3))
        covariance = (source_centered * translated_centered).mean(
            dim=(-2, -1, -3)
        )
        denominator = torch.sqrt(
            (source_var + self.eps) * (translated_var + self.eps)
        )
        correlation = (covariance / denominator).clamp(-1.0, 1.0)
        valid = source_var > self.min_source_rms ** 2
        return correlation, valid

    def coarse_correlation(self, source, translated, source_weights=None,
                           translated_weights=None):
        """Return one coarse normalized intensity correlation per image."""
        correlation, _ = self._correlation_and_validity(
            source,
            translated,
            source_weights=source_weights,
            translated_weights=translated_weights,
        )
        return correlation

    def forward(self, source, translated, source_weights=None,
                translated_weights=None):
        correlation, valid = self._correlation_and_validity(
            source,
            translated,
            source_weights=source_weights,
            translated_weights=translated_weights,
        )
        penalties = 1.0 - correlation
        valid_weights = valid.to(dtype=penalties.dtype)
        return torch.sum(penalties * valid_weights) / valid_weights.sum().clamp_min(1)
