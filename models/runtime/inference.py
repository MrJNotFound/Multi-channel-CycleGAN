import numpy as np
import torch

from models.base_model import BaseModel
from models import networks
from models.registry import legacy_name


class TestModel(BaseModel):
    """This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test instruction for more details.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        assert not is_train, "TestModel cannot be used during training time"

        # TestModel is a generic one-way runner.  Let the module that produced
        # the checkpoint contribute its defaults and architecture options so a
        # new model plugin can be used by the unified inference entry without
        # adding another dedicated script.
        source_opt, _ = parser.parse_known_args()
        source_model = getattr(source_opt, "source_model", "cycle_gan")
        if source_model and source_model != "test":
            from models import get_option_setter

            parser = get_option_setter(source_model)(parser, False)

        parser.set_defaults(model="test", dataset_mode="single")
        if not any(action.dest == "model_suffix" for action in parser._actions):
            parser.add_argument(
                "--model_suffix",
                type=str,
                default="",
                help=(
                    "network suffix in checkpoints_dir: "
                    "[epoch]_net_G[model_suffix].pth"
                ),
            )

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert not opt.isTrain
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.dual_channel = (opt.input_nc == 2)
        if legacy_name(getattr(opt, "source_model", "")) == "colorization":
            self.visual_names = ["real", "fake_rgb"]
        else:
            self.visual_names = (
                ["real_bf", "real_af", "fake"]
                if self.dual_channel
                else ["real", "fake"]
            )
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ["G" + opt.model_suffix]  # only generator is needed.
        netG_type = opt.netG
        source_model = legacy_name(getattr(opt, "source_model", "cycle_gan"))
        self.source_model = source_model
        # Generator selection is driven by the module that produced the
        # checkpoint, not by which dedicated inference script happened to be
        # launched. Stacked baselines therefore remain standard generators.
        use_spif_fusion = source_model == "spif"
        if use_spif_fusion:
            if opt.input_nc == 2 and netG_type in ("resnet_9blocks", "resnet_6blocks"):
                n_blocks = "9blocks" if "9blocks" in netG_type else "6blocks"
                if opt.spif_arch == "v2":
                    netG_type = f"spif_v2_{n_blocks}"
                elif opt.spif_arch == "legacy":
                    netG_type = f"spif_{n_blocks}"
            elif opt.output_nc == 2 and netG_type in ("resnet_9blocks", "resnet_6blocks"):
                n_blocks = "9blocks" if "9blocks" in netG_type else "6blocks"
                if opt.spif_arch == "v2":
                    netG_type = f"spif_dual_v2_{n_blocks}"
                elif opt.spif_arch == "legacy":
                    netG_type = f"spif_dual_{n_blocks}"
        norm_type = opt.norm
        no_antialias = True
        no_antialias_up = True
        if source_model in {"cut", "cut_stack", "sincut"}:
            norm_type = opt.normG if opt.normG else opt.norm
            no_antialias = opt.no_antialias
            no_antialias_up = opt.no_antialias_up
        self.netG = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            netG_type,
            norm_type,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            gpu_ids=[],
            no_antialias=no_antialias,
            no_antialias_up=no_antialias_up,
            opt=opt,
        )

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, "netG" + opt.model_suffix, self.netG)  # store netG in self.

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        self.real = input["A"].to(self.device)
        self.image_paths = input["A_paths"]

    def forward(self):
        """Run forward pass."""
        self.fake = self.netG(self.real)  # G(real)

    def result_image(self, batch_index=0):
        """Return one generated image using the source module's output adapter."""
        if self.source_model == "colorization":
            return self._colorization_image(batch_index)

        from util import util

        return util.tensor2im(self.fake[batch_index : batch_index + 1])

    def _colorization_image(self, batch_index):
        from skimage import color

        luminance = self.real[batch_index].detach().cpu().float().numpy()
        chroma = self.fake[batch_index].detach().cpu().float().numpy()
        lab = np.concatenate(
            [
                np.transpose((luminance + 1.0) * 50.0, (1, 2, 0)),
                np.transpose(chroma * 110.0, (1, 2, 0)),
            ],
            axis=2,
        )
        return (color.lab2rgb(lab) * 255.0).clip(0, 255).astype(np.uint8)

    def compute_visuals(self):
        if self.dual_channel:
            self.real_bf = self.real[:, 0:1, :, :].repeat(1, 3, 1, 1)
            self.real_af = self.real[:, 1:2, :, :].repeat(1, 3, 1, 1)
        if self.source_model == "colorization":
            converted = np.stack(
                [self._colorization_image(index) for index in range(len(self.real))]
            )
            self.fake_rgb = (
                torch.from_numpy(converted.transpose(0, 3, 1, 2))
                .float()
                .div(127.5)
                .sub(1.0)
            )

    def optimize_parameters(self):
        """No optimization for test model."""
        pass
