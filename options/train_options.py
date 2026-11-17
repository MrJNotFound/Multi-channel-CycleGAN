from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--save_visuals_epoch_freq', type=int, default=0,
                            help='fine-tuning: save one set of visuals at the end of every N epochs; 0 disables it')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--stop_after_epoch', type=int, default=None,
                            help='stop this invocation after the given epoch without changing n_epochs or n_epochs_decay')
        parser.add_argument('--complete_epoch_state', action='store_true',
                            help='atomically save a full latest G/D/optimizer/RNG/pool bundle every epoch')
        parser.add_argument('--weights_only', action='store_true',
                            help='CycleGAN-style checkpoints: save G/D weights and small progress JSON only')
        parser.add_argument('--cyclegan_runtime', action='store_true',
                            help='use original CycleGAN backend settings: cuDNN benchmark/TF32 on, strict determinism off')
        parser.add_argument('--record_initial_hashes', action='store_true',
                            help='record initial network hashes once to audit same-seed initialization')
        parser.add_argument('--fixed_preview_manifest', type=str, default=None,
                            help='fixed BF/AF diagnostic locations and deterministic transform JSON')
        parser.add_argument('--fixed_preview_epochs', type=int, nargs='+', default=[])
        parser.add_argument('--pause_file', type=str, default=None,
                            help='if this file exists, pause after saving the current epoch')
        parser.add_argument('--deterministic', action='store_true',
                            help='require deterministic PyTorch algorithms and disable TF32/cuDNN benchmarking')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam (CUT compatibility)')
        parser.add_argument('--seed', type=int, default=None, help='seed model initialization and independent per-epoch data randomness')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        parser.add_argument('--counter', type=int, default=0, help='running count of training samples seen (set automatically)')
        parser.add_argument('--data_size', type=int, default=0, help='number of training samples in the dataset (set automatically)')

        # Fine-tuning
        parser.add_argument('--pretrained_name', type=str, default=None,
                            help='name of the pretrained experiment (checkpoints/<name>/). '
                            'Load weights from a different experiment for fine-tuning.')
        parser.add_argument('--freeze_D_epochs', type=int, default=0,
                            help='keep discriminator frozen for first N epochs (fine-tuning)')

        self.isTrain = True
        return parser
