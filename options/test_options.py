from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.add_argument('--source_model', type=str, default='cycle_gan_single_none_base',
                            help='training model family that produced the generator checkpoint')
        parser.add_argument('--save_inputs', action='store_true',
                            help='also save input/auxiliary visuals next to generated images')
        parser.add_argument('--make_report', action='store_true',
                            help='create an HTML report for all saved inference visuals')
        parser.add_argument('--output_suffix', type=str, default='_fake',
                            help='suffix appended to generated output filenames')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
