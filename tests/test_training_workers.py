"""Exercise Windows spawn through the actual unified training loop."""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from data.base_dataset import get_transform
from gui.launcher import build_option_parser
from models.base_model import BaseModel
from training.train import run_training


class RecordingModel(BaseModel):
    """Keep the regression focused on data workers rather than GAN compute."""

    def __init__(self, opt):
        super().__init__(opt)
        self.batches = 0
        self.schedulers = []

    def setup(self, opt):
        pass

    def set_input(self, batch):
        assert batch['A'].shape == (1, 3, 32, 32)
        assert batch['B'].shape == (1, 3, 32, 32)

    def forward(self):
        pass

    def optimize_parameters(self):
        self.batches += 1

    def update_learning_rate(self):
        pass


def make_options(root, preprocess):
    parser, _ = build_option_parser('train', 'cycle_gan_single_none_base', 'unaligned')
    opt = parser.parse_args([
        '--dataroot', str(root), '--name', 'worker_test',
        '--checkpoints_dir', str(root / 'checkpoints'),
        '--num_threads', '1', '--batch_size', '1',
        '--n_epochs', '1', '--n_epochs_decay', '0',
        '--save_epoch_freq', '1', '--save_latest_freq', '0',
        '--display_freq', '0', '--load_size', '32', '--crop_size', '32',
        '--preprocess', preprocess, '--serial_batches', '--no_flip',
    ])
    opt.isTrain = True
    opt.device = torch.device('cpu')
    opt.gpu_ids = []
    return opt


class TrainingWorkerTests(unittest.TestCase):
    def test_training_with_html_and_spawn_workers(self):
        for mode in ('resize_and_crop', 'none', 'scale_width_and_crop'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for domain in ('trainA', 'trainB'):
                    (root / domain).mkdir()
                    for index in range(2):
                        Image.new('RGB', (32, 32), (40, 80, 120)).save(
                            root / domain / f'{index}.png'
                        )
                opt = make_options(root, mode)
                model = RecordingModel(opt)
                # Explicit spawn also reproduces Windows semantics on Linux.
                from torch.utils.data import DataLoader

                def spawn_loader(*args, **kwargs):
                    return DataLoader(*args, multiprocessing_context='spawn', **kwargs)

                with patch('training.train.create_model', return_value=model), patch(
                    'data.torch.utils.data.DataLoader', side_effect=spawn_loader
                ):
                    run_training(opt, [])
                self.assertEqual(model.batches, 2)
                self.assertFalse(hasattr(opt, 'visualizer'))
                self.assertTrue((root / 'checkpoints/worker_test/web').is_dir())

    def test_fixed_crop_and_flip_survive_serialization(self):
        opt = make_options(Path('.'), 'scale_width_and_crop')
        opt.no_flip = False
        transform = get_transform(opt, {'crop_pos': (1, 2), 'flip': True})
        restored = pickle.loads(pickle.dumps(transform))
        image = Image.new('RGB', (40, 48), (40, 80, 120))
        self.assertTrue(torch.equal(transform(image), restored(image)))


if __name__ == '__main__':
    unittest.main()
