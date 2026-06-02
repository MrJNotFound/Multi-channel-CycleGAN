"""Test script for single-channel (BF or AF) to H&E inference.

This is a variant of test_dual.py for single-channel input (--input_nc 1).
Loads a trained G_A generator and runs inference on testA/ single-channel images.

Example:
    python test_single.py
"""

import sys
from pathlib import Path
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == "__main__":

    sys.argv = [
        "infer_single.py",
        "--dataroot", "./datasets/mouse_kidney_dual_BF_AF_HE_256/testA_BF",
        "--name", "mouse_kidney_Trans_HE_256",
        "--model", "test",
        "--model_suffix", "_A",
        "--input_nc", "3",
        "--output_nc", "3",
        "--no_dropout",
        "--preprocess", "none",
        "--results_dir", r"C:\Users\30927\Desktop\img_histology\stain_kidney\kidney_fake\Trans_kidney_20x_24_1024",
        "--num_test", "10000",
        "--epoch", "latest",
    ]

    opt = TestOptions().parse()
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    web_dir = Path(opt.results_dir) / opt.name / f"{opt.phase}_{opt.epoch}"
    if opt.load_iter > 0:
        web_dir = Path(f"{web_dir}_iter{opt.load_iter}")
    print(f"creating web directory {web_dir}")
    webpage = html.HTML(web_dir, f"Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}")
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()

        img_path = model.get_image_paths()
        if i % 5 == 0:
            print(f"processing ({i:04d})-th image... {img_path}")
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()
