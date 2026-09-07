"""Test script for single-channel (BF or AF) to H&E inference.

This is a variant of test_dual.py for single-channel input (--input_nc 1).
Loads a trained G_A generator and runs inference on testA/ single-channel images.

Example:
    python test_single.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util.util import init_ddp

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == "__main__":

    if len(sys.argv) == 1:
        sys.argv = [
            "infer_single.py",
            "--dataroot", "./datasets/test_spleen_single_AF_HE_1024_256",
            "--name", "spleen_single_AF_1024_256",
            "--model", "test",
            "--model_suffix", "_A",
            "--dataset_mode", "single",
            "--input_nc", "3",
            "--output_nc", "3",
            "--no_dropout",
            "--preprocess", "none",
            "--results_dir", "./results",
            "--num_test", "10000",
            "--epoch", "latest",
            "--serial_batches",
            "--no_flip",
            "--num_threads", "0",
            "--batch_size", "1",
            "--gpu_ids", "0",
        ]

    opt = TestOptions().parse()
    opt.device = init_ddp(opt.gpu_ids)

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
    total_images = min(len(dataset), opt.num_test)
    print(f"Inference dataset size = {len(dataset)}; processing = {total_images}")
    processed = 0
    for data in dataset:
        if processed >= total_images:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        image_paths = model.get_image_paths()
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]
        remaining = total_images - processed
        batch_count = min(len(image_paths), remaining)
        for batch_index in range(batch_count):
            batch_visuals = {
                name: image[batch_index:batch_index + 1]
                for name, image in visuals.items()
            }
            save_images(
                webpage,
                batch_visuals,
                [image_paths[batch_index]],
                aspect_ratio=opt.aspect_ratio,
                width=opt.display_winsize,
            )
        processed += batch_count
        print(f"Processed {processed}/{total_images} images")
    webpage.save()
