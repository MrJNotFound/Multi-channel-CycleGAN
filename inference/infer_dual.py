"""General-purpose test script for image-to-image translation.

Once you have trained your model with train_dual.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It creates the model and dataset from command-line options, then runs inference
for '--num_test' images and saves only the generated images. Batch size and data
loader settings are configurable; every image in a batch is saved.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test_dual.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test_dual.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test_dual.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util


if __name__ == "__main__":

    if len(sys.argv) == 1:
        sys.argv = [
            "infer_dual.py",
            "--dataroot", "./datasets/test_validation_spleen_dual_BF_AF_HE_512",
            "--name", "spleen_dual_spif_v2_1024_256",
            "--model", "test",
            "--model_suffix", "_A",
            "--dataset_mode", "dual_channel",
            "--input_nc", "2",
            "--output_nc", "3",
            "--spif_arch", "v2",
            # "--netG", "resnet_9blocks",  # explicitly set for stack model
            "--no_dropout",
            "--preprocess", "none",
            "--results_dir", "./results",
            "--num_test", "20000",
            "--epoch", "latest",
            "--serial_batches",
            "--no_flip",
            "--num_threads", "0",
            "--batch_size", "1",
            "--gpu_ids", "0",
        ]

    opt = TestOptions().parse()  # get test options
    opt.device = util.init_ddp(opt.gpu_ids)

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    # Keep the historical output layout for downstream stitching tools, but do
    # not create an HTML report or save input/reconstruction visualizations.
    web_dir = Path(opt.results_dir) / opt.name / f"{opt.phase}_{opt.epoch}"  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = Path(f"{web_dir}_iter{opt.load_iter}")
    output_dir = web_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"saving generated images to {output_dir}")
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    total_images = min(len(dataset), opt.num_test)
    print(f"Inference dataset size = {len(dataset)}; processing = {total_images}")
    processed = 0
    for data in dataset:
        if processed >= total_images:
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        image_paths = model.get_image_paths()
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]
        remaining = total_images - processed
        batch_count = min(len(image_paths), remaining)
        for batch_index in range(batch_count):
            image_name = f"{Path(image_paths[batch_index]).stem}_fake.png"
            generated = util.tensor2im(model.fake[batch_index:batch_index + 1])
            util.save_image(
                generated,
                output_dir / image_name,
                aspect_ratio=opt.aspect_ratio,
            )
        processed += batch_count
        print(f"Processed {processed}/{total_images} images")
