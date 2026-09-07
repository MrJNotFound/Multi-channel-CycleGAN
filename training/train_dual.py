"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train_dual.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train_dual.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import init_ddp, cleanup_ddp
import sys


if __name__ == "__main__":

    if len(sys.argv) == 1:
        sys.argv = [
            "train_dual.py",
            "--dataroot", "./datasets/train_liver_dual_BF_AF_HE_512",
            "--name", "liver_dual_spif_v2_512_256",
            "--model", "spif",
            "--spif_arch", "v2",
            "--dataset_mode", "dual_channel",
            "--input_nc", "2",
            "--output_nc", "3",
            "--lambda_identity", "0",
            # V2 keeps separate BF/AF encoders through H/2, then uses a shared
            # H/4 ResNet body. Batch 8 leaves safe headroom for the two cycle
            # passes and both discriminators on a 48 GB GPU.
            "--batch_size", "8",
            "--n_epochs", "100",
            "--n_epochs_decay", "100",
            "--load_size", "286",
            "--crop_size", "256",
            "--gpu_ids", "0",
            # Original-resolution gradients guide fine structures during the
            # first 10% of training. Coarse intensity correlation guides the
            # large-scale brightness distribution only at the beginning and
            # reaches zero at 2.5% of total epochs.
            "--lambda_sp", "10.0",
            "--lambda_intensity", "0.5",
            "--sp_grad_decay_fraction", "0.1",
            "--sp_intensity_decay_fraction", "0.025",
            "--intensity_pool_size", "8",
            # "--epoch", "200"
            # "--epoch_count", "201",
        ]

    opt = TrainOptions().parse()  # get training options
    opt.device = init_ddp(opt.gpu_ids)
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    opt.data_size = dataset_size
    print(f"The number of training images = {dataset_size}")

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    # Auto-detect resume epoch when continue_train
    if opt.continue_train and opt.epoch_count == 1:
        import re, glob
        ckpt_dir = Path(opt.checkpoints_dir) / opt.name
        epochs = set()
        for f in glob.glob(str(ckpt_dir / "*_net_G_A.pth")):
            m = re.match(r"(\d+)_net_", Path(f).name)
            if m:
                epochs.add(int(m.group(1)))
        if epochs:
            opt.epoch_count = max(epochs)
            print(f"Auto-detected resume epoch: {opt.epoch_count} "
                  f"(nearest save, actual may be -{opt.save_epoch_freq}+1 epochs)")

    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = (opt.epoch_count - 1) * dataset_size
    configured_final_epoch = opt.n_epochs + opt.n_epochs_decay
    final_epoch = configured_final_epoch
    if opt.stop_after_epoch is not None:
        if opt.stop_after_epoch < opt.epoch_count:
            raise ValueError(
                "--stop_after_epoch must be greater than or equal to "
                "--epoch_count."
            )
        final_epoch = min(opt.stop_after_epoch, configured_final_epoch)
        print(
            f"This invocation will stop after epoch {final_epoch}; "
            f"loss and learning-rate schedules remain configured for "
            f"{configured_final_epoch} epochs."
        )

    for epoch in range(opt.epoch_count, final_epoch + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()
        # Set epoch for DistributedSampler
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        if hasattr(model, "set_current_epoch"):
            model.set_current_epoch(epoch)

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            t_data = iter_start_time - iter_data_time
            print_now = (total_iters % opt.print_freq == 0)

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            opt.counter = total_iters
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, total_iters, save_result)

            if print_now:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(total_iters, losses)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print(f"saving the latest model (epoch {epoch}, total_iters {total_iters})")
                save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        model.update_learning_rate()  # update learning rates at the end of every epoch

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print(f"saving the model at the end of epoch {epoch}, iters {total_iters}")
            model.save_networks("latest")
            model.save_networks(epoch)

        print(f"End of epoch {epoch} / {configured_final_epoch} \t Time Taken: {time.time() - epoch_start_time:.0f} sec")

    if final_epoch < configured_final_epoch:
        print(
            f"Stopped after epoch {final_epoch} as requested "
            f"(--stop_after_epoch={opt.stop_after_epoch})."
        )

    cleanup_ddp()
