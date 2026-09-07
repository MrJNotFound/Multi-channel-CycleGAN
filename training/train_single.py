"""Training script for RGB-to-RGB image translation with identity loss.

This is a variant of train_dual.py for RGB input and output (--input_nc 3, --output_nc 3).
Uses standard unaligned dataset mode — both domains contain RGB images.
Identity loss is enabled (default lambda_identity=0.5) since input_nc == output_nc.

Example:
    python train_single.py
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

    # Keep convenient IDE defaults, while allowing the GUI and normal CLI
    # invocations to supply their own complete configuration.
    if len(sys.argv) == 1:
        sys.argv = [
            "train_single.py",
            "--dataroot", "./datasets/train_liver_single_AF_HE_512",
            "--name", "liver_single_AF_HE_512_256",
            "--model", "cycle_gan",
            "--dataset_mode", "unaligned",
            "--input_nc", "3",
            "--output_nc", "3",
            "--lambda_identity", "0.5",
            "--batch_size", "8",
            "--n_epochs", "100",
            "--n_epochs_decay", "100",
            "--load_size", "286",
            "--crop_size", "256",
            "--num_threads", "0",
            "--gpu_ids", "0",
        ]

    opt = TrainOptions().parse()
    opt.device = init_ddp(opt.gpu_ids)
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    opt.data_size = dataset_size
    print(f"The number of training images = {dataset_size}")

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
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
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        if hasattr(model, "set_current_epoch"):
            model.set_current_epoch(epoch)

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            t_data = iter_start_time - iter_data_time
            print_now = (total_iters % opt.print_freq == 0)

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            opt.counter = total_iters
            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, total_iters, save_result)

            if print_now:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(total_iters, losses)

            if total_iters % opt.save_latest_freq == 0:
                print(f"saving the latest model (epoch {epoch}, total_iters {total_iters})")
                save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        model.update_learning_rate()

        if epoch % opt.save_epoch_freq == 0:
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
