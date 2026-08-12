"""SPIF-Single (BF): gradient-domain content loss + BF single-channel input.

Standard ResNet generator, no dual heads, no SPIFFusion.
Isolates the contribution of SPIF's gradient-domain + low-frequency anchor loss
on brightfield input.

Dataset: datasets/train_mouse_kidney_single_BF_HE_512
"""

import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import init_ddp, cleanup_ddp

if __name__ == "__main__":
    import sys
    sys.argv = [
        "train_spif_bf.py",
        "--dataroot", "./datasets/train_mouse_kidney_single_BF_HE_512",
        "--name", "mouse_kidney_sp_bf_256",
        "--model", "spif_single",
        "--dataset_mode", "unaligned",
        "--input_nc", "3", "--output_nc", "3",
        "--load_size", "286", "--crop_size", "256",
        "--lambda_identity", "1",
        "--batch_size", "8",
        "--n_epochs", "100", "--n_epochs_decay", "100",
        "--lambda_low_freq", "0.5",
    ]

    opt = TrainOptions().parse()
    opt.device = init_ddp()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    opt.data_size = dataset_size
    print(f"The number of training images = {dataset_size}")

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = (opt.epoch_count - 1) * dataset_size

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

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

        print(f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} "
              f"\t Time Taken: {time.time() - epoch_start_time:.0f} sec")

    cleanup_ddp()
    print("Training complete!")
