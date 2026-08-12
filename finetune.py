"""Transfer learning / fine-tuning script.

Loads a pretrained model checkpoint and fine-tunes it on a (typically smaller)
target dataset. Uses lower learning rate and fewer epochs by default.

Key features:
  --pretrained_name   Name of the pretrained experiment (checkpoints/<name>/)
                      If not set, falls back to --name (same experiment, continue training)
  --lr                Learning rate for fine-tuning (default: 0.00005 = 1/4 of train LR)
  --freeze_D_epochs   Keep discriminator frozen for first N epochs (default: 5)
                      Helps prevent D from overfitting on the small dataset early on.

Usage examples:
  # Fine-tune SPIF from mouse kidney checkpoint on a lung dataset
  python finetune.py \
      --dataroot ./datasets/lung_dual_BF_AF_HE_256 \
      --name lung_spif_finetune \
      --pretrained_name mouse_kidney_dual_spif_256 \
      --model spif --dataset_mode dual_channel --input_nc 2 --output_nc 3

  # Fine-tune CycleGAN on a small stained-slide subset
  python finetune.py \
      --dataroot ./datasets/small_he_stain --name he_cyclegan_ft \
      --pretrained_name mouse_kidney_cyclegan --model cycle_gan
"""

import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import init_ddp, cleanup_ddp
from pathlib import Path
import torch


if __name__ == "__main__":
    import sys

    # ---------- default fine-tuning config ----------
    # Only applied when the user hasn't passed explicit CLI arguments.
    BASE = [
        "finetune.py",
        "--dataroot", "C:/Users/30927/Desktop/Multi-layer-CycleGAN/datasets/finetune_mouse_kidney_dual_BF_AF_HE_512",
        "--name", "finetune_mouse_kidney",
        "--pretrained_name", "mouse_kidney_dual_spif_256",            # set to the source experiment name
        "--model", "spif",
        "--dataset_mode", "dual_channel",
        "--input_nc", "2", "--output_nc", "3",
        "--lambda_identity", "0",
        # -- fine-tuning hyperparams --
        "--lr", "0.00010",                   # 1/4 of regular training LR
        "--n_epochs", "20",                  # fewer epochs
        "--n_epochs_decay", "20",
        "--batch_size", "4",
        "--load_size", "286", "--crop_size", "256",
        "--lambda_low_freq", "0.5",
        "--lambda_content", "0",             # disable content loss during fine-tuning
        "--save_epoch_freq", "5",
        # -- freeze D early --
        "--freeze_D_epochs", "5",
    ]

    # Merge user-supplied args (if any) over the defaults
    user_args = sys.argv[1:]
    if user_args:
        # Use user args entirely; base is ignored except for script name
        final_args = ["finetune.py"] + user_args
    else:
        final_args = BASE

    sys.argv = final_args

    # Parse options
    opt = TrainOptions().parse()
    opt.device = init_ddp()

    # --- Validate pretrained checkpoint ---
    pretrained_name = getattr(opt, "pretrained_name", None) or opt.name
    pretrained_dir = Path(opt.checkpoints_dir) / pretrained_name
    if pretrained_name != opt.name and not pretrained_dir.is_dir():
        print(f"WARNING: pretrained checkpoint dir not found: {pretrained_dir}")
        print(f"  Training from scratch (no pretrained weights loaded).")
    else:
        print(f"Pretrained checkpoint: {pretrained_dir}")

    # --- Dataset ---
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    opt.data_size = dataset_size
    print(f"Fine-tuning dataset size = {dataset_size}")

    # --- Model ---
    model = create_model(opt)

    model.setup(opt)

    # Load pretrained weights AFTER setup (setup calls init_net which re-inits weights)
    if pretrained_name != opt.name:
        print(f"Loading pretrained weights from '{pretrained_name}' ...")
        model.save_dir = pretrained_dir  # temporarily redirect save_dir
        model.load_networks(opt.epoch)
        model.save_dir = Path(opt.checkpoints_dir) / opt.name  # restore

    # Freeze-D epochs
    freeze_D_epochs = int(getattr(opt, "freeze_D_epochs", 0))
    if freeze_D_epochs > 0:
        print(f"Discriminator frozen for first {freeze_D_epochs} epoch(s)")

    # CUT: lazy init on first batch (data_dependent_initialize → setup → parallelize)
    is_cut = opt.model in ("cut", "cut_stack", "sincut")

    # --- Visualizer ---
    visualizer = Visualizer(opt)
    total_iters = (opt.epoch_count - 1) * dataset_size

    # --- Training loop ---
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        # Freeze / unfreeze discriminator
        freeze_D = (epoch - opt.epoch_count) < freeze_D_epochs
        if hasattr(model, "netD_A"):
            model.set_requires_grad(model.netD_A, not freeze_D)
        if hasattr(model, "netD_B"):
            model.set_requires_grad(model.netD_B, not freeze_D)
        if hasattr(model, "netD") and not hasattr(model, "netD_A"):
            model.set_requires_grad(model.netD, not freeze_D)

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            t_data = iter_start_time - iter_data_time
            print_now = (total_iters % opt.print_freq == 0)

            # CUT lazy init on first batch
            if is_cut and epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)
                model.parallelize()

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            opt.counter = total_iters
            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, total_iters, save_result)

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
    print("Fine-tuning complete!")
