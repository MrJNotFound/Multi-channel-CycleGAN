"""Transfer learning / fine-tuning entry point.

Fresh fine-tuning loads network weights from ``checkpoints/<pretrained_name>``
and starts a new optimizer schedule under ``checkpoints/<name>``. Passing
``--continue_train`` instead resumes the target experiment and takes precedence
over ``--pretrained_name``.

The script also supports a genuine discriminator warm-up freeze and guarantees
an end-of-epoch visual sample when ``--save_visuals_epoch_freq`` is positive.
"""

import re
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import create_dataset
from models import create_model
from options.train_options import TrainOptions
from util.util import cleanup_ddp, init_ddp
from util.visualizer import Visualizer


def _argument_was_set(arguments, option):
    """Return whether an option was supplied as ``--x value`` or ``--x=value``."""
    return option in arguments or any(arg.startswith(f"{option}=") for arg in arguments)


def _frequency_due(previous_total, current_total, frequency):
    """Detect crossing a frequency boundary even when batch increments skip it."""
    return frequency > 0 and previous_total // frequency < current_total // frequency


def _batch_size(data, fallback):
    """Infer the actual batch size, including a smaller final batch."""
    if isinstance(data, dict):
        for value in data.values():
            if torch.is_tensor(value) and value.ndim > 0:
                return int(value.shape[0])
    return int(fallback)


def _load_suffix(opt):
    return f"iter_{opt.load_iter}" if opt.load_iter > 0 else str(opt.epoch)


def _checkpoint_paths(checkpoint_dir, suffix, model_names):
    return [checkpoint_dir / f"{suffix}_net_{name}.pth"
            for name in model_names if isinstance(name, str)]


def _require_checkpoint_set(checkpoint_dir, suffix, model_names, purpose):
    missing = [path for path in _checkpoint_paths(checkpoint_dir, suffix, model_names)
               if not path.is_file()]
    if missing:
        formatted = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Cannot {purpose}; the following checkpoint files are missing:\n  {formatted}"
        )


def _training_state_path(checkpoint_dir, suffix):
    return checkpoint_dir / f"{suffix}_training_state.pth"


def _read_training_state(checkpoint_dir, suffix):
    state_path = _training_state_path(checkpoint_dir, suffix)
    if not state_path.is_file():
        return None
    return torch.load(state_path, map_location="cpu", weights_only=True)


def _latest_numeric_epoch(checkpoint_dir, model_names):
    first_name = next((name for name in model_names if isinstance(name, str)), None)
    if first_name is None:
        return None
    pattern = re.compile(rf"^(\d+)_net_{re.escape(first_name)}\.pth$")
    epochs = []
    for path in checkpoint_dir.glob(f"*_net_{first_name}.pth"):
        match = pattern.match(path.name)
        if match:
            epochs.append(int(match.group(1)))
    return max(epochs) if epochs else None


def _prepare_resume_position(opt, model, checkpoint_dir, dataset_size,
                             epoch_count_explicit):
    """Infer the next epoch/counter before schedulers are constructed."""
    suffix = _load_suffix(opt)
    state = _read_training_state(checkpoint_dir, suffix)

    if state is not None:
        if not epoch_count_explicit:
            opt.epoch_count = int(state.get("next_epoch", opt.epoch_count))
        total_iters = int(state.get(
            "total_iters", (opt.epoch_count - 1) * dataset_size))
        print(f"Resuming training state from {checkpoint_dir} "
              f"(next epoch {opt.epoch_count}, total_iters {total_iters})")
        return state, total_iters

    if not epoch_count_explicit:
        if opt.load_iter > 0:
            opt.epoch_count = max(1, opt.load_iter // max(dataset_size, 1) + 1)
        elif str(opt.epoch).isdigit():
            opt.epoch_count = int(opt.epoch) + 1
        elif opt.epoch == "latest":
            latest_epoch = _latest_numeric_epoch(checkpoint_dir, model.model_names)
            if latest_epoch is not None:
                opt.epoch_count = latest_epoch + 1

    total_iters = (opt.epoch_count - 1) * dataset_size
    print("No optimizer/scheduler state was found; network weights will resume "
          f"with newly initialized optimizer state at epoch {opt.epoch_count}.")
    return None, total_iters


def _restore_optimizer_state(model, state):
    if state is None:
        return
    optimizer_states = state.get("optimizers", [])
    scheduler_states = state.get("schedulers", [])
    if len(optimizer_states) != len(model.optimizers):
        raise RuntimeError("Optimizer count in the training state does not match the model.")
    if len(scheduler_states) != len(model.schedulers):
        raise RuntimeError("Scheduler count in the training state does not match the model.")
    for optimizer, optimizer_state in zip(model.optimizers, optimizer_states):
        optimizer.load_state_dict(optimizer_state)
    for scheduler, scheduler_state in zip(model.schedulers, scheduler_states):
        scheduler.load_state_dict(scheduler_state)


def _is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def _save_training_state(model, suffix, next_epoch, total_iters):
    if not _is_main_process():
        return
    model.save_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "next_epoch": int(next_epoch),
        "total_iters": int(total_iters),
        "optimizers": [optimizer.state_dict() for optimizer in model.optimizers],
        "schedulers": [scheduler.state_dict() for scheduler in model.schedulers],
    }
    state_path = _training_state_path(model.save_dir, str(suffix))
    temporary_path = state_path.with_suffix(state_path.suffix + ".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(state_path)


def _setup_model(model, opt, source_dir, resume_state):
    """Initialize the model, then load either target-resume or source weights."""
    suffix = _load_suffix(opt)
    if opt.continue_train:
        _require_checkpoint_set(model.save_dir, suffix, model.model_names,
                                "resume fine-tuning")
        model.setup(opt)
        _restore_optimizer_state(model, resume_state)
        return

    _require_checkpoint_set(source_dir, suffix, model.model_names,
                            "start fine-tuning")
    model.setup(opt)
    target_dir = model.save_dir
    try:
        model.save_dir = source_dir
        print(f"Loading pretrained network weights from {source_dir}")
        model.load_networks(suffix)
    finally:
        model.save_dir = target_dir


def _set_discriminator_updates(model, enabled):
    setter = getattr(model, "set_discriminator_updates", None)
    if setter is None:
        if enabled:
            return
        raise NotImplementedError(
            f"Model '{model.__class__.__name__}' does not implement "
            "--freeze_D_epochs. Set --freeze_D_epochs 0 for this model."
        )
    setter(enabled)


def run_fine_tuning(opt, user_args):
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    if dataset_size <= 0:
        raise RuntimeError(f"Fine-tuning dataset is empty: {opt.dataroot}")
    opt.data_size = dataset_size
    print(f"Fine-tuning dataset size = {dataset_size}")

    model = create_model(opt)
    target_dir = Path(opt.checkpoints_dir) / opt.name
    source_name = getattr(opt, "pretrained_name", None) or opt.name
    source_dir = Path(opt.checkpoints_dir) / source_name

    epoch_count_explicit = _argument_was_set(user_args, "--epoch_count")
    resume_state = None
    if opt.continue_train:
        # Resuming the target experiment must never be overwritten by the
        # original --pretrained_name checkpoint.
        resume_state, total_iters = _prepare_resume_position(
            opt, model, target_dir, dataset_size, epoch_count_explicit)
        print(f"Resuming target experiment: {target_dir}")
    else:
        total_iters = (opt.epoch_count - 1) * dataset_size
        print(f"Pretrained source experiment: {source_dir}")

    final_epoch = opt.n_epochs + opt.n_epochs_decay
    if opt.epoch_count > final_epoch:
        raise ValueError(
            f"Starting epoch {opt.epoch_count} is beyond the configured final "
            f"epoch {final_epoch}. Increase --n_epochs/--n_epochs_decay to "
            "continue this experiment."
        )

    freeze_D_epochs = int(getattr(opt, "freeze_D_epochs", 0))
    save_visuals_epoch_freq = int(getattr(opt, "save_visuals_epoch_freq", 0))
    if freeze_D_epochs < 0:
        raise ValueError("--freeze_D_epochs must be non-negative.")
    if save_visuals_epoch_freq < 0:
        raise ValueError("--save_visuals_epoch_freq must be non-negative.")
    if freeze_D_epochs > 0 and not hasattr(model, "set_discriminator_updates"):
        _set_discriminator_updates(model, False)  # raises a model-specific error

    # CUT variants create data-dependent layers from their first batch.
    from models.registry import legacy_name
    is_cut = legacy_name(opt.model) in ("cut", "cut_stack", "sincut")
    model_ready = False
    if not is_cut:
        _setup_model(model, opt, source_dir, resume_state)
        model_ready = True

    visualizer = Visualizer(opt)
    last_freeze_state = None

    for epoch in range(opt.epoch_count, final_epoch + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        epoch_had_batches = False
        visualizer.reset()
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        if hasattr(model, "set_current_epoch"):
            model.set_current_epoch(epoch)

        freeze_D = epoch <= freeze_D_epochs
        if model_ready:
            _set_discriminator_updates(model, not freeze_D)
        if freeze_D != last_freeze_state:
            status = "frozen" if freeze_D else "trainable"
            print(f"Epoch {epoch}: discriminator is {status}")
            last_freeze_state = freeze_D

        for i, data in enumerate(dataset):
            epoch_had_batches = True
            iter_start_time = time.time()
            t_data = iter_start_time - iter_data_time

            if not model_ready:
                model.data_dependent_initialize(data)
                _setup_model(model, opt, source_dir, resume_state)
                model.parallelize()
                _set_discriminator_updates(model, not freeze_D)
                model_ready = True

            current_batch_size = _batch_size(data, opt.batch_size)
            previous_total_iters = total_iters
            total_iters += current_batch_size
            epoch_iter += current_batch_size
            opt.counter = total_iters

            model.set_input(data)
            model.optimize_parameters()

            display_due = _frequency_due(
                previous_total_iters, total_iters, opt.display_freq)
            if display_due:
                save_result = _frequency_due(
                    previous_total_iters, total_iters, opt.update_html_freq)
                model.compute_visuals()
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, total_iters, save_result)

            print_due = i == 0 or _frequency_due(
                previous_total_iters, total_iters, opt.print_freq)
            if print_due:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / current_batch_size
                visualizer.print_current_losses(
                    epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(total_iters, losses)

            if _frequency_due(
                    previous_total_iters, total_iters, opt.save_latest_freq):
                print(f"saving the latest model (epoch {epoch}, "
                      f"total_iters {total_iters})")
                save_suffix = (f"iter_{total_iters}"
                               if opt.save_by_iter else "latest")
                model.save_networks(save_suffix)
                _save_training_state(
                    model, save_suffix, epoch, total_iters)

            iter_data_time = time.time()

        if not epoch_had_batches:
            raise RuntimeError("The data loader yielded no batches.")

        # Always save the final batch of the selected epochs. This removes the
        # accidental dependence on display_freq that previously omitted every
        # odd epoch for datasets with a 200-sample epoch and display_freq=400.
        if (save_visuals_epoch_freq > 0
                and epoch % save_visuals_epoch_freq == 0):
            model.compute_visuals()
            visualizer.display_current_results(
                model.get_current_visuals(), epoch, total_iters, save_result=True)

        model.update_learning_rate()

        if epoch % opt.save_epoch_freq == 0:
            print(f"saving the model at the end of epoch {epoch}, "
                  f"iters {total_iters}")
            model.save_networks("latest")
            _save_training_state(model, "latest", epoch + 1, total_iters)
            model.save_networks(epoch)

        print(f"End of epoch {epoch} / {final_epoch} "
              f"\t Time Taken: {time.time() - epoch_start_time:.0f} sec")

    print("Fine-tuning complete!")


def main():
    # Defaults for IDE/no-argument runs. CLI arguments are appended so that
    # argparse's last-value-wins behavior lets users override individual values.
    base_args = [
        "finetune.py",
        "--dataroot", "C:/Users/30927/Desktop/Multi-layer-CycleGAN/datasets/finetune_cancer_dual_512",
        "--name", "finetune_cancer_dual_spif_256",
        "--pretrained_name", "cancer_dual_spif_256",
        "--model", "spif",
        "--dataset_mode", "dual_channel",
        "--input_nc", "2",
        "--output_nc", "3",
        "--lambda_identity", "0",
        "--lr", "0.00005",
        "--n_epochs", "20",
        "--n_epochs_decay", "20",
        "--batch_size", "8",
        "--load_size", "286",
        "--crop_size", "256",
        "--lambda_content", "0",
        "--save_epoch_freq", "5",
        "--save_visuals_epoch_freq", "1",
        "--gpu_ids", "0",
    ]

    user_args = sys.argv[1:]
    if not user_args:
        sys.argv = base_args
    opt = TrainOptions().parse()
    opt.device = init_ddp(opt.gpu_ids)
    try:
        run_fine_tuning(opt, user_args)
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
