"""Unified, plugin-driven training entry point.

The selected ``--model`` and ``--dataset_mode`` modules define the workflow.
Fresh training, resume training, and fine-tuning share one loop; models such as
CUT that require data-dependent initialization are handled through a capability
branch instead of a dedicated script.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import create_dataset  # noqa: E402
from models import create_model  # noqa: E402
from options.train_options import TrainOptions  # noqa: E402
from util.util import cleanup_ddp, init_ddp  # noqa: E402
from util.visualizer import Visualizer  # noqa: E402


GUI_EVENTS = False


def _is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def emit_event(event_type: str, **payload) -> None:
    """Write a machine-readable progress event alongside normal CLI logs."""
    if not GUI_EVENTS or not _is_main_process():
        return
    event = {"type": event_type, **payload}
    print("@@MLCGUI " + json.dumps(event, ensure_ascii=False), flush=True)


def _argument_was_set(arguments: list[str], option: str) -> bool:
    return option in arguments or any(arg.startswith(f"{option}=") for arg in arguments)


def _frequency_due(previous: int, current: int, frequency: int) -> bool:
    return frequency > 0 and previous // frequency < current // frequency


def _batch_size(data, fallback: int) -> int:
    if isinstance(data, dict):
        for value in data.values():
            if torch.is_tensor(value) and value.ndim > 0:
                return int(value.shape[0])
    return int(fallback)


def _load_suffix(opt) -> str:
    return f"iter_{opt.load_iter}" if opt.load_iter > 0 else str(opt.epoch)


def _checkpoint_paths(directory: Path, suffix: str, model_names) -> list[Path]:
    return [
        directory / f"{suffix}_net_{name}.pth"
        for name in model_names
        if isinstance(name, str)
    ]


def _require_checkpoints(directory: Path, suffix: str, model_names, purpose: str) -> None:
    missing = [path for path in _checkpoint_paths(directory, suffix, model_names) if not path.is_file()]
    if missing:
        listing = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Cannot {purpose}; missing checkpoint files:\n  {listing}")


def _state_path(directory: Path, suffix: str) -> Path:
    return directory / f"{suffix}_training_state.pth"


def _read_state(directory: Path, suffix: str):
    path = _state_path(directory, suffix)
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


def _latest_numeric_epoch(directory: Path, model_names) -> int | None:
    first_name = next((name for name in model_names if isinstance(name, str)), None)
    if first_name is None:
        return None
    pattern = re.compile(rf"^(\d+)_net_{re.escape(first_name)}\.pth$")
    epochs = []
    for path in directory.glob(f"*_net_{first_name}.pth"):
        match = pattern.match(path.name)
        if match:
            epochs.append(int(match.group(1)))
    return max(epochs) if epochs else None


def _prepare_resume(opt, model, directory: Path, dataset_size: int, user_args: list[str]):
    suffix = _load_suffix(opt)
    state = _read_state(directory, suffix)
    explicit_epoch = _argument_was_set(user_args, "--epoch_count")
    if state is not None:
        if not explicit_epoch:
            opt.epoch_count = int(state.get("next_epoch", opt.epoch_count))
        total_iters = int(state.get("total_iters", (opt.epoch_count - 1) * dataset_size))
        print(
            f"Resuming training state from {directory} "
            f"(next epoch {opt.epoch_count}, total_iters {total_iters})"
        )
        return state, total_iters

    if not explicit_epoch:
        if opt.load_iter > 0:
            opt.epoch_count = max(1, opt.load_iter // max(dataset_size, 1) + 1)
        elif str(opt.epoch).isdigit():
            opt.epoch_count = int(opt.epoch) + 1
        elif opt.epoch == "latest":
            latest = _latest_numeric_epoch(directory, model.model_names)
            if latest is not None:
                opt.epoch_count = latest + 1
    print(
        "No optimizer/scheduler state was found; network weights will resume "
        f"with newly initialized optimizer state at epoch {opt.epoch_count}."
    )
    return None, (opt.epoch_count - 1) * dataset_size


def _restore_state(model, state) -> None:
    if state is None:
        return
    optimizer_states = state.get("optimizers", [])
    scheduler_states = state.get("schedulers", [])
    if len(optimizer_states) != len(model.optimizers):
        raise RuntimeError("Optimizer count in training state does not match the model.")
    if len(scheduler_states) != len(model.schedulers):
        raise RuntimeError("Scheduler count in training state does not match the model.")
    for optimizer, optimizer_state in zip(model.optimizers, optimizer_states):
        optimizer.load_state_dict(optimizer_state)
    for scheduler, scheduler_state in zip(model.schedulers, scheduler_states):
        scheduler.load_state_dict(scheduler_state)


def _save_state(model, suffix: str, next_epoch: int, total_iters: int) -> None:
    if not _is_main_process():
        return
    model.save_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "next_epoch": int(next_epoch),
        "total_iters": int(total_iters),
        "optimizers": [optimizer.state_dict() for optimizer in model.optimizers],
        "schedulers": [scheduler.state_dict() for scheduler in model.schedulers],
    }
    destination = _state_path(model.save_dir, str(suffix))
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(destination)


def _initialize_model(model, opt, mode: str, source_dir: Path, resume_state) -> None:
    suffix = _load_suffix(opt)
    if mode == "resume":
        _require_checkpoints(model.save_dir, suffix, model.model_names, "resume training")
        model.setup(opt)
        _restore_state(model, resume_state)
        return

    model.setup(opt)
    if mode == "finetune":
        _require_checkpoints(source_dir, suffix, model.model_names, "start fine-tuning")
        target_dir = model.save_dir
        try:
            model.save_dir = source_dir
            print(f"Loading pretrained network weights from {source_dir}")
            model.load_networks(suffix)
        finally:
            model.save_dir = target_dir


def _set_discriminator_updates(model, enabled: bool) -> None:
    setter = getattr(model, "set_discriminator_updates", None)
    if setter is None:
        if enabled:
            return
        raise NotImplementedError(
            f"Model '{model.__class__.__name__}' does not support discriminator "
            "freezing; set --freeze_D_epochs 0."
        )
    setter(enabled)


def run_training(opt, user_args: list[str]) -> None:
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    if dataset_size <= 0:
        raise RuntimeError(f"Training dataset is empty: {opt.dataroot}")
    opt.data_size = dataset_size
    print(f"Training dataset size = {dataset_size}")

    model = create_model(opt)
    target_dir = Path(opt.checkpoints_dir) / opt.name
    source_name = str(opt.pretrained_name).strip() if opt.pretrained_name else ""
    source_dir = Path(opt.checkpoints_dir) / source_name if source_name else target_dir

    if opt.continue_train:
        initialization = "resume"
    elif source_name:
        initialization = "finetune"
    else:
        initialization = "fresh"

    resume_state = None
    if initialization == "resume":
        resume_state, total_iters = _prepare_resume(
            opt, model, target_dir, dataset_size, user_args
        )
    else:
        total_iters = (opt.epoch_count - 1) * dataset_size

    configured_final_epoch = opt.n_epochs + opt.n_epochs_decay
    final_epoch = configured_final_epoch
    if opt.stop_after_epoch is not None:
        if opt.stop_after_epoch < opt.epoch_count:
            raise ValueError("--stop_after_epoch must be >= --epoch_count.")
        final_epoch = min(final_epoch, opt.stop_after_epoch)
    if opt.epoch_count > final_epoch:
        raise ValueError(
            f"Starting epoch {opt.epoch_count} is beyond final epoch {final_epoch}."
        )

    freeze_epochs = int(opt.freeze_D_epochs)
    if freeze_epochs < 0:
        raise ValueError("--freeze_D_epochs must be non-negative.")
    if freeze_epochs and not hasattr(model, "set_discriminator_updates"):
        _set_discriminator_updates(model, False)

    # Capability detection keeps the engine open to future plugins: a model
    # opts into first-batch initialization simply by overriding the base hook.
    from models.base_model import BaseModel

    requires_data = (
        type(model).data_dependent_initialize
        is not BaseModel.data_dependent_initialize
    )
    model_ready = False
    if not requires_data:
        _initialize_model(model, opt, initialization, source_dir, resume_state)
        model_ready = True

    visualizer = Visualizer(opt)
    # Dataset workers receive opt through dataset.opt under Windows spawn.
    # Keep runtime objects (HTML modules, W&B sessions) local to this process.
    emit_event(
        "run_started",
        task="train",
        model=opt.model,
        dataset_mode=opt.dataset_mode,
        initialization=initialization,
        dataset_size=dataset_size,
        start_epoch=opt.epoch_count,
        final_epoch=final_epoch,
        configured_final_epoch=configured_final_epoch,
        output_dir=str(target_dir.resolve()),
    )

    last_freeze_state = None
    for epoch in range(opt.epoch_count, final_epoch + 1):
        epoch_start = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        had_batches = False
        visualizer.reset()
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        if hasattr(model, "set_current_epoch"):
            model.set_current_epoch(epoch)

        discriminator_frozen = epoch <= freeze_epochs
        if model_ready:
            _set_discriminator_updates(model, not discriminator_frozen)
        if discriminator_frozen != last_freeze_state:
            print(
                f"Epoch {epoch}: discriminator is "
                f"{'frozen' if discriminator_frozen else 'trainable'}"
            )
            last_freeze_state = discriminator_frozen

        for batch_index, data in enumerate(dataset):
            had_batches = True
            iter_start = time.time()
            data_time = iter_start - iter_data_time

            if not model_ready:
                model.data_dependent_initialize(data)
                _initialize_model(model, opt, initialization, source_dir, resume_state)
                model.parallelize()
                _set_discriminator_updates(model, not discriminator_frozen)
                model_ready = True

            current_batch = _batch_size(data, opt.batch_size)
            previous_total = total_iters
            total_iters += current_batch
            epoch_iter += current_batch
            opt.counter = total_iters

            model.set_input(data)
            model.optimize_parameters()
            emit_event(
                "progress",
                task="train",
                epoch=epoch,
                final_epoch=final_epoch,
                epoch_iter=min(epoch_iter, dataset_size),
                dataset_size=dataset_size,
                total_iters=total_iters,
            )

            if _frequency_due(previous_total, total_iters, opt.display_freq):
                save_result = _frequency_due(
                    previous_total, total_iters, opt.update_html_freq
                )
                model.compute_visuals()
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, total_iters, save_result
                )

            print_due = batch_index == 0 or _frequency_due(
                previous_total, total_iters, opt.print_freq
            )
            if print_due:
                losses = model.get_current_losses()
                compute_time = (time.time() - iter_start) / max(current_batch, 1)
                visualizer.print_current_losses(
                    epoch, epoch_iter, losses, compute_time, data_time
                )
                visualizer.plot_current_losses(total_iters, losses)
                emit_event(
                    "metrics",
                    epoch=epoch,
                    total_iters=total_iters,
                    values=losses,
                )

            if _frequency_due(previous_total, total_iters, opt.save_latest_freq):
                suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                print(
                    f"saving the latest model (epoch {epoch}, "
                    f"total_iters {total_iters})"
                )
                model.save_networks(suffix)
                _save_state(model, suffix, epoch, total_iters)
                emit_event("checkpoint", suffix=str(suffix), epoch=epoch)

            iter_data_time = time.time()

        if not had_batches:
            raise RuntimeError("The data loader yielded no batches.")

        if opt.save_visuals_epoch_freq > 0 and epoch % opt.save_visuals_epoch_freq == 0:
            model.compute_visuals()
            visualizer.display_current_results(
                model.get_current_visuals(), epoch, total_iters, save_result=True
            )

        model.update_learning_rate()
        if epoch % opt.save_epoch_freq == 0:
            print(f"saving model at end of epoch {epoch}, iters {total_iters}")
            model.save_networks("latest")
            _save_state(model, "latest", epoch + 1, total_iters)
            model.save_networks(epoch)
            _save_state(model, str(epoch), epoch + 1, total_iters)
            emit_event("checkpoint", suffix=str(epoch), epoch=epoch)

        duration = time.time() - epoch_start
        print(
            f"End of epoch {epoch} / {configured_final_epoch} "
            f"\t Time Taken: {duration:.0f} sec"
        )
        emit_event(
            "epoch_end",
            epoch=epoch,
            final_epoch=final_epoch,
            duration_seconds=duration,
        )

    emit_event("run_finished", task="train", total_iters=total_iters)
    print("Training complete!")


def main() -> None:
    global GUI_EVENTS
    user_args = sys.argv[1:]
    opt = TrainOptions().parse()
    GUI_EVENTS = bool(opt.gui_events)
    opt.device = init_ddp(opt.gpu_ids)
    try:
        run_training(opt, user_args)
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
