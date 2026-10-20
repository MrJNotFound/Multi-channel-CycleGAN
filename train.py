"""Train SPIF from datasets/trainA_BF, trainA_AF, and trainB."""

import argparse
import csv
from dataclasses import asdict, fields
import json
import time

import torch
from torch.utils.data import DataLoader

from spif.data import SPIFDataset
from spif.model import SPIFTrainer, TrainConfig
from spif.runtime import (
    ROOT, atomic_save, capture_rng, experiment_directory, read_checkpoint,
    restore_rng, save_image, seed_everything, seed_worker, select_device,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataroot", default=str(ROOT / "datasets"), help="Root containing trainA_BF, trainA_AF, and trainB.")
    parser.add_argument("--name", default="spif", help="Experiment name.")
    parser.add_argument("--checkpoints_dir", default=str(ROOT / "checkpoints"), help="Checkpoint output root.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--workers", type=int, default=0, help="Data-loading worker processes.")
    parser.add_argument("--cpu_threads", type=int, default=4, help="PyTorch CPU threads.")
    parser.add_argument("--resume", action="store_true", help="Resume latest.pt, restoring saved hyperparameters and optimizer state.")
    parser.add_argument("--save_every", type=int, default=0, help="Also retain a numbered checkpoint every N epochs; 0 keeps latest.pt only.")
    parser.add_argument("--print_every", type=int, default=100, help="Print losses every N batches.")
    parser.add_argument("--stop_after_epoch", type=int, default=None,
                        help="Stop after this epoch without changing the full learning-rate or SP-loss schedules.")
    descriptions = {
        "ngf": "Generator base channels", "ndf": "Discriminator base channels",
        "n_blocks": "Generator ResNet blocks", "load_size": "Square training resize",
        "crop_size": "Random crop size (multiple of 4, >=32)", "batch_size": "Training batch size",
        "n_epochs": "Epochs at the initial learning rate", "n_epochs_decay": "Linear decay epochs",
        "lr": "Initial Adam learning rate", "beta1": "Adam beta1",
        "lambda_A": "BF/AF cycle loss weight", "lambda_B": "H&E cycle loss weight",
        "lambda_sp": "Structure-preserving loss weight",
        "lambda_intensity": "Relative coarse-intensity weight",
        "sp_grad_decay_fraction": "Fraction of total epochs for gradient guidance",
        "sp_intensity_decay_fraction": "Fraction of total epochs for intensity guidance",
        "intensity_pool_size": "Coarse intensity grid size",
        "grad_channel_weights": "BF and AF weights (two values; default: equal)",
        "pool_size": "Discriminator replay-pool capacity", "no_flip": "Disable random horizontal flips",
        "seed": "Random seed",
    }
    group = parser.add_argument_group("SPIF hyperparameters (loaded from the checkpoint when resuming)")
    defaults = TrainConfig()
    for field in fields(TrainConfig):
        value = getattr(defaults, field.name)
        kwargs = {"default": None, "help": f"{descriptions[field.name]}; fresh-training default: {value}"}
        if field.name == "grad_channel_weights":
            kwargs.update(type=float, nargs=2, metavar=("BF", "AF"))
        elif isinstance(value, bool):
            kwargs["action"] = "store_true"
        else:
            kwargs["type"] = type(value)
        group.add_argument("--" + field.name, **kwargs)
    return parser


def run(args):
    if args.workers < 0 or args.save_every < 0 or args.print_every < 1:
        raise ValueError("workers and save_every must be non-negative; print_every must be positive.")
    directory = experiment_directory(args.checkpoints_dir, args.name)
    checkpoint_path = directory / "latest.pt"
    checkpoint = read_checkpoint(checkpoint_path) if args.resume else None
    if not args.resume and directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Experiment folder is not empty: {directory}. Use --resume or a new --name.")
    config = TrainConfig(**checkpoint["config"]) if checkpoint else TrainConfig()
    for field in fields(TrainConfig):
        value = getattr(args, field.name)
        if value is not None:
            if checkpoint and value != getattr(config, field.name):
                raise ValueError(f"Cannot change --{field.name} on resume; the checkpoint contains {getattr(config, field.name)!r}.")
            setattr(config, field.name, value)
    config.validate()
    first_epoch = checkpoint["epoch"] + 1 if checkpoint else 1
    final_epoch = config.total_epochs
    if args.stop_after_epoch is not None:
        if args.stop_after_epoch < first_epoch:
            raise ValueError(f"stop_after_epoch must be >= the next epoch ({first_epoch}).")
        final_epoch = min(final_epoch, args.stop_after_epoch)
    if first_epoch > final_epoch:
        print(f"Training already completed through epoch {checkpoint['epoch']}.")
        return
    device = select_device(args.device, args.cpu_threads)
    seed_everything(config.seed)
    dataset = SPIFDataset(args.dataroot, training=True, load_size=config.load_size,
                          crop_size=config.crop_size, no_flip=config.no_flip)
    loader_generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                        num_workers=args.workers, worker_init_fn=seed_worker,
                        generator=loader_generator, pin_memory=device.type == "cuda")
    trainer = SPIFTrainer(config, device)
    steps = 0
    if checkpoint:
        trainer.restore(checkpoint)
        restore_rng(checkpoint["rng"], loader_generator)
        steps = checkpoint["steps"]
        del checkpoint
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    log_path = directory / "losses.csv"
    if args.resume and log_path.exists():
        # Discard a row from an epoch interrupted before its checkpoint was saved.
        with log_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames
            rows = [row for row in reader if int(row["epoch"]) < first_epoch]
        with log_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    print(f"SPIF | device={device} | BF/AF pairs={len(dataset.pairs)} | H&E images={len(dataset.he_files)}")
    print(f"Checkpoints: {directory}", flush=True)
    for epoch in range(first_epoch, final_epoch + 1):
        start = time.perf_counter()
        trainer.set_epoch(epoch)
        totals, samples = {}, 0
        for batch_index, batch in enumerate(loader, 1):
            losses = trainer.step(batch)
            size = batch["A"].shape[0]
            samples += size
            steps += 1
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + value * size
            if batch_index == 1 or batch_index % args.print_every == 0:
                print(f"Epoch {epoch}/{config.total_epochs} | batch {batch_index}/{len(loader)} | "
                      f"G={losses['G']:.4f} D={losses['D']:.4f} content={losses['content']:.4f}", flush=True)
        row = {"epoch": epoch, "steps": steps, "lr": trainer.lr,
               **{name: value / samples for name, value in totals.items()}}
        new_log = not log_path.exists() or log_path.stat().st_size == 0
        with log_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if new_log:
                writer.writeheader()
            writer.writerow(row)
        for name, tensor in trainer.preview.items():
            save_image(tensor, directory / "samples" / f"epoch_{epoch:04d}_{name}.png")
        state = trainer.checkpoint(epoch, steps, capture_rng(loader_generator))
        atomic_save(state, checkpoint_path)
        if args.save_every and epoch % args.save_every == 0:
            atomic_save(state, directory / f"epoch_{epoch:04d}.pt")
        del state
        print(f"Saved epoch {epoch} -> {checkpoint_path} ({time.perf_counter() - start:.1f}s)", flush=True)
    print("Training complete.")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
