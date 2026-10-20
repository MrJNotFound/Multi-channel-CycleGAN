"""Device selection, reproducibility, checkpoint I/O, and image output."""

import os
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = "spif-v2-single-stage"
CHECKPOINT_VERSION = 1


def experiment_directory(root, name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Experiment name must start with a letter or digit and contain only letters, digits, _, -, or .")
    return Path(root).expanduser().resolve() / name


def select_device(name="auto", cpu_threads=4):
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive.")
    torch.set_num_threads(cpu_threads)
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices: auto, cpu, cuda, or cuda:N.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable in this PyTorch installation. Use --device cpu.")
        torch.cuda.set_device(device)
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def capture_rng(loader_generator):
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": loader_generator.get_state(),
    }


def restore_rng(state, loader_generator):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    loader_generator.set_state(state["loader"])
    if torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_save(checkpoint, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def read_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Run train.py first or pass --checkpoint.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(checkpoint, dict)
            or checkpoint.get("format_version") != CHECKPOINT_VERSION
            or checkpoint.get("architecture") != ARCHITECTURE):
        raise ValueError("Expected a SPIF release checkpoint produced by train.py.")
    return checkpoint


def save_image(tensor, path):
    """Save one normalized CHW image as an 8-bit PNG."""
    array = tensor.detach().float().cpu().clamp(-1, 1).add(1).mul(127.5)
    array = array.round().byte().permute(1, 2, 0).numpy()
    if array.shape[2] == 1:
        array = array[:, :, 0]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
