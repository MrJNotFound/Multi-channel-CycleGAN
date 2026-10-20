"""Load aligned BF/AF pairs and unpaired H&E images for SPIF."""

from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def image_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Required image folder does not exist: {directory}")
    files = sorted(p for p in directory.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise ValueError(f"No supported images found in: {directory}")
    return files


def read_image(path, mode):
    with Image.open(path) as image:
        if image.mode in {"I", "F"} or image.mode.startswith("I;16"):
            raise ValueError(f"Expected an 8-bit image, got {image.mode}: {path}")
        return image.convert(mode)


def to_tensor(image):
    array = np.array(image, dtype=np.float32, copy=True)
    if array.ndim == 2:
        array = array[:, :, None]
    return torch.from_numpy(array.transpose(2, 0, 1).copy()).div_(127.5).sub_(1.0)


class SPIFDataset(Dataset):
    """Apply identical spatial augmentation to each aligned BF/AF pair."""

    def __init__(self, dataroot, training=False, load_size=286, crop_size=256,
                 no_flip=False):
        self.training = training
        self.load_size = load_size
        self.crop_size = crop_size
        self.no_flip = no_flip
        root = Path(dataroot)
        phase = "train" if training else "test"
        self.bf_dir = root / f"{phase}A_BF"
        self.af_dir = root / f"{phase}A_AF"
        bf = {p.relative_to(self.bf_dir): p for p in image_files(self.bf_dir)}
        af = {p.relative_to(self.af_dir): p for p in image_files(self.af_dir)}
        if bf.keys() != af.keys():
            missing_af = sorted(str(p) for p in bf.keys() - af.keys())[:5]
            missing_bf = sorted(str(p) for p in af.keys() - bf.keys())[:5]
            raise ValueError(
                "BF and AF must have identical relative filenames. "
                f"Missing AF: {missing_af}; missing BF: {missing_bf}"
            )
        self.pairs = [(bf[name], af[name], name) for name in sorted(bf)]
        self.he_files = image_files(root / "trainB") if training else []
        if training and (crop_size < 32 or crop_size % 4 or load_size < crop_size):
            raise ValueError("crop_size must be >= 32 and divisible by 4; load_size must be >= crop_size.")

    def __len__(self):
        return max(len(self.pairs), len(self.he_files))

    def _augmentation(self):
        limit = self.load_size - self.crop_size
        return (random.randint(0, limit), random.randint(0, limit),
                not self.no_flip and random.random() > 0.5)

    def _transform(self, image, params):
        x, y, flip = params
        image = image.resize((self.load_size, self.load_size), Image.Resampling.BICUBIC)
        image = image.crop((x, y, x + self.crop_size, y + self.crop_size))
        if flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return to_tensor(image)

    def __getitem__(self, index):
        bf_path, af_path, relative = self.pairs[index % len(self.pairs)]
        bf = read_image(bf_path, "L")
        af = read_image(af_path, "L")
        if bf.size != af.size:
            raise ValueError(f"Aligned BF/AF images must have the same size: {relative}, {bf.size} vs {af.size}")
        if self.training:
            params = self._augmentation()
            source = torch.cat((self._transform(bf, params), self._transform(af, params)))
            he_path = random.choice(self.he_files)
            target = self._transform(read_image(he_path, "RGB"), self._augmentation())
            return {"A": source, "B": target, "name": relative.as_posix()}
        if min(bf.size) < 16:
            raise ValueError(f"Test images must be at least 16x16 pixels: {relative}")
        return {"A": torch.cat((to_tensor(bf), to_tensor(af))), "name": relative.as_posix()}
