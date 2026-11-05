import glob
import os
import re
import cv2
import imageio.v3 as iio
import numpy as np


def _natural_sort_key(text: str):
    parts = re.split(r"(\d+)", text)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def list_image_files(folder: str, patterns):
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(folder, pattern)))
    files = sorted(files, key=_natural_sort_key)
    if not files:
        raise FileNotFoundError(f"No image files found in {folder} with patterns {patterns}")
    return files


def load_image_stack(folder: str, patterns, downsample: float = 1.0, interp=cv2.INTER_AREA):
    files = list_image_files(folder, patterns)
    sample = iio.imread(files[0])
    if sample.ndim == 4 and sample.shape[0] == 1:
        sample = np.squeeze(sample, axis=0)
    if sample.ndim == 4 and sample.shape[-1] == 1:
        sample = np.squeeze(sample, axis=-1)
    if sample.ndim == 3 and sample.shape[0] in (1, 3, 4) and sample.shape[-1] not in (1, 3, 4):
        sample = np.transpose(sample, (1, 2, 0))
    is_color = sample.ndim == 3 and sample.shape[-1] >= 3

    stack = []
    for fpath in files:
        img = iio.imread(fpath)
        if img.ndim == 4 and img.shape[0] == 1:
            img = np.squeeze(img, axis=0)
        if img.ndim == 4 and img.shape[-1] == 1:
            img = np.squeeze(img, axis=-1)
        if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))

        if img.ndim == 2:
            if is_color:
                img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3:
            if img.shape[-1] == 1:
                img = img[..., 0]
                if is_color:
                    img = np.stack([img] * 3, axis=-1)
            else:
                if img.shape[-1] > 3:
                    img = img[..., :3]
                if not is_color:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"Unsupported image shape: {img.shape} for file {fpath}")

        if downsample > 1:
            h, w = img.shape[:2]
            new_w = max(1, int(w / downsample))
            new_h = max(1, int(h / downsample))
            img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        stack.append(img)

    arr = np.stack(stack, axis=0)
    return arr, is_color

