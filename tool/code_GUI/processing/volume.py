"""Volume loading and mask preparation used by the 3D viewer."""

import glob
import os

import cv2
import imageio.v3 as iio
import numpy as np


def load_image_stack(folder, patterns, downsample=1, interp=cv2.INTER_AREA):
    files = []
    for pattern in patterns:
        files += glob.glob(os.path.join(folder, pattern))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No files found in {folder} with patterns {patterns}")
    stack = []
    for f in files:
        img = iio.imread(f)
        if img.ndim == 2:
            img = img[..., np.newaxis]
        if downsample > 1:
            h, w = img.shape[:2]
            img = cv2.resize(img, (w // downsample, h // downsample), interpolation=interp)
        stack.append(img)
    arr = np.stack(stack, axis=0)
    if arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)
    return arr


def repair_mask(mask, closing_size=3, dilation_size=1):
    from scipy.ndimage import generate_binary_structure, binary_closing, binary_dilation
    pad_width = max(closing_size, dilation_size)
    # 三维padding
    mask_padded = np.pad(mask, pad_width, mode='constant', constant_values=0)
    structure = generate_binary_structure(3, 2)
    closed = binary_closing(mask_padded, structure=structure, iterations=closing_size)
    if dilation_size > 0:
        closed = binary_dilation(closed, structure=structure, iterations=dilation_size)
    # 裁剪回原始大小
    slices = tuple(slice(pad_width, -pad_width) for _ in range(3))
    closed = closed[slices]
    return closed
