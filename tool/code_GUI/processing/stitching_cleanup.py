"""Correct tile brightness, blend overlaps, and fill gaps in a mosaic."""

from itertools import product
from math import ceil

import cv2
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import lsqr


_BLOCK = 768
_GAP_HALO = 64


def _blocks(height, width):
    for y, x in product(range(0, height, _BLOCK), range(0, width, _BLOCK)):
        yield y, min(y + _BLOCK, height), x, min(x + _BLOCK, width)


def _overlap_offsets(tiles, masks, rows, cols, overlap_w, overlap_h):
    """Fit additive corrections from corresponding pixels in adjacent tiles."""
    height, width, channels = tiles[0].shape
    color_channels = min(3, channels)
    pairs, differences = [], []
    for row, col in product(range(rows), range(cols)):
        index = row * cols + col
        neighbors = []
        if col + 1 < cols and overlap_w:
            neighbors.append((index + 1, (0, width - overlap_w), (height, overlap_w)))
        if row + 1 < rows and overlap_h:
            neighbors.append((index + cols, (height - overlap_h, 0), (overlap_h, width)))
        for other, (y, x), (h, w) in neighbors:
            dy, dx = ceil(h / 96), ceil(w / 96)
            first = np.s_[y:y + h:dy, x:x + w:dx]
            second = np.s_[:h:dy, :w:dx]
            valid = np.ones_like(tiles[other][second][..., 0], dtype=bool)
            if masks[index] is not None:
                valid &= masks[index][first] != 0
            if masks[other] is not None:
                valid &= masks[other][second] != 0
            if np.count_nonzero(valid) < 8:
                continue
            left = tiles[index][first][valid, :color_channels].astype(np.float64)
            right = tiles[other][second][valid, :color_channels]
            difference = left - right
            difference = difference[np.isfinite(difference).all(axis=1)]
            if len(difference) >= 8:
                pairs.append((index, other))
                differences.append(np.median(difference, axis=0))

    offsets = np.zeros((len(tiles), channels))
    if pairs:
        # A small damping term keeps the original overall brightness and limits drift.
        matrix = sparse.csr_matrix(
            (np.tile([-1.0, 1.0], len(pairs)), (np.repeat(np.arange(len(pairs)), 2), np.ravel(pairs))),
            shape=(len(pairs), len(tiles)),
        )
        for channel, values in enumerate(np.asarray(differences).T):
            offsets[:, channel] = lsqr(matrix, values, damp=0.1)[0]
    return offsets


def _feather(size, overlap, index, count):
    weights = np.ones(size, dtype=np.float32)
    if overlap:
        ramp = (np.arange(overlap, dtype=np.float32) + 0.5) / overlap
        if index > 0:
            weights[:overlap] *= ramp
        if index + 1 < count:
            weights[-overlap:] *= ramp[::-1]
    return weights


def _cast_pixels(pixels, dtype):
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        np.rint(pixels, out=pixels)
        np.clip(pixels, limits.min, limits.max, out=pixels)
    return pixels.astype(dtype, copy=False)


def _coarse_gap_reference(image, coverage):
    height, width = coverage.shape
    step = ceil(max(height, width) / 1024)
    ys = np.arange(0, height, step)
    xs = np.arange(0, width, step)
    valid = coverage[np.ix_(ys, xs)]
    if not valid.any():
        row = int(coverage.any(axis=1).argmax())
        col = int(coverage[row].argmax())
        return image[row:row + 1, col:col + 1], max(height, width)
    indices = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return image[ys[indices[0]], xs[indices[1]]], step


def _fill_uncovered(image, coverage):
    """Extrapolate nearby pixels into uncovered areas, including outer black margins."""
    if coverage.all() or not coverage.any():
        return
    height, width = coverage.shape
    reference = None
    float_dtype = np.float32 if image.dtype.itemsize <= 2 else np.float64
    for y0, y1, x0, x1 in _blocks(height, width):
        missing = ~coverage[y0:y1, x0:x1]
        if not missing.any():
            continue
        top, bottom = max(0, y0 - _GAP_HALO), min(height, y1 + _GAP_HALO)
        left, right = max(0, x0 - _GAP_HALO), min(width, x1 + _GAP_HALO)
        valid = coverage[top:bottom, left:right]
        region = image[top:bottom, left:right]
        if valid.any():
            distance, indices = ndimage.distance_transform_edt(~valid, return_indices=True)
            patch = region[indices[0], indices[1]].astype(float_dtype)
            far = np.clip((distance - _GAP_HALO / 2) / (_GAP_HALO / 2 - 4), 0, 1)
        else:
            patch = np.empty(region.shape, dtype=float_dtype)
            far = np.ones(valid.shape)
        remote_y, remote_x = np.nonzero(far)
        if remote_y.size:
            if reference is None:
                reference = _coarse_gap_reference(image, coverage)
            coarse, step = reference
            ys = np.minimum((remote_y + top + step // 2) // step, coarse.shape[0] - 1)
            xs = np.minimum((remote_x + left + step // 2) // step, coarse.shape[1] - 1)
            replace = far[remote_y, remote_x] == 1
            patch[remote_y[replace], remote_x[replace]] = coarse[ys[replace], xs[replace]]
            ys, xs = ys[~replace], xs[~replace]
            iy, ix = remote_y[~replace], remote_x[~replace]
            weight = far[iy, ix, None]
            patch[iy, ix] += weight * (coarse[ys, xs] - patch[iy, ix])
        smoothed = cv2.GaussianBlur(patch, (0, 0), sigmaX=1.5)
        if smoothed.ndim == 2:
            smoothed = smoothed[..., None]
        center = smoothed[y0 - top:y1 - top, x0 - left:x1 - left]
        image[y0:y1, x0:x1][missing] = _cast_pixels(center[missing], image.dtype)


def stitch_with_cleanup(
    images, grid_shape, overlap_w, overlap_h, black_threshold,
    direction_width, direction_height, coverage_masks, fill_gaps, blend_overlaps,
):
    rows, cols = grid_shape
    first = images[0]
    height, width = first.shape[:2]
    if (cols > 1 and not 0 <= overlap_w < width) or (rows > 1 and not 0 <= overlap_h < height):
        raise ValueError("重叠像素必须小于对应方向的图块尺寸。")
    if any(image.shape != first.shape or image.dtype != first.dtype for image in images):
        raise ValueError("组内图像尺寸、通道数和位深必须一致。")
    if coverage_masks is None:
        coverage_masks = [None] * len(images)
    if len(coverage_masks) != len(images) or any(
        mask is not None and mask.shape != (height, width) for mask in coverage_masks
    ):
        raise ValueError("图块覆盖范围与图块尺寸不一致。")

    # Use the same physical order as the existing stitcher for each orientation.
    row_order = range(rows - 1, -1, -1) if direction_height == '向上' else range(rows)
    col_order = range(cols - 1, -1, -1) if direction_width == '向右' else range(cols)
    order = [row * cols + col for row, col in product(row_order, col_order)]
    tiles = [images[i][..., None] if first.ndim == 2 else images[i] for i in order]
    masks = [coverage_masks[i] for i in order]
    channels = tiles[0].shape[2]
    overlap_w = overlap_w if cols > 1 else 0
    overlap_h = overlap_h if rows > 1 else 0
    step_x, step_y = width - overlap_w, height - overlap_h
    out_h, out_w = height + (rows - 1) * step_y, width + (cols - 1) * step_x
    result = np.zeros((out_h, out_w, channels), dtype=first.dtype)
    coverage = np.zeros((out_h, out_w), dtype=bool) if fill_gaps and any(m is not None for m in masks) else None
    if blend_overlaps:
        offsets = _overlap_offsets(tiles, masks, rows, cols, overlap_w, overlap_h)
        offsets = offsets.astype(np.float32 if first.dtype.itemsize <= 2 else np.float64)
        weights_y = [_feather(height, overlap_h, row, rows) for row in range(rows)]
        weights_x = [_feather(width, overlap_w, col, cols) for col in range(cols)]

    for y0, y1, x0, x1 in _blocks(out_h, out_w):
        block = result[y0:y1, x0:x1]
        if blend_overlaps:
            total = np.zeros(block.shape, dtype=offsets.dtype)
            weight = np.zeros(block.shape[:2], dtype=offsets.dtype)
        else:
            covered = np.zeros(block.shape[:2], dtype=bool)

        intersecting_rows = range(max(0, (y0 - height) // step_y + 1), min(rows, (y1 - 1) // step_y + 1))
        intersecting_cols = range(max(0, (x0 - width) // step_x + 1), min(cols, (x1 - 1) // step_x + 1))
        for row, col in product(intersecting_rows, intersecting_cols):
            index = row * cols + col
            y, x = row * step_y, col * step_x
            top, bottom = max(y, y0), min(y + height, y1)
            left, right = max(x, x0), min(x + width, x1)
            src = np.s_[top - y:bottom - y, left - x:right - x]
            dst = np.s_[top - y0:bottom - y0, left - x0:right - x0]
            pixels = tiles[index][src]
            valid = None if masks[index] is None else masks[index][src].astype(bool)
            if blend_overlaps:
                feather = weights_y[row][src[0], None] * weights_x[col][src[1]]
                if valid is not None:
                    feather *= valid
                adjusted = pixels + offsets[index]
                if valid is not None:
                    adjusted[~valid] = 0
                total[dst] += adjusted * feather[..., None]
                weight[dst] += feather
            else:
                use = ~covered[dst] | np.any(pixels >= black_threshold, axis=2)
                if valid is not None:
                    use &= valid
                np.copyto(block[dst], pixels, where=use[..., None])
                covered[dst] |= use

        if blend_overlaps:
            covered = weight > 0
            np.divide(total, weight[..., None], out=total, where=covered[..., None])
            block[:] = _cast_pixels(total, first.dtype)
        if coverage is not None:
            coverage[y0:y1, x0:x1] = covered

    if coverage is not None:
        _fill_uncovered(result, coverage)
    return result[..., 0] if first.ndim == 2 else result
