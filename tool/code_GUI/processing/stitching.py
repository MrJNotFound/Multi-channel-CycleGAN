"""Compose the same mosaic geometry for the preview and for file export."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .images import read_image, rotate_image, write_image


@dataclass(frozen=True)
class MosaicOptions:
    rows: int
    cols: int
    overlap_w: int
    overlap_h: int
    black_threshold: int
    angle: float
    direction_width: str
    direction_height: str
    fill_gaps: bool
    blend_overlaps: bool

    @property
    def track_coverage(self):
        return self.fill_gaps or self.blend_overlaps


def prepare_tile(image, angle, scale, track_coverage):
    coverage = None
    if angle:
        if track_coverage:
            coverage = rotate_image(np.ones(image.shape[:2], dtype=np.uint8), angle)
        image = rotate_image(image, angle)
    if scale < 1:
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        if coverage is not None:
            coverage = cv2.resize(coverage, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return image, coverage


def local_preview_indices(index, count, rows, cols):
    """Return a real rectangular window up to 2×2 within the selected group."""
    group_start = index // (rows * cols) * (rows * cols)
    row, col = divmod(index - group_start, cols)
    preview_cols = min(2, cols - col, count - index)
    preview_rows = 1 + int(row + 1 < rows and index + cols + preview_cols <= count)
    indices = [index + r * cols + c for r in range(preview_rows) for c in range(preview_cols)]
    return (preview_rows, preview_cols), indices


def _join_along_axis(images, axis, overlap, threshold, append):
    """Join tiles with the same overlap priority in either dimension."""
    ordered_axis = (np.moveaxis(image, axis, 0) for image in images)
    result = next(ordered_axis).copy()
    for tile in ordered_axis:
        before, after = (result, tile) if append else (tile, result)
        if overlap == 0:
            result = np.concatenate((before, after), axis=0)
            continue
        seam = before[-overlap:].copy()
        incoming = after[:overlap]
        black = incoming < threshold
        if incoming.ndim == 3:
            black = np.all(black, axis=-1)
        seam[~black] = incoming[~black]
        result = np.concatenate((before[:-overlap], seam, after[overlap:]), axis=0)
    return np.moveaxis(result, 0, axis)


def stitch_images_mosaic(
    images, grid_shape, overlap_width=60, overlap_height=60, black_threshold=10,
    direction_width="向左", direction_height="向下", *,
    fill_gaps=False, blend_overlaps=False, coverage_masks=None,
):
    """Compose row-major tiles. Coverage masks mark valid pixels after rotation."""
    rows, cols = grid_shape
    if rows < 1 or cols < 1 or len(images) != rows * cols:
        raise ValueError("图片数量必须与拼接行列数匹配。")
    if direction_width not in ("向左", "向右") or direction_height not in ("向上", "向下"):
        raise ValueError("无效的拼接朝向。")
    if fill_gaps or blend_overlaps:
        from .stitching_cleanup import stitch_with_cleanup

        return stitch_with_cleanup(
            images, grid_shape, overlap_width, overlap_height, black_threshold,
            direction_width, direction_height, coverage_masks, fill_gaps, blend_overlaps,
        )
    row_images = [
        _join_along_axis(images[start:start + cols], 1, overlap_width, black_threshold, direction_width == "向左")
        for start in range(0, len(images), cols)
    ]
    return _join_along_axis(row_images, 0, overlap_height, black_threshold, direction_height == "向下")


def render_mosaic(images, options, *, scale=1.0, overlap_scale=None):
    tiles, masks = [], []
    for image in images:
        tile, mask = prepare_tile(image, options.angle, scale, options.track_coverage)
        if tiles and tile.shape != tiles[0].shape:
            raise ValueError("组内图像尺寸或通道数不一致。")
        tiles.append(tile)
        masks.append(mask)
    if not tiles:
        raise ValueError("没有可拼接的图块。")
    scale_y, scale_x = overlap_scale or (scale, scale)
    overlap_w = int(options.overlap_w * scale_x)
    overlap_h = int(options.overlap_h * scale_y)
    height, width = tiles[0].shape[:2]
    if (options.cols > 1 and not 0 <= overlap_w < width) or (options.rows > 1 and not 0 <= overlap_h < height):
        raise ValueError("重叠像素必须小于对应方向的图块尺寸。")
    return stitch_images_mosaic(
        tiles, (options.rows, options.cols), overlap_w, overlap_h, options.black_threshold,
        options.direction_width, options.direction_height,
        fill_gaps=options.fill_gaps, blend_overlaps=options.blend_overlaps, coverage_masks=masks,
    )


def export_mosaics(paths, destination, options, scale, *, progress, check_interrupted):
    group_size = options.rows * options.cols
    groups, remainder = divmod(len(paths), group_size)
    if not groups:
        raise ValueError(f"当前网格需要至少 {group_size} 张图像。")
    target = Path(destination)
    targets = [target] if groups == 1 else [
        target.with_name(f"{target.stem}_{index:04d}{target.suffix}") for index in range(1, groups + 1)
    ]
    source_paths = {Path(path).resolve() for path in paths}
    if any(output.resolve() in source_paths for output in targets):
        raise ValueError("输出路径会覆盖输入图像，请更换文件名或目录。")
    if groups > 1:
        existing = next((path for path in targets if path.exists()), None)
        if existing is not None:
            raise FileExistsError(f"编号结果 {existing.name} 已存在，请更换文件名或目录。")

    def read_tiles(group):
        for path in group:
            check_interrupted()
            tile = read_image(path)
            if tile is None:
                raise ValueError(f"无法读取图像：{Path(path).name}")
            yield tile

    saved, errors = [], []
    for index, output in enumerate(targets, 1):
        check_interrupted()
        start = (index - 1) * group_size
        try:
            result = render_mosaic(read_tiles(paths[start:start + group_size]), options, scale=scale)
            check_interrupted()
            write_image(output, result)
            saved.append(output)
        except (OSError, ValueError, cv2.error) as error:
            errors.append(f"第 {index} 组：{error}")
        progress(index)
    return saved, errors, remainder
