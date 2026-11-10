from __future__ import annotations

import math
import os
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .geometry import mosaic_bounds
from .image_ops import read_rgb, resize_rgb
from .models import TileSet


ProgressCallback = Callable[[int, int, str], None]


class TileCache:
    def __init__(self, maximum: int = 8, tile_rotation_deg: float = 0.0) -> None:
        self.maximum = maximum
        self.tile_rotation_deg = tile_rotation_deg
        self._items: OrderedDict[Path, np.ndarray] = OrderedDict()

    def get(self, path: Path) -> np.ndarray:
        if path in self._items:
            image = self._items.pop(path)
            self._items[path] = image
            return image
        image = read_rgb(path)
        if abs(self.tile_rotation_deg) > 1e-8:
            image = np.asarray(
                Image.fromarray(image, mode="RGB").rotate(
                    -self.tile_rotation_deg,
                    resample=Image.Resampling.BICUBIC,
                    expand=True,
                    fillcolor=(0, 0, 0),
                )
            ).copy()
        self._items[path] = image
        while len(self._items) > self.maximum:
            self._items.popitem(last=False)
        return image


def render_mosaic(
    tile_set: TileSet,
    positions: Mapping[tuple[int, int], np.ndarray],
    output_path: str | Path,
    tile_rotation_deg: float = 0.0,
    overlap_blend_percent: float = 50.0,
    chunk_size: int = 1024,
    compression: str | None = "deflate",
    progress: ProgressCallback | None = None,
    keep_working_array: bool = False,
) -> tuple[Path, Path]:
    """Render original or mark-filtered tiles into a tiled pyramidal BigTIFF.

    Output is accumulated one chunk at a time into a disk-backed array, so RAM
    use is bounded by the chunk and a small source-tile cache.
    """
    try:
        import tifffile
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("Writing OME-TIFF requires tifffile") from exc

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tile_w = tile_set.tile_width or 0
    tile_h = tile_set.tile_height or 0
    min_x, min_y, max_x, max_y = mosaic_bounds(
        positions, tile_w, tile_h, tile_rotation_deg
    )
    width, height = max_x - min_x, max_y - min_y
    if width <= 0 or height <= 0:
        raise ValueError("The mosaic has no drawable area")
    shifted = {
        key: np.asarray(value, dtype=np.float64) - np.array([min_x, min_y])
        for key, value in positions.items()
    }
    working_path = output.with_suffix(output.suffix + ".working.uint8")
    mosaic = np.memmap(working_path, dtype=np.uint8, mode="w+", shape=(height, width, 3))
    mosaic[:] = 0

    records = tile_set.by_rc
    rotated_w, rotated_h = _rotated_size(tile_w, tile_h, tile_rotation_deg)
    spatial = _build_spatial_index(
        shifted, tile_w, tile_h, rotated_w, rotated_h, chunk_size
    )
    cache = TileCache(maximum=8, tile_rotation_deg=tile_rotation_deg)
    chunks_x = math.ceil(width / chunk_size)
    chunks_y = math.ceil(height / chunk_size)
    total = chunks_x * chunks_y
    validity_full = np.full((tile_h, tile_w), 255, dtype=np.uint8)
    if abs(tile_rotation_deg) > 1e-8:
        validity_full = np.asarray(
            Image.fromarray(validity_full, mode="L").rotate(
                -tile_rotation_deg,
                resample=Image.Resampling.BICUBIC,
                expand=True,
                fillcolor=0,
            ),
            dtype=np.uint8,
        )
        rotated_h, rotated_w = validity_full.shape
    done = 0
    for chunk_y in range(chunks_y):
        y0 = chunk_y * chunk_size
        y1 = min(height, y0 + chunk_size)
        for chunk_x in range(chunks_x):
            x0 = chunk_x * chunk_size
            x1 = min(width, x0 + chunk_size)
            premultiplied = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.float32)
            coverage = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
            chunk_keys = sorted(
                spatial.get((chunk_x, chunk_y), ()),
                key=lambda key: records[key].slot if key in records else -1,
            )
            later_weight = float(np.clip(overlap_blend_percent / 100.0, 0.0, 1.0))
            for key in chunk_keys:
                record = records.get(key)
                if record is None:
                    continue
                position = shifted[key]
                center_x = float(position[0]) + tile_w / 2.0
                center_y = float(position[1]) + tile_h / 2.0
                tx = int(round(center_x - rotated_w / 2.0))
                ty = int(round(center_y - rotated_h / 2.0))
                ix0, iy0 = max(x0, tx), max(y0, ty)
                ix1, iy1 = min(x1, tx + rotated_w), min(y1, ty + rotated_h)
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                source = cache.get(record.path)
                source_slice = source[iy0 - ty : iy1 - ty, ix0 - tx : ix1 - tx]
                source_alpha = (
                    validity_full[iy0 - ty : iy1 - ty, ix0 - tx : ix1 - tx] >= 128
                ).astype(np.float32)
                cy0, cy1 = iy0 - y0, iy1 - y0
                cx0, cx1 = ix0 - x0, ix1 - x0
                target_coverage = coverage[cy0:cy1, cx0:cx1]
                overlap = (target_coverage > 1e-8) & (source_alpha > 0)
                effective_alpha = source_alpha
                effective_alpha[overlap] *= later_weight
                inverse_alpha = 1.0 - effective_alpha
                target_premultiplied = premultiplied[cy0:cy1, cx0:cx1]
                target_premultiplied *= inverse_alpha[..., None]
                target_premultiplied += source_slice.astype(np.float32) * effective_alpha[..., None]
                target_coverage *= inverse_alpha
                target_coverage += effective_alpha
            valid = coverage > 1e-8
            output_chunk = np.zeros(premultiplied.shape, dtype=np.uint8)
            output_chunk[valid] = np.clip(
                premultiplied[valid] / coverage[valid, None], 0, 255
            ).astype(np.uint8)
            mosaic[y0:y1, x0:x1] = output_chunk
            done += 1
            if progress:
                progress(done, total, "合成大图")
    mosaic.flush()

    write_pyramidal_ome(mosaic, output, compression=compression, tifffile_module=tifffile)

    preview_path = output.with_suffix(".preview.png")
    factor = max(1, math.ceil(max(width, height) / 2200))
    preview = np.asarray(mosaic[::factor, ::factor]).copy()
    Image.fromarray(preview, mode="RGB").save(preview_path, compress_level=4)
    del mosaic
    if not keep_working_array:
        working_path.unlink(missing_ok=True)
    return output, preview_path


def render_mosaic_thumbnail(
    tile_set: TileSet,
    positions: Mapping[tuple[int, int], np.ndarray],
    output_path: str | Path,
    tile_rotation_deg: float = 0.0,
    overlap_blend_percent: float = 50.0,
    maximum_edge: int = 2400,
    progress: ProgressCallback | None = None,
) -> Path:
    """Render a directly downsampled PNG without creating a full-size mosaic.

    Each source tile is reduced before rotation and composition.  Memory use is
    therefore bounded by the requested thumbnail canvas and one source image,
    rather than by the dimensions of the full-resolution stitched image.
    """
    if maximum_edge < 256:
        raise ValueError("缩略图最长边不能小于 256 像素")
    tile_w = tile_set.tile_width or 0
    tile_h = tile_set.tile_height or 0
    if tile_w <= 0 or tile_h <= 0:
        raise ValueError("无法确定图块尺寸")
    if not positions:
        raise ValueError("拼接方案中没有可用位置")

    min_x, min_y, max_x, max_y = mosaic_bounds(
        positions, tile_w, tile_h, tile_rotation_deg
    )
    full_width, full_height = max_x - min_x, max_y - min_y
    if full_width <= 0 or full_height <= 0:
        raise ValueError("拼接画布没有可绘制区域")
    scale = min(1.0, maximum_edge / max(full_width, full_height))
    canvas_width = max(1, min(maximum_edge, round(full_width * scale)))
    canvas_height = max(1, min(maximum_edge, round(full_height * scale)))
    scaled_tile_w = max(1, round(tile_w * scale))
    scaled_tile_h = max(1, round(tile_h * scale))

    canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
    records = tile_set.by_rc
    ordered_keys = sorted(
        (key for key in positions if key in records),
        key=lambda key: records[key].slot,
    )
    if not ordered_keys:
        raise ValueError("没有可用于生成缩略图的图像")
    later_weight = float(np.clip(overlap_blend_percent / 100.0, 0.0, 1.0))

    for index, key in enumerate(ordered_keys, start=1):
        record = records[key]
        source = read_rgb(record.path, max_size=max(scaled_tile_w, scaled_tile_h))
        if source.shape[1] != scaled_tile_w or source.shape[0] != scaled_tile_h:
            source = resize_rgb(source, (scaled_tile_w, scaled_tile_h))
        tile = Image.fromarray(source, mode="RGB").convert("RGBA")
        if abs(tile_rotation_deg) > 1e-8:
            tile = tile.rotate(
                -tile_rotation_deg,
                resample=Image.Resampling.BICUBIC,
                expand=True,
                fillcolor=(0, 0, 0, 0),
            )
        position = np.asarray(positions[key], dtype=np.float64)
        center_x = (float(position[0]) + tile_w / 2.0 - min_x) * scale
        center_y = (float(position[1]) + tile_h / 2.0 - min_y) * scale
        x = int(round(center_x - tile.width / 2.0))
        y = int(round(center_y - tile.height / 2.0))
        _composite_thumbnail_tile(canvas, tile, x, y, later_weight)
        if progress:
            progress(index, len(ordered_keys), "生成拼接缩略图")

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.writing")
    try:
        canvas.convert("RGB").save(temporary, format="PNG", compress_level=4)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _composite_thumbnail_tile(
    destination: Image.Image,
    source: Image.Image,
    x: int,
    y: int,
    later_weight: float,
) -> None:
    """Composite a thumbnail tile using the same overlap choice as previews."""
    left, top = max(0, x), max(0, y)
    right = min(destination.width, x + source.width)
    bottom = min(destination.height, y + source.height)
    if right <= left or bottom <= top:
        return
    source_crop = source.crop((left - x, top - y, right - x, bottom - y))
    destination_crop = destination.crop((left, top, right, bottom))
    source_array = np.asarray(source_crop, dtype=np.uint8).copy()
    destination_alpha = np.asarray(
        destination_crop.getchannel("A"), dtype=np.uint8
    )
    source_alpha = source_array[..., 3]
    overlap = (destination_alpha > 0) & (source_alpha > 0)
    if overlap.any():
        source_alpha[overlap] = np.rint(
            source_alpha[overlap] * later_weight
        ).astype(np.uint8)
    source_array[..., 3] = source_alpha
    destination.alpha_composite(
        Image.fromarray(source_array, mode="RGBA"), (left, top)
    )


def _build_spatial_index(
    positions: Mapping[tuple[int, int], np.ndarray],
    tile_width: int,
    tile_height: int,
    rotated_width: int,
    rotated_height: int,
    chunk_size: int,
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    index: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for key, position in positions.items():
        center_x = float(position[0]) + tile_width / 2.0
        center_y = float(position[1]) + tile_height / 2.0
        x = int(math.floor(center_x - rotated_width / 2.0))
        y = int(math.floor(center_y - rotated_height / 2.0))
        x0, y0 = x // chunk_size, y // chunk_size
        x1 = (x + rotated_width - 1) // chunk_size
        y1 = (y + rotated_height - 1) // chunk_size
        for cy in range(y0, y1 + 1):
            for cx in range(x0, x1 + 1):
                index[(cx, cy)].append(key)
    return index


def write_pyramidal_ome(
    image: np.ndarray,
    output_path: str | Path,
    compression: str | None = "deflate",
    tifffile_module=None,
) -> Path:
    if tifffile_module is None:
        try:
            import tifffile as tifffile_module
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("写入 OME-TIFF 需要 tifffile") from exc
    output = Path(output_path).expanduser().resolve()
    height, width = image.shape[:2]
    sublevels = max(0, min(6, int(math.ceil(math.log2(max(width, height) / 1400.0)))))
    tile_edge = min(512, max(16, 2 ** int(math.floor(math.log2(min(512, width, height))))))
    options = {
        "tile": (tile_edge, tile_edge),
        "compression": compression,
        "photometric": "rgb",
    }
    with tifffile_module.TiffWriter(output, bigtiff=True, kind="ome") as writer:
        writer.write(image, subifds=sublevels, metadata={"axes": "YXS"}, **options)
        for level in range(1, sublevels + 1):
            factor = 2**level
            writer.write(
                image[::factor, ::factor],
                subfiletype=1,
                metadata=None,
                **options,
            )
    return output


def _rotated_size(width: int, height: int, angle_deg: float) -> tuple[int, int]:
    if abs(angle_deg) <= 1e-8:
        return width, height
    probe = Image.new("L", (width, height))
    rotated = probe.rotate(-angle_deg, expand=True)
    return rotated.size
