from __future__ import annotations

import math
import threading
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
from PIL import Image, ImageDraw

from .geometry import step_vectors
from .image_ops import read_rgb, resize_rgb, suppress_tool_marks
from .models import GridGeometry, MarkFilterSettings, TileSet


_THUMBNAIL_CACHE: OrderedDict[tuple[str, int, int], np.ndarray] = OrderedDict()
_THUMBNAIL_CACHE_LOCK = threading.Lock()
_THUMBNAIL_CACHE_LIMIT = 40


def _read_cached_thumbnail(path, maximum: int) -> np.ndarray:
    resolved = path.resolve()
    try:
        modified = resolved.stat().st_mtime_ns
    except OSError:
        modified = 0
    key = (str(resolved), modified, maximum)
    with _THUMBNAIL_CACHE_LOCK:
        cached = _THUMBNAIL_CACHE.pop(key, None)
        if cached is not None:
            _THUMBNAIL_CACHE[key] = cached
            return cached
    image = read_rgb(resolved, max_size=maximum)
    image.setflags(write=False)
    with _THUMBNAIL_CACHE_LOCK:
        _THUMBNAIL_CACHE[key] = image
        while len(_THUMBNAIL_CACHE) > _THUMBNAIL_CACHE_LIMIT:
            _THUMBNAIL_CACHE.popitem(last=False)
    return image


def local_records(
    tile_set: TileSet,
    center_slot: int,
    window_size: int,
) -> list:
    if window_size not in {2, 3, 4}:
        raise ValueError("Preview size must be 2, 3, or 4")
    center_slot = min(max(0, center_slot), tile_set.rows * tile_set.cols - 1)
    center_row, center_col = tile_set.pattern.slot_to_rc(center_slot, tile_set.rows, tile_set.cols)
    before = (window_size - 1) // 2
    start_row = min(max(0, center_row - before), max(0, tile_set.rows - window_size))
    start_col = min(max(0, center_col - before), max(0, tile_set.cols - window_size))
    by_rc = tile_set.by_rc
    return [
        by_rc[(row, col)]
        for row in range(start_row, min(tile_set.rows, start_row + window_size))
        for col in range(start_col, min(tile_set.cols, start_col + window_size))
        if (row, col) in by_rc
    ]


def compose_local_preview(
    tile_set: TileSet,
    geometry: GridGeometry,
    center_slot: int,
    window_size: int = 3,
    thumb_width: int = 480,
    positions: Mapping[tuple[int, int], np.ndarray] | None = None,
    show_labels: bool = True,
    mark_filter: MarkFilterSettings | None = None,
    overlap_blend_percent: float = 50.0,
) -> Image.Image:
    records = sorted(
        local_records(tile_set, center_slot, window_size),
        key=lambda record: record.slot,
    )
    if not records:
        return Image.new("RGB", (640, 420), (25, 31, 38))
    full_w = tile_set.tile_width or 1
    full_h = tile_set.tile_height or 1
    scale = min(1.0, thumb_width / full_w)
    thumb_h = max(1, round(full_h * scale))
    if positions is None:
        column, row = step_vectors(geometry, full_w, full_h)
        positions = {
            (record.row, record.col): record.col * column + record.row * row
            for record in records
        }
    scaled_positions = {
        key: np.asarray(value, dtype=np.float64) * scale for key, value in positions.items()
        if key in {(record.row, record.col) for record in records}
    }
    probe = Image.new("L", (thumb_width, thumb_h)).rotate(
        -geometry.tile_rotation_deg, expand=True
    )
    rotated_w, rotated_h = probe.size
    coords = np.vstack(list(scaled_positions.values()))
    centers = coords + np.array([thumb_width / 2.0, thumb_h / 2.0])
    minimum = np.floor(centers.min(axis=0) - np.array([rotated_w / 2.0, rotated_h / 2.0]))
    maximum = np.ceil(centers.max(axis=0) + np.array([rotated_w / 2.0, rotated_h / 2.0]))
    canvas_size = tuple(np.maximum(1, (maximum - minimum).astype(int)))
    # Keep the image layer transparent while composing.  Rotated tile corners
    # are empty pixels, not black image data, and therefore must never cover a
    # valid pixel from another tile.
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    label_layer = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(label_layer)

    for record in records:
        array = _read_cached_thumbnail(record.path, thumb_width)
        if array.shape[1] != thumb_width:
            array = resize_rgb(array, (thumb_width, thumb_h))
        if mark_filter and mark_filter.enabled:
            array, _report = suppress_tool_marks(array, mark_filter)
        tile = Image.fromarray(array, mode="RGB").convert("RGBA")
        if abs(geometry.tile_rotation_deg) > 1e-8:
            tile = tile.rotate(
                -geometry.tile_rotation_deg,
                resample=Image.Resampling.BICUBIC,
                expand=True,
                fillcolor=(0, 0, 0, 0),
            )
        center = (
            scaled_positions[(record.row, record.col)]
            + np.array([thumb_width / 2.0, thumb_h / 2.0])
            - minimum
        )
        x = int(round(center[0] - tile.width / 2.0))
        y = int(round(center[1] - tile.height / 2.0))
        _composite_with_overlap_choice(
            canvas,
            tile,
            x,
            y,
            overlap_blend_percent / 100.0,
        )
        if show_labels:
            draw.rectangle((x, y, x + tile.width - 1, y + tile.height - 1), outline=(228, 240, 246, 205), width=1)
            draw.rectangle((x + 5, y + 5, x + 122, y + 24), fill=(10, 16, 21, 176))
            draw.text((x + 9, y + 8), f"{record.number}  r{record.row} c{record.col}", fill=(245, 248, 250, 255))
    background = Image.new("RGBA", canvas_size, (19, 25, 31, 255))
    background.alpha_composite(canvas)
    background.alpha_composite(label_layer)
    return background.convert("RGB")


def _composite_with_overlap_choice(
    destination: Image.Image,
    source: Image.Image,
    x: int,
    y: int,
    later_weight: float,
) -> None:
    """Composite one tile while selecting earlier/later pixels in overlaps."""
    later_weight = float(np.clip(later_weight, 0.0, 1.0))
    left, top = max(0, x), max(0, y)
    right = min(destination.width, x + source.width)
    bottom = min(destination.height, y + source.height)
    if right <= left or bottom <= top:
        return
    src = source.crop((left - x, top - y, right - x, bottom - y))
    dst = destination.crop((left, top, right, bottom))
    src_array = np.asarray(src, dtype=np.uint8).copy()
    dst_alpha = np.asarray(dst.getchannel("A"), dtype=np.uint8)
    src_alpha = src_array[..., 3]
    overlap = (dst_alpha > 0) & (src_alpha > 0)
    if overlap.any():
        src_alpha[overlap] = np.rint(src_alpha[overlap] * later_weight).astype(np.uint8)
    src_array[..., 3] = src_alpha
    destination.alpha_composite(Image.fromarray(src_array, mode="RGBA"), (left, top))


def estimated_preview_memory_mb(window_size: int, tile_width: int, tile_height: int) -> float:
    return window_size * window_size * tile_width * tile_height * 3 / 1_000_000.0
