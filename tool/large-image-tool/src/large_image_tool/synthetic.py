from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .geometry import step_vectors
from .models import GridGeometry, ScanPattern


def generate_turning_dataset(
    output_dir: str | Path,
    rows: int = 5,
    cols: int = 6,
    tile_size: tuple[int, int] = (640, 360),
    geometry: GridGeometry | None = None,
    pattern: ScanPattern | None = None,
    mark_period: float = 18.0,
    mark_strength: float = 0.34,
    jitter_px: float = 1.2,
    seed: int = 7,
) -> Path:
    """Generate a colored, lathe-like grid with known ground-truth positions."""
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    geometry = geometry or GridGeometry(14.0, 12.0, 1.4)
    pattern = pattern or ScanPattern()
    tile_w, tile_h = tile_size
    column, row = step_vectors(geometry, tile_w, tile_h)
    nominal = {
        (r, c): c * column + r * row for r in range(rows) for c in range(cols)
    }
    coords = np.vstack(list(nominal.values()))
    offset = -np.floor(coords.min(axis=0)) + np.array([tile_w * 0.8, tile_h * 0.8])
    rng = np.random.default_rng(seed)
    jitter = {
        key: rng.normal(0.0, jitter_px, size=2) for key in nominal
    }
    positions = {key: value + offset + jitter[key] for key, value in nominal.items()}
    values = np.vstack(list(positions.values()))
    base_w = int(math.ceil(values[:, 0].max() + tile_w + tile_w * 0.8))
    base_h = int(math.ceil(values[:, 1].max() + tile_h + tile_h * 0.8))
    base = _turning_surface(base_w, base_h, mark_period, mark_strength, rng)
    base_image = Image.fromarray(base, mode="RGB")

    metadata: dict[str, object] = {
        "rows": rows,
        "cols": cols,
        "tile_width": tile_w,
        "tile_height": tile_h,
        "geometry": {
            "overlap_x_percent": geometry.overlap_x_percent,
            "overlap_y_percent": geometry.overlap_y_percent,
            "tile_rotation_deg": geometry.tile_rotation_deg,
        },
        "positions": {},
    }
    for slot in range(rows * cols):
        r, c = pattern.slot_to_rc(slot, rows, cols)
        x, y = positions[(r, c)]
        angle = math.radians(geometry.tile_rotation_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        center_x, center_y = x + tile_w / 2.0, y + tile_h / 2.0
        coefficients = (
            cos_a,
            -sin_a,
            center_x - tile_w * cos_a / 2.0 + tile_h * sin_a / 2.0,
            sin_a,
            cos_a,
            center_y - tile_w * sin_a / 2.0 - tile_h * cos_a / 2.0,
        )
        tile = np.asarray(
            base_image.transform(
                (tile_w, tile_h),
                Image.Transform.AFFINE,
                coefficients,
                resample=Image.Resampling.BICUBIC,
            )
        ).astype(np.float32)
        yy, xx = np.mgrid[:tile_h, :tile_w]
        vignette = 1.0 - 0.10 * (((xx - tile_w / 2) / tile_w) ** 2 + ((yy - tile_h / 2) / tile_h) ** 2)
        gain = 0.94 + 0.12 * rng.random()
        bias = rng.normal(0.0, 2.0)
        tile = np.clip(tile * vignette[..., None] * gain + bias + rng.normal(0, 1.2, tile.shape), 0, 255).astype(np.uint8)
        number = slot + 1
        filename = f"{number:06d}.png"
        Image.fromarray(tile, mode="RGB").save(target / filename, compress_level=3)
        metadata["positions"][filename] = {
            "row": r,
            "col": c,
            "x": float(x - offset[0]),
            "y": float(y - offset[1]),
        }

    (target / "ground_truth.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


def _turning_surface(
    width: int,
    height: int,
    period: float,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    center_x = -0.38 * width
    center_y = 0.56 * height
    radius = np.hypot(x - center_x, y - center_y)
    rings = np.sin(2.0 * np.pi * radius / period)
    rings += 0.28 * np.sin(4.0 * np.pi * radius / period + 0.7)
    low_noise = ndimage.gaussian_filter(rng.normal(0, 1, (height, width)), sigma=8.0)
    low_noise /= max(float(low_noise.std()), 1e-6)
    fine = ndimage.gaussian_filter(rng.normal(0, 1, (height, width)), sigma=1.1)
    fine /= max(float(fine.std()), 1e-6)
    material = 0.40 * low_noise + 0.17 * fine
    defect = np.zeros((height, width), dtype=np.float32)
    for _ in range(max(18, width * height // 180_000)):
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        sx, sy = rng.uniform(3, 18), rng.uniform(3, 18)
        amplitude = rng.uniform(-2.2, 2.2)
        defect += amplitude * np.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))
    luminance = 128 + 55 * material + 74 * strength * rings + 24 * defect
    red = luminance * 1.05 + 7 * np.sin(radius / 83.0)
    green = luminance * 0.98 + 4 * low_noise
    blue = luminance * 0.89 - 5 * np.sin(radius / 121.0)
    return np.clip(np.stack([red, green, blue], axis=-1), 0, 255).astype(np.uint8)
