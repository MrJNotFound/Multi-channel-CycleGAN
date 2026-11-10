from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from PIL import Image

from .raster import RasterSource, open_raster


@dataclass(slots=True)
class CropSettings:
    mode: Literal["size", "count"] = "size"
    patch_width: int = 1024
    patch_height: int = 1024
    rows: int = 10
    cols: int = 10
    overlap_x_percent: float = 0.0
    overlap_y_percent: float = 0.0
    angle_deg: float = 0.0
    edge_mode: Literal["pad", "reflect", "skip"] = "pad"
    output_format: Literal["png", "tiff", "jpeg"] = "png"

    def validate(self) -> None:
        if self.patch_width < 1 or self.patch_height < 1:
            raise ValueError("Patch dimensions must be positive")
        if self.rows < 1 or self.cols < 1:
            raise ValueError("Rows and columns must be positive")
        if not 0 <= self.overlap_x_percent <= 30 or not 0 <= self.overlap_y_percent <= 30:
            raise ValueError("Crop overlap must be between 0 and 30 percent")


@dataclass(frozen=True, slots=True)
class PatchSpec:
    row: int
    col: int
    center_x: float
    center_y: float
    width: int
    height: int
    angle_deg: float


def patch_layout(width: int, height: int, settings: CropSettings) -> list[PatchSpec]:
    settings.validate()
    ox = settings.overlap_x_percent / 100.0
    oy = settings.overlap_y_percent / 100.0
    if settings.mode == "count":
        cols, rows = settings.cols, settings.rows
        patch_w = int(math.ceil(width / (cols - (cols - 1) * ox)))
        patch_h = int(math.ceil(height / (rows - (rows - 1) * oy)))
    else:
        patch_w, patch_h = settings.patch_width, settings.patch_height
        step_x = max(1.0, patch_w * (1.0 - ox))
        step_y = max(1.0, patch_h * (1.0 - oy))
        cols = max(1, math.ceil((width - patch_w) / step_x) + 1)
        rows = max(1, math.ceil((height - patch_h) / step_y) + 1)
    step_x = patch_w * (1.0 - ox)
    step_y = patch_h * (1.0 - oy)
    angle = math.radians(settings.angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    image_center = np.array([width / 2.0, height / 2.0])
    specs: list[PatchSpec] = []
    for row in range(rows):
        for col in range(cols):
            unrotated = np.array([col * step_x + patch_w / 2.0, row * step_y + patch_h / 2.0])
            delta = unrotated - image_center
            center = image_center + np.array(
                [cos_a * delta[0] - sin_a * delta[1], sin_a * delta[0] + cos_a * delta[1]]
            )
            specs.append(PatchSpec(row, col, float(center[0]), float(center[1]), patch_w, patch_h, settings.angle_deg))
    return specs


def export_patches(
    source_path: str | Path,
    output_dir: str | Path,
    settings: CropSettings,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[Path, int]:
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "manifest.csv"
    written = 0
    with open_raster(source_path) as source, manifest_path.open("w", newline="", encoding="utf-8") as handle:
        specs = patch_layout(source.width, source.height, settings)
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "row", "col", "center_x", "center_y", "width", "height", "angle_deg", "source"],
        )
        writer.writeheader()
        for index, spec in enumerate(specs, 1):
            patch, complete = _read_rotated_patch(source, spec, settings.edge_mode)
            if patch is None:
                if progress:
                    progress(index, len(specs), "分割")
                continue
            suffix = {"png": ".png", "tiff": ".tif", "jpeg": ".jpg"}[settings.output_format]
            filename = f"patch_r{spec.row:04d}_c{spec.col:04d}{suffix}"
            save_kwargs = {"quality": 95} if settings.output_format == "jpeg" else {}
            Image.fromarray(patch, mode="RGB").save(target / filename, **save_kwargs)
            writer.writerow(
                {
                    "filename": filename,
                    "row": spec.row,
                    "col": spec.col,
                    "center_x": f"{spec.center_x:.3f}",
                    "center_y": f"{spec.center_y:.3f}",
                    "width": spec.width,
                    "height": spec.height,
                    "angle_deg": f"{spec.angle_deg:.4f}",
                    "source": str(Path(source_path).resolve()),
                }
            )
            written += 1
            if progress:
                progress(index, len(specs), "分割")
    settings_path = target / "split_settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "large_image_split",
                "source": str(Path(source_path).resolve()),
                "output_folder": str(target),
                "settings": asdict(settings),
                "written_count": written,
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path, written


def save_crop_settings(path: str | Path, settings: CropSettings) -> Path:
    target = Path(path).expanduser().resolve()
    target.write_text(
        json.dumps(
            {"schema_version": 2, "kind": "large_image_split", "settings": asdict(settings)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return target


def load_crop_settings(path: str | Path) -> CropSettings:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CropSettings(**payload.get("settings", payload))


def _read_rotated_patch(
    source: RasterSource,
    spec: PatchSpec,
    edge_mode: str,
) -> tuple[np.ndarray | None, bool]:
    angle = math.radians(spec.angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    half_w, half_h = spec.width / 2.0, spec.height / 2.0
    bound_w = abs(spec.width * cos_a) + abs(spec.height * sin_a)
    bound_h = abs(spec.width * sin_a) + abs(spec.height * cos_a)
    bx0 = math.floor(spec.center_x - bound_w / 2.0) - 2
    by0 = math.floor(spec.center_y - bound_h / 2.0) - 2
    bx1 = math.ceil(spec.center_x + bound_w / 2.0) + 2
    by1 = math.ceil(spec.center_y + bound_h / 2.0) + 2
    complete = bx0 >= 0 and by0 >= 0 and bx1 <= source.width and by1 <= source.height
    if edge_mode == "skip" and not complete:
        return None, False

    ix0, iy0 = max(0, bx0), max(0, by0)
    ix1, iy1 = min(source.width, bx1), min(source.height, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        if edge_mode == "skip":
            return None, False
        region = np.zeros((max(1, by1 - by0), max(1, bx1 - bx0), 3), dtype=np.uint8)
    else:
        read = source.read_region(ix0, iy0, ix1 - ix0, iy1 - iy0)
        pad_top, pad_left = iy0 - by0, ix0 - bx0
        pad_bottom, pad_right = by1 - iy1, bx1 - ix1
        pad_mode = "reflect" if edge_mode == "reflect" and min(read.shape[:2]) > 1 else "constant"
        region = np.pad(read, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode=pad_mode)

    local_cx, local_cy = spec.center_x - bx0, spec.center_y - by0
    coefficients = (
        cos_a,
        -sin_a,
        local_cx - half_w * cos_a + half_h * sin_a,
        sin_a,
        cos_a,
        local_cy - half_w * sin_a - half_h * cos_a,
    )
    patch = Image.fromarray(region, mode="RGB").transform(
        (spec.width, spec.height),
        Image.Transform.AFFINE,
        coefficients,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0),
    )
    return np.asarray(patch).copy(), complete
