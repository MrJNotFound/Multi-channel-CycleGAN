from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class StartCorner(str, Enum):
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


class FastAxis(str, Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class ToolMarkMethod(str, Enum):
    FFT_GAUSSIAN_NOTCH = "fft_gaussian_notch"
    PERIODIC_TEMPLATE = "periodic_template"
    DIRECTIONAL_1D = "directional_1d"


@dataclass(frozen=True, slots=True)
class ScanPattern:
    start_corner: StartCorner = StartCorner.TOP_LEFT
    fast_axis: FastAxis = FastAxis.HORIZONTAL
    snake: bool = True

    def slot_to_rc(self, slot: int, rows: int, cols: int) -> tuple[int, int]:
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be positive")
        if not 0 <= slot < rows * cols:
            raise IndexError(f"slot {slot} is outside a {rows}x{cols} grid")

        starts_top = self.start_corner in {
            StartCorner.TOP_LEFT,
            StartCorner.TOP_RIGHT,
        }
        starts_left = self.start_corner in {
            StartCorner.TOP_LEFT,
            StartCorner.BOTTOM_LEFT,
        }

        if self.fast_axis is FastAxis.HORIZONTAL:
            acquisition_row, within = divmod(slot, cols)
            row = acquisition_row if starts_top else rows - 1 - acquisition_row
            left_to_right = starts_left
            if self.snake and acquisition_row % 2:
                left_to_right = not left_to_right
            col = within if left_to_right else cols - 1 - within
            return row, col

        acquisition_col, within = divmod(slot, rows)
        col = acquisition_col if starts_left else cols - 1 - acquisition_col
        top_to_bottom = starts_top
        if self.snake and acquisition_col % 2:
            top_to_bottom = not top_to_bottom
        row = within if top_to_bottom else rows - 1 - within
        return row, col


@dataclass(frozen=True, slots=True)
class TileRecord:
    number: int
    slot: int
    row: int
    col: int
    path: Path


@dataclass(slots=True)
class TileSet:
    folder: Path
    rows: int
    cols: int
    pattern: ScanPattern
    records: list[TileRecord] = field(default_factory=list)
    missing_slots: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tile_width: int | None = None
    tile_height: int | None = None

    @property
    def by_rc(self) -> dict[tuple[int, int], TileRecord]:
        return {(tile.row, tile.col): tile for tile in self.records}

    @property
    def count(self) -> int:
        return len(self.records)


@dataclass(slots=True)
class GridGeometry:
    overlap_x_percent: float = 10.0
    overlap_y_percent: float = 10.0
    tile_rotation_deg: float = 0.0

    def validate(self) -> None:
        for name, value in (
            ("overlap_x_percent", self.overlap_x_percent),
            ("overlap_y_percent", self.overlap_y_percent),
        ):
            if not 0.0 <= value <= 30.0:
                raise ValueError(f"{name} must be between 0 and 30")
        if not -15.0 <= self.tile_rotation_deg <= 15.0:
            raise ValueError("tile_rotation_deg must be between -15 and 15")


@dataclass(slots=True)
class AutoSearchSettings:
    overlap_min_percent: float = 0.0
    overlap_max_percent: float = 30.0
    rotation_min_deg: float = -8.0
    rotation_max_deg: float = 8.0
    sample_pairs_per_direction: int = 24
    registration_size: int = 1024

    def validate(self) -> None:
        if not 0.0 <= self.overlap_min_percent <= self.overlap_max_percent <= 30.0:
            raise ValueError("自动重叠范围必须在 0%–30% 内且下限不大于上限")
        if not -15.0 <= self.rotation_min_deg <= self.rotation_max_deg <= 15.0:
            raise ValueError("自动旋转范围必须在 -15°–15° 内且下限不大于上限")
        if not 2 <= self.sample_pairs_per_direction <= 200:
            raise ValueError("每个方向的抽样相邻对必须在 2–200 之间")
        if self.registration_size not in {512, 768, 1024, 1536, 2048}:
            raise ValueError("不支持的配准最长边")


@dataclass(slots=True)
class MarkFilterSettings:
    enabled: bool = True
    method: str = ToolMarkMethod.FFT_GAUSSIAN_NOTCH.value
    strength: float = 0.72
    auto_orientation: bool = True
    orientation_deg: float = 0.0
    auto_period: bool = True
    period_px: float = 18.0
    angular_width_deg: float = 5.0
    frequency_width: float = 0.0035
    min_frequency: float = 0.008
    max_frequency: float = 0.32
    max_peaks: int = 14
    template_smoothing: float = 1.2
    directional_kernel: int = 51
    block_size: int = 1024
    block_halo: int = 128

    def validate(self) -> None:
        ToolMarkMethod(self.method)
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("去除强度必须在 0–1 之间")
        if not -90.0 <= self.orientation_deg <= 90.0:
            raise ValueError("刀纹法向角必须在 -90°–90° 之间")
        if not 3.0 <= self.period_px <= 512.0:
            raise ValueError("刀纹周期必须在 3–512 像素之间")
        if self.directional_kernel < 3:
            raise ValueError("一维滤波核至少为 3")


@dataclass(slots=True)
class StitchProject:
    schema_version: int = 4
    source_folder: str = ""
    source_folder_relative: str = ""
    rows: int = 1
    cols: int = 1
    start_corner: str = StartCorner.TOP_LEFT.value
    fast_axis: str = FastAxis.HORIZONTAL.value
    snake: bool = True
    stitch_mode: str = "automatic"
    tile_width: int = 0
    tile_height: int = 0
    geometry: GridGeometry = field(default_factory=GridGeometry)
    overlap_blend_percent: float = 50.0
    auto_search: AutoSearchSettings = field(default_factory=AutoSearchSettings)
    excluded_numbers: list[int] = field(default_factory=list)
    refined_positions: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StitchProject":
        data = dict(payload)
        previous_schema = int(data.get("schema_version", 1))
        geometry = dict(data.get("geometry", {}))
        if "tile_rotation_deg" not in geometry:
            # Version 1 tilted the grid instead of the camera frame. Rotating
            # the tile by the opposite angle is the closest useful migration;
            # the obsolete non-orthogonality value is intentionally ignored.
            geometry["tile_rotation_deg"] = -float(geometry.pop("angle_deg", 0.0))
            geometry.pop("orthogonality_error_deg", None)
        data["geometry"] = GridGeometry(**geometry)
        data["auto_search"] = AutoSearchSettings(**data.get("auto_search", {}))
        data.pop("mark_filter", None)
        if previous_schema < 2:
            # Version 1 refined coordinates could contain a tilted or
            # non-orthogonal grid, which is invalid under the machine model.
            data["refined_positions"] = {}
        if data.get("stitch_mode") not in {"automatic", "manual"}:
            data["stitch_mode"] = "automatic"
        data["overlap_blend_percent"] = float(data.get("overlap_blend_percent", 50.0))
        if not 0.0 <= data["overlap_blend_percent"] <= 100.0:
            data["overlap_blend_percent"] = 50.0
        data["tile_width"] = max(0, int(data.get("tile_width", 0)))
        data["tile_height"] = max(0, int(data.get("tile_height", 0)))
        data["schema_version"] = 4
        return cls(**data)
