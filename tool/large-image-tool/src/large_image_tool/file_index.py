from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from .models import ScanPattern, TileRecord, TileSet


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
NUMERIC_STEM = re.compile(r"^\d+$")
ROW_COL_STEM = re.compile(r"^tile_r(?P<row>\d+)_c(?P<col>\d+)$", re.IGNORECASE)


class TileIndexError(ValueError):
    pass


def discover_numeric_tiles(
    folder: str | Path,
    rows: int,
    cols: int,
    pattern: ScanPattern,
) -> TileSet:
    """Index files whose entire stem is numeric.

    Numeric gaps are preserved as empty acquisition slots. Numbering is
    explicitly interpreted as one-based, unless a file named ``0`` exists.
    This avoids silently shifting every tile when the first numbered image is
    missing.
    """
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise TileIndexError(f"Folder does not exist: {root}")
    if rows < 1 or cols < 1:
        raise TileIndexError("Rows and columns must be positive")

    numbered: dict[int, Path] = {}
    ignored = 0
    for path in root.iterdir():
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if not NUMERIC_STEM.fullmatch(path.stem):
            ignored += 1
            continue
        number = int(path.stem)
        if number in numbered:
            raise TileIndexError(
                f"Duplicate numeric name {number}: {numbered[number].name}, {path.name}"
            )
        numbered[number] = path

    if not numbered:
        raise TileIndexError("No images with a purely numeric filename were found")

    first = 0 if 0 in numbered else 1
    capacity = rows * cols
    records: list[TileRecord] = []
    outside: list[int] = []
    for number, path in sorted(numbered.items()):
        slot = number - first
        if slot >= capacity:
            outside.append(number)
            continue
        row, col = pattern.slot_to_rc(slot, rows, cols)
        records.append(TileRecord(number, slot, row, col, path))

    present_slots = {record.slot for record in records}
    missing = sorted(set(range(capacity)) - present_slots)
    warnings: list[str] = []
    if missing:
        warnings.append(f"缺少 {len(missing)} 个编号，已保留为空位")
    if outside:
        warnings.append(
            f"{len(outside)} 个文件超出 {rows}×{cols} 网格，未加载"
        )
    if ignored:
        warnings.append(f"{ignored} 个图像文件不是纯数字命名，未加载")

    width, height = _read_common_size(records)
    return TileSet(root, rows, cols, pattern, records, missing, warnings, width, height)


def discover_row_col_tiles(folder: str | Path, rows: int, cols: int) -> TileSet:
    root = Path(folder).expanduser().resolve()
    records: list[TileRecord] = []
    warnings: list[str] = []
    occupied: set[tuple[int, int]] = set()
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        match = ROW_COL_STEM.fullmatch(path.stem)
        if not match:
            continue
        row, col = int(match.group("row")), int(match.group("col"))
        if not (0 <= row < rows and 0 <= col < cols):
            warnings.append(f"Ignored out-of-grid file: {path.name}")
            continue
        if (row, col) in occupied:
            raise TileIndexError(f"Duplicate row/column position: r{row} c{col}")
        occupied.add((row, col))
        slot = row * cols + col
        records.append(TileRecord(slot, slot, row, col, path))

    if not records:
        raise TileIndexError("No tile_r####_c#### image files were found")
    missing = [r * cols + c for r in range(rows) for c in range(cols) if (r, c) not in occupied]
    if missing:
        warnings.append(f"缺少 {len(missing)} 个位置，已保留为空位")
    width, height = _read_common_size(records)
    return TileSet(root, rows, cols, ScanPattern(), records, missing, warnings, width, height)


def _read_common_size(records: list[TileRecord]) -> tuple[int, int]:
    first_size: tuple[int, int] | None = None
    for record in records[: min(24, len(records))]:
        with Image.open(record.path) as image:
            size = image.size
        if first_size is None:
            first_size = size
        elif size != first_size:
            raise TileIndexError(
                f"Image size mismatch: {record.path.name} is {size}, expected {first_size}"
            )
    assert first_size is not None
    return first_size
