from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image


class RasterSource(ABC):
    path: Path
    width: int
    height: int
    channels: int

    @abstractmethod
    def read_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def __enter__(self) -> "RasterSource":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class PillowRasterSource(RasterSource):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self._image = Image.open(self.path)
        self.width, self.height = self._image.size
        self.channels = len(self._image.getbands())

    def read_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        if width <= 0 or height <= 0:
            return np.zeros((0, 0, 3), dtype=np.uint8)
        region = self._image.crop((x, y, x + width, y + height)).convert("RGB")
        return np.asarray(region, dtype=np.uint8).copy()

    def close(self) -> None:
        self._image.close()


class TiffRasterSource(RasterSource):
    def __init__(self, path: str | Path) -> None:
        try:
            import tifffile
            import zarr
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("Opening tiled TIFF regions requires tifffile and zarr") from exc
        self.path = Path(path).resolve()
        self._tiff = tifffile.TiffFile(self.path)
        self._store = self._tiff.series[0].aszarr(level=0)
        self._array = zarr.open(self._store, mode="r")
        shape = self._array.shape
        if len(shape) == 2:
            self.height, self.width = shape
            self.channels = 1
        elif len(shape) == 3 and shape[-1] in (3, 4):
            self.height, self.width, self.channels = shape
        else:
            raise ValueError(f"Unsupported TIFF array shape: {shape}")

    def read_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        region = np.asarray(self._array[y : y + height, x : x + width])
        if region.ndim == 2:
            region = np.repeat(region[..., None], 3, axis=2)
        return region[..., :3].copy()

    def close(self) -> None:
        close = getattr(self._store, "close", None)
        if close:
            close()
        self._tiff.close()


def open_raster(path: str | Path) -> RasterSource:
    suffix = Path(path).suffix.lower()
    if suffix in {".tif", ".tiff"}:
        try:
            return TiffRasterSource(path)
        except Exception:
            return PillowRasterSource(path)
    return PillowRasterSource(path)
