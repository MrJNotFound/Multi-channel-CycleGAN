from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .models import GridGeometry, TileSet


def step_vectors(
    geometry: GridGeometry, tile_width: int, tile_height: int
) -> tuple[np.ndarray, np.ndarray]:
    geometry.validate()
    sx = tile_width * (1.0 - geometry.overlap_x_percent / 100.0)
    sy = tile_height * (1.0 - geometry.overlap_y_percent / 100.0)
    # Machine X/Y guideways define the global frame and are orthogonal.  CCD
    # installation error rotates image content, never the grid of tile centers.
    column = np.array([sx, 0.0], dtype=np.float64)
    row = np.array([0.0, sy], dtype=np.float64)
    return column, row


def regular_positions(
    tile_set: TileSet, geometry: GridGeometry
) -> dict[tuple[int, int], np.ndarray]:
    if tile_set.tile_width is None or tile_set.tile_height is None:
        raise ValueError("Tile dimensions are unknown")
    column, row = step_vectors(geometry, tile_set.tile_width, tile_set.tile_height)
    return {
        (grid_row, grid_col): grid_col * column + grid_row * row
        for grid_row in range(tile_set.rows)
        for grid_col in range(tile_set.cols)
    }


def normalized_positions(
    positions: Mapping[tuple[int, int], np.ndarray],
) -> tuple[dict[tuple[int, int], np.ndarray], np.ndarray]:
    if not positions:
        return {}, np.zeros(2, dtype=np.float64)
    values = np.vstack(list(positions.values()))
    offset = values.min(axis=0)
    return {key: value - offset for key, value in positions.items()}, offset


def mosaic_bounds(
    positions: Mapping[tuple[int, int], np.ndarray],
    tile_width: int,
    tile_height: int,
    tile_rotation_deg: float = 0.0,
) -> tuple[int, int, int, int]:
    if not positions:
        return 0, 0, 0, 0
    values = np.vstack(list(positions.values()))
    angle = math.radians(tile_rotation_deg)
    rotated_width = abs(tile_width * math.cos(angle)) + abs(tile_height * math.sin(angle))
    rotated_height = abs(tile_width * math.sin(angle)) + abs(tile_height * math.cos(angle))
    centers = values + np.array([tile_width / 2.0, tile_height / 2.0])
    min_x = math.floor(float(centers[:, 0].min() - rotated_width / 2.0))
    min_y = math.floor(float(centers[:, 1].min() - rotated_height / 2.0))
    max_x = math.ceil(float(centers[:, 0].max() + rotated_width / 2.0))
    max_y = math.ceil(float(centers[:, 1].max() + rotated_height / 2.0))
    return min_x, min_y, max_x, max_y


def fit_grid_geometry(
    horizontal_vectors: np.ndarray,
    vertical_vectors: np.ndarray,
    tile_width: int,
    tile_height: int,
) -> GridGeometry:
    if horizontal_vectors.size == 0 or vertical_vectors.size == 0:
        raise ValueError("Both horizontal and vertical vectors are required")
    horizontal = np.median(horizontal_vectors, axis=0)
    vertical = np.median(vertical_vectors, axis=0)
    h_length = float(np.linalg.norm(horizontal))
    v_length = float(np.linalg.norm(vertical))
    h_angle = math.degrees(math.atan2(horizontal[1], horizontal[0]))
    v_angle = math.degrees(math.atan2(vertical[1], vertical[0])) - 90.0
    camera_frame_angle = _axial_mean_deg(h_angle, v_angle)
    return GridGeometry(
        overlap_x_percent=float(np.clip(100.0 * (1.0 - h_length / tile_width), 0.0, 30.0)),
        overlap_y_percent=float(np.clip(100.0 * (1.0 - v_length / tile_height), 0.0, 30.0)),
        tile_rotation_deg=float(np.clip(-camera_frame_angle, -15.0, 15.0)),
    )


def _wrap_angle(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _axial_mean_deg(first: float, second: float) -> float:
    first_rad, second_rad = math.radians(first), math.radians(second)
    x = math.cos(first_rad) + math.cos(second_rad)
    y = math.sin(first_rad) + math.sin(second_rad)
    return _wrap_angle(math.degrees(math.atan2(y, x)))
