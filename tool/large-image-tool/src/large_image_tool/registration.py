from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import linalg as sparse_linalg

from .geometry import fit_grid_geometry, regular_positions
from .image_ops import read_rgb, registration_channel, suppress_tool_marks
from .models import GridGeometry, MarkFilterSettings, TileRecord, TileSet


Direction = Literal["horizontal", "vertical"]


@dataclass(slots=True)
class ShiftCandidate:
    dx: float
    dy: float
    phase_value: float
    ncc: float
    score: float


@dataclass(slots=True)
class PairwiseShift:
    source: tuple[int, int]
    target: tuple[int, int]
    dx: float
    dy: float
    confidence: float
    ncc: float
    used_filtered: bool = False
    valid: bool = True


@dataclass(slots=True)
class AutoRegistrationResult:
    geometry: GridGeometry
    positions: dict[tuple[int, int], np.ndarray]
    edges: list[PairwiseShift] = field(default_factory=list)
    low_confidence_edges: list[PairwiseShift] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


def phase_correlation_candidates(
    reference: np.ndarray,
    moving: np.ndarray,
    direction: Direction,
    overlap_range: tuple[float, float] = (0.0, 30.0),
    rotation_range: tuple[float, float] = (-8.0, 8.0),
    top_k: int = 8,
    prior: np.ndarray | None = None,
) -> list[ShiftCandidate]:
    """Find plausible moving-image placements relative to reference.

    The FFT correlation is circular, so every selected peak is expanded into
    equivalent translations. Only offsets compatible with a right or lower
    grid neighbor and the configured overlap range are retained.
    """
    a = _prepare_channel(reference)
    b = _prepare_channel(moving)
    if a.shape != b.shape:
        raise ValueError("Pairwise registration requires equal image shapes")
    height, width = a.shape
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = fa * np.conj(fb)
    cross /= np.maximum(np.abs(cross), 1e-8)
    corr = np.abs(np.fft.ifft2(cross))
    maxima = corr == ndimage.maximum_filter(corr, size=5, mode="wrap")
    peak_indices = np.flatnonzero(maxima)
    if peak_indices.size == 0:
        return []
    values = corr.ravel()[peak_indices]
    take = min(max(top_k * 3, top_k), peak_indices.size)
    chosen = peak_indices[np.argpartition(values, -take)[-take:]]
    chosen = chosen[np.argsort(corr.ravel()[chosen])[::-1]]

    min_overlap, max_overlap = overlap_range
    min_step_x = width * (1.0 - max_overlap / 100.0)
    max_step_x = width * (1.0 - min_overlap / 100.0)
    min_step_y = height * (1.0 - max_overlap / 100.0)
    max_step_y = height * (1.0 - min_overlap / 100.0)
    rotation_min, rotation_max = rotation_range
    camera_angle_min, camera_angle_max = -rotation_max, -rotation_min
    candidates: list[ShiftCandidate] = []
    seen: set[tuple[int, int]] = set()
    for flat in chosen:
        py, px = np.unravel_index(flat, corr.shape)
        raw_y = float(py if py <= height // 2 else py - height)
        raw_x = float(px if px <= width // 2 else px - width)
        for ky in (-1, 0, 1):
            for kx in (-1, 0, 1):
                dy = raw_y + ky * height
                dx = raw_x + kx * width
                length = math.hypot(dx, dy)
                if direction == "horizontal":
                    deviation = _wrap_degrees(math.degrees(math.atan2(dy, dx)))
                    plausible = (
                        min_step_x - 1 <= length <= max_step_x + 1
                        and camera_angle_min - 0.5 <= deviation <= camera_angle_max + 0.5
                    )
                else:
                    deviation = _wrap_degrees(math.degrees(math.atan2(dy, dx)) - 90.0)
                    plausible = (
                        min_step_y - 1 <= length <= max_step_y + 1
                        and camera_angle_min - 0.5 <= deviation <= camera_angle_max + 0.5
                    )
                key = (round(dx), round(dy))
                if not plausible or key in seen:
                    continue
                seen.add(key)
                ncc = overlap_ncc(a, b, dx, dy)
                prior_penalty = 0.0
                if prior is not None:
                    scale = max(width, height)
                    prior_penalty = float(np.linalg.norm(np.array([dx, dy]) - prior) / scale)
                phase_value = float(corr[py, px])
                score = ncc + 0.06 * math.log1p(max(0.0, phase_value)) - 0.35 * prior_penalty
                candidates.append(ShiftCandidate(dx, dy, phase_value, ncc, score))
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:top_k]


def overlap_ncc(a: np.ndarray, b: np.ndarray, dx: float, dy: float) -> float:
    x = int(round(dx))
    y = int(round(dy))
    height, width = a.shape
    gx0, gy0 = max(0, x), max(0, y)
    gx1, gy1 = min(width, x + width), min(height, y + height)
    if gx1 - gx0 < 8 or gy1 - gy0 < 8:
        return -1.0
    aa = a[gy0:gy1, gx0:gx1]
    bb = b[gy0 - y : gy1 - y, gx0 - x : gx1 - x]
    aa = aa.astype(np.float64, copy=False)
    bb = bb.astype(np.float64, copy=False)
    aa -= aa.mean()
    bb -= bb.mean()
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float((aa * bb).sum() / denom) if denom > 1e-10 else -1.0


def register_pair(
    reference: np.ndarray,
    moving: np.ndarray,
    direction: Direction,
    overlap_range: tuple[float, float] = (0.0, 30.0),
    rotation_range: tuple[float, float] = (-8.0, 8.0),
    prior: np.ndarray | None = None,
    mark_filter: MarkFilterSettings | None = None,
    filtered_pair: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[ShiftCandidate | None, bool, float]:
    original = phase_correlation_candidates(
        reference, moving, direction, overlap_range, rotation_range, top_k=8, prior=prior
    )
    candidate_groups: list[tuple[ShiftCandidate, bool]] = [(item, False) for item in original]
    if mark_filter and mark_filter.enabled:
        if filtered_pair is None:
            filtered_a, report_a = suppress_tool_marks(reference, mark_filter)
            filtered_b, report_b = suppress_tool_marks(moving, mark_filter)
            usable_filtered = report_a.applied and report_b.applied
        else:
            filtered_a, filtered_b = filtered_pair
            usable_filtered = True
        if usable_filtered:
            filtered = phase_correlation_candidates(
                filtered_a, filtered_b, direction, overlap_range, rotation_range, top_k=8, prior=prior
            )
            for item in filtered:
                original_ncc = overlap_ncc(_prepare_channel(reference), _prepare_channel(moving), item.dx, item.dy)
                item.ncc = 0.55 * original_ncc + 0.45 * item.ncc
                item.score = item.ncc + 0.04 * math.log1p(max(0.0, item.phase_value))
                if prior is not None:
                    item.score -= 0.35 * float(
                        np.linalg.norm(np.array([item.dx, item.dy]) - prior) / max(reference.shape[:2])
                    )
                candidate_groups.append((item, True))
    if not candidate_groups:
        return None, False, 0.0

    # Reward agreement between raw and filtered registration candidates.
    for candidate, is_filtered in candidate_groups:
        for other, other_filtered in candidate_groups:
            if is_filtered == other_filtered:
                continue
            distance = math.hypot(candidate.dx - other.dx, candidate.dy - other.dy)
            if distance <= 2.5:
                candidate.score += 0.10
                break
    candidate_groups.sort(key=lambda pair: pair[0].score, reverse=True)
    best, used_filtered = candidate_groups[0]
    alternatives = [
        candidate
        for candidate, _filtered in candidate_groups[1:]
        if math.hypot(candidate.dx - best.dx, candidate.dy - best.dy) > 2.5
    ]
    runner_up_score = alternatives[0].score if alternatives else best.score - 0.25
    return best, used_filtered, best.score - runner_up_score


class AutoStitcher:
    def __init__(
        self,
        tile_set: TileSet,
        overlap_range: tuple[float, float] = (0.0, 30.0),
        rotation_range: tuple[float, float] = (-8.0, 8.0),
        max_registration_size: int = 1024,
        mark_filter: MarkFilterSettings | None = None,
        cache_tiles: int = 64,
        sample_pairs_per_direction: int = 24,
    ) -> None:
        self.tile_set = tile_set
        self.overlap_range = overlap_range
        self.rotation_range = rotation_range
        self.max_registration_size = max_registration_size
        self.mark_filter = mark_filter or MarkFilterSettings(enabled=False)
        self.cache_tiles = max(8, cache_tiles)
        self.sample_pairs_per_direction = sample_pairs_per_direction
        self._cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self._filtered_cache: OrderedDict[Path, tuple[np.ndarray, bool]] = OrderedDict()

    def estimate_geometry(
        self,
        sample_pairs_per_direction: int | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> tuple[GridGeometry, list[PairwiseShift]]:
        horizontal, vertical = _neighbor_pairs(self.tile_set)
        sample_count = sample_pairs_per_direction or self.sample_pairs_per_direction
        selected_h = _even_sample(horizontal, sample_count)
        selected_v = _even_sample(vertical, sample_count)
        jobs = [(pair, "horizontal") for pair in selected_h] + [
            (pair, "vertical") for pair in selected_v
        ]
        edges: list[PairwiseShift] = []
        for index, ((source, target), direction) in enumerate(jobs, 1):
            edge = self._register_records(source, target, direction, prior=None)
            if edge is not None:
                edges.append(edge)
            if progress:
                progress(index, len(jobs), "估算重叠和角度")

        horizontal_vectors = np.array(
            [[edge.dx, edge.dy] for edge in edges if edge.target[1] > edge.source[1] and edge.valid]
        )
        vertical_vectors = np.array(
            [[edge.dx, edge.dy] for edge in edges if edge.target[0] > edge.source[0] and edge.valid]
        )
        if horizontal_vectors.size == 0 or vertical_vectors.size == 0:
            raise RuntimeError("Not enough reliable horizontal and vertical pairs")
        horizontal_vectors = _robust_vectors(horizontal_vectors)
        vertical_vectors = _robust_vectors(vertical_vectors)
        geometry = fit_grid_geometry(
            horizontal_vectors,
            vertical_vectors,
            self.tile_set.tile_width or 1,
            self.tile_set.tile_height or 1,
        )
        return geometry, edges

    def refine_all(
        self,
        geometry: GridGeometry,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> AutoRegistrationResult:
        initial = regular_positions(self.tile_set, geometry)
        horizontal, vertical = _neighbor_pairs(self.tile_set)
        # The fitted model has only three global parameters.  Re-registering
        # every edge in a grid of hundreds of tiles adds substantial FFT work
        # without adding independent model parameters, so use the same evenly
        # distributed sampling budget for this verification pass.
        selected_h = _even_sample(horizontal, self.sample_pairs_per_direction)
        selected_v = _even_sample(vertical, self.sample_pairs_per_direction)
        jobs = [(pair, "horizontal") for pair in selected_h] + [
            (pair, "vertical") for pair in selected_v
        ]
        edges: list[PairwiseShift] = []
        for index, ((source, target), direction) in enumerate(jobs, 1):
            prior_full = initial[(target.row, target.col)] - initial[(source.row, source.col)]
            scale = self._scale(source.path)
            prior_camera = _rotate_vector(prior_full, -geometry.tile_rotation_deg)
            edge = self._register_records(source, target, direction, prior=prior_camera * scale)
            if edge is not None:
                edge.dx, edge.dy = _rotate_vector(
                    np.array([edge.dx, edge.dy]), geometry.tile_rotation_deg
                )
                edges.append(edge)
            if progress:
                progress(index, len(jobs), "检查相邻图")

        tile_width = self.tile_set.tile_width or 1
        tile_height = self.tile_set.tile_height or 1
        _mark_outliers(edges, geometry, tile_width, tile_height)

        # Use the reliable sampled neighbors to refine the three global model
        # parameters. Individual tile centers remain on a strictly
        # horizontal/vertical orthogonal machine grid.
        horizontal_vectors = np.array(
            [[edge.dx, edge.dy] for edge in edges if edge.valid and edge.target[1] > edge.source[1]]
        )
        vertical_vectors = np.array(
            [[edge.dx, edge.dy] for edge in edges if edge.valid and edge.target[0] > edge.source[0]]
        )
        refined_geometry = geometry
        if horizontal_vectors.size and vertical_vectors.size:
            correction = fit_grid_geometry(
                _robust_vectors(horizontal_vectors),
                _robust_vectors(vertical_vectors),
                tile_width,
                tile_height,
            )
            correction_angle = correction.tile_rotation_deg
            refined_geometry = GridGeometry(
                correction.overlap_x_percent,
                correction.overlap_y_percent,
                float(np.clip(
                    geometry.tile_rotation_deg + correction_angle,
                    self.rotation_range[0],
                    self.rotation_range[1],
                )),
            )
            for edge in edges:
                edge.dx, edge.dy = _rotate_vector(
                    np.array([edge.dx, edge.dy]), correction_angle
                )

        _mark_outliers(edges, refined_geometry, tile_width, tile_height)
        _apply_loop_consistency(edges)
        positions = regular_positions(self.tile_set, refined_geometry)
        low = [edge for edge in edges if not edge.valid or edge.confidence < 0.18]
        messages = []
        if low:
            messages.append(f"{len(low)} 条低可信抽样接缝已由规则网格约束稳定")
        return AutoRegistrationResult(refined_geometry, positions, edges, low, messages)

    def run(
        self,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> AutoRegistrationResult:
        estimate_progress = None
        refine_progress = None
        if progress:
            estimate_progress = lambda current, total, message: progress(
                round(40 * current / max(1, total)), 100, message
            )
            refine_progress = lambda current, total, message: progress(
                40 + round(60 * current / max(1, total)), 100, message
            )
        geometry, sampled = self.estimate_geometry(progress=estimate_progress)
        result = self.refine_all(geometry, progress=refine_progress)
        result.messages.insert(0, f"已用 {len(sampled)} 对抽样相邻图估算全局几何")
        return result

    def _register_records(
        self,
        source: TileRecord,
        target: TileRecord,
        direction: Direction,
        prior: np.ndarray | None,
    ) -> PairwiseShift | None:
        a = self._load(source.path)
        b = self._load(target.path)
        filtered_pair = None
        pair_filter = self.mark_filter
        if self.mark_filter.enabled:
            filtered_a, applied_a = self._load_filtered(source.path)
            filtered_b, applied_b = self._load_filtered(target.path)
            if applied_a and applied_b:
                filtered_pair = (filtered_a, filtered_b)
            else:
                pair_filter = MarkFilterSettings(enabled=False)
        best, used_filtered, uniqueness = register_pair(
            a,
            b,
            direction,
            self.overlap_range,
            self.rotation_range,
            prior=prior,
            mark_filter=pair_filter,
            filtered_pair=filtered_pair,
        )
        if best is None:
            return None
        scale = self._scale(source.path)
        confidence = float(np.clip(0.55 * (best.ncc + 1.0) / 2.0 + 0.45 * max(0.0, uniqueness), 0.0, 1.0))
        return PairwiseShift(
            (source.row, source.col),
            (target.row, target.col),
            best.dx / scale,
            best.dy / scale,
            confidence,
            best.ncc,
            used_filtered,
            best.ncc > -0.2,
        )

    def _load(self, path: Path) -> np.ndarray:
        if path in self._cache:
            image = self._cache.pop(path)
            self._cache[path] = image
            return image
        image = read_rgb(path, self.max_registration_size)
        self._cache[path] = image
        while len(self._cache) > self.cache_tiles:
            self._cache.popitem(last=False)
        return image

    def _load_filtered(self, path: Path) -> tuple[np.ndarray, bool]:
        if path in self._filtered_cache:
            item = self._filtered_cache.pop(path)
            self._filtered_cache[path] = item
            return item
        image, report = suppress_tool_marks(self._load(path), self.mark_filter)
        item = (image, report.applied)
        self._filtered_cache[path] = item
        while len(self._filtered_cache) > self.cache_tiles:
            self._filtered_cache.popitem(last=False)
        return item

    def _scale(self, path: Path) -> float:
        image = self._load(path)
        return image.shape[1] / float(self.tile_set.tile_width or image.shape[1])


def solve_global_positions(
    tile_set: TileSet,
    edges: list[PairwiseShift],
    initial: dict[tuple[int, int], np.ndarray],
    regularization: float = 0.08,
    iterations: int = 4,
) -> dict[tuple[int, int], np.ndarray]:
    keys = sorted(tile_set.by_rc)
    index = {key: i for i, key in enumerate(keys)}
    if not keys:
        return {}
    current = np.vstack([initial[key] for key in keys]).astype(np.float64)
    valid_edges = [edge for edge in edges if edge.source in index and edge.target in index and edge.valid]
    if not valid_edges:
        return initial

    for _ in range(iterations):
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        bx: list[float] = []
        by: list[float] = []
        equation = 0
        residuals = []
        for edge in valid_edges:
            i, j = index[edge.source], index[edge.target]
            measured = np.array([edge.dx, edge.dy])
            residuals.append(float(np.linalg.norm((current[j] - current[i]) - measured)))
        scale = max(float(np.median(residuals)) * 2.5, 1.0)
        for edge, residual in zip(valid_edges, residuals):
            i, j = index[edge.source], index[edge.target]
            huber = 1.0 if residual <= scale else scale / residual
            weight = math.sqrt(max(0.02, edge.confidence) * huber)
            rows.extend([equation, equation])
            cols.extend([i, j])
            data.extend([-weight, weight])
            bx.append(weight * edge.dx)
            by.append(weight * edge.dy)
            equation += 1
        reg_weight = math.sqrt(max(0.0, regularization))
        for key, i in index.items():
            rows.append(equation)
            cols.append(i)
            data.append(reg_weight)
            bx.append(reg_weight * float(initial[key][0]))
            by.append(reg_weight * float(initial[key][1]))
            equation += 1
        # Anchor one tile to remove translational null space.
        rows.append(equation)
        cols.append(0)
        data.append(10.0)
        bx.append(10.0 * float(initial[keys[0]][0]))
        by.append(10.0 * float(initial[keys[0]][1]))
        matrix = sparse.coo_matrix((data, (rows, cols)), shape=(equation + 1, len(keys))).tocsr()
        current[:, 0] = sparse_linalg.lsqr(matrix, np.asarray(bx), atol=1e-6, btol=1e-6)[0]
        current[:, 1] = sparse_linalg.lsqr(matrix, np.asarray(by), atol=1e-6, btol=1e-6)[0]
    # Keep regular positions for missing or explicitly excluded tiles.  They
    # are not rendered, but their blank cells still contribute to the mosaic
    # bounds as promised by the numeric import rule.
    result = {key: np.asarray(value, dtype=np.float64).copy() for key, value in initial.items()}
    result.update({key: current[i] for key, i in index.items()})
    return result


def _prepare_channel(image: np.ndarray) -> np.ndarray:
    channel = registration_channel(image)
    channel -= channel.mean()
    # Do not apodize the full tile here.  Adjacent microscope frames share
    # content at opposite borders; a Hann window would attenuate precisely
    # that useful overlap and can make a periodic interior peak win instead.
    return channel.astype(np.float32, copy=False)


def _neighbor_pairs(tile_set: TileSet) -> tuple[list[tuple[TileRecord, TileRecord]], list[tuple[TileRecord, TileRecord]]]:
    by_rc = tile_set.by_rc
    horizontal: list[tuple[TileRecord, TileRecord]] = []
    vertical: list[tuple[TileRecord, TileRecord]] = []
    for (row, col), tile in by_rc.items():
        right = by_rc.get((row, col + 1))
        below = by_rc.get((row + 1, col))
        if right:
            horizontal.append((tile, right))
        if below:
            vertical.append((tile, below))
    return horizontal, vertical


def _even_sample(items: list, maximum: int) -> list:
    if len(items) <= maximum:
        return items
    indices = np.linspace(0, len(items) - 1, maximum, dtype=int)
    return [items[index] for index in indices]


def _robust_vectors(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) < 3:
        return vectors
    median = np.median(vectors, axis=0)
    distance = np.linalg.norm(vectors - median, axis=1)
    mad = np.median(np.abs(distance - np.median(distance))) + 1e-6
    keep = distance <= np.median(distance) + 3.5 * mad
    return vectors[keep] if np.any(keep) else vectors


def _mark_outliers(edges: list[PairwiseShift], geometry: GridGeometry, width: int, height: int) -> None:
    expected_h = np.array([width * (1 - geometry.overlap_x_percent / 100), 0.0])
    expected_v = np.array([0.0, height * (1 - geometry.overlap_y_percent / 100)])
    for edge in edges:
        expected = expected_h if edge.target[1] > edge.source[1] else expected_v
        error = float(np.linalg.norm(np.array([edge.dx, edge.dy]) - expected))
        limit = max(width, height) * 0.06
        if error > limit or edge.ncc < -0.1:
            edge.valid = False
            edge.confidence *= 0.15


def _apply_loop_consistency(edges: list[PairwiseShift]) -> None:
    lookup = {(edge.source, edge.target): edge for edge in edges if edge.valid}
    cells = {source for source, _ in lookup}
    for row, col in cells:
        top = lookup.get(((row, col), (row, col + 1)))
        left = lookup.get(((row, col), (row + 1, col)))
        right = lookup.get(((row, col + 1), (row + 1, col + 1)))
        bottom = lookup.get(((row + 1, col), (row + 1, col + 1)))
        if not all((top, left, right, bottom)):
            continue
        closure = np.array([top.dx, top.dy]) + np.array([right.dx, right.dy]) - np.array([bottom.dx, bottom.dy]) - np.array([left.dx, left.dy])
        residual = float(np.linalg.norm(closure))
        if residual > 8.0:
            weakest = min((top, left, right, bottom), key=lambda edge: edge.confidence)
            weakest.confidence *= 0.25


def _rotate_vector(vector: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return np.array(
        [cos_a * vector[0] - sin_a * vector[1], sin_a * vector[0] + cos_a * vector[1]],
        dtype=np.float64,
    )


def _wrap_degrees(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0
