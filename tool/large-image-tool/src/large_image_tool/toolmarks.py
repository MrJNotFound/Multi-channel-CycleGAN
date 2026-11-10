from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from scipy import ndimage

from .file_index import SUPPORTED_EXTENSIONS
from .image_ops import as_uint8, merge_filtered_region, read_rgb, suppress_tool_marks, to_gray
from .models import MarkFilterSettings, ToolMarkMethod
from .mosaic import write_pyramidal_ome
from .raster import open_raster
from .stripe_analysis import MIN_ORIENTATION_CONFIDENCE, Region, detect_orientation, estimate_period, scaled_normal


ProgressCallback = Callable[[int, int, str], None]
MarkAnalysis = tuple[float, float, float, float, Region | None]


@dataclass(slots=True)
class ToolMarkReport:
    method: str
    orientation_deg: float
    orientation_confidence: float
    period_px: float
    period_confidence: float
    applied: bool
    processing_region: Region | None = None


@dataclass(slots=True)
class ToolMarkOutput:
    output_path: Path
    preview_path: Path | None
    settings_path: Path
    processed_count: int


def analyze_tool_marks(
    image: np.ndarray,
    settings: MarkFilterSettings,
) -> tuple[float, float, float, float]:
    return _analyze_tool_marks(image, settings)[:4]


def _analyze_tool_marks(image: np.ndarray, settings: MarkFilterSettings) -> MarkAnalysis:
    settings.validate()
    gray = to_gray(image)
    region = None
    if settings.auto_orientation:
        evidence = detect_orientation(gray)
        orientation, orientation_confidence = evidence.angle_deg, evidence.confidence
        region = evidence.region
    else:
        orientation, orientation_confidence = settings.orientation_deg, 1.0
    orientation = (float(orientation) + 90.0) % 180.0 - 90.0
    if settings.auto_period:
        if region is not None:
            x0, y0, x1, y1 = region
            gray = gray[y0:y1, x0:x1]
        period, period_confidence = estimate_period(gray, orientation)
    else:
        period, period_confidence = settings.period_px, 1.0
    return orientation, orientation_confidence, period, period_confidence, region


def remove_tool_marks(
    image: np.ndarray,
    settings: MarkFilterSettings,
) -> tuple[np.ndarray, ToolMarkReport]:
    settings.validate()
    source = as_uint8(image)
    orientation, orientation_confidence, period, period_confidence, region = _analyze_tool_marks(
        source, settings
    )
    if (not settings.enabled or settings.strength <= 0
            or (settings.auto_orientation and orientation_confidence < MIN_ORIENTATION_CONFIDENCE)):
        return source.copy(), ToolMarkReport(
            settings.method, orientation, orientation_confidence, period, period_confidence, False
        )

    if region is not None:
        x0, y0, x1, y1 = region
        local = replace(settings, auto_orientation=False, orientation_deg=orientation)
        filtered, report = remove_tool_marks(source[y0:y1, x0:x1], local)
        report.orientation_confidence = orientation_confidence
        report.processing_region = region
        return merge_filtered_region(source, filtered, region), report

    method = ToolMarkMethod(settings.method)
    if method is ToolMarkMethod.FFT_GAUSSIAN_NOTCH:
        fft_settings = MarkFilterSettings(**asdict(settings))
        fft_settings.auto_orientation = False
        fft_settings.orientation_deg = orientation
        fft_settings.auto_period = settings.auto_period
        fft_settings.period_px = period
        filtered, fft_report = suppress_tool_marks(source, fft_settings)
        return filtered, ToolMarkReport(
            settings.method,
            orientation,
            orientation_confidence,
            period,
            period_confidence,
            fft_report.applied,
        )

    # Rotate only the analysis/correction, never the image being preserved.
    # Inverse sampling uses pixel centers explicitly, avoiding half-pixel shifts
    # from rotating an expanded canvas back and cropping it by integer offsets.
    gray = to_gray(source)
    aligned = ndimage.rotate(gray, orientation, reshape=True, order=1, mode="reflect")
    valid = ndimage.rotate(
        np.ones(gray.shape, dtype=np.uint8), orientation,
        reshape=True, order=0, mode="constant", cval=0,
    ).astype(bool)
    if method is ToolMarkMethod.PERIODIC_TEMPLATE:
        correction = _stripe_template(aligned, valid, period, settings.template_smoothing)
    else:
        kernel = max(3, int(settings.directional_kernel)) | 1
        residual = aligned - ndimage.median_filter(aligned, size=(1, kernel), mode="reflect")
        # Retain residuals coherent along a stripe, suppressing isolated tissue
        # detail that a direct normal-axis median would otherwise erase.
        correction = ndimage.median_filter(residual, size=(max(31, 2 * kernel + 1), 1), mode="reflect")
    correction = _restore_correction(correction, gray.shape, orientation)
    if source.ndim == 3:
        correction = correction[..., None]
    restored = source.astype(np.float32) - settings.strength * correction
    filtered = np.clip(np.rint(restored), 0, 255).astype(np.uint8)
    return filtered, ToolMarkReport(
        settings.method, orientation, orientation_confidence, period, period_confidence,
        bool(np.any(filtered != source)),
    )


def preview_tool_marks(
    image: np.ndarray, settings: MarkFilterSettings, scale: tuple[float, float],
) -> tuple[np.ndarray, ToolMarkReport]:
    """Apply original-pixel settings to a thumbnail and report original units."""
    angle, normal_scale = scaled_normal(settings.orientation_deg, scale)
    local = replace(
        settings,
        orientation_deg=(angle + 90) % 180 - 90,
        period_px=float(np.clip(settings.period_px * normal_scale, 3, 512)),
        frequency_width=settings.frequency_width / normal_scale,
        min_frequency=settings.min_frequency / normal_scale,
        max_frequency=min(0.49, settings.max_frequency / normal_scale),
        template_smoothing=settings.template_smoothing * normal_scale,
        directional_kernel=max(3, round(settings.directional_kernel * normal_scale)) | 1,
    )
    filtered, report = remove_tool_marks(image, local)
    radians = math.radians(report.orientation_deg)
    report.orientation_deg = math.degrees(math.atan2(math.sin(radians) * scale[0], math.cos(radians) * scale[1]))
    report.orientation_deg = (report.orientation_deg + 90) % 180 - 90
    _, normal_scale = scaled_normal(report.orientation_deg, scale)
    report.period_px /= normal_scale
    if report.processing_region is not None:
        x0, y0, x1, y1 = report.processing_region
        report.processing_region = (
            round(x0 / scale[1]), round(y0 / scale[0]),
            round(x1 / scale[1]), round(y1 / scale[0]),
        )
    return filtered, report


def process_toolmark_input(
    input_path: str | Path,
    output_path: str | Path,
    settings: MarkFilterSettings,
    progress: ProgressCallback | None = None,
) -> ToolMarkOutput:
    source = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    settings.validate()
    if source.is_dir():
        return _process_folder(source, output, settings, progress)
    if not source.is_file():
        raise FileNotFoundError(source)
    return _process_raster(source, output, settings, progress)


def suggested_toolmark_output(input_path: str | Path) -> Path:
    source = Path(input_path)
    if source.is_dir():
        return source.with_name(source.name + "_marks_removed")
    name = source.name
    lower = name.lower()
    if lower.endswith(".ome.tiff"):
        stem = name[:-9]
    elif lower.endswith(".ome.tif"):
        stem = name[:-8]
    else:
        stem = source.stem
    return source.with_name(stem + "_marks_removed.ome.tif")


def save_toolmark_settings(path: str | Path, settings: MarkFilterSettings) -> Path:
    target = Path(path).expanduser().resolve()
    payload = {
        "schema_version": 2,
        "kind": "tool_mark_removal",
        "settings": asdict(settings),
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_toolmark_settings(path: str | Path) -> MarkFilterSettings:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return MarkFilterSettings(**payload.get("settings", payload))


def linked_source_folder(folder: str | Path) -> Path | None:
    """Return the pre-filter source recorded for a processed tile folder.

    This lets stitching use unfiltered pixels for registration while rendering
    the filtered tiles.  Both an absolute and a portable relative reference
    are accepted; a broken link simply falls back to the selected folder.
    """
    root = Path(folder).expanduser().resolve()
    metadata_path = root / "toolmark_settings.json"
    if not metadata_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if payload.get("kind") != "tool_mark_removal":
        return None
    candidates = []
    if payload.get("source"):
        candidates.append(Path(payload["source"]).expanduser())
    if payload.get("source_relative_to_output"):
        candidates.append(root / payload["source_relative_to_output"])
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_dir() and resolved != root:
            return resolved
    return None


def _stripe_template(
    aligned: np.ndarray,
    valid: np.ndarray,
    period_px: float,
    smoothing: float,
) -> np.ndarray:
    """Robust local profiles; no integer-period folding or reflected samples.

    A phase-folded template cancels itself for fractional/drifting periods and
    cannot represent irregular scratches. Tangential bands allow a stripe's
    position and contrast to change slowly along its length.
    """
    residual = aligned - ndimage.gaussian_filter1d(aligned, max(2.0, period_px * 0.4), axis=1)
    height, width = aligned.shape
    band = min(height, max(96, round(period_px * 4)))
    correction = np.zeros_like(aligned)
    coverage = np.zeros_like(aligned)
    for start in range(0, height, max(1, band // 2)):
        stop = min(height, start + band)
        mask = valid[start:stop]
        supported = mask.sum(axis=0) >= min(8, stop - start)
        if not np.any(supported):
            continue
        values = np.where(mask[:, supported], residual[start:stop, supported], np.nan)
        profile = np.zeros(width, dtype=np.float32)
        profile[supported] = np.nanmedian(values, axis=0)
        # Fill only for smoothing; unsupported columns retain zero blend weight.
        profile = np.interp(np.arange(width), np.flatnonzero(supported), profile[supported])
        if smoothing > 0:
            profile = ndimage.gaussian_filter1d(profile, smoothing)
        weights = np.hanning(stop - start + 2)[1:-1, None] * supported[None, :]
        correction[start:stop] += (weights * profile).astype(np.float32)
        coverage[start:stop] += weights.astype(np.float32)
    return correction / np.maximum(coverage, 1e-8)


def _restore_correction(correction: np.ndarray, shape: tuple[int, int], orientation: float) -> np.ndarray:
    height, width = shape
    angle = math.radians(orientation)
    cosine, sine = math.cos(angle), math.sin(angle)
    # Coordinates are (row, column), rotating the original centers into the
    # expanded analysis canvas. Only the correction undergoes interpolation.
    matrix = np.array([[cosine, -sine], [sine, cosine]])
    offset = (np.asarray(correction.shape) - 1) / 2 - matrix @ ((np.asarray(shape) - 1) / 2)
    return ndimage.affine_transform(correction, matrix, offset=offset, output_shape=shape, order=1, mode="nearest")


def _process_folder(
    source: Path,
    output: Path,
    settings: MarkFilterSettings,
    progress: ProgressCallback | None,
) -> ToolMarkOutput:
    if source.resolve() == output.resolve():
        raise ValueError("批量输出文件夹不能与输入文件夹相同")
    output.mkdir(parents=True, exist_ok=True)
    files = [
        path for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    files.sort(key=lambda path: (0, int(path.stem)) if path.stem.isdigit() else (1, path.name.lower()))
    if not files:
        raise ValueError("文件夹中没有支持的图像")
    reports: dict[str, dict] = {}
    for index, path in enumerate(files, 1):
        filtered, report = remove_tool_marks(read_rgb(path), settings)
        destination = output / path.name
        _save_rgb_atomic(destination, filtered)
        reports[path.name] = asdict(report)
        if progress:
            progress(index, len(files), "批量去除刀纹")
    settings_path = output / "toolmark_settings.json"
    _write_run_metadata(settings_path, source, output, settings, reports)
    return ToolMarkOutput(output, None, settings_path, len(files))


def _process_raster(
    source: Path,
    output: Path,
    settings: MarkFilterSettings,
    progress: ProgressCallback | None,
) -> ToolMarkOutput:
    if not output.name.lower().endswith((".ome.tif", ".ome.tiff")):
        output = output.with_name(output.name + ".ome.tif")
    output.parent.mkdir(parents=True, exist_ok=True)
    working_path = output.with_suffix(output.suffix + ".working.uint8")
    weights_path = output.with_suffix(output.suffix + ".working.weights.float16")
    reports: dict[str, dict] = {}
    result: np.memmap | None = None
    blend_weights: np.memmap | None = None
    try:
        with open_raster(source) as raster:
            result = np.memmap(
                working_path, dtype=np.uint8, mode="w+", shape=(raster.height, raster.width, 3)
            )
            result[:] = 0
            blend_weights = np.memmap(
                weights_path, dtype=np.float16, mode="w+", shape=(raster.height, raster.width)
            )
            blend_weights[:] = 0
            block = settings.block_size
            halo = settings.block_halo
            blocks_x = math.ceil(raster.width / block)
            blocks_y = math.ceil(raster.height / block)
            block_count = blocks_x * blocks_y
            needs_analysis = settings.auto_orientation or settings.auto_period
            total = block_count * (2 if needs_analysis else 1)
            done = 0
            local_analysis: dict[tuple[int, int], MarkAnalysis] = {}
            overview = None
            if settings.auto_orientation:
                overview_factor = min(1.0, 1024 / max(raster.width, raster.height))
                overview = np.zeros((
                    max(1, round(raster.height * overview_factor)),
                    max(1, round(raster.width * overview_factor)), 3,
                ), dtype=np.uint8)
                overview_sy = overview.shape[0] / raster.height
                overview_sx = overview.shape[1] / raster.width
            if needs_analysis:
                for block_y in range(blocks_y):
                    y0, y1 = block_y * block, min(raster.height, (block_y + 1) * block)
                    for block_x in range(blocks_x):
                        x0, x1 = block_x * block, min(raster.width, (block_x + 1) * block)
                        rx0, ry0 = max(0, x0 - halo), max(0, y0 - halo)
                        rx1, ry1 = min(raster.width, x1 + halo), min(raster.height, y1 + halo)
                        region = raster.read_region(rx0, ry0, rx1 - rx0, ry1 - ry0)
                        if overview is not None:
                            # Reuse this pass to make a bounded whole-image view.
                            # Otherwise separate export blocks can lose the
                            # preview's regional scope and filter clean tissue.
                            ox0, ox1 = round(x0 * overview_sx), round(x1 * overview_sx)
                            oy0, oy1 = round(y0 * overview_sy), round(y1 * overview_sy)
                            if ox1 > ox0 and oy1 > oy0:
                                core = region[y0-ry0:y1-ry0, x0-rx0:x1-rx0]
                                overview[oy0:oy1, ox0:ox1] = np.asarray(Image.fromarray(core).resize(
                                    (ox1-ox0, oy1-oy0), Image.Resampling.BILINEAR,
                                ))
                        local_analysis[(block_y, block_x)] = _analyze_tool_marks(region, settings)
                        done += 1
                        if progress:
                            progress(done, total, "分析局部刀纹")
            source_evidence = detect_orientation(to_gray(overview)) if overview is not None else None
            if source_evidence is not None and source_evidence.region is not None:
                ax0, ay0, ax1, ay1 = source_evidence.region
                angle = math.radians(source_evidence.angle_deg)
                angle_deg = math.degrees(math.atan2(math.sin(angle) * overview_sy, math.cos(angle) * overview_sx))
                source_evidence = replace(source_evidence,
                    angle_deg=(angle_deg + 90) % 180 - 90,
                    region=(round(ax0/overview_sx), round(ay0/overview_sy),
                            round(ax1/overview_sx), round(ay1/overview_sy)),
                )
            for block_y in range(blocks_y):
                y0, y1 = block_y * block, min(raster.height, (block_y + 1) * block)
                for block_x in range(blocks_x):
                    x0, x1 = block_x * block, min(raster.width, (block_x + 1) * block)
                    rx0, ry0 = max(0, x0 - halo), max(0, y0 - halo)
                    rx1, ry1 = min(raster.width, x1 + halo), min(raster.height, y1 + halo)
                    region = raster.read_region(rx0, ry0, rx1 - rx0, ry1 - ry0)
                    if source_evidence is not None and source_evidence.region is not None:
                        ax0, ay0, ax1, ay1 = source_evidence.region
                        local_region = (max(rx0, ax0)-rx0, max(ry0, ay0)-ry0,
                                        min(rx1, ax1)-rx0, min(ry1, ay1)-ry0)
                        lx0, ly0, lx1, ly1 = local_region
                        if lx1 > lx0 and ly1 > ly0:
                            local_settings = replace(settings, auto_orientation=False,
                                                     orientation_deg=source_evidence.angle_deg)
                            correction, report = remove_tool_marks(region[ly0:ly1, lx0:lx1], local_settings)
                            filtered = merge_filtered_region(region, correction, local_region)
                            report.processing_region = local_region
                            report.orientation_confidence = source_evidence.confidence
                        else:
                            filtered = region
                            report = ToolMarkReport(settings.method, source_evidence.angle_deg,
                                source_evidence.confidence, settings.period_px, 0.0, False)
                    else:
                        local_settings = _local_smoothed_settings(settings, local_analysis, block_y, block_x)
                        filtered, report = remove_tool_marks(region, local_settings)
                    if report.processing_region is not None:
                        ax0, ay0, ax1, ay1 = report.processing_region
                        report.processing_region = (rx0 + ax0, ry0 + ay0, rx0 + ax1, ry0 + ay1)
                    weight = _block_blend_weight(
                        rx0, ry0, rx1, ry1, x0, y0, x1, y1,
                        raster.width, raster.height,
                    )
                    old_weight = np.asarray(blend_weights[ry0:ry1, rx0:rx1], dtype=np.float32)
                    total_weight = old_weight + weight
                    previous = np.asarray(result[ry0:ry1, rx0:rx1], dtype=np.float32)
                    combined = (
                        previous * old_weight[..., None]
                        + filtered.astype(np.float32) * weight[..., None]
                    ) / np.maximum(total_weight[..., None], 1e-6)
                    result[ry0:ry1, rx0:rx1] = np.clip(np.rint(combined), 0, 255).astype(np.uint8)
                    blend_weights[ry0:ry1, rx0:rx1] = total_weight.astype(np.float16)
                    reports[f"block_{block_y:04d}_{block_x:04d}"] = asdict(report)
                    done += 1
                    if progress:
                        progress(done, total, "分块去除刀纹")
            result.flush()
            blend_weights.flush()
            write_pyramidal_ome(result, output)
            preview_path = output.with_suffix(".preview.png")
            factor = max(1, math.ceil(max(raster.width, raster.height) / 2200))
            Image.fromarray(np.asarray(result[::factor, ::factor]).copy(), mode="RGB").save(
                preview_path, compress_level=4
            )
    finally:
        if result is not None:
            del result
        if blend_weights is not None:
            del blend_weights
        working_path.unlink(missing_ok=True)
        weights_path.unlink(missing_ok=True)
    settings_path = output.with_suffix(".toolmarks.json")
    _write_run_metadata(settings_path, source, output, settings, reports)
    return ToolMarkOutput(output, preview_path, settings_path, 1)


def _local_smoothed_settings(
    settings: MarkFilterSettings,
    analysis: dict[tuple[int, int], MarkAnalysis],
    block_y: int,
    block_x: int,
) -> MarkFilterSettings:
    if not analysis:
        return settings
    center = analysis[(block_y, block_x)]
    if settings.auto_orientation and center[4] is not None:
        # Keep local detection enabled: locking a regional angle here would
        # apply that corner's correction to the entire export block.
        return settings
    neighbors = [
        report
        for (row, col), report in analysis.items()
        if abs(row - block_y) <= 1 and abs(col - block_x) <= 1
    ]
    local = MarkFilterSettings(**asdict(settings))
    if settings.auto_orientation:
        reliable = [report for report in neighbors if report[1] >= MIN_ORIENTATION_CONFIDENCE and report[4] is None]
        if not reliable or center[1] < MIN_ORIENTATION_CONFIDENCE:
            return replace(settings, enabled=False)
        if center[1] >= MIN_ORIENTATION_CONFIDENCE:
            reliable = [report for report in reliable if abs((report[0] - center[0] + 90) % 180 - 90) <= 10]
        angles = np.radians([report[0] * 2.0 for report in reliable])
        weights = np.array([report[1] for report in reliable])
        vector = np.sum(weights * np.exp(1j * angles))
        local.orientation_deg = float(np.degrees(np.angle(vector)) / 2.0)
        local.auto_orientation = False
    if settings.auto_period and settings.method != ToolMarkMethod.FFT_GAUSSIAN_NOTCH.value:
        periods = [report[2] for report in neighbors if report[2] >= 3.0]
        local.period_px = float(np.median(periods)) if periods else settings.period_px
        local.auto_period = False
    return local


def _block_blend_weight(
    rx0: int,
    ry0: int,
    rx1: int,
    ry1: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    wx = np.ones(rx1 - rx0, dtype=np.float32)
    wy = np.ones(ry1 - ry0, dtype=np.float32)
    left = x0 - rx0
    right = rx1 - x1
    top = y0 - ry0
    bottom = ry1 - y1
    if left > 0 and rx0 > 0:
        wx[:left] = np.linspace(0.02, 1.0, left, endpoint=False, dtype=np.float32)
    if right > 0 and rx1 < image_width:
        wx[-right:] = np.linspace(1.0, 0.02, right, endpoint=True, dtype=np.float32)
    if top > 0 and ry0 > 0:
        wy[:top] = np.linspace(0.02, 1.0, top, endpoint=False, dtype=np.float32)
    if bottom > 0 and ry1 < image_height:
        wy[-bottom:] = np.linspace(1.0, 0.02, bottom, endpoint=True, dtype=np.float32)
    return wy[:, None] * wx[None, :]


def _write_run_metadata(
    path: Path,
    source: Path,
    output: Path,
    settings: MarkFilterSettings,
    reports: dict[str, dict],
) -> None:
    payload = {
        "schema_version": 2,
        "kind": "tool_mark_removal",
        "source": str(source),
        "source_relative_to_output": os.path.relpath(source, output if output.is_dir() else output.parent),
        "output": str(output),
        "settings": asdict(settings),
        "analysis": reports,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_rgb(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix.lower()
    kwargs = {"quality": 95} if suffix in {".jpg", ".jpeg"} else {}
    Image.fromarray(as_uint8(image), mode="RGB").save(path, **kwargs)


def _save_rgb_atomic(path: Path, image: np.ndarray) -> None:
    """Write one batch result completely before replacing an existing file."""
    temporary = path.with_name(f".{path.stem}.writing{path.suffix}")
    try:
        _save_rgb(temporary, image)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
