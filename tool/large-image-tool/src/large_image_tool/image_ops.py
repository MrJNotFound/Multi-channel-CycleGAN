from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import fft as scipy_fft
from scipy import ndimage, signal

from .models import MarkFilterSettings
from .stripe_analysis import (
    MIN_ORIENTATION_CONFIDENCE, Region, analysis_image, detect_orientation,
    estimate_orientation, scaled_normal,
)


@dataclass(slots=True)
class MarkFilterReport:
    normal_angle_deg: float
    orientation_confidence: float
    peak_frequencies: list[float]
    applied: bool
    processing_region: Region | None = None


def read_rgb(path: str | Path, max_size: int | None = None) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if max_size and max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        image = image.convert("RGB")
        return np.asarray(image).copy()


def resize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(as_uint8(image), mode="RGB")
    return np.asarray(pil.resize(size, Image.Resampling.LANCZOS)).copy()


def as_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        return np.clip(array.astype(np.float32) * (255.0 / info.max), 0, 255).astype(np.uint8)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, (0.5, 99.5))
    if hi <= lo:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.clip((array - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def to_gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        return array
    if array.shape[-1] == 1:
        return array[..., 0]
    return 0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]


def registration_channel(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    sigma = max(1.0, min(gray.shape) / 180.0)
    high_pass = gray - ndimage.gaussian_filter(gray, sigma=sigma, mode="reflect")
    gy = ndimage.sobel(high_pass, axis=0, mode="reflect")
    gx = ndimage.sobel(high_pass, axis=1, mode="reflect")
    magnitude = np.hypot(gx, gy)
    scale = np.percentile(magnitude, 99.0)
    if scale > 0:
        magnitude = np.clip(magnitude / scale, 0.0, 1.0)
    return magnitude.astype(np.float32, copy=False)


def estimate_mark_orientation(gray: np.ndarray) -> tuple[float, float]:
    """Return the coherent stripe normal in image coordinates, modulo 180 degrees."""
    return estimate_orientation(gray)


def suppress_tool_marks(
    image: np.ndarray,
    settings: MarkFilterSettings | None = None,
) -> tuple[np.ndarray, MarkFilterReport]:
    """Suppress locally parallel quasi-periodic marks with soft FFT notches.

    The notch locations are estimated from luminance and the same mask is
    applied independently to every color channel. This function is suitable
    for registration copies and for an explicitly requested filtered output.
    """
    cfg = settings or MarkFilterSettings(enabled=True)
    cfg.validate()
    source = np.asarray(image)
    if not cfg.enabled or cfg.strength <= 0:
        return source.copy(), MarkFilterReport(cfg.orientation_deg, 0.0, [], False)
    gray = to_gray(source)
    if cfg.auto_orientation:
        evidence = detect_orientation(gray)
        normal_deg, confidence = evidence.angle_deg, evidence.confidence
        if evidence.region is not None:
            x0, y0, x1, y1 = evidence.region
            local = replace(cfg, auto_orientation=False, orientation_deg=normal_deg)
            filtered, report = suppress_tool_marks(source[y0:y1, x0:x1], local)
            report.orientation_confidence = confidence
            report.processing_region = evidence.region
            return merge_filtered_region(source, filtered, evidence.region), report
    else:
        normal_deg, confidence = cfg.orientation_deg, 1.0
    if cfg.auto_orientation and confidence < MIN_ORIENTATION_CONFIDENCE:
        return source.copy(), MarkFilterReport(normal_deg, confidence, [], False)

    if cfg.auto_period:
        # Frequencies must remain in original pixels after analysis resizing.
        analysis, scale = analysis_image(gray)
        analysis_angle, normal_scale = scaled_normal(normal_deg, scale)
        analysis_cfg = replace(
            cfg,
            min_frequency=cfg.min_frequency / normal_scale,
            max_frequency=min(0.49, cfg.max_frequency / normal_scale),
        )
        peak_frequencies = [
            frequency * normal_scale
            for frequency in _detect_periodic_frequencies(
                analysis, math.radians(analysis_angle), analysis_cfg,
            )
        ]
    else:
        fundamental = 1.0 / max(cfg.period_px, 1.0)
        peak_frequencies = [
            harmonic * fundamental
            for harmonic in range(1, cfg.max_peaks + 1)
            if cfg.min_frequency <= harmonic * fundamental <= cfg.max_frequency
        ]
    if not peak_frequencies:
        return source.copy(), MarkFilterReport(normal_deg, confidence, [], False)

    original_dtype = source.dtype
    working = source[..., None] if source.ndim == 2 else source
    integer_output = np.issubdtype(original_dtype, np.integer)
    result = np.empty(working.shape, dtype=original_dtype if integer_output else np.float32)
    mask = _frequency_notch_mask(
        working.shape[:2],
        math.radians(normal_deg),
        peak_frequencies,
        cfg,
    )
    for channel in range(working.shape[-1]):
        plane = working[..., channel].astype(np.float32)
        mean = float(plane.mean())
        transformed = scipy_fft.fftshift(scipy_fft.fft2(plane - mean, workers=1))
        transformed *= mask
        restored = scipy_fft.ifft2(scipy_fft.ifftshift(transformed), workers=1).real + mean
        if integer_output:
            info = np.iinfo(original_dtype)
            result[..., channel] = np.clip(np.rint(restored), info.min, info.max).astype(original_dtype)
        else:
            result[..., channel] = restored.astype(np.float32)

    if not integer_output:
        result = result.astype(original_dtype)
    if source.ndim == 2:
        result = result[..., 0]
    return result, MarkFilterReport(normal_deg, confidence, peak_frequencies, True)


def merge_filtered_region(source: np.ndarray, filtered: np.ndarray, region: Region) -> np.ndarray:
    """Feather a local correction at internal borders; preserve all other pixels."""
    x0, y0, x1, y1 = region
    height, width = filtered.shape[:2]
    feather = min(32, height, width, max(2, min(height, width) // 8))
    wx, wy = np.ones(width, dtype=np.float32), np.ones(height, dtype=np.float32)
    ramp = np.linspace(0, 1, feather, dtype=np.float32)
    if x0 > 0:
        wx[:feather] *= ramp
    if x1 < source.shape[1]:
        wx[-feather:] *= ramp[::-1]
    if y0 > 0:
        wy[:feather] *= ramp
    if y1 < source.shape[0]:
        wy[-feather:] *= ramp[::-1]
    weights = wy[:, None] * wx[None, :]
    if source.ndim == 3:
        weights = weights[..., None]
    original = source[y0:y1, x0:x1].astype(np.float32)
    blended = original + weights * (filtered.astype(np.float32) - original)
    if np.issubdtype(source.dtype, np.integer):
        limits = np.iinfo(source.dtype)
        blended = np.clip(np.rint(blended), limits.min, limits.max)
    result = source.copy()
    result[y0:y1, x0:x1] = blended.astype(source.dtype)
    return result


def _detect_periodic_frequencies(
    gray: np.ndarray,
    normal_angle: float,
    cfg: MarkFilterSettings,
) -> list[float]:
    data = np.array(gray, dtype=np.float32, copy=True)
    data -= ndimage.gaussian_filter(data, sigma=max(2.0, min(data.shape) / 90.0))
    window = np.outer(np.hanning(data.shape[0]), np.hanning(data.shape[1])).astype(np.float32)
    # Keep peak amplitude when averaging the wedge. Log compression can let
    # numerous weak off-line samples shift a narrow stripe's actual frequency.
    spectrum = np.abs(scipy_fft.fftshift(scipy_fft.fft2(data * window, workers=1)))
    fy = np.fft.fftshift(np.fft.fftfreq(data.shape[0]))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(data.shape[1]))[None, :]
    radius = np.hypot(fx, fy)
    angle = np.mod(np.arctan2(fy, fx), math.pi)
    distance = np.abs(np.arctan2(np.sin(angle - normal_angle), np.cos(angle - normal_angle)))
    distance = np.minimum(distance, math.pi - distance)
    wedge = distance <= math.radians(max(2.0, cfg.angular_width_deg * 1.5))
    valid = wedge & (radius >= cfg.min_frequency) & (radius <= cfg.max_frequency)
    if not np.any(valid):
        return []

    # Small/oblique regions have sparse Fourier samples. A fixed 512-bin
    # profile inserted zeros between them and mistook those holes for peaks.
    resolution = math.hypot(math.cos(normal_angle) / data.shape[1], math.sin(normal_angle) / data.shape[0])
    bins = min(512, max(32, round(2 * cfg.max_frequency / resolution)))
    index = np.clip((radius[valid] / cfg.max_frequency * (bins - 1)).astype(int), 0, bins - 1)
    sums = np.bincount(index, weights=spectrum[valid], minlength=bins)
    counts = np.bincount(index, minlength=bins)
    profile = sums / np.maximum(counts, 1)
    supported = np.flatnonzero(counts)
    if len(supported) < 3:
        return []
    profile = np.interp(np.arange(bins), supported, profile[supported])
    profile = ndimage.gaussian_filter1d(profile, sigma=1.0)
    baseline = ndimage.gaussian_filter1d(profile, sigma=12.0)
    residual = profile - baseline
    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med))) + 1e-6
    peaks, properties = signal.find_peaks(
        residual,
        prominence=max(2.8 * mad, float(np.percentile(residual, 80)) * 0.25),
        distance=max(1, round(bins * 0.0025 / cfg.max_frequency)),
    )
    if peaks.size == 0:
        return []
    prominences = properties["prominences"]
    order = np.argsort(prominences)[::-1][: cfg.max_peaks]
    frequencies = peaks[order] / (bins - 1) * cfg.max_frequency
    return sorted(float(f) for f in frequencies if cfg.min_frequency <= f <= cfg.max_frequency)


def _frequency_notch_mask(
    shape: tuple[int, int],
    normal_angle: float,
    peaks: list[float],
    cfg: MarkFilterSettings,
) -> np.ndarray:
    angular_sigma = math.radians(max(1.0, cfg.angular_width_deg))
    mask = np.ones(shape, dtype=np.float32)
    fy_values = scipy_fft.fftshift(scipy_fft.fftfreq(shape[0])).astype(np.float32)
    fx = scipy_fft.fftshift(scipy_fft.fftfreq(shape[1])).astype(np.float32)[None, :]
    strength = float(np.clip(cfg.strength, 0.0, 0.98))
    frequency_sigma = max(cfg.frequency_width, 1e-5)
    for start in range(0, shape[0], 192):
        stop = min(shape[0], start + 192)
        fy = fy_values[start:stop, None]
        radius = np.hypot(fx, fy)
        angle = np.mod(np.arctan2(fy, fx), math.pi)
        angular_distance = np.abs(
            np.arctan2(np.sin(angle - normal_angle), np.cos(angle - normal_angle))
        )
        angular_distance = np.minimum(angular_distance, math.pi - angular_distance)
        angle_weight = np.exp(-0.5 * (angular_distance / angular_sigma) ** 2)
        chunk = mask[start:stop]
        for frequency in peaks:
            radial_weight = np.exp(-0.5 * ((radius - frequency) / frequency_sigma) ** 2)
            chunk *= 1.0 - strength * angle_weight * radial_weight
        chunk[radius < cfg.min_frequency] = 1.0
    return np.clip(mask, 0.02, 1.0)


def feather_weight(height: int, width: int, feather: int) -> np.ndarray:
    if feather <= 0:
        return np.ones((height, width), dtype=np.float32)
    y = np.minimum(np.arange(height) + 1, np.arange(height, 0, -1))
    x = np.minimum(np.arange(width) + 1, np.arange(width, 0, -1))
    wy = np.clip(y / feather, 0.02, 1.0)
    wx = np.clip(x / feather, 0.02, 1.0)
    return np.minimum(wy[:, None], wx[None, :]).astype(np.float32)
