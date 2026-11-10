"""Estimate stripe normals from coherent lines, rather than individual edges.

Angles use image coordinates (x right, y down) and are axial, modulo 180°.
Public periods are always measured in pixels of the input image.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image
from scipy import fft, ndimage


MIN_ORIENTATION_CONFIDENCE = 0.4
Region = tuple[int, int, int, int]  # x0, y0, x1, y1 in input pixels


@dataclass(frozen=True, slots=True)
class OrientationEvidence:
    angle_deg: float
    confidence: float
    coherence: float
    region: Region | None = None


def analysis_image(gray: np.ndarray, maximum: int = 1024) -> tuple[np.ndarray, tuple[float, float]]:
    data = np.array(gray, dtype=np.float32, copy=True)
    height, width = data.shape
    if max(height, width) > maximum:
        scale = maximum / max(height, width)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        data = np.array(Image.fromarray(data).resize(size, Image.Resampling.BILINEAR), copy=True)
    return data, (data.shape[0] / height, data.shape[1] / width)


def scaled_normal(angle_deg: float, scale: tuple[float, float]) -> tuple[float, float]:
    """Return normal and normal-axis pixel scale after resizing an image."""
    angle = math.radians(angle_deg)
    x, y = math.cos(angle) / scale[1], math.sin(angle) / scale[0]
    return math.degrees(math.atan2(y, x)), 1.0 / math.hypot(x, y)


def normal_profile(data: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Windowed line projection, with linear binning and coverage normalization.

    Unlike rotating with reflected padding, this only uses measured pixels.
    """
    height, width = data.shape
    angle = math.radians(angle_deg)
    x = np.arange(width, dtype=np.float32) - (width - 1) / 2
    y = np.arange(height, dtype=np.float32) - (height - 1) / 2
    length = int(math.ceil(math.hypot(height, width))) + 3
    coordinate = (x[None, :] * math.cos(angle) + y[:, None] * math.sin(angle) + length / 2).ravel()
    index = coordinate.astype(np.intp)
    fraction = coordinate - index
    window = np.outer(np.hanning(height), np.hanning(width)).astype(np.float32).ravel()
    values = data.ravel() * window
    sums = (
        np.bincount(index, weights=values * (1 - fraction), minlength=length)
        + np.bincount(index + 1, weights=values * fraction, minlength=length)
    )
    coverage = (
        np.bincount(index, weights=window * (1 - fraction), minlength=length)
        + np.bincount(index + 1, weights=window * fraction, minlength=length)
    )
    return sums / np.maximum(coverage, 1e-8), coverage


def _orientation_evidence(gray: np.ndarray, *, refine: bool = True) -> OrientationEvidence:
    data, scale = analysis_image(gray)
    height, width = data.shape
    if min(height, width) < 16 or float(np.ptp(data)) < 1e-6:
        return OrientationEvidence(0.0, 0.0, 0.0)
    data -= ndimage.gaussian_filter(data, sigma=6.0, mode="reflect")
    window = np.outer(np.hanning(height), np.hanning(width)).astype(np.float32)
    power = np.abs(fft.fftshift(fft.fft2(data * window, workers=1))) ** 2
    fy = fft.fftshift(fft.fftfreq(height))[:, None]
    fx = fft.fftshift(fft.fftfreq(width))[None, :]
    radius = np.hypot(fx, fy)
    radial_bin = (radius * min(height, width)).astype(np.intp)
    radial = np.bincount(radial_bin.ravel(), weights=power.ravel()) / np.maximum(
        1, np.bincount(radial_bin.ravel())
    )
    radial = ndimage.gaussian_filter1d(radial, 2.0)
    # A floor prevents numerical FFT leakage from dominating a clean sinusoid.
    normalized = power / np.maximum(radial[radial_bin], max(float(radial.max()) * 1e-3, 1e-8))
    excess = np.maximum(np.sqrt(normalized) - 1.5, 0.0) ** 2
    angles = np.mod(np.degrees(np.arctan2(fy, fx)), 180.0)
    angle_bin = (angles * 4).astype(np.intp) % 720
    valid = (radius >= 0.015) & (radius <= 0.35)
    scores = np.bincount(angle_bin[valid], weights=excess[valid], minlength=720)
    scores /= np.maximum(1, np.bincount(angle_bin[valid], minlength=720))
    scores = ndimage.gaussian_filter1d(scores, 3.0, mode="wrap")
    peak = int(np.argmax(scores))
    baseline = float(np.median(scores))
    mad = float(np.median(np.abs(scores - baseline)))
    confidence = float(np.clip((scores[peak] - baseline - 6 * mad) / (scores[peak] + 1e-8), 0, 1))
    angle = (peak + 0.5) / 4
    coherence = 0.0
    if confidence >= MIN_ORIENTATION_CONFIDENCE:
        # Narrow angular FFT bins are quantized on a Cartesian grid. Refine in
        # image space, where the correct normal maximizes coherent line energy.
        # A short image dimension spreads a spectral line over a wider angle.
        # Include that uncertainty before refining (e.g. a very wide panorama).
        radius_deg = max(4.0, min(30.0, math.degrees(math.atan2(2.0 / min(height, width), 0.06))))
        candidates = np.arange(angle - radius_deg, angle + radius_deg + 0.01, 0.25) if refine else [angle]
        energies = []
        for candidate in candidates:
            profile, coverage = normal_profile(data, float(candidate))
            energies.append(float(np.sum(profile ** 2 * coverage) / max(coverage.sum(), 1e-8)))
        best = int(np.argmax(energies))
        coherence = float(energies[best] / max(float(np.mean(data ** 2)), 1e-8))
        angle = float(candidates[best])
        if 0 < best < len(energies) - 1:
            left, center, right = energies[best - 1:best + 2]
            curvature = left - 2 * center + right
            if abs(curvature) > 1e-12:
                angle += 0.25 * float(np.clip(0.5 * (left - right) / curvature, -1, 1))
    radians = math.radians(angle)
    original_angle = math.degrees(math.atan2(math.sin(radians) * scale[0], math.cos(radians) * scale[1]))
    return OrientationEvidence((original_angle + 90.0) % 180.0 - 90.0, confidence, coherence)


def detect_orientation(gray: np.ndarray) -> OrientationEvidence:
    """Use local evidence when the full-frame window misses a corner scratch.

    Angular confidence alone is insufficient: fine camera/tissue texture can
    have a sharp spectral peak without forming a coherent line in the image.
    Local candidates must also survive a line-projection energy check.
    """
    global_evidence = _orientation_evidence(gray)
    if global_evidence.confidence >= MIN_ORIENTATION_CONFIDENCE and global_evidence.coherence >= 0.02:
        return global_evidence
    data, scale = analysis_image(gray)
    height, width = data.shape
    if min(height, width) < 64 or float(np.ptp(data)) < 1e-6:
        return replace(global_evidence, confidence=0.0)
    candidates = []
    for fraction_h, fraction_w in ((0.5, 0.65), (0.3, 0.4)):
        block_h, block_w = round(height * fraction_h), round(width * fraction_w)
        minimum_coherence = max(0.03, 6.0 / min(block_h, block_w))
        for y0 in np.linspace(0, height - block_h, 3, dtype=int):
            for x0 in np.linspace(0, width - block_w, 3, dtype=int):
                patch = data[y0:y0 + block_h, x0:x0 + block_w]
                evidence = _orientation_evidence(patch, refine=False)
                if evidence.confidence >= 0.75 and evidence.coherence >= minimum_coherence:
                    score = evidence.coherence * evidence.confidence * math.sqrt(block_h * block_w)
                    candidates.append((score, int(x0), int(y0), block_w, block_h, evidence.angle_deg))
    for _, x0, y0, block_w, block_h, _ in sorted(candidates, reverse=True):
        evidence = _orientation_evidence(data[y0:y0 + block_h, x0:x0 + block_w])
        minimum_coherence = max(0.03, 6.0 / min(block_h, block_w))
        if evidence.confidence < 0.75 or evidence.coherence < minimum_coherence:
            continue
        # Merge overlapping evidence at the same angle, retaining the coverage
        # of a larger window when a smaller corner crop is more confident.
        x1, y1 = x0 + block_w, y0 + block_h
        bounds = [x0, y0, x1, y1]
        for _, cx, cy, cw, ch, candidate_angle in candidates:
            agrees = abs((candidate_angle - evidence.angle_deg + 90) % 180 - 90) <= 5
            overlaps = cx < x1 and cx + cw > x0 and cy < y1 and cy + ch > y0
            if agrees and overlaps:
                bounds = [min(bounds[0], cx), min(bounds[1], cy), max(bounds[2], cx + cw), max(bounds[3], cy + ch)]
        # Return the evidence box in original coordinates, so filtering can
        # leave the rest of the image untouched. Do not recurse on this box.
        region = (
            round(bounds[0] / scale[1]), round(bounds[1] / scale[0]),
            round(bounds[2] / scale[1]), round(bounds[3] / scale[0]),
        )
        angle = math.radians(evidence.angle_deg)
        angle_deg = math.degrees(math.atan2(math.sin(angle) * scale[0], math.cos(angle) * scale[1]))
        return replace(evidence, angle_deg=(angle_deg + 90) % 180 - 90, region=region)
    return replace(global_evidence, confidence=0.0)


def estimate_orientation(gray: np.ndarray) -> tuple[float, float]:
    evidence = detect_orientation(gray)
    return evidence.angle_deg, evidence.confidence


def estimate_period(gray: np.ndarray, orientation_deg: float) -> tuple[float, float]:
    data, scale = analysis_image(gray, maximum=1400)
    if min(data.shape) < 8 or float(np.ptp(data)) < 1e-6:
        return 18.0, 0.0
    angle, normal_scale = scaled_normal(orientation_deg, scale)
    profile, coverage = normal_profile(data, angle)
    supported = coverage > float(coverage.max()) * 0.08
    if np.count_nonzero(supported) < 16:
        return 18.0, 0.0
    profile = profile[supported]
    profile -= ndimage.gaussian_filter1d(profile, max(8.0, len(profile) / 16.0), mode="reflect")
    profile *= np.hanning(len(profile))
    nfft = fft.next_fast_len(len(profile) * 4)
    power = np.abs(fft.rfft(profile, n=nfft)) ** 2
    frequencies = fft.rfftfreq(nfft) * normal_scale
    valid = (frequencies >= 1 / 512) & (frequencies <= 1 / 3)
    indices = np.flatnonzero(valid)
    if not len(indices) or float(power[valid].max()) < 1e-8:
        return 18.0, 0.0
    peak = int(indices[np.argmax(power[valid])])
    position = float(peak)
    if 0 < peak < len(power) - 1:
        left, center, right = np.log(np.maximum(power[peak - 1:peak + 2], 1e-12))
        curvature = left - 2 * center + right
        if abs(curvature) > 1e-12:
            position += float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    frequency = position / nfft * normal_scale
    # A narrow peak must explain a meaningful fraction of the profile energy.
    # Irregular scratches may have a dominant scale without a reliable period.
    half_width = max(1, math.ceil(nfft / len(profile)))
    concentration = float(power[max(0, peak-half_width):peak+half_width+1].sum() / max(power[valid].sum(), 1e-8))
    confidence = float(np.clip((concentration - 0.1) / 0.65, 0, 1))
    return float(np.clip(1.0 / frequency, 3, 512)), confidence
