"""削弱明场组织图像中的圆周车削纹，保存独立结果和处理记录。

示例（在仓库根目录运行）：
  python tool/remove_circular_marks.py input.png output.png
  python tool/remove_circular_marks.py input.png output.png --strength 0.7
  python tool/remove_circular_marks.py input.png output.png --center 2609 1370

先稳健拟合圆心，再在极坐标中估计沿圆周持续的亮度纹路。
仅将估计的纹路层映回原图并扣除，原图本身不重采样。
高亮边界、显著色彩和强局部反差自动受到保护；--protect-mask 的非零像素完全保留。
无法保证区分与车削纹同心、同尺度的真实组织；应检查对比图和去除分量。
支持 uint8/uint16 灰度、BGR、BGRA；输出 PNG/TIFF，不覆盖输入或已有结果。
当前处理上限 4000 万像素，超大图请显式使用 --preview-max-side 5000 生成缩略结果。

依赖：numpy, scipy, opencv-python, Pillow。
极坐标去环纹的背景资料（本实现针对车削图像，未调用 Algotom）：
https://algotom.github.io/toc/section4/section4_3.html#transformation-between-cartesian-and-polar-coordinate-system
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage, optimize


MAX_PIXELS = 40_000_000


@dataclass
class RingSettings:
    strength: float = 0.9
    radial_width: float = 24.0
    arc_length: float = 30.0
    iterations: int = 2
    max_correction: float = 20.0  # Expressed in 8-bit-equivalent gray levels.

    def validate(self):
        values = (self.strength, self.radial_width, self.arc_length, self.max_correction)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("参数必须是有限数值")
        if not 0 <= self.strength <= 1:
            raise ValueError("strength 必须在 0–1 之间")
        if not 1 <= self.radial_width <= 100 or not 5 <= self.arc_length <= 200:
            raise ValueError("width 范围为 1–100，arc-length 范围为 5–200 像素")
        if not 1 <= self.iterations <= 4 or not 0 < self.max_correction <= 40:
            raise ValueError("iterations 范围为 1–4，max-correction 范围为 (0, 40]")


def _analysis(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if image.dtype not in (np.uint8, np.uint16):
        raise ValueError("仅支持 uint8/uint16 图像，不会自动改变原图动态范围")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
        raise ValueError("仅支持灰度、三通道或四通道图像")
    if min(image.shape[:2]) < 16 or image.shape[0] * image.shape[1] > MAX_PIXELS:
        raise ValueError("图像尺寸须至少为 16×16，且不超过 4000 万像素")
    if max(image.shape[:2]) >= 16000:
        raise ValueError("当前极坐标实现要求单边小于 16000 像素，请使用缩略预览")
    scale = 255.0 / np.iinfo(image.dtype).max
    color = image[..., :3] if image.ndim == 3 else image
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else color
    gray = gray.astype(np.float32) * scale
    chroma = np.ptp(color.astype(np.float32), axis=2) * scale if image.ndim == 3 else np.zeros_like(gray)
    contrast = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 2))
    protected = np.maximum(np.clip((gray - 205) / 20, 0, 1), np.clip((chroma - 20) / 25, 0, 1))
    protected = np.maximum(protected, np.clip((contrast - 8) / 15, 0, 1))
    protected = cv2.dilate(protected, np.ones((3, 3), np.uint8))
    weight = (1 - protected)**2 * np.clip((gray - 60) / 40, 0, 1)
    if image.ndim == 3 and image.shape[2] == 4:
        weight *= image[..., 3] == np.iinfo(image.dtype).max
    return gray, weight, scale


def estimate_center(gray: np.ndarray, weight: np.ndarray) -> dict:
    """Fit the intersection of reliable local gradient normals with a robust loss."""
    h, w = gray.shape
    size = max(24, min(160, min(h, w) // 10))
    smooth = cv2.GaussianBlur(gray, (0, 0), 0.7)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3) / 8
    points, normals, weights = [], [], []
    for y in range(size // 2, h - size // 2, size):
        for x in range(size // 2, w - size // 2, size):
            sl = np.s_[y-size//2:y+size//2, x-size//2:x+size//2]
            mask = cv2.erode((weight[sl] > .3).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            if mask.mean() < .3:
                continue
            dx, dy = gx[sl][mask], gy[sl][mask]
            magnitude = np.hypot(dx, dy)
            cap = float(np.quantile(magnitude, .7))
            if cap < .08:
                continue
            robust = np.minimum(1, (cap / (magnitude + 1e-6))**2)
            xx, xy, yy = np.mean(robust*dx*dx), np.mean(robust*dx*dy), np.mean(robust*dy*dy)
            coherence = np.hypot(xx-yy, 2*xy) / (xx+yy+1e-6)
            if coherence < .25:
                continue
            angle = .5 * np.arctan2(2*xy, xx-yy)
            points.append((x, y))
            normals.append((np.cos(angle), np.sin(angle)))
            weights.append(coherence**2 * mask.mean())
    report = {"center_xy": None, "reliable": False, "direction_samples": len(points),
              "inlier_fraction": 0.0, "median_angle_error_deg": None, "source": "automatic"}
    if len(points) < 12:
        return report
    points, normals, weights = np.array(points), np.array(normals), np.array(weights)
    tangent = normals[:, ::-1] * [-1, 1]

    def residual(center):
        delta = points - center
        return np.sum(delta*tangent, axis=1) / np.maximum(np.linalg.norm(delta, axis=1), size)

    candidates = [optimize.least_squares(lambda c: np.sqrt(weights)*residual(c), initial,
                   loss="cauchy", f_scale=.05, bounds=([-w, -h], [2*w, 2*h]))
                  for initial in [(w/2, h/2), (.6*w, .4*h), (.5*w, .65*h)]]
    fit = min(candidates, key=lambda item: item.cost)
    error = np.abs(residual(fit.x))
    inliers = error < math.sin(math.radians(6))
    fraction = float(np.sum(weights[inliers]) / np.sum(weights))
    diversity = float(np.linalg.eigvalsh((normals.T*weights) @ normals / np.sum(weights))[0])
    reliable = fraction >= .55 and inliers.sum() >= 12 and diversity > .07
    report.update(center_xy=fit.x.tolist(), reliable=bool(reliable), inlier_fraction=fraction,
                  median_angle_error_deg=float(np.rad2deg(np.arcsin(np.clip(np.median(error), 0, 1)))))
    return report


def _angular_template(residual, weight, radial, arc_length):
    """Normalized angular averaging, with overlaps to avoid radial band seams."""
    nt, nr = weight.shape
    result = np.zeros_like(weight)
    total = np.zeros(nr, np.float32)
    for middle in range(0, nr + 128, 128):
        left, right = max(0, middle-128), min(nr, middle+129)
        if right <= left:
            continue
        blend = np.maximum(0, 1-np.abs(np.arange(left, right)-middle)/128).astype(np.float32)
        radius = max(60, float(radial[min(middle, nr-1)]))
        sigma = arc_length / radius * nt / (2*np.pi)
        denominator = ndimage.gaussian_filter1d(weight[:, left:right], sigma, axis=0, mode="wrap")
        numerator = ndimage.gaussian_filter1d(residual[:, left:right], sigma, axis=0, mode="wrap")
        correction = numerator / np.maximum(denominator, .05) * np.clip(denominator/.35, 0, 1)
        result[:, left:right] += correction * blend
        total[left:right] += blend
    return result / np.maximum(total, 1e-6)


def remove_circular_marks(image: np.ndarray, settings: RingSettings | None = None,
                          center: tuple[float, float] | None = None,
                          protect_mask: np.ndarray | None = None,
                          progress: Callable[[str], None] | None = None) -> tuple[np.ndarray, dict, np.ndarray]:
    settings = settings or RingSettings()
    settings.validate()
    gray, weight, scale = _analysis(image)
    h, w = gray.shape
    if protect_mask is not None:
        if protect_mask.shape != gray.shape:
            raise ValueError("保护蒙版尺寸必须与处理图像一致")
        weight[protect_mask != 0] = 0
    if center is None:
        evidence = estimate_center(gray, weight)
    else:
        if len(center) != 2 or not all(math.isfinite(v) for v in center):
            raise ValueError("圆心应为两个有限坐标 x y")
        if not -w <= center[0] <= 2*w or not -h <= center[1] <= 2*h:
            raise ValueError("圆心超出支持的范围：图像四周一倍宽高以内")
        evidence = {"center_xy": list(center), "reliable": True, "source": "manual"}
    report = {"method": "protected_polar_residual", "settings": asdict(settings),
              "center_estimation": evidence, "applied": False,
              "protected_pixels": int(np.count_nonzero(weight == 0)),
              "protected_pixels_changed": 0,
              "limitation": "同心、同尺度的真实组织可能与车削纹混淆；应结合局部对比和去除分量复核。"}
    cumulative = np.zeros_like(gray)
    if settings.strength == 0 or not evidence["reliable"]:
        report["reason"] = "强度为零" if settings.strength == 0 else "未检出可靠圆心，原图保持不变；可手动指定 --center x y"
        return image.copy(), report, cumulative
    cx, cy = evidence["center_xy"]
    if progress:
        progress(f"圆心 ({cx:.1f}, {cy:.1f})；提取圆周纹分量…")
    max_radius = math.hypot(max(abs(cx), abs(w-1-cx)), max(abs(cy), abs(h-1-cy)))+1
    step = .5
    radial = np.arange(0, max_radius, step, dtype=np.float32)
    if len(radial) >= 32767:
        raise ValueError("极坐标半径过大，请缩小图像或调整圆心")
    nt = int(np.clip(math.ceil(np.pi * max_radius), 512, 4096))
    theta = np.arange(nt, dtype=np.float32)[:, None] * (2*np.pi/nt)
    mapx, mapy = cx+np.cos(theta)*radial[None, :], cy+np.sin(theta)*radial[None, :]
    polar_weight = cv2.remap(weight, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    denominator = cv2.GaussianBlur(polar_weight, (0, 1), settings.radial_width/step)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    radius = np.hypot(xx-cx, yy-cy)
    angle = (np.arctan2(yy-cy, xx-cx) % (2*np.pi)) * (nt/(2*np.pi))
    gate = weight * np.clip((radius-20)/40, 0, 1)
    del xx, yy
    for iteration in range(settings.iterations):
        polar = cv2.remap(gray-cumulative, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        background = cv2.GaussianBlur(polar*polar_weight, (0, 1), settings.radial_width/step) / np.maximum(denominator, .001)
        residual = np.clip(polar-background, -25, 25) * polar_weight
        del polar, background
        template = _angular_template(residual, polar_weight, radial, settings.arc_length)
        del residual
        # Duplicate the first angular row to make interpolation across 2*pi continuous.
        field = cv2.remap(np.vstack([template, template[:1]]), radius/step, angle, cv2.INTER_LINEAR)
        del template
        cumulative += settings.strength * field * gate
        np.clip(cumulative, -settings.max_correction, settings.max_correction, out=cumulative)
        if progress:
            progress(f"完成第 {iteration+1}/{settings.iterations} 次纹路估计")
    # Use one shared integer offset for all color channels. Clamp that offset,
    # not the individual channels, to preserve chromatic differences exactly.
    color = image[..., :3] if image.ndim == 3 else image
    channel_min = color.min(axis=2) if image.ndim == 3 else color
    channel_max = color.max(axis=2) if image.ndim == 3 else color
    delta = np.rint(cumulative/scale).astype(np.int32)
    delta = np.maximum(channel_max.astype(np.int32)-np.iinfo(image.dtype).max,
                       np.minimum(delta, channel_min.astype(np.int32)))
    filtered = image.copy()
    corrected = color.astype(np.int32) - (delta[..., None] if image.ndim == 3 else delta)
    if image.ndim == 3:
        filtered[..., :3] = corrected.astype(image.dtype)
    else:
        filtered = corrected.astype(image.dtype)
    report.update(applied=bool(np.any(delta)), changed_pixels=int(np.count_nonzero(delta)),
                  protected_pixels_changed=int(np.count_nonzero(delta[weight == 0])),
                  maximum_change_8bit=float(np.max(np.abs(delta))*scale),
                  rms_change_8bit=float(np.sqrt(np.mean((delta.astype(np.float32)*scale)**2))),
                  color_differences_preserved=True, alpha_preserved=True)
    return filtered, report, delta.astype(np.float32)*scale


def read_image(path: Path, preview_max_side: int | None = None) -> tuple[np.ndarray, dict]:
    # Read dimensions before OpenCV allocates a potentially gigapixel image.
    limit = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as header:
            width, height = header.size
            info = {"source_size_wh": [width, height], "preview": False}
            if preview_max_side is not None and max(width, height) > preview_max_side:
                if header.mode not in ("RGB", "RGBA", "L"):
                    raise ValueError("缩略预览仅支持 8 位 RGB/RGBA/灰度；16 位图像请先生成预览")
                if width*height > MAX_PIXELS and header.format != "JPEG":
                    raise ValueError("当前仅支持对超大 JPEG 直接缩略解码；其他超大格式请先生成独立预览图")
                ratio = preview_max_side / max(width, height)
                target = (max(1, round(width*ratio)), max(1, round(height*ratio)))
                header.draft(header.mode, target)
                header.thumbnail(target, Image.Resampling.LANCZOS)
                data = np.array(header)
                if data.ndim == 3:
                    data = cv2.cvtColor(data, cv2.COLOR_RGBA2BGRA if data.shape[2] == 4 else cv2.COLOR_RGB2BGR)
                info.update(preview=True, processed_size_wh=[data.shape[1], data.shape[0]])
                return data, info
            if width*height > MAX_PIXELS or max(width, height) >= 16000:
                raise ValueError(f"原图为 {width}×{height}，超过当前处理上限；请指定 --preview-max-side 5000，输出将是缩略结果")
    finally:
        Image.MAX_IMAGE_PIXELS = limit
    data = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise ValueError(f"无法读取图像：{path}")
    info["processed_size_wh"] = [data.shape[1], data.shape[0]]
    return data, info


def write_image(path: Path, image: np.ndarray):
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"图像编码失败：{path}")
    # Exclusive creation avoids silently replacing a previous analysis result.
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(memoryview(encoded))
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise


def make_comparison(before: np.ndarray, after: np.ndarray, delta: np.ndarray) -> np.ndarray:
    scale = 255.0 / np.iinfo(before.dtype).max
    a, b = np.rint(before.astype(np.float32)*scale).astype(np.uint8), np.rint(after.astype(np.float32)*scale).astype(np.uint8)
    if a.ndim == 2:
        a, b = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR), cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    else:
        a, b = a[..., :3], b[..., :3]
    h, w = a.shape[:2]
    # Broad markings, tissue boundaries, and dense tissue, all at native scale.
    rectangles = [(int(.75*w), int(.06*h)), (int(.24*w), int(.19*h)), (int(.57*w), int(.68*h))]
    cw, ch = min(720, w), min(400, h)
    rows = []
    for x, y in rectangles:
        x, y = min(x, w-cw), min(y, h-ch)
        sl = np.s_[y:y+ch, x:x+cw]
        removed = cv2.cvtColor(np.clip(128+delta[sl]*5, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        row = np.hstack([a[sl], b[sl], removed])
        row = cv2.copyMakeBorder(row, 35, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        for i, label in enumerate([f"Original ({x}, {y})", "Corrected", "Removed component x5"]):
            cv2.putText(row, label, (i*cw+10, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1)
        rows.append(row)
    return np.vstack(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="新建的 PNG/TIFF 输出路径")
    parser.add_argument("--center", nargs=2, type=float, metavar=("X", "Y"), help="手动圆心，坐标以实际处理图像为准")
    parser.add_argument("--strength", type=float, default=.9)
    parser.add_argument("--width", type=float, default=24, help="径向背景分离尺度，默认 24 像素")
    parser.add_argument("--arc-length", type=float, default=30, help="沿圆周平滑尺度，默认 30 像素")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--max-correction", type=float, default=20, help="累计最大改变量，以 8 位灰度级计")
    parser.add_argument("--protect-mask", type=Path, help="非零像素完全保留；尺寸与实际处理图像相同")
    parser.add_argument("--preview-max-side", type=int, help="显式生成缩略结果，适用于超大 JPEG")
    args = parser.parse_args(argv)
    started = perf_counter()
    try:
        settings = RingSettings(args.strength, args.width, args.arc_length, args.iterations, args.max_correction)
        settings.validate()
        if args.preview_max_side is not None and not 128 <= args.preview_max_side <= 6000:
            raise ValueError("preview-max-side 范围为 128–6000")
        if args.output.suffix.lower() not in (".png", ".tif", ".tiff"):
            raise ValueError("请使用 PNG 或 TIFF 无损输出")
        paths = [args.output, args.output.with_suffix(".report.json"), args.output.with_suffix(".comparison.png"),
                 args.output.with_suffix(".removed.png")]
        if any(path.exists() for path in paths):
            raise FileExistsError("输出或配套报告已存在，请指定新的输出文件名")
        image, info = read_image(args.input, args.preview_max_side)
        mask = None
        if args.protect_mask is not None:
            mask, _ = read_image(args.protect_mask)
            if mask.ndim == 3:
                mask = np.max(mask[..., :3], axis=2)
        filtered, report, delta = remove_circular_marks(
            image, settings, args.center, mask, lambda message: print(message, flush=True))
        report.update(info, input=str(args.input.resolve()), output=str(args.output.resolve()),
                      elapsed_seconds=round(perf_counter()-started, 2))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_image(paths[0], filtered)
        write_image(paths[2], make_comparison(image, filtered, delta))
        write_image(paths[3], np.clip(128+delta*5, 0, 255).astype(np.uint8))
        with paths[1].open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        print(f"结果：{paths[0]}\n局部对比：{paths[2]}\n处理记录：{paths[1]}")
        if not report["applied"]:
            print(report.get("reason", "没有需要修正的分量"))
        return 0
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f"错误：{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
