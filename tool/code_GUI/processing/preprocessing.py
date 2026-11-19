"""Image processing shared by preview, single export, and batch export."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .images import read_image, write_image
from .large_image_backend import directional_median


@dataclass(frozen=True)
class PreprocessingOptions:
    clip: int
    tile: tuple[int, int]
    frequency: tuple[int, int] | None = None
    toolmarks: dict | None = None


def enhance_image(
    img: np.ndarray,
    clahe_clip_limit: float = 8,
    clahe_tile_grid_size: tuple = (10, 10),
    mode: str = "bgr"
):
    """
    对图像应用 CLAHE 增强，兼容灰度与彩色图像。
    要求输入为 8-bit（uint8），灰度图保持单通道输出，彩色图按三个通道分别增强后再合并。

    参数:
    - img: 输入图像（np.ndarray），可为灰度 (H,W) 或 BGR 彩色 (H,W,3) 或 BGRA (H,W,4)
    - clahe_clip_limit: CLAHE 的 clipLimit 参数
    - clahe_tile_grid_size: CLAHE 的 tileGridSize 参数
    - mode:
        "bgr"  : 彩色图对 B、G、R 三个通道分别应用 CLAHE（满足每通道增强需求）
        "gray" : 强制按灰度增强（若彩色将转换为灰度后增强并返回灰度）

    返回:
    - 增强后的图像（与输入通道数保持一致；"gray"模式返回灰度图）
    """
    if img is None:
        return img

    # 仅支持 8-bit 图像进行 CLAHE；若不是 uint8，则直接返回以保持位深一致
    if img.dtype != np.uint8:
        return img

    has_alpha = (img.ndim == 3 and img.shape[2] == 4)
    alpha = None
    if has_alpha:
        alpha = img[:, :, 3].copy()
        img = img[:, :, :3]

    # 灰度图
    if img.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
        out = clahe.apply(img)
        return out

    # 彩色图 (BGR)
    if mode == "gray":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
        out = clahe.apply(gray)
        return out

    # HSV增强，只增强S和V
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # CLAHE
    clahe_v = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
    clahe_s = cv2.createCLAHE(clipLimit=clahe_clip_limit/2, tileGridSize=clahe_tile_grid_size)

    v2 = clahe_v.apply(v)
    s2 = clahe_s.apply(s)

    hsv2 = cv2.merge((h, s2, v2))
    bgr_eq = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

    if has_alpha:
        bgr_eq = cv2.merge((bgr_eq, alpha))
    return bgr_eq


def freq_filter(
        img,
        angle,
        bandwidth=20
):
    """
    频域带阻滤波，输入图像、频谱图条纹方向（由交互获得）、带宽，返回滤波后图像。
    注意：angle为频谱图中条纹方向，直接用于mask。
    优化掩膜生成部分，提升效率。
    """
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    # 生成坐标网格
    y = np.arange(rows) - crow
    x = np.arange(cols) - ccol
    X, Y = np.meshgrid(x, y)
    theta = np.degrees(np.arctan2(Y, X))
    # 计算与目标方向的最小夹角（周期性）
    diff = (theta - angle + 180) % 360 - 180
    mask = np.ones((rows, cols), np.float32)
    mask[np.abs(diff) < bandwidth] = 0
    mask[np.abs(diff) > (180 - bandwidth)] = 0
    # 傅里叶变换与滤波
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift_masked = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_masked)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back_norm = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back_uint8 = np.uint8(img_back_norm)
    return img_back_uint8


def _as_uint8(image):
    if image.dtype == np.uint8:
        return image
    image = np.nan_to_num(image)
    if image.ndim == 2:
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    channels = [_as_uint8(channel) for channel in cv2.split(image)]
    return cv2.merge(channels)


def process_image(source, options, *, preview_scale=None):
    base = source
    report = None
    if options.toolmarks is not None:
        base, report = directional_median(source, options.toolmarks, preview_scale=preview_scale)
    result = enhance_image(base, options.clip, options.tile, mode="bgr")
    if options.frequency is not None:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if result.ndim == 3 else result
        result = freq_filter(gray, *options.frequency)
        if base.ndim == 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            if base.shape[2] == 4:
                result = cv2.merge((*cv2.split(result), _as_uint8(base[..., 3])))
    return _as_uint8(result), report


def export_images(paths, destination, options, *, single_file, progress, check_interrupted):
    destination = Path(destination)
    paths = tuple(map(Path, paths))
    targets = [destination] if single_file else [destination / f"processed_{path.name}" for path in paths]
    resolved_targets = {path.resolve() for path in targets}
    if len(resolved_targets) != len(paths):
        raise ValueError("选中的输入文件会生成同名结果，请分开保存。")
    if not single_file and resolved_targets.intersection(path.resolve() for path in paths):
        raise ValueError("批量输出路径与输入图像冲突，请选择其他目录。")

    saved = 0
    errors = []
    for index, (path, target) in enumerate(zip(paths, targets), 1):
        check_interrupted()
        try:
            source = read_image(path)
            if source is None:
                raise ValueError("无法读取图像")
            result, _ = process_image(source, options)
            write_image(target, result)
            saved += 1
        except (OSError, ValueError, RuntimeError, cv2.error) as error:
            errors.append(f"{path.name}：{error}")
        progress(index)
    return saved, errors
