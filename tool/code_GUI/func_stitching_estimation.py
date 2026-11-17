"""将 large-image-tool 的全局几何估计适配为现有拼接页的像素参数。"""

import math
from pathlib import Path
import sys


def estimate_stitching_parameters(
    image_paths, rows, cols, direction_width, direction_height, progress=None
):
    """按现有拼接顺序估计第一组完整网格，返回旋转后画布中的重叠像素。"""
    if rows < 2 or cols < 2:
        raise ValueError("自动估计需要至少 2 行、2 列；请先设置横向数量和纵向数量。")
    count = rows * cols
    if len(image_paths) < count:
        raise ValueError(f"当前设置需要 {count} 张图像，已选择 {len(image_paths)} 张。")
    if direction_width not in ("向左", "向右") or direction_height not in ("向上", "向下"):
        raise ValueError("不支持的拼接朝向。")

    # 只加载计算模块，不引入 large-image-tool 的 PySide6 界面。
    source_root = Path(__file__).resolve().parents[1] / "large-image-tool" / "src"
    if source_root.is_dir() and str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    try:
        from PIL import Image
        from large_image_tool.models import ScanPattern, StartCorner, TileRecord, TileSet
        from large_image_tool.registration import AutoStitcher
    except ImportError as error:
        raise RuntimeError(
            "无法加载自动估计模块，请保留同级 large-image-tool 目录，并安装 code_GUI/requirements.txt 中的依赖。"
        ) from error

    paths = [Path(path) for path in image_paths[:count]]
    size = None
    for path in paths:
        try:
            with Image.open(path) as image:
                current_size = image.size
                if image.getexif().get(274, 1) != 1:
                    raise ValueError("图像含 EXIF 旋转标记，请先将其转换为方向确定的图像")
        except (OSError, ValueError) as error:
            raise ValueError(f"无法用于自动估计：{path.name}（{error}）") from error
        if size is None:
            size = current_size
        elif current_size != size:
            raise ValueError("第一组图像尺寸不一致，无法自动估计。")

    # 沿用 func_img_stitching 的实际放置顺序；不改变已有朝向控件的行为。
    corner = {
        ("向左", "向下"): StartCorner.TOP_LEFT,
        ("向右", "向下"): StartCorner.TOP_RIGHT,
        ("向左", "向上"): StartCorner.BOTTOM_LEFT,
        ("向右", "向上"): StartCorner.BOTTOM_RIGHT,
    }[(direction_width, direction_height)]
    pattern = ScanPattern(start_corner=corner, snake=False)
    width, height = size
    records = []
    for slot, path in enumerate(paths):
        row, col = pattern.slot_to_rc(slot, rows, cols)
        records.append(TileRecord(slot + 1, slot, row, col, path))
    tiles = TileSet(
        paths[0].parent, rows, cols, pattern, records=records,
        tile_width=width, tile_height=height,
    )
    stitcher = AutoStitcher(tiles, max_registration_size=1024, cache_tiles=16)
    try:
        geometry, edges = stitcher.estimate_geometry(progress=progress)
    except RuntimeError as error:
        raise ValueError("未找到足够可靠的相邻图像，请检查行列数量、图像顺序和拼接朝向。") from error

    # large-image-tool 的几何角度与现有 rotate_image 的逆时针角度相反。
    # 按界面的一位小数计算画布，重叠像素需包含旋转扩大后的黑边。
    angle = round(-geometry.tile_rotation_deg, 1)
    radians = math.radians(angle)
    rotated_width = int(height * abs(math.sin(radians)) + width * abs(math.cos(radians)))
    rotated_height = int(height * abs(math.cos(radians)) + width * abs(math.sin(radians)))
    overlap_width = round(rotated_width - width * (1 - geometry.overlap_x_percent / 100))
    overlap_height = round(rotated_height - height * (1 - geometry.overlap_y_percent / 100))
    return {
        "overlap_width": max(1, min(rotated_width - 1, overlap_width)),
        "overlap_height": max(1, min(rotated_height - 1, overlap_height)),
        "angle": angle,
        "sample_pairs": len(edges),
    }
