import numpy as np


def is_black(
        pixel,
        threshold=10
):
    """判断一个像素是否为黑色（支持多通道）"""
    if pixel.ndim == 3:
        return np.all(pixel < threshold, axis=-1)
    else:
        return pixel < threshold


def stitch_images_width(
        images,
        overlap_width=60,
        black_threshold=10,
        direction='向左'
):
    """
    横向拼接图像序列，支持向左或向右拼接。
    images: list of np.ndarray, shape (H, W, C)
    overlap_width: 重叠宽度
    black_threshold: 判断黑色的阈值
    direction: 'right'（默认，后图拼接在右侧）或 'left'（后图拼接在左侧）
    return: 拼接后的图像
    """
    if not images:
        return None
    result = images[0].copy()
    for img in images[1:]:
        ow = overlap_width
        if direction == '向左':
            left_part = result[:, :-ow]
            right_part = img[:, ow:]
            overlap_result = result[:, -ow:]
            overlap_img = img[:, :ow]
        elif direction == '向右':
            left_part = img[:, :-ow]
            right_part = result[:, ow:]
            overlap_result = img[:, -ow:]
            overlap_img = result[:, :ow]
        else:
            raise ValueError("direction must be 'right' or 'left'")
        # 生成掩码
        mask = ~is_black(overlap_img, black_threshold)
        overlap_combined = overlap_result.copy()
        overlap_combined[mask] = overlap_img[mask]
        if direction == '向左':
            result = np.hstack((left_part, overlap_combined, right_part))
        else:
            result = np.hstack((left_part, overlap_combined, right_part))
    return result


def stitch_images_height(
        images,
        overlap_height=60,
        black_threshold=10,
        direction='向下'
):
    """
    纵向拼接图像序列，支持向上或向下拼接。
    images: list of np.ndarray, shape (H, W, C)
    overlap_height: 重叠高度
    black_threshold: 判断黑色的阈值
    direction: 'down'（默认，后图拼接在下方）或 'up'（后图拼接在上方）
    return: 拼接后的图像
    """
    if not images:
        return None
    result = images[0].copy()
    for img in images[1:]:
        oh = overlap_height
        if direction == '向下':
            top_part = result[:-oh, :]
            bottom_part = img[oh:, :]
            overlap_result = result[-oh:, :]
            overlap_img = img[:oh, :]
        elif direction == '向上':
            top_part = img[:-oh, :]
            bottom_part = result[oh:, :]
            overlap_result = img[-oh:, :]
            overlap_img = result[:oh, :]
        else:
            raise ValueError("direction must be 'down' or 'up'")
        # 生成掩码
        mask = ~is_black(overlap_img, black_threshold)
        overlap_combined = overlap_result.copy()
        overlap_combined[mask] = overlap_img[mask]
        if direction == '向下':
            result = np.vstack((top_part, overlap_combined, bottom_part))
        else:
            result = np.vstack((top_part, overlap_combined, bottom_part))
    return result


def stitch_images_mosaic(
        images, grid_shape,
        overlap_width=60, overlap_height=60,
        black_threshold=10,
        direction_width='向左', direction_height='向下'
):
    """
    images: list of np.ndarray, 按行优先排列，数量=rows*cols
    grid_shape: (rows, cols)
    其他参数见 stitch_images_width/height
    返回: 马赛克拼接好的图像
    """
    rows, cols = grid_shape
    assert len(images) == rows * cols, "图片数量必须等于 rows*cols！"

    # 1. 先拼接每一行
    row_images = []
    for r in range(rows):
        imgs_row = images[r * cols: (r + 1) * cols]
        stitched_row = stitch_images_width(
            imgs_row,
            overlap_width=overlap_width,
            black_threshold=black_threshold,
            direction=direction_width
        )
        row_images.append(stitched_row)

    # 2. 再把所有行纵向拼接
    mosaic_img = stitch_images_height(
        row_images,
        overlap_height=overlap_height,
        black_threshold=black_threshold,
        direction=direction_height
    )
    return mosaic_img