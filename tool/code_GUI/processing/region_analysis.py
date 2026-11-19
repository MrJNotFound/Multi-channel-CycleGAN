"""Batch segmentation, label filtering and IoU output for the region page."""

from pathlib import Path
import os

import cv2
import numpy as np

from skimage.measure import regionprops

from .images import preview_geometry, read_image, write_image


def read_labels(path):
    labels = read_image(path)
    if labels is None:
        raise ValueError(f"无法读取标签图像：{Path(path).name}")
    return cv2.cvtColor(labels, cv2.COLOR_BGR2GRAY) if labels.ndim == 3 else labels


def write_labels(path, labels):
    maximum = labels.max(initial=0)
    if maximum > np.iinfo(np.uint16).max or labels.min(initial=0) < 0:
        raise ValueError("标签超出 PNG 支持的整数范围。")
    dtype = np.uint8 if maximum <= 255 else np.uint16
    write_image(path, labels.astype(dtype))


def kmeans_watershed_segmentation(
        image
):
    """
    输入原始图像（支持灰度或BGR/RGB），输出标签图像（markers）。
    步骤：KMeans聚类分割+形态学处理+分水岭算法。
    """
    # 判断输入图像通道数，兼容灰度和彩色
    if image.ndim == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_for_ws = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_for_ws = image
    else:
        raise ValueError('输入图像必须为灰度或3通道彩色图像')
    # 图像拉平成二维像素数组，便于KMeans聚类
    pixel_values = image_rgb.reshape((-1, 3)).astype(np.float32)
    k = 2  # 聚类类别数
    # KMeans聚类参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # KMeans聚类
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    # 聚类结果还原为图像
    segmented_image = centers[labels.flatten()].reshape(image_rgb.shape)
    # 转为灰度图
    gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
    # 大津阈值分割为二值图
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 形态学闭运算去除小黑洞
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(~binary_image, cv2.MORPH_CLOSE, kernel, iterations=3)
    # 形态学开运算去除小白点
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=5)
    # 膨胀得到确定背景
    sure_bg = cv2.dilate(opening, kernel, iterations=2)
    # 腐蚀得到确定前景
    sure_fg = cv2.erode(opening, kernel, iterations=2)
    sure_fg = np.uint8(sure_fg)
    # 背景-前景得到未知区域
    unknown = cv2.subtract(sure_bg, sure_fg)
    # 连通域标记前景
    _, markers = cv2.connectedComponents(sure_fg)
    # 所有标记+1，背景为1
    markers = markers + 1
    # 未知区域标记为0
    markers[unknown == 255] = 0
    # 分水岭算法分割（需3通道BGR图像）
    markers = cv2.watershed(image_for_ws, markers)
    # 创建全零的label图像
    label_img = np.zeros_like(markers, dtype=np.int32)
    # The int32 watershed boundary is -1. Marker 255 is a regular region ID.
    foreground = markers > 1
    _, compact_labels = np.unique(markers[foreground], return_inverse=True)
    label_img[foreground] = compact_labels + 1
    return label_img


def filter_regions(
        label_image,
        area_threshold_low=600,
        area_threshold_up=10000,
        circularity_threshold=0.3
):
    """
    区域筛选函数：根据面积和圆度筛选标签图像中的目标区域。
    参数：
        label_image: 输入标签图像（int类型，0为背景，其他为区域标签）
        area_threshold_low: 最小面积
        area_threshold_up: 最大面积
        circularity_threshold: 最小圆度
    返回：
        filtered_label_image: 筛选后的标签图像
    """
    # skimage要求标签为int类型
    label_image = label_image.astype(np.int32)
    props = regionprops(label_image)
    filtered_label_image = np.zeros_like(label_image)
    for prop in props:
        area = prop.area
        perimeter = prop.perimeter
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if area_threshold_low <= area <= area_threshold_up and circularity >= circularity_threshold:
            filtered_label_image[label_image == prop.label] = prop.label
    return filtered_label_image


def calculate_iou_mask(mask1, mask2):
    """
    用掩码高效计算IoU。
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def region_mask(region, shape):
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(region.coords.T)] = True
    return mask


def bbox_overlap(bbox1, bbox2):
    return not (bbox1[2] <= bbox2[0] or bbox1[0] >= bbox2[2] or bbox1[3] <= bbox2[1] or bbox1[1] >= bbox2[3])


def _check_label_image(label_img, path=None, max_label_count=1000):
    """
    检查输入是否为标签图像（单通道、整数类型、有限标签数）。
    若不合格则抛出异常。
    """
    if label_img.ndim != 2:
        raise ValueError(f"{path or ''} 不是单通道标签图像 (ndim={label_img.ndim})")
    if not np.issubdtype(label_img.dtype, np.integer):
        raise ValueError(f"{path or ''} 标签图像数据类型应为整数，实际为{label_img.dtype}")
    unique_vals = np.unique(label_img)
    if len(unique_vals) > max_label_count:
        raise ValueError(f"{path or ''} 标签种类过多（{len(unique_vals)}），可能不是标签图像")
    return True


def analyze_layer_iou_by_paths(
        label_image_paths,
        layer_index,
        weight_vec=None
):
    """
    计算指定层每个区域与前后各三层的最大IoU加权和。
    输入：
        label_image_paths: 标签图像路径列表（list of str）
        layer_index: 要分析的图片编号（int）
        weight_vec: 前后各层权重（list, 长度为3），默认[0.5, 0.3, 0.2]
    输出：
        max_iou_for_current_layer: list，每个区域的IoU指标
    """
    if weight_vec is None:
        weight_vec = [0.5, 0.3, 0.2]
    if layer_index < 0 or layer_index >= len(label_image_paths):
        raise ValueError(f"layer_index超出范围，应在0~{len(label_image_paths)-1}之间")
    # 只读取需要的图像和regionprops
    region_props = []
    imgs = []
    for i, path in enumerate(label_image_paths):
        if abs(i - layer_index) <= 3:
            img = read_labels(path)
            _check_label_image(img, path)
            region_props.append(regionprops(img))
            imgs.append(img)
        else:
            region_props.append(None)
            imgs.append(None)
    current_regions = region_props[layer_index]
    img_shape = imgs[layer_index].shape
    # 预生成当前层所有区域掩码和bbox
    current_masks = [region_mask(r, img_shape) for r in current_regions]
    current_bboxes = [r.bbox for r in current_regions]
    max_iou_for_current_layer = []
    for idx, region in enumerate(current_regions):
        mask1 = current_masks[idx]
        bbox1 = current_bboxes[idx]
        iou_sum = 0
        for i in range(1, 4):
            prev_index = layer_index - i
            next_index = layer_index + i
            prev_max_iou = 0
            next_max_iou = 0
            # 前一层
            if prev_index >= 0 and region_props[prev_index] is not None:
                for j, prev_region in enumerate(region_props[prev_index]):
                    if not bbox_overlap(bbox1, prev_region.bbox):
                        continue
                    mask2 = region_mask(prev_region, img_shape)
                    iou = calculate_iou_mask(mask1, mask2)
                    if iou > prev_max_iou:
                        prev_max_iou = iou
            # 后一层
            if next_index < len(label_image_paths) and region_props[next_index] is not None:
                for j, next_region in enumerate(region_props[next_index]):
                    if not bbox_overlap(bbox1, next_region.bbox):
                        continue
                    mask2 = region_mask(next_region, img_shape)
                    iou = calculate_iou_mask(mask1, mask2)
                    if iou > next_max_iou:
                        next_max_iou = iou
            iou_sum += weight_vec[i-1] * max(prev_max_iou, next_max_iou)
        max_iou_for_current_layer.append(iou_sum)
    return max_iou_for_current_layer


def batch_process(paths, operation, *, progress, check_interrupted):
    if len({os.path.normcase(Path(path).stem) for path in paths}) != len(paths):
        raise ValueError("选中的输入文件会生成同名结果，请分开保存。")
    saved, errors = [], []
    for index, path in enumerate(paths, 1):
        check_interrupted()
        try:
            saved.append(operation(index - 1, Path(path)))
        except (OSError, ValueError, RuntimeError, ImportError, cv2.error) as error:
            errors.append(f"{Path(path).name}：{error}")
        progress(index)
    return saved, errors


def segment_image(path, output_dir):
    image = read_image(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法读取图像")
    target = output_dir / f"{path.stem}_mask.png"
    write_labels(target, kmeans_watershed_segmentation(image))
    return target


def filter_image(path, output_dir, low, high, circularity):
    result = filter_regions(read_labels(path), low, high, circularity)
    target = output_dir / f"{path.stem}_filtered.png"
    write_labels(target, result)
    return target


def iou_score_map(original_labels, visible_labels, scores):
    # Filtering and thumbnail sizing can remove labels. Associate scores by the
    # original label identifiers so subsequent labels do not inherit them.
    ids = np.unique(original_labels)
    result = np.zeros(visible_labels.shape, dtype=np.float32)
    for label, score in zip(ids[ids > 0], scores):
        result[visible_labels == label] = score
    return result


def iou_colors(score_map):
    levels = np.rint(np.clip(score_map, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(levels, cv2.COLORMAP_VIRIDIS)


def iou_preview(paths, index, source_paths, max_side, *, label_filter=None, minimum=0):
    labels = read_labels(paths[index])
    visible = labels if label_filter is None else filter_regions(labels, *label_filter)
    scores = analyze_layer_iou_by_paths(paths, index)
    source = read_image(source_paths[index], cv2.IMREAD_COLOR) if index < len(source_paths) else None
    height, width = source.shape[:2] if source is not None else labels.shape
    size, _ = preview_geometry((height, width), max_side)
    if source is None:
        thumb = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    else:
        thumb = cv2.resize(source, size, interpolation=cv2.INTER_LINEAR)
    thumbnail_labels = cv2.resize(visible, size, interpolation=cv2.INTER_NEAREST)
    score_map = iou_score_map(labels, thumbnail_labels, scores)
    colors = iou_colors(score_map)
    if label_filter is None:
        return score_map, thumbnail_labels, thumb, colors
    mask = (score_map >= minimum) & (thumbnail_labels > 0)
    thumb[mask] = colors[mask]
    return visible, thumb


def save_iou(index, path, output_dir, paths, source_paths, max_side, limits, minimum):
    filtered, overlay = iou_preview(
        paths, index, source_paths, max_side, label_filter=limits, minimum=minimum,
    )
    ids = np.unique(filtered)
    relabeled = np.zeros(filtered.shape, dtype=filtered.dtype)
    for number, label in enumerate(ids[ids > 0], 1):
        relabeled[filtered == label] = number
    write_labels(output_dir / f"{path.stem}_filtered_relabel.png", relabeled)
    target = output_dir / f"{path.stem}_iou_analysis.png"
    write_image(target, overlay)
    return target
