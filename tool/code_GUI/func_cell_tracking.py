import cv2
import numpy as np
from skimage.measure import regionprops


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
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"无法读取图像: {path}")
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
