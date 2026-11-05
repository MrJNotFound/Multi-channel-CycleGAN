import cv2
import numpy as np
from skimage import measure


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
    # 找出所有区域标签（排除背景1和分水岭线-1/255）
    unique_labels = np.unique(markers)
    # 只取大于1的值作为有效区域
    region_labels = [l for l in unique_labels if l > 1]
    # 依次赋予递增标签（从1开始）
    for idx, label in enumerate(region_labels, start=1):
        label_img[markers == label] = idx
    # 背景设为0，分水岭线（-1和255）也设为0
    label_img[markers == 1] = 0
    label_img[markers == -1] = 0
    label_img[markers == 255] = 0
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
    props = measure.regionprops(label_image)
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


