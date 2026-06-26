import glob
import os

import cv2
import imageio.v3 as iio
import napari
import numpy as np
from napari.utils.notifications import show_info, show_warning
from natsort import natsorted

# ====== 常量 ======
IMG_GLOB = ("*.bmp", "*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")
LABEL_GLOB = ("*.png", "*.tif", "*.tiff")
IMG_FILTER = "Images (*.bmp *.tif *.tiff *.png *.jpg *.jpeg);;All files (*.*)"
RENDERING_MODES = ["mip", "attenuated_mip", "minip", "translucent", "iso", "average", "additive"]


# ====== 通用辅助 ======
def files_from_dir(folder, patterns=IMG_GLOB):
    """从文件夹 glob 收集文件并自然排序，无文件时报错。"""
    files = []
    for p in patterns:
        files += glob.glob(os.path.join(folder, p))
    files = natsorted(files)
    if not files:
        raise FileNotFoundError(f"在 {folder} 中找不到匹配 {patterns} 的文件")
    return files


def _resize(img, downsample, interp):
    """按 downsample 倍率缩小图像；downsample<=1 时原样返回。"""
    if downsample > 1:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w / downsample), int(h / downsample)), interpolation=interp)
    return img


def _to_cvt_dtype(img):
    """cv2.cvtColor 只接受 uint8/float32，其余类型转 uint8。"""
    if img.dtype not in (np.uint8, np.float32):
        img = img.astype(np.uint8)
    return img


# ====== 单张图像读取 ======
def _read_as_rgb(path, downsample, interp=cv2.INTER_AREA):
    """读取单张图像，统一为 3 通道 RGB uint8。"""
    img = iio.imread(path)
    if img.ndim == 2:  # 灰度
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 1:
        img = np.concatenate([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 3:
        pass
    elif img.ndim == 3 and img.shape[-1] > 3:
        img = img[..., :3]
    else:
        raise ValueError(f"不支持的图像形状: {img.shape}（文件 {path}）")
    return _resize(img, downsample, interp)


def _read_as_label(path, downsample, interp=cv2.INTER_NEAREST):
    """读取单张标签图，统一为单通道整数。"""
    img = iio.imread(path)
    if img.ndim == 3:
        img = img[..., 0] if img.shape[-1] == 1 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if not np.issubdtype(img.dtype, np.integer):
        img = img.astype(np.uint16)
    return _resize(img, downsample, interp)


# ====== 堆叠加载（接受文件路径列表） ======
def load_img_stack(files, downsample=1.0, interp=cv2.INTER_AREA,
                   pad_z=0, pad_h=0, pad_w=0):
    """
    从文件路径列表读取图像序列，堆叠为 (Z, H, W, 3) 彩色数组并可选补白。
    """
    if isinstance(files, str):
        raise TypeError("files 应为路径列表，不是字符串。若需从文件夹加载请用 files_from_dir()")
    stack = [_read_as_rgb(f, downsample, interp) for f in files]
    arr = np.stack(stack, axis=0)
    if pad_z > 0 or pad_h > 0 or pad_w > 0:
        arr = np.pad(arr, ((pad_z, pad_z), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                     mode="constant", constant_values=255)
    return arr


def load_lbl_stack(files, downsample=1.0, interp=cv2.INTER_NEAREST):
    """
    从文件路径列表读取标签图像序列，堆叠为 (Z, H, W) 整数数组。
    """
    if isinstance(files, str):
        raise TypeError("files 应为路径列表，不是字符串。若需从文件夹加载请用 files_from_dir()")
    stack = [_read_as_label(f, downsample, interp) for f in files]
    arr = np.stack(stack, axis=0)
    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.uint16)
    return arr


# ====== 加入 viewer ======
def add_rgb_img(viewer, rgb_stack, spacing, opacity, rendering,
                name_prefix="Image", transparent_zero=False):
    """将彩色 stack 的 R/G/B 三通道分开加入 napari，additive 叠加。"""
    if rgb_stack.ndim != 4 or rgb_stack.shape[-1] != 3:
        raise ValueError("rgb_stack 必须是形状为 (Z, H, W, 3) 的 4D 数组")

    colormaps = ["red", "green", "blue"]
    channel_names = ["Red", "Green", "Blue"]
    for i in range(3):
        channel = rgb_stack[..., i].astype(np.float32)
        if transparent_zero:
            channel[channel == 0] = np.nan
        viewer.add_image(
            channel,
            scale=spacing,
            name=f"{name_prefix}-{channel_names[i]}",
            colormap=colormaps[i],
            blending="additive",
            opacity=opacity,
            rendering=rendering,
        )


def add_gray_img(viewer, gray_stack, spacing, opacity, rendering,
                 name="Gray", transparent_zero=False):
    """将灰度或彩色 stack 加入 napari；彩色输入自动用 OpenCV 转灰度。"""
    if gray_stack.ndim == 4:
        if gray_stack.shape[-1] == 1:
            gray_stack = np.squeeze(gray_stack, axis=-1)
        elif gray_stack.shape[-1] == 3:
            gray_stack = np.stack(
                [cv2.cvtColor(_to_cvt_dtype(gray_stack[i]), cv2.COLOR_RGB2GRAY)
                 for i in range(gray_stack.shape[0])],
                axis=0,
            )
        else:
            raise ValueError("4D 输入的 shape[-1] 必须为 1 或 3")
    if gray_stack.ndim != 3:
        raise ValueError("gray_stack 必须是 (Z,H,W) / (Z,H,W,1) / (Z,H,W,3)")

    gray = gray_stack.astype(np.float32)
    if transparent_zero:
        gray[gray == 0] = np.nan
    viewer.add_image(
        gray,
        scale=spacing,
        name=name,
        colormap="gray",
        blending="additive",
        opacity=opacity,
        rendering=rendering,
    )


def add_lbl(viewer, label_stack, spacing, name="Label"):
    """将标签 stack 加入 napari Label 层。"""
    if label_stack.ndim == 4 and label_stack.shape[-1] == 1:
        label_stack = label_stack.squeeze(-1)
    if label_stack.ndim != 3:
        raise ValueError("label_stack 必须是 (Z,H,W) 或 (Z,H,W,1)")
    if not np.issubdtype(label_stack.dtype, np.integer):
        label_stack = label_stack.astype(np.uint16)
    viewer.add_labels(label_stack, scale=spacing, name=name)


# ====== 核心：把文件列表加载为图层并加入 viewer ======
def add_files_to_viewer(viewer, files, layer_kind="image", color="rgb",
                         spacing=(1.0, 1.0, 1.0), downsample=1.0,
                         rendering="mip", opacity=1.0, name=None):
    """
    将一组图像文件加载为图层并加入 viewer。

    Args:
        viewer: napari.Viewer
        files: 图像文件路径列表
        layer_kind: 'image' / 'label' / 'cell'（cell 会将背景置透明）
        color: 'rgb'（三通道分开）或 'gray'（灰度）
        spacing: (z, y, x) 物理间距
        downsample: 下采样倍率，1.0 为原始尺寸
        rendering: 渲染模式，如 'mip' / 'attenuated_mip' / 'iso' 等
        opacity: 图层不透明度 0~1
        name: 图层名，默认取首文件所在文件夹名
    """
    spacing = [float(s) for s in spacing]
    if name is None:
        name = os.path.basename(os.path.dirname(files[0])) or layer_kind

    if layer_kind == "label":
        lbl_stack = load_lbl_stack(files, downsample=downsample)
        add_lbl(viewer, lbl_stack, spacing, name=f"Label-{name}")
        return

    transparent = layer_kind == "cell"
    img_stack = load_img_stack(files, downsample=downsample)
    if color == "rgb":
        add_rgb_img(viewer, img_stack, spacing, opacity, rendering,
                    name_prefix=name, transparent_zero=transparent)
    else:
        add_gray_img(viewer, img_stack, spacing, opacity, rendering,
                     name=f"{name}-Gray", transparent_zero=transparent)


# ====== napari 内嵌控件面板 ======
def build_control_panel(viewer):
    """构造 magicgui 控件面板：多选文件 + 设参数 + 点按钮加入视图。"""
    from magicgui.widgets import (
        Container, FileEdit, ComboBox, FloatSpinBox, FloatSlider, PushButton,
    )

    kind_map = {"图像": "image", "标签": "label", "细胞": "cell"}

    file_edit = FileEdit(
        mode="rm",
        filter=IMG_FILTER,
        label="选择图像文件（可多选）",
    )
    kind_combo = ComboBox(
        choices=list(kind_map.keys()),
        label="图层类型",
        value="图像",
    )
    color_combo = ComboBox(
        choices=["rgb", "gray"],
        label="颜色模式",
        value="rgb",
    )
    rendering_combo = ComboBox(
        choices=RENDERING_MODES,
        label="渲染模式",
        value="mip",
    )
    downsample_spin = FloatSpinBox(
        label="下采样倍率",
        value=1.0,
        min=1.0,
        max=32.0,
        step=0.5,
    )
    space_z = FloatSpinBox(label="间距 Z", value=1.0, min=0.001, step=0.1)
    space_y = FloatSpinBox(label="间距 Y", value=1.0, min=0.001, step=0.1)
    space_x = FloatSpinBox(label="间距 X", value=1.0, min=0.001, step=0.1)
    opacity_slider = FloatSlider(label="不透明度", value=1.0, min=0.0, max=1.0)
    add_btn = PushButton(label="添加到视图")

    @add_btn.clicked.connect
    def _on_add():
        paths = file_edit.value
        if not paths:
            show_warning("请先选择图像文件")
            return
        # mode="rm" 返回 tuple[Path, ...]
        files = natsorted([str(p) for p in paths])
        kind = kind_map[kind_combo.value]
        try:
            add_files_to_viewer(
                viewer, files,
                layer_kind=kind,
                color=color_combo.value,
                spacing=(space_z.value, space_y.value, space_x.value),
                downsample=downsample_spin.value,
                rendering=rendering_combo.value,
                opacity=opacity_slider.value,
            )
            label = os.path.basename(os.path.dirname(files[0])) or kind
            show_info(f"已加入 {len(files)} 个文件 — {label}")
        except Exception as e:  # noqa: BLE001
            show_warning(f"加载失败: {e}")

    return Container(
        widgets=[
            file_edit,
            kind_combo,
            color_combo,
            rendering_combo,
            downsample_spin,
            space_z,
            space_y,
            space_x,
            opacity_slider,
            add_btn,
        ],
        labels=True,
    )


# ====== 批量接口（通过文件夹 glob，仍可脚本化调用）======
def rendering_with_label(img_path=(), label_path=(), cell_path=(),
                         spacing_z=1.0, spacing_y=1.0, spacing_x=1.0,
                         downsample=1.0, rendering="mip", color="rgb"):
    """一次性从多个文件夹 glob 加载图像并打开 napari。"""
    spacing = (float(spacing_z), float(spacing_y), float(spacing_x))
    viewer = napari.Viewer(ndisplay=3)

    for folder in img_path:
        files = files_from_dir(folder, IMG_GLOB)
        add_files_to_viewer(viewer, files, "image", color, spacing, downsample, rendering)
    for folder in label_path:
        files = files_from_dir(folder, LABEL_GLOB)
        add_files_to_viewer(viewer, files, "label", color, spacing, downsample, rendering)
    for folder in cell_path:
        files = files_from_dir(folder, IMG_GLOB)
        add_files_to_viewer(viewer, files, "cell", color, spacing, downsample, rendering)

    viewer.window.add_dock_widget(build_control_panel(viewer), area="right", name="体渲染加载器")
    napari.run()


def main():
    """打开空的 3D napari 窗口，右侧停靠控件面板，由用户交互加载。"""
    viewer = napari.Viewer(ndisplay=3)
    viewer.window.add_dock_widget(build_control_panel(viewer), area="right", name="体渲染加载器")
    napari.run()


if __name__ == "__main__":
    main()
