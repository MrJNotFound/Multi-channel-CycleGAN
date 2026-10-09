"""Explicit model composition registry; no network or optimizer code lives here.

Canonical IDs describe architecture / input modalities / fusion / extra loss.
Only registered compositions are accepted. RGB channels are not input modalities.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelSpec:
    key: str
    architecture: str
    modalities: str
    fusion: str
    extra_loss: str
    implementation: str
    adapter: str = "translation"
    defaults: Mapping[str, Any] = field(default_factory=dict)

    @property
    def components(self):
        return {name: getattr(self, name) for name in COMPONENT_LABELS}


COMPONENT_LABELS = {
    "architecture": {"cycle_gan": "CycleGAN · 双向循环", "cut": "CUT · 单向对比", "pix2pix": "Pix2Pix · 配对监督"},
    "modalities": {"single": "单类输入（灰度或 RGB）", "dual": "双类输入（BF + AF）"},
    "fusion": {"none": "不适用（单类）", "stack": "Stack · 通道堆叠", "fusion": "Fusion · 双分支特征融合"},
    "extra_loss": {"base": "无额外损失", "structure": "结构保持 · 梯度 + 强度（SPIF）", "saliency": "阈值内容保持（UTOM）"},
    "adapter": {"translation": "常规图像转换", "colorization": "Lab 着色"},
}
COMPONENT_TITLES = {
    "architecture": "① 模型架构", "modalities": "② 输入图像类别",
    "fusion": "③ 跨类融合策略", "extra_loss": "④ 额外损失",
    "adapter": "任务适配（不改变上述分类）",
}


def _spec(architecture, modalities, fusion, extra_loss, implementation,
          adapter="translation", **defaults):
    key = f"{architecture}_{modalities}_{fusion}_{extra_loss}"
    if adapter != "translation":
        key += f"_{adapter}"
    return ModelSpec(key, architecture, modalities, fusion, extra_loss,
                     implementation, adapter, defaults)


_SPECS = (
    _spec("cycle_gan", "single", "none", "base", "cycle_gan", input_nc=3),
    _spec("cycle_gan", "dual", "stack", "base", "cycle_gan", input_nc=2, lambda_identity=0),
    _spec("cycle_gan", "single", "none", "structure", "structure", input_nc=1, lambda_identity=0),
    _spec("cycle_gan", "dual", "stack", "structure", "structure", input_nc=2, lambda_identity=0),
    _spec("cycle_gan", "dual", "fusion", "structure", "structure", input_nc=2, lambda_identity=0),
    # The same fusion generator with optional structure guidance disabled.
    _spec("cycle_gan", "dual", "fusion", "base", "structure", input_nc=2, lambda_identity=0, lambda_content=0),
    _spec("cycle_gan", "single", "none", "saliency", "utom", input_nc=3),
    _spec("cycle_gan", "dual", "stack", "saliency", "utom", input_nc=2, lambda_identity=0),
    _spec("cut", "single", "none", "base", "cut", input_nc=3),
    _spec("cut", "dual", "stack", "base", "cut", input_nc=2, nce_idt=False),
    _spec("pix2pix", "single", "none", "base", "pix2pix", input_nc=3),
    _spec("pix2pix", "single", "none", "base", "colorization", adapter="colorization", input_nc=1, output_nc=2),
)
MODEL_SPECS = {spec.key: spec for spec in _SPECS}

# Runtime implementations, not command-line aliases.
IMPLEMENTATIONS = {
    "cycle_gan": ("models.cycle_gan_model", "CycleGANModel"),
    "structure": ("models.cycle_gan_structure_model", "SPIFModel"),
    "utom": ("models.cycle_gan_saliency_model", "UTOMModel"),
    "cut": ("models.cut_model", "CUTModel"),
    "pix2pix": ("models.pix2pix_model", "Pix2PixModel"),
    "colorization": ("models.pix2pix_colorization_model", "ColorizationModel"),
}


def model_spec(name):
    try:
        return MODEL_SPECS[name]
    except KeyError:
        raise ValueError(f"未知模型组合 {name!r}，请使用注册表中的完整组合名称。") from None


def generator_names(name, values):
    """One architecture decision shared by training and inference."""
    spec = model_spec(name)
    backbone = values.get("netG", "resnet_9blocks")
    if spec.fusion != "fusion":
        return backbone, backbone
    version = values.get("spif_version", "v2")
    if version not in {"v1", "v2"}:
        raise ValueError("SPIF 版本仅支持 v1 / v2。")
    if backbone not in {"resnet_6blocks", "resnet_9blocks"}:
        raise ValueError("SPIF Fusion 需要 ResNet 6/9 blocks。")
    blocks = backbone.removeprefix("resnet_")
    return f"spif_{version}_{blocks}", f"spif_dual_{version}_{blocks}"


def composition_candidates(filters=None, names=None):
    filters = filters or {}
    return [spec for spec in _SPECS
            if (names is None or spec.key in names)
            and all(getattr(spec, key) == value for key, value in filters.items())]


def composition_errors(name, values):
    """Validate the current composition contract in both CLI and GUI."""
    spec = MODEL_SPECS.get(name)
    if not spec:
        return [f"未知模型组合：{name}。"]
    errors = []
    if "spif_arch" in values:
        errors.append("spif_arch 已移除，请用 spif_version=v1/v2，Stack 由融合策略选择。")
    if spec.fusion != "fusion" and "spif_version" in values:
        errors.append("仅 Fusion 组合可选择 SPIF 版本。")
    try:
        nc = int(values.get("input_nc", spec.defaults.get("input_nc", 3)))
    except (TypeError, ValueError):
        return ["输入通道数必须是整数。"]
    if spec.modalities == "dual" and nc != 2:
        errors.append("双类 BF + AF 输入要求 input_nc=2（各一路灰度图）。")
    if spec.modalities == "single" and nc not in {1, 3}:
        errors.append("单类输入使用 input_nc=1（灰度）或 3（RGB）；双类输入请切换类别选择。")
    dataset = values.get("dataset_mode")
    if dataset == "dual_channel" and spec.modalities != "dual":
        errors.append("双类数据模块不能与单类模型组合。")
    if spec.modalities == "dual" and dataset not in {None, "dual_channel", "single"}:
        errors.append("双类组合需要 dual_channel 数据模块（推理也可用 single 双目录格式）。")
    if spec.fusion == "fusion":
        if values.get("netG", "resnet_9blocks") not in {"resnet_9blocks", "resnet_6blocks"}:
            errors.append("Fusion 目前只支持 ResNet 6/9 blocks；U-Net 不会执行双分支融合。")
        if values.get("spif_version", "v2") not in {"v1", "v2"}:
            errors.append("SPIF 版本请选择 v1（通道）或 v2（通道—空间）。")
    elif str(values.get("netG", "")).startswith("spif"):
        errors.append("此组合使用标准生成器；SPIF 双分支生成器请通过 Fusion 组合选择。")
    if spec.fusion == "fusion" and spec.extra_loss == "base":
        try:
            if float(values.get("lambda_content", 0)) != 0:
                errors.append("无额外损失要求 lambda_content=0；要启用请切换到结构保持。")
        except (TypeError, ValueError):
            errors.append("lambda_content 必须是数字。")
    if spec.adapter == "colorization":
        if nc != 1 or str(values.get("output_nc", 2)) != "2":
            errors.append("Lab 着色要求 input_nc=1、output_nc=2。")
        if dataset not in {None, "colorization"}:
            errors.append("Lab 着色需要 colorization 数据模块。")
    elif dataset == "colorization":
        errors.append("colorization 数据需要选择 Lab 着色任务适配。")
    if spec.architecture == "cut" and spec.modalities == "dual":
        if str(values.get("nce_idt", False)).lower() in {"true", "1", "yes"}:
            errors.append("CUT 双类输入不能开启恒等 NCE（nce_idt）。")
    return errors
