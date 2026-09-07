"""Explicit model composition registry; no network or optimizer code lives here.

Canonical IDs describe architecture / input modalities / fusion / extra loss.
Legacy names keep their original implementation and defaults. New IDs supply
composition defaults and reject contradictory parameters instead of silently
selecting a different generator. RGB channels are not input modalities.
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
    legacy_name: str
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


def _spec(architecture, modalities, fusion, extra_loss, legacy_name,
          adapter="translation", **defaults):
    key = f"{architecture}_{modalities}_{fusion}_{extra_loss}"
    if adapter != "translation":
        key += f"_{adapter}"
    return ModelSpec(key, architecture, modalities, fusion, extra_loss,
                     legacy_name, adapter, defaults)


_SPECS = (
    _spec("cycle_gan", "single", "none", "base", "cycle_gan", input_nc=3),
    _spec("cycle_gan", "dual", "stack", "base", "cycle_gan_stack", input_nc=2, lambda_identity=0),
    _spec("cycle_gan", "single", "none", "structure", "spif_single", input_nc=1, lambda_identity=0),
    _spec("cycle_gan", "dual", "stack", "structure", "spif_stack", input_nc=2, spif_arch="stack", lambda_identity=0),
    _spec("cycle_gan", "dual", "fusion", "structure", "spif", input_nc=2, spif_arch="v2", lambda_identity=0),
    # Existing SPIF fusion architecture with its optional content loss disabled.
    # Do not use the historical cycle_gan_fusion alias: it is actually stacked.
    _spec("cycle_gan", "dual", "fusion", "base", "spif", input_nc=2, spif_arch="v2", lambda_identity=0, lambda_content=0),
    _spec("cycle_gan", "single", "none", "saliency", "utom", input_nc=3),
    _spec("cycle_gan", "dual", "stack", "saliency", "utom_stack", input_nc=2, lambda_identity=0),
    _spec("cut", "single", "none", "base", "cut", input_nc=3),
    _spec("cut", "dual", "stack", "base", "cut_stack", input_nc=2, nce_idt=False),
    _spec("pix2pix", "single", "none", "base", "pix2pix", input_nc=3),
    _spec("pix2pix", "single", "none", "base", "colorization", adapter="colorization", input_nc=1, output_nc=2),
)
MODEL_SPECS = {spec.key: spec for spec in _SPECS}

# The old class names are intentionally retained for Python imports/checkpoints.
IMPLEMENTATIONS = {
    "cycle_gan": ("models.cycle_gan.standard", "CycleGANModel"),
    "cycle_gan_stack": ("models.cycle_gan.dual_stack", "CycleGANStackModel"),
    "cycle_gan_fusion": ("models.legacy.fusion_alias", "CycleGANFusionModel"),
    "spif": ("models.cycle_gan.structure", "SPIFModel"),
    "spif_single": ("models.cycle_gan.single_structure", "SPIFSingleModel"),
    "spif_stack": ("models.cycle_gan.dual_stack_structure", "SPIFStackModel"),
    "utom": ("models.cycle_gan.saliency", "UTOMModel"),
    "utom_stack": ("models.cycle_gan.dual_stack_saliency", "UTOMStackModel"),
    "cut": ("models.cut.standard", "CUTModel"),
    "cut_stack": ("models.cut.dual_stack", "CUTStackModel"),
    "sincut": ("models.cut.single_image", "SinCUTModel"),
    "pix2pix": ("models.pix2pix.standard", "Pix2PixModel"),
    "colorization": ("models.pix2pix.colorization", "ColorizationModel"),
    "test": ("models.runtime.inference", "TestModel"),
    "template": ("models.examples.template", "TemplateModel"),
}


def legacy_name(name):
    spec = MODEL_SPECS.get(name)
    return spec.legacy_name if spec else name


def composition_candidates(filters=None, names=None):
    filters = filters or {}
    return [spec for spec in _SPECS
            if (names is None or spec.key in names)
            and all(getattr(spec, key) == value for key, value in filters.items())]


def canonical_name(name, values=None):
    """Classify old saved options by actual behavior, not the historical label.

    Unsupported/ambiguous old configurations retain their legacy ID. The UI
    then displays a compatibility warning rather than silently changing weights.
    """
    if name in MODEL_SPECS:
        return name
    values = values or {}
    candidates = [spec for spec in _SPECS if spec.legacy_name == name]
    if not candidates and name != "cycle_gan_fusion":
        return name
    default_nc = candidates[0].defaults.get("input_nc", 3) if candidates else 2
    try:
        dual = int(values.get("input_nc", default_nc)) == 2
    except (ValueError, TypeError):
        return name
    modality = "dual" if dual else "single"
    if name in {"spif", "spif_single", "spif_stack"}:
        default_arch = "stack" if name == "spif_stack" else "v2"
        arch = values.get("spif_arch", default_arch)
        fusion = ("stack" if arch == "stack" else "fusion") if dual else "none"
        if fusion == "fusion" and values.get("netG", "resnet_9blocks") not in {"resnet_9blocks", "resnet_6blocks"}:
            return name
        extra = "base" if fusion == "fusion" and str(values.get("lambda_content", 1)) in {"0", "0.0"} else "structure"
        return f"cycle_gan_{modality}_{fusion}_{extra}"
    if name in {"cycle_gan", "cycle_gan_fusion", "utom"} and dual:
        # These classes use standard generators, including the misleading alias.
        # Keep the original implementation: stack subclasses have different hooks.
        return name
    if name in {"cycle_gan_stack", "utom_stack", "cut_stack"} and not dual:
        return name
    if name == "cut" and dual:
        return name
    proposed = candidates[0].key if candidates else name
    return name if composition_errors(proposed, values) else proposed


def composition_errors(name, values):
    """Validate canonical compositions in both CLI and GUI (legacy untouched)."""
    spec = MODEL_SPECS.get(name)
    if not spec:
        return []
    errors = []
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
        if values.get("spif_arch", "v2") not in {"v2", "legacy"}:
            errors.append("Fusion 请选择 v2 或 legacy；Stack 请在融合策略中选择。")
    elif str(values.get("netG", "")).startswith("spif"):
        errors.append("此组合使用标准生成器；SPIF 双分支生成器请通过 Fusion 组合选择。")
    if spec.fusion == "stack" and spec.extra_loss == "structure" and values.get("spif_arch", "stack") != "stack":
        errors.append("Stack + 结构保持要求 spif_arch=stack；Fusion 请切换融合策略。")
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
