"""Module catalogue, option introspection, validation, and process helpers.

The GUI has only two executable workflows: training and inference. Models and
datasets are plugins selected through their existing ``modify_commandline_options``
hooks; no dedicated experiment script is mapped to a UI preset.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from models.registry import (
    COMPONENT_LABELS, MODEL_SPECS, composition_errors,
)


ROOT = Path(__file__).resolve().parent.parent


class NonExitingArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


@dataclass(frozen=True)
class TaskInfo:
    key: str
    label: str
    description: str
    script: str


TASKS = {
    "train": TaskInfo(
        "train",
        "训练",
        "按所选模型和数据模块启动统一训练循环。",
        "training/train.py",
    ),
    "inference": TaskInfo(
        "inference",
        "推理",
        "加载所选训练模块生成的生成器检查点并批量输出图像。",
        "inference/infer.py",
    ),
}


@dataclass(frozen=True)
class ModuleInfo:
    key: str
    label: str
    family: str
    description: str
    recommended_dataset: str
    recommendations: Mapping[str, Any] = field(default_factory=dict)
    featured: tuple[str, ...] = ()
    checkpoint_suffix: str = "_A"
    inference_dataset: str | None = None


def _module_info(spec) -> ModuleInfo:
    """UI guidance comes from composition dimensions, not historical aliases."""
    recommendations = {"output_nc": 3, **spec.defaults}
    if spec.architecture == "cycle_gan":
        featured = ("lambda_A", "lambda_B", "lambda_identity", "gan_mode", "pool_size")
        description = "双向循环一致性训练。"
    elif spec.architecture == "cut":
        recommendations.update(CUT_mode="CUT", no_dropout=True)
        featured = ("CUT_mode", "lambda_GAN", "lambda_NCE", "nce_idt", "nce_layers", "nce_T", "num_patches", "netF")
        description = "使用 PatchNCE 的单向对比学习，自动执行首批数据初始化。"
    else:
        featured = ("lambda_L1", "gan_mode", "netG", "norm")
        description = "配对监督图像转换。"
    if spec.extra_loss == "structure":
        featured += ("lambda_content", "lambda_sp", "lambda_intensity",
                     "sp_grad_decay_fraction", "sp_intensity_decay_fraction",
                     "intensity_pool_size", "grad_channel_weights")
        description += " 加入原尺寸梯度与粗粒度强度引导。"
        if spec.fusion == "fusion":
            recommendations.update(lambda_sp=10.0, lambda_intensity=0.5,
                                   sp_grad_decay_fraction=0.1, sp_intensity_decay_fraction=0.025,
                                   intensity_pool_size=8)
    elif spec.extra_loss == "saliency":
        recommendations["netG"] = "unet_256"
        featured += ("threshold_A", "threshold_B", "domain_A_tissue")
        description += " 加入 UTOM 阈值内容保持。"
    if spec.fusion == "fusion":
        description += " SPIF v1 为通道融合，v2 为通道—空间融合。"
        recommendations["spif_version"] = "v2"
    if spec.adapter == "colorization":
        recommendations.update(netG="unet_256", norm="batch")
        description = "Lab 色彩空间的 L → ab 配对着色。"
    dataset = ("colorization" if spec.adapter == "colorization" else
               "dual_channel" if spec.modalities == "dual" else
               "aligned" if spec.architecture == "pix2pix" else "unaligned")
    label = " / ".join(COMPONENT_LABELS[part][spec.components[part]]
                       for part in ("architecture", "modalities", "fusion", "extra_loss"))
    if spec.adapter == "colorization":
        label += " / Lab 着色"
    return ModuleInfo(spec.key, label, COMPONENT_LABELS["architecture"][spec.architecture],
                      description, dataset, recommendations, featured,
                      "_A" if spec.architecture == "cycle_gan" else "")


MODULES = {name: _module_info(spec) for name, spec in MODEL_SPECS.items()}


@dataclass(frozen=True)
class DatasetInfo:
    key: str
    label: str
    description: str
    layout_train: tuple[str, ...]
    layout_inference: tuple[str, ...]


DATASETS: dict[str, DatasetInfo] = {
    "dual_channel": DatasetInfo(
        "dual_channel",
        "双通道 BF + AF",
        "按文件名匹配 BF/AF，并对两路图像应用完全相同的空间变换。",
        ("trainA_BF/", "trainA_AF/", "trainB/"),
        ("testA_BF/", "testA_AF/"),
    ),
    "unaligned": DatasetInfo(
        "unaligned",
        "无配对 A / B",
        "从两个域独立采样，适合 CycleGAN、SPIF-Single、CUT 和 UTOM。",
        ("trainA/", "trainB/"),
        ("testA/", "testB/"),
    ),
    "aligned": DatasetInfo(
        "aligned",
        "配对左右拼接",
        "每张图像左半为 A、右半为 B，适合 Pix2Pix。",
        ("train/",),
        ("test/",),
    ),
    "single": DatasetInfo(
        "single",
        "单目录推理",
        "直接读取 dataroot 中的图像；双通道时读取 dataroot_BF 与 dataroot_AF。",
        (),
        ("<dataroot>/",),
    ),
    "colorization": DatasetInfo(
        "colorization",
        "着色数据集",
        "为 colorization 模型提供 Lab 色彩空间数据。",
        ("train/",),
        ("test/",),
    ),
}


@dataclass(frozen=True)
class OptionSpec:
    dest: str
    flag: str
    aliases: tuple[str, ...]
    default: Any
    help: str
    choices: tuple[Any, ...]
    required: bool
    boolean: bool
    store_false: bool
    nargs: Any
    value_type: Any
    section: str
    boolean_value: bool = False


AUTO_OPTIONS = {
    "counter",
    "data_size",
    "model",
    "dataset_mode",
    "source_model",
    "gui_events",
}

SECTION_ORDER = ("运行与设备", "数据与增强", "网络结构", "优化与损失", "保存与记录", "其他")

SECTION_FIELDS = {
    "运行与设备": {
        "dataroot",
        "name",
        "checkpoints_dir",
        "results_dir",
        "gpu_ids",
        "phase",
        "epoch",
        "load_iter",
        "pretrained_name",
        "continue_train",
        "source_model",
        "model_suffix",
    },
    "数据与增强": {
        "input_nc",
        "output_nc",
        "batch_size",
        "num_threads",
        "load_size",
        "crop_size",
        "preprocess",
        "direction",
        "serial_batches",
        "no_flip",
        "max_dataset_size",
        "trainA_normalize",
        "trainB_normalize",
        "num_test",
        "aspect_ratio",
    },
    "网络结构": {
        "netG",
        "netD",
        "ngf",
        "ndf",
        "n_layers_D",
        "norm",
        "normG",
        "normD",
        "no_dropout",
        "no_antialias",
        "no_antialias_up",
        "init_type",
        "init_gain",
        "spif_version",
    },
    "优化与损失": {
        "n_epochs",
        "n_epochs_decay",
        "epoch_count",
        "stop_after_epoch",
        "lr",
        "beta1",
        "beta2",
        "gan_mode",
        "pool_size",
        "lr_policy",
        "lr_decay_iters",
        "freeze_D_epochs",
    },
    "保存与记录": {
        "display_freq",
        "update_html_freq",
        "save_visuals_epoch_freq",
        "print_freq",
        "no_html",
        "save_latest_freq",
        "save_epoch_freq",
        "save_by_iter",
        "display_winsize",
        "use_wandb",
        "wandb_project_name",
        "save_inputs",
        "make_report",
        "output_suffix",
        "verbose",
    },
}


LABELS = {
    "dataroot": "数据路径",
    "name": "实验名称",
    "checkpoints_dir": "检查点根目录",
    "results_dir": "结果根目录",
    "gpu_ids": "计算设备",
    "phase": "数据阶段",
    "pretrained_name": "预训练实验名称",
    "continue_train": "续训当前实验",
    "source_model": "检查点来源模块",
    "model_suffix": "检查点网络后缀",
    "epoch": "加载 Epoch",
    "load_iter": "加载 Iteration",
    "input_nc": "输入通道",
    "output_nc": "输出通道",
    "batch_size": "批大小",
    "num_threads": "数据线程",
    "load_size": "载入尺寸",
    "crop_size": "裁剪尺寸",
    "preprocess": "图像预处理",
    "direction": "转换方向",
    "serial_batches": "顺序读取",
    "no_flip": "关闭随机翻转",
    "max_dataset_size": "最大样本数",
    "trainA_normalize": "A 域归一化值",
    "trainB_normalize": "B 域归一化值",
    "num_test": "最多推理图像",
    "eval": "使用评估模式",
    "aspect_ratio": "输出宽高比",
    "netG": "生成器",
    "netD": "判别器",
    "ngf": "生成器基础通道",
    "ndf": "判别器基础通道",
    "n_layers_D": "判别器层数",
    "norm": "归一化",
    "normG": "生成器归一化",
    "normD": "判别器归一化",
    "no_dropout": "关闭 Dropout",
    "no_antialias": "关闭下采样抗锯齿",
    "no_antialias_up": "关闭上采样抗锯齿",
    "init_type": "权重初始化",
    "init_gain": "初始化增益",
    "spif_version": "SPIF 版本（v1 / v2）",
    "n_epochs": "恒定学习率 Epoch",
    "n_epochs_decay": "衰减 Epoch",
    "epoch_count": "起始 Epoch",
    "stop_after_epoch": "本次停止 Epoch",
    "lr": "学习率",
    "beta1": "Adam β1",
    "beta2": "Adam β2",
    "gan_mode": "GAN 损失",
    "pool_size": "历史图像池",
    "lr_policy": "学习率策略",
    "lr_decay_iters": "学习率衰减间隔",
    "freeze_D_epochs": "冻结判别器 Epoch",
    "display_freq": "训练预览间隔",
    "update_html_freq": "HTML 更新间隔",
    "save_visuals_epoch_freq": "每 N Epoch 保存预览",
    "print_freq": "损失输出间隔",
    "no_html": "关闭训练 HTML",
    "save_latest_freq": "Latest 保存间隔",
    "save_epoch_freq": "Epoch 保存间隔",
    "save_by_iter": "按 Iteration 保存",
    "display_winsize": "预览图尺寸",
    "use_wandb": "启用 W&B",
    "wandb_project_name": "W&B 项目名",
    "save_inputs": "保存推理输入",
    "make_report": "生成推理 HTML 报告",
    "output_suffix": "输出文件后缀",
    "verbose": "打印网络详情",
    "gui_events": "输出 GUI 进度事件",
    "lambda_A": "A 循环损失",
    "lambda_B": "B 循环损失",
    "lambda_identity": "恒等映射损失",
    "lambda_content": "内容损失总权重",
    "lambda_sp": "细结构梯度损失",
    "lambda_intensity": "粗粒度强度损失",
    "sp_grad_decay_fraction": "梯度引导占比",
    "sp_intensity_decay_fraction": "强度引导占比",
    "intensity_pool_size": "强度网格大小",
    "grad_channel_weights": "输入通道梯度权重",
    "lambda_L1": "L1 损失",
    "CUT_mode": "CUT 模式",
    "lambda_GAN": "GAN 损失权重",
    "lambda_NCE": "PatchNCE 权重",
    "nce_idt": "恒等 NCE",
    "nce_layers": "NCE 特征层",
    "nce_T": "NCE 温度",
    "num_patches": "每层 Patch 数",
    "netF": "特征采样网络",
    "netF_nc": "特征网络通道",
    "threshold_A": "A 域阈值",
    "threshold_B": "B 域阈值",
    "domain_A_tissue": "A 域组织明暗",
}


ESSENTIAL_FIELDS = {
    "train": ("dataroot", "name", "checkpoints_dir", "gpu_ids"),
    "inference": (
        "dataroot",
        "name",
        "checkpoints_dir",
        "results_dir",
        "gpu_ids",
        "epoch",
        "load_iter",
    ),
}

RECOMMENDED_COMMON = {
    "train": (
        "input_nc",
        "output_nc",
        "batch_size",
        "load_size",
        "crop_size",
        "n_epochs",
        "n_epochs_decay",
        "lr",
        "save_epoch_freq",
    ),
    "inference": (
        "input_nc",
        "output_nc",
        "batch_size",
        "num_threads",
        "preprocess",
        "num_test",
        "model_suffix",
        "eval",
    ),
}


def module_info(name: str) -> ModuleInfo:
    return MODULES[name]


def recommended_dataset_for(task: str, model_module: str) -> str:
    """Return guidance without coupling either engine to a script preset."""
    info = module_info(model_module)
    if task == "train":
        return info.recommended_dataset
    if info.inference_dataset:
        return info.inference_dataset
    if info.recommended_dataset == "dual_channel":
        return "dual_channel"
    if info.recommended_dataset == "colorization":
        return "colorization"
    return "single"


def dataset_info(name: str) -> DatasetInfo:
    if name in DATASETS:
        return DATASETS[name]
    return DatasetInfo(
        name,
        name.replace("_", " ").title(),
        "由仓库动态发现的数据模块。",
        (),
        (),
    )


def option_label(dest: str) -> str:
    return LABELS.get(dest, dest.replace("_", " ").title())


def _section_for(dest: str, origin: str) -> str:
    for section, fields in SECTION_FIELDS.items():
        if dest in fields:
            return section
    if origin == "model":
        return "优化与损失"
    if origin == "dataset":
        return "数据与增强"
    return "其他"


def _safe_import_data():
    try:
        import data

        return data
    except (ImportError, ModuleNotFoundError):
        return None


def discover_models() -> list[str]:
    import models

    names = []
    candidates = list(MODEL_SPECS)
    for name in candidates:
        try:
            models.get_option_setter(name)
        except (ImportError, ModuleNotFoundError, NotImplementedError, AttributeError):
            continue
        names.append(name)
    return names


def model_has_capability(model_name: str, method_name: str) -> bool:
    """Inspect an optional model hook without constructing the network."""
    try:
        import models

        model_class = models.find_model_using_name(model_name)
    except (ImportError, ModuleNotFoundError, NotImplementedError, AttributeError):
        return False
    return callable(getattr(model_class, method_name, None))


def discover_datasets(task: str = "train") -> list[str]:
    data = _safe_import_data()
    names = []
    for path in sorted((ROOT / "data").glob("*_dataset.py")):
        name = path.stem.removesuffix("_dataset")
        if name in {"base", "template"}:
            continue
        if task == "train" and name == "single":
            continue
        if data is not None:
            try:
                data.get_option_setter(name)
            except (ImportError, ModuleNotFoundError, NotImplementedError, AttributeError):
                continue
        names.append(name)
    return names


def build_option_parser(
    task: str,
    model_module: str,
    dataset_mode: str,
    dynamic_values: Mapping[str, Any] | None = None,
) -> tuple[argparse.ArgumentParser, dict[str, str]]:
    from options.test_options import TestOptions
    from options.train_options import TrainOptions
    import models

    source = TrainOptions() if task == "train" else TestOptions()
    parser = NonExitingArgumentParser(
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser = source.initialize(parser)
    origins = {action.dest: "base" for action in parser._actions}

    parser_model = model_module if task == "train" else "test"
    original_argv = sys.argv
    sys.argv = [
        original_argv[0],
        "--dataroot",
        ".",
        "--model",
        parser_model,
        "--dataset_mode",
        dataset_mode,
    ]
    if task == "inference":
        sys.argv.extend(["--source_model", model_module])
    if dynamic_values and dynamic_values.get("CUT_mode"):
        sys.argv.extend(["--CUT_mode", str(dynamic_values["CUT_mode"])])
    before = {action.dest for action in parser._actions}
    try:
        parser = models.get_option_setter(parser_model)(parser, source.isTrain)
    finally:
        sys.argv = original_argv
    for action in parser._actions:
        if action.dest not in before:
            origins[action.dest] = "model"

    data = _safe_import_data()
    if data is not None:
        before = {action.dest for action in parser._actions}
        parser = data.get_option_setter(dataset_mode)(parser, source.isTrain)
        for action in parser._actions:
            if action.dest not in before:
                origins[action.dest] = "dataset"

    parser.set_defaults(model=parser_model, dataset_mode=dataset_mode)
    return parser, origins


def option_specs(
    task: str,
    model_module: str,
    dataset_mode: str,
    dynamic_values: Mapping[str, Any] | None = None,
) -> tuple[argparse.ArgumentParser, list[OptionSpec]]:
    parser, origins = build_option_parser(
        task, model_module, dataset_mode, dynamic_values
    )
    specs = []
    for action in parser._actions:
        if not action.option_strings or action.dest in AUTO_OPTIONS:
            continue
        long_flags = tuple(flag for flag in action.option_strings if flag.startswith("--"))
        flag = long_flags[0] if long_flags else action.option_strings[0]
        store_true = isinstance(action, argparse._StoreTrueAction)
        store_false = isinstance(action, argparse._StoreFalseAction)
        boolean_value = (
            isinstance(action.default, bool)
            and action.nargs == "?"
            and action.const is True
        )
        specs.append(
            OptionSpec(
                action.dest,
                flag,
                tuple(action.option_strings),
                action.default,
                action.help or "",
                tuple(action.choices or ()),
                bool(action.required),
                store_true or store_false or boolean_value,
                store_false,
                action.nargs,
                action.type,
                _section_for(action.dest, origins.get(action.dest, "base")),
                boolean_value,
            )
        )
    return parser, specs


def display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return str(value)


def coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def default_values(task: str, model_module: str, dataset_mode: str) -> dict[str, Any]:
    _, specs = option_specs(task, model_module, dataset_mode)
    values = {spec.dest: spec.default for spec in specs}
    values.update(module_info(model_module).recommendations)
    values["model"] = model_module if task == "train" else "test"
    values["source_model"] = model_module
    values["dataset_mode"] = dataset_mode
    values["dataroot"] = ""
    values["name"] = ""
    values["gpu_ids"] = "0"
    values["gui_events"] = True
    if task == "inference":
        values.update(
            {
                "phase": "test",
                "preprocess": "none",
                "serial_batches": True,
                "no_flip": True,
                "num_threads": 0,
                "batch_size": 1,
                "eval": True,
                "model_suffix": module_info(model_module).checkpoint_suffix,
            }
        )
    return values


def recommended_fields(task: str, model_module: str) -> tuple[str, ...]:
    seen = set(ESSENTIAL_FIELDS[task])
    result = []
    for dest in (*RECOMMENDED_COMMON[task], *module_info(model_module).featured):
        if dest not in seen and dest not in result:
            result.append(dest)
    return tuple(result)


def _same_as_default(raw: Any, default: Any) -> bool:
    if isinstance(raw, bool) or isinstance(default, bool):
        return coerce_bool(raw) == coerce_bool(default)
    if default is None:
        return str(raw).strip() == ""
    return str(raw).strip() == display_value(default)


def _split_multi(raw: str) -> list[str]:
    return shlex.split(raw.replace(",", " "), posix=False)


def build_cli_args(
    parser: argparse.ArgumentParser,
    specs: Iterable[OptionSpec],
    values: Mapping[str, Any],
) -> list[str]:
    args: list[str] = []
    for spec in specs:
        if spec.boolean:
            raw = values.get(spec.dest, spec.default)
            enabled = coerce_bool(raw)
            if spec.boolean_value:
                if enabled != coerce_bool(spec.default):
                    args.extend([spec.flag, str(enabled)])
            elif spec.store_false:
                if not enabled and bool(spec.default):
                    args.append(spec.flag)
            elif enabled and not bool(spec.default):
                args.append(spec.flag)
            continue

        raw = values.get(spec.dest, spec.default)
        text = display_value(raw).strip()
        if spec.required and not text:
            raise ValueError(f"{option_label(spec.dest)}不能为空。")
        if not text or _same_as_default(raw, spec.default):
            continue
        args.append(spec.flag)
        if spec.nargs in ("+", "*") or isinstance(spec.nargs, int):
            args.extend(_split_multi(text))
        else:
            args.append(text)

    try:
        parser.parse_args(args)
    except (SystemExit, argparse.ArgumentError, ValueError) as exc:
        raise ValueError(f"参数格式无效：{exc}") from exc
    return args


def build_command(
    task: str,
    model_module: str,
    dataset_mode: str,
    values: Mapping[str, Any],
    python_executable: str | None = None,
) -> list[str]:
    enriched = dict(values)
    parser_model = model_module if task == "train" else "test"
    enriched["model"] = parser_model
    enriched["source_model"] = model_module
    enriched["dataset_mode"] = dataset_mode
    errors = composition_errors(model_module, enriched)
    if errors:
        raise ValueError("\n".join(errors))
    parser, specs = option_specs(task, model_module, dataset_mode, enriched)
    args = build_cli_args(parser, specs, enriched)
    args.extend(["--model", parser_model, "--dataset_mode", dataset_mode])
    if task == "inference":
        args.extend(["--source_model", model_module])
    args.append("--gui_events")
    return [
        python_executable or sys.executable,
        str(ROOT / TASKS[task].script),
        *args,
    ]


def quote_command(command: Iterable[str]) -> str:
    return subprocess.list2cmdline(list(command))


def _saved_option_value(text: str) -> Any:
    lowered = text.lower()
    if lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"inf", "+inf"}:
        return float("inf")
    if lowered == "-inf":
        return float("-inf")
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def load_checkpoint_configuration(directory: Path) -> dict[str, Any]:
    """Read the options saved by BaseOptions from one experiment directory."""
    json_path = directory / "train_opt.json"
    text_path = directory / "train_opt.txt"
    if json_path.is_file():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"检查点配置不是 JSON 对象：{json_path}")
        return payload
    if not text_path.is_file():
        raise FileNotFoundError(
            f"未找到 train_opt.json 或 train_opt.txt：{directory}"
        )

    result: dict[str, Any] = {}
    for line in text_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("-") or ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        raw = raw.split("\t[default:", 1)[0].strip()
        if key:
            result[key] = _saved_option_value(raw)
    if not result:
        raise ValueError(f"无法从检查点配置中解析参数：{text_path}")
    return result


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors


def _resolved(path_value: Any) -> Path:
    path = Path(str(path_value)).expanduser()
    return path if path.is_absolute() else ROOT / path


def _required_dataset_dirs(task: str, dataset_mode: str, phase: str, root: Path) -> list[Path]:
    if dataset_mode == "dual_channel":
        result = [root / f"{phase}A_BF", root / f"{phase}A_AF"]
        if task == "train":
            result.append(root / f"{phase}B")
        return result
    if dataset_mode == "unaligned":
        return [root / f"{phase}A", root / f"{phase}B"]
    if dataset_mode in {"aligned", "colorization"}:
        return [root / phase]
    return []


def validate_configuration(
    task: str,
    model_module: str,
    dataset_mode: str,
    initialization: str,
    values: Mapping[str, Any],
) -> ValidationReport:
    errors: list[str] = composition_errors(model_module, {**values, "dataset_mode": dataset_mode})
    warnings: list[str] = []
    name = str(values.get("name", "")).strip()
    dataroot = str(values.get("dataroot", "")).strip()
    if not name:
        errors.append("请填写实验名称；它同时决定检查点目录。")
    if not dataroot:
        errors.append("请选择数据路径。")
        root = None
    else:
        root = _resolved(dataroot)

    info = module_info(model_module)
    recommended_dataset = recommended_dataset_for(task, model_module)
    if dataset_mode != recommended_dataset:
        warnings.append(
            f"{info.label} 在当前任务中通常搭配 "
            f"{dataset_info(recommended_dataset).label}；"
            f"当前选择为 {dataset_info(dataset_mode).label}。"
        )

    try:
        input_nc = int(str(values.get("input_nc", 3)))
        output_nc = int(str(values.get("output_nc", 3)))
        if input_nc <= 0 or output_nc <= 0:
            errors.append("输入、输出通道数必须大于 0。")
    except ValueError:
        input_nc = output_nc = -1
        errors.append("输入、输出通道数必须是整数。")
    if dataset_mode == "dual_channel" and input_nc != 2:
        errors.append("双通道数据模块要求 input_nc=2。")

    phase = str(values.get("phase", "train" if task == "train" else "test"))
    if root:
        if dataset_mode == "single" and input_nc == 2:
            # The repository's single dataset treats dataroot as a shared
            # prefix and reads sibling <dataroot>_BF / <dataroot>_AF folders.
            required_dirs = [Path(f"{root}_BF"), Path(f"{root}_AF")]
        elif root.is_dir():
            required_dirs = _required_dataset_dirs(
                task, dataset_mode, phase, root
            )
        else:
            required_dirs = []
            errors.append(f"数据路径不存在或不是文件夹：{root}")
        missing = [path for path in required_dirs if not path.is_dir()]
        if missing:
            errors.append("数据目录结构缺少：\n" + "\n".join(str(path) for path in missing))

    try:
        batch_size = int(str(values.get("batch_size", 1)))
        if batch_size <= 0:
            errors.append("批大小必须大于 0。")
    except ValueError:
        errors.append("批大小必须是整数。")
    try:
        load_size = int(str(values.get("load_size", 0)))
        crop_size = int(str(values.get("crop_size", 0)))
        if "crop" in str(values.get("preprocess", "")) and load_size < crop_size:
            errors.append("启用裁剪时，载入尺寸不能小于裁剪尺寸。")
    except ValueError:
        errors.append("载入尺寸和裁剪尺寸必须是整数。")

    gpu_ids = str(values.get("gpu_ids", "-1")).replace(" ", "")
    if not re.fullmatch(r"-1|\d+(?:,\d+)*", gpu_ids):
        errors.append("计算设备应填写 -1、0 或逗号分隔的 GPU 编号。")
    elif "," in gpu_ids:
        warnings.append(
            "桌面界面直接启动单进程；多个 GPU 编号不会自动创建 torchrun "
            "进程，请使用单个编号或复制命令后由分布式启动器执行。"
        )

    checkpoint_root = _resolved(values.get("checkpoints_dir", "./checkpoints"))
    suffix = str(values.get("epoch", "latest"))
    try:
        load_iter = int(str(values.get("load_iter", 0) or 0))
        if load_iter > 0:
            suffix = f"iter_{load_iter}"
    except ValueError:
        errors.append("加载 Iteration 必须是整数。")

    if task == "train":
        if initialization not in {"fresh", "resume", "finetune"}:
            errors.append("未知的训练初始化方式。")
        source_name = name
        if initialization == "finetune":
            source_name = str(values.get("pretrained_name", "")).strip()
            if not source_name:
                errors.append("微调模式需要填写预训练实验名称。")
        if initialization in {"resume", "finetune"} and source_name:
            source_dir = checkpoint_root / source_name
            if not source_dir.is_dir():
                errors.append(f"找不到权重目录：{source_dir}")
            elif not list(source_dir.glob(f"{suffix}_net_*.pth")):
                errors.append(f"目录中没有 {suffix} 检查点：{source_dir}")
        try:
            identity_weight = float(values.get("lambda_identity", 0) or 0)
        except (TypeError, ValueError):
            identity_weight = 0
            errors.append("恒等映射损失权重必须是数字。")
        if input_nc != output_nc and identity_weight > 0:
            warnings.append("输入输出通道不同，模型会自动关闭恒等映射损失。")
        if model_module == "cut_dual_stack_base" and coerce_bool(values.get("nce_idt")):
            errors.append("CUT-Stack 的输入输出通道不同，不能启用恒等 NCE。")

        for field_name in ("n_epochs", "n_epochs_decay"):
            try:
                if int(str(values.get(field_name, 0))) < 0:
                    errors.append(f"{option_label(field_name)}不能小于 0。")
            except (TypeError, ValueError):
                errors.append(f"{option_label(field_name)}必须是整数。")
        try:
            epoch_count = int(str(values.get("epoch_count", 1)))
            if epoch_count < 1:
                errors.append("起始 Epoch 必须大于等于 1。")
        except (TypeError, ValueError):
            epoch_count = 1
            errors.append("起始 Epoch 必须是整数。")
        try:
            save_epoch_freq = int(str(values.get("save_epoch_freq", 1)))
            if save_epoch_freq < 1:
                errors.append("Epoch 保存间隔必须大于等于 1。")
        except (TypeError, ValueError):
            errors.append("Epoch 保存间隔必须是整数。")
        stop_after = values.get("stop_after_epoch")
        if stop_after not in (None, ""):
            try:
                if int(str(stop_after)) < epoch_count:
                    errors.append("本次停止 Epoch 不能早于起始 Epoch。")
            except (TypeError, ValueError):
                errors.append("本次停止 Epoch 必须是整数。")
        try:
            freeze_epochs = int(str(values.get("freeze_D_epochs", 0)))
            if freeze_epochs < 0:
                errors.append("冻结判别器 Epoch 不能小于 0。")
            elif freeze_epochs and not model_has_capability(
                model_module, "set_discriminator_updates"
            ):
                errors.append(
                    f"{info.label} 没有公开判别器冻结能力，请将冻结 Epoch 设为 0。"
                )
        except (TypeError, ValueError):
            errors.append("冻结判别器 Epoch 必须是整数。")
        try:
            if float(str(values.get("lr", 0))) <= 0:
                errors.append("学习率必须大于 0。")
        except (TypeError, ValueError):
            errors.append("学习率必须是数字。")
    else:
        try:
            if int(str(values.get("num_test", 0))) <= 0:
                errors.append("最多推理图像必须大于 0。")
        except (TypeError, ValueError):
            errors.append("最多推理图像必须是整数。")
        model_suffix = str(values.get("model_suffix", ""))
        checkpoint = checkpoint_root / name / f"{suffix}_net_G{model_suffix}.pth"
        if name and not checkpoint.is_file():
            errors.append(f"找不到生成器检查点：{checkpoint}")

    return ValidationReport(tuple(errors), tuple(warnings))


def output_directory(task: str, values: Mapping[str, Any]) -> Path:
    name = str(values.get("name", "experiment_name"))
    if task == "train":
        return _resolved(values.get("checkpoints_dir", "./checkpoints")) / name
    base = _resolved(values.get("results_dir", "./results"))
    stage = f"{values.get('phase', 'test')}_{values.get('epoch', 'latest')}"
    try:
        load_iter = int(str(values.get("load_iter", 0) or 0))
    except ValueError:
        load_iter = 0
    if load_iter > 0:
        stage += f"_iter{load_iter}"
    return base / name / stage


@dataclass
class ProgressState:
    phase: str = "等待启动"
    current: int = 0
    total: int = 0
    epoch: int = 0
    final_epoch: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    output_dir: str = ""
    last_event: dict[str, Any] = field(default_factory=dict)

    @property
    def fraction(self) -> float:
        return 0.0 if self.total <= 0 else max(0.0, min(1.0, self.current / self.total))


class ProgressParser:
    EVENT_PREFIX = "@@MLCGUI "
    DATASET_RE = re.compile(r"(?:training images|dataset size)\s*=\s*(\d+)", re.I)
    LOSS_RE = re.compile(r"\(epoch:\s*(\d+),\s*iters:\s*(\d+),[^)]*\)\s*(.*)", re.I)
    METRIC_RE = re.compile(
        r",?\s*([A-Za-z][A-Za-z0-9_]*)\s*:\s*"
        r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)",
        re.I,
    )
    EPOCH_RE = re.compile(r"End of epoch\s+(\d+)\s*/\s*(\d+)", re.I)
    INFER_RE = re.compile(r"Processed\s+(\d+)\s*/\s*(\d+)\s+images", re.I)

    def __init__(self, task: str):
        self.task = task
        self.state = ProgressState()
        self.dataset_size = 0

    def feed(self, line: str) -> ProgressState:
        if line.startswith(self.EVENT_PREFIX):
            try:
                event = json.loads(line[len(self.EVENT_PREFIX) :])
            except json.JSONDecodeError:
                return self.state
            self._apply_event(event)
            return self.state

        dataset = self.DATASET_RE.search(line)
        if dataset:
            self.dataset_size = int(dataset.group(1))
        loss = self.LOSS_RE.search(line)
        if loss:
            self.state.epoch = int(loss.group(1))
            epoch_iter = int(loss.group(2))
            if self.dataset_size and self.state.final_epoch:
                self.state.total = self.dataset_size * self.state.final_epoch
                self.state.current = (
                    (self.state.epoch - 1) * self.dataset_size
                    + min(epoch_iter, self.dataset_size)
                )
            self.state.metrics = {
                key: float(value) for key, value in self.METRIC_RE.findall(loss.group(3))
            }
        epoch = self.EPOCH_RE.search(line)
        if epoch:
            self.state.epoch = int(epoch.group(1))
            self.state.final_epoch = int(epoch.group(2))
            self.state.phase = f"完成 Epoch {self.state.epoch}/{self.state.final_epoch}"
        inference = self.INFER_RE.search(line)
        if inference:
            self.state.current = int(inference.group(1))
            self.state.total = int(inference.group(2))
            self.state.phase = f"推理 {self.state.current}/{self.state.total}"
        return self.state

    def _apply_event(self, event: dict[str, Any]) -> None:
        self.state.last_event = event
        event_type = event.get("type")
        if event_type == "run_started":
            self.state.output_dir = str(event.get("output_dir", ""))
            if event.get("task") == "train":
                self.dataset_size = int(event.get("dataset_size", 0))
                self.state.epoch = int(event.get("start_epoch", 1))
                self.state.final_epoch = int(event.get("final_epoch", 0))
                self.state.total = self.dataset_size * self.state.final_epoch
                self.state.current = self.dataset_size * max(self.state.epoch - 1, 0)
                self.state.phase = "模型与数据已就绪"
            else:
                self.state.total = int(event.get("total", 0))
                self.state.phase = "模型已加载"
        elif event_type == "progress":
            if event.get("task") == "train":
                epoch = int(event.get("epoch", 1))
                epoch_iter = int(event.get("epoch_iter", 0))
                size = int(event.get("dataset_size", self.dataset_size))
                self.state.epoch = epoch
                self.state.final_epoch = int(event.get("final_epoch", self.state.final_epoch))
                self.state.current = (epoch - 1) * size + epoch_iter
                self.state.total = size * self.state.final_epoch
                self.state.phase = f"训练 Epoch {epoch}/{self.state.final_epoch}"
            else:
                self.state.current = int(event.get("current", 0))
                self.state.total = int(event.get("total", 0))
                self.state.phase = f"推理 {self.state.current}/{self.state.total}"
        elif event_type == "metrics":
            self.state.metrics = {
                str(key): float(value)
                for key, value in dict(event.get("values", {})).items()
            }
        elif event_type == "epoch_end":
            self.state.epoch = int(event.get("epoch", self.state.epoch))
            self.state.phase = f"完成 Epoch {self.state.epoch}/{self.state.final_epoch}"
        elif event_type == "checkpoint":
            self.state.phase = f"已保存检查点 {event.get('suffix', '')}"
        elif event_type == "run_finished":
            self.state.current = self.state.total
            self.state.phase = "运行完成"
