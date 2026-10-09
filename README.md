# Multi-layer CycleGAN — Virtual H&E Staining

基于 CycleGAN 的多层虚拟 H&E 染色项目，支持双通道（明场 BF + 自发荧光 AF）输入到 RGB H&E 图像的转换。

## 项目结构

Git 保留范围、忽略规则和本地文件说明见 [`docs/REPOSITORY_SCOPE.md`](docs/REPOSITORY_SCOPE.md)。

```
Multi-layer-CycleGAN/
├── models/                     # 模型定义
│   ├── registry.py            # 架构 / 输入类别 / 融合 / 额外损失组合
│   ├── cycle_gan_model.py      # 基础 CycleGAN，单类与 Stack 共用
│   ├── cycle_gan_saliency_model.py # 阈值内容保持
│   ├── cycle_gan_structure_model.py # SPIF 融合，内含结构与强度损失
│   ├── cut_model.py            # CUT / FastCUT，内含 PatchNCE 损失
│   ├── pix2pix_model.py        # 配对监督
│   ├── pix2pix_colorization_model.py # Lab 着色适配
│   ├── test_model.py          # 通用生成器推理
│   ├── template_model.py      # 模型开发模板
│   ├── networks.py             # 生成器 / 判别器 / SPIF 双分支融合网络
│   └── base_model.py           # 抽象基类
├── data/                       # 数据加载
│   ├── dual_channel_dataset.py # 双通道数据集（BF + AF 配对）
│   └── unaligned_dataset.py    # 标准非配对数据集
├── options/                    # 命令行参数配置
├── util/                       # 工具函数（可视化、DDP 等）
├── training/train.py           # 模块化统一训练入口
├── inference/infer.py          # 模块化统一推理入口
├── launch_gui.py               # 图形化训练 / 推理控制台
└── tool/                       # 辅助工具脚本
    ├── patch_generate_multi_mode.py  # 滑窗切块
    ├── patch_stitching.py            # 图像拼接
    ├── visualize_content_loss.py     # SPIF 内容损失可视化
    ├── auto_freq_filter.py           # 频域滤波
    ├── img_registrating_pair.py      # 成对图像配准（依赖 series 模块）
    ├── color_normalization.py        # 颜色归一化
    ├── threshold_seg.py              # 阈值分割
    └── ...
```

## 核心特性

### 1. SPIF V2 生成器（SPIFV2Generator）

`--model cycle_gan_dual_fusion_structure --spif_version v2` 使用单阶段空间—通道竞争融合结构：

```
BF → 独立编码器 ─┐
                  ├→ H/2 竞争式融合 → 共享下采样 → ResNet → 跳连解码 → H&E
AF → 独立编码器 ─┘
```

- **单阶段竞争融合**：在 `H/2` 联合空间权重与通道权重，BF/AF 权重互补归一化，可在局部区域抑制不可靠模态
- **共享语义编码**：融合结果经共享下采样进入 `H/4` ResNet 主干，避免从已融合特征重复构造并再次融合两条高度相关的分支
- **结构跳连**：`H/2` 融合特征直接参与解码，减少细边缘经过瓶颈后的损失
- **改进反向生成器**：共享 H&E 编码和低分辨率语义，从 `H/2` 开始使用独立 BF/AF 解码器
- **两个明确版本**：`--spif_version v1` 为通道融合，`v2` 为通道—空间融合；Stack 由独立组合选择，不是版本参数。

### 2. SPIF 模型

SPIF（Structure-Preserving Image Fusion）在标准 CycleGAN 基础上添加了 **结构保持损失** 与 **跨通道特征融合**：

- 只在原始图像分辨率计算逐通道 Sobel 梯度幅值，通过加权算术平均融合，专门约束细微结构
- 梯度不做逐图 RMS 归一化，避免将近均匀背景中的微弱噪声放大为强结构
- 使用反射填充避免随机裁剪边界产生人工梯度
- 强度项沿用原论文的通道均值与 21×21 Gaussian 低通，在 8×8 粗网格上最小化 `1−ρ`，以较软的方向性监督维持大块组织的明暗分布，不回归具体亮度值
- 梯度与强度使用独立的前置余弦衰减：默认分别在总 epoch 的 10% 和 2.5% 处归零；前者负责细微结构，后者只在最初期引导粗粒度明暗布局并避免前景/背景反转

### 3. 双通道数据集

`DualChannelDataset` 从 `trainA_BF/` 和 `trainA_AF/` 目录加载同名文件，对两张图像施加相同的空间变换以保证像素级对齐，最终堆叠为 `(2, H, W)` 张量。

目录结构要求：
```
dataroot/
├── trainA_BF/    # 明场图像
├── trainA_AF/    # 自发荧光图像（与 BF 同名）
├── trainB/       # H&E RGB 图像
├── testA_BF/     # 测试集明场
└── testA_AF/     # 测试集自发荧光
```

## 快速开始

### 图形界面

```bash
python launch_gui.py
```

界面通过“架构、输入类别、融合策略、额外损失”四项联动选择组合，只保留“训练 / 推理”两个执行入口。
使用说明见 [`docs/GUI_CN.md`](docs/GUI_CN.md)，目录与新旧命名映射见 [`models/README.md`](models/README.md)。

### 训练

```bash
# SPIF 模型（推荐用于双通道虚拟染色）
python training/train.py \
    --dataroot ./datasets/mouse_kidney_dual_BF_AF_HE \
    --name mouse_kidney_dual_BF_AF_HE_v2 \
    --model cycle_gan_dual_fusion_structure \
    --spif_version v2 \
    --dataset_mode dual_channel \
    --input_nc 2 \
    --output_nc 3 \
    --lambda_identity 0 \
    --batch_size 8 \
    --n_epochs 100 \
    --load_size 286 \
    --crop_size 256

# 标准 CycleGAN（单通道 / RGB）
python training/train.py \
    --dataroot ./datasets/maps \
    --name maps_cyclegan \
    --model cycle_gan_single_none_base \
    --input_nc 3 \
    --output_nc 3
```

### 测试

```bash
python inference/infer.py \
    --dataroot <test_data_path> \
    --name mouse_kidney_dual_BF_AF_HE_v2 \
    --model test \
    --source_model cycle_gan_dual_fusion_structure \
    --model_suffix _A \
    --dataset_mode dual_channel \
    --spif_version v2 \
    --input_nc 2 \
    --output_nc 3 \
    --no_dropout \
    --preprocess none \
    --epoch latest
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `cycle_gan_single_none_base` | 架构_类别_融合_额外损失，仅支持注册组合 |
| `--dataset_mode` | `unaligned` | 数据集模式：`dual_channel` / `unaligned` / `aligned` |
| `--input_nc` | `3` | 输入通道数：`2`=双通道 BF+AF, `3`=RGB, `1`=灰度 |
| `--output_nc` | `3` | 输出通道数 |
| `--netG` | `resnet_9blocks` | SPIF 根据 `--spif_version` 自动选择对应的前向和反向生成器 |
| `--spif_version` | `v2` | `v1`=通道融合，`v2`=通道—空间融合；仅 Fusion 组合使用 |
| `--lambda_A/B` | `10.0` | 循环一致性损失权重 |
| `--lambda_identity` | `0.5` | 身份映射损失权重（input_nc ≠ output_nc 时自动禁用） |
| `--lambda_content` | `1.0` | SP loss 总倍率；设为 0 可关闭整个 SP loss |
| `--lambda_sp` | `1.0` | SPIF 结构保持损失的初始强度 |
| `--lambda_intensity` | `0.5` | 粗粒度组织明暗分布项相对权重 |
| `--grad_channel_weights` | 等权 | BF/AF 在梯度融合及强度通道均值中的权重，例如 `0.5 0.5` |
| `--sp_grad_decay_fraction` | `0.1` | 原尺寸细微结构梯度引导的总 epoch 比例 |
| `--sp_intensity_decay_fraction` | `0.025` | 粗粒度组织明暗分布引导的总 epoch 比例，应短于梯度窗口 |
| `--intensity_pool_size` | `8` | 粗粒度强度相关性使用的网格尺寸 |
| `--gan_mode` | `lsgan` | GAN 目标函数：`lsgan` / `vanilla` / `wgangp` |
| `--norm` | `instance` | 归一化层：`instance` / `batch` / `syncbatch` / `none` |

## 辅助工具

`tool/` 目录提供了完整的图像预处理和后处理工具链：

- **patch_generate_multi_mode.py** — 大图滑窗切块，支持多种采样模式
- **patch_stitching.py** — 将推理结果拼接回大图
- **visualize_content_loss.py** — SPIF 内容损失中间结果可视化
- **auto_freq_filter.py** — 自动频域滤波去噪
- **img_registrating_pair.py** — 成对图像刚性/弹性配准，依赖同目录的 `img_registrating_series.py`
- **color_normalization.py** — H&E 颜色归一化

## 依赖

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy, PIL, opencv-python
- visdom（可视化）
- dominate（HTML 结果页面）

## 参考

- [CycleGAN: Unpaired Image-to-Image Translation (Zhu et al., ICCV 2017)](https://arxiv.org/abs/1703.10593)
- [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) — 本项目的基础代码框架
