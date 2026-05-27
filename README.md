# Multi-layer CycleGAN — Virtual H&E Staining

基于 CycleGAN 的多层虚拟 H&E 染色项目，支持双通道（明场 BF + 自发荧光 AF）输入到 RGB H&E 图像的转换。

## 项目结构

```
Multi-layer-CycleGAN/
├── models/                     # 模型定义
│   ├── UTOM_model.py           # UTOM 模型（CycleGAN + 梯度域结构保持损失）
│   ├── cycle_gan_model.py      # 标准 CycleGAN（支持双通道输入）
│   ├── networks.py             # 生成器 / 判别器 / 双分支融合网络
│   └── base_model.py           # 抽象基类
├── data/                       # 数据加载
│   ├── dual_channel_dataset.py # 双通道数据集（BF + AF 配对）
│   └── unaligned_dataset.py    # 标准非配对数据集
├── options/                    # 命令行参数配置
├── util/                       # 工具函数（可视化、DDP 等）
├── train.py                    # 训练入口
├── test.py                     # 测试 / 推理入口
└── tool/                       # 辅助工具脚本
    ├── patch_generate_multi_mode.py  # 滑窗切块
    ├── patch_stitching.py            # 图像拼接
    ├── UTOM_mask.py                  # UTOM 掩码生成
    ├── auto_freq_filter.py           # 频域滤波
    ├── paired_img_regist.py          # 图像配准
    ├── color_normalization.py        # 颜色归一化
    ├── threshold_seg.py              # 阈值分割
    └── ...
```

## 核心特性

### 1. 双分支生成器（DualBranchResnetGenerator）

当 `input_nc=2` 时，G_A 自动切换为双分支结构：

```
BF (1, H, W) → Head_BF → feat_BF ─┐
                                   ├→ CrossChannelFusion → Shared Backbone → Output
AF (1, H, W) → Head_AF → feat_AF ─┘
```

- **Head_BF / Head_AF**：独立的 7×7 卷积分支，各自编码单通道特征
- **CrossChannelFusion**：通过通道注意力（Squeeze-and-Excitation）实现跨模态特征交换
- **Shared Backbone**：标准 ResNet 下采样 → ResNet Blocks → 上采样

### 2. UTOM 模型

UTOM（Unpaired Translation with Overlapping Modalities）在标准 CycleGAN 基础上添加了 **梯度域结构保持损失**（Sobel content loss）：

- 对输入和生成图像计算 Sobel 梯度幅值
- 用 L1 损失匹配梯度结构，引导生成器保留边缘和纹理
- 损失权重按 $15 \cdot e^{-counter / data\_size}$ 指数衰减，使训练初期以结构引导为主，后期交给 CycleGAN 目标主导

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

### 训练

```bash
# UTOM 模型（推荐用于双通道虚拟染色）
python train_dual.py \
    --dataroot ./datasets/mouse_kidney_dual_BF_AF_HE \
    --name mouse_kidney_dual_BF_AF_HE \
    --model utom \
    --dataset_mode dual_channel \
    --input_nc 2 \
    --output_nc 3 \
    --lambda_identity 0 \
    --batch_size 8 \
    --n_epochs 100 \
    --load_size 286 \
    --crop_size 256

# 标准 CycleGAN（单通道 / RGB）
python train_dual.py \
    --dataroot ./datasets/maps \
    --name maps_cyclegan \
    --model cycle_gan \
    --input_nc 3 \
    --output_nc 3
```

### 测试

```bash
python test_dual.py \
    --dataroot <test_data_path> \
    --name mouse_kidney_dual_BF_AF_HE \
    --model utom \
    --input_nc 2 \
    --output_nc 3 \
    --no_dropout \
    --preprocess none \
    --epoch latest
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `cycle_gan` | 模型类型：`cycle_gan` / `utom` / `pix2pix` |
| `--dataset_mode` | `unaligned` | 数据集模式：`dual_channel` / `unaligned` / `aligned` |
| `--input_nc` | `3` | 输入通道数：`2`=双通道 BF+AF, `3`=RGB, `1`=灰度 |
| `--output_nc` | `3` | 输出通道数 |
| `--netG` | `resnet_9blocks` | 生成器架构（2 通道时自动切换为 `dual_resnet_9blocks`） |
| `--lambda_A/B` | `10.0` | 循环一致性损失权重 |
| `--lambda_identity` | `0.5` | 身份映射损失权重（input_nc ≠ output_nc 时自动禁用） |
| `--gan_mode` | `lsgan` | GAN 目标函数：`lsgan` / `vanilla` / `wgangp` |
| `--norm` | `instance` | 归一化层：`instance` / `batch` / `syncbatch` / `none` |

## 辅助工具

`tool/` 目录提供了完整的图像预处理和后处理工具链：

- **patch_generate_multi_mode.py** — 大图滑窗切块，支持多种采样模式
- **patch_stitching.py** — 将推理结果拼接回大图
- **UTOM_mask.py** — 基于 UTOM 思想的边缘掩码生成
- **auto_freq_filter.py** — 自动频域滤波去噪
- **paired_img_regist.py** — 非配对图像刚性/仿射配准
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
