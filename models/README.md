# 模型组织与命名

新组合 ID 统一为 `<架构>_<输入类别>_<融合策略>_<额外损失>`。它描述配置，不要求每个组合复制一份实现。
注册、兼容和约束集中在 `registry.py`，GUI 与命令行共用；网络层定义仍在 `networks.py`，未修改。

## 四个组成元素

| 元素 | 含义与取值 |
| --- | --- |
| 模型架构 | `cycle_gan`（双向循环）、`cut`（单向对比）、`pix2pix`（配对监督）；具体生成器骨干由 `netG` 选择 |
| 输入类别 | `single`：一种来源，可为灰度或 RGB；`dual`：BF 与 AF 两种来源，本仓库按两路灰度读取 |
| 融合策略 | `none`：单类不融合；`stack`：输入通道堆叠；`fusion`：SPIF 双分支特征融合 |
| 额外损失 | `base`：无额外损失；`structure`：SPIF 梯度与粗强度引导；`saliency`：UTOM 阈值内容损失 |

“无额外损失”不等于没有损失：CycleGAN 自带 GAN、循环一致性及可选恒等损失；CUT 自带 GAN、PatchNCE；Pix2Pix 自带条件 GAN、L1。
`lambda_content`、`lambda_sp`、`lambda_intensity` 等仍可在结构保持组合内调节。UTOM 的权重仍按原实现指数衰减，不伪装成 SPIF 的 `lambda_content` 开关。

## 目录

```text
models/
├── registry.py                 # 四维组合、实现映射、参数一致性检查
├── compat.py                   # 旧 Python 导入路径的延迟兼容
├── base_model.py               # 公共训练/权重接口
├── networks.py                 # 原有网络定义（不变）
├── cycle_gan/
│   ├── standard.py             # CycleGAN 基础实现
│   ├── dual_stack.py           # 原双类 Stack 基线
│   ├── structure.py            # 基础损失 + 结构保持；按参数选择融合
│   ├── single_structure.py     # 单类结构保持的薄适配
│   ├── dual_stack_structure.py # 双类 Stack 结构保持的薄适配
│   ├── saliency.py             # 基础损失 + UTOM 阈值内容保持
│   └── dual_stack_saliency.py  # 双类 Stack 内容保持
├── cut/
│   ├── standard.py             # CUT / FastCUT
│   ├── dual_stack.py           # 双类 Stack 参数适配
│   └── single_image.py         # SinCUT 单图实验扩展
├── pix2pix/
│   ├── standard.py             # 配对监督
│   └── colorization.py         # Lab 着色任务适配
├── losses/
│   ├── structure.py            # 梯度、粗强度损失与早期衰减函数
│   └── patchnce.py             # PatchNCE
├── runtime/inference.py        # 仅加载生成器的推理适配
├── examples/template.py        # 新模型开发模板
└── legacy/fusion_alias.py      # 历史误名兼容，不作为 Fusion 展示
```

这里没有合并或重写各实现的 forward、反向传播、优化器，也没有移动网络内部层级。原有的重复初始化代码暂时保留，避免整理目录时改变训练行为。

## 旧名称到新组合

以下为原有推荐参数对应关系；旧配置中的实际参数优先，不能准确归类时会以兼容模块载入。

| 旧名称 | 新组合 ID |
| --- | --- |
| `cycle_gan` | `cycle_gan_single_none_base` |
| `cycle_gan_stack` | `cycle_gan_dual_stack_base` |
| `spif_single` | `cycle_gan_single_none_structure` |
| `spif_stack` | `cycle_gan_dual_stack_structure` |
| `spif` | `cycle_gan_dual_fusion_structure` |
| `utom` | `cycle_gan_single_none_saliency` |
| `utom_stack` | `cycle_gan_dual_stack_saliency` |
| `cut` | `cut_single_none_base` |
| `cut_stack` | `cut_dual_stack_base` |
| `pix2pix` | `pix2pix_single_none_base` |
| `colorization` | `pix2pix_single_none_base_colorization` |

此外，`cycle_gan_dual_fusion_base` 使用现有 SPIF 生成器与 `lambda_content=0`，提供纯融合架构消融；不新增网络。

**特别注意 `cycle_gan_fusion`：** 当前原始实现只是 `CycleGANModel` 的空继承，实际使用标准生成器，没有特征融合。旧 ID 与导入路径仍原样保留，绝不自动换成 SPIF，以免旧权重不匹配。真正的 Fusion 请选上表的融合组合。

Colorization 与普通 Pix2Pix 的差异还包括 Lab 数据及输出转换，因此作为任务适配后缀，并在 GUI 的 Pix2Pix 组合下额外显示“任务适配”。SinCUT 指单张图像训练，不是输入类别数；仓库缺少其 StyleGAN2 / singleimage 配套实现，保留代码但不在 GUI 宣称可用。

## 使用与兼容

```bash
python training/train.py --dataroot datasets/sample --name example --model cycle_gan_dual_fusion_structure --dataset_mode dual_channel
python inference/infer.py --dataroot datasets/sample --name example --model test --source_model cycle_gan_dual_fusion_structure --dataset_mode dual_channel --model_suffix _A
```

旧命令 `--model spif`、旧代码 `from models.spif_model import SPIFModel` 和旧 schema 2 配置继续支持。新代码请使用组织后的导入路径。类名、`G_A/G_B/G/D` 权重键、文件后缀不因这次整理改变；恢复实验仍须使用原始 `netG`、`spif_arch`、通道数等参数。

Fusion 只支持原代码已有的 ResNet 6/9 blocks + `spif_arch=v2/legacy`。CUT + Fusion、UTOM + Fusion 等未注册组合不会出现在 GUI 中。不以名称暗中切换网络，冲突参数会在启动前报错。

扩展时把实现放在对应架构目录，在 `IMPLEMENTATIONS` 登记路径与类，再增加 `ModelSpec` 组合。公共损失可放到 `losses/`。旧式 `models/<name>_model.py` 自定义插件仍可自动发现，并显示在“兼容 / 自定义插件”区域。

所有已注册模型、兼容适配与独立损失均纳入 Git 保留范围，避免新克隆仓库缺少界面选项或测试依赖。实验启动预设、数据及权重等仍排除，详见 [`docs/REPOSITORY_SCOPE.md`](../docs/REPOSITORY_SCOPE.md)。
