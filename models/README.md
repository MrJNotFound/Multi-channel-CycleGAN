# 模型与 SPIF 版本

模型组合统一命名为 `<架构>_<输入类别>_<融合策略>_<额外损失>`，不再提供旧名称或旧导入路径别名。
注册表描述组合，具体实现可被多个组合复用，不为不同参数复制模型文件。
源码采用扁平目录和架构前缀，贴近原始 CycleGAN 的 `*_model.py` 风格；文件名不等同于命令行组合 ID。
模型专用损失直接放在对应模型文件内：PatchNCE 在 `cut_model.py`，梯度、强度损失及其调度函数在 `cycle_gan_structure_model.py`，不再单设损失文件。

## SPIF v1 / v2

| 版本 | 融合机制 | 前向 / 反向生成器 |
| --- | --- | --- |
| v1 | 通道融合 | `SPIFV1Generator` / `SPIFDualV1Generator` |
| v2 | 通道—空间竞争融合 | `SPIFV2Generator` / `SPIFDualV2Generator` |

界面选择 Fusion 后直接显示“SPIF 版本”下拉框。命令行用 `--spif_version v1` 或 `--spif_version v2`，默认 v2。
Stack 是独立的融合策略，不是 SPIF 的某个版本，也不接收版本参数。两版都保留原网络计算和 state_dict 参数键。

## 四个组成元素

- 架构：`cycle_gan`（双向循环）、`cut`（单向对比）、`pix2pix`（配对监督）。骨干由 `netG` 选择。
- 输入类别：`single` 为一种来源，可为灰度或 RGB；`dual` 为 BF + AF 两种来源，各一路灰度。
- 融合：`none` 为单类不融合；`stack` 为输入通道堆叠；`fusion` 为 SPIF 双分支融合。
- 额外损失：`base` 为无额外损失；`structure` 为梯度与粗强度引导；`saliency` 为 UTOM 阈值内容保持。

基础目标始终存在：CycleGAN 自带 GAN、循环及可选恒等损失；CUT 自带 GAN、PatchNCE；Pix2Pix 自带条件 GAN、L1。
结构保持权重仍可单独调整；UTOM 保留原有指数衰减。Lab 着色是 Pix2Pix 的任务适配，使用后缀 `_colorization`。

## 目录

```text
models/
├── registry.py                 # 组合、参数约束、统一生成器选择
├── __init__.py                 # 只加载注册组合
├── base_model.py               # 训练与严格权重加载接口
├── networks.py                 # v1/v2 及其他网络层定义
├── cycle_gan_model.py          # 基础 CycleGAN，单类与 Stack 共用
├── cycle_gan_saliency_model.py # UTOM 内容保持，单类与 Stack 共用
├── cycle_gan_structure_model.py # 单类 / Stack / Fusion 共用的结构保持实现
├── cut_model.py                # CUT / FastCUT，单类与 Stack 共用
├── pix2pix_model.py            # 配对监督
├── pix2pix_colorization_model.py # Lab 着色
├── test_model.py               # 通用生成器推理
└── template_model.py           # 开发模板，不作为运行模型
```

## 使用

```bash
python training/train.py --dataroot datasets/sample --name spif_v1 --model cycle_gan_dual_fusion_structure --spif_version v1
python training/train.py --dataroot datasets/sample --name spif_v2 --model cycle_gan_dual_fusion_structure --spif_version v2
python inference/infer.py --dataroot datasets/sample --name spif_v1 --model test --source_model cycle_gan_dual_fusion_structure --dataset_mode dual_channel --spif_version v1 --model_suffix _A
```

单类结构保持使用 `cycle_gan_single_none_structure`；Stack 结构保持使用 `cycle_gan_dual_stack_structure`。
Stack 仅改变输入组织与通道配置，不单独实现模型：基础目标共用 `cycle_gan_model.py`，
阈值内容保持共用 `cycle_gan_saliency_model.py`，结构保持共用 `cycle_gan_structure_model.py`，CUT 共用 `cut_model.py`。
界面选择“双类 → Stack”时联动双目录数据加载与 `input_nc=2`；单类可以选择灰度或 RGB。
输入、输出通道不一致时，CycleGAN 自动禁用 identity loss；双类预览分别展示 BF、AF。
组合 ID、GUI schema 4 和权重参数键保持不变，已有 Stack 配置无需重新保存。
融合消融使用 `cycle_gan_dual_fusion_base`，它仅将内容损失关闭，不另建网络。
仅支持已实现组合：Fusion 使用 ResNet 6/9 blocks；没有 CUT + Fusion 或 UTOM + Fusion。

## 清理边界与已有权重

已删除旧导入兼容器、旧模型别名、旧 GUI 配置迁移、误名的融合空壳、纯参数包装、
缺少配套实现的 SinCUT，以及重复的单/双路与微调入口。现在仅有 `training/train.py` 和 `inference/infer.py`；
续训、微调仍由统一训练入口的 `--continue_train`、`--pretrained_name` 等参数控制。

旧命令与旧 JSON 请按照当前接口重新配置，不再静默迁移。GUI 仅保存/加载 schema 4。
`spif_arch` 和损失参数的旧拼写已移除；使用 `spif_version`、`lambda_intensity`、`sp_grad_decay_fraction`。

已有权重文件无需改名，也没有被修改；选择对应 v1/v2、原骨干、通道数和网络宽度即可加载。
检查点参数键必须严格匹配，不再删除未知键或忽略缺失参数。版本/结构不匹配会明确报错，不会假装加载成功。
这不承诺加载历史开发中已废弃的第三种结构或非常早期 PyTorch 格式。

扩展时在 `models/<架构>_<变体>_model.py` 实现模型，并在 `IMPLEMENTATIONS` / `MODEL_SPECS` 显式注册；不自动扫描文件名。
原架构子目录已移除，Python 导入使用新的扁平路径；训练/推理组合 ID、GUI 配置格式和 state_dict 权重格式不变。
Git 保留范围见 [仓库范围说明](../docs/REPOSITORY_SCOPE.md)。被 Git 排除的本地实验脚本与论文记录不属于新版运行接口，未自动改写。
