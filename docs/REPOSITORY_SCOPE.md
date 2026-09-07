# Git 保留范围

本仓库以“从代码仓库恢复完整的模型、训练/推理与界面功能”为保留标准，而不是只保留一个模型文件。
`.gitignore` 采用根目录白名单，再对训练入口、工具与文档细分。新增顶层文件或工具默认不纳入 Git，需要显式放行。

## 保留

- `models/`：SPIF 主方法、单类/Stack 消融、CycleGAN 基线、已注册的 CUT/UTOM/Pix2Pix 实现、独立损失、注册表与兼容适配。后几项也是界面和兼容测试的一部分，不能只忽略其文件而保留注册记录。
- `data/`、`options/`、`util/`：数据加载源码、参数接口与运行支撑；注意 `data/` 是源码，`datasets/` 才是本地数据。
- `gui/`、`launch_gui.py` 与 `tests/`：交互界面、启动器和回归测试。
- `training/train.py`、`training/finetune.py`、`inference/infer.py`：通用入口；保留原 `train_single/dual.py` 与 `infer_single/dual.py` 兼容入口。
- `tool/` 中明确放行的数据组织、切块/拼接、BF/AF 反相与配准、归一化、去噪、ROI/阈值处理、评价及损失可视化工具。成对配准依赖 `img_registrating_series.py`，两者一并保留。
- 项目说明、许可证、GUI 使用文档、模型命名说明、SPIF 架构说明与示意图，以及未来添加的依赖清单和 CI 配置。

## 仅保留在本地

- 数据集、检查点、推理结果、TensorBoard/W&B 日志、缓存、虚拟环境、IDE 与本机代理设置。
- `paper/`、`output/`、`out/`、`tmp/` 中的论文、导出文档、编译文件和预览图。
- 投稿检查表、手稿修订记录、对比实验笔记。
- 固定数据路径的批量重复实验、专用 CUT/UTOM/SPIF 启动预设、根目录旧实验入口。
- `scripts/` 中上游仓库的演示、下载、HED/Cityscapes 评价及旧环境安装脚本。
- 工具回收站、外部 3D 工程、通用文件整理和体渲染等未进入白名单的工具。论文绘图脚本依赖被排除的手稿源图，也仅本地保留；发布的架构说明和示意图仍纳入 Git。

没有全局忽略 PNG、SVG、JSON、YAML 或 TXT，避免误排除文档示意图、配置样例和依赖清单。权重、日志及秘密配置单独排除。

## Git 索引与本地文件

`.gitignore` 只阻止未跟踪文件被加入，不会自动移除已跟踪文件。首次整理时，仅对明确命中排除规则的已跟踪路径执行 `git rm --cached -- <准确路径>`：本地现存文件不删除，历史版本也不改写。

这会产生待提交的“删除”记录，表示停止版本管理，而不是删除硬盘文件。忽略规则、核心源码新增/搬迁及索引变化仍需正常审核、暂存并提交；本次整理不自动提交。

检查规则：

```bash
git check-ignore -v --no-index -- output/example.pdf models/networks.py
git ls-files -ci --exclude-standard
python -m unittest discover -s tests -p test_gitignore.py -v
```

第二条命令应没有输出。需要新增核心工具或文档时，修改对应白名单，并扩展 `tests/test_gitignore.py` 的保留/排除用例。
