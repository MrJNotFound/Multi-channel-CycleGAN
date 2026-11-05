# 顺序实验训练

`training/experiment_queue.py` 从 JSON 清单选择实验，为每项调用统一的
`training/train.py`。一次只运行一个训练进程，完整 SPIF 始终放在最后。
每项只运行一个指定种子；完成状态写入输出目录，再次运行不会重复训练已完成项。

当前肺组织清单位于本地 `output/lung_experiment_design/experiment_plan.json`，
其中包含原始数据目录、已校准的 UTOM 阈值及每项完整参数。
该清单及图像分析结果不纳入 Git；通用调度器与训练代码纳入 Git。

在仓库根目录运行以下命令。Windows 使用 `.venv\Scripts\python.exe`，
其他环境可替换解释器路径。

```powershell
# 列出顺序，不训练
.\.venv\Scripts\python.exe training\experiment_queue.py --config output\lung_experiment_design\experiment_plan.json

# 对每项执行真实图像的一步优化检查，不保存模型权重
.\.venv\Scripts\python.exe training\experiment_queue.py --config output\lung_experiment_design\experiment_plan.json --output runs\lung_12_once --smoke-test

# 顺序训练全部 12 项，每项一次，seed=2026
.\.venv\Scripts\python.exe training\experiment_queue.py --config output\lung_experiment_design\experiment_plan.json --output runs\lung_12_once --run

# 查看状态
.\.venv\Scripts\python.exe training\experiment_queue.py --config output\lung_experiment_design\experiment_plan.json --output runs\lung_12_once --status

# 中断后从最新检查点续训；已完成项仍然跳过
.\.venv\Scripts\python.exe training\experiment_queue.py --config output\lung_experiment_design\experiment_plan.json --output runs\lung_12_once --run --resume

# 只选择部分实验，需要独立输出目录
.\.venv\Scripts\python.exe training\experiment_queue.py --config output\lung_experiment_design\experiment_plan.json --output runs\lung_ablation --select C03 A01 A02 C10 --run
```

`queue_state.json` 保存当前实验、epoch、损失、进程 ID 和完成状态；
`logs/<ID>.log` 保存单项完整控制台日志；`checkpoints/<实验名称>/`
保存参数、权重、优化器状态和训练预览。`queue.lock` 防止同一输出目录被同时运行。
发生错误时队列停止，日志中记录原因。续训要求已有检查点；调度器不会静默从头重训。
更换实验列表、参数、种子或数据后，应使用新的输出目录。

单模态使用 `--source_dir` 与 `--target_dir` 直接读取原图目录；
双模态使用 `--bf_dir`、`--af_dir` 与 `--target_dir`。
BF/AF 同名配对、共享空间增强；H&E 独立随机采样。
训练种子控制模型初始化，独立的数据随机流由种子、epoch 和样本索引决定。
这避免了不同模型消耗随机数的差异改变数据增强，但不承诺 CUDA 运算逐位确定性。

当前清单采用 200 epochs、batch size 8、学习率 0.0002（100 轮固定＋100 轮衰减）。
UTOM 使用官方 U-Net-256；CUT 保留抗混叠采样。CUT 的不等通道输入适配会关闭
identity NCE，并对 RGB 输出取通道均值后回送编码器，因此清单将其标注为 CUT-adapted。
SPIF 完整模型以及两项消融沿用手稿的 SP 权重和衰减日程。

数据检查会比对训练集文件数、BF/AF 文件名集合，以及清单同目录下
`per_image_statistics.csv` 的 SHA-256（如有）。阈值应仅由训练集确定，
验证、测试数据不参与校准。未配对的 H&E 文件即使同名也不能用作配对指标的真值。
