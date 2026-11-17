# SCCF 损失消融训练队列

当前批次按用户要求采用旧 `CycleGAN-and-pix2pix` 仓库的计算与权重保存方式：
开启 cuDNN benchmark 和卷积 TF32，关闭严格确定性；仅保存 G/D 权重，不写
`training_state.pth`。此前严格模式的结果保留在旧目录，不计入重新启动的这批实验。

入口是 `training/loss_ablation_queue.py`，实际训练仍调用统一的 `training/train.py`。
默认数据目录为 `datasets/train_mouse_kidney_dual_BF_AF_HE_512`，包含
`trainA_BF`、`trainA_AF`、`trainB`。BF/AF 同名配对，分别转灰度后按 `[BF, AF]`
堆叠；RGB H&E 随机非配对采样。全部训练图像参与训练，不自动划分验证集。

## 默认实验清单

先执行 A：种子 101–120，各依次训练 A1、A2，共 40 次，停止于第 5 轮。
再执行 B：种子 101–103，各依次训练 B0–B6，共 21 次，每次 200 轮。
所有运行重新初始化，同种子采用相同网络初始化和数据随机序列；B 不接续 A。

| 组 | 梯度有效初始系数 | 强度有效初始系数 | 梯度模式 | 强度模式 |
|---|---:|---:|---|---|
| A1 / B1 | 10 | 0 | 余弦衰减 | 关闭 |
| A2 / B3 | 10 | 5 | 余弦衰减 | 余弦衰减 |
| B0 | 0 | 0 | 关闭 | 关闭 |
| B2 | 0 | 5 | 关闭 | 余弦衰减 |
| B4 | 10 | 5 | 恒定 | 余弦衰减 |
| B5 | 10 | 5 | 余弦衰减 | 恒定 |
| B6 | 10 | 5 | 恒定 | 恒定 |

均使用 `cycle_gan_dual_fusion_structure`、SCCF v2、9 个残差块、ngf/ndf=64、
InstanceNorm、Basic PatchGAN、LSGAN、Adam(lr=0.0002, β=(0.5,0.999))、batch=8、
cycle 权重 10/10、identity=0、pool=50、Normal(0.02)、无 dropout。
数据增强保留方案中的 512→286→随机裁剪 256 和随机水平翻转。

梯度第 20 轮归零、强度第 5 轮归零；恒定组使用独立 constant 模式。
五轮短跑使用 `--stop_after_epoch 5`，学习率与损失调度总长度仍是 200。
新参数 `--sp_grad_weight`、`--sp_intensity_weight` 是独立的最终系数，显式指定时覆盖
旧 lambda 系数；未指定时保留旧行为。`--sp_grad_schedule`、
`--sp_intensity_schedule` 分别选择 `cosine` 或 `constant`。

## 准备、检查和启动

在仓库根目录的 PowerShell 中运行：

```powershell
.\.venv\Scripts\python.exe training\loss_ablation_queue.py
.\.venv\Scripts\python.exe training\loss_ablation_queue.py --preflight
.\.venv\Scripts\python.exe training\loss_ablation_queue.py --run
```

不加 `--run` 会校验数据、保存固定诊断位置和完整清单，但不训练。
预检查用每域前 8 张真实图像检查七种条件、前后向计算、权重保存和预览，验证
同种子初始权重一致、仅用权重可继续训练，且没有生成 training_state；其结果不计入正式实验。
失败检查保留记录，用 `--preflight-dir runs/其他检查目录` 重新检查。

输出默认在 `runs/mouse_kidney_loss_ablation_cyclegan`。`--select B2 B3` 按条件选择，
`--select A1_seed101` 按单次运行选择；修改选择、参数、数据或训练代码须使用新的
`--output`，以免同一批次混用设置。默认单 GPU 0、4 个 worker、float32，
启用 cuDNN 卷积 TF32 和 benchmark，matmul TF32 保持 PyTorch 2.x 默认关闭，
不启用严格确定性。固定种子和独立的数据随机流仍保留；不保证逐位复现连续训练。
256→8 的平均池化采用相同的非重叠平均分箱。

## 进度、暂停和恢复

```powershell
.\.venv\Scripts\python.exe training\loss_ablation_queue.py --status
New-Item runs\mouse_kidney_loss_ablation_cyclegan\PAUSE -ItemType File
```

创建 `PAUSE` 后，当前训练完成本轮并保存，再停止队列。恢复：

```powershell
Remove-Item -LiteralPath runs\mouse_kidney_loss_ablation_cyclegan\PAUSE
.\.venv\Scripts\python.exe training\loss_ablation_queue.py --run --resume
```

异常退出后也使用 `--run --resume`。完成的运行会跳过，失败不会自动换种子或重跑。
续训从最近完整的编号权重继续；没有完整编号权重时拒绝静默重跑。Adam、历史池等
重新初始化，与旧框架只加载权重的方式一致，因此不等价于不中断的训练。
此类续训会在队列中标记，实验汇报时应保留该记录。

每处理 5,000 张图像保存 `latest_net_*.pth`；每 5 轮保存编号权重及 latest；
短跑第 5 轮、完整训练第 200 轮一定保存最终权重。主动暂停时额外保存本轮编号权重。
保存内容是 G_A/G_B/D_A/D_B 和几百字节的 `*_checkpoint.json` 进度信息，
不保存优化器、随机状态、历史池，不产生 `*_training_state.pth`。
旧批次的完整状态恢复功能仍保留在统一训练入口中，新队列使用 `--weights_only`。

## 固定诊断和实验材料

默认从训练集按 BF 组织覆盖比例和纹理固定选 16 对，`diagnostic_sources.jpg`
供检查。每个位置统一 resize 到 286 后中心裁剪 256、无翻转。A 保存第 1–5 轮；
B 保存第 1、5、10、20、50、100、200 轮。预览使用 eval 和 inference_mode，
恢复全部模型训练状态及 RNG，不参与优化。

这些是训练过程诊断，不是独立验证/测试。单独的诊断清单也可通过 `--diagnostics`
提供，格式见默认生成的 `diagnostic_manifest.json`。所有组须使用同一份固定清单。

- `study_plan.json`：全部 61 次配置、数据与代码摘要；`weight_schedule.csv`：逐轮有效系数和学习率。
- `dataset_manifest.csv`：每张训练图像的尺寸和 SHA256；`code_snapshot/`、`environment.*`：代码与环境。
- `queue_state.json`：当前任务及完成/失败/暂停状态；`logs/`：每次训练日志。
- `checkpoints/<运行名>/fixed_previews/`：固定位置输出及辅助相关性。
- `blinded_review/`：完成训练后自动复制匿名预览与待填写的 `scores.csv`；`review_key.json` 为对应关系。
- `paired_inversion_results.csv`：逐种子的配对记录模板；`final_quality_results.csv`：完整训练评价模板。

运行 `--export-review` 可随时导出已经产生的匿名预览。前背景反转需统一人工判读
yes/no/uncertain，相关性正负不自动构成反转结论。杂乱不可辨认输出另记失败或不确定。
尚未提供独立配准的真实 H&E 评价集时，不计算或填造 SSIM、PSNR、LPIPS、FID。
