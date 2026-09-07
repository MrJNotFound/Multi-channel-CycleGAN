# SPIF V2 网络数据流详解

本文档详细描述 SPIF V2 架构中每个模块的运算、输入输出维度与层间数据流。

**约定**（全文示例均采用以下配置）：

| 符号 | 含义 | 示例值 |
|------|------|--------|
| `B` | batch size | 8 |
| `ngf` | 生成器基础通道数 | 64 |
| `H, W` | 输入图像尺寸 | 256×256（`--crop_size 256`） |
| `n_blocks` | ResNet 块数量 | 9（`--netG resnet_9blocks`） |

输入张量统一为 NCHW 格式，生成器输入输出范围 [-1, 1]。

---

## 1. 整体架构

```
                 G_A (前向: BF+AF → H&E)                     G_B (反向: H&E → BF+AF)

BF ─→ Head_BF ─→ f_bf@H/2 ─┐                                    H&E ─→ stem ─→ down ─→ down
                             ├→ Fusion ─→ fused_half ──┐                     │      │      │
AF ─→ Head_AF ─→ f_af@H/2 ─┘                           │   skip              ↓      ↓      ↓
                             shared_down@H/4            │   ────────────→   body ← body 特征
                                  ↓                     │                      ↓
                               body (9×ResBlock)        │                   up_shared (H/2)
                                  ↓                     │                      ↓
                               up_to_half ──┐           │                   skip_merge
                                            ├→ skip_merge┘                      ↓
                                  ↓        （拼接 fused_half 跳跃连接）       decoder_bf ─→ BF
                               up_to_full ─→ output ─→ H&E                   decoder_af ─→ AF
```

**设计要点**：

1. **单阶段竞争融合**：在 H/2（2ngf 通道）根据联合特征生成互补的空间—通道权重，随后进入共享编码路径
2. **避免重复融合**：融合后直接共享下采样，不再从同一融合结果构造两条高度相关的分支并立即进行第二次融合
3. **H/2 跳跃连接**：`fused_half` 同时作为解码器跳跃连接，精细边界无需穿过 ResNet 瓶颈
4. **反向上采样**：全部使用 resize-convolution（双线性上采样 + 卷积），避免转置卷积棋盘伪影

---

## 2. 前向生成器 G_A（`SPIFV2Generator`）

**总参数量：11.57M**（ngf=64, 9 blocks, input_nc=2）

### 2.1 模态专用编码头（`head_bf` / `head_af`）

两个头结构完全相同，各 77.2K 参数。

| 层 | 运算 | 输入 → 输出 |
|----|------|------------|
| 1 | `ReflectionPad2d(3)` | (B,1,256,256) → (B,1,262,262) |
| 2 | `Conv2d(1, 64, k=7, pad=0, bias=True)` | (B,1,262,262) → (B,64,256,256) |
| 3 | `InstanceNorm(64)` + `ReLU` | 尺寸不变 |
| 4 | `Conv2d(64, 128, k=3, s=2, p=1)` | (B,64,256,256) → (B,128,128,128) |
| 5 | `InstanceNorm(128)` + `ReLU` | 尺寸不变 |

> `bias=True` 对应 InstanceNorm 约定（与 CycleGAN ResNet 一致，见 `_spif_uses_bias`）。
> 默认 `--norm instance`；若使用 BatchNorm 则此处 `bias=False`。

输出：
```
feat_bf_half: (B, 128, 128, 128)     # H/2 分辨率
feat_af_half: (B, 128, 128, 128)
```

### 2.2 单阶段竞争融合（`fusion_half`）

`SPIFCompetitiveFusion(channels=128)`，86.5K 参数。详见第 4 节模块内部剖析。

```
输入: feat_bf_half (B,128,128,128), feat_af_half (B,128,128,128)
输出: fused_half   (B,128,128,128)
```

### 2.3 共享下采样（`down_shared`）

`down_shared` 含 295.2K 参数，将融合特征送入共享的 H/4 语义空间：

| 层 | 运算 | 输入 → 输出 |
|----|------|------------|
| 1 | `Conv2d(128, 256, k=3, s=2, p=1)` | (B,128,128,128) → (B,256,64,64) |
| 2 | `InstanceNorm(256)` + `ReLU` | 尺寸不变 |

输出：
```
shared_quarter: (B, 256, 64, 64)    # H/4 分辨率
```

### 2.4 ResNet 主体（`body`）

9 × `ResnetBlock(256)`，共 10.62M 参数（占 G_A 的 91.8%）。

每个 ResnetBlock（1.18M 参数）的数据流：

```
x (B,256,64,64)
 ├─→ ReflectionPad2d(1) → Conv2d(256,256,k=3) → Norm → ReLU
 │   → ReflectionPad2d(1) → Conv2d(256,256,k=3) → Norm ─┐
 └─────────────────────────────────────────────────────(+)→ out (B,256,64,64)
```

输入输出均为 `(B, 256, 64, 64)`，分辨率与通道数不变。

### 2.5 解码与跳跃连接（`up_to_half` + `skip_merge`）

| 模块 | 层 | 运算 | 输入 → 输出 |
|------|----|------|------------|
| `up_to_half` | 1 | `Upsample(×2, bilinear)` | (B,256,64,64) → (B,256,128,128) |
| | 2 | `ReflectionPad2d(1)` | → (B,256,130,130) |
| | 3 | `Conv2d(256, 128, k=3, pad=0)` | → (B,128,128,128) |
| | 4 | `Norm(128)` + `ReLU` | 尺寸不变 |
| `skip_merge` | 拼接 | `cat[decoded_half, fused_half]` | (B,128,...)+(B,128,...) → (B,256,128,128) |
| | 1 | `Conv2d(256, 128, k=1)` | → (B,128,128,128) |
| | 2 | `Norm(128)` + `ReLU` | 尺寸不变 |

> `fused_half` 是第一级融合的输出（2.2 节），在此作为跳跃连接与解码器上采样结果拼接。

### 2.6 最终上采样与输出

| 模块 | 层 | 运算 | 输入 → 输出 |
|------|----|------|------------|
| `up_to_full` | 1 | `Upsample(×2, bilinear)` | (B,128,128,128) → (B,128,256,256) |
| | 2 | `ReflectionPad2d(1)` + `Conv2d(128, 64, k=3, pad=0)` | → (B,64,256,256) |
| | 3 | `Norm(64)` + `ReLU` | 尺寸不变 |
| `output` | 1 | `ReflectionPad2d(3)` | → (B,64,262,262) |
| | 2 | `Conv2d(64, 3, k=7, pad=0)` | → (B,3,256,256) |
| | 3 | `Tanh` | 值域压缩到 [-1,1] |

最终输出 `fake_B: (B, 3, 256, 256)` — 虚拟 H&E 图像。

---

## 3. 反向生成器 G_B（`SPIFDualV2Generator`）

**总参数量：11.49M**（ngf=64, 9 blocks, output_nc=2）

设计思想：**几何共享、纹理独立** — H&E 编码与 ResNet 主体共享，H/2 起 BF 与 AF 使用独立解码器。

### 3.1 共享编码路径

| 模块 | 层 | 运算 | 输入 → 输出 |
|------|----|------|------------|
| `stem` | 1-3 | `ReflectionPad2d(3)` → `Conv2d(3,64,k=7)` → `Norm(64)` + `ReLU` | (B,3,256,256) → (B,64,256,256) |
| `down_to_half` | 1-2 | `Conv2d(64,128,k=3,s=2,p=1)` → `Norm(128)` + `ReLU` | → (B,128,128,128) |
| `down_to_quarter` | 1-2 | `Conv2d(128,256,k=3,s=2,p=1)` → `Norm(256)` + `ReLU` | → (B,256,64,64) |
| `body` | — | 9 × `ResnetBlock(256)` | 尺寸不变 |

### 3.2 共享解码至 H/2

| 模块 | 层 | 运算 | 输入 → 输出 |
|------|----|------|------------|
| `up_shared` | 1 | `Upsample(×2)` | (B,256,64,64) → (B,256,128,128) |
| | 2 | `ReflectionPad2d(1)` + `Conv2d(256,128,k=3,pad=0)` | → (B,128,128,128) |
| `skip_merge` | 拼接 | `cat[decoded_half, feat_half]` | → (B,256,128,128) |
| | 1-2 | `Conv2d(256,128,k=1)` → `Norm(128)` + `ReLU` | → (B,128,128,128) |

> 跳跃连接来自 `down_to_half` 的编码输出 `feat_half`。

### 3.3 模态独立解码器（`decoder_bf` / `decoder_af`）

结构相同，各 77.0K 参数：

| 层 | 运算 | 输入 → 输出 |
|----|------|------------|
| 1 | `Upsample(×2)` | (B,128,128,128) → (B,128,256,256) |
| 2 | `ReflectionPad2d(1)` + `Conv2d(128,64,k=3,pad=0)` + `Norm` + `ReLU` | → (B,64,256,256) |
| 3 | `ReflectionPad2d(3)` + `Conv2d(64,1,k=7,pad=0)` | → (B,1,256,256) |
| 4 | `Tanh` | 值域 [-1,1] |

最终输出：
```
fake_A = cat[bf, af] = (B, 2, 256, 256)    # 重建的 BF+AF 双通道
```

---

## 4. 竞争融合模块内部剖析（`SPIFCompetitiveFusion`）

参数：`channels=C=128`，`reduction=4`。该模块只在 H/2 使用一次。

### 4.1 结构与维度（C=128）

```
feat_bf (B,128,128,128)          feat_af (B,128,128,128)
        └────────────┬──────────────┘
                     ↓ cat
              joint (B,256,128,128)
                     ↓
        ┌────────────┴─────────────┐
        ↓                          ↓
channel_logits (全局)       spatial_logits (局部)
  AdaptiveAvgPool2d(1)        Conv2d(256→16, k=3, p=1)
  (B,256,1,1)                 (B,16,128,128)
  Conv2d(256→32, k=1)         ReLU
  ReLU                        Conv2d(16→2, k=1)
  Conv2d(32→256, k=1)         (B,2,128,128)
  (B,256,1,1)                 ↓ view
  ↓ view                      (B,2,1,128,128)
  (B,2,128,1,1)               └──────┬───────┘
        └──────────────┬─────────────┘
                       ↓  logit_delta = (c₀-c₁) + (s₀-s₁)
              delta (B,128,128,128)   [广播相加]
                       ↓
        weight_bf = σ(delta)          (B,128,128,128)
        weight_af = 1 - σ(delta)      (B,128,128,128)
                       ↓
        selected = w_bf ⊙ feat_bf + w_af ⊙ feat_af   (B,128,128,128)
                       ↓
        interaction = Conv1x1(cat[feat_bf⊙feat_af, |feat_bf-feat_af|])
                     = Conv2d(256→128, k=1) → Norm(128)      (B,128,128,128)
                       ↓
        fused = selected + tanh(interaction_scale) · interaction
                       ↓
        output: Norm(128) → ReLU → fused  (B,128,128,128)
```

### 4.2 逐项运算说明

| 组件 | 运算 | 物理含义 |
|------|------|---------|
| `channel_logits` | 全局平均池化 + 2 层 1×1 卷积 MLP | 通道级模态可靠性（整个特征图共享的偏好） |
| `spatial_logits` | 3×3 卷积 + 1×1 卷积 | 空间级模态偏好（允许不同组织区域信任不同模态） |
| `logit_delta` | `(c₀-c₁)+(s₀-s₁)` | 两模态 logits 之差，经 sigmoid 等价于 2 类 softmax |
| `weight_bf/af` | `σ(Δ)` 与 `1-σ(Δ)` | **互补权重**：和为 1，不可靠模态可被压制 |
| `selected` | 加权和 | 竞争选择结果 |
| `interaction` | 1×1 卷积作用于 `[bf⊙af, \|bf-af\|]` | 跨模态一致性（乘积）与分歧（绝对差）特征 |
| `interaction_scale` | 零初始化参数，经 `tanh` 约束到 [-1,1] | 训练初期为纯凸混合，逐步学习跨模态修正 |

### 4.3 与旧版 CCF 的区别

| 项目 | 旧版 `SPIFFusion` | V2 `SPIFCompetitiveFusion` |
|------|-------------------|----------------------------|
| 权重依据 | 另一模态经全局池化产生通道门控 | BF/AF 联合特征同时产生通道与空间 logits |
| 空间分辨率 | 每个通道整幅图共享一个权重 | 每个通道、每个位置均可改变模态偏好 |
| BF/AF 权重关系 | 相互独立 | 互补且逐元素和为 1 |
| 主融合路径 | 独立门控后投影 | 竞争加权形成凸组合 |
| 旁路 | 无条件加入原始 `BF+AF` | 零初始化、可学习缩放的交互修正 |

---

## 5. 判别器（`NLayerDiscriminator`，70×70 PatchGAN）

D_A 与 D_B 结构相同，仅输入通道不同：D_A 输入 3ch（H&E），D_B 输入 2ch（BF+AF）。各约 2.77M 参数。

以 D_A 为例（输入 fake_B 或 real_B，B×3×256×256）：

| 层 | 运算 | 输入 → 输出 |
|----|------|------------|
| 1 | `Conv2d(3, 64, k=4, s=2, p=1)` → `LeakyReLU(0.2)` | (B,3,256,256) → (B,64,128,128) |
| 2 | `Conv2d(64, 128, k=4, s=2, p=1)` → `Norm` → `LReLU` | → (B,128,64,64) |
| 3 | `Conv2d(128, 256, k=4, s=2, p=1)` → `Norm` → `LReLU` | → (B,256,32,32) |
| 4 | `Conv2d(256, 512, k=4, s=1, p=1)` → `Norm` → `LReLU` | → (B,512,31,31) |
| 5 | `Conv2d(512, 1, k=4, s=1, p=1)` | → (B,1,30,30) |

输出 `(B,1,30,30)` 真伪预测图——每个像素对应原图 70×70 的感受野。LSGAN 损失作用于整个预测图。

---

## 6. 结构保持损失的数据流

### 6.1 原尺寸细微结构梯度损失（`FineStructureGradientLoss`）

对每对 (source, translated)，只在原始图像分辨率计算：

```
输入原图 (B,C,256,256)
  ReflectionPad2d(1)
  Sobel_x (÷8归一化):  Conv2d(k=[[-1,0,1],[-2,0,2],[-1,0,1]]/8, groups=C)
  Sobel_y:             Conv2d(k=[[-1,-2,-1],[0,0,0],[1,2,1]]/8, groups=C)
  magnitude = √(gx² + gy² + ε²) − ε     # 常数图 → 精确 0
  channel-weighted mean → (B,1,H,W)

L_grad = L1(translated_grad, source_grad)
```

该分支不再构建图像金字塔，也不进行空间降采样，专门利用原尺寸局部梯度约束细微边缘、细胞轮廓和组织边界。大块组织的明暗布局不由梯度项承担，而由第 6.2 节的粗粒度强度项约束。

> 源域（BF+AF, 2ch）可用 `--grad_channel_weights` 自定义通道权重；目标域（H&E, 3ch）恒用等权。

### 6.2 粗粒度组织明暗分布损失（`CoarseIntensityDistributionLoss`）

```
输入 x (B,C,256,256)
  → channel-weighted mean              (B,1,256,256)
  → 可分离高斯模糊 (radius=10, σ=5)
  → adaptive average pool              (B,1,8,8)
  → 空间均值中心化 + RMS 归一化

ρ = corr(source_coarse, translated_coarse)
L_int = 1 − ρ
```

该项保留原论文“通道均值 + 21×21 Gaussian 低通”的强度路径，但不再对低频亮度值做 L1 回归：减均值消除了全局亮度偏移，RMS 归一化消除了对比度幅值，8×8 池化只保留组织尺度布局。损失直接最小化 `1−ρ`，不使用相关性阈值或铰链截断；它仅通过第 6.3 节的快速 epoch 衰减限制在训练初期，用于维持大块组织区域的正确明暗分布，尤其避免前景/背景反转。近似常量的源图块不参与该项。

> 与原论文一致，BF/AF 在低通前先做通道加权均值；因此数据准备阶段应保证 AF 已翻转，使其前景/背景明暗关系与 BF 一致。`--grad_channel_weights` 同时作为此处的源通道权重。

### 6.3 有效权重调度（`frontloaded_cosine_decay`）

```
total_epochs = n_epochs + n_epochs_decay = 200

S_grad: active = ceil(200 × 0.1) = 20
        epoch=1 时 1.0 → epoch=20 时 0                 # --sp_grad_decay_fraction

S_int:  active = ceil(200 × 0.025) = 5
        epoch=1 时 1.0 → epoch=5 时 0                  # --sp_intensity_decay_fraction

w_grad = λ_content × λ_sp × S_grad(epoch)
w_int  = λ_content × λ_sp × λ_intensity × S_int(epoch)
```

两个调度均在 epoch 域计算，与 batch size、数据集大小和 DDP world size 无关。对应权重归零后分别跳过原尺寸 Sobel 或低通相关性计算。强度项只在最初约 4 个有效 epoch 内引导大块组织的明暗分布，梯度项则保留更长的细微结构引导窗口。

### 6.4 总损失

```
L_G = L_GAN(G_A) + L_GAN(G_B)
    + λ_A · L1(rec_A, real_A) + λ_B · L1(rec_B, real_B)      (λ=10)
    + λ_idt · [L1(idt_A, B) + L1(idt_B, A)]                   (2→3 时自动禁用)
    + w_grad · [L_grad(real_A, fake_B) + L_grad(real_B, fake_A)]
    + w_int  · [L_int(real_A, fake_B) + L_int(real_B, fake_A)]
```

---

## 7. 维度速查表（B=8, ngf=64, 输入 256×256）

### G_A 全程维度

| 阶段 | 张量 | 维度 |
|------|------|------|
| 输入 | BF / AF | (8,1,256,256) ×2 |
| 编码头输出 | feat_bf_half / feat_af_half | (8,128,128,128) |
| 竞争融合输出 | fused_half | (8,128,128,128) |
| 共享下采样输出 | shared_quarter | (8,256,64,64) |
| ResNet 主体 | body 输入/输出 | (8,256,64,64) |
| 上采样至 H/2 | decoded_half | (8,128,128,128) |
| skip 拼接 | cat[decoded_half, fused_half] | (8,256,128,128) |
| skip_merge 输出 | | (8,128,128,128) |
| 上采样至 H | | (8,64,256,256) |
| **输出** | **fake_B (H&E)** | **(8,3,256,256)** |

### G_B 全程维度

| 阶段 | 张量 | 维度 |
|------|------|------|
| 输入 | H&E | (8,3,256,256) |
| stem | | (8,64,256,256) |
| down_to_half | feat_half | (8,128,128,128) |
| down_to_quarter | | (8,256,64,64) |
| body | | (8,256,64,64) |
| up_shared | decoded_half | (8,128,128,128) |
| skip_merge | cat[decoded_half, feat_half] | (8,256,128,128) → (8,128,128,128) |
| decoder_bf / decoder_af | | (8,1,256,256) ×2 |
| **输出** | **fake_A (BF+AF)** | **(8,2,256,256)** |

### 判别器

| 判别器 | 输入 | 输出预测图 |
|--------|------|-----------|
| D_A | (8,3,256,256) H&E | (8,1,30,30) |
| D_B | (8,2,256,256) BF+AF | (8,1,30,30) |

---

## 8. 参数量分布（ngf=64, 9 blocks）

### G_A（11.57M）

| 模块 | 参数量 | 占比 |
|------|--------|------|
| head_bf + head_af | 154.1K | 1.3% |
| fusion_half | 86.5K | 0.7% |
| down_shared | 295.2K | 2.6% |
| body (9×ResBlock) | 10.62M | 91.8% |
| up_to_half + skip_merge | 327.9K | 2.8% |
| up_to_full + output | 83.2K | 0.7% |

### G_B（11.48M）

| 模块 | 参数量 | 占比 |
|------|--------|------|
| stem + down ×2 | 378.5K | 3.3% |
| body | 10.62M | 92.5% |
| up_shared + skip_merge | 327.9K | 2.9% |
| decoder_bf + decoder_af | 153.9K | 1.3% |

### 总计

| 网络 | 参数量 |
|------|--------|
| G_A | 11.57M |
| G_B | 11.48M |
| D_A | 2.765M |
| D_B | 2.764M |
| **训练总计** | **28.58M** |

> 竞争融合模块约 86.5K 参数（G_A 的 0.7%），主体仍是标准 ResNet。是否带来性能收益应通过与旧 CCF 或简单融合的消融实验验证。
