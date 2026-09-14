# 多尺度可变形注意力（Multi-Scale Deformable Attention）详解

> **一句话**：把标准注意力的「每个 query 和**所有** key 算相似度」换成「每个 query 只在自己**预测出来的 16 个采样点**上做加权求和」——采样位置由 query 特征自己算，所以既能跟着目标形状跑，复杂度又从 `O(N²)` 掉到 `O(N·K)`。
>
> **出处**：Zhu et al., *Deformable DETR: Deformable Transformers for End-to-End Object Detection*, ICLR 2021（§2、Appendix A）
>
> **思想来源**：Dai et al., *Deformable Convolutional Networks*（ICCV 2017）与 *DCNv2*（CVPR 2019）
>
> **贯穿例子**：一张 `800×1066` 的图（跟 `DETR.md` 用同一张），文中 **3 辆车 + 2 个人**。多尺度特征共 `L = 4` 层、`N = 17821` 个 token。全文所有维度都跟着这张图走。
>
> **配套阅读**：完整检测流程见 `Deformable-DETR.md`，原始 DETR 见 `DETR.md`。

---

## 1. 先看标准注意力"浪费"在哪

DETR 的 encoder 自注意力，`N = 850` 个 token，每个 token 都要和**其余全部** token 算相似度：

```
softmax(QKᵀ / √d) V        # Q,K,V 形状 [850, 256]，注意力矩阵 [850, 850]
```

三个浪费：

| 浪费 | 表现 | 后果 |
|---|---|---|
| **先验错了** | 刚训练时 Q/K 是随机的，注意力权重接近**均匀分布** | 模型要花几百 epoch 才学会"该往哪看" → DETR 收敛极慢 |
| **内容早知道了** | 一个 token 真正需要的邻居，通常在它**自己的局部**（同一个物体上），全图 850 个 token 里 99% 无关 | 大部分算力浪费在无关 token 上 |
| **二次复杂度** | `N² × 头数 × 层数`。`850² × 8 = 578 万` | **加不起多尺度**：4 个尺度共 17821 个 token，全注意力矩阵是 `17821² × 8 ≈ 25.4 亿`，fp32 单层就要 **≈10 GB** |

第 3 条是致命的：**DETR 想用小目标友好的多尺度特征，但全注意力在算力上根本不允许**。Deformable DETR 的整个动机就在这里。

---

## 2. 从可变形卷积（DCN）借了什么

DCN 的想法：卷积核的 3×3 采样点不必固定在规则网格上，让网络**自己学偏移**：

```
标准卷积： y(p₀) = Σ_{pₙ∈grid} w(pₙ) · x(p₀ + pₙ)
可变形卷积：y(p₀) = Σ_{pₙ∈grid} w(pₙ) · x(p₀ + pₙ + Δpₙ)      ← Δpₙ 是可学的连续偏移
DCNv2 再加一项：× Δmₙ（把权重也变成预测出来的调制系数，等价于"注意力"）
```

| | 可变形卷积 DCN | 可变形注意力 |
|---|---|---|
| 偏移怎么来 | 一个卷积核在特征图上**逐位置**预测，同层共享逻辑 | **由 query 特征直接预测**（一层 `Linear`），每个 query 一套 |
| 采样范围 | 限制在 `3×3`（或 `k×k`）**窗口内** | 参考点可以在**全图任意位置**，采样半径不受限 |
| 权重 | DCNv2 的 `Δm` 无归一化 | `A` 做 **softmax 归一化**，`Σ A = 1` |
| 层级 | 单尺度 | 多尺度（跨 4 层采样） |

**论文原话（关键等价）**：当 `L = 1`、`K = 1`、且 `W'_m` 取单位矩阵时，可变形注意力**退化成一个可变形卷积**。所以可变形注意力可以看成 "DCN + 注意力 + 多尺度" 的推广。

【贯穿例子】那辆 `32×32` 的**小汽车**：标准注意力要它去和全图 17821 个 token 逐一比较；可变形注意力只让它在"大概位置"周围采 16 个点——其中 `K = 4` 个点落在 C3（stride 8，小目标最清楚）的**汽车像素上**，另外几层的点负责提供上下文。这就是小目标 AP 提升的来源。

---

## 3. 单尺度可变形注意力：公式逐项拆

$$\text{DeformAttn}(z_q,\;p_q,\;x)=\sum_{m=1}^{M}W_m\left[\sum_{k=1}^{K}A_{mqk}\cdot W'_m\,x\!\left(p_q+\Delta p_{mqk}\right)\right]$$

| 符号 | 含义 | 形状/取值 |
|---|---|---|
| `z_q` | query 特征（**唯一的信息源**） | `[C]`，C=256 |
| `p_q` | **参考点**，连续 2D 坐标（带小数） | `[2]` |
| `x` | 输入特征图 | `[H,W,C]` |
| `M` | 注意力头数 | 8 |
| `K` | 每头的采样点数 | 4 |
| `Δp_mqk` | 采样偏移，`Linear(z_q)` 预测 | `[M,K,2]` |
| `A_mqk` | 注意力权重，`Linear(z_q)` 预测 | `[M,K]`，`Σ_k A = 1` |
| `W'_m` | 每个头的 value 投影 | 实现里 = `value_proj` + 分头 |
| `W_m` | 输出投影 | 实现里 = `output_proj` |
| `x(·)` | **双线性插值**取特征（坐标是小数） | `[C]` |

### 3.1 三个必须记住的"反常"点

第一次看这个公式，最容易忽略的是它**跟标准注意力到底哪里不一样**：

1. **没有 Q/K 投影，只有 V 投影**。官方实现里只有 `value_proj` 和 `output_proj`，没有 `query_proj` / `key_proj`：

   ```python
   self.sampling_offsets   = nn.Linear(d_model, n_heads * n_levels * n_points * 2)   # → Δp
   self.attention_weights  = nn.Linear(d_model, n_heads * n_levels * n_points)       # → A
   self.value_proj         = nn.Linear(d_model, d_model)                             # → V
   self.output_proj        = nn.Linear(d_model, d_model)                             # → W_m
   ```

   **注意力权重不是 `softmax(QKᵀ)` 算出来的，而是 query 自己"猜"出来的**。这是表达能力上的本质差别（也是它快的代价）：它学不到"内容↔内容"的相似度匹配，只能学"我这个 query 该看哪儿"。

2. **key 的位置不再是固定的像素网格**，而是 `p_q + Δp_mqk` —— 连续、可学习、每个 query 都不同。

3. **`x(·)` 是双线性插值**：采样点坐标是小数，落在 4 个像素中间，用 `grid_sample` 双线性加权取到那个"虚拟位置"的特征。**这一步对坐标可导**，所以 `Δp` 的梯度能回传到 query 特征——端到端可训练，跟 DCN 同理。

### 3.2 标准注意力 vs 可变形注意力

| | 标准注意力 | 可变形注意力 |
|---|---|---|
| key 位置 | 固定网格 `{1..N}` | `p_q + Δp`，**连续可学** |
| Q/K 投影 | 有 | **无**（只有 V 投影） |
| 权重 `A` | `softmax(QKᵀ/√d)` | **`Linear(z_q)` + softmax** |
| 采样点数 | `N`（全部） | `K`（4，每头） |
| 复杂度 | `O(N²C)` | `O(NKC)` |

---

## 4. 多尺度可变形注意力：换一个公式

让采样点**同时跨层**。这是把 4 个尺度的特征图拼在一起后用的形式：

$$\text{MSDeformAttn}(z_q,\;\hat p_q,\;\{x^l\}_{l=1}^{L})=\sum_{m=1}^{M}W_m\left[\sum_{l=1}^{L}\sum_{k=1}^{K}A_{mlqk}\cdot W'_m\,x^l\!\left(\phi_l(\hat p_q)+\Delta p_{mlqk}\right)\right]$$

新增/变化的部分：

| 符号 | 含义 |
|---|---|
| `l = 1..L` | 特征层级索引，`L = 4` |
| `x^l` | 第 `l` 层的特征图，`[H_l, W_l, C]` |
| `φ_l` | 把归一化坐标映射到第 `l` 层的坐标系：`φ_l(p̂) = p̂ ⊙ (W_l, H_l)` |
| `Δp_mlqk` | 第 `l` 层第 `k` 个点的偏移 |
| `A_mlqk` | **约束：`Σ_{l=1}^{L} Σ_{k=1}^{K} A_mlqk = 1`** |

### 4.1 三个关键设计

**① 参考点用「归一化图像坐标」`p̂_q ∈ [0,1]²`，而不是某层的像素坐标。**
这样 4 个尺度可以共享同一个参考点，靠 `φ_l` 各自换算。**同一个 `(0.6, 0.5)` 在 C3 上是像素 `(80.8, 50)`，在 C5 上是 `(20.2, 12.5)`。**

**② `Δp` 要按每层的尺寸归一化。**
官方实现（mmcv）里：

```python
offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)  # (W_l, H_l)
sampling_locations = reference_points[:, :, None, :, None, :] \
                   + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
```

除以 `(W_l, H_l)` 后，`Δp = 0.01` 在每一层都表示"该层宽度的 1%"——**同一个偏移量在所有尺度上语义一致**，模型不必为每个尺度学一套不同量纲的偏移。

**③ softmax 在 `L × K = 16` 个点上做一次全局归一化**，不是每层各做一次。

```python
attention_weights = self.attention_weights(query)              # [N, M, L*K]
attention_weights = attention_weights.view(N, M, L * K).softmax(-1)   # 一次 softmax 覆盖 L 和 K
```

*为什么*（直觉解释，非论文原文）：让 **"投给哪一层" 变成模型可以显式选择的事**。同一辆小汽车在 C3 上是清晰的一团像素、在 C6 上只是一个点，模型需要有权决定"这一票主要投给 C3"。如果每层各自 softmax，每层都会分到 `1/L` 的份额，等于强迫模型均摊。

### 4.2 层级/位置编码：`scale-level embedding`

一个纯工程但很关键的细节：C3 的 `(12, 20)` 和 C6 的 `(12, 20)` 在归一化坐标下**可能是同一个值**（低分辨率层的坐标是高分辨率层的子集），光靠 sin 位置编码模型分不清"我来自哪一层"。所以官方额外加一个**可学习的 scale-level embedding**（每层一个 256 维向量）：

```python
self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))   # [4, 256]
lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
```

【贯穿例子】C3 的某个 token 的位置编码 = 自己的 2D sin 编码 + `level_embed[0]`；C5 的同坐标 token = 自己的 2D sin 编码 + `level_embed[2]`。这样"位置"和"层级"两个信息都有。

---

## 5. 维度流水账（跟着贯穿例子算）

### 5.1 多尺度特征与 token 数

用 `ceil` 算（官方代码如此）：输入 `H=800, W=1066`

| 层级 | 来源 | stride | 尺寸 | token 数 |
|---|---|---|---|---|
| x⁰ = C3 | res3 最后一张特征图 | 8 | `100 × 134` | 13,400 |
| x¹ = C4 | res4 | 16 | `50 × 67` | 3,350 |
| x² = C5 | res5 | 32 | `25 × 34` | 850 |
| x³ = C6 | C5 上做 `3×3, stride 2` 卷积 | 64 | `13 × 17` | 221 |
| | | | **合计** | **17,821** |

每层各过一个 `1×1 conv` 压到 256 通道（和 FPN 的横向连接同一个作用），拼成 `value [17821, 256]`。

### 5.2 一次 MSDeformAttn 前向（编码器，`N = 17821`）

| 步骤 | 操作 | 形状 |
|---|---|---|
| query | = value（自注意力） | `[17821, 256]` |
| 预测偏移 | `Linear(256 → 8·4·4·2 = 256)` | `[17821, 8, 4, 4, 2]` |
| 预测权重 | `Linear(256 → 8·4·4 = 128)` → `softmax(-1)` | `[17821, 8, 4, 4]` |
| value 投影 | `Linear(256 → 256)` → 分头 | `[17821, 8, 32]` |
| 采样 | 每层 `grid_sample`（K=4 点/头/层） | `[17821, 8, 4, 4, 32]` |
| 加权求和 | 对 `(l, k)` 求和 | `[17821, 8, 32]` |
| 输出投影 | reshape → `Linear(256 → 256)` | `[17821, 256]` |

### 5.3 手工算一个 query（拿笔跟着走一遍）

取 C5 上 `(行 12, 列 20)` 这个 token，它的参考点是自己的像素中心归一化：

```
p̂ = ((20 + 0.5) / 34, (12 + 0.5) / 25) = (0.603, 0.500)
```

`φ_l(p̂)` 换算到各层：

| 层 | `φ_l(p̂) = p̂ ⊙ (W_l, H_l)` | 在哪 |
|---|---|---|
| C3 | `(0.603×134, 0.500×100)` = `(80.8, 50.0)` | 第 50 行、第 80 列附近 |
| C4 | `(0.603×67, 0.500×50)` = `(40.4, 25.0)` | — |
| C5 | `(0.603×34, 0.500×25)` = `(20.5, 12.5)` | 自己的中心，✓ |
| C6 | `(0.603×17, 0.500×13)` = `(10.3, 6.5)` | — |

再假设模型为 **C3 层**预测出 `Δp = (-0.02, +0.01)`（归一化单位），则实际采样位置：

```
loc = φ_8(p̂) + Δp ⊙ (W_l, H_l)          # 等价于官方代码的 Δp / offset_normalizer
    = (80.8, 50.0) + (-0.02 × 134, 0.01 × 100)
    = (80.8 - 2.68, 50.0 + 1.00) = (78.1, 51.0)
```

`(78.1, 51.0)` 是小数坐标 → 在 C3 上做双线性插值取出该点 256 维特征（落在 4 个像素 `(78,51) (79,51) (78,52) (79,52)` 中间，按距离加权）。

16 个采样点都取完后，用 softmax 归一化过的 `A` 加权求和。**如果那辆小汽车正好在 C3 的 `(78, 51)` 附近，模型就会把大权重给 C3 的这几个点** —— 这就是"可变形"三个字的实际含义：采样位置跟着目标跑。

---

## 6. 实现细节（容易踩坑的 6 点）

### 6.1 `grid_sample` 的两个参数

```python
F.grid_sample(v_l, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
```

- **`align_corners=False`**：此时 `W` 个像素的中心在 `(i+0.5)/W`。这也是参考点用 `torch.linspace(0.5, H - 0.5, H) / H` 生成的原因，两者必须配套，否则会整体偏移半个像素；
- **`padding_mode='zeros'`**：采样点落到图外（`Δp` 预测得太大时会）**返回 0，不报错**——等价于"这个点没采到东西"。但要注意：**它的 softmax 权重还在**，所以会白白浪费一份概率质量。

### 6.2 坐标要转成 `[-1, 1]`

`grid_sample` 的 grid 是归一化到 `[-1,1]` 的，而我们的参考点是 `[0,1]`：

```python
sampling_grid = 2 * sampling_locations - 1
```

### 6.3 偏移的初始化（很讲究）

```python
nn.init.constant_(self.sampling_offsets.weight, 0.)      # 权重置 0 → 位置只由 bias 决定
thetas = torch.arange(self.n_heads) * (2.0 * math.pi / self.n_heads)   # 8 个头 8 个方向
grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]       # 归一化方向
grid_init = grid_init.view(-1, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
for i in range(self.n_points):
    grid_init[:, :, i, :] *= i + 1                       # 第 k 个点距离 k+1 像素
with torch.no_grad():                                    # 官方代码写的是 nn.init.constant_(..., .data)，
    self.sampling_offsets.bias.copy_(grid_init.view(-1)) # 但新版 PyTorch 的 fill_ 只收 0-dim，用 copy_ 更稳
```

**为什么这么绕？** 如果 bias 也置 0，那么初始化时 16 个采样点会**全部塌在参考点这一个位置上**，取回来的特征全是同一个向量，加权平均之后输出恒等于该点特征——空间信息完全丢失，训练初期梯度极差。
官方这个初始化等价于：**每个头负责一个方向，4 个采样点沿该方向排开 1/2/3/4 个像素**，铺成一个 8 方向的"星形感受野"（很像 DCN 的膨胀卷积核）。

### 6.4 注意力权重初始化成全 0 → softmax 后均匀

```python
nn.init.constant_(self.attention_weights.weight, 0.)
nn.init.constant_(self.attention_weights.bias, 0.)   # logits 全 0 → A 全 = 1/(L·K) = 1/16
```

初始状态：**在 16 个已铺开的采样点上做均匀平均**。此时仍是有意义的（采样位置已经不同），不像标准注意力那样退化成"全图平均"。

### 6.5 必须用 CUDA 算子才快

双线性采样 + 加权求和会产生**无序内存访问**（unordered memory access），用纯 PyTorch 循环写会非常慢。官方实现是 mmcv 里的 CUDA 算子：

```python
from mmcv.ops import MultiScaleDeformableAttention
# embed_dims=256, num_heads=8, num_levels=4, num_points=4, dropout=0.1（默认值完全一致）
```

这也是 `Deformable-DETR` 官方仓库**必须编译 `MultiScaleDeformableAttention` 扩展**才能跑的原因（`cd models/ops && python setup.py build install`）。

### 6.6 工程约束

- `d_model` 必须能被 `n_heads` 整除；
- 每头维度取 **2 的幂**（如 32）CUDA 实现效率最高；
- 内存瓶颈往往**不在采样结果上，而在偏移张量上**：`N×M×L×K×2` 个 fp32。`17821 × 8 × 4 × 4 × 2 = 456 万` 个数 ≈ 18 MB —— 单层还好，6 层 + 反向传播要乘几倍，这是论文里专门讨论过的开销。

---

## 7. 复杂度：为什么这是"能用多尺度"的前提

### 7.1 公式

| 用在哪 | 标准注意力 | 可变形注意力 |
|---|---|---|
| **Encoder 自注意力** | `O(H²W²C)`（对空间尺寸二次） | **`O(HW·L·K·C)`（对空间尺寸线性）** |
| **Decoder 交叉注意力** | `O(N_q·H·W·C)` | **`O(N_q·L·K·C)`（与图像尺寸无关！）** |
| **Decoder 自注意力** | `O(N_q²·C)` | 不变（`N_q = 300` 本来就小，不值得改） |

关键：decoder 交叉注意力对空间尺寸是**常数**——不管图多大、几层特征图，每个 query 永远只采 `L×K = 16` 个点。

### 7.2 在贯穿例子上算一遍注意力权重个数

| 方案 | 注意力权重个数（单层、8 头） | 规模 |
|---|---|---|
| DETR encoder（单尺度 850 token） | `850 × 850 × 8` | 578 万 |
| **假想**：DETR 换成全注意力 + 多尺度 | `17821² × 8` | **25.4 亿**（fp32 ≈ **10 GB/层**，不可行） |
| **Deformable DETR encoder** | `17821 × (4×4) × 8` | **228 万** |

**结论**：token 数变成 21 倍（850 → 17821），注意力开销反而降到 **1/2.5**；相比"多尺度 + 全注意力"则省了 **约 1100 倍**。

### 7.3 一个诚实的补充

可变形注意力解决的是 **encoder 的二次复杂度**，不是"让模型整体更快"。看论文 Table 1 的 FLOPs：DETR-R50 **86 GFLOPs**，Deformable DETR **173 GFLOPs**——**反而更高**，因为它老老实实处理了 4 个尺度的高分辨率特征图（C3 一层就有 13,400 个 token）。它真正的收益是：

- **训练轮数** 500 → 50 epoch，总训练成本 `2000 GPU 小时 → 325 GPU 小时`；
- **小目标** `AP_S: 22.5 → 26.4`（DETR-DC5 500 epoch vs Deformable 50 epoch）；
- 推理 19 FPS，比 DETR-DC5（12 FPS）快，但比 DETR（28 FPS）慢。

---

## 8. 最小实现（纯 PyTorch，跑一遍就能看懂）

官方 CUDA 算子看不到内部，这里用 `grid_sample` 写一个等价版本。**只用于理解，速度很慢**。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSDeformAttn(nn.Module):
    """多尺度可变形注意力（纯 PyTorch 版）
    query: [N_q, C]；value: [N_v, C]（所有层级的 token 按 shapes 顺序拼接）
    """
    def __init__(self, d_model=256, n_heads=8, n_levels=4, n_points=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.n_levels, self.n_points = n_levels, n_points
        self.head_dim = d_model // n_heads

        # ① 由 query 直接预测 偏移 与 权重（注意：没有 Q/K 投影）
        self.sampling_offsets  = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        # ② 只对 value 做投影
        self.value_proj  = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        # 偏移：权重置 0，bias 铺成"每头一个方向、K 个点 1..K 像素"的星形
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(-1, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        # 注意：新版 PyTorch 的 nn.init.constant_ 底层 fill_ 只收 0-dim 张量，
        # 传 1 维张量会报错，所以这里用 copy_
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.view(-1))
        # 权重：全 0 → softmax 后 = 1/(L*K)，均匀平均
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)

    @staticmethod
    def reference_points(shapes, device):
        """每个 token 的参考点 = 自身像素中心，归一化到 [0,1] 的图像坐标 -> [N, 2]"""
        refs = []
        for H, W in shapes:
            ys = torch.linspace(0.5, H - 0.5, H, device=device) / H
            xs = torch.linspace(0.5, W - 0.5, W, device=device) / W
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            refs.append(torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1))
        return torch.cat(refs, dim=0)

    def forward(self, query, value, shapes, ref_points=None):
        N, M, L, K, D = query.shape[0], self.n_heads, self.n_levels, self.n_points, self.head_dim
        device = query.device
        if ref_points is None:                       # 编码器：参考点就是自己的位置
            ref_points = self.reference_points(shapes, device)

        # ── ① query → 偏移 & 权重 ──
        offset = self.sampling_offsets(query).view(N, M, L, K, 2)
        weight = self.attention_weights(query).view(N, M, L * K).softmax(-1)  # Σ_l Σ_k A = 1
        weight = weight.view(N, M, L, K)

        # ── ② 采样位置 = φ_l(p̂) + Δp / (W_l, H_l) ──
        sizes = torch.tensor(shapes, device=device, dtype=torch.float32)      # [L,2] = (H,W)
        normalizer = sizes.flip(-1)                                           # (W_l, H_l)
        loc = ref_points.view(N, 1, 1, 1, 2) + offset / normalizer.view(1, 1, L, 1, 2)

        # ── ③ 逐层逐头双线性采样 + 加权求和 ──
        v = self.value_proj(value).view(-1, M, D)
        out = query.new_zeros(N, M, D)
        start = 0
        for l, (H, W) in enumerate(shapes):
            v_l = v[start:start + H * W]                                      # [H*W, M, D]
            start += H * W
            for m in range(M):
                feat = v_l[:, m, :].view(H, W, D).permute(2, 0, 1)[None]      # [1, D, H, W]
                grid = (2 * loc[:, m, l] - 1)[None]                           # [0,1] → [-1,1]
                s = F.grid_sample(feat, grid, mode="bilinear",
                                  padding_mode="zeros", align_corners=False)[0]
                s = s.permute(1, 2, 0)                                        # [N, K, D]
                out[:, m] += (s * weight[:, m, l].unsqueeze(-1)).sum(1)
        return self.output_proj(out.flatten(1))                               # [N, C]
```

### 8.1 跑一遍（贯穿例子的缩小版）

```python
torch.manual_seed(0)
shapes = [(100, 134), (50, 67), (25, 34), (13, 17)]      # C3/C4/C5/C6
N = sum(h * w for h, w in shapes)
print(N)                       # 17821 ← 和手算的一致

feat = torch.randn(N, 256)                             # 4 层特征拼在一起
ref  = MSDeformAttn.reference_points(shapes, "cpu")     # [17821, 2]

attn = MSDeformAttn(d_model=256, n_heads=8, n_levels=4, n_points=4)
out = attn(feat, feat, shapes)                          # 编码器用法：自注意力
print(out.shape)               # torch.Size([17821, 256])

# 解码器用法：query 只有 300 个，参考点由 query 自己预测（见 Deformable-DETR.md）
q = torch.randn(300, 256)
ref_dec = torch.rand(300, 2)                            # 实际是 Linear(q)+sigmoid 得到
print(attn(q, feat, shapes, ref_points=ref_dec).shape)  # torch.Size([300, 256])
```

### 8.2 验证一个关键性质

```python
# 初始状态下 A 是均匀的，但 16 个采样位置各不相同
w = attn.attention_weights(feat).view(N, 8, 16)
print(w.softmax(-1)[0, 0])      # 全部 = 0.0625 = 1/16 ✓

# 偏移的 bias 就是那个"星形"初始化：第 0 个头沿 θ=0 方向，4 个点距离 1,2,3,4
print(attn.sampling_offsets.bias.view(8, 4, 4, 2)[0, 0])   # 4 个点的 (dx,dy)
```

---

## 9. 面试常见追问

- **Q：可变形注意力和标准注意力的本质区别是什么？**
  A：四条——① key 的位置从固定网格变成 `p_q + Δp`（连续可学）；② 注意力权重不是 `QKᵀ` 算的，而是由 query 特征 **直接预测**；③ **没有 Q/K 投影，只对 V 投影**；④ 复杂度从 `O(N²)` 变 `O(NK)`，`K` 是常数（4）。

- **Q：它和可变形卷积（DCN）什么关系？**
  A：`L=1, K=1` 且 `W'_m` 为单位矩阵时退化成 DCN。差别在：DCN 的偏移由局部卷积核产生、采样限制在 `k×k` 窗口内、权重无归一化；可变形注意力的偏移是 **per-query 预测**、参考点可在全图任意位置、权重做 **softmax 归一化**，还多了多尺度。

- **Q：为什么要在 `L×K` 上做一次全局 softmax，而不是每层各自 softmax？**
  A：让模型能显式地选择"**投给哪一层**"。同一物体在不同分辨率下的可用信息不同（小目标在 C3 清楚、大目标在 C6 才看得到全貌），全局归一化把"选层"变成一个可学习的权衡；每层各自 softmax 相当于强迫每层固定分到 `1/L` 的权重。

- **Q：`Δp` 为什么要按每层的 `(W_l, H_l)` 归一化？**
  A：统一量纲。归一化后 `Δp = 0.01` 在任何层都表示"该层宽度的 1%"，同一个偏移量在不同尺度上语义一致，模型不用为每个尺度学一套不同尺度的偏移参数。

- **Q：采样点落到特征图外面怎么办？**
  A：`grid_sample(padding_mode='zeros')` 返回 0，不报错。代价是"这个点没采到内容，但它的 softmax 权重仍然占着一份"，会造成轻微的信息损失——所以偏移量本身也需要学得别太野。

- **Q：为什么 encoder 用可变形注意力、decoder 的自注意力却保持原样？**
  A：因为收益和成本不对称。Encoder 有 17821 个 token，全注意力是 `17821²`，绝对不可行；decoder 自注意力只有 `300² = 9 万`，本来就便宜，改成稀疏采样省不了什么，反而丢掉 query 之间的全局交互（query 互相看一眼才能避免"多个 query 抢同一个物体"）。

- **Q：为什么它能让 DETR 收敛快 10 倍？**
  A：根本上是因为**引入了位置先验**。标准注意力从"均匀分布"开始，要靠几百 epoch 学出空间结构；可变形注意力的参考点 + 初始化好的星形采样，一开始就告诉模型"先看自己周围这一小圈"，把搜索空间砍掉了一大截。

- **Q：它是不是比标准注意力快？**
  A：**要看和谁比、比什么**。注意力模块本身快很多（线性 vs 二次），但 Deformable DETR 整体 FLOPs（173G）比 DETR（86G）**更高**，因为它处理了 4 个尺度的高分辨率特征。它的收益是**训练轮数减少 10 倍、小目标 AP 提升**，推理 FPS 19 介于 DETR（28）和 DETR-DC5（12）之间。另外无序内存访问让它比纯卷积慢。

---

## 10. 速记卡

- **一句话**：每个 query 只在自己**预测出来的 `L×K = 16` 个采样点**上做加权求和，采样位置可学、可导；
- **两个 Linear 决定一切**：`sampling_offsets`（→ `Δp`）+ `attention_weights`（→ `A`，softmax 后 `Σ_l Σ_k A = 1`）；**没有 Q/K 投影，只有 `value_proj` / `output_proj`**；
- **三处坐标细节**：参考点是**归一化的图像坐标** `p̂ ∈ [0,1]²`；`Δp` 要除以每层的 `(W_l, H_l)`；进 `grid_sample` 前要 `2x - 1` 映射到 `[-1,1]`；
- **初始化**：偏移的 bias 铺成 "8 个头 8 个方向、4 个点 1~4 像素" 的星形（否则 16 个点全塌在参考点上）；权重 bias 全 0 → 均匀 `1/16`；
- **复杂度**：encoder `O(H·W·L·K·C)`（对空间线性）、decoder 交叉注意力 `O(N_q·L·K·C)`（**与图像尺寸无关**）；相比全注意力 + 多尺度省约 **1100 倍**；
- **贯穿例子记住两组数**：`800×1066 → 4 层共 17821 个 token`；全注意力矩阵 `17821²×8 ≈ 25.4 亿`（fp32 ≈10 GB）→ 可变形 `17821×16×8 = 228 万`（省约 1100 倍）；
- **别忘了**：`scale-level embedding`（区分 token 来自哪一层）；官方用 mmcv 的 CUDA 算子，纯 PyTorch 循环慢但等价。
