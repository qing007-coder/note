# 稀疏卷积（Sparse Conv）：Regular vs Submanifold

3D 检测/分割里到处都在用（SECOND、CenterPoint、BEVFusion 的 LiDAR 分支、MinkowskiNet…），面试也爱问。
本文只讲清两件事：**两种变种的区别**，以及**它和普通卷积到底差在哪**。

- 关键论文：
  - Graham & van der Maaten, *3D Semantic Segmentation with Submanifold Sparse Convolutional Networks*, CVPR 2018（arXiv 1706.01307）→ submanifold 的来源
  - Choy et al., *4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks*, CVPR 2019（arXiv 1904.08755）→ 把两种统一成"广义稀疏卷积"
- 代码：`spconv`（SECOND/OpenPCDet 用）、`MinkowskiEngine`、`torchsparse`

---

## 1. 为什么需要它

体素网格里 99% 以上是空的（nuScenes 那种 0.075m 的体素，一帧有效点占比零点几个百分点）。
稠密 3D 卷积的计算量是 `H·W·D · K³ · C_in · C_out`，显存也是 `H·W·D·C`——跟"有没有数据"无关，纯浪费。

稀疏卷积的做法：**只存非空体素的坐标 + 特征**，卷积也只在需要的位置上算。

```
indices  (N, 4)   # [batch, z, y, x]，N = 非空体素数
features (N, C)
```

显存和计算量 ∝ N（非空点数），而不是网格体积 H·W·D。这就是"稀疏"的全部意义。

---

## 2. 统一公式：输出位置由谁决定

MinkowskiEngine 给的广义稀疏卷积（`C_in` / `C_out` 是输入/输出坐标集，`V^D(K)` 是 K³ 核的偏移集合）：

```
x_out[u] = Σ_{i ∈ N(u, C_in)}  W_i · x_in[u + i]        u ∈ C_out
其中 N(u, C_in) = { i | u + i ∈ C_in, i ∈ V^D(K) }
```

（Choy et al. CVPR'19, `sparse_tensor_network.html`）

稠密卷积是它的特例：`C_in = C_out = Z^D`（全网格）。
而稀疏卷积只看 **`u + i` 存在（非空）的那些偏移**——**空邻居直接跳过**。这就是你说的"跟着数据走"准确的含义。

两个变种，**唯一区别就是 `C_out` 怎么定**：

| | **Regular**（`SparseConv3d` / 普通稀疏卷积） | **Submanifold**（`SubMConv3d`） |
|---|---|---|
| 输出坐标集 | `C_out` = `C_in` **沿核足迹膨胀**出来的位置 | **`C_out = C_in`**（子流形） |
| 一句话 | 只要有邻居活着，这个位置就产生输出 | 输入哪个位置活着，才在哪个位置输出 |
| 稀疏性 | 每层向外长 1 圈，多层后退化成稠密 | 保持不变，可以堆很深 |
| 感受野/信息流 | 能把信息"铺"到原来空的位置 | 信息只在活跃点之间流动，不产生新活跃点 |
| 主要用途 | stride=1 用来打通隔离区域；stride=2 做下采样；encoder/decoder 之间 | 主干中堆深的那些层 |
| 代价 | 膨胀 → 计算量涨、失去稀疏优势 | 孤立点上看不到任何上下文（见下） |

### 1D 小例子（核 3，偏移 -1/0/1）

输入活跃位置 `C_in = {0, 5}`：

```
SubM:      输出活跃 = {0, 5}         ← 一模一样，只算这两处
Regular:   输出活跃 = {-1,0,1} ∪ {4,5,6} = {-1,0,1,4,5,6}
           再来一层 → {-2..2} ∪ {3..7}，L 层后 → ±L 圈，稀疏性就没了
```

**Submanifold 名字的来历**：Graham 原话是它 "operates strictly on submanifolds, rather than *dilating* the observation with every layer"——支撑集（submanifold）不变。

### 一个容易忽略的推论

SubM 里，某个位置 `u` 的输出只由 `u` 的**活跃邻居**贡献。所以**一个被空体素包住的孤立点，SubM 卷积退化成 1×1 卷积（只有中心权重起作用）**。

这带来 SubM 的真正代价：**信息无法跨过空的区域传播**。两个物体之间隔着一个空体素，SubM 卷积永远把信息传不过去。所以网络设计上是：

```
SubM 为主干（保稀疏） + 少量 Regular / stride=2 的稀疏卷积（膨胀、打通、下采样）
```

典型例子：SECOND 的 backbone 前几层用 `SparseConv3d` 下采样，中间用 `SubMConv3d`；CenterPoint、BEVFusion 的 LiDAR 分支同理。

---

## 3. 实现：gather 和 scatter（你说的"汇总累加 / 发射贡献"）

你的直觉正好点到这件事，但注意——**这不是 Regular 和 Submanifold 的区别**，而是同一件事的两种实现方向，两者数学上完全等价：

| | 输入驱动（scatter，**发射贡献**） | 输出驱动（gather，**汇总累加**） |
|---|---|---|
| 遍历谁 | 遍历每个输入活跃点 × K³ 个核偏移 | 遍历每个输出活跃点 × K³ 个核偏移 |
| 做什么 | 往 `pos + offset` 累加 `W·f` | 去 `pos + offset` 找输入，乘加求和 |
| 复杂度 | ∝ N_in · K³ | ∝ N_out · K³ |
| 关系 | 互为转置：**gather 卷积的反向传播就是 scatter 卷积** | |

工程上怎么做快：

- **rulebook / indice_pairs**：先离线算好 `(input_idx, output_idx, kernel_offset)` 这张映射表。spconv 里叫 `get_indice_pairs(...) -> (outids, indice_pairs, indice_pair_num)`（`spconv/pytorch/conv.py`），MinkowskiEngine 里叫 **kernel map**（"how each non-zero element in an input is mapped to the output"）。
- 有了这张表，卷积就变成 **implicit GEMM**：把相同 kernel offset 的点聚成一批做矩阵乘（spconv 走的就是 `get_indice_pairs_implicit_gemm`），所以不是逐点 for 循环。
- `subm` 就是这个 API 里的一个 bool：`subm=True` → `outids` 直接取输入的 indices；`subm=False` → `outids` 由输入坐标膨胀算出。
- spconv 还有 `indice_key` 缓存：同一个 `indice_key` 的层共享同一张 rulebook（稀疏模式没变就不重算），SECOND/CenterPoint 的 config 里能看到这个参数。

---

## 4. 面试问答

- **为什么不全用 SubM？** 信息传不出去，感受野只有活跃点构成的连通块；而且孤立点等于 1×1 卷积，等于没有空间聚合。
- **为什么不全用 Regular？** 每层膨胀一圈，几层之后支撑集就变成稠密了，稀疏加速完全失效（Graham 那篇的动机就是这个）。
- **stride > 1 的稀疏卷积怎么做下采样？** 输出坐标是"下采样后的网格上、被某个活跃输入点覆盖到的位置"，输出空间分辨率减半，和稠密卷积的 stride=2 对应；上采样用 inverse conv / transposed conv（spconv 的 `SparseInverseConv3d`，把 rulebook 反过来用）。
- **和 PointPillars 的关系？** PointPillars 的 backbone 是 2D 卷积，没有 3D 稀疏卷积——pillar 特征 scatter 成一张 BEV 特征图之后就是普通 2D 网。稀疏卷积出现在 voxel-based 的路线里（VoxelNet 用稠密 3D 卷积，很贵；SECOND 起改用稀疏卷积，这也是 SECOND 比 VoxelNet 快得多的原因之一）。
- **和稀疏化加速（pruning）区别？** 那是权重稀疏、激活还是稠密的；这里说的是**空间稀疏**（数据本身稀疏），完全是两回事。

---

## 5. 你的说法核对

- ✅ "在 voxel 里找不为 0 的点，不为 0 才发射/累加" —— 对，就是 sc 只对 `u + i ∈ C_in` 求和。
- ✅ "发射贡献 / 汇总累加 / 看重的对象不同" —— 就是 scatter（输入驱动）vs gather（输出驱动），互为转置、结果一样。
- ⚠️ "普通卷积跟着卷积核走、稀疏卷积跟着数据走" —— 方向对，但**真正区分 Submanifold 和 Regular 的不是这个**：两者都"跟着数据走"，区别只在**输出位置要不要膨胀**（`C_out = C_in` 还是膨胀集）。这才是这两个变种的核心差别，也是面试最常追问的点。
