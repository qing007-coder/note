# PVTFormer：Point-Voxel Transformer

## 一句话

PVTFormer 的核心是把**点特征和 voxel 特征结合起来**：point 分支保留点云的细节和几何位置，voxel 分支提供规则网格上的局部上下文，再用 Transformer/注意力让两种表示互相补充。

## 为什么需要 point + voxel

```text
纯 point：保留细节，计算和邻域建模不够规则
纯 voxel：方便用卷积，速度快，但量化会损失细节
PVTFormer：point 保细节，voxel 保结构和高效计算
```

voxel 化之后，相邻点被放进同一个网格，容易丢失精确位置；但点云直接做全局 attention 又很贵。因此模型通常让两条分支分别处理，再进行点到 voxel、voxel 到点的特征交互。

## 整体数据流

```text
原始点云
   │
   ├─ voxelization / pillarization → voxel features
   │                                  │
   │                                  └─ sparse convolution 提取局部上下文
   │
   └─ point branch → point features
                                      │
              point-voxel feature interaction / Transformer
                                      │
                         融合后的 3D 特征
                                      │
                         检测头输出类别和 3D 框
```

不同实现的具体模块名字可能不同，但面试时抓住这条主线即可。

## 两种表示分别做什么

### Point 特征

每个点带有坐标和其他属性，例如：

```text
(x, y, z, intensity) → point feature
```

point 分支更容易保留物体边界、稀疏几何细节和点的精确位置。

### Voxel 特征

把空间划成规则网格，同一 voxel 内的点经过 pooling、linear 或小型 point network 得到一个 voxel feature：

```text
多个点 → 一个 voxel feature
```

之后可以用 SparseConv 在非空 voxel 上做高效的 3D 局部建模。

## Point 和 Voxel 怎么交互

常见方式有两类：

### Point → Voxel

把 point feature 按点所在的 voxel 坐标聚合到 voxel 上，例如平均、最大池化或加权聚合。这样 voxel 分支得到点级的细节信息。

### Voxel → Point

根据每个点所在的 voxel 坐标，从 voxel feature 中查询对应特征，再和 point feature 拼接或相加。这样点分支得到规则网格中的上下文信息。

```text
point feature  ← 查询对应 voxel feature
voxel feature  ← 聚合落入该 voxel 的 point feature
```

如果使用 Transformer，还会通过 attention 学习哪些点、哪些 voxel 更值得交互，而不是固定地把所有邻居等权相加。

## Transformer 在这里做什么

它主要负责建立 point 和 voxel 之间的对应关系：

- query 可以来自 point，也可以来自 voxel；
- key/value 来自另一种表示；
- attention 权重表示当前 point/voxel 应该从哪些特征中取信息；
- 位置编码或相对坐标用于保留 3D 几何关系。

因此它不是简单 concat，而是**根据内容和空间关系进行自适应融合**。

## SparseConv 在 PVTFormer 中的位置

SparseConv 通常在 voxel 分支中使用：

```text
非空 voxel features + coordinates
          ↓
SubMConv：保持活跃 voxel 位置不变，提取局部特征
          ↓
SparseConv(stride=2)：下采样并扩大有效空间范围
          ↓
多尺度 voxel features
```

SparseConv 只对有效坐标建立卷积连接，不会把整个 3D 空间真的展开成稠密 tensor。SubMConv 主要保稀疏，stride sparse conv 负责下采样和扩大感受野。

## PVTFormer 的优点

- point 分支减少 voxel 化造成的细节损失；
- voxel 分支适合 SparseConv，局部建模效率高；
- Transformer 让两种表示能够自适应互补；
- 比直接在稠密 3D voxel 上做 Transformer 更省计算。

## 面试版回答

> PVTFormer 的核心是 point-voxel 双分支。点分支保留原始点云的精细几何和边界信息，voxel 分支把点云规则化后用 SparseConv 做高效的局部上下文建模。随后通过 point-to-voxel 和 voxel-to-point 的交互，或者用 Transformer attention，让两种表示互相查询和补充，最后得到融合的 3D 特征用于检测。它解决的主要问题是：纯 voxel 有量化损失，纯 point 计算和局部结构建模效率较低。

## 容易混淆的点

- PVTFormer 里的 voxel 分支可以使用 SparseConv，但 PVTFormer 本身不等于 SparseConv。
- SparseConv 解决的是**如何高效处理稀疏 voxel 网格**；Transformer 解决的是**不同特征之间如何建立自适应关系**。
- Point 和 voxel 的融合不一定只有一种实现，可能是 pooling、feature lookup、concat、sum 或 attention，具体要以代码版本为准。
