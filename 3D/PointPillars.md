# PointPillars 完整流程

> PointPillars：把 LiDAR 点云组织成**竖直柱子（Pillar）**，再拍扁成 **BEV 2D 伪图像**，最后用 **2D CNN** 做 3D 目标检测。
>
> 核心流程：
>
> **Point Cloud → Pillarization → Point Decoration → PillarFeatureNet → Scatter(BEV) → 2D CNN Backbone + FPN → Detection Head → 3D Box**
>
> 它解决的核心问题是：
>
> > VoxelNet / SECOND 的 3D 卷积又慢又重，能不能**不用 3D 卷积**，只在 BEV 上做 2D 检测？
>
> 一句话：**PointPillars = VoxelNet 的"降维打击"版本。**

---

# 一、先看整体流程

```text
LiDAR Point Cloud  (N × 4)
       │
       ▼
┌──────────────────────────┐
│      Pillarization       │
│  3D 空间 → BEV 网格 (H × W) │
│  每格内所有 z 上的点归为一柱 │
└──────────────────────────┘
       │
       ▼
Pillar 集合  (P × N × D)
       │
       ▼
┌──────────────────────────┐
│     Point Decoration     │
│  补几何信息 (D=9)         │
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│    PillarFeatureNet      │
│  1×1 Conv + BN + ReLU    │
│  + Max over Points       │
└──────────────────────────┘
       │
       ▼
每个 Pillar → 一个 64 维向量  (C × P)
       │
       ▼
┌──────────────────────────┐
│        Scatter           │
│  按 (x,y) 放回 BEV 网格   │
└──────────────────────────┘
       │
       ▼
BEV 伪图像  (C × H × W)
       │
       ▼
┌──────────────────────────┐
│   2D CNN Backbone        │
│   stride 2 / 4 / 8       │
└──────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│   FPN Neck (upsample+cat)│
└──────────────────────────┘
       │
       ▼
多尺度融合特征  (384 × H/2 × W/2)
       │
       ▼
┌──────────────────────────┐
│    Detection Head        │
│    (SSD 风格 Anchor)     │
└──────────────────────────┘
       │
       ├──────────────► Classification  (A × 1)
       │
       └──────────────► Box Regression  (A × 7)
                              │
                              ▼
                 Decode → Rotated NMS → 3D Boxes
```

---

# 二、为什么要有 PointPillars

## 2.1 先回顾 VoxelNet / SECOND 的痛点

VoxelNet 的做法是：

```text
Point Cloud → 3D Voxel → VFE → 3D CNN → RPN
```

问题在 **3D CNN** 这里。看一组真实量级的数字（KITTI）：

```text
voxel_size = (0.2, 0.2, 0.4)
point_cloud_range = [0, -40, -3, 70.4, 40, 1]

X = (70.4 - 0)    / 0.2 = 352
Y = (40 - (-40))  / 0.2 = 400
Z = (1 - (-3))    / 0.4 = 10

特征图 = 352 × 400 × 10
```

如果通道是 128：

```text
352 × 400 × 10 × 128 ≈ 1.8 × 10^8 个浮点数
```

而 LiDAR 大部分空间是**空的**（>95% 的 Voxel 里没有点）。

所以 3D 卷积的问题是：

```text
① 大部分计算浪费在空区域
② 3D 卷积核参数量大、显存吃紧
③ CUDA 上 3D 卷积的 kernel 效率低，推理慢
```

实测速度（KITTI val，V100，论文 Table 3）：

```text
VoxelNet     ≈  4.4 FPS
SECOND       ≈ 20   FPS
PointPillars ≈ 62   FPS   ← 快了一个数量级
```

## 2.2 PointPillars 的关键观察

> **对于 3D 目标检测，z 方向（高度）的精细结构其实没那么重要，重要的是 BEV（鸟瞰俯视图）上的位置、尺寸和朝向。**

车辆、行人、骑行者在地上的"投影轮廓"已经足够区分它们了。

于是思路变成：

```text
VoxelNet：  在 (X, Y, Z) 上做 3D 卷积
PointPillars：把 Z 抹掉，在 (X, Y) 上做 2D 卷积
```

**怎么做？把 z 方向的整根柱子当成一个整体。**

```text
VoxelNet 的 Voxel：          PointPillars 的 Pillar：

   ┌──┬──┬──┐                    │  │  │
   ├──┼──┼──┤  高 0.4m           │  │  │
   ├──┼──┼──┤                    │  │  │
   └──┴──┴──┘                    │  │  │  整根柱子，高度不限
   3D 小方块                      └──┴──┘
                                 竖直的"柱子"
```

---

# 三、第一步：Pillarization

## 3.1 Pillar 是什么

Pillar = 一根**无限高（或者说是整个 z 范围）的竖直方柱**。

它只在 BEV 平面上有尺寸：

```text
pillar_size = (dx, dy)      ← 没有 dz
```

3D 空间被切成规则网格：

```text
            BEV 俯视 (X-Y 平面)
    y ↑
      │  ┌────┬────┬────┬────┐
      │  │    │    │    │    │
      │  ├────┼────┼────┼────┤   ← 每格 = 一根 pillar
      │  │    │    │    │    │
      │  ├────┼────┼────┼────┤
      │  │    │    │    │    │
      │  └────┴────┴────┴────┘
      └───────────────────────→ x
```

## 3.2 具体维度例子（KITTI）

这是 PointPillars 用得最多的配置：

```text
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
                     x_min  y_min   z_min  x_max   y_max  z_max

pillar_size = (0.16, 0.16)     # 单位：米
```

划分网格：

```text
W = (x_max - x_min) / 0.16 = 69.12 / 0.16 = 432      ← x 方向格子数
H = (y_max - y_min) / 0.16 = 79.36 / 0.16 = 496      ← y 方向格子数

BEV 网格 = 432 × 496 = 214,272 个 pillar 位置
```

## 3.3 一个点属于哪根 Pillar

对某个点 `(x, y, z)`：

```text
pillar_x = floor((x - x_min) / 0.16)
pillar_y = floor((y - y_min) / 0.16)
```

**注意：z 完全没有参与！** 不管这个点是 0.5m 高还是 1.8m 高，只要 (x, y) 落在同一格，就属于同一根 pillar。

举例：

```text
点 A = (12.30, 4.52, 0.31, 0.8)
点 B = (12.35, 4.48, 1.62, 0.7)
点 C = (12.31, 4.55, 0.90, 0.6)

pillar_x = floor(12.30 / 0.16) = 76
pillar_y = floor((4.52 + 39.68) / 0.16) = floor(276.25) = 276

→ A、B、C 三点的 (pillar_x, pillar_y) 都是 (76, 276)
→ 它们属于同一根 Pillar
```

这就是关键：

> **同一根 pillar 里，装着竖直方向上一整条"点云柱"。**

```text
        一辆车上的点
             │
    z ↑      ●  1.6m
      │      ●  1.4m
      │      ●  ●  1.0m
      │   ●  ●  ●  ●  0.6m
      │   ●  ●  ●  ●  0.3m
      └─────────────────→ x,y
          └────────┘
           一根 pillar（俯视看只占一个格子）
```

## 3.4 Pillar 的张量化：为什么需要 padding

LiDAR 一帧有 ~100000 个点，分到 ~20000 根非空 pillar 里，每根 pillar 的点数**不一样**：

```text
Pillar 1  → 23 个点
Pillar 2  →  4 个点
Pillar 3  → 41 个点
Pillar 4  →  1 个点
...
```

神经网络需要固定形状，所以要做两件事：

```text
① 每根 pillar 最多保留 N 个点        （N = max_points_per_pillar）
   - 点多于 N：随机采样 N 个
   - 点少于 N：用 0 padding，并记录一个 mask

② 最多保留 P 根 pillar               （P = max_num_pillars）
   - 非空 pillar 多于 P：随机采样
   - 少于 P：用全 0 pillar 补齐
```

于是张量形状变成：

```text
(P, N, D)

P = 最多 pillar 数          例如 16000
N = 每根 pillar 最多点数    例如 32
D = 每个点的特征维度        例如 9
```

也就是：

```text
┌─────────────────────────────┐
│ Pillar Tensor: (16000, 32, 9)│
│                             │
│  pillar_0 ┌ point_0 (9,)    │
│           │ point_1 (9,)    │
│           │ ...             │
│           └ point_31(9,)    │
│  pillar_1 ┌ ...             │
│  ...                        │
└─────────────────────────────┘
```

同时还要存辅助信息，后面 scatter 时要用：

```text
coors:      (P, 3)   每根 pillar 的 (batch_idx, pillar_y, pillar_x)  ← 用来知道放回哪一格
num_points: (P,)     每根 pillar 真实点数  ← 用来算均值、剔除 padding
```

---

# 四、第二步：Point Decoration（点特征增强）

## 4.1 为什么不能直接用 (x, y, z, intensity)

原始点只有 4 维：

```text
(x, y, z, intensity)
```

如果直接送进网络，会有问题：

```text
全局坐标 x, y 数值很大（比如 x=45.3）
    ↓
对网络来说，同一个物体出现在 x=10 和 x=50
其绝对坐标完全不同 → 网络很难学
```

但我们真正想知道的其实是**相对位置**：

```text
这个点在这根 pillar 里的什么位置？
这个点相对于同 pillar 其他点的分布如何？
```

所以要做 **feature augmentation / decoration**。

## 4.2 9 维增强特征（论文的做法）

对某根 pillar 内的点集合，定义：

```text
(x, y, z, r)          ← 原始坐标 + 反射强度
xc = x - x_mean       ← 到本 pillar 内所有点均值中心的距离
yc = y - y_mean
zc = z - z_mean
xp = x - x_pillar_center    ← 到本 pillar 几何中心的 x 偏移
yp = y - y_pillar_center
```

其中：

```text
x_mean, y_mean, z_mean = 该 pillar 内所有真实点的算术平均
x_pillar_center = 该 pillar 格子的中心 x 坐标
y_pillar_center = 该 pillar 格子的中心 y 坐标
```

拼起来就是 **D = 9**：

```text
[ x, y, z, r, xc, yc, zc, xp, yp ]
  └─ 全局 ─┘ └─ 相对均值 ─┘ └─ 相对柱心 ─┘
```

具体数值举例：

```text
pillar 中心 = (12.32, 4.48)
该 pillar 内 3 个真实点：
  P1 = (12.30, 4.45, 0.31, 0.8)
  P2 = (12.35, 4.50, 1.62, 0.7)
  P3 = (12.28, 4.52, 0.90, 0.6)

均值 = ((12.30+12.35+12.28)/3, (4.45+4.50+4.52)/3, (0.31+1.62+0.90)/3)
     = (12.31, 4.49, 0.943)

对 P1：
  xc = 12.30 - 12.31 = -0.01
  yc = 4.45  - 4.49  = -0.04
  zc = 0.31  - 0.943 = -0.633
  xp = 12.30 - 12.32 = -0.02
  yp = 4.45  - 4.48  = -0.03

P1 的 9 维特征 = [12.30, 4.45, 0.31, 0.8, -0.01, -0.04, -0.633, -0.02, -0.03]
```

## 4.3 关键点：padding 点怎么处理

```text
真实点：用上式正常计算
padding 点：9 维全置 0
```

并且**算均值时只统计真实点**，否则 padding 的 0 会把均值拉偏。

这一点在实现里靠 `num_points` 来做：

```python
# 伪代码
num = num_points[v]                       # 本 pillar 真实点数
pts = pillar[v, :num, :3]                 # 只取真实点
center = pts.mean(axis=0)                 # 真实点均值
```

## 4.4 为什么加 xp, yp 不够，还要加 xc, yc, zc

```text
xp, yp：告诉网络"这根柱子在哪"（绝对位置，一格内的）
xc, yc, zc：告诉网络"这根柱子内部的几何形状"
```

举个例子，同样是"一根 pillar 里有 5 个点"：

```text
情况 A（一根细杆，比如电线杆）：      情况 B（一片平面，比如车顶）：
   x,y 都挤在一起                        x,y 散得很开
   z 拉得很长                            z 几乎一样

xc, yc ≈ 0,  zc 变化大               xc, yc 变化大,  zc ≈ 0
```

这两种几何形状，用 (xc, yc, zc) 就能区分开。这是 Pillar 内部**唯一的形状信息来源**——因为 z 轴被拍扁了，不补这个信息网络就不知道柱子里的点长什么样。

---

# 五、第三步：PillarFeatureNet（柱特征编码器）

这是 PointPillars 里唯一"接触原始点"的部分，本质是一个**简化版 PointNet**。

## 5.1 网络结构

```text
输入: (P, N, D) = (16000, 32, 9)
       │
       ▼
┌──────────────────────────┐
│ 1×1 Conv (等价于共享 Linear)│   D: 9 → 64
│ BatchNorm                 │
│ ReLU                      │
└──────────────────────────┘
       │
       ▼
(P, N, 64)
       │
       ▼
┌──────────────────────────┐
│ Max Pooling over N        │  在"点"这个维度上取最大
│ (沿 point 维度求 max)      │
└──────────────────────────┘
       │
       ▼
(P, 64)
       │
       ▼
输出: 每根 pillar → 一个 64 维向量
```

## 5.2 为什么用 1×1 Conv 而不是 Linear

两者的数学运算完全一样：

```text
Linear(D→64) 作用在 (P, N, D) 上：

   view 成 (P*N, D) → Linear → (P*N, 64) → view 回 (P, N, 64)

1×1 Conv(D→64) 作用在 (P, D, N) 上：

   直接支持任意 P、N，不用 reshape
```

工程上 1×1 Conv 更自然，因为后面的 BatchNorm 默认按 channel 归一化。

## 5.3 维度例子：完整走一遍

```text
输入 pillar tensor：

(P, N, D) = (16000, 32, 9)

        ↓ 维度置换 (transpose) 便于 1x1 Conv

(P, D, N) = (16000, 9, 32)

        ↓ Conv1d(9 → 64, kernel=1) + BN1d(64) + ReLU

(P, 64, 32) = (16000, 64, 32)

        ↓ MaxPool1d(kernel=32)   在 N=32 这个维度取 max

(P, 64, 1)

        ↓ squeeze

(P, 64) = (16000, 64)
```

代码对照：

```python
x = x.permute(0, 2, 1)          # (P, 9, 32)   把点维度挪到最后
x = self.conv1(x)               # (P, 64, 32)  Conv1d(9, 64, 1)
x = self.bn1(x)
x = self.relu(x)
x = self.conv2(x)               # (P, 64, 32)  Conv1d(64, 64, 1)
x = self.bn2(x)
x = self.relu(x)
x = torch.max(x, dim=2)[0]      # (P, 64)      沿点维度 max
```

> 注意：官方 repo 用的是**两层** 64 通道的 PFN；mmdet3d 默认只用一个 `feat_channels=(64,)`。差别不大，一层是标准配置。

## 5.4 为什么要 Max Pool

和 PointNet 一样的理由——**置换不变性（Permutation Invariance）**：

```text
pillar 内的点没有顺序：
  [P1, P2, P3] ≡ [P3, P1, P2]

max 满足：
  max(a, b, c) = max(c, a, b)
```

这是点云网络的基本要求。

另一个原因是：max 保留的是"这个 pillar 里最显著的特征"，对稀疏点云比较鲁棒（点少也不怕，只取最大值）。

---

# 六、第四步：Scatter → BEV 伪图像

## 6.1 做了什么

现在有：

```text
(P, 64)  = 16000 根 pillar，每根一个 64 维特征
coors: (P, 3) = 每根 pillar 的 (batch, y, x) 位置
```

要把它们按位置"撒"回 BEV 网格：

```text
        (16000, 64)
              │
              │ scatter by (batch, y, x)
              ▼
        (B, 64, 496, 432)
              │
              └── 这就是 BEV 伪图像（Pseudo Image）
```

## 6.2 为什么叫"伪图像"

因为形状和普通图像一模一样：

```text
普通 RGB 图像：  (3,  H,  W)     = (3, 496, 432)
BEV 伪图像：     (64, H,  W)     = (64, 496, 432)
```

区别只是：

```text
图像：每个像素是颜色
BEV：每个像素是"从地面到天上这一整柱点云"的特征
```

所以后面所有 2D 检测的套路都能直接搬过来用。

## 6.3 空 pillar 怎么办

大多数格子是空的（214272 个格子只有约 16000 个非空）：

```text
非空 pillar → 填入 64 维特征
空 pillar   → 全 0
```

网络会自动学会"全 0 = 这里没东西"。

## 6.4 一个经验性的检查

```text
214,272 个理论 pillar
  ↓ 实际非空（KITTI 一帧）
约 6,000 ~ 20,000 个
  ↓ max_num_pillars 截断
16,000 个（训练）/ 40,000 个（推理，官方 repo 设置）
```

非空率大约 **3% ~ 9%**，这就是 3D 点云的稀疏性，也是 PointPillars 能比 VoxelNet 快这么多的直接原因之一——它把稀疏的 3D 结构压成了稠密但**通道数很小**的 2D 图。

---

# 七、第五步：2D CNN Backbone

## 7.1 为什么可以直接用 2D CNN

因为现在数据已经是规则网格了：

```text
(B, 64, 496, 432)
```

这就是一张普通的多通道特征图，2D 卷积可以正常在 (y, x) 平面上滑动，提取：

```text
局部形状
目标轮廓
上下文关系
相邻目标的关系
```

## 7.2 网络结构（SECOND 风格的 Backbone，3 个 Block）

```text
输入 (B, 64, 496, 432)
       │
       ▼
┌─────────────────────────────────────┐
│ Block 1                             │
│  Conv(64→64,  k=3, s=2, p=1) + BN + ReLU   ← 下采样 2×
│  Conv(64→64,  k=3, s=1, p=1) + BN + ReLU   ┐
│  Conv(64→64,  k=3, s=1, p=1) + BN + ReLU   ├ 3 层
│  Conv(64→64,  k=3, s=1, p=1) + BN + ReLU   ┘
└─────────────────────────────────────┘
       │
       ▼
(B, 64, 248, 216)      stride 2
       │
       ▼
┌─────────────────────────────────────┐
│ Block 2                             │
│  Conv(64→128, k=3, s=2, p=1) + BN + ReLU   ← 再次下采样
│  Conv(128→128, k=3, s=1, p=1) + BN + ReLU  ┐
│  ... 共 5 层                                ┘
└─────────────────────────────────────┘
       │
       ▼
(B, 128, 124, 108)     stride 4
       │
       ▼
┌─────────────────────────────────────┐
│ Block 3                             │
│  Conv(128→256, k=3, s=2, p=1) + BN + ReLU  ← 再次下采样
│  Conv(256→256, k=3, s=1, p=1) + BN + ReLU  ┐
│  ... 共 5 层                                ┘
└─────────────────────────────────────┘
       │
       ▼
(B, 256, 62, 54)       stride 8
```

卷积的尺寸计算：

```text
H_out = floor((H_in + 2p - k) / s) + 1
```

代入 `k=3, p=1, s=2`：

```text
496 → floor((496 + 2 - 3) / 2) + 1 = floor(495/2) + 1 = 247 + 1 = 248
432 → floor((432 + 2 - 3) / 2) + 1 = floor(431/2) + 1 = 215 + 1 = 216
```

> 注意这里**不能直接整除**：431/2 = 215.5，向下取整是 215 而不是 216，
> 加 1 之后才恰好等于 216。尺寸是偶数时容易算错，按公式走。

所以 `(496, 432) → (248, 216)`。同理：

```text
(248, 216) → (124, 108)
(124, 108) → (62, 54)
```

## 7.3 FPN Neck：把多尺度拼起来

三个尺度的特征图尺寸不同，需要上采样对齐：

```text
Block 1 输出   (64,  248, 216)   stride 2   ──────────────┐
                                                          │
Block 2 输出   (128, 124, 108)   stride 4   ──Deconv×2──→ (128, 248, 216) ─┤
                                                          │
Block 3 输出   (256,  62,  54)   stride 8   ──Deconv×4──→ (128, 248, 216) ─┤
                                                          │
                                                          ▼
                                            Concat → (64+128+128, 248, 216)
                                                   = (320, 248, 216)
```

mmdet3d 里用 `out_channels=[128,128,128]`，所以是：

```text
Block 1 → 1×1 Conv 调到 128   → (128, 248, 216)
Block 2 → Deconv(s=2) 到 128  → (128, 248, 216)
Block 3 → Deconv(s=4) 到 128  → (128, 248, 216)
                       ↓
             Concat → (384, 248, 216)
```

> 这和你写的 [[FPN]] 笔记是一个思路：**深层语义 + 浅层细节，上采样后相加/拼接**。
> 区别是 FPN 走的是 top-down + 逐元素相加，PointPillars 的 SECONDFPN 是**各自上采样到同一尺度再 concat**。

## 7.4 为什么最后停在 stride 2 而不是 stride 1

```text
stride 1 (496 × 432)  →  太贵，anchors 数量爆炸
stride 2 (248 × 216)  →  一个格子 0.32m × 0.32m，对车/人足够精细
stride 4 或 8         →  0.64m / 1.28m，小目标（行人）定位会太糙
```

KITTI 上 `out_size_factor=2`，也就是检测头工作在 **1/2 分辨率**。

---

# 八、第六步：Detection Head

## 8.1 结构

PointPillars 用的是 **SSD 风格的单阶段 Anchor Head**，而且**每个类别一个独立的头**。

```text
FPN 输出 (B, 384, 248, 216)
       │
       ├───────────────┬───────────────┐
       ▼               ▼               ▼
   Car Head      Pedestrian Head   Cyclist Head
       │               │               │
       ├─ cls: Conv3x3(384→256) → 1×1 → (2, 248, 216)
       │                                 ↑ 2 个 anchor（0° 和 90°）
       │
       └─ reg: Conv3x3(384→256) → 1×1 → (14, 248, 216)
                                         ↑ 2 anchor × 7 个回归量
```

把三个类的输出拼起来：

```text
分类输出:   (B, 3 类 × 2 anchor, 248, 216)  = (B,  6, 248, 216)
回归输出:   (B, 3 类 × 2 anchor × 7, 248, 216) = (B, 42, 248, 216)
```

## 8.2 Anchor 数量算一下

```text
每类 anchor 数 = 248 × 216 × 2 = 107,136
总 anchor 数   = 107,136 × 3     = 321,408

回归输出元素数 = 321,408 × 7 = 2,249,856
```

三十多万个 anchor，听起来恐怖，但这是 2D 的，而且全部是**向量化算的**——这就是它能跑到 62 FPS 的原因。对比一下 VoxelNet 是 3D anchor，量级完全不同。

## 8.3 KITTI 的 Anchor 定义（具体数值）

```text
anchor 尺寸 (l, w, h) 和 z 位置：

Car         : l=3.9,  w=1.6,  h=1.56,  z=-1.78
Pedestrian  : l=0.8,  w=0.6,  h=1.73,  z=-0.60
Cyclist     : l=1.76, w=0.6,  h=1.73,  z=-0.60

旋转角 rotations = [0°, 90°]      ← 每个位置 2 个 anchor
```

> `z` 是**框底面的高度**，不是中心高度。
> 比如 Car 的 z=-1.78，h=1.56 → 中心 z = -1.78 + 1.56/2 = -1.0。

为什么只有两个旋转角？

```text
0°   → 车头朝 x 正方向（朝前）
90°  → 车头朝 y 方向（横着）
```

KITTI 里车基本都是这两种朝向，用 2 个 anchor 就够；nuScenes 里要精细得多（后面第十七节讲）。

## 8.4 Anchor 是怎么"铺"出来的

在 248 × 216 的每个格子上，按 `out_size_factor=2` 映射回 BEV 物理坐标：

```text
格子 (i, j) 的中心在 BEV 上的坐标：

x_center = x_min + (j + 0.5) × 0.16 × 2 = x_min + (j + 0.5) × 0.32
y_center = y_min + (i + 0.5) × 0.16 × 2 = y_min + (i + 0.5) × 0.32
```

每个位置放 3 类 × 2 个旋转 = 6 个 anchor。

## 8.5 回归目标（7 个量）

对每个 anchor，网络预测 7 个偏移：

```text
dx, dy, dz, dl, dw, dh, dθ
```

**编码公式**（SECOND 的 parameterization）：

```text
d_a = sqrt(w_a² + l_a²)

dx     = (x_gt     - x_a) / d_a
dy     = (y_gt     - y_a) / d_a
dz     = (z_gt     - z_a) / h_a

dl     = log(l_gt / l_a)
dw     = log(w_gt / w_a)
dh     = log(h_gt / h_a)

dθ     = sin(θ_gt - θ_a)          ← 用 sin 自动处理角度周期性
```

**解码**：

```text
x = dx × d_a + x_a
y = dy × d_a + y_a
z = dz × h_a + z_a

l = exp(dl) × l_a
w = exp(dw) × w_a
h = exp(dh) × h_a

θ = dθ + θ_a
```

注意几个设计细节：

```text
① 用 d_a = sqrt(w²+l²) 归一化 x,y 偏移
   → 让大车的偏移量和小车的偏移量处于同一量级

② 尺寸用 log 空间
   → 保证 exp() 出来永远是正数，且对小尺寸更敏感

③ 角度用 sin()
   → sin(θ+2π) = sin(θ)，避免 ±π 附近的不连续
```

---

# 九、Loss

```text
L = L_cls + L_reg
```

## 9.1 分类 Loss：Focal Loss

```text
FL(p_t) = -α (1 - p_t)^γ log(p_t)

α = 0.25
γ = 2.0
```

为什么用 Focal Loss？

```text
32 万个 anchor，正样本可能只有几十个
正负样本比 ≈ 1 : 10000
```

普通的 CrossEntropy 会被负样本淹没，Focal Loss 通过 `(1-p_t)^γ` 把"容易分的样本"的 loss 压下去，让网络专注在难样本上。

## 9.2 回归 Loss：Smooth L1

```text
SmoothL1(x) = 0.5 x²          if |x| < 1
              |x| - 0.5       otherwise
```

只对**正样本**算回归 loss。

## 9.3 正负样本划分

```text
Anchor 与 GT 的 BEV IoU：

IoU > 0.6   → 正样本
IoU < 0.45  → 负样本（忽略 0.45 ~ 0.6 之间的）
```

在 KITTI 上，还会先过滤掉：

```text
① 超出 point_cloud_range 的 anchor        ← 大部分都在这被滤掉
② 旋转 IoU 与某个 GT < 0.1 的 anchor       ← 再过滤一大片
```

这也是为什么 32 万个 anchor 实际能算得动——**大部分在预处理阶段就被剔除了**。

---

# 十、后处理

```text
网络输出 (B, 6, 248, 216) 和 (B, 42, 248, 216)
       │
       ▼
① Decode：用 §8.5 的解码公式还原成真实 3D Box
       │
       ▼
② 按类别取 sigmoid / softmax 得到 confidence
       │
       ▼
③ 置信度阈值过滤（比如 score > 0.1）
       │
       ▼
④ 范围/尺寸过滤（去掉超出 point_cloud_range、尺寸不合理的框）
       │
       ▼
⑤ Rotated NMS
       │   KITTI 用的 IoU 阈值很小，比如 0.01（因为车很密集，怕误删）
       │   nuScenes 通常 0.2
       ▼
⑥ 每张图保留最多 100 个框
       │
       ▼
最终 3D Bounding Boxes
```

**Rotated NMS 和普通 NMS 的区别**：IoU 要在**旋转框**上算，不是轴对齐矩形。这是在 BEV 平面（只有 yaw 一个旋转角）上算的，比 3D 旋转 IoU 简单。

---

# 十一、完整维度流转表（KITTI，背下来）

这是最值得记住的一张表。

```text
阶段                                   形状                      说明
──────────────────────────────────────────────────────────────────────────────
① 原始点云                              (N, 4)                    N ≈ 100000
                                                                  [x,y,z,intensity]

② 硬编码到内存 buffer、
   按 pillar 归组                        (P, N_p, D)               P = max_num_pillars
                                                                  N_p = max_points_per_pillar
                                                                  D = 9（增强后）
                                        (16000, 32, 9)

③ 辅助信息                              coors:      (P, 3)        (batch, pillar_y, pillar_x)
                                        num_points: (P,)

④ PillarFeatureNet - 维度置换            (P, 9, 32)

⑤ PFN Conv1d(9→64)+BN+ReLU              (P, 64, 32)

⑥ PFN MaxPool over N                    (P, 64, 1) → (P, 64)

⑦ Scatter 到 BEV 网格                    (B, 64, 496, 432)         496 = H(y), 432 = W(x)

⑧ Backbone Block1 (s=2)                 (B, 64, 248, 216)

⑨ Backbone Block2 (s=2)                 (B, 128, 124, 108)

⑩ Backbone Block3 (s=2)                 (B, 256, 62, 54)

⑪ FPN: 三级各自上采样到 248×216 后 concat  (B, 384, 248, 216)       128 × 3 = 384

⑫ Head - 分类                           (B, 6, 248, 216)         3 类 × 2 anchor

⑬ Head - 回归                           (B, 42, 248, 216)        3 类 × 2 anchor × 7

⑭ Decode + NMS                          (num_boxes, 7 + 1)       [x,y,z,l,w,h,yaw, score]
```

## 记忆抓手

```text
496 × 432  ← BEV 网格（y × x）
      ↓ ÷2
248 × 216  ← 检测头工作分辨率
      ↓

通道：64 → 64 → 128 → 256 → 384
      ↑        ↑     ↑
   scatter   Block1 Block2 Block3
```

---

# 十二、第二个维度例子：nuScenes

KITTI 是单帧 LiDAR，nuScenes 是 32 线 / 360°，配置完全不同。对比着看，能更清楚哪些是"设计"、哪些是"配置"。

```text
                        KITTI                    nuScenes
─────────────────────────────────────────────────────────────────────
point_cloud_range    [0,-39.68,-3,              [-51.2,-51.2,-5,
                       69.12,39.68,1]             51.2,51.2,3]

pillar_size          (0.16, 0.16)               (0.2, 0.2)

BEV 网格             432 × 496                 512 × 512
                     (x × y)                   (x × y)

max_points_per_pillar  32                      20

max_num_pillars      16000 / 40000             30000 / 40000
                     (train / eval)            (train / eval)

scatter 输出         (B, 64, 496, 432)         (B, 64, 512, 512)

Block1 输出          (B, 64, 248, 216)         (B, 64, 256, 256)
Block2 输出          (B, 128, 124, 108)        (B, 128, 128, 128)
Block3 输出          (B, 256, 62, 54)          (B, 256, 64, 64)

FPN 输出             (B, 384, 248, 216)        (B, 384, 256, 256)

检测头 anchor 数      248×216×6 = 321,408       256×256×6 = 393,216

类别                 3（Car/Ped/Cyclist）       10（car/truck/bus/
                                                 trailer/construction/
                                                 pedestrian/motorcycle/
                                                 bicycle/barrier/traffic_cone）

anchor rotations     [0°, 90°]                 [0°, 90°]（2 个）

loss                 Focal + SmoothL1           Focal + SmoothL1
```

注意几个变化点：

```text
① pillar_size 从 0.16m 变成 0.2m
   → nuScenes 范围大得多（102.4m vs 69.12m），格子太细 anchor 数会爆

② max_points_per_pillar 从 32 降到 20
   → nuScenes 点更稠密，但不希望单根 pillar 太贵

③ z 范围从 -3~1 变成 -5~3
   → nuScenes 的 32 线雷达更低，也有更高的目标（卡车）

④ 类别从 3 变 10
   → anchor 定义要按数据集的尺寸统计重新设计（这是很关键的一步）
```

## 一个实操提醒

换个数据集，**最先要调的往往不是网络结构，而是这三样**：

```text
① point_cloud_range    ← 决定 BEV 覆盖多大
② pillar_size          ← 决定 BEV 分辨率（进而决定 anchor 数）
③ anchor sizes/z       ← 用训练集的 GT 统计出来（k-means 或直接均值）
```

网络主体（PFN + Backbone + FPN + Head）几乎不用动。

---

# 十三、网络设计的几个关键决策（为什么这么设计）

这一节是"设计意图"，比记结构更重要。

## 决策 1：为什么用 Pillar 而不是 Voxel

```text
Voxel:  (dx, dy, dz) 三维剖分 → 3D 卷积 → 慢
Pillar: (dx, dy) 只在 BEV 剖分 → 2D 卷积 → 快

代价：丢了 z 方向的精细结构
      → 靠 Point Decoration 的 (xc, yc, zc) 补救
```

**本质是一次"精度换速度"的交易**，而且这笔交易很赚——KITTI 上 3D AP 几乎没降，速度涨了 3 倍。

## 决策 2：为什么 PillarFeatureNet 这么浅

```text
输入 (P, N, D)
→ 一层 1×1 Conv
→ max pool
→ 输出 (P, 64)
```

对比一下 PointNet++ 的 SA 模块，PillarFeatureNet 浅得多。原因是：

```text
① Pillar 内点数很少（N=32），且已经被 padding 成规则张量，
   不需要 FPS + ball query 那套复杂的采样分组

② 真正的特征提取交给后面的 2D CNN
   PFN 只需要做一个"把点集压成一个向量"的编码器

③ 要快 —— PFN 是整条流水线里唯一必须处理全部 100000 个点的部分
```

分工是：

```text
PFN：      点级 → 柱级   （局部编码）
2D CNN：   柱级 → 目标级 （空间推理）
```

## 决策 3：为什么用 Anchor 而不是 Anchor-Free

PointPillars（2019）还是 anchor-based 时代。它的好处：

```text
① 用先验尺寸，回归目标小，容易收敛
② 训练稳定，不需要复杂的正样本分配策略
```

代价：

```text
① anchor 超参要按数据集调
② 密集场景下正样本重叠，NMS 压力大
③ nuScenes 上 10 个类别尺寸差异大，anchor 设计很麻烦
```

这就是后来 [[CenterPoint]]（anchor-free，中心点 + 回归）、[[BEVFormer]] 这类方法出现的原因之一。

## 决策 4：为什么 FPN 要 concat 到 stride 2

```text
小目标（行人，0.8m × 0.6m）：
   stride 2 时占 248×216 里约 2.5 × 1.9 个格子  → 能看清
   stride 8 时只占 0.6 × 0.5 个格子             → 直接丢了
```

而深层特征（stride 8）语义强、感受野大，对大车很重要。

**所以必须把两者融合**——这就是 FPN 存在的意义，和你写的 [[FPN]] 笔记完全一致。

## 决策 5：为什么每类一个独立 Head

```text
Car:        3.9 × 1.6 × 1.56
Pedestrian: 0.8 × 0.6 × 1.73
Cyclist:    1.76 × 0.6 × 1.73
```

尺寸差异太大（车长 3.9m vs 人长 0.8m，差 5 倍）。如果共享一个 head + 所有 anchor：

```text
回归目标 dl = log(l_gt / l_a) 的分布会非常分散
→ 训练困难
```

每类独立 head + 独立 anchor 定义，能让每类的回归目标都集中在 0 附近。

---

# 十四、PyTorch 骨架代码

结合上面所有维度，把关键部分写出来。

## 14.1 PillarFeatureNet

```python
import torch
import torch.nn as nn


class PillarFeatureNet(nn.Module):
    """(P, N, D) -> (P, 64)"""

    def __init__(self, in_channels=9, feat_channels=(64,)):
        super().__init__()
        self.in_channels = in_channels
        channels = (in_channels,) + tuple(feat_channels)   # (9, 64)

        layers = []
        for i in range(len(feat_channels)):
            layers.extend([
                nn.Conv1d(channels[i], channels[i + 1], kernel_size=1, bias=False),
                nn.BatchNorm1d(channels[i + 1]),
                nn.ReLU(inplace=True),
            ])
        self.pfn_layers = nn.Sequential(*layers)

    def forward(self, features, num_points, coors):
        """
        features:   (P, N, D)  = (16000, 32, 9)
        num_points: (P,)       每根 pillar 的真实点数
        coors:      (P, 3)     (batch_idx, y, x)

        返回 (P, C) 和 coors
        """
        P, N, D = features.shape

        # ---- 1. 计算均值，只统计真实点（排除 padding）----
        # 生成 mask: (P, N)
        mask = torch.arange(N, device=features.device)[None, :] < num_points[:, None]
        mask = mask.unsqueeze(-1).float()                       # (P, N, 1)

        # 真实点之和 / 真实点数
        pts_sum = (features[..., :3] * mask).sum(dim=1)          # (P, 3)
        pts_mean = pts_sum / mask.sum(dim=1).clamp(min=1)        # (P, 3)

        # ---- 2. 计算 pillar 中心的 x,y ----
        voxel_size = features.new_tensor([0.16, 0.16])           # 来自配置
        pc_range = features.new_tensor([0.0, -39.68])
        # 左上角 + (grid + 0.5) * voxel_size
        pillar_xy = coors[:, [2, 1]].float() * voxel_size + pc_range + voxel_size / 2
        #                                                     (P, 2)

        # ---- 3. Point Decoration，拼成 9 维 ----
        xyz = features[..., :3]                                  # (P, N, 3)
        xc_yc_zc = xyz - pts_mean[:, None, :]                    # (P, N, 3)
        xp_yp = xyz[..., :2] - pillar_xy[:, None, :]             # (P, N, 2)

        features = torch.cat(
            [features, xc_yc_zc, xp_yp], dim=-1
        )                                                        # (P, N, 9)

        # padding 点置零
        features = features * mask

        # ---- 4. PFN ----
        features = features.permute(0, 2, 1)                     # (P, 9, 32)
        features = self.pfn_layers(features)                     # (P, 64, 32)
        features = torch.max(features, dim=2)[0]                 # (P, 64)

        return features, coors
```

## 14.2 Scatter → BEV

```python
class PointPillarsScatter(nn.Module):
    """(P, C) + coors -> (B, C, H, W)"""

    def __init__(self, in_channels=64, output_shape=(496, 432)):
        super().__init__()
        self.in_channels = in_channels
        self.ny, self.nx = output_shape
        self.register_buffer(
            'canvas', torch.zeros(1, in_channels, self.ny, self.nx), persistent=False
        )

    def forward(self, x, coors):
        """
        x:     (P, C)
        coors: (P, 3) = (batch_idx, y, x)
        """
        B = int(coors[:, 0].max().item()) + 1
        canvas = self.canvas.repeat(B, 1, 1, 1).clone()

        # 展平成索引，一次性 scatter
        batch_idx = coors[:, 0].long()
        y = coors[:, 1].long()
        x_idx = coors[:, 2].long()

        canvas[batch_idx, :, y, x_idx] = x.float()
        return canvas                                          # (B, C, 496, 432)
```

## 14.3 Backbone + FPN

```python
class SECONDBlock(nn.Module):
    """Conv(s=2) 下采样 + 若干 Conv(s=1)"""

    def __init__(self, in_ch, out_ch, num_layers, stride=2):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
                  nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 1):
            layers += [nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
                       nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SECOND(nn.Module):
    def __init__(self, in_channels=64, layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2), out_channels=(64, 128, 256)):
        super().__init__()
        blocks, ch = [], in_channels
        for n, s, o in zip(layer_nums, layer_strides, out_channels):
            blocks.append(SECONDBlock(ch, o, n, s))
            ch = o
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outs = []
        for blk in self.blocks:
            x = blk(x)
            outs.append(x)
        return outs          # [(B,64,248,216), (B,128,124,108), (B,256,62,54)]


class SECONDFPN(nn.Module):
    def __init__(self, in_channels=(64, 128, 256),
                 upsample_strides=(1, 2, 4), out_channels=(128, 128, 128)):
        super().__init__()
        deblocks = []
        for i, u in zip(in_channels, upsample_strides):
            deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(i, out_channels[0], 2 * u, u, u // 2, bias=False),
                nn.BatchNorm2d(out_channels[0]),
                nn.ReLU(inplace=True),
            ))
        self.deblocks = nn.ModuleList(deblocks)

    def forward(self, xs):
        ups = [de(x) for de, x in zip(self.deblocks, xs)]
        # 尺寸对齐（奇数尺寸时会有 1 像素差）
        h = min(u.shape[2] for u in ups)
        w = min(u.shape[3] for u in ups)
        ups = [u[:, :, :h, :w] for u in ups]
        return torch.cat(ups, dim=1)      # (B, 384, 248, 216)
```

## 14.4 完整前向

```python
class PointPillars(nn.Module):
    def __init__(self):
        super().__init__()
        self.pfn = PillarFeatureNet(in_channels=9, feat_channels=(64,))
        self.scatter = PointPillarsScatter(64, output_shape=(496, 432))
        self.backbone = SECOND(64, (3, 5, 5), (2, 2, 2), (64, 128, 256))
        self.neck = SECONDFPN((64, 128, 256), (1, 2, 4), (128, 128, 128))
        # head 略

    def forward(self, points):
        # 1. 体素化（实际在 dataset 里做，这里示意）
        features, coors, num_points = voxelize(points)
        # features:  (P, 32, 9)
        # coors:     (P, 3)
        # num_points:(P,)

        # 2. PFN
        x, coors = self.pfn(features, num_points, coors)   # (P, 64)

        # 3. Scatter
        x = self.scatter(x, coors)                          # (B, 64, 496, 432)

        # 4. Backbone
        xs = self.backbone(x)                               # 3 个尺度

        # 5. FPN
        x = self.neck(xs)                                   # (B, 384, 248, 216)

        # 6. Head（略）
        return x
```

---

# 十五、训练配置（KITTI 参考值）

```text
数据集          KITTI 3D Object Detection
                (7481 训练帧 / 7518 测试帧，只用 LiDAR)

point_cloud_range   [0, -39.68, -3, 69.12, 39.68, 1]
pillar_size         (0.16, 0.16, 4)
max_points_per_pillar  32
max_num_pillars     16000（训练）/ 40000（推理）

优化器          Adam / AdamW
学习率          ≈ 2e-4
weight_decay    ≈ 0.01
batch_size      ≈ 48（8 卡 × 6）
epochs          160
梯度裁剪         max_norm ≈ 35

Loss            Focal Loss (cls) + SmoothL1 (reg)
NMS IoU 阈值     0.01（BEV 旋转 IoU）
保留框数         最多 100 / 帧
```

## 数据增强

```text
① GT-Sampling（最重要）
   把其他帧里的 GT 目标"粘贴"到当前帧的 LiDAR 里
   → 为什么：KITTI 每帧目标太少（平均不到 15 个），
              靠粘贴把每帧目标数提到 15 个以上，大幅提升小目标

② 全局翻转       x 轴 / y 轴随机翻转
③ 全局旋转       绕 z 轴随机旋转 [-π/4, π/4]
④ 全局缩放       [0.95, 1.05]
⑤ 全局平移       (0.2, 0.2, 0.2) 米范围内
⑥ 逐 GT 旋转/平移 
⑦ 点云 dropout   （可选）
```

> 注：增强必须在**点云级别**做，不能对 BEV 特征图做——因为特征图已经丢失了 z 信息。

---

# 十六、性能与对比

## 16.1 KITTI test 3D AP（论文数值）

```text
                     Easy    Mod     Hard
Car                  79.09   74.99   68.82
Pedestrian           52.08   46.42   42.08
Cyclist              75.78   59.07   56.65
(mAP)                68.98   60.16   55.85

BEV AP:
Car                  88.35   86.10   79.83
```

## 16.2 速度对比（V100）

```text
方法             FPS      
────────────────────────────
VoxelNet         4.4
SECOND          20
PointPillars    62           ← PyTorch/TensorRT
PointPillars   105           ← TensorRT FP16（论文里的上限）
```

## 16.3 方法对比

| 方法 | Point 处理 | 空间表示 | Backbone | 特点 |
|---|---|---|---|---|
| PointNet++ | 直接 Point | Hierarchical Point | PointNet | 直接提点特征，慢 |
| VoxelNet | Point → Voxel | 3D Voxel | 3D CNN | 精度高，慢 |
| SECOND | Point → Sparse Voxel | Sparse Voxel | Sparse CNN | 去掉空体素，快 5× |
| **PointPillars** | **Point → Pillar** | **BEV 2D** | **2D CNN** | **快 3×，精度几乎不掉** |
| CenterPoint | Point → 中心点 | BEV | 2D CNN | Anchor-free |
| PV-RCNN | Point + Voxel | Hybrid | 3D + Point | 精度最高，慢 |

## 16.4 PointPillars 的优缺点

**优点**：

```text
① 快 —— 全 2D 卷积，TensorRT 友好，适合车端部署
② 简单 —— 结构清晰，没有 3D 稀疏卷积那套复杂实现
③ 通用 —— 换成 4D 雷达、深度图等其他 BEV 数据也容易改造
④ 没有 3D 卷积/稀疏卷积 → 端侧和嵌入式平台部署门槛低
```

**缺点**：

```text
① 丢 z 信息 —— 靠 (xc,yc,zc) 补救，但对"高度上重叠的物体"仍然弱
   （比如高架桥上的车 vs 桥下的车，BEV 上是同一点 → 分不开）
② Anchor-based —— 超参要按数据集调，10 类的 nuScenes 很麻烦
③ 小目标弱 —— 行人、锥桶这类，KITTI 上 Pedestrian AP 只有 52
④ BEV 分辨率/范围二选一 —— pillar 太细 anchor 爆炸，太粗小目标丢
```

---

# 十七、常见坑

```text
坑 1：算均值时没排除 padding 点
    → 均值被 0 拉偏，xc/yc/zc 全错，训练不收敛
    → 必须用 num_points 生成 mask

坑 2：pillar 中心和体素中心搞混
    → 有 3 套坐标：全局 (x,y)、pillar 索引 (i,j)、pillar 中心
    → 官方约定：pillar 中心 = 左上角 + (grid + 0.5) × voxel_size
    → 注意 KITTI 的 point_cloud_range 左上角是 [0, -39.68]，不是 [0, 0]

坑 3：coors 里 (y, x) 的顺序
    → mmdet3d 的 coors 是 (batch_idx, z_idx, y_idx, x_idx)
    → 但对应到特征图 (C, H=ny, W=nx)，即 H 对应 y，W 对应 x
    → 转置错了后面 anchor 会整体转 90°

坑 4：卷积尺寸算不对（496/2 = 248，但 432/2 要小心）
    → 用 (H + 2p - k)/s + 1，别直接整除
    → FPN 里不同尺度的上采样结果可能差 1 个像素，要裁齐

坑 5：max_num_pillars 设太小
    → 随机丢弃大量 pillar，远处目标直接消失
    → KITTI 训练 16000 够用，nuScenes 要 30000+

坑 6：anchor 的 z 是底面不是中心
    → Car: z=-1.78, h=1.56 → 中心在 z=-1.0
    → 这个搞错了回归目标 dz 会有系统性偏差

坑 7：换数据集直接套 KITTI 的 anchor
    → nuScenes 的 car 是 4.63m 长，KITTI 是 3.9m
    → 必须重新统计
```

---

# 十八、和 VoxelNet / SECOND / CenterPoint 的关系

```text
VoxelNet (2018)
   │  开创：Point → Voxel → VFE → 3D CNN
   │  问题：3D 卷积太慢（4.4 FPS）
   ▼
SECOND (2018)
   │  改进：Sparse Convolution，跳过空体素
   │  结果：20 FPS
   │  遗留：仍然是 3D 卷积
   ▼
PointPillars (2019)  ★
   │  改进：把 z 拍扁 → BEV 2D 伪图像 → 纯 2D 卷积
   │  结果：62 FPS，精度几乎不降
   │  副作用：这套"BEV + 2D CNN"成了后续所有方法的标配
   ▼
CenterPoint (2021)
   │  改进：Anchor-based → Anchor-free（中心点热力图）
   ▼
BEVFormer / BEVFusion (2022)
   │  改进：把相机也投到同一个 BEV 空间做多模态融合
   ▼
     BEV Perception 时代
```

关键的承接关系：

```text
PointPillars 有两个贡献：

① 效率贡献：证明了"3D 检测不需要 3D 卷积"
   → 这个思想直接启发了后面的所有 BEV 方法

② 表示贡献：把 "BEV + 2D 检测头" 这个范式固定下来了
   → BEVFormer、BEVFusion、BEVDet 全都在用这个范式
   → 你现在学的 [[BEVFormer]] 里的 BEV Feature，本质就是 PointPillars 的伪图像
```

所以 PointPillars 是**承上启下**的一篇：

```text
上承：VoxelNet / SECOND 的"如何把点云规则化"
下启：BEV 感知的整个时代
```

---

# 十九、和你现有学习路线的关系

```text
PointNet++    → 怎么直接在点上提特征
VoxelNet      → 怎么把点云变成 3D Voxel
PointPillars  → 怎么把 Voxel 再降维成 BEV  ← 你现在在这
BEVFormer     → 怎么把相机特征也投影到 BEV
BEVFusion     → 怎么在 BEV 上融合多模态
```

可以这样总结表示方式的演进：

```text
              z 信息保留程度         计算量
PointNet++    ★★★★★（原始点）       ★★★★★
VoxelNet      ★★★★（3D Voxel）      ★★★★
SECOND        ★★★★（稀疏 3D）       ★★★
PointPillars  ★（拍扁 + 补偿）       ★          ← 甜点
CenterPoint   ★（拍扁）              ★
```

> **PointPillars 最值得学的地方，是它做的那笔"交易"：**
> 主动丢掉 z 方向的精细结构，换来 10 倍速度，而且精度只掉一点点。
> 想清楚"这笔交易为什么能成立"，比背下网络结构有价值得多。

---

# 二十、一句话记忆

> **PointPillars = 把 LiDAR 点云按 BEV 网格切成一根根竖直的柱子，用 PointNet 式的 1×1 Conv + MaxPool 把每根柱子编码成一个 64 维向量，按位置撒成 BEV 伪图像，再用带 FPN 的 2D CNN + Anchor Head 做 3D 检测。**

流程一句话：

```text
Pillar 化 → 点特征增强(9维) → PFN(64维) → Scatter → 2D CNN → FPN(384) → Head → NMS
```

维度一句话：

```text
(N,4) → (P,N,9) → (P,64) → (64,496,432) → (64,248,216) → (384,248,216) → (6/42,248,216)
```

---

# 二十一、相关笔记

- [[VoxelNet]] —— 上一代，3D Voxel + VFE + 3D CNN
- [[FPN]] —— 多尺度特征融合，PointPillars 的 Neck 是同一个思路
- [[BEVFormer]] —— BEV 表示的后续发展，把相机也投到 BEV
- [[PointNet++]] —— 点云直接提特征的路线

对应文件：

```text
3D/VoxelNet.md
3D/BEVFormer.md
3D/pointnet++.py
2D/FPN.md
```
