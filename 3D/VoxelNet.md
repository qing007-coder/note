# VoxelNet 完整流程

> VoxelNet：一种经典的 LiDAR 点云 3D 目标检测网络。
>
> 核心思想：
>
> **Point Cloud → Voxelization → VFE → Sparse 3D CNN → RPN → 3D Bounding Box**
>
> 它解决的核心问题是：
>
> > 如何把无规则、稀疏的 LiDAR 点云，转换成神经网络可以高效处理的规则 3D 特征。

---

# 一、先看整体流程

VoxelNet 可以理解成下面这条流水线：

```text
LiDAR Point Cloud
       │
       ▼
┌─────────────────────┐
│   Voxelization      │
│ 点云 → 3D Voxel      │
└─────────────────────┘
       │
       ▼
每个 Voxel 内包含若干 Point
       │
       ▼
┌─────────────────────┐
│       VFE           │
│ Voxel Feature        │
│ Encoding             │
└─────────────────────┘
       │
       ▼
每个 Voxel → 一个特征向量
       │
       ▼
重新组织成 3D Feature Map
       │
       ▼
┌─────────────────────┐
│   3D Convolution    │
│   / 3D Middle CNN   │
└─────────────────────┘
       │
       ▼
高层 3D 空间特征
       │
       ▼
┌─────────────────────┐
│        RPN          │
│ Region Proposal      │
│ Network              │
└─────────────────────┘
       │
       ├──────────────► 分类 Classification
       │
       └──────────────► 回归 Box Regression
                              │
                              ▼
                     3D Bounding Boxes
```

---

# 二、VoxelNet 到底解决什么问题？

LiDAR 原始数据通常是：

```text
(x, y, z, intensity)
```

例如：

```text
Point 1 = (12.3, 4.5, 1.2, 0.8)
Point 2 = (12.5, 4.7, 1.3, 0.7)
Point 3 = (20.1, 8.2, 0.4, 0.3)
...
```

问题是：

> 点云不是规则排列的图像。

图像可以直接表示成：

```text
H × W × C
```

例如：

```text
640 × 480 × 3
```

每个像素的位置是固定的。

但是点云：

```text
(x1, y1, z1)
(x2, y2, z2)
(x3, y3, z3)
...
```

点的数量不固定，空间分布也不规则。

所以不能简单地：

```text
Point Cloud → CNN
```

VoxelNet 的解决方法就是：

> **先把 3D 空间划分成很多规则的小立方体 Voxel。**

这样就可以把无规则点云变成规则的 3D 空间结构。

---

# 三、第一步：输入 Point Cloud

假设 LiDAR 得到：

```text
N × 4
```

的数据：

```text
[x, y, z, intensity]
```

例如：

```text
[
    [1.2, 2.3, 0.5, 0.8],
    [1.3, 2.4, 0.6, 0.7],
    [5.1, 3.2, 1.0, 0.4],
    ...
]
```

其中：

```text
x        → 前后方向
y        → 左右方向
z        → 高度
intensity → 激光反射强度
```

假设一帧 LiDAR 有：

```text
100000 个点
```

那么输入就是：

```text
100000 × 4
```

---

# 四、第二步：Voxelization

## 4.1 什么是 Voxel？

Voxel = Volumetric Pixel

可以理解为：

> **3D 空间中的“小立方体像素”。**

二维图像：

```text
Pixel

┌───┬───┬───┐
│   │   │   │
├───┼───┼───┤
│   │   │   │
├───┼───┼───┤
│   │   │   │
└───┴───┴───┘
```

3D 空间：

```text
Voxel

      ┌──────┐
     /      /│
    ┌──────┐ │
    │      │ │
    │      │/
    └──────┘
```

也就是：

```text
Pixel → 2D
Voxel → 3D
```

---

# 五、Voxelization 怎么做？

假设我们定义：

```text
Voxel Size：

vx = 0.2 m
vy = 0.2 m
vz = 0.4 m
```

意思是：

```text
一个 Voxel：

0.2m × 0.2m × 0.4m
```

然后把整个 3D 空间切成很多小立方体。

例如：

```text
                 z
                 ↑
                 │
          ┌──────┼──────┐
         /│     /│     /│
        ┌─┼─────┐│    / │
        │ │     ││   /  │
        │ └─────┼┘  /   │
        │/      │  /    │
        └───────┘───────┘ → x
       /
      y
```

每个点根据自己的：

```text
(x, y, z)
```

确定自己属于哪个 Voxel。

---

# 六、点如何确定属于哪个 Voxel？

假设：

```text
Voxel Size = (vx, vy, vz)
```

对于一个点：

```text
(x, y, z)
```

可以计算：

```text
voxel_x = floor(x / vx)
voxel_y = floor(y / vy)
voxel_z = floor(z / vz)
```

例如：

```text
Point：

x = 1.25
y = 2.31
z = 0.72

Voxel Size：

0.2 × 0.2 × 0.4
```

那么：

```text
voxel_x = floor(1.25 / 0.2)
        = 6

voxel_y = floor(2.31 / 0.2)
        = 11

voxel_z = floor(0.72 / 0.4)
        = 1
```

因此这个点属于：

```text
Voxel(6, 11, 1)
```

---

# 七、Voxelization 后发生了什么？

原始点云：

```text
100000 × 4
```

经过 Voxelization：

```text
Voxel 1
    ├── Point 1
    ├── Point 2
    ├── Point 3
    └── ...

Voxel 2
    ├── Point 10
    ├── Point 11
    └── ...

Voxel 3
    ├── Point 20
    ├── Point 21
    ├── Point 22
    └── ...

...
```

注意：

> **一个 Voxel 里面可以有多个 Point。**

因此：

```text
Point Cloud
     ↓
Voxel
     ↓
每个 Voxel 内部又包含若干 Point
```

---

# 八、为什么不能直接把 Voxel 当成一个值？

这是 VoxelNet 非常重要的一点。

假设：

```text
Voxel A：

Point 1
Point 2
Point 3
Point 4
```

如果简单地：

```text
Voxel A → 一个平均值
```

就会损失大量点云的几何信息。

例如：

```text
        Point
          ●

          ●
     ●

              ●
```

这些点在 Voxel 内部的空间分布，本身就是非常重要的信息。

因此 VoxelNet 不直接平均。

而是：

> **使用 VFE（Voxel Feature Encoding）对 Voxel 内部的 Point 进行学习。**

---

# 九、第三步：VFE —— Voxel Feature Encoding

这是原始 VoxelNet 最核心的创新之一。

VFE 的目标：

```text
Voxel 内的多个 Point
          ↓
学习
          ↓
一个 Voxel Feature
```

例如：

```text
Voxel：

P1 = [x1,y1,z1,i1]
P2 = [x2,y2,z2,i2]
P3 = [x3,y3,z3,i3]
P4 = [x4,y4,z4,i4]

          ↓ VFE

Voxel Feature

F = [f1,f2,f3,...,fd]
```

最终：

```text
一个 Voxel
      ↓
一个固定维度的 Feature Vector
```

---

# 十、VFE 的核心思想

VFE 可以理解成：

```text
Point Feature
     ↓
Point-wise FC
     ↓
Point Feature
     ↓
Aggregation
     ↓
Voxel Feature
```

也就是：

> **先处理每个 Point，再把 Voxel 内的 Point 信息聚合起来。**

---

# 十一、Point-wise Feature

假设一个 Voxel 中有：

```text
K 个 Point
```

每个 Point：

```text
[x,y,z,intensity]
```

因此输入：

```text
K × 4
```

例如：

```text
5 × 4
```

然后通过一个共享的全连接层：

```text
FC / Linear
```

得到：

```text
K × C
```

例如：

```text
5 × 4
   ↓
FC
   ↓
5 × 32
```

这里的关键：

> **所有 Point 使用同一个 FC。**

也就是：

```text
P1 → FC
P2 → FC
P3 → FC
P4 → FC
```

实际上是：

```text
Shared FC
```

这样可以提取每个点自己的局部特征。

---

# 十二、为什么要 Shared FC？

因为 Point Cloud 本身没有固定顺序。

例如：

```text
[P1,P2,P3]
```

和：

```text
[P3,P1,P2]
```

实际上表示的是同一组点。

因此网络应该满足：

```text
Point 顺序改变
       ↓
结果基本不应该改变
```

这就是点云网络非常重要的：

> **Permutation Invariance（置换不变性）**

VoxelNet 的 VFE 就通过：

```text
Shared FC
+
Aggregation
```

来处理这个问题。

---

# 十三、Aggregation：怎么把 Point 聚合成 Voxel？

经过 Point-wise FC 后：

```text
P1 → F1
P2 → F2
P3 → F3
P4 → F4
```

然后使用：

```text
Element-wise Max Pooling
```

得到：

```text
Fmax
```

例如：

```text
F1 = [1, 5, 2]
F2 = [3, 2, 7]
F3 = [4, 1, 3]
```

Max Pool：

```text
Fmax = [4, 5, 7]
```

因为：

```text
max(1,3,4) = 4
max(5,2,1) = 5
max(2,7,3) = 7
```

这样就得到了一个：

```text
Voxel-level Feature
```

---

# 十四、VFE Layer 的完整过程

可以记成：

```text
Point Features
      │
      ▼
Shared FC
      │
      ▼
Point-wise Features
      │
      ├─────────────┐
      │             │
      ▼             │
 Max Pool           │
      │             │
      ▼             │
Voxel-wise Feature  │
      │             │
      └──────┬──────┘
             ▼
      Feature Fusion
             │
             ▼
       VFE Output
```

更简单地：

```text
Point
 ↓
FC
 ↓
Point Feature
 ↓
Max Pool
 ↓
Voxel Feature
```

---

# 十五、VFE 为什么不只是 Max Pool？

因为只使用：

```text
Max Pool
```

虽然可以获得 Voxel 内的整体信息，但是可能会损失某些 Point-specific information。

因此 VoxelNet 的 VFE 会将：

```text
Point Feature
+
Voxel-wise Feature
```

进行拼接。

也就是：

```text
f_i
+
f_v
```

得到：

```text
[f_i, f_v]
```

然后继续经过下一层 VFE。

所以：

```text
VFE 1
    ↓
VFE 2
    ↓
Voxel Feature
```

可以逐步学习更丰富的局部几何结构。

---

# 十六、VFE 的最终输出

假设：

```text
N_voxel = 12000
```

每个 Voxel 最终得到：

```text
128 维
```

Feature。

那么：

```text
12000 × 128
```

就代表：

```text
12000 个非空 Voxel
```

每一个都有一个：

```text
128-dimensional feature
```

---

# 十七、一个非常重要的问题：Voxel 数量为什么不是固定的？

整个 3D 空间可能有：

```text
100 × 100 × 20
```

个 Voxel。

理论上：

```text
100 × 100 × 20
= 200000
```

但是实际 LiDAR 点云非常稀疏。

可能只有：

```text
12000 个 Voxel
```

里面真正有点。

所以：

```text
200000 个理论 Voxel
        ↓
只有 12000 个非空 Voxel
```

这就是 LiDAR 的：

> **Sparsity（稀疏性）**

---

# 十八、第四步：把 Voxel Feature 放回 3D 空间

现在我们有：

```text
Voxel Feature
```

例如：

```text
Voxel(1,2,3) → Feature A
Voxel(1,2,4) → Feature B
Voxel(5,8,2) → Feature C
...
```

需要重新把这些 Feature 放回原来的空间位置。

形成：

```text
X × Y × Z × C
```

的 3D Feature Map。

例如：

```text
100 × 100 × 20 × 128
```

---

# 十九、为什么要重新放回去？

因为后面要使用：

```text
3D CNN
```

而 CNN 最擅长处理的是：

```text
规则网格上的数据
```

所以：

```text
Point Cloud
      ↓
Voxel
      ↓
Voxel Feature
      ↓
3D Feature Map
      ↓
3D CNN
```

这一步实际上完成了：

> **从 Point-level 表示 → Voxel-level 空间表示**

---

# 二十、第五步：3D Convolution / Middle Network

现在：

```text
Voxel Feature
```

已经具有空间位置。

接下来通过：

```text
3D Convolution
```

提取空间特征。

例如：

```text
3D Feature Map
      ↓
3D Conv
      ↓
3D Conv
      ↓
3D Conv
      ↓
High-level 3D Feature
```

它可以学习：

```text
局部几何关系
空间结构
物体形状
上下文关系
```

---

# 二十一、为什么这里需要 3D CNN？

因为 LiDAR 本身是：

```text
3D 数据
```

所以我们希望卷积能够同时观察：

```text
x
y
z
```

三个方向。

普通 2D CNN：

```text
H × W × C
```

卷积主要在：

```text
H × W
```

上进行。

而 3D CNN：

```text
X × Y × Z × C
```

卷积核可以在：

```text
X
Y
Z
```

三个空间方向移动。

例如：

```text
3 × 3 × 3
```

的 3D Kernel。

---

# 二十二、但是 3D CNN 有一个严重问题

3D Feature Map 非常大。

例如：

```text
X = 400
Y = 400
Z = 40
C = 128
```

数据量：

```text
400 × 400 × 40 × 128
```

非常恐怖。

而且 LiDAR 大部分空间是：

```text
空的
```

所以：

> **VoxelNet 的原始思想虽然解决了点云规则化问题，但计算量依然很大。**

这也是后来：

```text
SECOND
PointPillars
CenterPoint
Voxel R-CNN
```

等方法不断优化的原因之一。

---

# 二十三、第六步：RPN

经过 3D CNN 后：

```text
High-level 3D Feature
```

接下来进入：

```text
RPN
Region Proposal Network
```

RPN 的任务：

> **从特征图中找到可能存在目标的位置，并预测目标的 3D Bounding Box。**

---

# 二十四、RPN 在干什么？

可以简单理解：

```text
3D Feature
     ↓
RPN
     ↓
哪里可能有车？
哪里可能有人？
哪里可能有其他目标？
```

然后：

```text
分类
+
Bounding Box Regression
```

---

# 二十五、3D Bounding Box 是什么？

2D 检测：

```text
x
y
w
h
```

3D 检测：

```text
x
y
z
l
w
h
yaw
```

也就是：

```text
中心位置：

(x, y, z)

尺寸：

(length, width, height)

方向：

yaw
```

例如：

```text
Car：

x = 12.3
y = 4.5
z = 1.2

l = 4.5
w = 1.8
h = 1.6

yaw = 1.57
```

---

# 二十六、RPN 输出什么？

一般可以理解成两个主要输出：

```text
                    ┌── Classification
Feature ──→ RPN ────┤
                    └── Box Regression
```

## 1. Classification

判断：

```text
这里是不是目标？
```

例如：

```text
Car      → 0.92
Pedestrian → 0.03
Background → 0.05
```

---

## 2. Box Regression

预测：

```text
x
y
z
l
w
h
yaw
```

也就是：

```text
目标在哪里？
目标有多大？
目标朝哪个方向？
```

---

# 二十七、Anchor 是什么？

VoxelNet 的检测阶段会使用：

```text
Anchor
```

Anchor 可以理解成：

> **提前放在空间中的一个“参考框”。**

例如：

```text
             Car Anchor
          ┌─────────────┐
          │             │
          │             │
          └─────────────┘
```

网络不是直接凭空预测一个 Box。

而是：

```text
Anchor
   ↓
网络预测偏移量
   ↓
最终 Bounding Box
```

例如：

```text
Anchor：

(xa, ya, za, la, wa, ha, θa)

网络预测：

Δx
Δy
Δz
Δl
Δw
Δh
Δθ

最终：

Box = Anchor + Offset
```

实际会采用经过参数化的 Box Regression 公式。

---

# 二十八、训练阶段怎么训练？

VoxelNet 的训练过程大致是：

```text
Ground Truth 3D Box
          │
          ▼
       Anchor
          │
          ▼
计算 IoU
          │
     ┌────┴────┐
     ▼         ▼
 Positive    Negative
 Anchor      Anchor
     │
     └────┬────┘
          ▼
       RPN
          │
          ▼
预测 Classification
预测 Bounding Box
          │
          ▼
        Loss
```

---

# 二十九、Loss

VoxelNet 的训练目标主要包括：

```text
Classification Loss
+
Regression Loss
```

可以简单理解为：

```text
L = L_cls + L_reg
```

其中：

```text
L_cls
```

负责：

> 这个位置是不是目标。

而：

```text
L_reg
```

负责：

> Box 应该往哪里移动、变多大、旋转多少。

---

# 三十、完整训练过程

完整训练：

```text
LiDAR Point Cloud
       ↓
Voxelization
       ↓
VFE
       ↓
Voxel Feature
       ↓
3D CNN
       ↓
RPN
       ↓
Classification
+
Box Regression
       ↓
与 Ground Truth 比较
       ↓
计算 Loss
       ↓
Backpropagation
       ↓
更新网络参数
```

---

# 三十一、完整推理过程

真正测试的时候：

```text
LiDAR
  ↓
Voxelization
  ↓
VFE
  ↓
3D CNN
  ↓
RPN
  ↓
Classification
  ↓
Box Regression
  ↓
Confidence Filtering
  ↓
NMS
  ↓
最终 3D Bounding Boxes
```

---

# 三十二、NMS 在干什么？

网络可能预测出很多重复 Box。

例如：

```text
       ┌─────────────┐
      ┌───────────────┐
     ┌───────────────┐
```

三个 Box 实际上都是：

```text
同一辆车
```

所以需要：

```text
NMS
Non-Maximum Suppression
```

例如：

```text
Box A → 0.95
Box B → 0.91
Box C → 0.87
```

如果：

```text
IoU(A,B) > threshold
```

就保留：

```text
A
```

删除：

```text
B
```

最后：

```text
重复 Box
   ↓
NMS
   ↓
一个最终 Box
```

---

# 三十三、VoxelNet 最核心的三层理解

学习 VoxelNet 的时候，不要把它理解成一堆代码。

抓住三个核心：

```text
① Voxelization
② VFE
③ 3D CNN + RPN
```

---

## ① Voxelization

解决：

> 无规则点云如何变成规则空间？

```text
Point Cloud
      ↓
Voxel
```

---

## ② VFE

解决：

> 一个 Voxel 里面有很多 Point，如何提取它们的特征？

```text
Multiple Points
      ↓
Shared FC
      ↓
Aggregation
      ↓
Voxel Feature
```

---

## ③ 3D CNN + RPN

解决：

> 如何利用整个 3D 空间的信息完成目标检测？

```text
Voxel Features
      ↓
3D CNN
      ↓
Spatial Features
      ↓
RPN
      ↓
3D Boxes
```

---

# 三十四、VoxelNet 最重要的数据形状变化

这个非常建议记下来。

```text
原始点云：

N × 4

例如：

100000 × 4
```

↓

```text
Voxelization：

M × K × 4

M = Voxel 数量
K = 每个 Voxel 最大 Point 数

例如：

12000 × 35 × 4
```

注意：

```text
K
```

通常需要设定上限。

例如一个 Voxel 里面最多保留：

```text
35 个 Point
```

如果：

```text
Point 数 > 35
```

就需要采样/截断。

如果：

```text
Point 数 < 35
```

则需要 padding。

---

↓

```text
VFE：

M × K × 4
       ↓
M × K × C
       ↓
M × C
```

也就是：

```text
每个 Voxel
    ↓
一个固定维度 Feature
```

---

↓

```text
重新 Scatter：

M × C
    ↓
X × Y × Z × C
```

也就是：

```text
Sparse Voxel Features
```

重新放回 3D 空间。

---

↓

```text
3D CNN：

X × Y × Z × C
       ↓
High-level 3D Feature
```

---

↓

```text
RPN：

Feature
   ↓
Classification
+
Regression
   ↓
3D Bounding Box
```

---

# 三十五、VoxelNet 和 PointNet++ 的区别

你最近正在学 PointNet++，这两个非常容易混。

## PointNet++

思路：

```text
Point Cloud
    ↓
局部邻域
    ↓
PointNet
    ↓
层级化特征
    ↓
Feature
```

重点是：

> **直接在 Point 上做层级特征提取。**

---

## VoxelNet

思路：

```text
Point Cloud
    ↓
Voxelization
    ↓
Voxel
    ↓
VFE
    ↓
3D CNN
```

重点是：

> **先把 Point Cloud 离散成 Voxel，再利用规则空间结构进行卷积。**

---

# 三十六、VoxelNet 和 PointPillars 的区别

这个对你后面学习非常重要。

VoxelNet：

```text
Point Cloud
    ↓
3D Voxel
    ↓
VFE
    ↓
3D CNN
    ↓
RPN
```

PointPillars：

```text
Point Cloud
    ↓
Pillar
    ↓
PointNet-like Encoder
    ↓
BEV 2D Feature Map
    ↓
2D CNN
    ↓
Detection Head
```

核心区别：

```text
VoxelNet：

3D Voxel
↓
3D CNN
```

而：

```text
PointPillars：

Vertical Pillar
↓
BEV
↓
2D CNN
```

所以 PointPillars 通常计算效率更高。

---

# 三十七、VoxelNet → SECOND → 后续方法

可以把 3D 检测的发展理解成：

```text
VoxelNet
   │
   │ 解决：
   │  Point → Voxel
   │
   ▼
SECOND
   │
   │ 改进：
   │  Sparse Convolution
   │
   ▼
PointPillars
   │
   │ 改进：
   │  3D → BEV
   │  使用 2D CNN
   │
   ▼
CenterPoint
   │
   │ 改进：
   │  Center-based Detection
   │
   ▼
Voxel R-CNN / PV-RCNN 等
```

VoxelNet 在这条技术路线里非常重要。

---

# 三十八、你应该如何理解 VoxelNet？

不要把它理解成：

```text
VoxelNet = 一个复杂的网络
```

而应该理解成：

```text
VoxelNet
=
一种完整的 Point Cloud → 3D Detection 思路
```

核心流程：

```text
                Point Cloud
                     │
                     ▼
              ┌─────────────┐
              │ Voxelization│
              └─────────────┘
                     │
                     ▼
              ┌─────────────┐
              │     VFE     │
              └─────────────┘
                     │
                     ▼
              Voxel Features
                     │
                     ▼
              ┌─────────────┐
              │   3D CNN    │
              └─────────────┘
                     │
                     ▼
              Spatial Features
                     │
                     ▼
              ┌─────────────┐
              │     RPN     │
              └─────────────┘
                     │
             ┌───────┴────────┐
             ▼                ▼
       Classification    Box Regression
             │                │
             └───────┬────────┘
                     ▼
                3D Bounding Box
                     │
                     ▼
                    NMS
                     │
                     ▼
              Final Detection
```

---

# 三十九、一句话记忆 VoxelNet

> **VoxelNet 就是先把无规则的 LiDAR 点云划分成规则的 3D Voxel，再用 VFE 学习每个 Voxel 内部的点云特征，之后利用 3D CNN 学习空间结构，最后通过 RPN 完成 3D 目标检测。**

---

# 四十、从工程角度理解

如果以后你自己使用 OpenPCDet 等框架，不需要手写整个 VoxelNet。

你真正需要理解的是：

```text
Dataset
   ↓
Point Cloud
   ↓
Voxelization
   ↓
Voxel Features
   ↓
Backbone
   ↓
Detection Head
   ↓
3D Boxes
```

对应到代码里通常会看到：

```text
Dataset
Voxelization
Voxel Encoder
Sparse Encoder / Backbone
Dense Head / RPN
Post Processing
```

---

# 四十一、和你现在学习路线的关系

你现在学习：

```text
PointNet++
VoxelNet
PointPillars
3D Transformer
BEVFusion
```

其实是在理解不同的：

> **3D 点云特征表示方式。**

可以这样分类：

| 方法           | Point 如何处理           | 空间表示               | Backbone     |
| ------------ | -------------------- | ------------------ | ------------ |
| PointNet     | 直接 Point             | Point              | MLP          |
| PointNet++   | 局部 Point             | Hierarchical Point | PointNet     |
| VoxelNet     | Point → Voxel        | 3D Voxel           | 3D CNN       |
| SECOND       | Point → Sparse Voxel | Sparse Voxel       | Sparse CNN   |
| PointPillars | Point → Pillar       | BEV                | 2D CNN       |
| PV-RCNN      | Point + Voxel        | Point + Voxel      | Hybrid       |
| BEVFusion    | 多模态 → BEV            | BEV                | BEV Backbone |

所以你现在学 VoxelNet 的目的不是：

> “我要把 VoxelNet 代码全部背下来。”

而是要搞清楚：

> **为什么 Point Cloud 要进行 Voxelization？Voxel Feature 是怎么来的？为什么之后可以使用 CNN？以及最终怎么从 3D Feature 得到 Bounding Box？**

这几个问题搞明白，后面的：

```text
SECOND
PointPillars
PV-RCNN
Voxel R-CNN
CenterPoint
BEVFusion
```

都会好理解很多。
