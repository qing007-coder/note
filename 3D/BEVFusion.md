# BEVFusion 数据流与维度变换

只讲两件事：**每一步的张量形状怎么变**、**数据从哪流到哪**。

- 论文：Liu et al., *BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation*, ICRA 2023（arXiv 2205.13542）
- 代码：`github.com/mit-han-lab/bevfusion`
- 取的是 **nuScenes 检测（camera + LiDAR）** 这一条线，配置入口 `configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml`，对应 nuScenes val **68.52 mAP / 71.38 NDS**（纯相机基线 35.56，纯 LiDAR 基线 64.68）。
- 数字以**配置项和源码**为准，正文里都标了出处；凡是配置里没直接写、需要算或推的地方，都把推导过程写出来了。

---

## 0. 一张图看完全流程

以 **batch = 1** 为例写形状（batch 维写成 B）。

```
【相机分支】img (1, 6, 3, 256, 704)
   │  view → (6, 3, 256, 704)
   ▼  Swin-T  (out_indices=[1,2,3])
   (6,192,32,88)@1/8  (6,384,16,44)@1/16  (6,768,8,22)@1/32
   │  GeneralizedLSSFPN (in=[192,384,768], out=256)  ← 只取 x[0]
   ▼  (6, 256, 32, 88)          ← 这就是 feature_size=[32,88] 的来历
   │  DepthLSSTransform：118 个深度平面做外积
   ▼  (1, 6, 118, 32, 88, 80)   [B, N, D, fH, fW, C]
   │  get_geometry：每个格子按 lidar2img 算 3D 点 → (1, 6, 118, 32, 88, 3)
   │  bev_pool：按 0.3 m 落到 360×360×1 的格子里
   ▼  (1, 80, 360, 360)
   │  downsample ×2
   ▼  (1, 80, 180, 180)          ← 格 0.6 m

【LiDAR 分支】points (N, 5)（10 个 sweep 合并）
   │  体素化 voxel=[0.075,0.075,0.2], range=[-54,-54,-5,54,54,3]
   ▼  voxels (120000, 10, 5) → 格内均值 → (120000, 5)
   │  SparseEncoder（稀疏卷积，3 次 stride 2）
   ▼  (1, 128, 180, 180, 2)      ← 空间 180×180，z 只剩 2 格
   │  permute + view：把 z 折进通道
   ▼  (1, 256, 180, 180)         ← 256 = 128 × 2

【融合】cat([相机 80, LiDAR 256], dim=1) → (1, 336, 180, 180)
   ▼  ConvFuser: Conv3×3 336→256 → (1, 256, 180, 180)
   ▼  SECOND: (1,128,180,180) + (1,256,90,90)
   ▼  SECONDFPN: cat → (1, 512, 180, 180)
   ▼  TransFusionHead → 200 个 proposal 的 (center/height/dim/rot/vel/heatmap)
```

**整条线的支点**：两个分支必须在**同一个 BEV 网格**上收口。

| | 相机分支 | LiDAR 分支 |
|---|---|---|
| 原始步长 | xbound 0.3 m | voxel 0.075 m |
| 下采样 | `downsample: 2` | 稀疏卷积 stride 8 |
| 最终步长 | 0.3 × 2 = **0.6 m** | 0.075 × 8 = **0.6 m** |
| 网格数 | 108 m / 0.6 = **180×180** | 1440 / 8 = **180×180** |
| 原点 | (-54, -54) | (-54, -54) |

步长、网格数、原点三者全对上，两个 `(1, ·, 180, 180)` 才能直接 `cat`。**任何一个对不上，融合就没意义**——这是 BEVFusion 里最容易出错的地方。

---

## 1. 配置是怎么拼出来的

入口文件 `convfuser.yaml` 全文只有 6 行：

```yaml
model:
  fuser:
    type: ConvFuser
    in_channels: [80, 256]
    out_channels: 256
```

它是怎么变成一个完整模型的？`tools/train.py` 用的是 **torchpack 的递归配置加载**（不是 mmcv 的 `_base_`）：

```python
from torchpack.utils.config import configs
configs.load(args.config, recursive=True)     # tools/train.py:29
```

它会把从 `configs/` 到目标文件**路径上每一层目录里的 yaml 依次合并**。所以真实配置链是：

| 顺序 | 文件 | 提供了什么 |
|---|---|---|
| 1 | `configs/nuscenes/default.yaml` | `image_size [256,704]`、`point_cloud_range`、`voxel_size`、数据 pipeline |
| 2 | `configs/nuscenes/det/default.yaml` | `model.type = BEVFusion` |
| 3 | `configs/nuscenes/det/transfusion/default.yaml` | `heads.object = TransFusionHead` 全套（含 `grid_size`、`out_size_factor`） |
| 4 | `.../secfpn/default.yaml` | `decoder` = SECOND + SECONDFPN，head `in_channels: 512` |
| 5 | `.../camera+lidar/default.yaml` | 相机 neck / vtransform、LiDAR backbone `SparseEncoder` |
| 6 | `.../swint_v0p075/default.yaml` | voxel 0.075、范围 ±54、Swin-T、`sparse_shape`、`grid_size` |
| 7 | `convfuser.yaml` | 只有 fuser |

所以看代码时**不能只翻 `convfuser.yaml`**，第 5、6 层才是主干。

---

## 2. 相机分支

### 2.1 输入侧有哪些东西

`BEVFusion.extract_camera_features()`（`mmdet3d/models/fusion_models/bevfusion.py:105`）拿到的 meta 有：

| 变量 | shape | 含义 |
|---|---|---|
| `img` | (B, 6, 3, 256, 704) | 6 路环视，已经 resize 到 256×704 |
| `camera_intrinsics` | (B, 6, 4, 4) | **原始 1600×900 上的 K**（后面用 `[..., :3,:3]`） |
| `img_aug_matrix` | (B, 6, 4, 4) | 2D 增强矩阵（crop + resize），把增强后的图**还原**回原图用 |
| `camera2lidar` / `camera2ego` | (B, 6, 4, 4) | 相机外参（每路一个） |
| `lidar2ego` | (B, 4, 4) | LiDAR 外参（每帧一个，这条线上算了但没用） |
| `lidar2image` | (B, 6, 4, 4) | LiDAR → 每张图的投影矩阵（造深度图用） |
| `lidar_aug_matrix` | (B, 4, 4) | 点云的 3D 增强（`GlobalRotScaleTrans`） |

注意 `camera_intrinsics` 是**原图**的内参，`img_aug_matrix` 负责把 256×704 这个坐标系换算回原图坐标系。这个先后顺序在 2.6 节会再用到。

### 2.2 Backbone：Swin-T 出三个尺度

```python
B, N, C, H, W = x.size()      # 1, 6, 3, 256, 704
x = x.view(B * N, C, H, W)    # (6, 3, 256, 704)  ← 6 路图拼进 batch 维
x = self.encoders["camera"]["backbone"](x)
```

```yaml
# swint_v0p075/default.yaml
embed_dims: 96
depths: [2, 2, 6, 2]
num_heads: [3, 6, 12, 24]
out_indices: [1, 2, 3]      # ← 关键
```

patch 4×4 / stride 4 → `(6, 96, 64, 176)`，然后四个 stage：

| stage | 通道 | 分辨率 | 相对原图 |
|---|---|---|---|
| 0 | 96 | 64×176 | 1/4 |
| 1 | 192 | 32×88 | 1/8 |
| 2 | 384 | 16×44 | 1/16 |
| **3** | 768 | 8×22 | 1/32 |

`out_indices: [1, 2, 3]` 只取后三个 → neck 的 `in_channels: [192, 384, 768]` 正好对上。**1/4 那一级（96 通道 64×176）在这里就被丢掉了**，所以后面 LSS 用的特征分辨率是 1/8 而不是 1/4。

### 2.3 Neck：GeneralizedLSSFPN 只做两级融合

`mmdet3d/models/necks/generalized_lss.py`。配置 `in_channels: [192,384,768]`, `out_channels: 256`, `start_level: 0`。

构造时 `backbone_end_level = num_ins - 1 = 2`，所以只建 **2 组** 1×1 卷积：

- `lateral_convs[0]`: `192 + 256 = 448` → 256
- `lateral_convs[1]`: `384 + 768 = 1152` → 256

forward（自顶向下）：

```
i=1: upsample(768 @ 8×22  →  16×44)
     cat([384 @ 16×44, 256 @ 16×44]) = 1152 → conv1×1 → conv3×3 → (6, 256, 16, 44)
i=0: upsample(256 @ 16×44 →  32×88)
     cat([192 @ 32×88, 256 @ 32×88]) = 448  → conv1×1 → conv3×3 → (6, 256, 32, 88)
outs = [laterals[0], laterals[1]]      # 只有 2 个
```

两点值得注意：

1. 第 3 级（768 @ 8×22）**只作为 i=1 的输入，没有自己的输出**。配置写 `num_outs: 3`，但代码里 `outs = [laterals[i] for i in range(used_backbone_levels)]`，`used_backbone_levels = 3 - 1 = 2`，实际返回 **2** 个张量。
2. 回模型里只看第一个：

```python
x = self.encoders["camera"]["neck"](x)
if not isinstance(x, torch.Tensor):
    x = x[0]                  # ← 取 stride 1/8 那一级
BN, C, H, W = x.size()        # 6, 256, 32, 88
x = x.view(B, int(BN / B), C, H, W)   # (1, 6, 256, 32, 88)  ← 把 6 路拆回来
```

**这一步决定了 `vtransform.feature_size = [32, 88]`。** 如果哪天改了 Swin 的 `out_indices`，`feature_size` 必须跟着改，否则 2.5 节的 `cat` 会因为分辨率不一致直接报错。

### 2.4 深度图：相机分支要用 LiDAR 点云

这是最容易忽略的一条数据流：**相机分支需要 LiDAR 点云**。

`BaseDepthTransform.forward()`（`mmdet3d/models/vtransforms/base.py`）在跑网络之前，先拿 LiDAR 点投影造一张深度图：

```python
depth = torch.zeros(batch_size, img.shape[1], depth_in_channels, *self.image_size)
#                  (B, 6, 1, 256, 704)      depth_input='scalar' → 1 通道
for b in range(batch_size):
    cur_coords = points[b][:, :3]
    # 1. 反向 undo 3D 增强
    cur_coords -= cur_lidar_aug_matrix[:3, 3]
    cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(cur_coords.transpose(1, 0))
    # 2. lidar2image 投到 6 张图上，得到 (6, 3, N)
    cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
    cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
    # 3. 归一化 → 2D 像素
    dist = cur_coords[:, 2, :]
    cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
    cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]
    # 4. 再过一遍 2D 增强矩阵，得到增强后图像上的坐标
    ...
    # 5. 落在图内的点，把深度值写进 depth[b, c, 0, v, u]
    depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist
```

所以 `depth` 的形状是 **(B, 6, 1, 256, 704)**：一张和输入图同分辨率的、**由 LiDAR 点稀疏填充**的深度图。绝大部分像素是 0（没有点落上去）。

这让 `DepthLSSTransform` 和普通 `LSSTransform` 区分开了：普通 LSS 只用图像特征自己猜深度，DepthLSS 额外吃一张 LiDAR 给的深度提示。**这也是为什么相机分支的 forward 签名里必须有 `points`**——纯相机基线（`camera-only`）用的是另一套 `lssfpn` 配置，没有这一步。

### 2.5 DepthLSSTransform：2D 特征 → 3D 视锥体

`mmdet3d/models/vtransforms/depth_lss.py:82`：

```python
B, N, C, fH, fW = x.shape       # 1, 6, 256, 32, 88
d = d.view(B * N, *d.shape[2:]) # (6, 1, 256, 704)
x = x.view(B * N, C, fH, fW)    # (6, 256, 32, 88)
```

**第一段：把深度图降到特征分辨率**

```python
d = self.dtransform(d)
# (6,1,256,704)
# → Conv2d(1, 8, 1)                  (6, 8, 256, 704)
# → Conv2d(8, 32, 5, stride=4, pad=2)  (6, 32, 64, 176)
# → Conv2d(32, 64, 5, stride=2, pad=2) (6, 64, 32, 88)   ← 正好落到 1/8
```

**第二段：拼上图像特征，预测深度分布 + 上下文**

```python
x = torch.cat([d, x], dim=1)    # (6, 64+256=320, 32, 88)
x = self.depthnet(x)
# 320 → 256 → 256 → (D + C) = 118 + 80 = 198
```

- `D = 118`：`dbound: [1.0, 60.0, 0.5]` → `torch.arange(1, 60, 0.5)` → 118 个深度平面
- `C = 80`：`out_channels: 80`，即最终相机 BEV 的通道数

**第三段：外积，把 2D 摊成 3D**

```python
depth = x[:, :118].softmax(dim=1)                 # (6, 118, 32, 88) 每格 118 层概率
x = depth.unsqueeze(1) * x[:, 118:198].unsqueeze(2)
#   (6, 1, 118, 32, 88) ⊗ (6, 80, 1, 32, 88) → (6, 80, 118, 32, 88)
x = x.view(B, N, self.C, self.D, fH, fW)          # (1, 6, 80, 118, 32, 88)
x = x.permute(0, 1, 3, 4, 5, 2)                   # (1, 6, 118, 32, 88, 80)
```

最后这个 `(B, N, D, fH, fW, C)` 的 6 维布局是 `bev_pool` 约定的输入格式。

> 118 个深度平面是**固定的**（就是 dbound 的等差值 1.0, 1.5, …, 59.5），网络只负责预测"这个格子落在哪一层"的分布（那 118 通道的 softmax），不预测具体深度值。80 通道的特征则按这个分布加权铺到 118 层上。

### 2.6 get_geometry：从像素反推 3D 点

`base.py:92`。输入是固定视锥体 `frustum`：

```python
# create_frustum(): (D, fH, fW, 3) = (118, 32, 88, 3)
xs = torch.linspace(0, iW - 1, fW)   # 0 → 703，88 个采样点
ys = torch.linspace(0, iH - 1, fH)   # 0 → 255，32 个采样点
ds = torch.arange(*dbound)           # 1.0 → 59.5，118 层
frustum = torch.stack((xs, ys, ds), -1)   # (u, v, d)
```

注意 `xs` 是从 0 到 **703**（不是 87），说明这 88 个采样点是**在 256×704 那张图的像素坐标系里**取的，不是特征图坐标系里的格心。所以后面必须先把坐标还原回原图，再乘内参。

```python
# 1. undo 2D 增强：把 (u,v,d) 变回原图坐标系
points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

# 2. 把 (u,v) 乘上 d，凑成齐次坐标 (u·d, v·d, d)：K⁻¹ 作用在 (u,v,1) 上等于对 (u·d, v·d, d) 作用
points = torch.cat((points[..., :2] * points[..., 2:3], points[..., 2:3]), 5)

# 3. K⁻¹ 和外参一起乘，再加平移 → LiDAR 坐标系
combine = camera2lidar_rots.matmul(torch.inverse(intrins))
points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

# 4. 再叠上点云的 3D 增强（GlobalRotScaleTrans）
points = extra_rots.view(...).matmul(points.unsqueeze(-1)).squeeze(-1)
points += extra_trans.view(...)
```

输出 `(1, 6, 118, 32, 88, 3)`，单位米，**在增强后的 LiDAR 坐标系里**。总的点数：

```
6 × 118 × 32 × 88 = 1,993,728
```

也就是每帧要往 BEV 里投 **约 200 万个视锥点**——所以 `bev_pool` 必须是 CUDA 实现。

第 4 步（叠 `lidar_aug_matrix`）是保证两个分支坐标系一致的关键：点云在 `GlobalRotScaleTrans` 里被旋转/缩放过，相机这边算出来的 3D 点也必须做同一个变换，否则两边落在不同网格上。

### 2.7 bev_pool：3D 点 → BEV 格

`base.py:141`：

```python
B, N, D, H, W, C = x.shape      # 1, 6, 118, 32, 88, 80
Nprime = B * N * D * H * W      # 1993728
x = x.reshape(Nprime, C)        # (1993728, 80)
```

索引计算：

```python
geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
# bx = 边界 + dx/2，所以等价于 (坐标 - 边界) / dx，再向下取整 → 格号
```

`dx/bx/nx` 由 `gen_dx_bx()` 从 xbound/ybound/zbound 算出来：

```
dx = [0.3, 0.3, 20]              # 每格边长（z 方向 20 m 只有一个格）
bx = [-54+0.15, -54+0.15, -10+10]  # 格心偏移
nx = [360, 360, 1]               # 每维格数：108/0.3, 108/0.3, 20/20
```

然后过滤掉落在范围外的点，再调 CUDA 算子：

```python
x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])  # → (B, C, D, H, W)
final = torch.cat(x.unbind(dim=2), 1)                              # 折 z
```

`mmdet3d/ops/bev_pool/bev_pool.py` 里做的事：按格号排序 → `cumsum` 求段内和 → 同一格的特征**相加**（不是平均）。返回 `(1, 80, 1, 360, 360)`。

> `zbound: [-10, 10, 20]` → `nz = 1`。所以检测配置里"折 z"这步 `unbind` + `cat` 其实是**恒等操作**（只有一个 z 平面）。`zbound` 只有一层是相机做 BEV 的常见设定（BEVDet/BEVDepth 也一样），因为相机 BEV 本来就是 2D 的；要真正利用 z 维分级（体素/占用那类任务），得先把 `zbound` 的步长改小、让 `nz > 1`。

### 2.8 收尾：downsample ×2

```python
def forward(self, *args, **kwargs):
    x = super().forward(*args, **kwargs)   # (1, 80, 360, 360)
    x = self.downsample(x)                 # (1, 80, 180, 180)
    return x
```

`downsample: 2` 触发的是一个三层卷积（中间那层 `stride=2`），把 0.3 m 的 360×360 降到 0.6 m 的 180×180，通道保持 80。

**相机分支产物：(B, 80, 180, 180)。**

---

## 3. LiDAR 分支

### 3.1 体素化

```yaml
voxelize:
  max_num_points: 10
  point_cloud_range: [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
  voxel_size: [0.075, 0.075, 0.2]
  max_voxels: [120000, 160000]      # 训练 / 测试
```

点云从 pipeline 来：`load_dim: 5, use_dim: 5`（x, y, z, intensity, ring），`LoadPointsFromMultiSweeps(sweeps_num=9)` 把当前帧 + 9 个历史 sweep 合并成 `(N, 5)`。

网格大小：

```
x: 108 / 0.075 = 1440
y: 108 / 0.075 = 1440
z: (3 - (-5)) / 0.2 = 40  → sparse_shape 里写 41
```

**z 为什么是 41 而不是 40**：`z = 3.0` 这个边界点算出来的下标是 `floor((3-(-5))/0.2) = 40`，要装得下就得 41 个格子。mmdet3d 的稀疏卷积约定比实际网格多留一格，这也解释了 head 里 `grid_size: [1440, 1440, 41]`。

体素化输出：

| 张量 | shape |
|---|---|
| `voxels` | (120000, 10, 5) ← 最多 12 万个非空格子，每格最多 10 点 |
| `coords` | (120000, 4) ← batch + 3 个空间下标，**最后一列是 z**（见 3.2、3.3） |
| `num_points_per_voxel` | (120000,) |

整张网格是 `1440 × 1440 × 41 ≈ 8500 万` 格，实际只有 ≤12 万格非空——**这就是必须用稀疏卷积的原因**。

`BEVFusion.voxelize()` 里还有一步 `voxelize_reduce`：

```python
feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
```

把 `(120000, 10, 5)` 在"格内 10 个点"这一维求平均 → **`(120000, 5)`**。所以进 backbone 的是每格一个 5 维特征（正好对上 `SparseEncoder.in_channels: 5`），而不是 10 个点各自的特征。

### 3.2 SparseEncoder：稀疏卷积把 1440 压到 180

`mmdet3d/models/backbones/sparse_encoder.py`，配置：

```yaml
type: SparseEncoder
in_channels: 5
sparse_shape: [1440, 1440, 41]
output_channels: 128
encoder_channels: [[16, 16, 32], [32, 32, 64], [64, 64, 128], [128, 128]]
encoder_paddings: [[0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]], [0, 0]]
block_type: basicblock
```

结构（`block_type: basicblock` 下的实际搭建逻辑）：

| 层 | 操作 | 通道 | x / y | z |
|---|---|---|---|---|
| `conv_input` | SubMConv3d 3×3 | 5 → 16 | 1440 | 41 |
| stage1 | 2× BasicBlock + **stride 2** | 16→16→16→**32** | 720 | 21 |
| stage2 | 2× BasicBlock + **stride 2** | 32→32→32→**64** | 360 | 11 |
| stage3 | 2× BasicBlock + **stride 2** | 64→64→64→**128** | 180 | 5 |
| stage4 | 2× BasicBlock（不下采样） | 128→128 | 180 | 5 |
| `conv_out` | SparseConv3d k=(1,1,3) s=(1,1,2) | 128→128 | **180** | **2** |

每个 stage 的最后一个 block 是 `SparseConv3d(stride=2)`，其余是 `SparseBasicBlock`（子流形卷积，形状不变）。

**z 是最后一个空间维**，两处代码都指向这一点：

- stage3 那个 stride-2 的卷积，padding 写的是 `[1,1,0]`——前两维补 1（保证 360 → 180），第三维不补（保证 11 → 5 而不是 6）；
- `conv_out` 的 kernel 是 `(1,1,3)`、stride 是 `(1,1,2)`，只有第三维被 kernel 3 碰到，`5 → 2`，x/y 一动不动。

（`SparseEncoder` 的 docstring 写的是 "columns in the order of (batch_idx, z_idx, y_idx, x_idx)"，那是从 KITTI 版 SECOND 抄来的——KITTI 的 `sparse_shape=[41,1600,1408]` 确实是 z 在最前。nuScenes 这里 `sparse_shape=[1440,1440,41]`，41 在最后，**以配置和 padding 为准**。）

### 3.3 3D → 2D：把 z 折进通道

```python
out = self.conv_out(encode_features[-1])
spatial_features = out.dense()                    # (1, 128, 180, 180, 2)

N, C, H, W, D = spatial_features.shape
spatial_features = spatial_features.permute(0, 1, 4, 2, 3).contiguous()
#                                                 (1, 128, 2, 180, 180)
spatial_features = spatial_features.view(N, C * D, H, W)
#                                                 (1, 256, 180, 180)   ← 128 × 2 = 256
```

**LiDAR 分支那 256 通道，不是卷积出来的，是 `128 通道 × 剩下的 2 格 z` 折出来的。** 这是 BEVFusion 把 3D 稀疏特征变成 2D BEV 的那一步——不同深度/高度的信息没有丢，而是被塞进了通道维。

到这里：

- 相机 `(1, 80, 180, 180)`，格 0.6 m
- LiDAR `(1, 256, 180, 180)`，格 0.6 m

**可以拼了。**

---

## 4. 融合、解码、检测头

### 4.1 拼接顺序是有约束的

`bevfusion.py:297`：

```python
for sensor in (self.encoders if self.training else list(self.encoders.keys())[::-1]):
    ...
    features.append(feature)

if not self.training:
    features = features[::-1]      # 推理时倒着跑（省显存），最后再倒回来

if self.fuser is not None:
    x = self.fuser(features)
```

训练时 `self.encoders` 的插入顺序是 `camera` 在前、`lidar` 在后；推理时先跑 LiDAR 再跑相机，但最后 `features[::-1]` 又倒回来了。**所以不管训练还是推理，`features` 永远是 `[相机, LiDAR]`。**

这正好对上 `ConvFuser.in_channels: [80, 256]`——通道数的顺序和 list 的顺序是绑死的，写反了 `cat` 不会报 shape 错，但结果全错。

### 4.2 ConvFuser

`mmdet3d/models/fusers/conv.py`，整个文件只有一句有效代码：

```python
def forward(self, inputs):
    return super().forward(torch.cat(inputs, dim=1))
# nn.Sequential: Conv2d(336, 256, 3, padding=1, bias=False), BN2d(256), ReLU
```

```
cat([(1,80,180,180), (1,256,180,180)], dim=1) → (1, 336, 180, 180)
Conv3×3 336→256                              → (1, 256, 180, 180)
```

（`mmdet3d/models/fusers/add.py` 里另有一个 `AddFuser`，直接相加，要求两个分支通道数相同；检测配置用的是 `ConvFuser`。）

### 4.3 Decoder：SECOND + SECONDFPN

```yaml
# secfpn/default.yaml
backbone:                      # SECOND
  in_channels: 256
  out_channels: [128, 256]
  layer_nums: [5, 5]
  layer_strides: [1, 2]
neck:                          # SECONDFPN
  in_channels: [128, 256]
  out_channels: [256, 256]
  upsample_strides: [1, 2]
  use_conv_for_no_stride: true
```

```
(1, 256, 180, 180)
 ├─ block1: Conv3×3 s1 256→128 + 5×(Conv3×3 128→128)      → (1, 128, 180, 180)
 └─ block2: Conv3×3 s2 128→256 + 5×(Conv3×3 256→256)      → (1, 256,  90, 90)

SECONDFPN：
 level0: stride=1 → use_conv_for_no_stride → Conv2d k=1 s=1 128→256  → (1, 256, 180, 180)
 level1: stride=2 → ConvTranspose2d k=2 s=2 256→256                  → (1, 256, 180, 180)
 cat(dim=1)                                                          → (1, 512, 180, 180)
```

即 **256 + 256 = 512**，对上 head 的 `in_channels: 512`。注意 `SECONDFPN.forward` 返回的是 `[out]`（一个 list），所以 head 收到的其实是"只有一层特征"的列表。

### 4.4 TransFusionHead

```yaml
in_channels: 512
hidden_channel: 128
num_classes: 10
num_proposals: 200
num_decoder_layers: 1
num_heads: 8
ffn_channel: 256
train_cfg:
  grid_size: [1440, 1440, 41]
  out_size_factor: 8
  voxel_size: [0.075, 0.075, 0.2]
```

先把离线网格换算清楚：

```
特征图尺寸 = grid_size[:2] // out_size_factor = 1440 / 8 = 180
每格实际边长 = voxel_size × out_size_factor = 0.075 × 8 = 0.6 m
```

和相机分支的 180×180 / 0.6 m 完全一致。

**流程（`mmdet3d/models/heads/bbox/transfusion.py:215` forward_single）：**

```
输入 (1, 512, 180, 180)
  │ shared_conv (Conv3×3 512→128)
  ▼ (1, 128, 180, 180)                    ← lidar_feat
  │
  ├─ heatmap_head: Conv3×3 128→128 + Conv3×3 128→10
  ▼ (1, 10, 180, 180)                     ← dense_heatmap（10 类目标）
  │
  │ NMS 预热：3×3 max_pool 取局部极大（行人和锥桶用 1×1，因为它们密集）
  ▼ (1, 10, 180, 180) → view (1, 10, 32400)
  │ 展平后 argsort 取 top-200  ← 200 个 query 的来历
  ▼ top_proposals_index (1, 200)
```

然后用这 200 个索引去特征图上 gather 出 query：

```
query_feat    = gather(lidar_feat)              → (1, 128, 200)
              + class_encoding(one_hot(类别))   → 类别嵌入也加进来

bev_pos       = 180×180 每个格心坐标 (1, 32400, 2)
                gather(top-200)                 → (1, 200, 2)

query_feat = decoder[0](query_feat, lidar_feat_flatten, query_pos, bev_pos)
```

decoder 只有 1 层，K/V 就是 LiDAR BEV 特征本身（`lidar_feat_flatten`，`(1, 128, 32400)`）。输出经 FFN：

| 输出 | shape | 含义 |
|---|---|---|
| `heatmap` | (1, 10, 200) | 每个 proposal 的分类分数 |
| `center` | (1, 2, 200) | 格心偏移，**再加回 `bev_pos`** 得到绝对格坐标 |
| `height` | (1, 1, 200) | 高度 |
| `dim` | (1, 3, 200) | 长宽高 |
| `rot` | (1, 2, 200) | 朝向（sin/cos） |
| `vel` | (1, 2, 200) | 速度 |

最后 `bbox_coder`（`out_size_factor: 8`）把格坐标乘回 **0.6 m/格 + 原点 (-54, -54)** 还原成米制 3D 框。

> 这个 `TransFusionHead` 在本实现里**并没有用图像特征**——`forward_single` 的签名里有 `img_inputs`，但代码里 `img_feat_pos`/`img_feat_collapsed_pos` 两处变量声明后再没被使用，K/V 从头到尾只有 LiDAR BEV。图像信息早在 `ConvFuser` 那一步就混进 BEV 特征了。

---

## 5. 形状速查表

| # | 阶段 | 形状 | 备注 |
|---|---|---|---|
| 1 | 输入图像 | (B, 6, 3, 256, 704) | 6 路环视 |
| 2 | Swin-T 输出 | (B·6, 192, 32, 88) / 384,16,44 / 768,8,22 | `out_indices=[1,2,3]` |
| 3 | neck 输出（取 x[0]） | (B·6, 256, 32, 88) | 1/8 尺度 |
| 4 | reshape | (B, 6, 256, 32, 88) | 通道当 C |
| 5 | LiDAR 投影深度图 | (B, 6, 1, 256, 704) | 稀疏，大部分是 0 |
| 6 | dtransform 后 | (B·6, 64, 32, 88) | 深度图降到 1/8 |
| 7 | cat + depthnet | (B·6, 320, 32, 88) → (B·6, 198, 32, 88) | 118 深度 + 80 特征 |
| 8 | 外积 | (B, 6, 118, 32, 88, 80) | `[B,N,D,fH,fW,C]` |
| 9 | get_geometry | (B, 6, 118, 32, 88, 3) | 增广 LiDAR 坐标系，米 |
| 10 | bev_pool | (B, 80, 360, 360) | 0.3 m/格 |
| 11 | downsample ×2 | **(B, 80, 180, 180)** | 0.6 m/格 |
| 12 | 体素化 | (120000, 10, 5) → (120000, 5) | 格内均值 |
| 13 | SparseEncoder 稠密化 | (B, 128, 180, 180, 2) | 每 stage stride 2 |
| 14 | 折 z | **(B, 256, 180, 180)** | 128 × 2 |
| 15 | cat | (B, 336, 180, 180) | [相机 80, LiDAR 256] |
| 16 | ConvFuser | (B, 256, 180, 180) | |
| 17 | SECOND | (B, 128, 180, 180) + (B, 256, 90, 90) | |
| 18 | SECONDFPN | **(B, 512, 180, 180)** | 256+256 |
| 19 | head shared_conv | (B, 128, 180, 180) | |
| 20 | dense_heatmap | (B, 10, 180, 180) | 32400 个位置取 top-200 |
| 21 | proposal 输出 | (B, 10, 200) 等 | center/height/dim/rot/vel |

---

## 6. 几个容易搞错的点

1. **两个分支都有"折 z"，但不是一回事。**
   - 相机分支：`zbound: [-10, 10, 20]` → `nz = 1`，`bev_pool` 之后的 `unbind(dim=2) + cat` 是**恒等操作**（相机 BEV 本来就是 2D 的，没有 z 可以折）。
   - LiDAR 分支：稀疏卷积真的分过 z（41 → 5 → 2 格），最后把 2 格折进通道变成 256，**信息没丢，只是换了位置**（见 3.3）。

   别把这两处混在一起看。

2. **LiDAR 的 256 通道是折出来的。** 不是某个卷积层输出 256。`output_channels: 128`，乘上 `conv_out` 留下的 2 格 z 才是 256。改 `sparse_shape` 的 z 或 `conv_out` 的 stride，这个数会变，`ConvFuser.in_channels` 也得跟着改。

3. **`features` 的顺序绑定 `in_channels`。** `[相机, LiDAR]` ↔ `[80, 256]`，训练/推理两条路径靠 `features[::-1]` 保证一致。

4. **Swin 的第 0 级被丢了。** `out_indices: [1,2,3]` + neck 取 `x[0]` → LSS 的输入是 1/8（32×88），不是 1/4。`vtransform.feature_size` 写的就是 `image_size // 8`。

5. **相机分支依赖点云。** DepthLSS 的深度图来自 LiDAR 点投影，所以 `extract_camera_features` 的参数里有 `points`。纯相机模型走的是 `lssfpn` 那套配置（普通 `LSSTransform`）。

6. **两个分支共用一个刚体变换。** 相机侧 frustum 走到最后要乘 `lidar_aug_matrix`，点云本身也被 `GlobalRotScaleTrans` 处理过——同一个矩阵。少了这步，两条 180×180 的图就对不齐。

7. **判断某个数从哪来，要看配置链而不是 `convfuser.yaml`。** 这个文件只有 6 行，真正的超参在路径上的 5 层父配置里（torchpack 递归合并，不是 mmcv 的 `_base_`）。

8. **`main` 分支比论文更杂，读的时候注意版本。** 当前 `mmdet3d/models/vtransforms/base.py` 里有 `use_points='radar'`、`height_expand`、`add_depth_features`、`boolmask2idx`（ONNX 兼容用）这些论文里没有的分支，`bevfusion.py` 里也有成套注释掉的 radar 代码。其中 `add_depth_features` 默认 `True` 时会把点特征再拼到深度图上（depth 不再是 1 通道），和 `DepthLSSTransform.dtransform` 第一层 `Conv2d(1, 8, 1)` 对不上——**本文第 2.4～2.5 节按 1 通道这条主线写**，读主线以外的分支时以自己那份配置为准。

---

## 7. 参考

- 论文：<https://arxiv.org/abs/2205.13542>
- 代码：<https://github.com/mit-han-lab/bevfusion>
- 配置入口：`configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml`
- 关键文件：
  - `mmdet3d/models/fusion_models/bevfusion.py` — 整体 forward
  - `mmdet3d/models/vtransforms/depth_lss.py` / `base.py` — 2D→3D→BEV
  - `mmdet3d/models/necks/generalized_lss.py` — 相机 FPN
  - `mmdet3d/models/backbones/sparse_encoder.py` — 稀疏卷积 + 折 z
  - `mmdet3d/ops/bev_pool/bev_pool.py` — BEV pooling 算子
  - `mmdet3d/models/heads/bbox/transfusion.py` — 检测头
