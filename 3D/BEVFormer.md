# BEVFormer 完整流程笔记

> **一句话**：BEVFormer 预先在地上铺一张 `200×200` 的 **BEV 网格当 query**（每格一个 256 维向量），用 **空间交叉注意力** 让每个格子去 6 路环视相机里"找自己那一小块地方"的图像特征，再用 **时序自注意力** 把上一帧按自车运动对齐过来的 BEV 特征融进来——一次前向得到一张 `200×200×256` 的统一 BEV 特征图，3D 检测和 BEV 分割共用。
>
> **论文**：Li et al., *BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers*, ECCV 2022（arXiv **2203.17270**）
>
> **代码**：`github.com/fundamentalvision/BEVFormer`。本篇所有代码引用均来自该仓库 `master` 分支，超参以 `projects/configs/bevformer/bevformer_base.py` 为准——**代码和论文有几处对不上，本文都标出来了**。
>
> **前置**：**`Deformable-Attention.md`（必读）**——BEVFormer 的两个注意力模块都是可变形注意力的 3D 改造版，不理解多尺度可变形注意力读不下去；`DETR.md`、`Deformable-DETR.md`——检测头完全是 DETR 那一套（匈牙利一对一匹配 + 集合损失 + **无 NMS**）。
>
> **贯穿例子**：nuScenes 的一帧，**6 路环视相机、每张 `1600×900`**。BEV 网格 `200×200`、每格 `0.512 m`、覆盖 X/Y ∈ `[-51.2, 51.2] m`、Z ∈ `[-5, 3] m`。场景里 **3 辆车 + 2 个行人**。全文跟着 **BEV 第 91 行第 124 列的那个格子**走——它对应真实位置 `(x = 12.544 m, y = -4.352 m)`，那里停着 3 辆车中的一辆。

---

## 1. 要解决什么问题：相机做 BEV 的四条路

相机做 3D 检测的根本难题只有一句话：**图像里没有深度**。2D 检测器能告诉你"那有辆车"，但说不出"多远"。围绕"深度从哪来"，学界走了四条路：

| 路线 | 代表 | 深度怎么来 | 多相机怎么融 | 时序怎么融 | 参考点 |
|---|---|---|---|---|---|
| ① 单目直接回归 3D | FCOS3D | **网络隐式学** | 各相机各跑一遍，后处理合并 | 无 | 无 |
| ② 显式估深度 → 抬升 | Lift-Splat / VPN / BEVDet / BEVDepth | **每个像素预测深度分布**，按深度把特征"喷"进体素 | BEV 空间 | 叠历史 BEV | 无（栅格化，不可学） |
| ③ query 去图像里查 | DETR3D | 无显式深度 | object query → 图像 | 无 | **网络逐层预测** 3D 点 |
| ④ **BEVFormer** | —— | 无显式深度 | **BEV 空间** | **BEV 空间 + 注意力** | **固定 BEV 网格 + 相机投影** |

②③的毛病很具体：

- **②依赖深度估计的质量**——深度错，特征就喷到错误的位置；BEVDepth 要额外加深度监督才稳；
- **③的参考点是网络自己预测的，逐层累积误差**，每层只采极少几个点，整个性能被"参考点猜得准不准"卡住。这一点在 `Deformable-DETR.md` 里已经见过：decoder 里 query 预测参考点，而这里更极端——**参考点就是唯一的空间线索**。

BEVFormer 的答案是**把方向反过来**：

> **不要"从 query 去猜图像里的位置"，而是"先在地上铺好格子，让每个格子按相机内外参算出自己在图像里对应的位置"。**

参考点不再是网络预测的，而是**由 BEV 网格 + `lidar2img` 矩阵直接算出来的固定值**——训练第一个 iteration 就是准的。这个"固定参考点"是 BEVFormer 和 DETR3D 最本质的分野。

**顺带，BEV 表示本身就是收益**：6 路相机的视角跳变、不同任务（检测/分割/占用）的表示差异、时序融合的坐标系漂移，全都被"先折进一张 BEV 图"这一步抹平了。

---

## 2. 整体结构（一张图）

```
      6 路环视相机（FRONT / FL / FR / BACK / BL / BR），每张 1600×900
        │
        ▼  ResNet-101-DCN（out_indices=(1,2,3) → C3/C4/C5）+ FPN（num_outs=4）
       P3 1/8    P4 1/16    P5 1/32    P6 1/64        每层 256 通道
      116×200    58×100     29×50      15×25          每相机 30825 个 token
        │  + cams_embeds(6×256) + level_embeds(4×256)
        ▼  flatten → (num_cam=6, 30825, bs, 256)
        │
  ┌──────────────────────────────────────────────────────────────┐
  │  BEV query  = nn.Embedding(200×200, 256)     → [40000, 256]  │  ★ 可学习栅格
  │  +  can_bus MLP(18 → 128 → 256)                              │  ★ 自车状态
  │  +  可学习位置编码（LearnedPositionalEncoding, 200×200）     │
  └──────────────────────────────────────────────────────────────┘
        │
        ▼  Encoder × 6 层（每层：TSA → Norm → SCA → Norm → FFN → Norm，Post-Norm）
  ┌──────────────────────────────────────────────────────────────┐
  │ ① TemporalSelfAttention (TSA)                                 │
  │      K/V = [对齐后的上一帧 BEV (40000) ; 当前 BEV (40000)]     │
  │ ② SpatialCrossAttention (SCA)                                 │
  │      K/V = 6 个相机的 30825×6 个 token                         │
  │      每个 BEV 格子 → 4 个高度锚点 → 投影到图像 → 采样            │
  └──────────────────────────────────────────────────────────────┘
        │
        ▼  bev_embed [40000, 256]  ← 这就是产物：200×200×256 的 BEV 特征图
        │
        ├──────────► 分割头 / 占用头 / ……（论文里验证了多任务）
        │
        ▼  检测头：900 个 object query（Embedding(900, 512) 拆成 query + query_pos）
           reference_points = sigmoid(Linear(256→3)(query_pos))      ★ 3D 参考点
        │
        ▼  Decoder × 6 层
           ① 自注意力：900 个 query 互相看
           ② 交叉注意力 → CustomMSDeformableAttention，在 BEV 平面上采样
           ③ FFN；每层回归出新的 3D 参考点（inverse_sigmoid 残差 + detach）
        │
        ▼  每层 decoder 各接一套独立头：分类 Linear(256→10)、回归 Linear(256→10)
           [cx, cy, w, l, cz, h, sinθ, cosθ, vx, vy]
        │
        ▼  训练：匈牙利匹配 + 集合损失（Focal + L1）；推理：卡阈值，**无 NMS**
```

**三个和 DETR 系列比"一眼看出不一样"的地方**：

1. **query 不再只有 900 个，而是 40000 个**——BEV 网格本身就是 query，这是一次稠密化；
2. **encoder 里有两个注意力，而且是串联的**（DETR 的 encoder 只有自注意力）：先时序、再空间；
3. **decoder 的参考点是 3 维的**（`Linear(256→3)`），比 Deformable DETR 的 2D 多了一个高度 `z`。

---

## 3. 逐模块拆解 + 维度流水账

### 3.0 维度总表（贯穿例子）

| 步骤 | 操作 | 张量形状 |
|---|---|---|
| 输入 | 6 路相机，每张 `1600×900`，pad 到 32 的倍数 → `1600×928` | `[1, 6, 3, 928, 1600]` |
| Backbone | ResNet-101-DCN，`out_indices=(1,2,3)` | C3 `[1,6,512,116,200]`、C4 `[1,6,1024,58,100]`、C5 `[1,6,2048,29,50]` |
| FPN | `num_outs=4`，全部 256 通道 | P3 `116×200`、P4 `58×100`、P5 `29×50`、P6 `15×25` |
| 每相机 token 数 | `23200+5800+1450+375` | **30825** |
| 相机/层级编码 | `cams_embeds(6×256)`、`level_embeds(4×256)` | 逐元素相加 |
| flatten | 每层 `flatten(3)` 后 4 层拼接，再 permute 成 encoder 要的布局 | `[6, 30825, 1, 256]`（num_cam, token, bs, C）|
| BEV query | `nn.Embedding(40000, 256)` | `[40000, 256]` |
| can_bus | `Linear(18→128) → ReLU → Linear(128→256) → ReLU → LN` | `[1, 256]` 广播加 |
| BEV 位置编码 | `LearnedPositionalEncoding(num_feats=128, row=200, col=200)` | `[40000, 256]` |
| Encoder | 6 层 ×（TSA + SCA + FFN） | `[40000, 256]` |
| **产物** | `bev_embed`（拆开就是 BEV 特征图） | `[40000, 256]` ≡ `200×200×256` |
| object query | `nn.Embedding(900, 512)` 拆两半 | query `[900,256]`、query_pos `[900,256]` |
| 3D 参考点 | `sigmoid(Linear(256→3)(query_pos))` | `[900, 3]` |
| Decoder | 6 层 ×（自注意力 + 交叉注意力 + FFN） | `[6, 900, 256]` |
| 分类头 | `2×(Linear+LN+ReLU) + Linear(256→10)` | `[6, 900, 10]` |
| 回归头 | `2×(Linear+ReLU) + Linear(256→10)` | `[6, 900, 10]` |

**几个数怎么来的**：

```python
# BEV 网格与分辨率（config 里写死）
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
bev_h_ = bev_w_ = 200
# grid_length = (real_h/bev_h, real_w/bev_w)
#             = (102.4/200, 102.4/200) = (0.512, 0.512)   ← 每格 0.512 米
```

【贯穿例子】**格子 ↔ 真实坐标的换算**（后面所有计算都靠它）：

```
x_real = (col + 0.5) / 200 × 102.4 − 51.2
y_real = (row + 0.5) / 200 × 102.4 − 51.2
```

代进第 91 行第 124 列：

```
x = (124 + 0.5)/200 × 102.4 − 51.2 = 0.6225 × 102.4 − 51.2 = 63.744 − 51.2 =  12.544 m
y = ( 91 + 0.5)/200 × 102.4 − 51.2 = 0.4575 × 102.4 − 51.2 = 46.848 − 51.2 =  −4.352 m
```

多出来的 `0.5` 是**格心惯例**——`get_reference_points` 用的是 `linspace(0.5, W-0.5, W)`，指的是格子正中心，不是左边界。这个 `0.5` 后面还会惹出一件事（见 §3.3 高度锚点）。

### 3.1 Backbone + FPN：一个论文和代码对不上的地方

```python
img_backbone=dict(
    type='ResNet', depth=101, num_stages=4,
    out_indices=(1, 2, 3),                    # 只取 C3/C4/C5（1/8、1/16、1/32）
    frozen_stages=1,                          # stem 冻结
    norm_cfg=dict(type='BN2d', requires_grad=False), norm_eval=True,
    style='caffe',
    dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
    stage_with_dcn=(False, False, True, True))    # 只在 stage3/4 用可变形卷积
img_neck=dict(type='FPN', in_channels=[512, 1024, 2048], out_channels=256,
              start_level=0, add_extra_convs='on_output', num_outs=4, ...)
```

三件事：

1. **backbone 用 `ResNet-101-DCN`，并且加载 FCOS3D 的预训练权重**（`load_from='ckpts/r101_dcn_fcos3d_pretrain.pth'`）。这不是小事：BEVFormer 从零训 24 epoch 很难追上，**必须从单目 3D 检测的 checkpoint 起步**。换 backbone 时这一条最容易漏；
2. **DCN 只加在 stage3/4**（`stage_with_dcn=(False, False, True, True)`）——浅层的几何信息还是老老实实用普通卷积；
3. FPN 的 `num_outs=4` + `add_extra_convs='on_output'`：3 个输入（C3/C4/C5）经横向连接出 P3/P4/P5，再在 **P5 的输出（已经是 256 通道）** 上做一个 `3×3, stride 2` 卷积造出 **P6**。这就是 `FPN.md` §3.4 里说的"P6 的第二种来源"。

> ⚠️ **论文 vs 代码 #1**：论文 4.2 节写的是"multi-scale features from FPN with sizes of `1/16`, `1/32`, `1/64`"——**三层**。而官方 base config 实际喂了 **四层**（多一个 `1/8`）：`num_feature_levels=4`、`level_embeds` 是 4 行、`get_bev_features` 里 `for lvl, feat in enumerate(mlvl_feats)` 遍历全部。照代码实现，别照论文。

【贯穿例子】每相机 token 数 = `116×200 + 58×100 + 29×50 + 15×25 = 23200 + 5800 + 1450 + 375 = **30825**`；6 个相机 = **184950** 个图像 token。这个数记住，§3.3 要用来算账。

### 3.2 BEV query 与 can_bus：`40000` 个可学习栅格

```python
# BEVFormerHead._init_layers
self.bev_embedding   = nn.Embedding(bev_h * bev_w, embed_dims)      # 40000 × 256  ★
self.query_embedding = nn.Embedding(num_query, embed_dims * 2)      # 900 × 512（检测头的）
```

`bev_queries` 是**纯可学习的参数**（没有从图像初始化，也没有从点云初始化）。它和 DETR 的 object query 是同一个思想：**一组学出来的"槽位"，让注意力去填内容**。区别是这里的槽位有明确的**空间含义**——第 `p` 个槽位就代表 BEV 上第 `p` 个格子。

位置上再叠两样东西：

```python
bev_mask = torch.zeros((bs, bev_h, bev_w))
bev_pos  = self.positional_encoding(bev_mask)      # LearnedPositionalEncoding → [40000, 256]
```

**位置编码是学出来的**（`nn.Embedding(200,128)` + `nn.Embedding(200,128)` 拼起来，`num_feats=128` 是"对数"所以输出 256 维），不是正弦的。因为 BEV 网格是固定的 200×200，直接学一张位置表最省事。

**can_bus 加到 query 上**（这一步很多人第一次看会愣）：

```python
can_bus   = bev_queries.new_tensor([each['can_bus'] for each in img_metas])   # [bs, 18]
can_bus   = self.can_bus_mlp(can_bus)[None, :, :]                             # [1, bs, 256]
bev_queries = bev_queries + can_bus * self.use_can_bus
```

`can_bus` 是自车状态向量，代码里用到四项：

| 下标 | 含义 | 单位 |
|---|---|---|
| `[0]`, `[1]` | 帧间自车位移 `delta_x`, `delta_y` | 米 |
| `[-2]` | 自车朝向 `ego_angle` | **弧度**（代码 `can_bus[-2] / np.pi * 180` 转成度） |
| `[-1]` | 相对偏航角 `rotation_angle` | **度**（直接喂给 `torchvision` 的 `rotate`） |

**为什么要把它加进 query？** 因为同一张 BEV 栅格在自车转弯时对应的绝对朝向是变的。把自车姿态注入 query，等于告诉模型"我现在是斜着的"，让 BEV 特征有机会解耦"世界坐标"和"自车坐标"。

【贯穿例子】这一帧自车正在右转且向前走了 `delta = (0.8 m, 0.3 m)`：`translation_length = √(0.8²+0.3²) = 0.854 m`，`ego_angle ≈ 12°`，`rotation_angle ≈ 3°`。这两个数下面 §3.4 立刻要用。

### 3.3 空间交叉注意力（SCA）：全文最核心的一块

#### 论文公式

$$\text{SCA}(Q_p, F_t) = \frac{1}{|\mathcal{V}_{\text{hit}}|}\sum_{i \in \mathcal{V}_{\text{hit}}} \sum_{j=1}^{N_{\text{ref}}} \text{DeformAttn}\big(Q_p,\ \mathcal{P}(p,i,j),\ F_t^i\big)$$

翻译成人话：

1. 取 BEV 平面上第 `p` 个格子，**沿高度方向竖起来变成一根柱子**（pillar），在柱子上取 `N_ref = 4` 个 3D 参考点；
2. 用相机投影矩阵把这 4 个点投到 6 张图里，**只落在某些视图里**（这些叫命中视图 `V_hit`，其余的压根不参与）；
3. 在每个命中的视图里，以投影点为参考点做**可变形注意力**采样（`DeformAttn`）；
4. 对命中的视图**取平均**（就是那个 `1/|V_hit|`）。

#### 参考点：由 BEV 网格和相机参数直接算出，不经过网络

```python
# BEVFormerEncoder.get_reference_points，dim='3d'
zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar) / Z          # Z = 8 (= 3.0 − (−5.0))
xs = torch.linspace(0.5, W - 0.5, W) / W                             # W = 200
ys = torch.linspace(0.5, H - 0.5, H) / H                             # H = 200
ref_3d = torch.stack((xs, ys, zs), -1)                               # [4, 200, 200, 3]，全部归一化到 [0,1]
```

然后 `point_sampling` 把它反归一化回真实世界坐标，再做一次齐次变换：

```python
reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]   # x
reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]   # y
reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]   # z
reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)      # 齐次
reference_points_cam = torch.matmul(lidar2img, reference_points)     # [4, B, 6, 40000, 4]
```

**四个高度锚点到底在哪？** 论文 4.2 节说"uniformly sampled from `-5` to `3` meters"，但代码是 `linspace(0.5, Z-0.5, 4)/Z`——**两端各内缩半格**：

| 锚点 | 归一化值 | 反归一化后 (m) |
|---|---|---|
| #0 | 0.5/8 = 0.0625 | **−4.5** |
| #1 | 2.8333/8 = 0.35417 | **−2.167** |
| #2 | 5.1667/8 = 0.64583 | **+0.167** |
| #3 | 7.5/8 = 0.9375 | **+2.5** |

间距均匀（`2.333 m`），但整条区间是 `[-4.5, 2.5]` 而不是 `[-5, 3]`。**4 米高的卡车顶部就在采样范围外**——论文里 `mATE` 高、大车/远景差，跟这条 `Z` 范围有关系。

#### 命中判定：`bev_mask`（两层过滤 + 一层视锥判断）

```python
bev_mask = (reference_points_cam[..., 2:3] > 1e-5)                    # ① 深度为正（在相机前方）
reference_points_cam[..., 0:2] /= torch.maximum(reference_points_cam[..., 2:3], eps)   # ② 除以深度 → 归一化平面
reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]       # ③ 除以图像宽 → [0,1]
reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]       # ④ 除以图像高 → [0,1]
bev_mask = bev_mask & (0 < x < 1) & (0 < y < 1)                       # ⑤ 落在图像范围内
```

维度上：`reference_points_cam` 最终是 `[num_cams, bs, num_query, D, 2]`，`bev_mask` 是 `[num_cams, bs, num_query, D]`（`D = 4` 个高度锚点）。

【贯穿例子】第 91 行第 124 列的格子，4 个高度锚点投影下来：

| 相机 | 高度 −4.5m | −2.17m | +0.17m | +2.5m | 命中？ |
|---|---|---|---|---|---|
| FRONT | 图外(y<0) | ✓ | ✓ | ✓ | **✓** |
| FRONT_LEFT | 图外 | 图外 | 图外(x>1) | 图外 | ✗ |
| **FRONT_RIGHT** | 图外 | ✓ | ✓ | ✓ | **✓** |
| BACK / BL / BR | 深度 < 0 | 深度 < 0 | 深度 < 0 | 深度 < 0 | ✗ |

→ `V_hit = {FRONT, FRONT_RIGHT}`，**`|V_hit| = 2`**。这辆车高 `1.5 m`，所以真正装在车身上的采样点来自 `+0.17m` 那个锚点附近；`−4.5m`（地下 4.5 米）和 `+2.5m`（车顶上方 1 米）那两个点投到的图像区域是背景/天空，**这是设计上允许的噪声**——`DeformAttn` 的注意力权重会把它们压低。

#### 可变形注意力在采样什么（这是最容易糊涂的地方）

官方注释写得最清楚，直接引：

```python
"""
For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
For each referent point, we sample `num_points` sampling points.
For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
"""
```

代码里的乘法关系是：

```python
self.sampling_offsets    = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)   # 8×4×8×2 = 512
self.attention_weights   = nn.Linear(embed_dims, num_heads * num_levels * num_points)       # 8×4×8   = 256

sampling_offsets  = sampling_offsets.view(bs, nq, heads, num_levels, num_points, 2)         # num_points = 8
sampling_offsets  = sampling_offsets.view(bs, nq, heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, 2)
#                                                                     ↑ 8 // 4 = 2
```

> ⚠️ **论文 vs 代码 #2**：config 里写的是 `num_points=8`，但代码把这个 `8` 又按 `num_Z_anchors = 4` 除开了。**真正每个高度锚点分到的是 `8 / 4 = 2` 个采样偏移**，而论文 4.2 节说的是"For each reference point on 2D view features, we use **four** sampling points around this reference point for each head"——**四对二**。所以"每个 BEV 格子一次采样几个点"这个数，**以代码为准**：
>
> ```
> 每个 BEV 格子 × 每个头 × 每个命中相机
>   = num_levels(4) × num_points(8)
>   = 4 尺度 × (4 个高度锚点 × 2 个偏移)
>   = 32 个采样位置
> ```
>
> 8 个头 → **每个格子每个相机 256 个采样点**。注意力权重也是 32 个/头，在这 32 个位置上 `softmax` 归一化（和为 1，没有重复使用）。

#### 省了多少算力：`SpatialCrossAttention` 的 rebatch

这一段代码是**全文最值得抄的工程技巧**：

```python
D = reference_points_cam.size(3)
indexes = []
for i, mask_per_img in enumerate(bev_mask):
    index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)   # 这个相机命中了哪些格子
    indexes.append(index_query_per_img)
max_len = max([len(each) for each in indexes])                            # 取最长的那个

queries_rebatch             = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
reference_points_rebatch    = reference_points_cam.new_zeros([bs, self.num_cams, max_len, D, 2])
for j in range(bs):
    for i, reference_points_per_img in enumerate(reference_points_cam):
        index_query_per_img = indexes[i]
        queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
        reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
```

> **每个相机只处理"落在自己视野里的那些格子"，其余格子这条 GPU 路径上根本不存在。**

**这一步能省多少？** 用贯穿例子算账（单层 encoder、8 头合计）：

| 方案 | 注意力权重个数 | 显存 |
|---|---|---|
| 全局注意力（vanilla MHSA） | `40000 × 184950 × 8` ≈ **592 亿** | 光注意力矩阵就是 `40000×184950×4B ≈ 29.6 GB / 头`；论文退化成 `100×100` + 单尺度 + fp16 才压到 `~36 GB` |
| **BEVFormer SCA** | `Σ_cam(hit 格子数) × 32 × 8` | 假设 6 个相机共命中约 2 万个格子 → `20000 × 32 × 8` ≈ **512 万** |

**差了约 1.2 万倍。** 这就是 §1 里说"全局注意力为什么不行"的量化答案，也是论文 Table 5 里 global attention 只能靠 `100×100` 小网格 + 单尺度 + fp16 硬撑的原因。

> 注意别把这里的倍数误当成"总 FLOPs 也差一万倍"——论文 Table 5 里 global 版和 local 版的总 FLOPs 是 `1245.1G` vs `1303.5G`，**几乎一样**，因为**总 FLOPs 被 backbone 主导**（见 §5.5：backbone 391 ms，BEVFormer 才 130 ms）。省下来的是**注意力矩阵的显存**，而显存才是这里真正的瓶颈。

#### 收尾：按相机数取平均

```python
count = bev_mask.sum(-1) > 0                # [num_cams, bs, num_query]：这个相机命中了吗
count = count.permute(1, 2, 0).sum(-1)      # [bs, num_query]：一共几个相机命中
count = torch.clamp(count, min=1.0)
slots = slots / count[..., None]            # 就是论文里的 1/|V_hit|
slots = self.output_proj(slots)
return self.dropout(slots) + inp_residual
```

注意 `bev_mask.sum(-1)` 是**对 4 个高度锚点求和**——只要**任意一个高度**投进图像，这个相机就算命中。所以"回车顶上方"不会让车旁边那个格子丢掉前视相机。

【贯穿例子】我们的格子 `|V_hit| = 2`，所以两个相机的结果相加后除以 2。

### 3.4 时序自注意力（TSA）：怎么把上一帧"搬"过来

#### 论文公式

$$\text{TSA}(Q_p,\ \{Q, B'_{t-1}\}) = \sum_{V \in \{Q,\ B'_{t-1}\}} \text{DeformAttn}(Q_p,\ p,\ V)$$

`B_{t-1}` 是上一帧存下来的 BEV 特征，`B'_{t-1}` 是按自车运动**对齐到当前帧坐标系**之后的版本。**注意这里是求和不是平均**（代码里是 `output.mean(-1)`，见下）。

#### 第一步：对齐（旋转 + 平移）

```python
# PerceptionTransformer.get_bev_features —— 旋转：直接对 BEV 特征图做 2D 旋转
if self.rotate_prev_bev:
    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]                 # 度
    tmp_prev_bev = prev_bev[:, i].reshape(bev_h, bev_w, -1).permute(2, 0, 1)
    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center)   # center=[100, 100]
    prev_bev[:, i] = tmp_prev_bev.permute(1, 2, 0).reshape(bev_h * bev_w, 1, -1)[:, 0]
```

用的是 `torchvision.transforms.functional.rotate`——**在 BEV 特征图上直接做双线性旋转**，旋转中心是图像中心 `[100, 100]`（正好是 `200×200` 的中心，即自车位置）。这是一个很"粗暴但有效"的做法：不做任何几何推导，就是把特征图转一下。

```python
# 平移：转成"归一化的 BEV 栅格偏移"
grid_length_y, grid_length_x = grid_length[0], grid_length[1]              # 0.512, 0.512
translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
translation_angle  = np.arctan2(delta_y, delta_x) / np.pi * 180
bev_angle = ego_angle - translation_angle
shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
shift   = bev_queries.new_tensor([shift_x, shift_y]).permute(1, 0)         # [bs, 2]
```

**为什么平移不用改特征图，而是改成"参考点偏移"？** 因为**平移等价于"每个格子去看原来的邻居"**——把参考点整体挪一下就行，不需要动特征图本身。这个思路和卷积的平移等变性是同一件事。

【贯穿例子】`translation_length = 0.854 m`，`bev_angle = 12° − 30° = −18°`：

```
shift_x = 0.854 × sin(−18°) / 0.512 / 200 = 0.854 × (−0.309) / 102.4 = −0.00258
shift_y = 0.854 × cos(−18°) / 0.512 / 200 = 0.854 × ( 0.951) / 102.4 = +0.00793
```

也就是说，历史 BEV 的参考点整体挪了 **`(−0.0026, +0.0079)`**（归一化单位），换算成格数约 `(−0.51, +1.59)` 格——**大约挪了一格半**，和自车走了 0.85 米 ≈ 1.66 格是对得上的（0.85/0.512 = 1.66）。

> ⚠️ **官方自己标了一个 bug**。编码器里这段：
>
> ```python
> # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
> shift_ref_2d = ref_2d.clone()
> shift_ref_2d += shift[:, None, None, :]
> ```
>
> 官方注释明确说**这里保留了一个 bug，只是为了复现论文数值**。有意思的是注释要求的写法和实际写法是同一句——争议落在紧接着的 `+= shift` 上（社区普遍认为是 `shift` 被重复施加了一次）。**我在代码层面没能单独确认这一点**，但结论很明确：**迁移/复现时不要把这段当成正确实现照抄**，重新推一遍自车运动补偿反而更靠谱。

#### 第二步：`K/V` 里放什么

```python
if prev_bev is not None:
    prev_bev = prev_bev.permute(1, 0, 2)
    prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)     # ★ 拼起来
    hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(bs*2, len_bev, 1, 2)
else:
    hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(bs*2, len_bev, 1, 2)
```

- **K/V = `[对齐后的 prev_bev ; 当前 BEV query]`**，两条各 `40000` 个 token，拼成 `bs*2 × 40000 × 256`；
- **参考点也拼**：`[shift_ref_2d ; ref_2d]`，前一半是**平移过的位置**（去历史特征里找），后一半是**自己的位置**（在当前特征里找）；
- **`prev_bev is None` 时（序列第一帧）**：K/V 退化成 `[Q, Q]`，参考点退化成 `[p, p]`——就是**两个分支看同一份东西**，TSA 退化成普通的可变形自注意力。这正是论文说的 "for the first sample of each sequence, the temporal self-attention will degenerate into a self-attention without temporal information"。

#### 第三步：offset 由 `Q` 和 `B'` 拼接后预测（和标准可变形注意力不同）

```python
self.sampling_offsets = nn.Linear(embed_dims*self.num_bev_queue, num_bev_queue*num_heads*num_levels*num_points*2)
#                       ↑ 输入是 512 = 256×2，输出也翻倍 —— 两个分支各自一套 offset/weight
...
query = torch.cat([value[:bs], query], -1)      # ★ 512 维：当前 query 拼上一帧的 BEV 特征
sampling_offsets  = self.sampling_offsets(query).view(bs, nq, heads, 2, num_levels, num_points, 2)
attention_weights = self.attention_weights(query).view(bs, nq, heads, 2, num_levels * num_points)
```

这是论文里点名的设计差异：**offset 不是只由 query 预测，而是由"当前 query + 上一帧 BEV 特征"拼接后预测**。理由很直接——**要判断自己该往哪个方向找，得先知道上一帧那儿有什么**。消融实验（Table 8 的 `B.` 列）显示这一项值 `+0.4 NDS`（51.3 → 51.7）。

【贯穿例子】我们那个格子的 query `q` 拼上它在上一帧对齐后的特征 `b'`，凑成 512 维 → 预测出 2 组 offset/weight：**一组指向对齐后的历史 BEV（4 个采样点），一组指向当前 BEV（4 个采样点）**。

#### 第四步：两个分支求平均

```python
output = output.permute(1, 2, 0)                                  # (bs*2, nq, 256) → (nq, 256, bs*2)
output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
output = output.mean(-1)                                          # ★ 两个分支取平均
output = self.output_proj(output.permute(2, 0, 1))
return self.dropout(output) + identity
```

公式里写的是 `Σ`（求和），代码里是 `mean`——**以代码为准，是平均**。`num_levels=1`（只有一张 200×200 的 BEV 特征图）、`num_points=4`（类默认值，config 没改）。

### 3.5 检测头：把 BEV 特征图变成 3D 框

#### object query 的反常初始化：**输入全零**

```python
# PerceptionTransformer.forward
query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)   # 512 拆成两个 256
query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)                        # [900, 256] 位置
query     = query.unsqueeze(0).expand(bs, -1, -1)                            # [900, 256] 内容
reference_points = self.reference_points(query_pos).sigmoid()                # [900, 3]  ★ 3D 参考点
```

和 Deformable DETR 一模一样的配方——**`nn.Embedding(900, 512)` 拆成"内容"和"位置"两半，参考点由位置那一半预测**。但这里多了一维：**`Linear(256→2)` 变成了 `Linear(256→3)`**，参考点是 `(x, y, z)` 的 3D 点，不是 2D 点。

为什么需要 `z`？因为 BEV 只有平面信息，**箱子有多高必须自己带一个先验**。这也是 `z` 会被 decoder 逐层精修的原因（见下）。

#### 交叉注意力：在 BEV 平面上做一个普通的 2D 可变形注意力

```python
# DetectionTransformerDecoder.forward
reference_points_input = reference_points[..., :2].unsqueeze(2)   # 只取 (x,y)，丢掉 z  ★
output = layer(output, ..., reference_points=reference_points_input, ...)
```

`CustomMSDeformableAttention` 就是 `Deformable-Attention.md` 里那个标准 2D 可变形注意力，`num_levels=1`、`num_points=4`——**900 个 query 各自在 200×200 的 BEV 特征图上采 4 个点**。`z` 只是被用来算参考点，不参与采样。

算力上完全不成问题：`900 × 4 = 3600` 个采样点 vs 40000 个 BEV token，比 encoder 便宜 3 个数量级。

#### 逐层精修参考点（和 Deformable DETR 的迭代框精修同源）

```python
tmp = reg_branches[lid](output)                                   # 10 维：[cx,cy,w,l,cz,h,sin,cos,vx,vy]
new_reference_points = torch.zeros_like(reference_points)
new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])    # (x, y)
new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3]) # (z) ← 高度
new_reference_points = new_reference_points.sigmoid()
reference_points = new_reference_points.detach()                  # ★ 和 Deformable DETR 一样 detach
```

三个和 `Deformable-DETR.md` §4.5 完全对得上的点：

1. **必须在 logit 空间做残差**，所以有 `inverse_sigmoid`；
2. **只有中心点 `(x, y, z)` 参与残差**，`w/l/h` 是直接预测的。注意 **`z` 取的是回归输出的第 5 个元素 `tmp[..., 4:5]`，也就是 `cz`**——因为输出顺序被重排成了 `[cx,cy,w,l,cz,h,...]`（见下），**不是** nuScenes 标准的第 3 位。这就是"3D 参考点"里那一维的来源；
3. **参考点 `detach`**（"look forward once"）——跨层不回传梯度。

#### 两个头

```python
cls_branch: Linear(256,256)+LN+ReLU ×2 → Linear(256, 10)      # 10 类，无背景类（sigmoid + focal）
reg_branch: Linear(256,256)+ReLU    ×2 → Linear(256, 10)      # [cx,cy,w,l,cz,h,sinθ,cosθ,vx,vy]
code_weights = [1,1,1,1,1,1,1,1, 0.2, 0.2]                   # 速度两项降权
```

三个容易看漏的点：

1. **回归是 10 维**。3D 框本体是 9 维（中心 `x,y,z` + 尺寸 `w,l,h` + 朝向 `θ`），BEVFormer 把 `θ` 拆成 `(sinθ, cosθ)` 以避免 `±π` 处的角度回归跳变，再加 `(vx, vy)` 速度——**速度不是白送的：它正是时序信息带来的红利**（见 §5）；
2. **网络输出的顺序和 nuScenes 的标准顺序不一样**。`normalize_bbox` 做了一次重排：标准框是 `[cx,cy,cz,w,l,h,θ,vx,vy]`，而**回归头吐的是 `[cx,cy,w,l,cz,h,sinθ,cosθ,vx,vy]`**——`cz` 被挪到了第 5 位。这就是解码器里 `new_reference_points[..., 2:3] = tmp[..., 4:5]`（**取第 5 位当 `z`**）的原因，也解释了 `code_weights` 里为什么是第 9、10 位降权（对应 `vx, vy`）；
3. **`w, l, h` 是在 log 空间回归的**（`w = bboxes[..., 3:4].log()`，解码时 `.exp()` 回去）。尺寸是乘性量，log 空间里梯度尺度更均匀。

---

## 4. 训练

### 4.1 时序数据怎么组装：4 帧滑动窗口

```python
queue_length = 4    # 每个序列含 4 帧

# BEVFormer.forward_train
len_queue   = img.size(1)          # 4
prev_img    = img[:, :-1, ...]     # t-3, t-2, t-1
img         = img[:, -1,  ...]     # t（当前帧，唯一算梯度的）
prev_bev    = self.obtain_history_bev(prev_img, prev_img_metas)
```

```python
def obtain_history_bev(self, imgs_queue, img_metas_list):
    """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated."""
    self.eval()
    with torch.no_grad():                                    # ★ 历史帧全程 no_grad
        prev_bev = None
        for i in range(len_queue):                           # 逐帧滚动
            if not img_metas[0]['prev_bev_exists']:
                prev_bev = None
            prev_bev = self.pts_bbox_head(img_feats, img_metas, prev_bev, only_bev=True)
    self.train()
    return prev_bev
```

四个要点：

1. **只对当前帧算梯度**。历史 3 帧在 `torch.no_grad()` + `self.eval()` 下跑，**只跑 encoder**（`only_bev=True`）——检测头完全不参与；
2. **历史 BEV 是逐帧迭代出来的**（`t-3 → t-2 → t-1`），不是各自独立算再拼——所以时序信息是**递归**传递的，和 RNN 的展开是一回事；
3. **`self.eval()` / `self.train()` 的切换**：因为 backbone 里有 BN，历史帧前向必须走 eval 模式，否则会污染 running stats。这个细节很容易在改写时丢掉；
4. **`prev_bev_exists`**：序列第一帧没有历史，`prev_bev = None`，TSA 自动退化（§3.4）。

> **所以 BEVFormer 的时序是"递推"而不是"堆叠"**：理论上第 `t` 帧能看到 `t-3` 的信息，因为 `B_{t-1}` 里已经含有 `B_{t-2}` 的东西。这比 "BEVDet4D / BEVFormer 之前那批工作" 直接把几帧 BEV 摞在通道维上更省算力，也更不容易被干扰（论文 3.4 节原话）。

### 4.2 损失：只有两项，没有"ego-motion loss"

```python
loss_cls = dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0)
loss_bbox = dict(type='L1Loss', loss_weight=0.25)
loss_iou  = dict(type='GIoULoss', loss_weight=0.0)     # ← 权重 0，等于关掉
```

**匹配仍是匈牙利一对一**（`HungarianAssigner3D`）：

| 代价项 | 类型 | 权重 |
|---|---|---|
| 分类 | `FocalLossCost` | **2.0** |
| 回归 | `BBox3DL1Cost` | 0.25 |
| IoU | `IoUCost` | **0.0**（配置里直接注释 *"Fake cost. This is just to make it compatible with DETR head."*）|

**没有 IoU 损失、没有 ego-motion 一致性损失**——和 Deformable DETR 一样只有"分类 focal + 框 L1"两项。6 层 decoder 每层都算一份损失（辅助损失）：`loss_dict['loss_cls']` 取最后一层，`d0~d4` 是其余各层，**每份权重都是 1**（`losses_cls[-1]` 加上 5 个 `d{i}.loss_cls`，一共 6 项相加）。

`code_weights = [1,1,1,1,1,1,1,1,0.2,0.2]`：**速度两项降权到 0.2**。速度的量纲和位置不一样（m/s vs m），不降权会主导梯度。

### 4.3 超参（`bevformer_base.py`）

| 项 | 值 |
|---|---|
| epoch | **24** |
| optimizer | AdamW，`lr = 2e-4`，backbone `lr_mult = 0.1`（→ 2e-5） |
| weight decay | 0.01 |
| 梯度裁剪 | `max_norm = 35`（比 DETR 系的 0.1 松得多） |
| lr 调度 | CosineAnnealing，warmup 500 iter（`warmup_ratio = 1/3`），`min_lr_ratio = 1e-3` |
| 数据增强 | `PhotoMetricDistortionMultiViewImage`、**`GridMask`**（`ratio=0.5, prob=0.7`）|
| 预训练 | `r101_dcn_fcos3d_pretrain.pth`（**必须的**）|
| 训练显存 | 28500 MB（base，R101-DCN）|

归一化的 mean/std 是 `[103.530, 116.280, 123.675] / [1.0, 1.0, 1.0]` 且 **`to_rgb=False`**——**BGR 顺序**，因为 mmdet3d 读图默认走 OpenCV。这个坑换框架时必踩。

**`GridMask`** 值得单独说：以 0.7 的概率在图上盖一层 `50%` 遮盖率的网格块。目的是**不让模型靠"某个相机里目标长什么样"作弊**，逼它学会跨相机、跨时空推理——在 BEVFormer 里效果显著。

---

## 5. 结果与消融

### 5.1 nuScenes **test**（论文 Table 1）

| 方法 | 模态 | Backbone | NDS↑ | mAP↑ | mATE↓ | mAVE↓ |
|---|---|---|---|---|---|---|
| FCOS3D | C | R101 | 0.428 | 0.358 | 0.690 | 1.434 |
| PGD | C | R101 | 0.448 | 0.386 | 0.626 | 1.509 |
| DETR3D | C | V2-99* | 0.479 | 0.412 | 0.641 | 0.845 |
| DD3D | C | V2-99* | 0.477 | 0.418 | 0.572 | 1.014 |
| **BEVFormer-S** | C | R101 | 0.462 | 0.409 | 0.650 | 0.925 |
| **BEVFormer** | C | R101 | **0.535** | **0.445** | 0.631 | 0.435 |
| **BEVFormer-S** | C | V2-99* | 0.495 | 0.435 | 0.589 | 0.842 |
| **BEVFormer** | C | V2-99* | **0.569** | **0.481** | 0.582 | **0.378** |
| CenterPoint-Voxel | **L** | - | 0.655 | 0.580 | - | - |
| PointPainting | L&C | - | 0.581 | 0.464 | 0.388 | 0.247 |

\* V2-99 在深度估计任务上用额外数据预训练过。

**三行字读懂**：

1. **R101 版 BEVFormer（53.5 NDS）已经超过 R101 版 FCOS3D（42.8）10.7 个点**，和 V2-99 的 DETR3D（47.9）比也领先 5.6 个点；
2. **`mAVE` 从 DETR3D 的 0.845 掉到 0.378（腰斩一半以上）**——这是**时序信息的直接红利**。相机方法第一次在"速度估计"这一项上接近 LiDAR 方法（PointPainting 是 0.247）；
3. **但它仍然是相机方法**：CenterPoint-Voxel 用 LiDAR 有 65.5 NDS，差距还在。

### 5.2 nuScenes **val**（官方 Model Zoo）

| Backbone | 方法 | epoch | NDS | mAP | 训练显存 |
|---|---|---|---|---|---|
| R50 | BEVFormer-tiny | 24 | 35.4 | 25.2 | 6500 MB |
| R50 | BEVFormer-tiny_fp16 | 24 | 35.9 | 25.7 | - |
| R101-DCN | BEVFormer-small | 24 | 47.9 | 37.0 | 10500 MB |
| R101-DCN | **BEVFormer-base** | 24 | **51.7** | **41.6** | **28500 MB** |

**注意 val 和 test 的差距**：base 在 val 上是 51.7 / 41.6，在 test 上是 53.5 / 44.5——test 集更大更难但也更全，一般报 test。

### 5.3 消融：时序信息到底值多少（论文 Table 7 / Table 8）

**训练时用几帧**（Table 7）：

| #Frame | 1 | 2 | 3 | **4** | 5 |
|---|---|---|---|---|---|
| NDS | 0.448 | 0.490 | 0.510 | **0.517** | 0.517 |
| mAP | 0.375 | 0.388 | 0.410 | **0.416** | 0.412 |
| **mAVE** | 0.802 | 0.467 | 0.423 | **0.394** | 0.387 |

**这张表信息量最大**：

- **第 1 帧 = BEVFormer-S，44.8 NDS**；加上第 2 帧直接跳到 **49.0（+4.2）**，第 3 帧 51.0，**第 4 帧 51.7 后开始饱和**——这就是 `queue_length = 4` 的由来；
- **`mAVE` 从 0.802 掉到 0.467**，**一帧历史就把速度误差砍了 42%**。速度只能靠"两帧之间的位移"推出来，这是最直白的证据；
- **`mAP` 的增益远小于 `NDS`**（37.5 → 41.6，+4.1）。`NDS` 是 `mAP` 和 5 个误差项的加权，**时序主要在"误差项"上得分，不在"检出"上得分**。

**几个设计细节**（Table 8，baseline 第 4 行 = 51.7）：

| # | 对齐 ego-motion | 随机采 4/5 帧 | offset 用 Q+B' 预测 | NDS | mAP |
|---|---|---|---|---|---|
| 1 | ✗ | ✓ | ✓ | 51.0 | 41.0 |
| 2 | ✓ | ✗ | ✓ | 51.3 | 41.0 |
| 3 | ✓ | ✓ | ✗ | 51.3 | - |
| **4** | ✓ | ✓ | ✓ | **51.7** | **41.6** |

- **不做 ego-motion 对齐：−0.7 NDS**（51.0 vs 51.7）——两个时刻的 BEV 格子对不上同一个真实位置，融合就是错的；
- **offset 只用 query 预测（不拼 `B'`）：−0.4 NDS**；
- **随机采帧的数据增强：−0.4 NDS**。

### 5.4 消融：SCA 用什么注意力（论文 Table 5，在 **BEVFormer-S** 上做以排除时序干扰）

| 方法 | 注意力 | NDS↑ | mAP↑ | 参数 | FLOPs | 显存 |
|---|---|---|---|---|---|---|
| VPN* | - | 0.334 | 0.252 | 111.2M | 924.5G | ~20G |
| Lift-Splat* | - | 0.397 | 0.348 | 74.0M | 1087.7G | ~20G |
| BEVFormer-S† | **Global** | 0.404 | 0.325 | 62.1M | 1245.1G | **~36G** |
| BEVFormer-S‡ | Points | 0.423 | 0.351 | 68.1M | 1264.3G | ~20G |
| **BEVFormer-S** | **Local（可变形）** | **0.448** | **0.375** | 68.7M | 1303.5G | ~20G |

† global 版被迫退化成 `100×100` BEV + 单尺度 + fp16 权重（否则显存爆掉）；‡ 去掉预测的 offset/weight，只跟参考点交互。

**结论**：

- **全局注意力最差（40.4）而且最贵（36G）**——感受野最大不代表更好，**先验位置（参考点）比感受野更重要**；
- **只跟参考点交互（42.3）比全局好，但比可变形（44.8）差**——**参考点周围那一小圈局部区域是有用的**，这正是"可变形"相对"点采样"的价值；
- **BeV 系列（VPN / Lift-Splat）明显更差**（33.4 / 39.7）——**不估深度也能做好 BEV**，这是 BEVFormer 最重要的一个反直觉结论。

### 5.5 模型规模与延迟（论文 Table 6，单张 V100，输入 900×1600，R101-DCN）

| 配置 | 多尺度 | BEV 形状 | #Layer | Backbone(ms) | BEVFormer(ms) | Head(ms) | 端到端 FPS | NDS | mAP |
|---|---|---|---|---|---|---|---|---|---|
| **BEVFormer** | ✓ | 200×200 | 6 | **391** | 130 | 19 | **1.7** | **0.517** | **0.416** |
| A | ✗ | 200×200 | 6 | 387 | 87 | 19 | 1.9 | 0.511 | 0.406 |
| B | ✓ | 100×100 | 6 | 391 | 53 | 18 | 2.0 | 0.504 | 0.402 |
| C | ✓ | 200×200 | **1** | 391 | 25 | 19 | 2.1 | 0.501 | 0.396 |
| D | ✗ | 100×100 | **1** | 387 | **7** | 18 | **2.3** | 0.478 | 0.374 |

**这张表的核心不是"BEVFormer 慢"，而是"瓶颈根本不在 BEVFormer"**：

- **backbone 占了 391 ms，BEVFormer 自己只占 130 ms**——把 BEV 部分从 130 ms 砍到 7 ms（**快 18 倍**），端到端 FPS 只从 1.7 涨到 2.3。论文自己的结论：*"the bottleneck that limits the efficiency lies in the backbone"*；
- **6 层 encoder 从 130 ms 砍到 25 ms（×5.2 快），NDS 只掉 1.6 个点**（51.7 → 50.1）——**encoder 层数是性价比最高的旋钮**；
- **BEV 从 200×200 降到 100×100（4 倍省），NDS 只掉 1.3**（51.7 → 50.4）。想做轻量化，先动这两个。

### 5.6 Waymo / 多任务

- **Waymo**：BEV query 改成 `200×220`，X ∈ `[-35.0, 75.0] m`，Y ∈ `[-75.0, 75.0] m`，分辨率 `0.5 m`，自车在 BEV 的 `(70, 150)` 位置（因为 Waymo 相机拍不全四周，所以不是居中的）;
- **多任务**：同一次前向 + 检测头再加分割头，**联合训练反而检测更好**（52.0 NDS vs Lift-Splat 的 41.0），车道线分割 IoU 也涨 5.6 个点；
- **相机外参噪声**（附录 Fig. 6）：训练时加噪声能显著提升鲁棒性——**因为 BEVFormer 的参考点全靠外参算**，外参一歪，采样位置全错。这是它相对 DETR3D 的一个软肋，也是它必须做外参扰动的数据增强的原因。

---

## 6. 面试常见追问

- **Q：BEVFormer 和 DETR3D 的本质区别是什么？**
  A：**参考点从哪来**。DETR3D 的参考点是 object query 逐层预测的 3D 点，从零学起、逐层累积误差，整个性能被"参考点猜得准不准"卡死，而且每层只采 1 个点。BEVFormer 把 query 换成 **40000 个固定的 BEV 格子**，参考点由**网格位置 + `lidar2img` 直接算出**，第一个 iteration 就精确；再配合可变形注意力在参考点周围采一圈。一个是"学出来的稀疏点"，一个是"算出来的稠密网格"。

- **Q：为什么 BEV query 用可学习参数初始化，而不是从图像特征生成？**
  A：因为它代表的是**地面上的固定位置**，不含任何图像信息——它就是一个"空槽位"，等着 SCA 往里填。可学习参数在训练中会收敛成"哪种位置上的 BEV 特征长什么样比较有用"的先验。位置信息由**可学习位置编码 + can_bus** 提供，不靠查询内容。

- **Q：空间交叉注意力里 4 个高度锚点是干什么的？**
  A：BEV 是**平面**，只有 `(x, y)`，所以模型不知道"这个格子里有东西在多高"。把格子沿 `z` 竖成一根柱子，在 `[-4.5, 2.5] m`（代码实际值）上均匀取 4 个点投影到图像——**让同一个格子的不同高度分别去图像里"看一眼"**，由注意力权重决定哪个高度上的证据有用。没有这 4 个锚点，贴在远处的路面和近处的车顶会采到同一个点，无法区分。

- **Q：`num_points=8` 到底是"8 个采样点"还是"每个锚点 8 个"？**
  A：**都不是**。代码里 `sampling_offsets` 算出来 8 个，然后 `view(..., num_all_points // num_Z_anchors, num_Z_anchors, 2)` 把它按 4 个高度锚点除开，**每个锚点实际只分到 2 个偏移**。所以是 `4 个高度锚点 × 2 个偏移 × 4 个尺度 = 32 个采样位置 / 头`。论文正文写的是"每个参考点采 4 个点"，**和代码对不上，以代码为准**。

- **Q：为什么 decoder 里 `reference_points` 只取前两维，`z` 不就浪费了？**
  A：`z` 不参与 BEV 上的采样（BEV 特征图是平面的，只有 `x, y`），但它是**被逐层精修的连续变量**——每层用回归头输出的 `cz` 修正它（`inverse_sigmoid` 残差）。它的作用是让 box 的**高度估计**有一个可迭代的先验，同时参与 `mATE`/`mAAE` 这些高度相关指标的优化。

- **Q：时序是"堆叠"还是"递推"？梯度怎么传？**
  A：**递推**（RNN 式）。`B_{t-1}` 是由 `t-3, t-2, t-1` 逐帧滚动出来的，理论上有无限长的时序感受野。但**梯度不跨帧回传**：历史帧全程在 `torch.no_grad()` + `self.eval()` 下跑，只有当前帧算梯度。这是显存和稳定性的取舍，也让训练可以按序列并行。

- **Q：为什么用注意力融合时序，而不是直接把上一帧 BEV 加/拼过来？**
  A：因为**物体在动**。直接叠加假设"同一个格子在两帧里是同一个东西"，只对静止物体成立。TSA 让当前帧的 query **自己去历史 BEV 里找**（参考点是平移后的位置 + 可学习的 offset），运动物体也能对上。论文 Table 7 里 `mAVE` 从 0.802 掉到 0.394（−51%），就是这个机制在起作用。

- **Q：`prev_bev` 在第一帧怎么办？**
  A：`prev_bev_exists = False` → `prev_bev = None` → K/V 退化成 `{Q, Q}`、参考点退化成 `{p, p}`，**TSA 变成普通可变形自注意力**。所以 BEVFormer 天然兼容单帧输入，这也是 `BEVFormer-S`（永远传 `None`）能作为一个独立 baseline 的原因。

- **Q：BEVFormer 对相机外参敏感吗？**
  A：**很敏感，而且这是它的固有代价**。参考点是外参直接算出来的，外参有噪声时采样位置全错；而 DETR3D 的参考点是网络预测的，反而有一定"自适应"能力。论文附录专门做了外参噪声实验，结论是**训练时加噪声能显著改善鲁棒性**——实践中这几乎是一项必须做的数据增强。

- **Q：为什么 IoU 损失被关掉了？**
  A：官方配置里 `loss_iou` 的 `loss_weight = 0.0`，注释写着 *"Fake cost. This is just to make it compatible with DETR head."*——**纯粹是为了复用 mmdetection 的 DETR head 接口**。匹配代价里的 `iou_cost` 同理。

- **Q：代码里有几处论文没写的事？**
  A：至少四处，都在上面标了 ⚠️：① FPN 是 **4 层**（多一个 1/8）不是论文说的 3 层；② `num_points=8` 要按 4 个高度锚点除开，每个锚点 **2 个**偏移，不是论文说的 4 个；③ 高度锚点实际是 **`[-4.5, 2.5] m`** 不是 `[-5, 3]`；④ 编码器里 `shift_ref_2d` 那一行，官方注释明确写了 **"we keep this bug for reproducing our results in paper"**。

---

## 7. 最小实现（PyTorch，跟贯穿例子跑一遍）

可变形注意力本身见 `Deformable-Attention.md` §8。这里搭 BEVFormer 的特有部分：**参考点生成 → 投影 → 命中判定 → rebatch → TSA 对齐**。

### 7.1 参考点生成与投影（照抄官方 `get_reference_points` / `point_sampling`）

```python
import math
import numpy as np
import torch
import torch.nn as nn

# ── 贯穿例子：nuScenes base 配置 ──
pc_range    = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
bev_h = bev_w = 200
D             = 4                       # num_points_in_pillar
Z             = pc_range[5] - pc_range[2]      # = 8.0
grid_length   = (102.4 / bev_h, 102.4 / bev_w) # (0.512, 0.512)
img_h, img_w  = 928, 1600               # pad 到 32 的倍数之后

def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1):
    if dim == '3d':        # SCA 用：柱状 3D 参考点
        zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1).permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
        return ref_3d[None].repeat(bs, 1, 1, 1)          # [bs, 4, 40000, 3]
    else:                  # TSA 用：BEV 平面 2D 参考点
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H), torch.linspace(0.5, W - 0.5, W))
        ref_2d = torch.stack((ref_x.reshape(-1) / W, ref_y.reshape(-1) / H), -1)
        return ref_2d.repeat(bs, 1, 1).unsqueeze(2)      # [bs, 40000, 1, 2]

def point_sampling(reference_points, pc_range, lidar2img):
    """reference_points: [1, D, N, 3] 归一化；lidar2img: [N_cam, 4, 4]"""
    ref = reference_points.clone()
    for i in range(3):                                   # 反归一化回真实世界坐标
        ref[..., i] = ref[..., i] * (pc_range[3 + i] - pc_range[i]) + pc_range[i]
    ref = torch.cat((ref, torch.ones_like(ref[..., :1])), -1)      # 齐次坐标
    ref = ref.permute(1, 0, 2, 3)                                  # [D, bs, N, 4]
    D_, B, num_query = ref.size()[:3]
    num_cam = lidar2img.size(0)
    ref = ref.view(D_, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)   # [D,B,cam,N,4,1]
    l2i = lidar2img.view(1, 1, num_cam, 1, 4, 4).repeat(D_, B, 1, num_query, 1, 1)
    ref_cam = torch.matmul(l2i, ref).squeeze(-1)          # [D, B, cam, N, 4]
    eps = 1e-5
    mask = ref_cam[..., 2:3] > eps                        # ① 在相机前方
    ref_cam = ref_cam[..., 0:2] / torch.maximum(ref_cam[..., 2:3], torch.full_like(ref_cam[..., 2:3], eps))
    ref_cam[..., 0] /= img_w                              # ② 归一化到 [0,1]
    ref_cam[..., 1] /= img_h
    mask = mask & (ref_cam[..., 1:2] > 0) & (ref_cam[..., 1:2] < 1) \
                & (ref_cam[..., 0:1] > 0) & (ref_cam[..., 0:1] < 1)
    return ref_cam.permute(2, 1, 3, 0, 4), mask.permute(2, 1, 3, 0, 4).squeeze(-1)   # [cam,bs,N,D,2], [cam,bs,N,D]
```

**跑一遍贯穿例子的关键数字**：

```python
ref_3d = get_reference_points(bev_h, bev_w, Z, D, dim='3d', bs=1)
print(ref_3d.shape)                     # torch.Size([1, 4, 40000, 3])

# 高度锚点：linspace(0.5, 7.5, 4)/8，反归一化后（×8 − 5）
zs = ref_3d[0, :, 0, 2]
print(zs * Z + pc_range[2])             # tensor([-4.5000, -2.1667,  0.1667,  2.5000])  ← 论文说 [-5, 3]

# 第 91 行第 124 列的格子（N = 91*200 + 124 = 18324）对应哪个真实位置？
n = 91 * bev_w + 124
x_norm, y_norm = ref_3d[0, 0, n, 0], ref_3d[0, 0, n, 1]
print((x_norm * 102.4 - 51.2).item(), (y_norm * 102.4 - 51.2).item())   # 12.544 −4.352
```

### 7.2 命中判定 + rebatch（SCA 的核心骨架）

```python
def spatial_cross_attention_min(query, feats_per_cam, ref_3d, lidar2img, pc_range):
    """query: [bs, N_bev, C]；feats_per_cam: list[num_cam]，每个是 [bs, sum(h*w), C]（已 flatten + 加编码）"""
    bs, num_query, C = query.shape
    num_cams = len(feats_per_cam)
    ref_cam, bev_mask = point_sampling(ref_3d, pc_range, lidar2img)   # [cam,bs,N,D,2] / [cam,bs,N,D]

    # ① rebatch：每个相机只留自己命中的格子
    indexes = []
    for i in range(num_cams):
        idx = bev_mask[i][0].sum(-1).nonzero().squeeze(-1)           # 任一高度命中即算命中
        indexes.append(idx)
    per_cam_len = [len(e) for e in indexes]
    print("每相机命中的 BEV 格子数:", per_cam_len, "→ max_len =", max(per_cam_len))  # 几千量级，远小于 40000

    slots  = torch.zeros_like(query)
    counts = torch.zeros(bs, num_query)
    for i in range(num_cams):
        if per_cam_len[i] == 0:
            continue
        q_i = query[:, indexes[i]]                                   # [bs, len_i, C]
        r_i = ref_cam[i][:, indexes[i]]                              # [bs, len_i, D, 2]  ★ 带 D=4 个高度锚点
        # 这里接 Deformable-Attention.md 的 MSDeformAttn：
        #   out_i = MSDeformAttn(256, n_heads=8, n_levels=4, n_points=8)(q_i, feats_per_cam[i], shapes, r_i)
        # 注意 r_i 的最后一维是 D 个 2D 参考点（不是 1 个），必须走 "reference + offsets" 的 Z-anchor 分支
        out_i = placeholder_attn(q_i)                                # 占位，见下
        slots[:, indexes[i]] += out_i
        counts[:, indexes[i]] += 1

    slots = slots / counts.clamp(min=1)[..., None]                   # ★ 就是论文的 1/|V_hit|
    return slots                                                     # [bs, N_bev, C]

def placeholder_attn(q):
    """只为让骨架跑通，不产生有意义的结果；真实实现见 Deformable-Attention.md §8"""
    return q
```

### 7.3 TSA 的 ego-motion 对齐（照抄官方 `get_bev_features`）

```python
def compute_shift_and_align(prev_bev, delta_xy, ego_angle_rad, rotation_angle_deg,
                            grid_length=(0.512, 0.512), bev_h=200, bev_w=200):
    delta_x, delta_y = delta_xy
    translation_length = math.hypot(delta_x, delta_y)
    translation_angle  = math.degrees(math.atan2(delta_y, delta_x))
    bev_angle          = math.degrees(ego_angle_rad) - translation_angle

    shift_y = translation_length * math.cos(math.radians(bev_angle)) / grid_length[0] / bev_h
    shift_x = translation_length * math.sin(math.radians(bev_angle)) / grid_length[1] / bev_w
    shift   = torch.tensor([[shift_x, shift_y]])                     # [1, 2] 归一化偏移

    # 旋转：B 是 [N_bev, C]，先还原成 200×200 的特征图再转
    n, C = prev_bev.shape
    fmap = prev_bev.reshape(bev_h, bev_w, C).permute(2, 0, 1)        # [C, H, W]
    from torchvision.transforms.functional import rotate
    fmap = rotate(fmap, rotation_angle_deg, center=[bev_h // 2, bev_w // 2])
    return fmap.permute(1, 2, 0).reshape(n, C), shift

# 贯穿例子：delta = (0.8, 0.3) m，ego_angle = 12°，rotation = 3°
prev_bev = torch.randn(40000, 256)
aligned, shift = compute_shift_and_align(prev_bev, (0.8, 0.3), math.radians(12), 3.0)
print(shift)        # tensor([[-0.00258,  0.00793]])  → 约 (−0.51 格, +1.59 格)
```

### 7.4 维度自检清单

```python
assert get_reference_points(200, 200, 8, 4, '3d', 1).shape == (1, 4, 40000, 3)
assert get_reference_points(200, 200, 8, 4, '2d', 1).shape == (1, 40000, 1, 2)

# 高度锚点：代码 −4.5/−2.167/+0.167/+2.5（论文写 −5~3）
zs = get_reference_points(200, 200, 8, 4, '3d', 1)[0, :, 0, 2] * 8 - 5
assert torch.allclose(zs, torch.tensor([-4.5, -2.1667, 0.1667, 2.5]), atol=1e-3)

# 每个格子每个相机的采样数 = num_levels × (num_Z_anchors × 每锚点偏移数)
#                          = 4 × (4 × (8 // 4)) = 32
assert 4 * (4 * (8 // 4)) == 32
assert 32 * 8 == 256          # 8 个头 → 每格子每相机 256 个采样点

# BEV 格子 ↔ 真实坐标（贯穿例子的第 91 行第 124 列）
n = 91 * 200 + 124
r = get_reference_points(200, 200, 8, 4, '3d', 1)[0, 0, n]
assert abs((r[0] * 102.4 - 51.2).item() - 12.544) < 1e-3
assert abs((r[1] * 102.4 - 51.2).item() - (-4.352)) < 1e-3

# 图像 token 总数
assert 116*200 + 58*100 + 29*50 + 15*25 == 30825     # 每相机
assert 30825 * 6 == 184950                            # 6 相机
```

---

## 8. 速记卡

- **一句话**：铺一张 `200×200` 的 BEV 网格当 query，用**空间交叉注意力**去 6 路相机取特征、用**时序自注意力**融上一帧 BEV，吐出 `200×200×256` 的统一 BEV 特征图，检测/分割共用；
- **两个核心设计**：
  - **SCA**：每个格子沿 `z` 竖成柱子 → 取 4 个高度锚点 → 用 `lidar2img` 投影 → **只跟命中的相机交互**（`1/|V_hit|` 平均）→ 在参考点周围做可变形采样。**参考点是算出来的（固定），不是预测的（这是和 DETR3D 的本质区别）**；
  - **TSA**：`K/V = [旋转+平移对齐后的上一帧 BEV ; 当前 BEV]`，**offset 由 `Q` 和 `B'_{t-1}` 拼接后预测**，两个分支输出取平均。第一帧无历史 → 退化成 `{Q, Q}`；
- **关键数字**：BEV `200×200`、`0.512 m/格`、`±51.2 m`、`Z ∈ [-5, 3] m`、`embed_dim = 256`、encoder **6 层**、decoder **6 层**、`num_query = 900`、`queue_length = 4`、每相机 token **30825**、BEV query **40000** 个、每个格子每相机采样 **32×8 = 256** 点；
- **省算力的关键**：SCA 的 **rebatch**——每个相机只处理落在自己视野里的那几千个格子。注意力权重 `40000 × 184950 × 8 ≈ 592 亿` vs BEVFormer 约 `512 万`，**差约 1.2 万倍**（省的是**显存**：全局版光注意力矩阵就 `≈29.6 GB/头`，论文只好退化成 `100×100` + 单尺度 + fp16 压到 `~36 GB`；总 FLOPs 其实差不多，因为被 backbone 主导）；
- **时序买到了什么**：`mAVE` **0.845 → 0.378（test）**、`0.802 → 0.394`（1 帧 → 4 帧），NDS `44.8 → 51.7`。**速度估计是时序最大的红利，检出的提升反而小**；
- **成绩单**：nuScenes test V2-99 版 **56.9 NDS / 48.1 mAP**；R101-DCN 版 53.5 / 44.5（val 51.7 / 41.6，训练 28500 MB，24 epoch，`lr=2e-4`）；
- **延迟别赖 BEVFormer**：端到端 1.7 FPS 里 **backbone 占 391 ms，BEVFormer 只占 130 ms**。BEV 部分砍到 7 ms（快 18 倍），FPS 才从 1.7 → 2.3。**瓶颈在 backbone**；
- **四处代码 ≠ 论文**（实践必踩）：① FPN 用 **4 层**（多一个 1/8），论文写 3 层；② `num_points=8` 要按 4 个高度锚点除开 → 每锚点 **2 个**偏移，论文写 4 个；③ 高度锚点实际 **`[-4.5, 2.5] m`**，论文写 `[-5, 3]`；④ 编码器 `shift_ref_2d` 官方注释明写 **"we keep this bug for reproducing our results in paper"**，别照抄；
- **两个避不开的依赖**：必须从 **FCOS3D 预训练** 的 R101-DCN 起步；必须编译**自定义 CUDA 算子**（`MSDeformableAttn`），否则只能用慢速的 `multi_scale_deformable_attn_pytorch` 兜底；
- **演进位置**：Lift-Splat / VPN（显式深度）→ DETR3D（query 查图像）→ **BEVFormer（本篇）** → BEVFormer v2（透视监督 + 现代 backbone）→ BEVDet4D / BEVDepth / PETR / SparseBEV / Fast-BEV → 占用预测（BEVFormer 的 BEV 特征天然支持）。
