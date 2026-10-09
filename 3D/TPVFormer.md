# TPVFormer 流程与要点

- 论文：Huang et al., *Tri-Perspective View for Vision-Based 3D Semantic Occupancy Prediction*, CVPR 2023（arXiv 2302.07817）
- 代码：`github.com/wzzheng/TPVFormer`（基于 BEVFormer 改，Cylinder3D 的数据/损失）
- 任务：nuScenes，6 相机 → 3D 语义占据 / LiDAR 语义分割（mIoU）

---

## 0. 一句话

BEV 把高度压没了，voxel 是 O(H·W·D)。TPV 用**三个互相垂直的平面**表示 3D 场，任意点的特征 = 它在三个平面上的投影点采样后**相加**：

```
f(x,y,z) = t_hw(y,x) + t_zh(y,z) + t_wz(x,z)
```

采样是双线性插值，聚合就是**求和**（论文 Eq.4/5，没有可学权重）。表示复杂度 **O(HW + DH + WD)**，比 voxel 低一个数量级；只有解码成体素那一刻才临时展开 3D。所以三个平面是同一个 3D 场的三视图分解，每个平面只负责对应柱体的"视角专属"信息，正交方向的多样性由另外两个平面补。

三个平面（注意代码里 `h↔y`、`w↔x`；论文里写的 x/y 有反的地方，以代码为准）：

| 代码名 | 平面 | 尺寸（Base，`config/tpv_lidarseg.py`） | 法向（采样方向） |
|---|---|---|---|
| `hw` | XY 俯视 | 200×200 | z |
| `zh` | YZ 侧视 | 16×200 | x |
| `wz` | XZ 前视 | 200×16 | y |

---

## 1. 全流程

以 batch = 1、Base 配置（dim = 128，输入 6×900×1600）为例：

```
img (1, 6, 3, 900, 1600)
  │ ResNet101-DCN (layer2/3/4) + FPN(4 levels, 128ch)
  ▼ (113,200) (57,100) (29,50) (15,25)   ← stride 8/16/32/64，每相机 30125 个 token
  │ 拍平： (6, 30125, 1, 128)            ← (num_cam, num_value, bs, c)

TPV queries（可学习 nn.Embedding + 3D 位置编码）
     hw (1, 40000, 128) / zh (1, 3200, 128) / wz (1, 3200, 128)
  │
  ▼ Encoder: N1 = 3 个 HCAB + N2 = 2 个 HAB
  │   HCAB = CVHA → LN → ICA → LN → FFN → LN   （先灌图像信息）
  │   HAB  = CVHA → LN → FFN → LN              （纯上下文编码）
  ▼ 三个平面 feature（尺寸不变）
  │ TPVAggregator：reshape 回 2D → 沿各自正交轴 broadcast 到同一网格 → 相加
  ▼ (1, 128, 200, 200, 16) → MLP(128→256→128, Softplus) → Linear(→17)
     logits (1, 17, 200, 200, 16)          ← 逐体素语义
     点级：(1, 17, n_points)                ← 对 LiDAR 点双线性采样三个平面求和
```

关键：**TPV queries 是初始化的可学习参数**，不是从图像里抽出来的；图像信息全部靠 ICA 灌进去。每个 query 只加它自己那两个轴上的位置编码（`CustomPositionalEncoding`，48+48+32 = 128），第三个轴补零。

---

## 2. 一个 encoder layer 里的两个注意力

### ICA 图像交叉注意力（把图像信息灌进平面）

1. 每个 query 沿平面的法向均匀采 3D anchor：`hw` 平面 4 个（沿 z）、`zh` / `wz` 各 32 个（沿 x / 沿 y）。
2. `point_sampling` 用 `lidar2img` 把 anchor 投到 6 个相机的像素系，得到 `reference_points_cam (6, bs, N, #p, 2)` 和 `tpv_mask`（打掉图像外 / 相机后面的点）。
3. 只有"至少有一个 anchor 落在该相机里"的 query 才参与该相机 → 按相机 rebatch → deformable attention 在 4 个图像尺度上采样。
4. scatter 回去后**除以有效相机数**（论文 Eq.9）——同一个 query 会被多个相机看到，是平均，不是单相机。

### CVHA 跨视图混合注意力（三个平面互相交互）

- 实现上很巧：**把三个平面直接当作 deformable attention 的 3 个 "level"**。
  `spatial_shapes = [[200,200], [16,200], [200,16]]`，value = 三个平面各自 `value_proj` 后 concat。
- 每个 query 的参考点由 `get_cross_view_ref_points` 预生成，形状 `(N_total, 3, 16, 2)`：即"我的位置"分别投到三个平面上的 16 个 anchor。以 `hw` 平面上的 query (h,w) 为例 → 自己在 hw 上的位置、ZH 上的 `{(h, zᵢ)}₁₆`、WZ 上的 `{(zᵢ, w)}₁₆`（论文 Eq.10-11 说的把参考点分成 top / side / front 三个子集）。
- 每个 anchor 再学 2 个偏移 → 每个平面 32 个采样点。
- 注意力权重里有一路 **level 权重**在 3 个平面上 softmax，初始化时 bias 给自己的平面 `+10`（`init_weight` 里 `attn_bias[:, i, -1] = 10`）——即初始主要看自己，训练中再放开。

---

## 3. 融合与监督

- **体素**：三个平面 reshape 回 `(bs,c,200,200)` / `(bs,c,16,200)` / `(bs,c,200,16)`，沿各自正交轴 broadcast 到同一 (x,y,z) 网格后**直接相加**，再过一个 2 层 MLP + 线性分类器。三平面融合本身**没有可学权重**，唯一可学的"加权"在 CVHA 内部（采样点权重 + level 权重）。
- **点**：给 LiDAR 点用 `grid_sample` 在三个平面上双线性取三个值相加（lidarseg 时点走 lovasz loss、体素走 CE loss）。
- **监督只有稀疏 LiDAR 语义标签**，没有稠密 occupancy GT——这是论文强调的点（28k 帧 / ~300 GPU 小时，对比 Tesla OccNet 的百万帧级）。
- **任意分辨率**：平面就是 2D 特征图，推理时直接插值就能提高分辨率，不用重训（论文把 50×50×4 放大 8 倍，细节还在）。

---

## 4. 面试常问

- **为什么用 sum 不用 concat / 加权？** TPV 的定义就是把同一 3D 场按三视图分解，展开成体素时是逐元素相加，只有 sum 才和这个等价关系一致，也才能保持"存储只有三个平面"。
- **和 BEVFormer 的关系？** 代码就是 BEVFormer 改的：BEV 单平面 → 三平面；SCA（空间交叉注意力）→ ICA（多尺度多相机）；时序 TSA 去掉了（future work 就是加视频上下文）。所以没有时序上下文。
- **block 设计为什么分两种？** HCAB（CVHA + ICA）在前 N1 层负责从图像取视觉信息，HAB（只有 CVHA）在后 N2 层做纯上下文编码。消融显示两者都要。
- **数字 / 配置**：Base = TPV 200×200×16、dim 128、ResNet101-DCN、1600×900 多尺度；Small = 100×100×8、dim 128、ResNet50、800×450 单尺度（靠 ×2 上采样做更细监督）。A100 上约 290 ms/帧。
- **局限（问"有什么问题"答这个）**：三个平面之间只有注意力层面的交互、没有真正的 3D 卷积/3D 交互；稀疏监督信号弱；无时序；精度受平面分辨率限制。后续工作：SurroundOcc（稠密 GT + 体素头）、OccFormer（更强的 occupancy 头）、GaussianFormer（object query）。

---

## 5. 读代码的坑

仓库里有两套实现，别搞混：

- `tpvformer10/`：新的，lidarseg 配置用（200×200×16）。**CVHA 是真正的三平面交互**（上面讲的那套），`get_cross_view_ref_points` 也在这套里。
- `tpvformer04/`：旧的，occupancy 配置用（`config/tpv04_occupancy.py`，100×100×8、dim 256）。它的 CVHA 那一路退化了：`tpvformer04/modules/tpvformer_layer.py` 的 `self_attn` 只把 `query[0]`（hw 平面）传进去，而 `cross_view_hybrid_attention.py` 里 `value = torch.cat([query, query], 0)` 是自复制、ref 也自复制两份，`(out[:bs] + out[bs:]) / 2` 等于没做——实际等价于 hw 平面自己的可形变自注意力。`tpvformer10/modules/encoder.py` 的注释自己也写了 "used in self attention in tpvformer04 which is an older version. Now we use get_cross_view_ref_points instead."。要复现 occupancy 精度的话这里值得自己确认。
