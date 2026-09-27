# CenterPoint 数据流 / 网络结构

依据源码：`tianweiy/CenterPoint` master。

- 一阶段主配置（nuScenes）：`configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py`
- 一阶段对照配置：`configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep.py`
- 二阶段配置：`configs/waymo/voxelnet/two_stage/waymo_centerpoint_voxelnet_two_stage_bev_5point_ft_6epoch_freeze.py`

下面每个数字都标了出处；没标出处的就是由这些配置算出来的。

---

## 1. 端到端 shape 流

两个 backbone 的差异只在 `points → (B, C, H, W)` 这一段，之后完全一样。

### 1.1 VoxelNet 分支（nuScenes 主结果）

```
points                        (N, 5)                # x,y,z,intensity,时间戳差(10 sweep)
  │ Voxelization              voxel_generator: range=[-54,-54,-5,54,54,3]
  │                           voxel_size=[0.075,0.075,0.2]
  ├── voxels                  (M, 10, 5)            # max_points_in_voxel=10
  ├── coords                  (M, 4)                # (batch_idx, z, y, x)
  ├── num_points              (M,)
  └── shape                   (1440, 1440, 40)      # (nx, ny, nz)

VoxelFeatureExtractorV3       (M, 5)                # 体素内点做均值，无参数
  │
SpMiddleResNetFHD             (B, 256, 180, 180)    # 稀疏卷积，最后 dense+view
  │   sparse_shape = shape[::-1] + [1,0,0] = (41, 1440, 1440)
  │   conv_input 5→16  submanifold
  │   conv1     16→16   ×2 SparseBasicBlock
  │   conv2     16→32   stride2   z 41→21, xy 1440→720
  │   conv3     32→64   stride2   z 21→11, xy 720→360
  │   conv4     64→128  stride2   z 11→5,  xy 360→180   pad=[0,1,1]
  │   extra_conv 128→128 k=(3,1,1) s=(2,1,1)  z 5→2, xy 180
  │   dense → (B,128,2,180,180) → view(B, 256, 180, 180)
  │
RPN (neck)                    (B, 512, 180, 180)
  │   block0 256→128 s1, 5 conv → (B,128,180,180)
  │   block1 128→256 s2, 5 conv → (B,256, 90, 90)
  │   deblock0 us=1 → Conv2d k1 s1   → (B,256,180,180)
  │   deblock1 us=2 → ConvTranspose2d k2 s2 → (B,256,180,180)
  │   cat(dim=1)                (B, 512, 180, 180)
  │
CenterHead                    shared_conv 512→64 → (B, 64, 180, 180)
  │
  ├─ task0 (car)              hm (B,1,180,180)  + reg/height/dim/rot/vel 共 (B,10,180,180)
  ├─ task1 (truck, constr.)   hm (B,2,180,180)  + (B,10,180,180)
  ├─ task2 (bus, trailer)     hm (B,2,180,180)  + (B,10,180,180)
  ├─ task3 (barrier)          hm (B,1,180,180)  + (B,10,180,180)
  ├─ task4 (motor, bicycle)   hm (B,2,180,180)  + (B,10,180,180)
  └─ task5 (ped, cone)        hm (B,2,180,180)  + (B,10,180,180)
```

BEV 分辨率核对：`out_size_factor = 8`（见 3.1），BEV 格子 = `0.075 × 8 = 0.6 m`，
`180 × 0.6 = 108 m = 54 × 2`，与 `voxel_generator.range` 的 ±54 对上。

### 1.2 PointPillars 分支

```
points                        (N, 5)
  │ Voxelization              range=[-51.2,-51.2,-5,51.2,51.2,3]
  │                           voxel_size=[0.2,0.2,8]  # z 方向一整柱
  ├── voxels                  (M, 20, 5)            # max_points_in_voxel=20
  ├── coords                  (M, 4)
  └── shape                   (512, 512, 1)

PillarFeatureNet              (M, 64)
  │   f_cluster = xyz - 柱内均值                      (M,20,3)
  │   f_center  = xy - 柱中心                         (M,20,2)
  │   cat [f(5), f_cluster(3), f_center(2)] → (M,20,10)
  │   PFNLayer0: Linear 10→32, BN1d, ReLU, max over 20
  │      → 再 repeat 后 cat 回自己 → (M,20,64)
  │   PFNLayer1: Linear 64→64, BN1d, ReLU, max over 20 → (M,1,64) → squeeze → (M,64)
  │
PointPillarsScatter           (B, 64, 512, 512)
  │   indices = coords[:,2]*nx + coords[:,3]   # = y*nx + x
  │   canvas (64, 512*512) ← scatter → view(B, 64, ny=512, nx=512)
  │
RPN (neck)                    (B, 384, 128, 128)
  │   block0 64→64  s2, 3 conv → (B, 64,256,256)
  │   block1 64→128 s2, 5 conv → (B,128,128,128)
  │   block2 128→256 s2, 5 conv → (B,256, 64, 64)
  │   deblock0 us=0.5 → stride=round(1/0.5)=2 → Conv2d k2 s2      → (B,128,128,128)
  │   deblock1 us=1   → Conv2d k1 s1                              → (B,128,128,128)
  │   deblock2 us=2   → ConvTranspose2d k2 s2                     → (B,128,128,128)
  │   cat(dim=1)                (B, 384, 128, 128)
  │
CenterHead                    shared_conv 384→64 → (B, 64, 128, 128) → 同 1.1 的 6 个 task
```

核对：`out_size_factor = 4`，BEV 格子 = `0.2 × 4 = 0.8 m`，`128 × 0.8 = 102.4 = 51.2 × 2`。

### 1.3 训练时的并行数据流

推理只有一条线，训练时多一条标签线（`AssignLabel` pipeline，在 dataloader 里）：

```
gt_boxes (num_gt, 9) = [x, y, z, w, l, h, vx, vy, rot]
  │ AssignLabel (det3d/datasets/pipelines/preprocess.py)
  │ 每个 task 独立生成一份，共 6 组：
  ├── hm      (num_cls_in_task, 180, 180)     float32   # ny, nx
  ├── anno_box(max_objs=500, 10)              float32   # 见 5.2
  ├── ind     (500,)                          int64     # y*W + x
  ├── mask    (500,)                          uint8
  ├── cat     (500,)                          int64     # task 内类别 id，0-based
  └── gt_boxes_and_cls (500, 10)              float32   # 只给二阶段用
```

`hm` 的 shape 是 `(num_cls, ny, nx)` —— 注意 H 是 ny，W 是 nx。

---

## 2. Reader：关键实现细节

**VoxelFeatureExtractorV3**（`det3d/models/readers/voxel_encoder.py`）只有 3 行有效代码：

```python
points_mean = features[:, :, :self.num_input_features].sum(dim=1, keepdim=False) \
              / num_voxels.type_as(features).view(-1, 1)
return points_mean.contiguous()
```

`(M, 10, 5) → (M, 5)`。**没有 padding 屏蔽**——空槽位必须靠体素化保证全 0，否则均值会被污染。

**PillarFeatureNet** 里有个反直觉的点：`PFNLayer` 如果不是最后一层，输出通道会被**砍半**：

```python
if not self.last_vfe:
    out_channels = out_channels // 2
```

所以 `num_filters=[64, 64]` 实际是 `Linear(10→32)` 再 cat 回自己凑成 64 喂给下一层。
每层都是 `BN1d`，并且 `forward` 里先 `permute(0,2,1)` 过 BN 再 permute 回来。

---

## 3. Neck (RPN) 的输出 stride

### 3.1 out_size_factor 的算法

`det3d/utils/config_tool.py::get_downsample_factor`：

```python
downsample_factor  = np.prod(neck_cfg.get("ds_layer_strides", [1]))
downsample_factor /= neck_cfg.get("us_layer_strides", [])[-1]
downsample_factor *= backbone_cfg["ds_factor"]
```

代进两个配置：

| 配置 | `prod(ds_layer_strides)` | `us_layer_strides[-1]` | `backbone.ds_factor` | 结果 |
|---|---|---|---|---|
| VoxelNet | `1*2 = 2` | `2` | `8`（`SpMiddleResNetFHD`） | **8** |
| PointPillars | `2*2*2 = 8` | `2` | `1`（`PointPillarsScatter`） | **4** |

这个值同时用于：`AssignLabel` 里算 `feature_map_size`、`test_cfg` 里解码反量化、`gaussian_radius` 里把米换成格子数。**改 backbone 的 `ds_factor` 而忘了它会连带改变热图分辨率，是最容易错的地方。**

### 3.2 deblock 的两种形态

`RPN.forward` 里对每个 layer，若 `stride > 1` 用 `ConvTranspose2d`，否则用 `Conv2d`：

```python
if stride > 1:
    ConvTranspose2d(..., stride, stride=stride)
else:
    stride = np.round(1 / stride).astype(np.int64)
    Conv2d(..., stride, stride=stride)
```

所以 PointPillars 的 `us_layer_strides=[0.5, 1, 2]` 中，`0.5` 走的是 `Conv2d(k=2, s=2)` 这条下采样分支，不是上采样。
`downsample_factor` 那个 `property` 在 RPN 类里算出来是相对值（VoxelNet 是 1，PP 是 4），**它只对 PP 凑巧等于最终值**，别拿它当 `out_size_factor`。

---

## 4. CenterHead

`det3d/models/bbox_heads/center_head.py`。

### 4.1 先分 task，再分 head

`tasks` 把 10 个类别切成 **6 组**，每组一个 `SepHead`（不是 10 个类 10 个 head）：

```python
tasks = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]
```

理由在 `common_heads` 的通道数上：**回归是按 task 共享的**（`common_heads` 是固定通道数，只有 `hm` 拿到 `num_cls`）。
也就是说 `truck` 和 `construction_vehicle` 共用同一个 `dim` head，但 `bus`/`trailer` 用的是另一套参数。

### 4.2 通道分配

`common_heads`（config 里写的是 `(output_channel, num_conv)`）：

| key | 通道 | num_conv | 目标维度 |
|---|---|---|---|
| `reg` | 2 | 2 | `ct - ct_int`，**特征图格子单位的亚格偏移**，不是米 |
| `height` | 1 | 2 | `z`，**绝对米** |
| `dim` | 3 | 2 | `log(w), log(l), log(h)` |
| `rot` | 2 | 2 | `sin(rot), cos(rot)` |
| `vel` | 2 | 2 | `vx, vy`（`dataset='nuscenes'` 才有） |
| `hm` | `num_cls` | 2 | 热图 |

`SepHead` 的每个子头结构是：`num_conv-1` 个 `Conv2d(64→64, k=3) + BN + ReLU`（`bn=True`）再接一个
`Conv2d(64→classes, k=3)`（`final_kernel=3`）。

`hm` 头最后一层 bias 初始化为 `init_bias=-2.19`，其余走 kaiming。
`-2.19 ≈ -log((1-0.1)/0.1)`，即让初始热图背景概率约 0.1。

### 4.3 forward 的返回值

```python
def forward(self, x, *kwargs):
    x = self.shared_conv(x)            # (B, 64, H, W)
    ret_dicts = [task(x) for task in self.tasks]
    return ret_dicts, x                # 第二个返回值给二阶段用
```

返回的是 **list of dict**，每个 task 一个 dict，key 就是上面表里的 `hm/reg/height/dim/rot/vel`。

`dcn_head=True` 时换成 `DCNSepHead`：用 DCNv1 (`FeatureAdaption`) 分别给 cls 和 reg 生成 offset，
`conv_offset.weight` 零初始化。nuScenes 主配置是 `dcn_head=False`。

---

## 5. 训练目标（AssignLabel）

`det3d/datasets/pipelines/preprocess.py`。

### 5.1 正样本位置

每个 GT 落在一个格子上：

```python
coor_x = (x - pc_range[0]) / voxel_size[0] / out_size_factor
coor_y = (y - pc_range[1]) / voxel_size[1] / out_size_factor
ct      = np.array([coor_x, coor_y])
ct_int  = ct.astype(np.int32)
ind     = ct_int[1] * feature_map_size[0] + ct_int[0]     # = y*W + x
```

`ind` 是展平的 `y*W+x`；越过 `feature_map_size` 的 GT 直接 `continue` 丢掉。
`ct_int` 是 `astype(int32)` 截断，不是四舍五入——`reg` 的 offset 目标就是补偿这个截断。

### 5.2 anno_box 编码（10 维）

```python
anno_box[new_idx] = np.concatenate(
    (ct - ct_int,                        # 2  dx, dy  (格子单位, 0~1)
     z,                                  # 1  z 绝对米
     np.log(gt_boxes[k][3:6]),           # 3  log(w), log(l), log(h)
     vx, vy,                             # 2
     np.sin(rot), np.cos(rot)),          # 2  用 sin/cos 表达角度，绕开 ±π 回绕
    axis=None)
```

对应到 head 通道的顺序是 `cat([reg, height, dim, vel, rot], dim=1)`（`CenterHead.loss` 里就是这么拼的），
所以通道和 anno_box 列是**一一对应**的。

`dim` 走 log 空间，反解在 `predict` 里：`batch_dim = torch.exp(preds_dict['dim'])`。
`z` 是**绝对米**，解码时不加 offset 也不乘 voxel_size。

### 5.3 热图半径

```python
w, l = w / voxel_size[0] / out_size_factor, l / voxel_size[1] / out_size_factor
radius = gaussian_radius((l, w), min_overlap=self.gaussian_overlap)
radius = max(self._min_radius, int(radius))
```

注意两点：

1. `w, l` 先被换算成**特征图格子数**才送进 `gaussian_radius`，所以半径和网络输出分辨率绑死。
2. 传参顺序是 `(l, w)`——`gaussian_radius` 形参名是 `(height, width)`，这里 height 对应 l。
3. `min_overlap=0.1`（config 的 `gaussian_overlap`），不是 CenterNet 常见的 0.7。
   `_min_radius=2` 是兜底下限。

`draw_umich_gaussian(hm[cls_id], ct, radius)`：`sigma = diameter/6`，用 `np.maximum` 写入（重叠处取大）。

---

## 6. Loss

```
loss = hm_loss + weight * loc_loss
```

- `weight = 0.25`（nuScenes）/ `2.0`（Waymo）
- `hm_loss`：`FastFocalLoss`，CornerNet 的 penalty-reduced focal，`gt = (1-target)^4`，
  正样本按 `cat` 只取对应那一类通道。负数部分对**所有**格子求和，正样本部分只对 `ind` 位置求和后除以正样本数。
- `loc_loss`：`RegLoss` = L1，先用 `_transpose_and_gather_feat` 把 `(B,C,H,W)` 在 `ind` 处 gather 成 `(B, M, C)`,
  再乘 `code_weights` 加权、除以正样本数。

`code_weights` 是**逐通道**权重，长度必须等于通道数：

| 配置 | code_weights | 通道 |
|---|---|---|
| nuScenes | `[1,1,1,1,1,1, 0.2,0.2, 1,1]` | 10 = x,y,z,logdim×3,vx,vy,sin,cos（速度降权 0.2） |
| Waymo | `[1]*8` | 8 = 去掉 vel |

`RegLoss` 的归一化方式值得注意：

```python
loss = F.l1_loss(pred*mask, target*mask, reduction='none')   # (B, M, C)
loss = loss / (mask.sum() + 1e-4)                            # 分母是 batch 内全局正样本数
loss = loss.transpose(2, 0).sum(dim=2).sum(dim=1)            # → (C,)
```

`mask` 先把无效槽位乘成 0，分母是**整个 batch 的正样本总数**，
所以逐通道 loss 是「所有正样本在该通道上的平均 L1」，而不是逐样本平均。

---

## 7. 推理解码

### 7.1 反量化

`CenterHead.predict` 先把 `N C H W` permute 成 `N H W C`，然后：

```python
batch_hm   = torch.sigmoid(preds_dict['hm'])
batch_dim  = torch.exp(preds_dict['dim'])
batch_rot  = torch.atan2(preds_dict['rot'][..., 0:1], preds_dict['rot'][..., 1:2])
#                                 ↑ sin                        ↑ cos
```

通道顺序是 `[sin, cos]`（对应 5.2 的 `np.sin(rot), np.cos(rot)`），`atan2(sin, cos)`。

```python
ys, xs = meshgrid(arange(H), arange(W))
xs = (xs + batch_reg[..., 0:1]) * out_size_factor * voxel_size[0] + pc_range[0]
ys = (ys + batch_reg[..., 1:2]) * out_size_factor * voxel_size[1] + pc_range[1]
```

- `reg[0]` 配 x，`reg[1]` 配 y（`meshgrid` 的第一个输出是沿 W 变的索引 = x）
- **z 直接用 `height` 头的输出，不反量化**（nuScenes 的 z 范围只有 8 m）
- 最终 box 拼成 `[xs, ys, height, dim(3), vel(2), rot]` = 9 维，
  即 `(x, y, z, w, l, h, vx, vy, rot)` —— 提交格式要求 rot 在最后

### 7.2 过滤 + NMS

```python
scores, labels = torch.max(hm_preds, dim=-1)       # 每格取最大类，class-agnostic 取峰
mask = (scores > test_cfg.score_threshold) & 中心点在 post_center_limit_range 内
```

- `score_threshold=0.1`，`post_center_limit_range=[-61.2,-61.2,-10,61.2,61.2,10]`（比检测范围略大）
- 分数用 `max over class` 而不是逐类分别取峰，所以天然是 class-agnostic NMS（`use_multi_class_nms` 在实现里没被读）
- 两条 NMS 分支，由 `test_cfg.circular_nms` 决定：

| | rotate NMS | circle NMS |
|---|---|---|
| 条件 | 默认 | `circular_nms=True` |
| 参数 | `nms_pre_max_size=1000`, `nms_post_max_size=83`, `nms_iou_threshold=0.2` | `test_cfg.min_radius`（**list，按 task 索引**） |
| 半径 | — | `[4, 12, 10, 1, 0.85, 0.175]`，对应 6 个 task |

`nms_post_max_size=83` 是 nuScenes 官方评测每帧的框数上限。
circle NMS 的 `min_radius` 单位是**米**（BEV 中心距），和 5.3 里 assigner 的 `min_radius=2`（像素）不是一回事——
它俩名字一样但一个是 `train_cfg.assigner.min_radius`，一个是 `test_cfg.min_radius`。

`labels` 是 task 内 id，最后按 `flag += num_class` 累加还原成 0~9 的全局类别。

---

## 8. 二阶段（refine head）

`det3d/models/detectors/two_stage.py` + `det3d/models/second_stage/bird_eye_view.py`。

### 8.1 它不是 RoIAlign

这是最容易误解的地方。二阶段**没有**在 BEV 图上抠 patch 做 RoIAlign，而是**在若干个离散关键点上做双线性插值**：

```python
def absl_to_relative(self, absolute):
    a1 = (absolute[..., 0] - self.pc_start[0]) / self.voxel_size[0] / self.out_stride
    a2 = (absolute[..., 1] - self.pc_start[1]) / self.voxel_size[1] / self.out_stride
    return a1, a2

feature_map = bilinear_interpolate_torch(example['bev_feature'][batch_idx], xs, ys)  # (num_pt, C)
```

`bev_feature` 是 `N H W C`（`two_stage.py` 里 permute 过），插值输出每个点一个 C 维向量。
Waymo 配置的 `voxel_size=[0.1,0.1] × out_stride=8 = 0.8 m`，正好是 stage-1 的 BEV 格子大小。

### 8.2 取几个点

`num_point` 控制（`TwoStageDetector.get_box_center`）：

- `num_point=1`：只取框中心 `(N, 3)`
- `num_point=5`：中心 + 前/后/左/右四个边中点，用 `center_to_corner_box2d` 算角点再取边中点，
  `cat([center, front, back, left, right], dim=0)` → `(5N, 3)`

插值后按 `num_point` 分段拼接：

```python
section_size = len(feature_map) // num_point          # = N
feature_map = torch.cat([feature_map[i*N:(i+1)*N] for i in range(num_point)], dim=1)
```

→ `(N, 5 × 512 = 2560)`。这就解释了配置里 `input_channels=512*5`：
**5 是被当成通道数拼进去的，不是空间维度。**

### 8.3 特征来源开关

`TwoStageDetector` 的 `use_final_feature` 决定用哪张图：

| 值 | 用谁 | 通道 | 出处 |
|---|---|---|---|
| `False`（Waymo 配置，默认） | `bev_feature` = neck 输出 | 512 | `VoxelNet.forward_two_stage` 的第一个返回 |
| `True` | `final_feature` = `CenterHead.shared_conv` 输出 | 64 | `CenterHead.forward` 的第二个返回 |

### 8.4 RoI head 结构

`det3d/models/roi_heads/roi_head.py`。输入先被拉平成 `(B*N, C, 1)` 再用 **Conv1d 当 FC 用**：

```
roi_features (B, 500, 2560)      # 500 = NMS_POST_MAXSIZE，不足的补 0
  reshape → (B*500, 1, 2560) → permute → (B*500, 2560, 1)
  shared_fc_layer:
    Conv1d(2560→256) + BN1d + ReLU + Dropout(0.3)
    Conv1d(256→256)  + BN1d + ReLU
  ├─ cls_layers (CLS_FC=[256,256]): Conv1d(256→256)+BN+ReLU+Drop + Conv1d(256→256)+BN+ReLU + Conv1d(256→1)
  │    → (B*500, 1)
  └─ reg_layers (REG_FC=[256,256]): 同上，最后一层 Conv1d(256→7)
       → (B*500, 7)
```

`num_class=1`（`RoIHead.__init__` 默认值）→ **class-agnostic 二分类**，`cls` 只分前景/背景，
`reg` 出的 7 维被所有类共用。最后一层 `reg_layers[-1]` 权重用 `normal(std=0.001)` 单独初始化。

`Dropout` 的插入位置在 `make_fc_layers` 里是 `k == 0` 之后（只在第一层后加一次），
`shared_fc_layer` 里是「非最后一层」加。

### 8.5 标签：cls 回归的是 IoU，不是 0/1

`det3d/models/roi_heads/target_assigner/proposal_target_layer.py`。Waymo 配置 `CLS_SCORE_TYPE='roi_iou'`：

```python
fg_mask = iou > CLS_FG_THRESH      # 0.75
bg_mask = iou < CLS_BG_THRESH      # 0.25
batch_cls_labels = fg_mask.float()                                        # 前景 → 1.0
batch_cls_labels[interval_mask] = (iou - 0.25) / (0.75 - 0.25)            # 中间线性过渡
```

所以 `cls` 头学的是**预测框和 GT 的 3D IoU**（IoU-aware），中间区域变成软标签。
`interval_mask` 就是 `0.25 <= iou <= 0.75` 那部分，**不是 ignore**（对比 `CLS_SCORE_TYPE='cls'` 那条分支，那里会置 -1）。

采样（`subsample_rois`）：

| 参数 | 值 | 含义 |
|---|---|---|
| `ROI_PER_IMAGE` | 128 | 每帧采样 128 个 RoI |
| `FG_RATIO` | 0.5 | 64 前景 / 64 背景 |
| `REG_FG_THRESH` | 0.55 | IoU > 0.55 才算前景，且只有它参与回归损失 |
| `CLS_BG_THRESH_LO` | 0.1 | 低于此算 easy bg |
| `HARD_BG_RATIO` | 0.8 | 背景里 80% 取 hard bg（0.1~0.55） |

框回归目标是 **RoI 局部坐标系下的残差**（`roi_head_template.py::assign_targets`）：

```python
gt_of_rois[:, :, :6] -= rois[:, :, :6]          # 平移残差
gt_of_rois[:, :, 6]  -= roi_ry                   # 角度残差
gt_of_rois = rotate_points_along_z(gt_of_rois, angle=-roi_ry)   # 转到 RoI 朝向的局部系
```

角度残差还会做一次**翻转对齐**：把目标绕到 `(-π/2, π/2]`，这样画框框的长边朝向就不会左右颠倒。

损失：`cls` 用 BCE（`rcnn_cls_weight=1.0`），`reg` 用 L1（`rcnn_reg_weight=1.0`，`code_weights` 7 维全 1），
只对 `reg_valid_mask = iou > 0.55` 的位置回传。

### 8.6 输出融合

```python
scores = torch.sqrt(torch.sigmoid(cls_preds) * roi_scores)   # 几何平均
box_preds = (reg_residual + local_rois) 再 rotate 回全局系 + roi_xyz
```

一阶段分数和二阶段的 IoU 预测取几何平均。二阶段输出**不再做 NMS**（源码注释 `# currently don't need nms`），
直接丢掉 `label_preds == 0` 的背景框。

Waymo 配置 `freeze=True`：一阶段 backbone 冻结，只训二阶段，6 epoch，`lr_max=0.003`。

---

## 9. shape 速查表

以 nuScenes 为例。`B`=batch，`M`=非空体素/柱体数。

| 位置 | VoxelNet | PointPillars |
|---|---|---|
| voxel 网格 `(nx,ny,nz)` | `(1440,1440,40)` | `(512,512,1)` |
| voxel/柱 输入 | `voxels (M,10,5)` | `voxels (M,20,5)` |
| reader 输出 | `(M,5)` | `(M,64)` |
| backbone 输出 | `(B,256,180,180)` | `(B,64,512,512)` |
| neck 输出 | `(B,512,180,180)` | `(B,384,128,128)` |
| `out_size_factor` | `8` | `4` |
| BEV 格子大小 | `0.6 m` | `0.8 m` |
| BEV 覆盖范围 | `108 m` | `102.4 m` |
| `feature_map_size` | `(180,180)` | `(128,128)` |
| head 输入通道 | `512` | `384` |
| `shared_conv` 输出 | `(B,64,180,180)` | `(B,64,128,128)` |
| `hm`（每 task） | `(B,num_cls,180,180)` | `(B,num_cls,128,128)` |
| `reg/height/dim/rot/vel` | `(B,2/1/3/2/2,180,180)` | 同左 |
| 训练 `hm` 标签 | `(num_cls,180,180)` | `(num_cls,128,128)` |
| 训练 `anno_box` | `(500,10)` | `(500,10)` |
| 解码后 box | `(num_box,9)` | `(num_box,9)` |

head 输出通道汇总（nuScenes，6 个 task）：

| task | classes | hm 通道 | 回归通道 |
|---|---|---|---|
| 0 | car | 1 | 10 |
| 1 | truck, construction_vehicle | 2 | 10 |
| 2 | bus, trailer | 2 | 10 |
| 3 | barrier | 1 | 10 |
| 4 | motorcycle, bicycle | 2 | 10 |
| 5 | pedestrian, traffic_cone | 2 | 10 |
| 合计 | 10 | 10 | 60 |

二阶段（Waymo 配置）：

| 位置 | shape |
|---|---|
| `bev_feature` | `(B, 188, 188, 512)` |
| 关键点坐标 | `(5N, 3)` |
| 插值特征 | `(5N, 512)` → 分段 cat → `(N, 2560)` |
| `roi_features` | `(B, 500, 2560)` |
| `shared_fc_layer` 输出 | `(B*500, 256, 1)` |
| `rcnn_cls` | `(B*500, 1)` |
| `rcnn_reg` | `(B*500, 7)` |

---

## 10. 容易踩的点

1. **`out_size_factor` 由 `neck` 和 `backbone.ds_factor` 共同决定**，不是单一常量。
   `configs/nusc/pp/...` 里写 `out_size_factor=get_downsample_factor(model)`，改 `backbone.ds_factor` 会同时改变热图分辨率和解码反量化系数，两处必须一致。

2. **`height` 头回归的是绝对 z，`dim` 回归的是 log 后的尺寸。**
   只有 `reg`(x,y 的亚格偏移) 是相对的。解码时 `exp()` 只作用在 `dim` 上。

3. **`reg` 偏移单位是特征图格子（0~1），不是米。**
   反量化是 `(idx + reg) * out_size_factor * voxel_size + pc_range`，先加偏移再放大。

4. **`rot` 通道顺序是 `[sin, cos]`**，解码用 `atan2(sin, cos)`。
   写反了不会报错，只会让所有框的朝向错乱。

5. **回归头是按 task 共享的，不是按类。** 10 个类只出 60 个回归通道而不是 80 个。
   做多任务时 `hm` 通道数按 task 变，回归通道数恒定。

6. **`gaussian_radius` 的入参顺序是 `(l, w)`**，形参名却是 `(height, width)`。
   把 `w, l` 传反不会崩，但细长物体（bus/trailer）的热图半径会明显偏大或偏小。

7. **`gaussian_overlap=0.1`** 而不是 CenterNet 常用的 0.7。改大让正样本区域变大。

8. **`anno_box` 里的 `z` 是绝对米**，但同一个 10 维向量里 `dim` 是 log 空间。
   两个量在同一个 L1 loss 里，量纲不同，靠 `code_weights` 平衡——`code_weights` 长度必须精确等于通道数（nuScenes 10、Waymo 8），少了会广播错位。

9. **`min_radius` 有两个同名但完全不同的东西**：
   `train_cfg.assigner.min_radius`（像素，热图半径下限）vs `test_cfg.min_radius`（米，circle NMS 半径，且是 **list，按 task 索引**）。

10. **二阶段不是 RoIAlign**，是 `num_point` 个点上的双线性插值。
    `num_point=5` 时那 5 个点是拼在**通道维**上的，所以 `input_channels = 512*5`。
    点数或特征通道数一改，`input_channels` 必须跟着改，否则第一层 Conv1d 维度不匹配。

11. **`PointPillarsScatter` 里 `nx`/`ny` 的取法**：`nx = input_shape[0]`、`ny = input_shape[1]`，
    而 scatter 索引是 `coords[:,2]*nx + coords[:,3]`（y*nx + x），最后 `view(B, C, ny, nx)`。
    `input_shape` 的约定是 `(nx, ny, nz)`，别按 `(W, H, D)` 理解。

12. **`VoxelFeatureExtractorV3` 不做 padding 屏蔽**，直接对 `max_points_in_voxel` 全部槽位求均值。
    上游体素化必须把空槽位填 0，否则体素特征会被污染。

13. `test_cfg` 里有几个键**从头到尾没被读过**：`nms.use_rotate_nms`、`nms.use_multi_class_nms`、`max_per_img`。
    head 实际读取的只有 `double_flip`、`post_center_limit_range`、`out_size_factor`、`voxel_size`、`pc_range`、
    `per_class_nms`、`score_threshold`、`circular_nms`、`min_radius`、`nms.nms_*`。
    所以改 `use_rotate_nms=False` 不会有任何效果——真正的分支开关是 `circular_nms`。

14. 训练和推理走的是**不同代码路径**：训练用 dataloader 里的 `AssignLabel` 生成标签，
    推理用 `CenterHead.predict` 解码。两者的 `out_size_factor`、`pc_range`、`voxel_size` 都来自配置，
    任何一处改了 `ds_factor` 都要同时检查 `train_cfg` 和 `test_cfg` 里的 `get_downsample_factor(model)` 是否被重新求值。
