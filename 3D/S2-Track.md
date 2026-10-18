# S2-Track：面向端到端 3D 多目标跟踪的简单而强大框架

> **一句话**：S2-Track 将端到端 3D 多目标跟踪拆成 **query 初始化、query 传播、query 匹配** 三个环节，并分别加入 2D-Prompted Query Initialization（PQI）、Uncertainty-aware Probabilistic Decoder（UPD）和 Hierarchical Query Denoising（HQD），重点解决遮挡、小目标、远距离目标和训练匹配不稳定问题。
>
> **论文**：Tao Tang et al., *S2-Track: A Simple yet Strong Approach for End-to-End 3D Multi-Object Tracking*, ICML 2025（arXiv:2406.02147）。
>
> **任务设定**：输入自动驾驶车辆周围多摄像头在时刻 `t` 的图像，输出当前场景中的 3D 边界框，并在连续帧之间保持目标 ID 一致。

---

## 1. S2-Track 要解决什么问题

传统 3D MOT 通常采用 tracking-by-detection：先用检测器得到每帧 3D 框，再使用卡尔曼滤波、3D IoU、L2 距离或 ReID 特征做轨迹预测和数据关联。这种流程依赖较多后处理规则，检测和关联也不是一个统一的优化目标。

近年来的 query-based tracker 将目标表示为 Transformer query：上一帧的 query 传播到当前帧，decoder 同时更新旧目标并发现新目标。它避免了显式的帧间匹配，但朴素的 query 方案仍有三个薄弱点：

- **初始化弱**：随机 query 不一定落在目标附近，遮挡或小目标可能一开始就没有可靠 query；
- **传播不确定**：不同目标大小、运动模式和可见性差异很大，确定性的 attention 难以表达预测噪声；
- **训练匹配不稳**：query 与 GT 的二分图匹配会使训练收敛慢，简单的 denoising 也没有区分不同噪声样本的难度。

S2-Track 的思路不是重新设计一个复杂的 tracking pipeline，而是逐一增强这三个环节。

---

## 2. 从三个 query 环节理解整篇论文

```text
多摄像头图像 I_t
      │
      ▼
Backbone + FPN
      │                    ┌─ 2D detector ──► 2D boxes
      │                    └─ Depth network ─► depth
      ▼                              │
图像特征 F_t                 2D-Prompted Query Initialization
      │                              │
      ├───────────────┬──────────────┘
      │               │
上一帧 track queries   PQI queries + random init queries
      └───────────────┴──────────────► 当前 query 集合 Q_t
                                      │
                                      ▼
                         Uncertainty-aware Probabilistic Decoder
                                      │
                         当前 3D boxes B_t + 更新后的 queries Q_t
                                      │
                                      ▼
                            保持 ID，传播到下一帧

训练阶段：GT boxes → 加噪 → Hierarchical Query Denoising
```

每条 query 表示一个潜在的 3D 目标：

```text
q_t^i = { f_t^i, c_t^i }

f_t^i：query 的特征向量
c_t^i：query 对应的 3D 位置
```

当前帧的 query 由两部分组成：

1. 从上一帧传播来的 query，负责延续已有轨迹；
2. PQI 产生的 query 和随机初始化 query，负责定位或发现新目标。

基础递推可以写成：

```text
Q_t = Prop(Q_{t-1}, Q_init)
```

S2-Track 改为：

```text
Q_t = Prop(Q_{t-1}, Q_PQI-init, Q_init)
```

其中 `Q_init` 仍然保留，用于探索 2D 检测或深度分支漏掉的目标。

---

## 3. 2D-Prompted Query Initialization（PQI）

### 3.1 为什么要用 2D 和深度提示

只用随机可学习 embedding 初始化 query 时，query 的初始位置没有明确的几何意义。对于小目标、被遮挡目标和远处目标，decoder 需要从很大的搜索空间中找到对应的 3D 位置，容易漏检或丢轨。

PQI 复用图像 backbone 和 FPN 的特征，增加两个辅助任务：

- 2D 检测：预测每个摄像头中的 2D box；论文采用 YOLOX 风格的 2D detection head；
- 深度预测：预测图像位置的深度，深度网络由多个 residual block 构成，并用投影后的 LiDAR 点监督。

```text
F_t ──► 2D detection head ──► B_t^2d
  │
  └──► depth network ────────► D_t
```

辅助任务损失为：

```text
L_PQI = L_Det^2D + L_Depth
```

### 3.2 从 2D 中心和深度恢复 3D 位置

对 2D box 中心 `(u, v)`，使用相机内参和相机到 LiDAR 坐标系的变换，将深度预测转换为初始 3D 中心：

```text
c_t = T^{lidar}_{cam} K^{-1} D_t[u, v, 1]^T
```

其中：

- `K` 是摄像头内参矩阵；
- `K^{-1}[u, v, 1]^T` 给出相机坐标系中的射线方向；
- `D_t` 提供该位置的深度；
- `T^{lidar}_{cam}`（从 camera 到 lidar 的变换）将坐标转换到跟踪使用的 3D 坐标系。实现时应以代码中变换矩阵的定义为准，避免把变换方向写反。

由此得到带有初始几何位置的 `Q_PQI-init`。它不直接替代随机 query，而是和随机 query 一起输入 decoder：

```text
Q_t ← Prop(Q_{t-1}, Q_PQI-init, Q_init)
```

### 3.3 PQI 的作用

- 2D 检测提供目标在图像平面中的候选位置；
- 深度网络把 2D 候选提升到粗略 3D 位置；
- decoder 从更合理的空间先验开始细化 3D box；
- 随机 query 保留对漏检目标的探索能力。

一个容易混淆的点是：PQI 产生的是**初始 query/位置提示**，不是最终 3D 检测结果。最终的尺寸、朝向、速度和类别仍由 3D decoder 预测。

---

## 4. Uncertainty-aware Probabilistic Decoder（UPD）

### 4.1 确定性 attention 的问题

普通 Transformer attention 使用确定的 query-key 相似度：

```text
α_ij = q_i · k_j / √d
```

对于复杂驾驶场景，这个分数没有直接表达“不确定程度”：遮挡、远距离小目标和目标间交叠都可能让同一个 query 对多个位置产生模糊响应。

### 4.2 用高斯分布表示 attention

UPD 让每个 query-key 对应的 attention score 服从高斯分布：

```text
α_ij ~ N(μ_ij, σ_ij²)
```

`μ_ij` 和 `σ_ij²` 由一个 MLP 根据 `q_i` 与 `k_j` 预测：

```text
(μ_ij, σ_ij²) = MLP(q_i, k_j)
```

然后通过重参数化技巧采样 attention：

```text
α_ij = μ_ij + σ_ij ε,    ε ~ N(0, 1)
```

这样 decoder 在训练时可以将输入噪声和场景复杂性反映到 attention 中，而不是始终使用一个固定的相似度。

### 4.3 UPD 的损失

论文使用高斯负对数似然约束概率 attention，使预测分布既要靠近确定性 attention 的目标值，又要学习合适的方差。按高斯分布展开后，其核心形式可写为：

```text
L_UPD = Σ_i,j [ log(σ_ij²) +
        (α_ij - q_i k_j / √d)² / σ_ij² ]
```

直观理解：

- `μ` 描述 attention 的中心趋势；
- `σ` 描述当前 query-key 关系的不确定性；
- 难样本可以有更大的预测方差，模型不必被迫给出过度自信的单点注意力。

UPD 保留标准 decoder 的整体结构：self-attention、概率 cross-attention 和 FFN 堆叠多层，输出 3D boxes 以及下一帧使用的 query 表示。

---

## 5. Hierarchical Query Denoising（HQD）

### 5.1 普通 query denoising 的局限

DN-DETR 类方法会将加噪的 GT box 作为额外 query，让 decoder 重建干净 box，以减少二分图匹配的学习难度。但 3D MOT 中不同噪声 query 的难度可能差异很大：轻微扰动的 query 接近 GT，严重扰动的 query 可能接近背景或另一个目标。如果所有 query 使用同一种 box 和分类监督，容易干扰正常 query 的学习。

### 5.2 按 3D IoU 划分难度

先对 GT 3D box 加噪生成 noised query，再按照 noised query 与对应 GT 的 3D IoU 分为三类：

| 类别 | 条件 | 训练含义 |
|---|---|---|
| Pos | `IoU > β_upper` | 低难度，接近 GT，回归 box 并预测目标类别 |
| Ign | `β_lower ≤ IoU ≤ β_upper` | 难度不明确，忽略，避免干扰正常 query |
| Neg | `IoU < β_lower` | 高难度，训练其识别为背景 |

论文默认的最佳阈值为：

```text
β_lower = 0.30
β_upper = 0.70
```

### 5.3 HQD 损失

```text
L_HQD = L_box^pos + L_cls^pos + L_cls^neg
```

- `L_box^pos`：正样本的 L1 box 回归损失；
- `L_cls^pos`：正样本的 focal 分类损失；
- `L_cls^neg`：负样本区分背景的 focal 分类损失；
- Ign 样本不参与这部分监督。

训练时还要用 mask 分离普通 query 和 denoising query，避免 denoising 分支向正常 query 泄露信息。

HQD 的关键不是单纯“加噪”，而是**根据噪声造成的匹配难度选择监督方式**：接近 GT 的 query 学 box，偏离很远的 query 学背景，中间样本暂时忽略。

---

## 6. 整体优化目标

跟踪 query 的基础损失由分类和 box 回归组成：

```text
L_tracking = L_cls + L_box
```

S2-Track 的总损失为：

```text
L = λ_tracking L_tracking
  + λ_PQI L_PQI
  + λ_UPD L_UPD
  + λ_HQD L_HQD
```

论文默认各项权重均为 `1.0`。因此三个模块分别从不同位置改善系统：

```text
PQI：给 query 更好的空间起点
UPD：让 decoder 能表达预测不确定性
HQD：让 query-GT 学习过程更稳定
```

---

## 7. 训练和推理流程

### 7.1 训练

```text
多摄像头帧 I_t
  │
  ├─ Backbone + FPN → F_t
  ├─ 2D head → 2D boxes
  └─ Depth head → depth
          │
          └─ 投影 → PQI queries

上一帧 queries + PQI queries + random queries
          │
          ▼
UPD decoder → 3D boxes + updated queries
          │
          ├─ tracking classification/box loss
          ├─ PQI auxiliary loss
          ├─ probabilistic attention loss
          └─ HQD denoising loss
```

在视频训练中，上一帧 query 代表已有轨迹，随机和 PQI query 用于发现新目标。GT box 的加噪副本只在训练阶段作为 HQD query 使用。

### 7.2 在线推理

推理时不需要 GT，也不需要 HQD 的加噪 query：

1. 对当前多摄像头图像提取 FPN 特征；
2. 通过 2D 检测和深度预测生成 PQI query；
3. 合并上一帧 track query、PQI query 和随机初始化 query；
4. 用 UPD decoder 预测当前帧 3D boxes；
5. 由 query 的历史来源继承 track ID；
6. 保留有效轨迹 query 传播到下一帧。

因此 S2-Track 仍是端到端 query-based tracker，没有依赖显式的 IoU/Kalman 后处理关联器。

---

## 8. 实验结果和消融

### 8.1 nuScenes 主结果

论文在 nuScenes benchmark 上评估，主要指标包括 AMOTA、AMOTP、RECALL、MOTA、MOTP 和 IDS。

在 nuScenes test split 的 camera-based 3D MOT 对比中：

| 方法 | Backbone | AMOTA | AMOTP | RECALL | MOTA | MOTP | IDS |
|---|---|---:|---:|---:|---:|---:|---:|
| Sparse4D-v3 | V2-99 | 0.574 | 0.970 | 66.9% | 0.521 | 0.525 | 669 |
| S2-Track-F | V2-99 | 0.608 | 0.925 | 75.8% | 0.547 | 0.559 | 963 |
| **S2-Track-F** | **ViT-L** | **0.663** | **0.815** | **72.3%** | **0.554** | **0.530** | **844** |

论文报告的核心结果是：ViT-L 版本在 test split 达到 **66.3% AMOTA**，比此前最佳端到端方案高 **8.9 个百分点**。

在 nuScenes val split 上，ViT-L full-resolution 版本达到 `0.652 AMOTA`；V2-99 full-resolution 版本为 `0.566 AMOTA`。论文同时给出 small-resolution（S）和 full-resolution（F）设置，二者不能直接混作同一计算配置比较。

### 8.2 模块消融

以 small-resolution 设置为例，基线和模块组合的 AMOTA 如下：

| UPD | PQI | HQD | AMOTA |
|---|---|---|---:|
| ✗ | ✗ | ✗ | 0.394 |
| ✓ | ✗ | ✗ | 0.438（+4.4） |
| ✗ | ✓ | ✗ | 0.418（+2.4） |
| ✗ | ✗ | ✓ | 0.423（+2.9） |
| ✓ | ✓ | ✓ | 0.458 |

三个模块单独都带来收益，组合后进一步提升。UPD 的单模块增益最大，但它并不意味着 PQI 或 HQD 可以省略：PQI 改善初始化，HQD 改善训练，两者作用位置不同。

### 8.3 复杂场景

与 PF-Track 对比，S2-Track 在更困难的条件下增益更明显：

| 条件 | PF-Track AMOTA | S2-Track AMOTA | 提升 |
|---|---:|---:|---:|
| 可见性 0–40% | 38.3 | 44.6 | +6.3 |
| 目标尺寸 0–2 m | 22.8 | 28.3 | +5.5 |
| 距离 >30 m | 7.5 | 11.6 | +4.1 |

这与模块设计相符：PQI 提供位置提示，UPD 对不确定的 attention 建模，HQD 让不同难度的 query 获得更合适的监督。

### 8.4 关键超参数消融

- HQD 的最佳阈值为 `β_lower=0.30`、`β_upper=0.70`；
- PQI 使用 stride 16 的 FPN 特征时表现最好；
- 在所测试的 decoder 中，DETR3D 略优于 PETR，论文将 DETR3D 作为默认 decoder；
- 不分层处理所有噪声 query，或不给中间 IoU 样本设置 Ign 区域，都会降低性能。

---

## 9. 和其他 3D MOT 方法的区别

| 方法类别 | 核心流程 | 关联方式 |
|---|---|---|
| Tracking-by-detection | 3D detector → 运动预测/匹配 | 3D IoU、L2、Kalman 或外观特征 |
| MUTR3D / query tracker | query 跨帧传播 + decoder | 由 query 状态隐式继承身份 |
| PF-Track | 加强时空 object representation | query-based 端到端跟踪 |
| **S2-Track** | 初始化、传播、匹配三环分别增强 | PQI + UPD + HQD，仍端到端 |

S2-Track 的贡献点不是提出新的轨迹后处理器，而是把 query tracker 中容易被混在一起的问题拆开：

```text
初始化质量 → PQI
传播时的不确定性 → UPD
训练时的匹配稳定性 → HQD
```

---

## 10. 优点、局限与实现注意点

### 优点

- **端到端**：检测与跟踪共享 query decoder，不依赖显式的帧间数据关联后处理；
- **几何提示简单有效**：2D box 中心与深度能直接生成粗略 3D 位置；
- **覆盖困难场景**：小目标、遮挡和远距离目标上都有更明显收益；
- **训练更稳**：HQD 将加噪 query 按难度分为 Pos、Ign、Neg；
- **可插拔**：论文显示 PQI/UPD 能和 PETR、DETR3D 等 decoder 配合。

### 局限与论文未完全展开的部分

- PQI 依赖额外的 2D 检测和深度预测头，推理时会增加计算和工程维护成本；
- 相机深度误差会直接影响 PQI 的 3D 初始位置，尤其在远距离或遮挡区域；
- UPD 只对 attention 分布建模，`σ` 并不等同于完整的贝叶斯后验或严格校准的不确定性；
- 论文主要验证 nuScenes 的 camera-based 设置，传感器融合和跨数据集泛化仍需额外实验；
- 轨迹长时间消失后如何保留或恢复原 ID，仍受 query 生命周期和 decoder 预测质量影响；
- 复现时需要明确 full/small resolution、backbone、FPN stride、HQD 阈值和 decoder 配置，否则表格中的 AMOTA 不可直接比较。

### 实现时的关键检查

1. 确认相机内参、坐标变换方向和深度单位一致，避免 PQI 位置整体偏移；
2. PQI query 与 random query 要同时保留，不能让辅助 2D/深度分支完全替代随机探索；
3. UPD 的 `σ` 要保持正值，通常需要正值参数化或稳定的数值处理；
4. HQD 要用 3D IoU 按 `β_lower`、`β_upper` 分组，并对 Pos/Neg/Ign 使用不同 mask；
5. denoising query 与正常 query 之间要阻断信息泄露；
6. 测试时移除 GT-based HQD 分支，只保留 PQI、历史 query 和正常 decoder。

---

## 11. 面试版总结

> S2-Track 是一个 camera-based、端到端的 3D 多目标跟踪器。它把 query tracker 拆成初始化、传播和匹配三个部分：初始化阶段用 2D 检测框中心和深度预测生成带 3D 位置先验的 query，同时保留随机 query；传播阶段把传统确定性 cross-attention 改成高斯分布形式的 probabilistic attention，用均值和方差表示预测不确定性；训练阶段对 GT box 加噪，并按 3D IoU 将 query 分成正样本、忽略样本和负样本，形成分层 query denoising。三部分分别提升了初始定位、复杂场景建模和训练收敛，在 nuScenes test 上达到 66.3% AMOTA。

