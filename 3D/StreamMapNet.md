# StreamMapNet：用流式时序记忆构建矢量地图

> **一句话**：StreamMapNet 面向视频输入的在线矢量地图构建。它让地图 query 的每个折线点都成为 BEV 特征采样参考点（Multi-Point Attention），并在帧间传播高置信地图 query 与经过位姿对齐的 BEV 特征，兼顾长距离几何建模和时间一致性。
>
> **论文**：Yuan et al., *StreamMapNet: Streaming Mapping Network for Vectorized Online HD Map Construction*, arXiv:2308.12570（2023）。
>
> **代码**：[`yuantianyuan01/StreamMapNet`](https://github.com/yuantianyuan01/StreamMapNet)。文中实验重点讨论 nuScenes 与 Argoverse 2，感知范围、数据划分会影响指标，比较数值时需核对设置。

---

## 1. 任务与动机

在线建图根据车载传感器实时构建车辆周围的 HD 地图，输出车道分界线、道路边界、人行横道等矢量地图元素。地图以实例和折线表示，便于运动预测、规划等模块直接消费。

MapTR 等单帧方法只看当前时刻，遇到遮挡、强曝光变化或大路口时，当前图像可能缺少足够证据；它们通常也在较小范围内评估，范围扩大后精度容易下降。StreamMapNet 主要针对两个问题：

1. **长而不规则的地图几何**：一条道路边界可能横跨很长的 BEV 区域，单个中心参考点附近的局部采样不足以覆盖整条线；
2. **跨帧不稳定**：当前帧暂时不可见的地图结构，可由之前帧补充，但简单堆叠多帧会增加显存和延迟。

论文提出的第三点是评测层面：常用训练/验证划分可能包含大量相同地理位置，导致验证结果偏乐观。因此作者还构造了地理位置重叠更少的新划分。

## 2. 整体流程

```text
当前多相机图像 I_t
      │
      ▼
Image Backbone + BEV Encoder
      │ 当前 BEV 特征 F_t
      ├───────────────┐
      │               │
      ▼               ▼
历史 BEV 对齐融合    地图 Transformer Decoder
Warp + GRU          Multi-Point Attention
      │               │
      └───────┬───────┘
              ▼
   类别分数 + 地图折线点坐标
              │
              ├─ 当前帧高置信 query 选入下一帧
              └─ BEV 特征作为 recurrent memory
```

主干可以看成三部分：图像到 BEV 的特征编码器、用于矢量地图集合预测的 Transformer decoder，以及跨帧保留历史信息的 memory。论文采用 BEVFormer 风格 BEV encoder；时序融合本身由 query propagation 和 BEV fusion 两条路径组成。

## 3. 地图实例与 Query

一个 query 对应一个地图实例，例如一段连续道路边界或一条车道分界线。Decoder 对每个 query 预测类别分数，以及 `N_p` 个 BEV 点：

```text
query_i → (类别 c_i, 折线 P_i)
P_i = {p_i^1, p_i^2, ..., p_i^Np},  p_i^j = (x_i^j, y_i^j)
```

实例 query 将语义和整体几何信息带入 query self-attention；点坐标则描述折线形状。论文实验中的一个设置使用 `N_q=100` 个 query、每条折线 `N_p=20` 个点。它们是实验配置，不是方法必须固定的常数。

## 4. Multi-Point Attention：沿折线采样，而非只看中心

### 4.1 常规可变形注意力的不足

常规 Deformable DETR 风格交叉注意力为每个 query 设一个参考点，并在其附近预测少量采样偏移。对车辆这类局部目标，中心附近通常足以聚合特征；但道路边界和车道线是细长结构，中心点附近的局部区域代表不了整个实例。

### 4.2 用预测折线的所有点作参考点

StreamMapNet 在第 `i` 个 decoder 层使用上一层预测的 `N_p` 个点作为参考点。每个点附近预测偏移与采样权重，再将采样结果加权聚合：

```text
Q_i = Σ(j=1..N_p) Σ(k=1..N_off)
      w_i(j,k) · DA(Q_(i-1), P_i^j + Δ_i(j,k), F_BEV)
```

其中 `DA` 是可变形注意力采样，`P_i^j` 是上一层折线的第 `j` 个点，`Δ` 是局部偏移。这样注意力覆盖整条折线周围，而不是只覆盖一个中心区域。复杂度随点数和采样数增长，约为 `O(N_p × N_off)`，不需要对整个 `H×W` BEV 网格做全局注意力。

坐标 refinement 也有相应调整：论文对各 decoder 层共享回归 MLP，直接预测绝对点坐标，而不是像传统检测式迭代那样只回归相对中心参考点的偏移。预测点既是输出几何，也是下一层采样的几何锚点。

## 5. Streaming 时序融合

与将多帧同时堆叠处理的 **stacking** 策略不同，streaming 每次处理一帧，只维护前序帧压缩后的记忆状态。StreamMapNet 传递两类状态：稀疏的地图 query 和稠密的 BEV 特征。

### 5.1 Query Propagation：传递地图实例

静态地图实例通常会在相邻帧继续出现。模型从前一帧选出置信度最高的 `k` 个 refined queries，将它们传播到下一帧作为位置和语义先验。

由于 query 的坐标位于 ego 坐标系，传播前必须依据两帧之间的 `4×4` 位姿变换 `T` 转换折线点坐标。Query 的隐特征也会结合 `T` 经带残差的 MLP 做变换。下一帧 decoder 把这些历史 queries 与当前帧初始候选拼接；论文设置中 `N_q=100`、`k=33`。

作者加入 transformation loss，要求变换后的 query 解码出的折线点接近经坐标变换得到的目标折线：

```text
L_trans = Σ_j SmoothL1(P_hat^j, P_t^j)
```

这一损失帮助模型学会跨坐标系传递 query。若位姿不准或坐标变换约定出错，传播的几何先验也会偏移。

### 5.2 BEV Fusion：传递稠密场景记忆

另一条路径传播 BEV 特征。先把上一帧记忆按自车位姿变换 warp 到当前 ego 坐标，再用 GRU 与当前帧 BEV 特征融合，最后做 LayerNorm：

```text
F_mem(t-1) = Warp(F_BEV(t-1), T)
F_BEV(t)   = LayerNorm(GRU(F_mem(t-1), F_current(t)))
```

Query propagation 保留实例级的稀疏历史状态；BEV fusion 则保留稠密空间上下文。两条路径互补：前者对“以前识别到的这条线”做显式关联，后者让当前特征图携带过去的空间证据。

### 5.3 Streaming 与 Stacking

| 策略 | 如何处理历史 | 代价与特性 |
|---|---|---|
| Stacking | 多帧特征同时进入一次前向 | 计算、显存随帧数增加；通常只能融合较短历史 |
| Streaming | 每帧单独处理，传递压缩 memory | 单步开销不随历史帧数线性累积；可建立更长时间关联 |

论文训练时 memory 特征不向前序帧反向传播（截断时序梯度）。训练序列每个 epoch 会重新切分；前 4 个 epoch 使用单帧输入以稳定训练，之后训练 streaming 序列，推理则沿完整序列运行。

## 6. 匹配与损失

预测实例数量固定而 GT 数量可变，训练先用二分图匹配建立预测 query 与 GT 地图实例的一对一对应。类别代价使用 Focal Loss；折线代价在 MapTR 定义的等价点序集合 `Γ` 中选损失最小者：

```text
L_line(P_hat, P) = min_(γ∈Γ) (1/N_p) Σ_j SmoothL1(p_hat^j, p^(γ(j)))
L_match = λ1 L_line + λ2 L_Focal
L_train = λ1 L_line + λ2 L_Focal + λ3 L_trans
```

这样，匈牙利匹配负责“实例对实例”，`Γ` 负责“折线内部点序对齐”，Transformation Loss 负责时序 query 的坐标变换学习。推理时输出 query 对应的类别与点集，再按置信度筛选地图实例。

## 7. 数据划分与实验结论

论文在 nuScenes 和 Argoverse 2 上评估，地图类别包括 pedestrian crossing、lane divider、road boundary。除先前常用的较小范围 `前后各 30 m、左右各 15 m` 外，也评估了 `前后各 50 m、左右各 25 m` 的较大范围；AP 阈值随范围设置而变化，因此不同范围的 mAP 不宜直接横向比较。

作者检查训练集和验证集覆盖的地理区域后发现：原始 nuScenes 划分的地理区域重叠约为 **85%**，原始 Argoverse 2 划分约为 **54%**。他们提出新划分，nuScenes 重叠降至约 **11%**，Argoverse 2 为 **0%**。这说明原始 split 上的验证分数可能受地点记忆影响；论文同时报告原始和新划分结果。

论文报告 StreamMapNet 在多个数据集、范围和划分设定下超过对比方法，摘要给出的在线推理速度为 **14.2 FPS**。在新 Argoverse 2 划分、较大范围的消融中，从单帧基线逐步加入 Direct Predict、query propagation、transformation loss、BEV fusion 和更大输入分辨率后，mAP 由 `33.7` 提升到 `51.2`。这些数字只对应论文所列实验配置与划分。

## 8. 和 MapTR 的关系

| 方面 | MapTR | StreamMapNet |
|---|---|---|
| 输出 | 地图实例类别 + 折线点集 | 同样是实例级矢量地图 |
| 点序歧义 | 置换等价建模 | 沿用等价点序匹配思想 |
| 交叉注意力 | MapTR 的地图 decoder 注意力 | Multi-Point Attention，以折线各点为参考点 |
| 时序 | 原始 MapTR 主要为单帧 | query 和 BEV 两路 streaming memory |
| 重点 | 结构化点集表示与学习 | 长折线注意力、跨帧一致性和更大感知范围 |

因此 StreamMapNet 不是简单把多帧图像拼进 MapTR：它调整了 decoder 的采样几何，并显式设计地图 query 与 BEV 特征的 recurrent 传播。

## 9. 优点、限制与复现关注点

**优点**

- Multi-Point Attention 将采样沿折线展开，适合车道线和道路边界这类长形状；
- streaming 记忆避免堆叠长视频带来的线性显存和计算开销；
- query 与 BEV 两条记忆路径分别提供实例级关联和空间上下文；
- 对数据地理重叠的分析提醒后续工作重视地点隔离的泛化评估。

**限制与注意事项**

- 时序收益依赖准确的自车位姿变换；位姿误差会同时影响 BEV warp 和 query 几何；
- memory 特征截断反传，训练并非对完整长序列做端到端 BPTT；
- 大范围建图仍受图像分辨率、BEV encoder 和地图点采样密度影响；
- 14.2 FPS 和 mAP 与硬件、图像尺寸、数据划分及感知范围绑定，不应脱离配置引用；
- 原始 split 上的高分不能单独证明跨地理区域泛化，宜优先报告新 split 或明确空间隔离协议。

## 10. 复习时记住

1. **长折线怎么采特征？** 每条预测折线的多个点都成为 deformable attention 参考点。
2. **历史信息如何保留？** Top-k map queries 做实例级传播；warp 后的 BEV feature 经 GRU 做稠密融合。
3. **为什么 streaming？** 每帧只更新有限 memory，避免把许多历史帧全部堆入一次前向。
4. **训练还包含什么？** 匈牙利匹配、等价点序折线损失、分类损失和 query transformation loss。
5. **论文的评测提醒是什么？** 验证地点与训练地点重叠会高估泛化表现，地图建图 benchmark 应考虑地理隔离。

> 记忆口诀：**沿线多点采样，Query 与 BEV 双路记忆，逐帧流式更新；评测要看地点是否真正隔离。**

## 参考

- 论文: https://arxiv.org/abs/2308.12570
- 官方代码: https://github.com/yuantianyuan01/StreamMapNet
- MapTR 笔记: [MapTR.md](MapTR.md)
