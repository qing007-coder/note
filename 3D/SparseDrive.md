# SparseDrive：基于稀疏场景表示的端到端自动驾驶

> **一句话**：SparseDrive 用稀疏的实例特征和几何 anchor 表示周围车辆、行人以及地图元素，统一完成检测、跟踪和在线建图；随后让周围目标的运动预测与自车规划并行进行，并通过分层规划选择和碰撞感知重打分选出安全轨迹。
>
> **论文**：Wenchao Sun et al., *SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation*, ICRA 2025（arXiv:2405.19620）。
>
> **核心观点**：端到端自动驾驶不一定要先构造稠密 BEV。只保留有限数量的目标实例和地图实例，也可以完成场景理解、时序建模、预测和规划，同时显著降低计算量。

---

## 1. SparseDrive 要解决什么问题

传统自动驾驶系统通常是模块化流水线：感知得到目标和地图，预测周围交通参与者的未来轨迹，规划器再根据这些结果生成自车轨迹。模块化设计便于解释和调试，但模块之间存在信息损失和误差累积，而且每个模块的优化目标不一定与最终规划安全一致。

端到端方法试图把检测、跟踪、建图、预测和规划放进一个可微框架。现有方法的主要问题有两类：

- **表示代价高**：许多方法依赖稠密 BEV 特征，显存和计算开销大；
- **任务组织简单**：运动预测和规划经常按顺序执行，规划只使用预测结果，忽略自车对周围交通参与者的影响；规划通常只输出一条确定性轨迹，难以表达多模态意图和场景不确定性。

SparseDrive 从两个方向改造端到端框架：

1. 用稀疏实例表示替代贯穿整个网络的稠密 BEV，形成 **Sparse-Centric** 感知范式；
2. 将运动预测和自车规划放进同一个并行运动规划器，生成多模态轨迹，再进行安全选择。

---

## 2. 整体数据流

```text
多摄像头图像
      │
      ▼
Image Backbone + Neck
      │
多视角、多尺度图像特征 I
      │
      ▼
对称稀疏感知 Symmetric Sparse Perception
      ├─ 稀疏检测 / 跟踪：agent instances
      └─ 稀疏在线建图：map instances
      │
      ├───────────────────────────────┐
      │                               │
      ▼                               ▼
周围目标实例                   自车实例初始化
      └──────────────┬────────────────┘
                     ▼
          并行运动规划器 Parallel Motion Planner
      ├─ agent-temporal cross-attention
      ├─ agent-agent self-attention
      ├─ agent-map cross-attention
      ├─ 周围目标多模态运动预测
      └─ 自车多模态规划
                     │
                     ▼
            分层规划选择 + 碰撞感知重打分
                     │
                     ▼
                最终自车规划轨迹
```

论文将模型划分为三个部分：

- **Image encoder**：从多摄像头图像得到多尺度特征；
- **Symmetric sparse perception**：用结构对称的 decoder 处理动态目标和静态地图；
- **Parallel motion planner**：同时预测周围目标和自车轨迹，并选择安全规划结果。

---

## 3. 稀疏场景表示：一个实例由什么组成

SparseDrive 不把整个空间编码成密集的 BEV 网格，而是用有限个实例表示场景中的重要元素。完整的实例表示由两部分组成：

```text
实例 = instance feature + geometric anchor
```

- **instance feature**：语义、外观、上下文和时序信息；
- **geometric anchor**：该实例的空间位置、形状、朝向或折线几何。

这种表示方式与 Sparse4D 的 object-centric 思路接近：anchor 提供明确的几何查询位置，deformable aggregation 只从 anchor 周围的少量关键点采样图像特征。模型不需要让每个 BEV 网格都参与全局计算。

### 3.1 动态目标实例

周围车辆、行人等动态目标由检测实例特征和 3D anchor box 表示：

```text
F_d ∈ R^(N_d × C)
B_d ∈ R^(N_d × 11)
```

每个 11 维 anchor box 为：

```text
{x, y, z,
 ln w, ln h, ln l,
 sin(yaw), cos(yaw),
 vx, vy, vz}
```

其中包含中心位置、尺寸、朝向和速度。使用 `sin(yaw)` 与 `cos(yaw)` 可以避免角度在 `π` 和 `-π` 附近产生不连续回归误差。

### 3.2 地图实例

静态地图元素用折线 anchor 表示，例如车道分界线、人行横道和道路边界：

```text
F_m ∈ R^(N_m × C)
L_m ∈ R^(N_m × N_p × 2)
```

每条折线由 `N_p` 个二维点组成：

```text
{x_0, y_0, x_1, y_1, ..., x_(N_p-1), y_(N_p-1)}
```

因此，检测分支处理 box anchor，建图分支处理 polyline anchor；二者的 feature 更新和 decoder 结构保持对称。

---

## 4. Symmetric Sparse Perception：统一检测、跟踪和在线建图

### 4.1 稀疏检测 decoder

检测分支由 `N_dec` 个 decoder 组成：

- 1 个非时序 decoder：处理当前帧随机初始化的实例；
- `N_dec - 1` 个时序 decoder：融合当前帧和历史实例。

每个 decoder 的基本结构为：

```text
anchor → 生成关键点并投影到图像特征
       → deformable aggregation
       → FFN
       → box refinement + classification
```

设当前帧图像特征为 `I`，实例特征为 `F_d`，anchor box 为 `B_d`。deformable aggregation 在 box 周围生成固定或可学习的 keypoints，将这些点投影到各摄像头、各尺度的图像特征上采样，再与实例特征相加：

```text
F_d' = F_d + Aggregate(I, B_d)
```

输出层根据更新后的特征预测分类分数和 box offset，迭代细化 `B_d`。

### 4.2 时序 decoder

时序 decoder 比非时序 decoder 多两个注意力层：

1. **Temporal cross-attention**：当前实例查询上一帧及历史实例；
2. **Self-attention**：当前帧实例之间交互，处理目标间竞争和场景上下文。

在多头注意力中，anchor box 会被映射成高维 anchor embedding，作为位置编码：

```text
E_d = AnchorEmbedding(B_d)
```

这样 query 的语义 feature 和几何位置不是分开的：实例知道“它是什么”，也知道“它大概在哪里”。

### 4.3 稀疏在线建图

在线建图分支沿用检测分支的结构，但将实例定义从 3D box 换成 2D polyline：

```text
map feature + polyline anchor
        │
        ▼
deformable aggregation + FFN + refinement/classification
```

因此检测和地图都可以用相同的稀疏 decoder 模式处理，只需替换几何 anchor、回归头和类别定义。论文将这种结构对称性作为减少工程复杂度、统一场景表示的重要设计。

### 4.4 稀疏跟踪：简单的 ID 继承

SparseDrive 沿用 Sparse4Dv3 的简洁 ID 分配方式：

1. 当前帧检测实例的置信度超过阈值 `T_thresh`；
2. 该实例锁定到一个目标并分配 ID；
3. 之后随着实例跨帧传播，ID 保持不变。

论文不额外加入 track loss 或复杂的身份匹配约束。它依赖 query/instance 的时序传播保持身份一致，因此检测、跟踪和建图可以共享统一的稀疏结构。

实现配置中 `T_thresh = 0.2`。

---

## 5. Parallel Motion Planner：预测和规划同时进行

### 5.1 为什么预测和规划要并行

周围目标未来怎么走，与自车准备怎么走存在双向关系：自车改变速度或方向后，其他交通参与者可能产生不同反应。若严格采用：

```text
先预测周围目标 → 再规划自车
```

预测阶段没有看到自车的候选动作，规划阶段只能被动使用固定的预测结果。

SparseDrive 将自车和周围目标放进同一个 agent-level 表示，联合建模它们的交互：

```text
F_a = Concat(F_d, F_e)
B_a = Concat(B_d, B_e)
```

其中 `F_e, B_e` 是自车实例，`F_d, B_d` 是周围目标实例。

### 5.2 自车实例初始化

以前的做法常把自车 feature 随机初始化，但自车也需要场景语义和几何信息。由于摄像头通常无法直接看到自车本体，SparseDrive 使用前视摄像头最小尺度特征图的全局平均池化初始化自车 feature：

```text
F_e = AveragePool(I_front,S)
```

这样做有两个作用：

- 最小尺度特征图包含更强的全局场景语义；
- 保留一份稠密图像上下文，弥补稀疏感知可能漏掉的障碍物。

自车 anchor 的位置、尺寸和朝向可以直接根据车辆几何设置。速度不能直接使用当前帧 GT，否则会发生 ego status leakage。论文的做法是：

- 训练一个辅助头预测当前自车状态 `E_ST`，包括速度、加速度、角速度和转向角；
- 当前帧自车 anchor 的速度使用上一帧预测速度初始化。

```text
E_ST = MLP(F_e)
```

### 5.3 实例级时空交互

为了让自车拥有历史信息，论文维护一个实例 memory queue，大小为：

```text
(N_d + 1) × H
```

其中 `H` 是保存的历史帧数，默认 `H = 3`。规划器进行三种交互：

#### Agent-temporal cross-attention

每个当前实例查询自己在历史帧中的实例，重点聚合自己的运动状态。它与稀疏感知中的 scene-level temporal attention 不同：后者让当前实例与历史所有实例交互，而规划器这里采用 instance-level interaction，使每个实例主要关注自身历史。

#### Agent-agent self-attention

自车和周围目标之间进行 self-attention，建模车辆之间的空间竞争、让行和跟驰关系。

#### Agent-map cross-attention

agent instances 查询 map instances，将车道边界、道路结构等静态几何信息注入动态目标和自车表示。

---

## 6. 多模态运动预测与规划

道路参与者的未来行为本来就具有多种可能，SparseDrive 不直接预测一条确定性轨迹，而是输出多个 mode：

### 6.1 周围目标的运动预测

```text
τ_m ∈ R^(N_d × K_m × T_m × 2)
s_m ∈ R^(N_d × K_m)
```

- `N_d`：周围目标数量；
- `K_m`：运动模式数量；
- `T_m`：未来时间步数量；
- 每个轨迹点为二维平面坐标；
- `s_m`：每个模式的分数。

### 6.2 自车的规划候选

```text
τ_p ∈ R^(N_cmd × K_p × T_p × 2)
s_p ∈ R^(N_cmd × K_p)
```

规划考虑三种高层驾驶指令：

```text
turn left / turn right / go straight
```

对每个 command 生成多个规划模式，再由后续选择模块确定最终轨迹。

论文默认 `K_m = K_p = 6`，运动预测未来 `T_m = 12` 个时间步，规划未来 `T_p = 6` 个时间步。

### 6.3 模式 query

论文用训练集轨迹的 K-Means 聚类得到先验意图点，再将这些先验点加上正弦位置编码，生成 motion mode query 和 planning mode query：

```text
τ_m = MLP(F_d + MQ_m)
s_m = MLP(F_d + MQ_m)

τ_p = MLP(F_e + MQ_p)
s_p = MLP(F_e + MQ_p)
```

模式 query 让同一个实例可以同时提出“直行、减速、转向”等不同候选，而不是让一个回归头平均多个互相冲突的未来。

---

## 7. Hierarchical Planning Selection：从候选到安全轨迹

规划器输出多条候选轨迹后，SparseDrive 使用分层策略选择最终轨迹 `τ_p*`。

### 第一步：按驾驶指令筛选

根据当前 command 只保留对应的规划候选：

```text
τ_p → τ_p,cmd
```

例如当前指令为 turn right，则只在右转模式中选择。

### 第二步：碰撞感知重打分

使用周围目标的运动预测估计每条自车候选轨迹的碰撞风险。论文采用两个最有置信度的目标运动预测轨迹进行碰撞判断；如果某条规划轨迹与障碍物存在碰撞，则将该轨迹的分数置为 0：

```text
score(τ_p) = 0,   if collision(τ_p, predicted agents)
```

这一步是一个轻量的 differentiable pipeline 外部选择操作，不重新优化轨迹，也不依赖传统的规划后优化器。

### 第三步：选择最高分

```text
τ_p* = argmax score(τ_p,cmd)
```

它与先生成单一轨迹再做碰撞修正的方式不同：模型先保留多模态候选，再在选择阶段过滤不安全候选，因此更容易同时兼顾轨迹质量和安全性。

---

## 8. 训练策略和损失

### 8.1 两阶段训练

SparseDrive 使用两阶段训练：

**Stage 1：稀疏感知预训练**

- 从头训练 symmetric sparse perception；
- 学习稀疏检测、跟踪和在线地图表示；
- 同时使用深度估计辅助任务增强感知训练稳定性。

**Stage 2：端到端联合训练**

- 将稀疏感知和 parallel motion planner 联合优化；
- 不冻结模型权重；
- 感知、预测和规划共同接受最终损失监督。

### 8.2 总损失

论文的整体损失为：

```text
L = L_det + L_map + L_motion + L_plan + L_depth
```

检测损失由 focal 分类和 L1 box 回归组成：

```text
L_det = λ_det_cls L_det_cls + λ_det_reg L_det_reg
```

在线建图的损失形式相同：

```text
L_map = λ_map_cls L_map_cls + λ_map_reg L_map_reg
```

深度估计使用 L1 回归：

```text
L_depth = λ_depth L_depth
```

对于多模态运动预测和规划，使用 winner-takes-all：

1. 计算每个 mode 与 GT 轨迹的 ADE；
2. ADE 最小的 mode 作为正样本；
3. 其余 mode 作为负样本；
4. 分类头学习模式分数，回归头学习正模式的轨迹。

联合的预测/规划损失可写为：

```text
L_motion_planning =
    λ_motion_cls L_motion_cls
  + λ_motion_reg L_motion_reg
  + λ_plan_cls L_plan_cls
  + λ_plan_reg L_plan_reg
  + λ_plan_status L_plan_status
```

其中 `L_plan_status` 监督自车当前状态预测。

实现中的一组损失权重为：

```text
λ_det_cls = 2,    λ_det_reg = 0.25
λ_map_cls = 1,    λ_map_reg = 10
λ_depth = 0.2
λ_motion_cls = 0.2, λ_motion_reg = 0.2
λ_plan_cls = 0.5,   λ_plan_reg = 1.0
λ_plan_status = 1.0
```

检测和地图实例与 GT 通过 Hungarian matching 对齐；跟踪不使用额外 track loss，ID 由置信度阈值和时序实例传播维护。

---

## 9. 关键实现配置

论文给出的默认设置如下：

| 部件 | 配置 |
|---|---|
| 检测 decoder | 6 层：1 个非时序 + 5 个时序 |
| 检测 anchor 数量 | `N_d = 900` |
| 地图 polyline anchor 数量 | `N_m = 100` |
| 每条地图折线点数 | `N_p = 20` |
| 历史检测实例数 | 600 |
| 历史地图实例数 | 33 |
| tracking threshold | `T_thresh = 0.2` |
| 检测范围 | 半径 55 m 的圆形区域 |
| 建图范围 | 纵向 60 m、横向 30 m |
| 规划 memory queue | `H = 3` 帧 |
| 运动模式/规划模式 | `K_m = K_p = 6` |
| 运动预测时间步 | `T_m = 12` |
| 规划时间步 | `T_p = 6` |
| 优化器 | AdamW |
| 学习率策略 | Cosine Annealing |

anchor box 和 polyline 的位置使用训练集 K-Means 聚类得到；其他 box 参数按论文给出的固定初值初始化。多头注意力采用 FlashAttention 节省显存。

---

## 10. nuScenes 实验结果

### 10.1 感知任务

在 nuScenes validation 上，SparseDrive-B 使用 ResNet-101 和 `512×1408` 输入，SparseDrive-S 使用 ResNet-50 和 `256×704` 输入。

| 方法 | Backbone | 3D mAP | NDS |
|---|---|---:|---:|
| UniAD | ResNet-101 | 0.380 | 0.498 |
| SparseDrive-S | ResNet-50 | 0.418 | 0.525 |
| **SparseDrive-B** | **ResNet-101** | **0.496** | **0.588** |

SparseDrive-B 相对 UniAD 提升 `11.6` 个百分点 mAP 和 `9.0` 个百分点 NDS。

跟踪结果：

| 方法 | AMOTA | AMOTP | Recall | IDS |
|---|---:|---:|---:|---:|
| UniAD | 0.359 | 1.320 | 0.467 | 906 |
| SparseDrive-S | 0.386 | 1.254 | 0.499 | 886 |
| **SparseDrive-B** | **0.501** | **1.085** | **0.601** | **632** |

在线建图方面，SparseDrive-B 的 mAP 为 `56.2`，高于 VAD 的 `47.6`。

### 10.2 运动预测与规划

运动预测结果：

| 方法 | minADE | minFDE | Miss Rate | EPA |
|---|---:|---:|---:|---:|
| UniAD | 0.71 | 1.02 | 0.151 | 0.456 |
| SparseDrive-S | 0.62 | 0.99 | 0.136 | 0.482 |
| **SparseDrive-B** | **0.60** | **0.96** | **0.132** | **0.555** |

规划结果：

| 方法 | Avg. L2 (m) | Avg. Collision Rate |
|---|---:|---:|
| UniAD | 0.73 | 0.61% |
| VAD | 0.72 | 0.21% |
| SparseDrive-S | 0.61 | 0.08% |
| **SparseDrive-B** | **0.58** | **0.06%** |

相对 VAD，SparseDrive-B 将平均 L2 误差降低 `19.4%`，碰撞率降低 `71.4%`。

### 10.3 效率

| 方法 | FLOPs | 参数量 | FPS |
|---|---:|---:|---:|
| UniAD | 1709 G | 125.0 M | 1.8 |
| SparseDrive-S | 192 G | 85.9 M | 9.0 |
| SparseDrive-B | 787 G | 104.7 M | 7.3 |

论文报告：相同 backbone 下，SparseDrive-B 的训练速度约为 UniAD 的 `4.8×`，推理速度约为 `4.1×`；轻量的 SparseDrive-S 训练约快 `7.2×`，推理约快 `5.0×`。

---

## 11. 消融实验说明了什么

### 11.1 并行设计

将预测和规划改回顺序执行后，运动预测和碰撞率都变差，说明自车动作与周围目标行为之间的交互不能完全忽略。

### 11.2 自车实例初始化

随机初始化自车 feature、将自车 anchor 参数设为 0 会降低规划 L2 和碰撞表现。自车实例需要语义特征和几何状态，不能只作为一个无内容的 query。

### 11.3 多模态规划

只输出一条确定性轨迹时，碰撞率最高。规划模式数从 1 增加到 6 时性能持续改善，超过 6 后收益趋于饱和。这说明多模态候选是表达驾驶意图不确定性的关键。

### 11.4 Agent-temporal cross-attention

去掉实例级的历史交互后，规划平均 L2 从约 `0.61 m` 恶化到 `0.77 m`。历史信息要按实例聚合，不能只依赖当前帧的场景级融合。

### 11.5 Collision-aware rescore

论文比较了碰撞感知重打分和传统 post-optimization：

- SparseDrive 不使用重打分时，平均碰撞率约 `0.12%`；
- 加入 collision-aware rescore 后降至 `0.08%`，L2 几乎不变；
- 传统后优化会明显增加 L2 和碰撞率，破坏原本的端到端规划行为。

这个结果支持论文的设计：提前生成多模态轨迹并在候选选择阶段屏蔽碰撞轨迹，比先生成单一轨迹再进行大幅后处理更稳健。

---

## 12. 与 BEV-Centric 方法的区别

| 维度 | BEV-Centric 端到端方法 | SparseDrive |
|---|---|---|
| 场景表示 | 稠密 BEV feature | 稀疏实例 feature + 几何 anchor |
| 感知任务 | 检测、跟踪、建图可能使用不同表示 | 对称 sparse decoder 统一处理 |
| 时序建模 | 常在 BEV 或 query 上融合 | instance memory queue + temporal attention |
| 预测/规划 | 常见顺序设计 | 预测和规划并行 |
| 未来轨迹 | 可为确定性单轨迹 | 多模态轨迹候选 |
| 安全选择 | 常依赖规划后优化 | command 筛选 + collision-aware rescore |
| 计算效率 | 稠密特征开销大 | 只处理有限实例和局部采样点 |

需要注意：SparseDrive 并不是完全不使用稠密特征。图像 backbone/neck 仍输出多尺度特征图，自车初始化也使用前视图像的最小尺度特征图。它的“稀疏”主要指场景表示和 decoder 计算围绕实例 anchor 展开，而不是把所有中间图像特征都稀疏化。

---

## 13. 优点、局限与实现注意点

### 优点

- **表示高效**：实例级稀疏表示减少了稠密 BEV 的显存和计算成本；
- **结构统一**：检测、跟踪和在线建图共享对称 decoder；
- **时序自然**：实例 feature 和 anchor 通过 memory queue 传播；
- **预测与规划协同**：自车与周围目标在同一个 agent-level planner 中交互；
- **多模态和安全兼顾**：先生成候选，再按驾驶指令和碰撞风险选择；
- **端到端训练**：第二阶段不冻结感知权重，感知、预测和规划可以联合优化。

### 局限

论文明确指出，SparseDrive 的端到端在线建图性能仍落后于专门的单任务方法；nuScenes 数据规模也不足以完全释放端到端自动驾驶的潜力；此外，开放环评估不能完整代表真实闭环驾驶表现。

从实现角度还需要注意：

- 稀疏实例数量有限，漏检目标不会像稠密 BEV 那样自然存在，因此 anchor 数量、置信度阈值和初始化质量会影响召回；
- tracking 没有额外 track loss，ID 稳定性依赖实例传播和阈值策略；
- 自车速度不能直接读取当前帧 GT，否则会产生 ego status leakage；
- 规划碰撞判断依赖周围目标的运动预测质量，预测漏掉的障碍物仍可能造成风险；
- 论文的碰撞率实现考虑了自车 yaw，并使用车辆 box 重叠判断，不能直接和使用 `0.5 m` occupancy grid 的旧结果混比；
- 复现时必须同时记录 backbone、输入分辨率、anchor 数量、历史帧数、模式数量和评估协议。

---

## 14. 面试版总结

> SparseDrive 是一个基于稀疏场景表示的端到端自动驾驶框架。它用实例 feature 和几何 anchor 表示周围动态目标和地图折线，并用对称的 sparse decoder 统一完成检测、跟踪和在线建图。规划部分先初始化带语义和几何信息的自车实例，再通过 agent-temporal、agent-agent 和 agent-map 三类注意力，同时进行周围目标运动预测和自车规划，输出多模态轨迹。最后按驾驶指令筛选候选，并用碰撞感知重打分将有碰撞风险的轨迹置零，选择最高分轨迹作为规划结果。它的核心价值是用稀疏表示降低 BEV 计算成本，用并行多模态规划改善预测、规划和安全性，在 nuScenes 上同时取得更好的感知、预测、规划和效率结果。

