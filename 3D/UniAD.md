# UniAD：面向规划的统一自动驾驶框架

> **一句话**：UniAD（Unified Autonomous Driving）把感知、预测、占用建模和规划组织成一个有层次的端到端系统，让后续规划直接使用前面任务产生的结构化结果，而不是把各个模块孤立训练。
>
> **论文**：Yihan Hu et al., *Planning-oriented Autonomous Driving*, CVPR 2023（arXiv:2212.10156，CVPR 2023 Best Paper Award）。
>
> **核心观点**：自动驾驶中的任务并不是并列的多任务学习问题。跟踪结果应当服务于运动预测，地图与目标预测应当服务于占用和规划，最终规划目标可以反向约束前面的表示。

## 1. UniAD 要解决什么问题

传统自动驾驶系统通常把检测、跟踪、建图、预测和规划拆成多个模块，再通过人工设计的接口连接。这样容易出现误差累积：检测框略有偏差，跟踪、预测和规划会继续放大这个偏差；各模块单独优化的指标也不一定对应最终驾驶安全。

另一类端到端方法只从图像直接回归控制量，虽然链路短，但缺少可解释的中间表示，也难以利用目标的身份、地图拓扑和未来运动等信息。UniAD 的折中方案是：保留可解释的中间任务，同时让这些任务在一个可微框架中层级连接，并以规划为最终目标。

## 2. 整体数据流

```text
多摄像头图像 + ego status + 高层驾驶指令
                    │
                    ▼
          BEVFormer 时空 BEV 编码器
                    │
                    ▼
     ┌──────────────┴──────────────┐
     ▼                             ▼
TrackFormer                     MapFormer
目标检测 + 时序跟踪              在线地图元素
     │                             │
     └──────────────┬──────────────┘
                    ▼
              MotionFormer
          周围交通参与者运动预测
                    │
                    ▼
              OccFormer
         未来占用 / 语义占用预测
                    │
                    ▼
                Planner
              自车未来轨迹
```

论文采用 BEV 特征作为各任务共享的空间中间表示。Stage 1 先训练稳定的跟踪和地图模块；Stage 2 再联合优化跟踪、地图、运动、占用和规划。

## 3. BEVFormer：把多视角图像变成时空 BEV

输入是 nuScenes 的多摄像头图像。图像 backbone 和 FPN 提取多尺度特征后，BEVFormer 用一组 BEV queries 表示车辆周围的地面网格。每个 query 通过空间交叉注意力，从对应三维参考点投影到各相机特征图并采样；通过时间自注意力把历史 BEV 融合到当前帧。

可以把一帧的 BEV 更新概括为：

```text
BEV_t = TemporalSelfAttention(BEV_{t-1}, BEV query)
        + SpatialCrossAttention(image features, 3D reference points)
```

历史帧队列使速度、朝向和遮挡目标的状态能够传播。UniAD Stage 1 使用 5 帧队列，Stage 2 使用 3 帧队列。

## 4. 感知与建图：TrackFormer 和 MapFormer

### 4.1 TrackFormer

TrackFormer 采用 object query 在 BEV 上检测车辆、行人等动态目标，并通过上一帧的实例 query 进行时序传播。一个实例包含类别、三维框、速度和历史特征。当前帧 query 与历史 query 的匹配让目标 ID 跨帧保持一致。

跟踪输出不仅用于 AMOTA 评估，也作为 MotionFormer 的输入。因此，跟踪分支需要保留目标的身份和运动状态，而不只是输出当前帧检测框。

### 4.2 MapFormer

MapFormer 使用 map queries 预测车道线、道路边界和人行横道等地图元素。地图元素通常用折线表示，并回归一系列 BEV 平面点。相比离线 HD map，在线地图可以直接从相机观测更新，并能向规划器提供局部道路几何。

TrackFormer 和 MapFormer 都在共享 BEV 上运行，但使用不同类型的 query、类别和几何回归头。两者的输出分别是动态目标实例和静态地图实例。

## 5. MotionFormer：预测周围目标的未来

MotionFormer 接收 TrackFormer 的目标实例、MapFormer 的地图特征和历史轨迹，预测每个交通参与者的多模态未来轨迹。对一个目标，模型输出 `K` 条候选轨迹及对应置信度：

```text
τ ∈ R^(N_agent × K × T × 2)
score ∈ R^(N_agent × K)
```

`N_agent` 是目标数量，`K` 是运动模式数量，`T` 是未来时间步。多模态输出可以表达直行、变道、转弯和停车等互相冲突的意图。

运动预测不是独立的轨迹回归头：目标之间通过 agent-agent attention 交互，目标与地图通过 agent-map attention 交互，历史实例通过 temporal attention 交互。这样模型能同时利用“目标在哪里”“道路允许怎样行驶”和“目标过去如何运动”。

## 6. OccFormer：未来占用建模

单纯预测目标框无法覆盖静态障碍物、被遮挡目标和不规则物体。OccFormer 根据当前 BEV、地图和预测的交通参与者运动，预测未来多个时间步的 BEV occupancy。每个网格可以带有占用状态和语义类别。

占用预测为规划器提供一种比离散框更密集的安全表示：规划器可以检查候选自车轨迹是否进入未来被占用区域。论文报告的占用指标是非车辆类别 IoU（`IoU-n. = 63.7`，UniAD-B Stage 2）。

## 7. Planner：从任务表示生成自车轨迹

规划器读取目标跟踪、地图、运动预测和未来占用特征，并结合高层指令（左转、右转、直行）生成自车未来轨迹：

```text
planning query + ego status + map / agents / occupancy
                         │
                         ▼
                  planning decoder
                         │
                         ▼
                 ego trajectory (x, y)
```

规划器以轨迹点为输出，而不是直接输出油门和方向盘角度。轨迹可以再交给传统控制器执行，也便于用 L2 误差和碰撞率进行开放环评估。规划损失会对未来轨迹进行回归，并通过碰撞相关约束鼓励安全轨迹。

UniAD 的关键是任务之间的层级关系：

```text
tracking → motion prediction → planning
map      → motion / occupancy → planning
occupancy --------------------→ planning
```

因此，规划质量下降时，问题可以追溯到具体的中间表示；同时规划任务的梯度可以反向影响前面的共享 BEV 特征。

## 8. 训练策略和损失

### 8.1 两阶段训练

**Stage 1：感知预训练**

- 训练 TrackFormer 和 MapFormer；
- 使用 5 帧 BEV 队列；
- 获得稳定的跟踪和在线建图初始化。

**Stage 2：端到端联合训练**

- 加入 MotionFormer、OccFormer 和 Planner；
- 使用 3 帧 BEV 队列；
- 联合优化跟踪、地图、运动、占用和规划模块。

这样做的原因是：如果一开始就让所有任务同时学习，跟踪和地图 query 尚未稳定，预测与规划的梯度会使训练过程难以收敛。

### 8.2 总损失

可将总损失写成：

```text
L = L_track + L_map + L_motion + L_occ + L_plan
```

其中：

- `L_track`：目标分类、三维框回归、速度和身份相关损失；
- `L_map`：地图类别和折线点回归损失；
- `L_motion`：多模态轨迹回归和模式分类损失；
- `L_occ`：未来占用和语义占用损失；
- `L_plan`：自车轨迹回归及规划安全相关损失。

多模态运动预测通常采用 best-mode / winner-takes-all 训练：选择与 GT 轨迹误差最小的模式作为正模式，其余模式主要通过分类损失学习较低置信度。

## 9. nuScenes 结果

官方实现给出的 UniAD-B 结果如下：

| 阶段 | AMOTA | IoU-lane | Motion minADE | Occupancy IoU-n. | Planning avg.Col. |
|---|---:|---:|---:|---:|---:|
| Stage 1 感知 | 0.390 | 0.297 | — | — | — |
| Stage 2 端到端 | 0.363 | 0.313 | 0.705 | 63.7 | 0.29% |

这些指标分别对应跟踪、车道地图、周围目标预测、非车辆占用和自车规划碰撞率。SparseDrive 等后续工作常把 UniAD 作为规划导向端到端系统的基线。

## 10. UniAD 的优点和局限

### 优点

- **规划导向**：所有中间任务都围绕最终驾驶规划组织；
- **结构化可解释**：目标、地图、未来占用和轨迹都可以可视化；
- **任务协同**：共享 BEV 与层级连接减少模块间信息损失；
- **可扩展**：可以替换 BEV 编码器，只要提供兼容的 `bev_embed` 和 `bev_pos`；
- **训练可复现**：官方公开了 Stage 1 / Stage 2 配置和 checkpoint。

### 局限

- **计算量较大**：稠密 BEV、时序队列和多个 decoder 带来较高显存与延迟；
- **误差仍会传播**：跟踪漏检或地图错误会影响后续预测和规划；
- **开放环与闭环有差距**：nuScenes 的轨迹指标不能完全反映车辆与其他交通参与者交互后的闭环安全；
- **多任务权重敏感**：各损失的尺度和训练阶段会显著影响收敛；
- **占用与规划依赖数据覆盖**：对长尾场景和未观测区域，模型仍缺少可靠的未来信息。

## 11. 与 SparseDrive、PRIX 的关系

| 维度 | UniAD | SparseDrive | PRIX |
|---|---|---|---|
| 主表示 | 稠密时空 BEV | 稀疏实例 + 几何 anchor | 图像特征，不显式构造 3D BEV |
| 任务组织 | 跟踪 → 预测 → 占用 → 规划 | 稀疏感知 + 并行预测规划 | CaRT 特征增强 + 条件扩散规划 |
| 传感器 | 多摄像头 | 多摄像头 | 摄像头 |
| 规划输出 | 轨迹 | 多模态轨迹并安全筛选 | 两步去噪得到候选轨迹 |
| 主要取舍 | 解释性和任务完整性 | 效率和稀疏表示 | 速度、简洁输入与规划质量 |

## 12. 面试版总结

> UniAD 是一个以规划为中心的端到端自动驾驶框架。它先用 BEVFormer 将多摄像头图像编码成时空 BEV，再依次通过 TrackFormer、MapFormer、MotionFormer、OccFormer 和 Planner 完成目标跟踪、在线建图、周围目标运动预测、未来占用预测和自车轨迹规划。UniAD 的核心不是简单地把多个 head 放在一起，而是让任务形成层级依赖：跟踪服务于预测，地图和占用服务于规划，最终规划损失又反向优化共享 BEV 表示。它在 nuScenes 上取得了较好的跟踪、预测、占用和规划结果，但稠密 BEV 带来的计算成本以及开放环到闭环的差距，是后续 SparseDrive、PRIX 等方法继续改进的重点。

## 13. 论文与代码

- 论文：[Planning-oriented Autonomous Driving](https://arxiv.org/abs/2212.10156)
- 官方实现：[OpenDriveLab/UniAD](https://github.com/OpenDriveLab/UniAD)
- 官方仓库给出的 UniAD-B Stage 2 checkpoint 指标：AMOTA `0.363`、IoU-lane `0.313`、minADE `0.705`、occupancy IoU-n. `63.7`、avg. collision `0.29%`。
