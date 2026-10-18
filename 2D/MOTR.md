# MOTR：用 Tracklet Query 做端到端多目标跟踪

> **一句话**：MOTR（*MOTR: End-to-End Multiple-Object Tracking with Transformer*）把每条轨迹表示成一个会跨帧传递的 **tracklet query**。当前帧中，旧轨迹由 tracklet query 延续，新目标由 detection query 发现；训练时用 **tracklet-aware label assignment（TALA）** 保证同一个目标在不同帧始终由同一个 query 负责。
>
> **论文**：Zeng et al., *MOTR: End-to-End Multiple-Object Tracking with Transformer*, ICCV 2021（arXiv:2105.03247）
>
> **贯穿例子**：视频片段中，`ID=17` 和 `ID=42` 已经在上一帧出现，`ID=43` 在当前帧刚进入画面。MOTR 要让两个 tracklet query 延续前两条轨迹，再由 detection query 发现 `ID=43`。

---

## 1. MOTR 要解决什么问题

多目标跟踪（MOT）同时需要完成逐帧检测和跨帧身份关联。经典 tracking-by-detection 先用检测器得到每帧的框，再用 IoU、卡尔曼滤波或 ReID 特征匹配不同帧的框。这样做通常有三个问题：

- 检测器和关联器分开优化，误差会逐级传递；
- 一个目标的身份需要在后处理阶段维护，训练时没有直接的端到端监督；
- 遮挡、新目标出现和轨迹终止需要额外规则。

MOTR 的思路是把身份变成 Transformer decoder 的状态：**一个 tracklet query 对应一条轨迹，query 在视频帧之间传递，decoder 负责在新帧中更新它的位置和内容**。因此，跟踪不再是检测结果生成后的独立匹配步骤，而是集合预测模型内部的时序递推。

---

## 2. 整体结构

MOTR 建立在 Deformable DETR 的检测框架上。backbone 提取多尺度图像特征，Transformer encoder 编码特征，decoder 同时接收两类 query：

```text
第 t 帧图像 I_t
      │
      ▼
Backbone + 多尺度特征 + 位置编码
      │
      ▼
Deformable Transformer Encoder
      │  memory_t
      ├────────────────────────────────────────────┐
      │                                            │
      ▼                                            ▼
tracklet queries Q^trk_t                 detection queries Q^det
（来自上一帧的有效轨迹）                 （固定的可学习检测槽位）
      │                                            │
      └────────────────┬───────────────────────────┘
                       ▼
             Transformer Decoder
                       │
          每个 query 输出类别、边界框、状态
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
       延续旧轨迹                 发现新目标
     继承原 track ID              分配新 track ID
           └───────────┬───────────┘
                       ▼
             Query Interaction Module
                       │
               生成 Q^trk_{t+1}
```

与 DETR 一样，decoder 的每个 query 都产生类别和框；与 TrackFormer 类似，MOTR 把有效的历史 query 带入下一帧。MOTR 的重点在于：**训练时怎样让 query 的身份沿时间稳定传播**，以及怎样处理一条轨迹在多帧上的监督。

---

## 3. 两类 query

### 3.1 Tracklet query：一条轨迹的状态

设上一帧有 `M` 条有效轨迹：

```text
Q^trk_t = [q_17, q_42, ..., q_M]
```

每个 query 与一个 track ID 绑定。它包含目标的语义、位置和外观线索，送入当前帧 decoder 后输出更新框和新的 query 表示：

```text
q_17^t  +  memory_t  →  box_17^t, score_17^t, q_17^{t+1}
q_42^t  +  memory_t  →  box_42^t, score_42^t, q_42^{t+1}
```

如果目标在当前帧仍可见，query 会继续传递；如果其预测被判为无效，对应 query 会从下一帧的轨迹集合中移除。**ID 来自 query 的历史来源，而不是每帧重新靠 IoU 匹配。**

### 3.2 Detection queries：发现新目标

detection queries 是一组固定的可学习 query，作用类似 DETR 的 object query。它们在每帧从图像 memory 中寻找尚未被已有轨迹覆盖的目标。一个 detection query 预测出新目标后，经过筛选并分配新的轨迹身份，随后转化为 tracklet query。

贯穿例子可以写成：

```text
decoder([q_17, q_42, d_1, d_2, ...], memory_t)
  ├─ q_17 → ID=17 的当前框
  ├─ q_42 → ID=42 的当前框
  └─ d_j  → 新目标，创建 ID=43
```

两类 query 的分工是：**tracklet query 负责延续，detection query 负责出生**。如果只保留前者，画面中新进入的目标无法被初始化；如果只保留后者，每帧的检测没有身份状态。

---

## 4. 在线推理：轨迹如何递推

对视频帧 `I_1, I_2, ...`，MOTR 的在线过程可以写成：

```text
(Y_t, Q^trk_{t+1}) = MOTR(I_t, Q^trk_t, Q^det)
```

其中 `Y_t` 是当前帧的跟踪结果，`Q^trk_t` 是输入的历史轨迹状态。

逐帧流程：

1. 第一帧没有历史 query，输入空的 tracklet query 和 detection queries；
2. decoder 预测当前帧的框和类别；
3. 高置信度的 detection query 预测初始化新轨迹；
4. 将已有轨迹和新轨迹的 query 状态送入下一帧；
5. 下一帧同时更新旧轨迹并发现新目标；
6. 低置信度或预测为 no-object 的 tracklet query 被过滤。

### Query Interaction Module（QIM）

QIM 位于当前帧 decoder 输出和下一帧 tracklet query 之间，负责处理轨迹 query 的生命周期：

- 根据当前预测置信度筛掉失效轨迹；
- 保留有效轨迹的 decoder 表示，使其携带到下一帧；
- 将新检测到的目标纳入轨迹 query 集合；
- 在 query 数量变化时整理输入，避免无效 query 持续占用 decoder。

因此，MOTR 的“记忆”不是一个单独的外部特征库，而是**一组与 track ID 对齐的动态 query**。

---

## 5. 训练核心：Tracklet-Aware Label Assignment

### 5.1 普通 DETR 匹配的不足

在单帧 DETR 中，匈牙利匹配只需要回答：“这一帧的哪个 query 对应这个 GT？”如果每帧独立匹配，同一个行人在相邻帧可能被不同 query 负责：

```text
第 t 帧：    q_3  ↔  GT(ID=17)
第 t+1 帧：  q_8  ↔  GT(ID=17)
```

即使每帧检测框都正确，query 身份也会漂移，推理时就无法稳定传递 track ID。

### 5.2 TALA 的约束

MOTR 引入 **tracklet-aware label assignment（轨迹感知标签分配）**。核心规则是：

- 已有轨迹对应的 GT，在当前帧优先分配给与其 tracklet 绑定的 track query；
- 同一个 GT identity 在连续帧中始终由同一个 track query 负责；
- 当前帧新出现的 GT 再由 detection queries 进行集合匹配；
- 没有匹配到目标的 query 监督为 no-object。

可以把当前帧 GT 集合拆成两部分：

```text
G_t = G_t^tracked ∪ G_t^new

G_t^tracked：在上一帧已有轨迹，使用对应 tracklet query 监督
G_t^new：    新出现的目标，与 detection query 做一对一匹配
```

这不是简单地把上一帧框复制到当前帧，而是用 identity 约束匹配关系；框的位置仍由当前帧图像特征重新估计。TALA 让模型同时学会“检测得准”和“query 身份不变”。

### 5.3 损失组成

匹配完成后，损失仍然沿用 DETR/Deformable DETR 的集合损失：

```text
L = λ_cls L_cls + λ_box L_L1 + λ_iou L_GIoU
```

- `L_cls`：类别与 no-object 分类损失；
- `L_L1`：边界框坐标回归；
- `L_GIoU`：框重叠与几何关系；
- 对 tracklet query 的监督额外带有跨帧 identity 约束，保证它持续负责同一轨迹。

MOTR 通常按视频片段而不是完全独立的图像训练：模型在片段内逐帧前向，把上帧的 query 状态传给下帧，并在每个时间步计算监督。这种训练让 query 真正经历“出生—延续—终止”的生命周期。

---

## 6. 为什么 MOTR 比 TrackFormer 更强调训练机制

TrackFormer 已经提出了 object query 与 track query 的基本分工，但如果只把 query 状态机械地传到下一帧，训练仍可能出现：

- query 在相邻帧之间交换身份；
- 目标新出现、消失时的 query 数量变化不稳定；
- 单帧 Hungarian 匹配与跨帧身份传递目标冲突。

MOTR 的改进重点是**轨迹级的标签分配与时序训练**。TALA 把“同一目标的多帧监督”绑定到同一个 tracklet query，使 decoder 的输出不只是当前帧的检测结果，也成为下一帧的身份状态。

这也是理解 MOTR 的关键：它的贡献不只是“增加一个 track query”，而是重新设计了 query 在视频训练中的监督方式。

---

## 7. 和 TrackFormer、MOTR 后续方法的比较

| 方法 | 核心 query | 训练/关联重点 |
|---|---|---|
| DETR | object query | 单帧集合预测，无跨帧身份 |
| TrackFormer | object query + track query | 将历史 query 传给下一帧，推理时由 query 继承身份 |
| MOTR | detection query + tracklet query | TALA 进行轨迹感知标签分配，按视频片段递推训练 |
| MOTRv2 等后续方法 | 延续 tracklet query | 继续改进 query 交互、训练稳定性和检测/跟踪性能 |
| SORT / DeepSORT | 检测框 + 外部关联器 | 依赖运动或 ReID 的后处理关联 |

一句话区分：**TrackFormer 说明 query 可以跨帧跟踪，MOTR 进一步规定这些 query 应如何按轨迹被监督。**

---

## 8. 优点、局限与工程注意点

### 优点

- **端到端统一检测与跟踪**：检测、身份关联和状态更新在同一 Transformer 中完成；
- **无需核心外部 ReID 匹配器**：tracklet query 自身携带轨迹状态；
- **身份监督直接**：TALA 显式利用视频标注中的 identity；
- **在线推理**：只依赖当前帧和历史 query，不需要未来帧；
- **可继承 Deformable DETR**：多尺度可变形注意力适合处理不同大小的目标。

### 局限

- **遮挡和长时间消失**：query 被过滤后，目标重新出现可能获得新 ID；
- **计算量仍然较大**：每帧要运行 backbone、encoder 和 decoder，query 递推不等于免费；
- **训练需要连续视频身份标注**：只有单帧检测框无法充分提供 TALA 所需监督；
- **对轨迹生命周期规则敏感**：置信度阈值、query 保留和终止策略会影响 IDF1、HOTA 等指标；
- **视频片段训练显存较高**：片段越长，时序展开和反向传播成本越大。

工程实现时还要注意：历史 query 的顺序必须与 track ID 对齐；新目标分配 ID 后，下一帧必须把它放入 tracklet query；无效轨迹不能无限累积，否则 decoder 的输入规模和误匹配风险都会上升。

---

## 9. 最小伪代码

```python
# history_tracks: 每项包含 track_id 和 query_state
history_tracks = []
next_id = 0

for frame in video:
    image_memory = encoder(backbone(frame))

    track_queries = stack([x.query_state for x in history_tracks]) \
        if history_tracks else empty_queries()
    detect_queries = learned_detection_queries()

    queries = concat(track_queries, detect_queries)
    hs = decoder(queries, image_memory)
    logits, boxes = prediction_heads(hs)

    # 前一部分 query 继承已有 ID；后一部分负责发现新目标
    updated = []
    for i, track in enumerate(history_tracks):
        if is_valid(logits[i], boxes[i]):
            updated.append({
                "track_id": track["track_id"],
                "query_state": hs[i],
                "box": boxes[i],
            })

    for j in valid_new_detections(logits[len(history_tracks):]):
        updated.append({
            "track_id": next_id,
            "query_state": hs[len(history_tracks) + j],
            "box": boxes[len(history_tracks) + j],
        })
        next_id += 1

    emit(updated)                 # 当前帧输出
    history_tracks = updated       # 进入下一帧
```

上面代码省略了真正训练中的 TALA 和 Hungarian 代价，但体现了 MOTR 的状态递推：**tracklet query 保留原 ID，detection query 产生新 ID，decoder 输出成为下一帧 query 状态**。

---

## 10. 面试常见追问

- **Q：MOTR 的核心创新是什么？**  
  A：用 tracklet query 表示并传递轨迹状态，并提出 tracklet-aware label assignment，让同一个 GT identity 在连续帧中由同一个 query 负责，避免逐帧独立匹配造成 query 身份漂移。

- **Q：MOTR 中 tracklet query 和 detection query 分别做什么？**  
  A：tracklet query 延续已有轨迹并继承 ID；detection query 在当前帧发现新目标，经过筛选后初始化新的 tracklet query。

- **Q：TALA 为什么必要？**  
  A：普通 DETR 的 Hungarian 匹配是逐帧独立的，同一个目标在相邻帧可能被不同 query 匹配。TALA 把 GT identity 纳入分配规则，使 query 的职责跨帧稳定。

- **Q：MOTR 推理阶段还需要外部 Hungarian 匹配吗？**  
  A：训练阶段仍需要集合匹配来计算检测损失，但在线推理的主要身份关联来自 tracklet query 的 ID 继承，不需要把所有当前框和上一帧框再做一次外部匹配。

- **Q：目标遮挡后一定能恢复原 ID 吗？**  
  A：不一定。只要对应 tracklet query 没有被终止，它可能在后续帧重新定位目标；如果 query 已被删除，目标再次出现通常会由 detection query 创建新轨迹。

- **Q：MOTR 和 TrackFormer 的最大区别是什么？**  
  A：两者都传递 track query，但 MOTR 更系统地设计了视频片段训练和 TALA，把轨迹 identity 纳入标签分配，重点解决 query 跨帧身份不稳定的问题。

- **Q：为什么不能只用 tracklet query？**  
  A：只用已有轨迹无法发现新进入画面的目标。detection query 提供持续的“出生通道”，使模型能在每帧初始化新轨迹。

---

## 11. 速记卡

- **一句话**：MOTR = Deformable DETR + tracklet query + TALA；
- **两类 query**：tracklet query 延续旧轨迹，detection query 发现新目标；
- **身份来源**：tracklet query 与 track ID 绑定，当前帧输出直接继承该 ID；
- **训练关键**：同一 GT identity 跨帧分配给同一个 tracklet query，防止 query 漂移；
- **推理递推**：`(结果_t, Q^trk_{t+1}) = MOTR(I_t, Q^trk_t, Q^det)`；
- **QIM 作用**：筛除失效轨迹、保留有效状态、把新目标加入下一帧 query 集合；
- **和 TrackFormer 的区别**：MOTR 把轨迹级标签分配和视频片段训练做得更明确；
- **主要难点**：遮挡后的 query 丢失、轨迹生命周期管理、逐帧 Transformer 计算成本。
