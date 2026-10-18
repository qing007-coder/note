# TrackFormer：用 Transformer Query 做多目标跟踪

> **一句话**：TrackFormer 把视频多目标跟踪写成逐帧的集合预测：上一帧检测到的目标变成 **track query** 带到下一帧继续找，新出现的目标由 **object query** 发现；检测和跟踪共享 DETR 风格的 Transformer decoder，不需要外置的外观 ReID 网络或传统框匹配器。
>
> **论文**：Meinhardt et al., *TrackFormer: Multi-Object Tracking with Transformers*, CVPR 2022（arXiv:2101.02702）
>
> **贯穿例子**：视频第 `t` 帧有 3 个行人，其中 2 个在上一帧已经出现，另 1 个刚进入画面。模型要延续 2 条旧轨迹，并为新行人开一条轨迹。

---

## 1. 多目标跟踪要解决什么

多目标跟踪（MOT）要在每一帧同时回答两个问题：

1. **检测**：这一帧有哪些目标、类别和边界框是什么？
2. **关联**：这一帧的目标分别对应哪条已有轨迹？新目标何时开始，消失的轨迹何时结束？

传统 tracking-by-detection 通常拆成两个系统：检测器先逐帧出框，再用 IoU、运动模型或 ReID 外观特征做跨帧关联。分阶段设计灵活，但检测误差会传给关联器，而且两个模块的目标未必一致。

TrackFormer 的核心转变是：**把“身份”编码进 query 的状态里**。已存在目标的 query 跨帧传递，下一帧 decoder 直接用它去更新该目标的位置和表示；另有一组 object query 负责发现此前没有轨迹的新目标。

---

## 2. 整体结构

```text
第 t 帧图像 I_t
      │
      ▼
CNN backbone + 位置编码
      │  图像特征 memory_t
      ▼
Transformer encoder
      │
      ├─────────────────────────────────────────────┐
      │                                             │
      ▼                                             ▼
object queries O（固定可学习槽位）       track queries T_t（从 t-1 帧传入）
      │                                             │
      └──────────────────┬──────────────────────────┘
                         ▼
                 Transformer decoder
                         │
             每个 query 输出类别、框、特征
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
 object query 高分预测              track query 预测
 新目标：分配新 track ID             延续原 track ID
          └──────────────┬──────────────┘
                         ▼
             生成下一帧的 track queries T_{t+1}
```

和 DETR 一样，图像通过 backbone 与 encoder 编码成 memory，decoder 中每个 query 经交叉注意力读取图像特征，再由分类头和框头产生预测。不同点在于 decoder 的 query 集合不再只有一类：**object query 负责出生，track query 负责延续**。

---

## 3. 两类 query：新目标与已有轨迹

### 3.1 Object queries：检测尚未跟踪的目标

object queries 是一组可学习向量，作用类似 DETR 的 object query。它们每帧都参与 decoder，与当前图像特征交互，尝试找出当前帧中**还没有对应 track query 的目标**。

当 object query 预测出有效目标时，模型为其创建新的轨迹身份，并把对应的 decoder 表示作为新 track query 状态，供下一帧继续使用。

### 3.2 Track queries：带身份的时序状态

track query 来自上一帧已有轨迹的 decoder 表示。除了表示“这是哪条轨迹”，它也携带上一帧目标的语义与外观线索。它在当前帧通过交叉注意力寻找目标的新位置，输出更新后的框和 query 表示。

贯穿例子中，假设上一帧有两条轨迹 `ID=17` 和 `ID=42`：

```text
T_t = [q_17, q_42]
O   = [o_1, o_2, ..., o_K]

decoder([q_17, q_42, o_1, ..., o_K], memory_t)
  ├─ q_17 → 当前帧的框，仍标记为 ID=17
  ├─ q_42 → 当前帧的框，仍标记为 ID=42
  └─ o_j  → 新行人框，创建新 ID，例如 ID=43
```

**关键点**：TrackFormer 的身份关联不是在所有当前框和上一帧框之间再跑一次 Hungarian/IoU 匹配，而是通过“哪个 track query 产生了这个预测”直接继承身份。Transformer 负责在当前图像中为该 query 找到对应目标。

---

## 4. 一帧到下一帧：轨迹如何流动

逐帧在线推理可概括为以下步骤：

1. 第一帧没有历史轨迹，输入仅有 object queries；
2. decoder 对 object queries 和当前帧特征做预测；
3. 过滤低置信度的目标预测，为有效新目标创建 track ID；
4. 下一帧把有效轨迹对应的 query 状态作为 track queries，同时仍输入 object queries；
5. decoder 更新已有轨迹，并由 object queries 寻找新生目标；
6. 低置信度或无效的轨迹 query 被移除，持续有效的 query 继续传递。

可写成递推形式：

```text
(pred_t, T_{t+1}) = TrackFormer(I_t, T_t, O)
```

其中 `T_t` 是当前帧输入的历史轨迹状态；`pred_t` 包含当前帧输出；`T_{t+1}` 由当前帧仍有效的轨迹预测与新生轨迹组成。**每帧处理完就能输出结果**，因此这是在线跟踪，不依赖未来帧。

### 轨迹的出生、延续与终止

| 事件 | Query 来源 | 行为 |
|---|---|---|
| 新生目标 | object query | 预测有效目标后创建 track ID，状态进入下一帧 |
| 已有目标 | track query | 直接沿用 query 对应的 ID，更新框与 query 表示 |
| 目标离开或跟丢 | track query | 预测置信度低时不再保留该 query |

置信度阈值影响误检、漏检和轨迹寿命：阈值过低会保留错误目标，过高则可能让短暂遮挡或检测变弱的轨迹提前结束。具体阈值属于推理配置，不能和模型的核心 query 机制混为一谈。

---

## 5. 训练：检测损失与跨帧身份监督

### 5.1 单帧检测仍是集合预测

每个 decoder query 输出类别和框。训练时，预测集合与当前帧标注通过匈牙利算法进行一对一匹配，并使用分类和边界框损失监督；未匹配 query 学习为 no-object。这继承了 DETR 的集合预测方式。

### 5.2 视频身份信息如何监督 query

MOT 标注除了类别和框，还包含跨帧一致的 identity。训练时模型需要学会：

- 对于上一帧已经建立的目标，track query 在当前帧应继续预测同一 GT identity；
- object query 应补充发现当前帧中没有由已有 track query 覆盖的新目标；
- 当前帧输出的 query 表示要能作为后续帧的轨迹状态。

因此训练不是“每帧独立跑 DETR”再做后处理，而是让 decoder query 在视频序列中承担身份状态。常见训练流程以相邻帧或采样的帧对构造时序监督，让模型反复经历 query 传播与更新。

### 5.3 为什么还需要 object queries

若只传播已有 track query，模型就只能延续已知对象，无法自然处理新进入画面的目标；若只用 object query 独立检测每帧，又没有携带身份的状态。两类 query 分工使模型把**目标发现**和**轨迹延续**放进同一个 decoder 中协同训练。

---

## 6. 和 tracking-by-detection 的差异

| 维度 | 传统 tracking-by-detection | TrackFormer |
|---|---|---|
| 检测与关联 | 检测器、关联器分开 | 同一个 Transformer decoder 里联合建模 |
| 跨帧身份 | 根据框、运动或 ReID 特征做显式匹配 | 由 track query 的来源直接携带身份 |
| 新生目标 | 新检测框进入关联器后初始化轨迹 | object query 发现目标并初始化轨迹 |
| 主要优点 | 模块可替换、工程生态成熟 | 端到端、关联流程简洁、特征可联合学习 |
| 主要挑战 | 检测与关联误差相互传递 | query 丢失后轨迹可能中断；遮挡和长时重现仍困难 |

TrackFormer 仍然是逐帧检测并维护轨迹状态，只是把关键的关联线索融进 query，而不是说“跟踪问题从此没有状态管理”。轨迹阈值、ID 分配和无效 query 清理仍属于推理流程的一部分。

---

## 7. 和相近方法的关系

| 方法 | Query / 关联思路 | 与 TrackFormer 的区别 |
|---|---|---|
| DETR | object query 做单帧集合检测 | 没有跨帧身份状态 |
| TrackFormer | object query 找新目标，track query 延续轨迹 | 显式把上帧 query 状态带入当前帧 |
| MOTR | 也用 track query 做端到端跟踪 | 更强调端到端视频训练与 tracklet-aware label assignment 等训练机制，是后续的重要发展 |
| SORT / DeepSORT | 检测后做运动或 ReID 关联 | TrackFormer 不依赖独立的后置匹配器作为核心关联机制 |

可以把 TrackFormer 看作 DETR 从图像集合预测走向视频序列建模的一步：**query 不只是检测槽位，也成为了跨帧状态载体**。

---

## 8. 优点、局限与适用场景

### 优点

- **检测与关联统一**：不必单独设计 IoU/ReID 匹配代价并串接两个训练目标；
- **在线输出**：当前帧推理只需当前帧图像和已有 query 状态；
- **身份继承直接**：track query 与 track ID 一一对应，输出关联关系清楚；
- **兼容 DETR 组件**：backbone、Transformer encoder/decoder、分类和框回归都建立在熟悉的检测框架上。

### 局限

- **遮挡与长时间消失**：轨迹 query 若被过滤或状态漂移，重新出现时可能无法恢复原 ID；
- **计算成本**：每帧仍需 backbone 和 Transformer 前向，不能因为 query 机制就默认实时；
- **训练数据要求**：训练需要连续视频和跨帧 identity 标注，不只是静态框；
- **query 管理仍有超参**：初始化、置信度阈值和轨迹终止策略会影响实际表现；
- **一对一监督的收敛特性**：继承 DETR 类模型的训练成本和优化难点。

---

## 9. 阅读论文时抓住的主线

建议按下面顺序理解，不要一开始陷入所有损失项和配置：

1. **问题重述**：视频 MOT 如何转成逐帧集合预测？
2. **两类 query**：object query 为什么负责出生、track query 为什么负责延续？
3. **状态递推**：当前帧 decoder 的输出如何形成下一帧输入？
4. **身份监督**：训练如何利用 GT identity 让 query 学会保持轨迹？
5. **生命周期**：置信度阈值如何控制轨迹初始化和终止？
6. **与后续工作的关系**：MOTR 在时序训练和 tracklet assignment 上做了哪些进一步设计？

---

## 10. 面试常见追问

- **Q：TrackFormer 的核心创新是什么？**  
  A：把已有轨迹表示为跨帧传递的 track query，并与负责发现新目标的 object query 一起送入 DETR decoder，从而在同一模型中完成检测、关联和轨迹状态更新。

- **Q：Track query 和 object query 有什么区别？**  
  A：object query 是共享的可学习检测槽位，每帧负责发现新目标；track query 来自已有轨迹的 query 状态，和一个 track ID 绑定，负责在当前帧继续定位该目标。

- **Q：它还需要 Hungarian 算法吗？**  
  A：训练仍沿用集合预测的一对一匹配来建立预测与 GT 的监督关系。推理阶段的核心跨帧关联则由 track query 继承身份完成，不需要再把全部检测框和上一帧框做一次外部 Hungarian 匹配。

- **Q：新目标如何获得 ID？**  
  A：有效 object query 预测到当前帧中新出现的目标后，系统为它分配新的 ID，并把对应 decoder 状态转为后续帧使用的 track query。

- **Q：目标漏检一帧后还一定能保持 ID 吗？**  
  A：不保证。若对应 track query 仍被保留，后续帧可能继续找回目标；若 query 已被置信度规则移除，模型通常会把重新出现的目标当作新轨迹。遮挡后的重识别能力是这类在线 query 跟踪器的挑战之一。

- **Q：TrackFormer 是 tracking-by-detection 吗？**  
  A：它仍逐帧检测，但不采用“独立检测器 + 外置关联器”的经典拆分；身份信息通过 track query 在模型内部传递。

- **Q：和 MOTR 的关系是什么？**  
  A：二者都以 track query 做端到端多目标跟踪。MOTR 进一步围绕视频序列训练、轨迹级标签分配等机制发展，适合对照阅读它们怎样改进时序监督。

---

## 11. 速记卡

- **一句话**：TrackFormer = DETR 式检测器 + 跨帧传递的 track queries；object queries 找新目标，track queries 续旧轨迹；
- **核心数据流**：`当前图像 → encoder memory`，`[object queries, track queries] → decoder → 框/类别/更新后的 query`；
- **身份从哪来**：track query 与轨迹 ID 绑定，query 的输出直接继承该 ID；
- **Hungarian 的位置**：训练时用于集合预测监督；推理时不承担主要的跨帧关联；
- **新轨迹**：object query 检出新目标 → 分配 ID → 状态转成后续帧的 track query；
- **在线性**：每帧只依赖当前图像和过去传来的 query，不看未来帧；
- **主要难点**：遮挡导致 query 丢失、轨迹终止阈值敏感、逐帧 Transformer 计算成本；
- **方法脉络**：DETR（单帧集合检测）→ TrackFormer（query 跨帧带身份）→ MOTR（更强的端到端时序训练与轨迹监督）。
