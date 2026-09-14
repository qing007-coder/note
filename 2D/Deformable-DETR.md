# Deformable DETR（Deformable Transformers for End-to-End Object Detection）完整流程笔记

> **一句话**：Deformable DETR 保留 DETR 的"CNN + Transformer + 100 个 query + 匈牙利匹配 + 无 NMS"整套骨架，只做**两处替换**——把 encoder 自注意力和 decoder 交叉注意力换成**多尺度可变形注意力**。副作用是：终于能加**多尺度特征**了，于是小目标变好、收敛快了 10 倍。
>
> **论文**：Zhu et al., *Deformable DETR: Deformable Transformers for End-to-End Object Detection*, ICLR 2021（arXiv 2010.04159）
>
> **前置**：`DETR.md`（骨架完全一样，本篇只讲改了什么）；可变形注意力模块本身的详解见 **`Deformable-Attention.md`**（第一次见这个机制，建议先看那篇的第 1~4 节）
>
> **贯穿例子**：输入一张 `800×1066` 的图（和 `DETR.md` 用同一张，方便对照），图中 **3 辆车 + 2 个人**共 5 个目标。其中**一辆只有 32×32 的小汽车**是本篇的主角——DETR 在它身上栽跟头，Deformable DETR 就是为它来的。

---

## 1. 要解决什么问题：DETR 的三宗罪

| 问题 | 现象 | 根因 |
|---|---|---|
| **① 收敛慢** | 300 epoch 才追上 Faster R-CNN，需 16 张 V100 训 3 天（2000 GPU 小时） | 注意力从"近似均匀分布"起步，模型要靠试错学会"该往哪看"；每个 GT 只有 1 个正样本，监督稀疏 |
| **② 小目标差** | `AP_S` 只有 20.5（Faster R-CNN + FPN 是 26.6） | 只用 C5，stride 32。`32×32` 的小车在特征图上只剩 **1 个格子** |
| **③ Encoder 太贵** | 850² 自注意力占了全模型 **88%** 的注意力开销 | 全注意力对 token 数是 `O(N²)` |

②和③是**同一个死结**，这点最容易看漏：

> **小目标要靠多尺度特征，但 DETR 的全注意力加不起多尺度。**
> 把 C3~C6 四层一起送进 DETR 的 encoder，token 从 850 涨到 **17821**（21 倍），
> 注意力矩阵变成 `17821² × 8 ≈ 25.4 亿`，fp32 光单层就 **≈10 GB**。
> ——**不是"效果不好"，是"根本跑不起来"。**

后面所有内容都是这一句话的展开。

---

## 2. 动了哪两处？一张 diff 表

| 部件 | DETR | Deformable DETR | 为什么 |
|---|---|---|---|
| **输入特征** | C5 单尺度，stride 32，**850** token | C3/C4/C5/C6 四尺度，stride 8/16/32/64，**17821** token | 治小目标 |
| **Encoder 自注意力** | 全注意力 `850×850` | **MSDeformAttn**：每 token 采样 `L×K = 16` 点 | 全注意力在多尺度下不可行 |
| **Decoder 自注意力** | 全注意力 `100×100` | **保持不变**（`300×300`） | query 数量小，本来就便宜；还要靠它做 query 间去重 |
| **Decoder 交叉注意力** | 全注意力 `100×850` | **MSDeformAttn**：每 query 采样 16 点 | 同上，且引入**参考点先验** |
| **位置编码** | 2D sin，加在 Q/K 上，**单独张量**传 | 加到特征上 + **scale-level embedding** | 多尺度必须能区分 token 来自哪层 |
| **object query** | 100 | **300** | 配合多尺度，槽位不够用 |
| **分类损失** | softmax CE + ∅ 类降权 0.1 | **Focal Loss** | 与稀疏采样更搭 |
| **训练轮数** | 500 epoch | **50 epoch** | —— |
| **额外可选技巧** | — | 迭代框精修、两阶段 | 再涨 1~2 AP |

**没动的部分（仍然是 DETR）**：匈牙利一对一匹配、集合损失、无 anchor、无 NMS、辅助损失、ResNet-50 backbone、`d_model=256 / 8 头 / 6 层 encoder / 6 层 decoder`。

---

## 3. 整体结构（一张图）

```
[1,3,800,1066]  输入图（短边缩放到 800）
      │
      ▼  ResNet-50（stride 32，BN 冻结，lr_backbone = 1e-5）
  ┌───────┬────────┬───────────────────────────────────────┐
  C3      C4       C5 ──(1×1conv 投影后)──3×3,s2 conv──► C6
100×134  50×67    25×34                                 13×17
 512ch   1024ch   2048ch                                256ch
  └───────┴────────┴───────────────────────────────────────┘
      │  C3/C4/C5 各过 1×1 conv、C6 过 3×3 conv（input_proj），全部压到 256 通道
      ▼
  {x⁰:100×134, x¹:50×67, x²:25×34, x³:13×17}    共 N = 17821 个 token
      │
      ▼  2D sin 位置编码 + scale-level embedding（4 个可学习向量）
      ▼  四层 flatten 后拼接成 [17821, 1, 256]，记作 src
      │
      ▼  ┌────────────────────────────────────────────────────┐
         │  Encoder × 6 层                                     │
         │  MSDeformAttn 自注意力（替代原来的 850² 全注意力）   │ ← 每个 token 只采 L×K=16 点
         │  + FFN（256→1024→256）                              │   参考点 = 自己的位置
         └────────────────────────────────────────────────────┘
      │
      ▼  memory [17821, 1, 256]
      │
      │  query_embed [300, 256]（可学习）
      │  + reference_points = sigmoid(Linear(query_embed))  → [300, 2]  ★ 新增
      ▼  ┌────────────────────────────────────────────────────┐
         │  Decoder × 6 层                                     │
         │  ① 自注意力：300 个 query 互相看（**保持 dense**）   │
         │  ② 交叉注意力 → MSDeformAttn                        │ ← 参考点 = ①的 (x,y)，query 看 17821 个 token
         │  ③ FFN                                              │
         └────────────────────────────────────────────────────┘
      │
      ▼  hs [6,1,300,256]（6 层每层都输出一份，算辅助损失）
[1,300,256]
      ├─► class_embed: Linear(256, 92)              → [1,300,92]
      └─► bbox_embed : MLP(256→256→256→4) + sigmoid → [1,300,4]
      │
      ▼
300 个预测 ──训练：匈牙利匹配 + 集合损失──► 推理：卡阈值，**依然无 NMS**
```

**和 DETR 图的两个视觉差异**（对照 `DETR.md` 第 2 节看）：
- 上面多了一个 **C6 分支 + 4 路 1×1 conv**，token 总数从 850 变成 17821；
- decoder 里多了一条 `query_embed → sigmoid(Linear) → reference_points` 的支路。

【贯穿例子】那辆 `32×32` 的小车在哪？
- C3（stride 8）：占 `32/8 × 32/8 = 4×4 = 16` 个格子，位置很清楚；
- C6（stride 64）：只占不到 1 个格子，但"这里有东西"的语义最强。
多尺度可变形注意力让 300 个 query 里的某一个**同时**看到这 4 层的 16 个采样点，选出最有用的一层——这是 DETR 做不到的。

---

## 4. 逐模块拆解 + 维度流水账

### 4.0 维度总表（输入 `800×1066`）

| 步骤 | 操作 | 张量形状 |
|---|---|---|
| 输入 | 短边缩放到 800 | `[1, 3, 800, 1066]` |
| Backbone | ResNet-50 | C3~C5 |
| C3 | res3 输出 | `[1, 512, 100, 134]` |
| C4 | res4 输出 | `[1, 1024, 50, 67]` |
| C5 | res5 输出 | `[1, 2048, 25, 34]` |
| C6 | 在**投影后**的 C5（256ch）上做 `3×3, stride 2` | `[1, 256, 13, 17]` |
| input_proj | 3 个 `Conv2d(C, 256, 1)` + 1 个 `Conv2d(256, 256, 3, s2)` | 4 张 `[1, 256, H_l, W_l]` |
| 位置编码 | 2D sin + `level_embed[l]` | 4 张 `[1, 256, H_l, W_l]` |
| flatten + cat | 4 层按顺序拼接 | `[17821, 1, 256]` |
| Encoder | 6 层 ×（MSDeformAttn + FFN） | `[17821, 1, 256]` |
| query_embed | `nn.Embedding(300, 256)` | `[300, 1, 256]` |
| reference_points | `sigmoid(Linear(256→2)(query_embed))` | `[300, 1, 2]` |
| Decoder | 6 层 ×（自注意力 + MSDeformAttn + FFN） | `[6, 1, 300, 256]` |
| 取最后一层 | `hs[-1]` | `[1, 300, 256]` |
| class_embed | `Linear(256, 92)` | `[1, 300, 92]` |
| bbox_embed | `MLP(256→256→256→4)` + sigmoid | `[1, 300, 4]` |

**多尺度尺寸怎么算**：官方统一用 `ceil`。`800/8=100`、`1066/8=133.25→134`……于是得到 `100×134 / 50×67 / 25×34 / 13×17 = 13400+3350+850+221 = 17821`。

### 4.1 多尺度特征构造（其实不需要 FPN）

```python
# 前 3 个：各层 1×1 conv 对齐通道（接 GroupNorm）
input_proj = nn.ModuleList([nn.Sequential(nn.Conv2d(c, 256, 1), nn.GroupNorm(32, 256))
                            for c in [512, 1024, 2048]])
# 第 4 个：作用在【投影之后的 C5】上 —— 注意输入通道是 256，不是 2048
if num_levels == 4:
    input_proj.append(nn.Sequential(nn.Conv2d(256, 256, 3, stride=2, padding=1),
                                    nn.GroupNorm(32, 256)))       # → C6: 13×17
```

三个要点：

1. **只取各 stage 的"最后一张特征图"**，然后**自顶向下一条 FPN 通路都没有**——没有上采样、没有横向相加。
   论文专门验证过：**已经用了多尺度可变形注意力之后，再补一条 FPN 通路几乎没有增益**。原因很好理解：可变形注意力允许一个 query **跨层直接采样**（C3 的点和 C6 的点被同一个 query 的 `A` 加权求和），信息融合在注意力里已经做了，FPN 那套"逐级回流"就多余了；
2. **C6 是造出来的**：注意是 **`1×1 conv` 投影之后**的 C5（256 通道）再过 `3×3, stride 2`，**不是**直接拿 2048 通道的 C5 去卷。`25×34 → 13×17`，正好等于 `ceil(800/64) × ceil(1066/64)`。它提供 stride 64 的全局视野，专门服务大目标（论文报告 `AP_L` 因此受益）；
3. **通道统一 256**：和 FPN 的横向连接一个道理，1×1 conv 只对齐通道、不动分辨率。

【贯穿例子】`2048 → 256` 的那两个 1×1 conv 各约 `2048×256 ≈ 52 万` 参数，几乎不值一提。

### 4.2 Encoder：`MSDeformAttn` 自注意力

每层结构（**仍然是 Post-Norm**，和 DETR 一致）：

```
src = norm1(src + dropout(MSDeformAttn(q=src, v=src, ref=自身位置)))
src = norm2(src + dropout(FFN(src)))            # FFN: 256 → 1024 → 256
```

**编码器里的三个"是什么"**（这是第一次看最容易卡住的地方）：

| 问题 | 答案 |
|---|---|
| **query 是什么？** | **就是特征图上的每个 token 自己**（自注意力，`query = value = src`） |
| **参考点 `p̂_q` 是什么？** | **这个 token 自己的位置**。由 `torch.linspace(0.5, H-0.5, H) / H` 生成网格，即归一化到 `[0,1]` 的像素中心 |
| **所以它在做什么？** | "以我自己的位置为中心，预测 16 个采样偏移，去 4 个尺度上取 16 个特征点，加权平均成我的新表示" |

```python
# 参考点生成（mmcv，单张无 padding 的简化版）
ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_-0.5, H_), torch.linspace(0.5, W_-0.5, W_))
ref = torch.stack((ref_x.reshape(-1) / W_, ref_y.reshape(-1) / H_), -1)   # [H*W, 2]，∈[0,1]
# 官方代码还额外除以 valid_ratios（处理一个 batch 里长短边不同的 padding），这里省略
```

**位置编码这次是直接加到 `src` 上的**（DETR 是当独立张量传给 Q/K），并且额外加 `level_embed`：

```python
lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)   # [4, 256] 可学习
src = src + lvl_pos_embed
```

【贯穿例子】encoder 的算力账（单层、8 头）：

| 方案 | 注意力权重个数 | 说明 |
|---|---|---|
| DETR（单尺度 850 token） | `850 × 850 × 8` = 578 万 | 尺度太少 |
| **假想**：DETR 全注意力 + 多尺度 | `17821² × 8` ≈ **25.4 亿** | fp32 **≈10 GB/层**，跑不起来 |
| **Deformable DETR** | `17821 × 16 × 8` = **228 万** | token 数 ×21，开销反而 ÷2.5 |

### 4.3 Decoder：自注意力不动，交叉注意力换掉

每层三个子层（子层数量与 DETR 完全相同）：

```
① 自注意力（**保持 dense 全注意力**）：300 个 query 互相看
② 交叉注意力 → MSDeformAttn：q = tgt，v = memory [17821]，ref = query 自己预测的参考点
③ FFN
```

**为什么 decoder 自注意力不改？** 两个理由：
- `300² = 9 万`，本来就便宜，改稀疏采样省不下什么；
- **更要紧的是它有用**：300 个 query 互相看一眼，才能协商出"谁去负责哪个物体"，这是 DETR 避免重复框的机制之一，不能丢。

**新增的关键部件：query 自己的参考点。** 这是 decoder 与 DETR 最大的语义差别：

```python
self.reference_points = nn.Linear(d_model, 2)          # 只输出 (x, y)
...
reference_points = self.reference_points(query_embed).sigmoid()   # [300, 2] ∈ [0,1]²
```

- DETR 里 object query 是一个**纯语义向量**，通过交叉注意力在全图 850 个 token 里"泛泛地找"；
- Deformable DETR 的每个 query 额外带一个 **2D 空间先验 `(x, y)`**：`(0.62, 0.31)` 就是"我大概在图的右上方"。
  交叉注意力采样时以它为原点，视觉上非常直观——**每个 query 一开始就"站"在图上的某个位置**。

【贯穿例子】`hs [6,1,300,256]`，6 层每层都输出一份（辅助损失）。300 个 query 里，大概有 5 个会分别落在 3 辆车 + 2 个人的位置上，其余 295 个被判为 ∅。

### 4.4 两个预测头（与 DETR 基本一致）

```python
self.class_embed = nn.Linear(256, 91 + 1)      # +1 是 no-object
self.bbox_embed  = MLP(256, 256, 4, 3)         # Linear→ReLU→Linear→ReLU→Linear + sigmoid
```

- 分类用 **focal loss**（不再是 softmax CE + ∅ 降权）。官方配置 `--focal_alpha 0.25`，`γ = 2`。论文正文没展开解释这个选择，但它正是 DETR 论文试过、觉得"略好但更复杂"而**放弃**的那个变体（见 `DETR.md` §4.3）——Deformable DETR 把它捡了回来。合理的猜测是：可变形注意力的监督信号比全注意力更稀疏，更需要 focal loss 来抑制大量易分负样本的梯度；
- 框回归仍是**归一化的 `cxcywh` + sigmoid**，相对整张图；
- 官方配置里分类项权重被提到了 2（`set_cost_class = 2`、`cls_loss_coef = 2`），框仍是 L1 权重 5、GIoU 权重 2。

【贯穿例子】`[1,300,256] → 分类 [1,300,92] + 框 [1,300,4]`，和 DETR 一模一样，只是 100 → 300。

### 4.5 可选技巧一：迭代框精修（Iterative Bounding Box Refinement）

思路和级联检测头（Cascade R-CNN）一样：**每一层 decoder 的输出拿去修正下一层的参考点**。

```python
# 第 d 层的框预测 = 上一层的参考点 + 本层回归的"增量"
reference = inverse_sigmoid(reference)                  # ① 先反 sigmoid 回到 logit 空间
tmp = bbox_embed(hs[d])                                 # ② 预测增量 (Δx, Δy, Δw, Δh)
tmp[..., :2] += reference                               # ③ 只对中心点做"残差相加"
outputs_coord = tmp.sigmoid()                           # ④ 再 sigmoid 回 [0,1]
# ⑤ 修正后的框中心 → detach（切断梯度）→ 作为第 d+1 层的参考点
```

四个容易踩的点：

1. **要有 `inverse_sigmoid` 这一步**：参考点是 `[0,1]` 的归一化坐标（sigmoid 的输出），而残差相加必须在 logit 空间做，否则"加一个增量"的几何意义是错的；
2. **只有中心点 `(x, y)` 参与残差**，`w, h` 是直接预测的；
3. **参考点要 `detach`**（官方做法叫 "look forward once"）：第 `d` 层预测出的框，`detach` 之后才作为第 `d+1` 层的参考点。论文的定位是——参考点在各层之间充当**先验**，不是待优化的连续变量，所以第 `d+1` 层的损失只回传到第 `d+1` 层的参数。后来 DINO 的 "look forward twice" 反其道而行，把这条梯度通路打开，报告约 **+0.4 AP**（47.0 → 47.4），这是个后续改进点；
4. **各层检测头不共享参数**。

【贯穿例子】那辆小汽车：第 1 层参考点 `(0.62, 0.31)` 粗粗地框住了它的左上角；第 2 层以修正后的中心重新采样 16 个点；到第 6 层，采样点已经全部汇聚到车身上。**功能上等于"注意力知道自己该看哪儿了"**。

### 4.6 可选技巧二：两阶段 Deformable DETR

借两阶段检测器的思路，给 encoder 加一个"提案生成"：

1. **把 encoder 的每个特征点当成一个 anchor**：中心是该特征点的位置，宽高取一个固定基值（论文用 `0.05`，按层级缩放）；
2. **加一组检测头**（回归 + 分类）预测相对这些 anchor 的偏移，得到**第一阶段的候选框**；
3. **取 top-k 个得分最高的候选框**，把它们的中心作为 **decoder 的参考点**；query 和 query embedding 也由这些参考点的位置编码生成。

区别在于：一阶段的参考点来自"可学习的 `Linear(query_embed)`"，两阶段的参考点来自"encoder 特征预测的真实提案"——后者质量更高，因此再涨约 0.8~1 AP（45.4 → 46.2）。

---

## 5. 训练（和 DETR 完全同构，只有超参变了）

匹配与损失的形式和 DETR 一模一样（匈牙利一对一 + 集合损失 + 6 层辅助损失），**变的只是数值**：

| 超参 | DETR | Deformable DETR |
|---|---|---|
| epochs | 500 | **50** |
| object queries | 100 | **300** |
| 分类损失 | softmax CE + ∅ 降权 0.1 | **focal loss**（α=0.25, γ=2） |
| 分类项权重 | 1 | 2 |
| 匹配代价 | `-p̂(c) + 5·L1 + 2·GIoU` | `2·(-p̂(c)) + 5·L1 + 2·GIoU` |
| 学习率 | 1e-4（backbone 1e-5），200 epoch ÷10 | 2e-4（backbone 2e-5），40 epoch ÷10 |
| 梯度裁剪 | max_norm 0.1 | max_norm 0.1 |

**为什么 50 epoch 能顶 500 epoch？** 核心是**引入了位置先验**：

- DETR 的注意力从近似均匀起步，等于没有任何空间假设，要从零学会"看哪里"；
- Deformable DETR 的参考点 + 初始化好的星形采样，一开始就告诉模型"先看你周围这一圈 + 顺便扫一眼别的尺度"，把搜索空间砍掉了一大截。

**训练时的额外注意**：官方仓需要先编译 CUDA 算子（`cd models/ops && python setup.py build install`），否则跑不起来——因为 `MSDeformAttn` 的双线性采样 + 加权求和是自定义 kernel。

---

## 6. 结果与消融

### 6.1 COCO 2017 val（ResNet-50）

| 方法 | epochs | AP | AP₅₀ | **AP_S** | AP_M | AP_L | FLOPs | 训练 GPU 小时 | FPS |
|---|---|---|---|---|---|---|---|---|---|
| Faster R-CNN + FPN | 109 | 42.0 | 62.1 | 26.6 | 45.4 | 53.4 | 180G | 380 | 26 |
| DETR | 500 | 42.0 | 62.4 | **20.5** | 45.8 | 61.1 | 86G | **2000** | **28** |
| DETR-DC5 | 500 | 43.3 | 63.1 | 22.5 | 47.3 | 61.1 | 187G | 7000 | 12 |
| DETR-DC5 | **50** | 35.3 | 55.7 | 15.2 | 37.5 | 53.6 | 187G | 700 | 12 |
| **Deformable DETR** | **50** | **43.8** | 62.6 | **26.4** | 47.1 | 58.0 | 173G | **325** | 19 |
| + 迭代框精修 | 50 | 45.4 | 64.7 | 26.8 | 48.3 | 61.7 | 173G | 325 | 19 |
| ++ 两阶段 | 50 | 46.2 | 65.2 | **28.8** | 49.2 | 61.7 | 173G | 340 | 19 |

（**数值口径提示**：上表用的是论文 Table 1 的原始数字。官方仓库 README 与 MMDetection 复现报的数略高——README 是 base **44.5** AP、+迭代框精炼 **46.2** AP；MMDetection 是 44.5 / 46.1 / 46.8，且提示有 ±0.3 mAP 波动。"50 epoch 追平 DETR 500 epoch""`AP_S` 从 20.5 提到 26.4"这两个结论在所有版本里都成立。）

**三行字读懂这张表**：

1. **50 epoch 的 Deformable DETR（43.8）> 500 epoch 的 DETR-DC5（43.3）**——同样是 50 epoch 时 DETR-DC5 只有 35.3 AP，差距 8.5 个点；
2. **`AP_S` 从 20.5（DETR）提到 26.4，两阶段到 28.8**——比 Faster R-CNN + FPN 还高。小目标这个短板补上了；
3. **但 FLOPs 更高（173G vs 86G）、推理更慢（19 vs 28 FPS）**。因为它老老实实处理了 4 个尺度的高分辨率特征图。它买到的不是"更快"，而是"**训练便宜 6 倍 + 小目标更好**"。

### 6.2 消融要点

| 消融项 | 结论 |
|---|---|
| 采样点数 `K`（1→4） | AP 单调提升，`K=4` 是默认值；`K` 再大收益递减 |
| 注意力头数 `M` | `M=8` 默认，多头的意义和普通注意力一样：不同头学不同方向的采样 |
| 单尺度 vs 多尺度 | 多尺度约 **+1.7 AP、+2.9 AP_S** |
| 多尺度 + 再补 FPN | **几乎无增益** → 可变形注意力自己就能跨尺度融合信息 |
| 从 C3 开始（不用 C2） | C3 是性价比拐点，再往浅走收益不抵开销 |

---

## 7. 面试常见追问

- **Q：Deformable DETR 相对 DETR 到底改了什么？**
  A：两处替换 + 一个副产品。**替换** encoder 自注意力、decoder 交叉注意力为多尺度可变形注意力；**副产品**是终于能用多尺度特征（C3~C6）。其余（匈牙利匹配、集合损失、无 anchor、无 NMS、辅助损失、两个预测头）全都不变。

- **Q：为什么 DETR 不能直接加多尺度特征？**
  A：全注意力对 token 数是 `O(N²)`。四层特征会让 token 数从 850 涨到 17821，注意力矩阵 `17821²×8 ≈ 25.4 亿`，fp32 单层就要 ≈10 GB，显存和算力都撑不住。**不是效果问题，是可运行性问题**——这正是可变形注意力存在的前提。

- **Q：可变形注意力的 encoder 参考点和 decoder 参考点有什么不同？**
  A：encoder 里 query 就是特征图的 token 自身，参考点 = **它自己的位置**（归一化坐标的固定网格）；decoder 里 query 是 object query，参考点 = **`sigmoid(Linear(query_embed))` 预测出来的 2D 坐标**，是可学习的空间先验。前者是"我站在这里看周围"，后者是"我要去这里看"。

- **Q：为什么用多尺度可变形注意力之后就不需要 FPN 了？**
  A：因为**跨尺度融合已经发生在注意力内部**了。同一个 query 在 C3 上采 4 个点、C4 上 4 个点……最后用一组 softmax 权重把它们加权求和，等价于"模型自己去挑最有用的尺度"。FPN 的逐级上采样 + 相加是一种**固定、手工**的融合方式，在这里就成了多余的一层。

- **Q：`scale-level embedding` 是干什么的？为什么不能只用 sin 位置编码？**
  A：因为**同一个归一化坐标在不同层是同一个值**（C6 的坐标是 C3 的子集），sin 位置编码分不清"我来自 stride 8 还是 stride 64"。加一个每层一个的可学习 256 维向量，模型才能区分层级。

- **Q：decoder 的自注意力为什么还保留全注意力？**
  A：① 300 个 query 算 `300²` 本来就便宜；② 更重要的，它承担"query 之间协商分工"的职责——正因为有它，多个 query 才不会抢同一个目标，DETR 系列"无 NMS"的性质依赖于此。

- **Q：Deformable DETR 比 DETR 快吗？**
  A：**看比什么**。注意力模块本身快得多（线性 vs 二次），训练总成本从 2000 GPU 小时降到 325 GPU 小时（**约 6 倍**），收敛轮数快 **10 倍**。但**单张图的 FLOPs 反而更高**（173G vs 86G），推理 19 FPS 也慢于 DETR 的 28 FPS。它买的是"训练便宜 + 小目标好"，不是"推理快"。

- **Q：迭代框精修里为什么要 `detach` 参考点？**
  A：论文把参考点定位成**每一层的先验**，不是待优化的连续变量——所以第 `d+1` 层的损失只回传到第 `d+1` 层的参数，不跨层回传（官方注释叫 "look forward once"）。注意这不是终点：DINO 的 "look forward twice" 把这条梯度通路打开后还涨了约 0.4 AP，说明"该不该 detach"本身是个可调的设计选择。

- **Q：可变形注意力和 DCN（可变形卷积）的关系？**
  A：可变形注意力是 DCN 的推广：`L=1, K=1` 且 `W'_m` 取单位矩阵时退化成可变形卷积。区别是 DCN 的偏移在 `k×k` 窗口内、由共享的卷积核产生；可变形注意力的偏移 **per-query 预测**、参考点可以在全图任意位置、权重做了 softmax 归一化，还支持跨尺度。详见 `Deformable-Attention.md` 第 2 节。

---

## 8. 最小实现（PyTorch，跟贯穿例子跑一遍）

可变形注意力模块本身见 `Deformable-Attention.md` 第 8 节（`MSDeformAttn`），这里只搭"骨架接线"，确认维度对得上。

### 8.1 多尺度特征 + 参考点

```python
import math
import torch
import torch.nn as nn

# 贯穿例子：800×1066，短边 800
H, W = 800, 1066
strides     = [8, 16, 32, 64]
in_channels = [512, 1024, 2048]      # backbone 只给 3 张：C3 / C4 / C5
hidden_dim  = 256

shapes = [(math.ceil(H / s), math.ceil(W / s)) for s in strides]
print(shapes)          # [(100, 134), (50, 67), (25, 34), (13, 17)]

feats = [torch.randn(1, c, h, w) for c, (h, w) in zip(in_channels, shapes[:3])]

# input_proj：3 个 1×1 conv（C3/C4/C5） + 1 个 3×3 stride2 conv（造 C6）
input_proj = nn.ModuleList(
    [nn.Sequential(nn.Conv2d(c, hidden_dim, 1), nn.GroupNorm(32, hidden_dim)) for c in in_channels]
    + [nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
                     nn.GroupNorm(32, hidden_dim))])

srcs = [p(f) for p, f in zip(input_proj[:3], feats)]         # 3 张 [1,256,H_l,W_l]
srcs.append(input_proj[3](srcs[2]))                          # ★ C6 来自【投影后】的 C5
print(tuple(srcs[3].shape))                                  # (1, 256, 13, 17) ✓ 对上 shapes[3]

# 位置编码 + scale-level embedding
level_embed = nn.Parameter(torch.randn(len(shapes), 256))
pos_embeds  = [torch.randn(1, 256, h, w) for h, w in shapes]
srcs = [s.flatten(2).permute(2, 0, 1) + (p.flatten(2).permute(2, 0, 1)
                                        + level_embed[l].view(1, 1, -1))
        for l, (s, p) in enumerate(zip(srcs, pos_embeds))]

src = torch.cat(srcs, dim=0)                                 # [17821, 1, 256]
print(src.shape)       # torch.Size([17821, 1, 256])
```

### 8.2 Encoder / Decoder 接线

```python
from Deformable_Attention_impl import MSDeformAttn   # 见另一篇笔记的实现

enc_attn = MSDeformAttn(256, n_heads=8, n_levels=4, n_points=4)
dec_attn = MSDeformAttn(256, n_heads=8, n_levels=4, n_points=4)

# ── Encoder：query = value = src，参考点 = 自己位置 ──
memory = enc_attn(src[:, 0], src[:, 0], shapes)              # [17821, 256]

# ── Decoder ──
num_queries, d_model = 300, 256
query_embed = nn.Embedding(num_queries, d_model).weight       # [300, 256]
tgt         = torch.zeros_like(query_embed)                   # 同 DETR：输入全 0
ref_linear  = nn.Linear(d_model, 2)
ref_points  = ref_linear(query_embed).sigmoid()               # [300, 2]  ★ 新增的空间先验

# ① 自注意力仍是 dense：300×300，很便宜
# ② 交叉注意力换成可变形注意力：q 是 query，v 是 17821 个 token，ref 是上面预测的坐标
out = dec_attn(tgt, memory, shapes, ref_points=ref_points)    # [300, 256]

class_embed = nn.Linear(256, 91 + 1)
bbox_embed  = nn.Sequential(nn.Linear(256, 256), nn.ReLU(),
                            nn.Linear(256, 256), nn.ReLU(),
                            nn.Linear(256, 4))
pred_logits = class_embed(out)                                # [300, 92]
pred_boxes  = bbox_embed(out).sigmoid()                       # [300, 4]
print(pred_logits.shape, pred_boxes.shape)
# torch.Size([300, 92]) torch.Size([300, 4])
```

### 8.3 维度自检清单

```python
assert src.shape[0] == 17821                                   # 13400+3350+850+221
assert int(sum(h * w for h, w in shapes)) == src.shape[0]
assert memory.shape == src[:, 0].shape                          # encoder 不改变 token 数
assert ref_points.shape == (300, 2) and ref_points.min() >= 0 and ref_points.max() <= 1
```

---

## 9. 速记卡

- **一句话**：DETR 的骨架一动不动，只把 **encoder 自注意力 + decoder 交叉注意力** 换成**多尺度可变形注意力**，于是**终于能加多尺度特征**了；
- **为什么必须换**：全注意力 + 多尺度 = `17821²×8 ≈ 25.4 亿` 权重、fp32 **≈10 GB/层**，跑不起来。换成每 query 采样 `L×K=16` 点后降到 **228 万**（省约 1100 倍）；
- **四个尺度**：C3/C4/C5（stride 8/16/32）+ **C6**（C5 上 `3×3, s2` 卷积，stride 64），各过 1×1 conv 到 256 通道；`800×1066 → 100×134/50×67/25×34/13×17 = **17821** token`；
- **参考点的两种身份**：encoder 里 = **token 自己的位置**（固定网格）；decoder 里 = **`sigmoid(Linear(query_embed))` 预测的 (x,y)**（可学习空间先验）。这是和 DETR 最大的语义差别；
- **不改的**：decoder 自注意力（300×300，负责 query 间分工）、匈牙利匹配、集合损失、无 anchor、**无 NMS**、辅助损失；
- **新加的小件**：`scale-level embedding`（区分层级）、focal loss、object query 100 → **300**、迭代框精修、两阶段；
- **成绩单**：50 epoch 达到 43.8 AP（> DETR-DC5 训 500 epoch 的 43.3），`AP_S` 20.5 → 26.4 → 28.8，训练成本 2000 → 325 GPU 小时。**代价**是 FLOPs 86G → 173G、推理 28 → 19 FPS：它买的是"训练便宜 + 小目标好"，不是"推理快"；
- **演进位置**：DETR → **Deformable DETR（本篇）** → DAB-DETR（4D anchor query）→ DN-DETR（去噪）→ DINO → RT-DETR（实时）。
