# TokenLearner 与 TopKTokenPruner 面试八股

> **一句话区分**：TokenLearner 不挑选原来的 token，而是学习若干空间注意力图，对所有视觉特征做加权池化，生成新的少量 token；TopKTokenPruner 为每个已有 token 打分，直接保留分数最高的 `K` 个 token。

---

## 1. 为什么需要 token 压缩？

视觉 Transformer 的 Self-Attention 复杂度近似为：

```text
O(N² · D)
```

其中：

- `N`：视觉 token 数量；
- `D`：token 的通道维度。

以一张 `224×224` 图片为例：

```text
patch size = 16
patch 数量 = 14 × 14 = 196
加 CLS token 后 = 197
```

如果是视频，假设有 `T=16` 帧：

```text
N = 16 × 196 = 3136
```

如果直接做全局 Self-Attention，注意力矩阵大小是：

```text
[3136, 3136]
```

把 token 压缩到 `M=8` 或 `K=64` 后，后续 Transformer 只处理少量 token，计算大幅下降：

```text
原始：O(N² · D)
压缩后：O(M² · D) 或 O(K² · D)
```

TokenLearner 和 TopKTokenPruner 都是在“视觉编码器之后、昂贵的 Transformer 层之前”减少 token 数量，但它们保留信息的方式不同。

---

## 2. 贯穿全文的维度例子

统一使用下面的图像特征：

| 符号 | 含义 | 示例值 |
|---|---|---:|
| `B` | batch size | 2 |
| `H, W` | 特征图空间尺寸 | 14, 14 |
| `N` | token 数量，`H×W` | 196 |
| `C` | token 通道维度 | 768 |
| `M` | TokenLearner 输出 token 数 | 8 |
| `K` | TopKTokenPruner 保留 token 数 | 32 |
| `S` | TokenLearner 注意力图数量 | 8，通常等于输出 token 数 |

输入可以写成两种等价形式：

```text
feature map: [B, H, W, C] = [2, 14, 14, 768]
token sequence: [B, N, C] = [2, 196, 768]
```

注意代码中可能使用 `[B, C, H, W]`，也可能使用 `[B, H, W, C]`。先确认布局，再判断 `flatten` 和 `transpose` 后的维度。

---

## 3. TokenLearner 是什么？

TokenLearner 来自论文：

> *TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?*，NeurIPS 2021。

它的目标是从高分辨率视觉特征中**学习少量、任务相关的新 token**。

输入：

```text
X: [B, H, W, C]
```

TokenLearner 生成 `S` 张空间注意力图：

```text
A: [B, S, H, W]
```

每张注意力图对应一个可学习的视觉聚合区域。对每一张图做加权池化：

```text
output: [B, S, C]
```

示例：

```text
[2, 14, 14, 768]
→ 8 张注意力图 [2, 8, 14, 14]
→ 加权池化
[2, 8, 768]
```

它不是从 196 个 token 中挑 8 个，而是把 196 个 token 按不同注意力分布混合，重新生成 8 个 token。

---

## 4. TokenLearner 的公式

先把空间维度展平：

```text
X_flat: [B, N, C]
A_flat: [B, S, N]
```

对第 `s` 个注意力图和第 `c` 个通道：

```text
z[b, s, c] = Σ(n=1..N) A[b, s, n] · X[b, n, c]
```

矩阵写法：

```text
A_flat @ X_flat
[B, S, N] @ [B, N, C]
= [B, S, C]
```

如果实现里先把 `A` 写成 `[B, N, S]`，则计算会变成：

```text
A_flat.transpose(1, 2) @ X_flat
[B, S, N] @ [B, N, C]
= [B, S, C]
```

### 4.1 是否一定要 Softmax？

不一定。TokenLearner 的实现可能使用：

- `sigmoid` 得到独立空间门控；
- `softmax(dim=空间维)` 得到归一化加权池化；
- `sigmoid` 后再做加权平均或缩放；
- 不同版本中加入 LayerNorm、MLP、depthwise convolution。

因此阅读源码时不要只凭“注意力图”三个字就假设它一定是标准 Transformer Softmax Attention。核心不变的是：**先生成空间权重，再按空间权重聚合特征。**

---

## 5. TokenLearner 的典型计算流程

### 第一步：输入视觉特征

```text
X: [B, H, W, C]
 = [2, 14, 14, 768]
```

### 第二步：生成空间注意力图

通过一个轻量网络得到 `S` 个 attention map：

```text
attention_net(X)
→ A: [B, H, W, S]
 = [2, 14, 14, 8]
```

再换轴：

```text
[B, 14, 14, 8]
→ [B, 8, 14, 14]
```

### 第三步：展平空间维度

```text
X: [B, 14, 14, 768]
→ [B, 196, 768]

A: [B, 8, 14, 14]
→ [B, 8, 196]
```

### 第四步：加权求和

```text
A @ X
[B, 8, 196] @ [B, 196, 768]
= [B, 8, 768]
```

### 第五步：送入后续 Transformer

```text
原始 token： [B, 196, 768]
压缩 token： [B, 8, 768]
```

注意 TokenLearner 通常保留通道维度 `C`，主要改变 token 数 `N → S`。

---

## 6. TokenLearner 的注意力图代表什么？

第 `s` 张图：

```text
A[:, s, :, :]  # [B, H, W]
```

表示第 `s` 个输出 token 从哪些空间位置聚合信息。

可以把它理解为一个软区域池化器：

```text
第 1 张图：可能关注主体
第 2 张图：可能关注背景或局部纹理
第 3 张图：可能关注运动区域
```

但不要把它说成硬编码的目标检测框，也不要声称每一张图一定对应一个固定语义。不同 Query/attention map 的含义是训练中形成的，并且可能随输入和任务变化。

TokenLearner 具有两个重要特点：

1. **软选择**：每个输出 token 通常融合很多输入位置，而不是只取一个位置。
2. **内容自适应**：权重由当前特征 `X` 生成，不同图片得到的空间权重可以不同。

---

## 7. TokenLearner 和普通平均池化的区别

平均池化：

```text
z[c] = 1/N · Σ_n X[n, c]
```

所有空间位置权重固定且相同。

TokenLearner：

```text
z[s, c] = Σ_n A[s, n] · X[n, c]
```

每个输出 token 有自己的空间权重，而且权重由输入内容决定。

因此 TokenLearner 可以产生多个不同的全局摘要：

```text
平均池化： [B, 196, 768] → [B, 1, 768]
TokenLearner：[B, 196, 768] → [B, 8, 768]
```

它的表达能力比单个平均池化更强，但也引入了额外的注意力图预测网络和训练目标依赖。

---

## 8. TopKTokenPruner 是什么？

`TopKTokenPruner` 不是所有论文都统一采用的标准类名。工程中通常用它泛指：

```text
1. 为每个输入 token 预测一个重要性分数；
2. 取分数最高的 K 个 token；
3. 保留这些 token 的原始特征；
4. 把剪枝后的序列交给后续 Transformer。
```

输入：

```text
X: [B, N, C] = [2, 196, 768]
```

重要性打分器：

```text
score_net(X)
→ scores: [B, N, 1]
→ squeeze
scores: [B, N]
 = [2, 196]
```

选择索引：

```python
topk_scores, topk_indices = scores.topk(K, dim=1)
```

形状：

```text
topk_scores:   [B, K] = [2, 32]
topk_indices:  [B, K] = [2, 32]
```

按索引收集原始 token：

```python
indices = topk_indices.unsqueeze(-1).expand(-1, -1, C)
selected = torch.gather(X, dim=1, index=indices)
```

输出：

```text
selected: [B, K, C] = [2, 32, 768]
```

它做的是**离散 token 保留**，不是加权重构新 token。

---

## 9. TopKTokenPruner 的一个教学版实现

下面代码展示核心数据流。真实工程还需要考虑位置编码、padding、训练策略和是否保留 CLS token。

```python
import torch
import torch.nn as nn


class TopKTokenPruner(nn.Module):
    """为每个 token 打分，并保留分数最高的 K 个 token。"""

    def __init__(self, dim: int, keep_tokens: int):
        super().__init__()
        self.keep_tokens = keep_tokens

        # 最简单的 token importance scorer。
        # [B, N, C] -> [B, N, 1]
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, N, C]
        b, n, c = x.shape
        k = min(self.keep_tokens, n)

        scores = self.score(x).squeeze(-1)  # [B, N]
        topk_scores, topk_indices = scores.topk(k, dim=1)

        # gather 需要 index 和 x 的维度数一致。
        gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, c)
        selected_tokens = torch.gather(x, dim=1, index=gather_index)

        return selected_tokens, topk_scores, topk_indices
```

维度跟踪：

```text
x                  [B, N, C]
score(x)           [B, N, 1]
squeeze            [B, N]
topk_indices       [B, K]
expand             [B, K, C]
gather(x, dim=1)   [B, K, C]
```

---

## 10. TopKTokenPruner 的位置编码问题

剪枝后 token 顺序通常按照 `topk` 返回的分数顺序排列，而不是原始空间顺序。这样会带来位置编码问题。

原始 token：

```text
index: 0, 1, 2, ..., 195
```

TopK 后可能变成：

```text
index: 17, 104, 3, 88, ...
```

常见处理方式：

### 10.1 同时 gather 位置编码

如果位置编码是显式的：

```python
selected_pos = torch.gather(pos, 1, gather_index)
selected_tokens = selected_tokens + selected_pos
```

其中：

```text
pos: [B, N, C]
selected_pos: [B, K, C]
```

### 10.2 保留原始索引，再由二维坐标重建位置

如果 token 来自 `H×W` 网格，先把一维索引还原成：

```python
row = index // W
col = index % W
```

再根据 `(row, col)` 查相对位置或二维位置编码。

### 10.3 使用 RoPE 或相对位置偏置

这类位置机制可以根据当前序列位置或显式坐标计算，但要确认剪枝后是否仍然保留原始坐标信息。

面试时要指出：**只 gather token、不处理位置编码，可能破坏视觉空间关系。**

---

## 11. CLS token 要不要参与 TopK？

视觉 Transformer 里常见：

```text
x = [CLS, patch_1, patch_2, ..., patch_N]
```

有三种做法：

### 做法一：CLS 永远保留

```text
patches: [B, N, C]
TopK(patches): [B, K, C]
concat(CLS, selected): [B, K+1, C]
```

这是比较常见的工程选择，因为 CLS 是全局汇总 token。

### 做法二：CLS 也参与竞争

直接在 `[B, N+1, C]` 上取 TopK。这样可能把 CLS 剪掉，除非任务明确允许。

### 做法三：不使用 CLS

有些 ViT 或视觉编码器使用平均池化或纯 patch token，剪枝时不需要特殊处理。

面试回答：

> 是否保留 CLS 取决于后续模型接口。若 CLS 承担全局表示，通常固定保留，再对 patch token 做 TopK；不能默认所有实现都把 CLS 放进竞争集合。

---

## 12. TopK 选择为什么有梯度问题？

`torch.topk` 返回离散索引。对索引本身不能做普通的连续梯度更新：

```text
scores → topk indices → gather
```

当两个 token 分数交换排序时，选择结果可能突然改变，训练目标对排序边界不平滑。

但是要区分两条梯度路径：

1. **被选中的 token 特征**仍然可以通过后续网络获得梯度；
2. **未被选中的 token**通常在这一次 forward 中收不到来自后续网络的 token 特征梯度；
3. 排序索引的选择本身不是一个普通可微函数。

常见解决方案：

- 训练前期使用 soft weighting，后期再 hard TopK；
- 用 softmax / sigmoid 门控代替硬选择；
- 使用 Gumbel-Softmax、Straight-Through Estimator 等近似；
- 对 scorer 使用额外的蒸馏、稀疏正则或辅助任务；
- 训练时保留更多 token，推理时再剪枝。

因此 TopKTokenPruner 不一定“完全不可训练”，但它的离散选择会让 scorer 的优化更困难。

---

## 13. TopK 分数从哪里来？

`TopKTokenPruner` 只是功能名，重要性分数可以有多种来源：

### 13.1 学习一个 scorer

```text
score_n = MLP(x_n)
```

优点是端到端学习；缺点是需要额外训练和稀疏约束。

### 13.2 使用 CLS 对 token 的注意力

从某个 Transformer 层取：

```text
attention[CLS, patch_n]
```

优点是无需额外 scorer；缺点是注意力权重不一定等于 token 重要性，而且依赖具体层和 head。

### 13.3 使用 token 范数

```python
score = x.norm(dim=-1)
```

简单、无需参数，但特征范数大不一定表示任务相关。

### 13.4 使用类别或文本查询相关性

如果已经有文本 embedding 或任务 query，可以计算：

```text
score_n = token_n · task_query
```

这样选择是任务条件化的，但需要额外的查询向量。

### 13.5 使用梯度或扰动重要性

例如根据梯度、遮挡后 loss 变化评估 token，但通常计算成本较高，不适合每次在线推理。

面试时要说：**TopK 的关键是排序机制，不是固定某一种 score 定义。**

---

## 14. TokenLearner 和 TopKTokenPruner 的核心区别

| 对比项 | TokenLearner | TopKTokenPruner |
|---|---|---|
| 输出来源 | 对所有 token 加权池化，生成新 token | 从原 token 中直接选择部分 token |
| 输出数量 | `S` 个 | `K` 个 |
| 是否保留原 token | 通常不直接保留 | 直接保留 |
| 选择方式 | 连续软权重 | 离散 TopK 索引 |
| 是否可微 | 权重聚合通常可微 | 索引选择存在非平滑问题 |
| 空间信息 | 通过注意力图软聚合，可能模糊位置 | 选中的 token 保留原位置，但需处理位置编码 |
| 信息压缩 | 融合多个位置，可能重构语义 | 丢弃未选 token，细节可能直接消失 |
| 结果顺序 | attention map / token slot 的固定顺序 | 常按分数排序，也可按原位置重新排序 |
| 典型输入输出 | `[B,N,C] → [B,S,C]` | `[B,N,C] → [B,K,C]` |
| 适合场景 | 学习任务相关摘要、视频长序列压缩 | 保留关键 patch、减少后续计算 |

记忆方法：

```text
TokenLearner = learn to aggregate
TopKPruner   = score and select
```

---

## 15. 一个具体的数值例子

假设一张图经过视觉编码器得到 6 个 token，每个 token 维度为 4：

```text
X: [1, 6, 4]
```

### 15.1 TokenLearner，输出 2 个 token

注意力权重：

```text
A: [1, 2, 6]
```

第一个输出 token：

```text
z_1 = 0.1*x_1 + 0.1*x_2 + 0.4*x_3
   + 0.2*x_4 + 0.1*x_5 + 0.1*x_6
```

第二个输出 token 使用另一张权重图：

```text
z_2 = 0.0*x_1 + 0.2*x_2 + 0.1*x_3
   + 0.0*x_4 + 0.6*x_5 + 0.1*x_6
```

结果：

```text
[1, 6, 4] → [1, 2, 4]
```

`z_1` 和 `z_2` 都是新的向量，不一定等于某个原始 token。

### 15.2 TopKTokenPruner，保留 2 个 token

分数：

```text
scores = [0.2, 0.9, 0.1, 0.8, 0.3, 0.4]
```

TopK 索引：

```text
indices = [1, 3]
```

结果：

```text
[X_2, X_4]
[1, 6, 4] → [1, 2, 4]
```

输出向量就是原来的 `X_2` 和 `X_4`，只是顺序可能按分数重新排列。

---

## 16. 计算复杂度怎么比较？

设输入 token 数 `N`，通道维度 `C`，输出 token 数 `M/K`。

### TokenLearner

注意力图生成网络通常是线性于空间 token 数的：

```text
打分/生成 attention map：约 O(N · C · S)
加权池化：               O(S · N · C)
```

随后 Transformer 对 `S` 个 token 做 Attention：

```text
O(S² · C)
```

具体常数取决于 attention map 网络、卷积核、是否有降维和实现版本。

### TopKTokenPruner

```text
scorer：约 O(N · C)
TopK：  通常约 O(N log K) 或由底层实现决定
gather：O(K · C)
```

之后 Transformer 处理 `K` 个 token：

```text
O(K² · C)
```

`TopK` 的理论复杂度不要死记成唯一公式，不同硬件和选择算法可能不同；面试时说“排序/选择加上线性打分，底层复杂度依实现而定”更稳妥。

---

## 17. TokenLearner 的优缺点

### 优点

1. 输出 token 数固定，便于接入后续 Transformer。
2. 软聚合可以融合多个空间区域，不容易只依赖一个 patch。
3. attention map 有一定可视化解释性。
4. 对视频可以在空间和时间维度上学习任务相关摘要。
5. 通过减少后续 token 数降低二次注意力开销。

### 缺点

1. 生成的新 token 可能丢失精确空间位置。
2. 注意力图可能塌缩到相似区域，导致多个 token 冗余。
3. 需要训练 attention map 网络和后续任务目标。
4. 输出 token 不一定对应可解释的单个物体。
5. 如果任务依赖小目标、文字或精确计数，过强压缩可能损失细节。

---

## 18. TopKTokenPruner 的优缺点

### 优点

1. 直接保留原 token，局部特征不需要重新混合。
2. 输出 token 数可控，适合做动态计算。
3. 选中的 token 可以追溯到原图 patch，空间定位较直接。
4. 结构简单，容易插入已有 ViT 或多模态模型。

### 缺点

1. 硬 TopK 有排序边界不连续问题。
2. 未选 token 的信息会直接丢失，可能漏掉分散的小目标。
3. 需要处理位置编码、CLS、padding 和 token 顺序。
4. score 不可靠时，早期错误剪枝会影响后续所有层。
5. 固定 `K` 可能不适合内容复杂度不同的图片。

---

## 19. 高频面试问题与标准回答

### Q1：TokenLearner 和普通 Attention Pooling 有什么关系？

**答：** TokenLearner 可以看作内容自适应的多槽位 Attention Pooling。它为每个输出 token 学习一张空间权重图，再对输入特征做加权聚合；区别在于它通常同时生成多个 token，并且权重由输入内容动态产生。

### Q2：TokenLearner 是不是从输入 token 里选出几个？

**答：** 不是。TokenLearner 通过注意力权重对所有输入位置做加权求和，生成新的 token；输出向量通常不等于任何一个原始 token。直接从原 token 中挑选是 TopK Pruning 的行为。

### Q3：TokenLearner 为什么可以减少 Transformer 的计算量？

**答：** 它把 `N` 个视觉 token 压缩成 `S` 个 token，后续 Self-Attention 从 `O(N²C)` 变成 `O(S²C)`。注意 TokenLearner 自身的 attention map 生成和加权池化也有开销，所以应该比较整个模块和后续网络的总成本。

### Q4：TokenLearner 的输出 token 数由什么决定？

**答：** 由 attention map 数 `S` 决定，通常在模块初始化时设定。输入图像的 patch 数可以变化，但只要 attention map 网络支持对应空间尺寸，输出长度仍由 `S` 控制。

### Q5：TokenLearner 的注意力图是不是标准 Transformer Attention？

**答：** 不一定。它通常是由卷积或 MLP 从输入特征预测空间权重，可能用 sigmoid 或 softmax；它不一定有标准 Self-Attention 中的 `QK^T`。应该根据源码确认权重生成方式。

### Q6：TopKTokenPruner 的 score 怎么得到？

**答：** 没有统一规定，可以用可学习 MLP、CLS 对 patch 的注意力、token 范数、任务 query 相似度或梯度重要性。`TopKTokenPruner` 描述的是“打分加选择”的流程，不规定唯一 scorer。

### Q7：TopK 操作可导吗？

**答：** 选中的 token 特征可以参与后续反向传播，但 TopK 返回的离散索引对排序选择本身不可按普通连续函数求导，未选 token 也通常收不到后续路径的特征梯度。需要 soft gate、Straight-Through 或其他近似方法改善 scorer 训练。

### Q8：TopKTokenPruner 和 token merging 有什么区别？

**答：** TopK 是选择，输出 token 是原 token 的子集；token merging 是合并，把多个相似 token 聚合成新 token。前者可能直接丢信息，后者会保留某种加权汇总，但需要定义相似度和合并策略。

### Q9：为什么 TopK 后要处理位置编码？

**答：** TopK 改变了 token 的集合和顺序。如果继续使用剪枝后序列位置 `0,1,...,K-1`，模型可能把原本相距很远的 patch 当成相邻 token。应同步 gather 原始位置编码、保留二维坐标或使用支持坐标的相对位置机制。

### Q10：TokenLearner 输出 8 个 token，能理解成 8 个物体吗？

**答：** 不能。8 个 token 是 8 个可学习聚合槽位，没有目标检测框或实例级一一对应监督。它们可能编码主体、背景、动作等任务相关信息，也可能存在冗余。

### Q11：为什么不用固定的平均池化？

**答：** 平均池化对所有位置使用相同权重，只产生一个固定的全局统计；TokenLearner 为多个输出槽位学习输入相关的空间权重，可以聚合不同区域和不同语义，表达能力更强。

### Q12：TokenLearner 和 Q-Former 的 Query 有什么相似与区别？

**答：** 都用少量槽位从长序列中提取信息。TokenLearner 通常在视觉特征图上生成空间注意力图并加权池化；Q-Former 使用可学习 Query 作为 Q，与视觉特征做 Transformer Cross-Attention，还承担图文对齐和连接 LLM 的任务。二者都能压缩 token，但结构和训练目标不同。

### Q13：什么时候更适合 TokenLearner，什么时候更适合 TopK？

**答：** 如果希望多个输出 token 融合分散区域、追求任务相关摘要，可以用 TokenLearner；如果希望保留原 patch、保留可追溯坐标、实现简单的动态裁剪，可以用 TopK。对于小目标密集、空间关系重要的任务，需要特别评估硬 TopK 的漏检风险。

### Q14：如何动态决定 K？

**答：** 不必固定 K，可以根据分数阈值、累计质量、熵或预算约束选择数量。例如保留累计 softmax 分数达到 `p` 的 token，或者训练一个停止策略。但动态 K 会增加 batch padding、部署和延迟控制的复杂度。

---

## 20. 源码阅读时的维度检查清单

看到 TokenLearner 或 TopKPruner，按下面顺序读：

```text
1. 输入布局是 [B, C, H, W] 还是 [B, H, W, C]？
2. N 是 H×W，还是包含 CLS / 时间维？
3. attention map / score 的 shape 是 [B, S, N] 还是 [B, N, S]？
4. 权重使用 sigmoid 还是 softmax？softmax 沿哪个维度？
5. 加权池化是否有除以权重和的归一化？
6. TopK 的 dim 是 token 维还是通道维？
7. gather 的 index 是否 expand 到最后的 C 维？
8. 剪枝后是否同步处理 pos embedding、CLS、padding 和 mask？
9. TopK 返回顺序是分数降序，还是重新按原空间索引排序？
10. 后续 Transformer 的序列长度是否真的变成 S/K？
```

最常见的两个 bug：

```python
# 错：沿通道维 topk
scores.topk(k, dim=-1)

# 对 [B, N] 的 score，应该沿 token 维
scores.topk(k, dim=1)
```

以及：

```python
# 错：index 只有 [B, K]，不能直接 gather [B, N, C]
selected = torch.gather(x, 1, topk_indices)

# 对 [B, N, C]，需要扩成 [B, K, C]
gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, C)
selected = torch.gather(x, 1, gather_index)
```

---

## 21. 一页纸背诵版

```text
TokenLearner：
输入 X: [B, N, C]
生成 S 张空间注意力图 A: [B, S, N]
加权池化：A @ X
[B, S, N] @ [B, N, C] = [B, S, C]
输出是新 token，通常可微，属于软聚合。

TopKTokenPruner：
输入 X: [B, N, C]
打分 score: [B, N]
topk index: [B, K]
gather 后：[B, K, C]
输出是原 token 子集，属于硬选择。

TokenLearner = learn to aggregate
TopKPruner   = score and select

共同目标：减少后续 Self-Attention 的 token 数量，降低 O(N²C) 计算。
主要风险：TokenLearner 可能模糊空间细节；TopK 可能硬丢 token、破坏位置关系。
```

---

## 22. 面试中的 30 秒回答

> TokenLearner 和 TopKTokenPruner 都是视觉 token 压缩模块。TokenLearner 从输入特征预测多张空间注意力图，对所有 token 做加权池化，生成少量新的任务相关 token，属于可微的软聚合；TopKTokenPruner 为每个已有 token 预测重要性分数，通过 `topk` 和 `gather` 直接保留分数最高的 `K` 个原 token，属于离散硬选择。前者表达能力更强但会重构空间信息，后者实现简单且坐标可追溯但存在不可平滑选择和漏掉细节的问题。两者都通过将 `N` 降到 `S/K`，减少后续 Transformer 的二次注意力开销。

---

## 23. 论文与源码阅读入口

1. [TokenLearner 原论文：What Can 8 Learned Tokens Do for Images and Videos?](https://arxiv.org/abs/2106.11297)
2. [Google Research Scenic TokenLearner 实现](https://github.com/google-research/scenic/tree/main/scenic/projects/token_learner)
3. `TopKTokenPruner` 不是统一的论文名称。阅读具体项目时，应优先确认：score 的定义、TopK 维度、是否保留 CLS、位置编码处理、训练时是否使用 soft/hard 两阶段。
4. 如果项目把它叫作 `TokenPruner`、`TokenSelector`、`DynamicTokenPruning` 或 `TokenImportance`，通常都可以沿着本笔记的“打分 → 选择 → gather → 位置处理”主线阅读。
