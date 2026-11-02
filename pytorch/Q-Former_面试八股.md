# Q-Former 面试八股：从结构、维度到 BLIP-2

> **一句话记忆**：Q-Former 是一个用少量可学习 Query 从冻结视觉编码器输出中提取视觉信息的轻量 Transformer。它把大量视觉 token 压缩成固定数量的 query 表示，再投影给大语言模型。

---

## 1. Q-Former 解决什么问题？

视觉编码器通常输出很多视觉 token。例如对 `224×224` 图片使用 `14×14` 的 patch，得到 `16×16=256` 个 patch token，再加一个 CLS token，共 `257` 个 token。注意这里 `14×14` 指每个 patch 的像素大小；如果是 `16×16` 的 patch，则得到 `14×14+1=197` 个 token。

如果把所有视觉 token 直接送入 LLM，会有三个问题：

1. **序列太长**：视觉 token 数量大，会占用 LLM 上下文窗口。
2. **特征维度不匹配**：视觉编码器维度可能是 `1408`，LLM 的输入 embedding 维度可能是 `4096`。
3. **视觉信息和语言模型之间缺少语义接口**：即使线性投影对齐了维度，也需要通过训练让视觉表示能被 LLM 利用。

Q-Former 充当视觉编码器和 LLM 之间的桥接器：

```text
图片
  │
  ▼
冻结的视觉编码器
  │  [B, Nv, Dv]
  ▼
Q-Former + Nq 个可学习 Query
  │  [B, Nq, Dq]
  ▼
线性投影
  │  [B, Nq, Dllm]
  ▼
冻结的 LLM
```

它让固定数量的 Query 主动去“询问”视觉特征，提取与语言有关的信息。原始 BLIP-2 在第二阶段提取 Query 时不输入问题文本，因此这些 Query 不是随每个用户问题动态生成的；InstructBLIP 等后续方法才进一步引入指令条件。

---

## 2. 贯穿全文的维度例子

以下用 BLIP-2 中常见的配置来说明：

| 符号 | 含义 | 示例值 |
|---|---|---:|
| `B` | batch size | 2 |
| `H, W` | 输入图像大小 | 224, 224 |
| `P` | 每个视觉 patch 的大小 | 14 |
| `Nv` | 视觉 token 数量 | 257 |
| `Dv` | 视觉编码器输出维度 | 1408 |
| `Nq` | Query 数量 | 32 |
| `Dq` | Q-Former 隐藏维度 | 768 |
| `L` | 示例文本 token 长度，实际会受 tokenizer 和配置影响 | 32 |
| `Dc` | 阶段一对比学习的投影维度 | 256 |
| `Dllm` | LLM 输入 embedding 维度 | 4096 |

这里采用 EVA-CLIP ViT-g/14 的视觉维度作示例，LLM 侧以 OPT-2.7B 的 `4096` 维作示例；不是所有 BLIP-2 配置都相同。

因此视觉编码器输出为：

```text
image_embeds: [B, Nv, Dv] = [2, 257, 1408]
```

Q-Former 最终只输出：

```text
query_output: [B, Nq, Dq] = [2, 32, 768]
```

这就是视觉 token 压缩：

```text
[2, 257, 1408] → [2, 32, 768]
```

---

## 3. Q-Former 的核心组件

Q-Former 可以理解为一个 BERT-like Transformer，但加入了专门访问视觉特征的 Cross-Attention。BLIP-2 使用 BERT-base 初始化已有权重，新加入的 Cross-Attention 层和 Query 参数需要训练。常见配置为 12 层、12 个 head、hidden size 768，Q-Former 约 1.88 亿参数；“轻量”是相对于它连接的视觉模型和 LLM 而言。

阶段一还包含文本分支：图像侧 Query 和文本 token 共享 Self-Attention 层，并根据任务通过不同 mask 控制交互。具体实现中 Query 与文本可以使用独立 FFN，因此不能把它理解成完全相同的一套计算路径。

主要包含：

```text
1. 可学习 Query Token
2. Query Self-Attention
3. Query 到视觉特征的 Cross-Attention
4. Transformer MLP、LayerNorm 和残差连接
```

在 BLIP-2 的典型设计中，Cross-Attention 不是每一层都插入，而是隔层插入，例如每隔一个 Transformer block 插入一次。这样可以降低计算量，同时让 Query 多次读取视觉信息。

---

## 4. 可学习 Query 是什么？

Q-Former 有一组和输入图片无关的可学习向量：

```python
self.query_tokens = nn.Parameter(
    torch.randn(1, num_query_tokens, hidden_size)
)
```

以 `Nq=32、Dq=768` 为例：

```text
query_tokens 参数: [1, 32, 768]
```

前向时扩展到 batch，所有样本共享同一组初始 Query 参数：

```text
[1, 32, 768]
→ expand(B, -1, -1)
[2, 32, 768]
```

这些 Query 不是图片中的 patch，也不是文本 token。它们是模型学习出来的信息槽位，可以关注不同视觉内容；但没有保证“第一个 Query 专门看物体、第二个专门看颜色”，也没有 Query 与目标框的一一对应监督。

面试回答可以说：

> Query 是一组固定数量的可学习 latent tokens。它们通过 Cross-Attention 从视觉 token 中主动读取信息，因此输出长度由 `Nq` 决定，与图片产生多少 patch token 无关。

---

## 5. Query Self-Attention 的维度变化

假设 Query 已经扩展到 batch：

```text
Q_in: [B, Nq, Dq] = [2, 32, 768]
```

在仅输入 Query 的视觉提取路径中，Self-Attention 只在 Query 之间交互。阶段一的 ITM 等任务还允许 Query 与文本 token 交互，详见后文的 mask 表。

这里先看只有 Query 的情况：

```text
Q/K/V: [2, 32, 768]
```

假设有 `12` 个 attention heads：

```text
每个 head 维度 = 768 / 12 = 64
拆头后：        [B, 12, 32, 64]
```

计算注意力分数：

```text
Q @ K^T
[B, 12, 32, 64] @ [B, 12, 64, 32]
= [B, 12, 32, 32]
```

这里的 `[32, 32]` 表示 32 个 Query 两两之间的关系。

再乘以 V：

```text
[B, 12, 32, 32] @ [B, 12, 32, 64]
= [B, 12, 32, 64]
```

合并多个 head：

```text
[B, 12, 32, 64]
→ [B, 32, 768]
```

Self-Attention 的作用是让不同 Query 之间交换信息，避免每个 Query 完全独立工作。

---

## 6. Query 到图像的 Cross-Attention

这是 Q-Former 最关键的地方。

Query 作为 **Query**，视觉编码器输出作为 **Key/Value**：

```text
query hidden states: [B, Nq, Dq] = [2, 32, 768]
image_embeds:        [B, Nv, Dv] = [2, 257, 1408]
```

由于 Query 维度是 `768`，视觉特征维度是 `1408`，Cross-Attention 内部需要投影：

```text
Q = Linear(768 → 768)
K = Linear(1408 → 768)
V = Linear(1408 → 768)
```

投影后：

```text
Q: [B, 32, 768]
K: [B, 257, 768]
V: [B, 257, 768]
```

拆成 12 个 head：

```text
Q: [B, 12, 32, 64]
K: [B, 12, 257, 64]
V: [B, 12, 257, 64]
```

计算 Cross-Attention 分数：

```text
Q @ K^T
[B, 12, 32, 64] @ [B, 12, 64, 257]
= [B, 12, 32, 257]
```

这个形状非常重要：

```text
[32, 257]
```

表示每个 Query 对 257 个视觉 token 的关注权重。

再加权求和：

```text
[B, 12, 32, 257] @ [B, 12, 257, 64]
= [B, 12, 32, 64]
```

合并 heads 后：

```text
[B, 12, 32, 64]
→ [B, 32, 768]
```

所以 Cross-Attention 前后 Query 的序列长度不变：

```text
[B, 32, 768] → [B, 32, 768]
```

变化的是 Query 内容，它已经融合了视觉信息。

---

## 7. 为什么是 Query 读取图像，而不是图像读取 Query？

标准 Cross-Attention 中，谁提供 Q，谁就是主动发起查询的一方。

Q-Former 的设计是：

```text
Q = Query Token
K/V = Image Features
```

因此每个 Query 都在问视觉编码器：

> “和我当前需要的信息相关的视觉 token 是哪些？”

如果反过来使用：

```text
Q = Image Features
K/V = Query Token
```

输出长度会变成 `Nv=257`，无法直接得到固定的 32 个视觉摘要，而且信息瓶颈的位置也发生了变化。

Q-Former 选择 Query 作为 Q，是为了让输出长度固定为 `Nq`，实现可控的信息压缩。

---

## 8. Q-Former 的一次前向过程

以一组图片为例：

```text
输入图片：
[B, 3, 224, 224]
```

### 第一步：视觉编码器

```text
image = vision_encoder(images)
[B, 3, 224, 224]
→ [B, 257, 1408]
```

视觉编码器通常被冻结：

```python
with torch.no_grad():
    image_embeds = vision_encoder(images)
```

### 第二步：准备 Query

```text
query_tokens: [1, 32, 768]
expand batch 后： [B, 32, 768]
```

### 第三步：Self-Attention

```text
[B, 32, 768]
→ [B, 32, 768]
```

### 第四步：Cross-Attention

```text
Q: [B, 32, 768]
K/V 来源于 image_embeds: [B, 257, 1408]
输出: [B, 32, 768]
```

### 第五步：取 Query 输出

```text
last_hidden_state: [B, 32, 768]
```

这些 Query 输出就是 Q-Former 的视觉摘要。

### 第六步：连接 LLM

通过线性层把 Q-Former 的维度映射到 LLM 维度：

```text
Linear(768 → 4096)

[B, 32, 768]
→ [B, 32, 4096]
```

这 32 个向量可以作为 LLM 的视觉 prefix 或 soft prompt。

---

## 9. Q-Former 如何压缩视觉信息？

它不是简单的平均池化，也不是随机采样，而是一个可学习的注意力池化过程。

平均池化：

```text
[B, 257, 1408]
→ 对 257 个 token 求平均
[B, 1408]
```

Q-Former：

```text
[B, 257, 1408]
→ 32 个 Query 分别读取视觉 token
[B, 32, 768]
```

每个 Query 有独立的注意力分布：

```text
逐 head 的 attention map: [B, heads, Nq, Nv]
对 head 求平均后：         [B, Nq, Nv] = [2, 32, 257]
```

因此不同 Query 可以提取不同的视觉子空间。`Nq=32` 是一个信息瓶颈：模型必须用 32 个向量概括一张图片。

---

## 10. BLIP-2 中的两阶段训练

### 阶段一：视觉表示学习

目标是训练 Q-Former，让它从冻结的视觉编码器中提取和文本相关的视觉信息。

典型设置：

```text
冻结：视觉编码器
训练：Q-Former、Query Token、相关投影层
```

阶段一通常包含三个图文目标：

#### 10.1 Image-Text Contrastive Learning（ITC）

让图像表示和文本表示对齐。

**关键区别：不会先把 32 个 Query 平均成一个图像向量。**

Query 与文本分别编码，Self-Attention 不允许两者互相看，防止对比学习时提前泄漏配对文本。图像侧保留全部 Query，文本侧取 `[CLS]` 表示，再投影到对比空间并归一化。

```text
image query features: [B, Nq, 768]
→ Linear(768 → 256) + L2 normalize
image features:       [B, Nq, 256]

text [CLS]:           [B, 768]
→ Linear(768 → 256) + L2 normalize
text features:        [B, 256]
```

每张图片的每个 Query 都与候选文本算点积，再取最大的 Query 相似度作为图文对的分数：

```text
s(i, j, q) = image_features[i, q] · text_features[j]
S(i, j)    = max_q s(i, j, q)

所有候选图文对的逐 Query 分数：[B, B, Nq]
沿 Query 维取 max 后：          [B, B]
```

这里的分数张量轴顺序只是教学约定，源码可能先得到 `[B, Nq, B]` 再归约。

用温度参数缩放分数，计算 image-to-text 与 text-to-image 两个方向的交叉熵。多卡训练通常把特征跨卡收集；以上 `B` 暂按候选池的 batch 大小理解。

**面试追问：为什么取 max？** 一条描述可能只对应图片中部分内容，取 max 允许最相关的 Query 与文本匹配，不要求所有 Query 都表达同一语义。这也不代表剩余 Query 不会通过其他训练目标学习。

#### 10.2 Image-Text Matching（ITM）

判断一对图片和文本是否匹配。Query 和文本允许双向 Self-Attention，Query 再通过 Cross-Attention 访问视觉特征，文本间接获取图像信息。对各个融合后的 Query 使用共享二分类头，再对 Query 的 logits 求平均：

```text
[B, Nq, 768]
→ Linear(768 → 2)
[B, Nq, 2]
→ mean(dim=1)
[B, 2]
```

输出对应“匹配 / 不匹配”两类：

```text
[B, 2]
```

ITC 判断独立编码后的语义接近程度，ITM 在图文融合之后做更细致的匹配判断。训练还可利用 ITC 分数选取难负样本，让模型区分外观相似但描述不匹配的图文对。

#### 10.3 Image-grounded Text Generation（ITG）

给定图片，让 Q-Former 的文本生成分支预测描述。这里的注意力是多模态 causal mask：

- Query 可以互相看，并通过 Cross-Attention 读取图片，但不能看描述文本；
- 文本位置可以看所有 Query 和前面的文本 token，不能看未来文本；
- 文本从 Query 获得视觉信息，而不是直接在视觉 token 上做 Cross-Attention。

```text
视觉 Query：       [B, Nq, 768]
文本 hidden state：[B, L, 768]
词表预测 logits： [B, L, Vq]
```

`Vq` 是 Q-Former 文本生成头使用的词表大小，不是后端 LLM 的词表。训练使用 shifted next-token cross-entropy。阶段一的 Q-Former 确实参与文本生成；第二阶段最终的回答则由外部 LLM 生成。

三个目标在同一个阶段联合优化，并不是先训练完 ITC、再分别训练 ITM 和 ITG。

**三种 mask 对照表：**

| 阶段一目标 | Query 看 Query | Query 看文本 | 文本看 Query | 文本看文本 |
|---|---|---|---|---|
| ITC：对比 | 可以 | 不可以 | 不可以 | 双向，屏蔽 padding |
| ITM：匹配 | 可以 | 可以 | 可以 | 双向，屏蔽 padding |
| ITG：生成 | 可以 | 不可以 | 可以 | 因果，只看过去及当前输入位置 |

这里“看”指 Self-Attention 是否允许访问对应位置；视觉 token 始终通过 Query 的 Cross-Attention 提供信息。

阶段一的关键结果是：Query 学会提取对语言有用的视觉信息，而不是只保留低层视觉纹理。

### 阶段二：连接冻结的 LLM

把 Q-Former 输出接到 LLM：

```text
Q-Former output: [B, 32, 768]
Linear projection: [B, 32, 4096]
LLM input embedding: [B, 32, 4096]
```

典型设置是：

```text
冻结：视觉编码器、LLM
继续训练：Q-Former、Query Token
新增并训练：Q-Former 到 LLM 的线性投影层
```

**注意：原始 BLIP-2 第二阶段不只训练投影层，Q-Former 也继续更新。** 阶段一学到的表示需要进一步适配冻结 LLM 的输入空间。

以 decoder-only 的 OPT 为例，将投影后的视觉 embedding 与文本 embedding 沿序列维拼接：

```text
visual prefix: [B, 32, 4096]
text embeds:   [B, L, 4096]
拼接结果：      [B, 32 + L, 4096]
```

监督目标是文本的自回归生成损失，视觉 prefix 和 padding 对应的 label 设为 `-100`，不直接预测视觉向量。只有文本预测位置参与 loss。

如果后端是 encoder-decoder 的 Flan-T5，则将视觉前缀与输入文本送给 encoder，由 decoder 预测目标文本；不能把所有 BLIP-2 都说成 decoder-only 路线。

**为什么冻结 LLM 还能训练桥接器？** 冻结是令 LLM 参数不更新，不是切断计算图。梯度仍通过 LLM 对输入 embedding 的导数传回投影层和 Q-Former。若给整个 LLM 前向套上 `torch.no_grad()`，桥接器就收不到这部分梯度。

视觉塔通常可以使用 `no_grad()`，因为不需要对输入图片求梯度；视觉塔后面的 LayerNorm 是否更新等细节，以具体实现为准。

---

## 11. Q-Former 和直接投影有什么取舍？

直接投影可以写成：

```text
image feature [B, 257, 1408]
→ Linear(1408 → 4096)
[B, 257, 4096]
```

问题是：

- token 数量仍然是 257；
- 逐 token 的 Linear/MLP 不额外引入跨 token 注意力，但输入的 ViT 特征本身已经经过全局交互；
- 没有显式的信息筛选机制；
- 视觉 token 可能包含很多 LLM 不需要的细节。

Q-Former 则是：

```text
[B, 257, 1408]
→ Cross-Attention 信息筛选
→ [B, 32, 768]
→ [B, 32, 4096]
```

它同时完成了：

1. 视觉 token 压缩；
2. 视觉信息选择；
3. 图像和文本语义对齐；
4. 视觉维度到 LLM 维度的桥接准备。

这不意味着 Q-Former 在所有任务上都优于 MLP。固定 32 个 Query 可能丢失小字、数量或空间关系等细节；直接保留较多视觉 token 的投影方案结构更简单，也可能保留更多细粒度信息。效果还取决于训练数据、视觉分辨率、LLM 和指令微调，不能只凭连接器结构下结论。

---

## 12. Q-Former 和 Perceiver Resampler 的区别

两者都有固定数量的可学习 Latent/Query，从长序列中提取固定长度表示，因此思想相近。

| 对比项 | Q-Former | Perceiver Resampler |
|---|---|---|
| 典型模型 | BLIP-2 | Flamingo 等模型 |
| 核心结构 | BERT-like Transformer | Latent Query + Cross-Attention |
| 是否处理文本 | 可以同时做图文任务 | 主要作为视觉重采样器 |
| 训练目标 | 对比、匹配、生成 | 依模型整体目标而定 |
| 输出 | 固定数量 Query 表示 | 固定数量 Latent 表示 |

面试时不要简单说两者完全一样。更准确的说法是：**都使用固定数量的可学习 latent 从长视觉序列中读取信息，但 Q-Former 还承担了图文预训练和语言对齐功能。**

---

## 13. 高频面试问题与标准回答

### Q1：Q-Former 是什么？

**答：** Q-Former 是 BLIP-2 中连接冻结视觉编码器和冻结 LLM 的轻量 Transformer。它使用固定数量的可学习 Query，通过 Cross-Attention 从视觉 token 中提取信息，将大量视觉特征压缩成少量、语言相关的视觉表示，再投影到 LLM 的 embedding 空间。

### Q2：Q-Former 的 Query 是从哪里来的？

**答：** Query 是模型参数，不是从图片中切出来的 patch，也不是文本 token。它们初始化为可学习向量，形状通常是 `[1, Nq, Dq]`，扩展到 batch 后与视觉特征做 Cross-Attention。

### Q3：Q-Former 中 Cross-Attention 的 Q、K、V 分别是什么？

**答：** Query token 产生 Q，视觉编码器输出产生 K 和 V。这样输出长度由 Query 数量 `Nq` 决定，能够把任意长度的视觉 token 压缩成固定长度的视觉摘要。

### Q4：Q-Former 的输出维度是什么？

**答：** 输出通常是 `[B, Nq, Dq]`。例如 batch 为 2、Query 数量为 32、隐藏维度为 768，则输出为 `[2, 32, 768]`。之后用线性层映射到 LLM 的维度，例如 `[2, 32, 4096]`。

### Q5：为什么 Q-Former 输出 32 个 Query，而不是一个向量？

**答：** 一个向量的信息容量太小，难以表达复杂场景。多个 Query 可以分别关注不同物体、属性和关系，同时仍然把视觉序列压缩到固定长度。Query 数量是信息容量和计算开销之间的折中。

### Q6：Q-Former 是如何减少计算量的？

**答：** 视觉提取时 Query Self-Attention 的注意力计算约为 `O(Nq²·Dq)`，Cross-Attention 的注意力计算约为 `O(Nq·Nv·Dq)`；这些只计分数与加权求和，还不包括投影、FFN 和视觉编码器。更直接的收益是 LLM 只接收 32 个视觉前缀，减少后端的视觉输入长度。不能把整模型复杂度只写成 `O(Nq·Nv)`。

### Q7：Q-Former 和普通 Transformer 有什么区别？

**答：** Q-Former 保留了 Transformer 的 Self-Attention、MLP、LayerNorm 和残差结构，但额外插入了 Query 到视觉特征的 Cross-Attention，并使用可学习 Query 作为视觉信息读取接口。

### Q8：为什么视觉编码器和 LLM 要冻结？

**答：** 冻结可以显著降低训练显存和计算成本，避免大模型在中小规模图文数据上灾难性遗忘。Q-Former 和连接投影层承担模态桥接任务，只用较少参数就能把两个预训练模型连接起来。

### Q9：Q-Former 第二阶段训练哪些参数？

**答：** 原始 BLIP-2 第二阶段冻结视觉编码器和 LLM，继续训练 Q-Former、可学习 Query，以及新增的 LLM 投影层。冻结 LLM 参数仍然允许梯度从生成损失传回视觉前缀，不能把整个 LLM 前向放进 `no_grad()`。

### Q10：为什么 Q-Former 的输出可以直接作为 LLM 的输入？

**答：** Q-Former 输出先经过线性投影，使最后一维等于 LLM 的 embedding 维度。投影后的 `[B, Nq, Dllm]` 可以和文本 token embedding 在序列维拼接，或者作为视觉 prefix 输入 LLM。

### Q11：Q-Former 会生成文字吗？

**答：** 要分阶段：阶段一的 ITG 中，Q-Former 使用文本分支和语言建模头参与生成；阶段二连接外部 LLM 时，Q-Former 主要输出视觉条件向量，由后端 LLM 生成最终文字。不能绝对地说 Q-Former 从来不生成文本。

### Q12：Q-Former 的 Query 数量越多越好吗？

**答：** 不一定。Query 越多，信息容量越大，但 Cross-Attention 计算量、LLM 输入长度和显存也会增加。Query 太少可能丢失细节，太多则削弱压缩优势，需要根据任务和模型上下文长度选择。

### Q13：Q-Former 能处理任意大小的图片吗？

**答：** 视觉编码器通常要求固定输入尺寸，或者通过插值位置编码支持不同尺寸。只要视觉编码器输出视觉 token，Q-Former 理论上可以处理不同的 `Nv`，因为 Query 数量 `Nq` 是固定的，Cross-Attention 支持可变的 Key/Value 长度。

### Q14：Q-Former 的注意力矩阵 `[Nq, Nv]` 表示什么？

**答：** 第 `i` 个 Query 对第 `j` 个视觉 token 的关注程度。它可以用于理解某个 Query 在提取哪些图像区域，但注意力权重不一定等价于严格的因果解释或目标检测框。

---

## 14. 源码阅读时最容易看错的地方

### 14.1 `query_tokens` 的 batch 维

参数本身通常是：

```text
[1, Nq, Dq]
```

不是 `[B, Nq, Dq]`。前向时需要通过 `expand` 或其他方式扩展到 batch。

### 14.2 Cross-Attention 的 `encoder_hidden_states`

在 Hugging Face 风格实现中，经常看到：

```python
query_output = self.bert(
    query_embeds=query_tokens,
    encoder_hidden_states=image_embeds,
    encoder_attention_mask=image_atts,
)
```

这里 `query_embeds` 是 Q-Former 的 Query，`encoder_hidden_states` 是视觉特征，也就是 Cross-Attention 的 K/V 来源。

### 14.3 `attention_mask` 的用途不同

- 文本 attention mask：屏蔽 padding token；
- causal mask：阻止看到未来文本 token；
- image attention mask：通常标记哪些视觉 token 有效；
- query mask：标记 Query 是否参与注意力。

不要把 padding mask 和 causal mask 混为一谈。

### 14.4 输出的 `last_hidden_state` 不等于 pooled feature

Q-Former 输出通常是：

```text
last_hidden_state: [B, Nq, Dq]
```

它不是 `[B, Dq]` 的单一向量。要不要 pooling，取决于具体下游任务。

### 14.5 `repeat` 和 `expand` 的区别

Query 扩展 batch 时，`expand` 通常只创建视图，不复制底层数据；`repeat` 会复制数据。两者形状相同，但显存和计算图行为可能不同。

### 14.6 区分阶段一的生成头与阶段二的视觉桥接器

阶段一 ITG 使用 Q-Former 的文本生成头；阶段二通常移除不再使用的文本 embedding、语言建模头或文本侧 FFN，只保留视觉提取路径，由外部 LLM 输出最终文字。阅读源码时先确认当前类对应哪个阶段。

### 14.7 `requires_grad=False` 不等于 `eval()` 或 `no_grad()`

- `requires_grad=False`：冻结参数；仍可能保留对输入的梯度。
- `eval()`：改变 dropout 等模块的训练行为，不负责冻结参数。
- `torch.no_grad()`：当前范围不记录反向图；用于冻结 LLM 的训练前向会阻断对桥接器的梯度。

冻结视觉塔经常同时使用前两者；冻结 LLM 则需要保留从 loss 到视觉 prefix 的梯度路径。

---

## 15. 一页纸背诵版

```text
Q-Former = Learnable Queries + Self-Attention + Vision Cross-Attention

视觉编码器：
[B, 3, 224, 224]
→ [B, Nv, Dv]
例如 [2, 257, 1408]

可学习 Query：
[1, Nq, Dq]
→ expand batch
[B, Nq, Dq]
例如 [2, 32, 768]

Cross-Attention：
Q:   [B, 32, 768]
K/V: [B, 257, 1408]
内部投影到相同 attention 维度
注意力分数：[B, heads, 32, 257]
输出：      [B, 32, 768]

连接 LLM：
[B, 32, 768]
→ Linear(768 → 4096)
[B, 32, 4096]
→ 作为视觉 prefix 送入 LLM

BLIP-2：
阶段一：冻结视觉编码器，训练 Q-Former、Query 和任务头
        ITC：隔离图文，逐 Query 相似度取 max
        ITM：双向图文交互，二分类
        ITG：Query 不看文本，文本看 Query 与历史文本
阶段二：冻结视觉编码器和 LLM
        继续训练 Q-Former、Query，训练新增投影层
        用文本生成损失适配 LLM
```

---

## 16. 面试中如何用 30 秒回答“Q-Former 是什么”？

> Q-Former 是 BLIP-2 的视觉语言桥接模块，它用固定数量的可学习 Query 通过 Cross-Attention 从冻结视觉编码器的长序列特征中提取语言相关信息，把 `[B, Nv, Dv]` 压缩成 `[B, Nq, Dq]`，再投影到 LLM 的 embedding 维度，从而以较低的训练成本连接视觉模型和大语言模型。

如果面试官继续问训练，再补一句：

> 第一阶段冻结视觉塔，用图文对比、匹配和生成三个目标训练 Q-Former；第二阶段冻结视觉塔与 LLM，继续训练 Q-Former 和连接投影，通过 LLM 的语言生成损失完成适配。

---

## 17. 和 CLIP、LLaVA、DETR、InstructBLIP 怎么区分？

### 17.1 Q-Former 和 CLIP

| 维度 | CLIP | BLIP-2 的 Q-Former |
|---|---|---|
| 核心用途 | 学习图文共享空间，支持相似度匹配 | 提取视觉信息，桥接视觉塔与 LLM |
| 典型图像输出 | 全局向量 `[B, D]` | 多 Query 表示 `[B, Nq, Dq]` |
| 图文交互 | 原始双塔编码期间不交互 | 阶段一按 ITC/ITM/ITG 使用不同交互 mask |
| 训练目标 | 以双向图文对比为核心 | 阶段一包含对比、匹配、生成 |
| 与 LLM 的连接 | 原始 CLIP 没有这个组件 | 投影后作为 LLM 的视觉条件输入 |

CLIP 类模型也可以作为 BLIP-2 的视觉塔。二者处于系统中的不同位置，不是只能二选一。

### 17.2 Q-Former 和 LLaVA 的 MLP projector

```text
Q-Former：视觉 token → Query 读取并压缩 → 线性投影 → LLM
MLP：     视觉 token → 逐 token 变换 → LLM
```

常见 LLaVA 的投影器保留所选视觉序列的 token 数量；Q-Former 用固定 Query 数量形成瓶颈。MLP 简单，保留更多 token 时后端开销更高；Q-Former 引入额外注意力和预训练设计，压缩也可能损失细节。具体 LLaVA 版本还可能有其他视觉处理模块，比较时要说明版本与配置。

### 17.3 Q-Former 和 DETR 的 object queries

相似点：都使用可学习 Query，并通过注意力从视觉特征中读取信息。

区别：DETR 的 Query 最终预测类别和边框，由匈牙利匹配形成集合预测监督；Q-Former 的 Query 通常输出供语言任务使用的语义表示，没有检测框监督，也不要求每个 Query 对应一个物体。

因此不能说“32 个 Query 就代表图片里最多 32 个目标”。

### 17.4 Q-Former 和 InstructBLIP

原始 BLIP-2 的第二阶段，Q-Former 主要读取图片，用户问题在后面的 LLM 中输入。

InstructBLIP 进一步将指令输入 Q-Former，使视觉特征提取依赖指令：同一张图，问颜色和问数量时，可以提取不同的信息。原始 Query 参数仍然是学习出来的，但在指令交互后，输出表示会随指令改变。

**面试记忆：BLIP-2 学视觉到语言的桥，InstructBLIP 进一步让桥接过程受指令引导。**

---

## 18. 从源码认出这条计算链

下面是说明阶段二 decoder-only 路线的伪代码，用于识别数据流，不是一个完整可运行的模型：

```python
# 已冻结视觉塔和 LLM 参数；Q-Former、Query、llm_proj 保持可训练。
# ln_vision 是否可训练由实现决定，放在 no_grad 块之外。
with torch.no_grad():
    vision_states = vision_encoder(images)  # [B, Nv, 1408]

image_embeds = ln_vision(vision_states)     # [B, Nv, 1408]
queries = query_tokens.expand(B, -1, -1)   # [B, 32, 768]
image_mask = torch.ones(B, Nv, device=images.device, dtype=torch.long)

query_states = qformer(
    query_embeds=queries,
    encoder_hidden_states=image_embeds,    # Cross-Attention K/V 来源
    encoder_attention_mask=image_mask,
).last_hidden_state                       # [B, 32, 768]

visual_prefix = llm_proj(query_states)     # [B, 32, 4096]
text_embeds = llm.get_input_embeddings()(input_ids)  # [B, L, 4096]
inputs_embeds = torch.cat([visual_prefix, text_embeds], dim=1)
# [B, 32 + L, 4096]，实际拼接前保证 device 和 dtype 一致。

# 同样拼接 attention_mask；prefix 标签与 padding 标签设为 -100。
# 原始文本的其他位置保留 token id，由模型内部完成 next-token shift。
outputs = llm(
    inputs_embeds=inputs_embeds,
    attention_mask=combined_mask,          # [B, 32 + L]
    labels=combined_labels,                # [B, 32 + L]
)
loss = outputs.loss
loss.backward()  # 梯度经过冻结的 LLM，传回 llm_proj 和 Q-Former。
```

`visual_prefix` 是连续向量，不需要先变成某个词表里的 token id。它和文字 embedding 的最后一维相同，但每个向量并不对应一个可读单词。

---

## 19. 必须避开的八个口误

1. **“Q-Former 第二阶段只训练 Linear。”** 原始 BLIP-2 还继续训练 Q-Former 和 Query。
2. **“Query 是根据每张图片初始化的。”** 初始参数跨图片共享，视觉交互后的输出随图片变化。
3. **“32 个 Query 对应 32 个物体。”** 没有这种一一对应约束。
4. **“三个目标都使用同一种 mask。”** ITC 隔离图文，ITM 双向交互，ITG 使用多模态因果约束。
5. **“ITC 先平均 Query 再算相似度。”** 原始设计是逐 Query 计算，再取最大相似度。
6. **“冻结 LLM 就不用反向传播经过它。”** 参数不更新，但仍需向输入前缀传梯度。
7. **“原始 BLIP-2 的 Query 已经根据用户问题提取视觉信息。”** 第二阶段通常不向 Q-Former 输入问题；这是理解 InstructBLIP 改进的关键。
8. **“只要维度对上，LLM 就理解图像。”** 维度对齐只是接口要求，语义适配还需要训练目标与数据。

---

## 20. 原论文和官方源码阅读顺序

本笔记以 BLIP-2 原始设计和 Salesforce LAVIS 官方实现为依据；下面的代码链接指向仓库 `main`，后续版本可能变化。

1. [BLIP-2 原论文，arXiv:2301.12597](https://arxiv.org/abs/2301.12597)：重点看 Figure 2、Figure 3 和方法部分，理解模块、三种 mask 与两阶段训练。
2. [blip2.py](https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2.py)：看 `init_Qformer`，理解 BERT 初始化、视觉维度、Cross-Attention 频率和 Query 参数。
3. [Qformer.py](https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/Qformer.py)：看 Attention 与 Transformer layer，区分共享 Self-Attention、Query Cross-Attention 和两条 FFN 路径。
4. [blip2_qformer.py](https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_qformer.py)：看阶段一 `forward`，重点找 `sim_q2t.max`、`itm_head`、`loss_itc + loss_itm + loss_lm`。
5. [blip2_opt.py](https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_opt.py)：看阶段二 `forward`，重点找参数冻结、`opt_proj`、视觉文本拼接和 `-100` 标签。
6. [blip2_t5.py](https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_t5.py)：对比 encoder-decoder 路线，不要把 T5 当成 OPT。

如果只剩十分钟，优先读第 6 节 Cross-Attention、第 10 节训练和 mask、第 15 节背诵版，以及第 19 节口误清单。
