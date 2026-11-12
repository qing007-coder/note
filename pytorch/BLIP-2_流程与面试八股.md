# BLIP-2 流程与面试八股：架构重点

> **一句话记忆**：BLIP-2 用冻结的视觉编码器提取图像特征，用 Q-Former 的少量可学习 Query 从视觉 token 中抽取语言相关信息，再用线性投影把 Query 输出接入冻结的大语言模型。它的关键是“冻结两端大模型，只训练中间桥接器”，并不是把图像直接变成 caption 再喂给 LLM。
>
> **本笔记重点**：架构、数据流、注意力关系、维度变化、OPT/Flan-T5 接口和面试追问。训练阶段只保留理解架构所需的内容。

---

## 1. 先用 30 秒回答“BLIP-2 是什么？”

> BLIP-2 是一种视觉语言桥接架构。它冻结一个预训练视觉编码器和一个预训练语言模型，在中间加入轻量的 Q-Former。Q-Former 使用固定数量的可学习 Query，通过 Cross-Attention 从视觉编码器的长序列特征中提取少量、语言相关的视觉表示，再经过线性投影输入语言模型。这样可以复用两个大模型，降低端到端训练成本，同时把视觉信息转换成 LLM 能处理的条件 embedding。

架构拆成三个部分：

```text
Vision Encoder：图片 → 视觉 token
Q-Former：      视觉 token → 固定数量 Query 表示
LLM：           Query 表示 + 文本上下文 → 自回归文本 / encoder-decoder 文本
```

必须记住：

```text
BLIP-2 的核心瓶颈是 Q-Former，不是视觉塔和 LLM 之间直接拼接全部 patch。
```

---

## 2. BLIP-2 为什么需要 Q-Former？

视觉编码器输出的 token 数和隐藏维度通常都很大：

```text
image features: [B, Nv, Dv]
```

如果直接送给 LLM：

1. `Nv` 很大，占用上下文窗口；
2. `Dv` 与 LLM hidden size 不一致；
3. 视觉特征分布没有经过语言相关的筛选；
4. 图像局部 patch 太多，语言模型需要自己学习从哪里读取有效信息。

Q-Former 提供一个固定容量的瓶颈：

```text
[B, Nv, Dv]
       │
       ▼
Q-Former + Nq 个 learnable queries
       │
       ▼
[B, Nq, Dq]
       │
       ▼ Linear projection
[B, Nq, Dllm]
       │
       ▼
LLM
```

典型示例：

```text
[2, 257, 1408] → [2, 32, 768] → [2, 32, 4096]
```

这一步同时完成：

- 视觉 token 数压缩；
- 视觉信息筛选；
- 视觉特征与语言表示对齐；
- 视觉塔与 LLM 的维度桥接。

---

## 3. 版本与组件范围

| 名称 | 说明 |
|---|---|
| BLIP-2 | 原始论文提出的桥接架构，包含 Q-Former 和两阶段设计 |
| Q-Former | BLIP-2 中负责视觉查询、图文交互和视觉摘要的 Transformer |
| BLIP-2 + OPT | Q-Former 接 decoder-only 的 OPT |
| BLIP-2 + Flan-T5 | Q-Former 接 encoder-decoder 的 Flan-T5 |
| InstructBLIP | 在 BLIP-2 基础上把指令也引入 Q-Former，使视觉提取受问题条件影响 |
| LAVIS BLIP-2 实现 | 常见开源实现，提供 OPT、T5 等具体接口和配置 |

“BLIP-2”不是一个单一参数量的 checkpoint。视觉塔、LLM、hidden size、tokenizer、输入模板都会改变具体张量维度。

---

## 4. 贯穿全文的维度例子

以下用常见的 ViT-g/14 + 32 Query + OPT-2.7B 配置说明。数值用于读流程，具体 checkpoint 以配置为准。

| 符号 | 含义 | 示例 |
|---|---|---:|
| `B` | batch size | 2 |
| `H,W` | 图片尺寸 | 224, 224 |
| `P` | 视觉 patch size | 14 |
| `Nv` | 视觉 token 数 | `16×16+1=257` |
| `Dv` | 视觉 encoder 输出宽度 | 1408 |
| `Nq` | Query 数量 | 32 |
| `Dq` | Q-Former hidden size | 768 |
| `L` | 文本 token 数 | 32 |
| `Dc` | 阶段一对比投影维度 | 256 |
| `Dllm` | LLM hidden size | OPT-2.7B 常见为 4096 |

视觉输入与视觉输出：

```text
images:       [B, 3, 224, 224]
image_embeds: [B, 257, 1408]
```

Q-Former 输出：

```text
query_output: [B, 32, 768]
```

接 OPT 的投影后：

```text
visual_prefix: [B, 32, 4096]
```

如果换成 Flan-T5，最后一维要改成对应 T5 的 `d_model`，不能默认也是 4096。

---

## 5. 总体流程

```text
图片 [B,3,224,224]
       │
       ▼
冻结视觉编码器（ViT）
       │ [B,Nv,Dv]
       ▼
LayerNorm / 视觉特征整理
       │
       ▼
Q-Former 的 32 个可学习 Query
       │ Q = Query，K/V = image features
       ▼
query_output [B,32,768]
       │
       ▼
Linear(768 → Dllm)
       │ [B,32,Dllm]
       ├──────────────► OPT：作为视觉 prefix + 文本 embedding
       │
       └──────────────► Flan-T5：作为 encoder 的视觉输入 embedding
                               │
                               ▼
                         文本生成结果
```

一个容易考的点：

```text
视觉编码器和 Q-Former 之间是 Cross-Attention；
Q-Former 输出进入 LLM 后，OPT/T5 使用自己的语言模型结构继续处理。
```

---

## 6. 视觉编码器：从图片到 patch token

以 `224×224`、patch size `14` 为例：

```text
224 / 14 = 16
patch 数 = 16 × 16 = 256
```

视觉 ViT 的 patch 序列：

```text
[ B, 3, 224, 224 ]
→ patch embedding
[ B, 256, 1408 ]
→ 加 CLS token
[ B, 257, 1408 ]
→ ViT blocks
[ B, 257, 1408 ]
```

BLIP-2 常使用冻结的 ViT-g/14 视觉编码器，并在输出后加 LayerNorm：

```text
image_embeds = ln_vision(visual_encoder(images))
```

仍然是：

```text
[B, 257, 1408]
```

这里的 `257` 包含 CLS；Q-Former 的 Cross-Attention 可以读取视觉编码器输出序列。某些实现会在视觉侧选择或整理 CLS/patch，不能脱离 checkpoint 假设所有视觉塔都完全一样。

视觉编码器的作用是提供视觉上下文，Q-Former 的作用是选择和压缩，不要把两者混成一个模块。

---

## 7. Q-Former 的 Query 参数

Q-Former 使用固定数量的可学习 Query：

```python
query_tokens = nn.Parameter(
    torch.randn(1, num_query_token, hidden_size)
)
```

以 32 个 Query 为例：

```text
参数本身： [1, 32, 768]
扩展 batch：[B, 32, 768] = [2, 32, 768]
```

这些 Query：

- 不是图片 patch；
- 不是文本 token；
- 不是每张图片动态初始化；
- 不保证一个 Query 对应一个物体；
- 通过训练学会从视觉特征提取不同的信息。

它们可以被理解成固定数量的信息槽位。对同一组初始 Query，不同图片经过 Cross-Attention 后会产生不同的 Query 输出。

---

## 8. Query Self-Attention 的维度变化

Q-Former hidden state：

```text
x: [B, Nq, Dq] = [2, 32, 768]
```

假设使用 12 个 attention heads：

```text
head_dim = 768 / 12 = 64
Q/K/V: [B, 12, 32, 64]
```

注意力矩阵：

```text
Q @ K^T
[B, 12, 32, 64] @ [B, 12, 64, 32]
= [B, 12, 32, 32]
```

再乘 V 并合并 heads：

```text
[B, 12, 32, 32] @ [B, 12, 32, 64]
= [B, 12, 32, 64]
→ [B, 32, 768]
```

Self-Attention 让 Query 之间交换信息；它不会改变 Query 数量。

---

## 9. Query 到视觉特征的 Cross-Attention

输入：

```text
query_states: [B, 32, 768]
image_embeds: [B, 257, 1408]
```

Cross-Attention 的角色：

```text
Q：Q-Former Query
K：视觉编码器输出
V：视觉编码器输出
```

因为 `Dq=768`、`Dv=1408` 不同，内部会投影到 attention 的共同维度：

```text
Q = Linear(768  → 768)
K = Linear(1408 → 768)
V = Linear(1408 → 768)
```

拆头后：

```text
Q: [B, 12, 32, 64]
K: [B, 12, 257, 64]
V: [B, 12, 257, 64]
```

分数矩阵：

```text
Q @ K^T
[B, 12, 32, 64] @ [B, 12, 64, 257]
= [B, 12, 32, 257]
```

`[32,257]` 的含义是：每个 Query 对每个视觉 token 的注意力。

输出：

```text
[B, 12, 32, 257] @ [B, 12, 257, 64]
= [B, 12, 32, 64]
→ [B, 32, 768]
```

所以 Cross-Attention 前后 Query 序列长度不变，改变的是 Query 内容：

```text
[B, 32, 768] → [B, 32, 768]
```

---

## 10. Cross-Attention 不是每一层都必须有

BLIP-2 的 Q-Former 常以 BERT-base 结构初始化，Cross-Attention 以一定频率插入，例如每隔一个 Transformer block 插入。

抽象成：

```text
Layer 1：Self-Attention + FFN
Layer 2：Self-Attention + Vision Cross-Attention + FFN
Layer 3：Self-Attention + FFN
Layer 4：Self-Attention + Vision Cross-Attention + FFN
...
```

这样 Query 可以多次读取图像，同时不需要每层都执行视觉 Cross-Attention。

面试不要绝对化地说“每一层都有 Cross-Attention”。正确说法是：

> Q-Former 是 BERT-like Transformer，视觉 Cross-Attention 按配置以固定频率插入；常见实现是隔层插入。

---

## 11. 为什么 Q-Former 是 BLIP-2 的核心？

直接平均池化：

```text
[B, 257, 1408] → [B, 1, 1408]
```

直接线性投影全部视觉 token：

```text
[B, 257, 1408] → [B, 257, Dllm]
```

Q-Former：

```text
[B, 257, 1408]
→ 32 Query Cross-Attention
→ [B, 32, 768]
→ [B, 32, Dllm]
```

它相对直接连接的收益：

1. 输出 token 数固定；
2. 每个 Query 可以聚合不同的视觉区域；
3. Query 之间可以通过 Self-Attention 交互；
4. 阶段一可以用图文对比、匹配和生成目标学习语言相关的视觉表示；
5. LLM 不必直接处理所有视觉 patch。

代价是：

- 32 个 Query 是信息瓶颈；
- 小目标、文字和精确空间关系可能丢失；
- Q-Former 增加了额外参数和前向计算。

---

## 12. 阶段一：冻结视觉塔的视觉语言表示学习

用户重点是架构，这里只记住模块关系和三个目标。

### 12.1 阶段一的参数与输入

```text
冻结：视觉编码器
训练：Q-Former、Query、阶段一投影头/任务头
文本：作为 Q-Former 的输入分支
```

Q-Former 不只是单纯的视觉池化器。阶段一让它学习“哪些视觉信息对文字有用”。

### 12.2 ITC：Image-Text Contrastive Learning

图像侧通过 Query 得到：

```text
query_output: [B, 32, 768]
vision_proj:
[B, 32, 768] → [B, 32, 256]
image_feats: [B, 32, 256]
```

文本侧经过 Q-Former 文本分支，取 `[CLS]` 等句子表示：

```text
text_hidden: [B, L, 768]
text_cls:    [B, 768]
text_proj:   [B, 768] → [B, 256]
text_feats:  [B, 256]
```

图像的每个 Query 与文本向量计算相似度：

```text
image_feats: [B, Nq, Dc]
text_feats:  [B, Dc]
```

对每个图文组合先得到逐 Query 分数，再沿 Query 维聚合；常见实现使用最大相似度：

```text
逐 Query 分数：[B, B, Nq]
聚合后：      [B, B]
```

然后做 image-to-text 和 text-to-image 的对比损失。这里的 `B×B` 来自 batch 内候选配对。

**重要：ITC 中通常让图像 Query 和文本独立编码，避免图像和文本在计算相似度前通过融合 Self-Attention 泄漏信息。**

### 12.3 ITM：Image-Text Matching

ITM 判断某张图和某段文本是否匹配。与 ITC 不同，ITM 允许 Query 和文本进行更充分的多模态交互：

```text
Query + text tokens
→ Q-Former Self-Attention
→ Query Cross-Attention 读取 image features
→ 取 Query 输出
→ Linear(768 → 2)
```

维度示意：

```text
vl_embeddings: [B, 32, 768]
ITM head:      [B, 32, 2]
沿 Query 聚合：  [B, 2]
```

ITC 偏向全局对齐和检索，ITM 偏向一对图文的细粒度匹配。工程实现还可能利用 ITC 分数选取难负样本。

### 12.4 ITG：Image-grounded Text Generation

ITG 让 Q-Former 根据视觉信息预测文本 token：

```text
视觉 Query： [B, 32, 768]
文本 token：  [B, L, 768]
语言模型头：  [B, L, Vq]
```

它使用多模态因果 mask：

- Query 可以互相看；
- Query 可以读取图像；
- Query 不能读取未来文本；
- 文本位置可以读取 Query 和前面的文本 token；
- 文本位置不能看到未来文本。

阶段一的总目标可抽象为：

```text
L_stage1 = L_ITC + L_ITM + L_ITG
```

不同实现的权重、负样本和细节会变化，但三类目标的功能区别要记住。

---

## 13. Q-Former 的三种注意力 mask

这是面试高频点。用“能否看到”来记：

| 任务 | Query 看 Query | Query 看文本 | 文本看 Query | 文本看未来文本 |
|---|---|---|---|---|
| ITC | 可以 | 不可以 | 不可以 | 不适用/文本独立编码 |
| ITM | 可以 | 可以 | 可以 | 不允许未来泄漏，通常文本双向编码 |
| ITG | 可以 | 不可以 | 可以 | 不可以 |

视觉 token 不直接进入 Self-Attention 序列，而是作为 Cross-Attention 的 K/V 来源。

因此不要把 Q-Former 的所有任务都说成同一个 mask，也不要把阶段一的 Q-Former mask 和后端 OPT 的 causal mask 混为一谈。

---

## 14. 阶段二：把 Q-Former 接入冻结语言模型

### 14.1 阶段二的参数关系

原始 BLIP-2 的核心思路是：

```text
冻结：视觉编码器、LLM
训练：Q-Former、Query、LLM projection
```

Q-Former 从阶段一初始化，继续适配 LLM。视觉塔和 LLM 参数不更新，但 LLM 仍需保留从输入 embedding 到 loss 的梯度路径。

```python
for parameter in llm.parameters():
    parameter.requires_grad = False

# 可以得到对 inputs_embeds 的梯度，供 projection/Q-Former 更新。
outputs = llm(inputs_embeds=visual_prefix, labels=labels)
loss = outputs.loss
loss.backward()
```

如果把整个 LLM forward 放在 `torch.no_grad()` 中，会阻断桥接层梯度。

### 14.2 Q-Former 到 LLM 的投影

```text
query_output: [B, 32, 768]
Linear(768 → Dllm)
visual features: [B, 32, Dllm]
```

这个投影不是把向量变成词表 id，而是直接得到 LLM 的连续输入 embedding。

---

## 15. 接 OPT：decoder-only 路线

OPT 是 decoder-only causal language model。假设文本 token embedding 为：

```text
text_embeds: [B, L, Dllm]
```

把视觉 Query 作为 prefix：

```text
visual_prefix: [B, 32, Dllm]
text_embeds:   [B, L, Dllm]
concat:        [B, 32+L, Dllm]
```

完整序列可以理解成：

```text
[视觉前缀, prompt 文本, answer 文本]
```

OPT 使用 causal Self-Attention：

- 视觉 prefix 处于序列前部；
- prompt 可以读取前面的视觉 prefix；
- answer 可以读取视觉 prefix、prompt 和已经生成的 answer；
- 未来 answer token 不可见。

训练标签通常是：

```text
视觉 prefix：-100
prompt：     -100 或按任务设置
answer：     真实 token id
padding：    -100
```

训练时对 answer 的 next-token 计算交叉熵。图片向量不直接分类，但 answer loss 可以通过 OPT 的输入梯度更新 Q-Former 和 projection。

---

## 16. 接 Flan-T5：encoder-decoder 路线

Flan-T5 不是 decoder-only，而是 encoder-decoder。不能把 OPT 的 prefix 逻辑原样套到 T5。

典型流程：

```text
Q-Former 输出
[B, 32, 768]
→ Linear(768 → d_model_T5)
[B, 32, d_model_T5]
```

视觉 embedding 作为 encoder 侧的条件输入，通常与文本 encoder 输入组织在一起或作为视觉前缀，具体由实现决定：

```text
encoder inputs：视觉 prefix + 文本输入
encoder hidden： [B, L_encoder, d_model_T5]
                       │
                       ▼
decoder：自回归生成目标文本
```

T5 decoder 通过 Cross-Attention 读取 encoder hidden states：

```text
Decoder Q：当前文本生成状态
Encoder K/V：视觉与文本的 encoder 表示
```

因此两种 LLM 接口的根本区别是：

| 接口 | 视觉条件进入哪里 | 语言融合结构 |
|---|---|---|
| OPT | decoder 输入序列的视觉 prefix | decoder causal Self-Attention |
| Flan-T5 | encoder 输入/视觉前缀 | decoder Cross-Attention 读取 encoder 输出 |

面试时先判断语言模型类型，再描述视觉 embedding 的插入位置。

---

## 17. BLIP-2 中视觉和文本在哪里交互？

分三层回答最清楚：

### 17.1 视觉塔内部

ViT 的视觉 patch 通过 Self-Attention 交互，获得图像内部上下文。

### 17.2 Q-Former 内部

- Query Self-Attention：不同 Query 互相交换信息；
- Vision Cross-Attention：Query 读取视觉 encoder 的 K/V；
- 阶段一中，文本 token 根据 ITC/ITM/ITG mask 与 Query 交互。

### 17.3 后端 LLM 内部

- OPT：视觉 prefix 与文本在 decoder causal Self-Attention 中融合；
- Flan-T5：视觉条件进入 encoder，decoder 通过 Cross-Attention 读取 encoder 状态。

因此不能只回答“BLIP-2 使用 Cross-Attention”，需要说明是 Q-Former 的视觉读取阶段；后端语言模型的融合取决于 OPT 还是 Flan-T5。

---

## 18. 为什么说 BLIP-2 比端到端微调更高效？

架构角度要回答：

```text
视觉编码器：已经有通用视觉表征
LLM：        已经有通用语言能力
Q-Former：   学习两个预训练模块之间的接口
```

这样不需要同时更新视觉塔和整个 LLM，训练显存、参数更新量和灾难性遗忘风险都更小。

但“高效”不是“完全没有代价”：

- Q-Former 仍有 Transformer 和 Cross-Attention 计算；
- 32 Query 是信息瓶颈；
- 冻结的视觉塔可能不适合特殊领域；
- 视觉细节损失会影响 OCR、计数、空间关系。

不要把“冻结大模型”误说成“视觉和语言完全不融合”。融合发生在 Q-Former 和后端 LLM 的输入/注意力路径中。

---

## 19. 推理流程：从图片到回答

### 19.1 图像和文本准备

```text
图片
→ image processor
→ [B, 3, 224, 224]

问题
→ tokenizer
→ input_ids: [B, L]
```

### 19.2 视觉侧

```text
images
→ vision encoder
→ image_embeds: [B, 257, 1408]
→ Q-Former Query Cross-Attention
→ query_output: [B, 32, 768]
→ projection
→ visual_prefix: [B, 32, Dllm]
```

### 19.3 语言侧

OPT 路线：

```text
visual_prefix + text_embeds
→ decoder prefill
→ KV cache
→ 逐 token decode
```

Flan-T5 路线：

```text
visual prefix / text encoder input
→ encoder hidden states
→ decoder Cross-Attention
→ 逐 token decode
```

同一请求中，视觉编码和 Q-Former 通常只运行一次，后续生成利用语言模型的缓存。视觉条件越长，prefill 和缓存成本越高。

---

## 20. 和 Qwen-VL、LLaVA 的架构区别

| 对比项 | BLIP-2 | LLaVA-1.5 | Qwen2-VL |
|---|---|---|---|
| 视觉桥 | Q-Former Query Cross-Attention | 两层 MLP | PatchMerger + 动态视觉接口 |
| 视觉 token | 常见 32 个 | 常见保留 576 个 patch | 动态可变 |
| LLM 接口 | OPT 或 Flan-T5 | Vicuna/LLaMA decoder | Qwen2 decoder |
| 训练结构 | 阶段一先训练 Q-Former；阶段二接 LLM | 对齐 projector 后做视觉指令微调 | 以具体 Qwen2-VL 配置为准 |
| 时空位置 | 基础视觉/文本接口 | 基础版本不强调三轴视频位置 | M-RoPE 处理时间、高度、宽度 |
| 主要取舍 | 强压缩、固定视觉 prefix | 结构简单、保留更多 patch | 动态分辨率与时空建模更完整 |

一句话记忆：

```text
BLIP-2：Q-Former 先问视觉，再把 32 个回答交给 LLM。
LLaVA：MLP 把 patch 特征直接翻译成 LLM embedding。
Qwen2-VL：动态网格和 PatchMerger 处理可变图像/视频 token，并用 M-RoPE 保留时空位置。
```

---

## 21. 高频面试问答

### Q1：BLIP-2 的整体架构是什么？

**答：** 冻结视觉编码器提取图像特征，Q-Former 用固定数量 Query 通过 Cross-Attention 读取并压缩视觉 token，线性层把 Query 输出映射到 LLM hidden size，最后由 OPT 或 Flan-T5 生成文本。

### Q2：Q-Former 的 Query 从哪里来？

**答：** Query 是形状通常为 `[1, Nq, Dq]` 的可学习参数，前向时扩展到 batch。它不是从图片切出的 patch，也不是文本 token，不保证一个 Query 对应一个物体。

### Q3：Q-Former 的 Cross-Attention 中 Q、K、V 是什么？

**答：** Q 是可学习 Query 经过当前层 Self-Attention 后的状态，K/V 来自冻结视觉编码器的 image features。输出序列长度由 Query 数量决定。

### Q4：为什么 BLIP-2 只使用 32 个 Query？

**答：** 32 是信息容量、细节保留和 LLM 上下文成本之间的折中。Query 越少，视觉 prefix 越短但可能丢细节；越多，信息容量更大但后端序列和计算开销增加。它不是目标数量，也不是理论固定值。

### Q5：Q-Former 是不是 BLIP-2 的视觉编码器？

**答：** 不是。视觉编码器负责从像素提取视觉表征，Q-Former 是读取和压缩视觉表征的桥接器。BLIP-2 通常冻结视觉编码器，训练 Q-Former 来学习语言相关的信息选择。

### Q6：Q-Former 为什么比平均池化好？

**答：** 平均池化对所有空间位置等权，只得到一个全局向量；Q-Former 有多个 Query，每个 Query 可以学习不同的 Cross-Attention 分布，并通过 Self-Attention 交互，输出多个任务相关视觉 token。

### Q7：阶段一的三个目标分别是什么？

**答：** ITC 做独立图文表示对齐，ITM 让图文充分交互后判断是否匹配，ITG 让 Q-Former 根据图像生成或预测文本。三者分别强调全局相似度、细粒度匹配和语言生成。

### Q8：ITC 和 ITM 的区别是什么？

**答：** ITC 通常让图像 Query 和文本独立编码，再计算 batch 内图文相似度；ITM 允许图像 Query 与文本 token 融合，通过二分类头判断一对图文是否匹配。ITC 适合检索式对齐，ITM 适合细粒度配对判断。

### Q9：BLIP-2 第二阶段冻结哪些模块？

**答：** 原始设计冻结视觉编码器和 LLM，继续训练已经经过阶段一初始化的 Q-Former，以及连接 Q-Former 和 LLM 的线性投影层。具体开源变体可能改变冻结策略，但不能把所有 BLIP-2 配置混为一谈。

### Q10：冻结 LLM 后，Q-Former 怎么得到生成损失的梯度？

**答：** 冻结的是 LLM 参数，不是从 LLM 输入到 loss 的计算路径。梯度可以通过 LLM 的输入 embedding 传回 projection 和 Q-Former；不能将整个 LLM forward 放进 `torch.no_grad()`。

### Q11：BLIP-2 接 OPT 和接 Flan-T5 有什么区别？

**答：** OPT 是 decoder-only，视觉 Query 投影后通常作为 prefix 与文本 embedding 拼接，靠 causal Self-Attention 生成；Flan-T5 是 encoder-decoder，视觉 embedding 通常作为 encoder 侧条件，T5 decoder 通过 Cross-Attention 读取 encoder 表示。不能用 OPT 的 prefix 描述替代 T5 的接口。

### Q12：BLIP-2 会把图像转成离散文本 token 吗？

**答：** 不会。Q-Former 输出的是连续视觉 embedding，线性投影后直接作为 LLM 的输入向量。它不是先生成 caption，也不是查 LLM 词表得到视觉词向量。

### Q13：为什么图像 Query 可以和文本一起训练？

**答：** Q-Former 是 BERT-like 多模态 Transformer：Query 通过视觉 Cross-Attention 获取图像信息，文本 token 根据不同任务 mask 与 Query 交互，图文共同服务于对比、匹配或生成目标。

### Q14：Q-Former 的视觉 Cross-Attention 每一层都有吗？

**答：** 通常按固定频率插入，例如隔层插入。具体要以 `cross_attention_freq` 或模型实现为准，不能绝对说每层都插入。

### Q15：BLIP-2 的 32 个 Query 对应 32 个物体吗？

**答：** 不对应。Query 没有默认实例级检测监督，它们是信息槽位，可能编码物体、属性、背景、文字或关系，具体语义由训练和输入决定。

### Q16：BLIP-2 为什么可能丢失小字和空间关系？

**答：** 视觉 token 被压缩成固定数量的 Query，细粒度局部信息需要通过有限瓶颈传递；视觉编码器分辨率、Query 数、Cross-Attention 和数据覆盖都会影响细节保留。

### Q17：BLIP-2 与 LLaVA 最大的结构区别是什么？

**答：** BLIP-2 用 Q-Former 将长视觉序列压缩成少量 Query，再连接 OPT/T5；LLaVA-1.5 常用两层 MLP 逐 token 映射视觉 patch，通常保留更多 patch token，结构更简单但 LLM 序列更长。

### Q18：BLIP-2 与 Qwen-VL 原始版有什么相似和区别？

**答：** 二者都可能用可学习 Query 读取视觉特征，但 Q-Former 是 BERT-like 多任务桥接器，包含 ITC/ITM/ITG；Qwen-VL 使用自己的视觉语言适配器和 Qwen LLM 接口。不能因为 Query 数量都固定，就说二者架构相同。

### Q19：BLIP-2 能处理视频吗？

**答：** 原始 BLIP-2 的核心论文主要面向图像。视频可以通过采样帧、增加时间建模或改造视觉输入来扩展，但不能把“将多帧图像拼起来”当成原始架构自带的视频时序能力。

### Q20：如果把 Query 数从 32 改成 64，哪些维度变化？

**答：** Query 参数从 `[1,32,768]` 变成 `[1,64,768]`，Q-Former 输出从 `[B,32,768]` 变成 `[B,64,768]`，连接 LLM 后视觉 prefix 长度翻倍。Cross-Attention 分数从 `[B,heads,32,Nv]` 变成 `[B,heads,64,Nv]`；后端 LLM 的上下文和 KV cache 也会增加。

### Q21：Q-Former 是不是 Cross-Attention 版本的 CLIP？

**答：** 不是。CLIP 核心是双塔图文对比学习，输出图像/文本共享 embedding；Q-Former 是视觉语言桥接器，使用 Query 读取视觉 token，并可在阶段一执行对比、匹配和生成任务。两者的输出接口和训练角色不同。

### Q22：面试官让你从源码找 BLIP-2 的关键路径，看哪些变量？

**答：** 视觉侧看 `visual_encoder`、`ln_vision`、`image_embeds`；Query 侧看 `query_tokens`、`Qformer.bert`、`encoder_hidden_states`；阶段一看 `vision_proj`、`text_proj`、`itm_head`、`loss_itc/loss_itm/loss_lm`；阶段二看 `opt_proj` 或 T5 投影、`inputs_embeds`、`attention_mask` 和语言模型 `loss`。

---

## 22. 源码阅读：从图片跟到 LLM

```text
1. image processor
   └── [B,3,H,W]

2. visual_encoder
   └── image_embeds: [B,Nv,Dv]

3. ln_vision
   └── 形状不变

4. query_tokens.expand(B,-1,-1)
   └── [B,Nq,Dq]

5. Qformer.bert(
       query_embeds=query_tokens,
       encoder_hidden_states=image_embeds,
       encoder_attention_mask=image_atts,
   )
   └── last_hidden_state: [B,Nq,Dq]

6. vision_proj / opt_proj / t5_proj
   └── [B,Nq,Dllm] 或阶段一 [B,Nq,Dc]

7. OPT 路线
   └── 与 text embedding 沿序列维拼接

8. Flan-T5 路线
   └── 作为 encoder 侧条件输入

9. language model
   └── 生成 logits 和文本
```

### 22.1 关键参数与形状

```python
query_tokens                  # [1, Nq, Dq]
image_embeds                  # [B, Nv, Dv]
image_atts                    # [B, Nv]
query_output.last_hidden_state# [B, Nq, Dq]
vision_proj(query_output)     # [B, Nq, Dc]
opt_proj(query_output)        # [B, Nq, D_OPT]
```

### 22.2 看到这些变量时怎么判断作用？

- `query_embeds`：Q-Former 的可学习 Query 输入；
- `encoder_hidden_states`：视觉 Cross-Attention 的 K/V 来源；
- `encoder_attention_mask`：哪些视觉 token 有效；
- `last_hidden_state`：每个 Query 的输出，不是自动池化成一个向量；
- `empty_targets`：为视觉 prefix 等非文本位置填充 `-100`；
- `inputs_embeds`：已经包含连续视觉 embedding 的语言模型输入；
- `opt_proj`：Q-Former hidden size 到 OPT hidden size 的投影。

### 22.3 最常见的源码误判

```text
1. 把 query_tokens 当成每张图的 patch。
2. 把 Q-Former 输出误认为 [B,Dq]，忽略 [B,Nq,Dq]。
3. 把 encoder_hidden_states 误看成文本，而它在视觉路径中是图像特征。
4. 把冻结参数和 no_grad 混为一谈，误以为冻结 LLM 后不能传输入梯度。
5. 用 OPT 的视觉 prefix 流程解释 Flan-T5。
6. 把 ITC、ITM、ITG 说成同一个注意力 mask。
7. 把 32 个 Query 说成 32 个检测目标。
8. 把 projection 说成 reshape；它是可学习的最后一维变换。
```

---

## 23. 复杂度和信息瓶颈

设视觉 token 数为 `Nv`，Query 数为 `Nq`，Q-Former hidden size 为 `Dq`：

```text
视觉 Cross-Attention 的主要注意力分数规模：
[B, heads, Nq, Nv]
```

计算量大致随 `Nq × Nv × Dq` 增长，而不是 `Nv² × Dq` 的完整视觉序列自注意力。

Q-Former 输出后，LLM 只接收 `Nq` 个视觉 token：

```text
原始视觉序列：Nv
Q-Former 输出：Nq
```

当 `Nq << Nv` 时，后端 LLM 的视觉上下文和 KV cache 更短。但固定 Query 数也构成信息瓶颈：

```text
压缩更强 → 计算更省、信息可能更少
Query 更多 → 细节更多、LLM 上下文更长
```

不要只根据 Q-Former 的 `Nq×Nv` 计算量推断整个 BLIP-2 的速度；视觉塔、Q-Former FFN、LLM prefill/decode 和缓存也有成本。

---

## 24. 架构一页纸背诵版

```text
BLIP-2 = Frozen Vision Encoder + Q-Former + Frozen/Adapted LLM

图片：
[B,3,224,224]
→ ViT-g/14
[B,257,1408]

Query：
[1,32,768]
→ expand batch
[B,32,768]

Q-Former：
Q = Query
K/V = image features
Cross-Attention 分数：[B,heads,32,257]
输出：[B,32,768]

连接 LLM：
Linear(768 → Dllm)
[B,32,Dllm]

OPT：
视觉 prefix + 文本 embedding
→ decoder causal Self-Attention
→ 自回归生成

Flan-T5：
视觉 prefix / 文本输入进入 encoder
→ decoder Cross-Attention
→ 自回归生成

阶段一：
Q-Former 做 ITC、ITM、ITG，学习语言相关视觉表示。

阶段二：
冻结视觉塔和 LLM，继续训练 Q-Former 与 LLM projection。

核心取舍：
Q-Former 减少视觉 token、降低 LLM 上下文成本；
固定 Query 也可能丢失小目标、文字和精确空间关系。
```

---

## 25. 面试中的 30 秒架构回答

> BLIP-2 用冻结的 ViT 提取图像 patch 特征，Q-Former 通过一组可学习 Query 对这些视觉 token 做 Cross-Attention，得到固定数量的语言相关视觉向量，再经过线性投影输入 OPT 或 Flan-T5。若接 OPT，视觉向量作为 decoder 的 prefix 与文本一起做 causal Self-Attention；若接 Flan-T5，视觉向量进入 encoder，decoder 通过 Cross-Attention 生成文本。它的主要优势是用轻量桥接器复用两个大模型，主要瓶颈是 Query 数量造成的信息压缩。

---

## 26. 原论文与源码阅读入口

1. [BLIP-2 原论文，arXiv:2301.12597](https://arxiv.org/abs/2301.12597)：重点看 Q-Former、三种阶段一目标和两阶段框架图。
2. [LAVIS blip2.py](https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2.py)：看 `init_Qformer`、Query 初始化和 Cross-Attention 频率。
3. [LAVIS blip2_qformer.py](https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_qformer.py)：看 ITC、ITM、ITG 的数据流和 mask。
4. [LAVIS blip2_opt.py](https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_opt.py)：看 OPT 的 `opt_proj`、视觉 prefix 与 `inputs_embeds` 拼接。
5. [LAVIS blip2_t5.py](https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_t5.py)：对比 encoder-decoder 的 Flan-T5 接口。
6. [Hugging Face BLIP-2 文档](https://huggingface.co/docs/transformers/model_doc/blip-2)：查看 processor、`query_embeds`、`pixel_values`、`inputs_embeds` 等统一接口。

配套复习：[Q-Former](Q-Former_面试八股.md)、[LLaVA](LLaVA_流程与面试八股.md)、[Qwen-VL](Qwen-VL_流程与面试八股.md)。
