# LLaVA 流程与面试八股

> **一句话记忆**：LLaVA 将预训练视觉编码器的 patch 特征通过一个可训练连接器投影到 LLM 的 embedding 维度，在文本序列的图片占位处插入这些连续向量，再由 decoder-only LLM 自回归生成回答。
>
> **阅读范围**：以常见的 **LLaVA-1.5-7B** 为主线：CLIP ViT-L/14@336 + 两层 MLP projector + Vicuna-7B v1.5。原版 LLaVA、1.5、NeXT 和 OneVision 的配置并不相同，后文单独区分。以下数值用于解释这个具体配置，不代表整个系列。

---

## 1. 先用 30 秒回答“LLaVA 是什么？”

> LLaVA 是一种视觉语言助手模型，用 CLIP 视觉塔提取图像 patch 特征，用线性层或 MLP 将视觉特征映射到语言模型的 embedding 空间，再将视觉 embedding 和问题文本 embedding 拼成一个序列，送入 LLM 生成回答。训练通常分两步：先冻结视觉塔和 LLM，只训练连接器完成初步对齐；再冻结视觉塔，训练连接器和语言模型完成视觉指令微调。LLaVA-1.5 的典型实现使用 CLIP ViT-L/14@336、两层 MLP 和 Vicuna。

需要记住三个角色：

```text
Vision Tower：把像素变成视觉特征。
Projector：把视觉特征映射到 LLM 输入维度，并学习接口适配。
LLM：联合利用视觉条件和文本上下文，逐 token 生成回答。
```

---

## 2. 为什么光有 CLIP 和 LLM 还不够？

CLIP 的视觉塔已经能提取语义相关的图像特征，LLM 已经会理解和生成文本，但二者的接口没有天然对齐：

- 视觉特征维度与语言 embedding 维度可能不同；
- 同维度也不代表语义分布相同；
- LLM 还需要学会根据图像回答问题、解释场景、遵循用户指令。

因此需要两件事：

1. **连接器和对齐训练**：把视觉特征转成 LLM 可以利用的连续输入。
2. **视觉指令微调**：让模型学习“看图并按问题回答”的行为。

不要把 projector 只理解成补齐 shape 的工具。维度一致是接口条件，图像条件能否被语言模型正确使用还依赖训练。

---

## 3. 整体数据流

```text
输入图片                                用户问题
   │                                 “图里的人在做什么？”
   ▼                                        │
图像预处理                                  ▼
[B, 3, 336, 336]                         tokenizer
   │                                    [B, Lt]
   ▼                                        │
CLIP ViT-L/14@336                            ▼
取指定层 hidden states，去 CLS           文本 embedding
[B, 576, 1024]                          [B, Lt, 4096]
   │                                        │
   ▼                                        │
两层 MLP projector                          │
1024 → 4096 → 4096                          │
   │                                        │
   ▼                                        │
[B, 576, 4096]                              │
   └────────────────┬───────────────────────┘
                    ▼
     在 <image> 的位置插入视觉 embedding
       同步构建 attention mask、labels
                    │
                    ▼
         decoder-only Vicuna / LLaMA
                    │
                    ▼
      下一个文本 token 的词表概率分布
                    │
                    ▼
       自回归生成：“他正在骑自行车。”
```

视觉和文本进入 LLM 后是同一个序列中的不同位置。经典 LLaVA 不额外给 LLM 插入专门的图文 Cross-Attention 层。

---

## 4. 全文统一的维度符号

| 符号 | 含义 | 示例 |
|---|---|---:|
| `B` | batch size | 2 |
| `H, W` | 预处理后的图像高、宽 | 336, 336 |
| `P` | 视觉 patch 的边长 | 14 |
| `N` | patch 数，不包含 CLS | `24×24=576` |
| `Dv` | CLIP ViT-L 的隐藏宽度 | 1024 |
| `Dl` | Vicuna-7B 的隐藏宽度 | 4096 |
| `Lt` | 展开前序列长度，包含一个图片占位 | 示例 41 |
| `L` | 图片占位展开后的长度 | `41-1+576=616` |
| `V` | LLM 词表大小 | 取决于 tokenizer 和 checkpoint |

注意三个容易混淆的维度：

```text
1024：视觉 Transformer 的 patch hidden state 维度。
4096：本例 LLM 的 embedding / hidden state 维度。
V：   语言模型输出的词表大小。
```

视觉宽度不是 CLIP 全局检索向量的维度，也不应把所有 LLaVA 模型的 LLM 宽度都写成 4096。典型 Vicuna-13B 的宽度为 5120。

---

## 5. 图像预处理：原图怎样变成模型输入？

输入通常是不同分辨率、不同宽高比的 RGB 图片。视觉塔需要适合其配置的张量：

```text
原图：例如 [720, 1280, 3]
→ 宽高比处理、resize、归一化
→ channel-first
[3, 336, 336]
→ batch
[B, 3, 336, 336]
```

LLaVA-1.5 官方训练配置常使用 `image_aspect_ratio=pad`，先将非正方形图片补成正方形，再进入图像处理流程，减少直接中心裁剪丢失边缘内容的问题。

注意：

- 具体 resize、crop、mean/std 应使用 checkpoint 对应的 image processor；
- 不能训练时用 CLIP 归一化，推理时随意改成另一套；
- 补边保留了内容范围，但固定分辨率下，小字和小目标仍可能被缩小到难以辨认。

---

## 6. 视觉塔：576 个 token 是怎么来的？

### 6.1 Patch embedding

对于 `336×336` 的输入，patch size 为 14：

```text
336 / 14 = 24
N = 24 × 24 = 576
```

以卷积等价形式看 patch embedding：

```text
[B, 3, 336, 336]
→ Conv2d(3, 1024, kernel_size=14, stride=14)
[B, 1024, 24, 24]
→ flatten(2)
[B, 1024, 576]
→ transpose(1, 2)
[B, 576, 1024]
```

### 6.2 CLS 与位置编码

CLIP ViT 内部会加入 CLS，并加上视觉位置编码：

```text
patch tokens: [B, 576, 1024]
CLS:          [B,   1, 1024]
拼接后：       [B, 577, 1024]
```

经过视觉 Transformer 后，每个 patch 已经通过视觉 Self-Attention 融合其他位置的信息。

### 6.3 选取特征层并丢弃 CLS

典型 LLaVA-1.5 配置选择 `mm_vision_select_layer=-2`，从视觉 hidden states 中取倒数第二层，再使用 patch 特征：

```python
# 示意：具体结构取决于视觉模型包装器。
features = vision_outputs.hidden_states[-2]  # [B, 577, 1024]
patch_features = features[:, 1:, :]          # [B, 576, 1024]
```

需要同时记住：

1. 使用 patch 序列保留局部区域信息；
2. 不是把 CLIP 的全局图像 embedding 当作整幅图唯一的视觉输入；
3. 不需要 CLIP 文本编码器参与这条看图问答路径。

**为什么经常取倒数第二层？** 这是常见的经验配置，可减少对最终 CLIP 对齐输出的依赖并保留适合下游的表征；不能把“倒数第二层一定最好”当成理论结论。层选择和是否包含 CLS 都应以 checkpoint 配置为准。

---

## 7. Projector：MLP 到底改变了什么？

LLaVA-1.5 常见的 `mlp2x_gelu`：

```python
projector = nn.Sequential(
    nn.Linear(1024, 4096),
    nn.GELU(),
    nn.Linear(4096, 4096),
)
```

维度变化：

```text
[B, 576, 1024]
→ Linear(1024, 4096)
[B, 576, 4096]
→ GELU
[B, 576, 4096]
→ Linear(4096, 4096)
[B, 576, 4096]
```

**它改变通道维度，不压缩 token 数量。**

MLP 对每个 token 的最后一维应用相同的变换，不在这两层中做跨 token 的注意力。不过输入 patch 特征已经经过视觉塔交互，输出也会在 LLM 中继续交互，因此不能说 LLaVA 的 patch 从来不交换信息。

输出是 576 个连续向量：

```text
visual_embeds: [B, 576, 4096]
```

它们不需要先变成文字，也不需要逐个对应词表里的某个单词。

---

## 8. `<image>` 是一个视觉 token 吗？

**不是。要区分文本占位和真正送入 LLM 的视觉 embedding。**

假设输入模板是：

```text
SYSTEM: ...
USER: <image>\n图里的人在做什么？
ASSISTANT:
```

原始 LLaVA 实现可用一个特殊图片索引标记插入位置，然后将序列拆成图片标记前后的文本片段：

```text
文本片段 A | 图片占位 | 文本片段 B
```

进入 LLM 前，图片占位被视觉 embedding 序列替换：

```text
emb(A) | 576 个 visual embeddings | emb(B)
```

如果占位前后实际文本共 40 个 token，展开前为 41 个位置：

```text
展开前：Lt = 41
展开后：L = Lt - 1 + 576 = 616
```

图片位置本身不代表“只有 1 个图像 token”。某些 Hugging Face 实现可能由 processor 提前展开图片占位，代码细节不同，但最终都要使视觉向量与对应序列位置匹配。

有的配置还会使用 image start/end 特殊符号；这些不是所有 checkpoint 的统一要求，也不是 576 个 patch token 的来源。

---

## 9. 混合序列如何进入 LLM？

假设某条样本展开后长度为 `L=616`：

```text
inputs_embeds:  [B, 616, 4096]
attention_mask: [B, 616]
position_ids:   与序列位置对应，具体 batch 布局依实现
```

LLM 的 Transformer 层保持隐藏宽度与序列长度：

```text
[B, 616, 4096]
→ LLM layers
[B, 616, 4096]
→ lm_head
[B, 616, V]
```

`[B, 616, V]` 是训练时的完整词表预测形状示意。推理实现可以只保留需要位置的 logits，以减少显存，不必总是物化整个张量。

这里存在两套位置机制：

1. **视觉塔的二维位置表示**：帮助视觉 patch 感知图片空间结构；
2. **LLM 的序列位置机制**：例如 LLaMA 的 RoPE，作用于展开后的混合序列。

LLM 的第 20 个序列位置不是图片坐标 `(20, 0)`。不能把语言模型的序列位置和原图二维坐标混为一谈。

---

## 10. 图文交互是 Self-Attention 还是 Cross-Attention？

经典 LLaVA 将视觉和文本 embedding 放到同一个 decoder-only 序列，使用 LLM 已有的 **causal Self-Attention**。

```text
sequence = [system / prefix, image tokens, question, answer]
```

后面的文本位置可以把前面的视觉位置作为 K/V 来读取，因此回答可以依赖图像。

但因果约束仍然存在：

- 回答位置可以读取之前的图片、问题与回答前缀；
- 问题位置能否看到图片，取决于图片是否位于它之前；
- 视觉位置不能在同一层读取位于它后面的用户问题；
- 视觉 patch 在进入 LLM 之前，已经通过 CLIP ViT 做过视觉上下文交互。

**不要说“拼起来后所有 token 都双向互看”。** 经典 LLaVA 的 LLM 使用因果掩码，不是 BERT 式双向编码器。这里描述的是 1.5 常见实现，不能推断所有后续模型都采用相同 mask。

Attention 中所有序列位置都会生成 Q/K/V，不是“文本只做 Q、图像只做 K/V”的专门 Cross-Attention 结构。

---

## 11. 第一阶段：视觉语言特征对齐

### 11.1 训练哪些模块？

```text
CLIP 视觉塔：冻结
LLM：        冻结
Projector：  训练
```

本阶段不是从零训练视觉模型和语言模型，而是先为两个预训练模块建立可用接口。

### 11.2 数据和目标

LLaVA-1.5 官方常用约 558K 图文对作为对齐数据。训练可以把图像描述写成目标回答：

```text
输入：图片 + “请描述这张图片。”
目标：一名穿红色衣服的人正在骑自行车。
```

目标是 LLM 的自回归文本生成损失：

```text
L_align = - Σ_t log p(caption_t | image, prompt, caption_<t)
```

**它不是 CLIP 的图文对比损失。** 不需要在 batch 中构造 `[B,B]` 图文相似度矩阵作为这里的核心目标。

### 11.3 冻结 LLM，Projector 怎么得到梯度？

```text
文本生成 loss
   ↓
经过 LLM 对输入 embedding 的导数
   ↓
visual_embeds
   ↓
projector 参数
```

冻结 LLM 表示参数不更新，不表示切断输入到 loss 的计算图。

```python
for parameter in llm.parameters():
    parameter.requires_grad = False

# 仍然需要梯度从 LLM 输入传回 projector。
outputs = llm(inputs_embeds=inputs_embeds, labels=labels)
loss = outputs.loss
loss.backward()
```

如果把整个 LLM 前向放进 `torch.no_grad()`，会阻断对连接器的梯度。冻结视觉塔且不需要优化输入图片时，视觉塔前向通常可以不记录反向图。

---

## 12. 第二阶段：视觉指令微调

### 12.1 训练哪些模块？

以官方常见的 full fine-tuning 路线为例：

```text
CLIP 视觉塔：冻结
Projector：  继续训练
LLM：        训练
```

如果使用 LoRA，更新的是 LLM 的低秩适配参数，以及配置允许训练的 projector 等模块，不是所有 LLM 权重。不能将 LoRA 设置和全量微调的参数冻结表混在一起。

### 12.2 数据是什么？

训练数据包含图片、多轮对话、问题与回答，例如：

```text
USER: <image> 这张图里有什么？
ASSISTANT: 一个人正在骑自行车。
USER: 他穿什么颜色的衣服？
ASSISTANT: 红色。
```

LLaVA-1.5 官方约 665K 混合指令数据包含约 150K GPT 生成的视觉指令数据，以及约 515K 其他学术任务导向的 VQA 等数据；混合中还包含用于保持语言能力的纯文本对话。

这些规模是对应版本的训练设置，不是模型架构对数据量的硬性要求。

### 12.3 指令微调学到了什么？

- 根据图片和问题选择回答内容；
- 遵循长度、格式等回答要求；
- 处理多轮视觉对话；
- 将预训练语言能力用于视觉问答、描述、推理等任务。

它既需要视觉信息，也可能利用语言先验。因此回答流畅不代表一定正确识别了图片。

### 12.4 为什么拆成两个阶段？

先训练连接器，可以在开始联合微调前建立初步视觉语言接口，降低随机初始化连接器对 LLM 的干扰；再用指令数据适配回答行为。

这是原始设计与常见训练策略，不意味着“一阶段训练理论上不可行”。其他模型可以采用不同的数据配比、初始化和训练调度。

---

## 13. Loss 算在哪里？为什么要设置 `-100`？

### 13.1 输入中有图像和问题，监督通常针对回答

SFT 使用 teacher forcing：训练时将正确回答也作为输入的一部分，令模型学习预测下一个词。

```text
输入序列：   system | user + image + question | assistant answer
监督标签：   忽略   | 忽略                    | 回答 token id
```

视觉 embedding、padding，以及模板中不参与监督的用户/系统内容，对应 label 使用 ignore index，常见是 `-100`：

```text
labels: [B, L]

visual positions  → -100
user/system      → 通常 -100
padding          → -100
assistant answer → 对应 token id
```

多轮数据可监督多个 assistant 回答，而不只监督最后一轮；角色分隔符和终止符的具体处理依模板而定。

### 13.2 为什么不是预测视觉 embedding？

LM head 输出的是词表 logits，训练目标是文本 token。图像向量在这里充当条件，不是要重建的像素或离散视觉码。

因此视觉位置不直接参与词表分类 loss，但回答 loss 仍然可以通过注意力传回视觉输入和连接器。

### 13.3 `labels=-100` 与 attention mask 完全不同

| 机制 | 作用 | 视觉 token 的常见设置 |
|---|---|---|
| `labels=-100` | 不直接计算该位置的监督目标 | 是 |
| `attention_mask=0` | 将该位置标记为无效或 padding，具体解释依实现 | 否，视觉位置应该有效 |

**图片位置不算 loss，但必须能被回答位置看到。** 如果把图片在 attention mask 中屏蔽，模型就失去了正常的视觉条件。

### 13.4 Next-token shift

常见 causal LM 在内部执行：

```python
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
loss = cross_entropy(
    shift_logits.reshape(-1, vocab_size),
    shift_labels.reshape(-1),
    ignore_index=-100,
)
```

不要在模型已经内部 shift 的情况下再次手动 shift 标签。更精确地说，是某位置的 logits 用来预测下一个位置的目标，而不是“同位置照抄输入词”。

---

## 14. 推理流程：首个 token 与后续 token

### 14.1 Prefill：处理图片和完整问题

```text
图片 → CLIP → projector → visual embeddings
问题 → tokenizer → text embeddings
展开 <image> → 混合序列
LLM 一次处理整个 prompt
得到最后有效位置的 logits
选择第一个回答 token
```

这里的最后有效位置要结合 padding side 和生成框架处理，不能在右侧 padding 的 batch 中无条件拿最后一个数组位置。

### 14.2 Decode：逐 token 生成

使用 KV cache 时：

```text
第 1 步：prompt → 预测 “他”
第 2 步：缓存 + “他” → 预测 “正在”
第 3 步：缓存 + “正在” → 预测 “骑”
...
```

后续通常不重新运行 CLIP，也不重新计算所有历史位置的 K/V。新 token 的 Q 会对缓存中的历史 K/V 做注意力，历史中包括视觉位置。

**KV cache 减少历史重复计算，不让每步计算成本完全变成常数。** 随着上下文变长，新 token 需要读取的历史 K/V 也会增多。

### 14.3 为什么图片越多、分辨率越高，推理越贵？

视觉 token 更多，会增加：

- 视觉塔的计算；
- 混合序列的 prefill 长度；
- KV cache 占用；
- 后续 decode 对历史视觉位置的读取成本。

使用 FlashAttention 可减少注意力的内存开销，但不会自动减少视觉 token 数，也不会将一般的全注意力算术复杂度变成线性。

---

## 15. 完整张量流程，一次串起来

```text
1. 图片预处理
   [2, 3, 336, 336]

2. CLIP ViT patch embedding
   [2, 576, 1024]

3. 视觉塔内部加 CLS、位置表示并编码
   [2, 577, 1024]

4. 取选定层 patch hidden states，去 CLS
   [2, 576, 1024]

5. 两层 MLP projector
   [2, 576, 1024]
   → [2, 576, 4096]
   → [2, 576, 4096]

6. 原始序列每条假设长 41，包含 1 个 <image> 占位
   文本片段共 40 个实际 token
   text embeddings 总计 [2, 40, 4096]

7. 按占位位置插入 576 个视觉 embedding
   [2, 616, 4096]，616 = 41 - 1 + 576

8. attention_mask / labels 同步展开
   [2, 616]

9. LLM 隐藏状态
   [2, 616, 4096]

10. LM head，训练时全序列形状示意
    [2, 616, V]

11. shifted CE，仅对有效回答目标计算损失
    标量 loss
```

真实 batch 内样本长度不同，需要先逐条展开，再 padding 到该 batch 的最大有效长度。不能仅修改 embedding 而不修改 labels、mask 与位置处理。

---

## 16. 用伪代码读懂一次训练前向

下面以单张图片、单条样本说明主要逻辑，不是可直接运行的完整模型；分词模板、batch padding 和生成缓存应交给对应实现处理。

```python
# 参数策略：阶段一只训练 projector；阶段二训练 projector + LLM。
# 两个阶段的典型设置都冻结 vision_tower。
with torch.no_grad():
    outputs = vision_tower(images, output_hidden_states=True)
    patch_features = outputs.hidden_states[-2][:, 1:, :]
    # [1, 576, 1024]

visual_embeds = projector(patch_features)
# [1, 576, 4096]，进入 LLM 前确保 dtype/device 匹配。

# 先在 input_ids 中定位图片占位，再拆分为两侧的普通文本 id。
# 不要把某些原实现中的负数图片占位索引直接送给普通 embedding 表。
left_ids, right_ids = split_at_image_placeholder(input_ids)
left_embeds = llm.get_input_embeddings()(left_ids)
right_embeds = llm.get_input_embeddings()(right_ids)

inputs_embeds = torch.cat(
    [left_embeds, visual_embeds, right_embeds], dim=1
)
# [1, L_left + 576 + L_right, 4096]

visual_labels = torch.full(
    (1, 576), -100, dtype=torch.long, device=visual_embeds.device
)
labels = torch.cat([left_labels, visual_labels, right_labels], dim=1)
# left/right labels 已将用户、系统等非监督位置设置为 -100。

attention_mask = build_valid_mask(inputs_embeds)
# 此处没有 padding，所有实际输入位置均有效，包括图像位置。

outputs = llm(
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    labels=labels,
)
loss = outputs.loss
loss.backward()
# 阶段一也必须保留穿过 LLM 到 projector 的梯度路径。
```

源码里不一定直接用上面这些辅助函数名，重点认出“视觉特征提取 → 连接器 → 占位替换 → 标签展开 → causal LM loss”这条链。

---

## 17. 原版、1.5、NeXT、OneVision 怎么区分？

| 版本 | 面试时应强调什么 | 不能混用的表述 |
|---|---|---|
| 原版 LLaVA | 视觉指令微调；CLIP 视觉塔、线性连接器与 Vicuna；GPT 辅助构建指令数据 | 不要直接把两层 MLP、336 分辨率和 665K 数据归给原版 |
| LLaVA-1.5 | 更强的简单基线：两层 MLP、常用 336 视觉输入、改进的数据混合 | 本文 576 token 的例子主要对应这一配置 |
| LLaVA-NeXT / 1.6 | 提升分辨率和长宽比适应能力，常用多裁剪/AnyRes；也有不同语言骨干 | 高分辨率图像不再一定只有 576 个视觉 token |
| LLaVA-OneVision | 图像、多图与视频场景的统一设计，采用更新的视觉与语言骨干 | 不要默认视觉塔一定是 CLIP、LLM 一定是 Vicuna |

### 17.1 AnyRes 怎么理解？

一种常见思路是将图片分成多个适配视觉塔的 crop，分别编码，再结合全局视图及空间组织方式送入 LLM。

```text
高分辨率图片
→ 全局视图 + 多个局部 crop
→ 各自视觉编码
→ 特征拼接、网格重排、unpad 等版本相关处理
→ 更多视觉 token
```

实际 token 数取决于裁剪网格、是否移除 padding、全局视图以及特殊分隔位置等，不应无条件写成固定的“crop 数 × 576”。

### 17.2 为什么更高分辨率有帮助？

小字、小物体、图表元素在低分辨率下可能已经消失。更高分辨率保留更多视觉证据，但会增加视觉计算、LLM 上下文和缓存开销。

高分辨率不是 OCR 或计数正确率的保证，还需要匹配的数据、训练与空间信息处理。

---

## 18. 原始视觉指令数据怎么来的？

原始 LLaVA 使用文本形式的图像描述、物体类别及边框等符号信息，构造提示并让文本 GPT-4 生成视觉相关的对话、详细描述和复杂推理问答。

```text
图像的 captions + 物体/边框等文本信息
→ 文本 GPT-4 生成问答
→ 将问答与原始图片配对
→ 视觉指令微调数据
```

**不要把这段历史说成“最初让具备视觉能力的 GPT-4 直接看所有原图生成回答”。** 当时的方法通过文本化图像信息生成指令。

这种办法降低了标注成本，但生成结果可能带有错误、语言先验和信息覆盖不足。后续 1.5 的数据混合也不只是这类合成问答，还加入了其他任务数据。

---

## 19. 和 CLIP、BLIP-2、Q-Former、TokenLearner 的关系

### 19.1 CLIP 与 LLaVA

| 维度 | 原始 CLIP | LLaVA-1.5 |
|---|---|---|
| 主要目标 | 学习图文相似度空间 | 根据图片与指令生成回答 |
| 视觉使用方式 | 常取全局特征做匹配 | 取 patch hidden states 输入 LLM |
| 文本侧 | CLIP 文本编码器 | Vicuna 等自回归 LLM |
| 核心训练目标 | 对称图文对比损失 | 文本生成 / 指令微调损失 |
| 输出 | 相似度、检索排序等 | 自由文本 |

LLaVA 复用 CLIP 的视觉表征，不等于照搬 CLIP 的双塔训练流程。

### 19.2 BLIP-2 / Q-Former 与 LLaVA

```text
BLIP-2：视觉塔 → Q-Former → 投影 → LLM
LLaVA： 视觉塔 → Linear / MLP → LLM
```

| 对比项 | 原始 BLIP-2 常见设计 | LLaVA-1.5 常见设计 |
|---|---|---|
| 桥接模块 | 可学习 Query + Cross-Attention | 逐 token 两层 MLP |
| 视觉压缩 | 常见 32 个 Query 输出 | patch 数通常保持不变，例如 576 |
| 前期训练 | ITC、ITM、ITG 训练 Q-Former | 文本生成目标训练 projector 对齐 |
| 连接 LLM 时 | 冻结视觉塔和 LLM，训练 Q-Former 与连接投影 | 第一阶段冻结两端；第二阶段训练 projector 和 LLM |
| 主要取舍 | 查询摘要减少后端 token，可能形成信息瓶颈 | 接口简单、保留较多 patch，后端序列更长 |

这张表说的是对应模型的基础预训练设计。BLIP-2 的后续任务微调、InstructBLIP 和各种改造不一定沿用同一参数冻结策略。

不能直接说“MLP 比 Q-Former 强”或相反，比较还受训练数据、LLM、视觉塔、分辨率和微调策略影响。

### 19.3 TokenLearner / TopK 剪枝插在哪里？

如果视觉 token 太多，可以在视觉塔和 LLM 之间引入聚合、选择或重采样模块：

```text
视觉塔 → token 压缩/选择 → projector → LLM
```

也可能设计在 projector 之后或 LLM 中间层，依具体方法而定。

- TokenLearner：加权聚合得到新 token；
- TopK pruner：打分后保留原 token 子集；
- Q-Former：Query 通过 Cross-Attention 提取固定长度表示。

**这些不是经典 LLaVA-1.5 必备组件。** 增加剪枝器需要评估信息损失、坐标/位置处理和训练匹配，不能直接把 token 删掉就假设性能不变。

---

## 20. 高频面试问答

### Q1：LLaVA 的创新只是把 CLIP 和 LLM 拼起来吗？

**答：** 结构确实简洁，但还包括视觉指令数据构造、两阶段训练和让模型遵循多模态指令的训练方式。连接结构解决接口问题，数据和指令微调决定模型怎样利用图片回答。

### Q2：图片会先被转成一段文字再输入 LLM 吗？

**答：** 经典 LLaVA 不需要先生成 caption。图片变成 patch 特征，经 projector 变为连续视觉 embedding，直接插入 LLM 的输入序列。

### Q3：为什么需要 projector？

**答：** 它将视觉宽度映射到语言模型的输入宽度，并通过训练学习两个预训练模块间的接口。仅仅维度相同并不意味着语义已经对齐。

### Q4：LLaVA-1.5 为什么是 576 个视觉 token？

**答：** 典型 CLIP ViT-L/14@336 对 `336×336` 输入产生 `24×24=576` 个 patch，视觉塔内部还有 CLS；LLaVA 常取 patch 特征并去掉 CLS，因此是 576 个。这不是所有 LLaVA 版本的固定数量。

### Q5：MLP 是不是把 576 个 token 压缩成一个？

**答：** 不是。两层 MLP 逐 token 改变最后一维：`[B,576,1024] → [B,576,4096]`，token 数保持不变。

### Q6：为什么不只使用 CLIP 的 CLS 向量？

**答：** 单一全局向量容易丢失细粒度内容。patch 序列保留更多区域信息，适合问答、空间理解和局部属性识别，但会增加 LLM 序列长度。

### Q7：为什么不用 CLIP 的文本编码器？

**答：** CLIP 文本塔面向图文相似度表示，不承担开放式自回归回答。LLaVA 用 Vicuna 等 LLM 来处理问题、对话与生成。

### Q8：两阶段分别冻结什么？

**答：** 典型 1.5 第一阶段冻结视觉塔和 LLM，只训练 projector；第二阶段冻结视觉塔，训练 projector 和 LLM。LoRA 版本则以低秩适配参数替代 LLM 全量更新。

### Q9：第一阶段是不是做图文对比学习？

**答：** 不是。它通过冻结 LLM 对图像描述等文本进行自回归预测，用语言生成损失训练连接器，与 CLIP 的 batch 内双向对比损失不同。

### Q10：冻结 LLM 后还要经过它反向传播吗？

**答：** 要。LLM 参数可以不更新，但需要保留输入 embedding 到 loss 的梯度，让梯度回传到 projector。不能把 LLM 前向整体包在 `no_grad()` 中。

### Q11：视觉 token 不算 loss，为什么 projector 仍能训练？

**答：** 后续回答位置通过注意力利用视觉 token，回答的生成误差会对视觉 embedding 产生梯度。`labels=-100` 只忽略直接监督，不会自动断开该输入对后续预测的影响。

### Q12：图文交互靠哪种注意力？

**答：** 经典 LLaVA 把图文 embedding 放入同一个 decoder-only 序列，依靠 LLM 原有的 causal Self-Attention 交互，没有额外的图文 Cross-Attention 层。

### Q13：视觉 token 能看到后面的问题吗？

**答：** 在经典 LLaVA 的因果 LLM 中，位于问题前面的视觉位置不能读取未来问题；后面的回答可以同时读取前面的图像和问题。视觉塔内的 patch 特征此前已经有视觉上下文。

### Q14：视觉 token 如何携带位置信息？

**答：** patch 特征首先在视觉塔中结合视觉位置表示，再按一定顺序插入 LLM，LLM 还会使用自身的序列位置机制。语言序列位置并不等价于原图二维坐标。

### Q15：推理每生成一个词都要重新跑视觉塔吗？

**答：** 通常不用。同一轮生成中，图片先编码一次，prefill 建立包含视觉条件的缓存，后续逐 token decode 利用 KV cache。新请求或上下文改变是否能复用缓存，需要由服务实现管理。

### Q16：为什么 LLaVA 会出现幻觉？

**答：** 视觉证据可能不足，训练问答可能有噪声，模型又有强语言先验；低分辨率、细节丢失和任务分布差异也会放大错误。生成流畅并不保证回答受真实图片充分约束。

### Q17：怎样缓解幻觉？

**答：** 改进数据质量和负例覆盖，匹配任务提高视觉分辨率，加强细粒度 grounding 或 OCR 证据，训练合理的不确定性表达，并用有针对性的评估检查。仅降低 temperature 可以减少采样随机性，不能从根本上修正视觉识别或事实错误。

### Q18：为什么计数、OCR 和空间关系容易错？

**答：** 这些任务依赖局部细节和精确位置；resize、patch 化、表征聚合可能丢信息，训练数据也未必充分监督精确数量与关系。高分辨率、专项数据和更合适的视觉处理能改善，但不是必然解决。

### Q19：如何支持多图或视频？

**答：** 技术上可以为不同图片或帧构造视觉 embedding 并组织进上下文，但还需要图像边界、时间信息、token 预算和对应任务数据。能拼接多段 embedding 不等于原始单图 checkpoint 已经具备可靠的多图或视频能力。

### Q20：如果想做领域适配，先考虑什么？

**答：** 先确认视觉分辨率是否能保留领域信息、数据与标签是否可靠，再匹配任务做指令微调。小数据场景可考虑 LLM LoRA、训练 projector 等受控更新；若视觉塔根本不能表达领域特征，再评估解冻或替换视觉塔，而不是只增加文本问答。

### Q21：评估 LLaVA 不能只看回答像不像人，还要看什么？

**答：** 分别检查感知、OCR、计数、关系理解、知识问答、幻觉与指令遵循；使用任务指标并抽查错误类型，留意数据泄漏和评测偏差。部署还要测首 token 延迟、生成速度、显存与不同图像尺寸下的质量。

### Q22：为什么不直接用 Q-Former 替代 MLP？

**答：** 可以改造，但结构、训练流程和信息瓶颈都变化了。Q-Former 可减少后端 token，MLP 更简单并保留较多 patch；不能仅以模块复杂度预测效果，需要在相同数据与预算下比较。

---

## 21. 工程追问与易错点

### 21.1 Padding、label 和图片展开必须同步

```text
展开前： [文本, <image>, 文本]
展开后： [文本 embedding, 576 个视觉 embedding, 文本 embedding]
```

同步更新序列长度、有效位置 mask、label 与位置处理。图片数量与占位数量不一致、展开后标签错位，都可能导致训练或生成错误。

### 21.2 上下文预算要按展开后计算

一张图片的占位可能只显示成一个符号，却占用几百个实际位置。多轮长对话、多图或高分辨率裁剪要预留回答空间，避免展开后截断图片或有效回答。

### 21.3 `eval()` 不会冻结参数

```text
eval()：切换 dropout 等模块的行为。
requires_grad=False：参数不累积梯度。
no_grad()：当前范围不记录反向图。
```

三者作用不同，尤其不能以 `eval()` 代替冻结，也不能用 `no_grad()` 切断训练中的连接器梯度。

### 21.4 特殊模板不能随意更换

角色标记、image 占位、分隔符、停止符与 tokenizer 需符合 checkpoint。模型参数没变，换错模板也可能导致回答异常或监督 mask 错误。

### 21.5 dtype 和 device 要一致

视觉塔、projector 与 LLM 可能使用不同精度。拼接前确保视觉和文本 embedding 的 device/dtype 合适；token id 与 label 通常仍使用整数类型。

### 21.6 更少 FLOPs 不保证同比例加速

视觉编码、MLP、LLM prefill/decode、数据搬运与 KV cache 的瓶颈不同。减少视觉 token 可能改善后端开销，但应测完整请求延迟，不能仅根据 `N²` 宣称整模型快了几百倍。

全注意力层除了注意力矩阵运算约 `O(L²Dl)`，还有投影与 FFN 等约 `O(LDl²)` 的计算；prefill 与单步 decode 的成本也不同。

---

## 22. 一页纸背诵版

```text
LLaVA = Vision Tower + Projector + Decoder-only LLM

以 LLaVA-1.5-7B 为例：
图片 [B,3,336,336]
→ CLIP ViT-L/14@336
→ 选定层、去 CLS
视觉 patch [B,576,1024]
→ Linear(1024,4096) + GELU + Linear(4096,4096)
视觉 embedding [B,576,4096]
→ 替换序列中的图片占位
→ 与文本 embedding 组成混合序列
→ LLM causal Self-Attention
→ 逐 token 生成文本

Projector 改通道，不减少 576 个 patch。
连续视觉向量不等于可读文字，不需要先转成 token id。

阶段一：
冻结视觉塔 + LLM，只训练 projector。
目标是文本生成，不是 CLIP 对比损失。

阶段二：
冻结视觉塔，训练 projector + LLM（或 LLM 的 LoRA）。
使用视觉指令和对话数据。

Loss：
监督 assistant 回答。
图片、padding、非监督上下文的 labels 使用 -100。
图片 attention mask 仍然有效。

推理：
先 prefill，再带 KV cache 逐词 decode。
同一轮不必反复跑视觉塔。

重点区别：
CLIP：图文相似度 / 对比学习。
BLIP-2：Q-Former 压缩并桥接冻结 LLM。
LLaVA-1.5：MLP 连接，视觉指令阶段更新 LLM。
NeXT：高分辨率 / AnyRes，视觉 token 数不固定为 576。
```

---

## 23. 原论文与官方源码阅读顺序

以下源码链接指向官方仓库的 `main`，包含后续功能；复现实验要固定版本和配置，不能将当前所有分支功能都归到 1.5。

1. [Visual Instruction Tuning，原版 LLaVA 论文](https://arxiv.org/abs/2304.08485)：理解视觉指令数据生成、初始模型与两阶段训练。
2. [Improved Baselines with Visual Instruction Tuning，LLaVA-1.5 论文](https://arxiv.org/abs/2310.03744)：理解 MLP、数据混合、分辨率等基线改进。
3. [官方 README 的训练说明](https://github.com/haotian-liu/LLaVA#train)：核对 558K 对齐数据、约 665K 指令数据与训练配置。
4. [clip_encoder.py](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_encoder/clip_encoder.py)：看 `feature_select`、`select_layer` 与 patch/CLS 选择。
5. [multimodal_projector/builder.py](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_projector/builder.py)：看 `linear`、`mlp2x_gelu` 的结构。
6. [llava_arch.py](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py)：看 `encode_images`、`prepare_inputs_labels_for_multimodal`，跟踪图片占位替换、mask、labels 和 padding。
7. [train.py](https://github.com/haotian-liu/LLaVA/blob/main/llava/train/train.py)：看数据预处理、角色监督 mask、冻结策略与训练参数。
8. [LLaVA-1.5 pretrain.sh](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/pretrain.sh) 与 [finetune.sh](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/finetune.sh)：核对视觉塔、特征层、projector、分辨率和训练阶段。
9. [LLaVA-NeXT 官方介绍](https://llava-vl.github.io/blog/2024-01-30-llava-next/) 与 [LLaVA-OneVision 论文](https://arxiv.org/abs/2408.03326)：了解高分辨率、多图和视频扩展，不与本文基础配置混写。

配套复习：[CLIP](CLIP_面试八股.md)、[Q-Former](Q-Former_面试八股.md)、[TokenLearner 与 TopKTokenPruner](TokenLearner与TopKTokenPruner_面试八股.md)。

如果时间有限，优先看第 3 节总流程、第 8 节图片占位、第 11–13 节训练与 loss、第 20 节面试问答，以及第 22 节背诵版。
