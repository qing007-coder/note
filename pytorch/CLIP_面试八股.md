# CLIP 面试八股：从原理到维度变化

> 适合面试前快速复习，也适合结合 PyTorch 源码阅读。
>
> **一句话记忆**：CLIP 用图像编码器和文本编码器把两种模态映射到同一个向量空间，通过 batch 内的对比学习，让正确的图文配对相似度变高，让错误配对相似度变低。

---

## 1. CLIP 解决什么问题？

传统图像分类通常需要：

- 固定的类别集合；
- 每张图片对应一个人工标注的类别 id；
- 测试时只能在训练过的类别中分类。

CLIP 的目标是学习一个**图像和文本共享的语义空间**。推理时，把类别写成文本，例如：

```text
"a photo of a dog"
"a photo of a cat"
"a photo of a car"
```

模型分别得到图像向量和文本向量，然后比较相似度。相似度最高的文本就是预测类别。

因此 CLIP 可以做 **zero-shot classification**：不针对某个下游分类数据集重新训练分类头，也能完成分类。

---

## 2. CLIP 的整体结构

CLIP 有两个独立编码器：

```text
图片 x                         文本 t
  │                              │
  ▼                              ▼
Image Encoder               Text Encoder
(ResNet / ViT)              (Transformer)
  │                              │
  ▼                              ▼
image feature                text feature
  [B, D_i]                    [B, D_t]
  │                              │
  └──── projection head ────────┘
             │
             ▼
      shared embedding space
          [B, D]
             │
             ▼
      L2 normalize + cosine similarity
             │
             ▼
          [B, B] logits
```

这里的关键是：两个编码器的原始输出维度可以不同，但经过 projection 后必须进入相同维度 `D` 的空间。

CLIP 不是把图像和文本 token 直接拼接后做一个 Transformer，而是**双塔结构**：图像塔和文本塔各自编码，最后用对比学习对齐。

---

## 3. 贯穿全文的维度例子

为了说明维度变化，统一使用下面这组配置：

| 符号 | 含义 | 示例值 |
|---|---|---:|
| `B` | batch size | 2 |
| `H, W` | 输入图像高、宽 | 224, 224 |
| `P` | ViT patch size | 16 |
| `N` | 图像 patch 数量 | `(224/16)^2 = 196` |
| `C` | 图像通道数 | 3 |
| `L` | 文本最大长度 | 77 |
| `V` | 文本词表大小 | 约 49,408 |
| `d_v` | ViT token 维度 | 768 |
| `d_t` | 文本 Transformer 宽度 | 512 |
| `D` | 共享 embedding 维度 | 512 |
| `h` | attention head 数 | 12（以 ViT-B/16 为例） |

实际模型的数值会因版本不同而变化，面试时要说明“以下以 ViT-B/16 为例”。

---

## 4. 图像编码器：以 ViT 为例

### 4.1 图片切成 patch

输入图片：

```text
x: [B, C, H, W] = [2, 3, 224, 224]
```

ViT 使用一个 `patch_size=16` 的卷积完成 patch 切分和线性投影：

```python
nn.Conv2d(
    in_channels=3,
    out_channels=768,
    kernel_size=16,
    stride=16,
)
```

卷积输出空间尺寸：

```text
H' = 224 / 16 = 14
W' = 224 / 16 = 14
```

因此输出：

```text
[B, d_v, H', W'] = [2, 768, 14, 14]
```

`14 × 14 = 196`，表示 196 个 patch。

### 4.2 展平 patch 序列

把空间维展平，并把 token 维移到最后：

```text
[2, 768, 14, 14]
    └─ flatten spatial dimensions
[2, 768, 196]
    └─ transpose
[2, 196, 768]
```

此时每个 patch 是一个 768 维 token。

### 4.3 加 CLS token 和位置编码

ViT 通常在序列开头添加一个可学习的 CLS token：

```text
patch tokens: [B, 196, 768]
cls token:    [B,   1, 768]
concat:       [B, 197, 768]
```

再加上位置编码：

```text
positional embedding: [1, 197, 768]
sequence:              [2, 197, 768]
```

广播后逐元素相加，形状不变。

### 4.4 经过 Transformer Encoder

Transformer Encoder 的输入和输出形状通常保持不变：

```text
[B, 197, 768]
      │ Transformer × 12
[B, 197, 768]
```

每层主要包含：

```text
x
│
├─ LayerNorm
├─ Multi-Head Self-Attention
├─ residual add
├─ LayerNorm
├─ MLP：768 → 3072 → 768
└─ residual add
```

Attention 内部的维度，以 12 个 head 为例：

```text
输入 x: [B, 197, 768]
Q/K/V:  [B, 197, 768]
拆头:   [B, 12, 197, 64]   # 768 / 12 = 64
QK^T:   [B, 12, 197, 197]
权重V:  [B, 12, 197, 64]
合并:   [B, 197, 768]
```

注意 `QK^T` 的最后两个维度是 `序列长度 × 序列长度`，因此是 `[197, 197]`，不是 `[768, 768]`。

### 4.5 取 CLS token 并投影

Transformer 输出：

```text
[B, 197, 768]
```

取第一个 token：

```text
image pooled feature = x[:, 0, :]
[B, 768]
```

再通过图像 projection：

```text
[B, 768] → Linear(768, 512) → [B, 512]
```

最终图像 embedding：

```text
image_embedding: [2, 512]
```

---

## 5. 文本编码器：Transformer 的维度变化

### 5.1 文本分词

CLIP 使用 BPE tokenizer。每个文本会被截断或补齐到固定长度 `L=77`。

假设 batch 中有 2 句话：

```text
input_ids: [B, L] = [2, 77]
```

每个元素是一个整数 token id，取值范围约为 `[0, V)`。

### 5.2 Token Embedding 和位置编码

词嵌入表：

```text
token_embedding: [V, d_t] = [49408, 512]
```

查表之后：

```text
input_ids:       [2, 77]
token embedding: [2, 77, 512]
```

位置编码：

```text
positional embedding: [77, 512]
```

广播相加后仍然是：

```text
[2, 77, 512]
```

### 5.3 文本 Transformer

假设文本 Transformer 有 12 层、8 个 head：

```text
输入: [B, L, d_t] = [2, 77, 512]
```

单层 attention 的维度：

```text
Q/K/V: [2, 77, 512]
拆成 8 个 head: [2, 8, 77, 64]
注意力分数:      [2, 8, 77, 77]
注意力输出:      [2, 8, 77, 64]
合并 head:       [2, 77, 512]
```

文本 Transformer 输出仍为：

```text
[2, 77, 512]
```

### 5.4 取文本结束位置的 token

CLIP 通常使用 EOT（end of text）位置的 hidden state 作为整句表示，而不是简单平均所有 token。

```text
text pooled feature: [B, 512]
```

如果实现中使用额外的文本 projection，则形状可能是：

```text
[B, 512] → Linear(512, 512) → [B, 512]
```

最终文本 embedding：

```text
text_embedding: [2, 512]
```

### 5.5 为什么不能直接取平均？

平均池化会把所有 token 等权混合，可能削弱句子末尾的语义信息。CLIP 训练时固定使用 EOT token 对应的位置作为句子级表示，推理时也要保持一致。

面试回答可以说：

> 文本编码器需要一个固定长度的句向量。CLIP 使用 EOT token 的 hidden state 作为全局文本表示，因为该位置在自回归 mask 下可以聚合前面的上下文，并且训练和推理的表示方式保持一致。

---

## 6. 图文对齐：归一化和相似度

图像和文本 embedding 都是：

```text
image_embedding: [B, D] = [2, 512]
text_embedding:  [B, D] = [2, 512]
```

先做 L2 归一化：

```python
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)
```

形状不变：

```text
[2, 512] → [2, 512]
```

归一化后点积就是余弦相似度：

```text
similarity(i, j) = image_features[i] · text_features[j]
```

矩阵乘法：

```text
image_features: [B, D]
text_features.T: [D, B]
logits:          [B, B]
```

本例中：

```text
[2, 512] @ [512, 2] = [2, 2]
```

`logits[i, j]` 表示第 `i` 张图片和第 `j` 条文本的相似度。

---

## 7. CLIP 的核心损失函数

一个 batch 中，第 `i` 张图片和第 `i` 条文本是正样本，其余组合是负样本。

相似度矩阵：

```text
              text 0   text 1
image 0        正样本    负样本
image 1        负样本    正样本
```

### 7.1 图像到文本的交叉熵

对 logits 的每一行做分类：图片应该找到对应文本。

```python
labels = torch.arange(B, device=device)
loss_i = cross_entropy(logits_per_image, labels)
```

`logits_per_image` 形状：

```text
[B, B]
```

每一行有 `B` 个候选文本，正确类别是对角线位置 `i`。

### 7.2 文本到图像的交叉熵

转置 logits 后，对每一行做分类：文本应该找到对应图片。

```python
loss_t = cross_entropy(logits_per_text, labels)
```

其中：

```python
logits_per_text = logits_per_image.T
```

### 7.3 对称损失

最终 CLIP loss：

```python
loss = (loss_i + loss_t) / 2
```

完整表达式：

```text
L = 1/2 [ CE(S, I) + CE(S^T, I) ]
```

其中：

- `S` 是图像到文本的相似度矩阵；
- `I = [0, 1, ..., B-1]` 是正确配对标签。

### 7.4 为什么要双向对比？

只做 image-to-text 只能保证每张图片能找到文本；加入 text-to-image 后，文本也必须能找到正确图片，约束更对称，通常训练更稳定，图文共享空间也更可靠。

---

## 8. temperature 参数是什么？

CLIP 会学习一个温度参数，代码中常见写法是：

```python
logit_scale = self.logit_scale.exp()
logits = logit_scale * image_features @ text_features.T
```

注意参数通常保存为 `logit_scale` 的对数形式，真正使用时要 `exp()`，这样可以保证温度缩放因子为正。

如果用温度 `tau` 表示：

```text
logits = cosine_similarity / tau
```

代码中的 `logit_scale` 通常等价于 `1 / tau`。

作用：

- `tau` 小：分布更尖锐，模型更强调最高相似度；
- `tau` 大：分布更平滑；
- 学习温度能让模型自动调节对比分类的难度。

面试常问：**为什么不用固定温度？**

回答：不同数据规模、模型规模和训练阶段对 logits 的合适尺度不同。把温度设为可学习参数，优化器可以自动找到更合适的相似度分布。

工程上通常会给 `logit_scale` 设置上限，避免 `exp()` 后数值过大导致训练不稳定。

---

## 9. Zero-shot 分类的维度变化

假设有 `K=3` 个候选类别：

```text
"a photo of a dog"
"a photo of a cat"
"a photo of a car"
```

### 文本侧

```text
K 条文本
→ tokenizer
[K, 77]
→ text encoder
[K, 512]
→ L2 normalize
[K, 512]
```

### 图片侧

一张图片：

```text
[1, 3, 224, 224]
→ image encoder
[1, 512]
→ L2 normalize
[1, 512]
```

### 相似度

```text
image_features: [1, 512]
text_features.T: [512, 3]
logits:           [1, 3]
```

对 `[1, 3]` 做 softmax，最高位置就是预测类别。

如果使用多个 prompt，例如：

```text
"a photo of a dog"
"a blurry photo of a dog"
"a close-up photo of a dog"
```

可以先对同一类别的多个文本 embedding 求平均，再归一化，得到该类别的 prototype。

---

## 10. 预训练数据和训练方式

CLIP 使用大规模互联网图文对进行预训练。每个样本通常是：

```text
(image, caption)
```

它不需要人工定义的分类类别，但需要图像和文本之间存在一定语义关联。

训练流程：

```text
1. batch 中读取 B 个图文对
2. 图像编码器得到 [B, D]
3. 文本编码器得到 [B, D]
4. 归一化
5. 计算 [B, B] 相似度矩阵
6. 对角线作为正样本
7. 计算双向交叉熵
8. 反向传播更新两个编码器和温度参数
```

### batch size 为什么重要？

标准 CLIP 中，同一个 batch 内其他图文组合都可以当作负样本。一个 batch 有 `B` 对样本，就能形成约 `B^2` 个图文组合。

`B` 越大，负样本越多，对比学习通常越有效，但显存和通信成本也越高。大规模训练常用多卡 all-gather，把所有 GPU 上的特征拼接后再计算全局 logits。

---

## 11. CLIP 和普通分类模型的区别

| 对比项 | 普通图像分类 | CLIP |
|---|---|---|
| 标签 | 固定类别 id | 文本描述 |
| 输出层 | 固定分类头 | 图像与文本相似度 |
| 新类别 | 通常要重新训练 | 改 prompt 即可尝试 zero-shot |
| 训练目标 | 分类交叉熵 | 图文对比损失 |
| 结构 | 通常单编码器 | 图像塔 + 文本塔 |
| 负样本 | 分类头隐式提供 | batch 内其他配对 |
| 泛化方式 | 类别边界泛化 | 语义空间泛化 |

---

## 12. CLIP 和 ALIGN、BLIP 的区别

### CLIP vs ALIGN

两者都是大规模图文对比学习方法，核心训练思想相近。ALIGN 使用更大规模、噪声更高的图文数据，模型和数据清洗策略不同。

### CLIP vs BLIP

BLIP 是更完整的视觉语言框架，通常包含图文理解、图像描述、视觉问答等生成或匹配任务；CLIP 的核心产物是共享 embedding，主要用于检索、分类、排序和下游迁移。

### CLIP vs cross-attention 模型

CLIP 是双塔结构，图像和文本最后才交互，推理速度快，适合大规模检索。

cross-attention 模型会让图像 token 和文本 token 在中间层直接交互，表达能力强，但每个图文对都要联合计算，检索时成本较高。

---

## 13. 高频面试问题与标准回答

### Q1：CLIP 为什么能 zero-shot？

**答：** CLIP 训练的不是固定类别分类器，而是图文共享语义空间。推理时把类别转换成文本描述，计算图片 embedding 与各个类别文本 embedding 的相似度，因此新类别不需要重新训练分类头，只要文本描述能表达类别语义即可。

### Q2：CLIP 的正样本和负样本怎么构造？

**答：** 一个 batch 中第 `i` 张图片和第 `i` 条文本是正样本，其他 `i != j` 的组合都是负样本，所以相似度矩阵的对角线是正样本。

### Q3：为什么 logits 是 `[B, B]`？

**答：** 图像特征是 `[B, D]`，文本特征转置后是 `[D, B]`，矩阵乘法得到 `[B, B]`。第 `i` 行表示第 `i` 张图片与 batch 中所有文本的相似度。

### Q4：为什么要做 L2 normalize？

**答：** 归一化后点积等于余弦相似度，去除了向量长度对相似度的影响，使模型主要学习方向上的语义关系，也让相似度尺度更稳定。

### Q5：CLIP 为什么使用对称 loss？

**答：** 图像到文本和文本到图像是两个方向的匹配任务。双向交叉熵同时约束两个检索方向，使共享空间更对称、更稳定。

### Q6：图像编码器为什么可以用 ResNet，也可以用 ViT？

**答：** CLIP 只要求图像编码器最终输出一个图像向量。ResNet 和 ViT 都可以作为图像塔，只要通过 pooling 或 CLS token 得到全局特征，再投影到共享 embedding 维度即可。

### Q7：ViT 中 patch 数量怎么计算？

**答：** 对输入尺寸 `H×W`、patch size `P`，patch 数量是 `(H/P)×(W/P)`。例如 `224×224`、`P=16`，得到 `14×14=196` 个 patch；加 CLS 后序列长度是 `197`。

### Q8：为什么文本最大长度是 77？

**答：** 这是 CLIP 预训练时设定的最大 token 长度，便于批量计算和固定位置编码。超过长度会截断，较短文本会补齐。它不是语言模型理论上的必然值，而是模型配置。

### Q9：CLIP 能理解中文吗？

**答：** 原始 OpenAI CLIP 主要使用英文图文数据和英文 tokenizer，中文效果通常有限。中文场景应使用中文或多语言 CLIP，例如 Chinese-CLIP、AltCLIP，或者针对中文数据做微调。

### Q10：CLIP 的主要缺点是什么？

**答：** 主要包括：互联网图文数据存在噪声和偏见；文本描述不一定精确对应图片；细粒度定位能力弱；对 prompt 敏感；对计数、空间关系和复杂组合关系的理解有限；zero-shot 结果不一定超过针对任务训练的监督模型。

### Q11：为什么 prompt 会影响结果？

**答：** CLIP 学到的是图文联合分布，文本模板本身会改变文本 embedding。`"a photo of a dog"` 和 `"a drawing of a dog"` 的 embedding 不同，所以模板、上下文和类别名称都会影响相似度。

### Q12：如何提升 zero-shot 分类效果？

**答：** 可以使用多个 prompt ensemble，对同一类别的多个文本模板分别编码并平均；使用更准确的类别描述；对文本特征和图像特征做一致的归一化；也可以采用 linear probe、prompt tuning 或 adapter 做轻量适配。

### Q13：CLIP 的 embedding 能直接用于下游任务吗？

**答：** 可以。常见做法有：冻结 CLIP 后训练线性分类器（linear probe）、微调整个模型、只训练 projection、训练 adapter，或者做 prompt tuning。不同方案在数据量、计算成本和过拟合风险上不同。

### Q14：为什么大 batch 对 CLIP 很重要？

**答：** batch 内其他样本就是负样本。大 batch 提供更多负样本，使对比任务更有区分度；但也需要更多显存，分布式训练时通常需要跨卡收集特征。

### Q15：如果 batch 中存在错误配对或重复图片怎么办？

**答：** 标准对比损失会把错误配对当成负样本，噪声会影响训练。可以做数据清洗、去重、软标签、去除疑似冲突样本，或者使用更鲁棒的对比目标。

---

## 14. 代码阅读时最容易看错的地方

### 14.1 `transpose` 后的维度顺序

图像 patch 通常经历：

```text
[B, C, H', W']
→ flatten(2)
[B, C, N]
→ transpose(1, 2)
[B, N, C]
```

最后一个维度才是 token embedding 维度。

### 14.2 `permute` 不等于改变数据内容

`permute` 只改变维度视图的顺序，不会自动完成线性投影。例如：

```python
x = x.permute(0, 2, 1)
```

只是 `[B, C, N] → [B, N, C]`，不会把 `C` 映射成别的维度。

### 14.3 `expand` 和 `repeat` 的区别

添加 CLS token 时常见：

```python
cls = self.class_embedding.to(x.dtype)
cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device)
```

这样利用广播创建 batch 维。`expand` 通常不复制数据，`repeat` 会实际复制，显存开销不同。

### 14.4 `logit_scale` 不是直接的 temperature

源码中常见：

```python
self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
```

真正计算：

```python
scale = self.logit_scale.exp()
```

所以参数保存的是 `log(1/tau)`，不是 `tau` 本身。

### 14.5 文本 mask 的方向

CLIP 文本 Transformer 常使用 causal mask，使当前位置不能看到未来 token。阅读源码时要确认 mask 的形状和上三角/下三角方向，不能只看变量名。

### 14.6 `argmax` 前要确认维度

zero-shot 分类中：

```python
logits = image_features @ text_features.T  # [B, K]
pred = logits.argmax(dim=-1)              # [B]
```

`dim=-1` 是在类别维度上取最大值，而不是在 batch 维度上取最大值。

---

## 15. 一页纸背诵版

```text
CLIP = Image Encoder + Text Encoder + Contrastive Learning

图像：
[B, 3, 224, 224]
→ ViT patch embedding
[B, 196, 768]
→ 加 CLS
[B, 197, 768]
→ Transformer
[B, 197, 768]
→ 取 CLS
[B, 768]
→ projection
[B, 512]

文本：
[B, 77]
→ token embedding
[B, 77, 512]
→ Transformer
[B, 77, 512]
→ 取 EOT
[B, 512]
→ projection
[B, 512]

对齐：
image [B, 512] @ text.T [512, B] = logits [B, B]

loss：
image-to-text CE + text-to-image CE，再取平均

推理：
图片和 K 个类别文本分别编码
[1, 512] @ [512, K] = [1, K]
argmax 得到 zero-shot 类别
```

---

## 16. 面试时可以主动补充的工程细节

1. **数值稳定性**：归一化时要考虑零向量和混合精度；`logit_scale.exp()` 需要关注溢出。
2. **分布式训练**：跨 GPU all-gather 图像和文本特征，否则每张卡只能看到局部负样本。
3. **数据质量**：图文对比学习对配对质量敏感，网页数据中的错配、重复和偏见会直接进入 embedding 空间。
4. **评估方式**：zero-shot 分类、图文检索 Recall@K、线性探测和下游微调要分开报告。
5. **冻结策略**：小数据集通常先冻结 backbone，只训练线性层或 adapter，避免全量微调过拟合。
6. **prompt 设计**：类别名称、模板、语言和上下文都会影响文本向量，实际应用中应验证多个模板。

---

## 17. 最后用一句话回答“CLIP 是什么”

> CLIP 是一个图文双塔模型：图像编码器和文本编码器分别把两种模态映射到同一个归一化向量空间，通过 batch 内的对称对比学习拉近正确图文对、推远错误图文对，因此推理时可以通过文本 prompt 进行 zero-shot 分类和图文检索。
