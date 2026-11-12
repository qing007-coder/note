# Qwen-VL 流程与面试八股：架构重点

> **一句话记忆**：Qwen-VL 把图片先编码成视觉特征，再通过视觉语言连接器变成 LLM 能接收的连续 embedding，插入文本序列后由 Qwen 自回归生成回答。原始 Qwen-VL 使用固定数量 Query 做视觉压缩；Qwen2-VL 改为动态分辨率、PatchMerger 和多模态旋转位置编码 M-RoPE。
>
> **本笔记的重点**：架构、数据流、维度变化和源码阅读。训练只保留理解架构所需的最少背景。

---

## 1. 先用 30 秒回答“Qwen-VL 是什么？”

> Qwen-VL 是 Qwen 系列的大视觉语言模型。它由视觉编码器、视觉语言连接器和自回归语言模型组成：视觉编码器把图片变成 patch 特征，连接器把视觉特征压缩或映射到语言模型隐藏维度，模型再把这些连续视觉向量和问题文本放进同一个多模态上下文中，依靠语言模型生成回答。原始 Qwen-VL 使用 OpenCLIP ViT-bigG、256 个 Query 的视觉语言适配器和 Qwen-7B；Qwen2-VL 进一步支持动态分辨率、图像/视频的时空网格和 M-RoPE，视觉 token 数量不再固定。

需要拆成三个角色：

```text
Vision Encoder：像素 → 视觉 patch / tubelet 特征。
Vision-Language Connector：视觉特征 → LLM hidden size，并完成压缩或重采样。
Qwen LLM：视觉 embedding + 文本 embedding → 因果自回归生成。
```

**面试先问版本。** “Qwen-VL”可能指 2023 年原始模型，也可能被口语化地用来指 Qwen2-VL / Qwen2.5-VL。架构回答必须说明版本，不能把固定 256 token 和动态分辨率混在一起。

---

## 2. Qwen-VL 家族的架构演进

| 版本 | 视觉输入与编码 | 视觉连接器 | 位置与序列特点 | 面试关键词 |
|---|---|---|---|---|
| Qwen-VL（原始） | OpenCLIP ViT-bigG，典型输入 448×448 | 单层 Cross-Attention + 256 个可学习 Query，再接 MLP | 常见固定数量视觉 token | 256 Query、视觉语言适配器 |
| Qwen-VL-Chat | 在 Qwen-VL 上加入多轮视觉对话能力 | 基本沿用 Qwen-VL 视觉接口 | 对话模板、图像特殊标记 | Chat template、多轮上下文 |
| Qwen2-VL | 动态分辨率 ViT，支持图像和视频 | PatchMerger，空间 2×2 合并后投影到 LLM | M-RoPE，区分时间/高度/宽度 | Naive Dynamic Resolution、image_grid_thw、video_grid_thw |
| Qwen2.5-VL | 在 Qwen2-VL 方向继续增强视觉、视频和文档能力 | 以具体 checkpoint 配置为准 | 视觉侧可见窗口/全局注意力、时序和多层特征增强 | 不要默认所有细节都与 Qwen2-VL 相同 |

后文先完整讲原始 Qwen-VL，再把 Qwen2-VL 的动态架构单独讲清楚。

---

## 3. 原始 Qwen-VL 的整体数据流

```text
图片
  │
  ▼
图像预处理：resize / normalize
  │ [B, 3, 448, 448]
  ▼
OpenCLIP ViT-bigG
  │ 大量 patch 特征
  ▼
视觉语言适配器
  │ 256 个可学习 Query 通过 Cross-Attention 读取视觉特征
  ▼
256 个视觉 token
  │ 通过 MLP 映射到 Qwen 的 hidden size
  ▼
[B, 256, D_llm]
  │
  ├── 插入 <img> 与 </img> 之间的视觉位置
  └── 与问题文本 embedding 拼成混合序列
          │
          ▼
       Qwen-7B decoder-only Transformer
          │ causal Self-Attention
          ▼
       LM Head → 下一个文本 token
          │
          ▼
       自回归生成回答
```

原始 Qwen-VL 的视觉 Query 类似 Q-Former 的信息瓶颈，但不要直接说它就是完整的 BLIP-2 Q-Former：Qwen-VL 使用的是自己的视觉语言适配器，组件、训练目标和语言模型接口都不同。

---

## 4. 原始 Qwen-VL 的维度例子

下面的数值用于帮助追维度；具体 checkpoint 的视觉 hidden size 和 Qwen hidden size 以配置为准。

| 符号 | 含义 | 教学示例 |
|---|---|---:|
| `B` | batch size | 2 |
| `H,W` | 视觉输入尺寸 | 448, 448 |
| `P` | patch size | 14 |
| `Nv` | 视觉 patch 数 | `32×32=1024` |
| `Dv` | 视觉编码器特征宽度 | 视觉配置决定 |
| `Nq` | Query 数量 | 256 |
| `Dq` | 适配器内部 Query 宽度 | 配置决定 |
| `Dllm` | Qwen LLM hidden size | Qwen-7B 配置决定 |
| `Lt` | 图片占位前文本序列长度 | 例如 41 |
| `L` | 展开后的混合序列长度 | `Lt-1+256=296` |

如果视觉编码器使用 `448×448`、`patch_size=14`：

```text
448 / 14 = 32
patch 数 = 32 × 32 = 1024
```

如果视觉塔内部带 CLS，内部序列可能是 `1025`；连接器具体是否使用 CLS、使用哪一层 hidden state，要看实现配置。经过 256 Query 的视觉语言适配器后，输出长度变为 256：

```text
视觉特征： [B, 1024, Dv]
Query：    [B, 256, Dq]
适配器输出：[B, 256, Dq]
投影后：   [B, 256, Dllm]
```

不要把 `Nv=1024` 和 `Nq=256` 混淆：前者是视觉编码器的输入序列长度，后者是连接器输出给语言模型的长度。

---

## 5. 原始 Qwen-VL 的视觉语言适配器

### 5.1 Query 做什么？

适配器有一组可学习 Query：

```text
query_tokens: [1, Nq, Dq]
Nq = 256
```

扩展到 batch：

```text
[1, 256, Dq]
→ [B, 256, Dq]
```

Cross-Attention 中：

```text
Q：Query token
K/V：视觉编码器的 patch 特征
```

若视觉 patch 特征为 `[B, 1024, Dv]`，则注意力分数的 token 维度是：

```text
[B, heads, 256, 1024]
```

每个 Query 可以从全部视觉 patch 中聚合信息，最终输出固定 256 个视觉 token。

### 5.2 为什么要压缩？

如果把 1024 个 patch 全部送入 7B 语言模型：

- 文本上下文被视觉 token 占用更多位置；
- prefill 和 KV cache 更贵；
- 多图和多轮对话更容易超过上下文预算。

压缩后：

```text
[B, 1024, Dv] → [B, 256, Dq]
```

这是一种“固定长度视觉摘要”，但也意味着细粒度信息可能丢失。Query 数量不是图片中目标的数量，也不保证一一对应物体。

### 5.3 再接 MLP 做维度对齐

适配器输出要变成 Qwen LLM 的 hidden size：

```text
[B, 256, Dq]
→ MLP / Linear
[B, 256, Dllm]
```

这一步既是维度接口，也承担视觉分布到语言 embedding 分布的适配。不能把它说成简单的 reshape。

---

## 6. 图片特殊标记：`<img>` 是什么？

原始 Qwen-VL 的文本模板会用图片相关特殊标记包住图片位置，概念上可以写成：

```text
<|user|>
<img> IMAGE_PLACEHOLDER </img>
请描述这张图。
<|assistant|>
```

要区分三类东西：

1. **特殊 token 的 tokenizer id**：例如图片开始/结束标记，在文本序列中有离散 id。
2. **图片占位位置**：告诉模型这里要插入视觉条件。
3. **真正的视觉 embedding**：由视觉塔和连接器产生的连续向量，形状是 `[Nq, Dllm]` 或 Qwen2-VL 中的可变长度 `[Nimg, Dllm]`。

图片 embedding 不是通过普通词表 embedding 查出来的单词向量。实现可能先把占位符编码成 token id，再在 `inputs_embeds` 阶段替换/拼接视觉向量。

图片开始和结束标记是否参与 attention、是否有额外的视觉边界 embedding，取决于版本和实现；不要把某个仓库的模板当成整个 Qwen-VL 家族的硬规则。

---

## 7. 原始 Qwen-VL 如何和文本组成序列？

假设图片展开成 256 个视觉 embedding，图片占位原来只占 1 个位置：

```text
展开前：文本 token 40 个 + 1 个图片占位
Lt = 41

展开后：
40 个文本 embedding + 256 个视觉 embedding
L = 40 + 256 = 296
```

进入 Qwen LLM 的张量：

```text
inputs_embeds:  [B, 296, Dllm]
attention_mask: [B, 296]
position_ids:   [B, 296] 或按实现构造
```

Qwen Transformer 处理后：

```text
hidden_states: [B, 296, Dllm]
lm_head:       [B, 296, V]
```

训练或推理时，视觉 embedding 只作为条件输入；LLM 的输出词表仍然是文本词表。模型不会在视觉位置预测“第 17 个 patch 的类别”。

---

## 8. Qwen-VL 的图文交互：主要是 LLM 的 causal Self-Attention

原始 Qwen-VL 将视觉 embedding 插入语言序列，语言模型内部仍然使用 decoder-only Transformer 的因果 Self-Attention：

```text
序列：[系统提示, 图片视觉向量, 用户问题, assistant 前缀, 回答]
```

回答位置可以读取之前的视觉向量和问题 token：

```text
answer token 的 Q
        │
        ├── 读取图片位置的 K/V
        ├── 读取问题位置的 K/V
        └── 读取已生成回答的 K/V
```

注意：

- 这不是“视觉塔和文本塔做一次全量双塔对比”；
- 也不是必须有一层额外的图文 Cross-Attention；
- 在 LLM 中，视觉和文本已经成为同一序列的不同位置；
- 因果 mask 仍然约束未来文本不能被当前位置看到。

视觉适配器内部有 Cross-Attention，LLM 融合阶段通常依赖 causal Self-Attention。面试时把这两层交互区分开。

---

## 9. Qwen2-VL：为什么架构发生了变化？

原始 Qwen-VL 的固定 448 输入和固定 256 视觉 token 对所有图片使用相同预算：

- 宽高比差异大的图片需要裁剪或变形；
- 长图、文档和小字可能因固定缩放损失细节；
- 所有图片都使用同样数量的视觉 token，简单图片和复杂图片的预算不匹配。

Qwen2-VL 引入 **Naive Dynamic Resolution**：让输入图片按照内容和宽高比选择合适的视觉网格，得到可变数量视觉 token。

```text
原图
  │
  ▼
选择满足像素上下限和 patch 对齐约束的尺寸
  │
  ▼
视觉 patch 网格 [H_grid, W_grid]
  │
  ▼
Vision Transformer
  │
  ▼
2×2 spatial PatchMerger
  │
  ▼
Nimg = T × (H_grid/2) × (W_grid/2) 个视觉 token
```

`Nimg` 不再固定为 256。图片更大或更细长时，可能输出更多 token；部署中仍受 min/max pixels、上下文长度和显存约束。

---

## 10. Qwen2-VL 的视觉输入维度

### 10.1 3D patch / tubelet 输入

Qwen2-VL 同时支持图像和视频，因此视觉 patch 可以带时间维。用符号表示输入网格：

```text
grid_t：时间块数量
grid_h：高度方向 patch 数
grid_w：宽度方向 patch 数
```

一个视觉序列可以写成：

```text
[grid_t, grid_h, grid_w, Dv]
→ flatten
[grid_t × grid_h × grid_w, Dv]
```

对单张图片，`grid_t` 通常相当于 1 个有效时间位置；视频则随帧数增加。具体 temporal patch 设置取决于 checkpoint 和 processor。

### 10.2 PatchMerger 的维度变化

设视觉编码器输出：

```text
vision_hidden: [B, T, H, W, Dv]
```

空间 `2×2` 合并后，每 4 个邻近 token 拼接：

```text
[B, T, H, W, Dv]
→ [B, T, H/2, W/2, 4Dv]
```

再经过 LayerNorm 和 MLP 投影到 LLM hidden size：

```text
[B, T, H/2, W/2, 4Dv]
→ flatten
[B, T×H/2×W/2, Dllm]
```

以一张 `448×448` 图片、patch size 14 为例：

```text
视觉 patch 网格：32 × 32
2×2 merge 后：   16 × 16
输出视觉 token： 256
```

这个例子与原始 Qwen-VL 的 256 token 数相同，但来源不同：

```text
原始 Qwen-VL：1024 patch → 256 个 Query 聚合 token
Qwen2-VL：    1024 patch → 2×2 空间合并 → 256 个 token
```

如果动态分辨率得到 `H_grid=48、W_grid=32`：

```text
merge 后：24 × 16 = 384 个视觉 token
```

实际代码还会处理图像/视频网格、padding 和 dtype，维度必须以 `image_grid_thw` / `video_grid_thw` 为准。

---

## 11. 视觉 Transformer 的注意力：区分 Qwen2-VL 与 Qwen2.5-VL

不要把两代视觉注意力实现混写。Qwen2-VL 的视觉塔会依据 `grid_thw` 组织变长视觉序列，并用视觉注意力处理每个视觉输入；Qwen2.5-VL 的实现进一步显式区分局部窗口注意力和 `fullatt_block_indexes` 指定的全局注意力层。面试官如果只问 Qwen2-VL，不要无条件回答“它每层都是 window attention”。

抽象流程：

```text
Qwen2-VL：
patch / tubelet tokens + grid_thw
        │
        ├── 变长视觉序列组织
        ├── 视觉位置旋转
        └── 视觉 Transformer 注意力

Qwen2.5-VL 的常见实现：
        │
        ├── 大多数 block：window attention
        ├── 指定 fullatt_block_indexes：跨窗口全局 attention
        └── 多层重复
```

不要把 window attention 误解成最终的 token 剪枝：

- window attention 改变注意力连接范围（主要是 Qwen2.5-VL 的视觉实现）；
- PatchMerger 才改变视觉 token 数量；
- 动态分辨率改变网格大小；
- 三者作用不同，可以同时出现。

---

## 12. Qwen2-VL 的 M-RoPE 是什么？

普通 RoPE 通常用一条一维序列位置描述 token 位置，但视觉数据天然有至少三条坐标轴：

```text
时间 t、图像高度 h、图像宽度 w
```

Qwen2-VL 使用 Multi-modal Rotary Position Embedding（M-RoPE），将位置编码拆成时间、高度、宽度三个部分，使图像和视频 token 保留时空位置信息。

概念上，位置 id 可以组织为：

```text
position_ids: [3, B, L]

第 0 维：temporal position
第 1 维：height position
第 2 维：width position
```

对视频：

```text
同一帧内不同 h/w 有不同空间位置；
不同帧之间 t 递增；
```

对文本：

```text
文本 token 仍按语言序列递增，具体三轴如何填充由实现的 rope index 逻辑决定。
```

M-RoPE 的价值是让 LLM 的注意力区分：

- 图像中上下左右不同的 patch；
- 视频中前后帧的同一位置；
- 图像 token 与文本 token 在混合上下文中的相对次序。

**它不是把视觉坐标直接变成文字 token，也不是给每个物体分配 ID。** 它是应用在注意力 Q/K 旋转上的位置机制。

---

## 13. Qwen2-VL 的图片/视频特殊 token

Qwen2-VL 的实现通常会使用视觉开始/结束标记和图像/视频 padding 占位标记，常见概念包括：

```text
<|vision_start|>
<|image_pad|> 或图像占位位置
<|vision_end|>
```

视频还有对应的视频占位形式。不同 tokenizer 和仓库版本的具体字符串可能不同，因此阅读源码时优先看 tokenizer 配置和 processor。

典型数据流：

```text
文本 input_ids
    │
    ├── 特殊视觉占位位置
    └── 普通文本 token

image_embeds / video_embeds
    │
    ▼
在视觉占位位置替换或拼入连续视觉 embedding
    │
    ▼
inputs_embeds + position_ids + attention_mask
```

重要约束：

```text
视觉特征数量 = 视觉占位位置数量
```

如果 processor 算出的 `image_grid_thw` 与占位数量不一致，通常会出现 shape mismatch 或视觉信息错位。

---

## 14. 多图和视频是怎么组织的？

### 14.1 多图

多张图可以抽象为：

```text
文本前缀
<vision_start> image_1_tokens <vision_end>
文本问题
<vision_start> image_2_tokens <vision_end>
继续问题或回答
```

每张图可以有不同的动态网格和不同的 token 数。模型需要依靠顺序、边界标记和文本描述区分图片来源。

在 batch 中，常见做法是将不同图片的视觉特征按样本组织，或者先拼接视觉序列，再依据每张图的网格元数据计算位置和替换位置。不要假设所有图片都必须 resize 到同一尺寸后再共用一个固定 `Nimg`。

### 14.2 视频

视频可以抽象为：

```text
视频帧序列
→ temporal patch / tubelet
→ 视频视觉 Transformer
→ PatchMerger
→ [Nvideo, Dllm]
→ 和文本放入同一个 LLM 上下文
```

视频的关键不只是把多张图片简单 concat：

- 需要时间轴位置；
- 需要控制帧数与 token 预算；
- 需要保留帧间运动或变化信息；
- 需要让文本回答能读取相应视频位置。

Qwen2-VL 的 `video_grid_thw` 就是视觉处理器和模型之间的重要形状元数据。

---

## 15. Qwen-VL 中视觉和文本如何交互？

需要区分三个位置的交互：

### 15.1 视觉编码器内部

视觉 patch 在 ViT 的 Self-Attention 中交互，获得图像内部上下文；Qwen2.5-VL 还可通过窗口/全局 attention 控制空间连接范围。具体模型应以 checkpoint 的视觉配置和源码为准。

### 15.2 视觉连接器内部

原始 Qwen-VL 的 Query 通过 Cross-Attention 读取视觉特征；Qwen2-VL 的 PatchMerger 更像空间重采样和通道投影，并不等价于 256 个 Query 的 Cross-Attention。

### 15.3 LLM 内部

视觉 embedding 与文本 embedding 进入同一 decoder-only Transformer，使用 causal Self-Attention。回答 token 读取之前的视觉和文本上下文。

面试可直接回答：

> Qwen-VL 的视觉塔内部先做视觉建模，连接器完成视觉特征压缩或投影，最终图文融合主要由 Qwen decoder 的 causal Self-Attention 完成。原始版的连接器包含 Query Cross-Attention；Qwen2-VL 的动态版本主要通过视觉 ViT、PatchMerger 和 M-RoPE 处理时空视觉输入。

---

## 16. 推理流程：prefill 与 decode

### 16.1 Prefill

```text
1. processor 处理图片/视频，得到 pixel_values 和 grid_thw
2. vision encoder 产生视觉 hidden states
3. connector / PatchMerger 产生视觉 embedding
4. tokenizer 处理文本和视觉特殊标记
5. 将视觉 embedding 放入对应占位位置
6. 计算 M-RoPE / position_ids 和 attention mask
7. LLM 一次处理整个多模态 prompt
8. 取得最后有效位置 logits，生成第一个回答 token
```

### 16.2 Decode

```text
第 1 步：prompt + 视觉上下文 → 生成 token_1
第 2 步：KV cache + token_1 → 生成 token_2
第 3 步：KV cache + token_1 + token_2 → 生成 token_3
```

同一轮生成通常不会每一步重新运行视觉编码器。视觉 token 作为历史上下文的一部分写入 KV cache。

但视觉 token 越多：

- prefill 越慢；
- KV cache 越大；
- 后续 token 读取历史视觉 K/V 的成本越高。

动态分辨率是在视觉细节与语言上下文成本之间做预算，而不是免费增加分辨率。

---

## 17. 训练只需记住什么？

用户希望重点看架构，训练只保留这几句：

### 原始 Qwen-VL

```text
视觉编码器通常先使用预训练权重；
视觉语言适配器和语言模型通过图文数据适配；
Qwen-VL-Chat 再通过对话/指令数据获得多轮视觉交互能力。
```

### Qwen2-VL

```text
架构重点是动态分辨率、时空视觉 token、PatchMerger 和 M-RoPE；
具体 checkpoint 的冻结、解冻、数据比例和训练阶段不能凭家族名称猜，需看配置。
```

面试问到训练时，先回答“哪个模块在什么阶段更新”，不要把训练规模、GPU 数量和架构本身混为一谈。

---

## 18. Qwen-VL 和 LLaVA 的架构区别

| 对比项 | Qwen-VL / Qwen2-VL | LLaVA-1.5 |
|---|---|---|
| 视觉连接器 | 原始版 Query Cross-Attention；Qwen2-VL PatchMerger | 两层 MLP projector |
| 视觉 token 数 | 原始版常见固定 256；Qwen2-VL 动态可变 | 典型 336 输入保留 576 patch |
| 视觉分辨率 | Qwen2-VL 支持动态分辨率 | LLaVA-1.5 常见固定 CLIP 输入 |
| 时空位置 | Qwen2-VL 使用 M-RoPE，适合图像/视频 | 基础 LLaVA 主要依赖视觉位置 + LLM 序列位置 |
| LLM | Qwen 系列 | Vicuna/LLaMA 系列 |
| 融合方式 | 视觉 embedding 插入 Qwen 上下文，使用 decoder 融合 | 视觉 embedding 插入 LLM 序列，使用 causal Self-Attention |
| 多图/视频接口 | 原生设计中有 grid 元数据和时空组织 | 需要具体版本/扩展支持 |

一句话：

```text
LLaVA-1.5：固定视觉 patch + MLP 接口，结构极简。
Qwen2-VL：动态视觉网格 + PatchMerger + M-RoPE，时空输入更系统。
```

---

## 19. Qwen-VL 和 BLIP-2 / Q-Former 的区别

| 对比项 | BLIP-2 | 原始 Qwen-VL | Qwen2-VL |
|---|---|---|---|
| 视觉压缩 | Q-Former 的 Query Cross-Attention | 256 Query 视觉语言适配器 | PatchMerger 的空间合并与投影 |
| 视觉 token 数 | 常见 32 | 常见 256 | 动态可变 |
| LLM 接口 | 视觉 prefix 接 OPT/Flan-T5 | 接 Qwen decoder | 接 Qwen2 decoder |
| 位置重点 | 普通文本/视觉接口 | 固定视觉 token 接入 | M-RoPE 的时间/高度/宽度 |
| 架构关系 | Q-Former 本身是主要桥接模块 | 有 Query 适配器，但不等于复刻 Q-Former 全部设计 | 视觉压缩更偏 PatchMerger 和动态网格 |

面试不要只用“都有 Query”来判断相同。应继续问：

- Query 数量是否固定？
- 输出 token 是否从原 token 聚合得到？
- 是否支持动态分辨率与时间轴？
- LLM 的位置编码如何感知视觉坐标？

---

## 20. 高频面试问答

### Q1：Qwen-VL 的整体架构是什么？

**答：** 可以拆为视觉编码器、视觉语言连接器和 Qwen 自回归语言模型。图片经视觉编码器得到 patch 特征，连接器将其压缩或投影到 LLM hidden size，再与文本 embedding 组成多模态序列，Qwen decoder 通过 causal Self-Attention 生成回答。

### Q2：原始 Qwen-VL 为什么有 256 个视觉 token？

**答：** 典型原始架构使用 256 个可学习 Query。Query 通过 Cross-Attention 从大量视觉 patch 中提取信息，输出长度由 Query 数量决定。它不是图片中固定存在的 256 个 patch，也不代表 256 个物体。

### Q3：Qwen2-VL 的 token 数为什么不固定？

**答：** Qwen2-VL 使用动态分辨率，根据图片宽高比和像素预算选择视觉网格。patch 数随网格变化，经过空间 PatchMerger 后输出 token 数仍然随 `grid_t、grid_h、grid_w` 变化，通常由 `image_grid_thw` 或 `video_grid_thw` 描述。

### Q4：Qwen2-VL 的 PatchMerger 做什么？

**答：** 它把相邻的 `2×2` 空间视觉 token 拼接或重排，再通过 LayerNorm 和 MLP 投影到 LLM hidden size。它既降低空间 token 数，又完成视觉维度到语言维度的接口变换。它不是简单丢掉 3/4 token，也不是原始版的 256 Query Cross-Attention。

### Q5：Qwen-VL 的图像是不是先转成 caption 再给 LLM？

**答：** 推理时不是。图片直接经过视觉编码器和连接器变成连续视觉 embedding，插入 LLM 的输入上下文。caption 可以作为训练数据或中间监督，但不是每次推理的必经步骤。

### Q6：图片特殊 token 和视觉 embedding 是一回事吗？

**答：** 不是。特殊 token 是 tokenizer 中的离散占位或边界标记；视觉 embedding 是视觉网络生成的连续向量。模型要在占位位置用视觉 embedding 替换或拼接，二者的数量必须对齐。

### Q7：Qwen-VL 中图文融合用的是 Cross-Attention 还是 Self-Attention？

**答：** 原始连接器内部有 Query 到视觉特征的 Cross-Attention；视觉 embedding 进入 Qwen LLM 后，图文融合主要依靠 decoder 的 causal Self-Attention。不要只用一个词概括整个模型。

### Q8：Qwen2-VL 为什么需要 M-RoPE？

**答：** 图像和视频 token 有时间、高度、宽度三类位置。普通一维 RoPE 无法直接表达时空网格，M-RoPE 将位置拆成多轴，使 LLM 能区分不同帧以及图像中的不同空间位置。

### Q9：M-RoPE 是不是给每个图片 token 加一个二维坐标向量？

**答：** 更准确地说，它构造多轴位置索引，并将相应位置用于 RoPE 对 Q/K 做旋转。它不是简单把 `(h,w)` 拼成一个额外 embedding 后相加，也不是目标检测框坐标监督。

### Q10：Qwen2-VL 如何支持视频？

**答：** processor 将帧序列组织成带时间网格的视觉输入，视觉编码器处理 temporal/spatial patch，PatchMerger 输出视觉 token，`video_grid_thw` 记录时间、高度和宽度网格，M-RoPE 让 LLM 看到时间位置，之后与文本一起通过 causal Self-Attention 生成回答。

### Q11：Qwen-VL 的视觉编码器会在每次生成 token 时重复运行吗？

**答：** 通常不会。同一轮请求先完成视觉编码和 prompt prefill，视觉 K/V 进入 KV cache，后续 decode 只追加新的文本 token。新图片、图片参数或上下文改变时才需要重新计算对应视觉条件。

### Q12：Qwen-VL 为什么能处理多张图片？

**答：** 将每张图片编码成一段视觉 embedding，并用视觉边界标记、序列顺序和相应的网格元数据组织在同一上下文中。每张图可以有不同的动态 token 数，模型通过上下文位置和文本问题区分它们。

### Q13：Qwen2-VL 的动态分辨率是不是无限提高图片尺寸？

**答：** 不是。processor 会受最小/最大像素、patch 对齐、上下文窗口、显存和延迟约束。分辨率越高通常保留更多细节，但视觉 token、prefill 和 KV cache 成本也更高。

### Q14：Qwen-VL 与 LLaVA 最本质的架构差异是什么？

**答：** 典型 LLaVA-1.5 使用固定分辨率 CLIP patch 加两层 MLP，保留固定数量 patch token；原始 Qwen-VL 使用 Query 适配器压缩到固定 256 token，Qwen2-VL 则使用动态视觉网格、PatchMerger 和 M-RoPE，重点解决分辨率、空间和视频时序问题。

### Q15：Qwen-VL 与 BLIP-2 的视觉桥接器一样吗？

**答：** 都可能用 Query 从视觉特征中提取信息，但不能说完全一样。BLIP-2 的 Q-Former 是 BERT-like 多任务桥接器，包含 ITC、ITM、ITG 等阶段一目标；原始 Qwen-VL 的视觉语言适配器是其自身架构，Qwen2-VL 又转向动态 PatchMerger，不应只看“有 Query”下结论。

### Q16：视觉 token 的 labels 需要预测吗？

**答：** 通常不直接预测视觉 token。视觉 embedding 是给回答提供条件，文本生成 loss 主要作用于回答 token；视觉位置可用 ignore index，不代表视觉位置在 attention 中无效。

### Q17：如果把 image token 的 attention mask 设为 0 会怎样？

**答：** 取决于具体 mask 语义，但通常会让视觉位置被视为无效或无法被后续回答读取，从而破坏视觉条件。应区分“该位置不计算 label loss”和“该位置不能参与注意力”。

### Q18：为什么视觉 token 不能直接用词表 embedding？

**答：** 视觉 token 不是离散文本 id，视觉信息是连续特征。它们需要通过连接器产生与 LLM hidden size 相同的连续向量，再作为 `inputs_embeds` 或等价的视觉输入进入模型。

### Q19：Qwen2-VL 的 PatchMerger 会不会丢失空间位置？

**答：** 2×2 合并减少了空间 token 数，但视觉位置在视觉编码器的位置表示和后续 M-RoPE / grid 索引中仍要被建模。合并是信息压缩，不等于完全删除坐标；小目标和细粒度关系仍可能因压缩与分辨率限制而变难。

### Q20：为什么不能只看模型参数量判断 Qwen-VL 的视觉能力？

**答：** 视觉输入分辨率、动态 token 数、视觉塔宽度、连接器的信息瓶颈、位置编码、数据覆盖和 LLM 融合方式都会影响能力。相同 7B 语言参数量，视觉前端和 token 预算不同也会导致明显差异。

### Q21：面试官让你从源码找图片插入位置，先看哪些变量？

**答：** 先看 processor 输出的 `pixel_values`、`image_grid_thw` / `video_grid_thw`，再看视觉 encoder 输出、PatchMerger 输出、`image_token_id` / `video_token_id`、`vision_start` / `vision_end`、`inputs_embeds` 替换逻辑、`position_ids` 或 `rope_index`。不同版本命名不同，但数据流就是“网格元数据 → 视觉 token 数 → 占位替换 → 多轴位置”。

### Q22：如果只允许一句话解释 M-RoPE？

**答：** M-RoPE 把混合序列中的视觉位置从一条一维序列扩展为时间、高度、宽度三条位置轴，并将这些轴用于 RoPE，使同一个 LLM 能感知图片空间和视频时间。

### Q23：Qwen-VL 是否自带目标检测头？

**答：** 基础生成架构的主要输出是文本，不等于有一个独立的检测 head。它可以通过视觉问答或坐标相关指令输出框坐标，但坐标能力来自模型、数据和格式学习；若要得到严格检测指标，通常需要专门的检测解码与评估设计。

### Q24：Qwen-VL 的回答为什么会幻觉？

**答：** 视觉证据可能没有被编码或读取，图片分辨率/视觉 token 预算有限，训练数据含噪声，LLM 语言先验又很强。模型能生成流畅文本不代表每个事实都由图片支持。

---

## 21. 源码阅读：从 processor 跟到 LLM

阅读 Qwen-VL / Qwen2-VL 源码时按这条链追：

```text
1. processor / image_processor
   ├── resize、pixel_values
   ├── min_pixels / max_pixels
   ├── image_grid_thw / video_grid_thw
   └── 特殊 token 与占位数量

2. vision encoder
   ├── patch / tubelet embedding
   ├── 视觉 attention；Qwen2.5-VL 可见 window / full attention
   ├── vision hidden states
   └── 视觉位置处理

3. connector
   ├── 原始 Qwen-VL：Query Cross-Attention + MLP
   └── Qwen2-VL：PatchMerger、2×2 spatial merge、投影到 LLM hidden size

4. multimodal preparation
   ├── 找 image/video token 位置
   ├── 替换或拼接 visual embeddings
   ├── 构造 attention mask
   └── 构造 M-RoPE position_ids / rope index

5. language model
   ├── 混合序列 causal Self-Attention
   ├── KV cache
   └── lm_head 自回归生成
```

### 21.1 关键断点

```python
# 形状名是教学示意，具体项目命名可能不同。
pixel_values       # [num_images_or_patches, C, H, W]
image_grid_thw     # [num_images, 3]
vision_hidden      # [num_visual_tokens_before_merge, Dv]
visual_embeds      # [num_visual_tokens_after_merge, Dllm]
inputs_embeds      # [B, mixed_sequence_length, Dllm]
position_ids       # 可能含多轴位置信息
```

对于多图 batch，`pixel_values` 可能沿视觉样本维拼接，而文本 batch 仍按对话样本组织；需要依靠 grid 元数据和每条样本的图片数量切分，不能看到第一维就直接当作 batch size。

### 21.2 最常见的源码错误判断

- 看到 `image_grid_thw` 就以为它是图片尺寸；它通常是视觉 token 网格的时间/高度/宽度元数据。
- 看到 `image_token_id` 就以为图片只有一个 token；它可能是可展开的占位标记。
- 看到视觉 hidden size 与 LLM hidden size 不同，就以为不能拼接；中间还有 PatchMerger/projector。
- 看到 `position_ids` 是三维，就以为 batch、序列、hidden 三个维度；Qwen2-VL 中它可能表达多轴位置索引。
- 看到 `torch.cat` 就以为只是普通文本拼接；多模态实现还要同步 mask、label、rope index 和 cache。

---

## 22. 复杂度与上下文预算

设语言模型混合序列长度为：

```text
L_total = L_text + N_visual
```

LLM prefill 的全局注意力部分近似随 `L_total²` 增长：

```text
O((L_text + N_visual)² · Dllm)
```

因此视觉 token 数从 256 增加到 1024，未必只是视觉前端多做 4 倍工作；还会增加 LLM 的混合序列注意力和 KV cache。

Qwen2-VL 选择动态 token 数的目的不是无条件追求更高分辨率，而是在：

```text
细节保留 ↔ 视觉编码成本 ↔ LLM 上下文 ↔ 推理延迟
```

之间分配预算。

视频还多了时间维，若帧数翻倍而每帧网格不变，视觉 token 和上下文成本也会明显增长。因此实际部署需要限制总帧数、像素数或视觉 token 预算。

---

## 23. 和 TokenLearner、TopK、Q-Former 的架构关系

| 模块 | 如何减少 token | 输出是否为原 token | 位置特点 |
|---|---|---|---|
| Qwen 原始视觉适配器 | Query Cross-Attention 从视觉 token 聚合 | 新的 Query 表示 | 固定 Query 数 |
| Qwen2-VL PatchMerger | 邻近 `2×2` token 重排、合并、投影 | 新的合并表示 | 由网格和 M-RoPE 继续编码 |
| TokenLearner | 多张注意力图加权池化 | 新的软聚合 token | 可视化注意力图，坐标可能模糊 |
| TopKTokenPruner | 打分后选择分数最高 token | 是原 token 子集 | 需同步保留原位置 |
| LLaVA MLP | 逐 token 改变通道 | 保留原 token 数 | 主要依赖视觉位置与 LLM 位置 |

它们都能被称为视觉 token 处理模块，但不是同一个概念：

```text
Query / TokenLearner：学习聚合
PatchMerger：局部结构合并
TopK：离散选择
MLP：通道投影，不改变 token 数
```

---

## 24. 架构八股一页纸

```text
原始 Qwen-VL：
图片 448×448
→ OpenCLIP ViT-bigG
→ 大量 patch 特征（示例 1024）
→ 256 个 learnable Query Cross-Attention 聚合
→ MLP 映射到 Qwen hidden size
→ <img> 位置插入视觉 embedding
→ Qwen decoder causal Self-Attention
→ 自回归回答

Qwen2-VL：
图片/视频动态分辨率
→ temporal + spatial patch/tubelet
→ 视觉 ViT（窗口/全局注意力）
→ 2×2 PatchMerger
→ 可变数量视觉 token
→ image_grid_thw / video_grid_thw
→ M-RoPE（时间、高度、宽度三轴）
→ Qwen2 decoder
→ 自回归回答

最重要的区别：
原始 Qwen-VL：固定 Query 压缩。
Qwen2-VL：动态网格 + PatchMerger + M-RoPE。
```

---

## 25. 面试中的 30 秒架构回答

> 如果说原始 Qwen-VL，它的链路是 OpenCLIP ViT-bigG 提取视觉 patch，256 个可学习 Query 通过 Cross-Attention 聚合视觉信息，再用 MLP 映射到 Qwen hidden size，插入图片特殊标记对应的序列位置，最后由 Qwen decoder 的 causal Self-Attention 融合图文并生成回答。若面试官指的是 Qwen2-VL，则要补充动态分辨率、`image_grid_thw/video_grid_thw`、2×2 PatchMerger 和 M-RoPE；此时视觉 token 数是可变的，M-RoPE 负责时间、高度、宽度位置，而不是固定 256 Query。

---

## 26. 原论文与源码阅读入口

1. [Qwen-VL 原论文，arXiv:2308.12966](https://arxiv.org/abs/2308.12966)：重点看 visual receptor / visual-language adapter 和 256 Query。
2. [Qwen-VL 官方仓库](https://github.com/QwenLM/Qwen-VL)：看原始图片标记、视觉接口和 Qwen-7B 连接方式。
3. [Qwen2-VL 原论文，arXiv:2409.12191](https://arxiv.org/abs/2409.12191)：重点看 Naive Dynamic Resolution、M-RoPE、图像/视频网格。
4. [Qwen2-VL 官方仓库](https://github.com/QwenLM/Qwen2-VL)：看 processor、`image_grid_thw`、`video_grid_thw` 和推理接口。
5. [Qwen2-VL Hugging Face modeling code](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2_vl)：重点跟 `Qwen2VisionTransformer`、`PatchMerger`、`get_rope_index` 和 multimodal input preparation。
6. [Qwen2.5-VL 官方仓库](https://github.com/QwenLM/Qwen2.5-VL)：当面试题指向 2.5-VL 时，重新核对视觉塔、processor 和位置实现，不要照抄 Qwen2-VL 的所有细节。

配套复习：[BLIP-2](BLIP-2_流程与面试八股.md)、[Q-Former](Q-Former_面试八股.md)、[LLaVA](LLaVA_流程与面试八股.md)、[TokenLearner 与 TopKTokenPruner](TokenLearner与TopKTokenPruner_面试八股.md)。
