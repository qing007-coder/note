# RT-2：把机器人动作变成 VLM 的语言 token

> **一句话**：RT-2 不额外设计独立的动作分类头，而是把连续机器人动作离散成词表中的 token，让预训练 VLM 直接用原来的语言建模方式预测动作。
>
> 本篇按本地 `vla_brain/models/rt2` 实现整理，重点看网络结构、动作与 token 的关系、训练和推理流程。

## 1. RT-2 想解决什么问题？

普通机器人策略通常只在机器人示范数据上训练，模型知道“怎么移动”，但对互联网图文预训练得到的开放世界知识利用不足。例如，指令要求把物体放到某个数字、图标或语义位置附近，单纯模仿学习可能没有见过这种组合。

RT-2 的思路是：

```text
预训练 VLM 已经会理解图像和语言
                  │
                  ▼ 继续用机器人数据微调
图像 + 指令 ──> VLM ──> 语言 token / 动作 token
                                      │
                                      ▼
                               还原成机器人动作
```

这样，机器人动作就进入了 VLM 已经熟悉的“文本生成”接口。图文数据提供语义泛化，机器人数据提供动作落地和控制对应关系。

## 2. 总体网络架构

本地实现的主要模块是：

```text
多相机图像
    │
    ▼
视觉编码器 ViT / SigLIP
    │ [B, V×Nv, Dv]
    ▼
多模态 projector
    │ [B, V×Nv, Dl]
    │ 加 camera embedding
    ▼
视觉 token 前缀
    │
指令 token ── 语言 embedding ─────────────┐
                                           ▼
                          视觉 token + 指令 token
                                           │ prefix attention
                                           ▼
                           Gemma / Llama 风格 decoder
                                           │
                                           ▼
                              共享 lm_head [B,N,Vocab]
                                           │
                                           ▼
                                 动作词表 token
```

本地代码的继承关系可以这样记：

```text
RT2Policy
   └── VocabularyActionPolicy
          └── VisionLanguageModel
                 ├── vision
                 ├── projector
                 └── language + lm_head
```

`RT2Policy.from_config()` 会构造视觉塔、projector 和语言 decoder，并把原文本词表后面追加 256 个动作 token。`from_vlm()` 则把已有 VLM 接入动作策略，仍然复用同一套视觉、语言和输出层。

## 3. 视觉到语言空间

假设有 `V` 个相机，每个视觉塔输出 `Nv` 个 patch token，视觉维度为 `Dv`，语言模型隐藏维度为 `Dl`：

```text
相机 v 的图像                         [B,3,H,W]
        │ ViT / SigLIP
        ▼
视觉 patch 特征                       [B,Nv,Dv]
        │ projector：Linear 或 MLP
        ▼
语言空间视觉 token                   [B,Nv,Dl]
        │ 加该相机的 camera_embedding
        ▼
多相机拼接                            [B,V×Nv,Dl]
```

本地 `VisionLanguageModel.encode_images()` 按 `camera_names` 的固定顺序处理相机，再沿 token 维拼接。`camera_embedding[index]` 用来告诉语言模型某个 token 来自哪个相机。

这里的 projector 只做维度和表示空间的对齐：

$$
z_{\mathrm{lang}}=Wz_{\mathrm{vision}}+b.
$$

它把视觉特征从 `Dv` 维变成语言模型可以接收的 `Dl` 维，但维度相等不代表语义已经自动对齐，projector 和 VLM 仍需要训练或加载匹配的预训练参数。

## 4. 动作如何变成语言 token？

本地实现固定使用 7 维动作接口：

```text
a = [dx, dy, dz, roll, pitch, yaw, gripper]
```

### 4.1 先归一化

`ActionNormalizer` 把每个动作维度按自己的上下界映射到 `[-1,1]`：

$$
a^{\mathrm{norm}}_j=
2\frac{a_j-l_j}{u_j-l_j}-1.
$$

直觉：不同动作维度的量纲和范围不同，先统一尺度，才能使用同一套分桶逻辑。

### 4.2 每个维度分成 256 个 bin

对归一化后的每个标量，映射到 `0,...,255`：

$$
q_j=\min\left(255,
\left\lfloor\frac{a^{\mathrm{norm}}_j+1}{2}\times256\right\rfloor\right).
$$

于是一个单步动作的形状变化是：

```text
连续动作                         [B,7]
归一化动作                       [B,7]
bin 编号                         [B,7]，每个值属于 0...255
```

如果训练的是动作序列 `[B,T,7]`，则每一步的 7 个动作标量都会变成 token：

```text
[B,T,7] → [B,T,7] 的 bin 编号 → flatten → [B,T×7]
```

### 4.3 bin 编号映射到真实 VLM 词表

RT-2 的关键不是单独建立 `action_head: hidden → 256`，而是建立一张映射：

```text
动作 bin 0   ──> 词表 token id_0
动作 bin 1   ──> 词表 token id_1
...
动作 bin 255 ──> 词表 token id_255
```

本地 `VocabularyActionCodec` 保存这 256 个 `token_ids`：

```text
VLM logits                       [B,N,Vocab]
        │ 取动作 token 对应的词表位置
动作 token logits                [B,7,256]  # 语义上的动作类别空间
        │ argmax / sampling
生成的词表 token id              [B,7]
        │ codec.decode
归一化动作                       [B,7]
        │ ActionNormalizer.unnormalize
原始机器人动作                   [B,7]
```

所以“256 类”描述的是每个动作标量可选的离散值数量；不是 7 个维度合起来只有 256 个动作，也不是为 7 个维度分别建立 7 个独立网络头。

### 4.4 为什么不能直接把数字字符串当作动作？

具体 VLM 的 tokenizer 可能把 `128` 拆成多个 token。RT-2 需要保证一个动作 bin 对应一个可控的词表 token。因此本地 `from_pretrained()` 会优先检查 `0...255` 是否能各自编码成单 token；如果不满足，就注册 `<action_000>` 到 `<action_255>` 这类特殊 token，并扩展 embedding 与 `lm_head`。

这一步的目标是得到**一维动作标量对应一个词表 token**，从而可以使用标准 next-token prediction。

## 5. 训练流程：机器人数据和 VQA 共用 CE

### 5.1 机器人样本的序列

本地 `pack_multimodal_sequence()` 将视觉和文字放入同一个语言序列：

```text
prompt 中的 BOS / 指令 token
              │
              ▼
视觉 token + 指令 token + 动作答案 token
```

更具体地，代码会把 prompt 的第一个 token 放在前面，再拼视觉 token，再拼 prompt 的剩余 token，最后拼动作答案：

```text
[prompt_BOS] + [visual tokens] + [prompt_remaining] + [action tokens]
```

答案 token 位置才有监督 label，视觉和指令位置的 label 设为 `-100`，不参与交叉熵。

### 5.2 prefix attention

RT-2 使用 prefix attention：

```text
视觉 / 指令前缀：相互可见
动作答案序列：因果可见，只读取前缀和之前的动作 token
```

直觉：视觉和指令需要共同理解，答案 token 则按语言模型方式逐个预测。

### 5.3 next-token cross entropy

语言模型第 `i` 个位置的 logits 用来预测第 `i+1` 个 token：

$$
\mathcal L_{\mathrm{CE}}
=-\sum_i \log p_\theta(y_{i+1}\mid y_{\le i},\text{image},\text{instruction}).
$$

本地代码中，`logits[:, :-1]` 对齐 `labels[:, 1:]`，只对动作答案或 VQA 答案的有效位置计算 loss。

### 5.4 图文与机器人数据联合微调

本地 `co_finetune_loss()` 同时计算两类 loss：

```text
机器人 batch → 动作 token CE ──┐
                              ├─ robot_weight × robot_loss
VQA batch    → 文本 token CE ──┘  + (1-robot_weight) × vqa_loss
```

这样同一套 VLM 同时学习：

- 图像与语言的语义理解；
- 图像、指令到低层动作 token 的映射。

## 6. 推理流程：VLM 生成动作 token

```text
多相机图像 + 指令
       │ 视觉塔 + projector
       ▼
视觉 / 指令前缀                     [B,Nprefix,D]
       │ prefill，一次性进入语言 decoder
       ▼
每层 KV-Cache                        [B,H,Nprefix,Dh]
       │
       ▼
读取最后位置 logits                  [B,1,Vocab]
       │ 只允许 256 个动作 token
       ▼
生成第 1 个动作 token
       │ token embedding + 增量 decoder
       ▼
更新 KV-Cache，生成第 2 个 token
       │
       ▼
重复直到 7 个动作 token
       │ codec.decode + 反归一化
       ▼
连续机器人动作                     [B,7]
```

本地 `predict_action()` 调用 `generate(max_new_tokens=7, allowed_token_ids=codec.token_ids)`，因此一次闭环控制输出 7 个动作标量。`generate()` 的每轮输入是上一步刚生成的 token；该 token 只有再次经过 embedding 和 Transformer 各层，才会产生并追加自己的 K/V。

动作 token 生成后，`VocabularyActionCodec.decode()` 先把词表 ID 找回对应的 bin 中心，再交给 `ActionNormalizer.unnormalize()` 还原到机器人实际量纲。

## 7. RT-2 的优点与局限

| 优点 | 局限 |
|---|---|
| 直接复用 VLM 的视觉、语言表示和词表预测框架 | 动作分桶产生量化误差 |
| 网络结构统一，动作输出不需要独立 action head | 逐 token 自回归生成，动作维度越多延迟越高 |
| 图文任务和机器人任务可以共享主干进行 co-finetuning | 语言模型的 token 顺序不一定天然适合连续控制 |
| 互联网知识有机会迁移到新物体、新指令和语义组合 | 低层动力学和精细连续控制仍受机器人数据覆盖限制 |
| 允许对动作 token 做受限解码，降低生成非法 token 的概率 | 256 个 bin 的离散分辨率与动作边界需要按机器人配置设计 |

RT-2 的核心收益是**语义泛化迁移**，不是把离散 token 自动变成高精度控制器。词表 token 提供统一接口，真正的动作质量仍取决于机器人数据、归一化范围、动作顺序和闭环执行频率。

## 8. 和 RT-1 的关系

```text
RT-1：专门设计视觉编码器 + Transformer + 动作 token 预测
RT-2：预训练 VLM 作为底座，把动作 token 接入原有语言生成接口
```

RT-2 延续了 RT-1 的动作离散化思想，但把“动作 token 预测器”放入更大的预训练视觉语言模型中。RT-1 的视觉编码可以是 EfficientNet、FiLM、TokenLearner；本地 RT-2 则使用 ViT / SigLIP 风格视觉塔加 projector，并直接连接语言 decoder。

## 9. 面试问题

| 问题 | 精炼回答 |
|---|---|
| 1. RT-2 的核心思想是什么？ | 把机器人动作离散成语言词表 token，让预训练 VLM 用原来的 next-token 预测框架直接输出动作。 |
| 2. RT-2 为什么能迁移 VLM 的图文知识？ | 图像和指令仍经过预训练 VLM 的视觉、投影和语言主干，机器人数据只把输出空间扩展到动作 token，并进行联合微调。 |
| 3. RT-2 有独立的 action head 吗？ | 本地实现没有独立动作头，动作 token 使用语言模型共享的 `lm_head` 输出。 |
| 4. 视觉特征如何进入语言模型？ | 视觉塔输出 patch token，projector 把 `Dv` 映射到 `Dl`，再作为语言 decoder 的视觉前缀；多相机还加 camera embedding 并拼接。 |
| 5. 一个 7 维动作对应多少 token？ | 单步对应 7 个 token，每个标量先归一化，再映射到 256 个 bin，最后映射到 256 个词表 token。 |
| 6. 256 bin 是 256 个完整动作类别吗？ | 不是；每个动作维度各有 256 个离散值，7 个 token 合起来表示一个动作。 |
| 7. RT-2 的训练 loss 是什么？ | 标准 next-token cross entropy，只在动作答案或文本答案的有效位置计算。 |
| 8. 为什么可以图文和机器人数据联合训练？ | 两类任务都被组织成“输入多模态上下文，输出 token 序列”，共用同一个 VLM 和同一个 CE 接口。 |
| 9. 训练时动作 token 能并行计算吗？ | 可以，已知整段答案 token 时一次前向计算所有位置；因果 / prefix mask 防止读取未来答案。 |
| 10. 推理时为什么要逐个生成 7 个 token？ | 每轮只能看到已经生成的前缀，生成 q1 后才能把 q1 作为输入预测 q2，直到动作维度全部完成。 |
| 11. KV-Cache 存的是什么？ | 每层注意力的 K/V 张量，不是词表 ID；新 token 生成后，下一轮输入它，才会产生自己的 K/V。 |
| 12. RT-2 的连续动作如何恢复？ | 词表 ID → 动作 bin 中心 → `[-1,1]` 归一化动作 → 按每维边界反归一化为机器人动作。 |
| 13. RT-2 的主要缺点是什么？ | 量化误差、自回归延迟、动作 token 顺序限制，以及机器人动力学泛化仍依赖真实示范覆盖。 |
| 14. RT-2 和 RT-1 最大区别是什么？ | RT-1 主要是专用机器人 Transformer；RT-2 把动作作为语言 token 接入预训练 VLM，重点利用图文知识迁移。 |

## 10. 一句话复述

**RT-2 把“机器人控制”改写成“视觉语言模型生成动作词”，用 256-bin 动作 token、共享 lm_head 和交叉熵，把图文预训练的语义能力迁移到闭环机器人控制。**

## 参考

- [RT-2 原论文：Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)
- [本地 RT-2 策略实现](D:/work/embodied-intelligence/vla_brain/models/rt2/model.py)
- [本地动作归一化与词表 codec](D:/work/embodied-intelligence/vla_brain/models/actions.py)
- [本地 VLM 序列、prefix attention 与 KV-Cache](D:/work/embodied-intelligence/vla_brain/models/vlm/model.py)
