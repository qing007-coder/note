# RT-1：语言条件视觉编码、动作离散化与自回归变体

> 复习主线：**指令调制视觉 → TokenLearner 压缩 → Transformer → 离散动作 → 反量化执行。**
>
> 本篇先说明原版 RT-1，再完整展开“多相机 + 7 维动作 + BOS 右移训练 + KV-Cache 推理”的 RT-1 风格自回归变体。二者共享很多模块，但不能当成同一个原版配置。

## 1. 先分清原版与本篇自回归变体

RT-1（Robotics Transformer 1）是一种语言条件的模仿学习策略：读取短期图像历史与任务指令，为当前控制时刻预测动作，然后闭环执行。

| 项目 | 原论文 / 官方基础实现 | 本篇展开的自回归变体 |
|---|---|---|
| 视觉输入 | 6 帧图像历史，不把它说成固定的多相机配置 | 多相机、多帧历史 |
| 动作维度 | 7 个机械臂变量 + 3 个底盘变量 + 1 个模式变量，共 11 个 | 简化为 7 维机械臂动作 |
| 动作维度间的依赖 | 基础配置不使用前面动作值来条件化后续维度 | 按固定顺序逐标量自回归 |
| BOS、右移动作、KV-Cache | 不能直接说成原版默认机制 | 用它们实现标准 next-token 训练与增量推理 |
| time_position / camera_position | 官方有序列位置编码，不能断言含有这两个独立命名模块 | 显式加入时间身份与相机身份 embedding |

原论文的消融指出：增加动作自回归条件化没有提升效果，并使推理变慢超过 2 倍。因此，**decoder-only Transformer 不自动等于按动作维度自回归**。

官方源码的推理函数虽然有逐动作 token 的循环，但组装输入时把动作 embedding 清零，并使用特殊 mask，不能仅凭循环就断言后续动作依赖前面生成的动作值；该循环也不是 KV-Cache 实现。

## 2. 总流程：三个模块分别负责什么？

```text
任务指令 ── 文本编码器，如 USE ──> 语言向量 e
                                        │
图像历史 ── EfficientNet-B3 + FiLM ◀──────┘ 让视觉特征受任务条件影响
                       │ 空间特征
                       ▼
                 TokenLearner               减少视觉 token 数
                       │ 少量视觉 token
                       ▼
              位置编码 + Transformer        融合视觉历史并预测动作类别
                       │ 每个动作变量的分类分布
                       ▼
               256 档动作 token
                       │ 按各维范围反量化
                       ▼
              机器人动作 → 新图像 → 下次控制
```

原 RT-1 约以 3 Hz 闭环控制；这是原系统配置，不是所有复现必须使用的频率。它以大规模真实示范做行为克隆，不是靠在线强化学习从零探索。

## 3. 视觉编码：FiLM-EfficientNet → TokenLearner

### 3.1 贯穿全文的多相机形状例子

以下是自回归变体的讲解配置，原版视觉基线可把 `Nc` 设为 1。

| 符号 | 含义 | 示例 |
|---|---|---:|
| `B` | batch size | 2 |
| `T` | 图像历史长度，包含当前帧 | 6 |
| `Nc` | 相机数 | 2 |
| `H,W` | 编码前图像高、宽 | 300,300 |
| `D` | 视觉 / Transformer token 维度 | 512 |
| `S` | 每张图 TokenLearner 输出数 | 8 |
| `L=T*Nc*S` | 视觉前缀长度 | 96 |
| `Da` | 当前动作的标量维度数 | 7 |

```text
多帧多相机图像                         [B,T,Nc,3,H,W]
                                      [2,6,2,3,300,300]
       │ 合并 batch、时间、相机维
       ▼
待编码的独立图像                       [B*T*Nc,3,H,W] = [24,3,300,300]
       │ 共享 FiLM-EfficientNet-B3 + 特征投影
       ▼
空间特征                              [24,512,Hf,Wf]
       │ 以 RT-1 的 9×9 token 接口举例
       ▼
展平空间                              [24,81,512]
       │ TokenLearner：81 个位置 → 8 个新 token
       ▼
                                      [24,8,512]
       │ 恢复时间、相机维
       ▼
                                      [2,6,2,8,512]
       │ 加位置 / 身份 embedding，再展平
       ▼
视觉前缀 P                            [2,96,512]
```

论文报告的视觉 token 接口是 `9×9×512`，具体空间网格随预处理、裁剪和实现而变；不要把 512 误记成原始 EfficientNet-B3 在所有配置下的末层通道数。

原版单路 6 帧、每帧 8 个 token，对应 `6×8=48` 个视觉 token。这里的 96 来自额外的第二个相机。

### 3.2 FiLM：指令在 CNN 里就开始影响视觉

指令先经文本编码器得到句向量 `e=[B,512]`，为同一样本的各帧、各相机广播使用；不是给不同样本配错语言。

$$
\gamma=\operatorname{Linear}_{\gamma}(e),\qquad
\beta=\operatorname{Linear}_{\beta}(e),\qquad
f'=(1+\gamma)\odot f+\beta.
$$

直觉：指令决定哪些视觉通道需要增强、压低或平移，让“拿可乐”和“拿海绵”从视觉提取阶段就有所区别。

对某层 `[B*T*Nc,C,Hf,Wf]` 特征，缩放与偏置从 `[B*T*Nc,C]` 扩展到 `[B*T*Nc,C,1,1]`，在空间位置上广播。官方用零初始化条件投影，使初始 FiLM 接近恒等变换。

FiLM 是**通道调制**，不是文本 token 到图像 token 的 Cross-Attention。原版的语言主要通过 FiLM 注入，不能默认还向 Transformer 拼接了一串完整文本 token。

### 3.3 TokenLearner：不是 Top-K，也不只是平均池化

对每张图的 81 个空间特征，预测 8 张空间权重图，各自做加权池化：

$$
\alpha_{s,n}=\operatorname{softmax}_{n}(g(f)_{s,n}),\qquad
z_s=\sum_{n=1}^{81}\alpha_{s,n}f_n,\quad s=1,\ldots,8.
$$

直觉：每个输出 token 用一组学到的权重汇总空间信息；生成的是新的聚合 token，不是挑出 8 个原始位置。

```text
特征          [24,81,512]
权重 logits   [24,8,81] ── 空间 softmax ──> 权重 [24,8,81]
                                   │ 与特征加权求和
                                   ▼
新 token                               [24,8,512]
```

视觉 token 越多，后续注意力的计算与缓存越贵。TokenLearner 在进入 Transformer 前压缩序列，但也可能丢失细粒度信息；8 个 token 不是自动保证足够的理论值。

### 3.4 time_position、camera_position 分别告诉网络什么？

多相机变体可在视觉 token 上加入：

$$
p_{t,c,s}=z_{t,c,s}+e^{\mathrm{time}}_t+
e^{\mathrm{camera}}_c+e^{\mathrm{slot}}_s.
$$

直觉：内容向量描述“看到了什么”，时间与相机身份说明“什么时候、从哪个视角看到”，slot 或其他序列位置编码区分 token 槽位。

| 编码 | 作用 | 不要误解成 |
|---|---|---|
| `time_position` | 区分历史帧先后，帮助理解运动与状态变化 | 真实控制命令或扩散噪声时间 |
| `camera_position` | 区分头部、腕部等相机来源 | 相机标定矩阵或三维坐标本身 |
| token / 序列位置 | 区分压缩 token 及后续动作槽位 | TokenLearner 输出必然对应固定像素 |

这些 embedding 广播到相应的 token 上，相加不改变 `[B,T,Nc,S,D]` 形状；展平时须保持固定的时间—相机—slot 顺序。官方原版的序列位置编码与此类显式分解方案应区别表述。

## 4. 动作离散化：7 个标量，各做 256 类分类

以下简化机械臂动作顺序为：

```text
a = [dx, dy, dz, droll, dpitch, dyaw, gripper]   [B,7]
                 │ 每个维度按自己的范围离散化
                 ▼
q = [q1, q2, q3, q4, q5, q6, q7]               [B,7]
每个 qj ∈ {0,...,255}
```

以官方 tokenizer 的均匀量化约定为例，对范围 `[l_j,u_j]` 内的连续量：

$$
q_j=\left\lfloor255\cdot
\frac{\operatorname{clip}(a_j,l_j,u_j)-l_j}{u_j-l_j}\right\rfloor,
\qquad
\hat a_j=l_j+\frac{q_j}{255}(u_j-l_j).
$$

直觉：先裁剪并映射到 0～255 的类别，执行时再把类别映射回该动作维度的物理范围；不同维度的同一个 token ID 不代表同一个物理量。

其他实现可能用四舍五入或 bin 中心反量化，要和自己的真实代码一致。量化会带来精度损失，分类也不天然知道相邻类别在数值上更接近。

原版的模式变量是控制机械臂、底盘或终止等有限类别，官方 tokenizer 对这种离散变量直接处理；不要把它也说成连续物理量均匀切 256 档。

## 5. 原版训练 / 推理先记到什么程度？

```text
训练：图像历史 + 指令 → 视觉 token → Transformer
                              │
真实动作 → 动作类别标签 ────────┤ 各动作输出位置计算交叉熵

推理：当前视觉历史 + 指令 → 各动作类别预测 → 反量化 → 执行
```

原版基础配置不把前面预测的动作值作为后续维度的自回归条件，因此不应拿下一节的标准语言模型概率分解去描述它。训练中保留动作位置、做 next-position 对齐或使用因果 mask，也不等于使用了动作值自回归。

**下面第 6～8 节专讲用户关心的自回归变体**，完整展示 teacher forcing 与 KV-Cache。

## 6. 自回归变体训练：视觉前缀 + BOS + 右移动作

### 6.1 概率分解

$$
p(q_1,\ldots,q_7\mid P)=\prod_{j=1}^{7}p(q_j\mid P,q_{<j}).
$$

直觉：先根据视觉预测 dx，再结合已知 dx 预测 dy，逐维建立当前动作内部的依赖；7 个 token 是**同一个控制时刻的 7 个标量**，不是未来 7 步动作。

### 6.2 右移与标签逐位置对齐

```text
视觉前缀 P                BOS    q1    q2    q3    q4    q5    q6
                           │      │     │     │     │     │     │
各位置监督目标              q1     q2    q3    q4    q5    q6    q7
```

训练输入动作部分是 `[BOS,q1,...,q6]`，标签是 `[q1,...,q7]`；动作输入使用**真实示范 token**，这就是 teacher forcing。

```text
视觉前缀                         [B,L,D] = [2,96,512]
右移动作 token                   [B,7]
       │ token embedding
       ▼
动作输入 embedding               [B,7,D]
       │ 与视觉前缀拼接，加入匹配的位置编码
       ▼
Transformer 输入                 [B,L+7,D] = [2,103,512]
       │ 多层自注意力 + FFN
       ▼
取 7 个动作输入位置的隐特征       [B,7,D]
       │ 分类头 Linear(D → 256)
       ▼
动作 logits                      [B,7,256]
```

BOS 是动作开始标记，不占用 256 个动作 bin 中的任何一个。可以用独立 BOS embedding，或让输入 embedding 额外包含 BOS；**动作输出头仍可只有 256 类**。

### 6.3 为什么 teacher forcing 可以并行训练？

整条真实动作序列已知，可以一次送入 Transformer；mask 保证预测 qj 的位置只能读取视觉和 qj 之前的动作，不能看到正确答案或未来动作。

```text
预测 q1：可读 P、BOS
预测 q2：可读 P、BOS、q1
预测 q3：可读 P、BOS、q1、q2
...
```

本篇按整个序列的因果 mask 讲：视觉位置不读取后面的动作，动作位置可读取全部视觉前缀和此前动作。也可设计视觉前缀内部双向可见的 prefix mask，但必须训练、推理保持一致。

**并行的是一次前向里所有预测位置的计算，不是取消了动作之间的因果依赖。** 推理没有未来真实 token，仍需逐维解码。若不右移，或者标签和读出位置错位，可能把答案泄漏给模型。

### 6.4 交叉熵 loss

$$
\mathcal L=-\frac{1}{7B}\sum_{b=1}^{B}\sum_{j=1}^{7}
\log p_\theta(q_{b,j}\mid P_b,q_{b,<j}).
$$

直觉：提高正确动作类别的概率；只在 7 个目标位置计算动作损失，视觉前缀不作为动作标签监督。填充、无效样本另用 loss mask 屏蔽。

## 7. 自回归变体推理：prefill 一次，再逐标量生成

### 7.1 prefill 与第一维预测

```text
图像历史 + 指令 → 编码、FiLM、TokenLearner → P [B,L,D]
                           │ Transformer prefill 一次
                           ▼
                  各层视觉 K、V 缓存，长度 L
                           │ 输入 BOS
                           ▼
                  缓存长度 L+1；logits → 选 q1
```

视觉前缀可以和 BOS 合并为一次 prefill。上图拆开是为了看清：**输入 BOS 的隐状态预测 q1，q1 此时只是从 logits 选出的 ID。**

### 7.2 完整的缓存时间线

| 本轮实际送进网络的输入 | 前向后缓存长度 | 本轮从 logits 选出的 token |
|---|---:|---|
| 视觉前缀 P | `L` | 此处先不读动作 |
| BOS | `L+1` | q1 |
| q1 | `L+2` | q2 |
| q2 | `L+3` | q3 |
| q3 | `L+4` | q4 |
| q4 | `L+5` | q5 |
| q5 | `L+6` | q6 |
| q6 | `L+7` | q7，当前 7 维动作齐全 |

最后 q7 如果不再用于后续 token 预测，可以直接反量化执行，**无需专门再前向一次只为缓存 q7**。

动作选择可以用 argmax 或按分布采样；训练、评估与执行需要明确使用哪种策略。自回归表示提供跨维依赖，但会带来串行延迟和错误传播。


## 8. 与 ACT、扩散 / FM 的区别

| 项目 | 本篇 RT-1 风格自回归变体 | ACT | 扩散 / Flow Matching 动作策略 |
|---|---|---|---|
| 输出表示 | 逐标量离散类别 | 连续动作块 | 连续动作块 |
| 一次生成什么 | 当前时刻的多维动作 | 未来多步动作 | 未来多步动作 |
| 主训练目标 | 动作分类交叉熵 | 动作 L1 + KL | 噪声 / 干净动作 / 速度等回归 |
| 主要串行开销 | 动作维度逐 token 解码 | 通常并行预测动作槽位 | 多轮去噪或 ODE 积分 |
| 条件如何进入 | 指令经 FiLM 影响视觉，视觉 token 进入 Transformer | 图像、状态、潜变量进入条件生成网络 | 全局条件调制或 Cross-Attention 等 |

分类能保留多个动作 bin 的概率峰，避免直接 MSE 回归均值的某些问题，但量化存在精度代价，argmax 不保证探索所有模式。原版非动作自回归配置与本表第一列的串行机制要区别记忆。

## 9. 面试问题：看完能复述流程

| 问题 | 精炼回答 |
|---|---|
| 1. RT-1 的完整流程是什么？ | 指令句向量通过 FiLM 调制 EfficientNet-B3，视觉特征经 TokenLearner 压缩，再由 Transformer 输出离散动作类别并反量化执行。 |
| 2. 原版一定是多相机、7 维、自回归吗？ | 不是；原版用 6 帧图像历史，完整动作含 11 个变量，基础配置没有动作值自回归条件化。本篇多相机 7 维 AR 是明确标注的变体。 |
| 3. FiLM 的语言条件怎么作用？ | 语言向量产生逐通道缩放与偏置，广播到空间特征，视觉提取阶段就受任务影响；它不等于 Cross-Attention。 |
| 4. TokenLearner 和 Top-K 有何区别？ | TokenLearner 对所有空间特征做多组学习到的加权池化，产生新 token；Top-K 直接保留部分原 token。 |
| 5. time_position、camera_position 干什么？ | 分别提供帧先后和相机身份；相加不改 token 形状，相机身份不等于标定信息，也不是原版固定命名模块。 |
| 6. 256 bin 是整个 7 维动作共 256 类吗？ | 不是；每个连续维度各有 256 个可选值。7 个标量构成一个动作，分类头可共享，槽位与上下文区分维度。 |
| 7. BOS 与右移怎么对齐？ | 输入 `[BOS,q1,...,q6]`，监督 `[q1,...,q7]`；BOS 不占动作 bin，输出头仍可为 256 类。 |
| 8. teacher forcing 为什么能并行？ | 所有真实历史 token 已知，一次前向计算所有位置；因果 mask 保证每个位置只看允许的前缀。 |
| 9. 训练并行，推理为什么不能一次出完 AR 动作？ | 推理缺少真实历史动作 token，需要先选出 q1 再预测依赖它的 q2，依次继续。 |
| 10. 新选出的 q1 为什么还不在缓存里？ | q1 是本轮 logits 之后选出的 ID，尚未参与 Transformer 前向；下一轮喂入后才有 q1 对应的逐层 K/V。 |
| 11. 生成 7 个 token 后，最后一个必须再喂一次吗？ | 若当前动作已经结束则不必；只有还要基于它继续预测时，才需要让它参与前向。 |
| 12. KV-Cache 与视觉特征缓存有什么区别？ | 前者缓存 Transformer 各层 K/V；后者缓存 CNN / TokenLearner 的结果。缓存图像 token 不等于已经缓存各层注意力表示。 |
| 13. 新图像到来能直接沿用旧 KV 吗？ | 不能无条件沿用；视觉前缀变化可能影响位置与各层表示，需重新 prefill 或有明确的流式缓存设计。 |
| 14. decoder-only 就是动作自回归吗？ | 不是；要看输入动作值是否进入网络、mask、输出位置及训练依赖，不能只看结构名或代码中的循环。 |

## 10. 一句话与两条流程速记

**原版 RT-1：语言调视觉、TokenLearner 压 token、Transformer 预测离散动作；不要自动补上动作值自回归。**

```text
AR 变体训练：多帧多相机 → FiLM-CNN → TokenLearner → 视觉前缀
             + [BOS,真实 q1...q6] → 因果并行前向 → [q1...q7] 交叉熵。

AR 变体推理：视觉 prefill → 输入 BOS 选 q1 → 输入 q1 选 q2 → ... → 输入 q6 选 q7
             → 7 维反量化执行；被选出的 token 要参与下一次前向，才会进入自己的 KV 缓存。
```

## 参考

- [RT-1 原论文：Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)
- [项目页：原版视觉架构与动作组成](https://robotics-transformer1.github.io/)
- [官方 TransformerNetwork：mask、动作 embedding 清零、训练和推理](https://github.com/google-research/robotics_transformer/blob/master/transformer_network.py)
- [官方图像 tokenizer：81 个空间 token 与 TokenLearner](https://github.com/google-research/robotics_transformer/blob/master/tokenizers/image_tokenizer.py)
- [官方动作 tokenizer：量化与反量化](https://github.com/google-research/robotics_transformer/blob/master/tokenizers/action_tokenizer.py)
- [官方 FiLM 通道调制](https://github.com/google-research/robotics_transformer/blob/master/film_efficientnet/film_conditioning_layer.py)
- [已有笔记：TokenLearner 与 TopKTokenPruner](../pytorch/TokenLearner与TopKTokenPruner_面试八股.md)
