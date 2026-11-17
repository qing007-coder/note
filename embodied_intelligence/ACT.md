# ACT：CVAE + Transformer 一次预测一个动作块

> ACT = Action Chunking with Transformers。重点是从多相机图像、关节状态、潜变量，到未来 K 步动作的流程。
>
> 前置：[VAE](VAE.md)。ACT 是机械臂操作路线，不是 GNM → ViNT → NoMaD 导航路线中的前一代。

## 1. ACT 要解决什么？

如果每一时刻只预测下一步动作，机器人就需要不断做局部决策，误差会沿执行过程累积，细粒度操作也容易抖动。

ACT 一次预测未来 `K` 步，称为 **action chunk（动作块）**：

```text
逐步策略：当前观测 ──> a_t
ACT：    当前观测 ──> [a_t, a_(t+1), ..., a_(t+K-1)]
```

三个组成部分分别承担不同工作：

| 组成 | 主要作用 |
|---|---|
| CVAE | 训练时用潜变量解释示范中的行为差异 |
| Action chunking | 一次建模一段连续动作，减少逐步决策的负担 |
| Temporal ensembling | 融合不同时刻对同一执行时刻的预测，平滑控制 |

## 2. 全文的形状例子

以下用 ALOHA 风格的双臂输入举例；相机数量、动作块长度等由任务配置决定。

| 符号 | 含义 | 示例 |
|---|---|---:|
| `B` | batch size | 2 |
| `Nc` | 相机数 | 4 |
| `H, W` | 图像高、宽 | 480, 640 |
| `Dq` | 本体状态维度 | 14 |
| `Da` | 每步动作维度 | 14 |
| `K` | 动作块长度 / action query 数 | 100 |
| `D` | Transformer 隐藏维度 | 512 |
| `Dz` | 潜变量维度 | 32 |

这里 14 维对应双臂的关节和夹爪量。动作通常是目标关节位置及夹爪指令，不是固定的末端六维位姿；状态、动作训练前需要按数据统计量归一化。

```text
images  [B, Nc, 3, H, W] = [2, 4, 3, 480, 640]
qpos    [B, Dq]          = [2, 14]
actions [B, K, Da]       = [2, 100, 14]  # 只在训练重建时提供
is_pad  [B, K]           = [2, 100]      # True 表示填充动作
```

## 3. 先分清两个 Encoder

ACT 里容易混淆的不是形状，而是两条不同的编码路径：

```text
① CVAE 后验 Encoder（训练重建时使用）
   qpos + 示范动作块 ──> mu, logvar ──> z

② 条件动作生成网络（训练和推理都使用）
   图像 + qpos + z ──> Transformer Encoder / Decoder ──> 预测动作块
```

CVAE 的“Decoder”指整个条件动作生成网络，里面又包含视觉 Backbone、Transformer Encoder 和 Transformer Decoder。

## 4. 路径一：训练时从示范动作推断 z

官方实现的后验为 `q_phi(z | qpos, A)`，**这一支不读相机图像**。

```text
示范动作 A [B, K, 14] ── Linear(14 → 512) ──> [B, K, 512]
关节状态 q [B, 14]    ── Linear(14 → 512) ──> [B, 1, 512]
可学习 CLS token                           [B, 1, 512]
                                                │
                          按序列维拼接 [CLS, q, A]
                                                ▼
                                   [B, K+2, 512]
                                   [2, 102, 512]
                                                │ 加位置编码、padding mask
                                                ▼ Transformer Encoder
                                   [B, K+2, 512]
                                                │ 取 CLS 对应输出
                                                ▼
                                   [B, 512]
                                                │ Linear(512 → 64)
                                                ▼
                                   [B, 64]
                                      ├─ mu     [B, 32]
                                      └─ logvar [B, 32]
                                                │ 重参数化
                                                ▼
                                   z [B, 32]
```

注意是 **`K+2`** 个 token：一个 CLS、一个状态、K 个动作。官方源码部分注释写成 `seq+1`，应以实际拼接为准。

整段动作块共享一个 `z`，有利于用一个潜变量描述这段示范的整体风格。填充的动作位置由 `is_pad` 屏蔽，CLS 和状态 token 不屏蔽。

## 5. 路径二：图像变成空间 token

每个相机分别经过共享的 ResNet-18 视觉 Backbone。以未启用 dilation、总步长 32 的配置举例：

```text
单个相机图像                    [B, 3, 480, 640]
           │ ResNet-18
           ▼
空间特征图                      [B, 512, 15, 20]
           │ 1×1 Conv，投影到 D=512
           ▼
投影特征图                      [B, 512, 15, 20]

4 个相机沿特征图宽度拼接         [B, 512, 15, 80]
           │ flatten 空间维 + 转置
           ▼
视觉 token                      [B, 1200, 512]
```

`1200 = Nc × Hf × Wf = 4 × 15 × 20`。这里一个 token 对应一个相机特征图上的空间位置，**不是一张图只变成一个 token**。

图像使用二维位置编码，关节状态和潜变量使用额外的位置 embedding。原版实现把各相机的特征及其位置编码一起拼接。

## 6. 加入状态、z，再用 K 个 query 生成动作块

```text
视觉 token                     [B, 1200, 512]
qpos [B, 14] ── Linear ──>      [B, 1, 512]
z    [B, 32] ── Linear ──>      [B, 1, 512]
                    │
                    ▼ 拼接：[z token, qpos token, 视觉 token]
条件序列                       [B, 1202, 512]
                    │ Transformer Encoder
                    ▼
memory                         [B, 1202, 512]
                    │
                    │ 供 Cross-Attention 读取
                    ▼
K 个可学习 action query        [K, 512] → 扩展为 [B, K, 512]
                    │ Transformer Decoder
                    │ query 自注意力 + 读取 memory
                    ▼
动作隐特征                     [B, K, 512]
                    │ Linear(512 → 14)
                    ▼
预测动作块 A_hat               [B, K, 14] = [2, 100, 14]
```

每个 query 对应一个未来时间槽位。Decoder 并行输出 K 步，**不需要先生成第一步，再把第一步喂回去才能生成第二步**。

query embedding 提供槽位身份；官方实现的 Decoder 内容输入 `tgt` 从全零开始。源码 Transformer 常用 `[序列长度, B, D]`，上图统一用 `[B, 序列长度, D]` 方便阅读；转置不改变语义。

## 7. 训练完整流程与损失

```text
当前图像、qpos、示范未来动作 A
        │
        ├─ qpos + A ── 后验 Encoder ──> mu, logvar ──> 采样 z
        │                                                        │
        └─ 图像 + qpos ────────────────────────────────────────────┤
                                                                 ▼
                                                     条件动作生成网络
                                                                 │
                                                                 ▼
                                                       A_hat [B,K,Da]
                                                                 │
                                                       L1(A_hat, A)

mu, logvar ──> KL(q(z | qpos,A) || N(0,I))
```

$$
\mathcal L=\mathcal L_{\mathrm{L1,masked}}
+\beta D_{\mathrm{KL}}\!\left(q_\phi(z\mid qpos,A)\|\mathcal N(0,I)\right).
$$

L1 比较预测动作与示范，填充位置不贡献误差。KL 约束潜空间。官方模型还定义了 `is_pad_head`，但公开 `ACTPolicy` 的主损失由 L1 与 KL 构成。

## 8. 推理：没有未来动作，官方固定 z=0

```text
当前多相机图像 [B,Nc,3,H,W] + qpos [B,14]
                                  │
固定 z = zeros([B,32]) ────────────┤
                                  ▼
                         条件动作生成网络
                                  ▼
                         A_hat [B,100,14]
                                  │ 反归一化
                                  ▼
                         未来 100 步目标动作
```

推理不运行后验动作 Encoder，因为此时没有示范的未来动作。通用 CVAE 可以从先验采样，但 **ACT 官方默认实现直接取 `z=0`**。

这也解释了为什么不能把 ACT 简化成“推理时随机采样 z 来挑左右”。CVAE 为训练提供了表达示范变化的方式；固定潜变量、模型是否有效使用 `z`、数据覆盖等因素仍会影响歧义场景下的表现。

## 9. 预测一个块之后怎么执行？

### 方式一：按块执行

预测 K 步，执行一段，再读取新观测预测下一块。它减少网络查询次数，但长时间开环执行会降低对环境变化的响应能力。

### 方式二：每步重新预测 + 时间集成

每步都生成新动作块，同一个物理时刻会被多个块覆盖：

```text
t=0 的预测： a_0|0  a_1|0  a_2|0  a_3|0 ...
t=1 的预测：        a_1|1  a_2|1  a_3|1 ...
t=2 的预测：               a_2|2  a_3|2 ...
                                 ↑
执行时刻 2：融合 a_2|0、a_2|1、a_2|2
```

$$
a_t^{\mathrm{exec}}=\sum_{s\in\mathcal S_t}w_s\hat a_{t\mid s},
\qquad \sum_s w_s=1.
$$

只融合**对同一个执行时刻**的预测，不是把一个块里的 K 个不同时间动作平均。原版采用指数权重；具体谁权重大依赖候选排序，不能直接断言“最新预测一定权重最大”。

时间集成主要用于平滑重叠预测，也不保证跨不同策略模式的平均合理。

## 10. 一条线复述

```text
训练：qpos + 示范动作 → 后验 z；图像空间 token + qpos + z → Transformer → K 步动作。
推理：图像 + qpos + 固定零 z → K 步动作 → 反归一化 → 按块执行或时间集成。
```

## 参考

- [论文：Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
- [官方实现：DETRVAE](https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py)
- [Transformer 的 token 拼接与 action query](https://github.com/tonyzhaozh/act/blob/main/detr/models/transformer.py)
- [训练损失与推理接口](https://github.com/tonyzhaozh/act/blob/main/policy.py)
- [执行循环与 temporal ensembling](https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py)
