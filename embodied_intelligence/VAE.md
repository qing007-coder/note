# VAE：从“动作取平均”到潜变量生成

> 重点：为什么需要潜变量，以及 Encoder → 分布参数 → 采样 → Decoder 的数据流。
>
> 阅读顺序：本篇 → [ACT](ACT.md)。导航中的另一条生成路线见 [NoMaD](NoMaD.md)。

## 1. 为什么不直接回归动作？

假设机器人前面有障碍物，**从左边绕和从右边绕都正确，但从中间走会撞上去**。当输入观测不足以区分两种选择时，示范动作可能是：

```text
相同观测 o
  ├─ 50% 的示范：向左，横向动作 a = -1
  └─ 50% 的示范：向右，横向动作 a = +1
```

如果模型只有一个确定性输出，并用均方误差 MSE 训练：

$$
\mathcal L(f)=\mathbb E[(f(o)-a)^2\mid o],
\qquad f^*(o)=\mathbb E[a\mid o].
$$

那么最优输出就是 `(-1 + 1) / 2 = 0`。**损失意义上的最优，可能是任务意义上的错误动作。** 对整段轨迹逐点取均值，也可能得到一条穿过障碍物的轨迹。

这叫多模态动作分布：同一条件下有多个合理的行为模式。这里的“多模态”是多个概率峰，不是图像、语言等多种输入模态。

几点边界：

- 不是所有直接回归都会失败；如果条件足够明确、动作分布接近单峰，回归很合适。
- MSE 对应条件均值；L1 对应条件中位数，不能把所有回归损失都说成“必然求均值”。
- 把 MSE 改成 L1，并不等于获得了能表达多个行为模式的生成模型。

## 2. 潜变量 z 在流程中起什么作用？

希望把原来的映射：

```text
观测 o ── 确定性网络 ──> 一个动作 / 一段轨迹
```

改成：

```text
观测 o + 潜变量 z ── Decoder ──> 一段轨迹
               │
               ├─ 某些 z 区域：生成左绕轨迹
               └─ 另一些 z 区域：生成右绕轨迹
```

`z` 可以携带行为风格、未被观测明确决定的动作选择。它通常是连续向量，不是人工规定的“左 / 右”标签；训练也不保证某一维天然就表示左右。

条件生成的表达是：

$$
p_\theta(A\mid o)=\int p_\theta(A\mid o,z)p(z)\,dz.
$$

即使 `p(z)` 是单峰高斯，通过非线性 Decoder 后，动作空间中的分布仍然可以具有多个模式。

**关键是对不同 z 分别生成完整轨迹。若再把左绕、右绕样本逐点平均，仍可能重新得到危险的中间轨迹。**

## 3. 先看普通 VAE 的完整形状变化

普通 VAE 学的是输入 `x` 的生成分布。以下用一个便于算形状的 MLP 示例，不是固定架构：

| 符号 | 含义 | 示例 |
|---|---|---:|
| `B` | batch size | 2 |
| `Dx` | 输入维度 | 784，展平的 28×28 图像 |
| `Dh` | 隐层维度 | 256 |
| `Dz` | 潜变量维度 | 32 |

```text
x                                  [B, 784]
 │
 ▼ Encoder：Linear + 激活
h                                  [B, 256]
 ├─ Linear → mu                    [B, 32]
 └─ Linear → logvar                [B, 32]
                │
                ▼ std = exp(0.5 * logvar)
epsilon ~ N(0, I)                   [B, 32]
                │
                ▼ z = mu + std * epsilon
z                                  [B, 32]
 │
 ▼ Decoder：MLP
x_hat                              [B, 784]
```

`mu`、`logvar` 是 Encoder 输出的分布参数：

$$
q_\phi(z\mid x)=\mathcal N\!\left(\mu_\phi(x),
\operatorname{diag}(\exp(\operatorname{logvar}_\phi(x)))\right).
$$

不要把 `mu` 当成最终的动作预测，也不要把 `logvar` 当成标准差。这里 `logvar = log(sigma²)`，所以标准差是 `exp(0.5 * logvar)`。

## 4. 为什么使用重参数化？

训练需要把重建误差传回 Encoder，但“直接从一个由网络决定的分布中随机抽样”不方便按普通计算图求导。

把随机性搬到独立噪声 `epsilon` 上：

$$
\epsilon\sim\mathcal N(0,I),\qquad
z=\mu+\exp(\tfrac12\operatorname{logvar})\odot\epsilon.
$$

此时给定 `epsilon`，`z` 对 `mu`、`logvar` 是可微的，重建损失就能沿 Decoder → z → Encoder 反向传播。

```python
# 仅演示形状；encoder、decoder 是待定义的网络。
mu, logvar = encoder(x)                # 都是 [B, Dz]
std = (0.5 * logvar).exp()             # [B, Dz]
eps = torch.randn_like(std)            # [B, Dz]
z = mu + std * eps                    # [B, Dz]
x_hat = decoder(z)                    # [B, Dx]
```

## 5. 用于动作生成时，通常是 CVAE

普通 VAE 学 `p(x)`；条件 VAE（CVAE）则在条件 `o` 下学 `p(A | o)`。

用动作块作为要生成的对象：`A = [a_t, ..., a_(t+K-1)]`。

| 张量 | 形状示例 | 含义 |
|---|---|---|
| 观测特征 `o` | `[B, Do] = [2, 512]` | 图像、状态等编码后的条件 |
| 示范动作块 `A` | `[B, K, Da] = [2, 100, 14]` | 未来 100 步，每步 14 维 |
| 后验参数 `mu, logvar` | 各 `[2, 32]` | 从示范中推断潜变量分布 |
| 采样 `z` | `[2, 32]` | 一整个动作块共享一个潜变量 |
| 预测动作块 `A_hat` | `[2, 100, 14]` | Decoder 生成的未来动作 |

```text
训练：
示范动作块 A + 条件 o
       │
       ▼ 后验 Encoder q_phi(z | A, o)
    mu, logvar ── 重参数化 ──> z [B, Dz]
                                  │
观测条件 o ─────────────────────────┤
                                  ▼ 条件 Decoder p_theta(A | o, z)
                             A_hat [B, K, Da]
                                  │
                            与示范 A 比较

推理：
先验 p(z) 采样 z + 当前条件 o ── Decoder ──> A_hat
```

上图是通用 CVAE。**ACT 的具体后验 Encoder 只输入关节状态与示范动作，不输入图像；图像进入动作生成侧。** 具体拆解见 [ACT](ACT.md)。

## 6. 损失：重建 + KL

$$
\mathcal L=
\underbrace{\mathbb E_{q_\phi(z\mid A,o)}[-\log p_\theta(A\mid o,z)]}_{\text{重建项}}
+\beta\underbrace{D_{\mathrm{KL}}(q_\phi(z\mid A,o)\|\mathcal N(0,I))}_{\text{先验约束}}.
$$

- **重建项**：给定观测和从示范推断出的 `z`，能否还原示范动作。实践中可使用 L1、MSE 等形式；ACT 用 L1。
- **KL 项**：让训练时的后验接近推理时可用的标准正态先验，约束潜空间，而不是让推理依赖不可获得的未来动作。

对角高斯的 KL 可以直接计算：

$$
D_{\mathrm{KL}}=
\frac12\sum_{j=1}^{D_z}
\left(\mu_j^2+\exp(\operatorname{logvar}_j)-1-\operatorname{logvar}_j\right).
$$

`beta=1` 对应标准形式；改变权重会影响重建与正则化的取舍。权重过强可能让模型忽略 `z`，出现后验坍塌；有 VAE 结构也不保证一定学好了多个行为模式。

## 7. 训练和推理一定要分开记

| 阶段 | 能看到示范动作吗？ | z 从哪里来？ | 使用哪些模块？ |
|---|---|---|---|
| 训练重建 | 能 | Encoder 后验采样 | Encoder + Decoder |
| 通用 CVAE 推理 | 不能 | 从先验采样 | Decoder |
| ACT 官方默认推理 | 不能 | 固定 `z=0` | 条件动作生成网络 |

**`z=0` 是潜变量先验的均值，不是把生成动作取均值。** 但固定 `z=0` 也不等于显式采样多个行为，不能据此声称 ACT 默认推理保证解决所有多峰歧义。

## 8. 最后只记这条流程

```text
直接 MSE 回归：同一观测下左右都对 → 可能回归到中间。

CVAE 训练：观测 + 示范 → 后验分布 → 采样 z → 条件生成动作 → 重建 + KL。
CVAE 推理：当前观测 + 先验 z → 生成一段动作；不再需要未来示范。
```

## 参考

- [VAE 原论文：Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [ACT 官方实现：后验编码、重参数化、推理 z=0](https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py)
