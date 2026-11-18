# DDPM：把动作加噪，再学会条件去噪

> 本篇只围绕具身智能：生成未来动作块 / 航点序列。重点是参数、前向与反向流程，以及 1D U-Net、DiT 如何接收观测条件。
>
> 阅读顺序：本篇 → [DDIM](DDIM.md)。导航中的应用见 [NoMaD](NoMaD.md)。公式采用最常见的 **epsilon 噪声预测**版本，不展开变分推导。

## 1. DDPM 在机器人里生成什么？

生成对象是未来一段动作，而不是观测图像：

```text
图像历史、机器人状态、目标 / 指令
               │ 观测编码器
               ▼
条件 c ──────────────────┐
                         │
纯高斯动作噪声 ── 多步条件去噪 ──> 未来动作块
                                   │
                                   ▼
                            反归一化 → 执行一段 → 重新规划
```

同一观测下，从左边绕、从右边绕都可能合理。确定性 MSE 动作回归可能输出折中路线；扩散策略学习动作分布，可以从不同噪声出发生成不同的完整动作块。

训练同样可以用 MSE，但它回归的是**给定带噪动作、噪声等级与观测条件时的噪声**，不等于直接从观测回归一条平均动作轨迹。

## 2. 先分清三种“时间”

| 记号 | 含义 | 讲解示例 |
|---|---|---:|
| `To` | 输入观测历史长度 | 2 帧 |
| `Ha` | 一次预测的未来动作长度 | 16 步 |
| `t` | 扩散噪声等级 | `1 ... N` |
| `N` | 训练噪声表的长度 | 1000，示例值 |

**扩散去噪 1000 次，不代表机器人执行 1000 个动作；预测 16 步动作，也不代表只去噪 16 次。**

本文数学下标用 `x_0` 表示干净动作，`x_N` 表示高噪声动作。许多代码把 N 个噪声等级存为 `0 ... N-1`，不能直接把代码里的 `t=0` 当成本文干净动作 `x_0`。

## 3. 参数表：每个量到底控制什么？

| 参数 / 符号 | 含义 | 要记住的关系 |
|---|---|---|
| `x_0` | 归一化后的真实动作块 | `[B,Ha,Da]` |
| `x_t` | 噪声等级 t 的动作块 | 形状始终与 `x_0` 相同 |
| `beta_t` | 第 t 次前向加噪的方差 | 是方差，噪声标准差是 `sqrt(beta_t)` |
| `alpha_t` | 单步保留系数 | `alpha_t = 1 - beta_t` |
| `alpha_bar_t` | 从干净动作到 t 的累计保留系数 | `alpha_bar_t = prod(alpha_i, i=1...t)` |
| `beta_schedule` | 整张噪声表的生成方式 | 如 linear、cosine |
| `beta_start / beta_end` | 线性噪声表的起止值 | 如 `1e-4 → 0.02`，不是具身任务的固定配置 |
| `N / num_train_timesteps` | 训练可采样的噪声等级数 | 不是 epoch 数，也不是动作长度 |
| `epsilon` | 用来构造带噪样本的标准高斯噪声 | `[B,Ha,Da]` |
| `epsilon_theta` | 神经网络预测的噪声 | 由 1D U-Net、DiT 等实现 |
| `c / global_cond` | 图像、状态、指令等编码出的条件 | 如 `[B,256]` |
| `sigma_t²` | 反向采样时添加噪声的方差 | 可取后验方差等，需要看 scheduler 配置 |
| `prediction_type` | 网络输出参数化 | 如 epsilon、x0、v；本文按 epsilon 讲 |
| `Ha / Da` | 动作块长度 / 每步动作维度 | 如 16 步、每步 7 维 |

`beta_t`、`alpha_t`、`alpha_bar_t` 通常是预先计算的标量表，不是靠反向传播训练出来的网络权重。学习的主要是去噪网络，以及任务中参与训练的观测编码器。

## 4. 前向过程：给真实动作逐步加噪

假设机械臂每步输出 7 维动作，例如 3 维位置增量、3 维旋转参数、1 维夹爪量。具体动作定义由任务决定。

```text
示范动作 A                      [B,Ha,Da] = [2,16,7]
      │ 按训练数据统计量归一化
      ▼
干净动作 x_0                    [2,16,7]
      │ 前向加噪
      ▼
x_1 → x_2 → ... → x_N            每一步都是 [2,16,7]
```

单步前向公式：

$$
x_t=\sqrt{1-\beta_t}\,x_{t-1}+\sqrt{\beta_t}\,\epsilon_t,
\qquad \epsilon_t\sim\mathcal N(0,I).
$$

这里把 `alpha_t = 1 - beta_t` 直接代入公式。原来写的 `z_t` 就是标准高斯噪声，与写作 `epsilon_t` 没有本质区别，本篇统一使用后者。每个前向步骤独立采样新的 `epsilon_t`。

随着噪声累计，动作结构逐渐被淹没。噪声表通常设计成 `alpha_bar_N` 足够小，使终点近似标准高斯，从而允许推理从纯噪声开始。

### 训练不用真的从 1 加到 t

利用累计系数，可以从干净动作一步得到任意噪声等级：

$$
\boxed{x_t=\sqrt{\bar\alpha_t}x_0+
\sqrt{1-\bar\alpha_t}\epsilon,\qquad\epsilon\sim\mathcal N(0,I).}
$$

前面的 `epsilon` 是从 `x_0` 到 `x_t` 的**等效累计噪声**，不是把每个单步的 `epsilon_t` 简单相加。

```text
x_0 [2,16,7] + epsilon [2,16,7] + 随机 t [2]
                        │
                        ▼ 查表、广播系数到 [2,1,1]
                     x_t [2,16,7]
```

同一个 batch 中不同样本可以选不同的 t。标准做法是同一动作块里的所有动作位置共享一个噪声等级，但每个动作元素的高斯噪声独立采样。

## 5. 训练流程：网络学的是预测噪声

```text
图像、状态、目标 ── Encoder ──> c [B,256] ─────────┐
                                                  │
示范动作 ── 归一化 ──> x_0 [B,16,7]                │
随机 t [B] + epsilon [B,16,7]                       │
                │ 一步构造 x_t                     │
                ▼                                 │
          x_t [B,16,7] ────────────────────────────┤
          t   [B] ────────────────────────────────┤
                                                  ▼
                               epsilon_theta(x_t,t,c) [B,16,7]
                                                  │
                                                  ▼ 与已知 epsilon 比较
                                                MSE
```

$$
\mathcal L_{\mathrm{simple}}=
\mathbb E\left[\|\epsilon-\epsilon_\theta(x_t,t,c)\|_2^2\right].
$$

面试先记这个简化噪声损失即可。实际任务可能加有效动作 mask、不同动作维度权重等。

一次训练更新通常为每个样本随机抽一个噪声等级，只调用一次去噪网络，不需要先跑完整个反向链。

## 6. 反向过程：网络预测噪声，scheduler 更新动作

推理没有示范动作，先采样 `x_N ~ N(0,I)`，然后逐步去噪。

最常见的 epsilon 参数化反向均值：

$$
\mu_\theta(x_t,t,c)=\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t,c)\right).
$$

反向更新：

$$
\boxed{x_{t-1}=\mu_\theta(x_t,t,c)+\sigma_t z,
\qquad z\sim\mathcal N(0,I).}
$$

这里的 `z` 也是标准高斯噪声，换个字母不改变含义。保留这个记号是为了区分：`epsilon_theta` 是网络预测的噪声，`z` 是反向这一步新采样的独立噪声，不能把二者当成同一个数值。

理解成两步：先沿模型给出的方向更新动作，再按反向方差加入适量随机噪声。**不是直接做 `x_t - epsilon_hat`，也不是每次把动作重置成纯噪声。**

### 为什么反向还要加随机噪声？

这里的随机项有两个相关但不同的作用：

1. **保留动作分布的多样性**：同一观测下左绕、右绕都可能是正确示范。随机项为反向路径提供不同分支，使模型通过采样表达合理动作之间的不确定性。但多样性也来自初始 `x_N`；DDIM 即使逐步不加随机噪声，也能生成多模态动作，不能说“不加这项就生成不出多模态”。
2. **匹配反向条件分布**：真实反向条件 `q(x_{t-1}|x_t,c)` 通常有方差，模型用带方差的 `p_theta(x_{t-1}|x_t,c)` 近似它。反向随机项是这个采样分布的一部分。直接删掉 DDPM 随机项、只保留均值，一般会改变原定的采样分布；这也不等于正确切换到了 DDIM。

所以，“帮助保留多模态动作的多样性”可以作为直观解释；更准确的回答是：**反向噪声用于实现带方差的反向条件采样，而非只输出条件均值。动作的多模态来自整个条件生成过程，不由这项噪声单独保证。**

### 一个相关的纠错视角：Langevin Dynamics

在 score-based diffusion 中，若网络估计的是数据分布的 score
`s_theta(x,t,c) ≈ ∇_x log p_t(x | c)`，Langevin 更新可以写成：

$$
x' = x + \rho\,s_\theta(x,t,c)+\sqrt{2\rho}\,\xi,
\qquad \xi\sim\mathcal N(0,I).
$$

`rho` 是更新步长。它的直觉是：score 把样本推向**当前噪声等级下概率更高的动作区域**，随机项帮助探索附近区域。两者配合用于采样分布，而不是只沿梯度追逐某个最高概率动作。这可以作为一种**带随机探索的纠错视角**，但不是噪声本身会识别错误，也不保证修复所有模型误差。

DDPM 的 epsilon 预测和 score 有关系：

$$
s_\theta(x_t,t,c)\approx-
\frac{\epsilon_\theta(x_t,t,c)}{\sqrt{1-\bar\alpha_t}}.
$$

因此 DDPM 的反向均值更新也可以从“沿 score 方向修正 + 注入随机性”的角度理解。但面试要说清楚：**标准 DDPM 不是简单额外插入一轮独立的 Langevin Dynamics；它的 scheduler 反向公式本身就是根据噪声等级推导的离散反向采样。** 某些 score-based sampler 会显式使用 predictor-corrector，在 predictor 后再执行 Langevin corrector；那是采样器设计，不是所有 DDPM 实现的必备步骤。

常见的 fixed-small 方差选择为：

$$
\sigma_t^2=\tilde\beta_t
=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t.
$$

另有取 `sigma_t²=beta_t` 的 fixed-large 版本，以及学习方差的版本。下面和 DDIM 的等价讨论以 **fixed-small** 为准。最后到干净动作的步骤不再添加噪声。

```text
固定本次规划条件 c                    [B,256]
采样初始动作噪声 x_N                  [B,16,7]
          │
          ▼ 重复：t = N, N-1, ..., 1
去噪网络预测噪声                      [B,16,7]
scheduler 按反向公式更新动作           [B,16,7]
          │
          ▼
生成的干净动作块                      [B,16,7]
          │ 反归一化
          ▼
与当前控制时刻对齐后，执行其中一小段
          │
          ▼ 新观测 → 编码新条件 → 重新规划
```

这就是滚动执行。比如预测 16 步、只执行其中连续 4 步后重新规划；实际取哪个片段要遵循训练数据的时间对齐方式。如果输出表示航点增量，还要像 NoMaD 那样 `cumsum` 还原位置。

## 7. 1D U-Net 怎么嵌进 DDPM？

**DDPM 是训练与采样方法；1D U-Net 是实现 `epsilon_theta` 的网络。** 网络负责预测噪声，采样公式由 scheduler 负责。

```text
观测历史 / 状态 / 目标 ── Encoder ──> c [B,256]
t [B] ── 正弦时间嵌入 + MLP ──> e_t [B,256]
                         │ concat
                         ▼
                 global feature g [B,512]
                         │ 注入各层条件残差块
                         │
x_t [B,16,7] ── transpose ──> [B,7,16]
                                  │
                                  ▼ 1D U-Net，沿动作序列长度卷积
                            [B,64,16]
                                  ↓ 下采样
                            [B,128,8]
                                  ↓ 下采样
                            [B,256,4]
                                  ↑ 上采样 + skip connection
                            [B,128,8]
                                  ↑ 上采样 + skip connection
                            [B,64,16]
                                  │ 输出投影 + transpose
                                  ▼
                           eps_hat [B,16,7]
```

以上通道数是讲解配置。具体模型可以用其他宽度，但输入、输出的动作形状相同。

### global condition 怎么进入某一层？

假设某层特征 `h=[B,128,8]`：

```text
g [B,512] ── MLP ──> scale、bias 各 [B,128]
                            │ unsqueeze
                            ▼
                     各 [B,128,1]
                            │ 广播到长度维
                            ▼
                   h' = scale * h + bias
```

这是一种 FiLM 式的条件调制：每个样本、每个通道的缩放和平移由条件决定，并在该层所有动作位置共享。部分实现仅注入 `bias`，也有写成 `(1+scale)*h+bias` 的参数化。

`global` 表示条件向量影响整段动作，不表示机器人一定拥有全局地图。它也不是把 `c` 当成额外的动作塞进输出序列。

## 8. DiT 怎么嵌进 DDPM？

可以把 U-Net 替换为动作序列上的 Diffusion Transformer，继续使用相同的加噪目标与采样公式。以下是**将 DiT 的条件调制思路用于动作的示意结构**，不是把原图像 DiT 不改代码直接用于机器人。

```text
带噪动作 x_t                          [B,16,7]
       │ 每一步动作做 Linear(7 → 512)
       ▼
动作 token                            [B,16,512]
       │ 加动作位置编码，表示第几个未来动作
       ▼
Transformer blocks × L                 [B,16,512]
       │ Linear(512 → 7)
       ▼
eps_hat                               [B,16,7]
```

一个动作位置对应一个 token，不需要图像 patch 化。动作位置编码与扩散步 embedding 是两个不同概念：前者区分 16 个未来位置，后者告诉模型当前噪声有多大。

### 用 adaLN 注入全局条件

```text
扩散步 t ── embedding + MLP ──> [B,512] ──┐
观测条件 c [B,256] ── Linear ──> [B,512] ──┤ 相加得到 g [B,512]
                                           │
                                           ▼ 每个 block 的条件 MLP
                              shift、scale、gate 等 [B,512]
                                           │ 扩展为 [B,1,512]
                                           ▼ 广播到 16 个动作 token
动作特征 h [B,16,512] → LayerNorm → 条件缩放 / 平移 → Attention / MLP
```

典型调制公式（条件参数沿动作序列维广播）：

$$
h_{\mathrm{mod}}=(1+\mathrm{scale})\odot\operatorname{LayerNorm}(h)+\mathrm{shift},
\qquad h_{\mathrm{out}}=h+\mathrm{gate}\odot\operatorname{Attention}(h_{\mathrm{mod}}).
$$

原 DiT 的 adaLN-Zero 风格 block 会为 Attention 和 MLP 分别生成调制参数与残差门控。动作扩散模型也可保留视觉 / 语言 token，用 Cross-Attention 读取条件，不一定先压成一个全局向量。

**DiT / U-Net 决定如何预测噪声；DDPM / DDIM 决定如何利用预测更新样本。** 但把已训练 U-Net 直接换成 DiT，需要重新训练或适配网络，不能把它和“切换 DDIM 采样器无需重训”混为一谈。

## 9. 面试八股：容易说错的点

| 问题 | 建议回答 |
|---|---|
| 前向过程学不学？ | 标准 DDPM 前向加噪是固定过程；学习的是条件去噪网络等参数。 |
| 网络预测动作还是噪声？ | 本文 epsilon 版本预测噪声；也有 x0、v 参数化，必须和 scheduler 对齐。 |
| 反向每步为什么还加噪声？ | 实现有方差的反向条件采样，表达动作不确定性；但多模态不以逐步加噪为必要条件，DDIM 也能多模态生成。 |
| 反向加噪是不是一种纠错机制？ | 可用 Langevin 的“score 修正 + 噪声探索”帮助理解；标准 DDPM 不等于额外运行 Langevin corrector，噪声本身也不负责识别错误。 |
| 训练是 1000 步，就要每个 batch 跑 1000 次网络吗？ | 不需要；随机抽噪声等级，一步构造带噪样本，通常一次网络前向。 |
| 推理能直接把 for 循环少跑几次吗？ | 不能沿用相邻步公式随便跳；要用支持跳步的采样更新与对应时间表，见 DDIM。 |
| 为什么网络一定要知道 t？ | 相同数值的带噪动作在不同噪声等级下，所需去噪方向 / 强度不同。 |
| c 在去噪中每步都变吗？ | 普通条件采样中，本次规划的 c 固定；扩散步 t 和动作 x_t 持续变化。 |
| 训练和推理的观测编码各做几次？ | 通常每个训练 batch / 每次规划编码观测，再供各去噪步复用；下一次闭环规划更新 c。 |
| 用 MSE 预测噪声为什么不等于平均轨迹？ | 回归对象是带噪动作条件下的噪声；最终经整条采样过程得到动作分布。 |
| 多采几条轨迹后取均值好吗？ | 跨左右模式直接取均值可能产生错误轨迹；应按任务采用或筛选样本。 |

## 10. 面试够用的复述

> 在具身智能里，我把未来动作块当作扩散变量，图像、状态或指令编码成条件。训练随机选一个噪声等级，对真实动作直接加噪，让 U-Net 或 DiT 预测噪声。推理从高斯动作噪声出发，网络给出噪声预测，scheduler 按反向均值和方差逐步更新，最后得到动作块，反归一化后滚动执行。动作长度和扩散步数是两条不同的时间轴。

## 参考

- [DDPM：Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Diffusion Policy：动作扩散与视觉条件策略](https://arxiv.org/abs/2303.04137)
- [Diffusion Policy 的 ConditionalUnet1D](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conditional_unet1d.py)
- [DiT 条件调制实现：参考 adaLN 思路，动作 token 结构需按任务改造](https://github.com/facebookresearch/DiT/blob/main/models.py)
