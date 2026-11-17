# NoMaD：目标掩码 + 条件扩散的导航策略

> 重点：视觉信息如何变成条件向量，目标如何被屏蔽，以及带噪动作如何通过 1D U-Net 逐步变成轨迹。
>
> 导航阅读顺序：[GNM](GNM.md) → [ViNT](ViNT.md) → NoMaD。理解动作多峰问题可先看 [VAE](VAE.md)。

## 1. 从 ViNT 再往前走一步

GNM 和 ViNT 的核心航点网络给定输入后输出一条确定轨迹。假设前方有障碍物，左右两条绕行路线都合理，确定性 MSE 回归可能得到两条轨迹的平均，而平均轨迹可能穿过障碍物。

NoMaD 用两个改动处理导航中的不同需求：

| 改动 | 作用 |
|---|---|
| **目标掩码 Goal Masking** | 有目标时朝目标走；屏蔽目标时，根据观测生成探索行为 |
| **条件扩散 Diffusion Policy** | 学习轨迹分布，从不同噪声出发生成不同的合理轨迹 |

```text
GNM： 图像 → CNN + MLP → 一条航点轨迹
ViNT：图像 → CNN + Transformer → 一条航点轨迹
NoMaD：图像 → 带目标掩码的视觉编码器 → 条件扩散 → 可采样的航点轨迹
```

NoMaD 不是 VAE，也没有 `mu、logvar、KL` 那条训练路径。它通过“给动作加噪，再学习去噪”训练生成模型。

## 2. 全文采用的配置

按官方 `nomad.yaml` 的 ViNT 编码器版本说明，不混入可选的其他视觉编码器。

| 符号 | 含义 | 示例 |
|---|---|---:|
| `B` | batch size | 2 |
| `C` | 历史帧数 | 3 |
| `T=C+1` | 含当前帧的观测数 | 4 |
| `H, W` | 图像大小 | 96, 96 |
| `D` | 视觉条件维度 | 256 |
| `P` | 未来航点数 | 8 |
| `Da` | 每航点位置维度 | 2，不预测朝向 |
| `Ndiff` | 配置中的扩散步数 | 10 |
| `Ns` | 同一观测的采样轨迹数 | 4，讲解示例 |

```text
obs_img      [B,3T,H,W] = [2,12,96,96]
goal_img     [B,3,H,W]  = [2,3,96,96]
goal_mask    [B]        = [2]，每个元素为 0 或 1
航点标签 W   [B,P,2]    = [2,8,2]
```

`P=8` 是要生成的未来航点数，`Ndiff=10` 是去噪迭代数，二者不是同一时间轴。

## 3. 总网络：视觉编码器 + 去噪网络 + 距离头

```text
最近 T 帧 + 当前/目标图像对 + goal_mask
                       │
                       ▼ NoMaD-ViNT 视觉编码器
视觉条件 c             [B,256]
                       ├─ 距离 MLP ──> d_hat [B,1]
                       │
                       └───────────────────┐
带噪动作 X_k           [B,8,2] ────────────┤
扩散步索引 k           [B] ────────────────┤
                                           ▼ Conditional 1D U-Net
预测噪声 epsilon_hat                       [B,8,2]
                                           │ 调度器更新样本
                                           ▼
                                  更干净的动作 X_(k-1)
```

视觉条件用于指导动作去噪。距离头用于子目标定位等导航逻辑，不能误读为给每一条采样轨迹自动做碰撞评分的网络。

## 4. 视觉编码器：继承 ViNT 思路，但改变汇聚

### 4.1 构造 4 个观测 token + 1 个目标条件 token

```text
obs_img [B,12,96,96]
     │ 拆成 4 帧，在 batch 维拼接
     ▼
[T*B,3,96,96]
     │ 共享 EfficientNet-B0 → GAP → Linear(1280 → 256)
     ▼
[T*B,256]
     │ reshape(T,B,D) + transpose
     ▼
观测 token                         [B,4,256]

当前帧 [B,3,96,96] + 目标 [B,3,96,96]
     │ 沿通道拼接
     ▼
[B,6,96,96]
     │ 独立 EfficientNet-B0 → GAP → Linear(1280 → 256)
     ▼
目标条件 token                     [B,1,256]

两者沿序列维拼接                   [B,5,256]
     │ 位置编码 + 带目标 mask 的 Transformer Encoder
     ▼
输出 token                         [B,5,256]
     │ 对有效 token 做平均池化
     ▼
视觉条件 c                         [B,256]
```

这个版本还把 EfficientNet 的 BatchNorm 替换为 GroupNorm。更关键的输出差异是：**ViNT 将所有 token 展平并压到 32 维；NoMaD 对有效 token 平均，保留 D 维条件向量给扩散网络。**

### 4.2 目标 mask 到底屏蔽什么？

官方约定：

```text
goal_mask = 0：目标可见，生成朝该目标走的轨迹。
goal_mask = 1：目标被屏蔽，生成依赖观测的探索轨迹。
```

对一个样本，Transformer 的 key padding mask 可以写成：

```text
观测0  观测1  观测2  当前观测  目标条件 token
False  False  False   False       True         # 屏蔽目标
```

还需要在最终池化时去掉目标 token，避免其输出残留进入条件向量：

$$
c=\frac{\sum_i v_i h_i}{\sum_i v_i},
\qquad v_i=\begin{cases}0,&\text{被屏蔽的目标 token},\\1,&\text{有效 token}.\end{cases}
$$

因此，屏蔽目标时平均 4 个观测 token；不屏蔽时平均全部 5 个 token。官方通过权重缩放后调用 `mean` 实现等价的有效 token 平均。

**屏蔽目标不等于把目标图片变黑，也不等于屏蔽所有输入。** 探索策略仍然依赖当前与历史视觉观测。

## 5. 训练的第一步：航点先变成归一化增量

真实未来位置先转换到当前机器人局部坐标系，再构造相邻点增量：

```text
未来局部航点 W                            [B,8,2]
      │ 在最前面补当前原点 (0,0)，再做相邻差分
      ▼
位置增量 Delta W                          [B,8,2]
      │ 按动作统计量做归一化
      ▼
干净动作 X_0                              [B,8,2]
```

官方使用数据统计量做 min-max 归一化，映射到约 `[-1,1]` 的训练范围。**扩散模型直接去噪的是归一化位置增量，去噪结束后才还原成航点。**

## 6. 训练：随机选一个噪声等级，预测噪声

对每个样本随机选扩散步 `k`，采样与动作同形状的高斯噪声：

$$
\epsilon\sim\mathcal N(0,I),\qquad
X_k=\sqrt{\bar\alpha_k}X_0+\sqrt{1-\bar\alpha_k}\epsilon.
$$

`bar_alpha_k` 由噪声调度器决定，随扩散步控制信号与噪声的比例。

```text
干净增量 X_0                  [B,8,2]
随机噪声 epsilon              [B,8,2]
随机扩散步 k                  [B]
        │ scheduler.add_noise
        ▼
带噪动作 X_k                  [B,8,2]
        │
        ├─ 视觉条件 c         [B,256]
        └─ 扩散步 k           [B]
        │
        ▼ 条件 1D U-Net
预测噪声 epsilon_hat          [B,8,2]
        │
        ▼ MSE(epsilon_hat, epsilon)
```

训练一次更新通常只为每个样本抽一个 `k`，不需要先完整跑完全部反向去噪步骤再计算主噪声损失。

## 7. 1D U-Net 内部的形状如何变化？

1D 卷积沿**未来航点序列长度 P**做计算，不沿图像宽高做卷积。

官方配置 `down_dims=[64,128,256]`。省略残差块内部细节后：

```text
X_k                                  [B,8,2]
     │ 转为 Conv1d 的 [B,通道,长度]
     ▼
                                     [B,2,8]
     │ 条件残差块，通道 2 → 64
     ▼
                                     [B,64,8]
     │ 下采样长度 8 → 4
     ▼
                                     [B,64,4]
     │ 条件残差块，通道 64 → 128
     ▼
                                     [B,128,4]
     │ 下采样长度 4 → 2
     ▼
                                     [B,128,2]
     │ 条件残差块，通道 128 → 256
     ▼
瓶颈                                 [B,256,2]
     │ 与编码侧特征做 skip 拼接，残差块 + 上采样
     ▼
                                     [B,128,4]
     │ 再做 skip 拼接，残差块 + 上采样
     ▼
                                     [B,64,8]
     │ 输出卷积，通道 64 → 2
     ▼
                                     [B,2,8]
     │ 转回 [B,长度,通道]
     ▼
epsilon_hat                          [B,8,2]
```

每个条件残差块都接收视觉条件与扩散步 embedding。依赖实现默认的 256 维时间嵌入时：

```text
k [B] ── 正弦 embedding + MLP ──> e_k [B,256]
c                                               [B,256]
                │ concat
                ▼
块的全局条件                                    [B,512]
```

条件经线性层映射到各层通道数并广播到序列长度。当前 NoMaD 配置 `cond_predict_scale=False`，使用条件偏置注入；不是把图片 token 与 8 个动作直接沿时间维拼成一个序列。

## 8. 两种任务怎么一起训练？

训练时随机为每个样本屏蔽目标，官方配置 `goal_mask_prob=0.5`：

```text
同一套网络、同一批轨迹数据
      ├─ 一部分样本保留目标 → 学 p(轨迹 | 观测, 目标)
      └─ 一部分样本屏蔽目标 → 学 p(轨迹 | 观测)
```

噪声预测损失在两种条件下都训练。距离监督只应作用于目标可见的样本，因为目标被隐藏后，不能要求网络推断某张不可见目标图像的距离。

主损失可概括为：

$$
\mathcal L=(1-\lambda)\mathcal L_{\mathrm{noise}}
+\lambda\mathcal L_{\mathrm{distance,unmasked}}.
$$

官方配置距离权重 `lambda=1e-4`，并使用动作有效性 mask 控制噪声损失的样本贡献。这里的损失权重 `lambda` 与扩散噪声调度的 `bar_alpha_k` 不是同一个量。

## 9. 推理：从纯噪声反复去噪

```text
当前观测 + 目标图像 + mask
                │ 视觉编码器，每次规划先计算条件
                ▼
c                                              [B,256]
                │ 为每个观测复制 Ns=4 份条件
                ▼
c_repeat                                       [B*Ns,256] = [8,256]

采样初始高斯噪声 X_K                          [B*Ns,8,2] = [8,8,2]
                │
                ▼ 从高噪声步到低噪声步，重复执行
epsilon_hat = U-Net(X_k, k, c_repeat)            [8,8,2]
X_(k-1) = scheduler.step(epsilon_hat, k, X_k)    [8,8,2]
                │ 全部去噪步骤结束
                ▼
生成的归一化位置增量                           [8,8,2]
                │ 反归一化
                ▼
位置增量                                       [8,8,2]
                │ 沿航点时间维 cumsum
                ▼
完整航点序列                                   [8,8,2]
                │ 恢复采样维
                ▼
每个观测的 4 条候选轨迹                        [B,Ns,P,2] = [2,4,8,2]
```

示意中的 `K` 表示反向过程的起始噪声等级；实际代码使用调度器给出的时间步序列，数量由 `Ndiff` 配置决定。每一步噪声预测和调度器更新都保持动作张量形状不变。

多样性来自不同的初始噪声，以及采样器可能注入的随机性。一次采样生成的是**有时间关联的一整条轨迹**，不是每个航点独立随机选左或右。

## 10. 生成轨迹以后怎么用？

```text
目标可见 mask=0：结合拓扑地图 / 子目标选择，朝指定图像目标导航。
目标屏蔽 mask=1：只用视觉观测，生成局部探索行为。

采样轨迹 → 采用某条轨迹 → 取前方航点 → 恢复机器人运动尺度
                                     → 底层控制器 → 执行 → 重新观测
```

不要把多条左右绕行轨迹直接逐点平均，否则又回到“平均后走中间”的问题。官方导航示例生成多个样本后使用第一个样本中的指定航点；若需要最优或安全筛选，应另行设计评分 / 规划模块，不能认为扩散模型自动完成了这一层。

扩散提供表达多种轨迹模式的能力，但不保证每个样本无碰撞，也不等于完成长距离全局规划。

## 11. 三代模型的数据流对照

| 环节 | GNM baseline | ViNT | NoMaD 的 ViNT 版本 |
|---|---|---|---|
| 观测编码 | 多帧通道堆叠 CNN | 每帧 CNN → 一个 token | 每帧 CNN → 一个 token |
| 目标编码 | 整段观测 + 目标的 CNN 分支 | 当前帧 + 目标 → token | 当前帧 + 目标 → 可屏蔽 token |
| 融合 / 汇聚 | 特征拼接 MLP | Transformer → 展平 MLP | 带 mask 的 Transformer → 有效 token 平均 |
| 示例最终条件 | `[B,32]` | `[B,32]` | `[B,256]` |
| 动作预测 | Linear 回归航点增量 | Linear 回归航点增量 | 条件 U-Net 迭代去噪增量 |
| 轨迹表达 | 确定性输出 | 确定性输出 | 可重复采样的轨迹分布 |
| 无目标探索 | 依赖外部目标 / 探索设计 | 可配外部目标生成与规划 | 目标掩码直接训练观测条件策略 |

```text
把三代记成：统一导航接口 → 更强的视觉时序融合 → 目标 / 探索统一的轨迹生成。

NoMaD 训练：示范增量 + 噪声 + 随机目标 mask → 预测噪声。
NoMaD 推理：视觉条件 + 纯噪声 → 多步去噪 → 反归一化 + 累积 → 航点。
```

## 参考

- [论文：NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration](https://arxiv.org/abs/2310.07896)
- [NoMaD 视觉编码器、目标 mask 与池化](https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/nomad/nomad_vint.py)
- [NoMaD 模块接口与距离头](https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/nomad/nomad.py)
- [官方配置：256 维条件、8 个航点、10 步扩散](https://github.com/robodhruv/visualnav-transformer/blob/main/train/config/nomad.yaml)
- [训练、动作差分、归一化和采样](https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/training/train_utils.py)
- [ConditionalUnet1D：扩散策略依赖中的 U-Net 实现](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conditional_unet1d.py)
