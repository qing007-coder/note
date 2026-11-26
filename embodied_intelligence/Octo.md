# Octo：分组时序 Transformer + DDPM 动作块

## 一、网络流程设计

设 batch 为 `B`、历史窗口为 `W`、动作块长度为 `H=4`、动作维度为 `7`、策略宽度为 `D`。

### 1. 任务文本编码

```text
task_ids                         [B,L]
    │ T5 encoder
    ▼
task_hidden                      [B,L,Dt]
    │ task_projector
    ▼
task_tokens                      [B,L,D]
```

任务 token 的 `kind=0`，`time=-1`。

### 2. 多相机历史观测编码

每路相机历史图像先把 batch 和时间维合并，再经过视觉编码器：

```text
images                           [B,W,3,H,W]
    │ flatten(0,1)
    ▼
                               [B×W,3,H,W]
    │ SmallStem / ViT
    ▼
                               [B×W,N,Dv]
    │ visual_projector
    ▼
                               [B×W,N,D]
    │ 可选 TokenLearner
    ▼
                               [B×W,N',D]
    │ reshape 回时间维
    ▼
                               [B,W,N',D]
```

每个观测 token 加上：

```text
spatial_position + time_embedding[帧号] + camera_embedding[相机号]
```

最终观测 token 为 `[B,W×N',D]`，`kind=1`，`time=帧号`。

### 3. Readout token

`readout` 是可学习参数 `[1,1,R,D]`，扩展到每个 batch 和每个历史时刻：

```text
readout 参数                    [1,1,R,D]
    │ expand + time_embedding
    ▼
readout tokens                  [B,W,R,D]
    │ flatten 时间和 readout 槽位
    ▼
                               [B,W×R,D]
```

它们的 `kind=2`，`time=帧号`。

### 4. 三类 token 拼接与分组 Transformer

顺序固定为：

```text
[任务 token | 观测 token | readout token]
```

对应总长度：

```text
N_total = L + W×N' + W×R
tokens   [B,N_total,D]
kind     [N_total]
time     [N_total]
valid    [B,N_total]
```

`OctoBlockTransformer` 由多层组成，每层是：

```text
Pre-LN → 多头自注意力 → 残差 → Pre-LN → MLP → 残差
```

注意力使用 `kind、time、valid` 生成分组掩码；`valid` 同时作用于 query 和 key。

### 5. Readout 汇聚成条件

Transformer 输出仍为 `[B,N_total,D]`。只取最后 `W×R` 个 readout token：

```text
[B,W×R,D]
    │ reshape
    ▼
[B,W,R,D]
    │ 对 R 个 readout 求均值
    ▼
condition                        [B,W,D]
```

### 6. DDPM 动作头

训练时每个历史时刻都有一个条件向量：

```text
condition                        [B,W,D]
动作样本 x0                      [B,W,H,7]
随机噪声 epsilon                  [B,W,H,7]
随机扩散步 t                     [B,W]
    │ x_t = sqrt(abar_t)x0 + sqrt(1-abar_t)epsilon
    ▼
noisy action                     [B,W,H,7]
    │ 展平 B×W，输入 ConditionedResidualMLP
    │ cat([noisy H×7, condition D, t 的正弦嵌入])
    ▼
epsilon_hat                      [B,W,H,7]
```

该动作头是 **DDPM + ConditionedResidualMLP**：不是 U-Net，不是 DiT；条件直接拼接到 MLP 输入，没有 FiLM、Cross-Attention 或 adaLN。

推理时 `encode()` 只运行一次，取最后一个有效时刻的条件 `[B,D]`；`sample()` 只重复小型 MLP 和 DDPM 更新，得到动作块 `[B,H,7]`。

## 二、核心设计

1. **Octo 是条件策略，不是生成式 VLM**：文本和视觉先编码成条件，动作由独立 DDPM 动作头生成。

2. **任务、观测、readout 三类 token 不交错拼接**：统一按 `[任务 | 观测 | readout]` 排列，并用 `kind/time` 控制信息流。

3. **观测 token 只读任务和当前 / 过去观测**：禁止读取 readout，避免动作条件反向污染观测表示。

4. **readout token 负责从每个历史时刻提取动作条件**：它可以读取任务、非未来观测和当前 / 过去 readout，保持时间因果。

5. **每层 Transformer 都使用分组掩码**：任务只读任务；观测不读未来；readout 不读未来观测和未来 readout；`valid` 同时屏蔽无效 query 与 key。

6. **动作头是条件残差 MLP**：输入为 noisy action、时刻条件和扩散时间嵌入，直接输出 `epsilon`，不使用 U-Net、DiT 或显式 Cross-Attention。

7. **训练按每个历史时刻计算 DDPM loss，推理只使用最后有效时刻**：这样编码器学习整个历史窗口的时序条件，部署时输出当前控制时刻的动作块。

8. **默认 DDPM 配置是 cosine `alpha_bar` schedule、20 个扩散步、动作块长度 4、动作维度 7**。

## 三、面试题

1. **Octo 的输入、输出分别是什么？**

   输入是多相机历史观测和任务文本，输出是当前控制时刻的动作块 `[B,H,7]`。

2. **任务 token、观测 token、readout token 的 `kind` 和 `time` 如何定义？**

   任务 token 为 `kind=0,time=-1`；观测 token 为 `kind=1,time=帧号`；readout token 为 `kind=2,time=帧号`。

3. **为什么观测 token 不能读取 readout token？**

   防止动作条件通过 readout 反向污染观测表示，让观测编码保持独立、稳定并符合信息流设计。

4. **分组掩码中，任务、观测、readout 分别允许读取哪些 token？**

   任务只读任务；观测读任务和当前 / 过去观测；readout 读任务、非未来观测和当前 / 过去 readout；`valid` 同时屏蔽无效 query 与 key。

5. **为什么要为 readout token 加时间 embedding？**

   让不同历史时刻的 readout 保留时间身份，Transformer 才能区分“哪一帧提取出的动作条件”。

6. **readout token 在 Octo 中承担什么作用？**

   它是每个历史时刻的可学习查询槽位，用来从任务和历史观测中提取动作生成所需的条件表示。

7. **为什么训练时每个历史时刻都要计算动作 loss，而推理只取最后一个有效时刻？**

   训练对所有时刻监督可以扩大有效样本量并约束时序表示；推理只需要当前最新条件来输出当前动作块。

8. **Octo 的动作头为什么使用 ConditionedResidualMLP，而不是 U-Net 或 DiT？**

   动作块较短、条件维度明确，残差 MLP 足以完成噪声预测，结构和计算更轻；U-Net / DiT 不是 DDPM 的必需组件。

9. **Octo 的条件是如何融合进 DDPM 动作头的？**

   将 noisy action 展平后，与条件向量 `D` 和扩散时间的正弦嵌入直接拼接，送入 ConditionedResidualMLP，输出 `epsilon_hat [B,H,7]`。

10. **Octo 的 DDPM 训练和推理流程分别是什么？**

    训练随机采样 `t` 和噪声，构造 `x_t` 并回归噪声；推理从纯噪声开始，按 `t=steps-1...0` 逐步反推 `x0`、计算后验均值，`t>0` 加后验噪声，`t=0` 不加噪。

11. **为什么 `encode()` 只运行一次，而 `sample()` 可以重复多次？**

    视觉、文本和 Transformer 条件在一次规划内固定；DDPM 只需要在不同噪声状态上重复调用小型动作头，所以可以复用编码结果，减少计算。

12. **Octo 是生成式 VLM 吗？它和 RT-2 的动作 token 预测有什么区别？**

    不是。Octo 是条件策略，VLM / Transformer 负责编码条件，DDPM 动作头生成连续动作块；RT-2 则把离散动作 token 放入 VLM 词表，用共享 `lm_head` 自回归预测。
