# ViNT：用视觉 token 和 Transformer 建模导航

> ViNT = Visual Navigation Transformer。相对 [GNM](GNM.md)，主要变化是“逐帧 CNN 编码 → token 序列 → Transformer 融合”，输出仍是距离和确定性航点。
>
> 导航阅读顺序：[GNM](GNM.md) → ViNT → [NoMaD](NoMaD.md)。

## 1. 先看整体流程

```text
历史 + 当前图像 ── 共享 EfficientNet ──> T 个观测 token ──┐
                                                        │
当前图像 + 目标图像 ── 另一套 EfficientNet ──> 目标条件 token
                                                        │
                                                        ▼ 拼接 + 位置编码
                                             Transformer 自注意力
                                                        │
                                                        ▼ 展平 + MLP
                                                   [B,32]
                                                        ├─ 距离 [B,1]
                                                        └─ 未来航点 [B,P,Da]
```

这里 Transformer 处理的是压缩后的图像级 token，**不是把所有像素或所有图像 patch 直接送进 Transformer**。

## 2. 贯穿形状与配置

按官方 `vint.yaml` 的主要结构参数举例，batch size 为讲解用的小值。

| 符号 | 含义 | 示例 |
|---|---|---:|
| `B` | batch size | 2 |
| `C` | 历史帧数 | 5 |
| `T=C+1` | 含当前帧的观测数 | 6 |
| `H, W` | 图像高、宽 | 64, 85 |
| `D` | token 维度 | 512 |
| `P` | 预测航点数 | 5 |
| `Da` | 每航点维度 | 4，含朝向 |
| `L` | Transformer 层数 | 4 |
| `Nh` | 注意力头数 | 4 |

```text
obs_img  [B,3T,H,W] = [2,18,64,85]
goal_img [B,3,H,W]  = [2,3,64,85]
```

输入接口看起来仍把 T 帧堆在通道维，但进入观测 CNN 前会拆回单帧。这和 GNM 直接用 `3T` 通道 CNN 混合各帧的计算方式不同。

## 3. 观测分支：每张图压成一个 token

```text
obs_img                                    [B,3T,H,W]
     │ 每 3 个通道拆成一帧
     ▼
T 个张量，每个                             [B,3,H,W]
     │ 在 batch 维拼接，复用同一个 CNN
     ▼
所有帧                                    [T*B,3,H,W]
     │ EfficientNet-B0，取卷积特征
     ▼
特征图                                    [T*B,1280,hf,wf]
     │ 全局平均池化 GAP + flatten
     ▼
图像向量                                  [T*B,1280]
     │ Linear(1280 → 512)
     ▼
帧向量                                    [T*B,512]
     │ 恢复时间与 batch 维
     ▼
观测 token                                [B,T,512] = [2,6,512]
```

每帧的空间信息先由 CNN 提取，再被池化到一个向量。Transformer 主要处理帧间关系，以及观测与目标之间的关系。

### 一个容易写错的 reshape

官方先对每帧的 `[B,3,H,W]` 在 batch 维拼接，得到的是“时间优先”的内存顺序：

```text
[帧0的全部 B 个样本, 帧1的全部 B 个样本, ...]
```

所以编码后需要：

```python
# obs_encoding: [T*B, D]，排列顺序来自逐帧 concat(dim=0)
obs_tokens = obs_encoding.reshape(T, B, D).transpose(0, 1)
# [T*B,D] → [T,B,D] → [B,T,D]
```

不能直接 `reshape(B,T,D)`，否则即使形状看起来正确，也会把时间与样本对应关系弄错。

## 4. 目标分支：当前图像与目标图像先融合

默认 `late_fusion=False`，目标分支使用一个 **6 通道**的 EfficientNet-B0：

```text
当前帧 o_t     [B,3,H,W] ──┐
                          ├─ 通道拼接 ──> [B,6,H,W]
目标图像 g     [B,3,H,W] ──┘                  │
                                             ▼ 独立的 EfficientNet-B0
                                      [B,1280,hf,wf]
                                             │ GAP + flatten
                                             ▼
                                      [B,1280]
                                             │ Linear(1280 → 512)
                                             ▼
                                      [B,512]
                                             │ unsqueeze
                                             ▼
目标条件 token                        [B,1,512]
```

这个 token 表达的是“当前画面与目标画面的关系”，不是只对目标图像编码。目标 CNN 和观测 CNN 参数独立；观测分支则在不同时间帧间共享参数。

若开启 `late_fusion=True`，该实现的目标 CNN 只编码目标 RGB 图像，输入为 3 通道，再在后面的 Transformer 中与观测融合。本篇主流程按默认的 6 通道路径讲。

## 5. token 序列如何进入 Transformer？

```text
观测 token                                 [B,6,512]
目标条件 token                             [B,1,512]
      │ 沿序列维 concat
      ▼
tokens                                     [B,7,512]
      │ 加正弦位置编码
      ▼
Transformer Encoder × 4                     [B,7,512]
```

序列长度是 `T+1=C+2=7`，不是 6，也不是图像 patch 数量。

自注意力里，每个时间帧 token 都能读取其他帧和目标条件 token：

```text
Q、K、V：             [B,7,512]
拆成 4 个头：         [B,4,7,128]
注意力矩阵 QK^T：     [B,4,7,7]
聚合 V 并合并各头：   [B,7,512]
```

位置编码让网络知道帧的先后顺序。所有输入帧都来自当前及过去，因此这里不需要用因果 mask 去做未来动作的自回归生成。

虽然官方封装类名是 `MultiLayerDecoder`，内部实际调用的是 `nn.TransformerEncoder` 做自注意力；它没有 ACT 那种 K 个动作 query 读取视觉 memory 的结构。

## 6. Transformer 输出如何变成航点？

原版 ViNT 的这个实现是 **展平所有 token，再经过 MLP**，不是取 CLS，也不是直接做 token 平均：

```text
Transformer 输出                     [B,7,512]
       │ flatten，7×512=3584
       ▼
                                     [B,3584]
       │ MLP + ReLU
       ▼
3584 → 512 → 256 → 128 → 64 → 32
       │
       ▼
最终特征                             [B,32]
       ├─ Linear(32 → 1)             → 距离 [B,1]
       └─ Linear(32 → 5×4)           → [B,20]
                                          │ reshape
                                          ▼
                                      [B,5,4]
```

航点后处理与 GNM 相同：

- 前两维先作为位置增量预测，再沿航点时间维 `cumsum`，得到当前局部坐标系中的未来位置。
- 后两维是朝向的 `cos、sin` 表示，做 L2 归一化。
- 不预测朝向时，`Da=2`，输出为 `[B,P,2]`。

## 7. 训练与部署流程

### 训练

```text
多机器人轨迹数据
     │
     ▼ 采样历史帧、当前帧、未来目标帧
obs_img + goal_img
     │
     ▼ CNN → token → Transformer → MLP
距离预测 + 航点预测
     │
     ▼ 与时间距离标签、归一化局部航点标签比较
距离回归损失 + 动作回归损失
```

其数据接口和 GNM 接近：用图像作为目标，用归一化航点弱化机器人本体差异。动作监督必须匹配有效的观测—目标对，负目标不能直接当成有到达示范的正样本。

### 部署

```text
图像历史 + 候选目标 / 拓扑地图节点图像
     │
     ▼ ViNT
预测候选节点距离，选择要推进的子目标
     │
     ▼ 取该子目标对应的局部航点预测
恢复部署尺度 → 底层控制器 → 执行 → 更新图像历史
```

**局部网络输出和完整导航系统要分开理解。** 论文还讨论了拓扑规划、探索时生成候选目标图像等系统设计；上面逐层展开的是核心距离 / 航点网络，不把外部规划模块算作它的一个预测头。

## 8. 从 GNM 到 ViNT，再到 NoMaD

| 对比项 | GNM baseline | ViNT |
|---|---|---|
| 历史图像 | 堆叠进 CNN 输入通道 | 每帧经共享 CNN 成为一个 token |
| 目标条件 | 整段观测与目标图像进入目标 CNN | 默认用当前帧 + 目标图像生成一个 token |
| 融合方式 | 两路特征拼接 + MLP | token 自注意力 + 展平 MLP |
| 输出 | 一个确定性航点序列 + 距离 | 一个确定性航点序列 + 距离 |

ViNT 增强了视觉上下文和目标关系的建模，但把网络换成 Transformer，并不会自动把确定性回归变成多峰生成。

```text
前面障碍物，左右都能绕
              │
ViNT 确定性航点回归仍可能遇到模式折中
              │
              ▼ NoMaD
保留视觉 token 编码思路
加入目标掩码，并用条件扩散生成完整轨迹
```

后续见 [NoMaD](NoMaD.md)：它还修改了 token 汇聚方式，不能理解成原样保留 ViNT 全部头部，只把最后一层 Linear 换成 U-Net。

## 参考

- [论文：ViNT: A Foundation Model for Visual Navigation](https://arxiv.org/abs/2306.14846)
- [ViNT 官方网络实现](https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/vint/vint.py)
- [自注意力、展平与 MLP](https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/vint/self_attention.py)
- [本篇使用的配置](https://github.com/robodhruv/visualnav-transformer/blob/main/train/config/vint.yaml)
