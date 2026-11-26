# OpenVLA：双视觉塔 + Llama 的视觉语言动作策略

> OpenVLA 和 [RT-2](RT-2.md) 都把预训练 VLM 作为底座，并把机器人动作转成语言 token；OpenVLA 的主要特点是 **DINOv2 + SigLIP 双视觉塔、Prismatic projector、Llama-2 语言模型**。

## 1. 网络流程

本地实现的完整数据流如下，设 batch 为 `B`、单图 patch 数为 `N=256`、DINOv2 维度为 `1024`、SigLIP 维度为 `1152`、Llama 隐藏维度为 `D_lm=4096`：

```text
每个相机图像
    ├─ DINOv2-L/14     [B,3,224,224] → [B,256,1024]
    └─ SigLIP-So400m/14[B,3,224,224] → [B,256,1152]
                                      │ 同一空间 patch 位置拼接
                                      ▼
双塔视觉特征                         [B,256,2176]
                                      │ Prismatic projector
                                      │ 2176 → 8704 → 4096 → 4096
                                      ▼
语言空间视觉 token                   [B,256,4096]
                                      │ 多相机沿 token 维拼接
                                      ▼
视觉前缀                             [B,V×256,4096]
                                      │ 加相机 embedding
任务文本 token → Llama embedding ────┘
                                      ▼
视觉 token + 指令 token
                                      │ Llama-2 decoder，causal attention
                                      ▼
共享 lm_head                         [B,N_total,Vocab]
                                      │ 只允许动作 token 解码
                                      ▼
7 个动作 token → codec → 连续动作 [B,7]
```

双塔的关键是**沿通道维融合相同空间位置的 patch**：

```text
[B,256,1024] + [B,256,1152] → [B,256,2176]
```

它不是把 DINO 的 256 个 token 和 SigLIP 的 256 个 token 沿序列维拼成 512 个 token。多相机扩展才是沿 token 维拼接：`[B,V×256,4096]`。

## 2. 动作 token 与训练 / 推理

### 动作编码

动作接口是 7 维：

```text
[dx, dy, dz, roll, pitch, yaw, gripper]
```

先按数据集的 `q01/q99` 统计归一化，再用 256 个边界把每个维度分桶。OpenVLA 本地 codec 使用 **256 个边界、255 个可解码中心**，对应 255 个动作词表 token；这和 RT-2 的 256 个 bin / 256 个 token 映射约定不同。

训练动作形状：

```text
连续示范动作                       [B,T,7]
归一化 + 分桶                      [B,T,7]
flatten                            [B,T×7]
映射到 Llama 词表                  [B,T×7]
```

### 训练

```text
图像 + 指令 → 双视觉塔 → projector → 视觉前缀
示范动作 → 归一化 / 分桶 → 动作词表 token
视觉前缀 + 指令 + 动作答案 → Llama decoder
                                      │
                                      ▼
                          共享 lm_head + next-token CE
```

动作答案位置计算交叉熵，视觉与 prompt 位置用 `-100` 屏蔽。OpenVLA 使用 Llama 风格的 causal 语言建模接口，不额外创建独立的动作分类头。

### 推理

```text
图像 + 指令 → 视觉前缀 prefill → KV-Cache
                                  │
                                  ▼
                      每轮生成一个动作词表 token
                                  │ 只允许动作 token 集合
                                  ▼
                       生成 7 个 token
                                  │
                       token → bin 中心
                                  │
                       反归一化 → [B,7] 动作
```

推理时可以复用预填充后的视觉 / 指令 KV-Cache，每生成一个 token 后再增量更新缓存。动作 token 约束避免模型输出普通文本词。

## 3. 核心设计

1. **双视觉塔互补**：DINOv2 提供视觉表征，SigLIP 提供图文预训练表征，在相同 patch 网格上做通道级融合。

2. **Prismatic projector 完成跨模态对齐**：`2176 → 8704 → 4096 → 4096`，把双塔特征映射到 Llama 的语言隐藏空间。

3. **不使用独立 action head**：动作 token 直接由 Llama 共享 `lm_head` 预测，训练目标仍是 next-token cross entropy。

4. **OpenVLA 的动作 codec 与 RT-2 不完全相同**：OpenVLA 使用 256 个边界和 255 个中心，动作 token 从 Llama 词表末端选择；不能简单写成 RT-2 的 256 token 方案。

5. **多相机和双塔是两个不同拼接轴**：双塔在同一 patch 位置沿通道拼接，多相机在完成 projector 后沿 token 维拼接。

6. **动作归一化使用数据统计量**：本地实现支持选择 checkpoint 中不同数据集的 `q01/q99` 边界，反量化后才恢复机器人真实动作尺度。

7. **参数高效微调可用 LoRA**：`enable_lora()` 冻结主干，只给 Llama 注意力的 q/k/v/o 投影注入低秩增量。

## 4. 和 RT-2 对比

| 项目 | RT-2 | OpenVLA |
|---|---|---|
| 视觉底座 | 视觉塔 + projector，接 PaLI-X / PaLM-E / 本地 PaliGemma 风格适配 | DINOv2 + SigLIP 双视觉塔 |
| 视觉融合 | 通常单视觉塔投影到语言空间 | 同一 patch 位置通道拼接后再过 Prismatic projector |
| 语言模型 | Gemma / PaliGemma 风格 decoder | Llama-2 风格 decoder |
| 动作 token | 常用 256 个 bin 对应 256 个 token | 256 个边界、255 个中心，对应 255 个动作 token |
| 注意力接口 | 本地 RT-2 使用 prefix attention | 本地 OpenVLA 使用 causal attention |
| 动作输出 | 共享语言 `lm_head`，逐 token 生成 | 共享语言 `lm_head`，逐 token 生成 |
| 训练 | 机器人动作 CE，可和 VQA 共微调 | 动作 CE，可配合 Llama / 视觉塔预训练和 LoRA |

两者共同点是：**动作即语言 token，VLM 的视觉语言主干负责理解，词表预测负责输出动作**。主要差异在视觉塔组合、projector、语言底座、注意力方式和动作 codec。

## 5. 面试题

1. **OpenVLA 的完整网络流程是什么？**

   DINOv2 和 SigLIP 分别编码图像，在相同 patch 位置沿通道拼接，经 Prismatic projector 对齐到 Llama，再和文本组成序列，由 Llama 共享 lm_head 生成动作 token。

2. **OpenVLA 为什么使用两个视觉塔？**

   DINOv2 和 SigLIP 的预训练表征侧重点不同，双塔融合可以同时利用视觉表征和图文对齐表征。

3. **双视觉塔如何融合？**

   两路输出形状分别是 `[B,256,1024]` 和 `[B,256,1152]`，沿最后一维拼成 `[B,256,2176]`，不是沿 token 维拼接。

4. **Prismatic projector 的作用是什么？**

   将双塔视觉通道维 `2176` 映射到 Llama 隐藏维度 `4096`，使视觉 patch 可以作为语言模型输入 embedding。

5. **OpenVLA 有独立 action head 吗？**

   没有；动作 token 由 Llama 的共享 `lm_head` 输出，动作训练使用语言模型的 next-token 交叉熵。

6. **OpenVLA 的动作 token 和 RT-2 完全一样吗？**

   不完全一样。OpenVLA 的本地 codec 使用 256 个边界和 255 个中心，RT-2 的本地 codec 使用 256 个 bin 中心和 256 个动作 token。

7. **OpenVLA 推理时如何从 token 得到动作？**

   约束 logits 只在动作 token 集合内选择，生成 7 个 token，再映射到 bin 中心并按照数据集 `q01/q99` 统计反归一化。

8. **OpenVLA 和 RT-2 最大的共同点与差异是什么？**

   共同点是都把 VLM 作为动作生成底座；差异是 OpenVLA 使用 DINOv2 + SigLIP 双塔、Prismatic projector 和 Llama，动作 codec 也不同。

9. **为什么需要动作归一化？**

   7 个动作维度的物理范围不同，归一化后才能在统一的离散边界上分桶，推理时再恢复原始量纲。

10. **OpenVLA 如何减少微调参数？**

   可以冻结视觉和语言主干，仅对 Llama 注意力 q/k/v/o 投影注入 LoRA；但 projector 和动作映射仍需与任务数据匹配。

## 一句话总结

**OpenVLA 是用 DINOv2 + SigLIP 双视觉塔把图像编码到语言空间，再让 Llama 用共享词表逐 token 生成离散动作的 VLA；与 RT-2 的主线相同，重点差在双塔视觉融合、Llama 底座和动作 codec。**

## 参考

- [OpenVLA 论文](https://arxiv.org/abs/2406.09246)
- [本地 OpenVLA 主模型](D:/work/embodied-intelligence/vla_brain/models/openvla/model.py)
- [本地双视觉塔与 Prismatic projector](D:/work/embodied-intelligence/vla_brain/models/openvla/dual_vision.py)
- [本地动作词表 codec](D:/work/embodied-intelligence/vla_brain/models/actions.py)
