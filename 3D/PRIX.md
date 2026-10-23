# PRIX：从原始像素学习规划的端到端自动驾驶

> **一句话**：PRIX（Plan from Raw pIXels）只使用摄像头图像，通过 Context-aware Recalibration Transformer（CaRT）增强视觉特征，再用条件扩散规划器把带噪轨迹快速去噪成安全、可执行的自车路径。
>
> **论文**：Maciej Wozniak et al., *PRIX: Learning to Plan from Raw Pixels for End-to-End Autonomous Driving*, IEEE Robotics and Automation Letters，arXiv:2507.17596。
>
> **核心结果**：在 NavSim-v1 上达到 `87.8 PDMS` 和 `57 FPS`；输入只有 camera，ResNet-34 作为 backbone。

## 1. PRIX 要解决什么问题

自动驾驶感知系统常使用 LiDAR 或显式 3D / BEV 表示。LiDAR 会增加传感器成本和部署复杂度，稠密 BEV 则会增加显存和计算量。另一类直接从图像回归单条轨迹的方法虽然简单，但难以表达复杂场景中的多种驾驶意图，也容易输出不平滑或不安全的路径。

PRIX 的目标是同时满足三个条件：

1. 只依赖相机，适合大规模量产车辆；
2. 不需要显式构造昂贵的 3D 几何表示；
3. 仍然保留多模态规划能力，并以较少的采样步数得到轨迹。

## 2. 整体数据流

```text
多摄像头图像 + ego status
          │
          ▼
ResNet-34 图像 backbone / FPN
          │
          ▼
多尺度视觉特征
          │
          ▼
CaRT：Context-aware Recalibration Transformer
          │
          ▼
Transformer decoder + ego / scene queries
          │
          ▼
条件扩散规划器 Conditional Diffusion Planner
          │   （轨迹 anchor + 2 步去噪）
          ▼
未来 4 秒自车轨迹
```

PRIX 不输出检测框、地图折线或显式 occupancy 作为中间任务，而是直接学习从视觉上下文到规划轨迹的映射。代码实现保留了 status encoding、query embedding 和 Transformer decoder，用于把图像特征与车辆状态转成规划条件。

## 3. CaRT：不显式建 3D 的上下文增强

### 3.1 动机

普通卷积主要聚合局部邻域，而驾驶规划需要长距离上下文，例如远处车道方向、路口结构和前方车辆之间的关系。直接把高分辨率视觉 token 全部送入全局 Transformer 又会产生较高开销。

CaRT 将特征图压缩到固定空间大小（实现中为 `8×32`），再把每个空间位置展平为 token：

```text
F ∈ R^(B×C×H×W)
→ tokens ∈ R^(B×(H·W)×C)
→ positional embedding
→ Transformer blocks
→ reshape 回特征图
```

实现中使用可学习位置编码、LayerNorm、4 头 self-attention 和前馈网络。CaRT 通过全局空间交互重新校准每个位置的视觉特征，但不要求先估计深度、相机外参或 BEV 网格。

### 3.2 多尺度共享 CaRT

PRIX 的 backbone 对 ResNet-34 的多级特征进行如下处理：

```text
每一级特征 → AdaptiveAvgPool 到 8×32
          → 1×1 Conv 投影到共享维度
          → 同一个 CaRT
          → 1×1 Conv 投影回原通道
          → 残差加回原特征
```

共享 CaRT 让不同尺度具有一致的上下文建模方式，减少为每个尺度维护一套 Transformer 的参数和计算。随后使用轻量 FPN top-down 路径生成规划器使用的高分辨率特征。

## 4. Transformer decoder：把视觉上下文变成规划条件

CaRT 输出的视觉特征会被展平并加入 key-value positional embedding。模型还编码车辆状态，例如速度、加速度、角速度和转向信息，将其作为额外的 status token。查询端使用可学习的 query embedding，经过 Transformer decoder 与视觉 key-value 交互。

可将其抽象为：

```text
K,V = VisualTokens + SpatialEmbedding + StatusToken
Q   = LearnableQueryEmbedding
Z   = TransformerDecoder(Q, K, V)
condition = pool / select(Z)
```

这个条件向量包含前方视觉上下文和当前自车状态，供后面的扩散模型预测轨迹。PRIX 因而不需要显式检测每辆车再把检测结果交给规划器，训练目标直接来自规划质量。

## 5. Conditional Diffusion Planner

### 5.1 为什么使用扩散规划

在路口、变道和避障场景中，给定同一帧图像可能存在多条合理轨迹。单一回归头容易把这些模式平均成一条不安全的中间路径。扩散模型把轨迹生成看作从噪声逐步恢复数据分布的过程，天然可以表示多模态轨迹。

PRIX 不是从完全随机噪声开始长链采样，而是使用轨迹 anchor vocabulary 提供合理的初始轨迹模式，再进行少量去噪。论文和项目说明强调只需 **2 个 diffusion steps**，以满足实时推理要求。

### 5.2 轨迹表示

规划器预测未来一段时间的二维自车轨迹：

```text
τ = {(x_1, y_1), ..., (x_T, y_T)}
```

轨迹 anchor 是从训练数据中获得的典型驾驶路径集合，覆盖直行、转弯、变道和减速等模式。给定视觉条件 `c` 和带噪轨迹 `τ_t`，条件 U-Net 估计去噪方向：

```text
ε_θ = ε_θ(τ_t, t, c)
τ_{t-1} = SchedulerStep(τ_t, ε_θ, t)
```

代码中的 `ConditionalUnet1D` 由 1D Conv、GroupNorm、Mish、下采样 / 上采样和条件残差块组成；扩散时间步使用正弦位置编码，条件通过 FiLM 风格的 scale / bias 或加性调制注入每个残差块。

### 5.3 Anchor 与扩散的关系

```text
anchor vocabulary
        │
        ▼
选择 / 初始化候选轨迹
        │
        ▼
加噪得到 τ_t
        │
        ▼
条件 U-Net 进行 2 步去噪
        │
        ▼
候选轨迹与分数
```

anchor 提供几何先验，视觉条件负责根据当前道路和交通状态修正它。两者结合比从零生成轨迹更容易在极少采样步数下保持合理形状。

## 6. 训练与数据设置

PRIX 基于 NavSim 框架训练和评估。项目 README 给出的主要设置是：

- backbone：ResNet-34；
- 输入：camera；
- 优化器：AdamW；
- 训练轮数：100 epochs；
- 每 GPU batch size：64；
- NavSim-v1 规划时域：从初始传感器观测规划未来 4 秒轨迹；
- 评估硬件：消费级硬件，报告实时 FPS。

训练目标包括轨迹去噪 / 重建和多模态规划损失。多模态损失需要让至少一个候选轨迹贴近示范轨迹，同时避免所有模式塌缩到同一条路径。具体权重应以论文和对应配置为准，复现时需要记录 anchor 数量、轨迹采样点数、扩散 scheduler 和推理步数。

## 7. NavSim 评估指标

### 7.1 NavSim-v1

NavSim-v1 是非反应式仿真：模型根据初始传感器输入规划 4 秒轨迹。主要指标是 PDM Score（PDMS），综合碰撞、道路边界、进度和舒适性。PRIX 在该设置下报告 `87.8 PDMS` 和 `57 FPS`。

### 7.2 NavSim-v2

NavSim-v2 使用带反应交通参与者的 pseudo-simulation，并使用 Extended PDM Score（EPDMS）评估。它比纯开放环轨迹误差更接近交互式驾驶，但 PRIX README 的主表重点报告 NavSim-v1。

## 8. 结果与对比

### 8.1 NavSim-v1

| 方法 | 输入 | Backbone | PDMS ↑ | FPS ↑ |
|---|---|---|---:|---:|
| UniAD | Camera | ResNet-34 | 83.4 | 3 |
| Transfuser | Camera + LiDAR | ResNet-34 | 84.0 | 60 |
| DiffusionDrive | Camera + LiDAR | ResNet-34 | 88.1 | 45 |
| **PRIX** | **Camera** | **ResNet-34** | **87.8** | **57** |

PRIX 在只使用相机的情况下接近 DiffusionDrive 的规划分数，同时保持较高帧率；相较 UniAD，主要优势是避免了稠密 BEV 和完整多任务链路。

### 8.2 nuScenes 轨迹预测表

项目 README 还给出相机输入下的轨迹规划对比：

| 方法 | L2 Avg ↓ | Collision Rate ↓ | FPS ↑ |
|---|---:|---:|---:|
| VAD | 0.72 m | 0.22% | 4.5 |
| SparseDrive | 0.61 m | 0.08% | 9.0 |
| **PRIX** | **0.57 m** | **0.07%** | **11.2** |

这些结果应结合评估协议理解：不同方法的输入、规划时域、碰撞定义和实现硬件可能不同，不能只根据单个数字断言全面优于所有方法。

## 9. PRIX 的优点、局限和复现注意点

### 优点

- **传感器简单**：只依赖相机，不需要 LiDAR；
- **无需显式 3D 几何**：CaRT 在图像空间建模长距离上下文；
- **规划多模态**：anchor vocabulary 与扩散过程能表达多种驾驶意图；
- **推理高效**：只做两步去噪，适合实时部署；
- **结构紧凑**：视觉 backbone、Transformer 条件编码和规划器可以联合训练。

### 局限

- **缺少显式可解释中间任务**：模型不直接输出目标框、地图和占用，出错时较难定位原因；
- **对数据分布依赖较强**：anchor vocabulary 主要覆盖训练集中的轨迹模式，极端新场景可能缺少合适先验；
- **相机感知受天气和遮挡影响**：没有 LiDAR 的几何补充；
- **开放环评估限制**：NavSim-v1 不完全等价于真实闭环驾驶，仍需关注 NavSim-v2 或实车测试；
- **结果需协议对齐**：FPS 依赖硬件、batch、输入分辨率和是否包含预处理。

复现时重点检查以下配置：图像尺寸（项目提供 512 和 256 版本）、ResNet-34 预训练权重、CaRT 的固定池化尺寸 `8×32`、Transformer 的 query 数量、anchor 文件、扩散 scheduler 和推理步数。论文实现代码中的部分路径是作者集群路径，直接迁移到本地时要改成相对路径或配置项。

## 10. 与 UniAD、SparseDrive 的关系

| 维度 | UniAD | SparseDrive | PRIX |
|---|---|---|---|
| 表示 | 时空稠密 BEV | 稀疏实例 + anchor | 图像特征 + CaRT |
| 中间任务 | 跟踪、地图、运动、占用 | 稀疏检测、跟踪、建图、预测 | 主要直接服务规划 |
| 规划方式 | Transformer 轨迹规划 | 多模态轨迹 + 碰撞重打分 | anchor 条件扩散，两步去噪 |
| 输入 | 多摄像头 | 多摄像头 | camera |
| 主要目标 | 规划导向的任务统一 | 稀疏高效和预测规划协同 | 低成本、实时的原始像素规划 |

## 11. 面试版总结

> PRIX 是一个纯视觉端到端自动驾驶规划模型。它先用 ResNet-34 提取多尺度图像特征，再把各尺度压缩到 `8×32` 网格，通过共享的 CaRT Transformer 建模长距离空间上下文，并用 Transformer decoder 融合 ego status。规划阶段不回归单一路径，而是从轨迹 anchor vocabulary 出发，用条件 1D U-Net 执行两步扩散去噪，得到多模态、平滑且可执行的自车轨迹。PRIX 的关键取舍是用图像空间上下文和生成式规划换取低成本、低延迟部署；代价是缺少 UniAD 那样的显式目标与地图中间表示，调试和长尾场景泛化更依赖数据与轨迹先验。

## 12. 论文与代码

- 论文：[PRIX: Learning to Plan from Raw Pixels for End-to-End Autonomous Driving](https://arxiv.org/abs/2507.17596)
- 官方实现：[KTH-RPL/prix](https://github.com/KTH-RPL/prix)
- 项目模型：[PRIX on Hugging Face](https://huggingface.co/maciejw94/prix512/tree/main)
- NavSim：[autonomousvision/navsim](https://github.com/autonomousvision/navsim)

```bibtex
@article{wozniak2026prix,
  title={PRIX: Learning to Plan from Raw Pixels for End-to-End Autonomous Driving},
  author={Wozniak, Maciej and Liu, Lianhang and Cai, Yixi and Jensfelt, Patric},
  journal={IEEE Robotics and Automation Letters},
  year={2026}
}
```
