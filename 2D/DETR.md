# DETR（DEtection TRansformer）完整流程笔记

> **一句话**：DETR 把目标检测重新表述成**集合预测**问题——用 CNN 提特征，用 Transformer 编码器/解码器做全局建模，用 **100 个可学习的 object query** 一次性吐出 100 个框，再用**匈牙利算法**做一对一二分图匹配训练，**彻底扔掉了 anchor 和 NMS**。
>
> **论文**：Carion et al., *End-to-End Object Detection with Transformers*, ECCV 2020
>
> **贯穿例子**：输入一张 `800×1066` 的图（DETR 的标准缩放：短边固定 800，长边 ≤ 1333），图中有 **3 辆车 + 2 个人**共 5 个目标。全文所有维度都跟着这张图走。

---

## 1. 要解决什么问题

传统检测器（Faster R-CNN / RetinaNet / FCOS）都依赖两大手工组件：

| 组件 | 作用 | 带来的问题 |
|---|---|---|
| **Anchor** | 用预设的密铺框做先验 | 尺度/长宽比要调，对不同数据集敏感；正负样本分配规则（IoU 阈值）也是超参 |
| **NMS** | 后处理去重 | 不是端到端的梯度不可导步骤；阈值调不好就漏检/重复；拥挤场景尤其难受 |

DETR 的野心：**输入图片 → 直接输出一组不重复的框**，中间没有任何需要手调的"后处理规则"。它靠三件事做到：

1. **一对一匹配**：每个 GT 只匹配**一个**预测 → 天然不需要 NMS 去重；
2. **全局注意力**：每个 query 都能看到整张图 → 不需要 anchor 这种局部先验；
3. **匈牙利算法**：把"哪个预测对哪个 GT"变成一个可离线求解的最优分配问题 → 不需要 IoU 阈值这种手写规则。

**代价**：收敛极慢（300 epoch ≈ 16 张 V100 训 3 天，改进版 500 epoch），小目标性能弱。

---

## 2. 整体结构（一张图）

```
[1,3,800,1066]  输入图（短边缩放到 800）
      │
      ▼  ResNet-50（stride 32，BN 冻结）
[1,2048,25,34]  C5 特征图
      │
      ▼  1×1 conv 降维（input_proj）
[1,256,25,34]   src            ────┐
      │                            │ 2D sin 位置编码（独立张量）
      ▼  flatten(2).permute(2,0,1) │
[850,1,256]     src_seq            │
      │                            ▼
      ▼  ┌──────────────────────────────────┐
         │  Transformer Encoder × 6 层       │  ← 自注意力：850 个 token 互相看
         │  (自注意力 + FFN，Post-Norm)      │     Q/K 加 pos，V 不加
         └──────────────────────────────────┘
      │
      ▼  memory [850,1,256]
      │
      │   query_embed [100,256]（object queries，可学习）
      ▼  ┌──────────────────────────────────┐
         │  Transformer Decoder × 6 层       │  ① 100 个 query 自注意力
         │  自注意力 + 交叉注意力 + FFN       │  ② 交叉注意力：query 看 memory 的 850 个 token
         └──────────────────────────────────┘
      │
      ▼  hs [6,1,100,256]（6 层每层都输出一份）
[1,100,256]  取最后一层 hn
      │
      ├─► class_embed: Linear(256, 92)         → [1,100,92]   91 类 + no-object
      └─► bbox_embed : MLP(256→256→256→4) + sigmoid → [1,100,4] 归一化 cxcywh ∈ [0,1]
      │
      ▼
100 个预测  ──训练：匈牙利匹配 + 集合损失──►  推理：卡阈值 0.7 直接输出，无 NMS
```

---

## 3. 逐模块拆解 + 维度流水账（跟着贯穿例子算）

### 3.0 维度总表（输入 `800×1066`）

| 步骤 | 操作 | 张量形状 |
|---|---|---|
| 输入 | 短边缩放到 800 | `[1, 3, 800, 1066]` |
| Backbone | ResNet-50，stride 32 | `[1, 2048, 25, 34]` |
| input_proj | `Conv2d(2048, 256, 1)` | `[1, 256, 25, 34]` |
| 位置编码 | 2D sin，y 128 维 + x 128 维 | `[1, 256, 25, 34]` |
| flatten | `flatten(2).permute(2,0,1)` | `[850, 1, 256]` |
| Encoder | 6 层 ×（自注意力 + FFN） | `[850, 1, 256]` |
| query_embed | `nn.Embedding(100, 256)` | `[100, 1, 256]` |
| Decoder | 6 层 ×（自注意力 + 交叉注意力 + FFN） | `[6, 1, 100, 256]` |
| 取最后一层 | `hs[-1]` | `[1, 100, 256]` |
| class_embed | `Linear(256, 92)` | `[1, 100, 92]` |
| bbox_embed | `MLP(256→256→256→4)` + sigmoid | `[1, 100, 4]` |

**关键数字**：`800/32 = 25`，`1066/32 ≈ 33.3 → 34`，`25 × 34 = 850`。这就是 DETR 里到处出现的 **N = 850**。

### 3.1 Backbone：ResNet-50（改造点）

- 取 ResNet 的最后一个 stage 输出 C5，**stride = 32**，通道 2048；
- **BN 冻结**（`FrozenBatchNorm2d`）——因为 batch size 太小（DETR 只用了 2 张/卡）；
- **backbone 学习率极小**：`lr_backbone = 1e-5`，而 Transformer 用 `1e-4`。原因：预训练权重已经很好了，动太多反而破坏。

【贯穿例子】`800×1066 → 25×34`，整张图被压成 850 个"格子"，每个格子对应原图 `32×32` 的区域。

### 3.2 1×1 卷积降维（input_proj）

`Conv2d(2048, 256, 1)`：把 backbone 的 2048 通道压到 Transformer 的 `d_model = 256`。**参数量只有 `2048×256 ≈ 52 万`**，纯通道对齐，不动空间分辨率。

【贯穿例子】`[1,2048,25,34] → [1,256,25,34]`。

### 3.3 位置编码：2D 正弦编码（关键细节）

Transformer 本身**对顺序无感**（置换等变），但检测需要位置信息，所以 DETR 加 2D 正弦位置编码：

```
y 方向：128 维正弦编码（按行号 1..25 生成，再按 mask 归一化到 [0, 2π]）
x 方向：128 维正弦编码（按列号 1..34 生成）
pos = cat([pos_y, pos_x], dim=1)   → 256 通道
```

三个容易踩的细节：

1. **位置编码不进 backbone 之外的输入，是独立传进去的**：官方实现里 `pos` 从头到尾作为一个单独张量在传，**不是**在输入端就加到 `src` 上（Deformable DETR 才改成直接加到 `src`）；
2. **只加在 Q/K 上，不加在 V 上**：
   - Encoder 自注意力：`q = k = src + pos, v = src`
   - Decoder 自注意力：`q = k = tgt + query_pos, v = tgt`
   - Decoder 交叉注意力：`q = tgt + query_pos, k = memory + pos, v = memory`
   - 直觉：位置是用来决定"该看谁"的（注意力权重），而不是用来污染内容的；
3. **归一化 + scale = 2π**：图像尺寸不定，先用 mask 的累积和归一化，再乘 2π。

【贯穿例子】`pos` 形状 `[1,256,25,34]`，flatten 后 `[850,1,256]`，与 `src` 完全对齐。

### 3.4 Encoder：6 层，全局自注意力

每层结构（官方默认 **Post-Norm**，即 `x = norm(x + sublayer(x))`）：

```
q = k = src + pos ;  v = src
src = norm1(src + dropout(self_attn(q, k, v)))
src = norm2(src + dropout(FFN(src)))        # FFN: 256 → 2048 → 256
```

配置：`d_model=256`，`8 头`（每头 `d_k = 32`），`FFN 隐藏层 2048`，`dropout 0.1`，共 6 层。

【贯穿例子】**Encoder 是这个模型的性能瓶颈**，算一笔账：

| 注意力 | 尺寸 | 每头每层权重数 | × 8 头 × 6 层 |
|---|---|---|---|
| Encoder 自注意力 | 850 × 850 | 722,500 | **≈ 3468 万** |
| Decoder 自注意力 | 100 × 100 | 10,000 | 48 万 |
| Decoder 交叉注意力 | 100 × 850 | 85,000 | 408 万 |

Encoder 占了全模型注意力开销的 **88%**（3468 / 3924）。这就是后来 **Deformable DETR** 只用可变形注意力稀疏采样、把 encoder 重做的原因。

### 3.5 Decoder：100 个 query + 交叉注意力

配置与 encoder 相同（6 层、8 头、256 维），但多了一个**交叉注意力**子层，共三个子层：

```
① 自注意力：100 个 query 互相看      q = k = tgt + query_pos, v = tgt
② 交叉注意力：query 看 850 个记忆    q = tgt + query_pos, k = memory + pos, v = memory
③ FFN
```

**object query 到底是什么？** 就是 `nn.Embedding(100, 256)` 学出来的 100 个 256 维向量，每个 query 对应最终的一个预测框。官方实现里有个反直觉的细节：

```python
query_embed = self.query_embed.weight          # [100, 256]  可学习
tgt = torch.zeros_like(query_embed)            # [100, 256]  初始全 0！
```

即 **decoder 的输入 tgt 是全零，那 100 个可学习向量是当作 `query_pos` 用的**（每层都加到 Q/K 上）。论文插图里把 object queries 画成输入，实现上它更像"位置先验"。这一点经常被面试追问。

【贯穿例子】`tgt [100,1,256]` → 6 层 decoder → `hs [6,1,100,256]`。**6 层每层都输出一份**，因为要用辅助损失。

### 3.6 两个预测头

```python
self.class_embed = nn.Linear(256, 91 + 1)   # 91 类 + 1 个 no-object
self.bbox_embed  = MLP(256, 256, 4, 3)      # Linear→ReLU→Linear→ReLU→Linear
```

- **分类头用 softmax（不是 sigmoid）**：因为"没有物体"被显式建模成第 92 维（index 91），成为一个正常的类别；
- **框回归头输出归一化的 `cxcywh`，再 sigmoid 压到 [0,1]**（相对于整张图，不是相对于 anchor）。推理时乘回原图尺寸即可；
- 不用 anchor ⇒ 不需要"编码解码 offset"那一套，回归目标就是直接的归一化坐标。

【贯穿例子】`[1,100,256]` → 分类 `[1,100,92]`，框 `[1,100,4]`。

---

## 4. 训练核心：匈牙利匹配 + 集合损失

这是 DETR 最精髓、也最容易被问倒的部分。

### 4.1 为什么需要匹配

DETR 输出 100 个预测，GT 只有 5 个（3 车 + 2 人）。100 个预测是**无序的集合**，所以"A 预测对应 B 标注"这件事必须先定下来，才能算损失。DETR 用**二分图最优匹配**：让匹配代价总和最小。

### 4.2 匹配代价（Match Cost）

```
σ̂ = argmin_σ  Σ_{i=1}^{N} L_match(y_i, ŷ_σ(i))

L_match(y_i, ŷ_σ(i)) = -1{c_i ≠ ∅} · p̂_σ(i)(c_i)          ← 分类代价
                     + 1{c_i ≠ ∅} · L_box(b_i, b̂_σ(i))     ← 框代价

L_box = λ_l1 · ‖b_i - b̂‖₁ + λ_giou · L_giou(b_i, b̂)      λ_l1 = 5, λ_giou = 2
```

四个要点：

1. **只用"目标类别的预测概率的负值"做分类代价**，不是对所有类做交叉熵——因为这是"分配"不是"分类"；
2. **∅（no-object）不参与匹配代价**——一个预测框不该因为"它说自己是背景"而被选中，也不该因此被惩罚；
3. **框代价 = L1 + GIoU**：L1 管绝对位置，GIoU 管重叠度，两者互补（纯 L1 对大框小框一视同仁，尺度不敏感）；
4. **求解用 `scipy.optimize.linear_sum_assignment`**，复杂度 `O(N³)`。这里 N 是 100 和 5 取大值，实际很便宜，而且**只在训练时跑**，推理时不需要。

【贯穿例子】100 × 5 的代价矩阵，每条边算出 `1·cost_cls + 5·cost_l1 + 2·cost_giou`，然后用匈牙利算法找出唯一的最优 5 条边。

### 4.3 最终损失（Hungarian Loss）

```
L_Hungarian(y, ŷ) = Σ_{i=1}^{N} [ -log p̂_σ̂(i)(c_i) + 1{c_i ≠ ∅} · L_box(b_i, b̂_σ̂(i)) ]
```

- **分类项**：匹配上的 5 个 query 用真实类别做交叉熵；**其余 95 个 query 全部归到 ∅ 类**，但 **∅ 类的 log-prob 降权 0.1**（否则 100 个里 95 个是负样本，类别极度不平衡，模型会退化成"全预测背景"）；
- **框回归项**：只对匹配上的 5 个 query 计算，权重与匹配代价一致（L1×5，GIoU×2）；
- 论文也试过 focal loss 变体（`--focal_alpha`），效果略好，但最终选了更简单的"softmax CE + ∅ 降权 0.1"。

【贯穿例子】5 个正样本 + 95 个负样本 → 每个正样本贡献 1 项分类 + 1 项 L1 + 1 项 GIoU，每个负样本只贡献 1 项降权的分类。

### 4.4 辅助损失（Auxiliary Loss）

**6 层 decoder 每层的输出都接同一个预测头，都算一遍完整损失**（`hs` 的 6 份输出全部参与）。

- 目的：解决"深监督"问题，逼迫浅层 decoder 也学好，显著加快收敛；
- 匹配是**每层各自独立做**的（不是只在最后一层匹配好再共享）；
- 效果：去掉辅助损失会让 AP 掉 1~2 个点。后续 DINO 等模型还在此基础上发展出 **look-forward-twice**、**去噪训练（DN）**。

### 4.5 一对一匹配 vs 一对多

| | 传统检测器（RetinaNet 等） | DETR |
|---|---|---|
| 分配方式 | 一个 GT 配多个 anchor（一对多） | 一个 GT 只配一个预测（一对一） |
| 结果 | 同一物体有多个正样本预测 → 需要 **NMS** 去重 | 天然无重复 → **不需要 NMS** |
| 梯度信号 | 每个 GT 有几十个正样本，监督稠密 | 每个 GT 只有 1 个正样本，监督稀疏 → **收敛慢** |

---

## 5. 推理流程（比训练简单太多）

```python
# 训练完的推理：一次前向，没有任何后处理
outputs = model(img)                    # [1, 100, 92], [1, 100, 4]
probs = outputs['pred_logits'].softmax(-1)[..., :-1]      # 丢掉 no-object，[1,100,91]
scores, labels = probs.max(-1)          # 每个 query 取最高分的类
# 直接卡阈值（官方用 0.7），不做 NMS
keep = scores > 0.7
```

- **没有 anchor 生成**、**没有 ROI 池化**、**没有 NMS**——一张图一次前向就出结果；
- 阈值 0.7 之所以能这么高，正是因为一对一训练让每个物体只有一个高分预测，不需要靠 NMS 收拾重复框。

【贯穿例子】100 个 query 中，5 个分数 > 0.7（正好对应 3 车 + 2 人），另外 95 个全被判为 ∅。

---

## 6. 结果与后续演进

### 6.1 论文报告的结果

- **DETR-R50（300 epoch）：COCO val 42.0 AP**；训到 500 epoch 提升到 **43.3 AP**；
- **DETR-DC5-R50**（把 backbone 最后 stage 的 stride 换成 dilation，`--dilation`，输出 stride 变 16、token 数 ×4）：500 epoch **44.9 AP**；
- **大目标很强，小目标明显偏弱**（AP_L 高，AP_S 只有 20 出头），主要原因就是 stride 32 的特征图太粗（小目标只剩 1 个格子）——**这正是 FPN 当年要解决的问题，DETR 一开始把它丢了**。

### 6.2 后续工作（演进路线）

| 模型 | 核心改进 | 解决的问题 |
|---|---|---|
| **Deformable DETR** | 多尺度特征 + 可变形注意力（每个 query 只采样 4 个点） | 收敛慢（50 epoch 达到 DETR 500 epoch 水平）、小目标差、encoder 太慢 |
| **DAB-DETR** | 把 query 显式建模成 4D anchor box（x, y, w, h） | query 语义不明、收敛慢 |
| **DN-DETR** | 去噪训练：给 GT 框加噪声喂给 decoder，让模型学会"还原" | 匈牙利匹配不稳定（同一图两次训练匹配结果可能不同） |
| **DINO** | 对比去噪 + look-forward-twice + 混合 query 选择 | 集大成者，COCO 63 AP 级 |
| **RT-DETR** | 高效混合编码器 + IoU-aware query 选择 | 实时化（首个超过 YOLO 的实时端到端检测器） |

**一句话演进逻辑**：DETR 证明了"端到端集合预测"可行 → 后面所有人都在补它的三个短板：**收敛慢、小目标差、encoder 太贵**。

---

## 7. 面试常见追问

- **Q：DETR 为什么不需要 NMS？**
  A：训练时用匈牙利算法做**一对一**匹配，一个 GT 只监督一个预测框，模型被训练成"每个物体只出一个框"，所以推理时不存在重复框，不需要 NMS。

- **Q：object query 是什么？**
  A：`nn.Embedding(100, 256)` 学出来的 100 个向量，每个对应一个预测槽位。官方实现中 decoder 输入 `tgt` 初始化为全 0，这 100 个向量是作为 `query_pos` 用的，每层都加到 Q/K 上。它们通过交叉注意力"查询"图像特征，逐渐特化（有的专门找车、有的专门找人、有的专门找大物体）。

- **Q：位置编码为什么只加在 Q/K 上，不加在 V 上？**
  A：位置信息的作用是决定"注意力权重应该分给谁"，属于匹配信号；把它加进 V 会污染内容表示。DETR encoder/decoder 全部遵循这个原则。

- **Q：为什么 DETR 收敛这么慢？**
  A：三个原因叠加——① 每个 GT 只有一个正样本，监督信号极度稀疏；② 用的是 Post-Norm（`norm(x + sub(x))`），深层梯度不稳，后来大家改 Pre-Norm；③ 初期注意力接近均匀分布，模型没有 anchor 那样的尺度/位置先验，要从零学会"往哪看"。

- **Q：DETR 的分类头为什么用 softmax 而不是 sigmoid？**
  A：因为把"无物体"显式建模成第 92 个类别（no-object）。既然每个 query 必然属于"91 类之一 + 无物体"这个互斥集合，用 softmax 更自然。代价是类别不平衡，所以 ∅ 类降权 0.1。

- **Q：为什么小目标效果差？**
  A：特征图 stride 32，`800×1066` 只压成 `25×34`。一个 `32×32` 的目标在特征图上只占 1 个格子——FPN 当年就是为了解决这个问题，而 DETR 一开始没有多尺度。Deformable DETR 加上多尺度后才好转。

- **Q：匈牙利匹配的代价为什么用 L1 + GIoU 而不是 IoU？**
  A：GIoU 在框不重叠时仍有梯度（IoU 为 0 就没梯度了），L1 则提供与尺度无关的绝对位置约束，两者互补。权重 5:2 是消融实验调出来的。

---

## 8. 最小实现（PyTorch，跟贯穿例子跑一遍）

### 8.1 维度自检

```python
import torch
import torch.nn as nn

# 贯穿例子：一张 800×1066 的图（短边 800，长边 1066 ≤ 1333）
feat = torch.randn(1, 2048, 25, 34)          # ResNet-50 输出，stride 32

proj = nn.Conv2d(2048, 256, 1)               # input_proj
src = proj(feat)                             # [1, 256, 25, 34]

pos = torch.randn(1, 256, 25, 34)            # 2D sin 位置编码，同样是 256 通道

# flatten 成序列：flatten(2) 把 H,W 合并，permute 把序列维放最前
src_seq = src.flatten(2).permute(2, 0, 1)    # [850, 1, 256]
pos_seq = pos.flatten(2).permute(2, 0, 1)    # [850, 1, 256]

# Encoder 自注意力的 Q/K/V（位置只进 Q/K）
q = k = src_seq + pos_seq                    # [850, 1, 256]
v = src_seq                                  # [850, 1, 256]

query_embed = nn.Embedding(100, 256).weight  # [100, 256]  object queries
query_pos = query_embed.unsqueeze(1).repeat(1, 1, 1)   # [100, 1, 256]
tgt = torch.zeros_like(query_pos)            # [100, 1, 256]  官方实现：输入全 0

print(src_seq.shape, query_pos.shape, tgt.shape)
# torch.Size([850, 1, 256]) torch.Size([100, 1, 256]) torch.Size([100, 1, 256])
```

### 8.2 匈牙利匹配 + 集合损失（骨架，省略 batch / 辅助损失细节）

```python
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou


def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h,
                        cx + 0.5 * w, cy + 0.5 * h], dim=-1)


@torch.no_grad()
def hungarian_match(logits, boxes, targets):
    """logits: [100,92]  boxes: [100,4] (归一化 cxcywh)
       targets: {'labels': [M], 'boxes': [M,4]}"""
    prob = logits.softmax(-1)                       # [100, 92]
    tgt_cls, tgt_box = targets['labels'], targets['boxes']

    cost_cls  = -prob[:, tgt_cls]                   # [100, M]  取目标类概率的负值
    cost_l1   = torch.cdist(boxes, tgt_box, p=1)    # [100, M]
    cost_giou = -generalized_box_iou(               # [100, M]
        box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(tgt_box))

    cost = 1 * cost_cls + 5 * cost_l1 + 2 * cost_giou
    i, j = linear_sum_assignment(cost.cpu())        # 最优分配
    return torch.as_tensor(i), torch.as_tensor(j)


def detr_loss(logits, boxes, targets, num_classes=91, matched=None):
    """logits: [100,92]  boxes: [100,4]  targets 同上一对一匹配后的下标"""
    src_idx, tgt_idx = matched                      # 匹配上的 query 下标 / GT 下标

    # ── ① 分类：匹配上的用真实类，其余 95 个全归到 no-object(第 91 类) ──
    target_classes = torch.full((logits.shape[0],), num_classes, dtype=torch.long)
    target_classes[src_idx] = targets['labels'][tgt_idx]

    # ∅ 类降权 0.1：否则 95 个负样本会压垮 5 个正样本
    weight = torch.ones(num_classes + 1)
    weight[num_classes] = 0.1
    loss_ce = F.cross_entropy(logits, target_classes, weight=weight)

    # ── ② 框回归：只算匹配上的那 5 个 ──
    src_box = boxes[src_idx]                        # [M, 4]
    tgt_box = targets['boxes'][tgt_idx]             # [M, 4]
    loss_l1 = F.l1_loss(src_box, tgt_box, reduction='mean')
    loss_giou = (1 - torch.diag(generalized_box_iou(   # GIoU 的损失形式
        box_cxcywh_to_xyxy(src_box), box_cxcywh_to_xyxy(tgt_box)))).mean()

    return loss_ce + 5 * loss_l1 + 2 * loss_giou
```

**训练时还需要**：6 层 decoder 输出全部算损失（辅助损失）、`loss_dict` 里同时记录 `loss_ce / loss_bbox / loss_giou / cardinality_error`、梯度裁剪 `max_norm=0.1`、优化器 AdamW（`lr=1e-4`，backbone `1e-5`，weight decay `1e-4`）、学习率在第 200 epoch 除以 10。

---

## 9. 速记卡

- **一句话**：CNN 提特征 → Transformer 编码全局 → **100 个 object query** 解码 → **匈牙利一对一匹配**训练 → 直接输出集合，**无 anchor、无 NMS**；
- **维度记法**：`800×1066 → (stride 32) → 25×34 = 850 个 token`；`d_model=256`，`8 头`，`FFN=2048`，`6 层 encoder + 6 层 decoder`，`100 个 query`；
- **两个头**：分类 `Linear(256, 91+1)`、回归 `MLP(256→256→256→4) + sigmoid`，输出归一化 `cxcywh`；
- **位置编码**：2D 正弦，`128(y) + 128(x) = 256`，**只加 Q/K，不加 V**；
- **匹配代价**：`-p̂(c_i) + 5·L1 + 2·GIoU`，用 `linear_sum_assignment` 求解，**只在训练时**；
- **损失**：匹配上的算分类+框，其余 95 个 query 归到 ∅ 类并**降权 0.1**；6 层 decoder 输出全算损失（辅助损失）；
- **三大短板**：收敛慢（300 epoch / 16 卡 3 天）、小目标差（stride 32 太粗）、encoder 贵（850² 自注意力占 88% 注意力开销）；
- **演进**：Deformable DETR（可变形注意力 + 多尺度）→ DAB-DETR（4D anchor query）→ DN-DETR（去噪）→ DINO → RT-DETR（实时）。
