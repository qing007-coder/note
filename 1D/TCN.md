
# TCN（Temporal Convolutional Network）完整流程

> TCN = Temporal Convolutional Network，时间卷积网络  
> 核心思想：使用 **1D 因果卷积（Causal Convolution）+ 空洞卷积（也叫膨胀卷积，Dilated Convolution）+ 残差连接（Residual Connection）** 来处理时间序列。

---

# 一、TCN 是干什么的？

TCN 主要用于处理：

- 时间序列预测
- 时间序列分类
- 异常检测
- 状态识别
- 传感器数据分析
- 自然灾害预测

例如自然灾害项目中：

```text
时间
 ↓
t1    t2    t3    t4    t5    t6    t7
 ↓     ↓     ↓     ↓     ↓     ↓     ↓
雨量  土壤  IMU   位移  水压  姿态  ...
````

每一个时间点都有多个传感器数据：

```text
t1 = [雨量, 土壤含水率, IMU, 位移, 水压, 姿态]
t2 = [雨量, 土壤含水率, IMU, 位移, 水压, 姿态]
t3 = [雨量, 土壤含水率, IMU, 位移, 水压, 姿态]
...
```

最终形成：

```text
时间序列
    ↓
TCN
    ↓
预测未来状态
```

例如：

```text
过去一段时间：

雨量
土壤含水率
IMU
位移
孔隙水压力
姿态

        ↓

       TCN

        ↓

未来滑坡风险概率
```

---

# 二、TCN 的核心组成

TCN 最重要的三个组成部分：

```text
1. Causal Convolution
   因果卷积

2. Dilated Convolution
   空洞卷积

3. Residual Connection
   残差连接
```

可以理解为：

```text
TCN
│
├── Causal Conv
│      ↓
│   保证不看未来
│
├── Dilated Conv
│      ↓
│   扩大时间感受野
│
└── Residual Connection
       ↓
    保证深层网络容易训练
```

---

# 三、输入数据

假设我们有：

```text
Batch = 8
时间长度 = 100
传感器特征 = 6
```

原始数据：

```text
[B, T, Features]
```

即：

```text
[8, 100, 6]
```

其中：

```text
B = Batch Size
T = Time
Features = 每个时间点的特征数量
```

例如：

```text
6个特征：

1. 雨量
2. 土壤含水率
3. IMU_x
4. IMU_y
5. 位移
6. 孔隙水压力
```

那么：

```text
t1 = [r1, s1, imu1, imu2, d1, p1]
t2 = [r2, s2, imu1, imu2, d2, p2]
t3 = [r3, s3, imu1, imu2, d3, p3]
...
```

形成：

```text
[8, 100, 6]
```

---

# 四、为什么 Conv1d 输入要变成 [B, C, T]？

PyTorch 的 `Conv1d` 要求输入格式：

```text
[B, C, T]
```

而我们原始时间序列一般是：

```text
[B, T, Features]
```

所以需要：

```python
x = x.transpose(1, 2)
```

例如：

```text
原始：

[8, 100, 6]

        ↓ transpose

[8, 6, 100]
```

此时：

```text
Batch = 8
Channels = 6
Time = 100
```

所以：

```text
[B, T, Features]
        ↓
transpose
        ↓
[B, Features, T]
```

---

# 五、1D 卷积在 TCN 中干什么？

假设：

```text
输入：

[B, 6, 100]
```

使用：

```python
nn.Conv1d(
    in_channels=6,
    out_channels=16,
    kernel_size=3
)
```

那么：

```text
6个输入通道
        ↓
16个卷积核
        ↓
16个输出通道
```

可以理解为：

```text
[B, 6, 100]
      ↓
Conv1D
      ↓
[B, 16, ...]
```

这里卷积主要是在：

> **时间维度上提取局部时间变化特征。**

例如：

```text
t1 → t2 → t3
```

卷积核可以学习：

```text
短时间内：

雨量怎么变化
土壤含水率怎么变化
位移怎么变化
IMU 怎么变化
```

---

# 六、为什么 6 个输入通道可以变成 16 个输出通道？

这是理解卷积非常重要的一点。

假设：

```text
in_channels = 6
out_channels = 16
kernel_size = 3
```

意味着：

```text
输入：
6个通道

输出：
16个通道
```

这里的：

```text
out_channels = 16
```

可以理解为：

```text
有16个卷积核
```

每一个卷积核都会同时处理：

```text
6个输入通道
```

所以：

```text
6维输入
   ↓
16个卷积核
   ↓
16维输出
```

也就是：

```text
[B, 6, T]
      ↓
Conv1d(6 → 16)
      ↓
[B, 16, T']
```

---

# 七、1×1 卷积为什么可以改变通道数？

这个在 TCN 的残差连接中非常重要。

假设：

```text
输入：

[B, 16, T]
```

经过主分支后：

```text
[B, 32, T]
```

这时候如果直接：

```python
out + x
```

是不行的。

因为：

```text
16 ≠ 32
```

所以需要：

```python
nn.Conv1d(
    16,
    32,
    kernel_size=1
)
```

于是：

```text
[B, 16, T]
      ↓
1×1 Conv
      ↓
[B, 32, T]
```

这样就可以：

```text
主分支：[B, 32, T]
残差支：[B, 32, T]

        ↓

       相加
```

---

# 八、1×1 卷积本质上是什么？

假设某一个时间点的数据是：

```text
16维向量
```

经过：

```text
1×1 Conv
```

变成：

```text
32维向量
```

本质上就是一个线性变换：

```text
16维
 ↓
矩阵 W
 ↓
32维
```

数学上：

```text
y = Wx + b
```

其中：

```text
x ∈ R^16

W ∈ R^(32×16)

y ∈ R^32
```

所以：

```text
1×1 Conv

16维 → 32维
```

本质就是：

> **对每一个时间点独立进行一次通道维度上的线性映射。**

注意：

```text
1×1 Conv
```

并不是在时间上提取邻域信息。

它主要负责：

```text
通道变换
```

---

# 九、普通卷积存在的问题

假设：

```text
kernel_size = 3
```

普通卷积一次只能看到附近的数据：

```text
t1  t2  t3
 ↓   ↓   ↓
 ●   ●   ●
```

如果预测：

```text
t3
```

那么它只能利用附近几个时间点的信息。

但是对于自然灾害：

```text
过去10分钟的降雨
过去30分钟的土壤含水率
过去几小时的位移变化
```

都可能影响当前风险。

因此：

> **我们需要让卷积看到更长时间范围的数据。**

---

# 十、Dilated Convolution（空洞卷积）

TCN 使用：

> **空洞卷积扩大感受野。**

假设：

```text
kernel_size = 3
```

---

## 10.1 dilation = 1

普通卷积：

```text
● ● ●
```

相当于：

```text
t-2  t-1  t
```

间隔：

```text
1
```

---

## 10.2 dilation = 2

变成：

```text
●   ●   ●
```

相当于：

```text
t-4  t-2  t
```

中间跳过一个时间点。

---

## 10.3 dilation = 4

变成：

```text
●       ●       ●
```

相当于：

```text
t-8  t-4  t
```

---

# 十一、为什么 dilation 能扩大感受野？

假设：

```text
kernel_size = 3
```

那么：

```text
dilation = 1

感受野 = 3
```

```text
dilation = 2

感受野 = 5
```

```text
dilation = 4

感受野 = 9
```

```text
dilation = 8

感受野 = 17
```

所以：

```text
dilation 越大
      ↓
感受野越大
      ↓
可以利用更久远的历史信息
```

---

# 十二、为什么经常使用 1、2、4、8？

典型 TCN：

```text
TCN Block 1
dilation = 1

        ↓

TCN Block 2
dilation = 2

        ↓

TCN Block 3
dilation = 4

        ↓

TCN Block 4
dilation = 8

        ↓

TCN Block 5
dilation = 16
```

形成：

```text
1 → 2 → 4 → 8 → 16
```

这样能够快速扩大感受野。

---

# 十三、Causal Convolution（因果卷积）

TCN 的另一个核心是：

> **当前时刻的预测不能使用未来的数据。**

例如：

```text
预测 t=10
```

只能使用：

```text
t1 ~ t10
```

不能使用：

```text
t11
t12
t13
...
```

否则就是：

```text
未来信息泄漏
```

---

# 十四、普通卷积为什么可能看到未来？

假设：

```text
kernel_size = 3
```

普通卷积可能看到：

```text
t-1   t   t+1
```

如果我们预测：

```text
t
```

那么：

```text
t+1
```

属于未来。

因此普通卷积不满足严格的时间因果关系。

---

# 十五、因果卷积怎么解决？

因果卷积只使用：

```text
过去 + 当前
```

例如：

```text
t-4   t-2   t
 ↓     ↓     ↓
 ●     ●     ●
```

不会使用：

```text
t+1
t+2
...
```

所以：

```text
Causal
=
不看未来
```

---

# 十六、Causal + Dilated

TCN 最核心的卷积其实可以理解成：

```text
Causal Convolution
        +
Dilated Convolution
```

也就是：

```text
既要：

不看未来

又要：

能够看很远的历史
```

例如：

```text
kernel_size = 3
dilation = 4
```

预测当前时刻：

```text
t-8      t-4      t
 ↓        ↓        ↓
 ●        ●        ●
```

可以看到：

```text
当前
 ↑
过去很远的数据
```

但是不会看到：

```text
未来
```

---

# 十七、TCN Block

一个经典 TCN Block 通常包含：

```text
输入
 ↓
Causal Dilated Conv
 ↓
ReLU
 ↓
Dropout
 ↓
Causal Dilated Conv
 ↓
ReLU
 ↓
Dropout
 ↓
Residual Connection
 ↓
输出
```

结构：

```text
                ┌──────────────────┐
                │                  │
                │   Residual       │
                │                  │
Input ──────────┤                  │
  │             │                  │
  ↓             │                  │
Conv1           │                  │
  ↓             │                  │
ReLU            │                  │
  ↓             │                  │
Dropout         │                  │
  ↓             │                  │
Conv2           │                  │
  ↓             │                  │
ReLU            │                  │
  ↓             │                  │
Dropout         │                  │
  │             │                  │
  └─────────────┴────── + ─────────┘
                         ↓
                       Output
```

---

# 十八、为什么一个 Block 中通常有两个卷积？

例如：

```text
Conv1
 ↓
ReLU
 ↓
Dropout
 ↓
Conv2
 ↓
ReLU
 ↓
Dropout
```

这样一个 Block 中可以进行两次特征提取。

第一层：

```text
提取局部时间特征
```

第二层：

```text
进一步组合时间特征
```

然后通过：

```text
Residual
```

与原始输入进行融合。

---

# 十九、Residual Connection（残差连接）

残差连接：

```text
Output = F(x) + x
```

其中：

```text
F(x)
```

就是卷积网络学习到的特征变化。

结构：

```text
             ┌───────────────┐
             │               │
x ───────────┤               │
│            │               │
│            ↓               │
│          Conv              │
│            ↓               │
│          ReLU              │
│            ↓               │
│          Conv              │
│            ↓               │
│          ReLU              │
│            │               │
└────────────┴────── + ──────┘
                       ↓
                     Output
```

数学上：

```text
y = F(x) + x
```

---

# 二十、为什么需要残差连接？

如果网络越来越深：

```text
TCN Block 1
 ↓
TCN Block 2
 ↓
TCN Block 3
 ↓
TCN Block 4
 ↓
TCN Block 5
```

可能会出现：

```text
梯度传播困难
训练困难
原始信息丢失
```

残差连接提供了一条 shortcut：

```text
x
│
├──────────────────────→
│                         ↓
│                        +
↓                         ↑
F(x) ─────────────────────┘
```

因此：

```text
y = F(x) + x
```

可以让：

```text
原始信息
```

更容易传到后面。

---

# 二十一、如果输入输出通道不同怎么办？

例如：

```text
输入：

[B, 16, T]
```

主分支：

```text
16 → 32
```

得到：

```text
[B, 32, T]
```

但是：

```text
Residual = [B, 16, T]
```

无法直接相加。

所以：

```text
[B, 16, T]
      ↓
1×1 Conv
      ↓
[B, 32, T]
```

最终：

```text
主分支：

[B, 32, T]

        +

残差分支：

[B, 32, T]

        ↓

[B, 32, T]
```

所以 TCN 中的：

```python
self.downsample = nn.Conv1d(
    in_channels,
    out_channels,
    kernel_size=1
)
```

主要就是干这个事情。

---

# 二十二、一个完整 TCN 网络

假设：

```text
输入特征：

6
```

设计：

```text
6 → 16 → 32 → 64
```

那么：

```text
Input
[B, 6, T]
      ↓
TCN Block 1
6 → 16
dilation = 1
      ↓
[B, 16, T]
      ↓
TCN Block 2
16 → 32
dilation = 2
      ↓
[B, 32, T]
      ↓
TCN Block 3
32 → 64
dilation = 4
      ↓
[B, 64, T]
```

---

# 二十三、完整 TCN 流程

```text
原始时间序列

[B, T, Features]
        ↓
   transpose
        ↓
[B, Features, T]
        ↓
────────────────────────
     TCN Block 1
     dilation = 1
────────────────────────
        ↓
────────────────────────
     TCN Block 2
     dilation = 2
────────────────────────
        ↓
────────────────────────
     TCN Block 3
     dilation = 4
────────────────────────
        ↓
────────────────────────
     TCN Block 4
     dilation = 8
────────────────────────
        ↓
    时间序列特征
        ↓
取最后一个时间点
        ↓
      Linear
        ↓
     Prediction
```

---

# 二十四、以滑坡预测为例

假设过去：

```text
60个时间点
```

每个时间点有：

```text
6个传感器特征
```

包括：

```text
1. 雨量
2. 土壤含水率
3. IMU
4. 位移
5. 孔隙水压力
6. 姿态
```

输入：

```text
[B, 60, 6]
```

首先：

```text
transpose
```

得到：

```text
[B, 6, 60]
```

然后：

```text
TCN Block 1

6 → 16
dilation = 1

[B, 6, 60]
      ↓
[B, 16, 60]
```

然后：

```text
TCN Block 2

16 → 32
dilation = 2

[B, 16, 60]
      ↓
[B, 32, 60]
```

然后：

```text
TCN Block 3

32 → 64
dilation = 4

[B, 32, 60]
      ↓
[B, 64, 60]
```

然后取最后一个时间点：

```python
x = x[:, :, -1]
```

得到：

```text
[B, 64]
```

最后：

```text
Linear
64 → 1
```

得到：

```text
[B, 1]
```

如果使用：

```python
sigmoid()
```

得到：

```text
0 ~ 1
```

例如：

```text
0.05
0.21
0.67
0.93
```

可以作为：

```text
滑坡风险概率
```

---

# 二十五、TCN 中到底学到了什么？

例如：

```text
过去60个时间点
```

TCN 并不是简单地：

```text
把60个数据直接平均
```

而是在学习：

```text
时间变化规律
```

例如可能学习到：

```text
降雨增加
    ↓
土壤含水率增加
    ↓
孔隙水压力增加
    ↓
位移开始变化
    ↓
IMU出现异常
    ↓
滑坡风险增加
```

所以 TCN 的价值在于：

> **学习多个传感器变量随时间变化的模式。**

---

# 二十六、TCN 和 LSTM 的区别

## LSTM

LSTM：

```text
t1 → LSTM → h1
             ↓
t2 → LSTM → h2
             ↓
t3 → LSTM → h3
             ↓
...
```

特点：

```text
按照时间顺序递归处理
```

---

## TCN

TCN：

```text
t1 t2 t3 t4 t5 t6 t7
 │  │  │  │  │  │  │
 └──┴──┴──┴──┴──┴──┘
          ↓
       Conv1D
          ↓
       TCN Block
```

特点：

```text
通过卷积处理时间序列
```

同时利用：

```text
Dilated Conv
```

扩大感受野。

---

# 二十七、TCN 的优势

## 1. 可以学习长时间依赖

通过：

```text
Dilated Convolution
```

扩大感受野。

---

## 2. 可以并行计算

相比传统 RNN：

```text
t1 → t2 → t3 → t4
```

TCN 的卷积计算更加适合 GPU 并行。

---

## 3. 不容易受到传统 RNN 长序列递归结构的限制

TCN 不需要像传统 RNN 那样：

```text
一个时间点
一个时间点
一个时间点
```

递归处理。

---

## 4. 适合多变量传感器数据

例如：

```text
雨量
土壤含水率
位移
IMU
水压
姿态
```

都可以作为输入通道。

---

# 二十八、TCN 的缺点

## 1. 需要合理设置感受野

如果：

```text
实际需要看过去100个时间点
```

但是：

```text
TCN感受野只有30
```

那么模型无法看到足够的历史信息。

---

## 2. dilation 不能无限增大

例如：

```text
1 → 2 → 4 → 8 → 16 → 32 → 64
```

虽然感受野越来越大，但是采样会越来越稀疏。

因此需要根据：

```text
时间窗口
采样频率
任务特点
```

进行设计。

---

# 二十九、感受野怎么计算？

对于一个简单的堆叠结构：

```text
kernel_size = K
```

如果每层 dilation 为：

```text
d1, d2, ..., dn
```

常见计算：

```text
Receptive Field
=
1 + Σ[(K - 1) × di]
```

例如：

```text
kernel_size = 3

dilation：

1
2
4
8
```

那么：

```text
RF
=
1 + 2 × (1 + 2 + 4 + 8)

=
1 + 30

=
31
```

也就是说理论上可以利用大约：

```text
31个时间位置
```

的信息。

> 注意：实际网络的感受野需要结合具体 Block 数量、每个 Block 中的卷积层数量、padding 等结构一起计算。

---

# 三十、PyTorch 实现一个 TCN Block

```python
import torch
import torch.nn as nn


class TCNBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        dropout=0.2
    ):
        super().__init__()

        # 因果卷积需要的 padding
        padding = (kernel_size - 1) * dilation

        # 第一层卷积
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.relu1 = nn.ReLU()

        self.dropout1 = nn.Dropout(dropout)

        # 第二层卷积
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.relu2 = nn.ReLU()

        self.dropout2 = nn.Dropout(dropout)

        # 如果输入输出通道不一样
        # 使用1×1卷积调整通道
        if in_channels != out_channels:

            self.downsample = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1
            )

        else:

            self.downsample = None


    def forward(self, x):

        # 保存残差
        residual = x

        # 第一层卷积
        out = self.conv1(x)

        # 裁剪右侧
        # 保证因果关系
        out = out[:, :, :-self.conv1.padding[0]]

        # 激活
        out = self.relu1(out)

        # Dropout
        out = self.dropout1(out)

        # 第二层卷积
        out = self.conv2(out)

        # 再次裁剪
        out = out[:, :, :-self.conv2.padding[0]]

        # 激活
        out = self.relu2(out)

        # Dropout
        out = self.dropout2(out)

        # 如果通道数不同
        if self.downsample is not None:

            residual = self.downsample(residual)

        # 残差连接
        out = out + residual

        return out
```

---

# 三十一、完整 TCN 网络

```python
class TCN(nn.Module):

    def __init__(
        self,
        input_channels,
        channels,
        kernel_size=3,
        dropout=0.2
    ):
        super().__init__()

        layers = []

        for i, out_channels in enumerate(channels):

            # 第一个Block
            if i == 0:

                in_channels = input_channels

            else:

                in_channels = channels[i - 1]

            # dilation：
            # 1 → 2 → 4 → 8 → ...
            dilation = 2 ** i

            layers.append(
                TCNBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)


    def forward(self, x):

        # 原始输入：
        # [B, T, C]

        x = x.transpose(1, 2)

        # 变成：
        # [B, C, T]

        x = self.network(x)

        # 取最后一个时间点
        x = x[:, :, -1]

        return x
```

---

# 三十二、创建 TCN

例如：

```python
model = TCN(
    input_channels=6,
    channels=[16, 32, 64]
)
```

表示：

```text
输入：

6个特征

        ↓

Block 1

6 → 16
dilation = 1

        ↓

Block 2

16 → 32
dilation = 2

        ↓

Block 3

32 → 64
dilation = 4
```

---

# 三十三、如果做二分类

例如：

```text
是否发生滑坡
```

TCN 提取：

```text
[B, 64]
```

然后：

```python
classifier = nn.Linear(64, 1)
```

输出：

```text
[B, 1]
```

训练：

```python
criterion = nn.BCEWithLogitsLoss()
```

预测：

```python
prob = torch.sigmoid(logit)
```

最终：

```text
0 ~ 1
```

可以表示：

```text
滑坡风险概率
```

---

# 三十四、如果做多分类

例如：

```text
正常
注意
预警
严重预警
```

那么：

```python
classifier = nn.Linear(64, 4)
```

得到：

```text
[B, 4]
```

然后：

```python
prob = torch.softmax(logits, dim=1)
```

得到：

```text
[
    正常概率,
    注意概率,
    预警概率,
    严重预警概率
]
```

---

# 三十五、TCN 在多模态自然灾害预测中的位置

对于你的自然灾害项目，可以理解成：

```text
              多模态传感器
                    │
       ┌────────────┼────────────┐
       ↓            ↓            ↓
      雨量        土壤含水率      位移
       │            │            │
       ├────────────┼────────────┤
                    ↓
              时间序列数据
                    ↓
                  TCN
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
    短期变化特征             长期变化特征
        ↓                       ↓
        └───────────┬───────────┘
                    ↓
                 特征融合
                    ↓
                 分类器
                    ↓
              滑坡风险概率
```

这里需要注意：

> **TCN 本身并不是专门的“多模态融合算法”。**

它更准确地说是：

```text
时间序列特征提取器
```

如果把：

```text
雨量
土壤含水率
位移
IMU
水压
```

作为不同的输入通道，那么 TCN 可以学习：

```text
不同传感器之间
+
不同时间之间
```

的变化关系。

---

# 三十六、TCN 最终记忆版

把 TCN 压缩成下面这张图：

```text
TCN
│
├── 输入时间序列
│      │
│      ↓
│   [B,T,C]
│      │
│      ↓
│   transpose
│      │
│      ↓
│   [B,C,T]
│
├── TCN Block
│      │
│      ├── Causal Conv
│      │      ↓
│      │   不看未来
│      │
│      ├── Dilated Conv
│      │      ↓
│      │   扩大感受野
│      │
│      ├── ReLU
│      │
│      ├── Dropout
│      │
│      ├── Conv
│      │
│      └── Residual
│             ↓
│          F(x) + x
│
├── 多个 Block
│      │
│      ├── dilation = 1
│      ├── dilation = 2
│      ├── dilation = 4
│      └── dilation = 8
│
└── 输出
       ↓
    时间序列特征
       ↓
     Linear
       ↓
    Prediction
```

---

# 三十七、一句话理解 TCN

```text
普通卷积：
看附近

Dilated Conv：
看得更远

Causal Conv：
只看过去

Residual：
让网络更容易训练

1×1 Conv：
调整通道数

TCN：
把这些东西组合起来处理时间序列
```

最终可以记成：

```text
TCN
=
Causal Conv
+
Dilated Conv
+
Residual Connection
```

而它解决的核心问题就是：

```text
如何利用过去一段时间的多变量数据
来预测当前/未来的状态。
```

```

这版从**输入张量形状 → Conv1D → 1×1 Conv → Causal → Dilated → Residual → TCN Block → 多层 TCN → 滑坡预测 → PyTorch 实现**完整串起来了，直接复制里面的 `markdown` 代码块即可。
```
