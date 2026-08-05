import torch
import torch.nn as nn
import torch.optim as optim


class OptimizerDemo:
    
    def momentum_demo(self):
        """
        动量法（Momentum）
        
        核心思想：引入"惯性"概念，积累历史梯度信息
        
        更新公式：
        v_t = momentum * v_{t-1} + learning_rate * grad
        param_t = param_{t-1} - v_t
        
        关键参数：
        - momentum：动量系数，通常设为0.9，控制历史梯度的衰减速度
        - 可以理解为对过去梯度的指数加权平均，系数越大，历史梯度影响越大
        
        解决的问题：
        1. 标准SGD在遇到局部极小值或鞍点时梯度为0，更新停滞。动量法凭借历史积累的"冲量"可以冲过去
        2. 在梯度方向频繁变化的峡谷地形中，动量法可以平滑震荡，加速收敛
        
        通俗理解：好比下山时有了惯性，即使当前坡度变缓甚至有小坑，也能借助之前的势头继续前进
        """
        # 模拟一个简单的线性层参数
        params = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        # 模拟一个损失值（实际中由loss.backward()自动计算）
        loss = (params - torch.tensor([0.0, 0.0, 0.0])).sum()
        loss.backward()
        
        # 动量优化器，lr=0.01，momentum=0.9
        optimizer = optim.SGD([params], lr=0.01, momentum=0.9)
        optimizer.step()
        
        print(f"Momentum更新后的参数: {params.data}")
        # 注意：第一次step时v=0，实际就是标准SGD，第二次开始才有动量积累
        # 实际使用中需要多次迭代才能体现动量效果
    
    def adagrad_demo(self):
        """
        AdaGrad（Adaptive Gradient，自适应梯度）
        
        核心思想：为每个参数分配不同的学习率，频繁更新的参数学习率衰减更快
        
        更新公式：
        grad_square_sum_t = grad_square_sum_{t-1} + grad^2
        param_t = param_{t-1} - (learning_rate / sqrt(grad_square_sum_t + epsilon)) * grad
        
        关键点：
        - 每个参数有自己的学习率，由该参数历史所有梯度的平方和决定
        - 历史梯度平方和越大（更新越频繁），学习率越小
        - epsilon（通常1e-8）是为了防止除零
        
        解决的问题：
        - 稀疏数据场景：那些频繁出现的特征对应的参数，学习率会逐渐变小；而稀疏特征对应的参数，因为梯度平方和小，学习率仍然较大，能有效学习
        
        缺点：
        - 学习率单调递减，训练后期学习率趋近于0，模型提前停止学习
        - 这个问题催生了RMSprop和Adam
        
        适用场景：自然语言处理中的词嵌入训练等稀疏数据场景
        """
        params = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        loss = (params - torch.tensor([0.0, 0.0, 0.0])).sum()
        loss.backward()
        
        # AdaGrad优化器，lr是初始学习率，eps防止除零
        optimizer = optim.Adagrad([params], lr=0.01, eps=1e-8)
        optimizer.step()
        
        print(f"AdaGrad更新后的参数: {params.data}")
        # 实际使用中，不同参数的学习率差异会逐渐拉大
        # 频繁更新的参数学习率小，不常更新的参数学习率大
    
    def rmsprop_demo(self):
        """
        RMSprop（Root Mean Square Propagation）
        
        核心思想：改进AdaGrad学习率单调递减的问题，使用指数加权移动平均代替历史平方和
        
        更新公式：
        grad_square_avg_t = beta * grad_square_avg_{t-1} + (1 - beta) * grad^2
        param_t = param_{t-1} - (learning_rate / sqrt(grad_square_avg_t + epsilon)) * grad
        
        关键点：
        - beta（通常0.9）：控制历史梯度平方的衰减速度，与动量法中的momentum类似
        - 使用移动平均而非累加，所以历史信息会逐渐"遗忘"，学习率不会持续衰减
        - 有效解决了AdaGrad后期学习率消失的问题
        
        解决的问题：
        - 避免了AdaGrad的提前停止问题
        - 在非凸优化（神经网络）中表现稳定，能自适应调整学习率
        - 对RNN等梯度变化剧烈的网络效果较好
        
        与AdaGrad的区别：
        - AdaGrad：累加所有历史梯度平方 → 学习率只降不升
        - RMSprop：移动平均历史梯度平方 → 学习率可以上升也可以下降
        """
        params = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        loss = (params - torch.tensor([0.0, 0.0, 0.0])).sum()
        loss.backward()
        
        # RMSprop优化器，alpha即beta（衰减系数），eps防止除零
        optimizer = optim.RMSprop([params], lr=0.01, alpha=0.9, eps=1e-8)
        optimizer.step()
        
        print(f"RMSprop更新后的参数: {params.data}")
        # RMSprop在动态调整每个参数的学习率，但不会像AdaGrad那样降到零
    
    def adam_demo(self):
        """
        Adam（Adaptive Moment Estimation，自适应矩估计）
        
        核心思想：同时结合动量法（一阶矩）和RMSprop（二阶矩）的思想
        
        更新公式（简写）：
        m_t = beta1 * m_{t-1} + (1 - beta1) * grad          # 一阶矩：梯度的移动平均（动量）
        v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2        # 二阶矩：梯度平方的移动平均（RMSprop）
        
        m_hat_t = m_t / (1 - beta1^t)   # 偏差校正，防止初期偏向0
        v_hat_t = v_t / (1 - beta2^t)   # 偏差校正
        
        param_t = param_{t-1} - (lr / sqrt(v_hat_t) + eps) * m_hat_t
        
        关键参数（推荐默认值）：
        - beta1 = 0.9：一阶矩衰减系数（动量）
        - beta2 = 0.999：二阶矩衰减系数（RMSprop）
        - eps = 1e-8：防止除零
        
        核心设计：
        1. 一阶矩（m）：记录梯度方向，平滑梯度波动，加速收敛（动量法）
        2. 二阶矩（v）：记录梯度幅度，自适应调整学习率（RMSprop）
        3. 偏差校正：训练初期m和v偏向0，通过除以(1 - beta^t)进行修正
        
        解决的问题：
        - 结合了动量法的快速收敛和RMSprop的自适应学习率
        - 对超参数（学习率）不太敏感，鲁棒性好
        - 适用于大规模数据和复杂模型
        
        通俗理解：
        - 一阶矩决定了"往哪个方向走"（速度方向）
        - 二阶矩决定了"每步走多远"（自适应步长）
        - 两者结合 = 既能快速冲过平坦区域，又能在陡峭区域自动减小步长
        
        现状：目前深度学习中最主流的优化器，在CV、NLP、推荐系统等领域广泛应用
        """
        params = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        loss = (params - torch.tensor([0.0, 0.0, 0.0])).sum()
        loss.backward()
        
        # Adam优化器，betas=(beta1, beta2)
        optimizer = optim.Adam([params], lr=0.001, betas=(0.9, 0.999), eps=1e-8)
        optimizer.step()
        
        print(f"Adam更新后的参数: {params.data}")
        # 实际使用中，Adam结合了动量和自适应学习率，收敛速度快且稳定