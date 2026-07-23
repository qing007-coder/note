import torch 
from typing import List
import numpy as np
from numbers import Number


class Tensor:

    def __init__(self):
        pass

    def tensor_from_num(self, num: float):
        """
        将一个数字转换为 PyTorch 张量
        :param num: 输入的数字
        """
        return torch.tensor(num, dtype=torch.float32)

    def tensor_from_list(self, lst: List):
        """
        将一个列表转换为 PyTorch 张量
        :param lst: 输入的列表
        """
        return torch.tensor(lst, dtype=torch.float32)
    
    def tensor_from_numpy(self, np_array: np.ndarray):
        """
        将一个 NumPy 数组转换为 PyTorch 张量
        :param np_array: 输入的 NumPy 数组
        """
        return torch.tensor(np_array, dtype=torch.float32)
    

    def print(self, tensor: torch.Tensor):
        """
        打印 PyTorch 张量的内容
        :param tensor: 输入的 PyTorch 张量
        """
        print(f"Tensor: {tensor}  Shape: {tensor.shape}  Dtype: {tensor.dtype}")

    def ones(self, shape: List[int]):
        """
        创建一个全为 1 的 PyTorch 张量
        :param shape: 张量的形状
        """
        return torch.ones(shape, dtype=torch.float32)
    
    def zeros(self, shape: List[int]):
        """
        创建一个全为 0 的 PyTorch 张量
        :param shape: 张量的形状
        """
        return torch.zeros(shape, dtype=torch.float32)
    
    def one_like(self, tensor: torch.Tensor):
        """
        创建一个与给定张量形状相同的全为 1 的张量
        :param tensor: 输入的 PyTorch 张量
        """
        return torch.ones_like(tensor, dtype=torch.float32)
    
    def zero_like(self, tensor: torch.Tensor):
        """
        创建一个与给定张量形状相同的全为 0 的张量
        :param tensor: 输入的 PyTorch 张量
        """
        return torch.zeros_like(tensor, dtype=torch.float32)
    
    def full_like(self, tensor: torch.Tensor, fill_value: float):
        """
        创建一个与给定张量形状相同的张量，并用指定的值填充
        :param tensor: 输入的 PyTorch 张量
        :param fill_value: 填充的值
        """
        return torch.full_like(tensor, fill_value, dtype=torch.float32)
    
    def arange(self, start: float, end: float, step: float):
        """
        创建一个指定范围的 PyTorch 线性张量 
        :param start: 起始值
        :param end: 结束值
        :param step: 步长
        """
        return torch.arange(start, end, step, dtype=torch.float32)
    
    def linspace(self, start: float, end: float, steps: int):
        """
        创建一个指定范围的 PyTorch 线性张量，包含指定数量的步数
        :param start: 起始值
        :param end: 结束值
        :param steps: 步数
        """
        return torch.linspace(start, end, steps, dtype=torch.float32)
    
    def rand(self, shape: List[int]):
        """
        创建一个指定形状的 PyTorch 随机张量
        :param shape: 张量的形状
        """

        # torch.initial_seed()  # 默认采用当前系统的时间戳作为随机种子
        torch.manual_seed(42)  # 设置随机种子.
        return torch.rand(shape, dtype=torch.float32)
    
    def randn(self, shape: List[int]):
        """
        创建一个指定形状的 PyTorch 正态分布随机张量
        :param shape: 张量的形状
        """
        return torch.randn(shape, dtype=torch.float32)
    
    def randint(self, low: int, high: int, shape: List[int]):
        """
        创建一个指定范围的 PyTorch 随机整数张量
        :param low: 最小值（包含）
        :param high: 最大值（不包含）
        :param shape: 张量的形状
        """
        return torch.randint(low, high, shape, dtype=torch.int32)

    def convert_to_float64(self, tensor: torch.Tensor):
        """
        将 PyTorch 张量转换为 float64 类型
        :param tensor: 输入的 PyTorch 张量
        """
        return tensor.type(torch.float64)

    def tensor_to_numpy(self, tensor: torch.Tensor):
        """
        将 PyTorch 张量转换为 NumPy 数组 tensor.numpy() 方法返回的是一个共享内存的 NumPy 数组, tensor.numpy().copy() 返回的是一个独立的 NumPy 数组。
        :param tensor: 输入的 PyTorch 张量
        """
        return tensor.numpy()

    def numpy_to_tensor(self, np_array: np.ndarray):
        """
        将 NumPy 数组转换为 PyTorch 张量 torch.from_numpy() 方法返回的是一个共享内存的 PyTorch 张量, torch.tensor(np_array) 返回的是一个独立的 PyTorch 张量。
        :param np_array: 输入的 NumPy 数组
        """
        return torch.from_numpy(np_array)
    
    def get_value(self, tensor: torch.Tensor):
        """
        获取 PyTorch 张量的值 等价于
            if tensor.numel() == 1:      # 如果张量只有 1 个元素
                return tensor.item()      # 返回标量值（如 3.14）
            else:                         # 如果张量有多个元素
                return tensor.tolist()    # 返回列表（如 [[1, 2], [3, 4]]）
        :param tensor: 输入的 PyTorch 张量
        """
        return tensor.item() if tensor.numel() == 1 else tensor.tolist() 

    def add_num(self, tensor: torch.Tensor, num: Number):
        """
        add sub mul div neg 同理
        等价于 return tensor + num
        :param tensor: 输入的 PyTorch 张量
        :param num: 要添加的数字
        """

        return tensor.add(num)
    
    def mul(self, tensor1: torch.Tensor, temsor2: torch.Tensor):
        """
        等价于 return tensor1 * tensor2
        :param tensor1: 输入的 PyTorch 张量 1
        :param temsor2: 输入的 PyTorch 张量 2
        """
        return tensor1.mul(temsor2)

    def matmul(self, tensor1: torch.Tensor, tensor2: torch.Tensor):
        """
        等价于 return tensor1 @ tensor2
        :param tensor1: 输入的 PyTorch 张量 1
        :param tensor2: 输入的 PyTorch 张量 2
        """
        return tensor1.matmul(tensor2)

    def sum(self, tensor: torch.Tensor,dim:int=None):
        """
        dim=0按列求和，dim=1按行求和，dim=None对所有元素求和
        max min mean 同理
        :param tensor: 输入的 PyTorch 张量
        """
        return tensor.sum(dim=dim)
    
    def pow(self, tensor: torch.Tensor, exponent: Number):
        """
        等价于 return tensor ** exponent 对里面的每个元素进行指数运算
        :param tensor: 输入的 PyTorch 张量
        :param exponent: 指数
        """
        return tensor.pow(exponent)
    
    def index_demo(self):
        """
        张量索引示例
        """
        tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
        self.print(tensor)

        first_row = tensor[0]
        print("First Row:")
        self.print(first_row)

        # 获取第二列
        second_column = tensor[:, 1]
        print("Second Column:")
        self.print(second_column)

        # 获取子张量（第一行和第二行，第二列和第三列）
        sub_tensor = tensor[0:2, 1:3]
        print("Sub Tensor (Rows 0-1, Columns 1-2):")
        self.print(sub_tensor)    

    def reshape(self, tensor: torch.Tensor, new_shape: List[int]):
        """
        将 PyTorch 张量重新塑形 不改变数据的情况下改变张量的形状
        :param tensor: 输入的 PyTorch 张量
        :param new_shape: 新的形状
        """
        return tensor.reshape(new_shape)

    def squeeze(self, tensor: torch.Tensor, dim: int = None):
        """
        移除张量中指定维度的大小为 1 的维度
        :param tensor: 输入的 PyTorch 张量
        :param dim: 指定要移除的维度，如果为 None，则移除所有大小为 1 的维度
        """
        if dim is None:
            return tensor.data.squeeze()
        else:
            return tensor.data.squeeze(dim)

    def unsqueeze(self, tensor: torch.Tensor, dim: int):
        """
        在张量中指定维度插入一个大小为 1 的维度
        :param tensor: 输入的 PyTorch 张量
        :param dim: 指定要插入的维度
        """
        return tensor.unsqueeze(dim=dim)
    
    def transpose(self, tensor: torch.Tensor, dim0: int, dim1: int):
        """
        交换张量的两个维度 变得不连续了
        :param tensor: 输入的 PyTorch 张量
        :param dim0: 要交换的第一个维度
        :param dim1: 要交换的第二个维度
        """
        return tensor.transpose(dim0, dim1)
    
    def permute(self, tensor: torch.Tensor, dims: List[int]):
        """
        根据指定的维度顺序重新排列张量的维度 变得不连续了
        :param tensor: 输入的 PyTorch 张量
        :param dims: 指定的维度顺序列表
        """
        return tensor.permute(*dims)
    
    def view(self, tensor: torch.Tensor, new_shape: List[int]):
        """
        返回一个新的张量，具有相同的数据但不同的形状 只能处理连续数据 若处理不连续可以先用contiguous转成连续的 然后在处理
        :param tensor: 输入的 PyTorch 张量
        :param new_shape: 新的形状
        """
        return tensor.view(new_shape)
    
    def contiguous(self, tensor: torch.Tensor):
        """
        返回一个连续的张量，如果原始张量已经是连续的，则返回原始张量本身
        :param tensor: 输入的 PyTorch 张量
        """
        return tensor.contiguous()
    
    def is_contiguous(self, tensor: torch.Tensor):
        """
        检查张量是否是连续的
        :param tensor: 输入的 PyTorch 张量
        """
        return tensor.is_contiguous()

    def cat(self, tensors: List[torch.Tensor], dim: int = 0):
        """
        将多个张量沿指定维度进行拼接  不改变维度数，拼接张量，除了拼接的那个维度外，其它维度数必须保持一致.
        :param tensors: 要拼接的张量列表
        :param dim: 指定拼接的维度
        """
        return torch.cat(tensors, dim=dim)

    def stack(self, tensors: List[torch.Tensor], dim: int = 0):
        """
        将多个张量沿指定维度进行堆叠  改变维度数，堆叠张量，所有张量的形状必须相同. 
        :param tensors: 要堆叠的张量列表
        :param dim: 指定堆叠的维度
        """
        return torch.stack(tensors, dim=dim)

    def demo_backward(self):
        """
        
        """
        w = torch.tensor(10, dtype=torch.float32, requires_grad=True)

        # 激活函数
        loss = w ** 2 + 10

        for i in range(0, 10):
            loss = w ** 2 + 10

            if w.grad is not None:
                w.grad.zero_()

            # 反向传播
            loss.sum().backward()


            # w新 = w旧 - 学习率 * 梯度（激活函数的导数）
            w.data = w.data - 0.01 * w.grad

            print(f'第 {i} 次, 权重初值:{w} (0.01 * w.grad):{0.01 *w.grad},loss:{loss}')

        print(f'最终结果权重:{w},梯度:{w.grad},Loss:{loss}')    

    def detach_demo(self):
        """
        当一个张量被设置成自动微分的情况下 直接转numpy是不可以的 需要通过detach函数来转
        """    

        t1 = torch.tensor([10, 20], requires_grad=True, dtype=torch.float)

        # t2 = t1.numpy() 这样写是报错的 因为没有detach
        t2 = t1.detach().numpy()

        self.print(t2)





def nn_demo():
    """
    神经网络线性回归搭建（面向新手详解版）
    
    本函数完整演示了用 PyTorch 搭建神经网络进行线性回归的全过程：
    1. 生成模拟数据（sklearn）
    2. 数据转换为 PyTorch 张量
    3. 构建神经网络模型
    4. 定义损失函数和优化器
    5. 训练模型
    6. 可视化结果
    """
    
    # ============================================
    # 第一步：导入需要的库
    # ============================================
    
    import torch                          # PyTorch 核心库，用于张量计算和神经网络
    from torch.utils.data import TensorDataset   # 将张量数据封装成数据集对象
    from torch.utils.data import DataLoader      # 数据加载器，用于批量读取数据
    from torch import nn                         # nn 模块：包含神经网络的各种层和损失函数
    from torch import optim                      # optim 模块：包含各种优化算法（如 SGD、Adam）
    from sklearn.datasets import make_regression # 用于生成线性回归的模拟数据集
    import matplotlib.pyplot as plt              # 用于绘制图表，可视化训练结果
    
    # 设置 Matplotlib 显示中文（防止中文乱码）
    plt.rcParams['font.sans-serif'] = ['SimHei']      # 使用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False        # 正常显示负号（如 -1.5）
    
    # ============================================
    # 第二步：生成模拟数据集
    # ============================================
    # make_regression 是 sklearn 提供的函数，专门生成线性回归用的模拟数据
    
    x, y, coef = make_regression(
        n_samples=100,        # 样本数量：生成 100 个数据点
        n_features=1,         # 特征数量：每个样本只有 1 个特征（简单的一元线性回归）
        noise=10,             # 噪声强度：给数据添加随机干扰，模拟真实数据的波动
                              #         噪声越大，数据点越分散；噪声越小，数据越接近直线
        coef=True,            # 是否返回真实的回归系数（真实的斜率）
                              #         设为 True，函数会返回真实的权重值，方便我们对比模型学得好不好
        bias=14.5,            # 偏置项：真实的截距，即 y = kx + b 中的 b
        random_state=3        # 随机种子：固定后，每次运行生成的数据都一样，方便复现结果
    )
    
    # 打印真实参数，方便后续对比模型学习效果
    print(f"真实的斜率（权重）: {coef:.4f}")
    print(f"真实的截距（偏置）: 14.5")
    
    # ============================================
    # 第三步：将 NumPy 数组转换为 PyTorch 张量（Tensor）
    # ============================================
    # PyTorch 的神经网络只能处理 Tensor 类型的数据，所以需要转换
    
    # torch.tensor()：将 NumPy 数组转换为 PyTorch 张量
    # dtype=torch.float32：指定数据类型为 32 位浮点数（神经网络的标准输入类型）
    
    x = torch.tensor(x, dtype=torch.float32)   # 特征数据 X，形状为 (100, 1)，100个样本，每个1个特征
    y = torch.tensor(y, dtype=torch.float32)   # 标签数据 y，形状为 (100,)，100个目标值
    
    # 调整 y 的形状，使其从 (100,) 变为 (100, 1)，与模型输出维度一致
    # view(-1, 1) 中：-1 表示自动计算该维度的大小，1 表示每个样本的目标值维度为1
    y = y.view(-1, 1)
    
    # ============================================
    # 第四步：封装数据集，创建数据加载器
    # ============================================
    # TensorDataset：将特征 x 和标签 y 打包成一对一对的数据集
    #               这样每次取数据时，能同时拿到对应的 x 和 y
    
    dataset = TensorDataset(x, y)
    
    # DataLoader：数据加载器，负责按"批次"（batch）读取数据
    # batch_size=16：每次训练取 16 个样本作为一个批次
    #                 小批量训练比一次性训练全部数据更稳定，收敛更快
    # shuffle=True：每个 epoch 开始前打乱数据顺序，防止模型记住数据顺序
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # ============================================
    # 第五步：定义神经网络模型
    # ============================================
    # 线性回归本质上就是一个"单层的全连接神经网络"
    # 输入 1 个特征 → 输出 1 个预测值
    # 公式：y_pred = weight * x + bias
    
    class LinearRegressionModel(nn.Module):
        """
        自定义线性回归模型
        继承 nn.Module：这是 PyTorch 中所有神经网络模块的基类
        """
        
        def __init__(self):
            # 调用父类的构造函数，这是必须写的
            super(LinearRegressionModel, self).__init__()
            
            # 定义一个全连接层（线性层）
            # nn.Linear(in_features, out_features)
            # in_features=1：输入特征数量（我们的 x 只有1个特征）
            # out_features=1：输出预测值的数量（预测1个 y 值）
            self.linear = nn.Linear(1, 1)
            
            # 这一层内部会自动创建：
            # - weight（权重）：模型要学习的斜率，初始为随机值
            # - bias（偏置）：模型要学习的截距，初始为随机值
        
        def forward(self, x):
            """
            前向传播：定义数据如何流过网络
            x 输入 → linear 层计算 → 输出预测值
            
            参数：
                x: 输入的特征数据
            返回：
                模型对 x 的预测结果
            """
            return self.linear(x)
    
    # 创建模型实例
    model = LinearRegressionModel()
    
    # 查看模型初始参数（训练前的随机值）
    print("\n=== 训练前 ===")
    for name, param in model.named_parameters():
        print(f"{name}: {param.data.numpy()}")
        # name='weight' 是权重（斜率），name='bias' 是偏置（截距）
    
    # ============================================
    # 第六步：定义损失函数和优化器
    # ============================================
    
    # 损失函数：衡量模型预测值与真实值之间的差距
    # nn.MSELoss()：均方误差损失（Mean Squared Error）
    #               公式：loss = 平均( (预测值 - 真实值)² )
    #               值越小，说明预测越准确
    criterion = nn.MSELoss()
    
    # 优化器：负责更新模型的参数（权重和偏置），让损失越来越小
    # optim.SGD：随机梯度下降优化器
    #           lr=0.01：学习率（learning rate），控制每次参数更新的步长
    #                   学习率太大可能震荡不收敛，太小收敛太慢
    # model.parameters()：告诉优化器要优化哪些参数（即模型的 weight 和 bias）
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # ============================================
    # 第七步：训练模型
    # ============================================
    
    num_epochs = 100          # 训练轮数：把整个数据集完整过 100 遍
    loss_history = []         # 用于记录每轮训练的平均损失，方便后续画图观察收敛情况
    
    print("\n=== 开始训练 ===")
    
    for epoch in range(num_epochs):
        """
        外层循环：遍历所有 epoch（轮次）
        每一轮中，模型会把所有数据都学习一遍
        """
        
        epoch_loss = 0.0      # 记录本轮的总损失
        num_batches = 0       # 记录本轮的批次数量
        
        for batch_x, batch_y in dataloader:
            """
            内层循环：遍历每个 batch（批次）
            batch_x：当前批次的特征数据，形状 (16, 1)
            batch_y：当前批次的标签数据，形状 (16, 1)
            """
            
            # ---------- 1. 前向传播 ----------
            # 将当前批次的数据输入模型，得到预测值
            predictions = model(batch_x)
            
            # ---------- 2. 计算损失 ----------
            # 用损失函数比较预测值和真实值的差距
            loss = criterion(predictions, batch_y)
            
            # ---------- 3. 反向传播 ----------
            # 清空之前累积的梯度（ PyTorch 默认会累加梯度，所以每次要清零）
            optimizer.zero_grad()
            
            # 自动计算损失函数对各个参数的梯度（即 weight 和 bias 的梯度）
            # 梯度告诉我们：参数应该往哪个方向调整，才能减小损失
            loss.backward()
            
            # ---------- 4. 更新参数 ----------
            # 优化器根据梯度，按照学习率更新 weight 和 bias
            optimizer.step()
            
            # 累加损失，用于计算本轮平均损失
            epoch_loss += loss.item()   # .item() 将张量转换为 Python 数字
            num_batches += 1
        
        # 计算本轮的平均损失
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        # 每 10 轮打印一次训练进度
        if (epoch + 1) % 10 == 0:
            print(f"第 {epoch + 1:3d} 轮，平均损失: {avg_loss:.4f}")
    
    # ============================================
    # 第八步：查看训练后的模型参数
    # ============================================
    
    print("\n=== 训练后 ===")
    for name, param in model.named_parameters():
        print(f"{name}: {param.data.numpy()}")
    
    print(f"\n真实的斜率: {coef:.4f}")
    print(f"真实的截距: 14.5")
    print("可以看到，模型学到的参数非常接近真实值！")
    
    # ============================================
    # 第九步：可视化结果
    # ============================================
    
    # 创建一个大图，包含两个子图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ---------- 子图1：损失下降曲线 ----------
    ax1 = axes[0]
    ax1.plot(range(1, num_epochs + 1), loss_history, 'b-', linewidth=2)
    ax1.set_xlabel('训练轮次 (Epoch)', fontsize=12)
    ax1.set_ylabel('均方误差损失 (MSE)', fontsize=12)
    ax1.set_title('训练过程中损失的变化', fontsize=14)
    ax1.grid(True, alpha=0.3)  # 添加网格线，alpha 控制透明度
    
    # ---------- 子图2：数据点与拟合直线 ----------
    ax2 = axes[1]
    
    # 绘制原始数据点（散点图）
    ax2.scatter(x.numpy(), y.numpy(), color='blue', alpha=0.5, label='真实数据点')
    
    # 生成用于画拟合直线的连续 x 值
    x_line = torch.linspace(x.min(), x.max(), 100).view(-1, 1)
    
    # 用训练好的模型预测这些 x 对应的 y 值
    with torch.no_grad():   # 预测时不需要计算梯度，节省内存
        y_pred_line = model(x_line)
    
    # 绘制模型拟合的直线
    ax2.plot(x_line.numpy(), y_pred_line.numpy(), 'r-', linewidth=2, label='模型拟合的直线')
    
    ax2.set_xlabel('特征 X', fontsize=12)
    ax2.set_ylabel('标签 y', fontsize=12)
    ax2.set_title('线性回归拟合效果', fontsize=14)
    ax2.legend(fontsize=11)   # 显示图例
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()        # 自动调整子图间距
    plt.show()
    
    # ============================================
    # 第十步：使用模型进行预测（可选）
    # ============================================
    
    # 假设有一个新数据点 x=2.0，预测对应的 y 值
    test_x = torch.tensor([[2.0]], dtype=torch.float32)
    with torch.no_grad():
        predicted_y = model(test_x)
    print(f"\n当 x = 2.0 时，模型预测 y = {predicted_y.item():.4f}")
    
    # 用真实参数验证：y = coef * x + bias
    true_y = coef * 2.0 + 14.5
    print(f"用真实参数计算 y = {true_y:.4f}")

