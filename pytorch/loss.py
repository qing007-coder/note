import torch 
import torch.nn as nn


class LossFunction:
    def __init__(self):
        pass

    def cross_entropy_loss_demo(self):
        """
        计算交叉熵损失 用于softmax 多分类交叉熵
        """

        y_true = torch.tensor([1, 0])
        y_pred = torch.tensor([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]], requires_grad=True, dtype=torch.float)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_pred, y_true)

        print(f"Loss: {loss}")

        return

    def BCE_loss_demo(self):
        """
        计算二分类交叉熵损失 用于sigmoid 二分类交叉熵
        """

        y_true = torch.tensor([1, 0], dtype=torch.float)
        y_pred = torch.tensor([0.9, 0.2], requires_grad=True, dtype=torch.float)

        criterion = nn.BCELoss()
        loss = criterion(y_pred, y_true)

        print(f"Loss: {loss}")

        return

    def MSE_loss_demo(self):
        """
        计算均方误差损失（Mean Squared Error Loss）

        MSE = mean((y_pred - y_true)^2)

        特点：
        - 处处可导，梯度连续
        - 误差较大时梯度也大，收敛速度快
        - 缺点：对异常值敏感（误差平方会放大离群点的影响）
        - 常用于回归任务
        """

        y_true = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float)
        y_pred = torch.tensor([1.5, 2.5, 3.5], requires_grad=True, dtype=torch.float)

        criterion = nn.MSELoss()
        loss = criterion(y_pred, y_true)

        print(f"Loss: {loss}")

        return


    def MAE_loss_demo(self):
        """
        计算平均绝对误差损失（Mean Absolute Error Loss）
        
        MAE = mean(|y_pred - y_true|)
        
        特点：
        - 对异常值不敏感，鲁棒性比MSE好
        - 缺点：在零点处不可导，梯度不连续
        - 梯度恒定（为±1），误差较小时收敛不如MSE平滑
        - 常用于回归任务，尤其是数据中存在较多离群点时
        """
    
        y_true = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float)
        y_pred = torch.tensor([1.5, 2.5, 3.5], requires_grad=True, dtype=torch.float)
    
        criterion = nn.L1Loss()
        loss = criterion(y_pred, y_true)
    
        print(f"Loss: {loss}")
    
        return


    def smooth_L1_loss_demo(self):
        """
        计算平滑L1损失（Smooth L1 Loss）

        分段函数：
        - 当 |y_pred - y_true| < 1 时：0.5 * (y_pred - y_true)^2
        - 当 |y_pred - y_true| >= 1 时：|y_pred - y_true| - 0.5

        特点：
        - 结合了MSE和MAE的优点
        - 误差较小时（<1），采用MSE形式，梯度平滑、收敛稳定
        - 误差较大时（>=1），采用MAE形式，梯度恒为±1，防止梯度爆炸
        - 有效缓解了MSE在大误差时梯度爆炸的问题
        - 同时解决了MAE在零点不可导的问题
        - 对异常值比MSE更鲁棒
        - 在目标检测（如Faster R-CNN）和回归任务中广泛使用
        """

        y_true = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float)
        y_pred = torch.tensor([1.5, 2.5, 3.5], requires_grad=True, dtype=torch.float)

        criterion = nn.SmoothL1Loss()
        loss = criterion(y_pred, y_true)

        print(f"Loss: {loss}")

        return