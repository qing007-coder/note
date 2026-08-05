import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class LearningRateSchedulerDemo:
    """学习率衰减策略使用示例 - PyTorch实战"""
    
    def __init__(self):
        # 创建简单的线性模型
        self.model = nn.Linear(10, 1)
        
        # 创建模拟数据
        x = torch.randn(100, 10)
        y = torch.randn(100, 1)
        self.train_loader = DataLoader(TensorDataset(x, y), batch_size=16)
    
    def step_lr_demo(self):
        """StepLR - 等间隔衰减"""
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        criterion = nn.MSELoss()
        
        print("StepLR: 每30轮学习率乘以0.1")
        for epoch in range(100):
            # 训练
            for batch_x, batch_y in self.train_loader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
            
            scheduler.step()  # 更新学习率
            
            # 打印学习率
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}, LR: {current_lr:.6f}")
    
    def multi_step_lr_demo(self):
        """MultiStepLR - 指定间隔衰减"""
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 80], gamma=0.1)
        criterion = nn.MSELoss()
        
        print("MultiStepLR: 在第20、50、80轮衰减")
        for epoch in range(100):
            # 训练
            for batch_x, batch_y in self.train_loader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}, LR: {current_lr:.6f}")
    
    def exponential_lr_demo(self):
        """ExponentialLR - 指数衰减"""
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        criterion = nn.MSELoss()
        
        print("ExponentialLR: 每轮乘以0.95")
        for epoch in range(100):
            # 训练
            for batch_x, batch_y in self.train_loader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}, LR: {current_lr:.6f}")
    
    def cosine_annealing_lr_demo(self):
        """CosineAnnealingLR - 余弦退火"""
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
        criterion = nn.MSELoss()
        
        print("CosineAnnealingLR: 50轮完成一个余弦周期")
        for epoch in range(100):
            # 训练
            for batch_x, batch_y in self.train_loader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}, LR: {current_lr:.6f}")
    
    def reduce_lr_on_plateau_demo(self):
        """ReduceLROnPlateau - 自适应衰减"""
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',      # 监控loss下降
            patience=10,     # 10轮无改善触发衰减
            factor=0.1,      # 衰减因子
            verbose=True     # 打印提示
        )
        criterion = nn.MSELoss()
        
        print("ReduceLROnPlateau: 验证集loss停滞10轮自动衰减")
        for epoch in range(100):
            # 训练
            train_loss = 0
            for batch_x, batch_y in self.train_loader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 模拟验证集loss（实际训练中通过验证集计算）
            val_loss = train_loss / len(self.train_loader)
            
            # 更新学习率（传入验证集loss）
            scheduler.step(val_loss)
            
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
    
    def lambda_lr_demo(self):
        """LambdaLR - 自定义衰减函数"""
        # 定义自定义衰减函数：学习率 = 初始lr * (1 - epoch/总轮数)
        lambda_func = lambda epoch: 1 - epoch / 100
        
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)
        criterion = nn.MSELoss()
        
        print("LambdaLR: 自定义线性衰减")
        for epoch in range(100):
            # 训练
            for batch_x, batch_y in self.train_loader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}, LR: {current_lr:.6f}")
    
    def multi_lr_demo(self):
        """为不同参数组设置不同学习率"""
        # 为不同层设置不同学习率
        params = [
            {'params': self.model.weight, 'lr': 0.1},      # 权重层学习率0.1
            {'params': self.model.bias, 'lr': 0.01}        # 偏置层学习率0.01
        ]
        
        optimizer = optim.SGD(params, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        print("MultiLR: 不同参数组不同学习率")
        for epoch in range(100):
            # 训练
            for batch_x, batch_y in self.train_loader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = nn.MSELoss()(pred, batch_y)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            # 分别获取各组学习率
            lrs = [param_group['lr'] for param_group in optimizer.param_groups]
            print(f"Epoch {epoch:3d}, Weight LR: {lrs[0]:.6f}, Bias LR: {lrs[1]:.6f}")


# 快速使用模板
def quick_use():
    """最常用的训练模板"""
    # 1. 创建模型、优化器、损失函数
    model = nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    # 2. 选择衰减策略（最常用的是StepLR和ReduceLROnPlateau）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # 或
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1)
    
    criterion = nn.MSELoss()
    
    # 3. 创建数据加载器
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=16)
    
    # 4. 训练循环
    for epoch in range(100):
        # 训练
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
        
        # 更新学习率
        scheduler.step()  # StepLR/MultiStepLR/ExponentialLR等
        # 或
        # scheduler.step(val_loss)  # ReduceLROnPlateau需要传入验证loss
        
        # 打印学习率
        print(f"Epoch {epoch}, LR: {scheduler.get_last_lr()[0]:.6f}")


if __name__ == "__main__":
    # 运行所有demo
    demo = LearningRateSchedulerDemo()
    
    print("="*60)
    print("学习率衰减策略实战演示")
    print("="*60)
    
    # 取消注释想要运行的demo
    # demo.step_lr_demo()
    # demo.multi_step_lr_demo()
    # demo.exponential_lr_demo()
    # demo.cosine_annealing_lr_demo()
    # demo.reduce_lr_on_plateau_demo()
    # demo.lambda_lr_demo()
    # demo.multi_lr_demo()
    
    quick_use()  # 最常用模板