import torch
import torch.nn as nn


class RegularizationDemo:
    def __init__(self):
        pass


    def dropout_demo(self):
        """
        对激活值进行随机失活dropout处理  只有训练阶段有，测试阶段没有 
        """
        t1 = torch.randint(0, 10, (1, 5), dtype=torch.float32)
        print("原始张量:\n", t1)

        linear1 = nn.Linear(5, 4)
        nn.init.xavier_uniform_(linear1.weight)
        nn.init.zeros_(linear1.bias)

        l1 = linear1(t1)
        print("线性层输出:\n", l1)

        output = torch.relu(l1)
        print("Dropout前张量:\n", output)

        dropout = nn.Dropout(p=0.5) # 每个神经元都有50%的概率被 kill
        output_dropout = dropout(output)
        print("Dropout后张量:\n", output_dropout) # 未被失活的进行缩放，缩放比例为:1/(1-p)=2

    def bn_demo(self):
        """
        批归一化（Batch Normalization）- 图片数据演示
        """
        # 1. 创建模拟图片数据 [batch=4, channel=3, height=32, width=32]
        t1 = torch.randint(0, 10, (4, 3, 32, 32), dtype=torch.float32)
        print("原始张量形状:", t1.shape)
        print("原始张量(第1张图的R通道前5x5):\n", t1[0, 0, :5, :5])
        print("-" * 50)
        
        # 2. 卷积层 (输入3通道 → 输出4通道)
        conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(conv1.weight)
        nn.init.zeros_(conv1.bias)
        
        l1 = conv1(t1)
        print("卷积层输出形状:", l1.shape)  # [4, 4, 32, 32]
        print("卷积层输出(第1张图第1通道前5x5):\n", l1[0, 0, :5, :5])
        print("-" * 50)
        
        # 3. ReLU激活
        output = torch.relu(l1)
        print("ReLU后形状:", output.shape)
        print("ReLU后(第1张图第1通道前5x5):\n", output[0, 0, :5, :5])
        print("-" * 50)
        
        # 4. 创建批量归一化层(BatchNorm2d)
        # 参数1: num_features=4 (卷积层输出的通道数)
        # 参数2: eps=1e-5 (防止除零的小常数)
        # 参数3: momentum=0.1 (动量值，用于更新全局统计量)
        # 参数4: affine=True (使用可学习的缩放γ和平移β)
        bn2d = nn.BatchNorm2d(num_features=4, eps=1e-5, momentum=0.1, affine=True)
        
        # 5. 应用BN
        output_bn = bn2d(output)
        print("BN后形状:", output_bn.shape)
        print("BN后(第1张图第1通道前5x5):\n", output_bn[0, 0, :5, :5])
        



if __name__ == "__main__":
    demo = RegularizationDemo()
    demo.bn_demo()