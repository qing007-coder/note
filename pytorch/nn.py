import torch.nn as nn
from torchsummary import summary
import torch


class PytorchInit:
    def __init__(self):
        self.linear = nn.Linear(5,3)  # Example linear layer

    def init_weights(self, method='xavier_uniform'):
        """
        kaiming适合ReLU激活函数，xavier适合tanh，sigmod激活函数
        """
        if method == 'xavier_uniform':
            return nn.init.xavier_uniform_(self.linear.weight)
        elif method == 'xavier_normal':
            return nn.init.xavier_normal_(self.linear.weight)
        elif method == 'kaiming_uniform':
            return nn.init.kaiming_uniform_(self.linear.weight)
        elif method == 'kaiming_normal':
            return nn.init.kaiming_normal_(self.linear.weight)
        else:
            raise ValueError(f"Unknown initialization method: {method}")

    def init_bias(self, method='zeros'):
        if method == 'zeros':
            return nn.init.zeros_(self.linear.bias)
        elif method == 'ones':
            return nn.init.ones_(self.linear.bias)
        else:
            raise ValueError(f"Unknown bias initialization method: {method}")

    def test(self):
        print("Before initialization:")
        print("Weights:", self.linear.weight)
        print("Bias:", self.linear.bias)

        self.init_weights(method='xavier_uniform')
        self.init_bias(method='zeros')

        print("\nAfter initialization:")
        print("Weights:", self.linear.weight)
        print("Bias:", self.linear.bias)


class ModelDemo(nn.Module):
    def __init__(self):
        super().__init__()

        # 隐藏层1
        self.linear1 = nn.Linear(3, 3)

        # 隐藏层1
        self.linear2 = nn.Linear(3, 2)

        # 输出层
        self.output = nn.Linear(2, 2)

        # 初始化权重
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.kaiming_uniform_(self.linear2.weight)
        # nn.init.kaiming_uniform_(self.output.weight)

        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
        # nn.init.zeros_(self.output.bias)


    def forward(self, x):

        # 隐藏层1
        x = self.linear1(x)
        x = torch.sigmoid(x)

        # 隐藏层2
        x = self.linear2(x)
        x = torch.relu(x)

        # 输出层 dim=-1 按行计算
        x = self.output(x)
        x = torch.softmax(x, dim=-1)

        return x


def train():
    model = ModelDemo()

    print(f"Model structure:{model}")
    print(30 * "-")

    data = torch.randn(size=(5, 3))
    print(f"Input data:{data}")
    print(30 * "-")

    output = model(data)
    print(f"Output data:{output}")

    print("Model summary:")
    summary(model, input_size=(5, 3))  # 输入维度为3
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, shape: {param.shape}")        