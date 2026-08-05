import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from torchsummary import summary


def create_dataset():
    data = pd.read_csv('./pytorch/data.csv')
    x, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

    x = x.astype(np.float32)
    y = y.astype(np.float32).reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y)) # 返回训练集、测试集、输入特征维度、类别数


class NNModel(nn.Module):
    """简单的全连接神经网络模型"""
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.output = nn.Linear(256, output_size)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.softmax(self.output(x), dim=1)
        return x

def train(train_dataset, input_dim, output_dim):
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    model = NNModel(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs =  50
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.long().squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), './pytorch/model.pth')    


def evaluate(test_dataset, input_dim, output_dim):
    model = NNModel(input_size=input_dim, output_size=output_dim)

    model.load_state_dict(torch.load("./pytorch/model.pth"))

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    correct = 0

    for x, y in test_loader:
        model.eval()
        y_pred = model(x)
        print(f"预测值 {y_pred}")

        predicted_classes = torch.argmax(y_pred, dim=1)
        # 结果: tensor([3, 3, 3, 1, 0, 1])
        
        # 如果是分类问题，可以与真实标签比较计算准确率
        accuracy = (predicted_classes == y).float().mean()

        print(f"准确率 {accuracy}")

if __name__ == "__main__":
    train_dataset, test_dataset, input_size, num_classes = create_dataset()
    print(f"训练集样本数: {len(train_dataset)}, 测试集样本数: {len(test_dataset)}, 输入特征维度: {input_size}, 类别数: {num_classes}")
    model = NNModel(input_size, num_classes)
    # summary(model, (input_size,))
    # 一批训练16个条数据
    # summary(model, input_size=(16, input_size))

    # train(train_dataset, input_size, num_classes)

    evaluate(test_dataset, input_size, num_classes)