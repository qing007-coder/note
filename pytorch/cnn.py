import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 


class CNNModel(nn.Module):
    """
    简单的卷积神经网络模型
    输出尺寸 = (输入尺寸 - 卷积核尺寸 + 2 * padding) / stride + 1
    """
    def __init__(self, in_channels, output_dims):
        super(CNNModel, self).__init__()

        # 第一层卷积层：输入3通道，输出32通道，卷积核5x5
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=1, padding=1)
        # 第二层卷积层：输入32通道，输出64通道，卷积核5x5
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=1)
        # 池化层 2x2  步长 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层 将卷积层输出的特征图展平后输入到全连接层
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, output_dims)

    def forward(self, x):
        # 定义前向传播逻辑
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 3个通道！
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    sample_image, _ = train_dataset[0]
    in_channels = sample_image.shape[0]  # 第一个维度就是通道数
    output_dims = len(train_dataset.classes)  # 类别数

    return train_dataset, test_dataset, in_channels, output_dims


def train(train_dataset, inchannels, output_dims):
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    model = CNNModel(in_channels=inchannels, output_dims=output_dims)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 50
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), './pytorch/cnn_model.pth')    

def evaluate(test_dataset, inchannels, output_dims):
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    model = CNNModel(in_channels=inchannels, output_dims=output_dims)
    model.load_state_dict(torch.load("./pytorch/cnn_model.pth"))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    train_dataset, test_dataset, in_channels, output_dims = load_dataset()

    # train(train_dataset, in_channels, output_dims)

    evaluate(test_dataset, in_channels, output_dims)