import numpy as np


class DenseLayer:
    """
    全连接层（支持反向传播）
    """

    def __init__(self, weights: np.ndarray, biases: np.ndarray):
        self.weights = weights
        self.biases = biases
        # 用于保存前向传播时的状态，反向传播时要用
        self.X = None
        self.Z = None
        self.A = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, a):
        # Sigmoid的导数公式：f'(z) = a * (1 - a)，这里的 a 是已经过激活后的输出
        return a * (1 - a)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X  # 📌 记住输入，反向传播算 dW 时要用
        self.Z = np.dot(X, self.weights) + self.biases
        self.A = self._sigmoid(self.Z)
        return self.A

    def backward(self, dLoss_dA: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        反向传播计算梯度并更新参数
        :param dLoss_dA: 后面那一层传回来的“对当前层输出A”的敏感度 (样本数, 当前层神经元数)
        :return: 传递给前一层的误差梯度 (样本数, 输入特征数)
        """
        # 1. 连乘激活层的导数：得到对 Z 的敏感度
        dLoss_dZ = dLoss_dA * self._sigmoid_derivative(self.A)

        # 2. 计算当前层 W 和 b 的责任（梯度）
        # dW = X^T · dLoss_dZ (矩阵对齐，同时平均样本数)
        m = self.X.shape[0]
        dW = np.dot(self.X.T, dLoss_dZ) / m
        db = np.sum(dLoss_dZ, axis=0, keepdims=True) / m

        # 3. 📌 关键：计算要传给前一层的误差（链式法则连乘当前层的权重）
        # 前一层要为这一层的 Z 负责，所以乘上 weights
        dLoss_dX = np.dot(dLoss_dZ, self.weights.T)

        # 4. 梯度下降：更新当前层的参数
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * db

        # 把追责接力棒传给前一层
        return dLoss_dX


class MultiLayerNetwork:
    """
    完整的神经网络（带训练功能）
    """

    def __init__(self):
        # 隐藏层 (2输入, 3输出)
        w1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        b1 = np.array([[0.1, -0.2, 0.0]])
        self.layer1 = DenseLayer(w1, b1)

        # 输出层 (3输入, 1输出)
        w2 = np.array([[0.2], [0.1], [0.7]])
        b2 = np.array([[0.5]])
        self.layer2 = DenseLayer(w2, b2)

    def forward(self, X: np.ndarray) -> np.ndarray:
        A1 = self.layer1.forward(X)
        A2 = self.layer2.forward(A1)
        return A2

    def backward(self, dLoss_dA2: np.ndarray, learning_rate: float):
        """
        反向传播：必须从最后一层往第一层倒着传
        """
        # 先传第二层（输出层），拿到传给第一层的梯度 dLoss_dA1
        dLoss_dA1 = self.layer2.backward(dLoss_dA2, learning_rate)

        # 再传第一层（隐藏层）
        self.layer1.backward(dLoss_dA1, learning_rate)


# =====================================================================
# 运行验证（让网络学会做出正确决策）
# =====================================================================
if __name__ == "__main__":
    # 假设我们有一个训练集 (小车传感器：[左距离, 右距离])
    X_train = np.array([[1.0, 2.0], [2.0, 1.0], [0.5, 1.5], [1.5, 0.5]])

    # 对应的正确录用标签：1代表右转，0代表左转
    y_true = np.array([[1.0], [0.0], [1.0], [0.0]])

    net = MultiLayerNetwork()
    learning_rate = 1.0

    print("--- 开始循环迭代训练 ---")
    for epoch in range(2001):
        # 1. 前向传播
        y_pred = net.forward(X_train)

        # 2. 计算 Loss (减法)
        loss = 0.5 * np.mean((y_pred - y_true) ** 2)

        # 📌 检查误差阈值：如果误差已经非常微小，主动退出循环
        if loss < 0.001:
            print(f"检测到 Loss 低于阈值 0.001，在第 {epoch} 轮提前终止训练！")
            break

        # 3. 计算最后一层的初始追责 (预测值 - 真实值)
        # 本质上是 Loss 对 y_pred 的求导
        dLoss_dyPred = y_pred - y_true

        # 4. 触发反向传播 (连乘开始)
        net.backward(dLoss_dyPred, learning_rate)

        if epoch % 500 == 0:
            print(f"轮次 {epoch:4d} | 当前均方误差 Loss: {loss:.6f}")

    print("\n--- 训练结束，验证最终决策能力 ---")
    # 测试一个新输入
    test_X = np.array([[1.0, 2.0]])
    final_res = net.forward(test_X)
    print(f"输入 [1.0, 2.0]（左近右远），输出右转概率: {final_res[0][0]*100:.2f}%")