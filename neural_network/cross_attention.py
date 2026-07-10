import numpy as np


class CrossAttentionLayer:
    """
    交叉注意力层 (Cross-Attention Layer)
    让 图像(Query) 去关注 文本(Key/Value)
    """

    def __init__(self, img_dim: int, text_dim: int, embed_dim: int):
        """
        :param img_dim: 图像特征维度
        :param text_dim: 文本特征维度
        :param embed_dim: 映射后的注意力维度
        """
        # 初始化三个投影矩阵：W_q, W_k, W_v
        np.random.seed(42)
        self.W_q = np.random.randn(img_dim, embed_dim) * 0.1
        self.W_k = np.random.randn(text_dim, embed_dim) * 0.1
        self.W_v = np.random.randn(text_dim, embed_dim) * 0.1

        self.scale = np.sqrt(embed_dim)  # 缩放因子，防止数值过大

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, X_img: np.ndarray, X_text: np.ndarray) -> np.ndarray:
        # 记住输入，反向传播要用
        self.X_img = X_img
        self.X_text = X_text

        # 1. 矩阵乘法一：把图像和文本投影到同一个空间，生成 Q, K, V
        self.Q = np.dot(X_img, self.W_q)  # (样本数, embed_dim)
        self.K = np.dot(X_text, self.W_k)  # (样本数, embed_dim)
        self.V = np.dot(X_text, self.W_v)  # (样本数, embed_dim)

        # 2. 矩阵乘法二：Q 和 K 的转置相乘，算出图像和文本的相关度评分
        # Scores 形状: (样本数, 样本数)，代表图文匹配度
        self.Scores = np.dot(self.Q, self.K.T) / self.scale

        # 3. 过 Softmax 变成概率分布（注意力权重）
        self.A = self._softmax(self.Scores)

        # 4. 矩阵乘法三：把权重乘以 Value，得到融合后的多模态特征
        # Out 形状: (样本数, embed_dim)
        self.Out = np.dot(self.A, self.V)
        return self.Out

    def backward(self, dLoss_dOut: np.ndarray, learning_rate: float):
        """
        反向传播：顺着三条矩阵乘法路径，同时把误差传回给 W_q, W_k, W_v
        """
        m = self.X_img.shape[0]

        # 1. 对 矩阵乘法三 求导
        dLoss_dA = np.dot(dLoss_dOut, self.V.T)
        dLoss_dV = np.dot(self.A.T, dLoss_dOut)

        # 2. 对 Softmax 求导
        dLoss_dScores = self.A * (dLoss_dA - np.sum(dLoss_dA * self.A, axis=-1, keepdims=True)) / self.scale

        # 3. 对 矩阵乘法二 求导
        dLoss_dQ = np.dot(dLoss_dScores, self.K)
        dLoss_dK = np.dot(dLoss_dScores.T, self.Q)

        # 4. 对 矩阵乘法一（参数 W）求导，算出参数各自要负的责
        dW_q = np.dot(self.X_img.T, dLoss_dQ) / m
        dW_k = np.dot(self.X_text.T, dLoss_dK) / m
        dW_v = np.dot(self.X_text.T, dLoss_dV) / m

        # 5. 梯度下降更新交叉注意力的参数
        self.W_q -= learning_rate * dW_q
        self.W_k -= learning_rate * dW_k
        self.W_v -= learning_rate * dW_v

        # 把追责接力棒传回给底层的图像和文本网络（如果前段还有网络的话）
        dLoss_dXimg = np.dot(dLoss_dQ, self.W_q.T)
        return dLoss_dXimg


class DenseLayer:
    """这是你之前写好的全连接层"""

    def __init__(self, weights: np.ndarray, biases: np.ndarray):
        self.weights = weights
        self.biases = biases

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        self.Z = np.dot(X, self.weights) + self.biases
        self.A = 1 / (1 + np.exp(-self.Z))
        return self.A

    def backward(self, dLoss_dA: np.ndarray, learning_rate: float) -> np.ndarray:
        dLoss_dZ = dLoss_dA * (self.A * (1.0 - self.A))
        m = self.X.shape[0]
        dW = np.dot(self.X.T, dLoss_dZ) / m
        db = np.sum(dLoss_dZ, axis=0, keepdims=True) / m
        dLoss_dX = np.dot(dLoss_dZ, self.weights.T)

        self.weights -= learning_rate * dW
        self.biases -= learning_rate * db
        return dLoss_dX


# =====================================================================
# 组装网络并运行
# =====================================================================
if __name__ == "__main__":
    # 模拟具身智能输入 (4个样本)
    # 图像特征 (4, 4)：比如前、后、左、右物体的距离
    X_img = np.array([
        [0.1, 0.9, 0.5, 0.5],  # 样本1
        [0.8, 0.2, 0.5, 0.5],  # 样本2
        [0.2, 0.8, 0.5, 0.5],  # 样本3
        [0.9, 0.1, 0.5, 0.5]   # 样本4
    ])
    # 文本指令特征 (4, 3)：编码后的人类口令特征
    X_text = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    # 真实决策标签：1代表右转，0代表左转
    y_true = np.array([[1.0], [0.0], [1.0], [0.0]])

    # 实例化我们的组件
    # 1. 交叉注意力层：输入图像维度4，文本维度3，融合输出维度4
    attention_layer = CrossAttentionLayer(img_dim=4, text_dim=3, embed_dim=4)
    # 2. 全连接决策层：接收注意力融合后的4维特征，输出1维动作概率
    w_dense = np.random.randn(4, 1) * 0.1
    b_dense = np.zeros((1, 1))
    dense_layer = DenseLayer(w_dense, b_dense)

    learning_rate = 0.5

    print("--- 开始带『交叉注意力』的具身智能训练 ---")
    for epoch in range(1001):
        # ======= 前向传播 (Forward) =======
        # 1. 先过交叉注意力层，把图像和文本融合在一起
        attn_out = attention_layer.forward(X_img, X_text)
        # 2. 再把融合特征喂给决策层
        y_pred = dense_layer.forward(attn_out)

        # 计算 Loss
        loss = 0.5 * np.mean((y_pred - y_true) ** 2)

        # ======= 反向传播 (Backward) =======
        # 1. 算出最后一层的初始委屈度
        dLoss_dyPred = y_pred - y_true
        # 2. 倒着传回全连接决策层，拿到给前一层的梯度 dLoss_dAttnOut
        dLoss_dAttnOut = dense_layer.backward(dLoss_dyPred, learning_rate)
        # 3. 倒着传回交叉注意力层，更新 W_q, W_k, W_v
        attention_layer.backward(dLoss_dAttnOut, learning_rate)

        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | 当前均方误差 Loss: {loss:.6f}")

    print("\n--- 训练完成！ ---")