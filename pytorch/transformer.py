import numpy as np
import torch

from pytorch.nn import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms


class SingleHeadAttention:
    """
    单头 Self-Attention

    输入:
        X:
        shape = (seq_len, d_model)

    输出:
        output:
        shape = (seq_len, d_k)

    流程:

        X
        |
        |--- Linear(Wq) ---> Q
        |
        |--- Linear(Wk) ---> K
        |
        |--- Linear(Wv) ---> V
        |
        QK^T计算相关性
        |
        Softmax得到权重
        |
        权重乘V
        |
        输出新的特征
    """

    def __init__(self, d_model, d_k):

        # 输入特征维度
        self.d_model = d_model

        # Q/K/V的维度
        self.d_k = d_k


        # 三个可学习参数
        # 训练过程中会不断更新

        # Query投影矩阵
        self.W_q = np.random.randn(
            d_model,
            d_k
        )

        # Key投影矩阵
        self.W_k = np.random.randn(
            d_model,
            d_k
        )

        # Value投影矩阵
        self.W_v = np.random.randn(
            d_model,
            d_k
        )


    def forward(self, X):

        """
        X:
        输入token特征

        shape:
        (seq_len, d_model)
        """


        # ==========================
        # 1. 生成Q K V
        # ==========================

        # Query:
        # 表示当前token想寻找什么信息

        Q = np.dot(
            X,
            self.W_q
        )


        # Key:
        # 表示当前token有什么特征

        K = np.dot(
            X,
            self.W_k
        )


        # Value:
        # 表示当前token携带的实际信息

        V = np.dot(
            X,
            self.W_v
        )


        # ==========================
        # 2. 计算注意力分数
        # ==========================

        # Q和所有K计算相似程度
        #
        # Q:
        # (seq_len,d_k)
        #
        # K.T:
        # (d_k,seq_len)
        #
        # 结果:
        # (seq_len,seq_len)

        scores = np.dot(
            Q,
            K.T
        )


        # 缩放
        # 防止点积结果过大

        scores = scores / np.sqrt(self.d_k)



        # ==========================
        # 3. Softmax得到注意力权重
        # ==========================

        attention_weights = self.softmax(scores)



        # ==========================
        # 4. 加权求和
        # ==========================

        # attention_weights:
        #
        # 哪些token重要
        #
        # V:
        #
        # token的信息


        output = np.dot(
            attention_weights,
            V
        )


        return output



    def softmax(self,x):

        """
        将分数转换成概率
        """

        exp_x = np.exp(
            x - np.max(
                x,
                axis=-1,
                keepdims=True
            )
        )


        return exp_x / np.sum(
            exp_x,
            axis=-1,
            keepdims=True
        )



# if __name__ == "__main__":
#     # 测试
#     X = np.array([
#         [1, 0, 1],
#         [0, 2, 0],
#         [1, 1, 1]
#     ])

#     attention = SingleHeadAttention(
#         d_model=3,
#         d_k=2
#     )

#     output = attention.forward(X)

#     print("输出特征:")
#     print(output)



class PatchEmbed(nn.Module):
    """
    将图片切分成patches，并将每个patch展平后映射到指定维度
    """

    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [batch_size, in_chans, img_size, img_size] [4, 3, 32, 32]
        x = self.proj(x)  # [batch_size, embed_dim, num_patches**0.5, num_patches**0.5] [4, 128, 8, 8]
        x = x.flatten(2)  # [batch_size, embed_dim, num_patches] [4, 128, 64]
        x = x.transpose(1, 2)  # [batch_size, num_patches, embed_dim] [4, 64, 128]
        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力 —— 让每个Patch都能看到所有其他Patch"""
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 128/4=32
        self.scale = self.head_dim ** -0.5
        
        # ★ 这4个全连接层，就是你要的W_Q, W_K, W_V, W_O  W_O可以理解成 多头 Attention 的总出口 / 融合器。
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, D = x.shape  # [Batch, 64, 128]
        
        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ V).transpose(1, 2).contiguous().view(B, N, D)
        out = self.W_o(out)
        return out


class FFN(nn.Module):
    """前馈网络 —— 对每个Patch独立做非线性变换"""
    def __init__(self, embed_dim=128, mlp_ratio=2, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)  # 128*2=256
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.act = nn.GELU()  # ★ ViT用GELU，你的CNN用ReLU
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """一个完整的Transformer层 —— 相当于CNN里的一个卷积块"""
    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=2, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)  # ★ 这是LayerNorm，你的CNN里没有
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x):
        # Pre-Norm结构：先Norm，再Attention/FFN，最后加残差
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ViTForCIFAR10(nn.Module):
    """
    你的CNNModel:
        conv1 -> pool -> relu -> conv2 -> pool -> relu -> flatten -> fc1 -> relu -> fc2
    
    ViT:
        patch_embed -> pos_embed -> [TransformerBlock × N] -> norm -> cls_token -> head
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3, 
                 embed_dim=128, depth=4, num_heads=4, num_classes=10):
        super().__init__()
        
        # ★ Step 1: Patch Embedding (替代你的conv1)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches  # 64
        
        # ★ Step 2: CLS Token + 位置编码 (ViT特有，你的CNN没有)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        print(self.cls_token.shape)  # [1, 1, 128]
        
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.pos_dropout = nn.Dropout(0.1)
        
        # ★ Step 3: Transformer Encoder (替代你的conv2+pool)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=2, dropout=0.1) 
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # ★ Step 4: 分类头 (对应你的fc1+fc2)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        
        # 1. Patch Embedding (替代卷积)
        x = self.patch_embed(x)  # [B, 64, 128]
        
        # 2. 加上CLS token (ViT特有)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, 128]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 65, 128]
        
        # 3. 加上位置编码
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # 4. 经过N层Transformer (替代多层卷积)
        for block in self.blocks:
            x = block(x)
        
        # 5. 取CLS token做分类 (替代全局池化或flatten)
        x = self.norm(x)
        cls_out = x[:, 0, :]  # [B, 128]  取第一个token
        
        # 6. 分类头 (和你CNN的fc2一样)
        out = self.head(cls_out)  # [B, 10]
        return out


def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


def train(train_dataset):
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # ★ 唯一的区别：这里创建的是ViT，不是CNN
    model = ViTForCIFAR10(
        img_size=32, patch_size=4, in_channels=3,
        embed_dim=128, depth=4, num_heads=4, num_classes=10
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)  # ★ 用AdamW，你的CNN用Adam
    
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
    
    torch.save(model.state_dict(), './vit_cifar10.pth')


def evaluate(test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = ViTForCIFAR10(
        img_size=32, patch_size=4, in_channels=3,
        embed_dim=128, depth=4, num_heads=4, num_classes=10
    )
    model.load_state_dict(torch.load("./vit_cifar10.pth", map_location='cpu'))
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
    train_dataset, test_dataset = load_dataset()
    train(train_dataset)      # 训练ViT
    evaluate(test_dataset)    # 评估ViT