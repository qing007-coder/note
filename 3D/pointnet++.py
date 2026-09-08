import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, BatchNorm1d as BN
import numpy as np

# ============================================================
# 第一部分：基础工具函数（对应底层原理）
# ============================================================

def square_distance(src, dst):
    """
    计算两组点云之间的欧式距离平方矩阵
    对应 PointNet++ 中球查询(ball query)的距离计算
    src: [B, N, C]   dst: [B, M, C]
    返回: [B, N, M]  距离平方矩阵
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    # 计算 (x-y)^2 = x^2 + y^2 - 2xy
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz, npoint):
    """
    最远点采样 (Farthest Point Sampling)
    对应 PointNet++ 论文 Section 3.2 的关键步骤
    xyz: [B, N, 3]  点云坐标
    npoint: int     采样的目标点数
    返回: [B, npoint]  采样点的索引
    """
    device = xyz.device
    B, N, _ = xyz.shape
    
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    """
    根据索引从点云中取出对应的点
    对应 PointNet++ 中的分组(grouping)操作
    points: [B, N, C]   idx: [B, S] 或 [B, S, K]
    返回: [B, S, C] 或 [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    球查询 (Ball Query)
    对应 PointNet++ 论文 Section 3.3 的局部邻域构建
    在半径 radius 内为每个 new_xyz 点寻找最多 nsample 个邻居
    xyz: [B, N, 3]    原始点云
    new_xyz: [B, S, 3] 查询中心点
    返回: [B, S, nsample]  每个查询点对应的邻居索引
    """
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(xyz.device).view(1, 1, N).repeat([B, S, 1])
    
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N  # 超出半径的标记为 N (无效索引)
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    
    # 如果邻居不足 nsample 个，用第一个有效点填充（保证维度一致）
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


# ============================================================
# 第二部分：PointNet++ 核心模块
# ============================================================

class PointNetSetAbstraction(nn.Module):
    """
    Set Abstraction 模块 (对应论文 SA 层)
    包含: 采样 → 分组 → 局部特征提取 (PointNet)
    
    npoint: 采样点数 (None 表示不采样, group_all)
    radius: 球查询半径
    nsample: 每个局部区域的采样点数
    in_channel: 输入特征维度
    mlp: 多层感知机结构 [中间维度, ..., 输出维度]
    group_all: 是否对整个点云做全局特征提取
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        # MLP 用于提取局部特征
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz, points):
        """
        xyz: [B, N, 3]  点云坐标
        points: [B, N, C]  点云特征 (可能为 None)
        """
        B, N, _ = xyz.shape
        if points is not None:
            # points: [B, N, C] -> [B, C, N] 方便 Conv1d 处理
            points = points.permute(0, 2, 1)
        
        if self.group_all:
            # 不采样，直接对所有点做特征提取 (用于最终的全局特征)
            new_xyz = torch.zeros(B, 1, 3).to(xyz.device)
            new_points = points.unsqueeze(2)  # [B, C, 1, N]
            grouped_xyz = xyz.permute(0, 2, 1).unsqueeze(2)  # [B, 3, 1, N]
        else:
            # 1. 最远点采样
            fps_idx = farthest_point_sample(xyz, self.npoint)  # [B, npoint]
            new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
            
            # 2. 球查询
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, 3]
            
            # 3. 坐标归一化 (减去中心点)
            grouped_xyz -= new_xyz.view(B, self.npoint, 1, 3)
            
            if points is not None:
                # 获取分组对应的特征
                grouped_points = index_points(points.permute(0, 2, 1), idx)  # [B, npoint, nsample, C]
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # [B, npoint, nsample, C+3]
            else:
                new_points = grouped_xyz
        
        # 4. 将局部点云通过 MLP 提取特征 (类似 PointNet)
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C, nsample, npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        # 5. 最大池化聚合局部特征
        new_points = torch.max(new_points, 2)[0]  # [B, C, npoint]
        new_points = new_points.permute(0, 2, 1)  # [B, npoint, C]
        
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    """
    多尺度分组 (MSG) 版本的 Set Abstraction
    对应 PointNet++ 论文 Section 3.3 的 MSG 架构
    使用多个不同半径并行提取特征，然后拼接
    """
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        
        # 为每个尺度创建一个 SA 模块
        self.sa_modules = nn.ModuleList()
        for i in range(len(radius_list)):
            self.sa_modules.append(
                PointNetSetAbstraction(
                    npoint=npoint,
                    radius=radius_list[i],
                    nsample=nsample_list[i],
                    in_channel=in_channel,
                    mlp=mlp_list[i],
                    group_all=False
                )
            )
    
    def forward(self, xyz, points):
        """
        每个尺度独立进行 SA，最后拼接特征
        """
        # 所有尺度共享同一组采样点
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)
        
        new_points_list = []
        for sa in self.sa_modules:
            # 注意：每个 SA 内部会重新做球查询，但使用相同的采样点
            _, new_points = sa(xyz, points)
            new_points_list.append(new_points)
        
        # 特征拼接
        new_points = torch.cat(new_points_list, dim=-1)
        return new_xyz, new_points


# ============================================================
# 第三部分：完整模型 (分类 + 分割)
# ============================================================

class PointNet2Classification(nn.Module):
    """
    PointNet++ 分类模型
    输入: [B, N, 3+C]  点云坐标和特征
    输出: [B, num_class]  分类分数
    """
    def __init__(self, num_class, input_channels=0, use_xyz=True):
        super().__init__()
        # 输入特征维度 = 坐标(3) + 额外特征
        in_channel = input_channels + 3 if use_xyz else input_channels
        
        # SA 层1: 采样512点，半径0.2，每局部32点
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=in_channel, mlp=[64, 64, 128], group_all=False
        )
        # SA 层2: 采样128点，半径0.4，每局部64点
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256], group_all=False
        )
        # SA 层3: 全局特征聚合 (group_all)
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True
        )
        
        # 分类头
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)
    
    def forward(self, xyz, points=None):
        """
        xyz: [B, N, 3]
        points: [B, N, C]  可选
        """
        if points is not None:
            # 将坐标和特征拼接作为输入
            points = torch.cat([xyz, points], dim=-1)
        else:
            points = xyz
        
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # l3_points: [B, 1, 1024] -> [B, 1024]
        x = l3_points.squeeze(1)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)
        return x


class PointNet2Segmentation(nn.Module):
    """
    PointNet++ 分割模型 (编码器-解码器结构)
    输入: [B, N, 3+C]  输出: [B, N, num_class] 每个点的分类
    """
    def __init__(self, num_class, input_channels=0):
        super().__init__()
        in_channel = input_channels + 3
        
        # ===== 编码器 (Encoder) =====
        # 使用 MSG 架构的 SA 层
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.1, 0.2, 0.4],
            nsample_list=[16, 32, 64],
            in_channel=in_channel,
            mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=128,
            radius_list=[0.2, 0.4, 0.8],
            nsample_list=[32, 64, 128],
            in_channel=64 + 128 + 128,  # 上一层的拼接特征维度
            mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]]
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=128 + 256 + 256,  # 上一层的拼接特征维度
            mlp=[256, 512, 1024], group_all=True
        )
        
        # ===== 解码器 (Decoder) =====
        # 上采样和特征传播
        self.fp3 = PointNetFeaturePropagation(in_channel=128+256+256 + 1024, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=64+128+128 + 256, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=in_channel + 128, mlp=[128, 128, 128])
        
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_class, 1)
    
    def forward(self, xyz, points=None):
        if points is not None:
            points = torch.cat([xyz, points], dim=-1)
        else:
            points = xyz
        
        # ===== 编码过程 =====
        # 保存每层的坐标和特征用于跳连接
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # ===== 解码过程 (特征传播) =====
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, points, l1_points)
        
        # 输出
        x = l0_points.permute(0, 2, 1)  # [B, C, N]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        x = self.conv2(x)  # [B, num_class, N]
        x = x.permute(0, 2, 1)  # [B, N, num_class]
        return x


class PointNetFeaturePropagation(nn.Module):
    """
    特征传播模块 (Feature Propagation)
    用于分割网络的上采样，将高层的语义特征传播到低层
    通过最近邻插值 + 跳连接 + MLP 实现
    """
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz1, xyz2, points1, points2):
        """
        xyz1: [B, N1, 3]  目标点 (更多点，分辨率更高)
        xyz2: [B, N2, 3]  源点 (更少点，分辨率更低)
        points1: [B, N1, C1]  目标点的特征 (低层)
        points2: [B, N2, C2]  源点的特征 (高层)
        """
        B, N1, _ = xyz1.shape
        _, N2, _ = xyz2.shape
        
        if N2 == 1:
            # 如果源点只有1个，直接广播
            interpolated_points = points2.repeat(1, N1, 1)
        else:
            # 1. 计算距离矩阵
            dists = square_distance(xyz1, xyz2)  # [B, N1, N2]
            # 2. 找最近的3个点
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N1, 3]
            # 3. 计算权重 (逆距离加权)
            weight = 1.0 / (dists + 1e-8)
            # 4. 加权平均插值
            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.view(B, N1, 3, 1),
                dim=2
            )
        
        # 5. 跳连接: 将插值后的高层特征与低层特征拼接
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        
        # 6. MLP 融合
        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = new_points.permute(0, 2, 1)
        
        return new_points


# ============================================================
# 第四部分：使用示例
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 生成模拟数据: batch=4, 点云数=2048, 特征维度=6 (xyz + rgb)
    B, N, C = 4, 2048, 6
    xyz = torch.randn(B, N, 3).to(device)
    features = torch.randn(B, N, 3).to(device)
    
    print("=" * 50)
    print("1. 测试分类模型")
    classifier = PointNet2Classification(num_class=40, input_channels=3).to(device)
    out = classifier(xyz, features)
    print(f"分类模型输出形状: {out.shape}  (B, num_class)")
    
    print("\n" + "=" * 50)
    print("2. 测试分割模型")
    segmenter = PointNet2Segmentation(num_class=13, input_channels=3).to(device)
    out = segmenter(xyz, features)
    print(f"分割模型输出形状: {out.shape}  (B, N, num_class)")
    
    print("\n" + "=" * 50)
    print("3. 模型参数量统计")
    print(f"分类模型参数量: {sum(p.numel() for p in classifier.parameters()) / 1e6:.2f} M")
    print(f"分割模型参数量: {sum(p.numel() for p in segmenter.parameters()) / 1e6:.2f} M")