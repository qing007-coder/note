import open3d as o3d
import numpy as np


class PointCloudManager:
    """Open3D 点云对象管理器，封装了从数据注入、特征处理到可视化渲染的完整生命周期。"""

    def __init__(self, data: np.ndarray):
        """
        初始化管理器，并在内部直接构建、封装 Open3D 点云几何对象。

        :param data: 输入的原始点云数据，形状为 (N, 3) 的二维 NumPy 数组
        """
        self.raw_data: np.ndarray = data

        # 将 pcd 对象直接内聚为类属性，不再暴露给外部零散维护
        self.pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()

        # 初始化时自动完成数据注入
        self._inject_points()

    @classmethod
    def generate_random_3d_points(cls, n_points: int) -> np.ndarray:
        """
        静态生成符合标准正态分布的随机三维点云数据。

        :param n_points: 需要生成的点数量 (N)
        :return: 形状为 (n_points, 3) 的符合标准正态分布的 NumPy 数组
        """
        return np.random.randn(n_points, 3)

    def _inject_points(self) -> None:
        """
        内部私有方法：将内存中的 NumPy 数组数据转换为 C++ 底层 Vector3dVector 结构
        并绑定到类内部的 pcd 对象上。
        """
        self.pcd.points = o3d.utility.Vector3dVector(self.raw_data)

    def draw(self) -> None:
        """
        渲染并交互式展示当前类内部持有的点云几何对象。
        通过向上转型为 Geometry 列表，彻底解决 IDE 的 Expected type 'list[Geometry]' 警告。
        """
        # 1. 显式将其转为 Open3D 基础几何基类 Geometry 的列表，消除类型警告
        geometry_list: list[o3d.geometry.Geometry] = [self.pcd]

        # 2. 调用底层渲染引擎
        o3d.visualization.draw_geometries(geometry_list)

    def read_data(self, path: str):
        self.pcd = o3d.io.read_point_cloud(path)

    def get_shape(self) -> tuple[int, ...]:
        points = np.asarray(self.pcd.points)
        return points.shape

    def down_sample(self, voxel_size: float) -> None:
        """
        算子框定一个 0.05 * 0.05 * 0.05 的三维立方体空间。结果： 找出落在这个立方体里的所有点（比如你说的 100 个点），对它们的 X坐标、Y 坐标、Z 坐标分别求平均值（计算几何重心），融合成 1 个全新的物理点。
        降采样
        :param voxel_size:
        :return:
        """
        self.pcd = self.pcd.voxel_down_sample(voxel_size)

    def clean_noise_by_statistical(self, nb_neighbors: int = 20, std_ratio: float = 2.0) -> "PointCloudManager":
        """
        算子1：统计学离群点去除 (Statistical Outlier Removal)

        【物理图像】：通过高斯分布模型，剔除那些“邻居间平均距离异常大”的点。
        【适用场景】：去除雷达测量引起的雾状微小噪声、物体边缘散射的飞点。

        :param nb_neighbors: 考察的目标邻居点数量 (K-NN 检索)
        :param std_ratio: 标准差倍数阈值，越小去噪越严格（通常取 1.0 ~ 3.0）
        :return: 返回自身实例，支持链式调用
        """
        # 1. 计算离群点，返回的 cl 是处理后的点云（不常用），ind 是保留的有效点索引
        _, ind = self.pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )

        # 2. 原地更新内部点云对象
        self.pcd = self.pcd.select_by_index(ind)
        return self

    def clean_noise_by_radius(self, nb_points: int = 16, radius: float = 0.05) -> "PointCloudManager":
        """
        算子2：半径区域离群点去除 (Radius Outlier Removal)

        【物理图像】：在设定的空间球体半径内，如果小岛上的“居民”（点数）太少，则判定为孤立噪点。
        【适用场景】：专门精准打击那些距离主体几何极远的孤立飞点。

        对密度不均匀的数据极度不友好： 激光雷达采集的数据通常是“近密远疏”的。离雷达近的地方，物体表面点极多，这个算子工作得很好。但到了远处的物体（比如 50 米开外的路牌），由于距离远，它本身的物理点云就会变得非常稀疏，这时候 remove_radius_outlier 往往会把远处正常的物体误当成噪点全部抹除。

        :param nb_points: 半径球体内必须包含的最少点数门限
        :param radius: 空间搜索球体的半径大小（单位取决于点云数据的物理尺度）
        :return: 返回自身实例，支持链式调用
        """
        # 1. 计算半径离群点
        _, ind = self.pcd.remove_radius_outlier(
            nb_points=nb_points,
            radius=radius
        )

        # 2. 原地更新内部点云对象
        self.pcd = self.pcd.select_by_index(ind)
        return self

    # def calculate_normals(self, radius: float = 0.05) :
