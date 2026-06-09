import open3d as o3d
import numpy as np
from typing import Tuple


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

    def draw(self, point_show_normal: bool) -> None:
        """
        渲染并交互式展示当前类内部持有的点云几何对象。
        通过向上转型为 Geometry 列表，彻底解决 IDE 的 Expected type 'list[Geometry]' 警告。
        """
        # 1. 显式将其转为 Open3D 基础几何基类 Geometry 的列表，消除类型警告
        geometry_list: list[o3d.geometry.Geometry] = [self.pcd]

        # 2. 调用底层渲染引擎
        o3d.visualization.draw_geometries(geometry_list, point_show_normal=point_show_normal)

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

    def calculate_normals(self, radius: float = 0.05, max_nn: int = 100):
        """

        :param max_nn:
        :param radius:
        :return:
        """

        self.pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius,
                max_nn=max_nn,
            )
        )

    def find_knn_neighbors(self, source: int, k: int) -> Tuple[
        int, o3d.utility.IntVector, o3d.utility.DoubleVector]:
        """
        通过KNN（K近邻）算法寻找最近的K个邻居

        :param source: 目标点的坐标索引
        :param k: 想要寻找的邻居数量 (必须是正整数)
        :return: (找到的邻居数, 邻居在点云中的索引列表, 距离的平方列表)
        """
        # 健壮性检查：确保传入的是整数
        k = int(k)

        kdtree = o3d.geometry.KDTreeFlann(self.pcd)  # 建立k维二叉树索引

        # 执行搜索
        count, idx, dis = kdtree.search_knn_vector_3d(self.pcd.points[source], k)

        return count, idx, dis

    @classmethod
    def demo_point_to_point(cls) -> None:
        """
        Demo：使用 Open3D 自带测试数据集演示【点对点】ICP 配准。
        无需外部传参，一键运行查看配准前后的可视化对比。
        """
        print("\n" + "=" * 20 + " 启动点对点 ICP Demo " + "=" * 20)
        # 1. 载入内置的错位测试数据
        demo_data = o3d.data.DemoICPPointClouds()
        source_pcd = o3d.io.read_point_cloud(demo_data.paths[0])
        target_pcd = o3d.io.read_point_cloud(demo_data.paths[1])

        # 2. 染色（黄色为待配准源，蓝色为目标基准）
        source_pcd.paint_uniform_color([1, 0.706, 0])
        target_pcd.paint_uniform_color([0, 0.651, 0.929])

        # 3. 备份一份未对齐的源点云，用于前后期对比
        import copy
        source_old = copy.deepcopy(source_pcd)

        # 4. 运行点对点 ICP 配准
        threshold = 0.02
        result = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        print(f"配准重合度 (Fitness): {result.fitness:.4f}")
        print(f"均方根误差 (RMSE): {result.inlier_rmse:.4f}")

        # 5. 原地移动源点云
        source_pcd.transform(result.transformation)

        # 6. 渲染验证
        print("-> 正在展示：配准前的【错位】状态（请按 Q 键关闭窗口以继续）...")
        o3d.visualization.draw_geometries([source_old, target_pcd], window_name="Before Registration")

        print("-> 正在展示：点对点 ICP 配准后的【融合】状态...")
        o3d.visualization.draw_geometries([source_pcd, target_pcd], window_name="After Point-to-Point ICP")

    @classmethod
    def demo_point_to_plane(cls) -> None:
        """
        Demo：使用 Open3D 自带测试数据集演示【点对面】ICP 配准。
        由于点对面硬性要求目标点云具备法线，函数内部会自动前置计算法线。
        ICP 第一步是帮 source 的每个点在 target 里找最近的邻居。如果两个点距离大于这个 threshold，ICP 就认为“它们俩根本不是同一个地方，强行配准会带偏大部队”，于是直接丢弃这对匹配
        """
        print("\n" + "=" * 20 + " 启动点对面 ICP Demo " + "=" * 20)
        # 1. 载入内置的错位测试数据
        demo_data = o3d.data.DemoICPPointClouds()
        source_pcd = o3d.io.read_point_cloud(demo_data.paths[0])
        target_pcd = o3d.io.read_point_cloud(demo_data.paths[1])

        # 2. 染色
        source_pcd.paint_uniform_color([1, 0.706, 0])
        target_pcd.paint_uniform_color([0, 0.651, 0.929])

        import copy
        source_old = copy.deepcopy(source_pcd)

        # 3. 【核心前置步骤】：为 Target 估计法线（点对面算法的强依赖）
        # 这里模拟类内部的 calculate_normals 算子
        target_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )

        # 4. 运行点对面 ICP 配准
        threshold = 0.02
        result = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )

        print(f"配准重合度 (Fitness): {result.fitness:.4f}")
        print(f"均方根误差 (RMSE): {result.inlier_rmse:.4f}")

        # 5. 原地移动源点云
        source_pcd.transform(result.transformation)

        # 6. 渲染验证
        print("-> 正在展示：配准前的【错位】状态（请按 Q 键关闭窗口以继续）...")
        o3d.visualization.draw_geometries([source_old, target_pcd], window_name="Before Registration")

        print("-> 正在展示：点对面 ICP 配准后的【融合】状态...")
        o3d.visualization.draw_geometries([source_pcd, target_pcd], window_name="After Point-to-Plane ICP")

    def segment_and_separate_plane(
            self,
            distance_threshold: float = 0.02,
            ransac_n: int = 3,
            num_iterations: int = 1000
    ) -> o3d.geometry.PointCloud:
        """
        算子3：RANSAC 平面分割与宏观背景分离

        【物理图像】：利用 RANSAC 算法在三维空间中拟合出一个包含最多点数的宏观几何平面（如地面、墙面、桌面）。
                    通过阈值将点云切分为“平面内点”与“非平面外点”两部分，实现前背景的彻底分离。
        【适用场景】：移除地面等大型结构性背景（障碍物检测中的“数据清洗/降噪”），或者专门提取特定的平面结构。

        :param distance_threshold: 点到拟合平面的最大允许距离（单位：米）。此处 0.02 代表 2 厘米内的点均判定为平面内点。
        :param ransac_n: 每次迭代初始化平面的最少点数。几何学上三点确定一个平面，故固定为 3。
        :param num_iterations: 最大随机迭代优化次数。1000 次是兼顾计算效率与全局最优解的经验值。
        :return: 提取出的平面（地面）点云对象（o3d.geometry.PointCloud）。同时，类内部的 self.pcd 将原地更新为剔除平面后的物体点云。
        """
        # 1. 调用底层算子计算平面方程系数（plane_model）和属于该平面的点索引（inliers）
        plane_model, inliers = self.pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        # 打印平面方程 Ax + By + Cz + D = 0 的系数，便于调试
        print(
            f"[RANSAC] 平面方程参数: A={plane_model[0]:.4f}, B={plane_model[1]:.4f}, C={plane_model[2]:.4f}, D={plane_model[3]:.4f}")

        # 2. 提取平面点云（invert=False）：只捞出索引列表 inliers 包含的点（如地面）
        ground_pcd = self.pcd.select_by_index(inliers, invert=False)

        # 3. 原地更新内部点云对象（invert=True）：排除平面点，只保留剩下的点（如前景障碍物）
        # 目的是防止大面积的地面干扰后续的聚类或目标提取算法，并节省计算算力
        self.pcd = self.pcd.select_by_index(inliers, invert=True)

        # 4. 返回分离出来的平面点云，方便外部做单独渲染或进一步分析
        return ground_pcd

    def cluster_objects_dbscan(self, eps: float = 0.05, min_points: int = 20) -> np.ndarray:
        """
        算子4：DBSCAN 密度聚类 (Density-Based Spatial Clustering of Applications with Noise)

        【物理图像】：以每个点为中心、eps 为半径画球。若球内点数 >= min_points，则该点升级为“核心点”并拉帮结派形成一个独立的几何实体。
                    算法不需要提前指定聚类数量，并能自发识别孤立的噪声点。
        【适用场景】：在剔除地面等大面积背景后，将剩余散落的点云自动“分堆”，识别出具体的、独立的 3D 目标（如人、车、障碍物）。

        :param eps: 邻域半径（单位：米）。5 厘米是近距离物体（如桌面、室内场景）分块的常用阈值。
        :param min_points: 判定为核心点所需的最小邻居数量门限。越小对稀疏点云越友好，但越容易误把噪点聚类。
        :return: 聚类标签数组（形状与点云点数一致）。
                 非负整数(0, 1, 2...)代表物体编号，-1 代表被算法剔除的孤立噪点。
        """
        # 1. 调用底层 C++ 算子，返回一个 IntVector
        labels_vector = self.pcd.cluster_dbscan(eps=eps, min_points=min_points)

        # 2. 将结果强转为 NumPy 数组以便于高级索引和切片操作
        labels = np.array(labels_vector)

        # 3. 统计聚类结果
        max_label = labels.max()
        n_clusters = max_label + 1 if max_label >= 0 else 0
        n_noise = np.sum(labels == -1)
        print(f"[DBSCAN] 聚类完成。共切分出 {n_clusters} 个独立物体，识别出 {n_noise} 个孤立噪点。")

        return labels

    def extract_specific_cluster(self, labels: np.ndarray, cluster_idx: int) -> o3d.geometry.PointCloud:
        """
        工具算子：根据 DBSCAN 标签提取特定的物理目标。

        :param labels: 由 cluster_objects_dbscan 返回的标签数组
        :param cluster_idx: 想要单独过滤出的物体编号（例如 0 代表第一个物体，-1 代表噪声点）
        :return: 过滤出来的独立物体点云对象（o3d.geometry.PointCloud）
        """
        # 1. 使用 numpy 找出所有属于目标聚类的点索引
        target_indices = np.where(labels == cluster_idx)[0]

        # 2. 从当前点云中安全过滤并提取
        cluster_pcd = self.pcd.select_by_index(target_indices)
        return cluster_pcd

    def get_all_clusters_bounding_boxes(self, labels: np.ndarray, use_obb: bool = True) -> list[o3d.geometry.Geometry]:
        """
        [核心特征提取算子]
        遍历 DBSCAN 聚类标签，逐一抽离独立的点云对象，解算其三维空间包围盒及关键几何属性（质心、尺寸、体积）。

        :param labels: 由 cluster_dbscan 算法输出的形状为 (N,) 的一维 NumPy 标签数组。-1 代表噪声。
        :param use_obb: 是否启用定向包围盒(OBB)。True 则计算贴合朝向的 OBB（精度高）；False 则计算轴对齐包围盒(AABB)。
        :return: 包含所有计算生成的包围盒实例（ o3d.geometry.AxisAlignedBoundingBox 或 OrientedBoundingBox ）的列表。
        """
        import matplotlib.pyplot as plt

        # 1. 边界防御检测：检查是否存在有效聚类
        max_label = labels.max()
        if max_label < 0:
            print("[警告] 标签阵列未检测到任何有效聚类核心（max_label < 0），中断执行。")
            return []

        bboxes = []

        # 2. 动态生成调色盘：使用 tab20 离散颜色映射表，最多支持 20 种不重复的高对比度颜色标注
        cmap = plt.get_cmap("tab20")

        # 3. 顺序遍历每个独立的几何聚类簇 (Label 范围: 0 到 max_label)
        for cluster_idx in range(max_label + 1):

            # 3.1 调用内部工具算子，过滤并抽取当前 ID 对应的独立 PointCloud 实体
            cluster_pcd = self.extract_specific_cluster(labels, cluster_idx)

            # 3.2 鲁棒性检查：若该聚类的有效点数过少（如少于4个点），无法正确支撑 3D 空间的协方差矩阵或凸包解算
            if len(cluster_pcd.points) < 4:
                continue

            # 3.3 核心步骤：解算包围盒并提取三维尺寸物理特征 (Extent)
            if use_obb:
                # 计算定向包围盒（OBB）：适应物体的空间偏转角度
                bbox = cluster_pcd.get_oriented_bounding_box()
                extent = bbox.extent  # OBB 直接读取其局部坐标系下的三轴边长属性 [Length, Width, Height]
            else:
                # 计算轴对齐包围盒（AABB）：边严格平行于世界坐标轴
                bbox = cluster_pcd.get_axis_aligned_bounding_box()
                extent = bbox.get_extent()  # AABB 需通过内部 getter 函数获取三维尺寸

            # 3.4 几何体积推导：长 * 宽 * 高 (单位：立方米，取决于原始点云的物理尺度)
            volume = extent[0] * extent[1] * extent[2]

            # 3.5 提取物体的绝对三维绝对质心坐标 [X, Y, Z]
            center = bbox.get_center()

            # 4. 结构化日志中立输出（用于离线调试与感知结果验证）
            print(f"--- [Target Cluster #{cluster_idx}] ---")
            print(f"  空间质心 (Center X,Y,Z) : [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
            print(f"  物理长宽尺寸 (Extent L,W,H): [{extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f}]")
            print(f"  外接包围体积 (Volume)     : {volume:.4f} m³")

            # 5. 赋能可视化：将 matplotlib 转换出来的 RGBA 格式前三项(RGB)提取出来，注入 Open3D 渲染器
            # cmap(idx) 返回 (R, G, B, A)，其中数值范围已被归一化至 [0.0, 1.0]
            bbox.color = cmap(cluster_idx)[:3]

            # 6. 将当前处理完毕的包围盒压入队列，留待后续批量渲染渲染或数据传递
            bboxes.append(bbox)

        return bboxes
