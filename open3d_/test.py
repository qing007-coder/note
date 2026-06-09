from .common import PointCloudManager
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def test_open3d():
    # raw_points = PointCloudManager.generate_random_3d_points(10000)
    # manager = PointCloudManager(raw_points)
    #
    # manager.read_data("./resource/pcd/1.pcd")
    # manager.draw(point_show_normal=False)
    # # print(manager.pcd)
    #
    # print(manager.get_shape())
    #
    # manager.down_sample(0.005)
    #
    # manager.draw(False)
    # print(manager.get_shape())
    #
    # manager.calculate_normals()
    # print(manager.get_shape())
    # manager.draw(point_show_normal=True)
    #
    # count, idx, dis = manager.find_knn_neighbors(0, 10)
    # for i in range(count):
    #     print(idx[i], dis[i])
    #
    # PointCloudManager.demo_point_to_point()
    # PointCloudManager.demo_point_to_plane()

    # run_point_cloud_pipeline()

    np.random.seed(0)
    obj1 = np.random.randn(100, 3) * 0.1 + np.array([0, 0, 0])
    obj2 = np.random.randn(80, 3) * 0.08 + np.array([1, 1, 0.5])
    obj3 = np.random.randn(120, 3) * 0.12 + np.array([-1, 0.5, -0.2])
    # 模拟地面
    floor = np.random.uniform(-2, 2, (500, 3))
    floor[:, 2] = -0.5  # Z 轴固定在 -0.5 模拟平面

    mock_raw_data = np.vstack([obj1, obj2, obj3, floor])

    # 执行流水线
    boxes, foreground = object_detection_pipeline(mock_raw_data)

    # 可视化结果确认
    if boxes:
        o3d.visualization.draw_geometries([foreground] + boxes, window_name="Pipeline Output")


def run_point_cloud_pipeline() -> None:
    """
    工业级点云处理流水线演示函数：
    数据加载 -> 体素降采样 -> 统计去噪 -> 法线计算 -> 点对面 ICP 精配准
    """
    print("\n" + "█" * 20 + " 启动点云处理流水线 Pipeline " + "█" * 20)

    # ==========================================
    # 1. 数据准备阶段 (模拟从 ROS 或文件加载两帧原始错位数据)
    # ==========================================
    print("\n[Step 1] 正在加载原始两帧错位点云数据...")
    demo_data = o3d.data.DemoICPPointClouds()

    # 实例化两个管理器分别管理 Source 和 Target
    source_mgr = PointCloudManager(np.random.randn(10, 3))  # 临时初始化
    target_mgr = PointCloudManager(np.random.randn(10, 3))

    # 真正读入测试数据
    source_mgr.read_data(demo_data.paths[0])
    target_mgr.read_data(demo_data.paths[1])

    print(f"-> 原始 Source 点云形状: {source_mgr.get_shape()}")
    print(f"-> 原始 Target 点云形状: {target_mgr.get_shape()}")

    # 染色以便可视化区分
    source_mgr.pcd.paint_uniform_color([1, 0.706, 0])  # 源点云：黄色
    target_mgr.pcd.paint_uniform_color([0, 0.651, 0.929])  # 目标点云：蓝色

    # 备份未处理且未配准的原始源点云，用于最后的效果大对比
    import copy
    source_raw_backup = copy.deepcopy(source_mgr.pcd)

    # ==========================================
    # 2. 前处理阶段：降采样 (Downsampling)
    # ==========================================
    print("\n[Step 2] 正在进入前处理：执行体素下采样 (Voxel Downsampling)...")
    voxel_size = 0.02  # 定义 2 厘米的体素格子
    source_mgr.down_sample(voxel_size)
    target_mgr.down_sample(voxel_size)
    print(f"-> 降采样后 Source 点云形状: {source_mgr.get_shape()}")
    print(f"-> 降采样后 Target 点云形状: {target_mgr.get_shape()}")

    # ==========================================
    # 3. 前处理阶段：去噪 (Denoising)
    # ==========================================
    print("\n[Step 3] 正在进入前处理：执行统计学离群点剔除去噪...")
    # 利用链式调用完成连续过滤
    source_mgr.clean_noise_by_statistical(nb_neighbors=20, std_ratio=2.0)
    target_mgr.clean_noise_by_statistical(nb_neighbors=20, std_ratio=2.0)
    print(f"-> 去噪后 Source 点云形状: {source_mgr.get_shape()}")
    print(f"-> 去噪后 Target 点云形状: {target_mgr.get_shape()}")

    # ==========================================
    # 4. 特征提取阶段：法线计算 (Normals Estimation)
    # ==========================================
    print("\n[Step 4] 正在进入特征提取：计算表面法向量...")
    # 点对面 ICP 强依赖于 Target 的法线，这里两边都计算，便于观察
    source_mgr.calculate_normals(radius=voxel_size * 2, max_nn=30)
    target_mgr.calculate_normals(radius=voxel_size * 2, max_nn=30)
    print("-> 表面法向量计算完成。")

    # ==========================================
    # 5. 精配准阶段：点对面 ICP (Registration)
    # ==========================================
    print("\n[Step 5] 正在进入精配准：运行工业级【点对面】 ICP 算法...")
    threshold = 0.03  # 匹配对最大距离阈值设为 3 厘米

    # 直接调用底层的精配准算子
    result = o3d.pipelines.registration.registration_icp(
        source_mgr.pcd,
        target_mgr.pcd,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    print(f"   [成绩单] 配准重合度 (Fitness): {result.fitness:.4f}")
    print(f"   [成绩单] 均方根误差 (RMSE): {result.inlier_rmse:.4f}")
    print(f"   [核心解] 4x4 变换矩阵:\n{result.transformation}")

    # 原地移动经过前处理后的 source 点云
    source_mgr.pcd.transform(result.transformation)

    # ==========================================
    # 6. 结果可视化验证 (Pipeline Results Visualization)
    # ==========================================
    print("\n[Step 6] 正在拉起可视化窗口进行效果验证...")

    # 窗口 1：完全未处理过的原始错位数据
    print("-> 窗口 1 正在展示：没有任何处理的【原始错位点云】（请按 Q 键关闭窗口继续）...")
    o3d.visualization.draw_geometries([source_raw_backup, target_mgr.pcd], window_name="1. Raw Mismatched Clouds")

    # 窗口 2：经过流水线洗礼，完美配准融合后的高质量干净数据
    print("-> 窗口 2 正在展示：经过【降采样+去噪+法线精配准】后的最终完美状态...")
    o3d.visualization.draw_geometries([source_mgr.pcd, target_mgr.pcd], window_name="2. Final Aligned Pipeline Result")

    print("\n" + "█" * 20 + " 流水线演示结束 Pipeline Done " + "█" * 20)


def object_detection_pipeline(raw_data: np.ndarray) -> tuple[list, o3d.geometry.PointCloud]:
    """
    3D 点云目标提取与检测完整流水线

    :param raw_data: 形状为 (N, 3) 的原始点云 NumPy 数组
    :return: (包含所有物体 OBB 的列表, 清洗聚类后的前景点云)
    """
    print("\n" + "=" * 20 + " 启动点云处理流水线 " + "=" * 20)

    # ----------------------------------------------------
    # Step 1: 数据数据注入与初始化
    # ----------------------------------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(raw_data)
    print(f"[配置] 原始点云初始化完成，点数: {len(pcd.points)}")

    # ----------------------------------------------------
    # Step 2: 体素降采样 (控制数据规模)
    # ----------------------------------------------------
    voxel_size = 0.02  # 2cm 体素
    pcd = pcd.voxel_down_sample(voxel_size)
    print(f"[降采样] 当前点数: {len(pcd.points)}")

    # ----------------------------------------------------
    # Step 3: 统计学去噪 (剔除空气散射飞点)
    # ----------------------------------------------------
    _, inlier_indices = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(inlier_indices)
    print(f"[去噪] 统计学清洗完成，剩余点数: {len(pcd.points)}")

    # ----------------------------------------------------
    # Step 4: RANSAC 平面分割 (剥离地面背景)
    # ----------------------------------------------------
    # 距离平面 3cm 内的点均判定为地面
    plane_model, road_indices = pcd.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=1000)

    # 提取前景（非地面部分）
    foreground_pcd = pcd.select_by_index(road_indices, invert=True)
    print(f"[RANSAC] 地面剥离成功。前景点数: {len(foreground_pcd.points)}")

    # ----------------------------------------------------
    # Step 5: DBSCAN 密度聚类 (物体分堆)
    # ----------------------------------------------------
    # 半径 10cm，最少点数 15 个
    labels_vector = foreground_pcd.cluster_dbscan(eps=0.10, min_points=15)
    labels = np.array(labels_vector)

    max_label = labels.max()
    n_clusters = max_label + 1 if max_label >= 0 else 0
    print(f"[DBSCAN] 聚类完成。共切分出 {n_clusters} 个独立物体。")

    if n_clusters == 0:
        return [], foreground_pcd

    # ----------------------------------------------------
    # Step 6: 循环遍历各个语义标签，计算 OBB 几何特征
    # ----------------------------------------------------
    detected_boxes = []
    cmap = plt.get_cmap("tab20")  # 用于生成不同的框颜色

    for cluster_idx in range(n_clusters):
        # 1. 精确获取当前物体 ID 的点云索引
        target_indices = np.where(labels == cluster_idx)[0]
        cluster_pcd = foreground_pcd.select_by_index(target_indices)

        # 2. 计算物体的 定向包围盒 (OBB)
        obb = cluster_pcd.get_oriented_bounding_box()

        # 3. 提取空间位置与物理尺度
        center = obb.get_center()
        extent = obb.extent  # 局部坐标系下的长、宽、高
        volume = extent[0] * extent[1] * extent[2]  # 物理体积

        # 4. 打印格式化指标 (无修饰中立输出)
        print(f"  - 物体 #{cluster_idx}:")
        print(f"    中心坐标 (X, Y, Z) : [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
        print(f"    真实外形 (长, 宽, High): [{extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f}]")
        print(f"    外接体积 (Volume)   : {volume:.4f} m³")

        # 5. 渲染着色与容器收集
        obb.color = cmap(cluster_idx)[:3]
        detected_boxes.append(obb)

    print(f"[输出] 流水线执行完毕，成功捕获 {len(detected_boxes)} 个 3D 目标边界。")
    return detected_boxes, foreground_pcd
