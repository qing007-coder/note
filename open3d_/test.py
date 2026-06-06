from .common import PointCloudManager
import open3d as o3d
import numpy as np


def test_open3d():
    # raw_points = PointCloudManager.generate_random_3d_points(10000)
    # manager = PointCloudManager(raw_points)
    #
    # manager.read_data("./resource/pcd/1.pcd")
    # manager.draw(point_show_normal=False)
    # print(manager.pcd)
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

    run_point_cloud_pipeline()


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
