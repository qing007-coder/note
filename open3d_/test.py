from .common import PointCloudManager


def test_open3d():
    raw_points = PointCloudManager.generate_random_3d_points(10000)
    manager = PointCloudManager(raw_points)

    manager.read_data("./resource/pcd/plane.pcd")
    manager.draw()
    print(manager.pcd)

    print(manager.get_shape())

    manager.down_sample(0.05)

    manager.draw()
    print(manager.get_shape())

