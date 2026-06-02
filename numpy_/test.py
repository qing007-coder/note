import numpy as np
from .common import Numpy


def numpy_test():
    matrix1 = Numpy([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    ])
    print(matrix1.get_shape())

    matrix2 = Numpy.create_zero((3, 3))
    print(matrix2.get_shape())
    print(matrix2.data)

    matrix3 = Numpy.create_one((3, 3))
    print(matrix3.data)

    matrix4 = Numpy.create_eyes(2)
    print(matrix4.data)

    matrix1.reshape((4, 5))
    print(matrix1.get_shape())
    print(matrix1.data)

    matrix2.broadcast(2)
    print(matrix2.data)

    matrix5 = Numpy.random_matrix((5, 5))
    print(matrix5.data)

    matrix6 = Numpy.random_matrix((5, 5))
    print(matrix6.matmul(np.eye(5, 5)))

    print(matrix3 + matrix2)

    points1 = np.array([
        [1, 2, 3, 0.5],
        [4, 5, 6, 0.8],
        [7, 8, 9, 0.9]
    ])

    # 要前三列
    print(points1[:, :3])

    # 要最后一列
    print(points1[:, 3])

    points2 = np.array([
        [1, 2, 3],
        [100, 200, 300],
        [4, 5, 6]
    ])

    mask = points2[:, 0] < 50
    filtered = points2[mask]
    print(filtered)

    # axis是轴 0是最外层的轴 这个函数就是把最外层的轴进行压缩
    # 每一列平均
    print(points2.mean(axis=0))

    # 每一行平均
    print(points2.mean(axis=1))
