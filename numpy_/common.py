import numpy as np
from typing import Self


class Numpy:
    def __init__(self, data: list[int] | list[list[int]] | np.ndarray):
        self.data = np.array(data)

    def get_shape(self) -> tuple:
        """
        获取矩阵的形状
        :return:
        """
        return self.data.shape

    def __add__(self, other) -> Self:
        return Numpy(self.data + other.data)

    @classmethod
    def create_zero(cls, params: tuple[int, ...]) -> Self:
        """
        全是0的矩阵
        :param params:
        :return:
        """
        zero_matrix = np.zeros(params, dtype=int)
        return cls(zero_matrix)

    @classmethod
    def create_one(cls, params: tuple[int, ...]) -> Self:
        """
        全为1的矩阵
        :param params:
        :return:
        """
        one_matrix = np.ones(params, dtype=int)
        return cls(one_matrix)

    @classmethod
    def create_eyes(cls, param: int) -> Self:
        """
        Create an eye matrix 单位矩阵
        :param param:
        :return:
        """
        eyes = np.eye(param, dtype=int)
        return cls(eyes)

    def reshape(self, shape: tuple[int, ...]) -> Self:
        """
        Reshapes the data
        :param shape:
        :return:
        """
        return self.data.reshape(shape)

    def broadcast(self, num: int) -> None:
        """
        这个应该是做到__add__做兼容的 兼容Numpy对象和int的 用isinstance判断
        :param num:
        :return:
        """
        self.data += num

    def matmul(self, other: np.ndarray) -> np.ndarray:
        """
        也可以 a @ b 这样来做矩阵乘法
        :param other:
        :return:
        """
        return np.matmul(self.data, other.data)

    @classmethod
    def random_matrix(cls, shape: tuple[int, ...]) -> Self:
        return cls(np.random.random(shape))
