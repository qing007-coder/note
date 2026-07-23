from .tensor import Tensor, nn_demo
import numpy as np


def test_pytorch():
    tensor = Tensor()
    # tensor.print(tensor.tensor_from_num(5.0))
    # tensor.print(tensor.tensor_from_list([1.0, 2.0, 3.0]))
    # tensor.print(tensor.tensor_from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]])))

    # t1 = tensor.ones((2, 3))
    # tensor.print(t1)

    # t2 = tensor.tensor_from_list([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # t3 = tensor.one_like(t2)
    # tensor.print(t3)

    # t4 = tensor.zeros((2, 3))
    # tensor.print(t4)

    # t5 = tensor.tensor_from_list([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # t6 = tensor.one_like(t5)
    # tensor.print(t6)

    # t7 = tensor.tensor_from_list([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # t8 = tensor.full_like(t7, fill_value=9.0)
    # tensor.print(t8)

    # tensor.print(tensor.arange(0, 10, 2))
    # tensor.print(tensor.linspace(0, 10, 5))

    # tensor.print(tensor.randn((2, 3)))
    # tensor.print(tensor.rand((2, 3)))
    # tensor.print(tensor.randint(low=1, high=10, shape=(2, 3)))

    # tensor.print(tensor.convert_to_float64(tensor.tensor_from_list([[1.0, 2.0], [3.0, 4.0]])))

    # t1 = tensor.randint(low=1, high=10, shape=(2, 3))
    # t2 = tensor.reshape(t1, new_shape=(3, 2))
    # tensor.print(t2)

    # t1 = tensor.randint(low=1, high=10, shape=(1, 2, 3, 1))
    # tensor.print(t1)

    # t2 = tensor.unsqueeze(tensor=t1, dim=1)
    # tensor.print(t2)

    # t3 = tensor.squeeze(tensor=t2)
    # tensor.print(t3)

    # t1 = tensor.randint(low=1, high=10, shape=(2, 3, 4))
    # tensor.print(t1)
    # t2 = tensor.transpose(t1, dim0=0, dim1=2)
    # tensor.print(t2)

    # t3 = tensor.permute(t1, dims=(2, 0, 1))
    # tensor.print(t3)

    # tensor.demo_backward()

    # tensor.detach_demo()

    nn_demo()
