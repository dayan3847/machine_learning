import numpy as np


class ShapeChecker:

    @staticmethod
    def check_shape(_array: np.array, _expected_shape: tuple):
        if _array.shape != _expected_shape:
            raise Exception('Expected shape: {}, actual shape: {}'.format(_expected_shape, _array.shape))

    @staticmethod
    def check_shape_data(_data: np.array):
        if len(_data.shape) != 2 or _data.shape[0] != 3:
            raise Exception('Expected shape: (3, N), actual shape: {}'.format(_data.shape))

    # point or point list
    @staticmethod
    def check_shape_point(_point: np.array, dim: int = 2):
        if len(_point.shape) != 2 or _point.shape[0] != dim:
            raise Exception('Expected shape: ({},point_count), actual shape: {}'.format(dim, _point.shape))
