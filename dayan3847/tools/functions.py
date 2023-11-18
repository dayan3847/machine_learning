import numpy as np


def check_shape(array: np.array, expected_shape: tuple):
    if array.shape != expected_shape:
        raise Exception('Expected shape: {}, actual shape: {}'.format(expected_shape, array.shape))


def check_shape_len(array: np.array, expected_shape_len: int):
    if len(array.shape) != expected_shape_len:
        raise Exception('Expected shape len: {}, actual shape len: {}'.format(expected_shape_len, len(array.shape)))


def check_shape_vector(point: np.array, dim: int):
    check_shape(point, (dim,))


def check_shape_vector_set(point_set: np.array, dim: int):
    if len(point_set.shape) != 2 or point_set.shape[1] != dim:
        raise Exception('Expected shape: ({},point_count), actual shape: {}'.format(dim, point_set.shape))
