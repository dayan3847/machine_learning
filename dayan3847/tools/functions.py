import numpy as np


def check_shape(array: np.array, expected_shape: tuple):
    array = np.array(array)
    if array.shape != expected_shape:
        raise Exception('Expected shape: {}, actual shape: {}'.format(expected_shape, array.shape))


def check_shape_dim(array: np.array, expected_shape_dim: int):
    array = np.array(array)
    if array.ndim != expected_shape_dim:
        raise Exception('Expected shape len: {}, actual shape len: {}'.format(expected_shape_dim, len(array.shape)))


def check_vector_dim(point: np.array, dim: int):
    check_shape(point, (dim,))


def check_vector_set_dim(point_set: np.array, dim: int):
    point_set = np.array(point_set)
    if len(point_set.shape) != 2 or point_set.shape[1] != dim:
        raise Exception('Expected shape: ({},point_count), actual shape: {}'.format(dim, point_set.shape))
