import numpy as np

from dayan3847.models.Model import Model
from dayan3847.tools.functions import check_shape_len


class LinearModel(Model):

    def g(self, x):
        if isinstance(x, np.ndarray):
            check_shape_len(x, 1)
            return self.g_set(x)
        else:
            return self.g_single(x)

    def train_single(self, x: float, y: float, a: float):
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            raise Exception('x or y is np.ndarray')
        super().train_single(x, y, a)
