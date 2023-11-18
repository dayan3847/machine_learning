import numpy as np

from dayan3847.models.Model import Model
from dayan3847.tools.functions import check_vector_dim, check_vector_set_dim


class MultivariateModel(Model):
    def __init__(self,
                 factors_x_dim: list[int],  # Number of factors per dimension, Ex. [5,5]
                 init_weights: float | None = None,
                 ):

        self.factors_x_dim: list[int] = factors_x_dim
        self.dim: int = len(factors_x_dim)  # Number of dimensions, Ex. 2
        if self.dim == 0:
            raise Exception('factors_x_dim must have at least one element')

        f: int = 1
        for _f in self.factors_x_dim:
            f *= _f

        super().__init__(f, init_weights)

    def g(self, x: np.array) -> np.array:
        x = np.array(x)
        if x.ndim == 1:
            check_vector_dim(x, self.dim)
            return self.g_single(x)
        elif x.ndim == 2:
            check_vector_set_dim(x, self.dim)
            return self.g_set(x)
        else:
            raise Exception('x.ndim invalid')

    def bb(self, x: np.array) -> np.array:
        pass

    def train_single(self, x: np.array, y: float, a: float):
        check_vector_dim(x, self.dim)
        super().train_single(x, y, a)
