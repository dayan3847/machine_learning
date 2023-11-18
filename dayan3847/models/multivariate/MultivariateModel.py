import numpy as np

from dayan3847.models.Model import Model
from dayan3847.tools.functions import check_shape_vector, check_shape_vector_set


class MultivariateModel(Model):
    def __init__(self,
                 factors_x_dim: list[int],  # Number of factors per dimension, Ex. [5,5]
                 init_weights: float | None = None,  # None: random, float: init weights with this value
                 ):

        self.factors_x_dim: list[int] = factors_x_dim
        self.dim: int = len(factors_x_dim)  # Number of dimensions, Ex. 2
        if self.dim == 0:
            raise Exception('factors_x_dim must have at least one element')

        f: int = 1
        for _f in self.factors_x_dim:
            f *= _f

        super().__init__(f, init_weights)

    # Calculate the basis function a partir de un valor X
    # cada valor del vector es el resultado de aplicar la funcion base que se multiplica por cada peso
    def bb(self, x_: np.array) -> np.array:
        pass

    # Calculate the model value for a simple value

    def hi(self,
           x: np.array  # Expected shape: (x_dim)
           ) -> float:
        # Check shapes
        check_shape_vector(x, self.dim)
        bb: np.array = self.bb(x)
        # Check shapes
        check_shape_vector(bb, self.f)
        _r = self.ww @ bb
        return _r

    @staticmethod
    def activate(h: float) -> float:
        # return 1 / (1 + np.exp(-h))
        return h

    def gi(self, x_: np.array) -> float:
        x_ = np.array(x_)  # Convert to numpy array
        return self.activate(self.hi(x_))

    # Expected shape: (x_dim,N )
    def g(self, x_set: np.array) -> np.array:
        # Check shapes
        check_shape_vector_set(x_set, self.dim)
        r_set: list[float] = []
        for x in x_set:
            r = self.gi(x)
            r_set.append(r)
        r_set_np = np.array(r_set)  # Convert to numpy array
        check_shape_vector(r_set_np, x_set.shape[0])
        return r_set_np

    # Entrenar con un solo punto
    def train_single(self, x: np.array, y: float, a: float):
        x = np.array(x)
        g: float = self.gi(x)
        a_g__y: float = a * (g - y)
        bb: np.array = self.bb(x)
        check_shape_vector(bb, self.f)
        a_g__y_bb: np.array = a_g__y * bb
        self.ww -= a_g__y_bb

    def get_ww(self) -> np.array:
        return self.ww

    def set_ww(self, ww: np.array):
        check_shape_vector(ww, self.f)
        self.ww = ww
