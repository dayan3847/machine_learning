import numpy as np

from dayan3847.models.Model import Model
from dayan3847.tools.ShapeChecker import ShapeChecker


class MultivariateModel(Model):
    def __init__(self,
                 a: float,
                 factors_x_dim: list[int],
                 init_weights_random: bool = True):
        self.a: float = a  # Learning rate
        self.factors_x_dim: list[int] = factors_x_dim  # Number of factors per dimension, Ex. [5,5]
        self.dim: int = len(factors_x_dim)  # Number of dimensions, Ex. 2

        if self.dim == 0:
            raise Exception('factors_x_dim must have at least one element')

        self.f: int = 1
        for _f in self.factors_x_dim:
            self.f *= _f

        if self.f == 0:
            raise Exception('factors_x_dim can not have zero elements')

        # Weights: Factors Row Vector Ex. shape(1,25,)
        # este es el vector de todos los pesos, es un vector fila
        self.ww: np.array = np.random.rand(self.f)[np.newaxis, :] - .5 if init_weights_random \
            else np.zeros(self.f)[np.newaxis, :]

        # Check shapes, expected shape: (1, f)
        ShapeChecker.check_shape(self.ww, (1, self.f,))

    # Calculate the basis function a partir de un valor X
    # retorna un vector de shape(25,1,)
    # cada valor del vector es el resultado de aplicar la funcion base que se multiplica por cada peso
    # entrada esperada es un punto X (2,1) donde 2 es la cantidad de dimensiones de X
    def bb(self, x_: np.array) -> np.array:
        ShapeChecker.check_shape(x_, (self.dim, 1))

    # Calculate the model value for a simple value

    # entrada esperada: Ex: shape(2,1,) donde 2 es la cantidad de dimensiones de X
    # Expected shape: (x_dim,)
    def hi(self, x_: np.array) -> float:
        # Check shapes
        ShapeChecker.check_shape(x_, (self.dim, 1))
        bb: np.array = self.bb(x_)
        # Check shapes
        ShapeChecker.check_shape(bb, (self.f, 1))
        _r = self.ww @ bb
        return float(_r[0, 0])

    @staticmethod
    def activate(h: float) -> float:
        # return 1 / (1 + np.exp(-h))
        return h

    # Expected shape: (x_dim,1)
    def gi(self, x_: np.array) -> float:
        x_ = np.array(x_)
        if x_.shape == (self.dim,):
            x_ = x_[:, np.newaxis]
        # Check shapes
        ShapeChecker.check_shape(x_, (self.dim, 1))
        return self.activate(self.hi(x_))

    # Expected shape: (x_dim,N )
    def g(self, x_set_: np.array) -> np.array:
        # Check shapes
        ShapeChecker.check_shape_point_set(x_set_, self.dim)
        _r_set: np.array = np.array([])
        for _x_1d in x_set_:
            _x = _x_1d[:, np.newaxis]
            _r_i = self.gi(_x)
            _r_set = np.append(_r_set, _r_i)
        ShapeChecker.check_shape(_r_set, (x_set_.shape[0],))
        return _r_set

    # Entrenar con un solo punto
    def update_w_single(self, x: np.array, y: float):
        x = np.array(x)
        _x = x[:, np.newaxis]
        _g: float = self.gi(_x)
        _diff: float = _g - y
        _a_diff: float = self.a * _diff
        _bf_vfr: np.array = self.bb(_x)
        ShapeChecker.check_shape(_bf_vfr, (self.f, 1))
        _dw_vfc: np.array = _a_diff * _bf_vfr
        self.ww -= _dw_vfc.T
