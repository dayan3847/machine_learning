import numpy as np

from dayan3847.basis_functions import gaussian_multivariate_2d
from dayan3847.tools import ShapeChecker


class Model:
    def __init__(self, a: float, factors_x_dim: tuple):
        self.a: float = a  # Learning rate
        self.factors_x_dim: tuple = factors_x_dim  # Number of factors per dimension, Ex. (5,5)
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
        self.weights_vfr: np.array = np.random.rand(self.f)[np.newaxis, :] - .5  # Weights

        # Check shapes, expected shape: (1, f)
        ShapeChecker.check_shape(self.weights_vfr, (1, self.f,))

    # Calculate the basis function a partir de un valor X
    # retorna un vector de shape(25,1,)
    # cada valor del vector es el resultado de aplicar la funcion base que se multiplica por cada peso
    # entrada esperada es un punto X (2,1) donde 2 es la cantidad de dimensiones de X
    def basefun_vfc(self, x_: np.array) -> np.array:
        ShapeChecker.check_shape(x_, (self.dim, 1))

    # Calculate the model value for a simple value

    # entrada esperada: Ex: shape(2,1,) donde 2 es la cantidad de dimensiones de X
    # Expected shape: (x_dim,)
    def hi(self, x_: np.array) -> float:
        # Check shapes
        ShapeChecker.check_shape(x_, (self.dim, 1))
        basefun_vfc: np.array = self.basefun_vfc(x_)
        # Check shapes
        ShapeChecker.check_shape(basefun_vfc, (self.f, 1))
        _r = self.weights_vfr @ basefun_vfc
        return float(_r[0, 0])

    @staticmethod
    def activate(h: float) -> float:
        # return 1 / (1 + np.exp(-h))
        return h

    # Expected shape: (x_dim,1)
    def gi(self, x_: np.array) -> float:
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
        for _x_1d in x_set_.T:
            _x = _x_1d[:, np.newaxis]
            _r_i = self.gi(_x)
            _r_set = np.append(_r_set, _r_i)
        ShapeChecker.check_shape(_r_set, (x_set_.shape[1],))
        return _r_set

    # Entrenar con un solo punto
    def update_w(self, x_1d, y: float):
        _x = x_1d[:, np.newaxis]
        _g: float = self.gi(_x)
        _diff: float = _g - y
        _a_diff: float = self.a * _diff
        _bf_vfr: np.array = self.basefun_vfc(_x)
        ShapeChecker.check_shape(_bf_vfr, (self.f, 1))
        _dw_vfc: np.array = _a_diff * _bf_vfr
        self.weights_vfr -= _dw_vfc.T

    def data_to_plot_plotly(self, num=20):
        pass


class ModelGaussian(Model):

    def __init__(self, a: float, factors_x_dim: tuple, limits_x_dim: tuple, _s2: float):
        super().__init__(a, factors_x_dim)
        self.limits_x_dim: tuple[tuple] = limits_x_dim
        if len(self.limits_x_dim) != self.dim:
            raise Exception('limits_x_dim must have the same number of elements as factors_x_dim')
        # covariance matrix identity
        self.cov: np.array = np.identity(self.dim) * _s2
        self.cov_inv: np.array = np.linalg.inv(self.cov)
        # Aqui almacenaremos todas las mu, cada mu es un vector columna de shape (self.dim,1)
        self.mm__: list[np.array] = self.get_mu_list()
        # Verify shapes
        ShapeChecker.check_shape(self.mm__[0], (self.dim, 1))
        ShapeChecker.check_shape(self.cov, (self.dim, self.dim))

    # Lista de vectores mu, cada vector mu es un vector columna de shape (self.dim,1)
    # TODO: en este momento solo funciona para 2 dimensiones
    def get_mu_list(self) -> list[np.array]:
        if self.dim != 2:
            raise Exception('get_mu_list only works for 2 dimensions at the moment')
        _r: list[np.array] = []
        _m_0: np.array = np.linspace(self.limits_x_dim[0][0], self.limits_x_dim[0][1], self.factors_x_dim[0])
        _m_1: np.array = np.linspace(self.limits_x_dim[1][0], self.limits_x_dim[1][1], self.factors_x_dim[1])
        _m_0, _m_1 = np.meshgrid(_m_0, _m_1)
        _m_0, _m_1 = _m_0.flatten(), _m_1.flatten()
        for _m0_i, _m1_i in zip(_m_0, _m_1):
            _r_i = np.array([_m0_i, _m1_i])[:, np.newaxis]
            _r.append(_r_i)

        return _r

    def basefun_vfc(self, x_: np.array) -> np.array:
        super().basefun_vfc(x_)
        result: np.array = np.array([])
        for _mu_i in self.mm__:
            _r_i = gaussian_multivariate_2d(x_, _mu_i, self.cov_inv)
            result = np.append(result, _r_i)
        result = result[:, np.newaxis]
        # Verify shapes
        ShapeChecker.check_shape(result, (self.f, 1))
        return result

    def data_to_plot_plotly(self, num=20):
        _x = np.linspace(self.limits_x_dim[0][0], self.limits_x_dim[0][1], num)
        _y = np.linspace(self.limits_x_dim[1][0], self.limits_x_dim[1][1], num)
        _z = np.empty((num, num))
        for _y_i in range(num):
            for _x_i in range(num):
                _x_vector = np.array([_x[_x_i], _y[_y_i]])[:, np.newaxis]
                _z[_y_i, _x_i] = self.gi(_x_vector)

        ShapeChecker.check_shape(_z, (num, num))
        return _x, _y, _z

    def data_to_plot_matplotlib(self, num=20):
        _x_0 = np.linspace(self.limits_x_dim[0][0], self.limits_x_dim[0][1], num)
        _x_1 = np.linspace(self.limits_x_dim[1][0], self.limits_x_dim[1][1], num)
        _x_0, _x_1 = np.meshgrid(_x_0, _x_1)
        _x_0, _x_1 = _x_0.flatten(), _x_1.flatten()
        _x: np.array = np.array([_x_0, _x_1])
        _y: np.array = self.g(_x)
        return _x, _y
