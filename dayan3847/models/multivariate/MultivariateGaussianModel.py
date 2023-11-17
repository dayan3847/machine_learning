import numpy as np

from dayan3847.tools.gaussian import gaussian_multivariate_2d
from dayan3847.tools.ShapeChecker import ShapeChecker
from dayan3847.models.multivariate.MultivariateModel import MultivariateModel


class MultivariateGaussianModel(MultivariateModel):

    def __init__(self,
                 a: float,
                 factors_x_dim: list[int],
                 limits_x_dim: list[tuple[int, int]],
                 _s2: float,
                 init_weights_random: bool = True
                 ):
        super().__init__(a, factors_x_dim, init_weights_random)
        self.limits_x_dim: list[tuple[int, int]] = limits_x_dim
        if len(self.limits_x_dim) != self.dim:
            raise Exception('limits_x_dim must have the same number of elements as factors_x_dim')
        # covariance matrix identity
        self.cov: np.array = np.identity(self.dim) * _s2
        self.cov_inv: np.array = np.linalg.inv(self.cov)
        # Aqui almacenaremos todas las mu, cada mu es un vector columna de shape (self.dim,1)
        self.mm: list[np.array] = self.get_mu_list()
        # Verify shapes
        ShapeChecker.check_shape(self.mm[0], (self.dim, 1))
        ShapeChecker.check_shape(self.cov, (self.dim, self.dim))

    def get_mu_list(self) -> list[np.array]:
        _r: list[np.array] = []
        _m: list[np.array] = []
        for _i_lxd, _i_fxd in zip(self.limits_x_dim, self.factors_x_dim):
            _m.append(np.linspace(_i_lxd[0], _i_lxd[1], _i_fxd))
        _m = np.meshgrid(*_m)
        _m = [_mi.flatten() for _mi in _m]
        for _m_i in zip(*_m):
            _r_i = np.array(_m_i)[:, np.newaxis]
            _r.append(_r_i)

        return _r

    def bb(self, x_: np.array) -> np.array:
        x_ = np.array(x_)
        super().bb(x_)
        result: np.array = np.array([])
        for _mu_i in self.mm:
            _r_i = gaussian_multivariate_2d(x_, _mu_i, self.cov_inv)
            result = np.append(result, _r_i)
        result = result[:, np.newaxis]
        # Verify shapes
        ShapeChecker.check_shape(result, (self.f, 1))
        return result

    def data_to_plot_plotly(self, num=20):
        if self.dim != 2:
            raise Exception('plot only works for 2 dimensions')
        _x = np.linspace(self.limits_x_dim[0][0], self.limits_x_dim[0][1], num)
        _y = np.linspace(self.limits_x_dim[1][0], self.limits_x_dim[1][1], num)
        _z = np.empty((num, num))
        for _y_i in range(num):
            for _x_i in range(num):
                _x_vector = np.array([_x[_x_i], _y[_y_i]])[:, np.newaxis]
                _z[_y_i, _x_i] = self.gi(_x_vector)

        ShapeChecker.check_shape(_z, (num, num))
        return _x, _y, _z

    def data_to_plot_matplotlib(self, num=20) -> np.array:
        if self.dim != 2:
            raise Exception('plot only works for 2 dimensions')
        _x_0 = np.linspace(self.limits_x_dim[0][0], self.limits_x_dim[0][1], num)
        _x_1 = np.linspace(self.limits_x_dim[1][0], self.limits_x_dim[1][1], num)
        _x_0, _x_1 = np.meshgrid(_x_0, _x_1)
        _x_0, _x_1 = _x_0.flatten(), _x_1.flatten()
        _x: np.array = np.array([_x_0, _x_1]).T
        _y: np.array = self.g(_x)
        return np.array([_x_0, _x_1, _y]).T
