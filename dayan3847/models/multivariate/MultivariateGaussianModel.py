import numpy as np

from dayan3847.tools.gaussian import gaussian_multivariate
from dayan3847.models.multivariate.MultivariateModel import MultivariateModel
from dayan3847.tools.functions import check_shape


class MultivariateGaussianModel(MultivariateModel):

    def __init__(self,
                 gaussian_x_dim: list[
                     tuple[
                         float,  # min
                         float,  # max
                         int,  # number of gaussian
                     ]
                 ],
                 cov: np.array,  # covariance matrix
                 init_weights: float | None = None,
                 ):
        factors_x_dim: list[int] = [i[2] for i in gaussian_x_dim]
        super().__init__(factors_x_dim, init_weights)
        self.gaussian_x_dim: list[
            tuple[
                float,  # min
                float,  # max
                int,  # number of gaussian
            ]
        ] = gaussian_x_dim
        self.mm: np.array = self.get_mu_list()
        check_shape(self.mm, (self.f, self.dim))
        # # covariance matrix
        # cov: np.array = np.identity(self.dim)
        # for i in range(self.dim):
        #     cov[i, i] = gaussian_x_dim[i][3]
        check_shape(cov, (self.dim, self.dim))
        self.cov_inv: np.array = np.linalg.inv(cov)

    def get_mu_list(self) -> np.array:
        _m_list: list[np.array] = [np.linspace(*gaussian[:3]) for gaussian in self.gaussian_x_dim]
        _m_meshgrid = np.meshgrid(*_m_list)
        _m_flatten = [_mi.flatten() for _mi in _m_meshgrid]
        _m_flatten = np.array(_m_flatten).T
        return _m_flatten

    def bb(self, x_: np.array) -> np.array:
        x_ = np.array(x_)
        result_list: list[np.array] = []
        for m in self.mm:
            _r_i = gaussian_multivariate(x_, m, self.cov_inv)
            result_list.append(_r_i)
        return np.array(result_list)

    def data_to_plot_matplotlib(self, num=20) -> np.array:
        if self.dim != 2:
            raise Exception('plot only works for 2 dimensions')
        _x_0 = np.linspace(self.gaussian_x_dim[0][0], self.gaussian_x_dim[0][1], num)
        _x_1 = np.linspace(self.gaussian_x_dim[1][0], self.gaussian_x_dim[1][1], num)
        _x_0, _x_1 = np.meshgrid(_x_0, _x_1)
        _x_0, _x_1 = _x_0.flatten(), _x_1.flatten()
        _x: np.array = np.array([_x_0, _x_1]).T
        _y: np.array = self.g(_x)
        return np.array([_x_0, _x_1, _y]).T
