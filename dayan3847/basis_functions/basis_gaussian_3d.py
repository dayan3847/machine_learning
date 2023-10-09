import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from dayan3847.tools import ShapeChecker, PlotterData

np.random.seed(0)


class BasisFunction3d:

    def basis_function(self, x_: np.array) -> np.array:
        ShapeChecker.check_shape_point(x_)

    @staticmethod
    def activation_function(y: np.array) -> np.array:
        return 1 / (1 + np.exp(-1 * y))


class BasisGaussian3d(BasisFunction3d):

    # mu_: np.array: column vector (2, 1)
    def __init__(self, mu: np.array = np.zeros(2)[:, np.newaxis], cov: np.array = np.identity(2)):
        ShapeChecker.check_shape(mu, (2, 1))
        ShapeChecker.check_shape(cov, (2, 2,))
        self.mu: np.array = mu
        self.cov: np.array = cov

    def plot(self, fig_: plt.figure, pos: int, pos_act: int, w_: float = 1.) -> plt.axes:
        _s = 3
        _x_plot: np.array = PlotterData.get_x_plot_2d()
        _y_plot: np.array = self.basis_function(_x_plot)
        _y_plot_h: np.array = w_ * _y_plot
        _y_plot_g: np.array = self.activation_function(_y_plot_h)

        _ax: list[plt.axes] = [
            fig_.add_subplot(pos, projection='3d'),
            fig_.add_subplot(pos_act, projection='3d'),
        ]
        for _ax_i in _ax:
            _ax_i.set_xlim(-_s, _s)
            _ax_i.set_ylim(-_s, _s)
            _ax_i.set_zlim(0, 1)
            _ax_i.set_xlabel('x_0')
            _ax_i.set_ylabel('x_1')
            _ax_i.set_zlabel('y')

        _ax[0].set_title('{} w={}'.format(self.__class__.__name__, w_))
        _ax[0].plot_trisurf(_x_plot[0], _x_plot[1], _y_plot_h, cmap='viridis', edgecolor='none')

        _ax[1].set_title('{} w={} activation'.format(self.__class__.__name__, w_))
        _ax[1].plot_trisurf(_x_plot[0], _x_plot[1], _y_plot_g, cmap='viridis', edgecolor='none')


class BasisGaussian3dMultivariateNormal(BasisGaussian3d):
    def __init__(self, mu_: np.array = np.zeros(2)[:, np.newaxis], cov: np.array = np.identity(2)):
        super().__init__(mu_, cov)
        _mu_f = mu_.flatten()
        ShapeChecker.check_shape(_mu_f, (2,))
        self.multivariate_normal = multivariate_normal(mean=_mu_f, cov=self.cov)

    def basis_function(self, x_: np.array) -> np.array:
        super().basis_function(x_)
        _x_point = np.dstack(x_)
        return self.multivariate_normal.pdf(_x_point)


class BasisGaussian3dVictorUc(BasisGaussian3d):
    def basis_function_x_i(self, x_i_: np.array) -> np.array:
        ShapeChecker.check_shape(x_i_, (2, 1))
        # x - mu_
        _x_mu = x_i_ - self.mu
        # (x - mu_)T
        _x_mu_t = _x_mu.T
        # inv(cov)
        _cov_inv = np.linalg.inv(self.cov)
        _r_i: np.array = np.exp(-.5 * _x_mu_t @ _cov_inv @ _x_mu)
        return _r_i

    def basis_function(self, x_: np.array) -> np.array:
        super().basis_function(x_)
        _r: np.array = np.array([])
        for _x_i in x_.T:
            _x_i_c = _x_i[:, np.newaxis]
            _r_i = self.basis_function_x_i(_x_i_c)
            _r = np.append(_r, _r_i)
        return _r


class BasisGaussian3dDayanBravo(BasisGaussian3d):
    def basis_function(self, x_: np.array) -> np.array:
        super().basis_function(x_)
        _r: np.array = (
                (np.exp(-1 * (x_[0] ** 2) / 2))
                * (np.exp(-1 * (x_[1] ** 2) / 2))
        )

        return _r


if '__main__' == __name__:
    fig = plt.figure(figsize=(25, 10))

    g1 = BasisGaussian3dMultivariateNormal()
    g1.plot(fig, 231, 234, 6)

    g2 = BasisGaussian3dVictorUc()
    g2.plot(fig, 232, 235, 1)

    g3 = BasisGaussian3dDayanBravo()
    g3.plot(fig, 233, 236, 1)

    plt.tight_layout()
    plt.show()
