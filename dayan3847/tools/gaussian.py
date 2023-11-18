import numpy as np

from dayan3847.tools.ShapeChecker import ShapeChecker


def gaussian_multivariate(
        x_i_: np.array,  # point (2, 1)
        mu_: np.array,  # mean shape (2, 1)
        cov_inv_: np.array,  # inverse of covariance matrix shape (2, 2)
) -> float:
    # x - mu_
    _x_mu = x_i_ - mu_
    # (x - mu_)T
    # _x_mu_t = _x_mu.T
    # inv(cov)
    _x_mu_t_cov_inv_x_mu = _x_mu @ cov_inv_ @ _x_mu
    # ShapeChecker.check_shape(_x_mu_t_cov_inv_x_mu, (1, 1))
    _val = float(_x_mu_t_cov_inv_x_mu)
    _r_i: np.array = np.exp(-.5 * _val)
    return _r_i
