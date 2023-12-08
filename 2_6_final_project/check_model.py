import numpy as np
import matplotlib.pyplot as plt

from dayan3847.models.Model import Model
from dayan3847.models.multivariate.MultivariateGaussianModel import MultivariateGaussianModel

if __name__ == '__main__':
    np.random.seed(0)

    # gaussian_x_dim: list[
    #     tuple[
    #         float,  # min
    #         float,  # max
    #         int,  # number of gaussian
    #     ]
    # ] = [
    #     (-2, 2, 5),
    #     (-1, 1, 3),
    #     (-1, 1, 3),
    #     (-2, 2, 5),
    #     (-15, 15, 7),
    # ]
    # cov: np.array = np.array([
    #     [.1, 0, 0, 0, 0],
    #     [0, .1, 0, 0, 0],
    #     [0, 0, .1, 0, 0],
    #     [0, 0, 0, .1, 0],
    #     [0, 0, 0, 0, .1],
    # ])

    gaussian_x_dim: list[
        tuple[
            float,  # min
            float,  # max
            int,  # number of gaussian
        ]
    ] = [
        (-2, 2, 5),
        (-1, 1, 5),
        (-1, 1, 5),
        (-2, 2, 5),
        (-15, 15, 9),
    ]
    cov: np.array = np.array([
        [.08, 0, 0, 0, 0],
        [0, .02, 0, 0, 0],
        [0, 0, .02, 0, 0],
        [0, 0, 0, .08, 0],
        [0, 0, 0, 0, 1],
    ])

    model: Model = MultivariateGaussianModel(
        gaussian_x_dim,
        cov=cov,
        init_weights=1,
    )
    dim_count = len(gaussian_x_dim)

    for i, g in enumerate(gaussian_x_dim):
        x = np.linspace(g[0], g[1], 100)[:, np.newaxis]
        if i != 0:
            x = np.hstack((np.zeros((x.shape[0], i)), x))
        if i != dim_count - 1:
            x = np.hstack((x, np.zeros((x.shape[0], dim_count - i - 1))))

        y = model.g(x)

        fig = plt.figure()
        fig.suptitle('Gaussian Dim: {}'.format(i))
        ax = fig.add_subplot(111)
        ax.plot(x[:, i], y)
        ax.set_ylim(bottom=0)
        plt.show()
