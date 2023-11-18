import numpy as np
import matplotlib.pyplot as plt

from dayan3847.models.Model import Model
from dayan3847.models.multivariate.MultivariateGaussianModel import MultivariateGaussianModel
from dayan3847.models.functions import get_model_error, train_model

if __name__ == '__main__':
    np.random.seed(0)
    data: np.array = np.loadtxt('data_sin_2d.csv', delimiter=',')
    print("data.shape: {}".format(data.shape))

    model: Model = MultivariateGaussianModel(
        [
            (0, 1, 7),
            (0, 1, 7),
        ],
        cov=np.array([
            [.01, 0],
            [0, .01],
        ]),
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], label='data', c='r', marker='o')
    model_plot = model.data_to_plot_matplotlib()
    ax.plot_trisurf(model_plot[:, 0], model_plot[:, 1], model_plot[:, 2], cmap='viridis', edgecolor='none')
    ax.set_xlabel('x_0')
    ax.set_ylabel('x_1')
    ax.set_zlabel('y')
    ax.legend()

    plt.show()

    er = get_model_error(model, data[:, :2], data[:, 2])
    print('error: {}'.format(er))

    error_history = train_model(model,
                                data_x=data[:, :2],
                                data_y=data[:, 2],
                                a=.1,
                                epochs_count=10,
                                error_threshold=15,
                                )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], label='data', c='r', marker='o')
    model_plot = model.data_to_plot_matplotlib()
    ax.plot_trisurf(model_plot[:, 0], model_plot[:, 1], model_plot[:, 2], cmap='viridis', edgecolor='none')
    ax.set_xlabel('x_0')
    ax.set_ylabel('x_1')
    ax.set_zlabel('y')
    ax.legend()

    plt.show()

    # plot error history
    fig = plt.figure()
    fig.suptitle('Error history')
    ax = fig.add_subplot(111)
    ax.plot(error_history)
    ax.set_xlabel('epoch')
    ax.set_ylabel('error')
    plt.show()
