import numpy as np
import matplotlib.pyplot as plt

from dayan3847.models.Model import Model
from dayan3847.models.multivariate.MultivariateGaussianModel import MultivariateGaussianModel
from dayan3847.models.functions import get_model_error, train_model

if __name__ == '__main__':
    np.random.seed(0)
    data: np.array = np.array([
        [.5, .5, 1],
        [.6, .6, -1],
    ])
    print("data.shape: {}".format(data.shape))
    model: Model = MultivariateGaussianModel(a=.1,
                                             factors_x_dim=[10, 10],
                                             limits_x_dim=[(0, 1), (0, 1)],
                                             _s2=.01,
                                             )
    er = get_model_error(model, data[:, :2], data[:, 2])

    fig = plt.figure()
    fig.suptitle('Incializacion del modelo (epoca 0), error: {}'.format(er))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], label='data', c='r', marker='o')
    model_plot = model.data_to_plot_matplotlib()
    ax.plot_trisurf(model_plot[:, 0], model_plot[:, 1], model_plot[:, 2], cmap='viridis', edgecolor='none')
    ax.set_xlabel('x_0')
    ax.set_ylabel('x_1')
    ax.set_zlabel('y')
    ax.legend()

    plt.show()

    error_history: list[float] = train_model(model,
                                             data_x=data[:, :2],
                                             data_y=data[:, 2],
                                             )

    fig = plt.figure()
    fig.suptitle('epoch 1 error: {}'.format(error_history[-1]))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], label='data', c='r', marker='o')
    model_plot = model.data_to_plot_matplotlib()
    ax.plot_trisurf(model_plot[:, 0], model_plot[:, 1], model_plot[:, 2], cmap='viridis', edgecolor='none')
    ax.set_xlabel('x_0')
    ax.set_ylabel('x_1')
    ax.set_zlabel('y')
    ax.legend()

    plt.show()

    error_history = train_model(model,
                                data_x=data[:, :2],
                                data_y=data[:, 2],
                                epochs_count=100,
                                error_threshold=1e-3,
                                )

    ep = len(error_history)
    er = error_history[-1]

    fig = plt.figure()
    fig.suptitle('epoch {} error: {}'.format(ep, er))
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
    ax.legend()
    plt.show()
