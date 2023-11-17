import numpy as np
import matplotlib.pyplot as plt

from dayan3847.models.Model import Model
from dayan3847.models.multivariate.MultivariateGaussianModel import ModelGaussianMultivariate
from dayan3847.models.ModelTrainer import ModelTrainer

if __name__ == '__main__':
    np.random.seed(0)
    data: np.array = np.loadtxt('data_sin_2d.csv', delimiter=',')
    print("data.shape: {}".format(data.shape))
    model: Model = ModelGaussianMultivariate(a=.1,
                                             factors_x_dim=[7, 7],
                                             limits_x_dim=[(0, 1), (0, 1)],
                                             _s2=.01
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

    trainer: ModelTrainer = ModelTrainer(model,
                                         epochs_count=10,
                                         data_x=data[:, :2],
                                         data_y=data[:, 2],
                                         )
    trainer.train()

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
