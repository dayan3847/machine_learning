import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    np.random.seed(0)
    N = 1000

    # Generate Data
    x_0 = np.random.uniform(0, 1, N)
    x_1 = np.random.uniform(0, 1, N)
    _e: np.array = np.random.rand(N) * .6 - .3
    y = np.sin(2 * np.pi * x_0) / 2 + np.sin(2 * np.pi * x_1) / 2 + _e
    data: np.array = np.array([x_0, x_1, y]).T

    np.savetxt('data_sin_2d.csv', data, delimiter=',')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], label='data', c='r', marker='o')
    ax.set_xlabel('x_0')
    ax.set_ylabel('x_1')
    ax.set_zlabel('y')
    ax.legend()

    plt.show()
