import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
N = 1000


def generate_data_3d():
    global N
    x_0 = np.random.uniform(0, 1, N)
    x_1 = np.random.uniform(0, 1, N)
    _e: np.array = np.random.rand(N) * .6 - .3
    y = np.sin(2 * np.pi * x_0) / 2 + np.sin(2 * np.pi * x_1) / 2 + _e
    return np.array([x_0, x_1, y]).T


def plot_data_3d(_data: np.array):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(_data[:, 0], _data[:, 1], _data[:, 2], label='data', c='r', marker='o')
    ax.set_xlabel('x_0')
    ax.set_ylabel('x_1')
    ax.set_zlabel('y')
    ax.legend()

    plt.show()


if __name__ == '__main__':
    data: np.array = generate_data_3d()
    np.savetxt('data_sin_2d.csv', data, delimiter=',')
    plot_data_3d(data)
