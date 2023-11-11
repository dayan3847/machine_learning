import numpy as np
import matplotlib.pyplot as plt


def plot_data_3d(_data: np.array):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(_data[0], _data[1], _data[2], label='data', c='r', marker='o')
    ax.set_xlabel('x_0')
    ax.set_ylabel('x_1')
    ax.set_zlabel('y')
    ax.legend()

    plt.show()


if __name__ == '__main__':
    data: np.array = np.loadtxt('data.csv', delimiter=',').T
    plot_data_3d(data)
