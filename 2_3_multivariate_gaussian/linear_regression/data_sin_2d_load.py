import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data: np.array = np.loadtxt('data_sin_2d.csv', delimiter=',')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], label='data', c='r', marker='o')
    ax.set_xlabel('x_0')
    ax.set_ylabel('x_1')
    ax.set_zlabel('y')
    ax.legend()

    plt.show()
