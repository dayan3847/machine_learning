import numpy as np
from dayan3847.stochastic_gradient_descent import TrainingDataGenerator

if __name__ == '__main__':
    _file_name = 'data_points.csv'
    _, f = TrainingDataGenerator.generate_data_points_sinusoidal(_file_name)

    xf = np.arange(0, 1, 0.01)
    yf = [f.subs('x', x) for x in xf]

    # load data points
    data_points: np.array = np.loadtxt(_file_name, delimiter=',').T

    # plot data points with matplotlib
    import matplotlib.pyplot as plt

    # Data
    plt.clf()
    plt.title('Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axvline(color='black')
    plt.axhline(color='black')
    # points
    plt.scatter(data_points[0], data_points[1], color='gray', label='data points')
    # function
    plt.plot(xf, yf, color='blue', label='function')
    plt.legend()
    plt.grid()
    plt.show()
