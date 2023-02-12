from typing import List
import matplotlib.pyplot as plt
import numpy as np

from Artificial import Artificial
from Polynomial import Polynomial


# Training Data (data unit)
class GrapherMatplotlib:

    # plot data in 2D
    @staticmethod
    def plot_artificial_data_2d(artificial: List[Artificial], clf: bool = True, show: bool = True):
        if clf:
            plt.clf()
        plt.title('Data')
        plt.xlabel('x0')
        plt.ylabel('x1')
        plt.axvline(color='black')
        plt.axhline(color='black')
        for data in artificial:
            color: str = 'red' if data.y == 1 else 'blue' if data.y == 0 else 'gray'
            plt.scatter(data.x_vector[0], data.x_vector[1], color=color, marker='o')
        plt.grid()
        if show:
            plt.show()

    @staticmethod
    def plot_polynomial_2d(polinomial: Polynomial, clf: bool = True, show: bool = True):
        if polinomial.get_variables_count() > 2:
            return
        if polinomial.get_last_variable_degree() > 1:
            return
        if clf:
            plt.clf()
        plt.title('Polynomial')
        plt.xlabel('x0')
        plt.ylabel('x1')
        plt.axvline(color='black')
        plt.axhline(color='black')
        x0_list = np.arange(-3, 3, 0.01)
        x1_list = []
        for x0 in x0_list:
            x1 = polinomial.evaluate_despejando([x0], 1)
            x1_list.append(x1)
        plt.plot(x0_list, x1_list, color='orange', label='initial')
        plt.grid()
        if show:
            plt.show()

    # plot data in 3D
    @staticmethod
    def plot_artificial_data_3d(artificial: List[Artificial], clf: bool = True, show: bool = True, ax=None):
        if clf:
            plt.clf()
        fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111, projection='3d')
        for data in artificial:
            color: str = 'red' if data.y == 1 else 'blue' if data.y == 0 else 'gray'
            ax.scatter3D(data.x_vector[0], data.x_vector[1], data.y, c=color, marker='o')
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('y')
        if show:
            plt.show()
        return ax

    @staticmethod
    def plot_polynomial_3d(polinomial: Polynomial, clf: bool = True, show: bool = True, ax=None):
        if polinomial.get_variables_count() > 3:
            return
        if clf:
            plt.clf()
        fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111, projection='3d')
        x0_list = np.arange(-3, 3, .25)
        x1_list = np.arange(-7, 3, .25)
        x0_list, x1_list = np.meshgrid(x0_list, x1_list)
        y_list = []
        for x0 in x0_list:
            for x1 in x1_list:
                y = polinomial.evaluate([x0, x1])
                y_list.append(y)

        # ax.plot_surface(x0_list, x1_list, y_list, color='orange', label='initial')
        # ax.scatter(x0_list[0], x1_list[0], y_list[0], color='green', marker='o')
        # ax.plot3D(x0_list, x1_list, y_list, color='orange', label='initial')
        # ax.plot3D(xline, yline, zline, 'gray')
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('y')
        if show:
            plt.show()

        return ax
