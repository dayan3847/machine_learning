from typing import List
import matplotlib.pyplot as plt
import numpy as np

from Artificial import Artificial
from Polynomial import Polynomial


# Training Data (data unit)
class Grapher:
    # plot data in 3D
    @staticmethod
    def plot_artificial_data_3d(artificial: List[Artificial], clf: bool = True, show: bool = True):
        if clf:
            plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for data in artificial:
            color: str = 'red' if data.y_data == 1 else 'blue' if data.y_data == 0 else 'gray'
            ax.scatter(data.x_list_data[0], data.x_list_data[1], data.y_data, color=color, marker='o')
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('y')
        if show:
            plt.show()

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
            color: str = 'red' if data.y_data == 1 else 'blue' if data.y_data == 0 else 'gray'
            plt.scatter(data.x_list_data[0], data.x_list_data[1], color=color, marker='o')
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
