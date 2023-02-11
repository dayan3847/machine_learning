from typing import List
import matplotlib.pyplot as plt

from Artificial import Artificial


# Training Data (data unit)
class Grapher:
    # plot data in 3D
    @staticmethod
    def plot_artificial_data_3d(artificial: List[Artificial]):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for data in artificial:
            color: str = 'green' if data.y_data == 1 else 'red' if data.y_data == 0 else 'gray'
            ax.scatter(data.x_list_data[0], data.x_list_data[1], data.y_data, color=color, marker='o')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')
        plt.show()

    # plot data in 2D
    @staticmethod
    def plot_artificial_data_2d(artificial: List[Artificial]):
        plt.clf()
        plt.title('Data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axvline(color='black')
        plt.axhline(color='black')
        for data in artificial:
            color: str = 'green' if data.y_data == 1 else 'red' if data.y_data == 0 else 'gray'
            plt.scatter(data.x_list_data[0], data.x_list_data[1], color=color, marker='o')
        plt.grid()
        plt.show()
