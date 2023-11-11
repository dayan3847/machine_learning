import matplotlib.pyplot as plt
import numpy as np

from dayan3847.ada_boost.src import WeakClassifier


class Plotter:
    def __init__(self, subplots: bool = True):
        if subplots:
            self.fig, self.ax = plt.subplots()
        else:
            self.fig = plt
            self.ax = plt

    def clear(self):
        self.ax.clear()

    def show(self):
        self.ax.title('Data')
        self.ax.xlabel('x')
        self.ax.ylabel('y')
        self.ax.axvline(color='black')
        self.ax.axhline(color='black')
        self.ax.grid()
        self.ax.legend()
        self.fig.show()

    def plot_data(self, data_points: np.ndarray):
        index_positive = np.where(data_points[2].astype(int) > 0)
        index_negative = np.where(data_points[2].astype(int) <= 0)

        # Data Points
        self.ax.scatter(data_points[0][index_positive], data_points[1][index_positive], color='green', label='positive',
                        marker='o', s=data_points[3][index_positive] * 5000)
        self.ax.scatter(data_points[0][index_negative], data_points[1][index_negative], color='red', label='negative',
                        marker='o', s=data_points[3][index_negative] * 5000)

    def plot_classifier(self, classifier: WeakClassifier, line_style: str = '-'):
        color = 'green' if classifier.polarity else 'red'
        if classifier.feature == 0:
            self.ax.axvline(x=classifier.threshold, color=color, linestyle=line_style)
        elif classifier.feature == 1:
            self.ax.axhline(y=classifier.threshold, color=color, linestyle=line_style)
        else:
            raise Exception('Feature not supported')
