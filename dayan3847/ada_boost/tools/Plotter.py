import matplotlib.pyplot as plt

from dayan3847.ada_boost.entity import Data, WeakClassifier


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
        self.fig.show()

    def plot_data(self, data: Data):
        self.ax.scatter(data.x[0], data.x[1], c=data.target, s=data.weights * 1000)

    def plot_classifier(self, classifier: WeakClassifier, line_style: str = '-'):
        color = 'green' if classifier.polarity else 'red'
        if classifier.feature == 0:
            self.ax.axvline(x=classifier.threshold, color=color, linestyle=line_style)
        elif classifier.feature == 1:
            self.ax.axhline(y=classifier.threshold, color=color, linestyle=line_style)
        else:
            raise Exception('Feature not supported')
