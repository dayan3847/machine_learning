import numpy as np
import matplotlib.pyplot as plt

from dayan3847.ada_boost.src.AdaBoost import AdaBoost
from dayan3847.ada_boost.src.Plotter import Plotter

if __name__ == '__main__':
    data_t: np.ndarray = np.loadtxt('../data/dataCircle_fix.txt', delimiter=' ')
    ada_boost = AdaBoost(data_t.T)
    print(ada_boost.data)

    print(ada_boost.add_weight_row())

    print(ada_boost.data.shape)
    plotter: Plotter = Plotter(False)
    plotter.plot_data(ada_boost.data)
    plotter.show()
    print(ada_boost.data.shape)

    print(ada_boost.normalize_data())

    plotter.plot_data(ada_boost.data)
    plotter.show()

    ada_boost.add_classifier(True)
    ada_boost.add_classifier_loop(100)
    ada_boost.add_classifier(True)

    plt.clf()
    plt.title('Accuracy History')
    plt.plot(ada_boost.accuracy_history)
    plt.show()
