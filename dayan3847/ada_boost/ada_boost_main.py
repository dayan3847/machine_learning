from typing import List

from dayan3847.ada_boost.entity import WeakClassifier
from ada_boost.main import AdaBoostMain
from ada_boost.tools import Plotter

if __name__ == '__main__':
    plotter: Plotter = Plotter()
    ada_boost_main = AdaBoostMain()

    plotter.plot_data(ada_boost_main.data)
    plotter.show()

    ada_boost_main.data.normalize()
    plotter.clear()
    plotter.plot_data(ada_boost_main.data)
    plotter.show()

    classifiers: List[WeakClassifier] = []
    classifiers_count: int = 10
    for i in range(classifiers_count):
        best_classifier = ada_boost_main.search_best_classifier()
        plotter.plot_classifier(best_classifier)
        plotter.show()
        ada_boost_main.update_weights(best_classifier)
        plotter.clear()
        plotter.plot_data(ada_boost_main.data)
        for cla_anterior in classifiers:
            plotter.plot_classifier(cla_anterior, '--')
        plotter.plot_classifier(best_classifier)
        classifiers.append(best_classifier)
        plotter.show()
