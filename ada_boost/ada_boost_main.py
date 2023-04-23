from typing import List

from ada_boost.entity.WeakClassifier import WeakClassifier
from ada_boost.main import AdaBoostMain

if __name__ == '__main__':
    ada_boost_main = AdaBoostMain()

    ada_boost_main.plot_data()
    ada_boost_main.plotter.show()

    ada_boost_main.data.normalize()
    ada_boost_main.plotter.clear()
    ada_boost_main.plot_data()
    ada_boost_main.plotter.show()

    classifiers: List[WeakClassifier] = []
    classifiers_count: int = 10
    for i in range(classifiers_count):
        best_classifier = ada_boost_main.search_best_classifier()
        ada_boost_main.plotter.plot_classifier(best_classifier)
        ada_boost_main.plotter.show()
        ada_boost_main.update_weights(best_classifier)
        ada_boost_main.plotter.clear()
        ada_boost_main.plot_data()
        for cla_anterior in classifiers:
            ada_boost_main.plotter.plot_classifier(cla_anterior, '--')
        ada_boost_main.plotter.plot_classifier(best_classifier)
        classifiers.append(best_classifier)
        ada_boost_main.plotter.show()
