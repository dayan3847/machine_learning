import numpy as np

from typing import List

from dayan3847.ada_boost.src.Plotter import Plotter
from dayan3847.ada_boost.src.WeakClassifier import WeakClassifier
from dayan3847.tools.Normalizer import Normalizer


class AdaBoost:

    def __init__(self, data: np.ndarray):
        self.data: np.ndarray = data
        self.data_count: int = data.shape[1]
        self.classifiers: List[WeakClassifier] = []
        self.accuracy_history: np.ndarray = np.array([])

    def add_weight_row(self):
        weight_row: np.ndarray = np.ones((1, self.data_count)) / self.data_count
        self.data = np.vstack((self.data, weight_row))
        return self.data

    def normalize_data(self):
        self.data[0] = Normalizer.normalize_vector_points(self.data[0], 100)
        self.data[1] = Normalizer.normalize_vector_points(self.data[1], 100)
        return self.data

    def search_best_classifier(self) -> WeakClassifier:
        classifier_optimum = None
        error_optimum: float = 1.1
        features_count: int = 2
        for feature in range(features_count):
            for threshold in range(1, 100):
                classifier = WeakClassifier(feature, threshold)
                classifier_error = classifier.get_error(self.data)
                if classifier_error < error_optimum:
                    classifier_optimum = classifier
                    error_optimum = classifier_error

        return classifier_optimum

    def update_weights(self, classifier: WeakClassifier):
        classification = classifier.classify(self.data)
        for i in range(self.data_count):
            if classification[i] != self.data[2][i]:
                self.data[3][i] *= 2
        return self.normalize_weights()

    def normalize_weights(self):
        sum_weights = self.data[3].sum()
        self.data[3] = self.data[3] / sum_weights
        return self.data

    def add_classifier(self, plot: bool = False):
        best_classifier = self.search_best_classifier()
        self.update_weights(best_classifier)
        self.classifiers.append(best_classifier)
        self.accuracy_history = np.append(self.accuracy_history, self.get_accuracy(self.data))

        if plot:
            plotter: Plotter = Plotter(False)
            plotter.plot_data(self.data)
            plotter.plot_classifier(best_classifier)
            for cla_anterior in self.classifiers[:-1]:
                plotter.plot_classifier(cla_anterior, '--')
            plotter.show()

    def add_classifier_loop(self, count: int):
        for i in range(count):
            self.add_classifier()

    def classify(self, data: np.ndarray) -> np.ndarray:
        classification = np.zeros(data.shape[1])
        for classifier in self.classifiers:
            classification += classifier.get_alpha(self.data) * classifier.classify(data)
        return np.where(classification > 0, 1, -1)

    def get_accuracy(self, data: np.ndarray) -> float:
        classification = self.classify(data)
        return np.where(classification == data[2], 1, 0).sum() / data.shape[1]
