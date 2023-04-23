from ada_boost.entity import WeakClassifier, Data
from ada_boost.tools import Plotter


class AdaBoostMain:

    def __init__(self):
        self.plotter: Plotter = Plotter()
        self.data: Data = Data.load_data()

    def plot_data(self):
        self.plotter.plot_data(self.data)

    def search_best_classifier(self) -> WeakClassifier:
        features_count: int = len(self.data.x)
        classifier_optimo = None
        error_optimo = None
        for feature in range(features_count):
            for threshold in range(1, 100):
                classifier = WeakClassifier(feature, threshold)
                classifier_error = classifier.fix_polarity(self.data)
                if classifier_optimo is None or classifier_error < error_optimo:
                    classifier_optimo = classifier
                    error_optimo = classifier_error

        return classifier_optimo

    def update_weights(self, classifier: WeakClassifier):
        classification = classifier.classify(self.data.x)
        for i in range(self.data.count):
            if classification[i] != self.data.target[i]:
                self.data.weights[i] *= 2
        self.data.normalize_weights()