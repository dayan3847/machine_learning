from ada_boost.entity import WeakClassifier, Data


class AdaBoostMain:

    def __init__(self, corpus_dir: str = './corpus'):
        self.data: Data = Data.load_data(corpus_dir)

    def search_best_classifier(self) -> WeakClassifier:
        classifier_optimo = None
        error_optimo = None
        features_count: int = len(self.data.x)
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
