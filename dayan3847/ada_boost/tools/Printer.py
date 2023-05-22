from dayan3847.ada_boost.entity import WeakClassifier, Data


class Printer:

    @staticmethod
    def print_classifier(classifier: WeakClassifier, data: Data):
        eje = 'x' if classifier.feature == 0 else 'y'
        signo = '>' if classifier.polarity else '<'
        print(f'{eje} {signo} {classifier.threshold}')
        print(f'Error: {classifier.error}')
        print(f'Alpha: {classifier.get_alpha(data)}')
