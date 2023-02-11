from typing import List

from Artificial import Artificial
from ArtificialRepo import ArtificialRepo
from Grapher import Grapher


class BinaryClassificationThroughLogisticRegression:

    def __init__(self):
        self.training_data: List[Artificial] = []

    def init(self):
        self.load_training_data()
        Grapher.plot_artificial_data_3d(self.training_data)
        Grapher.plot_artificial_data_2d(self.training_data)

    # Generate Data Points
    def load_training_data(self):
        if not self.training_data:
            self.training_data = ArtificialRepo.load_training_data()

    def main(self, plot: bool = False):
        self.init()


if __name__ == '__main__':
    controller = BinaryClassificationThroughLogisticRegression()
    controller.main()
