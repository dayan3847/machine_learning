from typing import List

from Artificial import Artificial
from ArtificialRepo import ArtificialRepo
from Factor import Factor
from Grapher import Grapher
from Polynomial import Polynomial


class BinaryClassificationThroughLogisticRegression:

    def __init__(self):
        self.training_data: List[Artificial] = []

    def init(self):
        self.load_training_data()

    # Generate Data Points
    def load_training_data(self):
        if not self.training_data:
            self.training_data = ArtificialRepo.load_training_data()

    def main(self, plot: bool = False):
        self.init()

        polinomial: Polynomial = Polynomial(
            [
                Factor(),  # x1^0 (Independent Term)
                Factor(1),  # x1^1 (Linear Term of x1)
                Factor(1, 1)  # x2^1 (Linear Term of x2)
            ]
        )
        polinomial.init_thetas((-0.01, 0.01))

        Grapher.plot_artificial_data_2d(self.training_data, clf=True, show=False)
        Grapher.plot_polynomial_2d(polinomial, clf=False, show=True)

        Grapher.plot_artificial_data_3d(self.training_data)
        # Grapher.plot_polynomial_3d(polinomial)


if __name__ == '__main__':
    controller = BinaryClassificationThroughLogisticRegression()
    controller.main()
