from typing import List

from Artificial import Artificial
from ArtificialRepo import ArtificialRepo
from Factor import Factor
from GrapherMatplotlib import GrapherMatplotlib
from GrapherPlotly import GrapherPlotly
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

    def main(self):
        self.init()

        polinomial: Polynomial = Polynomial(
            [
                Factor(),  # x1^0 (Independent Term)
                Factor(1),  # x1^1 (Linear Term of x1)
                # Factor(2),  # x1^2 (Quadratic Term of x1)
                # Factor(3),  # x1^2 (Cubic Term of x1)
                Factor(1, 1)  # x2^1 (Linear Term of x2)
            ]
        )
        polinomial.init_thetas((-0.01, 0.01))
        # Grapher Matplotlib
        GrapherMatplotlib.plot_artificial_data_2d(self.training_data, clf=True, show=False)
        GrapherMatplotlib.plot_polynomial_2d(polinomial, clf=False, show=True)
        ax = GrapherMatplotlib.plot_artificial_data_3d(self.training_data, clf=True, show=False)
        GrapherMatplotlib.plot_polynomial_3d(polinomial, clf=False, show=True, ax=ax)
        # Grapher Plotly
        fig = GrapherPlotly.plot_artificial_data_2d(self.training_data)
        GrapherPlotly.plot_polynomial_2d(polinomial, fig=fig)
        fig = GrapherPlotly.plot_artificial_data_3d(self.training_data)
        GrapherPlotly.plot_polynomial_3d(polinomial, fig=fig)


if __name__ == '__main__':
    controller = BinaryClassificationThroughLogisticRegression()
    controller.main()
