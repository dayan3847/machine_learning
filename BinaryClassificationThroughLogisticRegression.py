from typing import List

import numpy as np

from Artificial import Artificial
from DataRepo import DataRepo
from Factor import Factor
from GrapherMatplotlib import GrapherMatplotlib
from GrapherPlotly import GrapherPlotly
from GrapherPlotly2D import GrapherPlotly2D
from GrapherPlotly3D import GrapherPlotly3D
from Polynomial import Polynomial


class BinaryClassificationThroughLogisticRegression:

    def __init__(self):
        self.training_data: List[Artificial] = []

    def main(self):
        if not self.training_data:
            self.training_data = DataRepo.load_training_data()

        factors: List[Factor] = [
            Factor(),  # 1 (Independent Term)
            Factor(0, 1),  # x0^1 (Linear Term of x1)
            Factor(0, 2),  # x0^2 (Quadratic Term of x1)
            # Factor(0,3),  # x0^2 (Cubic Term of x1)
            Factor(1, 1)  # x1^1 (Linear Term of x2)
        ]
        thetas: List[float] = DataRepo.load_thetas(len(factors))
        polinomial: Polynomial = Polynomial(factors, thetas)
        # Grapher Matplotlib
        # GrapherMatplotlib.plot_artificial_data_2d(self.training_data, clf=True, show=False)
        # GrapherMatplotlib.plot_polynomial_2d(polinomial, clf=False, show=True)
        # ax = GrapherMatplotlib.plot_artificial_data_3d(self.training_data, clf=True, show=False)
        # GrapherMatplotlib.plot_polynomial_3d(polinomial, clf=False, show=True, ax=ax)
        # Grapher Plotly
        grapher_plotly2d: GrapherPlotly2D = GrapherPlotly2D()
        # grapher_plotly2d.plot_sigmoid_function()
        grapher_plotly2d.plot_artificial_data_2d(self.training_data)
        grapher_plotly2d.plot_polynomial_2d(polinomial, show=True)
        grapher_plotly3d: GrapherPlotly3D = GrapherPlotly3D()
        grapher_plotly3d.plot_plane_y0()
        grapher_plotly3d.plot_artificial_data_3d(self.training_data)
        grapher_plotly3d.plot_polynomial_3d(polinomial)
        grapher_plotly3d.plot_polynomial_projection_3d(polinomial, show=True)

    # def sigmoid function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


if __name__ == '__main__':
    controller = BinaryClassificationThroughLogisticRegression()
    controller.main()
