from copy import deepcopy
from typing import List

import numpy as np

from Artificial import Artificial
from DataRepo import DataRepo
from Factor import Factor
from GrapherPlotly2D import GrapherPlotly2D
from GrapherPlotly3D import GrapherPlotly3D
from Polynomial import Polynomial


class BinaryClassificationThroughLogisticRegression:
    training_data: List[Artificial]
    polinomial: Polynomial
    polinomial_initial: Polynomial

    def __init__(self):
        # config
        #   Iterations Count
        self.iterations_count: int = 100
        #   Alpha
        self.a: float = .001
        #   Errors
        self.errors = []
        #   use sigmoid function
        self.use_sigmoid = True

        # init training data
        self.training_data = DataRepo.load_training_data()
        # init polinomial
        factors: List[Factor] = [
            Factor(),  # 1 (Independent Term)
            Factor(0, 1),  # x0^1 (Linear Term of x1)
            Factor(0, 2),  # x0^2 (Quadratic Term of x1)
            # Factor(0,3),  # x0^2 (Cubic Term of x1)
            Factor(1, 1)  # x1^1 (Linear Term of x2)
        ]
        thetas: List[float] = DataRepo.load_thetas(len(factors))
        self.polinomial = Polynomial(factors, thetas)
        # clone polinomial
        self.polinomial_initial = deepcopy(self.polinomial)

    # sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def plot_data(self):
        # Grapher Matplotlib
        # GrapherMatplotlib.plot_artificial_data_2d(self.training_data, clf=True, show=False)
        # GrapherMatplotlib.plot_polynomial_2d(polinomial, clf=False, show=True)
        # ax = GrapherMatplotlib.plot_artificial_data_3d(self.training_data, clf=True, show=False)
        # GrapherMatplotlib.plot_polynomial_3d(polinomial, clf=False, show=True, ax=ax)
        # Grapher Plotly 2D
        grapher_plotly2d: GrapherPlotly2D = GrapherPlotly2D()
        # grapher_plotly2d.plot_sigmoid_function()
        grapher_plotly2d.plot_artificial_data_2d(self.training_data)
        grapher_plotly2d.plot_polynomial(self.polinomial_initial, name='Initial', color='red')
        grapher_plotly2d.plot_polynomial(self.polinomial, name='Final', color='green', show=True)
        # grapher_plotly2d.plot_polynomial(self.errors, name='Final', color='red')
        # Grapher Plotly 3D
        grapher_plotly3d: GrapherPlotly3D = GrapherPlotly3D()
        grapher_plotly3d.plot_plane_y0()
        grapher_plotly3d.plot_artificial_data_3d(self.training_data)
        grapher_plotly3d.plot_polynomial(self.polinomial_initial, name='Initial', color='Reds')
        grapher_plotly3d.plot_polynomial(self.polinomial, name='Final', color='Greens', show=True)

    def error(self) -> float:
        error = 0
        for data in self.training_data:
            yi = data.y
            hi = self.polinomial.evaluate(data.x_vector)
            if self.use_sigmoid:
                hi = self.sigmoid(hi)
            error += (hi - yi) ** 2
        return error / 2

    def error_rms(self) -> float:
        e = self.error()
        m = len(self.training_data)
        return (2 * e / m) ** 0.5

    def main(self):
        self.iterations_count = 100
        self.a = .01
        self.errors.append(self.error_rms())

        for i in range(self.iterations_count):
            for data in self.training_data:
                yi = data.y
                xi_list = data.x_vector
                for j in range(len(self.polinomial.factors)):
                    # se evalua el polinomio con los thetas actuales
                    hi = self.polinomial.evaluate(xi_list)
                    if self.use_sigmoid:
                        hi = self.sigmoid(hi)
                    # se obtiene el factor y el valor de x correspondiente
                    factor_j = self.polinomial.factors[j]
                    xj = data.x_vector[factor_j.variable]
                    # se actualiza el theta
                    self.polinomial.thetas[j] += self.a * (yi - hi) * xj ** factor_j.degree
            self.errors.append(self.error_rms())

        # plot data
        self.plot_data()
        self.print_report()

    def print_report(self):
        print('Initial Polinomial')
        print(self.polinomial_initial)
        print('Final Polinomial')
        print(self.polinomial)
        print('Initial Error')
        print(self.errors[0])
        print('Final Error')
        print(self.errors[-1])


if __name__ == '__main__':
    controller = BinaryClassificationThroughLogisticRegression()
    controller.main()
