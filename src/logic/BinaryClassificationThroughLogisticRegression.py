import os
from datetime import datetime

import numpy as np
from copy import deepcopy
from typing import List
from src.models import Artificial
from src.models import Polynomial
from src.models import Factor
from src.repositories import DataRepo
from src.tools import GrapherPlotly2D
from src.tools import GrapherPlotly3D


class BinaryClassificationThroughLogisticRegression:
    training_data: List[Artificial]
    polinomial: Polynomial
    polinomial_initial: Polynomial

    def __init__(self):
        self.root = './../'
        self.path_reports: str = f'{self.root}reports/'
        # create sub folder with current date
        self.path_reports = f"{self.path_reports}{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
        os.mkdir(self.path_reports)
        self.path_files = f'{self.root}files/'
        # config
        #   Iterations Count
        self.iterations_count: int = 100
        #   Alpha
        self.a: float = .001
        #   Errors
        self.errors: List[float] = []
        #   use sigmoid function
        self.use_sigmoid = True

        self.data_repo: DataRepo = DataRepo(self.path_files)
        # init training data
        self.training_data = self.data_repo.load_training_data()
        # init polinomial
        factors: List[Factor] = [
            Factor(),  # 1 (Independent Term)
            Factor(0, 1),  # x0^1 (Linear Term of x1)
            Factor(0, 2),  # x0^2 (Quadratic Term of x1)
            # Factor(0,3),  # x0^2 (Cubic Term of x1)
            Factor(1, 1)  # x1^1 (Linear Term of x2)
        ]
        thetas: List[float] = self.data_repo.load_thetas(len(factors))
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
        grapher_plotly2d.plot_polynomial(self.polinomial, name='Final', color='green')
        grapher_plotly2d.plot_errors(self.errors)
        grapher_plotly2d.save(self.path_reports)
        # Grapher Plotly 3D
        grapher_plotly3d: GrapherPlotly3D = GrapherPlotly3D()
        grapher_plotly3d.plot_plane_y0()
        grapher_plotly3d.plot_artificial_data_3d(self.training_data)
        grapher_plotly3d.plot_polynomial(self.polinomial_initial, name='Initial', color='Reds')
        grapher_plotly3d.plot_polynomial(self.polinomial, name='Final', color='Greens')
        grapher_plotly3d.save(self.path_reports)

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
        self.save_report()

    def save_report(self):
        file = open(f'{self.path_reports}report.txt', 'w')
        file.write('Initial Polinomial\n')
        file.write(str(self.polinomial_initial) + '\n')
        file.write('Final Polinomial\n')
        file.write(str(self.polinomial) + '\n')
        file.write('Initial Error\n')
        file.write(str(self.errors[0]) + '\n')
        file.write('Final Error\n')
        file.write(str(self.errors[-1]))
        file.close()
