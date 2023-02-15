import os
from datetime import datetime

import numpy as np
from copy import deepcopy
from typing import List
from src.models import Artificial
from src.models import Polynomial
from src.models import Factor
from src.repositories import DataRepo
from src.tools import GrapherPlotlyData3D, GrapherPlotlyData2D, GrapherPlotlyErrors2D
from src.tools.GrapherPlotlyRoc2D import GrapherPlotlyRoc2D


class LogisticRegression:
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
        os.system(f'cp {self.path_files}data.txt {self.path_reports}')

        # ROC
        self.tp: List[int] = []
        self.tn: List[int] = []
        self.fp: List[int] = []
        self.fn: List[int] = []
        self.fpr: List[float] = []
        self.tpr: List[float] = []

        # config
        #   Iterations Count
        self.iterations_count: int = 1000
        #   Alpha
        self.a: float = .1
        #   Errors
        self.errors: List[float] = []
        #   use sigmoid function
        self.use_sigmoid = True

        self.data_repo: DataRepo = DataRepo(self.path_files)
        self.data_repo.distribute_data_if_not_exists()
        # init training data
        self.training_data = self.data_repo.load_training_data()
        self.validation_data = self.data_repo.load_validation_data()
        self.test_data = self.data_repo.load_test_data()
        # init polinomial
        factors: List[Factor] = [
            Factor(),  # 1 (Independent Term)
            Factor(0, 1),  # x0^1 (Linear Term of x1)
            Factor(0, 2),  # x0^2 (Quadratic Term of x1)
            Factor(0, 3),  # x0^3 (Cubic Term of x1)
            # Factor(0, 4),  # x0^4 (Quartic Term of x1)
            Factor(1, 1),  # x1^1 (Linear Term of x2)
            # Factor(1, 2),  # x1^2 (Quadratic Term of x2)
            # Factor(1, 3),  # x1^3 (Cubic Term of x2)
            # Factor(1, 4),  # x1^4 (Quartic Term of x2)
        ]
        thetas: List[float] = self.data_repo.load_thetas(len(factors))
        os.system(f'cp {self.path_files}thetas.txt {self.path_reports}')
        self.polinomial = Polynomial(factors, thetas)
        # clone polinomial
        self.polinomial_initial = deepcopy(self.polinomial)

    # sigmoid function
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def plot_data(self):
        # Grapher Matplotlib
        # GrapherMatplotlib.plot_artificial_data_2d(self.training_data, clf=True, show=False)
        # GrapherMatplotlib.plot_polynomial_2d(polinomial, clf=False, show=True)
        # ax = GrapherMatplotlib.plot_artificial_data_3d(self.training_data, clf=True, show=False)
        # GrapherMatplotlib.plot_polynomial_3d(polinomial, clf=False, show=True, ax=ax)

        # Grapher Plotly Data 3D
        grapher_plotly3d: GrapherPlotlyData3D = GrapherPlotlyData3D()
        grapher_plotly3d.plot_plane_y(.5)
        grapher_plotly3d.plot_artificial_data(self.training_data, name='Training Data', color='blue')
        grapher_plotly3d.plot_artificial_data(self.validation_data, name='Validation Data', color='orange')
        grapher_plotly3d.plot_artificial_data(self.test_data, name='Test Data', color='green')
        grapher_plotly3d.plot_polynomial(self.polinomial_initial, name='Initial', color='Reds')
        grapher_plotly3d.plot_polynomial(self.polinomial, name='Final', color='Greens')
        grapher_plotly3d.save(self.path_reports)
        # Grapher Plotly Data 2D
        grapher_plotly2d: GrapherPlotlyData2D = GrapherPlotlyData2D()
        grapher_plotly2d.plot_artificial_data(self.training_data, name='Training Data', color='blue')
        grapher_plotly2d.plot_artificial_data(self.validation_data, name='Validation Data', color='orange')
        grapher_plotly2d.plot_artificial_data(self.test_data, name='Test Data', color='green')
        grapher_plotly2d.plot_polynomial(self.polinomial_initial, name='Initial', color='red')
        # grapher_plotly2d.plot_polynomial(self.polinomial, name='Final 1', color='yellow', y=1)
        grapher_plotly2d.plot_polynomial(self.polinomial, name='Final', color='green')
        # grapher_plotly2d.plot_polynomial(self.polinomial, name='Final 0', color='violet', y=0)
        grapher_plotly2d.save(self.path_reports)
        # Grapher Plotly Errors 2D
        grapher_plotly2d_errors: GrapherPlotlyErrors2D = GrapherPlotlyErrors2D()
        grapher_plotly2d_errors.plot_errors(self.errors)
        grapher_plotly2d_errors.save(self.path_reports)
        grapher_plotly2d_errors.plot_sigmoid_function()
        grapher_plotly_roc: GrapherPlotlyRoc2D = GrapherPlotlyRoc2D()
        self.calculate_roc()
        grapher_plotly_roc.plot_roc(self.fpr, self.tpr)

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
        return (2 * e / m) ** .5

    def main(self):
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
            self.calculate_sensitivity_and_specificity()

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
        file.write(str(self.errors[-1]) + '\n')
        file.write('Iterations Count\n')
        file.write(str(self.iterations_count) + '\n')
        file.write('Alpha\n')
        file.write(str(self.a) + '\n')
        file.close()

    def calculate_sensitivity_and_specificity(self):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for data in self.validation_data:
            yi = data.y
            xi_list = data.x_vector
            hi = self.polinomial.evaluate(xi_list)
            if self.use_sigmoid:
                hi = self.sigmoid(hi)
            if yi == 1:
                if hi >= .5:
                    tp += 1
                else:
                    fn += 1
            else:
                if hi >= .5:
                    fp += 1
                else:
                    tn += 1
        self.tp.append(tp)
        self.fp.append(fp)
        self.tn.append(tn)
        self.fn.append(fn)

    def calculate_roc(self):
        for i in range(len(self.tp)):
            self.fpr.append(self.fp[i] / (self.fp[i] + self.tn[i]))
            self.tpr.append(self.tp[i] / (self.tp[i] + self.fn[i]))
