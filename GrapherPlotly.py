from typing import List
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go

from Artificial import Artificial
from Polynomial import Polynomial


# Training Data (data unit)
class GrapherPlotly:

    # plot data in 2D
    @staticmethod
    def plot_artificial_data_2d(artificial: List[Artificial], show: bool = True, fig=None):
        x0_list0 = []
        x1_list0 = []
        x0_list1 = []
        x1_list1 = []
        for data in artificial:
            if data.y_data == 0:
                x0_list0.append(data.x_list_data[0])
                x1_list0.append(data.x_list_data[1])
            else:
                x0_list1.append(data.x_list_data[0])
                x1_list1.append(data.x_list_data[1])
        if fig is None:
            fig = go.Figure()
        fig.add_trace(go.Scatter(x=x0_list0, y=x1_list0, mode='markers', name='0', marker_color='blue'))
        fig.add_trace(go.Scatter(x=x0_list1, y=x1_list1, mode='markers', name='1', marker_color='red'))
        if show:
            fig.show()
        return fig

    @staticmethod
    def plot_polynomial_2d(polinomial: Polynomial, show: bool = True, fig=None):
        if polinomial.get_variables_count() > 2:
            return
        if polinomial.get_last_variable_degree() > 1:
            return
        # x0_list = np.arange(-3, 3, 0.01)
        x0_list = np.linspace(-3, 3, 100)
        x1_list = []
        for x0 in x0_list:
            x1 = polinomial.evaluate_despejando([x0], 1)
            x1_list.append(x1)
        if fig is None:
            fig = go.Figure()
        fig.add_trace(go.Scatter(x=x0_list, y=x1_list, mode='lines', name='Polinomial', marker_color='green'))
        if show:
            fig.show()
        return fig

    # plot data in 3D
    @staticmethod
    def plot_artificial_data_3d(artificial: List[Artificial], show: bool = True, fig=None):

        x0_list = []
        x1_list = []
        y_list = []
        for data in artificial:
            x0_list.append(data.x_list_data[0])
            x1_list.append(data.x_list_data[1])
            y_list.append(data.y_data)
        if fig is None:
            fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=x0_list, y=x1_list, z=y_list, mode='markers',
                                   marker=dict(size=12, color=y_list, colorscale='Viridis', opacity=0.8)))
        if show:
            fig.show()
        return fig

    @staticmethod
    def plot_polynomial_3d(polinomial: Polynomial, show: bool = True, fig=None):
        if polinomial.get_variables_count() > 3:
            return
        x0 = np.linspace(-3, 3, 50)
        x1 = np.linspace(-7, 3, 50)
        x0, x1 = np.meshgrid(x0, x1)
        y = polinomial.evaluate([x0, x1])
        if fig is None:
            fig = go.Figure()
        fig.add_trace(go.Surface(z=y, x=x0, y=x1, colorscale='Viridis', opacity=0.8))
        if show:
            fig.show()
        return fig
