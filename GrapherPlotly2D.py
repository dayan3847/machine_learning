from typing import List
import numpy as np
import plotly.graph_objs as go
from plotly.graph_objs import Figure

from Artificial import Artificial
from GrapherPlotly import GrapherPlotly
from Polynomial import Polynomial


class GrapherPlotly2D(GrapherPlotly):

    def __init__(self):
        super().__init__()
        self.figure_2d_data: Figure = go.Figure()
        self.figure_2d_data.update_layout(
            xaxis_title="x0",
            yaxis_title="x1",
            xaxis_range=self.area_to_pot[0],
            yaxis_range=self.area_to_pot[1],
        )
        self.figure_2d_data.add_trace(
            go.Scatter(
                x=[-10, 10],
                y=[0, 0],
                mode='lines',
                name='x',
                marker_color='gray'
            )
        )
        self.figure_2d_data.add_trace(
            go.Scatter(
                x=[0, 0],
                y=[-10, 10],
                mode='lines',
                name='y',
                marker_color='gray',
            )
        )

        self.figure_2d_error: Figure = go.Figure()
        self.figure_2d_error.update_layout(
            xaxis_title="iteration",
            yaxis_title="error",
        )

    def plot_artificial_data_2d(self, artificial: List[Artificial], show: bool = True):
        data_to_print: dict = {}
        for data in artificial:
            if data.y_data not in data_to_print:
                data_to_print[data.y_data] = {'x0': [], 'x1': []}
            data_to_print[data.y_data]['x0'].append(data.x_list_data[0])
            data_to_print[data.y_data]['x1'].append(data.x_list_data[1])
        for key in data_to_print:
            color: str = 'blue' if key == 0 else 'red' if key == 1 else 'gray'
            self.figure_2d_data.add_trace(
                go.Scatter(
                    x=data_to_print[key]['x0'],
                    y=data_to_print[key]['x1'],
                    mode='markers',
                    name=str(key),
                    marker_color=color,
                )
            )
        if show:
            self.figure_2d_data.show()

    def plot_polynomial_2d(self, polinomial: Polynomial, show: bool = True):
        if polinomial.get_variables_count() > 2:
            return
        if polinomial.get_last_variable_degree() > 1:
            return
        x0_list = np.linspace(-3, 3, 100)
        x1_list = []
        for x0 in x0_list:
            x1 = polinomial.evaluate_despejando([x0], 1)
            x1_list.append(x1)
        self.figure_2d_data.add_trace(
            go.Scatter(
                x=x0_list,
                y=x1_list,
                mode='lines',
                name='Initial',
                marker_color='orange',
            )
        )
        if show:
            self.figure_2d_data.show()
