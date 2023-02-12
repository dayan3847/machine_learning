import numpy as np
import plotly.graph_objs as go
from typing import List
from plotly.graph_objs import Figure
from src.tools import GrapherPlotly


class GrapherPlotlyErrors2D(GrapherPlotly):

    def __init__(self):
        super().__init__()
        self.figure.update_layout(
            xaxis_title="iteration",
            yaxis_title="error",
        )

    def plot_errors(self, errors: List[float]):
        self.figure.add_trace(
            go.Scatter(
                x=list(range(len(errors))),
                y=errors,
                mode='lines',
                name='error',
                marker_color='red',
            )
        )
        # add axis lines
        self.figure.add_trace(
            go.Scatter(
                x=[-1, len(errors) - 1],
                y=[0, 0],
                mode='lines',
                name='x',
                marker_color='gray'
            )
        )
        self.figure.add_trace(
            go.Scatter(
                x=[0, 0],
                y=[-.01, max(errors) + .1],
                mode='lines',
                name='y',
                marker_color='gray'
            )
        )

    def file_name(self) -> str:
        return 'figure_2d_error.html'

    @staticmethod
    def plot_sigmoid_function():
        the_range = (-20, 20)
        x0_list = np.linspace(the_range[0], the_range[1], 100)
        x1_list = []
        for x0 in x0_list:
            x1 = 1 / (1 + np.exp(-x0))
            x1_list.append(x1)

        figure_sigmoid: Figure = go.Figure()
        figure_sigmoid.update_layout(
            xaxis_title="x",
            yaxis_title="sigmoid(x)",
            xaxis_range=(the_range[0], the_range[1]),
            yaxis_range=(-.1, 1.1),
        )
        figure_sigmoid.add_trace(
            go.Scatter(
                x=x0_list,
                y=x1_list,
                mode='lines',
                name='Sigmoid',
                marker_color='green',
            )
        )
        # add horizontal lines for 0 and 1
        figure_sigmoid.add_trace(
            go.Scatter(
                x=[the_range[0], the_range[1]],
                y=[0, 0],
                mode='lines',
                name='0',
                marker_color='gray'
            )
        )
        figure_sigmoid.add_trace(
            go.Scatter(
                x=[the_range[0], the_range[1]],
                y=[1, 1],
                mode='lines',
                name='1',
                marker_color='gray'
            )
        )
        # add vertical line for 0
        figure_sigmoid.add_trace(
            go.Scatter(
                x=[0, 0],
                y=[-.1, 1.1],
                mode='lines',
                name='v0',
                marker_color='gray'
            )
        )
        # add center point (0, 0.5)
        figure_sigmoid.add_trace(
            go.Scatter(
                x=[0],
                y=[0.5],
                mode='markers',
                name='center',
                marker_color='black'
            )
        )
        figure_sigmoid.show()
