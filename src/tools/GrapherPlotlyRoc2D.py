from typing import List

import plotly.graph_objs as go
from src.tools import GrapherPlotly


class GrapherPlotlyRoc2D(GrapherPlotly):

    def file_name(self) -> str:
        return 'roc_2d_data.html'

    def __init__(self):
        super().__init__()
        self.figure.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )

    def plot_roc(self, fpr: List[float], tpr: List[float]):
        self.figure.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name='ROC',
                marker_color='green',
            )
        )
        # add axis lines
        self.figure.add_trace(
            go.Scatter(
                x=[-1, 1],
                y=[0, 0],
                mode='lines',
                name='x',
                marker_color='gray'
            )
        )
        self.figure.add_trace(
            go.Scatter(
                x=[0, 0],
                y=[-.01, 1.01],
                mode='lines',
                name='y',
                marker_color='gray'
            )
        )
        # add diagonal line en lineas rojas discontinuas
        self.figure.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='diagonal',
                marker_color='red',
                line=dict(
                    dash='dash',
                )
            )
        )
        self.figure.show()
