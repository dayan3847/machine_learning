from typing import Callable

import numpy as np
from networkx import DiGraph

from dayan3847.multi_layer_perceptron.src.net.Neuron import Neuron


class NeuronPerceptron(Neuron):
    activation_function: Callable[[float], float]
    weight_bias: float

    def __init__(self, graph: DiGraph, weight_bias: float):
        super().__init__(graph)
        self.weight_bias = weight_bias
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    def get_value(self) -> float:
        if self._value is None:
            self._value = self.weight_bias
            in_edges = self._graph.in_edges(self, data=True)
            for in_edge in in_edges:
                prev_neuron: Neuron = in_edge[0]
                prev_neuron_value: float = prev_neuron.get_value()
                in_edge_weight: float = in_edge[2]['weight']
                self._value += prev_neuron_value * in_edge_weight

        self._value = self.activation_function(self._value)

        return self._value
