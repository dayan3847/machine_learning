from networkx import DiGraph

from src.net.Neuron import Neuron


class NeuronInput(Neuron):

    def __init__(self, graph: DiGraph, value: float = None):
        super().__init__(graph)
        self._value = value

    def get_value(self) -> float:
        if self._value is None:
            raise Exception("Input node has no value")
        return self._value
