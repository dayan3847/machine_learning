from typing import List

from Perceptron import Perceptron


class Network:
    neurons: List[List[Perceptron]]

    def __init__(self, topology: List[int]):
        self.neurons = []
        for i in range(1, len(topology)):
            layer: List[Perceptron] = []
            for j in range(topology[i]):
                layer.append(Perceptron(topology[i - 1] + 1))
            self.neurons.append(layer)

    def propagate(self, features: List[float]) -> List[float]:
        result: List[float] = features
        for layer in self.neurons:
            result = [perceptron.propagate(result) for perceptron in layer]
        return result
