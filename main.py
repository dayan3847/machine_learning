from typing import List

import numpy as np


class Perceptron:
    my_weights: List[float]

    def __init__(self, weights_count: int = 0):
        self.weights = self.generate_weights(weights_count) if 0 < weights_count else []

    def set_weights(self, weights: List[float]):
        self.weights = weights

    def net_input_numpy(self, features: List[float]) -> float:
        # b is weights[0], features[0] is 1
        features_copy: List[float] = features.copy()
        features_copy.insert(0, 1)
        return float(np.dot(features_copy, self.weights))

    def net_input_teacher(self, features: List[float]) -> float:
        result: float = self.weights[0]
        for i in range(1, len(features)):
            result += features[i - 1] * self.weights[i]
        return result

    @staticmethod
    def activation_function(x: float) -> float:  # sigmoid
        return 1 / (1 + np.exp(-x))

    def propagate(self, features: List[float]) -> float:
        sigma: float = self.net_input_numpy(features)
        return self.activation_function(sigma)

    @staticmethod
    def generate_weights(count: int) -> List[float]:
        return [np.random.uniform(-.5, .5) for _ in range(count)]


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


class DataSet:

    @staticmethod
    def data1() -> List[float]:
        return [
            0.14699916,
            -0.04568567,
            0.39091784,
            0.1823034,
            -0.48700912,
            -0.11667783,
            0.34573132,
            0.00219967,
            0.42239386,
            0.12168889,
            -0.16521032
        ]


if __name__ == '__main__':
    my_features = [1, 4, 7, 8, 10, -1, -4, -7, -8, -10]
    my_weights = DataSet.data1()
    my_perceptron: Perceptron = Perceptron()
    my_perceptron.set_weights(my_weights)

    print(f'Σ (numpy) is {my_perceptron.net_input_numpy(my_features)}')
    print(f'Σ (teacher) is {my_perceptron.net_input_teacher(my_features)}')
    print(f'σ is {my_perceptron.propagate(my_features)}')

    generate_weights = Perceptron.generate_weights(10)
    print(f'Generated weights {generate_weights}')

    network: Network = Network([10, 2, 2])
    print(f'Network {network.propagate(my_features)}')
