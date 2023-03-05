from typing import List
from DataSet import DataSet
from GrapherNetworkX import GrapherNetworkX
from Network import Network
from Perceptron import Perceptron

if __name__ == '__main__':
    features: List[int] = [1, 4, 7, 8, 10, -1, -4, -7, -8, -10]
    perceptron: Perceptron = Perceptron()
    weights: List[float] = DataSet.data1()
    perceptron.set_weights(weights)

    print(f'Σ (numpy) is {perceptron.net_input_numpy(features)}')
    print(f'Σ (teacher) is {perceptron.net_input_teacher(features)}')
    print(f'σ is {perceptron.propagate(features)}')

    generate_weights = perceptron.generate_weights(10)
    print(f'Generated weights {generate_weights}')

    network: Network = Network([10, 2, 2])
    print(f'Network {network.propagate(features)}')

    GrapherNetworkX.draw_network(network)
