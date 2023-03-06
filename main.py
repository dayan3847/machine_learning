import numpy as np
from typing import List

from src.net.Network import Network

if __name__ == '__main__':
    np.random.seed(123)

    features: List[int] = [1, 4, 7, 8, 10, -1, -4, -7, -8, -10]

    network: Network = Network([10, 2, 2])
    print(f'Network Output {network.propagate(features)}')
    network.draw()
