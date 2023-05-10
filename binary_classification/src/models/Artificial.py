from typing import List


# Training Data (data unit)
class Artificial:
    # vector x (valores de las variables independientes)
    x_vector: List[float]
    # escalar y (valor de la variable dependiente)
    y: float

    def __init__(self, x_vector: List[float], y: float = 0):
        self.x_vector = x_vector
        self.y = y
