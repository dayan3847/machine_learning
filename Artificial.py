from typing import List


# Training Data (data unit)
class Artificial:
    # valores de las variables independientes
    x_list_data: List[float]
    # valor de la variable dependiente
    y_data: float

    # init
    def __init__(self, x_list_data: List[float], y_data: float = 0):
        self.x_list_data = x_list_data
        self.y_data = y_data
