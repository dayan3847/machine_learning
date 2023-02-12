import random
from typing import List

from Artificial import Artificial


# Artificial Repository
class DataRepo:

    @staticmethod
    def load_training_data() -> List[Artificial]:
        result: List[Artificial] = []
        file = open('data.txt', 'r')
        for line in file:
            line = line.strip()
            data: List[str] = line.split()
            x_list_data: List[float] = []
            for x in data[:-1]:
                x_list_data.append(float(x))
            y_data: float = float(data[-1])
            result.append(Artificial(x_list_data, y_data))
        file.close()
        return result

    @staticmethod
    def load_thetas(count: int) -> List[float]:
        result: List[float] = DataRepo.read_thetas()
        if len(result) == count:
            return result
        result = []
        for i in range(count):
            # generar una theta aleatoria entre -0.01 y 0.01 diferente de 0
            theta_i = 0
            while 0 == theta_i:
                theta_i = random.uniform(-0.01, 0.01)
            result.append(theta_i)
        DataRepo.save_thetas(result)
        return result

    @staticmethod
    def save_thetas(thetas: List[float]):
        file = open('thetas.txt', 'w')
        file.truncate(0)  # clear file
        for theta in thetas:
            file.write(str(theta) + '\n')
        file.close()

    @staticmethod
    def read_thetas() -> List[float]:
        result: List[float] = []
        try:
            file = open('thetas.txt', 'r')
            lines = file.readlines()
            for line in lines:
                result.append(float(line))
            file.close()
        except FileNotFoundError:
            pass
        return result
