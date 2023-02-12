import random
from typing import List
from src.models import Artificial


# Artificial Repository
class DataRepo:

    def __init__(self, path_files: str):
        self.path_files = path_files

    def load_training_data(self) -> List[Artificial]:
        result: List[Artificial] = []
        file = open(f'{self.path_files}data.txt', 'r')
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

    def load_thetas(self, count: int) -> List[float]:
        result: List[float] = self.read_thetas()
        if len(result) == count:
            return result
        result = []
        for i in range(count):
            # generar una theta aleatoria entre -0.01 y 0.01 diferente de 0
            theta_i = 0
            while 0 == theta_i:
                theta_i = random.uniform(-0.01, 0.01)
            result.append(theta_i)
        self.save_thetas(result)
        return result

    def save_thetas(self, thetas: List[float]):
        file = open(f'{self.path_files}thetas.txt', 'w')
        file.truncate(0)  # clear file
        for theta in thetas:
            file.write(str(theta) + '\n')
        file.close()

    def read_thetas(self) -> List[float]:
        result: List[float] = []
        try:
            file = open(f'{self.path_files}thetas.txt', 'r')
            lines = file.readlines()
            for line in lines:
                result.append(float(line))
            file.close()
        except FileNotFoundError:
            pass
        return result
