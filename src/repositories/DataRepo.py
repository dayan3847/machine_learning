import random
from typing import List
from src.models import Artificial


# Artificial Repository
class DataRepo:

    def __init__(self, path_files: str):
        self.path_files: str = path_files

    def load_data(self, file_name: str = 'data.txt') -> List[Artificial]:
        result: List[Artificial] = []
        file = open(self.path_files + file_name, 'r')
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
            theta_i = random.uniform(.01, .1)
            # cambiar el signo
            if 0 == random.randint(0, 1):
                theta_i *= -1
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

    def save_data(self, data: List[Artificial], file_name: str = 'data.txt'):
        file = open(self.path_files + file_name, 'w')
        file.truncate(0)
        for artificial in data:
            for x in artificial.x_vector:
                file.write(str(x) + ' ')
            file.write(str(artificial.y) + '\n')
        file.close()

    def load_training_data(self) -> List[Artificial]:
        return self.load_data('data_training.txt')

    def load_test_data(self) -> List[Artificial]:
        return self.load_data('data_test.txt')

    def load_validation_data(self) -> List[Artificial]:
        return self.load_data('data_validation.txt')

    def distribute_data(self):
        data: List[Artificial] = self.load_data()
        training_data: List[Artificial] = []
        test_data: List[Artificial] = []
        validation_data: List[Artificial] = []
        count_data: int = len(data)
        count_training_data: int = int(count_data * .7)
        count_validation_data: int = int(count_data * .2)

        # distribuir los datos aleatoriamente
        for i in range(count_training_data):
            index = random.randint(0, len(data) - 1)
            training_data.append(data[index])
            data.pop(index)
        for i in range(count_validation_data):
            index = random.randint(0, len(data) - 1)
            validation_data.append(data[index])
            data.pop(index)
        test_data = data

        self.save_data(training_data, 'data_training.txt')
        self.save_data(test_data, 'data_test.txt')
        self.save_data(validation_data, 'data_validation.txt')

        # distribuir datos si no existen los archivos

    def distribute_data_if_not_exists(self):
        try:
            self.load_training_data()
            self.load_test_data()
            self.load_validation_data()
        except FileNotFoundError:
            self.distribute_data()
