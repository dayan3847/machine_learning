from typing import List

from Artificial import Artificial


# Artificial Repository
class ArtificialRepo:

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
