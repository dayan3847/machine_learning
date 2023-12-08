import numpy as np


class Agent:
    def __init__(self, action_count: int):
        self.action_count: int = action_count

    def select_an_action(self,
                         s: np.array,  # State
                         a: int | None = None,
                         ) -> int:
        pass

    def train_action(self, s: np.array, r_prev: float):
        pass

    def select_an_action_random(self) -> int:
        return np.random.randint(self.action_count)

    def save_knowledge(self, filepath: str):
        pass
