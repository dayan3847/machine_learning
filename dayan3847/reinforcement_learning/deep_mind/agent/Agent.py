import numpy as np
from dm_env import TimeStep


class Agent:
    def __init__(self, action_values: np.array, ):
        self.action_values: np.array = action_values
        self.action_count: int = len(action_values)

    @staticmethod
    def get_state(time_step: TimeStep) -> np.array:
        position: np.array = time_step.observation['position']
        velocity: np.array = time_step.observation['velocity']
        state = np.concatenate((position, velocity))
        return state

    def select_an_action(self,
                         time_step: TimeStep,
                         a: int | None = None,
                         ) -> tuple[int, float, bool]:  # action, q, is_random
        pass

    def select_an_action_random(self) -> int:
        return np.random.randint(self.action_count)

    def save_knowledge(self, filepath: str):
        pass
