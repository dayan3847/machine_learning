import time
from abc import ABC, abstractmethod

import numpy as np

from dayan3847.reinforcement_learning.case_1d import Environment, AgentPhysical


class ADynamic(AgentPhysical, ABC):

    def __init__(self, env: Environment):
        super().__init__(env)
        self.point = np.array([0, 0])
        self.actions: np.ndarray = np.array([np.array([_x, 0]) for _x in range(-1, 2)])
        self.actions_count: int = len(self.actions)

    def run_callback(self):
        while self.running:
            time.sleep(self.env.TIME_STEP)
            # random choice action
            action: np.array = self.get_action()
            self.apply_action(action)

    @abstractmethod
    def get_action(self) -> int:
        pass

    def apply_action(self, a: int):
        action: np.array = self.actions[a]
        self.point = self.point + action
        for i in range(2):
            self.point[i] = max(-1 * self.env.MAX[i] + 1, self.point[i])
            self.point[i] = min(self.env.MAX[i] - 1, self.point[i])
