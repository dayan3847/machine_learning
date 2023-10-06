import numpy as np

from dayan3847.reinforcement_learning.case_1d import Environment, AgentDynamic


class AgentRandom(AgentDynamic):

    def __init__(self, env: Environment):
        super().__init__(env)
        self.color = (255, 0, 0)
        self.point = np.array([0, 0])

    def get_action(self) -> int:
        return np.random.randint(0, self.actions_count)
