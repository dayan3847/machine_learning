import numpy as np

from dayan3847.reinforcement_learning.case_1d import AgentPhysical, Environment


class AStatic(AgentPhysical):

    def __init__(self, env: Environment):
        super().__init__(env, 'target')
        self.color = (0, 255, 0)
        self.point = np.array([env.MAX[0] - 2, 0])
