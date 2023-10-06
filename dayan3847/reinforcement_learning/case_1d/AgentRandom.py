import numpy as np
from dayan3847.reinforcement_learning.case_1d import AgentDynamic


class AgentRandom(AgentDynamic):

    def get_action(self) -> np.array:
        actions: np.ndarray = self.env.get_actions_available(self)
        index: int = np.random.randint(0, len(actions))
        return actions[index]
