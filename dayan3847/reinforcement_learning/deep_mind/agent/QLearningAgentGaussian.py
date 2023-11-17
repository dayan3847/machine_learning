import numpy as np
from dm_control.rl.control import Environment

from dayan3847.reinforcement_learning.deep_mind.agent.QLearningAgent import QLearningAgent, KnowledgeModel
from dayan3847.models.Model import Model
from dayan3847.models.multivariate.MultivariateGaussianModel import MultivariateGaussianModel


class KnowledgeModelGaussian(KnowledgeModel):

    def __init__(self, size_actions: int):
        self.models: list[Model] = [
            MultivariateGaussianModel(
                .1,
                [5, 3, 3, 5, 7],
                [(-2, 2), (-1, 1), (-1, 1), (-2, 2), (-15, 15)],
                .1,
            ) for _ in range(size_actions)
        ]

    def read_q_value(self,
                     s: np.array,  # state
                     a: int,  # action
                     ) -> float:
        print('read s', s)
        q = self.models[a].gi(s)
        return float(q)

    def update_q_value(self,
                       s: np.array,  # state (normalmente seria el estado previo)
                       a: int,  # action
                       q: float,  # q_value
                       ):
        print('fix s', s)
        self.models[a].update_w(s, q)

    def save_knowledge(self, filepath: str):
        w_data: np.array = np.array([
            mi.get_ww() for mi in self.models
        ])
        np.savetxt(filepath, w_data.T, delimiter=',')

    def load_knowledge(self, filepath: str):
        w_data: np.array = np.loadtxt(filepath, delimiter=',').T
        for i, mi in enumerate(self.models):
            mi.set_ww(w_data[i].T)


class QLearningAgentGaussian(QLearningAgent):

    def __init__(self,
                 env: Environment,
                 action_count: int,
                 ):
        self.knowledge_model: KnowledgeModelGaussian = KnowledgeModelGaussian(action_count)
        super().__init__(env, action_count, self.knowledge_model)

    def update_current_state(self) -> np.array:
        position: np.array = self.time_step.observation['position']
        velocity: np.array = self.time_step.observation['velocity']
        self.state_current = np.concatenate((position, velocity))
        return self.state_current
