import numpy as np

from dayan3847.reinforcement_learning.agent.TemporalDifferenceLearningAgent import QLearningAgent, KnowledgeModel
from dayan3847.models.Model import Model
from dayan3847.models.functions import train_model


class KnowledgeModelGaussian(KnowledgeModel):

    def __init__(self, size_actions: int, models: list[Model]):
        self.size_actions: int = size_actions
        self.models: list[Model] = models

    def reset_knowledge(self):
        for mi in self.models:
            mi.set_ww(np.zeros_like(mi.get_ww()))

    def read_q_value(self,
                     s: np.array,  # state
                     a: int,  # action
                     ) -> float:
        q = self.models[a].g(s)
        print('read: a:{} s:{} q:{}'.format(a, s, q))
        return float(q)

    def update_q_value(self,
                       s: np.array,  # state (normalmente seria el estado previo)
                       a: int,  # action
                       q: float,  # q_value
                       ):
        print('\033[92m', f'update: a:{a} s:{s} q:{q}', '\033[0m')
        train_model(self.models[a],
                    data_x=s,
                    data_y=q,
                    a=.3,
                    epochs_count=1000,
                    error_threshold=.001,
                    )

        print('fix: a:{} s:{} q:{}'.format(a, s, q))

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

    def __init__(self, action_count: int, models: list[Model]):
        super().__init__(
            action_count,
            KnowledgeModelGaussian(
                action_count,
                models
            )
        )
        self.algorithm_name: str = 'q_learning_gaussian'
