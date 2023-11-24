import numpy as np

from dayan3847.reinforcement_learning.deep_mind.agent.QLearningAgent import QLearningAgent, KnowledgeModel
from dayan3847.models.Model import Model
from dayan3847.models.multivariate.MultivariateGaussianModel import MultivariateGaussianModel
from dayan3847.models.functions import train_model


class KnowledgeModelGaussian(KnowledgeModel):

    def __init__(self, size_actions: int):
        self.models: list[Model] = [
            MultivariateGaussianModel(
                [
                    (-2, 2, 5),
                    (-1, 1, 5),
                    (-1, 1, 5),
                    (-2, 2, 5),
                    (-15, 15, 9),
                ],
                cov=np.array([
                    [.08, 0, 0, 0, 0],
                    [0, .02, 0, 0, 0],
                    [0, 0, .02, 0, 0],
                    [0, 0, 0, .08, 0],
                    [0, 0, 0, 0, 1],
                ]),
                init_weights=0,
            ) for _ in range(size_actions)
        ]

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
        train_model(self.models[a],
                    data_x=s,
                    data_y=q,
                    a=.1,
                    epochs_count=100,
                    error_threshold=1e-2,
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

    def __init__(self, action_count: int):
        self.knowledge_model: KnowledgeModelGaussian = KnowledgeModelGaussian(action_count)
        super().__init__(action_count, self.knowledge_model)
