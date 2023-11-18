import numpy as np
from dm_control.rl.control import Environment

from dayan3847.reinforcement_learning.deep_mind.agent.Agent import Agent


class KnowledgeModel:
    def read_q_value(self,
                     s: np.array,  # state
                     a: int,  # action
                     ) -> float:
        pass

    def update_q_value(self,
                       s: np.array,  # state (normalmente seria el estado previo)
                       a: int,  # action
                       q: float,  # q_value
                       ):
        pass

    def save_knowledge(self, filepath: str):
        pass

    def load_knowledge(self, filepath: str):
        pass


class QLearningAgent(Agent):
    def __init__(self,
                 env: Environment,
                 action_count: int,
                 knowledge_model: KnowledgeModel,
                 ):
        super().__init__(env, action_count)
        # Action:
        # self.action_best_q: int = 0  # best_action for current state
        # q-learning
        self.alpha = .1
        self.gamma = 1
        self.epsilon = .1
        self.knowledge_model: KnowledgeModel = knowledge_model

        (self.time_step, self.state_pre, self.state_current, self.step) = self.init_episode()

    def select_an_action(self) -> tuple[int, float, bool]:  # action, q, is_random
        # Realizara una accion aleatoria con probabilidad epsilon
        return self.select_an_action_random() if np.random.random() < self.epsilon \
            else self.select_an_action_best_q()

    def select_an_action_best_q(self) -> tuple[int, float, bool]:  # best_action, best_q_value
        # Obtener la lista de valores Q de todas las acciones para el estado "s"
        q_values: list[float] = self.read_q_values_x_actions()
        best_q_value = max(q_values)
        best_action = q_values.index(best_q_value)
        print('max: a: {} q: {}'.format(best_action, best_q_value))
        return int(best_action), float(best_q_value), False

    def read_q_values_x_actions(self) -> list[float]:
        # s = self.state_current if s_curr else self.state_pre
        return [self.knowledge_model.read_q_value(self.state_current, a) for a in range(self.action_count)]

    def run_step(self) -> tuple[float, int, float, bool] | None:
        sp = super().run_step()
        if sp is None:
            return None
        r = sp[0]
        a = sp[1]
        q = sp[2]
        _is_random = sp[3]
        print("apply Action: ", '\033[91m' if _is_random else '\033[92m', a, '\033[00m', 'r: ', r)
        if _is_random:
            print('obteniendo q del estado previo')
            q = self.knowledge_model.read_q_value(self.state_pre, a)
        else:
            print('reuse q')
        print('obteniendo q maximo del estado actual')
        q_max: float = self.select_an_action_best_q()[1]

        self.train_action(a, r, q, q_max)
        return sp

    def train_action(self, a: int, r: float, q_prev: float, q_max: float):
        _q_fixed: float = q_prev + self.alpha * (r + self.gamma * q_max - q_prev)
        self.knowledge_model.update_q_value(self.state_pre, a, _q_fixed)

    def save_knowledge(self, filepath: str):
        self.knowledge_model.save_knowledge(filepath)
