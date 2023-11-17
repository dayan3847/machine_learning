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
        self.action_best_q: int = 0  # best_action for current state
        # q-learning
        self.alpha = .1
        self.gamma = 1
        self.epsilon = .1
        self.knowledge_model: KnowledgeModel = knowledge_model

        (self.time_step, self.state_pre, self.state_current, self.step) = self.init_episode()

    def select_an_action(self) -> tuple[int, bool]:  # action, is_random
        # Realizara una accion aleatoria con probabilidad epsilon
        return (self.select_an_action_random(), True) if np.random.random() < self.epsilon \
            else (self.action_best_q, False)

    def select_an_action_best_q(self) -> tuple[int, float]:  # best_action, best_q_value
        # Obtener la lista de valores Q de todas las acciones para el estado "s"
        q_values_per_action = self.read_q_values_x_actions()
        # De la lista de valores Q, buscar el mejor
        best_action = q_values_per_action[0][0]
        best_q_value = q_values_per_action[0][1]
        for av in q_values_per_action[1:]:
            a: int = av[0]  # action
            v: float = av[1]  # q_value
            if v > best_q_value:
                best_action = a
                best_q_value = v
        return int(best_action), float(best_q_value)

    def read_q_values_x_actions(self) -> list[tuple[int, float]]:
        return [(a, self.knowledge_model.read_q_value(self.state_current, a)) for a in range(self.action_count)]

    def run_step(self) -> tuple[float, int, bool] | None:
        sp = super().run_step()
        if None == sp:
            return None
        r = sp[0]
        a = sp[1]
        self.action_best_q, q_max = self.select_an_action_best_q()
        self.train_action(a, q_max, r)
        return sp

    def train_action(self, a: int, q_max: float, reward: float):
        _q = self.knowledge_model.read_q_value(self.state_pre, a)
        _q_fixed: float = _q + self.alpha * (reward + self.gamma * q_max - _q)
        self.knowledge_model.update_q_value(self.state_pre, a, _q_fixed)

    def save_knowledge(self, filepath: str):
        self.knowledge_model.save_knowledge(filepath)
