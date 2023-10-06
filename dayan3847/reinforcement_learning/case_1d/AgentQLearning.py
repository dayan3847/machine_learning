import os
import time
import numpy as np
from dayan3847.reinforcement_learning.case_1d import Environment, AgentDynamic


# Q-Learning Agent
class AgentQLearning(AgentDynamic):

    def __init__(self, env: Environment, name: str):
        super().__init__(env, name)
        self._path_q = 'q'
        self.Q: np.array = np.load(self._path_q) if os.path.isfile(self._path_q) \
            else np.zeros((env.MAX[0], env.MAX[1], 4))
        self.alpha = .1
        self.gamma = 1
        self.epsilon = .1

    # reward
    def get_action(self) -> (np.array, int):
        actions: np.ndarray = self.env.get_actions_available(self)
        index: int = np.random.randint(0, len(actions)) if np.random.random() < self.epsilon \
            else self.get_q_max(self.point)[0]
        return actions[index], index

    def run_callback(self):
        while self.running:
            time.sleep(self.env.TIME_STEP)
            action, a = self.get_action()
            # last state
            last_state: np.array = self.point
            reward: int = self.env.apply_action(self, action)
            # current state
            current_state: np.array = self.point
            # update Q
            q_sa: float = self.get_q(last_state, a)
            q_sa_max: float = self.get_q_max(current_state)[1]
            fixed_q: float = q_sa + self.alpha * (reward + self.gamma * (q_sa_max - q_sa))
            self.fix_q(last_state, a, fixed_q)

    def get_q(self, status: np.array, action: int) -> float:
        return self.Q[status[0], status[1], action]

    def get_q_max(self, status: np.array) -> (int, float):
        return np.argmax(self.Q[status[0], status[1]]), np.max(self.Q[status[0], status[1]])

    def fix_q(self, status: np.array, action: int, q: float):
        self.Q[status[0], status[1], action] = q

    def save_q(self):
        np.save(self._path_q, self.Q)

    def stop(self):
        self.save_q()
        super().stop()
