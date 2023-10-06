import os
import time
import numpy as np
from dayan3847.reinforcement_learning.case_1d import Environment, Agent, AgentDynamic


# Q-Learning Agent
class AgentQLearning(AgentDynamic):

    def __init__(self, env: Environment, name: str):
        super().__init__(env, name)
        self._path_q = 'q.txt'
        self.Q: np.array = np.loadtxt(self._path_q) if os.path.isfile(self._path_q) \
            else np.zeros((env.MAX[0], 3))
        self.alpha = .1
        self.gamma = 1
        self.epsilon = .1

    # reward
    def get_action(self) -> np.array:
        actions: np.ndarray = self.env.get_actions_available(self)
        index: int = np.random.randint(0, len(actions)) if np.random.random() < self.epsilon \
            else np.argmax(self.Q[self.point[0]])
        return actions[index]

    def run_callback(self):
        while self.running:
            time.sleep(self.env.TIME_STEP)
            action: np.array = self.get_action()
            # last state
            last_state: np.array = self.point
            reward: int = self.env.apply_action(self, action)
            # current state
            current_state: np.array = self.point
            # update Q
            q_sa: float = self.Q[last_state[0], action]
            self.Q[last_state[0], action] = q_sa + self.alpha * (
                    reward + self.gamma * np.max(self.Q[current_state[0]]) - q_sa
            )

    def save_q(self):
        np.savetxt(self._path_q, self.Q)

    def stop(self):
        self.save_q()
        super().stop()
