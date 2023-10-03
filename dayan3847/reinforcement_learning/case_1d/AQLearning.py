import os
import numpy as np

from dayan3847.reinforcement_learning.case_1d import Environment, ADynamic, AgentPhysical, Agent


# Q-Learning Agent
class AQLearning(ADynamic):

    def __init__(self, env: Environment):
        super().__init__(env)
        self.name = 'q_learning'
        a_target: Agent = env.get_agent_by_name('target')
        self.target: np.array = a_target.point if isinstance(a_target, AgentPhysical) else np.array([0, 0])
        self.color = (255, 255, 0)
        self._path_q = 'q.txt'
        self.Q: np.array = np.loadtxt(self._path_q) if os.path.isfile(self._path_q) \
            else np.zeros((env.MAX[0], self.actions_count))
        self.alpha = .1
        self.gamma = 1
        self.epsilon = .01

    # reward
    def get_reward(self, action: np.array) -> float:
        # distance
        distance: float = np.linalg.norm(self.target - self.point)
        # movement_penalty
        penalty: int = 0 if np.all(action == 0) else .1
        # reward
        return -1 * distance - penalty

    def get_action(self) -> int:
        return np.random.randint(0, self.actions_count) if np.random.random() < self.epsilon \
            else np.argmax(self.Q[self.point[0]])

    def apply_action(self, a: int):
        action: np.array = self.actions[a]
        # last state
        last_state: np.array = self.point
        # apply action
        super().apply_action(a)
        # current state
        current_state: np.array = self.point
        # update Q
        q_sa: float = self.Q[last_state[0], a]
        self.Q[last_state[0], a] = q_sa + self.alpha * (
                self.get_reward(action) + self.gamma * np.max(self.Q[current_state[0]]) - q_sa
        )

    def save_q(self):
        np.savetxt(self._path_q, self.Q)

    def stop(self):
        self.save_q()
        super().stop()

    # def policy(self) -> int:
    #     return np.argmax(self.Q[self.point[0], self.point[1]])
    #
    # def q(self, state: np.array, action: np.array) -> float:
    #     return self.Q[state[0], state[1], action]
    #
    # def actions(self, state: np.array) -> np.array:
    #     return self.Q[state[0], state[1]]
