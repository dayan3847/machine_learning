import numpy as np
from typing import List
from dayan3847.bandit.src.BanditMachine import BanditMachine


class BanditMachinePlayerEpsilonGreedy:
    def __init__(self, bandit_machine: BanditMachine, q1: float = 0.0, epsilon: float = 0.1):
        self.bandit_machine: BanditMachine = bandit_machine
        # number of times each button has been pressed
        self.N: np.ndarray = np.zeros(self.bandit_machine.n_actions)
        # estimated value of each button
        self.Q: np.ndarray = np.full(self.bandit_machine.n_actions, q1)
        # best estimated value
        self.best_q: float = q1
        # best actions
        self.best_actions: List[int] = list(range(self.bandit_machine.n_actions))
        # epsilon
        self.epsilon: float = epsilon

    # Get next action type
    # True: exploration
    # False: exploitation
    def get_next_action_type(self) -> bool:
        return np.random.random() < self.epsilon

    def get_next_action_exploration(self) -> int:
        return np.random.randint(self.bandit_machine.n_actions)

    def get_next_action_exploitation(self) -> int:
        return np.random.choice(self.best_actions)

    def get_next_action(self) -> int:
        exploration: bool = self.get_next_action_type()
        if exploration:
            return self.get_next_action_exploration()
        else:
            return self.get_next_action_exploitation()

    def play(self) -> [float, int]:
        # action id
        a: int = self.get_next_action()
        reward: float = self.bandit_machine.push(a)
        self.N[a] += 1
        self.Q[a] += (reward - self.Q[a]) / self.N[a]
        # update best_q and best_actions
        if self.Q[a] > self.best_q:
            self.best_q = self.Q[a]
            self.best_actions = [a]
        elif self.Q[a] == self.best_q:
            self.best_actions.append(a)
        elif a in self.best_actions:
            self.best_actions.remove(a)
            if len(self.best_actions) == 0:
                self.calculate_best_q()

        return reward, a

    def calculate_best_q(self):
        self.best_q = self.Q.max()
        self.best_actions = np.where(self.Q == self.best_q)[0].tolist()
