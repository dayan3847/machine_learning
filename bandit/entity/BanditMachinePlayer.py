import numpy as np
from typing import List
from enum import Enum
from bandit.entity.BanditMachine import BanditMachine


class ActionType(Enum):
    EXPLORATION = 'exploration'
    EXPLOITATION = 'exploitation'


class BanditMachinePlayerEpsilonGreedy:
    def __init__(self, bandit_machine: BanditMachine, initial_q: float = 0.0, epsilon: float = 0.1):
        self.bandit_machine: BanditMachine = bandit_machine
        # number of times each button has been pressed
        self.N: np.ndarray = np.zeros(self.bandit_machine.n_actions)
        # estimated value of each button
        self.Q: np.ndarray = np.full(self.bandit_machine.n_actions, initial_q)
        # best estimated value
        self.best_q: float = initial_q
        # best actions
        self.best_actions: List[int] = list(range(self.bandit_machine.n_actions))
        # epsilon
        self.epsilon: float = epsilon

    def get_next_action_type(self) -> ActionType:
        if np.random.random() < self.epsilon:
            return ActionType.EXPLORATION
        else:
            return ActionType.EXPLOITATION

    def get_next_action_exploration(self) -> int:
        return np.random.randint(self.bandit_machine.n_actions)

    def get_next_action_exploitation(self) -> int:
        return np.random.choice(self.best_actions)

    def get_next_action(self) -> int:
        action_type = self.get_next_action_type()
        if action_type == ActionType.EXPLORATION:
            return self.get_next_action_exploration()
        elif action_type == ActionType.EXPLOITATION:
            return self.get_next_action_exploitation()
        else:
            raise ValueError('Invalid action type')

    def push(self, action_id: int):
        reward: float = self.bandit_machine.push(action_id)
        self.N[action_id] += 1
        self.Q[action_id] += (reward - self.Q[action_id]) / self.N[action_id]
        # update best_q and best_actions
        if self.Q[action_id] > self.best_q:
            self.best_q = self.Q[action_id]
            self.best_actions = [action_id]
        elif self.Q[action_id] == self.best_q:
            self.best_actions.append(action_id)

    def pay(self):
        action_id: int = self.get_next_action()
        self.push(action_id)
