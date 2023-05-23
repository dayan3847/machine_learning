import numpy as np


class BanditMachineAction:
    def __init__(self, action_id: int, median: float, variance: float = 1.0):
        self.action_id = action_id
        self.n_push = 0
        self.median = median
        self.variance = variance

    def push(self) -> float:
        self.n_push += 1
        return np.random.normal(self.median, self.variance)


class BanditMachine:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.action_list = []
        for i in range(self.n_actions):
            self.action_list.append(BanditMachineAction(i, np.random.normal(0, 1)))

    def push(self, action_id) -> float:
        return self.action_list[action_id].push()

    def get_optimal_action(self) -> int:
        return max(self.action_list, key=lambda action: action.median).action_id
