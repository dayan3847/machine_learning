import numpy as np

from dayan3847.reinforcement_learning.agent.TemporalDifferenceLearningAgent \
    import KnowledgeModel, QLearningAgent, SarsaAgent


class KnowledgeModelTable(KnowledgeModel):

    def __init__(self, actions_count: int, state_shape: tuple):
        self.actions_count: int = actions_count
        self.state_shape: tuple = state_shape
        self.table: np.array = np.zeros((actions_count, *state_shape), dtype=np.float64)

    def read_q_value(self,
                     s: np.array,  # state
                     a: int,  # action
                     ) -> float:
        _pos = (a, *tuple(np.array(s, dtype=np.int64)))
        q = self.table[_pos]
        return float(q)

    def update_q_value(self,
                       s: np.array,  # state (normalmente seria el estado previo)
                       a: int,  # action
                       q: float,  # q_value
                       ):
        _pos = (a, *tuple(np.array(s, dtype=np.int64)))
        self.table[_pos] = q

    def save_knowledge(self, filepath: str):
        np.save(filepath, self.table)

    def load_knowledge(self, filepath: str):
        self.table = np.load(filepath)

    def reset_knowledge(self):
        self.table = np.zeros_like(self.table)


class QLearningAgentTable(QLearningAgent):

    def __init__(self, action_count: int, state_shape: tuple):
        super().__init__(
            action_count,
            KnowledgeModelTable(
                actions_count=action_count,
                state_shape=state_shape,
            )
        )


class SarsaAgentTable(SarsaAgent):

    def __init__(self, action_count: int, state_shape: tuple):
        super().__init__(
            action_count,
            KnowledgeModelTable(
                actions_count=action_count,
                state_shape=state_shape,
            )
        )
