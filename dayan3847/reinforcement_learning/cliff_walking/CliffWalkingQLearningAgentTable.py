import numpy as np

from dayan3847.reinforcement_learning.agent.QLearningAgent import QLearningAgent, KnowledgeModel


class CliffWalkingKnowledgeModelTable(KnowledgeModel):

    def __init__(self, actions_count: int, board_shape: tuple[int, int]):
        self.table: np.array = np.zeros((actions_count, *board_shape), dtype=np.float64)

    def read_q_value(self,
                     s: np.array,  # state
                     a: int,  # action
                     ) -> float:
        q = self.table[a, s[0], s[1]]
        return float(q)

    def update_q_value(self,
                       s: np.array,  # state (normalmente seria el estado previo)
                       a: int,  # action
                       q: float,  # q_value
                       ):
        self.table[a, s[0], s[1]] = q

    def save_knowledge(self, filepath: str):
        np.save(filepath, self.table)

    def load_knowledge(self, filepath: str):
        self.table = np.load(filepath)


class CliffWalkingQLearningAgentTable(QLearningAgent):

    def __init__(self, actions_count: int, board_shape: tuple[int, int]):
        self.knowledge_model: CliffWalkingKnowledgeModelTable = CliffWalkingKnowledgeModelTable(
            actions_count=actions_count,
            board_shape=board_shape,
        )
        super().__init__(actions_count, self.knowledge_model)
