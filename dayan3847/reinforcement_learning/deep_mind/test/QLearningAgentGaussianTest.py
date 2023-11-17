import unittest
import numpy as np
from dm_control import suite
from dm_control.rl.control import Environment

from dayan3847.reinforcement_learning.deep_mind.agent.QLearningAgentGaussian import KnowledgeModelGaussian

np.random.seed(0)


class QLearningAgentGaussianTest(unittest.TestCase):
    def test_knowledge(self):
        knowledge1 = KnowledgeModelGaussian(7)
        knowledge1.save_knowledge('knowledge.csv')
        knowledge2 = KnowledgeModelGaussian(7)
        knowledge2.load_knowledge('knowledge.csv')
        knowledge1.save_knowledge('knowledge2.csv')

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
