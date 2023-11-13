import unittest
import numpy as np
from dm_control import suite
from dm_control.rl.control import Environment
from QLearningAgentPV import QLearningAgentPV


class QLearningAgentPVTest(unittest.TestCase):
    def test_1(self):
        random_state = np.random.RandomState(42)
        env: Environment = suite.load('cartpole', 'balance', task_kwargs={'random': random_state})

        ag: QLearningAgentPV = QLearningAgentPV(
            env=env,
            action_count=11,
        )
        ag.knowledge_model.model.summary()

        self.assertEqual(True, True)

    def test_2(self):
        random_state = np.random.RandomState(42)
        env: Environment = suite.load('cartpole', 'balance', task_kwargs={'random': random_state})

        ag: QLearningAgentPV = QLearningAgentPV(
            env=env,
            action_count=11,
        )
        ag.run_episode()

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
