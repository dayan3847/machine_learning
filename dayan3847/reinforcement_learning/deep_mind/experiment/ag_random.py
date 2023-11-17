import numpy as np
from dm_control import suite
from dm_control.rl.control import Environment

from dayan3847.reinforcement_learning.deep_mind.agent.RandomAgent import RandomAgent
from dayan3847.reinforcement_learning.deep_mind.deep_mind_experiment import deep_mind_experiment

if __name__ == '__main__':
    random_state = np.random.RandomState(42)
    env: Environment = suite.load('cartpole', 'balance', task_kwargs={'random': random_state})
    ag = RandomAgent(
        env=env,
        action_count=7,
    )
    deep_mind_experiment(ag, 'random', 'csv')
