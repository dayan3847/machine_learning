import numpy as np
from dm_control import suite
from dm_control.rl.control import Environment

from dayan3847.reinforcement_learning.agent.RandomAgent import RandomAgent
from dayan3847.reinforcement_learning.deep_mind.functions_deep_mind import deep_mind_experiment

if __name__ == '__main__':
    random_state = np.random.RandomState(42)
    env: Environment = suite.load('cartpole', 'balance', task_kwargs={'random': random_state})
    ag = RandomAgent(action_count=7)
    deep_mind_experiment(env, ag, 'random')
