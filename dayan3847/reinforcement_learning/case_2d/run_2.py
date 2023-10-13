import numpy as np

from dayan3847.reinforcement_learning.case_2d import Environment, AgentGrapherPyGame
from dayan3847.reinforcement_learning.case_1d import Agent, AgentRandom, AgentQLearning

if __name__ == '__main__':
    env: Environment = Environment((12, 4))
    grapher: AgentGrapherPyGame = AgentGrapherPyGame(env)

    a_good_target: Agent = Agent(env, 'target')
    a_good_target.color = (0, 255, 0)
    a_good_target.point = np.array([env.board_size[0] - 1, 0])
    a_good_target.reward = 100
    env.targets.append(a_good_target)

    for i in range(1, env.board_size[0] - 1):
        a_bad_target: Agent = Agent(env, 'bad_target')
        a_bad_target.point = np.array([i, 0])
        a_bad_target.reward = -100
        env.targets.append(a_bad_target)

    # a_random: AgentRandom = AgentRandom(env, 'random')
    # a_random.color = (255, 0, 0)

    a_q_learning: AgentQLearning = AgentQLearning(env, 'q_learning')
    a_q_learning.color = (255, 255, 0)

    env.run()
