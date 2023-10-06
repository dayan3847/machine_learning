import numpy as np

from dayan3847.reinforcement_learning.case_1d import Environment, Agent, AgentGrapherPyGame, AgentRandom, AgentQLearning

if __name__ == '__main__':
    env: Environment = Environment()
    grapher: AgentGrapherPyGame = AgentGrapherPyGame(env)

    a_target: Agent = Agent(env, 'target')
    a_target.color = (0, 255, 0)
    a_target.point = np.array([env.MAX[0] - 2, 0])
    a_target.reward = 100
    env.targets.append(a_target)

    a_random: AgentRandom = AgentRandom(env, 'random')
    a_random.color = (255, 0, 0)

    a_q_learning: AgentQLearning = AgentQLearning(env, 'q_learning')
    a_q_learning.color = (255, 255, 0)

    env.run()
