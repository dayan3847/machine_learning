import numpy as np

from dayan3847.reinforcement_learning.case_1d import Environment, Agent, AgentGrapherPyGame, AgentRandom, AgentQLearning

if __name__ == '__main__':
    env: Environment = Environment()
    grapher: AgentGrapherPyGame = AgentGrapherPyGame(env)

    a_static: Agent = Agent(env, 'target')
    a_static.color = (0, 255, 0)
    a_static.point = np.array([env.MAX[0] - 2, 0])

    a_random: AgentRandom = AgentRandom(env)
    a_q_learning: AgentQLearning = AgentQLearning(env)

    env.run()
