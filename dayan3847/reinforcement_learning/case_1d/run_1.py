from dayan3847.reinforcement_learning.case_1d import Environment, GrapherPyGame, AStatic, ARandom, AQLearning

if __name__ == '__main__':
    env: Environment = Environment()
    grapher: GrapherPyGame = GrapherPyGame(env)
    a_static: AStatic = AStatic(env)
    a_random: ARandom = ARandom(env)
    a_q_learning: AQLearning = AQLearning(env)

    env.run()
