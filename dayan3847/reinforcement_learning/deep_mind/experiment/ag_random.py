from dayan3847.reinforcement_learning.agent.RandomAgent import RandomAgent
from dayan3847.reinforcement_learning.deep_mind.functions_deep_mind import run_experiment

if __name__ == '__main__':
    ag = RandomAgent(action_count=7)
    run_experiment(ag, 'random')
