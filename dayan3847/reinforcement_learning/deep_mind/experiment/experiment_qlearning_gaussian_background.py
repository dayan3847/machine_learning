from dayan3847.reinforcement_learning.deep_mind.experiment.agests import balance_qlearning_gaussian_5_11111
from dayan3847.reinforcement_learning.deep_mind.functions_deep_mind import run_experiment

if __name__ == '__main__':
    ag, get_state = balance_qlearning_gaussian_5_11111()
    ag.epsilon = .1
    # ag.knowledge_model.load_knowledge('gaussian_knowledge.csv')
    run_experiment(ag, get_state, 'gaussian')
