from dayan3847.reinforcement_learning.deep_mind.experiment.agests import balance_qlearning_table_5
from dayan3847.reinforcement_learning.deep_mind.functions_deep_mind import run_experiment

FILE_PATH = 'knowledge_qlearning_table.npy'

if __name__ == '__main__':
    ag, get_state = balance_qlearning_table_5()
    ag.epsilon = .01
    ag.knowledge_model.load_knowledge(FILE_PATH)
    run_experiment(ag, get_state, 'table')
