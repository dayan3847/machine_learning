from dayan3847.reinforcement_learning.deep_mind.experiment.agests import balance_qlearning_table_5
from dayan3847.reinforcement_learning.deep_mind.functions_deep_mind import run_experiment


if __name__ == '__main__':
    ag, get_state = balance_qlearning_table_5()
    run_experiment(ag, get_state, 'table')
