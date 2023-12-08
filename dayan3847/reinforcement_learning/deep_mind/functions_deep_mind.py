import time
from datetime import datetime
import numpy as np
from dm_env import TimeStep
from dm_control import suite
from dm_control.rl.control import Environment

from dayan3847.reinforcement_learning.agent.Agent import Agent


def get_action_values(env_: Environment, action_count_: int) -> np.array:
    _spec = env_.action_spec()
    if action_count_ == 5:
        return np.array([
            -.3, -.1, 0, .1, .3,
        ], dtype=_spec.dtype)
    elif action_count_ == 7:
        return np.array([
            -.6, -.3, -.1, 0, .1, .3, .6,
        ], dtype=_spec.dtype)
    return np.linspace(_spec.minimum, _spec.maximum, action_count_)


def run_experiment(ag: Agent, get_state: callable, model_name: str, knowledge_extension: str = 'csv'):
    history_global_reward: list[float] = []
    history_global_state: list[np.array] = []

    def policy(time_step_: TimeStep) -> int:
        s = get_state(time_step_)

        if not time_step_.first():
            r = float(time_step_.reward)
            ag.train_action(s, r)

        a = ag.select_an_action(s)

        return a

    env: Environment = suite.load('cartpole', 'balance',
                                  task_kwargs={'random': np.random.RandomState(42)})
    action_values: np.array = get_action_values(env, ag.action_count)

    def run_episode() -> tuple[list[float], list[np.array], bool]:  # (reward,state,win)
        h_reward: list[float] = []
        h_state: list[np.array] = []
        lose: bool = False
        _time_step = env.reset()
        episode_step = 0
        while not _time_step.last() and not lose:
            if np.loadtxt('stop.txt') == 2:
                print('\033[96m' + 'stop' + '\033[00m')
                break
            episode_step += 1
            print('\033[96m{}\033[00m'.format(episode_step))
            h_state.append(get_state(_time_step))
            action: int = policy(_time_step)
            action_val: float = float(action_values[action])
            _time_step = env.step(action_val)
            r = float(_time_step.reward)
            h_reward.append(r)
            if r < .35:
                lose = True
        policy(_time_step)
        return h_reward, h_state, not lose

    while True:
        start_time = time.time()
        if np.loadtxt('stop.txt') != 0:
            print('stop')
            break
        name: str = datetime.now().strftime('%Y%m%d%H%M%S')
        print(f'running episode {name}')
        history_reward, history_state, _win = run_episode()
        history_global_reward += history_reward
        history_global_state += history_state
        if _win:  # green
            print('\033[92m' + 'WIN' + '\033[00m')
        else:  # red
            print('\033[91m' + 'LOSE' + '\033[00m')

        # history_frames: list[np.array] = [ag.env.physics.render(camera_id=0)]

        print('saving knowledge')
        ag.save_knowledge(f'{model_name}_knowledge.{knowledge_extension}')
        # ag.save_knowledge(f'ep/{name}_{model_name}_knowledge.{knowledge_extension}')
        print('saving reward')
        np.savetxt('reward.txt', history_reward)
        # np.savetxt(f'ep/{name}_{model_name}_reward.txt', history_reward)
        print('saving state')
        np.savetxt('state.csv', history_state, delimiter=',')
        np.savetxt(f'ep/{name}_{model_name}_state.csv', history_state, delimiter=',')
        # print('saving frames')
        # np.save('frames.npy', history_frames)

        print('saving global reward')
        np.savetxt('global_reward.txt', history_global_reward)
        print('saving global state')
        np.savetxt('global_state.csv', history_global_state, delimiter=',')

        # print in yellow
        print('\033[93m' + 'episode time: ' + str(time.time() - start_time) + '\033[00m')

    np.savetxt('stop.txt', [0])
