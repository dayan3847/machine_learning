from datetime import datetime
import numpy as np
from dm_control import suite
from dm_control.rl.control import Environment
from dm_env import StepType

from dayan3847.reinforcement_learning.deep_mind.agent.QLearningAgentNN import QLearningAgentNN

if __name__ == '__main__':
    random_state = np.random.RandomState(42)
    env: Environment = suite.load('cartpole', 'balance', task_kwargs={'random': random_state})

    ag: QLearningAgentNN = QLearningAgentNN(env=env, action_count=11)
    # ag.knowledge_model.load_knowledge('knowledge.h5')
    # ag.epsilon = .99

    while True:
        if np.loadtxt('stop.txt') != 0:
            print('stop')
            break
        name: str = datetime.now().strftime('%Y%m%d%H%M%S')
        print(f'running episode {name}')
        ag.init_episode()
        history_reward: list[float] = []
        # history_frames: list[np.array] = [ag.env.physics.render(camera_id=0)]
        while StepType.LAST != ag.time_step.step_type:
            if np.loadtxt('stop.txt') == 2:
                print('\033[96m' + 'stop' + '\033[00m')
                break
            print('\033[96m{}\033[00m'.format(ag.step))
            _r, _a, _is_random = ag.run_step()
            history_reward.append(_r)
            # history_frames.append(ag.env.physics.render(camera_id=0))
            print("Action: ", '\033[91m' if _is_random else '\033[92m', _a, '\033[00m')
            print("Reward: ", _r)

        print('saving knowledge')
        ag.knowledge_model.save_knowledge('knowledge.h5')
        print('saving reward')
        np.savetxt('../reward.txt', history_reward)
        np.savetxt(f'ep/{name}_reward.txt', history_reward)
        # print('saving frames')
        # np.save('frames.npy', history_frames)
