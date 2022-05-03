from typing import List, Tuple, Dict, Union
import numpy as np
import torch
import yaml
import datetime
from torch.utils.tensorboard import SummaryWriter

from absl import app
from absl import flags
import numpy as np
from tqdm import tqdm
import pybullet as p
from locomotion.agents.model import StochasticPolicy  # pytype: disable=import-error

from locomotion.envs import env_builder
from locomotion.robots import a1
from locomotion.robots import laikago
from locomotion.robots import robot_config

from locomotion.agents.sac import SAC
from locomotion.agents.utils import check_path, Buffer


FLAGS = flags.FLAGS
flags.DEFINE_enum('robot_type', 'A1', ['A1', 'Laikago'], 'Robot Type.')
flags.DEFINE_enum('motor_control_mode', 'Torque',
                  ['Torque', 'Position', 'Hybrid'], 'Motor Control Mode.')
flags.DEFINE_bool('on_rack', False, 'Whether to put the robot on rack.')
flags.DEFINE_string('video_dir', None,
                    'Where to save video (or None for not saving).')

ROBOT_CLASS_MAP = {'A1': a1.A1, 'Laikago': laikago.Laikago}

MOTOR_CONTROL_MODE_MAP = {
    'Torque': robot_config.MotorControlMode.TORQUE,
    'Position': robot_config.MotorControlMode.POSITION,
    'Hybrid': robot_config.MotorControlMode.HYBRID
}


def main():
    config = {
        'model_config': {
            'o_dim': None,
            'a_dim': None,
            'policy_hidden_layers': [256, 256],
            'value_hidden_layers': [256, 256],
            'a_min': -1.0,
            'a_max': 1.0,
            'logstd_min': -20,
            'logstd_max': 2,
        },
        'seed': 10,
        'manual_action_bound': 10,
        'buffer_size': 1000000,
        'lr': 0.0003,
        'gamma': 0.99,
        'tau': 0.001,
        'batch_size': 512,
        'initial_alpha': 10,
        'train_policy_delay': 2,
        'device': 'cuda',
        'max_timesteps': 1000000,
        'eval_interval': 5000,
        'eval_episode': 10,
        'result_path': '/home/xukang/Project/locomotion_simulation/locomotion/results/sac_forward_task/'
    }
    
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    config.update({
        'exp_path': config['result_path'] + f"{datetime.datetime.now().strftime('%m-%d_%H-%M')}/"
    })
    check_path(config['exp_path'])
    logger = SummaryWriter(config['exp_path'])

    robot = a1.A1
    env = env_builder.build_regular_env(robot,
                                        motor_control_mode=robot_config.MotorControlMode.TORQUE,
                                        enable_rendering=False,
                                        on_rack=False,
                                        wrap_trajectory_generator=False)

    config['model_config'].update({
        'o_dim': env.observation_space.shape[0],
        'a_dim': env.action_space.shape[0]
    })
    with open(config['exp_path'] + 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, indent=2)


    agent = SAC(config)
    #obs_filter = MeanStdFilter(shape=config['model_config']['o_dim'])
    buffer = Buffer(memory_size=config['buffer_size'])


    total_step, total_episode = 0, 0
    episode_step, episode_r = 0, 0
    best_score = 0
    obs = env.reset()
    while total_step < config['max_timesteps']:
        action = agent.policy.act(obs, True)
        transferred_action = action * config['manual_action_bound']
        next_obs, reward, done, info = env.step(transferred_action)
        buffer.save_trans((obs, action, reward, done, next_obs))
        loss_dict = agent.train_ac(buffer)

        episode_r += reward
        episode_step += 1

        if done:
            if total_episode % 100 == 0:
                print(f"- - - Episode: {total_episode} Episode Step: {episode_step} Episode R: {episode_r} - - -")
            episode_r = episode_step = 0
            total_episode += 1
            obs = env.reset()
        else:
            obs = next_obs

        if total_step % config['eval_interval'] == 0:
            eval_score = agent.evaluate(env, config['manual_action_bound'], config['eval_episode'])
            if eval_score > best_score:
                agent.save_policy(config['exp_path'], 'best')
                best_score = eval_score

            print(f"| Step: {total_step} | Episode: {total_episode} | Eval_Return: {eval_score} | Loss: {loss_dict} |")
            logger.add_scalar('Eval/Eval_Return', eval_score, total_step)
            for loss_name, loss_value in list(loss_dict.items()):
                logger.add_scalar(f'Train/{loss_name}', loss_value, total_step)

        total_step += 1


def demo(exp_path: str) -> None:
    with open(exp_path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    policy = StochasticPolicy(
        config['model_config'],
        torch.device('cpu')
    )
    policy.load_state_dict(torch.load(exp_path + 'policy_best'))
    env = env_builder.build_regular_env(robot_class=a1.A1,
                                        motor_control_mode=robot_config.MotorControlMode.TORQUE,
                                        enable_rendering=True,
                                        on_rack=False,
                                        wrap_trajectory_generator=False)

    for epi in range(1000):
        done = False
        obs = env.reset()
        episode_r = 0
        episode_step = 0
        while not done:
            a = policy.act(obs, False)
            obs, r, done, info = env.step(a * config['manual_action_bond'])
            episode_r += r
            episode_step += 1
        
        print(f"| Episode {epi} | Step {episode_step} | Return {episode_r}")



#main()
demo('/home/xukang/Project/locomotion_simulation/locomotion/results/sac_forward_task/05-03_08-44/')