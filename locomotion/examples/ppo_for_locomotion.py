from typing import List, Dict, Tuple, Union
import numpy as np
import yaml
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter

from absl import app
from absl import flags
import numpy as np
from tqdm import tqdm
import pybullet as p  # pytype: disable=import-error

from locomotion.envs import env_builder
from locomotion.robots import a1
from locomotion.robots import laikago
from locomotion.robots import robot_config

from locomotion.agents.ppo import PPO
from locomotion.agents.utils import check_path



FLAGS = flags.FLAGS
flags.DEFINE_enum('robot_type', 'A1', ['A1', 'Laikago'], 'Robot Type.')
flags.DEFINE_enum('motor_control_mode', 'Position',
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
            'policy_hidden_layers': [128, 128, 128],
            'value_hidden_layers': [128, 128, 128],
            'a_min': -1.0,
            'a_max': 1.0,
        },
        'seed': 10,
        'num_workers': 10,
        'manual_action_scale': 1,
        'lr': 0.0003,
        'gamma': 0.99,
        'tau': 0.005,
        'action_std': 0.4,
        'ratio_clip': 0.25,
        'temperature_coeff': 0.1,
        'num_epoch': 10,
        'batch_size': 256,
        'initial_alpha': 10,
        'train_policy_delay': 2,
        'device': 'cuda',
        'max_timesteps': 10000000,
        'eval_iteration_interval': 1,
        'eval_episode': 10,
        'result_path': '/home/xukang/Project/locomotion_simulation/locomotion/results/ppo_forward_task_positon_mode/'
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
                                        wrap_trajectory_generator=False,
                                        enable_clip_motor_commands=True)

    config['model_config'].update({
        'o_dim': env.observation_space.shape[0],
        'a_dim': env.action_space.shape[0]
    })
    with open(config['exp_path'] + 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, indent=2)

    agent = PPO(config)
    agent._init_workers(env, config['eval_episode'])

    total_step, total_episode, total_iteration = 0, 0, 0
    best_score = 0
    while total_step < config['max_timesteps']:
        train_score, worker_scores, loss_pi, loss_v = agent.roll_update()

        total_step = agent.total_steps
        total_episode = agent.total_episodes

        if total_iteration % config['eval_iteration_interval'] == 0:
            eval_score = agent.evaluation(env, config['eval_episode'])
            if eval_score > best_score:
                agent.save_policy(config['exp_path'], 'best')
                best_score = eval_score

            print(f"| Step: {total_step} | Episode: {total_episode} | Eval_Return: {eval_score} | Loss_pi: {loss_pi} | Loss_v: {loss_v}")
            logger.add_scalar('Eval/Train_Return', train_score, total_step)
            logger.add_scalar('Eval/Eval_Return', eval_score, total_step)
            logger.add_scalar('Train/loss_pi', loss_pi, total_step)
            logger.add_scalar('Train/loss_v', loss_v, total_step)

        total_iteration += 1

main()