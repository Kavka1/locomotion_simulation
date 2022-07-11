from typing import List, Dict, Tuple, Union
import numpy as np
import yaml
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter

from absl import flags
import numpy as np
from locomotion.agents.model import FixStdGaussianPolicy  # pytype: disable=import-error

from locomotion.envs import env_builder
from locomotion.robots import a1
from locomotion.robots import laikago
from locomotion.robots import robot_config

from locomotion.agents.ppo import PPO, Worker
from locomotion.agents.utils import check_path



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
            'policy_hiddens': [128, 128, 128],
            'value_hiddens': [128, 128, 128],
            'a_min': -1.0,
            'a_max': 1.0,
        },
        'seed': 10,
        'num_workers': 10,
        'manual_action_scale': [
            [
                -0.80285144, -1.0471976, -2.6965337, 
                -0.80285144, -1.0471976, -2.6965337, 
                -0.80285144, -1.0471976, -2.6965337, 
                -0.80285144, -1.0471976, -2.6965337
            ],
            [
                0.80285144, 4.1887903, -0.91629785, 
                0.80285144, 4.1887903, -0.91629785, 
                0.80285144, 4.1887903, -0.91629785, 
                0.80285144, 4.1887903, -0.91629785
            ]
        ],
        'lr': 0.0003,
        'gamma': 0.99,
        'lamda': 0.95,
        'tau': 0.005,
        'action_std': 0.2,
        'ratio_clip': 0.25,
        'temperature_coef': 0.1,
        'num_epoch': 10,
        'batch_size': 256,
        'device': 'cuda',
        'max_timesteps': 20000000,
        'eval_iteration_interval': 5,
        'eval_episode': 10,
        'result_path': '/home/xukang/Project/locomotion_simulation/locomotion/results/ppo_position_mode_forward_task/'
    }
    
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    config.update({
        'exp_path': config['result_path'] + f"{datetime.datetime.now().strftime('%m-%d_%H-%M')}/"
    })
    check_path(config['exp_path'])
    logger = SummaryWriter(config['exp_path'])

    env = env_builder.build_regular_env(robot_class=a1.A1,
                                        motor_control_mode=robot_config.MotorControlMode.POSITION,
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
    # Initialize workers
    for i in range(agent.num_workers):
        agent.workers.append(
            Worker.remote(
                i,
                agent.o_dim,
                agent.a_dim,
                agent.action_std,
                agent.policy_hiddens,
                agent.value_hiddens,
                agent.gamma, 
                agent.lamda,
                agent.num_worker_rollout,
                agent.action_bound,
                agent.manual_action_scale,
                control_mode = robot_config.MotorControlMode.POSITION
            )
        )

    total_step, total_episode, total_iteration = 0, 0, 0
    best_score = 0
    while total_step < config['max_timesteps']:
        train_score, worker_scores, loss_pi, loss_v = agent.roll_update()

        take_steps = agent.total_steps - total_step
        total_step = agent.total_steps
        total_episode = agent.total_episodes

        if total_iteration % config['eval_iteration_interval'] == 0:
            eval_score, eval_steps = agent.evaluation(env, config['eval_episode'])

            if eval_score > best_score:
                agent.save_policy(config['exp_path'], 'best')
                best_score = eval_score

            print(f"| Step: {total_step} | Episode: {total_episode} | Take_Step: {take_steps} | Eval_Return: {eval_score} | Loss_pi: {loss_pi} | Loss_v: {loss_v}")
            logger.add_scalar('Eval/Train_Return', train_score, total_step)
            logger.add_scalar('Eval/Eval_Return', eval_score, total_step)
            logger.add_scalar('Train/loss_pi', loss_pi, total_step)
            logger.add_scalar('Train/loss_v', loss_v, total_step)

        total_iteration += 1


def demo(exp_path: str) -> None:
    with open(exp_path + 'config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    policy = FixStdGaussianPolicy(
        config['model_config']['o_dim'],
        config['model_config']['a_dim'],
        config['model_config']['policy_hiddens'],
        config['action_std'],
        torch.device('cpu')
    )
    policy.load_state_dict(torch.load(exp_path + 'policy_best'))
    env = env_builder.build_regular_env(robot_class=a1.A1,
                                        motor_control_mode=robot_config.MotorControlMode.POSITION,
                                        enable_rendering=True,
                                        on_rack=False,
                                        wrap_trajectory_generator=False)

    for epi in range(1000):
        done = False
        obs = env.reset()
        episode_r = 0
        episode_step = 0
        while not done:
            a_dist = policy(torch.from_numpy(obs).float())
            a = a_dist.mean.detach().numpy()

            if isinstance(config['manual_action_scale'], List):
                action = np.array([
                        (config['manual_action_scale'][0][i] + config['manual_action_scale'][1][i]) / 2 + 
                        (config['manual_action_scale'][1][i] - config['manual_action_scale'][0][i]) / 2 * a[i] 
                        for i in range(len(a))
                    ], dtype=np.float64)
            elif isinstance(config['manual_action_scale'], float):
                action = a * config['manual_action_scale']
            else:
                raise ValueError(f"Invalid action scale : {config['manual_action_scale']}")


            obs, r, done, info = env.step(action)
            episode_r += r
            episode_step += 1
        
        print(f"| Episode {epi} | Step {episode_step} | Return {episode_r}")



main()
#demo('/home/xukang/Project/locomotion_simulation/locomotion/results/ppo_position_mode_forward_task/05-13_16-25/')