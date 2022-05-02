from typing import List, Tuple, Dict, Union
import os
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F


def check_path(path: str) -> None:
    if os.path.exists(path) is False:
        os.makedirs(path)

def array2tensor(obs: np.array, a: np.array, r: np.array, done: np.array, obs_: np.array, device: torch.device) -> Tuple:
    obs = torch.from_numpy(obs).to(device).float()
    a = torch.from_numpy(a).to(device).float()
    r = torch.from_numpy(r).to(device).float().unsqueeze(dim=-1)
    done = torch.from_numpy(done).to(device).int().unsqueeze(dim=-1)
    obs_ = torch.from_numpy(obs_).to(device).float()
    return obs, a, r, done, obs_


def hard_update(source_net: nn.Module, target_net: nn.Module) -> None:
    target_net.load_state_dict(source_net.state_dict())


def soft_update(source_net: nn.Module, target_net: nn.Module, tau: float) -> None:
    for param, param_tar in zip(source_net.parameters(), target_net.parameters()):
        param_tar.data.copy_(tau * param.data + (1 - tau) * param_tar.data)


class StochasticPolicy(nn.Module):
    def __init__(self, model_config: Dict, device: torch.device):
        super(StochasticPolicy, self).__init__()
        self.o_dim, self.a_dim = model_config['o_dim'], model_config['a_dim']
        self.hidden_layers = model_config['policy_hidden_layers']
        self.a_min, self.a_max = model_config['a_min'], model_config['a_max']
        self.logstd_min, self.logstd_max = model_config['logstd_min'], model_config['logstd_max']
        self.device = device
        
        encoder = []
        hidden_layers = [self.o_dim] + self.hidden_layers
        for i in range(len(hidden_layers)-1):
            encoder += [nn.Linear(hidden_layers[i], hidden_layers[i+1]), nn.ReLU()]
        
        self.encoder = nn.Sequential(*encoder)
        self.mean_head = nn.Linear(hidden_layers[-1], self.a_dim)
        self.logstd_head = nn.Linear(hidden_layers[-1], self.a_dim)

    def act(self, obs: np.array, with_noise: True) -> np.array:
        obs = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad():
            latent = self.encoder(obs)
            if with_noise:
                mean, log_std = self.mean_head(latent), self.logstd_head(latent)
                log_std = torch.clamp(log_std, self.logstd_min, self.logstd_max)
                std = torch.exp(log_std)
                dist = Normal(mean, std)
                action = dist.sample()
            else:
                action = self.mean_head(latent)
            action = torch.tanh(action).detach().cpu().numpy()
        action = np.clip(action, self.a_min, self.a_max)
        return action

    def __call__(self, obs: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        latent = self.encoder(obs)
        mean, logstd = self.mean_head(latent), self.logstd_head(latent)
        logstd = torch.clamp(logstd, self.logstd_min, self.logstd_max)
        std = torch.exp(logstd)

        dist = Normal(mean, std)
        arctanh_action = dist.rsample()

        action = torch.tanh(arctanh_action)
        logprob = dist.log_prob(arctanh_action) - torch.log(1 - action**2 + 1e-6)
        logprob = logprob.sum(dim=-1, keepdim=True)
        
        return action, logprob

    def load_model(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
        print(f"- - - - - - - - - Loaded model from {path} - - - - - - - - - - -")



class QFunction(nn.Module):
    def __init__(self, model_config: Dict) -> None:
        super(QFunction, self).__init__()
        self.o_dim, self.a_dim = model_config['o_dim'], model_config['a_dim']
        self.hidden_layers = model_config['value_hidden_layers']

        self.model = []
        layers = [self.o_dim + self.a_dim] + self.hidden_layers
        for i in range(len(layers)-1):
            self.model += [nn.Linear(layers[i], layers[i+1]), nn.ReLU()]
        self.model.append(nn.Linear(layers[-1], 1))
        self.model = nn.Sequential(*self.model)

    def __call__(self, obs: torch.tensor, action: torch.tensor) -> torch.tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.model(x)


class TwinQFunction(nn.Module):
    def __init__(self, model_config: Dict) -> None:
        super(TwinQFunction, self).__init__()
        self.Q1_model = QFunction(model_config)
        self.Q2_model = QFunction(model_config)

    def __call__(self, obs: torch.tensor, a: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        Q1_value, Q2_value = self.Q1_model(obs, a), self.Q2_model(obs, a)
        return Q1_value, Q2_value

    def call_Q1(self, obs: torch.tensor, a: torch.tensor) -> torch.tensor:
        return self.Q1_model(obs, a)
    
    def call_Q2(self, obs: torch.tensor, a: torch.tensor) -> torch.tensor:
        return self.Q2_model(obs, a)


class Buffer(object):
    def __init__(self, memory_size) -> None:
        super(Buffer, self).__init__()

        self.size = memory_size
        self.data = deque(maxlen=self.size)
    
    def save_trans(self, transition: Tuple) -> None:
        self.data.append(transition)

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.data, batch_size)
        obs, a, r, done, obs_ = zip(*batch)
        obs, a, r, done, obs_ = np.stack(obs, 0), np.stack(a, 0), np.array(r), np.array(done), np.stack(obs_, 0)
        return obs, a, r, done, obs_

    def __len__(self):
        return len(self.data)

class RunningStats(object):
    def __init__(self, shape: Union[int, Tuple]) -> None:
        super().__init__()

        self.shape = shape
        self._mean = np.zeros(shape = shape, dtype = np.float64)
        self._square_sum = np.zeros(shape=shape, dtype = np.float64)
        self._count = 0

    def push(self, x):
        n = self._count
        self._count += 1
        if self._count == 1:
            self._mean[...] = x
        else:
            delta = x - self._mean
            self._mean[...] += delta / self._count
            self._square_sum[...] += delta**2 * n / self._count

    @property
    def var(self) -> np.array:
        return self._square_sum / (self._count - 1) if self._count > 1 else np.square(self._mean)

    @property
    def std(self) -> np.array:
        return np.sqrt(self.var)

class MeanStdFilter(object):
    def __init__(self, shape: Union[int, Tuple]) -> None:
        super().__init__()
        self.shape = shape
        self.rs = RunningStats(shape)

    def __call__(self, x: np.array) -> np.array:
        assert x.shape[0] == self.shape, (f"Filter.__call__: x.shape-{x.shape} != filter.shape-{self.shape}")
        return (x - self.rs._mean) / (self.rs.std + 1e-6)

    def trans_batch(self, x_batch: np.array) -> np.array:
        for i in range(len(x_batch)):
            x_batch[i] = (x_batch[i] - self.rs._mean) / (self.rs.std + 1e-6)
        return x_batch

    def push_batch(self, x_batch: List) -> None:
        assert x_batch[0].shape[0] == self.shape
        for x in x_batch:
            self.rs.push(x)

    def load_params(self, path: str) -> None:
        assert os.path.exists(path)
        mix_params = np.load(path, allow_pickle=True)
        self.update(mix_params[0], mix_params[1], mix_params[2])
        print(f"------Loaded obs filter params from {path}------")

    def save_filter(self, path: str, remark: str) -> None:
        assert os.path.exists(path)
        filter_params = np.array([self.mean, self.square_sum, self.count])
        np.save(path + f'filter_{remark}', filter_params)
        print(f"-------Filter params saved to {path}-------")

    @property
    def mean(self) -> np.array:
        return self.rs._mean

    @property
    def square_sum(self) -> np.array:
        return self.rs._square_sum

    @property
    def count(self) -> int:
        return self.rs._count


class SAC(object):
    def __init__(self, config: Dict) -> None:
        super(SAC, self).__init__()

        self.model_config = config['model_config']

        self.lr = config['lr']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.batch_size = config['batch_size']
        self.initial_alpha = config['initial_alpha']
        self.train_policy_delay = config['train_policy_delay']
        self.device = torch.device(config['device'])

        self.target_entropy = - torch.tensor(config['model_config']['a_dim'], dtype=torch.float64)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = torch.exp(self.log_alpha)

        self._init_model()
        self._init_optimizer()
        self._init_env()
        self._init_logger()

    def _init_model(self) -> None:
        self.policy = StochasticPolicy(self.model_config, self.device).to(self.device)
        self.value = TwinQFunction(self.model_config).to(self.device)
        self.value_tar = TwinQFunction(self.model_config).to(self.device)
        hard_update(self.value, self.value_tar)

    def _init_optimizer(self) -> None:
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.optimizer_Q = optim.Adam(self.value.parameters(), lr=self.lr)
        self.optimizer_alpha = optim.Adam([self.log_alpha], lr=self.lr)
        self.optimizer_dynamic = optim.Adam(self.dynamic.parameters(), lr=self.lr)

    def _init_logger(self) -> None:
        self.logger_loss_q = 0.
        self.logger_loss_policy = 0.
        self.logger_loss_alpha = 0.
        self.logger_alpha = self.alpha.item()
        self.update_count = 0

    def train_ac(self, buffer: Buffer, obs_filter: MeanStdFilter) -> Dict:
        if len(buffer) < self.batch_size:
            return 0., 0.

        obs, a, r, done, obs_ = buffer.sample(self.batch_size)

        obs = obs_filter.trans_batch(obs)
        obs_ = obs_filter.trans_batch(obs_)

        obs, a, r, done, obs_ = array2tensor(obs, a, r, done, obs_, self.device)

        with torch.no_grad():
            next_a, next_a_logprob = self.policy(obs_)
            next_q1_tar, next_q2_tar = self.value_tar(obs_, next_a)
            next_q_tar = torch.min(next_q1_tar, next_q2_tar)
            q_update_tar = r + (1 - done) * self.gamma * (next_q_tar - self.alpha * next_a_logprob)
        q1_pred, q2_pred = self.value(obs, a)
        loss_q = F.mse_loss(q1_pred, q_update_tar) + F.mse_loss(q2_pred, q_update_tar)
        self.optimizer_Q.zero_grad()
        loss_q.backward(retain_graph=True)
        self.optimizer_Q.step()

        self.logger_loss_q = loss_q.item()
        self.update_count += 1

        if self.update_count % self.train_policy_delay == 0:
            a_new, a_new_logprob = self.policy(obs)
            loss_policy = (self.alpha * a_new_logprob - self.value.call_Q1(obs, a_new)).mean()
            self.optimizer_policy.zero_grad()
            loss_policy.backward()
            self.optimizer_policy.step()

            a_new_logprob = torch.tensor(a_new_logprob.tolist(), requires_grad=False, device=self.device)
            loss_alpha = (- torch.exp(self.log_alpha) * (a_new_logprob + self.target_entropy)).mean()
            self.optimizer_alpha.zero_grad()
            loss_alpha.backward()
            self.optimizer_alpha.step()

            self.alpha = torch.exp(self.log_alpha)

            soft_update(self.value, self.value_tar, self.tau)

            self.logger_alpha = self.alpha.item()
            self.logger_loss_alpha = loss_alpha.item()
            self.logger_loss_policy = loss_policy.item()
            
        return {
            'loss_q': self.logger_loss_q, 
            'loss_policy': self.logger_loss_policy, 
            'loss_alpha': self.logger_loss_alpha, 
            'alpha': self.logger_alpha
        }

    def evaluate(self, env, action_high, episodes: int, obs_filter: MeanStdFilter) -> float:
        reward = 0
        for i_episode in range(episodes):
            done = False
            obs = env.reset()
            obs = obs_filter(obs)
            while not done:
                action = self.policy.act(obs, with_noise=False)
                next_obs, r, done, info = env.step(action * action_high)
                reward += r
                next_obs = obs_filter(next_obs)
                obs = next_obs
        return reward / episodes

    def save_policy(self, path: str, remark: str) -> None:
        check_path(path)
        torch.save(self.policy.state_dict(), path+f'policy_{remark}')
        print(f"-------Model of all individuals saved to {path}-------")



from absl import app
from absl import flags
import numpy as np
from tqdm import tqdm
import pybullet as p  # pytype: disable=import-error

from locomotion.envs import env_builder
from locomotion.robots import a1
from locomotion.robots import laikago
from locomotion.robots import robot_config

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
            'policy_hidden_layers': [128, 128],
            'value_hidden_layers': [128, 128],
            'a_min': -1.0,
            'a_max': 1.0,
            'logstd_min': -2,
            'logstd_max': 20,
        },
        'buffer_size': 1e6,
        'lr': 0.0003,
        'gamma': 0.99,
        'tau': 0.001,
        'batch_size': 256,
        'initial_alpha': 1,
        'train_policy_delay': 2,
        'device': 'cuda',
        'max_timesteps': 5e6,
        'eval_interval': 50000,
        'eval_episode': 10,
        'result_path': '/home/xukang/Project/locomotion_simulation/locomotion/results/test_sac/'
    }

    robot = ROBOT_CLASS_MAP[FLAGS.robot_type]
    motor_control_mode = MOTOR_CONTROL_MODE_MAP[FLAGS.motor_control_mode]
    env = env_builder.build_regular_env(robot,
                                        motor_control_mode=motor_control_mode,
                                        enable_rendering=True,
                                        on_rack=FLAGS.on_rack,
                                        wrap_trajectory_generator=False)
    action_low, action_high = env.action_space.low, env.action_space.high
    action_median = (action_low + action_high) / 2.

    config['model_config'].update({
        'o_dim': env.observation_space.shape[0],
        'a_dim': env.action_space.shape[0]
    })
    agent = SAC(config)
    obs_filter = MeanStdFilter(shape=config['model_config']['o_dim'])
    buffer = Buffer(memory_size=config['buffer_size'])


    total_step, total_episode = 0, 0
    best_score = 0
    obs = env.reset()
    while total_step < config['max_timesteps']:
        normalized_obs = obs_filter(obs)
        action = agent.policy.act(normalized_obs, True)
        transferred_action = action * action_high[0]
        next_obs, reward, done, info = env.step(transferred_action)

        buffer.save_trans((obs, action, reward, done, next_obs))
        loss_dict = agent.train_ac(buffer, obs_filter)

        if done:
            total_episode += 1
            obs = env.reset()
        else:
            obs = next_obs

        if total_step % config['eval_interval'] == 0:
            eval_score = agent.evaluate(env, action_high[0], config['eval_episode'], obs_filter)
            if eval_score > best_score:
                agent.save_policy(config['result_path'], 'best')
                obs_filter.save_filter(config['result_path'], 'best')
                best_score = eval_score

            print(f"Step: {total_step} Episode: {total_episode} Eval_Return: {eval_score} Loss: {[key+': '+value for key, value in list(loss_dict.items())]}")

        total_step += 1
