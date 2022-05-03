from typing import Dict, List, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from locomotion.agents.utils import *
from locomotion.agents.model import *


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

    def _init_logger(self) -> None:
        self.logger_loss_q = 0.
        self.logger_loss_policy = 0.
        self.logger_loss_alpha = 0.
        self.logger_alpha = self.alpha.item()
        self.update_count = 0

    def train_ac(self, buffer: Buffer) -> Dict:
        if len(buffer) < self.batch_size:
            return {
            'loss_q': 0, 
            'loss_policy': 0, 
            'loss_alpha': 0, 
            'alpha': self.logger_alpha
        }

        obs, a, r, done, obs_ = buffer.sample(self.batch_size)
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

            self.logger_alpha = self.alpha.item()
            self.logger_loss_alpha = loss_alpha.item()
            self.logger_loss_policy = loss_policy.item()
            
        soft_update(self.value, self.value_tar, self.tau)

        return {
            'loss_q': self.logger_loss_q, 
            'loss_policy': self.logger_loss_policy, 
            'loss_alpha': self.logger_loss_alpha, 
            'alpha': self.logger_alpha
        }

    def evaluate(self, env, action_high, episodes: int) -> float:
        reward = 0
        for i_episode in range(episodes):
            done = False
            obs = env.reset()
            step = 0
            while not done:
                action = self.policy.act(obs, with_noise=False)
                next_obs, r, done, info = env.step(action * action_high)
                reward += r
                obs = next_obs
                step += 1
        return reward / episodes

    def save_policy(self, path: str, remark: str) -> None:
        check_path(path)
        torch.save(self.policy.state_dict(), path+f'policy_{remark}')
        print(f"-------Model of all individuals saved to {path}-------")

