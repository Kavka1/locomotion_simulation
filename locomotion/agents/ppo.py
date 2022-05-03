from copy import deepcopy
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ray

from locomotion.agents.model import FixStdGaussianPolicy, VFunction
from locomotion.agents.utils import check_path, gae_estimator, Dataset
from locomotion.envs.env_builder import build_regular_env
from locomotion.robots import a1
from locomotion.robots import robot_config


@ray.remote
class Worker(object):
    def __init__(
        self,
        worker_id: int,
        o_dim: int,
        a_dim: int,
        action_std: float,
        policy_hiddens: List,
        value_hiddens: List,
        gamma: float, 
        lamda: float,
        rollout_episodes: int,
        action_bound: int,
        manual_action_scale: int,
    ) -> None:
        self.id = worker_id
        self.policy = FixStdGaussianPolicy(o_dim, a_dim, policy_hiddens, action_std, torch.device('cpu'))
        self.value = VFunction(o_dim, value_hiddens).to(torch.device('cpu'))
        self.env = build_regular_env(
            robot_class=a1.A1,
            motor_control_mode=robot_config.MotorControlMode.TORQUE,
            enable_rendering=False,
            on_rack=False
        )
        self.gamma, self.lamda = gamma, lamda
        self.rollout_episodes = rollout_episodes
        self.action_bound = action_bound
        self.action_scale = manual_action_scale

    def rollout(self, policy_state_dict, value_state_dict) -> Dict:
        self.policy.load_state_dict(policy_state_dict)
        self.value.load_state_dict(value_state_dict)

        all_obs_seq = [] 
        all_a_seq = [] 
        all_logprob_seq = [] 
        all_ret_seq = [] 
        all_adv_seq = []
        total_steps, cumulative_r = 0, 0

        for episode in range(self.rollout_episodes):
            obs_seq, a_seq, r_seq, logprob_seq, value_seq = [], [], [], [], []

            obs = self.env.reset()
            done = False
            while not done:
                obs_tensor = torch.from_numpy(obs).float()
                a_dist = self.policy(obs_tensor)
                value = self.value(obs_tensor)

                a = a_dist.sample()
                log_prob = a_dist.log_prob(a).detach().numpy()
                a = a.detach().numpy()
                
                clipped_a = np.clip(a, -self.action_bound, self.action_bound)
                obs_, r, done, info = self.env.step(clipped_a * self.action_scale)
                
                obs_seq.append(obs)
                a_seq.append(a)
                r_seq.append(r)
                logprob_seq.append(log_prob)
                value_seq.append(value.squeeze(-1).detach().numpy())

                total_steps += 1
                cumulative_r += r
                obs = obs_

            ret_seq, adv_seq = gae_estimator(r_seq, value_seq, self.gamma, self.lamda)
            
            all_obs_seq += obs_seq
            all_a_seq += a_seq
            all_logprob_seq += logprob_seq
            all_ret_seq += ret_seq
            all_adv_seq += adv_seq
        
        return {
            'steps': total_steps,
            'cumulative_r': cumulative_r / self.rollout_episodes,
            'obs': np.stack(all_obs_seq, 0),
            'a': np.stack(all_a_seq, 0),
            'logprob': np.stack(all_logprob_seq, 0),
            'ret': np.array(all_ret_seq, dtype=np.float32)[:, np.newaxis],
            'adv': np.array(all_adv_seq, dtype=np.float32)[:, np.newaxis]
        }



class PPO(object):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.o_dim = config['model_config']['o_dim']
        self.a_dim = config['model_config']['a_dim']
        self.action_bound = config['model_config']['a_max']
        self.manual_action_scale = config['manual_action_scale']

        self.lr = config['lr']
        self.gamma = config['gamma']
        self.lamda = config['lamda']
        self.action_std = config['action_std']
        self.ratio_clip = config['ratio_clip']
        self.temperature_coef = config['temperature_coef']
        self.num_epoch = config['num_epoch']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.policy_hiddens = config['model_config']['policy_hiddens']
        self.value_hiddens = config['model_config']['value_hiddens']
        self.num_worker_rollout = config['eval_episode']
        self.device = torch.device(config['device'])

        self.policy = FixStdGaussianPolicy(self.o_dim, self.a_dim, self.policy_hiddens, self.action_std, torch.device(self.device)).to(self.device)
        self.value = VFunction(self.o_dim, self.value_hiddens).to(self.device)
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=self.lr)

        self.workers = []
        self.total_steps, self.total_episodes = 0, 0
        
    def _init_workers(self, env, rollout_episodes: int) -> None:
        raise NotImplementedError("Initialize workers in the main loop")

    def roll_update(self, ) -> Tuple[float, List, float, float]:
        policy_state_dict_remote = ray.put(deepcopy(self.policy).to(torch.device('cpu')).state_dict())
        value_state_dict_remote = ray.put(deepcopy(self.value).to(torch.device('cpu')).state_dict())

        rollout_remote = [
            worker.rollout.remote(policy_state_dict_remote, value_state_dict_remote) 
            for i, worker in enumerate(self.workers)
        ]
        results = ray.get(rollout_remote)

        train_score, log_loss_pi, log_loss_v, update_count = 0, 0, 0, 0
        worker_scores = []

        data_buffer = {'obs': [], 'a': [], 'logprob': [],'ret': [], 'adv': []}
        for item in results:
            self.total_steps += item['steps']
            self.total_episodes += self.num_worker_rollout * self.num_workers
            train_score += item['cumulative_r']
            
            worker_scores.append(item['cumulative_r'])
            data_buffer['obs'].append(item['obs'])
            data_buffer['a'].append(item['a'])
            data_buffer['logprob'].append(item['logprob'])
            data_buffer['ret'].append(item['ret'])
            data_buffer['adv'].append(item['adv'])
        train_score /= self.num_workers
        
        for key in list(data_buffer.keys()):
            data_buffer[key] = torch.from_numpy(np.concatenate(data_buffer[key], 0)).float().to(self.device)
        all_batch = Dataset(data_buffer)

        for i in range(self.num_epoch):
            for batch in all_batch.iterate_once(self.batch_size):
                o, a, logprob, ret, adv = batch['obs'], batch['a'], batch['logprob'], batch['ret'], batch['adv']
                
                if len(adv) != 1:   # length is 1, the std will be nan
                    adv = (adv - adv.mean()) / (adv.std() + 1e-5)

                dist = self.policy(o)
                value = self.value(o)

                new_logprob = dist.log_prob(a)
                entropy = dist.entropy()
                ratio = torch.exp(new_logprob.sum(-1, keepdim=True) - logprob.sum(-1, keepdim=True))
                surr1 = ratio * adv
                surr2 = torch.clip(ratio, 1-self.ratio_clip, 1 + self.ratio_clip) * adv
                
                loss_pi = (- torch.min(surr1, surr2) - self.temperature_coef * entropy).mean()
                self.optimizer_policy.zero_grad()
                loss_pi.backward()
                nn.utils.clip_grad_norm(self.policy.parameters(), 0.5)
                self.optimizer_policy.step()

                loss_v = 0.5 * F.mse_loss(value, ret)
                self.optimizer_value.zero_grad()
                loss_v.backward()
                nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
                self.optimizer_value.step()

                log_loss_pi += loss_pi.detach().cpu().item()
                log_loss_v += loss_v.detach().cpu().item()
                update_count += 1

        return train_score, worker_scores, log_loss_pi / update_count, log_loss_v / update_count
    
    def evaluation(self, env, num_episodes: int) -> Tuple[float, int]:
        assert num_episodes > 0
        score, steps = 0, 0
        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            while not done:
                obs = torch.from_numpy(obs).float().to(self.device)
                dist = self.policy(obs)
                a = dist.mean.cpu().detach().numpy()
                obs, r, done, info = env.step(a * self.manual_action_scale)
                score += r
                steps += 1
        return score / num_episodes, steps
    
    def save_policy(self, exp_path: str, remark: str) -> None:
        check_path(exp_path)
        model_path = exp_path + f'policy_{remark}'
        torch.save(self.policy.state_dict(), model_path)
        print(f"------- Policy saved to {model_path} ----------")
    