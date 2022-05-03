from typing import List, Dict, Tuple, Union
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import os


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


class Dataset(object):
    # For PPO training
    def __init__(self, data: Dict) -> None:
        super(Dataset, self).__init__()
        self.data = data
        self.n = len(data['ret'])
        self._next_id = 0
        self.shuffle()

    def shuffle(self) -> None:
        perm = np.arange(self.n)
        np.random.shuffle(perm)
        for key in list(self.data.keys()):
            self.data[key] = self.data[key][perm]

    def next_batch(self, batch_size: int) -> Tuple:
        cur_id = self._next_id
        cur_batch_size = min(batch_size, self.n - cur_id)
        self._next_id += cur_batch_size

        batch = dict()
        for key in list(self.data.keys()):
            batch[key] = self.data[key][cur_id: cur_id + cur_batch_size]
        return batch

    def iterate_once(self, batch_size: int) -> Tuple:
        self.shuffle()
        while self._next_id < self.n:
            yield self.next_batch(batch_size)
        self._next_id = 0


def gae_estimator(rewards: List[float], values: List[np.float32], gamma: float, lamda: float) -> Tuple[List, List]:
    ret_seq, adv_seq = [], []
    prev_ret, prev_adv, prev_value = 0., 0., 0.
    length = len(rewards)
    for i in reversed(range(length)):
        ret = rewards[i] + gamma * prev_ret
        delta = rewards[i] + gamma * prev_value - values[i]
        adv = delta + gamma * lamda * prev_adv

        ret_seq.insert(0, ret)
        adv_seq.insert(0, adv)
        
        prev_ret = ret
        prev_value = values[i]
        prev_adv = adv

    return ret_seq, adv_seq