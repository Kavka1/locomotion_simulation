from typing import Dict, List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


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
        logprob = dist.log_prob(arctanh_action).sum(dim=-1, keepdim=True)
        squashed_correction = torch.log(1 - action**2 + 1e-6).sum(dim=-1, keepdim=True)
        logprob = logprob - squashed_correction
        
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



class FixStdGaussianPolicy(nn.Module):
    def __init__(self, o_dim: int, a_dim: int, hiddens: List[int], fix_std: float, device: torch.device) -> None:
        super(FixStdGaussianPolicy, self).__init__()
        prev_layers = [o_dim] + hiddens + [a_dim]
        prev_module = []
        for i in range(len(prev_layers)-1):
            prev_module.append(nn.Linear(prev_layers[i], prev_layers[i+1]))
            prev_module.append(nn.Tanh())
        self.model = nn.Sequential(*prev_module)
        self.ac_std = torch.tensor([fix_std]*a_dim, dtype=torch.float32, device=device)

        # Orthogonal Initialize Weights
        #for m in self.model.modules():
        #    if isinstance(m, nn.Linear):
        #        nn.init.orthogonal_(m.weight)


    def __call__(self, obs: torch.tensor) -> torch.distributions.Distribution:
        mean = self.model(obs)
        dist = Normal(mean, self.ac_std)
        return dist

    def load_model(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
        print(f"- - - Loaded model from {path} - - -")


class VFunction(nn.Module):
    def __init__(self, o_dim: int, hiddens: List[int]) -> None:
        super(VFunction, self).__init__()
        prev_layers = [o_dim] + hiddens + [1]
        modules = []
        for i in range(len(prev_layers) - 1):
            modules.append(nn.Linear(prev_layers[i], prev_layers[i+1]))
            if i != len(prev_layers) - 2:
                modules.append(nn.Tanh())
        self.model = nn.Sequential(*modules)

        # Orthogonal Initialize Weights
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)

    def __call__(self, obs: torch.tensor) -> torch.tensor:
        return self.model(obs)