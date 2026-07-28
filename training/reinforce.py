"""Monte Carlo policy gradient (REINFORCE), written by hand.

Stable-Baselines3 has no REINFORCE. Network shape matches the SB3 MlpPolicy used
by PPO and A2C so the comparison is about the algorithm, not about capacity.

Logs the same Monitor CSV schema as the SB3 runs. If that drifts, every figure
in analysis/plots.py needs a special case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ReinforceConfig:
    lr: float = 1e-3
    gamma: float = 0.99
    baseline: str = "value"  # "none" or "value"
    ent_coef: float = 0.0
    episodes_per_update: int = 1
    normalise_advantage: bool = False
    net_arch: tuple[int, ...] = (256, 256)
    seed: int = 0


class PolicyNetwork(nn.Module):
    """Categorical policy over the 18 discrete actions."""

    def __init__(self, obs_dim: int, n_actions: int, net_arch: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for width in net_arch:
            layers += [nn.Linear(last, width), nn.Tanh()]
            last = width
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.net(obs))


class ValueNetwork(nn.Module):
    """State value baseline. Separate head, trained on MSE against returns."""

    def __init__(self, obs_dim: int, net_arch: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for width in net_arch:
            layers += [nn.Linear(last, width), nn.Tanh()]
            last = width
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    """Reward-to-go, computed backwards. No bootstrapping, this is Monte Carlo."""
    out = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running = rewards[i] + gamma * running
        out[i] = running
    return out


def train_reinforce(env, config: ReinforceConfig, total_timesteps: int, run_dir):
    """Collect whole episodes, then take one gradient step per batch.

    Logs policy entropy every update, since the report needs the entropy curve
    alongside PPO and A2C.
    """
    # TODO: rollout loop, returns, optional baseline subtraction, optional
    # advantage normalisation, policy loss -logp * advantage - ent_coef * entropy,
    # value loss MSE, Monitor CSV rows, TensorBoard scalars.
    raise NotImplementedError
