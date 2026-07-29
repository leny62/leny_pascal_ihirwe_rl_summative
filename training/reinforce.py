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


class ReinforceAgent:
    """Wraps the trained networks in the predict/save/load shape SB3 exposes,
    so evaluation and play.py treat all four algorithms identically."""

    def __init__(self, policy: PolicyNetwork, obs_dim: int, n_actions: int,
                 net_arch: tuple[int, ...]) -> None:
        self.policy = policy
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.net_arch = tuple(net_arch)

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        with torch.no_grad():
            dist = self.policy(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return int(action.item()), None

    def save(self, path) -> None:
        torch.save(
            {
                "state_dict": self.policy.state_dict(),
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "net_arch": list(self.net_arch),
            },
            str(path),
        )

    @classmethod
    def load(cls, path) -> ReinforceAgent:
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        net_arch = tuple(blob["net_arch"])
        policy = PolicyNetwork(blob["obs_dim"], blob["n_actions"], net_arch)
        policy.load_state_dict(blob["state_dict"])
        policy.eval()
        return cls(policy, blob["obs_dim"], blob["n_actions"], net_arch)


def train_reinforce(env, config: ReinforceConfig, total_timesteps: int, run_dir) -> ReinforceAgent:
    """Collect whole episodes, then take one gradient step per batch.

    Logs policy entropy every update, since the report needs the entropy curve
    alongside PPO and A2C.
    """
    torch.manual_seed(config.seed)
    obs_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)

    policy = PolicyNetwork(obs_dim, n_actions, config.net_arch)
    params = list(policy.parameters())
    value_net = None
    if config.baseline == "value":
        value_net = ValueNetwork(obs_dim, config.net_arch)
        params += list(value_net.parameters())
    optimiser = torch.optim.Adam(params, lr=config.lr)

    entropy_log: list[tuple[int, float]] = []
    steps = 0
    obs, _ = env.reset(seed=config.seed)

    while steps < total_timesteps:
        # --- collect a batch of complete episodes -----------------------------
        batch_obs: list[np.ndarray] = []
        batch_actions: list[int] = []
        batch_returns: list[np.ndarray] = []

        for _ in range(config.episodes_per_update):
            ep_obs: list[np.ndarray] = []
            ep_actions: list[int] = []
            ep_rewards: list[float] = []
            while True:
                with torch.no_grad():
                    dist = policy(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
                    action = int(dist.sample().item())
                ep_obs.append(np.asarray(obs, dtype=np.float32))
                ep_actions.append(action)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_rewards.append(float(reward))
                steps += 1
                if terminated or truncated:
                    obs, _ = env.reset()
                    break
            batch_obs.extend(ep_obs)
            batch_actions.extend(ep_actions)
            batch_returns.append(discounted_returns(ep_rewards, config.gamma))

        # --- one gradient step on the whole batch ------------------------------
        obs_t = torch.as_tensor(np.asarray(batch_obs), dtype=torch.float32)
        act_t = torch.as_tensor(batch_actions, dtype=torch.int64)
        ret_t = torch.as_tensor(np.concatenate(batch_returns), dtype=torch.float32)

        dist = policy(obs_t)
        logp = dist.log_prob(act_t)
        entropy = dist.entropy().mean()

        value_loss = torch.zeros(())
        if value_net is not None:
            values = value_net(obs_t)
            advantage = ret_t - values.detach()
            value_loss = nn.functional.mse_loss(values, ret_t)
        else:
            advantage = ret_t

        if config.normalise_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        policy_loss = -(logp * advantage).mean() - config.ent_coef * entropy
        loss = policy_loss + 0.5 * value_loss

        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 0.5)
        optimiser.step()

        entropy_log.append((steps, float(entropy.item())))

    if run_dir is not None:
        from pathlib import Path

        path = Path(run_dir) / "entropy.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = "\n".join(f"{s},{e:.6f}" for s, e in entropy_log)
        path.write_text("timesteps,entropy\n" + rows + "\n")

    policy.eval()
    return ReinforceAgent(policy, obs_dim, n_actions, config.net_arch)
