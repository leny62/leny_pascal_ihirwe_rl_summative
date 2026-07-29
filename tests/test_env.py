"""Environment acceptance criteria.

These are written before the implementation on purpose. Everything is skipped
until UmurimaEnv.reset stops raising, then the skip marker comes off.
"""

import gymnasium as gym
import numpy as np
import pytest

import environment  # noqa: F401  registers Umurima-v0
from environment.custom_env import (
    HORIZON,
    N_ZONES,
    OBS_DIM,
    Z_DEPLETION,
    ZONE_STRIDE,
    Action,
)


def _env_is_implemented() -> bool:
    try:
        gym.make("Umurima-v0").reset(seed=0)
    except NotImplementedError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _env_is_implemented(), reason="environment not implemented yet"
)


def test_passes_the_gymnasium_env_checker():
    from gymnasium.utils.env_checker import check_env

    check_env(gym.make("Umurima-v0").unwrapped, skip_render_check=True)


def test_spaces_match_the_spec():
    env = gym.make("Umurima-v0")
    assert env.action_space.n == len(Action) == 18
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.observation_space.dtype == np.float32


def test_same_seed_gives_identical_trajectories():
    actions = [3, 10, 0, 12, 7, 15, 0, 13]
    runs = []
    for _ in range(2):
        env = gym.make("Umurima-v0")
        obs, _ = env.reset(seed=42)
        trace = [obs.copy()]
        for a in actions:
            obs, _, _, _, _ = env.step(a)
            trace.append(obs.copy())
        runs.append(np.array(trace))
    np.testing.assert_array_equal(runs[0], runs[1])


def test_different_seeds_diverge():
    env = gym.make("Umurima-v0")
    a, _ = env.reset(seed=1)
    b, _ = env.reset(seed=2)
    assert not np.allclose(a, b)


def test_observations_stay_inside_the_declared_box():
    env = gym.make("Umurima-v0")
    obs, _ = env.reset(seed=0)
    for _ in range(HORIZON):
        assert env.observation_space.contains(obs)
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break


def _run(env, policy, seed):
    """Roll a callable policy to episode end, return the final info."""
    obs, _ = env.reset(seed=seed)
    while True:
        obs, _, terminated, truncated, info = env.step(policy(obs))
        if terminated or truncated:
            return info


def _keep_alive(obs: np.ndarray) -> int:
    """Irrigate whenever any bench is drying, never harvest. Reaches the horizon
    with the crop still standing, which is what truncation means here."""
    depletion = [obs[z * ZONE_STRIDE + Z_DEPLETION] for z in range(N_ZONES)]
    return int(Action.IRRIGATE_ALL_LIGHT) if max(depletion) > 0.35 else int(Action.IDLE)


def _causes_over_seeds(env, policy, seeds):
    return {_run(env, policy, s)["termination_cause"] for s in seeds}


def _mean_return(env, policy, seeds):
    totals = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        total = 0.0
        while True:
            obs, reward, terminated, truncated, _ = env.step(policy(obs))
            total += reward
            if terminated or truncated:
                totals.append(total)
                break
    return float(np.mean(totals))


def test_immature_harvest_does_not_end_the_season():
    """Pods do not exist early, so a harvest called on day one picks nothing and
    the season carries on."""
    env = gym.make("Umurima-v0")
    env.reset(seed=0)
    for _ in range(5):
        _, _, terminated, truncated, info = env.step(int(Action.HARVEST_ALL))
        assert not terminated and not truncated
        assert info["harvested_fraction"] == 0.0


def test_quitting_early_is_not_a_winning_strategy():
    """Tripwire for a local optimum that once dominated this environment.

    When harvest was ungated, ending the episode on day one scored better than
    exploring, so every training run converged on it. If this regresses, the
    sweep produces forty rows of an agent that refuses to farm.
    """
    env = gym.make("Umurima-v0")
    quitting = _mean_return(env, lambda o: int(Action.HARVEST_ALL), range(20))
    random_play = _mean_return(env, lambda o: env.action_space.sample(), range(20))
    assert quitting < random_play


def test_all_termination_causes_are_reachable():
    """The rubric wants the agent to explore edge cases, so each ending must be
    reachable. Pure random almost always harvests early, so reachability is shown
    with targeted policies. Spend-rate against opening cash is seed dependent, so
    each cause is checked for presence across a seed range, not on one seed."""
    env = gym.make("Umurima-v0")
    from training.baselines import ScriptedAgronomist

    scripted = ScriptedAgronomist()
    seeds = range(30)

    assert "harvested" in _causes_over_seeds(env, lambda o: scripted.predict(o)[0], seeds)
    assert "crop_failure" in _causes_over_seeds(env, lambda o: int(Action.IDLE), seeds)
    # Applying nitrogen every day is the fastest way to burn working capital.
    assert "insolvency" in _causes_over_seeds(env, lambda o: int(Action.APPLY_N_SPLIT), seeds)


def test_random_rollouts_run_without_error():
    """Robustness: many random episodes, no exception, every episode ends."""
    env = gym.make("Umurima-v0")
    for seed in range(300):
        info = _run(env, lambda o: env.action_space.sample(), seed)
        assert "termination_cause" in info


def test_truncation_is_not_reported_as_termination():
    """PPO and A2C bootstrap differently on the two. Collapsing them is a silent bug."""
    env = gym.make("Umurima-v0")
    obs, _ = env.reset(seed=7)
    terminated = truncated = False
    for _ in range(HORIZON):
        obs, _, terminated, truncated, _ = env.step(_keep_alive(obs))
        if terminated or truncated:
            break
    assert truncated
    assert not terminated


def test_scripted_agronomist_beats_random():
    """If this fails the reward weights are miscalibrated, fix that before training."""
    from training.baselines import RandomPolicy, ScriptedAgronomist

    env = gym.make("Umurima-v0")
    scores = {}
    for name, policy in [
        ("random", RandomPolicy(env.action_space)),
        ("scripted", ScriptedAgronomist()),
    ]:
        totals = []
        for seed in range(30):
            obs, _ = env.reset(seed=seed)
            total = 0.0
            while True:
                action, _ = policy.predict(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            totals.append(total)
        scores[name] = float(np.mean(totals))
    assert scores["scripted"] > scores["random"]
