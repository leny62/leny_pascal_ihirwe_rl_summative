"""Environment acceptance criteria.

These are written before the implementation on purpose. Everything is skipped
until UmurimaEnv.reset stops raising, then the skip marker comes off.
"""

import gymnasium as gym
import numpy as np
import pytest

import environment  # noqa: F401  registers Umurima-v0
from environment.custom_env import HORIZON, OBS_DIM, Action


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


def test_random_rollouts_hit_every_termination_cause():
    """Rubric asks the agent to explore edge cases, so they must be reachable."""
    causes = set()
    env = gym.make("Umurima-v0")
    for seed in range(1000):
        env.reset(seed=seed)
        while True:
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                causes.add(info.get("termination_cause", "truncated"))
                break
    assert {"harvested", "crop_failure", "insolvency", "truncated"} <= causes


def test_truncation_is_not_reported_as_termination():
    """PPO and A2C bootstrap differently on the two. Collapsing them is a silent bug."""
    env = gym.make("Umurima-v0")
    env.reset(seed=7)
    for _ in range(HORIZON):
        _, _, terminated, truncated, _ = env.step(Action.IDLE)
        if truncated:
            assert not terminated
            return
    pytest.fail("idling for the full horizon should truncate")


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
