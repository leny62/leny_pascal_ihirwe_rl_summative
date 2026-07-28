"""Umurima: irrigation and input scheduling environment."""

from gymnasium.envs.registration import register

from environment.custom_env import UmurimaEnv

register(
    id="Umurima-v0",
    entry_point="environment.custom_env:UmurimaEnv",
    max_episode_steps=120,
)

__all__ = ["UmurimaEnv"]
