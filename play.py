"""Watch a policy work a season.

    uv run play.py --model models/pg/ppo_best.zip --episodes 2 --verbose
    uv run play.py --baseline scripted --episodes 50 --no-render

--verbose prints one line per day. The video needs visible terminal output
alongside the GUI, so keep it on when recording.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym

import environment  # noqa: F401  registers Umurima-v0
from environment.custom_env import Action
from training.baselines import RandomPolicy, ScriptedAgronomist

# Simulated days per second. A season is only ~90 days, so 12 fps runs a whole
# episode in under 8 seconds. 4 is slow enough to narrate over.
DEFAULT_RENDER_FPS = 4.0


def load_policy(model_path: Path | None, baseline: str | None):
    if baseline == "random":
        return RandomPolicy(gym.make("Umurima-v0").action_space)
    if baseline == "scripted":
        return ScriptedAgronomist()

    if model_path is None:
        raise ValueError("pass --model or --baseline")

    suffix = model_path.suffix.lower()
    if suffix == ".zip":
        algo = model_path.parent.name
        if algo == "dqn":
            from stable_baselines3 import DQN
            return DQN.load(str(model_path))
        if algo == "ppo":
            from stable_baselines3 import PPO
            return PPO.load(str(model_path))
        if algo == "a2c":
            from stable_baselines3 import A2C
            return A2C.load(str(model_path))
        if algo == "pg":
            stem = model_path.stem
            if stem.startswith("ppo"):
                from stable_baselines3 import PPO
                return PPO.load(str(model_path))
            if stem.startswith("a2c"):
                from stable_baselines3 import A2C
                return A2C.load(str(model_path))
            if stem.startswith("reinforce"):
                from training.reinforce import ReinforceAgent
                return ReinforceAgent.load(str(model_path))
        raise ValueError(f"unknown algorithm for {model_path}")
    if suffix == ".pt":
        from training.reinforce import ReinforceAgent
        return ReinforceAgent.load(str(model_path))
    raise ValueError(f"unknown model suffix: {suffix}")


def run_episodes(
    model_path: Path | None = None,
    baseline: str | None = None,
    episodes: int = 1,
    seed: int | None = None,
    render: bool = True,
    verbose: bool = False,
    fps: float = DEFAULT_RENDER_FPS,
) -> list[float]:
    render_mode = "human" if render else None
    policy = load_policy(model_path, baseline)
    env = gym.make("Umurima-v0", render_mode=render_mode)
    clock = None
    if render:
        import pygame

        clock = pygame.time.Clock()
    returns: list[float] = []

    for ep in range(episodes):
        ep_seed = (seed or 0) + ep if seed is not None else None
        obs, _ = env.reset(seed=ep_seed)
        total = 0.0
        step_count = 0

        while True:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += float(reward)
            step_count += 1

            if render and not env.unwrapped.render_closed:
                env.render()
                clock.tick(fps)

            if verbose:
                act_name = Action(int(action)).name
                print(
                    f"day {info.get('day', step_count):3d} | "
                    f"action {act_name:<22s} | "
                    f"reward {reward:+.3f} | "
                    f"cash {info.get('cash_krwf', 0):+.1f} kRWF | "
                    f"harvested {info.get('harvested_fraction', 0):.0%}"
                )

            if terminated or truncated:
                cause = info.get("termination_cause", "truncated")
                returns.append(total)
                if verbose:
                    print(
                        f"--- episode {ep + 1} ended: {cause} | "
                        f"return {total:.2f} | "
                        f"days {info.get('day', step_count)}"
                    )
                break

    if render and not env.unwrapped.render_closed:
        _hold_final_frame(env, clock)
    env.close()
    return returns


def _hold_final_frame(env, clock) -> None:
    """Keep the window up after the last episode so the end state stays readable.

    Without this the window closes the instant the episode terminates, which at
    a season length of ~90 days is only a few seconds after it opened. The
    renderer owns event handling, so the camera stays interactive here.
    """
    print("Episode complete. Drag to orbit, scroll to zoom, Esc or close to exit.")
    while not env.unwrapped.render_closed:
        env.render()
        clock.tick(60)


def main() -> None:
    p = argparse.ArgumentParser(description="Run a trained policy on Umurima-v0")
    p.add_argument("--model", type=Path)
    p.add_argument("--baseline", choices=("random", "scripted"))
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int)
    p.add_argument("--no-render", dest="render", action="store_false")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_RENDER_FPS,
        help="simulated days per second when rendering; lower is easier to narrate",
    )
    args = p.parse_args()

    if args.model is None and args.baseline is None:
        p.error("pass --model or --baseline")

    returns = run_episodes(
        model_path=args.model,
        baseline=args.baseline,
        episodes=args.episodes,
        seed=args.seed,
        render=args.render,
        verbose=args.verbose,
        fps=args.fps,
    )
    print(f"mean return over {len(returns)} episodes: {sum(returns) / len(returns):.2f}")


if __name__ == "__main__":
    main()
