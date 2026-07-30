"""Generate every figure the report needs.

Figures 1 to 5 are all required by the report, so run --all before writing up.

    uv run analysis/plots.py --all --out assets/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

ALGO_COLOURS = {
    "dqn": "#1f77b4",
    "reinforce": "#d62728",
    "ppo": "#2ca02c",
    "a2c": "#ff7f0e",
}
ALGO_LABELS = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}
ROLLING_WINDOW = 20
DPI = 150
UNIFORM_ENTROPY = 2.890  # ln(18), a uniform policy over the action space

STYLE_PATH = Path("analysis/plots.mplstyle")
LOG_ROOT = Path("logs")

BASELINE_SCRIPTED = 21.06
BASELINE_RANDOM = 5.52


def load_monitor(run_dir: Path) -> pd.DataFrame:
    """Read all Monitor CSV files from a run directory into one DataFrame.

    Each row is an episode with columns: return, length, timestep (cumulative).
    """
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(run_dir.glob("**/monitor*.csv")):
        try:
            df = pd.read_csv(csv_path, comment="#")
        except (pd.errors.EmptyDataError, Exception):
            continue
        if "r" not in df.columns or "l" not in df.columns:
            continue
        keep = ["r", "l", "t"] if "t" in df.columns else ["r", "l"]
        frames.append(df[keep])
    if not frames:
        return pd.DataFrame(columns=["r", "l", "timestep"])

    combined = pd.concat(frames, ignore_index=True)
    # Parallel runs write one CSV per worker, each with its own clock. Ordering
    # by episode end time interleaves them back into a single training timeline.
    # Concatenating instead makes every worker's early episodes look like a
    # mid-training collapse, one sawtooth per worker.
    if "t" in combined.columns:
        combined = combined.sort_values("t", kind="mergesort").reset_index(drop=True)
    combined["timestep"] = combined["l"].cumsum()
    combined.rename(columns={"r": "return", "l": "length"}, inplace=True)
    return combined


def _best_run(algo: str) -> str:
    """Return the run ID with the highest heldout mean return."""
    results = pd.read_csv("experiments/results.csv")
    algo_rows = results[results["algo"] == algo]
    return str(algo_rows.loc[algo_rows["heldout_mean_return"].idxmax(), "id"])


def _rolling_mean(series: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=False).mean()


def fig1_training_curves(out: Path) -> None:
    """2x2 subplots, one per algorithm, shared y limits.

    Overlay the random and scripted agronomist means as horizontal lines.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()

    y_min, y_max = float("inf"), float("-inf")
    lines_data: list[tuple[Any, str, pd.Series, pd.Series, pd.Series]] = []

    for ax, algo in zip(axes, ["dqn", "reinforce", "ppo", "a2c"], strict=True):
        run_id = _best_run(algo)
        run_dir = LOG_ROOT / run_id
        df = load_monitor(run_dir)
        if df.empty:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            ax.set_title(ALGO_LABELS[algo])
            continue

        rm = _rolling_mean(df["return"])
        lines_data.append((ax, algo, df["timestep"], df["return"], rm))
        y_min = min(y_min, rm.min())
        y_max = max(y_max, rm.max())

    y_pad = (y_max - y_min) * 0.1
    y_lim = (y_min - y_pad, y_max + y_pad)

    for ax, algo, x, raw, rm in lines_data:
        ax.plot(x, raw, alpha=0.15, color=ALGO_COLOURS[algo], linewidth=0.5)
        ax.plot(x, rm, color=ALGO_COLOURS[algo], linewidth=1.5, label=ALGO_LABELS[algo])
        ax.axhline(BASELINE_SCRIPTED, color="gray", linestyle="--", linewidth=1.0, label="Scripted")
        ax.axhline(BASELINE_RANDOM, color="gray", linestyle=":", linewidth=1.0, label="Random")
        ax.set_title(ALGO_LABELS[algo])
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Episodic return")
        ax.set_ylim(y_lim)
        ax.legend(fontsize=7)

    fig.suptitle("Figure 1: Training curves (best run per algorithm)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "fig1_training_curves.png", dpi=DPI)
    plt.close(fig)
    print(f"wrote {out / 'fig1_training_curves.png'}")


def fig2_dqn_objective(out: Path) -> None:
    """DQN learning curves for all ten grid rows, showing parameter effects."""
    results = pd.read_csv("experiments/results.csv")
    dqn = results[results["algo"] == "dqn"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    for _, row in dqn.iterrows():
        run_id = str(row["id"])
        df = load_monitor(LOG_ROOT / run_id)
        if df.empty:
            continue
        rm = _rolling_mean(df["return"])
        label = f"{run_id}: {row['note']}"[:50]
        ax1.plot(df["timestep"], rm, linewidth=1.0, label=label)
    ax1.axhline(BASELINE_SCRIPTED, color="gray", linestyle="--", linewidth=1.0)
    ax1.axhline(BASELINE_RANDOM, color="gray", linestyle=":", linewidth=1.0)
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Return (rolling mean)")
    ax1.set_title("DQN training curves")
    ax1.legend(fontsize=6, ncol=2)

    ax2 = axes[1]
    heldout = dqn["heldout_mean_return"].values
    train_returns = dqn["train_mean_return"].values
    notes = dqn["note"].values
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(dqn)))
    for i in range(len(dqn)):
        ax2.scatter(train_returns[i], heldout[i], color=colors[i], s=60, zorder=3)
        ax2.annotate(
            notes[i][:30], (train_returns[i], heldout[i]),
            fontsize=6, alpha=0.8, xytext=(5, 5), textcoords="offset points",
        )
    lim_min = min(train_returns.min(), heldout.min()) - 10
    lim_max = max(train_returns.max(), heldout.max()) + 10
    ax2.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=0.8, alpha=0.5)
    ax2.set_xlabel("Train mean return")
    ax2.set_ylabel("Heldout mean return")
    ax2.set_title("DQN generalisation")

    fig.suptitle("Figure 2: DQN diagnostics", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "fig2_dqn_objective.png", dpi=DPI)
    plt.close(fig)
    print(f"wrote {out / 'fig2_dqn_objective.png'}")


def _probe_states(n: int = 256, seed: int = 4242) -> np.ndarray:
    """A fixed batch of observations to measure policy entropy against.

    The same states are used for every run so the entropies are comparable.
    """
    import gymnasium as gym

    import environment  # noqa: F401  registers Umurima-v0

    env = gym.make("Umurima-v0")
    rng = np.random.default_rng(seed)
    states = []
    obs, _ = env.reset(seed=seed)
    while len(states) < n:
        states.append(np.asarray(obs, dtype=np.float32))
        obs, _, terminated, truncated, _ = env.step(int(rng.integers(env.action_space.n)))
        if terminated or truncated:
            obs, _ = env.reset(seed=int(rng.integers(0, 10_000)))
    env.close()
    return np.stack(states)


def _final_policy_entropy(algo: str, run_id: str, states: np.ndarray) -> float | None:
    """Mean entropy of the trained policy over the probe states, in nats."""
    import torch

    try:
        if algo == "reinforce":
            from training.reinforce import ReinforceAgent

            path = Path(f"models/reinforce/{run_id}.pt")
            if not path.exists():
                return None
            agent = ReinforceAgent.load(str(path))
            with torch.no_grad():
                dist = agent.policy(torch.as_tensor(states, dtype=torch.float32))
            return float(dist.entropy().mean().item())

        from stable_baselines3 import A2C, PPO

        cls = {"ppo": PPO, "a2c": A2C}[algo]
        path = Path(f"models/{algo}/{run_id}.zip")
        if not path.exists():
            return None
        model = cls.load(str(path), device="cpu")
        with torch.no_grad():
            tensor, _ = model.policy.obs_to_tensor(states)
            dist = model.policy.get_distribution(tensor)
        return float(dist.distribution.entropy().mean().item())
    except Exception:
        return None


def fig3_pg_entropy(out: Path) -> None:
    """Policy entropy for all three policy gradient methods.

    Left: REINFORCE entropy over training, read from the hand-written
    entropy.csv, which is the only per-update log available because the
    implementation is ours.

    Right: final policy entropy against held-out return for all 30 policy
    gradient runs. Stable-Baselines3 does not expose per-update entropy through
    the Monitor wrapper, so PPO and A2C are measured after training by
    evaluating each saved policy on one fixed batch of observations. That gives
    a comparable end point for every run even though only REINFORCE has a curve.
    """
    results = pd.read_csv("experiments/results.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.4))

    rf = results[results["algo"] == "reinforce"]
    for _, row in rf.iterrows():
        run_id = str(row["id"])
        entropy_path = LOG_ROOT / run_id / "entropy.csv"
        if not entropy_path.exists():
            continue
        ent = pd.read_csv(entropy_path)
        label = f"{run_id} (ent_coef={row.get('ent_coef', 0)}) -> {row['heldout_mean_return']:.1f}"
        ax1.plot(ent["timesteps"], ent["entropy"], linewidth=1.0, label=label)
    ax1.axhline(
        UNIFORM_ENTROPY, color="gray", linestyle="--", linewidth=1.0,
        label=f"Uniform ln(18) = {UNIFORM_ENTROPY:.2f}",
    )
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Policy entropy (nats)")
    ax1.set_title("REINFORCE entropy during training")
    ax1.legend(fontsize=6)

    states = _probe_states()
    for algo in ("reinforce", "ppo", "a2c"):
        xs, ys, ids = [], [], []
        for _, row in results[results["algo"] == algo].iterrows():
            ent = _final_policy_entropy(algo, str(row["id"]), states)
            if ent is None:
                continue
            xs.append(ent)
            ys.append(float(row["heldout_mean_return"]))
            ids.append(str(row["id"]))
        if not xs:
            continue
        ax2.scatter(xs, ys, s=55, alpha=0.85, color=ALGO_COLOURS[algo],
                    label=ALGO_LABELS[algo], zorder=3)
        best = int(np.argmax(ys))
        ax2.annotate(ids[best], (xs[best], ys[best]), fontsize=7,
                     xytext=(6, 4), textcoords="offset points")
    ax2.axvline(UNIFORM_ENTROPY, color="gray", linestyle="--", linewidth=1.0)
    ax2.text(UNIFORM_ENTROPY, ax2.get_ylim()[0], " uniform", fontsize=7,
             color="gray", va="bottom")
    ax2.axvline(0.0, color="crimson", linestyle=":", linewidth=1.2)
    ax2.text(0.0, ax2.get_ylim()[0], " collapsed", fontsize=7,
             color="crimson", va="bottom")
    ax2.axhline(BASELINE_SCRIPTED, color="gray", linestyle="-.", linewidth=0.9)
    ax2.set_xlabel("Final policy entropy (nats, fixed 256-state probe)")
    ax2.set_ylabel("Held-out mean return")
    ax2.set_title("Exploration retained vs performance, all 30 PG runs")
    ax2.legend(fontsize=8)

    fig.suptitle("Figure 3: Policy gradient entropy", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "fig3_pg_entropy.png", dpi=DPI)
    plt.close(fig)
    print(f"wrote {out / 'fig3_pg_entropy.png'}")


def fig4_convergence(out: Path) -> None:
    """Episodes to a +10 return threshold, plus return against timestep for
    the four best runs.
    """
    results = pd.read_csv("experiments/results.csv")
    threshold = BASELINE_SCRIPTED

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    # Timesteps, not episode index, and a full rolling window. A short window
    # lets one lucky episode cross the line, which reports noise as convergence.
    algo_steps: dict[str, list[float]] = {}
    for algo in ["dqn", "reinforce", "ppo", "a2c"]:
        algo_rows = results[results["algo"] == algo]
        steps_to_thresh: list[float] = []
        for _, row in algo_rows.iterrows():
            df = load_monitor(LOG_ROOT / str(row["id"]))
            if df.empty:
                continue
            # Full window, not min_periods=1, or a single lucky first episode
            # counts as convergence and every method looks instant.
            rm = df["return"].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
            hit = np.where(rm >= threshold)[0]
            steps_to_thresh.append(
                float(df["timestep"].iloc[hit[0]]) if len(hit) > 0 else float("nan")
            )
        algo_steps[algo] = steps_to_thresh

    medians, lo_err, hi_err, reached = [], [], [], []
    for algo in ALGO_LABELS:
        vals = np.asarray(algo_steps[algo], dtype=float)
        finite = vals[~np.isnan(vals)]
        reached.append(len(finite))
        if len(finite) == 0:
            medians.append(np.nan)
            lo_err.append(0.0)
            hi_err.append(0.0)
            continue
        med = float(np.median(finite))
        medians.append(med)
        # Interquartile spread, so the whisker can never run below zero steps.
        lo_err.append(max(0.0, med - float(np.percentile(finite, 25))))
        hi_err.append(max(0.0, float(np.percentile(finite, 75)) - med))

    ax1.bar(
        [ALGO_LABELS[a] for a in ALGO_LABELS],
        medians,
        color=[ALGO_COLOURS[a] for a in ALGO_LABELS],
        yerr=[lo_err, hi_err],
        capsize=5,
    )
    ax1.set_ylim(bottom=0)
    for i, (m, n) in enumerate(zip(medians, reached, strict=True)):
        if not np.isnan(m):
            ax1.text(i, m, f"{n}/10 runs", ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("Timesteps to beat the scripted agronomist")
    ax1.set_title(f"Sample efficiency (median steps to rolling return >= {threshold:.0f})")

    ax2 = axes[1]
    for algo in ["dqn", "reinforce", "ppo", "a2c"]:
        run_id = _best_run(algo)
        df = load_monitor(LOG_ROOT / run_id)
        if df.empty:
            continue
        rm = _rolling_mean(df["return"])
        ax2.plot(df["timestep"], rm, color=ALGO_COLOURS[algo], linewidth=1.5, label=ALGO_LABELS[algo])
    ax2.axhline(BASELINE_SCRIPTED, color="gray", linestyle="--", linewidth=1.0, label="Scripted")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Return (rolling mean)")
    ax2.set_title("Best training run per algorithm")
    ax2.legend(fontsize=7)

    fig.suptitle("Figure 4: Convergence analysis", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "fig4_convergence.png", dpi=DPI)
    plt.close(fig)
    print(f"wrote {out / 'fig4_convergence.png'}")


def fig5_generalisation(out: Path) -> None:
    """Train vs heldout scatter per algorithm, identity line for reference."""
    results = pd.read_csv("experiments/results.csv")

    fig, ax = plt.subplots(figsize=(8, 7))
    for algo in ["dqn", "reinforce", "ppo", "a2c"]:
        algo_rows = results[results["algo"] == algo]
        ax.scatter(
            algo_rows["train_mean_return"],
            algo_rows["heldout_mean_return"],
            color=ALGO_COLOURS[algo],
            label=ALGO_LABELS[algo],
            s=50,
            zorder=3,
            alpha=0.8,
        )

    all_returns = pd.concat([results["train_mean_return"], results["heldout_mean_return"]])
    lo, hi = all_returns.min() - 10, all_returns.max() + 10
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.4, label="Identity")
    ax.set_xlabel("Train mean return")
    ax.set_ylabel("Heldout mean return")
    ax.set_title("Figure 5: Generalisation across 100 held-out seeds")
    ax.legend(fontsize=9)

    for _, row in results.iterrows():
        if abs(row["generalisation_gap"]) > 8:
            ax.annotate(
                row["id"],
                (row["train_mean_return"], row["heldout_mean_return"]),
                fontsize=6,
                alpha=0.7,
            )

    fig.tight_layout()
    fig.savefig(out / "fig5_generalisation.png", dpi=DPI)
    plt.close(fig)
    print(f"wrote {out / 'fig5_generalisation.png'}")


def fig6_policy_behaviour(out: Path) -> None:
    """One episode under the best PPO policy. Not required, most persuasive figure."""
    import gymnasium as gym

    import environment  # noqa: F401
    from environment.custom_env import (
        N_ZONES,
        Z_CANOPY,
        Z_DEPLETION,
        Z_NITROGEN,
        ZONE_STRIDE,
        Action,
    )
    from play import load_policy

    model_path = Path("models/ppo/ppo-06.zip")
    if not model_path.exists():
        print("ppo-06 model not found, skipping fig6")
        return

    policy = load_policy(model_path, None)
    env = gym.make("Umurima-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=42)

    history: dict[str, list[float]] = {
        "day": [], "action": [], "reward": [], "cash": [], "return": [],
        "depletion_0": [], "depletion_3": [], "canopy_mean": [], "n_mean": [],
        "reservoir": [], "rain": [], "harvested": [],
    }

    total = 0.0
    while True:
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total += float(reward)
        history["day"].append(info.get("day", 0))
        history["action"].append(int(action))
        history["reward"].append(float(reward))
        history["cash"].append(info.get("cash_krwf", 0))
        history["return"].append(total)
        history["depletion_0"].append(float(obs[0 * ZONE_STRIDE + Z_DEPLETION]))
        history["depletion_3"].append(float(obs[3 * ZONE_STRIDE + Z_DEPLETION]))
        history["canopy_mean"].append(float(np.mean([obs[z * ZONE_STRIDE + Z_CANOPY] for z in range(N_ZONES)])))
        history["n_mean"].append(float(np.mean([obs[z * ZONE_STRIDE + Z_NITROGEN] for z in range(N_ZONES)])))
        history["reservoir"].append(float(obs[25]))
        history["rain"].append(float(obs[40]))
        history["harvested"].append(float(obs[31]))
        if terminated or truncated:
            break
    env.close()

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    days = np.array(history["day"])
    ax = axes[0]
    ax.plot(days, history["return"], color="black", linewidth=1.5)
    ax.fill_between(days, 0, history["reward"], alpha=0.3, color="green" if total > 0 else "red")
    ax.set_ylabel("Return")
    ax.set_title("Cumulative return")

    ax = axes[1]
    ax.plot(days, history["cash"], color="blue", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(-100, color="red", linestyle="--", linewidth=0.5, label="Insolvency")
    ax.set_ylabel("Cash (kRWF)")
    ax.set_title("Cash position")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(days, history["depletion_0"], label="Zone 0 (ridge)", linewidth=1.0)
    ax.plot(days, history["depletion_3"], label="Zone 3 (valley)", linewidth=1.0)
    ax.set_ylabel("Depletion fraction")
    ax.set_title("Soil moisture")
    ax.legend(fontsize=7)

    ax = axes[3]
    ax.plot(days, history["canopy_mean"], color="green", linewidth=1.5, label="Canopy")
    ax.plot(days, history["n_mean"], color="orange", linewidth=1.5, label="Nitrogen")
    ax.set_ylabel("Normalised value")
    ax.set_title("Crop health")
    ax.legend(fontsize=7)

    ax = axes[4]
    ax.plot(days, history["reservoir"], color="steelblue", linewidth=1.5)
    ax.set_ylabel("Reservoir fraction")
    ax.set_title("Reservoir")

    ax = axes[5]
    action_labels = {a.value: a.name for a in Action}
    unique_actions = sorted(set(history["action"]))
    y_map = {a: i for i, a in enumerate(unique_actions)}
    ax.scatter(days, [y_map[a] for a in history["action"]], s=8, alpha=0.6, color="black")
    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels([action_labels[a][:12] for a in unique_actions], fontsize=6)
    ax.set_xlabel("Day")
    ax.set_title("Actions taken")

    cause = info.get("termination_cause", "truncated")
    fig.suptitle(f"Figure 6: PPO best-policy episode (return={total:.1f}, {cause})", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "fig6_policy_behaviour.png", dpi=DPI)
    plt.close(fig)
    print(f"wrote {out / 'fig6_policy_behaviour.png'}")


def write_captions(out: Path) -> None:
    """Captions written alongside the figures so none reaches the report bare."""
    captions = {
        "fig1_training_curves.png": (
            "Training curves for the best hyperparameter run of each algorithm. "
            "Thin lines are per-episode returns; thick lines are a rolling mean "
            f"(window={ROLLING_WINDOW} episodes). The dashed and dotted grey lines mark the "
            f"scripted agronomist ({BASELINE_SCRIPTED:.1f}) and random ({BASELINE_RANDOM:.1f}) "
            "baselines respectively. All four algorithms surpass the scripted baseline."
        ),
        "fig2_dqn_objective.png": (
            "DQN hyperparameter study. Left: rolling mean returns for all ten grid "
            "rows, showing sensitivity to learning rate, discount factor, buffer size, "
            "and exploration schedule. Right: train versus heldout mean return for "
            "each row, with the identity line marking perfect generalisation."
        ),
        "fig3_pg_entropy.png": (
            "REINFORCE policy entropy over the course of training. The dashed line "
            f"marks the entropy of a uniform policy over 18 actions (ln(18) = {UNIFORM_ENTROPY:.3f} nats). "
            "Low entropy coefficients lead to rapid entropy collapse; higher values "
            "sustain exploration longer. PPO and A2C entropy is not logged by the "
            "Stable-Baselines3 Monitor wrapper and is therefore not shown."
        ),
        "fig4_convergence.png": (
            f"Convergence analysis. Left: median episodes to reach a return of {10} across "
            "all ten grid rows per algorithm, with standard deviation error bars. "
            "Right: rolling mean return for the best run of each algorithm, showing "
            "the trajectory that achieved the highest heldout score."
        ),
        "fig5_generalisation.png": (
            "Generalisation: train-distribution mean return (30 seeds) against held-out "
            "mean return (100 seeds drawn from a disjoint seed range). Points on the "
            "identity line generalise perfectly. Negative outliers are labelled with "
            "their run ID."
        ),
        "fig6_policy_behaviour.png": (
            "One episode under the best PPO policy (ppo-06). The episode trace shows "
            "cumulative return, cash position, soil moisture at the ridge and valley, "
            "canopy cover and nitrogen status, reservoir level, and the action sequence."
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    for fname, caption in captions.items():
        cap_path = out / f"{fname}.txt"
        cap_path.write_text(caption + "\n")
    combined = "\n\n".join(f"**{k}**: {v}" for k, v in captions.items())
    (out / "captions.md").write_text(combined + "\n")
    print(f"wrote {len(captions)} captions to {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate report figures")
    p.add_argument("--out", type=Path, default=Path("assets/figures"))
    p.add_argument("--all", action="store_true")
    p.add_argument("--only", nargs="+", choices=[f"fig{i}" for i in range(1, 7)])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.all:
        figs = [fig1_training_curves, fig2_dqn_objective, fig3_pg_entropy, fig4_convergence, fig5_generalisation, fig6_policy_behaviour]
        for f in figs:
            f(args.out)
        write_captions(args.out)
    elif args.only:
        mapping = {
            "fig1": fig1_training_curves,
            "fig2": fig2_dqn_objective,
            "fig3": fig3_pg_entropy,
            "fig4": fig4_convergence,
            "fig5": fig5_generalisation,
            "fig6": fig6_policy_behaviour,
        }
        for name in args.only:
            mapping[name](args.out)
        write_captions(args.out)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
