# Umurima: RL for irrigation and input scheduling

A reinforcement learning agent that schedules a season of irrigation, fertiliser,
spray and labour decisions on a one hectare horticulture block on a Rwandan
hillside, and a comparison of four RL methods trained against it.

**Course:** Machine Learning Techniques II, Summative Assignment
**Author:** Leny Pascal Ihirwe

Report: [Report](https://docs.google.com/document/d/1s_DMaKDej38lyw0hKZn0jrFKVR5mTO_mWzu-sYL3Cxk/edit?usp=sharing)
Video: [Demo Video](https://drive.google.com/file/d/1dNKVyiTZj4k9TvHRn1OpK5C5YkptukIp/view?usp=sharing)

## The problem

Smallholder blocks under the Small Scale Irrigation Technology subsidy now have
pumps and drip lines, but scheduling is still done by eye. Irrigating before rain
wastes water and pushes nitrogen below the root zone. Irrigating too late during
flowering costs yield that never comes back. Spraying inside the pre-harvest
interval makes the crop unsellable to an export buyer.

Each of those is a sequential decision under uncertain weather, which is what
reinforcement learning is for.

## Environment

`Umurima-v0` simulates 120 days on a block split into four terrace zones, each
with its own soil depth, slope and water-holding capacity. One action per
simulated day.

| | |
|---|---|
| Observation | `Box(43,)`, float32, continuous |
| Action | `Discrete(18)` |
| Episode length | up to 120 days |
| Reward | marketable revenue at harvest, net of water, nitrogen loss, inputs and quality penalties |

Zones are terrace benches, not grid cells. Each has its own soil depth, slope and
water-holding capacity, and runoff cascades downslope, so irrigating the ridge
partly waters the benches below it and the valley bottom waterlogs if it is
irrigated during a wet spell.

Dynamics come from the FAO-56 dual crop coefficient water balance, thermal-time
canopy development, a mineral nitrogen pool with leaching and biological
fixation, and logistic pest and weed pressure. Weather is generated from a
wet/dry Markov chain with gamma rainfall depths fitted to Rwanda's bimodal
season.

Two details make the scheduling non-trivial. Water stress during flowering is
weighted 2.5 times heavier than elsewhere, because yield lost then is not
recovered by later irrigation, so timing matters more than volume. And spraying
within seven days of harvest breaches the pre-harvest interval, which cuts
marketable quality to 35 percent, a rejected consignment rather than a gradual
penalty.

Pest, weed and soil nitrogen readings degrade with time since the last scouting
visit, so the agent has to decide when paying for information is worth it.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                    # core: environment, training, analysis
uv sync --extra ui         # adds the Streamlit dashboard
uv sync --extra api        # adds the JSON API
uv sync --extra dev        # adds pytest and ruff
```

## Running

### 3D view

Watch the best trained policy work a season:

```bash
uv run main.py
```

The window is interactive: **drag** to orbit, **scroll** to zoom, **R** to reset
the view, **Esc** to quit. The final frame is held until you close it.

A season is only ~90 days, so pick a rate you can narrate over. `--fps` is
simulated days per second:

```bash
uv run main.py --fps 1      # ~90 s per season
uv run main.py --fps 0.5    # ~3 min per season
uv run main.py --no-render  # terminal only
```

The HUD shows day and growth stage, cash, reservoir level, market price,
standing yield, rainfall today and the three day forecast, running water stress,
and the current action. The stage turns amber during flowering, when stress costs
yield that later irrigation cannot recover, and a pre-harvest interval breach is
flagged in red.

### Pick a policy

```bash
uv run play.py --model models/ppo/ppo-06.zip --verbose
uv run play.py --model models/dqn/dqn-07.zip --episodes 3
uv run play.py --model models/reinforce/rf-09.pt --no-render --episodes 50
uv run play.py --baseline scripted --no-render --episodes 40
uv run play.py --baseline random --no-render --episodes 40
```

### Streamlit dashboard

A browser view of the same episode, useful for stepping through a season and
inspecting the audit trail:

```bash
uv run --extra ui streamlit run ui/dashboard.py
```

Choose a policy and seed, then step day by day or leave it on auto-play. It shows
KPI tiles, per-zone moisture, canopy and nitrogen indicators, the action log, the
hash-chained field ledger with its verification status, and the weather forecast.

### JSON API

```bash
uv run --extra api uvicorn api.server:app --reload
```

| Endpoint | Method | Returns |
|---|---|---|
| `/health` | GET | liveness |
| `/episode` | POST | creates an episode (`?seed=&policy=`), returns its id and state |
| `/episode/{id}/state` | GET | block state: four zones, weather, cash, reservoir |
| `/episode/{id}/advance` | POST | steps `?days=N`, returns actions taken and new state |
| `/episode/{id}/ledger` | GET | hash-chained field events with season totals |
| `/episode/{id}/ledger/verify` | GET | recomputes the chain, `false` means a record was altered |

`policy` accepts `scripted`, `random`, or a path to any model file.

### Train

```bash
uv run training/dqn_training.py --run-name dqn-baseline --lr 1e-4 --gamma 0.99
uv run training/pg_training.py --algo ppo --run-name ppo-baseline --lr 3e-4
uv run training/pg_training.py --algo reinforce --run-name rf-baseline --lr 1e-3
uv run training/pg_training.py --algo a2c --run-name a2c-baseline --lr 7e-4
```

### Reproduce the hyperparameter study (40 runs)

```bash
uv run experiments/sweep.py --all --workers 6     # writes experiments/results.csv
uv run analysis/plots.py --all --out assets/figures
uv run analysis/tables.py --out assets/tables
```

### Tests

```bash
uv run --extra dev pytest
```

## Layout

```
environment/
  custom_env.py      Gymnasium environment, spaces, reward, termination
  agronomy.py        soil water balance, canopy, nitrogen, pest and weed models
  stochastics.py     rainfall, evapotranspiration, temperature, market price
  rendering.py       PyOpenGL 3D scene and HUD
  traceability.py    hash-chained per-block event ledger
training/
  dqn_training.py    DQN
  pg_training.py     REINFORCE, PPO, A2C
  reinforce.py       Monte Carlo policy gradient with a learned baseline
  common.py          env factory, seeding, logging callbacks
experiments/
  sweep.py           runs the hyperparameter grid across cores
  configs/           one YAML per algorithm, 10 rows each
analysis/
  plots.py           reward curves, DQN diagnostics, PG entropy, convergence, generalisation
  tables.py          the four hyperparameter tables, built from results.csv
ui/
  dashboard.py       Streamlit block monitor
  _manager.py        episode lifecycle behind a narrow interface
api/
  server.py          read-only JSON state and ledger endpoints
```

## Results

From `experiments/results.csv`, 40 runs, 400k timesteps each. Train is 30 episodes
on seeds 0-29; held out is 100 episodes on seeds 10000-10099, which no training
run ever saw.

| Method | Best run | Train mean | Held-out mean | Harvest rate | What made it best |
|---|---|---|---|---|---|
| **PPO** | ppo-06 | **99.9** | **103.1** | 88% | rollout shorter than one episode |
| REINFORCE | rf-09 | 67.2 | 73.2 | 93% | batching 8 episodes per update |
| DQN | dqn-07 | 65.8 | 66.3 | 0% | large replay buffer |
| A2C | a2c-06 | 45.4 | 56.9 | 48% | entropy bonus |

Reference points: the scripted agronomist scores **+21.1** and a random policy
**-1.1**, so PPO is roughly five times the hand-coded expert rules.

DQN is the interesting failure. It scores respectably by keeping the crop alive
but almost never harvests on held-out seeds: with `gamma=0.99` and harvest ~90
steps away the terminal reward is discounted to about 40 percent of face value,
so the Q-function learns to avoid failure without learning to pursue the payoff.

Convergence curves are in `assets/figures/fig4_convergence.png`; the full
per-hyperparameter tables are in `assets/tables/`.

## Notes on the model

The agronomy is deliberately simplified from AquaCrop and DSSAT: single soil
layer per zone, no lateral flow between terraces beyond a runoff cascade, and a
harvest index that responds to stress only during flowering. That is enough to
make the scheduling problem non-trivial without pretending to calibration data I
do not have. Parameters are literature values, so absolute yields are indicative
and only the ranking between policies is meaningful.

| Group | Source |
|---|---|
| Water balance, Kc, depletion fraction, Ks | Allen et al., FAO Irrigation and Drainage Paper 56, 1998 |
| Canopy growth, water productivity, harvest index | Steduto et al., AquaCrop reference manual, FAO 2009 |
| Runoff curve numbers | USDA NRCS National Engineering Handbook, section 4 |
| Reference evapotranspiration | Hargreaves and Samani, 1985 |
| Rainfall statistics | Rwanda Meteorology Agency and TAHMO station summaries |
| Fertiliser rates and spacing | Rwanda Agriculture Board French bean production guidelines |
