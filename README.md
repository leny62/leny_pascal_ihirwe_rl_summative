# Umurima: RL for irrigation and input scheduling

A reinforcement learning agent that schedules a season of irrigation, fertiliser,
spray and labour decisions on a one hectare horticulture block on a Rwandan
hillside, and a comparison of four RL methods trained against it.

**Course:** Machine Learning Techniques II, Summative Assignment
**Author:** Leny Pascal Ihirwe

Report: `docs/report.pdf`
Video: TODO add link

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
uv sync
```

## Running

Watch the trained agent work a season in the 3D view:

```bash
uv run main.py
```

Pick a specific policy or run headless:

```bash
uv run play.py --model models/dqn/best.zip --episodes 3
uv run play.py --model models/pg/ppo_best.zip --no-render --episodes 50
```

Train:

```bash
uv run training/dqn_training.py --run-name dqn-baseline --lr 1e-4 --gamma 0.99
uv run training/pg_training.py --algo ppo --run-name ppo-baseline --lr 3e-4
uv run training/pg_training.py --algo reinforce --run-name rf-baseline --lr 1e-3
uv run training/pg_training.py --algo a2c --run-name a2c-baseline --lr 7e-4
```

Reproduce the full hyperparameter study (40 runs):

```bash
uv run experiments/sweep.py --all --workers 6
uv run analysis/plots.py --out assets/figures
```

Tests:

```bash
uv run pytest
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
  plots.py           reward curves, DQN loss, PG entropy, convergence, generalisation
api/
  server.py          read-only JSON state and ledger endpoints
```

## Results

Filled from `experiments/results.csv` once the sweep completes.

| Method | Best mean reward | Episodes to converge | Held-out seed mean |
|---|---|---|---|
| DQN | TODO | TODO | TODO |
| REINFORCE | TODO | TODO | TODO |
| PPO | TODO | TODO | TODO |
| A2C | TODO | TODO | TODO |

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
