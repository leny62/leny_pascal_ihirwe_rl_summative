## DQN

| Learning rate | $\gamma$ | Replay buffer | Batch size | Exploration frac | Final $\epsilon$ | Target update | Network | Mean reward (heldout) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1.00e-04 | 0.99 | 1.00e+05 | 64 | 0.2 | 0.05 | 1000 | [256, 256] | 51.45 | baseline |
| 5.00e-04 | 0.99 | 1.00e+05 | 64 | 0.2 | 0.05 | 1000 | [256, 256] | 43.71 | high lr destabilises the Q target |
| 5.00e-05 | 0.99 | 1.00e+05 | 64 | 0.2 | 0.05 | 1000 | [256, 256] | 27.36 | low lr underfits inside the budget |
| 1.00e-04 | 0.95 | 1.00e+05 | 64 | 0.2 | 0.05 | 1000 | [256, 256] | 46.35 | myopic discount ignores terminal revenue |
| 1.00e-04 | 0.999 | 1.00e+05 | 64 | 0.2 | 0.05 | 1000 | [256, 256] | 17.16 | long horizon credit assignment |
| 1.00e-04 | 0.99 | 2.00e+04 | 64 | 0.2 | 0.05 | 1000 | [256, 256] | 50.36 | small buffer, correlated samples |
| 1.00e-04 | 0.99 | 5.00e+05 | 64 | 0.2 | 0.05 | 1000 | [256, 256] | 66.25 | large buffer, stale off-policy data |
| 1.00e-04 | 0.99 | 1.00e+05 | 256 | 0.2 | 0.05 | 5000 | [256, 256] | 63.03 | big batch, slow target, max stability |
| 1.00e-04 | 0.99 | 1.00e+05 | 64 | 0.5 | 0.1 | 1000 | [256, 256] | 50.12 | sustained exploration to discover SCOUT |
| 1.00e-04 | 0.99 | 1.00e+05 | 128 | 0.3 | 0.05 | 2000 | [512, 512, 256] | -48.29 | more capacity, later learning start |

## REINFORCE

| Learning rate | $\gamma$ | Baseline | Entropy coef | Episodes/update | Norm. advantage | Mean reward (heldout) | Notes |
|---|---|---|---|---|---|---|---|
| 0.001 | 0.99 | none | 0 | 1 | False | 0.34 | textbook REINFORCE, expect high variance |
| 0.001 | 0.99 | value | 0 | 1 | False | -45.69 | learned baseline cuts variance |
| 3.00e-04 | 0.99 | value | 0 | 1 | False | 35.95 | lower lr, smoother but slower |
| 0.003 | 0.99 | value | 0 | 1 | False | -65.56 | too high, expect divergence |
| 0.001 | 0.95 | value | 0 | 1 | False | -45.69 | myopic on a terminal-reward task |
| 0.001 | 1 | value | 0 | 1 | False | 0.34 | undiscounted, maximum variance |
| 0.001 | 0.99 | value | 0.01 | 1 | False | -45.69 | mild entropy bonus delays collapse |
| 0.001 | 0.99 | value | 0.05 | 1 | False | 0.46 | strong entropy, expect no commitment |
| 0.001 | 0.99 | value | 0.01 | 8 | False | 73.17 | batching episodes cuts gradient variance |
| 0.001 | 0.99 | value | 0.01 | 8 | True | 50.11 | normalised advantages, expect best of ten |

## PPO

| Learning rate | $\gamma$ | Rollout steps | Batch size | Epochs | Clip range | GAE $\lambda$ | Entropy coef | Mean reward (heldout) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 3.00e-04 | 0.99 | 2048 | 64 | 10 | 0.2 | 0.95 | 0 | 84.59 | baseline |
| 1.00e-04 | 0.99 | 2048 | 64 | 10 | 0.2 | 0.95 | 0 | 59.43 | conservative lr |
| 0.001 | 0.99 | 2048 | 64 | 10 | 0.2 | 0.95 | 0 | 81.94 | aggressive lr |
| 3.00e-04 | 0.99 | 2048 | 64 | 10 | 0.1 | 0.95 | 0 | 74.60 | tight clip, small trust region |
| 3.00e-04 | 0.99 | 2048 | 64 | 10 | 0.3 | 0.95 | 0 | 90.81 | loose clip, larger policy jumps |
| 3.00e-04 | 0.99 | 512 | 64 | 10 | 0.2 | 0.95 | 0 | 103.12 | rollout shorter than one episode |
| 3.00e-04 | 0.99 | 4096 | 128 | 10 | 0.2 | 0.95 | 0 | 65.92 | long rollout, several episodes |
| 3.00e-04 | 0.99 | 2048 | 64 | 20 | 0.2 | 0.95 | 0 | 97.59 | over-optimising each batch |
| 3.00e-04 | 0.99 | 2048 | 64 | 10 | 0.2 | 0.95 | 0.01 | 90.73 | entropy bonus sustains exploration |
| 3.00e-04 | 0.995 | 2048 | 64 | 10 | 0.2 | 0.8 | 0.01 | 89.13 | shorter GAE, longer discount |

## A2C

| Learning rate | $\gamma$ | Rollout steps | Entropy coef | Value coef | Optimiser | Grad clip | Mean reward (heldout) | Notes |
|---|---|---|---|---|---|---|---|---|
| 7.00e-04 | 0.99 | 5 | 0 | 0.5 | RMSProp | 0.5 | -45.69 | SB3 default baseline |
| 3.00e-04 | 0.99 | 5 | 0 | 0.5 | RMSProp | 0.5 | -45.69 | lower lr |
| 0.001 | 0.99 | 5 | 0 | 0.5 | RMSProp | 0.5 | -45.69 | higher lr |
| 7.00e-04 | 0.99 | 16 | 0 | 0.5 | RMSProp | 0.5 | 23.09 | longer rollout, less biased returns |
| 7.00e-04 | 0.99 | 64 | 0 | 0.5 | RMSProp | 0.5 | 45.79 | much longer rollout, approaching PPO |
| 7.00e-04 | 0.99 | 16 | 0.01 | 0.5 | RMSProp | 0.5 | 56.94 | entropy bonus |
| 7.00e-04 | 0.99 | 16 | 0.05 | 0.5 | RMSProp | 0.5 | 55.47 | heavy entropy, expect no commitment |
| 7.00e-04 | 0.99 | 16 | 0.01 | 0.25 | RMSProp | 0.5 | 22.01 | weaker critic weighting |
| 7.00e-04 | 0.99 | 16 | 0.01 | 0.5 | Adam | 0.5 | 25.37 | Adam instead of RMSProp |
| 7.00e-04 | 0.995 | 16 | 0.01 | 0.5 | Adam | 0.3 | 0.34 | tighter gradient clipping |

