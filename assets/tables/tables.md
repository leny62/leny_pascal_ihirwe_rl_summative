## DQN

| Learning rate | $\gamma$ | Replay buffer | Batch size | Exploration frac | Final $\epsilon$ | Target update | Network | Mean reward (heldout) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1.00e-04 | 0.99 | 1.00e+05 | 64 | 0.2 | 0.05 | 1000 | [256, 256] | 53.40 | baseline |
| 5.00e-04 | 0.99 | 1.00e+05 | 64 | 0.2 | 0.05 | 1000 | [256, 256] | -15.56 | high lr destabilises the Q target |
| 5.00e-05 | 0.99 | 1.00e+05 | 64 | 0.2 | 0.05 | 1000 | [256, 256] | 29.89 | low lr underfits inside the budget |
| 1.00e-04 | 0.95 | 1.00e+05 | 64 | 0.2 | 0.05 | 1000 | [256, 256] | 33.90 | myopic discount ignores terminal revenue |
| 1.00e-04 | 0.999 | 1.00e+05 | 64 | 0.2 | 0.05 | 1000 | [256, 256] | 19.64 | long horizon credit assignment |
| 1.00e-04 | 0.99 | 2.00e+04 | 64 | 0.2 | 0.05 | 1000 | [256, 256] | 30.47 | small buffer, correlated samples |
| 1.00e-04 | 0.99 | 5.00e+05 | 64 | 0.2 | 0.05 | 1000 | [256, 256] | 48.19 | large buffer, stale off-policy data |
| 1.00e-04 | 0.99 | 1.00e+05 | 256 | 0.2 | 0.05 | 5000 | [256, 256] | 53.53 | big batch, slow target, max stability |
| 1.00e-04 | 0.99 | 1.00e+05 | 64 | 0.5 | 0.1 | 1000 | [256, 256] | 56.56 | sustained exploration to discover SCOUT |
| 1.00e-04 | 0.99 | 1.00e+05 | 128 | 0.3 | 0.05 | 2000 | [512, 512, 256] | -24.93 | more capacity, later learning start |

## REINFORCE

| Learning rate | $\gamma$ | Baseline | Entropy coef | Episodes/update | Norm. advantage | Mean reward (heldout) | Notes |
|---|---|---|---|---|---|---|---|
| 0.001 | 0.99 | none | 0 | 1 | False | -1.24 | textbook REINFORCE, expect high variance |
| 0.001 | 0.99 | value | 0 | 1 | False | -29.24 | learned baseline cuts variance |
| 3.00e-04 | 0.99 | value | 0 | 1 | False | 39.65 | lower lr, smoother but slower |
| 0.003 | 0.99 | value | 0 | 1 | False | -1.24 | too high, expect divergence |
| 0.001 | 0.95 | value | 0 | 1 | False | -1.24 | myopic on a terminal-reward task |
| 0.001 | 1 | value | 0 | 1 | False | -64.78 | undiscounted, maximum variance |
| 0.001 | 0.99 | value | 0.01 | 1 | False | 17.37 | mild entropy bonus delays collapse |
| 0.001 | 0.99 | value | 0.05 | 1 | False | -1.22 | strong entropy, expect no commitment |
| 0.001 | 0.99 | value | 0.01 | 8 | False | 29.38 | batching episodes cuts gradient variance |
| 0.001 | 0.99 | value | 0.01 | 8 | True | 33.14 | normalised advantages, expect best of ten |

## PPO

| Learning rate | $\gamma$ | Rollout steps | Batch size | Epochs | Clip range | GAE $\lambda$ | Entropy coef | Mean reward (heldout) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 3.00e-04 | 0.99 | 2048 | 64 | 10 | 0.2 | 0.95 | 0 | 74.79 | baseline |
| 1.00e-04 | 0.99 | 2048 | 64 | 10 | 0.2 | 0.95 | 0 | 36.89 | conservative lr |
| 0.001 | 0.99 | 2048 | 64 | 10 | 0.2 | 0.95 | 0 | 76.09 | aggressive lr |
| 3.00e-04 | 0.99 | 2048 | 64 | 10 | 0.1 | 0.95 | 0 | 63.09 | tight clip, small trust region |
| 3.00e-04 | 0.99 | 2048 | 64 | 10 | 0.3 | 0.95 | 0 | 90.25 | loose clip, larger policy jumps |
| 3.00e-04 | 0.99 | 512 | 64 | 10 | 0.2 | 0.95 | 0 | 93.77 | rollout shorter than one episode |
| 3.00e-04 | 0.99 | 4096 | 128 | 10 | 0.2 | 0.95 | 0 | 59.57 | long rollout, several episodes |
| 3.00e-04 | 0.99 | 2048 | 64 | 20 | 0.2 | 0.95 | 0 | 88.38 | over-optimising each batch |
| 3.00e-04 | 0.99 | 2048 | 64 | 10 | 0.2 | 0.95 | 0.01 | 91.62 | entropy bonus sustains exploration |
| 3.00e-04 | 0.995 | 2048 | 64 | 10 | 0.2 | 0.8 | 0.01 | 92.54 | shorter GAE, longer discount |

## A2C

| Learning rate | $\gamma$ | Rollout steps | Entropy coef | Value coef | Optimiser | Grad clip | Mean reward (heldout) | Notes |
|---|---|---|---|---|---|---|---|---|
| 7.00e-04 | 0.99 | 5 | 0 | 0.5 | RMSProp | 0.5 | 30.42 | SB3 default baseline |
| 3.00e-04 | 0.99 | 5 | 0 | 0.5 | RMSProp | 0.5 | 32.27 | lower lr |
| 0.001 | 0.99 | 5 | 0 | 0.5 | RMSProp | 0.5 | -1.22 | higher lr |
| 7.00e-04 | 0.99 | 16 | 0 | 0.5 | RMSProp | 0.5 | -1.22 | longer rollout, less biased returns |
| 7.00e-04 | 0.99 | 64 | 0 | 0.5 | RMSProp | 0.5 | -28.16 | much longer rollout, approaching PPO |
| 7.00e-04 | 0.99 | 16 | 0.01 | 0.5 | RMSProp | 0.5 | -23.51 | entropy bonus |
| 7.00e-04 | 0.99 | 16 | 0.05 | 0.5 | RMSProp | 0.5 | 46.62 | heavy entropy, expect no commitment |
| 7.00e-04 | 0.99 | 16 | 0.01 | 0.25 | RMSProp | 0.5 | 4.68 | weaker critic weighting |
| 7.00e-04 | 0.99 | 16 | 0.01 | 0.5 | Adam | 0.5 | 30.94 | Adam instead of RMSProp |
| 7.00e-04 | 0.995 | 16 | 0.01 | 0.5 | Adam | 0.3 | 12.56 | tighter gradient clipping |

