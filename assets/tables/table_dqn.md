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
