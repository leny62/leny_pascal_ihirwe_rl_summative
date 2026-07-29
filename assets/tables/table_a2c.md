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
