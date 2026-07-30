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
