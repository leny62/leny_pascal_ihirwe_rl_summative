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
