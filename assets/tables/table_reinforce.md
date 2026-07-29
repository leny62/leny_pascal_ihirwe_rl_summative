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
