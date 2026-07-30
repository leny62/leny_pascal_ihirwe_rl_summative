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
