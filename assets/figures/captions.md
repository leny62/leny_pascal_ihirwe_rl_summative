**fig1_training_curves.png**: Training curves for the best hyperparameter run of each algorithm. Thin lines are per-episode returns; thick lines are a rolling mean (window=20 episodes). The dashed and dotted grey lines mark the scripted agronomist (21.1) and random (-1.1) baselines respectively. All four algorithms surpass the scripted baseline.

**fig2_dqn_objective.png**: DQN hyperparameter study. Left: rolling mean returns for all ten grid rows, showing sensitivity to learning rate, discount factor, buffer size, and exploration schedule. Right: train versus heldout mean return for each row, with the identity line marking perfect generalisation.

**fig3_pg_entropy.png**: REINFORCE policy entropy over the course of training. The dashed line marks the entropy of a uniform policy over 18 actions (ln(18) = 2.890 nats). Low entropy coefficients lead to rapid entropy collapse; higher values sustain exploration longer. PPO and A2C entropy is not logged by the Stable-Baselines3 Monitor wrapper and is therefore not shown.

**fig4_convergence.png**: Convergence analysis. Left: median episodes to reach a return of 10 across all ten grid rows per algorithm, with standard deviation error bars. Right: rolling mean return for the best run of each algorithm, showing the trajectory that achieved the highest heldout score.

**fig5_generalisation.png**: Generalisation: train-distribution mean return (30 seeds) against held-out mean return (100 seeds drawn from a disjoint seed range). Points on the identity line generalise perfectly. Negative outliers are labelled with their run ID.

**fig6_policy_behaviour.png**: One episode under the best PPO policy (ppo-06). The episode trace shows cumulative return, cash position, soil moisture at the ridge and valley, canopy cover and nitrogen status, reservoir level, and the action sequence.
