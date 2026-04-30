# Next Steps

This project now has:

- a concrete simple MRP
- exact finite-horizon value computation via Bellman recursion
- Monte Carlo estimation with confidence intervals
- gamma sweeps
- trajectory-count sweeps
- MRP generation from scratch
- MRP visualization
- a first paper note in `MRP/paper_i`

The next sensible extensions are below, in roughly the order that gives the best payoff.

## 1. Infinite-Horizon Value Analysis

For `gamma < 1`, compute the exact infinite-horizon value function

```text
V = (I - gamma P)^{-1} r
```

and compare it against the finite-horizon value function for different horizons.

Why this matters:

- it gives a second exact benchmark, not just finite-horizon Bellman recursion
- it makes truncation error explicit
- it shows how fast finite-horizon values converge as horizon grows

Recommended outputs:

- `infinite_horizon_values.csv`
- `finite_vs_infinite_gap.csv`
- plot of truncation error versus horizon for several `gamma` values

## 2. Horizon Sensitivity Study

Right now horizon is a parameter, but it should become a first-class experiment axis.

Suggested sweep:

- `--horizon-list 1,2,5,10,20,50,100`

Questions to answer:

- how quickly do rankings stabilize?
- for which `gamma` values does horizon matter most?
- at what horizon is the finite-horizon approximation effectively converged?

Recommended plots:

- state value versus horizon for selected `gamma`
- truncation gap versus horizon

## 3. Trajectory-Count Study For Paper Quality Results

The code now supports `--traj-list`, but the analysis should be formalized into a proper experiment and then folded into the paper.

Suggested sweep:

- `10, 20, 50, 100, 200, 500, 1000`

Questions to answer:

- how does mean absolute error decay with trajectory count?
- how does confidence interval width decay?
- how does estimator variance behave near `gamma = 1`?

Recommended additions:

- separate error curves for `gamma = 0.1, 0.5, 0.9, 1.0`
- log-scale x-axis option for trajectory count plots

## 4. Ensemble Experiments Over Random MRPs

The random generator makes it possible to move from one hand-crafted MRP to a family of MRPs.

This would allow:

- measuring how stable the observed patterns are across random transition structures
- relating state value spread to self-transition probabilities
- exploring how reward range affects ranking volatility

Suggested workflow:

- generate `N` random MRPs
- run the same finite-horizon exact/MC analysis on each
- aggregate summary statistics across instances

Recommended outputs:

- `ensemble_summary.csv`
- histograms of value gaps
- distribution of MC error across randomly generated MRPs

## 5. Stationary Distribution And Long-Run Structure

For ergodic MRPs, compute:

- stationary distribution
- expected long-run occupancy of each state
- long-run average reward

Why this matters:

- it complements discounted finite-horizon analysis
- it explains which states dominate long-run behavior
- it gives a bridge from short-run return to asymptotic structure

Recommended outputs:

- `stationary_distribution.csv`
- occupancy plots
- long-run average reward summaries

## 6. Hitting-Time And Absorption Analysis

Introduce designated target states or absorbing states and study:

- expected time to hit a state
- probability of hitting a state within a horizon
- expected reward until absorption

This is a natural structural extension of the current framework and fits well with exact dynamic programming.

## 7. Better Visualization Variants

The current graph renderer is useful, but a few targeted improvements would make it stronger:

- optional edge filtering by minimum probability threshold
- separate rendering modes for dense vs sparse MRPs
- color edges by reward sign or magnitude
- export SVG in addition to PNG

This would be especially useful for larger randomly generated MRPs.

## 8. Paper II

The next paper should probably focus on one coherent theme rather than adding everything at once.

Best choice:

- `paper_ii`: finite-horizon versus infinite-horizon values, including truncation error and convergence with horizon

That gives a clean theoretical continuation of `paper_i` and makes the project feel cumulative rather than fragmented.

## Recommended Immediate Sequence

If the goal is to keep momentum and produce the strongest next result set, the best order is:

1. Add infinite-horizon exact values.
2. Add horizon sweeps and truncation-gap plots.
3. Run a formal trajectory-count experiment.
4. Update `paper_i` or write `paper_ii`.

That sequence gives:

- stronger theory
- better diagnostics
- a cleaner publication-style narrative
