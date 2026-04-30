# How To Run

This experiment package will live under `MRP/` with the following layout:

- `MRP/src`: source code
- `MRP/docs`: documentation
- `MRP/images`: image assets, including the MRP diagram
- `MRP/inputs`: structured input data, if any is introduced
- `MRP/outputs`: generated tables, plots, and summaries

## Goal

Given the simple Markov Reward Process in `MRP/images/simple_Markov_Reward_process.png`, compute state values across a sweep of discount factors `gamma` using two methods:

1. Monte Carlo rollouts with a finite horizon
2. Exact finite-horizon dynamic programming using Bellman equations

For each `gamma` and each start state:

- start the process from that state,
- evaluate the finite-horizon discounted return,
- produce a value estimate from Monte Carlo averaging,
- compute the exact finite-horizon value,
- rank states by decreasing value for that `gamma`.

## Planned CLI

The CLI should expose at least these flags:

- `--gamma-grid-num`: number of equally spaced `gamma` values in `[0, 1]`
- `--num-time-steps`: rollout horizon, default `10`
- `--num-trajectories`: number of Monte Carlo trajectories per `(gamma, start_state)`, default `10`

Reasonable future flags:

- `--seed`: random seed for reproducibility
- `--output-dir`: override default output directory
- `--save-format`: choose CSV, Markdown, PNG, or all

## Intended Workflow

1. Build the MRP transition and reward specification from the parsed diagram.
2. Construct a grid of discount factors using `gamma_grid = linspace(0, 1, gamma_grid_num)`.
3. For each `gamma`:
   - for each start state:
     - run `num_trajectories` finite-horizon rollouts of length `num_time_steps`,
     - compute the average discounted return,
     - compute the exact finite-horizon value using Bellman recursion.
4. Create ranked tables for that `gamma`.
5. Save plots showing state value as a function of `gamma`.

## Expected Outputs

All generated artifacts should be written to `MRP/outputs`.

Suggested outputs:

- `state_values_exact.csv`
- `state_values_mc.csv`
- `ranked_state_values.md`
- `value_vs_gamma.png`
- `comparison_exact_vs_mc.png`

Each tabular record should contain at least:

- `gamma`
- `state`
- `method`
- `value`
- `rank_within_gamma`

## Example Command Shape

The exact entry point is still to be implemented, but the intended usage is of the form:

```bash
python MRP/src/run_mrp_experiment.py \
  --gamma-grid-num 21 \
  --num-time-steps 10 \
  --num-trajectories 10
```

This would evaluate `gamma` on the grid:

```text
0.00, 0.05, 0.10, ..., 0.95, 1.00
```

when `--gamma-grid-num 21`.

## Notes

- The Monte Carlo estimate and the exact finite-horizon value should be reported side by side.
- Because the horizon is finite, values at `gamma = 1` remain well-defined.
- Exact dynamic programming is mandatory, not optional, because the MRP is small and the exact computation provides a clean validation target for the sampled estimates.
