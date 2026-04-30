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

## CLI

The CLI should expose at least these flags:

- `--gamma-grid-num`: number of equally spaced `gamma` values in `[0, 1]`
- `--num-time-steps`: rollout horizon, default `10`
- `--num-trajectories`: number of Monte Carlo trajectories per `(gamma, start_state)`, default `10`

Implemented flags:

- `--input-json`: path to the MRP specification, default `MRP/inputs/simple_mrp.json`
- `--output-dir`: output directory, default `MRP/outputs`
- `--seed`: random seed for reproducibility, default `0`
- `--traj-list`: comma-separated trajectory counts for a trajectory-sweep experiment
- `--num-horizon`: alias for `--num-time-steps`

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

Current outputs:

- `state_values_exact.csv`
- `state_values_infinite_exact.csv`
- `state_values_mc.csv`
- `state_value_comparison.csv`
- `finite_vs_infinite_gap.csv`
- `ranked_state_values.md`
- `value_vs_gamma.png`
- `comparison_exact_vs_mc.png`

Each tabular record should contain at least:

- `gamma`
- `state`
- `method`
- `value`
- `rank_within_gamma`

Monte Carlo outputs also include:

- sample standard deviation across trajectories
- standard error of the mean
- 95% confidence interval bounds

Infinite-horizon exact outputs:

- are defined only for `gamma < 1`
- are computed from the linear system `(I - gamma P)V = r`
- are omitted at `gamma = 1`

## Example Command

The runner is:

```bash
python MRP/src/run_mrp_experiment.py \
  --gamma-grid-num 21 \
  --num-time-steps 10 \
  --num-trajectories 10
```

The same script also works when run from inside `MRP/`:

```bash
python src/run_mrp_experiment.py \
  --gamma-grid-num 21 \
  --num-time-steps 10 \
  --num-trajectories 10
```

To compare Monte Carlo estimates against exact values across multiple trajectory counts:

```bash
python src/run_mrp_experiment.py \
  --gamma-grid-num 21 \
  --num-horizon 10 \
  --traj-list 10,20,50,100,500 \
  --output-dir outputs_traj_sweep
```

## Create A New MRP

You can generate a fresh random MRP JSON with:

```bash
python MRP/src/create_mrp.py \
  --num-states 5 \
  --self-prob 0.4 \
  --reward-min -2 \
  --reward-max 3 \
  --output-png MRP/outputs/generated_mrp.png \
  --output-json MRP/inputs/generated_mrp.json
```

Or from inside `MRP/`:

```bash
python src/create_mrp.py \
  --num-states 5 \
  --self-probs 0.2,0.3,0.4,0.5,0.6 \
  --reward-min -2 \
  --reward-max 3 \
  --output-png outputs/generated_mrp.png \
  --output-json inputs/generated_mrp.json
```

Generator flags:

- `--num-states`: number of states
- `--self-prob`: one shared self-transition probability for every state
- `--self-probs`: comma-separated per-state self-transition probabilities
- `--reward-min`, `--reward-max`: reward range for random transition rewards
- `--state-prefix`: state name prefix, default `s`
- `--seed`: random seed
- `--output-png`: optional path to save a rendered visualization immediately
- `--label-precision`: decimal precision for edge labels in the visualization

For each state, the remaining probability mass `1 - p_self` is distributed randomly across the other states and normalized to sum correctly.

## Visualize An Existing MRP

To render any MRP JSON as a PNG:

```bash
python MRP/src/visualize_mrp.py \
  --input-json MRP/inputs/simple_mrp.json \
  --output-png MRP/outputs/simple_mrp_visualization.png
```

Or from inside `MRP/`:

```bash
python src/visualize_mrp.py \
  --input-json inputs/simple_mrp.json \
  --output-png outputs/simple_mrp_visualization.png
```

The visualizer uses a circular layout, curved directed edges, offset label boxes, and separate self-loop placement so probability and reward annotations stay readable even when the graph is dense.

This would evaluate `gamma` on the grid:

```text
0.00, 0.05, 0.10, ..., 0.95, 1.00
```

when `--gamma-grid-num 21`.

## Notes

- The Monte Carlo estimate and the exact finite-horizon value should be reported side by side.
- `value_vs_gamma.png` shows exact curves and Monte Carlo curves, with 95% confidence bands on the Monte Carlo panel.
- `comparison_exact_vs_mc.png` is a scatter plot of Monte Carlo estimate against exact value, with a diagonal reference line and Monte Carlo error bars.
- `finite_vs_infinite_exact.png` compares exact finite-horizon and exact infinite-horizon values.
- `truncation_gap_vs_gamma.png` shows the gap between infinite-horizon and finite-horizon exact values as a function of `gamma`.
- In trajectory-sweep mode, the runner writes `value_vs_gamma_by_trajectory.png`, `comparison_exact_vs_mc_by_trajectory.png`, `error_vs_num_trajectories.png`, and `variance_vs_num_trajectories.png`.
- Because the horizon is finite, values at `gamma = 1` remain well-defined.
- Exact dynamic programming is mandatory, not optional, because the MRP is small and the exact computation provides a clean validation target for the sampled estimates.
- `state_value_comparison.csv` includes exact value, Monte Carlo value, ranks, and absolute error for each `(gamma, state)` pair.
