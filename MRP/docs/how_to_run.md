# How To Run

This file is intentionally command-first. Every supported workflow below is given as an exact command that can be run from inside `MRP/`.

## Working Directory

All commands below assume:

```bash
cd /Users/gn/work/learn/python/deep_rl/MRP
```

## Available Entry Points

The runnable CLIs in this project are:

- `src/run_mrp_experiment.py`
- `src/create_mrp.py`
- `src/visualize_mrp.py`
- `src/analyze_mrp_structure.py`

There is also a paper source in:

- `paper_i/paper_i.tex`

## 1. Single Finite-Horizon Experiment

This is the base experiment:

```bash
python src/run_mrp_experiment.py --gamma-grid-num 21 --num-horizon 10 --num-trajectories 100 --output-dir outputs_single_h10_100traj
```

What it writes:

- `state_values_exact.csv`
- `state_values_infinite_exact.csv`
- `state_values_mc.csv`
- `state_value_comparison.csv`
- `finite_vs_infinite_gap.csv`
- `ranked_state_values.md`
- `finite_vs_infinite_exact.png`
- `truncation_gap_vs_gamma.png`
- `value_vs_gamma.png`
- `comparison_exact_vs_mc.png`

## 2. Trajectory-Count Sweep

Compare Monte Carlo estimates across multiple trajectory counts at a fixed horizon:

```bash
python src/run_mrp_experiment.py --gamma-grid-num 21 --num-horizon 10 --traj-list 10,20,50,100,500 --output-dir outputs_traj_sweep
```

What it writes:

- `state_values_exact.csv`
- `state_values_infinite_exact.csv`
- `state_values_mc.csv`
- `state_value_comparison.csv`
- `finite_vs_infinite_gap.csv`
- `ranked_state_values.md`
- `finite_vs_infinite_exact.png`
- `truncation_gap_vs_gamma.png`
- `value_vs_gamma_by_trajectory.png`
- `comparison_exact_vs_mc_by_trajectory.png`
- `error_vs_num_trajectories.png`
- `variance_vs_num_trajectories.png`

## 3. Horizon Sweep Against Infinite-Horizon Exact Values

Compare finite-horizon exact values and Monte Carlo estimates against the infinite-horizon exact benchmark:

```bash
python src/run_mrp_experiment.py --gamma-grid-num 21 --num-trajectories 100 --horizon-list 10,20,50,100,300,500,1000 --output-dir outputs_horizon_sweep_100traj
```

Higher-precision Monte Carlo version:

```bash
python src/run_mrp_experiment.py --gamma-grid-num 21 --num-trajectories 1000 --horizon-list 10,20,50,100,300,500,1000 --output-dir outputs_horizon_sweep_1000traj
```

What horizon-sweep mode writes:

- `state_values_finite_exact_by_horizon.csv`
- `state_values_infinite_exact.csv`
- `state_values_mc_by_horizon.csv`
- `horizon_sweep_comparison.csv`
- `horizon_sweep_gap_vs_horizon.png`
- `horizon_sweep_gap_vs_horizon_loglog.png`
- `horizon_sweep_values_vs_horizon.png`

## 4. Single Long-Horizon Run

If you want one fixed large horizon without a sweep:

```bash
python src/run_mrp_experiment.py --gamma-grid-num 21 --num-horizon 1000 --num-trajectories 10000 --output-dir outputs_h1000_10000traj
```

This is a normal single-run mode, not a horizon sweep, so it does not write the horizon-sweep gap plots.

## 5. Create A New Random MRP

Shared self-transition probability:

```bash
python src/create_mrp.py --num-states 5 --self-prob 0.4 --reward-min -2 --reward-max 3 --output-json inputs/generated_mrp_shared.json --output-png outputs/generated_mrp_shared.png
```

Per-state self-transition probabilities:

```bash
python src/create_mrp.py --num-states 5 --self-probs 0.2,0.3,0.4,0.5,0.6 --reward-min -2 --reward-max 3 --output-json inputs/generated_mrp_vector.json --output-png outputs/generated_mrp_vector.png
```

Important behavior:

- there is no default self-transition probability
- you must specify either `--self-prob` or `--self-probs`
- the remaining probability mass is distributed randomly across the other states using normalized exponential draws

## 6. Visualize An Existing MRP

Visualize the simple MRP:

```bash
python src/visualize_mrp.py --input-json inputs/simple_mrp.json --output-png outputs/simple_mrp_visualization.png
```

Visualize a generated MRP:

```bash
python src/visualize_mrp.py --input-json inputs/generated_mrp_shared.json --output-png outputs/generated_mrp_shared_visualization.png
```

## 7. Analyze Transience / Ergodicity / Communicating Classes

Analyze the simple MRP:

```bash
python src/analyze_mrp_structure.py --input-json inputs/simple_mrp.json --output-dir outputs_structure_simple
```

Analyze a generated MRP:

```bash
python src/analyze_mrp_structure.py --input-json inputs/generated_mrp_shared.json --output-dir outputs_structure_generated_shared
```

What structure analysis writes:

- `markov_structure.json`
- `markov_structure.md`

Reported properties include:

- communicating classes
- closed classes
- transient and recurrent states
- absorbing states
- class periods
- irreducibility
- aperiodicity
- ergodicity
- unichain status

## 8. Build Paper I

Compile the current paper:

```bash
cd paper_i && pdflatex -interaction=nonstopmode paper_i.tex && pdflatex -interaction=nonstopmode paper_i.tex
```

This writes:

- `paper_i/paper_i.pdf`

## Notes

- Infinite-horizon exact values are only defined for `gamma < 1`.
- `gamma = 1` is still valid in finite-horizon mode.
- `--num-horizon` is just an alias for `--num-time-steps`.
- `--traj-list` and `--horizon-list` activate different experiment branches.
- The most complete currently supported analyses are the trajectory sweep and the horizon sweep.
