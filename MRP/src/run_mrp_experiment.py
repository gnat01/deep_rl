from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_MPLCONFIGDIR = Path("MRP/outputs/.mplconfig")
DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR.resolve()))

from experiment import ExperimentConfig, run_experiment
from mrp_definition import MarkovRewardProcess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run finite-horizon MRP value experiments over a gamma grid."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("MRP/inputs/simple_mrp.json"),
        help="Path to the MRP transition-reward specification.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("MRP/outputs"),
        help="Directory where CSVs, markdown, and plots will be written.",
    )
    parser.add_argument(
        "--gamma-grid-num",
        type=int,
        required=True,
        help="Number of equally spaced gamma values in [0, 1].",
    )
    parser.add_argument(
        "--num-time-steps",
        type=int,
        default=10,
        help="Finite horizon length used for exact and Monte Carlo values.",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=10,
        help="Monte Carlo trajectories per (gamma, start_state).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for Monte Carlo rollouts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_time_steps < 0:
        raise ValueError("--num-time-steps must be non-negative")
    if args.num_trajectories <= 0:
        raise ValueError("--num-trajectories must be positive")

    mrp = MarkovRewardProcess.from_json(args.input_json)
    config = ExperimentConfig(
        gamma_grid_num=args.gamma_grid_num,
        num_time_steps=args.num_time_steps,
        num_trajectories=args.num_trajectories,
        seed=args.seed,
    )
    outputs = run_experiment(mrp=mrp, config=config, output_dir=args.output_dir)

    print("Wrote outputs:")
    for label, path in outputs.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
