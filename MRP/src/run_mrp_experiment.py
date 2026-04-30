from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_JSON = PROJECT_ROOT / "inputs" / "simple_mrp.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MPLCONFIGDIR = DEFAULT_OUTPUT_DIR / ".mplconfig"
DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR.resolve()))

from experiment import ExperimentConfig, run_experiment
from mrp_definition import MarkovRewardProcess


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("--traj-list must contain at least one integer")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run finite-horizon MRP value experiments over a gamma grid."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=DEFAULT_INPUT_JSON,
        help="Path to the MRP transition-reward specification.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
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
        "--num-horizon",
        type=int,
        dest="num_time_steps",
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
        "--traj-list",
        type=str,
        default=None,
        help="Comma-separated trajectory counts, e.g. 10,20,50,100,500.",
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
    traj_list = parse_int_list(args.traj_list) if args.traj_list else None
    if traj_list is not None and any(value <= 0 for value in traj_list):
        raise ValueError("--traj-list entries must all be positive")

    mrp = MarkovRewardProcess.from_json(args.input_json)
    config = ExperimentConfig(
        gamma_grid_num=args.gamma_grid_num,
        num_time_steps=args.num_time_steps,
        num_trajectories=args.num_trajectories,
        seed=args.seed,
        traj_list=traj_list,
    )
    outputs = run_experiment(mrp=mrp, config=config, output_dir=args.output_dir)

    print("Wrote outputs:")
    for label, path in outputs.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
