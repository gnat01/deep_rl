from __future__ import annotations

import argparse
from pathlib import Path

from mrp_definition import MarkovRewardProcess
from visualization import visualize_mrp


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_JSON = PROJECT_ROOT / "inputs" / "simple_mrp.json"
DEFAULT_OUTPUT_PNG = PROJECT_ROOT / "outputs" / "mrp_visualization.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Markov Reward Process JSON specification as a PNG."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=DEFAULT_INPUT_JSON,
        help="Path to the input MRP JSON file.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=DEFAULT_OUTPUT_PNG,
        help="Path where the rendered PNG should be written.",
    )
    parser.add_argument(
        "--label-precision",
        type=int,
        default=2,
        help="Decimal precision for probability and reward labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.label_precision < 0:
        raise ValueError("--label-precision must be non-negative")

    mrp = MarkovRewardProcess.from_json(args.input_json)
    output_path = visualize_mrp(
        mrp=mrp,
        output_path=args.output_png,
        label_precision=args.label_precision,
    )
    print(f"Wrote visualization: {output_path.resolve()}")


if __name__ == "__main__":
    main()
