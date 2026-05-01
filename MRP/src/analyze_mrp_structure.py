from __future__ import annotations

import argparse
from pathlib import Path

from markov_structure import (
    analyze_markov_structure,
    write_structure_json,
    write_structure_markdown,
)
from mrp_definition import MarkovRewardProcess


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_JSON = PROJECT_ROOT / "inputs" / "simple_mrp.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs_structure"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the structural Markov-chain properties of an MRP."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=DEFAULT_INPUT_JSON,
        help="Path to the input MRP JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the structure analysis outputs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mrp = MarkovRewardProcess.from_json(args.input_json)
    payload = analyze_markov_structure(mrp)
    output_dir = args.output_dir
    json_path = output_dir / "markov_structure.json"
    markdown_path = output_dir / "markov_structure.md"
    write_structure_json(json_path, payload)
    write_structure_markdown(markdown_path, payload)
    print("Wrote outputs:")
    print(f"- structure_json: {json_path}")
    print(f"- structure_markdown: {markdown_path}")


if __name__ == "__main__":
    main()
