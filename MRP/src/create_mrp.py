from __future__ import annotations

import argparse
import random
from pathlib import Path

from mrp_definition import MarkovRewardProcess, Transition


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "inputs" / "generated_mrp.json"


def parse_self_probs(raw: str, num_states: int) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(values) != num_states:
        raise ValueError(
            f"--self-probs must provide exactly {num_states} comma-separated values"
        )
    return values


def validate_self_probs(self_probs: list[float], num_states: int) -> None:
    for index, value in enumerate(self_probs):
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Self probability for state index {index} must lie in [0, 1]"
            )
        if num_states > 1 and value == 1.0:
            raise ValueError(
                "Self probability cannot be 1.0 when there are other states to assign mass to"
            )
    if num_states == 1 and self_probs != [1.0]:
        raise ValueError("A one-state MRP must have self probability 1.0")


def random_probability_vector(length: int, rng: random.Random) -> list[float]:
    if length == 0:
        return []
    weights = [rng.expovariate(1.0) for _ in range(length)]
    total = sum(weights)
    return [weight / total for weight in weights]


def build_transitions_for_state(
    states: list[str],
    state_index: int,
    self_prob: float,
    reward_min: float,
    reward_max: float,
    rng: random.Random,
) -> list[Transition]:
    state = states[state_index]
    other_states = [other for other in states if other != state]
    transitions: list[Transition] = [
        Transition(
            next_state=state,
            probability=self_prob,
            reward=rng.uniform(reward_min, reward_max),
        )
    ]
    if not other_states:
        return transitions

    remainder = 1.0 - self_prob
    other_probs = random_probability_vector(len(other_states), rng)
    for next_state, other_prob in zip(other_states, other_probs, strict=True):
        transitions.append(
            Transition(
                next_state=next_state,
                probability=remainder * other_prob,
                reward=rng.uniform(reward_min, reward_max),
            )
        )
    return transitions


def create_random_mrp(
    num_states: int,
    self_probs: list[float],
    reward_min: float,
    reward_max: float,
    state_prefix: str,
    seed: int,
) -> MarkovRewardProcess:
    rng = random.Random(seed)
    states = [f"{state_prefix}{index}" for index in range(num_states)]
    transitions = {
        state: build_transitions_for_state(
            states=states,
            state_index=index,
            self_prob=self_probs[index],
            reward_min=reward_min,
            reward_max=reward_max,
            rng=rng,
        )
        for index, state in enumerate(states)
    }
    mrp = MarkovRewardProcess(states=states, transitions=transitions)
    mrp.validate()
    return mrp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a random Markov Reward Process JSON specification."
    )
    parser.add_argument(
        "--num-states",
        type=int,
        required=True,
        help="Number of states in the generated MRP.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Where to write the generated MRP JSON.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--self-prob",
        type=float,
        help="Shared self-transition probability to use for every state.",
    )
    group.add_argument(
        "--self-probs",
        type=str,
        help="Comma-separated self-transition probabilities, one per state.",
    )
    parser.add_argument(
        "--reward-min",
        type=float,
        default=-1.0,
        help="Minimum reward for a generated transition.",
    )
    parser.add_argument(
        "--reward-max",
        type=float,
        default=1.0,
        help="Maximum reward for a generated transition.",
    )
    parser.add_argument(
        "--state-prefix",
        type=str,
        default="s",
        help="Prefix used when naming states, e.g. s0, s1, ...",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_states <= 0:
        raise ValueError("--num-states must be positive")
    if args.reward_min > args.reward_max:
        raise ValueError("--reward-min must be <= --reward-max")

    if args.self_prob is not None:
        self_probs = [args.self_prob] * args.num_states
    else:
        self_probs = parse_self_probs(args.self_probs, args.num_states)
    validate_self_probs(self_probs, args.num_states)

    mrp = create_random_mrp(
        num_states=args.num_states,
        self_probs=self_probs,
        reward_min=args.reward_min,
        reward_max=args.reward_max,
        state_prefix=args.state_prefix,
        seed=args.seed,
    )
    mrp.to_json(args.output_json)
    print(f"Wrote MRP JSON: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
