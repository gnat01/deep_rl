from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Transition:
    next_state: str
    probability: float
    reward: float


@dataclass(frozen=True)
class MarkovRewardProcess:
    states: list[str]
    transitions: dict[str, list[Transition]]

    @classmethod
    def from_json(cls, path: Path) -> "MarkovRewardProcess":
        payload = json.loads(path.read_text())
        states = payload["states"]
        transitions = {
            state: [Transition(**item) for item in payload["transitions"][state]]
            for state in states
        }
        instance = cls(states=states, transitions=transitions)
        instance.validate()
        return instance

    def validate(self) -> None:
        state_set = set(self.states)
        for state in self.states:
            if state not in self.transitions:
                raise ValueError(f"Missing transitions for state {state!r}")
            total_probability = 0.0
            for transition in self.transitions[state]:
                if transition.next_state not in state_set:
                    raise ValueError(
                        f"Unknown next state {transition.next_state!r} from {state!r}"
                    )
                if transition.probability < 0.0:
                    raise ValueError(
                        f"Negative probability on transition from {state!r}"
                    )
                total_probability += transition.probability
            if abs(total_probability - 1.0) > 1e-9:
                raise ValueError(
                    f"Outgoing probabilities from {state!r} sum to {total_probability}, not 1.0"
                )

    def to_payload(self) -> dict[str, object]:
        return {
            "states": self.states,
            "transitions": {
                state: [
                    {
                        "next_state": transition.next_state,
                        "probability": transition.probability,
                        "reward": transition.reward,
                    }
                    for transition in self.transitions[state]
                ]
                for state in self.states
            },
        }

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_payload(), indent=2) + "\n")
