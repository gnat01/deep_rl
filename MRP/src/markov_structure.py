from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from mrp_definition import MarkovRewardProcess


@dataclass(frozen=True)
class CommunicatingClass:
    states: list[str]
    is_closed: bool
    period: int
    is_aperiodic: bool
    classification: str


def positive_adjacency(mrp: MarkovRewardProcess) -> dict[str, list[str]]:
    return {
        state: [
            transition.next_state
            for transition in mrp.transitions[state]
            if transition.probability > 0.0
        ]
        for state in mrp.states
    }


def reverse_adjacency(adjacency: dict[str, list[str]]) -> dict[str, list[str]]:
    reversed_graph = {state: [] for state in adjacency}
    for source, targets in adjacency.items():
        for target in targets:
            reversed_graph[target].append(source)
    return reversed_graph


def kosaraju_sccs(adjacency: dict[str, list[str]]) -> list[list[str]]:
    visited: set[str] = set()
    order: list[str] = []

    def dfs_first(node: str) -> None:
        visited.add(node)
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                dfs_first(neighbor)
        order.append(node)

    for state in adjacency:
        if state not in visited:
            dfs_first(state)

    reversed_graph = reverse_adjacency(adjacency)
    visited.clear()
    components: list[list[str]] = []

    def dfs_second(node: str, component: list[str]) -> None:
        visited.add(node)
        component.append(node)
        for neighbor in reversed_graph[node]:
            if neighbor not in visited:
                dfs_second(neighbor, component)

    for state in reversed(order):
        if state not in visited:
            component: list[str] = []
            dfs_second(state, component)
            components.append(sorted(component))
    return components


def is_closed_class(
    component: list[str], adjacency: dict[str, list[str]]
) -> bool:
    component_set = set(component)
    for state in component:
        for neighbor in adjacency[state]:
            if neighbor not in component_set:
                return False
    return True


def class_period(component: list[str], adjacency: dict[str, list[str]]) -> int:
    component_set = set(component)
    root = component[0]
    distances = {root: 0}
    queue = [root]
    while queue:
        current = queue.pop(0)
        for neighbor in adjacency[current]:
            if neighbor in component_set and neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)

    period = 0
    for source in component:
        for target in adjacency[source]:
            if target not in component_set:
                continue
            cycle_length_delta = distances[source] + 1 - distances[target]
            period = math.gcd(period, abs(cycle_length_delta))
    return max(period, 1)


def absorbing_states(mrp: MarkovRewardProcess) -> list[str]:
    absorbing: list[str] = []
    for state in mrp.states:
        transitions = mrp.transitions[state]
        if len(transitions) == 1:
            transition = transitions[0]
            if transition.next_state == state and abs(transition.probability - 1.0) < 1e-12:
                absorbing.append(state)
    return absorbing


def analyze_markov_structure(mrp: MarkovRewardProcess) -> dict[str, object]:
    adjacency = positive_adjacency(mrp)
    components = kosaraju_sccs(adjacency)

    classes: list[CommunicatingClass] = []
    transient_states: list[str] = []
    recurrent_states: list[str] = []

    for component in components:
        closed = is_closed_class(component, adjacency)
        period = class_period(component, adjacency)
        aperiodic = period == 1
        classification = "recurrent" if closed else "transient"
        if closed:
            recurrent_states.extend(component)
        else:
            transient_states.extend(component)
        classes.append(
            CommunicatingClass(
                states=component,
                is_closed=closed,
                period=period,
                is_aperiodic=aperiodic,
                classification=classification,
            )
        )

    irreducible = len(classes) == 1
    aperiodic = irreducible and classes[0].is_aperiodic
    ergodic = irreducible and aperiodic
    closed_classes = [cls for cls in classes if cls.is_closed]
    unichain = len(closed_classes) == 1

    payload = {
        "num_states": len(mrp.states),
        "states": mrp.states,
        "absorbing_states": sorted(absorbing_states(mrp)),
        "transient_states": sorted(transient_states),
        "recurrent_states": sorted(recurrent_states),
        "num_communicating_classes": len(classes),
        "num_closed_classes": len(closed_classes),
        "irreducible": irreducible,
        "aperiodic": aperiodic,
        "ergodic": ergodic,
        "unichain": unichain,
        "communicating_classes": [
            {
                "states": cls.states,
                "is_closed": cls.is_closed,
                "period": cls.period,
                "is_aperiodic": cls.is_aperiodic,
                "classification": cls.classification,
            }
            for cls in classes
        ],
    }
    return payload


def write_structure_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_structure_markdown(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# Markov Structure Analysis",
        "",
        f"- `num_states`: {payload['num_states']}",
        f"- `irreducible`: {payload['irreducible']}",
        f"- `aperiodic`: {payload['aperiodic']}",
        f"- `ergodic`: {payload['ergodic']}",
        f"- `unichain`: {payload['unichain']}",
        f"- `absorbing_states`: {', '.join(payload['absorbing_states']) or 'none'}",
        f"- `transient_states`: {', '.join(payload['transient_states']) or 'none'}",
        f"- `recurrent_states`: {', '.join(payload['recurrent_states']) or 'none'}",
        "",
        "## Communicating Classes",
        "",
    ]
    for index, cls in enumerate(payload["communicating_classes"], start=1):
        lines.extend(
            [
                f"### Class {index}",
                "",
                f"- `states`: {', '.join(cls['states'])}",
                f"- `is_closed`: {cls['is_closed']}",
                f"- `classification`: {cls['classification']}",
                f"- `period`: {cls['period']}",
                f"- `is_aperiodic`: {cls['is_aperiodic']}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
