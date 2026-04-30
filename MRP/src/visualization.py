from __future__ import annotations

import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MPLCONFIGDIR = PROJECT_ROOT / "outputs" / ".mplconfig"
DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR.resolve()))

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.patheffects as path_effects
import numpy as np

from mrp_definition import MarkovRewardProcess, Transition


def circular_layout(states: list[str], radius: float = 3.0) -> dict[str, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * math.pi, len(states), endpoint=False)
    return {
        state: np.array([radius * math.cos(angle), radius * math.sin(angle)])
        for state, angle in zip(states, angles, strict=True)
    }


def format_edge_label(transition: Transition, precision: int) -> str:
    return (
        f"p={transition.probability:.{precision}f}\n"
        f"r={transition.reward:.{precision}f}"
    )


def outward_unit_vector(point: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(point)
    if norm == 0.0:
        return np.array([1.0, 0.0])
    return point / norm


def perpendicular_unit_vector(vector: np.ndarray) -> np.ndarray:
    perpendicular = np.array([-vector[1], vector[0]])
    norm = np.linalg.norm(perpendicular)
    if norm == 0.0:
        return np.array([0.0, 1.0])
    return perpendicular / norm


def bezier_point(
    start: np.ndarray, control: np.ndarray, end: np.ndarray, t: float
) -> np.ndarray:
    return ((1.0 - t) ** 2) * start + 2.0 * (1.0 - t) * t * control + (t**2) * end


def bezier_tangent(
    start: np.ndarray, control: np.ndarray, end: np.ndarray, t: float
) -> np.ndarray:
    return 2.0 * (1.0 - t) * (control - start) + 2.0 * t * (end - control)


def pair_curvature(source_index: int, target_index: int) -> float:
    if source_index < target_index:
        return 0.18
    return -0.18


def angle_from_vector(vector: np.ndarray) -> float:
    angle = math.degrees(math.atan2(vector[1], vector[0]))
    if angle > 90.0:
        angle -= 180.0
    if angle < -90.0:
        angle += 180.0
    return angle


def draw_edge_label(
    ax: plt.Axes,
    position: np.ndarray,
    tangent: np.ndarray,
    transition: Transition,
    label_precision: int,
) -> None:
    text = ax.text(
        position[0],
        position[1],
        format_edge_label(transition, label_precision),
        fontsize=8,
        ha="center",
        va="center",
        rotation=angle_from_vector(tangent),
        rotation_mode="anchor",
        color="#222222",
    )
    text.set_path_effects(
        [path_effects.withStroke(linewidth=3.0, foreground="white", alpha=0.95)]
    )


def draw_self_loop(
    ax: plt.Axes,
    center: np.ndarray,
    transition: Transition,
    label_precision: int,
    node_radius: float,
) -> None:
    outward = outward_unit_vector(center)
    tangent = perpendicular_unit_vector(outward)
    start = center + node_radius * tangent
    end = center - node_radius * tangent
    control = center + outward * (node_radius * 3.0)
    arrow = FancyArrowPatch(
        posA=tuple(start),
        posB=tuple(end),
        connectionstyle="arc3,rad=1.5",
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.2,
        color="#555555",
    )
    ax.add_patch(arrow)
    label_t = 0.55
    label_position = bezier_point(start, control, end, label_t)
    label_tangent = bezier_tangent(start, control, end, label_t)
    draw_edge_label(
        ax=ax,
        position=label_position,
        tangent=label_tangent,
        transition=transition,
        label_precision=label_precision,
    )


def draw_directed_edge(
    ax: plt.Axes,
    source: np.ndarray,
    target: np.ndarray,
    transition: Transition,
    label_precision: int,
    curvature: float,
    node_radius: float,
) -> None:
    direction = target - source
    unit_direction = direction / np.linalg.norm(direction)
    start = source + unit_direction * node_radius
    end = target - unit_direction * node_radius
    normal = perpendicular_unit_vector(direction)
    control = (start + end) / 2.0 + normal * (curvature * np.linalg.norm(end - start))
    arrow = FancyArrowPatch(
        posA=tuple(start),
        posB=tuple(end),
        connectionstyle=f"arc3,rad={curvature}",
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.2,
        color="#555555",
        alpha=0.9,
    )
    ax.add_patch(arrow)

    label_t = 0.5
    label_position = bezier_point(start, control, end, label_t)
    label_tangent = bezier_tangent(start, control, end, label_t)
    draw_edge_label(
        ax=ax,
        position=label_position,
        tangent=label_tangent,
        transition=transition,
        label_precision=label_precision,
    )


def node_colors(num_states: int) -> list[str]:
    cmap = plt.get_cmap("Set2")
    return [cmap(index % cmap.N) for index in range(num_states)]


def visualize_mrp(
    mrp: MarkovRewardProcess,
    output_path: Path,
    *,
    figure_size: tuple[float, float] = (10.0, 10.0),
    label_precision: int = 2,
) -> Path:
    positions = circular_layout(mrp.states)
    node_radius = 0.62
    fig, ax = plt.subplots(figsize=figure_size)
    ax.set_aspect("equal")
    ax.axis("off")

    state_to_index = {state: index for index, state in enumerate(mrp.states)}
    colors = node_colors(len(mrp.states))

    for state, color in zip(mrp.states, colors, strict=True):
        center = positions[state]
        circle = Circle(
            tuple(center),
            radius=node_radius,
            facecolor=color,
            edgecolor="black",
            linewidth=1.8,
            alpha=0.9,
        )
        ax.add_patch(circle)
        ax.text(
            center[0],
            center[1],
            state,
            fontsize=12,
            ha="center",
            va="center",
            weight="bold",
        )

    for source in mrp.states:
        for transition in mrp.transitions[source]:
            source_position = positions[source]
            target_position = positions[transition.next_state]
            if source == transition.next_state:
                draw_self_loop(
                    ax=ax,
                    center=source_position,
                    transition=transition,
                    label_precision=label_precision,
                    node_radius=node_radius,
                )
            else:
                curvature = pair_curvature(
                    state_to_index[source], state_to_index[transition.next_state]
                )
                draw_directed_edge(
                    ax=ax,
                    source=source_position,
                    target=target_position,
                    transition=transition,
                    label_precision=label_precision,
                    curvature=curvature,
                    node_radius=node_radius,
                )

    radius = max(np.linalg.norm(position) for position in positions.values())
    padding = 2.0
    ax.set_xlim(-radius - padding, radius + padding)
    ax.set_ylim(-radius - padding, radius + padding)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path
