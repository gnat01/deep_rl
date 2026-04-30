from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mrp_definition import MarkovRewardProcess, Transition


@dataclass(frozen=True)
class ExperimentConfig:
    gamma_grid_num: int
    num_time_steps: int
    num_trajectories: int
    seed: int
    traj_list: list[int] | None = None
    horizon_list: list[int] | None = None


def gamma_grid(grid_num: int) -> np.ndarray:
    if grid_num < 2:
        raise ValueError("--gamma-grid-num must be at least 2")
    return np.linspace(0.0, 1.0, grid_num)


def trajectory_counts(config: ExperimentConfig) -> list[int]:
    if config.traj_list:
        return config.traj_list
    return [config.num_trajectories]


def horizon_values(config: ExperimentConfig) -> list[int]:
    if config.horizon_list:
        return config.horizon_list
    return [config.num_time_steps]


def exact_finite_horizon_values(
    mrp: MarkovRewardProcess, gamma: float, num_time_steps: int
) -> dict[str, float]:
    values = {state: 0.0 for state in mrp.states}
    for _ in range(num_time_steps):
        next_values: dict[str, float] = {}
        for state in mrp.states:
            total = 0.0
            for transition in mrp.transitions[state]:
                total += transition.probability * (
                    transition.reward + gamma * values[transition.next_state]
                )
            next_values[state] = total
        values = next_values
    return values


def transition_matrix_and_reward_vector(
    mrp: MarkovRewardProcess,
) -> tuple[np.ndarray, np.ndarray]:
    state_to_index = {state: index for index, state in enumerate(mrp.states)}
    num_states = len(mrp.states)
    transition_matrix = np.zeros((num_states, num_states))
    reward_vector = np.zeros(num_states)
    for state in mrp.states:
        row_index = state_to_index[state]
        for transition in mrp.transitions[state]:
            col_index = state_to_index[transition.next_state]
            transition_matrix[row_index, col_index] += transition.probability
            reward_vector[row_index] += transition.probability * transition.reward
    return transition_matrix, reward_vector


def exact_infinite_horizon_values(
    mrp: MarkovRewardProcess, gamma: float
) -> dict[str, float] | None:
    if gamma >= 1.0:
        return None
    transition_matrix, reward_vector = transition_matrix_and_reward_vector(mrp)
    identity = np.eye(len(mrp.states))
    values = np.linalg.solve(identity - gamma * transition_matrix, reward_vector)
    return {state: float(values[index]) for index, state in enumerate(mrp.states)}


def sample_transition(
    transitions: list[Transition], rng: random.Random
) -> Transition:
    threshold = rng.random()
    cumulative = 0.0
    for transition in transitions:
        cumulative += transition.probability
        if threshold <= cumulative:
            return transition
    return transitions[-1]


def monte_carlo_summary(
    mrp: MarkovRewardProcess,
    start_state: str,
    gamma: float,
    num_time_steps: int,
    num_trajectories: int,
    rng: random.Random,
) -> dict[str, float]:
    returns: list[float] = []
    for _ in range(num_trajectories):
        current_state = start_state
        discount = 1.0
        total_return = 0.0
        for _ in range(num_time_steps):
            transition = sample_transition(mrp.transitions[current_state], rng)
            total_return += discount * transition.reward
            discount *= gamma
            current_state = transition.next_state
        returns.append(total_return)

    mean_value = sum(returns) / len(returns)
    if len(returns) > 1:
        sample_variance = sum((value - mean_value) ** 2 for value in returns) / (
            len(returns) - 1
        )
        sample_std = math.sqrt(sample_variance)
        stderr = sample_std / math.sqrt(len(returns))
    else:
        sample_variance = 0.0
        sample_std = 0.0
        stderr = 0.0
    ci_half_width = 1.96 * stderr
    return {
        "value": mean_value,
        "sample_variance": sample_variance,
        "sample_std": sample_std,
        "stderr": stderr,
        "ci_lower_95": mean_value - ci_half_width,
        "ci_upper_95": mean_value + ci_half_width,
        "ci_width_95": 2.0 * ci_half_width,
    }


def build_rows(
    mrp: MarkovRewardProcess, config: ExperimentConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    exact_rows: list[dict[str, Any]] = []
    infinite_rows: list[dict[str, Any]] = []
    mc_rows: list[dict[str, Any]] = []
    gammas = gamma_grid(config.gamma_grid_num)
    traj_counts = trajectory_counts(config)

    for gamma_index, gamma in enumerate(gammas):
        exact_values = exact_finite_horizon_values(
            mrp=mrp, gamma=float(gamma), num_time_steps=config.num_time_steps
        )
        infinite_values = exact_infinite_horizon_values(mrp=mrp, gamma=float(gamma))
        exact_sorted = sorted(
            exact_values.items(), key=lambda item: item[1], reverse=True
        )
        for rank, (state, value) in enumerate(exact_sorted, start=1):
            exact_rows.append(
                {
                    "gamma": float(gamma),
                    "state": state,
                    "value": value,
                    "rank_within_gamma": rank,
                    "num_time_steps": config.num_time_steps,
                    "method": "exact",
                }
            )
        if infinite_values is not None:
            infinite_sorted = sorted(
                infinite_values.items(), key=lambda item: item[1], reverse=True
            )
            for rank, (state, value) in enumerate(infinite_sorted, start=1):
                infinite_rows.append(
                    {
                        "gamma": float(gamma),
                        "state": state,
                        "value": value,
                        "rank_within_gamma": rank,
                        "method": "exact_infinite",
                    }
                )

        for traj_index, traj_count in enumerate(traj_counts):
            rng = random.Random(
                config.seed + 10_000 * gamma_index + 1_000_000 * traj_index
            )
            mc_summaries = {
                state: monte_carlo_summary(
                    mrp=mrp,
                    start_state=state,
                    gamma=float(gamma),
                    num_time_steps=config.num_time_steps,
                    num_trajectories=traj_count,
                    rng=rng,
                )
                for state in mrp.states
            }
            mc_sorted = sorted(
                ((state, summary["value"]) for state, summary in mc_summaries.items()),
                key=lambda item: item[1],
                reverse=True,
            )
            for rank, (state, value) in enumerate(mc_sorted, start=1):
                summary = mc_summaries[state]
                mc_rows.append(
                    {
                        "gamma": float(gamma),
                        "state": state,
                        "value": value,
                        "sample_variance": summary["sample_variance"],
                        "sample_std": summary["sample_std"],
                        "stderr": summary["stderr"],
                        "ci_lower_95": summary["ci_lower_95"],
                        "ci_upper_95": summary["ci_upper_95"],
                        "ci_width_95": summary["ci_width_95"],
                        "rank_within_gamma": rank,
                        "num_time_steps": config.num_time_steps,
                        "num_trajectories": traj_count,
                        "method": "monte_carlo",
                    }
                )

    return exact_rows, infinite_rows, mc_rows


def build_horizon_sweep_rows(
    mrp: MarkovRewardProcess, config: ExperimentConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    finite_exact_rows: list[dict[str, Any]] = []
    infinite_rows: list[dict[str, Any]] = []
    mc_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    gammas = gamma_grid(config.gamma_grid_num)
    horizons = horizon_values(config)

    infinite_lookup: dict[tuple[float, str], float] = {}
    for gamma in gammas:
        infinite_values = exact_infinite_horizon_values(mrp=mrp, gamma=float(gamma))
        if infinite_values is None:
            continue
        infinite_sorted = sorted(infinite_values.items(), key=lambda item: item[1], reverse=True)
        for rank, (state, value) in enumerate(infinite_sorted, start=1):
            infinite_rows.append(
                {
                    "gamma": float(gamma),
                    "state": state,
                    "value": value,
                    "rank_within_gamma": rank,
                    "method": "exact_infinite",
                }
            )
            infinite_lookup[(float(gamma), state)] = value

    for horizon_index, horizon in enumerate(horizons):
        horizon_config = ExperimentConfig(
            gamma_grid_num=config.gamma_grid_num,
            num_time_steps=horizon,
            num_trajectories=config.num_trajectories,
            seed=config.seed + 1_000_000_000 * horizon_index,
            traj_list=None,
            horizon_list=None,
        )
        exact_rows, _, horizon_mc_rows = build_rows(mrp=mrp, config=horizon_config)
        finite_exact_rows.extend(exact_rows)
        mc_rows.extend(horizon_mc_rows)

        exact_lookup = {
            (row["gamma"], row["state"]): row for row in exact_rows
        }
        for mc_row in horizon_mc_rows:
            key = (mc_row["gamma"], mc_row["state"])
            exact_row = exact_lookup[key]
            comparison_record = {
                "gamma": mc_row["gamma"],
                "state": mc_row["state"],
                "num_time_steps": horizon,
                "num_trajectories": mc_row["num_trajectories"],
                "finite_exact_value": exact_row["value"],
                "mc_value": mc_row["value"],
                "mc_sample_variance": mc_row["sample_variance"],
                "mc_sample_std": mc_row["sample_std"],
                "mc_stderr": mc_row["stderr"],
                "mc_ci_lower_95": mc_row["ci_lower_95"],
                "mc_ci_upper_95": mc_row["ci_upper_95"],
                "mc_ci_width_95": mc_row["ci_width_95"],
                "finite_vs_mc_abs_error": abs(exact_row["value"] - mc_row["value"]),
            }
            if key in infinite_lookup:
                infinite_value = infinite_lookup[key]
                comparison_record["infinite_exact_value"] = infinite_value
                comparison_record["finite_vs_infinite_abs_gap"] = abs(exact_row["value"] - infinite_value)
                comparison_record["mc_vs_infinite_abs_gap"] = abs(mc_row["value"] - infinite_value)
            else:
                comparison_record["infinite_exact_value"] = ""
                comparison_record["finite_vs_infinite_abs_gap"] = ""
                comparison_record["mc_vs_infinite_abs_gap"] = ""
            comparison_rows.append(comparison_record)

    return finite_exact_rows, infinite_rows, mc_rows, comparison_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_ranked_markdown(
    path: Path,
    exact_rows: list[dict[str, Any]],
    infinite_rows: list[dict[str, Any]],
    mc_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = [
        "# Ranked State Values",
        "",
        "Each gamma block is sorted by decreasing value.",
        "",
        "## Exact",
        "",
    ]
    gammas = sorted({row["gamma"] for row in exact_rows})
    for gamma in gammas:
        lines.append(f"### gamma = {gamma:.2f}")
        lines.append("")
        lines.append("| Rank | State | Value |")
        lines.append("| --- | --- | ---: |")
        gamma_rows = [row for row in exact_rows if row["gamma"] == gamma]
        gamma_rows.sort(key=lambda row: row["rank_within_gamma"])
        for row in gamma_rows:
            lines.append(
                f"| {row['rank_within_gamma']} | {row['state']} | {row['value']:.6f} |"
            )
        lines.append("")

    if infinite_rows:
        lines.extend(["## Exact Infinite Horizon (gamma < 1)", ""])
        infinite_gammas = sorted({row["gamma"] for row in infinite_rows})
        for gamma in infinite_gammas:
            lines.append(f"### gamma = {gamma:.2f}")
            lines.append("")
            lines.append("| Rank | State | Value |")
            lines.append("| --- | --- | ---: |")
            gamma_rows = [row for row in infinite_rows if row["gamma"] == gamma]
            gamma_rows.sort(key=lambda row: row["rank_within_gamma"])
            for row in gamma_rows:
                lines.append(
                    f"| {row['rank_within_gamma']} | {row['state']} | {row['value']:.6f} |"
                )
            lines.append("")

    traj_counts = sorted({row["num_trajectories"] for row in mc_rows})
    for traj_count in traj_counts:
        lines.append(f"## Monte Carlo ({traj_count} trajectories)")
        lines.append("")
        for gamma in gammas:
            lines.append(f"### gamma = {gamma:.2f}")
            lines.append("")
            lines.append("| Rank | State | Value [95% CI] |")
            lines.append("| --- | --- | ---: |")
            gamma_rows = [
                row
                for row in mc_rows
                if row["gamma"] == gamma and row["num_trajectories"] == traj_count
            ]
            gamma_rows.sort(key=lambda row: row["rank_within_gamma"])
            for row in gamma_rows:
                lines.append(
                    f"| {row['rank_within_gamma']} | {row['state']} | "
                    f"{row['value']:.6f} [{row['ci_lower_95']:.6f}, {row['ci_upper_95']:.6f}] |"
                )
            lines.append("")
    path.write_text("\n".join(lines))


def build_comparison_rows(
    exact_rows: list[dict[str, Any]], mc_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    exact_lookup = {(row["gamma"], row["state"]): row for row in exact_rows}
    comparison_rows: list[dict[str, Any]] = []
    for mc_row in sorted(
        mc_rows, key=lambda row: (row["num_trajectories"], row["gamma"], row["state"])
    ):
        exact_row = exact_lookup[(mc_row["gamma"], mc_row["state"])]
        comparison_rows.append(
            {
                "gamma": exact_row["gamma"],
                "state": exact_row["state"],
                "num_trajectories": mc_row["num_trajectories"],
                "exact_value": exact_row["value"],
                "mc_value": mc_row["value"],
                "mc_sample_variance": mc_row["sample_variance"],
                "mc_sample_std": mc_row["sample_std"],
                "mc_stderr": mc_row["stderr"],
                "mc_ci_lower_95": mc_row["ci_lower_95"],
                "mc_ci_upper_95": mc_row["ci_upper_95"],
                "mc_ci_width_95": mc_row["ci_width_95"],
                "absolute_error": abs(exact_row["value"] - mc_row["value"]),
                "exact_rank_within_gamma": exact_row["rank_within_gamma"],
                "mc_rank_within_gamma": mc_row["rank_within_gamma"],
                "num_time_steps": exact_row["num_time_steps"],
            }
        )
    return comparison_rows


def build_finite_vs_infinite_rows(
    exact_rows: list[dict[str, Any]], infinite_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    finite_lookup = {(row["gamma"], row["state"]): row for row in exact_rows}
    gap_rows: list[dict[str, Any]] = []
    for infinite_row in sorted(infinite_rows, key=lambda row: (row["gamma"], row["state"])):
        finite_row = finite_lookup[(infinite_row["gamma"], infinite_row["state"])]
        gap_rows.append(
            {
                "gamma": infinite_row["gamma"],
                "state": infinite_row["state"],
                "finite_value": finite_row["value"],
                "infinite_value": infinite_row["value"],
                "absolute_gap": abs(infinite_row["value"] - finite_row["value"]),
                "signed_gap": infinite_row["value"] - finite_row["value"],
                "num_time_steps": finite_row["num_time_steps"],
            }
        )
    return gap_rows


def plot_single_run_value_vs_gamma(
    path: Path,
    exact_rows: list[dict[str, Any]],
    infinite_rows: list[dict[str, Any]],
    mc_rows: list[dict[str, Any]],
    states: list[str],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5), sharey=False)
    for axis, rows, title in (
        (axes[0], exact_rows, "Exact Finite-Horizon Values"),
        (axes[1], infinite_rows, "Exact Infinite-Horizon Values"),
        (axes[2], mc_rows, "Monte Carlo Estimates"),
    ):
        for state in states:
            state_rows = sorted(
                (row for row in rows if row["state"] == state),
                key=lambda row: row["gamma"],
            )
            x_values = [row["gamma"] for row in state_rows]
            y_values = [row["value"] for row in state_rows]
            axis.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=state,
            )
            if title == "Monte Carlo Estimates":
                axis.fill_between(
                    x_values,
                    [row["ci_lower_95"] for row in state_rows],
                    [row["ci_upper_95"] for row in state_rows],
                    alpha=0.15,
                )
        axis.set_title(title)
        axis.set_xlabel("gamma")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("state value")
    axes[1].set_ylabel("state value")
    axes[2].set_ylabel("state value")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_trajectory_sweep_value_vs_gamma(
    path: Path,
    exact_rows: list[dict[str, Any]],
    infinite_rows: list[dict[str, Any]],
    mc_rows: list[dict[str, Any]],
    states: list[str],
) -> None:
    traj_counts = sorted({row["num_trajectories"] for row in mc_rows})
    num_panels = len(traj_counts) + 2
    ncols = 2
    nrows = math.ceil(num_panels / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7 * ncols, 4.6 * nrows), sharex=True, sharey=True
    )
    axes_flat = list(np.atleast_1d(axes).flatten())

    panels: list[tuple[str, list[dict[str, Any]], int | None]] = [
        ("Exact finite-horizon values", exact_rows, None),
        ("Exact infinite-horizon values", infinite_rows, None),
    ]
    panels.extend(
        (f"Monte Carlo ({traj_count} trajectories)", mc_rows, traj_count)
        for traj_count in traj_counts
    )

    for axis, (title, rows, traj_count) in zip(axes_flat, panels):
        for state in states:
            if traj_count is None:
                state_rows = sorted(
                    (row for row in rows if row["state"] == state),
                    key=lambda row: row["gamma"],
                )
            else:
                state_rows = sorted(
                    (
                        row
                        for row in rows
                        if row["state"] == state
                        and row["num_trajectories"] == traj_count
                    ),
                    key=lambda row: row["gamma"],
                )
            x_values = [row["gamma"] for row in state_rows]
            y_values = [row["value"] for row in state_rows]
            axis.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=state,
            )
            if traj_count is not None:
                axis.fill_between(
                    x_values,
                    [row["ci_lower_95"] for row in state_rows],
                    [row["ci_upper_95"] for row in state_rows],
                    alpha=0.15,
                )
        axis.set_title(title)
        axis.set_xlabel("gamma")
        axis.set_ylabel("state value")
        axis.grid(True, alpha=0.3)

    for axis in axes_flat[num_panels:]:
        axis.axis("off")

    axes_flat[min(1, num_panels - 1)].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_single_run_exact_vs_mc(
    path: Path, exact_rows: list[dict[str, Any]], mc_rows: list[dict[str, Any]]
) -> None:
    exact_lookup = {(row["gamma"], row["state"]): row["value"] for row in exact_rows}
    fig, ax = plt.subplots(figsize=(8, 6))
    states = sorted({row["state"] for row in mc_rows})
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_lookup = {
        state: colors[index % len(colors)] for index, state in enumerate(states)
    }
    all_values: list[float] = []

    for state in states:
        state_rows = sorted(
            (row for row in mc_rows if row["state"] == state),
            key=lambda row: row["gamma"],
        )
        x_values = [exact_lookup[(row["gamma"], row["state"])] for row in state_rows]
        y_values = [row["value"] for row in state_rows]
        y_errors = [1.96 * row["stderr"] for row in state_rows]
        all_values.extend(x_values)
        all_values.extend(y_values)
        ax.errorbar(
            x_values,
            y_values,
            yerr=y_errors,
            fmt="o",
            capsize=3,
            markersize=5,
            color=color_lookup[state],
            ecolor=color_lookup[state],
            alpha=0.85,
            label=state,
        )

    lower = min(all_values)
    upper = max(all_values)
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1)
    ax.set_title("Monte Carlo vs Exact Values")
    ax.set_xlabel("exact value")
    ax.set_ylabel("Monte Carlo estimate")
    ax.grid(True, alpha=0.3)
    ax.legend(title="state")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_finite_vs_infinite_values(
    path: Path,
    exact_rows: list[dict[str, Any]],
    infinite_rows: list[dict[str, Any]],
    states: list[str],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for state in states:
        finite_state_rows = sorted(
            (row for row in exact_rows if row["state"] == state and row["gamma"] < 1.0),
            key=lambda row: row["gamma"],
        )
        infinite_state_rows = sorted(
            (row for row in infinite_rows if row["state"] == state),
            key=lambda row: row["gamma"],
        )
        axes[0].plot(
            [row["gamma"] for row in finite_state_rows],
            [row["value"] for row in finite_state_rows],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=f"{state} finite",
        )
        axes[0].plot(
            [row["gamma"] for row in infinite_state_rows],
            [row["value"] for row in infinite_state_rows],
            linestyle="--",
            linewidth=1.6,
            label=f"{state} infinite",
        )
    axes[0].set_title("Finite vs Infinite Exact Values")
    axes[0].set_xlabel("gamma")
    axes[0].set_ylabel("value")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=2, fontsize=8)

    for state in states:
        gap_rows = sorted(
            (
                {
                    "gamma": infinite_row["gamma"],
                    "gap": infinite_row["value"]
                    - next(
                        row["value"]
                        for row in exact_rows
                        if row["gamma"] == infinite_row["gamma"] and row["state"] == state
                    ),
                }
                for infinite_row in infinite_rows
                if infinite_row["state"] == state
            ),
            key=lambda row: row["gamma"],
        )
        axes[1].plot(
            [row["gamma"] for row in gap_rows],
            [row["gap"] for row in gap_rows],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=state,
        )
    axes[1].set_title("Infinite Minus Finite Exact Value")
    axes[1].set_xlabel("gamma")
    axes[1].set_ylabel("gap")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_truncation_gap_vs_gamma(path: Path, gap_rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    states = sorted({row["state"] for row in gap_rows})
    for state in states:
        state_rows = sorted(
            (row for row in gap_rows if row["state"] == state),
            key=lambda row: row["gamma"],
        )
        ax.plot(
            [row["gamma"] for row in state_rows],
            [row["absolute_gap"] for row in state_rows],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=state,
        )
    ax.set_title("Absolute Truncation Gap vs Gamma")
    ax.set_xlabel("gamma")
    ax.set_ylabel("|infinite - finite|")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_horizon_sweep_gap_vs_horizon(
    path: Path, comparison_rows: list[dict[str, Any]]
) -> None:
    horizons = sorted({row["num_time_steps"] for row in comparison_rows})
    gammas = sorted(
        {
            row["gamma"]
            for row in comparison_rows
            if row["infinite_exact_value"] != ""
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    for gamma in gammas:
        finite_gap = []
        mc_gap = []
        for horizon in horizons:
            rows = [
                row for row in comparison_rows
                if row["gamma"] == gamma and row["num_time_steps"] == horizon
                and row["infinite_exact_value"] != ""
            ]
            finite_gap.append(np.mean([row["finite_vs_infinite_abs_gap"] for row in rows]))
            mc_gap.append(np.mean([row["mc_vs_infinite_abs_gap"] for row in rows]))
        axes[0].plot(horizons, finite_gap, marker="o", linewidth=1.8, label=f"gamma={gamma:.2f}")
        axes[1].plot(horizons, mc_gap, marker="o", linewidth=1.8, label=f"gamma={gamma:.2f}")

    axes[0].set_title("Finite Exact vs Infinite Exact")
    axes[0].set_xlabel("horizon")
    axes[0].set_ylabel("mean absolute gap")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Monte Carlo vs Infinite Exact")
    axes[1].set_xlabel("horizon")
    axes[1].set_ylabel("mean absolute gap")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_horizon_sweep_values_vs_horizon(
    path: Path,
    finite_exact_rows: list[dict[str, Any]],
    infinite_rows: list[dict[str, Any]],
    mc_rows: list[dict[str, Any]],
    states: list[str],
) -> None:
    selected_gammas = sorted({row["gamma"] for row in infinite_rows})
    ncols = 2
    nrows = math.ceil(len(selected_gammas) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.8 * nrows), sharex=True)
    axes_flat = list(np.atleast_1d(axes).flatten())
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_lookup = {state: color_cycle[i % len(color_cycle)] for i, state in enumerate(states)}

    for axis, gamma in zip(axes_flat, selected_gammas):
        for state in states:
            finite_state_rows = sorted(
                (row for row in finite_exact_rows if row["gamma"] == gamma and row["state"] == state),
                key=lambda row: row["num_time_steps"],
            )
            mc_state_rows = sorted(
                (row for row in mc_rows if row["gamma"] == gamma and row["state"] == state),
                key=lambda row: row["num_time_steps"],
            )
            infinite_value = next(
                row["value"] for row in infinite_rows if row["gamma"] == gamma and row["state"] == state
            )
            horizons = [row["num_time_steps"] for row in finite_state_rows]
            axis.plot(
                horizons,
                [row["value"] for row in finite_state_rows],
                marker="o",
                linewidth=1.8,
                color=color_lookup[state],
                label=f"{state} finite",
            )
            axis.errorbar(
                horizons,
                [row["value"] for row in mc_state_rows],
                yerr=[1.96 * row["stderr"] for row in mc_state_rows],
                fmt="s",
                markersize=4,
                linewidth=1.2,
                color=color_lookup[state],
                alpha=0.8,
            )
            axis.hlines(
                infinite_value,
                xmin=min(horizons),
                xmax=max(horizons),
                colors=color_lookup[state],
                linestyles="--",
                linewidth=1.2,
            )
        axis.set_title(f"gamma = {gamma:.2f}")
        axis.set_xlabel("horizon")
        axis.set_ylabel("value")
        axis.grid(True, alpha=0.3)

    for axis in axes_flat[len(selected_gammas):]:
        axis.axis("off")

    axes_flat[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_trajectory_sweep_exact_vs_mc(
    path: Path, exact_rows: list[dict[str, Any]], mc_rows: list[dict[str, Any]]
) -> None:
    exact_lookup = {(row["gamma"], row["state"]): row["value"] for row in exact_rows}
    traj_counts = sorted({row["num_trajectories"] for row in mc_rows})
    ncols = 2
    nrows = math.ceil(len(traj_counts) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.2 * nrows))
    axes_flat = list(np.atleast_1d(axes).flatten())
    states = sorted({row["state"] for row in mc_rows})
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_lookup = {
        state: colors[index % len(colors)] for index, state in enumerate(states)
    }

    for axis, traj_count in zip(axes_flat, traj_counts):
        traj_rows = [row for row in mc_rows if row["num_trajectories"] == traj_count]
        all_values: list[float] = []
        for state in states:
            state_rows = sorted(
                (row for row in traj_rows if row["state"] == state),
                key=lambda row: row["gamma"],
            )
            x_values = [exact_lookup[(row["gamma"], row["state"])] for row in state_rows]
            y_values = [row["value"] for row in state_rows]
            y_errors = [1.96 * row["stderr"] for row in state_rows]
            all_values.extend(x_values)
            all_values.extend(y_values)
            axis.errorbar(
                x_values,
                y_values,
                yerr=y_errors,
                fmt="o",
                capsize=3,
                markersize=4,
                color=color_lookup[state],
                ecolor=color_lookup[state],
                alpha=0.85,
                label=state,
            )
        lower = min(all_values)
        upper = max(all_values)
        axis.plot(
            [lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1
        )
        axis.set_title(f"{traj_count} trajectories")
        axis.set_xlabel("exact value")
        axis.set_ylabel("Monte Carlo estimate")
        axis.grid(True, alpha=0.3)

    for axis in axes_flat[len(traj_counts):]:
        axis.axis("off")

    axes_flat[0].legend(title="state")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_error_vs_num_trajectories(path: Path, comparison_rows: list[dict[str, Any]]) -> None:
    traj_counts = sorted({row["num_trajectories"] for row in comparison_rows})
    mean_abs_error = [
        np.mean(
            [
                row["absolute_error"]
                for row in comparison_rows
                if row["num_trajectories"] == traj_count
            ]
        )
        for traj_count in traj_counts
    ]
    max_abs_error = [
        np.max(
            [
                row["absolute_error"]
                for row in comparison_rows
                if row["num_trajectories"] == traj_count
            ]
        )
        for traj_count in traj_counts
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(traj_counts, mean_abs_error, marker="o", linewidth=1.8, label="mean abs error")
    ax.plot(traj_counts, max_abs_error, marker="s", linewidth=1.8, label="max abs error")
    ax.set_title("Monte Carlo Error vs Number of Trajectories")
    ax.set_xlabel("number of trajectories")
    ax.set_ylabel("error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_variance_vs_num_trajectories(
    path: Path, comparison_rows: list[dict[str, Any]]
) -> None:
    traj_counts = sorted({row["num_trajectories"] for row in comparison_rows})
    mean_sample_variance = [
        np.mean(
            [
                row["mc_sample_variance"]
                for row in comparison_rows
                if row["num_trajectories"] == traj_count
            ]
        )
        for traj_count in traj_counts
    ]
    mean_stderr = [
        np.mean(
            [
                row["mc_stderr"]
                for row in comparison_rows
                if row["num_trajectories"] == traj_count
            ]
        )
        for traj_count in traj_counts
    ]
    mean_ci_width = [
        np.mean(
            [
                row["mc_ci_width_95"]
                for row in comparison_rows
                if row["num_trajectories"] == traj_count
            ]
        )
        for traj_count in traj_counts
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(
        traj_counts,
        mean_sample_variance,
        marker="o",
        linewidth=1.8,
        color="#1f77b4",
    )
    axes[0].set_title("Average Return Variance vs Number of Trajectories")
    axes[0].set_xlabel("number of trajectories")
    axes[0].set_ylabel("average sample variance")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        traj_counts,
        mean_stderr,
        marker="o",
        linewidth=1.8,
        label="mean stderr",
        color="#ff7f0e",
    )
    axes[1].plot(
        traj_counts,
        mean_ci_width,
        marker="s",
        linewidth=1.8,
        label="mean 95% CI width",
        color="#2ca02c",
    )
    axes[1].set_title("Estimator Uncertainty vs Number of Trajectories")
    axes[1].set_xlabel("number of trajectories")
    axes[1].set_ylabel("uncertainty")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_experiment(
    mrp: MarkovRewardProcess, config: ExperimentConfig, output_dir: Path
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.horizon_list:
        finite_exact_rows, infinite_rows, mc_rows, horizon_comparison_rows = build_horizon_sweep_rows(
            mrp=mrp, config=config
        )
        finite_exact_csv = output_dir / "state_values_finite_exact_by_horizon.csv"
        infinite_exact_csv = output_dir / "state_values_infinite_exact.csv"
        mc_csv = output_dir / "state_values_mc_by_horizon.csv"
        comparison_csv = output_dir / "horizon_sweep_comparison.csv"
        horizon_gap_plot = output_dir / "horizon_sweep_gap_vs_horizon.png"
        horizon_values_plot = output_dir / "horizon_sweep_values_vs_horizon.png"

        write_csv(finite_exact_csv, finite_exact_rows)
        write_csv(infinite_exact_csv, infinite_rows)
        write_csv(mc_csv, mc_rows)
        write_csv(comparison_csv, horizon_comparison_rows)
        plot_horizon_sweep_gap_vs_horizon(horizon_gap_plot, horizon_comparison_rows)
        plot_horizon_sweep_values_vs_horizon(
            horizon_values_plot, finite_exact_rows, infinite_rows, mc_rows, mrp.states
        )
        return {
            "finite_exact_csv": finite_exact_csv,
            "infinite_exact_csv": infinite_exact_csv,
            "mc_csv": mc_csv,
            "horizon_comparison_csv": comparison_csv,
            "horizon_gap_plot": horizon_gap_plot,
            "horizon_values_plot": horizon_values_plot,
        }

    exact_rows, infinite_rows, mc_rows = build_rows(mrp=mrp, config=config)
    comparison_rows = build_comparison_rows(exact_rows, mc_rows)
    gap_rows = build_finite_vs_infinite_rows(exact_rows, infinite_rows)

    exact_csv = output_dir / "state_values_exact.csv"
    infinite_exact_csv = output_dir / "state_values_infinite_exact.csv"
    mc_csv = output_dir / "state_values_mc.csv"
    comparison_csv = output_dir / "state_value_comparison.csv"
    gap_csv = output_dir / "finite_vs_infinite_gap.csv"
    ranked_md = output_dir / "ranked_state_values.md"
    value_plot = output_dir / "value_vs_gamma.png"
    comparison_plot = output_dir / "comparison_exact_vs_mc.png"
    finite_vs_infinite_plot = output_dir / "finite_vs_infinite_exact.png"
    truncation_gap_plot = output_dir / "truncation_gap_vs_gamma.png"

    write_csv(exact_csv, exact_rows)
    write_csv(infinite_exact_csv, infinite_rows)
    write_csv(mc_csv, mc_rows)
    write_csv(comparison_csv, comparison_rows)
    write_csv(gap_csv, gap_rows)
    write_ranked_markdown(ranked_md, exact_rows, infinite_rows, mc_rows)
    plot_finite_vs_infinite_values(
        finite_vs_infinite_plot, exact_rows, infinite_rows, mrp.states
    )
    plot_truncation_gap_vs_gamma(truncation_gap_plot, gap_rows)

    outputs = {
        "exact_csv": exact_csv,
        "infinite_exact_csv": infinite_exact_csv,
        "mc_csv": mc_csv,
        "comparison_csv": comparison_csv,
        "finite_vs_infinite_gap_csv": gap_csv,
        "ranked_markdown": ranked_md,
        "finite_vs_infinite_plot": finite_vs_infinite_plot,
        "truncation_gap_plot": truncation_gap_plot,
    }

    if len(trajectory_counts(config)) == 1:
        plot_single_run_value_vs_gamma(
            value_plot, exact_rows, infinite_rows, mc_rows, mrp.states
        )
        plot_single_run_exact_vs_mc(comparison_plot, exact_rows, mc_rows)
        outputs["value_plot"] = value_plot
        outputs["comparison_plot"] = comparison_plot
        return outputs

    traj_sweep_value_plot = output_dir / "value_vs_gamma_by_trajectory.png"
    traj_sweep_scatter_plot = output_dir / "comparison_exact_vs_mc_by_trajectory.png"
    error_plot = output_dir / "error_vs_num_trajectories.png"
    variance_plot = output_dir / "variance_vs_num_trajectories.png"

    plot_trajectory_sweep_value_vs_gamma(
        traj_sweep_value_plot, exact_rows, infinite_rows, mc_rows, mrp.states
    )
    plot_trajectory_sweep_exact_vs_mc(
        traj_sweep_scatter_plot, exact_rows, mc_rows
    )
    plot_error_vs_num_trajectories(error_plot, comparison_rows)
    plot_variance_vs_num_trajectories(variance_plot, comparison_rows)

    outputs["value_plot"] = traj_sweep_value_plot
    outputs["comparison_plot"] = traj_sweep_scatter_plot
    outputs["error_plot"] = error_plot
    outputs["variance_plot"] = variance_plot
    return outputs
