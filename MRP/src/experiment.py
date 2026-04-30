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


def gamma_grid(grid_num: int) -> np.ndarray:
    if grid_num < 2:
        raise ValueError("--gamma-grid-num must be at least 2")
    return np.linspace(0.0, 1.0, grid_num)


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
        sample_std = 0.0
        stderr = 0.0
    ci_half_width = 1.96 * stderr
    return {
        "value": mean_value,
        "sample_std": sample_std,
        "stderr": stderr,
        "ci_lower_95": mean_value - ci_half_width,
        "ci_upper_95": mean_value + ci_half_width,
    }


def build_rows(
    mrp: MarkovRewardProcess, config: ExperimentConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exact_rows: list[dict[str, Any]] = []
    mc_rows: list[dict[str, Any]] = []
    gammas = gamma_grid(config.gamma_grid_num)

    for gamma_index, gamma in enumerate(gammas):
        exact_values = exact_finite_horizon_values(
            mrp=mrp, gamma=float(gamma), num_time_steps=config.num_time_steps
        )
        rng = random.Random(config.seed + gamma_index)
        mc_summaries = {
            state: monte_carlo_summary(
                mrp=mrp,
                start_state=state,
                gamma=float(gamma),
                num_time_steps=config.num_time_steps,
                num_trajectories=config.num_trajectories,
                rng=rng,
            )
            for state in mrp.states
        }
        mc_values = {state: summary["value"] for state, summary in mc_summaries.items()}

        exact_sorted = sorted(
            exact_values.items(), key=lambda item: item[1], reverse=True
        )
        mc_sorted = sorted(mc_values.items(), key=lambda item: item[1], reverse=True)

        for rank, (state, value) in enumerate(exact_sorted, start=1):
            exact_rows.append(
                {
                    "gamma": float(gamma),
                    "state": state,
                    "value": value,
                    "rank_within_gamma": rank,
                    "num_time_steps": config.num_time_steps,
                    "num_trajectories": config.num_trajectories,
                    "method": "exact",
                }
            )
        for rank, (state, value) in enumerate(mc_sorted, start=1):
            summary = mc_summaries[state]
            mc_rows.append(
                {
                    "gamma": float(gamma),
                    "state": state,
                    "value": value,
                    "sample_std": summary["sample_std"],
                    "stderr": summary["stderr"],
                    "ci_lower_95": summary["ci_lower_95"],
                    "ci_upper_95": summary["ci_upper_95"],
                    "rank_within_gamma": rank,
                    "num_time_steps": config.num_time_steps,
                    "num_trajectories": config.num_trajectories,
                    "method": "monte_carlo",
                }
            )

    return exact_rows, mc_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_ranked_markdown(
    path: Path, exact_rows: list[dict[str, Any]], mc_rows: list[dict[str, Any]]
) -> None:
    lines: list[str] = [
        "# Ranked State Values",
        "",
        "Each gamma block is sorted by decreasing value.",
        "",
    ]
    for method_name, rows in (("Exact", exact_rows), ("Monte Carlo", mc_rows)):
        lines.append(f"## {method_name}")
        lines.append("")
        gammas = sorted({row["gamma"] for row in rows})
        for gamma in gammas:
            lines.append(f"### gamma = {gamma:.2f}")
            lines.append("")
            lines.append("| Rank | State | Value |")
            lines.append("| --- | --- | ---: |")
            gamma_rows = [row for row in rows if row["gamma"] == gamma]
            gamma_rows.sort(key=lambda row: row["rank_within_gamma"])
            for row in gamma_rows:
                if method_name == "Monte Carlo":
                    value_text = (
                        f"{row['value']:.6f} "
                        f"[{row['ci_lower_95']:.6f}, {row['ci_upper_95']:.6f}]"
                    )
                else:
                    value_text = f"{row['value']:.6f}"
                lines.append(f"| {row['rank_within_gamma']} | {row['state']} | {value_text} |")
            lines.append("")
    path.write_text("\n".join(lines))


def build_comparison_rows(
    exact_rows: list[dict[str, Any]], mc_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    exact_lookup = {
        (row["gamma"], row["state"]): row
        for row in exact_rows
    }
    mc_lookup = {
        (row["gamma"], row["state"]): row
        for row in mc_rows
    }
    comparison_rows: list[dict[str, Any]] = []
    for key in sorted(exact_lookup.keys()):
        exact_row = exact_lookup[key]
        mc_row = mc_lookup[key]
        comparison_rows.append(
            {
                "gamma": exact_row["gamma"],
                "state": exact_row["state"],
                "exact_value": exact_row["value"],
                "mc_value": mc_row["value"],
                "mc_sample_std": mc_row["sample_std"],
                "mc_stderr": mc_row["stderr"],
                "mc_ci_lower_95": mc_row["ci_lower_95"],
                "mc_ci_upper_95": mc_row["ci_upper_95"],
                "absolute_error": abs(exact_row["value"] - mc_row["value"]),
                "exact_rank_within_gamma": exact_row["rank_within_gamma"],
                "mc_rank_within_gamma": mc_row["rank_within_gamma"],
                "num_time_steps": exact_row["num_time_steps"],
                "num_trajectories": exact_row["num_trajectories"],
            }
        )
    return comparison_rows


def plot_value_vs_gamma(
    path: Path,
    exact_rows: list[dict[str, Any]],
    mc_rows: list[dict[str, Any]],
    states: list[str],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, rows, title in (
        (axes[0], exact_rows, "Exact Finite-Horizon Values"),
        (axes[1], mc_rows, "Monte Carlo Estimates"),
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
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_exact_vs_mc(
    path: Path, exact_rows: list[dict[str, Any]], mc_rows: list[dict[str, Any]]
) -> None:
    exact_lookup = {
        (row["gamma"], row["state"]): row["value"] for row in exact_rows
    }
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


def run_experiment(
    mrp: MarkovRewardProcess, config: ExperimentConfig, output_dir: Path
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exact_rows, mc_rows = build_rows(mrp=mrp, config=config)

    exact_csv = output_dir / "state_values_exact.csv"
    mc_csv = output_dir / "state_values_mc.csv"
    comparison_csv = output_dir / "state_value_comparison.csv"
    ranked_md = output_dir / "ranked_state_values.md"
    value_plot = output_dir / "value_vs_gamma.png"
    comparison_plot = output_dir / "comparison_exact_vs_mc.png"
    comparison_rows = build_comparison_rows(exact_rows, mc_rows)

    write_csv(exact_csv, exact_rows)
    write_csv(mc_csv, mc_rows)
    write_csv(comparison_csv, comparison_rows)
    write_ranked_markdown(ranked_md, exact_rows, mc_rows)
    plot_value_vs_gamma(value_plot, exact_rows, mc_rows, mrp.states)
    plot_exact_vs_mc(comparison_plot, exact_rows, mc_rows)

    return {
        "exact_csv": exact_csv,
        "mc_csv": mc_csv,
        "comparison_csv": comparison_csv,
        "ranked_markdown": ranked_md,
        "value_plot": value_plot,
        "comparison_plot": comparison_plot,
    }
