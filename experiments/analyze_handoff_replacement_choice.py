#!/usr/bin/env python3
"""Replacement-choice analysis for the clean QPU handoff experiment.

Each failure event is identified by (circuit, source backend, failure fraction)
and contains all four possible replacement backends. Uniform-random failover is
therefore evaluated exactly by averaging over the four observed candidates; no
Monte Carlo target selection is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVENT_COLS = [
    "circuit_key",
    "algorithm",
    "size",
    "source_backend",
    "failure_fraction",
]
TOLERANCES = [0.0, 0.01, 0.02, 0.05]


def deterministic_seed(base: int, *parts: object) -> int:
    payload = "|".join([str(base), *(str(part) for part in parts)])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def tolerance_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def cluster_bootstrap_median_ci(
    frame: pd.DataFrame,
    metric: str,
    repetitions: int,
    seed: int,
    cluster_col: str = "circuit_key",
    alpha: float = 0.05,
) -> Tuple[float, float]:
    work = frame[[cluster_col, metric]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work[np.isfinite(work[metric].to_numpy(dtype=float))]
    if work.empty:
        return float("nan"), float("nan")

    clusters = list(work[cluster_col].drop_duplicates())
    if len(clusters) < 2:
        value = float(work[metric].median())
        return value, value

    values_by_cluster = {
        cluster: work.loc[work[cluster_col] == cluster, metric].to_numpy(dtype=float)
        for cluster in clusters
    }
    rng = np.random.default_rng(seed)
    medians = np.empty(repetitions, dtype=float)
    for rep in range(repetitions):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        values = np.concatenate([values_by_cluster[clusters[index]] for index in sampled])
        medians[rep] = np.median(values)
    return (
        float(np.quantile(medians, alpha / 2)),
        float(np.quantile(medians, 1 - alpha / 2)),
    )


def validate_inputs(
    handoffs: pd.DataFrame,
    no_failure: pd.DataFrame,
    config: Mapping[str, object],
) -> None:
    expected_backends = [str(value) for value in config["backends"]]
    expected_fractions = [float(value) for value in config["failure_fractions"]]
    expected_circuits = len(config["algorithms"]) * len(config["sizes"])
    expected_events = expected_circuits * len(expected_backends) * len(expected_fractions)
    expected_handoffs = expected_events * (len(expected_backends) - 1)
    expected_no_failure = expected_circuits * len(expected_backends)

    if len(handoffs) != expected_handoffs:
        raise ValueError(f"expected {expected_handoffs} handoffs, found {len(handoffs)}")
    if len(no_failure) != expected_no_failure:
        raise ValueError(f"expected {expected_no_failure} no-failure rows, found {len(no_failure)}")

    if handoffs[EVENT_COLS + ["target_backend"]].duplicated().any():
        raise ValueError("duplicate handoff candidate rows")
    if no_failure[["circuit_key", "backend"]].duplicated().any():
        raise ValueError("duplicate no-failure rows")

    group_sizes = handoffs.groupby(EVENT_COLS, dropna=False).size()
    expected_targets = len(expected_backends) - 1
    if len(group_sizes) != expected_events or not (group_sizes == expected_targets).all():
        raise ValueError("each failure event must contain every alternative backend exactly once")

    if set(pd.to_numeric(handoffs["failure_fraction"]).unique()) != set(expected_fractions):
        raise ValueError("failure fractions do not match configuration")
    if (handoffs["source_backend"] == handoffs["target_backend"]).any():
        raise ValueError("source and target backend must differ")

    numeric = [
        "source_no_failure_shots",
        "source_no_failure_final_tvd_to_aer",
        "total_shots",
        "final_aggregated_tvd_to_aer",
    ]
    for column in numeric:
        values = pd.to_numeric(handoffs[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"missing/non-numeric values in {column}")


def pareto_dominated(group: pd.DataFrame) -> pd.Series:
    tvd = group["final_aggregated_tvd_to_aer"].to_numpy(dtype=float)
    shots = group["total_shots"].to_numpy(dtype=float)
    dominated = np.zeros(len(group), dtype=bool)
    for i in range(len(group)):
        no_worse = (tvd <= tvd[i]) & (shots <= shots[i])
        strictly_better = (tvd < tvd[i]) | (shots < shots[i])
        dominated[i] = bool(np.any(no_worse & strictly_better))
    return pd.Series(dominated, index=group.index)


def build_candidate_table(handoffs: pd.DataFrame, max_shots: int) -> pd.DataFrame:
    frame = handoffs.copy()
    frame["delta_tvd_vs_source"] = (
        frame["final_aggregated_tvd_to_aer"] - frame["source_no_failure_final_tvd_to_aer"]
    )
    frame["delta_shots_vs_source"] = frame["total_shots"] - frame["source_no_failure_shots"]
    frame["hit_max_budget"] = frame["total_shots"] >= max_shots

    for tolerance in TOLERANCES:
        label = tolerance_label(tolerance)
        frame[f"harm_gt_{label}"] = frame["delta_tvd_vs_source"] > tolerance
        frame[f"safe_within_{label}"] = frame["delta_tvd_vs_source"] <= tolerance

    frame["improves_tvd"] = frame["delta_tvd_vs_source"] < 0.0
    frame["pareto_dominated"] = False

    for _, group in frame.groupby(EVENT_COLS, sort=False, dropna=False):
        dominated = pareto_dominated(group)
        frame.loc[dominated.index, "pareto_dominated"] = dominated

        best_tvd = float(group["final_aggregated_tvd_to_aer"].min())
        min_shots = float(group["total_shots"].min())
        frame.loc[group.index, "oracle_best_tvd"] = best_tvd
        frame.loc[group.index, "oracle_min_shots"] = min_shots
        frame.loc[group.index, "tvd_regret_vs_oracle"] = (
            group["final_aggregated_tvd_to_aer"] - best_tvd
        )
        frame.loc[group.index, "shot_regret_vs_min"] = group["total_shots"] - min_shots
        frame.loc[group.index, "tvd_rank"] = group["final_aggregated_tvd_to_aer"].rank(
            method="min", ascending=True
        )
        frame.loc[group.index, "shots_rank"] = group["total_shots"].rank(
            method="min", ascending=True
        )

    return frame


def build_event_table(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for key, group in candidates.groupby(EVENT_COLS, sort=True, dropna=False):
        row: Dict[str, object] = {column: value for column, value in zip(EVENT_COLS, key)}
        row["candidate_targets"] = int(len(group))
        row["actual_failure_fraction"] = float(group["actual_failure_fraction"].iloc[0])
        row["source_no_failure_shots"] = int(group["source_no_failure_shots"].iloc[0])
        row["source_no_failure_tvd"] = float(group["source_no_failure_final_tvd_to_aer"].iloc[0])

        tvd = group["final_aggregated_tvd_to_aer"].to_numpy(dtype=float)
        delta_tvd = group["delta_tvd_vs_source"].to_numpy(dtype=float)
        shots = group["total_shots"].to_numpy(dtype=float)
        delta_shots = group["delta_shots_vs_source"].to_numpy(dtype=float)

        row.update({
            "random_expected_tvd": float(np.mean(tvd)),
            "random_median_tvd": float(np.median(tvd)),
            "random_expected_delta_tvd": float(np.mean(delta_tvd)),
            "random_median_delta_tvd": float(np.median(delta_tvd)),
            "random_expected_shots": float(np.mean(shots)),
            "random_median_shots": float(np.median(shots)),
            "random_expected_shot_overhead": float(np.mean(delta_shots)),
            "target_tvd_min": float(np.min(tvd)),
            "target_tvd_max": float(np.max(tvd)),
            "target_tvd_range": float(np.max(tvd) - np.min(tvd)),
            "target_tvd_std": float(np.std(tvd, ddof=1)),
            "target_shots_min": float(np.min(shots)),
            "target_shots_max": float(np.max(shots)),
            "target_shots_range": float(np.max(shots) - np.min(shots)),
            "target_shots_std": float(np.std(shots, ddof=1)),
            "p_random_improves": float(np.mean(group["improves_tvd"])),
            "p_random_hits_max_budget": float(np.mean(group["hit_max_budget"])),
            "p_random_dominated": float(np.mean(group["pareto_dominated"])),
            "oracle_best_tvd": float(np.min(tvd)),
            "oracle_best_delta_tvd": float(np.min(delta_tvd)),
            "random_tvd_regret": float(np.mean(tvd) - np.min(tvd)),
            "oracle_min_shots": float(np.min(shots)),
            "random_shot_regret": float(np.mean(shots) - np.min(shots)),
        })

        for tolerance in TOLERANCES:
            label = tolerance_label(tolerance)
            row[f"p_random_harm_gt_{label}"] = float(np.mean(group[f"harm_gt_{label}"]))
            row[f"p_random_safe_within_{label}"] = float(np.mean(group[f"safe_within_{label}"]))

        best_targets = group.loc[
            np.isclose(group["final_aggregated_tvd_to_aer"], row["oracle_best_tvd"]),
            "target_backend",
        ]
        min_shot_targets = group.loc[
            np.isclose(group["total_shots"], row["oracle_min_shots"]),
            "target_backend",
        ]
        row["oracle_best_tvd_targets"] = ";".join(sorted(best_targets.astype(str)))
        row["oracle_min_shot_targets"] = ";".join(sorted(min_shot_targets.astype(str)))
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_events(
    events: pd.DataFrame,
    group_cols: Sequence[str],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    metrics = [
        "random_expected_delta_tvd",
        "random_expected_tvd",
        "random_expected_shot_overhead",
        "random_expected_shots",
        "target_tvd_range",
        "target_tvd_std",
        "target_shots_range",
        "random_tvd_regret",
        "random_shot_regret",
        "p_random_improves",
        "p_random_hits_max_budget",
        "p_random_dominated",
        *[f"p_random_harm_gt_{tolerance_label(value)}" for value in TOLERANCES],
        *[f"p_random_safe_within_{tolerance_label(value)}" for value in TOLERANCES],
    ]
    rows: List[Dict[str, object]] = []
    for key, group in events.groupby(list(group_cols), sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row: Dict[str, object] = {
            column: value for column, value in zip(group_cols, key)
        }
        row["events"] = int(len(group))
        row["circuits"] = int(group["circuit_key"].nunique())
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"median_{metric}"] = float(np.median(values))
            row[f"q25_{metric}"] = float(np.quantile(values, 0.25))
            row[f"q75_{metric}"] = float(np.quantile(values, 0.75))
            low, high = cluster_bootstrap_median_ci(
                group,
                metric,
                repetitions,
                deterministic_seed(seed, *key, metric),
            )
            row[f"median_{metric}_ci_low"] = low
            row[f"median_{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def save_line(
    summary: pd.DataFrame,
    y: str,
    output: Path,
    ylabel: str,
    title: str,
    ci: bool = False,
) -> None:
    work = summary.sort_values("failure_fraction")
    x = work["failure_fraction"].to_numpy(dtype=float)
    yv = work[y].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(x, yv, marker="o")
    if ci:
        low = work[f"{y}_ci_low"].to_numpy(dtype=float)
        high = work[f"{y}_ci_high"].to_numpy(dtype=float)
        ax.fill_between(x, low, high, alpha=0.2)
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("Failure fraction")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_probability_lines(summary: pd.DataFrame, output: Path) -> None:
    work = summary.sort_values("failure_fraction")
    x = work["failure_fraction"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    for tolerance in TOLERANCES:
        label = tolerance_label(tolerance)
        column = f"mean_p_random_harm_gt_{label}"
        ax.plot(
            x,
            work[column].to_numpy(dtype=float),
            marker="o",
            label=f"Delta TVD > {tolerance:g}",
        )
    ax.set_xlabel("Failure fraction")
    ax.set_ylabel("Probability under uniform random replacement")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_title("Risk that a random replacement harms TVD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_heatmap(
    summary: pd.DataFrame,
    value_col: str,
    backends: Sequence[str],
    fractions: Sequence[float],
    output: Path,
    title: str,
    colorbar_label: str,
) -> None:
    matrix = np.full((len(backends), len(fractions)), np.nan)
    for i, backend in enumerate(backends):
        for j, fraction in enumerate(fractions):
            row = summary[
                (summary["source_backend"] == backend)
                & np.isclose(summary["failure_fraction"], fraction)
            ]
            if not row.empty:
                matrix[i, j] = float(row.iloc[0][value_col])

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(fractions)), labels=[f"{value:g}" for value in fractions])
    ax.set_yticks(range(len(backends)), labels=backends)
    ax.set_xlabel("Failure fraction")
    ax.set_ylabel("Failed source backend")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/handoff_baseline_config.json"),
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    output_dir = Path(str(config["output_dir"]))
    raw_dir = output_dir / "raw"
    analysis_dir = output_dir / "analysis"
    plots_dir = output_dir / "plots"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    handoffs = pd.read_csv(raw_dir / "handoff_runs.csv")
    no_failure = pd.read_csv(raw_dir / "no_failure_runs.csv")
    validate_inputs(handoffs, no_failure, config)

    max_shots = int(config["stableshots"]["max_shots"])
    repetitions = int(config["analysis"]["bootstrap_repetitions"])
    seed = int(config["analysis"]["bootstrap_seed"])

    candidates = build_candidate_table(handoffs, max_shots)
    events = build_event_table(candidates)

    expected_events = (
        len(config["algorithms"])
        * len(config["sizes"])
        * len(config["backends"])
        * len(config["failure_fractions"])
    )
    if len(events) != expected_events:
        raise RuntimeError(f"expected {expected_events} replacement events, found {len(events)}")

    failure_summary = summarize_events(
        events,
        ["failure_fraction"],
        repetitions,
        seed,
    )
    source_failure_summary = summarize_events(
        events,
        ["source_backend", "failure_fraction"],
        repetitions,
        seed,
    )
    source_summary = summarize_events(
        events,
        ["source_backend"],
        repetitions,
        seed,
    )

    candidates.to_csv(analysis_dir / "replacement_candidates_enriched.csv", index=False)
    events.to_csv(analysis_dir / "replacement_event_metrics.csv", index=False)
    failure_summary.to_csv(analysis_dir / "replacement_failure_summary.csv", index=False)
    source_failure_summary.to_csv(
        analysis_dir / "replacement_source_failure_summary.csv", index=False
    )
    source_summary.to_csv(analysis_dir / "replacement_source_summary.csv", index=False)

    save_line(
        failure_summary,
        "median_random_expected_delta_tvd",
        plots_dir / "replacement_random_delta_tvd_vs_failure.png",
        "Median expected Delta TVD vs staying on source",
        "Uniform-random replacement: accuracy effect",
        ci=True,
    )
    save_probability_lines(
        failure_summary,
        plots_dir / "replacement_random_harm_probability_vs_failure.png",
    )
    save_line(
        failure_summary,
        "median_target_tvd_range",
        plots_dir / "replacement_target_choice_tvd_range_vs_failure.png",
        "Median max-minus-min TVD across four targets",
        "How much replacement choice matters",
        ci=True,
    )
    save_line(
        failure_summary,
        "median_random_tvd_regret",
        plots_dir / "replacement_random_tvd_regret_vs_failure.png",
        "Median random-policy TVD regret vs oracle target",
        "Value left on the table by random replacement",
        ci=True,
    )
    save_line(
        failure_summary,
        "median_random_expected_shot_overhead",
        plots_dir / "replacement_random_shot_overhead_vs_failure.png",
        "Median expected shot overhead vs source no-failure",
        "Uniform-random replacement: shot cost",
        ci=True,
    )
    save_line(
        failure_summary,
        "mean_p_random_hits_max_budget",
        plots_dir / "replacement_random_max_budget_risk_vs_failure.png",
        "Probability of reaching max-shot budget",
        "Uniform-random replacement: max-budget risk",
    )
    save_line(
        failure_summary,
        "mean_p_random_dominated",
        plots_dir / "replacement_random_dominated_probability_vs_failure.png",
        "Probability random target is Pareto dominated",
        "Risk of choosing a strictly inferior replacement",
    )

    backends = [str(value) for value in config["backends"]]
    fractions = [float(value) for value in config["failure_fractions"]]
    save_heatmap(
        source_failure_summary,
        "median_random_expected_delta_tvd",
        backends,
        fractions,
        plots_dir / "replacement_source_delta_tvd_heatmap.png",
        "Expected random-replacement Delta TVD by failed source",
        "Median Delta TVD",
    )
    save_heatmap(
        source_failure_summary,
        "mean_p_random_harm_gt_0p02",
        backends,
        fractions,
        plots_dir / "replacement_source_harm_gt_0p02_heatmap.png",
        "Probability random replacement worsens TVD by > 0.02",
        "Probability",
    )
    save_heatmap(
        source_failure_summary,
        "median_random_tvd_regret",
        backends,
        fractions,
        plots_dir / "replacement_source_regret_heatmap.png",
        "Random-policy regret by failed source",
        "Median TVD regret",
    )

    manifest = {
        "candidate_rows": int(len(candidates)),
        "failure_events": int(len(events)),
        "candidate_targets_per_event": int(len(config["backends"]) - 1),
        "failure_summary_rows": int(len(failure_summary)),
        "source_failure_summary_rows": int(len(source_failure_summary)),
        "source_summary_rows": int(len(source_summary)),
        "random_policy": "uniform over the four available replacement QPUs",
        "random_policy_evaluation": "exact enumeration; no Monte Carlo target draws",
        "reference_counterfactual": "matched source-QPU no-failure StableShots run",
        "safety_tolerances_delta_tvd": TOLERANCES,
        "max_shots": max_shots,
    }
    (analysis_dir / "replacement_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
