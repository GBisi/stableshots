#!/usr/bin/env python3
"""Analysis for the clean QPU handoff policy matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POLICY_ORDER = [
    "stableshots_keep",
    "stableshots_discard",
    "fixed_10k_keep",
    "fixed_10k_discard",
    "fixed_20k_keep",
    "fixed_20k_discard",
]
STOPPING_RULE_ORDER = ["stableshots", "fixed_10k", "fixed_20k"]


def add_stats(
    row: Dict[str, object],
    values: Iterable[float],
    prefix: str,
) -> None:
    series = pd.Series(list(values), dtype=float).dropna()
    row[f"{prefix}_n"] = int(len(series))
    row[f"{prefix}_min"] = float(series.min())
    row[f"{prefix}_q25"] = float(series.quantile(0.25))
    row[f"{prefix}_median"] = float(series.median())
    row[f"{prefix}_mean"] = float(series.mean())
    row[f"{prefix}_q75"] = float(series.quantile(0.75))
    row[f"{prefix}_max"] = float(series.max())
    row[f"{prefix}_std"] = float(series.std(ddof=1))


def summary_table(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    metrics: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for key, group in frame.groupby(list(group_cols), sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row: Dict[str, object] = {
            col: value for col, value in zip(group_cols, key)
        }
        (
            policy_id,
            stopping_rule,
            history_policy,
            circuit_key,
            algorithm,
            size,
            source_backend,
            failure_fraction,
        ) = key
        matched = baseline_lookup.loc[
            (circuit_key, source_backend, stopping_rule)
        ]
        source_no_failure_tvd = float(matched["source_no_failure_tvd_to_aer"])
        source_no_failure_shots = float(
            matched["source_no_failure_physical_shots"]
        )
        event_shot_threshold = (
            shot_threshold_fraction * source_no_failure_shots
        )
        row["rows"] = int(len(group))
        for metric in metrics:
            add_stats(row, group[metric], metric)
        rows.append(row)
    return pd.DataFrame(rows)


def derive_selection_events(
    handoff: pd.DataFrame,
    no_failure: pd.DataFrame,
    tvd_threshold: float,
    shot_threshold_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = no_failure[
        [
            "circuit_key",
            "backend",
            "stopping_rule",
            "final_tvd_to_aer",
            "total_physical_shots",
        ]
    ].rename(
        columns={
            "backend": "source_backend",
            "final_tvd_to_aer": "source_no_failure_tvd_to_aer",
            "total_physical_shots": "source_no_failure_physical_shots",
        }
    )

    baseline_lookup = baseline.set_index(
        ["circuit_key", "source_backend", "stopping_rule"]
    )

    rows: List[Dict[str, object]] = []
    candidate_gap_rows: List[Dict[str, object]] = []
    group_cols = [
        "policy_id",
        "stopping_rule",
        "history_policy",
        "circuit_key",
        "algorithm",
        "size",
        "source_backend",
        "failure_fraction",
    ]
    for key, group in handoff.groupby(group_cols, sort=True, dropna=False):
        if len(group) != 4:
            raise RuntimeError(
                f"expected four candidate targets for {key}, got {len(group)}"
            )
        row: Dict[str, object] = {
            col: value for col, value in zip(group_cols, key)
        }
        tvd = group["final_tvd_to_aer"].to_numpy(dtype=float)
        shots = group["total_physical_shots"].to_numpy(dtype=float)
        random_tvd = float(np.mean(tvd))
        random_shots = float(np.mean(shots))
        best_tvd = float(np.min(tvd))
        best_shots = float(np.min(shots))

        best_tvd_rows = group[np.isclose(group["final_tvd_to_aer"], best_tvd)]
        best_shot_rows = group[np.isclose(group["total_physical_shots"], best_shots)]

        shots_at_best_tvd = float(best_tvd_rows["total_physical_shots"].mean())
        tvd_at_best_shots = float(best_shot_rows["final_tvd_to_aer"].mean())

        tvd_gaps = tvd - best_tvd
        shot_gaps = shots - best_shots
        for candidate_index, (_, candidate) in enumerate(group.iterrows()):
            candidate_gap_rows.append(
                {
                    **{
                        col: value for col, value in zip(group_cols, key)
                    },
                    "target_backend": str(candidate["target_backend"]),
                    "candidate_tvd_to_aer": float(candidate["final_tvd_to_aer"]),
                    "candidate_physical_shots": float(
                        candidate["total_physical_shots"]
                    ),
                    "best_tvd": best_tvd,
                    "best_physical_shots": best_shots,
                    "tvd_gap_from_best": float(tvd_gaps[candidate_index]),
                    "shot_gap_from_best": float(shot_gaps[candidate_index]),
                    "source_no_failure_physical_shots": source_no_failure_shots,
                    "shot_practical_threshold": event_shot_threshold,
                }
            )

        row.update(
            {
                "candidate_targets": 4,
                "random_expected_tvd": random_tvd,
                "best_tvd": best_tvd,
                "best_tvd_targets": ";".join(
                    sorted(best_tvd_rows["target_backend"].astype(str))
                ),
                "random_expected_physical_shots": random_shots,
                "shots_at_best_tvd": shots_at_best_tvd,
                "r_tvd": random_tvd - best_tvd,
                "candidate_tvd_gap_min": float(np.min(tvd_gaps)),
                "candidate_tvd_gap_mean": float(np.mean(tvd_gaps)),
                "candidate_tvd_gap_median": float(np.median(tvd_gaps)),
                "candidate_tvd_gap_max": float(np.max(tvd_gaps)),
                "candidate_tvd_gap_std": float(np.std(tvd_gaps, ddof=1)),
                "random_minus_best_tvd_shots": random_shots - shots_at_best_tvd,
                "best_physical_shots": best_shots,
                "best_shot_targets": ";".join(
                    sorted(best_shot_rows["target_backend"].astype(str))
                ),
                "tvd_at_best_shots": tvd_at_best_shots,
                "r_shots": random_shots - best_shots,
                "candidate_shot_gap_min": float(np.min(shot_gaps)),
                "candidate_shot_gap_mean": float(np.mean(shot_gaps)),
                "candidate_shot_gap_median": float(np.median(shot_gaps)),
                "candidate_shot_gap_max": float(np.max(shot_gaps)),
                "candidate_shot_gap_std": float(np.std(shot_gaps, ddof=1)),
                "random_minus_best_shots_tvd": random_tvd - tvd_at_best_shots,
                "shot_practical_threshold": event_shot_threshold,
                "shot_practical_threshold_fraction": shot_threshold_fraction,
                "random_within_tvd_threshold": (
                    random_tvd - best_tvd <= tvd_threshold
                ),
                "random_within_shot_threshold": (
                    random_shots - best_shots <= event_shot_threshold
                ),
                "random_within_both_thresholds": (
                    random_tvd - best_tvd <= tvd_threshold
                    and random_shots - best_shots <= event_shot_threshold
                ),
            }
        )
        rows.append(row)

    events = pd.DataFrame(rows)
    events = events.merge(
        baseline,
        on=["circuit_key", "source_backend", "stopping_rule"],
        how="left",
        validate="many_to_one",
    )
    if events[
        ["source_no_failure_tvd_to_aer", "source_no_failure_physical_shots"]
    ].isna().any().any():
        raise RuntimeError("failed to join matched no-failure source baseline")

    events["random_delta_tvd_vs_no_failure"] = (
        events["random_expected_tvd"] - events["source_no_failure_tvd_to_aer"]
    )
    events["best_tvd_delta_vs_no_failure"] = (
        events["best_tvd"] - events["source_no_failure_tvd_to_aer"]
    )
    events["shot_oracle_tvd_delta_vs_no_failure"] = (
        events["tvd_at_best_shots"] - events["source_no_failure_tvd_to_aer"]
    )
    events["random_delta_shots_vs_no_failure"] = (
        events["random_expected_physical_shots"]
        - events["source_no_failure_physical_shots"]
    )
    events["best_tvd_delta_shots_vs_no_failure"] = (
        events["shots_at_best_tvd"]
        - events["source_no_failure_physical_shots"]
    )
    events["best_shots_delta_vs_no_failure"] = (
        events["best_physical_shots"]
        - events["source_no_failure_physical_shots"]
    )
    candidate_gaps = pd.DataFrame(candidate_gap_rows)
    return events, candidate_gaps


def draw_boxes(
    ax: plt.Axes,
    arrays: Sequence[np.ndarray],
    labels: Sequence[str],
    ylabel: str,
    title: str,
    threshold: float | None = None,
) -> None:
    ax.boxplot(arrays, labels=labels, showfliers=False)
    if threshold is not None:
        ax.axhline(threshold, linestyle="--", linewidth=1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_no_failure(
    no_failure: pd.DataFrame,
    backends: Sequence[str],
    plots_dir: Path,
) -> None:
    for metric, ylabel, filename in [
        ("final_tvd_to_aer", "Final TVD to Aer", "no_failure_tvd_boxplots.png"),
        (
            "total_physical_shots",
            "Total physical shots",
            "no_failure_shots_boxplots.png",
        ),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=False)
        for ax, rule in zip(axes, STOPPING_RULE_ORDER):
            arrays = [
                no_failure.loc[
                    (no_failure["stopping_rule"] == rule)
                    & (no_failure["backend"] == backend),
                    metric,
                ].to_numpy(dtype=float)
                for backend in backends
            ]
            draw_boxes(
                ax,
                arrays,
                [backend.replace("fake_", "") for backend in backends],
                ylabel,
                rule.replace("_", " "),
            )
            ax.tick_params(axis="x", rotation=30)
        fig.suptitle(f"No-failure baseline: {ylabel}")
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=180)
        plt.close(fig)


def plot_source_failure_distributions(
    handoff: pd.DataFrame,
    backends: Sequence[str],
    fractions: Sequence[float],
    plots_dir: Path,
) -> None:
    for policy in POLICY_ORDER:
        work = handoff[handoff["policy_id"] == policy]
        for metric, ylabel, suffix in [
            ("final_tvd_to_aer", "Final TVD to Aer", "tvd"),
            ("total_physical_shots", "Total physical shots", "shots"),
        ]:
            fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.2))
            axes_flat = list(axes.flat)
            for ax, source in zip(axes_flat, backends):
                arrays = [
                    work.loc[
                        (work["source_backend"] == source)
                        & np.isclose(work["failure_fraction"], fraction),
                        metric,
                    ].to_numpy(dtype=float)
                    for fraction in fractions
                ]
                draw_boxes(
                    ax,
                    arrays,
                    [f"{fraction:g}" for fraction in fractions],
                    ylabel,
                    source.replace("fake_", ""),
                )
                ax.set_xlabel("Failure fraction")
            axes_flat[-1].axis("off")
            fig.suptitle(
                f"{policy}: distribution over circuits, sizes and four targets"
            )
            fig.tight_layout()
            fig.savefig(
                plots_dir / f"source_failure_{policy}_{suffix}.png",
                dpi=180,
            )
            plt.close(fig)


def plot_regrets(
    events: pd.DataFrame,
    fractions: Sequence[float],
    plots_dir: Path,
    tvd_threshold: float,
    shot_threshold: float,
) -> None:
    for metric, ylabel, threshold, filename in [
        ("r_tvd", "R_TVD: random - best TVD", tvd_threshold, "selection_tvd_regret.png"),
        (
            "r_shots",
            "R_shots: random - minimum shots",
            shot_threshold,
            "selection_shot_regret.png",
        ),
    ]:
        fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.2))
        for ax, policy in zip(axes.flat, POLICY_ORDER):
            work = events[events["policy_id"] == policy]
            arrays = [
                work.loc[
                    np.isclose(work["failure_fraction"], fraction),
                    metric,
                ].to_numpy(dtype=float)
                for fraction in fractions
            ]
            draw_boxes(
                ax,
                arrays,
                [f"{fraction:g}" for fraction in fractions],
                ylabel,
                policy,
                threshold=threshold,
            )
            ax.set_xlabel("Failure fraction")
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=180)
        plt.close(fig)


def paired_boxplot(
    ax: plt.Axes,
    frame: pd.DataFrame,
    fractions: Sequence[float],
    left_metric: str,
    right_metric: str,
    ylabel: str,
    left_label: str,
    right_label: str,
) -> None:
    positions: List[float] = []
    arrays: List[np.ndarray] = []
    labels: List[str] = []
    for index, fraction in enumerate(fractions):
        center = index * 3.0
        arrays.extend(
            [
                frame.loc[
                    np.isclose(frame["failure_fraction"], fraction),
                    left_metric,
                ].to_numpy(dtype=float),
                frame.loc[
                    np.isclose(frame["failure_fraction"], fraction),
                    right_metric,
                ].to_numpy(dtype=float),
            ]
        )
        positions.extend([center - 0.45, center + 0.45])
        labels.extend([left_label, right_label])
    ax.boxplot(
        arrays,
        positions=positions,
        widths=0.7,
        showfliers=False,
    )
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xticks(
        [index * 3.0 for index in range(len(fractions))],
        [f"{fraction:g}" for fraction in fractions],
    )
    ax.set_xlabel("Failure fraction")
    ax.set_ylabel(ylabel)
    # Compact legend via invisible handles.
    ax.plot([], [], label=left_label)
    ax.plot([], [], label=right_label)
    ax.legend()


def plot_failure_impact(
    events: pd.DataFrame,
    fractions: Sequence[float],
    plots_dir: Path,
) -> None:
    for policy in POLICY_ORDER:
        work = events[events["policy_id"] == policy]
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
        paired_boxplot(
            axes[0],
            work,
            fractions,
            "random_delta_tvd_vs_no_failure",
            "best_tvd_delta_vs_no_failure",
            "Delta TVD vs matched no-failure source",
            "Random",
            "Best TVD target",
        )
        paired_boxplot(
            axes[1],
            work,
            fractions,
            "random_delta_shots_vs_no_failure",
            "best_shots_delta_vs_no_failure",
            "Delta physical shots vs matched no-failure source",
            "Random",
            "Minimum-shot target",
        )
        fig.suptitle(f"Failure impact relative to no failure: {policy}")
        fig.tight_layout()
        fig.savefig(
            plots_dir / f"failure_impact_{policy}.png",
            dpi=180,
        )
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/handoff_policy_matrix_config.json"),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())

    output_dir = Path(str(config["output_dir"]))
    raw_dir = output_dir / "raw"
    analysis_dir = output_dir / "analysis"
    plots_dir = output_dir / "plots"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    no_failure = pd.read_csv(raw_dir / "no_failure_policy_runs.csv")
    handoff = pd.read_csv(raw_dir / "handoff_policy_runs.csv")

    expected_nf = 36 * 5 * 3
    expected_handoff = 36 * 20 * 5 * 6
    if len(no_failure) != expected_nf:
        raise RuntimeError(f"expected {expected_nf} no-failure rows")
    if len(handoff) != expected_handoff:
        raise RuntimeError(f"expected {expected_handoff} handoff rows")

    thresholds = config["practical_thresholds"]
    tvd_threshold = float(thresholds["tvd"])
    shot_threshold_fraction = float(
        thresholds["shots_fraction_of_source_no_failure"]
    )

    no_failure_summary = summary_table(
        no_failure,
        ["stopping_rule", "backend"],
        ["final_tvd_to_aer", "total_physical_shots"],
    )
    source_failure_summary = summary_table(
        handoff,
        ["policy_id", "source_backend", "failure_fraction"],
        ["final_tvd_to_aer", "total_physical_shots"],
    )

    events, candidate_gaps = derive_selection_events(
        handoff,
        no_failure,
        tvd_threshold,
        shot_threshold_fraction,
    )

    event_metrics = [
        "random_expected_tvd",
        "best_tvd",
        "r_tvd",
        "candidate_tvd_gap_min",
        "candidate_tvd_gap_mean",
        "candidate_tvd_gap_median",
        "candidate_tvd_gap_max",
        "candidate_tvd_gap_std",
        "random_minus_best_tvd_shots",
        "random_expected_physical_shots",
        "best_physical_shots",
        "r_shots",
        "candidate_shot_gap_min",
        "candidate_shot_gap_mean",
        "candidate_shot_gap_median",
        "candidate_shot_gap_max",
        "candidate_shot_gap_std",
        "shot_practical_threshold",
        "random_minus_best_shots_tvd",
        "random_delta_tvd_vs_no_failure",
        "best_tvd_delta_vs_no_failure",
        "shot_oracle_tvd_delta_vs_no_failure",
        "random_delta_shots_vs_no_failure",
        "best_tvd_delta_shots_vs_no_failure",
        "best_shots_delta_vs_no_failure",
    ]

    summaries = {
        "selection_by_policy_failure": ["policy_id", "failure_fraction"],
        "selection_by_policy_source": ["policy_id", "source_backend"],
        "selection_by_policy_algorithm": ["policy_id", "algorithm"],
        "selection_by_policy_size": ["policy_id", "size"],
        "selection_by_policy_source_failure": [
            "policy_id",
            "source_backend",
            "failure_fraction",
        ],
    }
    summary_frames: Dict[str, pd.DataFrame] = {}
    for name, cols in summaries.items():
        summary = summary_table(events, cols, event_metrics)
        # Add practical-equivalence rates.
        rate_rows: List[Dict[str, object]] = []
        for key, group in events.groupby(cols, sort=True, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            rate_row = {col: value for col, value in zip(cols, key)}
            rate_row["p_random_within_tvd_threshold"] = float(
                group["random_within_tvd_threshold"].mean()
            )
            rate_row["p_random_within_shot_threshold"] = float(
                group["random_within_shot_threshold"].mean()
            )
            rate_row["p_random_within_both_thresholds"] = float(
                group["random_within_both_thresholds"].mean()
            )
            rate_rows.append(rate_row)
        summary = summary.merge(
            pd.DataFrame(rate_rows),
            on=cols,
            how="left",
            validate="one_to_one",
        )
        summary_frames[name] = summary

    no_failure_summary.to_csv(
        analysis_dir / "no_failure_qpu_summary.csv",
        index=False,
    )
    source_failure_summary.to_csv(
        analysis_dir / "handoff_source_failure_summary.csv",
        index=False,
    )
    events.to_csv(
        analysis_dir / "selection_event_metrics.csv",
        index=False,
    )
    candidate_gaps.to_csv(
        analysis_dir / "selection_candidate_gaps.csv",
        index=False,
    )
    candidate_gap_summary = summary_table(
        candidate_gaps,
        ["policy_id", "failure_fraction"],
        ["tvd_gap_from_best", "shot_gap_from_best"],
    )
    candidate_gap_summary.to_csv(
        analysis_dir / "candidate_gap_by_policy_failure.csv",
        index=False,
    )
    candidate_gap_source_failure = summary_table(
        candidate_gaps,
        ["policy_id", "source_backend", "failure_fraction"],
        ["tvd_gap_from_best", "shot_gap_from_best"],
    )
    candidate_gap_source_failure.to_csv(
        analysis_dir / "candidate_gap_by_policy_source_failure.csv",
        index=False,
    )
    for name, frame in summary_frames.items():
        frame.to_csv(analysis_dir / f"{name}.csv", index=False)

    backends = [str(value) for value in config["backends"]]
    fractions = [float(value) for value in config["failure_fractions"]]
    plot_no_failure(no_failure, backends, plots_dir)
    plot_source_failure_distributions(
        handoff,
        backends,
        fractions,
        plots_dir,
    )
    plot_regrets(
        events,
        fractions,
        plots_dir,
        tvd_threshold,
        None,
    )
    plot_failure_impact(events, fractions, plots_dir)

    manifest = {
        "no_failure_rows": int(len(no_failure)),
        "handoff_rows": int(len(handoff)),
        "selection_events": int(len(events)),
        "candidate_gap_rows": int(len(candidate_gaps)),
        "candidate_targets_per_event": 4,
        "random_policy": "exact uniform mean over four alternative target QPUs",
        "tvd_practical_threshold": tvd_threshold,
        "shot_practical_threshold_fraction_of_matched_source_no_failure": (
            shot_threshold_fraction
        ),
        "shot_threshold_definition": (
            "5% of matched no-failure physical shots for the same "
            "(circuit, size, source backend, stopping rule)"
        ),
        "primary_proposal": "stableshots_keep",
        "failure_fraction_basis": (
            "fraction of matched no-failure stopping count for same stopping rule"
        ),
        "analysis_units": {
            "no_failure": "(circuit, backend, stopping_rule)",
            "handoff_candidate": (
                "(policy, failure_fraction, source, circuit, target)"
            ),
            "selection_event": (
                "(policy, failure_fraction, source, circuit)"
            ),
        },
    }
    (analysis_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
