#!/usr/bin/env python3
"""Policy-matrix analysis v2.

Changes relative to the first analysis:
- remove the synthetic random-circuit family from every analysis;
- evaluate practical equivalence at 5% and 10% for both TVD and shots;
- normalize shot regret by matched source no-failure physical shots;
- add Joint Normalized Regret (JNR), a single magnitude-aware score:
      J_tau = max(R_TVD / tau, (R_shots / N_source_nf) / tau)
  so J_tau <= 1 iff an event passes both thresholds;
- produce a strict cap-free StableShots version. Any backend/circuit/size triple
  whose no-failure StableShots run reaches the 20k max budget is excluded.
  Handoff events are retained only when all four alternative targets remain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POLICY_ORDER = [
    "stableshots_keep",
    "stableshots_discard",
    "fixed_20k_keep",
    "fixed_20k_discard",
]
STOPPING_RULE_ORDER = ["stableshots", "fixed_20k"]


def pct_label(value: float) -> str:
    return str(int(round(value * 100)))


def add_stats(row: Dict[str, object], values: Iterable[float], prefix: str) -> None:
    s = pd.Series(list(values), dtype=float).dropna()
    row[f"{prefix}_n"] = int(len(s))
    row[f"{prefix}_min"] = float(s.min())
    row[f"{prefix}_q25"] = float(s.quantile(0.25))
    row[f"{prefix}_median"] = float(s.median())
    row[f"{prefix}_mean"] = float(s.mean())
    row[f"{prefix}_q75"] = float(s.quantile(0.75))
    row[f"{prefix}_max"] = float(s.max())
    row[f"{prefix}_std"] = float(s.std(ddof=1))


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
        row["rows"] = int(len(group))
        for metric in metrics:
            add_stats(row, group[metric], metric)
        rows.append(row)
    return pd.DataFrame(rows)


def derive_selection_events(
    handoff: pd.DataFrame,
    no_failure: pd.DataFrame,
    thresholds: Sequence[float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    candidate_rows: List[Dict[str, object]] = []
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
        if len(group) < 1:
            continue

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

        lookup_key = (circuit_key, source_backend, stopping_rule)
        if lookup_key not in baseline_lookup.index:
            continue
        matched = baseline_lookup.loc[lookup_key]
        source_nf_tvd = float(matched["source_no_failure_tvd_to_aer"])
        source_nf_shots = float(matched["source_no_failure_physical_shots"])

        tvd = group["final_tvd_to_aer"].to_numpy(dtype=float)
        shots = group["total_physical_shots"].to_numpy(dtype=float)
        random_tvd = float(np.mean(tvd))
        random_shots = float(np.mean(shots))
        best_tvd = float(np.min(tvd))
        best_shots = float(np.min(shots))
        r_tvd = random_tvd - best_tvd
        r_shots = random_shots - best_shots
        r_shots_frac = r_shots / source_nf_shots

        best_tvd_rows = group[np.isclose(group["final_tvd_to_aer"], best_tvd)]
        best_shot_rows = group[np.isclose(group["total_physical_shots"], best_shots)]
        shots_at_best_tvd = float(best_tvd_rows["total_physical_shots"].mean())
        tvd_at_best_shots = float(best_shot_rows["final_tvd_to_aer"].mean())

        tvd_gaps = tvd - best_tvd
        shot_gaps = shots - best_shots
        shot_gap_fracs = shot_gaps / source_nf_shots

        base = {
            "policy_id": policy_id,
            "stopping_rule": stopping_rule,
            "history_policy": history_policy,
            "circuit_key": circuit_key,
            "algorithm": algorithm,
            "size": int(size),
            "source_backend": source_backend,
            "failure_fraction": float(failure_fraction),
            "candidate_targets": int(len(group)),
            "source_no_failure_tvd_to_aer": source_nf_tvd,
            "source_no_failure_physical_shots": source_nf_shots,
            "random_expected_tvd": random_tvd,
            "best_tvd": best_tvd,
            "best_tvd_targets": ";".join(
                sorted(best_tvd_rows["target_backend"].astype(str))
            ),
            "random_expected_physical_shots": random_shots,
            "best_physical_shots": best_shots,
            "best_shot_targets": ";".join(
                sorted(best_shot_rows["target_backend"].astype(str))
            ),
            "shots_at_best_tvd": shots_at_best_tvd,
            "tvd_at_best_shots": tvd_at_best_shots,
            "r_tvd": r_tvd,
            "r_shots": r_shots,
            "r_shots_fraction_of_source_no_failure": r_shots_frac,
            "random_minus_best_tvd_shots": random_shots - shots_at_best_tvd,
            "random_minus_best_shots_tvd": random_tvd - tvd_at_best_shots,
            "random_delta_tvd_vs_no_failure": random_tvd - source_nf_tvd,
            "best_tvd_delta_vs_no_failure": best_tvd - source_nf_tvd,
            "shot_oracle_tvd_delta_vs_no_failure": tvd_at_best_shots - source_nf_tvd,
            "random_delta_shots_vs_no_failure": random_shots - source_nf_shots,
            "best_tvd_delta_shots_vs_no_failure": (
                shots_at_best_tvd - source_nf_shots
            ),
            "best_shots_delta_vs_no_failure": best_shots - source_nf_shots,
            "candidate_tvd_gap_min": float(np.min(tvd_gaps)),
            "candidate_tvd_gap_mean": float(np.mean(tvd_gaps)),
            "candidate_tvd_gap_median": float(np.median(tvd_gaps)),
            "candidate_tvd_gap_max": float(np.max(tvd_gaps)),
            "candidate_tvd_gap_std": float(np.std(tvd_gaps, ddof=1)),
            "candidate_shot_gap_min": float(np.min(shot_gaps)),
            "candidate_shot_gap_mean": float(np.mean(shot_gaps)),
            "candidate_shot_gap_median": float(np.median(shot_gaps)),
            "candidate_shot_gap_max": float(np.max(shot_gaps)),
            "candidate_shot_gap_std": float(np.std(shot_gaps, ddof=1)),
        }

        for tau in thresholds:
            label = pct_label(tau)
            jnr = max(r_tvd / tau, r_shots_frac / tau)
            base[f"tvd_pass_{label}pct"] = bool(r_tvd <= tau)
            base[f"shots_pass_{label}pct"] = bool(r_shots_frac <= tau)
            base[f"joint_pass_{label}pct"] = bool(
                r_tvd <= tau and r_shots_frac <= tau
            )
            base[f"jnr_{label}pct"] = float(jnr)
            base[f"jnr_excess_{label}pct"] = float(max(0.0, jnr - 1.0))

        rows.append(base)

        for idx, (_, candidate) in enumerate(group.iterrows()):
            row = {
                "policy_id": policy_id,
                "stopping_rule": stopping_rule,
                "history_policy": history_policy,
                "circuit_key": circuit_key,
                "algorithm": algorithm,
                "size": int(size),
                "source_backend": source_backend,
                "failure_fraction": float(failure_fraction),
                "target_backend": str(candidate["target_backend"]),
                "candidate_tvd_to_aer": float(candidate["final_tvd_to_aer"]),
                "candidate_physical_shots": float(candidate["total_physical_shots"]),
                "best_tvd": best_tvd,
                "best_physical_shots": best_shots,
                "tvd_gap_from_best": float(tvd_gaps[idx]),
                "shot_gap_from_best": float(shot_gaps[idx]),
                "shot_gap_fraction_of_source_no_failure": float(
                    shot_gap_fracs[idx]
                ),
                "source_no_failure_physical_shots": source_nf_shots,
            }
            candidate_rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(candidate_rows)


def add_threshold_rates(
    summary: pd.DataFrame,
    events: pd.DataFrame,
    group_cols: Sequence[str],
    thresholds: Sequence[float],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for key, group in events.groupby(list(group_cols), sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row: Dict[str, object] = {
            col: value for col, value in zip(group_cols, key)
        }
        for tau in thresholds:
            label = pct_label(tau)
            row[f"p_tvd_pass_{label}pct"] = float(
                group[f"tvd_pass_{label}pct"].mean()
            )
            row[f"p_shots_pass_{label}pct"] = float(
                group[f"shots_pass_{label}pct"].mean()
            )
            row[f"p_joint_pass_{label}pct"] = float(
                group[f"joint_pass_{label}pct"].mean()
            )
            # Primary one-number magnitude-aware summary requested by user.
            row[f"jnr_{label}pct_mean"] = float(
                group[f"jnr_{label}pct"].mean()
            )
            row[f"jnr_{label}pct_median"] = float(
                group[f"jnr_{label}pct"].median()
            )
            row[f"jnr_excess_{label}pct_mean"] = float(
                group[f"jnr_excess_{label}pct"].mean()
            )
        rows.append(row)
    return summary.merge(
        pd.DataFrame(rows),
        on=list(group_cols),
        how="left",
        validate="one_to_one",
    )


def destination_shot_comparison(
    no_failure: pd.DataFrame,
    handoff: pd.DataFrame,
) -> pd.DataFrame:
    """Compare StableShots no-failure destination cost with handoff cost.

    One row per StableShots+keep handoff candidate. The matched reference is the
    no-failure StableShots execution of the destination backend for the same
    circuit/size.
    """
    dest_nf = no_failure[
        no_failure["stopping_rule"] == "stableshots"
    ][
        ["circuit_key", "backend", "total_physical_shots"]
    ].rename(
        columns={
            "backend": "target_backend",
            "total_physical_shots": "destination_no_failure_shots",
        }
    )
    work = handoff[handoff["policy_id"] == "stableshots_keep"].copy()
    work = work.merge(
        dest_nf,
        on=["circuit_key", "target_backend"],
        how="inner",
        validate="many_to_one",
    )
    work["source_pre_failure_shots"] = work["failure_shots"].astype(float)
    work["destination_post_failure_shots"] = work["post_failure_shots"].astype(float)
    work["delta_total_including_source_vs_dest_no_failure"] = (
        work["total_physical_shots"].astype(float)
        - work["destination_no_failure_shots"].astype(float)
    )
    work["delta_destination_only_vs_dest_no_failure"] = (
        work["destination_post_failure_shots"].astype(float)
        - work["destination_no_failure_shots"].astype(float)
    )
    return work[
        [
            "circuit_key",
            "algorithm",
            "size",
            "source_backend",
            "target_backend",
            "failure_fraction",
            "source_pre_failure_shots",
            "destination_post_failure_shots",
            "total_physical_shots",
            "destination_no_failure_shots",
            "delta_total_including_source_vs_dest_no_failure",
            "delta_destination_only_vs_dest_no_failure",
        ]
    ]


def save_variant(
    *,
    name: str,
    no_failure: pd.DataFrame,
    handoff: pd.DataFrame,
    thresholds: Sequence[float],
    analysis_dir: Path,
    plots_dir: Path,
    backends: Sequence[str],
    fractions: Sequence[float],
) -> Dict[str, object]:
    out_dir = analysis_dir / name
    out_plots = plots_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_plots.mkdir(parents=True, exist_ok=True)

    events, candidate_gaps = derive_selection_events(
        handoff,
        no_failure,
        thresholds,
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

    event_metrics = [
        "random_expected_tvd",
        "best_tvd",
        "r_tvd",
        "r_shots",
        "r_shots_fraction_of_source_no_failure",
        "random_expected_physical_shots",
        "best_physical_shots",
        "random_minus_best_tvd_shots",
        "random_minus_best_shots_tvd",
        "random_delta_tvd_vs_no_failure",
        "best_tvd_delta_vs_no_failure",
        "random_delta_shots_vs_no_failure",
        "best_shots_delta_vs_no_failure",
        "candidate_tvd_gap_mean",
        "candidate_tvd_gap_median",
        "candidate_tvd_gap_max",
        "candidate_tvd_gap_std",
        "candidate_shot_gap_mean",
        "candidate_shot_gap_median",
        "candidate_shot_gap_max",
        "candidate_shot_gap_std",
    ]
    for tau in thresholds:
        label = pct_label(tau)
        event_metrics += [
            f"jnr_{label}pct",
            f"jnr_excess_{label}pct",
        ]

    groupings = {
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

    no_failure_summary.to_csv(out_dir / "no_failure_qpu_summary.csv", index=False)
    source_failure_summary.to_csv(
        out_dir / "handoff_source_failure_summary.csv",
        index=False,
    )
    events.to_csv(out_dir / "selection_event_metrics.csv", index=False)
    candidate_gaps.to_csv(out_dir / "selection_candidate_gaps.csv", index=False)

    candidate_gap_summary = summary_table(
        candidate_gaps,
        ["policy_id", "failure_fraction"],
        [
            "tvd_gap_from_best",
            "shot_gap_from_best",
            "shot_gap_fraction_of_source_no_failure",
        ],
    )
    candidate_gap_summary.to_csv(
        out_dir / "candidate_gap_by_policy_failure.csv",
        index=False,
    )

    destination_compare = destination_shot_comparison(
        no_failure,
        handoff,
    )
    destination_compare.to_csv(
        out_dir / "proposal_destination_shot_comparison.csv",
        index=False,
    )

    for fname, cols in groupings.items():
        summary = summary_table(events, cols, event_metrics)
        summary = add_threshold_rates(summary, events, cols, thresholds)
        summary.to_csv(out_dir / f"{fname}.csv", index=False)

    # Proposal-only compact threshold table: one row per failure fraction.
    proposal = events[events["policy_id"] == "stableshots_keep"]
    proposal_rows: List[Dict[str, object]] = []
    for fraction, group in proposal.groupby("failure_fraction", sort=True):
        row: Dict[str, object] = {
            "failure_fraction": float(fraction),
            "events": int(len(group)),
            "r_tvd_mean": float(group["r_tvd"].mean()),
            "r_tvd_median": float(group["r_tvd"].median()),
            "r_shots_fraction_mean": float(
                group["r_shots_fraction_of_source_no_failure"].mean()
            ),
            "r_shots_fraction_median": float(
                group["r_shots_fraction_of_source_no_failure"].median()
            ),
        }
        for tau in thresholds:
            label = pct_label(tau)
            row[f"p_joint_pass_{label}pct"] = float(
                group[f"joint_pass_{label}pct"].mean()
            )
            row[f"jnr_{label}pct_mean"] = float(
                group[f"jnr_{label}pct"].mean()
            )
            row[f"jnr_{label}pct_median"] = float(
                group[f"jnr_{label}pct"].median()
            )
        proposal_rows.append(row)
    pd.DataFrame(proposal_rows).to_csv(
        out_dir / "proposal_threshold_summary.csv",
        index=False,
    )

    # ---- Plots ----
    # No-failure boxplots.
    for metric, ylabel, filename in [
        ("final_tvd_to_aer", "Final TVD to Aer", "no_failure_tvd_boxplots.png"),
        (
            "total_physical_shots",
            "Total physical shots",
            "no_failure_shots_boxplots.png",
        ),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
        for ax, rule in zip(axes, STOPPING_RULE_ORDER):
            arrays = [
                no_failure.loc[
                    (no_failure["stopping_rule"] == rule)
                    & (no_failure["backend"] == backend),
                    metric,
                ].to_numpy(dtype=float)
                for backend in backends
            ]
            ax.boxplot(
                arrays,
                tick_labels=[b.replace("fake_", "") for b in backends],
                showfliers=False,
            )
            ax.set_title(rule.replace("_", " "))
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=30)
        fig.suptitle(f"{name}: no-failure {ylabel}")
        fig.tight_layout()
        fig.savefig(out_plots / filename, dpi=180)
        plt.close(fig)

    # Proposal source/failure distributions.
    proposal_candidates = handoff[handoff["policy_id"] == "stableshots_keep"]
    for metric, ylabel, filename in [
        ("final_tvd_to_aer", "Final TVD to Aer", "proposal_source_failure_tvd.png"),
        (
            "total_physical_shots",
            "Total physical shots",
            "proposal_source_failure_shots.png",
        ),
    ]:
        fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.2))
        axes_flat = list(axes.flat)
        for ax, source in zip(axes_flat, backends):
            arrays = [
                proposal_candidates.loc[
                    (proposal_candidates["source_backend"] == source)
                    & np.isclose(
                        proposal_candidates["failure_fraction"],
                        fraction,
                    ),
                    metric,
                ].to_numpy(dtype=float)
                for fraction in fractions
            ]
            ax.boxplot(
                arrays,
                tick_labels=[f"{f:g}" for f in fractions],
                showfliers=False,
            )
            ax.set_title(source.replace("fake_", ""))
            ax.set_xlabel("Failure fraction")
            ax.set_ylabel(ylabel)
        axes_flat[-1].axis("off")
        fig.suptitle(f"{name}: StableShots keep")
        fig.tight_layout()
        fig.savefig(out_plots / filename, dpi=180)
        plt.close(fig)

    # Matched destination-shot differences for StableShots + keep.
    # Each panel is one destination QPU; boxes are failure fractions.
    for metric, ylabel, filename in [
        (
            "delta_total_including_source_vs_dest_no_failure",
            "Delta shots vs destination no-failure (source + destination)",
            "proposal_destination_shot_delta_source_counted.png",
        ),
        (
            "delta_destination_only_vs_dest_no_failure",
            "Delta shots vs destination no-failure (destination only)",
            "proposal_destination_shot_delta_destination_only.png",
        ),
    ]:
        fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.2))
        axes_flat = list(axes.flat)
        for ax, target in zip(axes_flat, backends):
            target_work = destination_compare[
                destination_compare["target_backend"] == target
            ]
            arrays = [
                target_work.loc[
                    np.isclose(target_work["failure_fraction"], fraction),
                    metric,
                ].to_numpy(dtype=float)
                for fraction in fractions
            ]
            ax.boxplot(
                arrays,
                tick_labels=[f"{f:g}" for f in fractions],
                showfliers=False,
            )
            ax.axhline(0.0, linewidth=0.9)
            ax.set_title(target.replace("fake_", ""))
            ax.set_xlabel("Failure fraction")
            ax.set_ylabel(ylabel)
        axes_flat[-1].axis("off")
        fig.suptitle(
            f"{name}: StableShots keep destination cost relative to no failure"
        )
        fig.tight_layout()
        fig.savefig(out_plots / filename, dpi=180)
        plt.close(fig)

    # Regret boxplots by policy.
    for metric, ylabel, filename in [
        ("r_tvd", "R_TVD", "selection_tvd_regret.png"),
        (
            "r_shots_fraction_of_source_no_failure",
            "R_shots / matched no-failure shots",
            "selection_shot_regret_fraction.png",
        ),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
        for ax, policy in zip(axes.flat, POLICY_ORDER):
            work = events[events["policy_id"] == policy]
            arrays = [
                work.loc[
                    np.isclose(work["failure_fraction"], fraction),
                    metric,
                ].to_numpy(dtype=float)
                for fraction in fractions
            ]
            ax.boxplot(
                arrays,
                tick_labels=[f"{f:g}" for f in fractions],
                showfliers=False,
            )
            for tau in thresholds:
                ax.axhline(tau, linestyle="--", linewidth=0.9)
            ax.set_title(policy)
            ax.set_xlabel("Failure fraction")
            ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(out_plots / filename, dpi=180)
        plt.close(fig)

    # JNR boxplots for proposal at 5% and 10%.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=False)
    for ax, tau in zip(axes, thresholds):
        label = pct_label(tau)
        arrays = [
            proposal.loc[
                np.isclose(proposal["failure_fraction"], fraction),
                f"jnr_{label}pct",
            ].to_numpy(dtype=float)
            for fraction in fractions
        ]
        ax.boxplot(
            arrays,
            tick_labels=[f"{f:g}" for f in fractions],
            showfliers=False,
        )
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_title(f"Joint normalized regret, {label}% threshold")
        ax.set_xlabel("Failure fraction")
        ax.set_ylabel("JNR")
    fig.suptitle(f"{name}: StableShots keep magnitude-aware practical regret")
    fig.tight_layout()
    fig.savefig(out_plots / "proposal_joint_normalized_regret.png", dpi=180)
    plt.close(fig)

    # Failure impact versus no failure for proposal.
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    for ax, left, right, ylabel in [
        (
            axes[0],
            "random_delta_tvd_vs_no_failure",
            "best_tvd_delta_vs_no_failure",
            "Delta TVD vs no failure",
        ),
        (
            axes[1],
            "random_delta_shots_vs_no_failure",
            "best_shots_delta_vs_no_failure",
            "Delta physical shots vs no failure",
        ),
    ]:
        positions: List[float] = []
        arrays: List[np.ndarray] = []
        ticks: List[float] = []
        for i, fraction in enumerate(fractions):
            center = i * 3.0
            arrays.extend(
                [
                    proposal.loc[
                        np.isclose(proposal["failure_fraction"], fraction),
                        left,
                    ].to_numpy(dtype=float),
                    proposal.loc[
                        np.isclose(proposal["failure_fraction"], fraction),
                        right,
                    ].to_numpy(dtype=float),
                ]
            )
            positions.extend([center - 0.45, center + 0.45])
            ticks.append(center)
        ax.boxplot(arrays, positions=positions, widths=0.7, showfliers=False)
        ax.axhline(0, linewidth=0.8)
        ax.set_xticks(ticks, [f"{f:g}" for f in fractions])
        ax.set_xlabel("Failure fraction")
        ax.set_ylabel(ylabel)
    fig.suptitle(f"{name}: StableShots keep failure impact")
    fig.tight_layout()
    fig.savefig(out_plots / "proposal_failure_impact.png", dpi=180)
    plt.close(fig)

    return {
        "variant": name,
        "no_failure_rows": int(len(no_failure)),
        "handoff_candidate_rows": int(len(handoff)),
        "selection_events": int(len(events)),
        "candidate_gap_rows": int(len(candidate_gaps)),
        "candidate_targets_min": int(events["candidate_targets"].min()),
        "candidate_targets_max": int(events["candidate_targets"].max()),
        "destination_shot_comparison_rows": int(len(destination_compare)),
        "algorithms": sorted(no_failure["algorithm"].unique().tolist()),
        "circuits": int(no_failure["circuit_key"].nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/handoff_policy_matrix_analysis_v4_config.json"),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())

    source_dir = Path(config["source_output_dir"])
    raw_dir = source_dir / "raw"
    analysis_dir = Path(config["analysis_output_dir"])
    plots_dir = Path(config["plots_output_dir"])
    analysis_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [float(v) for v in config["threshold_levels"]]
    excluded = set(str(v) for v in config["exclude_algorithms"])

    no_failure_raw = pd.read_csv(raw_dir / "no_failure_policy_runs.csv")
    handoff_raw = pd.read_csv(raw_dir / "handoff_policy_runs.csv")

    no_failure = no_failure_raw[
        (~no_failure_raw["algorithm"].isin(excluded))
        & (no_failure_raw["stopping_rule"] != "fixed_10k")
    ].copy()
    handoff = handoff_raw[
        (~handoff_raw["algorithm"].isin(excluded))
        & (handoff_raw["stopping_rule"] != "fixed_10k")
    ].copy()

    backends = sorted(no_failure["backend"].unique().tolist())
    fractions = sorted(handoff["failure_fraction"].unique().astype(float).tolist())

    full_manifest = save_variant(
        name="full",
        no_failure=no_failure,
        handoff=handoff,
        thresholds=thresholds,
        analysis_dir=analysis_dir,
        plots_dir=plots_dir,
        backends=backends,
        fractions=fractions,
    )

    manifest = {
        "excluded_algorithms": sorted(excluded),
        "excluded_stopping_rules": ["fixed_10k"],
        "threshold_levels": thresholds,
        "policies": POLICY_ORDER,
        "jnr_definition": (
            "max(R_TVD/tau, (R_shots/source_no_failure_shots)/tau)"
        ),
        "jnr_interpretation": (
            "JNR <= 1 passes both practical thresholds; magnitude above 1 "
            "measures how many threshold units the worse objective incurs"
        ),
        "full": full_manifest,
    }
    (analysis_dir / "analysis_v4_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
