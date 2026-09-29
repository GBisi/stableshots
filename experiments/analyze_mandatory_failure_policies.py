#!/usr/bin/env python3
"""Mandatory-failure policy comparison including simple fixed-shot baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

EVENT = ["circuit_key", "algorithm", "size", "source_backend", "failure_fraction"]
TVD_THRESHOLD = 0.05
SHOT_THRESHOLD = 0.05


def existing_stableshots_candidates(handoff: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "circuit_key": handoff["circuit_key"],
            "algorithm": handoff["algorithm"],
            "size": handoff["size"],
            "source_backend": handoff["source_backend"],
            "target_backend": handoff["target_backend"],
            "failure_fraction": handoff["failure_fraction"],
            "failure_shots": handoff["failure_shots"],
            "source_no_failure_shots": handoff["source_no_failure_shots"],
            "source_no_failure_final_tvd_to_aer": handoff[
                "source_no_failure_final_tvd_to_aer"
            ],
            "policy": "stableshots_continuation",
            "estimator_shots": handoff["total_shots"],
            "post_failure_shots": handoff["post_failure_shots"],
            "discarded_pre_failure_shots": 0,
            "total_physical_shots": handoff["total_shots"],
            "final_aggregated_tvd_to_aer": handoff[
                "final_aggregated_tvd_to_aer"
            ],
            "delta_tvd_vs_source_no_failure": handoff[
                "delta_tvd_vs_source_no_failure"
            ],
            "shot_delta_vs_source_no_failure": handoff[
                "shot_delta_vs_source_no_failure"
            ],
            "stop_reason": handoff["stop_reason"],
            "hit_stableshots_cap": handoff["stop_reason"].eq("max_budget"),
        }
    )
    return out


def derive_events(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    group_cols = EVENT + ["policy"]
    for key, group in candidates.groupby(group_cols, sort=True, dropna=False):
        row = {column: value for column, value in zip(group_cols, key)}
        tvd = group["final_aggregated_tvd_to_aer"].to_numpy(dtype=float)
        shots = group["total_physical_shots"].to_numpy(dtype=float)
        best_tvd = float(np.min(tvd))
        best_shots = float(np.min(shots))
        random_tvd = float(np.mean(tvd))
        random_shots = float(np.mean(shots))
        row.update(
            {
                "candidate_targets": int(len(group)),
                "random_expected_tvd": random_tvd,
                "best_tvd": best_tvd,
                "random_tvd_regret": random_tvd - best_tvd,
                "random_expected_shots": random_shots,
                "best_shots": best_shots,
                "random_shot_regret": random_shots - best_shots,
                "random_shot_regret_pct": (
                    (random_shots - best_shots) / best_shots
                    if best_shots > 0
                    else float("nan")
                ),
                "random_within_5pct_tvd": (random_tvd - best_tvd)
                <= TVD_THRESHOLD,
                "random_within_5pct_shots": (
                    (random_shots - best_shots) / best_shots
                    if best_shots > 0
                    else float("inf")
                )
                <= SHOT_THRESHOLD,
            }
        )
        row["random_practically_adequate"] = bool(
            row["random_within_5pct_tvd"] and row["random_within_5pct_shots"]
        )

        best_tvd_group = group[
            np.isclose(group["final_aggregated_tvd_to_aer"], best_tvd)
        ]
        best_shot_group = group[np.isclose(group["total_physical_shots"], best_shots)]
        row["shots_at_best_tvd"] = float(
            best_tvd_group["total_physical_shots"].mean()
        )
        row["best_tvd_shot_overhead_pct_vs_shot_oracle"] = (
            (float(row["shots_at_best_tvd"]) - best_shots) / best_shots
            if best_shots > 0
            else float("nan")
        )
        row["tvd_at_best_shots"] = float(
            best_shot_group["final_aggregated_tvd_to_aer"].mean()
        )
        row["shot_oracle_tvd_penalty_vs_tvd_oracle"] = (
            float(row["tvd_at_best_shots"]) - best_tvd
        )
        best_tvd_targets = set(best_tvd_group["target_backend"].astype(str))
        best_shot_targets = set(best_shot_group["target_backend"].astype(str))
        row["oracle_target_overlap"] = bool(best_tvd_targets & best_shot_targets)

        joint = group[
            (group["final_aggregated_tvd_to_aer"] <= best_tvd + TVD_THRESHOLD)
            & (group["total_physical_shots"] <= best_shots * (1.0 + SHOT_THRESHOLD))
        ]
        row["fraction_targets_joint_near_oracle"] = float(len(joint) / len(group))
        row["has_joint_near_oracle_target"] = bool(len(joint) > 0)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(events: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    metrics = [
        "random_tvd_regret",
        "random_shot_regret",
        "random_shot_regret_pct",
        "best_tvd_shot_overhead_pct_vs_shot_oracle",
        "shot_oracle_tvd_penalty_vs_tvd_oracle",
        "fraction_targets_joint_near_oracle",
    ]
    rows: List[Dict[str, object]] = []
    for key, group in events.groupby(list(group_cols), sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row: Dict[str, object] = {
            column: value for column, value in zip(group_cols, key)
        }
        row["events"] = int(len(group))
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"median_{metric}"] = float(values.median())
            row[f"mean_{metric}"] = float(values.mean())
            row[f"std_{metric}"] = float(values.std(ddof=1))
            row[f"min_{metric}"] = float(values.min())
            row[f"max_{metric}"] = float(values.max())
        row["p_random_tvd_regret_gt_0p05"] = float(
            (group["random_tvd_regret"] > TVD_THRESHOLD).mean()
        )
        row["p_random_shot_regret_gt_5pct"] = float(
            (group["random_shot_regret_pct"] > SHOT_THRESHOLD).mean()
        )
        row["p_random_practically_adequate"] = float(
            group["random_practically_adequate"].mean()
        )
        row["p_oracle_target_overlap"] = float(group["oracle_target_overlap"].mean())
        row["p_has_joint_near_oracle_target"] = float(
            group["has_joint_near_oracle_target"].mean()
        )
        rows.append(row)
    return pd.DataFrame(rows)


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
    analysis_dir.mkdir(parents=True, exist_ok=True)

    handoff = pd.read_csv(raw_dir / "handoff_runs.csv")
    extra = pd.read_csv(raw_dir / "mandatory_failure_baselines.csv")
    stable = existing_stableshots_candidates(handoff)
    candidates = pd.concat([stable, extra], ignore_index=True, sort=False)

    expected_policies = {
        "stableshots_continuation",
        "fixed_20k_continuation",
        "restart_stableshots",
        "restart_fixed_20k",
    }
    if set(candidates["policy"]) != expected_policies:
        raise RuntimeError("mandatory-failure policy set is incomplete")

    candidates.to_csv(
        analysis_dir / "mandatory_failure_policy_candidates.csv", index=False
    )
    events = derive_events(candidates)
    events.to_csv(
        analysis_dir / "mandatory_failure_policy_events.csv", index=False
    )

    summaries = {
        "policy_failure": ["policy", "failure_fraction"],
        "policy_source": ["policy", "source_backend"],
        "policy_algorithm": ["policy", "algorithm"],
        "policy_size": ["policy", "size"],
    }
    for name, columns in summaries.items():
        summarize(events, columns).to_csv(
            analysis_dir / f"mandatory_failure_{name}_summary.csv",
            index=False,
        )

    # Cap-free sensitivity: exclude only candidates where a StableShots
    # controller terminated at max_budget. Fixed-20k baselines intentionally
    # use 20k and are therefore retained.
    uncapped_candidates = candidates[~candidates["hit_stableshots_cap"].astype(bool)]
    counts = uncapped_candidates.groupby(EVENT + ["policy"]).size()
    valid = counts[counts >= 2].reset_index()[EVENT + ["policy"]]
    cap_free = uncapped_candidates.merge(
        valid,
        on=EVENT + ["policy"],
        how="inner",
    )
    cap_free_events = derive_events(cap_free)
    cap_free_events.to_csv(
        analysis_dir / "mandatory_failure_policy_events_cap_free.csv", index=False
    )
    for name, columns in summaries.items():
        summarize(cap_free_events, columns).to_csv(
            analysis_dir / f"mandatory_failure_{name}_summary_cap_free.csv",
            index=False,
        )

    manifest = {
        "candidate_rows": int(len(candidates)),
        "events": int(len(events)),
        "policies": sorted(expected_policies),
        "tvd_practical_threshold_absolute": TVD_THRESHOLD,
        "shot_practical_threshold_relative": SHOT_THRESHOLD,
        "cap_free_definition": (
            "exclude only max_budget StableShots candidates; fixed-20k "
            "baselines are retained because 20k is intentional"
        ),
        "cap_free_candidate_rows": int(len(cap_free)),
        "cap_free_events": int(len(cap_free_events)),
    }
    (analysis_dir / "mandatory_failure_policy_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
