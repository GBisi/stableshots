#!/usr/bin/env python3
"""Mandatory-failure regret analysis with one primary regret per objective.

Primary metrics:
- R_TVD = E_random[TVD_to_Aer] - min_target(TVD_to_Aer)
- R_shots = E_random[physical_shots] - min_target(physical_shots)

Cross-metrics are signed:
- random_minus_tvd_oracle_shots: shot difference when comparing random choice
  with the TVD-optimal target.
- random_minus_shot_oracle_tvd: Aer-TVD difference when comparing random choice
  with the shot-optimal target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

EVENT = ["circuit_key", "algorithm", "size", "source_backend", "failure_fraction"]
TVD_THRESHOLD = 0.05
SHOT_THRESHOLD = 1000.0


def derive_events(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for key, group in candidates.groupby(EVENT + ["policy"], sort=True, dropna=False):
        row: Dict[str, object] = {
            column: value for column, value in zip(EVENT + ["policy"], key)
        }
        tvd = group["final_aggregated_tvd_to_aer"].to_numpy(dtype=float)
        shots = group["total_physical_shots"].to_numpy(dtype=float)
        random_tvd = float(np.mean(tvd))
        random_shots = float(np.mean(shots))
        best_tvd = float(np.min(tvd))
        best_shots = float(np.min(shots))

        best_tvd_group = group[np.isclose(group["final_aggregated_tvd_to_aer"], best_tvd)]
        best_shot_group = group[np.isclose(group["total_physical_shots"], best_shots)]
        shots_at_best_tvd = float(best_tvd_group["total_physical_shots"].mean())
        tvd_at_best_shots = float(best_shot_group["final_aggregated_tvd_to_aer"].mean())

        r_tvd = random_tvd - best_tvd
        r_shots = random_shots - best_shots

        row.update(
            {
                "candidate_targets": int(len(group)),
                # TVD-optimal comparison.
                "random_tvd_to_aer": random_tvd,
                "best_tvd_to_aer": best_tvd,
                "r_tvd": r_tvd,
                "random_shots_for_tvd_comparison": random_shots,
                "shots_at_best_tvd": shots_at_best_tvd,
                "random_minus_tvd_oracle_shots": random_shots - shots_at_best_tvd,

                # Shot-optimal comparison.
                "random_physical_shots": random_shots,
                "best_physical_shots": best_shots,
                "r_shots": r_shots,
                "random_tvd_for_shot_comparison": random_tvd,
                "tvd_at_best_shots": tvd_at_best_shots,
                "random_minus_shot_oracle_tvd": random_tvd - tvd_at_best_shots,

                # Practical thresholds.
                "r_tvd_gt_0p05": r_tvd > TVD_THRESHOLD,
                "r_shots_gt_1000": r_shots > SHOT_THRESHOLD,
                "random_joint_practically_adequate": (
                    r_tvd <= TVD_THRESHOLD and r_shots <= SHOT_THRESHOLD
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(events: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    metrics = [
        "r_tvd",
        "r_shots",
        "random_tvd_to_aer",
        "best_tvd_to_aer",
        "random_physical_shots",
        "best_physical_shots",
        "random_minus_tvd_oracle_shots",
        "tvd_at_best_shots",
        "random_minus_shot_oracle_tvd",
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
            row[f"min_{metric}"] = float(values.min())
            row[f"q25_{metric}"] = float(values.quantile(0.25))
            row[f"median_{metric}"] = float(values.median())
            row[f"q75_{metric}"] = float(values.quantile(0.75))
            row[f"max_{metric}"] = float(values.max())
            row[f"mean_{metric}"] = float(values.mean())
            row[f"std_{metric}"] = float(values.std(ddof=1))
        row["p_r_tvd_gt_0p05"] = float(group["r_tvd_gt_0p05"].mean())
        row["p_r_shots_gt_1000"] = float(group["r_shots_gt_1000"].mean())
        row["p_random_joint_practically_adequate"] = float(
            group["random_joint_practically_adequate"].mean()
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
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(
        analysis_dir / "mandatory_failure_policy_candidates.csv"
    )
    events = derive_events(candidates)
    events.to_csv(
        analysis_dir / "mandatory_failure_regret_events.csv",
        index=False,
    )

    groupings = {
        "policy_failure": ["policy", "failure_fraction"],
        "policy_source": ["policy", "source_backend"],
        "policy_algorithm": ["policy", "algorithm"],
        "policy_size": ["policy", "size"],
        "policy_circuit": ["policy", "circuit_key"],
        "policy_overall": ["policy"],
    }
    for name, columns in groupings.items():
        summarize(events, columns).to_csv(
            analysis_dir / f"mandatory_failure_regret_{name}_summary.csv",
            index=False,
        )

    # Cap-free sensitivity: only StableShots max-budget terminations are
    # considered capped. Fixed-20k policies intentionally use 20k and remain.
    uncapped = candidates[~candidates["hit_stableshots_cap"].astype(bool)].copy()
    counts = uncapped.groupby(EVENT + ["policy"]).size()
    valid = counts[counts >= 2].reset_index()[EVENT + ["policy"]]
    uncapped = uncapped.merge(valid, on=EVENT + ["policy"], how="inner")
    cap_free_events = derive_events(uncapped)
    cap_free_events.to_csv(
        analysis_dir / "mandatory_failure_regret_events_cap_free.csv",
        index=False,
    )
    for name, columns in groupings.items():
        summarize(cap_free_events, columns).to_csv(
            analysis_dir / f"mandatory_failure_regret_{name}_summary_cap_free.csv",
            index=False,
        )

    manifest = {
        "events": int(len(events)),
        "policies": sorted(events["policy"].unique().tolist()),
        "r_tvd_definition": "random expected final TVD to Aer minus best-target final TVD to Aer",
        "r_shots_definition": "random expected physical shots minus minimum-target physical shots",
        "tvd_threshold": TVD_THRESHOLD,
        "shot_threshold_absolute": SHOT_THRESHOLD,
        "cap_free_events": int(len(cap_free_events)),
    }
    (analysis_dir / "mandatory_failure_regret_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
