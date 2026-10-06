#!/usr/bin/env python3
"""Simplified Q-SE 2027 analysis for stateful QPU handoffs.

RQ1 describes StableShots+keep trajectories by ordered source/destination pair
and keeps retain-vs-restart/fixed baselines as a corollary.

RQ2 defines retrospective quality bounds over the four realised candidate
handoffs for each circuit/source/failure event:
  BEST  = minimum realised final TVD to the 200k Aer reference;
  WORST = maximum realised final TVD to the 200k Aer reference.

BEST/WORST are descriptive oracles, not deployable selectors.  A separate
diagnostic reports whether the best standalone Fixed20k profile belongs to the
retrospective BEST set.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def clean_backend(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"^fake_", "", regex=True)


def describe(s: pd.Series) -> dict[str, float]:
    s = s.astype(float)
    return {
        "min": float(s.min()),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "max": float(s.max()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("results/qpu_handoff_policy_matrix/raw"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qpu_handoff_policy_matrix/analysis_v7_qse2027"),
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    handoff = pd.read_csv(args.raw_dir / "handoff_policy_runs.csv")
    no_failure = pd.read_csv(args.raw_dir / "no_failure_policy_runs.csv")

    keep = handoff[handoff.policy_id == "stableshots_keep"].copy()
    keep["source"] = clean_backend(keep.source_backend)
    keep["destination"] = clean_backend(keep.target_backend)
    keep["failure"] = keep.failure_fraction.astype(float)

    pair = (
        keep.groupby(["source", "destination", "failure"], as_index=False)
        .agg(
            n=("final_tvd_to_aer", "size"),
            tvd_median=("final_tvd_to_aer", "median"),
            tvd_mean=("final_tvd_to_aer", "mean"),
            shots_median=("total_physical_shots", "median"),
            shots_mean=("total_physical_shots", "mean"),
        )
        .sort_values(["source", "destination", "failure"])
    )
    pair.to_csv(args.output_dir / "pair_medians.csv", index=False)

    nf_ss = no_failure[no_failure.stopping_rule == "stableshots"].copy()
    nf_ss["source"] = clean_backend(nf_ss.backend)
    (
        nf_ss.groupby("source", as_index=False)
        .agg(
            n=("final_tvd_to_aer", "size"),
            tvd_median=("final_tvd_to_aer", "median"),
            tvd_mean=("final_tvd_to_aer", "mean"),
            shots_median=("total_physical_shots", "median"),
            shots_mean=("total_physical_shots", "mean"),
        )
        .sort_values("source")
        .to_csv(args.output_dir / "source_baseline.csv", index=False)
    )

    rq1 = []
    for f, g in keep.groupby("failure", sort=True):
        tvd, shots = describe(g.final_tvd_to_aer), describe(g.total_physical_shots)
        rq1.append({
            "failure": f,
            "n": len(g),
            **{f"tvd_{k}": v for k, v in tvd.items()},
            **{f"shots_{k}": v for k, v in shots.items()},
        })
    pd.DataFrame(rq1).to_csv(args.output_dir / "rq1_global.csv", index=False)

    policies = [
        "stableshots_keep",
        "stableshots_discard",
        "fixed_20k_keep",
        "fixed_20k_discard",
    ]
    rows = []
    for (f, policy), g in handoff[handoff.policy_id.isin(policies)].groupby(
        ["failure_fraction", "policy_id"], sort=True
    ):
        rows.append({
            "failure": float(f),
            "policy": policy,
            "n": len(g),
            "tvd_median": float(g.final_tvd_to_aer.median()),
            "tvd_mean": float(g.final_tvd_to_aer.mean()),
            "shots_median": float(g.total_physical_shots.median()),
            "shots_mean": float(g.total_physical_shots.mean()),
            "stable_rate": float((g.stop_reason == "stable").mean()),
        })
    pd.DataFrame(rows).to_csv(args.output_dir / "policy_by_failure.csv", index=False)

    nf_rows = []
    for policy, g in no_failure[
        no_failure.stopping_rule.isin(["stableshots", "fixed_20k"])
    ].groupby("stopping_rule", sort=True):
        nf_rows.append({
            "policy": policy,
            "n": len(g),
            "tvd_median": float(g.final_tvd_to_aer.median()),
            "tvd_mean": float(g.final_tvd_to_aer.mean()),
            "shots_median": float(g.total_physical_shots.median()),
            "shots_mean": float(g.total_physical_shots.mean()),
            "stable_rate": float((g.stop_reason == "stable").mean()),
        })
    pd.DataFrame(nf_rows).to_csv(args.output_dir / "no_failure_global.csv", index=False)

    fixed20 = no_failure[no_failure.stopping_rule == "fixed_20k"]
    profile = fixed20.set_index(["circuit_key", "backend"])["final_tvd_to_aer"]

    event_rows = []
    group_cols = [
        "circuit_key", "algorithm", "size", "source_backend", "failure_fraction"
    ]
    for key, g in keep.groupby(group_cols, sort=True):
        if len(g) != 4:
            raise RuntimeError(f"expected four destinations for {key}, got {len(g)}")
        circuit, algorithm, size, source, failure = key
        min_tvd = float(g.final_tvd_to_aer.min())
        max_tvd = float(g.final_tvd_to_aer.max())
        best_set = g[np.isclose(g.final_tvd_to_aer, min_tvd)]
        worst_set = g[np.isclose(g.final_tvd_to_aer, max_tvd)]
        best = best_set.sort_values(
            ["total_physical_shots", "target_backend"]
        ).iloc[0]
        worst = worst_set.sort_values(
            ["total_physical_shots", "target_backend"],
            ascending=[False, True],
        ).iloc[0]

        candidates = g.target_backend.astype(str).tolist()
        prof = {q: float(profile[(circuit, q)]) for q in candidates}
        min_profile = min(prof.values())
        offline_best = sorted(
            q for q, value in prof.items() if np.isclose(value, min_profile)
        )[0]
        agreement = offline_best in set(best_set.target_backend.astype(str))

        event_rows.append({
            "circuit_key": circuit,
            "algorithm": algorithm,
            "size": int(size),
            "source": str(source).replace("fake_", ""),
            "failure": float(failure),
            "best_target": str(best.target_backend).replace("fake_", ""),
            "worst_target": str(worst.target_backend).replace("fake_", ""),
            "offline20k_target": offline_best.replace("fake_", ""),
            "best_tvd": float(best.final_tvd_to_aer),
            "worst_tvd": float(worst.final_tvd_to_aer),
            "best_shots": float(best.total_physical_shots),
            "worst_shots": float(worst.total_physical_shots),
            "offline20k_matches_best": bool(agreement),
        })

    events = pd.DataFrame(event_rows)
    if len(events) != 900:
        raise RuntimeError(f"expected 900 RQ2 events, got {len(events)}")
    events.to_csv(args.output_dir / "rq2_oracle_events.csv", index=False)

    rows = []
    for (source, failure), g in events.groupby(["source", "failure"], sort=True):
        for strategy, prefix in [("BEST", "best"), ("WORST", "worst")]:
            rows.append({
                "source": source,
                "failure": failure,
                "strategy": strategy,
                "n": len(g),
                "tvd_median": float(g[f"{prefix}_tvd"].median()),
                "tvd_mean": float(g[f"{prefix}_tvd"].mean()),
                "shots_median": float(g[f"{prefix}_shots"].median()),
                "shots_mean": float(g[f"{prefix}_shots"].mean()),
            })
    pd.DataFrame(rows).to_csv(
        args.output_dir / "rq2_source_failure.csv", index=False
    )

    global_rows, gap_rows, agreement_rows = [], [], []
    for failure, g in events.groupby("failure", sort=True):
        for strategy, prefix in [("BEST", "best"), ("WORST", "worst")]:
            tvd, shots = describe(g[f"{prefix}_tvd"]), describe(g[f"{prefix}_shots"])
            global_rows.append({
                "failure": failure,
                "strategy": strategy,
                "n": len(g),
                **{f"tvd_{k}": v for k, v in tvd.items()},
                **{f"shots_{k}": v for k, v in shots.items()},
            })

        tvd_gap = describe(g.worst_tvd - g.best_tvd)
        shot_gap = describe(g.worst_shots - g.best_shots)
        matches = int(g.offline20k_matches_best.sum())
        gap_rows.append({
            "failure": failure,
            "n": len(g),
            **{f"tvd_gap_{k}": v for k, v in tvd_gap.items()},
            **{f"shots_gap_{k}": v for k, v in shot_gap.items()},
            "offline20k_match_count": matches,
            "offline20k_match_rate": matches / len(g),
        })
        agreement_rows.append({
            "failure": failure,
            "match_count": matches,
            "total": len(g),
            "match_rate": matches / len(g),
        })

    pd.DataFrame(global_rows).to_csv(args.output_dir / "rq2_global.csv", index=False)
    pd.DataFrame(gap_rows).to_csv(
        args.output_dir / "rq2_gaps_agreement.csv", index=False
    )
    pd.DataFrame(agreement_rows).to_csv(
        args.output_dir / "offline20k_agreement.csv", index=False
    )

    matches = int(events.offline20k_matches_best.sum())
    print(f"offline-20k agrees with retrospective BEST in "
          f"{matches}/{len(events)} events ({matches/len(events):.1%})")


if __name__ == "__main__":
    main()
