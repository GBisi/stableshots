#!/usr/bin/env python3
"""Oracle replacement-QPU strategy analysis for the Q-SE 2027 paper.

Selection strategies are defined from offline QPU quality profiles:
- oracle_best: lowest 30k-shot TVD to Aer among available destinations;
- oracle_nonworse: uniform expectation over destinations whose 20k-shot TVD
  to Aer is <= the failed source's 20k-shot TVD. If none exists, fall back to
  the destination with the lowest 20k-shot TVD and record the fallback;
- random: uniform expectation over all available destinations;
- oracle_worst: highest 30k-shot TVD to Aer among available destinations.

The 20k/30k profiling cost is offline oracle information and is excluded from
runtime handoff shot cost. Handoff outcomes are the already-recorded
StableShots+keep candidate runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from run_handoff_baseline_experiment import StableShotsConfig, materialize_circuit, tvd_weighted
from run_handoff_policy_matrix import take_exact_shots


STRATEGIES = ["oracle_best", "oracle_nonworse", "random", "oracle_worst"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/handoff_policy_matrix_config.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qpu_handoff_policy_matrix/analysis_v6_oracle"),
    )
    args = parser.parse_args()
    config: Dict[str, object] = json.loads(args.config.read_text())

    raw_dir = Path(str(config["output_dir"])) / "raw"
    handoff = pd.read_csv(raw_dir / "handoff_policy_runs.csv")
    handoff = handoff[
        (handoff["policy_id"] == "stableshots_keep")
        & (handoff["algorithm"] != "random")
    ].copy()

    stable_raw = dict(config["stableshots"])
    profile_cfg = StableShotsConfig(
        batch_size=int(stable_raw["batch_size"]),
        lookback_batches=int(stable_raw["lookback_batches"]),
        stability=int(stable_raw["stability"]),
        epsilon=float(stable_raw["epsilon"]),
        max_shots=30000,
    )

    profile_rows: List[Dict[str, object]] = []
    for algorithm in [str(x) for x in config["algorithms"] if str(x) != "random"]:
        for size in [int(x) for x in config["sizes"]]:
            print(f"profiling {algorithm}/{size}", flush=True)
            ideal, streams = materialize_circuit(
                algorithm,
                size,
                config,
                profile_cfg,
            )
            for backend, batches in streams.items():
                c20 = take_exact_shots(batches, 20000)
                c30 = take_exact_shots(batches, 30000)
                profile_rows.append(
                    {
                        "circuit_key": f"{algorithm}_{size}",
                        "algorithm": algorithm,
                        "size": size,
                        "backend": backend,
                        "tvd_20k_to_aer": float(tvd_weighted(c20, ideal)),
                        "tvd_30k_to_aer": float(tvd_weighted(c30, ideal)),
                    }
                )

    profiles = pd.DataFrame(profile_rows)
    if len(profiles) != 5 * 6 * 5:
        raise RuntimeError(f"expected 150 profile rows, got {len(profiles)}")

    lookup20 = profiles.set_index(["circuit_key", "backend"])["tvd_20k_to_aer"]
    lookup30 = profiles.set_index(["circuit_key", "backend"])["tvd_30k_to_aer"]

    event_rows: List[Dict[str, object]] = []
    group_cols = [
        "circuit_key", "algorithm", "size", "source_backend", "failure_fraction"
    ]
    for key, group in handoff.groupby(group_cols, sort=True):
        if len(group) != 4:
            raise RuntimeError(f"expected 4 targets for {key}, got {len(group)}")
        circuit_key, algorithm, size, source, failure_fraction = key
        candidates = group.copy()
        candidates["profile_20k"] = [
            float(lookup20[(circuit_key, b)]) for b in candidates["target_backend"]
        ]
        candidates["profile_30k"] = [
            float(lookup30[(circuit_key, b)]) for b in candidates["target_backend"]
        ]
        source20 = float(lookup20[(circuit_key, source)])

        min30 = float(candidates["profile_30k"].min())
        max30 = float(candidates["profile_30k"].max())
        best30 = candidates[np.isclose(candidates["profile_30k"], min30)]
        worst30 = candidates[np.isclose(candidates["profile_30k"], max30)]

        nonworse = candidates[candidates["profile_20k"] <= source20 + 1e-15]
        fallback = False
        if len(nonworse) == 0:
            fallback = True
            min20 = float(candidates["profile_20k"].min())
            nonworse = candidates[np.isclose(candidates["profile_20k"], min20)]

        strategy_groups = {
            "oracle_best": best30,
            "oracle_nonworse": nonworse,
            "random": candidates,
            "oracle_worst": worst30,
        }

        # Runtime oracle-best handoff outcome is the reference for strategy regret.
        best_runtime_tvd = float(best30["final_tvd_to_aer"].mean())
        best_runtime_shots = float(best30["total_physical_shots"].mean())

        for strategy, chosen in strategy_groups.items():
            final_tvd = float(chosen["final_tvd_to_aer"].mean())
            shots = float(chosen["total_physical_shots"].mean())
            event_rows.append(
                {
                    "circuit_key": circuit_key,
                    "algorithm": algorithm,
                    "size": int(size),
                    "source_backend": source,
                    "failure_fraction": float(failure_fraction),
                    "strategy": strategy,
                    "chosen_targets": ";".join(sorted(chosen["target_backend"].astype(str))),
                    "chosen_count": int(len(chosen)),
                    "nonworse_fallback": bool(fallback) if strategy == "oracle_nonworse" else False,
                    "source_profile_20k_tvd": source20,
                    "selected_profile_20k_tvd": float(chosen["profile_20k"].mean()),
                    "selected_profile_30k_tvd": float(chosen["profile_30k"].mean()),
                    "handoff_final_tvd_to_aer": final_tvd,
                    "handoff_total_physical_shots": shots,
                    "delta_tvd_vs_oracle_best": final_tvd - best_runtime_tvd,
                    "delta_shots_vs_oracle_best": shots - best_runtime_shots,
                }
            )

    events = pd.DataFrame(event_rows)
    expected = 5 * 6 * 5 * 5 * 4
    if len(events) != expected:
        raise RuntimeError(f"expected {expected} strategy-event rows, got {len(events)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(args.output_dir / "qpu_quality_profiles_20k_30k.csv", index=False)
    events.to_csv(args.output_dir / "oracle_strategy_events.csv", index=False)

    rows: List[Dict[str, object]] = []
    for (strategy, failure_fraction), group in events.groupby(
        ["strategy", "failure_fraction"], sort=True
    ):
        row: Dict[str, object] = {
            "strategy": strategy,
            "failure_fraction": float(failure_fraction),
            "events": int(len(group)),
            "nonworse_fallback_rate": float(group["nonworse_fallback"].mean()),
        }
        for metric in [
            "handoff_final_tvd_to_aer",
            "handoff_total_physical_shots",
            "delta_tvd_vs_oracle_best",
            "delta_shots_vs_oracle_best",
        ]:
            s = group[metric].astype(float)
            row[f"{metric}_min"] = float(s.min())
            row[f"{metric}_q25"] = float(s.quantile(0.25))
            row[f"{metric}_median"] = float(s.median())
            row[f"{metric}_mean"] = float(s.mean())
            row[f"{metric}_q75"] = float(s.quantile(0.75))
            row[f"{metric}_max"] = float(s.max())
            row[f"{metric}_std"] = float(s.std(ddof=1))
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        args.output_dir / "oracle_strategy_summary_by_failure.csv",
        index=False,
    )

    manifest = {
        "profile_rows": int(len(profiles)),
        "strategy_event_rows": int(len(events)),
        "strategies": STRATEGIES,
        "best_profile": "lowest 30k-shot TVD to Aer",
        "nonworse_profile": "20k-shot TVD to Aer <= failed source 20k-shot TVD",
        "nonworse_fallback": "lowest 20k-shot TVD destination if eligible set is empty",
        "worst_profile": "highest 30k-shot TVD to Aer",
        "profile_cost_accounting": "offline oracle profiling excluded from handoff cost",
    }
    (args.output_dir / "oracle_strategy_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
