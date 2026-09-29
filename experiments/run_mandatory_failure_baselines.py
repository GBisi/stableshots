#!/usr/bin/env python3
"""Generate mandatory-failure comparison baselines.

This replays the same deterministic backend streams used by the clean handoff
experiment and adds three mandatory-failure baselines:

1. fixed_20k_continuation:
   keep the source prefix, hand off, and keep accumulating until the combined
   estimator contains exactly 20,000 shots. No StableShots stopping is used
   after the failure.

2. restart_stableshots:
   discard all pre-failure evidence and restart StableShots from scratch on the
   replacement backend. Physical cost includes the discarded source prefix.

3. restart_fixed_20k:
   discard all pre-failure evidence and collect exactly 20,000 fresh shots on
   the replacement backend. Physical cost is failure_shots + 20,000.

The existing naive StableShots continuation remains in handoff_runs.csv and is
joined with these baselines by the analysis script.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from run_handoff_baseline_experiment import (
    Batch,
    Counts,
    StableShotsConfig,
    load_config,
    materialize_circuit,
    merge_counts,
    replay_source_prefix,
    rounded_failure_shots,
    run_no_failure_baselines,
    tvd_weighted,
)


def take_exact_shots(batches: Sequence[Batch], requested_shots: int) -> Counts:
    counts: Counts = Counter()
    used = 0
    for batch_shots, batch_counts in batches:
        if used + batch_shots > requested_shots:
            raise ValueError(
                f"requested_shots={requested_shots} is not aligned with batch size"
            )
        merge_counts(counts, batch_counts)
        used += int(batch_shots)
        if used == requested_shots:
            break
    if used != requested_shots:
        raise RuntimeError(
            f"stream ended after {used} shots; requested {requested_shots}"
        )
    return counts


def row_common(
    *,
    algorithm: str,
    size: int,
    source_backend: str,
    target_backend: str,
    failure_fraction: float,
    failure_shots: int,
    source_baseline: Mapping[str, object],
    target_baseline: Mapping[str, object],
) -> Dict[str, object]:
    return {
        "circuit_key": f"{algorithm}_{size}",
        "algorithm": algorithm,
        "size": int(size),
        "source_backend": source_backend,
        "target_backend": target_backend,
        "failure_fraction": float(failure_fraction),
        "failure_shots": int(failure_shots),
        "source_no_failure_shots": int(source_baseline["shots"]),
        "source_no_failure_final_tvd_to_aer": float(
            source_baseline["final_tvd_to_aer"]
        ),
        "target_no_failure_shots": int(target_baseline["shots"]),
        "target_no_failure_final_tvd_to_aer": float(
            target_baseline["final_tvd_to_aer"]
        ),
    }


def fixed_20k_continuation(
    *,
    common: Dict[str, object],
    source_counts: Counts,
    target_batches: Sequence[Batch],
    ideal: Counts,
    max_shots: int,
) -> Dict[str, object]:
    failure_shots = int(common["failure_shots"])
    target_shots = max_shots - failure_shots
    target_counts = take_exact_shots(target_batches, target_shots)
    final_counts = Counter(source_counts)
    merge_counts(final_counts, target_counts)
    final_tvd = tvd_weighted(final_counts, ideal)
    return {
        **common,
        "policy": "fixed_20k_continuation",
        "estimator_shots": int(max_shots),
        "post_failure_shots": int(target_shots),
        "discarded_pre_failure_shots": 0,
        "total_physical_shots": int(max_shots),
        "final_aggregated_tvd_to_aer": float(final_tvd),
        "delta_tvd_vs_source_no_failure": float(
            final_tvd - float(common["source_no_failure_final_tvd_to_aer"])
        ),
        "shot_delta_vs_source_no_failure": int(
            max_shots - int(common["source_no_failure_shots"])
        ),
        "stop_reason": "fixed_20k",
        "hit_stableshots_cap": False,
    }


def restart_stableshots(
    *,
    common: Dict[str, object],
    target_baseline: Mapping[str, object],
) -> Dict[str, object]:
    failure_shots = int(common["failure_shots"])
    target_shots = int(target_baseline["shots"])
    final_tvd = float(target_baseline["final_tvd_to_aer"])
    total_physical = failure_shots + target_shots
    return {
        **common,
        "policy": "restart_stableshots",
        "estimator_shots": target_shots,
        "post_failure_shots": target_shots,
        "discarded_pre_failure_shots": failure_shots,
        "total_physical_shots": int(total_physical),
        "final_aggregated_tvd_to_aer": final_tvd,
        "delta_tvd_vs_source_no_failure": float(
            final_tvd - float(common["source_no_failure_final_tvd_to_aer"])
        ),
        "shot_delta_vs_source_no_failure": int(
            total_physical - int(common["source_no_failure_shots"])
        ),
        "stop_reason": str(target_baseline["stop_reason"]),
        "hit_stableshots_cap": str(target_baseline["stop_reason"]) == "max_budget",
    }


def restart_fixed_20k(
    *,
    common: Dict[str, object],
    target_batches: Sequence[Batch],
    ideal: Counts,
    max_shots: int,
) -> Dict[str, object]:
    failure_shots = int(common["failure_shots"])
    target_counts = take_exact_shots(target_batches, max_shots)
    final_tvd = tvd_weighted(target_counts, ideal)
    total_physical = failure_shots + max_shots
    return {
        **common,
        "policy": "restart_fixed_20k",
        "estimator_shots": int(max_shots),
        "post_failure_shots": int(max_shots),
        "discarded_pre_failure_shots": failure_shots,
        "total_physical_shots": int(total_physical),
        "final_aggregated_tvd_to_aer": float(final_tvd),
        "delta_tvd_vs_source_no_failure": float(
            final_tvd - float(common["source_no_failure_final_tvd_to_aer"])
        ),
        "shot_delta_vs_source_no_failure": int(
            total_physical - int(common["source_no_failure_shots"])
        ),
        "stop_reason": "fixed_20k",
        "hit_stableshots_cap": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/handoff_baseline_config.json"),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    stable_raw = config["stableshots"]
    stable_config = StableShotsConfig(
        batch_size=int(stable_raw["batch_size"]),  # type: ignore[index]
        lookback_batches=int(stable_raw["lookback_batches"]),  # type: ignore[index]
        stability=int(stable_raw["stability"]),  # type: ignore[index]
        epsilon=float(stable_raw["epsilon"]),  # type: ignore[index]
        max_shots=int(stable_raw["max_shots"]),  # type: ignore[index]
    )

    output_dir = Path(str(config["output_dir"]))
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    algorithms = [str(value) for value in config["algorithms"]]
    sizes = [int(value) for value in config["sizes"]]
    backends = [str(value) for value in config["backends"]]
    fractions = [float(value) for value in config["failure_fractions"]]
    rows: List[Dict[str, object]] = []

    for algorithm in algorithms:
        for size in sizes:
            print(f"materializing baseline streams {algorithm}/{size}", flush=True)
            ideal, streams = materialize_circuit(
                algorithm, size, config, stable_config
            )
            _, baselines = run_no_failure_baselines(
                algorithm, size, streams, ideal, stable_config
            )

            for source_backend in backends:
                source_baseline = baselines[source_backend]
                for target_backend in backends:
                    if source_backend == target_backend:
                        continue
                    target_baseline = baselines[target_backend]
                    for failure_fraction in fractions:
                        failure_shots = rounded_failure_shots(
                            int(source_baseline["shots"]),
                            failure_fraction,
                            stable_config.batch_size,
                        )
                        source_controller = replay_source_prefix(
                            streams[source_backend],
                            stable_config,
                            failure_shots,
                        )
                        common = row_common(
                            algorithm=algorithm,
                            size=size,
                            source_backend=source_backend,
                            target_backend=target_backend,
                            failure_fraction=failure_fraction,
                            failure_shots=failure_shots,
                            source_baseline=source_baseline,
                            target_baseline=target_baseline,
                        )
                        rows.append(
                            fixed_20k_continuation(
                                common=common,
                                source_counts=Counter(source_controller.counts),
                                target_batches=streams[target_backend],
                                ideal=ideal,
                                max_shots=stable_config.max_shots,
                            )
                        )
                        rows.append(
                            restart_stableshots(
                                common=common,
                                target_baseline=target_baseline,
                            )
                        )
                        rows.append(
                            restart_fixed_20k(
                                common=common,
                                target_batches=streams[target_backend],
                                ideal=ideal,
                                max_shots=stable_config.max_shots,
                            )
                        )

    frame = pd.DataFrame(rows)
    expected = (
        len(algorithms)
        * len(sizes)
        * len(backends)
        * (len(backends) - 1)
        * len(fractions)
        * 3
    )
    if len(frame) != expected:
        raise RuntimeError(f"expected {expected} rows, found {len(frame)}")
    if frame[
        [
            "circuit_key",
            "source_backend",
            "target_backend",
            "failure_fraction",
            "policy",
        ]
    ].duplicated().any():
        raise RuntimeError("duplicate mandatory-failure baseline rows")

    frame.to_csv(raw_dir / "mandatory_failure_baselines.csv", index=False)
    manifest = {
        "rows": int(len(frame)),
        "policies": sorted(frame["policy"].unique().tolist()),
        "rows_per_policy": int(len(frame) // 3),
        "fixed_total_budget_continuation": stable_config.max_shots,
        "restart_fixed_target_shots": stable_config.max_shots,
        "restart_cost_definition": (
            "total physical shots include discarded pre-failure source shots"
        ),
    }
    (output_dir / "mandatory_failure_baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
