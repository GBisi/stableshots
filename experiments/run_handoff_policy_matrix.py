#!/usr/bin/env python3
"""Clean policy-matrix QPU failover experiment.

Policies cross stopping rule and history handling:
  1. StableShots + keep accumulated state (proposal)
  2. StableShots + discard state and restart on target
  3. Fixed 10k + keep accumulated counts
  4. Fixed 10k + discard counts and collect 10k fresh target shots
  5. Fixed 20k + keep accumulated counts
  6. Fixed 20k + discard counts and collect 20k fresh target shots

Failure fractions are defined relative to the matched no-failure stopping count
of the same stopping rule on the source backend. Thus fixed-10k failures occur
at fractions of 10k, fixed-20k at fractions of 20k, and StableShots failures at
fractions of the matched source StableShots stop.

For discard policies, total_physical_shots includes the pre-failure source shots
that are thrown away. estimator_shots counts only shots retained in the final
estimator.
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
    continue_on_target,
    load_config as load_base_config,
    materialize_circuit,
    merge_counts,
    replay_source_prefix,
    rounded_failure_shots,
    run_no_failure_baselines,
    tvd_weighted,
)


def load_config(path: Path) -> Dict[str, object]:
    config = json.loads(path.read_text())
    required = [
        "output_dir",
        "algorithms",
        "sizes",
        "backends",
        "failure_fractions",
        "fixed_budgets",
        "stableshots",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing configuration keys: {missing}")
    if [int(value) for value in config["fixed_budgets"]] != [10000, 20000]:
        raise ValueError("fixed_budgets must be exactly [10000, 20000]")
    if [float(value) for value in config["failure_fractions"]] != [
        0.1,
        0.25,
        0.5,
        0.75,
        0.9,
    ]:
        raise ValueError("unexpected failure fractions")
    return config


def take_exact_shots(batches: Sequence[Batch], requested_shots: int) -> Counts:
    counts: Counts = Counter()
    used = 0
    for batch_shots, batch_counts in batches:
        if used + int(batch_shots) > requested_shots:
            raise ValueError(
                f"requested_shots={requested_shots} is not aligned with batch size"
            )
        merge_counts(counts, batch_counts)
        used += int(batch_shots)
        if used == requested_shots:
            return counts
    raise RuntimeError(
        f"stream ended at {used} shots before requested {requested_shots}"
    )


def stable_no_failure_rows(
    algorithm: str,
    size: int,
    streams: Mapping[str, Sequence[Batch]],
    ideal: Counts,
    stable_config: StableShotsConfig,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    rows, baselines = run_no_failure_baselines(
        algorithm,
        size,
        streams,
        ideal,
        stable_config,
    )
    normalized: List[Dict[str, object]] = []
    for row in rows:
        normalized.append(
            {
                "circuit_key": row["circuit_key"],
                "algorithm": row["algorithm"],
                "size": row["size"],
                "backend": row["backend"],
                "stopping_rule": "stableshots",
                "nominal_budget": stable_config.max_shots,
                "estimator_shots": int(row["shots"]),
                "total_physical_shots": int(row["shots"]),
                "final_tvd_to_aer": float(row["final_tvd_to_aer"]),
                "stop_reason": row["stop_reason"],
                "hit_stableshots_cap": row["stop_reason"] == "max_budget",
            }
        )
    return normalized, baselines


def fixed_no_failure_rows(
    algorithm: str,
    size: int,
    streams: Mapping[str, Sequence[Batch]],
    ideal: Counts,
    budgets: Sequence[int],
) -> Tuple[List[Dict[str, object]], Dict[Tuple[str, int], Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    baselines: Dict[Tuple[str, int], Dict[str, object]] = {}
    for backend, batches in streams.items():
        for budget in budgets:
            counts = take_exact_shots(batches, budget)
            tvd = tvd_weighted(counts, ideal)
            row = {
                "circuit_key": f"{algorithm}_{size}",
                "algorithm": algorithm,
                "size": int(size),
                "backend": backend,
                "stopping_rule": f"fixed_{budget // 1000}k",
                "nominal_budget": int(budget),
                "estimator_shots": int(budget),
                "total_physical_shots": int(budget),
                "final_tvd_to_aer": float(tvd),
                "stop_reason": "fixed_budget",
                "hit_stableshots_cap": False,
            }
            rows.append(row)
            baselines[(backend, budget)] = {
                "counts": Counter(counts),
                "shots": int(budget),
                "final_tvd_to_aer": float(tvd),
                "stop_reason": "fixed_budget",
            }
    return rows, baselines


def common_handoff_fields(
    *,
    algorithm: str,
    size: int,
    source_backend: str,
    target_backend: str,
    failure_fraction: float,
    failure_shots: int,
    source_no_failure_shots: int,
    source_no_failure_tvd: float,
    stopping_rule: str,
    history_policy: str,
    nominal_budget: int,
) -> Dict[str, object]:
    policy_id = f"{stopping_rule}_{history_policy}"
    return {
        "circuit_key": f"{algorithm}_{size}",
        "algorithm": algorithm,
        "size": int(size),
        "source_backend": source_backend,
        "target_backend": target_backend,
        "failure_fraction": float(failure_fraction),
        "failure_shots": int(failure_shots),
        "actual_failure_fraction": float(
            failure_shots / source_no_failure_shots
        ),
        "stopping_rule": stopping_rule,
        "history_policy": history_policy,
        "policy_id": policy_id,
        "is_proposal": policy_id == "stableshots_keep",
        "nominal_budget": int(nominal_budget),
        "source_no_failure_shots": int(source_no_failure_shots),
        "source_no_failure_tvd_to_aer": float(source_no_failure_tvd),
    }


def stable_keep_row(
    *,
    algorithm: str,
    size: int,
    source_backend: str,
    target_backend: str,
    source_batches: Sequence[Batch],
    target_batches: Sequence[Batch],
    ideal: Counts,
    source_baseline: Mapping[str, object],
    failure_fraction: float,
    stable_config: StableShotsConfig,
) -> Dict[str, object]:
    source_nf_shots = int(source_baseline["shots"])
    failure_shots = rounded_failure_shots(
        source_nf_shots,
        failure_fraction,
        stable_config.batch_size,
    )
    controller = replay_source_prefix(
        source_batches,
        stable_config,
        failure_shots,
    )
    _, post_failure_shots, _ = continue_on_target(
        controller,
        target_batches,
    )
    final_tvd = tvd_weighted(controller.counts, ideal)
    total = int(controller.physical_shots)
    common = common_handoff_fields(
        algorithm=algorithm,
        size=size,
        source_backend=source_backend,
        target_backend=target_backend,
        failure_fraction=failure_fraction,
        failure_shots=failure_shots,
        source_no_failure_shots=source_nf_shots,
        source_no_failure_tvd=float(source_baseline["final_tvd_to_aer"]),
        stopping_rule="stableshots",
        history_policy="keep",
        nominal_budget=stable_config.max_shots,
    )
    return {
        **common,
        "estimator_shots": total,
        "post_failure_shots": int(post_failure_shots),
        "discarded_pre_failure_shots": 0,
        "total_physical_shots": total,
        "final_tvd_to_aer": float(final_tvd),
        "stop_reason": controller.stop_reason,
        "hit_stableshots_cap": controller.stop_reason == "max_budget",
    }


def stable_discard_row(
    *,
    algorithm: str,
    size: int,
    source_backend: str,
    target_backend: str,
    source_baseline: Mapping[str, object],
    target_baseline: Mapping[str, object],
    failure_fraction: float,
    stable_config: StableShotsConfig,
) -> Dict[str, object]:
    source_nf_shots = int(source_baseline["shots"])
    failure_shots = rounded_failure_shots(
        source_nf_shots,
        failure_fraction,
        stable_config.batch_size,
    )
    target_shots = int(target_baseline["shots"])
    total_physical = failure_shots + target_shots
    common = common_handoff_fields(
        algorithm=algorithm,
        size=size,
        source_backend=source_backend,
        target_backend=target_backend,
        failure_fraction=failure_fraction,
        failure_shots=failure_shots,
        source_no_failure_shots=source_nf_shots,
        source_no_failure_tvd=float(source_baseline["final_tvd_to_aer"]),
        stopping_rule="stableshots",
        history_policy="discard",
        nominal_budget=stable_config.max_shots,
    )
    return {
        **common,
        "estimator_shots": target_shots,
        "post_failure_shots": target_shots,
        "discarded_pre_failure_shots": failure_shots,
        "total_physical_shots": int(total_physical),
        "final_tvd_to_aer": float(target_baseline["final_tvd_to_aer"]),
        "stop_reason": str(target_baseline["stop_reason"]),
        "hit_stableshots_cap": str(target_baseline["stop_reason"]) == "max_budget",
    }


def fixed_keep_row(
    *,
    algorithm: str,
    size: int,
    source_backend: str,
    target_backend: str,
    source_batches: Sequence[Batch],
    target_batches: Sequence[Batch],
    ideal: Counts,
    source_baseline: Mapping[str, object],
    failure_fraction: float,
    budget: int,
    batch_size: int,
) -> Dict[str, object]:
    failure_shots = rounded_failure_shots(
        budget,
        failure_fraction,
        batch_size,
    )
    source_counts = take_exact_shots(source_batches, failure_shots)
    target_shots = budget - failure_shots
    target_counts = take_exact_shots(target_batches, target_shots)
    final_counts = Counter(source_counts)
    merge_counts(final_counts, target_counts)
    final_tvd = tvd_weighted(final_counts, ideal)
    stopping_rule = f"fixed_{budget // 1000}k"
    common = common_handoff_fields(
        algorithm=algorithm,
        size=size,
        source_backend=source_backend,
        target_backend=target_backend,
        failure_fraction=failure_fraction,
        failure_shots=failure_shots,
        source_no_failure_shots=budget,
        source_no_failure_tvd=float(source_baseline["final_tvd_to_aer"]),
        stopping_rule=stopping_rule,
        history_policy="keep",
        nominal_budget=budget,
    )
    return {
        **common,
        "estimator_shots": int(budget),
        "post_failure_shots": int(target_shots),
        "discarded_pre_failure_shots": 0,
        "total_physical_shots": int(budget),
        "final_tvd_to_aer": float(final_tvd),
        "stop_reason": "fixed_budget",
        "hit_stableshots_cap": False,
    }


def fixed_discard_row(
    *,
    algorithm: str,
    size: int,
    source_backend: str,
    target_backend: str,
    target_batches: Sequence[Batch],
    ideal: Counts,
    source_baseline: Mapping[str, object],
    failure_fraction: float,
    budget: int,
    batch_size: int,
) -> Dict[str, object]:
    failure_shots = rounded_failure_shots(
        budget,
        failure_fraction,
        batch_size,
    )
    target_counts = take_exact_shots(target_batches, budget)
    final_tvd = tvd_weighted(target_counts, ideal)
    stopping_rule = f"fixed_{budget // 1000}k"
    common = common_handoff_fields(
        algorithm=algorithm,
        size=size,
        source_backend=source_backend,
        target_backend=target_backend,
        failure_fraction=failure_fraction,
        failure_shots=failure_shots,
        source_no_failure_shots=budget,
        source_no_failure_tvd=float(source_baseline["final_tvd_to_aer"]),
        stopping_rule=stopping_rule,
        history_policy="discard",
        nominal_budget=budget,
    )
    return {
        **common,
        "estimator_shots": int(budget),
        "post_failure_shots": int(budget),
        "discarded_pre_failure_shots": int(failure_shots),
        "total_physical_shots": int(failure_shots + budget),
        "final_tvd_to_aer": float(final_tvd),
        "stop_reason": "fixed_budget",
        "hit_stableshots_cap": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/handoff_policy_matrix_config.json"),
    )
    args = parser.parse_args()
    config = load_config(args.config)

    stable_raw = config["stableshots"]
    stable_config = StableShotsConfig(
        batch_size=int(stable_raw["batch_size"]),
        lookback_batches=int(stable_raw["lookback_batches"]),
        stability=int(stable_raw["stability"]),
        epsilon=float(stable_raw["epsilon"]),
        max_shots=int(stable_raw["max_shots"]),
    )

    algorithms = [str(value) for value in config["algorithms"]]
    sizes = [int(value) for value in config["sizes"]]
    backends = [str(value) for value in config["backends"]]
    fractions = [float(value) for value in config["failure_fractions"]]
    budgets = [int(value) for value in config["fixed_budgets"]]

    output_dir = Path(str(config["output_dir"]))
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    no_failure_rows: List[Dict[str, object]] = []
    handoff_rows: List[Dict[str, object]] = []

    # materialize_circuit expects the same core fields as the baseline config.
    base_config = dict(config)

    for algorithm in algorithms:
        for size in sizes:
            print(f"materializing policy matrix {algorithm}/{size}", flush=True)
            ideal, streams = materialize_circuit(
                algorithm,
                size,
                base_config,
                stable_config,
            )

            stable_rows, stable_baselines = stable_no_failure_rows(
                algorithm,
                size,
                streams,
                ideal,
                stable_config,
            )
            fixed_rows, fixed_baselines = fixed_no_failure_rows(
                algorithm,
                size,
                streams,
                ideal,
                budgets,
            )
            no_failure_rows.extend(stable_rows)
            no_failure_rows.extend(fixed_rows)

            for source_backend in backends:
                for target_backend in backends:
                    if source_backend == target_backend:
                        continue
                    for failure_fraction in fractions:
                        handoff_rows.append(
                            stable_keep_row(
                                algorithm=algorithm,
                                size=size,
                                source_backend=source_backend,
                                target_backend=target_backend,
                                source_batches=streams[source_backend],
                                target_batches=streams[target_backend],
                                ideal=ideal,
                                source_baseline=stable_baselines[source_backend],
                                failure_fraction=failure_fraction,
                                stable_config=stable_config,
                            )
                        )
                        handoff_rows.append(
                            stable_discard_row(
                                algorithm=algorithm,
                                size=size,
                                source_backend=source_backend,
                                target_backend=target_backend,
                                source_baseline=stable_baselines[source_backend],
                                target_baseline=stable_baselines[target_backend],
                                failure_fraction=failure_fraction,
                                stable_config=stable_config,
                            )
                        )
                        for budget in budgets:
                            source_fixed = fixed_baselines[
                                (source_backend, budget)
                            ]
                            handoff_rows.append(
                                fixed_keep_row(
                                    algorithm=algorithm,
                                    size=size,
                                    source_backend=source_backend,
                                    target_backend=target_backend,
                                    source_batches=streams[source_backend],
                                    target_batches=streams[target_backend],
                                    ideal=ideal,
                                    source_baseline=source_fixed,
                                    failure_fraction=failure_fraction,
                                    budget=budget,
                                    batch_size=stable_config.batch_size,
                                )
                            )
                            handoff_rows.append(
                                fixed_discard_row(
                                    algorithm=algorithm,
                                    size=size,
                                    source_backend=source_backend,
                                    target_backend=target_backend,
                                    target_batches=streams[target_backend],
                                    ideal=ideal,
                                    source_baseline=source_fixed,
                                    failure_fraction=failure_fraction,
                                    budget=budget,
                                    batch_size=stable_config.batch_size,
                                )
                            )

    no_failure = pd.DataFrame(no_failure_rows)
    handoff = pd.DataFrame(handoff_rows)

    expected_no_failure = (
        len(algorithms) * len(sizes) * len(backends) * 3
    )
    expected_handoff = (
        len(algorithms)
        * len(sizes)
        * len(backends)
        * (len(backends) - 1)
        * len(fractions)
        * 6
    )
    if len(no_failure) != expected_no_failure:
        raise RuntimeError(
            f"expected {expected_no_failure} no-failure rows, got {len(no_failure)}"
        )
    if len(handoff) != expected_handoff:
        raise RuntimeError(
            f"expected {expected_handoff} handoff rows, got {len(handoff)}"
        )
    if handoff[
        [
            "circuit_key",
            "source_backend",
            "target_backend",
            "failure_fraction",
            "policy_id",
        ]
    ].duplicated().any():
        raise RuntimeError("duplicate handoff policy rows")

    no_failure.to_csv(raw_dir / "no_failure_policy_runs.csv", index=False)
    handoff.to_csv(raw_dir / "handoff_policy_runs.csv", index=False)

    manifest = {
        "circuits": len(algorithms) * len(sizes),
        "backends": len(backends),
        "directed_pairs": len(backends) * (len(backends) - 1),
        "failure_fractions": fractions,
        "fixed_budgets": budgets,
        "no_failure_rows": int(len(no_failure)),
        "handoff_rows": int(len(handoff)),
        "policies": sorted(handoff["policy_id"].unique().tolist()),
        "proposal": "stableshots_keep",
        "failure_fraction_basis": "matched no-failure stopping count of same stopping rule",
        "ideal_reference_shots": int(config["ideal_reference_shots"]),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
