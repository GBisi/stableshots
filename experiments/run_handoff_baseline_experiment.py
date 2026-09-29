#!/usr/bin/env python3
"""Clean directed QPU-handoff experiment for StableShots.

The experiment intentionally uses one handoff policy only: naive cumulative
continuation.  StableShots state is carried unchanged across the backend switch.

For every circuit:
  * run a no-failure baseline on every backend;
  * for every directed source != target pair;
  * fail at 10%, 25%, 50%, 75%, and 90% of the matched source no-failure
    stopping count;
  * continue on the target backend under the same total physical-shot budget.

The output records the estimator at the failure point, the post-failure-only
target segment, and the final cumulative estimator, all against the same
high-shot Aer reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import pandas as pd

from stableshot.main import (
    StableShotsConfig,
    TraceSpec,
    iter_execution_batches,
    load_qsimbench_batches,
    run_stable_shots,
)

Counts = Counter[str]
Batch = Tuple[int, Counts]


def deterministic_seed(base: int, *parts: object) -> int:
    payload = "|".join([str(base), *(str(part) for part in parts)])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def total_mass(counts: Mapping[str, float]) -> float:
    return float(sum(float(value) for value in counts.values()))


def merge_counts(
    target: MutableMapping[str, float],
    source: Mapping[str, float],
) -> None:
    for outcome, value in source.items():
        target[str(outcome)] = float(target.get(str(outcome), 0.0)) + float(value)


def tvd_weighted(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    left_mass = total_mass(left)
    right_mass = total_mass(right)
    if left_mass <= 0 or right_mass <= 0:
        raise ValueError("TVD requires non-empty distributions")
    support = set(left) | set(right)
    return 0.5 * sum(
        abs(
            float(left.get(outcome, 0.0)) / left_mass
            - float(right.get(outcome, 0.0)) / right_mass
        )
        for outcome in support
    )


def load_config(path: Path) -> Dict[str, object]:
    config = json.loads(path.read_text())
    required = [
        "output_dir",
        "algorithms",
        "sizes",
        "backends",
        "ideal_backend",
        "ideal_reference_shots",
        "source_batch_size",
        "sampling_strategy",
        "sampling_seed",
        "reference_seed",
        "failure_fractions",
        "stableshots",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing configuration keys: {missing}")
    fractions = [float(value) for value in config["failure_fractions"]]
    if fractions != [0.1, 0.25, 0.5, 0.75, 0.9]:
        raise ValueError("failure_fractions must be exactly [0.1, 0.25, 0.5, 0.75, 0.9]")
    return config


def execution_batches(
    raw_batches: Sequence[Batch],
    config: StableShotsConfig,
) -> List[Batch]:
    return list(iter_execution_batches(raw_batches, config.batch_size, config.max_shots))


def get_ideal_reference(
    algorithm: str,
    size: int,
    backend: str,
    shots: int,
    circuit_kind: str,
    seed: int,
    force: bool,
) -> Counts:
    from qsimbench import get_outcomes  # type: ignore

    counts = get_outcomes(
        algorithm=algorithm,
        size=size,
        backend=backend,
        shots=shots,
        circuit_kind=circuit_kind,
        exact=True,
        strategy="random",
        seed=seed,
        force=force,
    )
    return Counter({str(key): int(value) for key, value in counts.items() if int(value) > 0})


@dataclass
class ContinuationController:
    config: StableShotsConfig
    counts: Counts = field(default_factory=Counter)
    snapshots: List[Counts] = field(default_factory=list)
    stable_checks: int = 0
    checks_performed: int = 0
    last_delta: float = float("nan")
    physical_shots: int = 0
    stopped: bool = False
    stop_reason: str = ""

    def apply_batch(self, batch_shots: int, batch_counts: Mapping[str, int]) -> bool:
        if self.stopped:
            return True

        merge_counts(self.counts, batch_counts)
        self.physical_shots += int(batch_shots)
        self.snapshots.append(Counter(self.counts))

        if len(self.snapshots) > self.config.lookback_batches:
            previous = self.snapshots[-1 - self.config.lookback_batches]
            self.last_delta = tvd_weighted(self.counts, previous)
            self.checks_performed += 1
            if self.last_delta <= self.config.epsilon:
                self.stable_checks += 1
            else:
                self.stable_checks = 0
            if self.stable_checks >= self.config.stability:
                self.stopped = True
                self.stop_reason = "stable"
                return True

        if self.physical_shots >= self.config.max_shots:
            self.stopped = True
            self.stop_reason = "max_budget"
            return True
        return False


def rounded_failure_shots(
    no_failure_shots: int,
    fraction: float,
    batch_size: int,
) -> int:
    if not (0.0 < fraction < 1.0):
        raise ValueError(f"failure fraction must be in (0,1), got {fraction}")
    failure_shots = int(math.floor(no_failure_shots * fraction / batch_size)) * batch_size
    failure_shots = max(batch_size, failure_shots)
    if no_failure_shots > batch_size:
        failure_shots = min(failure_shots, no_failure_shots - batch_size)
    return failure_shots


def replay_source_prefix(
    batches: Sequence[Batch],
    config: StableShotsConfig,
    failure_shots: int,
) -> ContinuationController:
    controller = ContinuationController(config=config)
    for batch_shots, batch_counts in batches:
        if controller.physical_shots + batch_shots > failure_shots:
            break
        controller.apply_batch(batch_shots, batch_counts)
        if controller.stopped:
            raise RuntimeError(
                "source controller stopped before the requested failure point; "
                "the failure point must precede the matched no-failure stop"
            )
        if controller.physical_shots == failure_shots:
            break
    if controller.physical_shots != failure_shots:
        raise RuntimeError(
            f"could not materialize failure prefix: requested={failure_shots}, "
            f"materialized={controller.physical_shots}"
        )
    return controller


def continue_on_target(
    controller: ContinuationController,
    target_batches: Sequence[Batch],
) -> Tuple[Counts, int, int]:
    target_only: Counts = Counter()
    post_failure_shots = 0
    target_batches_consumed = 0

    for batch_shots, batch_counts in target_batches:
        if controller.physical_shots + batch_shots > controller.config.max_shots:
            break
        merge_counts(target_only, batch_counts)
        post_failure_shots += int(batch_shots)
        target_batches_consumed += 1
        if controller.apply_batch(batch_shots, batch_counts):
            break

    if not controller.stopped:
        controller.stopped = True
        controller.stop_reason = (
            "max_budget"
            if controller.physical_shots >= controller.config.max_shots
            else "input_exhausted"
        )

    return target_only, post_failure_shots, target_batches_consumed


def materialize_circuit(
    algorithm: str,
    size: int,
    config: Mapping[str, object],
    stable_config: StableShotsConfig,
) -> Tuple[Counts, Dict[str, List[Batch]]]:
    circuit_kind = str(config.get("circuit_kind", "circuit"))
    force = bool(config.get("force_download", False))
    ideal_seed = deterministic_seed(
        int(config["reference_seed"]),
        algorithm,
        size,
        config["ideal_backend"],
    )
    ideal = get_ideal_reference(
        algorithm=algorithm,
        size=size,
        backend=str(config["ideal_backend"]),
        shots=int(config["ideal_reference_shots"]),
        circuit_kind=circuit_kind,
        seed=ideal_seed,
        force=force,
    )

    streams: Dict[str, List[Batch]] = {}
    for backend in config["backends"]:  # type: ignore[assignment]
        backend_name = str(backend)
        spec = TraceSpec(
            algorithm=algorithm,
            size=size,
            backend=backend_name,
            circuit_kind=circuit_kind,
        )
        raw = load_qsimbench_batches(
            spec=spec,
            total_shots=stable_config.max_shots,
            source_batch_size=int(config["source_batch_size"]),
            sampling_strategy=str(config["sampling_strategy"]),
            sampling_seed=int(config["sampling_seed"]),
            force=force,
        )
        streams[backend_name] = execution_batches(raw, stable_config)
    return ideal, streams


def run_no_failure_baselines(
    algorithm: str,
    size: int,
    streams: Mapping[str, Sequence[Batch]],
    ideal: Counts,
    stable_config: StableShotsConfig,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    by_backend: Dict[str, Dict[str, object]] = {}

    for backend, batches in streams.items():
        counts, shots, last_delta, stop_reason = run_stable_shots(
            [(batch_shots, Counter(batch_counts)) for batch_shots, batch_counts in batches],
            stable_config,
        )
        final_tvd = tvd_weighted(counts, ideal)
        row = {
            "experiment": "no_failure",
            "circuit_key": f"{algorithm}_{size}",
            "algorithm": algorithm,
            "size": int(size),
            "backend": backend,
            "shots": int(shots),
            "final_tvd_to_aer": float(final_tvd),
            "stop_reason": stop_reason,
            "last_stability_tvd": float(last_delta),
        }
        rows.append(row)
        by_backend[backend] = {
            "counts": Counter(counts),
            "shots": int(shots),
            "final_tvd_to_aer": float(final_tvd),
            "stop_reason": stop_reason,
        }

    return rows, by_backend


def run_handoff(
    *,
    algorithm: str,
    size: int,
    source_backend: str,
    target_backend: str,
    source_batches: Sequence[Batch],
    target_batches: Sequence[Batch],
    ideal: Counts,
    source_baseline: Mapping[str, object],
    target_baseline: Mapping[str, object],
    failure_fraction: float,
    stable_config: StableShotsConfig,
) -> Dict[str, object]:
    source_no_failure_shots = int(source_baseline["shots"])
    failure_shots = rounded_failure_shots(
        source_no_failure_shots,
        failure_fraction,
        stable_config.batch_size,
    )

    controller = replay_source_prefix(source_batches, stable_config, failure_shots)
    failure_counts = Counter(controller.counts)
    failure_point_tvd = tvd_weighted(failure_counts, ideal)
    stable_streak_at_failure = int(controller.stable_checks)
    last_stability_tvd_at_failure = float(controller.last_delta)

    target_only_counts, post_failure_shots, target_batches_consumed = continue_on_target(
        controller,
        target_batches,
    )
    final_counts = Counter(controller.counts)
    final_tvd = tvd_weighted(final_counts, ideal)
    post_failure_only_tvd = (
        tvd_weighted(target_only_counts, ideal)
        if post_failure_shots > 0
        else float("nan")
    )
    total_shots = int(controller.physical_shots)

    return {
        "experiment": "directed_handoff",
        "policy": "naive_continuation",
        "circuit_key": f"{algorithm}_{size}",
        "algorithm": algorithm,
        "size": int(size),
        "source_backend": source_backend,
        "target_backend": target_backend,
        "failure_fraction": float(failure_fraction),
        "source_no_failure_shots": source_no_failure_shots,
        "source_no_failure_final_tvd_to_aer": float(source_baseline["final_tvd_to_aer"]),
        "target_no_failure_shots": int(target_baseline["shots"]),
        "target_no_failure_final_tvd_to_aer": float(target_baseline["final_tvd_to_aer"]),
        "failure_shots": int(failure_shots),
        "actual_failure_fraction": float(failure_shots / source_no_failure_shots),
        "failure_point_tvd_to_aer": float(failure_point_tvd),
        "stable_streak_at_failure": stable_streak_at_failure,
        "last_stability_tvd_at_failure": last_stability_tvd_at_failure,
        "post_failure_shots": int(post_failure_shots),
        "post_failure_only_tvd_to_aer": float(post_failure_only_tvd),
        "target_batches_consumed": int(target_batches_consumed),
        "total_shots": total_shots,
        "source_evidence_share": float(failure_shots / total_shots),
        "target_evidence_share": float(post_failure_shots / total_shots),
        "final_aggregated_tvd_to_aer": float(final_tvd),
        "delta_tvd_failure_to_final": float(final_tvd - failure_point_tvd),
        "delta_tvd_vs_source_no_failure": float(
            final_tvd - float(source_baseline["final_tvd_to_aer"])
        ),
        "delta_tvd_vs_target_no_failure": float(
            final_tvd - float(target_baseline["final_tvd_to_aer"])
        ),
        "shot_delta_vs_source_no_failure": int(total_shots - source_no_failure_shots),
        "stop_reason": controller.stop_reason,
        "last_stability_tvd": float(controller.last_delta),
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

    no_failure_rows: List[Dict[str, object]] = []
    handoff_rows: List[Dict[str, object]] = []

    algorithms = [str(value) for value in config["algorithms"]]
    sizes = [int(value) for value in config["sizes"]]
    backends = [str(value) for value in config["backends"]]
    fractions = [float(value) for value in config["failure_fractions"]]

    for algorithm in algorithms:
        for size in sizes:
            print(f"materializing {algorithm}/{size}", flush=True)
            ideal, streams = materialize_circuit(
                algorithm,
                size,
                config,
                stable_config,
            )
            baseline_rows, baselines = run_no_failure_baselines(
                algorithm,
                size,
                streams,
                ideal,
                stable_config,
            )
            no_failure_rows.extend(baseline_rows)

            for source_backend in backends:
                for target_backend in backends:
                    if source_backend == target_backend:
                        continue
                    for failure_fraction in fractions:
                        handoff_rows.append(
                            run_handoff(
                                algorithm=algorithm,
                                size=size,
                                source_backend=source_backend,
                                target_backend=target_backend,
                                source_batches=streams[source_backend],
                                target_batches=streams[target_backend],
                                ideal=ideal,
                                source_baseline=baselines[source_backend],
                                target_baseline=baselines[target_backend],
                                failure_fraction=failure_fraction,
                                stable_config=stable_config,
                            )
                        )

    no_failure = pd.DataFrame(no_failure_rows)
    handoffs = pd.DataFrame(handoff_rows)
    no_failure.to_csv(raw_dir / "no_failure_runs.csv", index=False)
    handoffs.to_csv(raw_dir / "handoff_runs.csv", index=False)

    expected_no_failure = len(algorithms) * len(sizes) * len(backends)
    expected_handoffs = (
        len(algorithms)
        * len(sizes)
        * len(backends)
        * (len(backends) - 1)
        * len(fractions)
    )
    if len(no_failure) != expected_no_failure:
        raise RuntimeError(
            f"expected {expected_no_failure} no-failure rows, got {len(no_failure)}"
        )
    if len(handoffs) != expected_handoffs:
        raise RuntimeError(
            f"expected {expected_handoffs} handoff rows, got {len(handoffs)}"
        )

    manifest = {
        "experiment_name": config.get("experiment_name", "qpu_handoff_baseline"),
        "circuits": len(algorithms) * len(sizes),
        "backend_count": len(backends),
        "directed_backend_pairs": len(backends) * (len(backends) - 1),
        "failure_fractions": fractions,
        "no_failure_rows": int(len(no_failure)),
        "handoff_rows": int(len(handoffs)),
        "policy": "naive_continuation",
        "total_physical_shot_budget": stable_config.max_shots,
        "ideal_reference_shots": int(config["ideal_reference_shots"]),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
