#!/usr/bin/env python3
"""
StableShots multi-QPU experiment runner.

Version: grid-search-fast-v4

This script implements a two-stage workflow:

1. materialize:
   - load QSimBench traces once,
   - store raw batch counts, reference counts, fixed baselines, and local
     StableShots outputs.

2. aggregate:
   - enumerate every non-empty backend subset,
   - evaluate three multi-QPU policies offline from the same materialized
     raw batches:

     a) local_only_offline_aggregate
        Each QPU runs StableShots independently; final stopped distributions
        are aggregated offline. This is the original multi-QPU extension.

     b) aggregate_only_sync
        All QPUs in a subset are queried synchronously with one equal batch per
        backend per global round. StableShots is applied only to the aggregate
        distribution. The whole ensemble stops when the aggregate stabilizes.

     c) hierarchical_local_aggregate
        Every QPU has an independent local StableShots controller and the
        aggregate has its own controller. When one QPU stabilizes locally, it is
        frozen and no longer queried. The aggregate continues using fixed
        subset weights over frozen plus live backend distributions. When the
        aggregate stabilizes, all remaining QPUs stop.

The aggregate-only and hierarchical policies require raw_batches in
trace_payloads.jsonl.gz. Keep raw_batches_saved=True for those policies.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_ALGORITHMS = ["dj", "qaoa", "qft", "qnn", "random", "vqe"]
DEFAULT_SIZES = list(range(4, 15))
DEFAULT_BACKENDS = [
    "fake_fez",
    "fake_kyiv",
    "fake_marrakesh",
    "fake_sherbrooke",
    "fake_torino",
]
DEFAULT_FIXED_BASELINES = [1000, 2500, 5000, 10000, 15000, 18000]
DEFAULT_REFERENCE_SHOTS = 20000
DEFAULT_SOURCE_BATCH_SIZE = 50
DEFAULT_SELECTED_CONFIG_ID = "b50_lb3_k5_eps0p005"
DEFAULT_WEIGHTING_SCHEMES = ["uniform", "shot_proportional", "confidence"]
DEFAULT_POLICIES = [
    "local_only_offline_aggregate",
    "aggregate_only_sync",
    "hierarchical_local_aggregate",
]
DEFAULT_GRID_SPLIT_MODE = "stratified_size"
GRID_SPLIT_MODES = {"stratified_size", "grouped_circuit_random"}
GRID_SELECTION_RISK_STATS = {"q95", "max"}
GRID_SELECTION_OBJECTIVES = {"max_ssr", "max_parallel_ssr", "min_tvd"}

Counts = Counter[str]
Batch = Tuple[int, Counts]

# Caches are intentionally process-local. Grid search repeatedly touches the same
# payloads across 31 backend subsets, 3 policies, and many configurations. Without
# caching, raw_batches are converted from JSON and batch prefixes are rebuilt thousands
# of times.
_RAW_BATCH_CACHE: Dict[str, List[Batch]] = {}
_PREFIX_BATCH_CACHE: Dict[Tuple[str, int, int], List[Batch]] = {}
_LOCAL_STABLE_CACHE: Dict[Tuple[str, str], Dict[str, object]] = {}


@dataclass(frozen=True)
class StableShotsConfig:
    batch_size: int = 50
    lookback_batches: int = 3
    stability: int = 5
    epsilon: float = 0.005
    max_shots: int = DEFAULT_REFERENCE_SHOTS

    @property
    def config_id(self) -> str:
        eps = f"{self.epsilon:g}".replace(".", "p")
        return f"b{self.batch_size}_lb{self.lookback_batches}_k{self.stability}_eps{eps}"

    @property
    def strategy_name(self) -> str:
        return f"stable_shots_{self.config_id}"


@dataclass(frozen=True)
class StableShotsGrid:
    batch_sizes: Sequence[int]
    lookback_batches: Sequence[int]
    stabilities: Sequence[int]
    epsilons: Sequence[float]
    max_shots: int = DEFAULT_REFERENCE_SHOTS

    def expand(self) -> List[StableShotsConfig]:
        configs: List[StableShotsConfig] = []
        for batch_size, lookback, stability, epsilon in product(
            self.batch_sizes, self.lookback_batches, self.stabilities, self.epsilons
        ):
            configs.append(
                StableShotsConfig(
                    batch_size=int(batch_size),
                    lookback_batches=int(lookback),
                    stability=int(stability),
                    epsilon=float(epsilon),
                    max_shots=int(self.max_shots),
                )
            )
        return configs


@dataclass(frozen=True)
class TraceSpec:
    algorithm: str
    size: int
    backend: str
    circuit_kind: str = "circuit"

    @property
    def trace_id(self) -> str:
        return f"{self.algorithm}_{self.size}_{self.backend}"

    @property
    def circuit_key(self) -> str:
        return f"{self.algorithm}_{self.size}_{self.circuit_kind}"


@dataclass
class LocalState:
    backend: str
    cumulative: Counts
    snapshots: List[Counts]
    stable_checks: int
    shots: int
    last_delta: float
    checks_performed: int
    stopped: bool
    stop_reason: str
    stop_round: Optional[int]


@dataclass
class AggregateState:
    snapshots: List[Dict[str, float]]
    stable_checks: int
    last_delta: float
    checks_performed: int
    stopped: bool
    stop_reason: str
    stop_round: Optional[int]


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------


def now_seconds() -> float:
    return time.perf_counter()


def parse_csv_strings(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_csv_ints(value: str) -> List[int]:
    return [int(item) for item in parse_csv_strings(value)]


def parse_csv_floats(value: str) -> List[float]:
    return [float(item) for item in parse_csv_strings(value)]


def require_non_empty(name: str, values: Sequence[object]) -> None:
    if not values:
        raise ValueError(f"{name} must contain at least one value")


def total_count(counts: Mapping[str, int]) -> int:
    return int(sum(int(v) for v in counts.values()))


def merge_counts(target: Counts, source: Mapping[str, int]) -> None:
    for bitstring, count in source.items():
        value = int(count)
        if value > 0:
            target[str(bitstring)] += value


def counts_from_json(obj: Mapping[str, object]) -> Counts:
    return Counter({str(k): int(v) for k, v in obj.items() if int(v) > 0})


def counts_to_json(counts: Mapping[str, int]) -> Dict[str, int]:
    return {str(k): int(v) for k, v in counts.items() if int(v) > 0}


def stable_sampling_seed(base_seed: int, spec: TraceSpec, batch_index: int) -> int:
    payload = f"{base_seed}|{spec.trace_id}|{spec.circuit_kind}|{batch_index}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def deterministic_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


# ---------------------------------------------------------------------------
# Distribution and prefix operations
# ---------------------------------------------------------------------------


def deterministic_subcount(counts: Mapping[str, int], target_shots: int) -> Counts:
    source_total = total_count(counts)
    if target_shots > source_total:
        raise ValueError("cannot take more shots than available")
    if target_shots == source_total:
        return Counter({str(k): int(v) for k, v in counts.items()})
    if target_shots == 0:
        return Counter()

    quotas = []
    for key, value in counts.items():
        exact = int(value) * target_shots / source_total
        base = int(math.floor(exact))
        quotas.append((str(key), base, exact - base))

    result: Counts = Counter({key: base for key, base, _ in quotas if base > 0})
    missing = target_shots - total_count(result)
    quotas.sort(key=lambda item: (-item[2], item[0]))
    for key, _, _ in quotas[:missing]:
        result[key] += 1
    return result


def prefix_counts(raw_batches: Sequence[Batch], shots: int) -> Counts:
    counts: Counts = Counter()
    used = 0
    for batch_shots, batch_counts in raw_batches:
        if used >= shots:
            break
        remaining = shots - used
        if batch_shots <= remaining:
            merge_counts(counts, batch_counts)
            used += batch_shots
        else:
            merge_counts(counts, deterministic_subcount(batch_counts, remaining))
            used += remaining
    if used < shots:
        raise RuntimeError(f"requested {shots} shots but found only {used}")
    return counts


def iter_execution_batches(raw_batches: Sequence[Batch], batch_size: int, max_shots: int) -> Iterator[Batch]:
    emitted = 0
    buffer: Counts = Counter()
    buffer_shots = 0
    for shots, counts in raw_batches:
        if emitted >= max_shots:
            break
        remaining_record = Counter(counts)
        remaining_shots = int(shots)
        while remaining_shots > 0 and emitted < max_shots:
            needed_batch = batch_size - buffer_shots
            needed_budget = max_shots - emitted - buffer_shots
            take = min(remaining_shots, needed_batch, needed_budget)
            if take == remaining_shots:
                take_counts = Counter(remaining_record)
                remaining_record = Counter()
            else:
                take_counts = deterministic_subcount(remaining_record, take)
                for key, value in take_counts.items():
                    remaining_record[key] -= value
                    if remaining_record[key] <= 0:
                        del remaining_record[key]
            merge_counts(buffer, take_counts)
            buffer_shots += take
            remaining_shots -= take
            if buffer_shots == batch_size or emitted + buffer_shots == max_shots:
                yield buffer_shots, Counter(buffer)
                emitted += buffer_shots
                buffer = Counter()
                buffer_shots = 0


def make_batch_prefixes(raw_batches: Sequence[Batch], batch_size: int, max_shots: int) -> List[Batch]:
    return list(iter_execution_batches(raw_batches, batch_size, max_shots))


def tvd_counts(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    """Fast TVD between sparse count dictionaries.

    Avoids constructing the full union set. This matters for 14/15-qubit
    traces where supports can contain many thousands of observed bitstrings and
    TVD is evaluated repeatedly during StableShots checks.
    """
    n_left = total_count(left)
    n_right = total_count(right)
    if n_left <= 0 or n_right <= 0:
        raise ValueError("TVD requires non-empty distributions")
    inv_left = 1.0 / n_left
    inv_right = 1.0 / n_right
    total = 0.0
    right_get = right.get
    for key, left_value in left.items():
        total += abs(float(left_value) * inv_left - float(right_get(key, 0)) * inv_right)
    for key, right_value in right.items():
        if key not in left:
            total += abs(float(right_value) * inv_right)
    return 0.5 * total


def counts_to_probs(counts: Mapping[str, int]) -> Dict[str, float]:
    n = total_count(counts)
    if n <= 0:
        raise ValueError("cannot normalize empty counts")
    return {str(k): int(v) / n for k, v in counts.items() if int(v) > 0}


def weighted_distribution(counts_list: Sequence[Mapping[str, int]], weights: Sequence[float]) -> Dict[str, float]:
    if len(counts_list) != len(weights):
        raise ValueError("counts_list and weights must have equal length")
    out: Dict[str, float] = {}
    for counts, weight in zip(counts_list, weights):
        probs = counts_to_probs(counts)
        for bitstring, value in probs.items():
            out[bitstring] = out.get(bitstring, 0.0) + float(weight) * value
    return out


def weighted_distribution_from_prob_list(prob_list: Sequence[Mapping[str, float]], weights: Sequence[float]) -> Dict[str, float]:
    if len(prob_list) != len(weights):
        raise ValueError("prob_list and weights must have equal length")
    out: Dict[str, float] = {}
    for probs, weight in zip(prob_list, weights):
        for bitstring, value in probs.items():
            out[bitstring] = out.get(bitstring, 0.0) + float(weight) * float(value)
    return out


def tvd_probs(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    total = 0.0
    right_get = right.get
    for key, left_value in left.items():
        total += abs(float(left_value) - float(right_get(key, 0.0)))
    for key, right_value in right.items():
        if key not in left:
            total += abs(float(right_value))
    return 0.5 * total


# ---------------------------------------------------------------------------
# QSimBench loading and StableShots
# ---------------------------------------------------------------------------


def load_qsimbench_batches(
    spec: TraceSpec,
    total_shots: int,
    source_batch_size: int,
    sampling_strategy: str,
    sampling_seed: int,
    force: bool,
) -> List[Batch]:
    try:
        from qsimbench import get_outcomes  # type: ignore
    except Exception as exc:
        raise RuntimeError("could not import qsimbench.get_outcomes; install qsimbench") from exc

    if total_shots <= 0:
        raise ValueError("total_shots must be positive")
    if source_batch_size <= 0:
        raise ValueError("source_batch_size must be positive")
    if sampling_strategy not in {"sequential", "random"}:
        raise ValueError("sampling_strategy must be 'sequential' or 'random'")

    batches: List[Batch] = []
    produced = 0
    batch_index = 0
    while produced < total_shots:
        requested = min(source_batch_size, total_shots - produced)
        counts = get_outcomes(
            algorithm=spec.algorithm,
            size=spec.size,
            backend=spec.backend,
            shots=requested,
            circuit_kind=spec.circuit_kind,
            exact=True,
            strategy=sampling_strategy,
            seed=stable_sampling_seed(sampling_seed, spec, batch_index),
            force=force and batch_index == 0,
        )
        batch_counts: Counts = Counter(
            {str(bitstring): int(count) for bitstring, count in counts.items() if int(count) > 0}
        )
        actual = total_count(batch_counts)
        if actual != requested:
            raise RuntimeError(
                f"get_outcomes returned {actual} shots for {spec.trace_id}; expected {requested}"
            )
        batches.append((actual, batch_counts))
        produced += actual
        batch_index += 1
    return batches


def update_local_state_with_batch(state: LocalState, batch_counts: Mapping[str, int], batch_shots: int, config: StableShotsConfig, round_index: int) -> None:
    if state.stopped:
        return
    merge_counts(state.cumulative, batch_counts)
    state.shots += int(batch_shots)
    state.snapshots.append(Counter(state.cumulative))
    if len(state.snapshots) > config.lookback_batches:
        previous = state.snapshots[-1 - config.lookback_batches]
        state.last_delta = tvd_counts(state.cumulative, previous)
        state.checks_performed += 1
        state.stable_checks = state.stable_checks + 1 if state.last_delta <= config.epsilon else 0
        if state.stable_checks >= config.stability:
            state.stopped = True
            state.stop_reason = "stable"
            state.stop_round = round_index
    if (not state.stopped) and state.shots >= config.max_shots:
        state.stopped = True
        state.stop_reason = "max_budget"
        state.stop_round = round_index


def run_stable_shots(raw_batches: Sequence[Batch], config: StableShotsConfig) -> Tuple[Counts, int, float, str, int]:
    state = LocalState(
        backend="",
        cumulative=Counter(),
        snapshots=[],
        stable_checks=0,
        shots=0,
        last_delta=float("nan"),
        checks_performed=0,
        stopped=False,
        stop_reason="running",
        stop_round=None,
    )
    for round_index, (batch_shots, batch_counts) in enumerate(
        iter_execution_batches(raw_batches, config.batch_size, config.max_shots), start=1
    ):
        update_local_state_with_batch(state, batch_counts, batch_shots, config, round_index)
        if state.stopped:
            break
    if not state.stopped:
        state.stop_reason = "max_budget"
    return Counter(state.cumulative), state.shots, state.last_delta, state.stop_reason, state.checks_performed


# ---------------------------------------------------------------------------
# JSONL materialization format
# ---------------------------------------------------------------------------


def raw_batches_to_json(raw_batches: Sequence[Batch]) -> List[Dict[str, object]]:
    return [{"shots": int(shots), "counts": counts_to_json(counts)} for shots, counts in raw_batches]


def raw_batches_from_json(rows: Sequence[Mapping[str, object]]) -> List[Batch]:
    batches: List[Batch] = []
    for row in rows:
        batches.append((int(row["shots"]), counts_from_json(row["counts"])))  # type: ignore[arg-type]
    return batches


def open_jsonl_gz(path: Path, mode: str):
    if "b" in mode:
        raise ValueError("open_jsonl_gz expects text mode")
    return gzip.open(path, mode, encoding="utf-8")


def write_payloads_jsonl(path: Path, payloads: Sequence[Mapping[str, object]]) -> None:
    with open_jsonl_gz(path, "wt") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_payloads_jsonl(path: Path) -> List[Dict[str, object]]:
    payloads: List[Dict[str, object]] = []
    with open_jsonl_gz(path, "rt") as handle:
        for line in handle:
            if line.strip():
                payloads.append(json.loads(line))
    return payloads


# ---------------------------------------------------------------------------
# Materialization stage
# ---------------------------------------------------------------------------


def trace_payload_to_metric_rows(payload: Mapping[str, object]) -> List[Dict[str, object]]:
    base = {
        "trace_id": payload["trace_id"],
        "circuit_key": payload["circuit_key"],
        "algorithm": payload["algorithm"],
        "size": payload["size"],
        "backend": payload["backend"],
        "circuit_kind": payload["circuit_kind"],
        "reference_shots": payload["reference_shots"],
        "source_batch_size": payload["source_batch_size"],
        "sampling_strategy": payload["sampling_strategy"],
        "sampling_seed": payload["sampling_seed"],
        "qsimbench_materialization_time_seconds": payload["qsimbench_materialization_time_seconds"],
    }
    rows: List[Dict[str, object]] = []

    fixed = payload.get("fixed", {})
    if isinstance(fixed, Mapping):
        for budget_str, item in fixed.items():
            if not isinstance(item, Mapping):
                continue
            rows.append(
                {
                    **base,
                    "strategy_family": "fixed",
                    "strategy": f"fixed_{budget_str}",
                    "config_id": "",
                    "shots": int(item["shots"]),
                    "tvd_to_reference": float(item["tvd_to_reference"]),
                    "saving_vs_reference": float(item["saving_vs_reference"]),
                    "stop_reason": "fixed_budget",
                    "last_stability_tvd": float("nan"),
                    "checks_performed": 0,
                    "strategy_eval_time_seconds": float(item.get("eval_time_seconds", float("nan"))),
                }
            )

    stable = payload.get("stable", {})
    if isinstance(stable, Mapping):
        for config_id, item in stable.items():
            if not isinstance(item, Mapping):
                continue
            rows.append(
                {
                    **base,
                    "strategy_family": "stable_shots",
                    "strategy": f"stable_shots_{config_id}",
                    "config_id": config_id,
                    "shots": int(item["shots"]),
                    "tvd_to_reference": float(item["tvd_to_reference"]),
                    "saving_vs_reference": float(item["saving_vs_reference"]),
                    "stop_reason": item["stop_reason"],
                    "last_stability_tvd": float(item.get("last_stability_tvd", float("nan"))),
                    "checks_performed": int(item.get("checks_performed", 0)),
                    "strategy_eval_time_seconds": float(item.get("eval_time_seconds", float("nan"))),
                    "batch_size": item.get("batch_size", ""),
                    "lookback_batches": item.get("lookback_batches", ""),
                    "stability": item.get("stability", ""),
                    "epsilon": item.get("epsilon", ""),
                }
            )
    return rows


def materialize_one_trace(
    spec: TraceSpec,
    configs: Sequence[StableShotsConfig],
    fixed_baselines: Sequence[int],
    reference_shots: int,
    source_batch_size: int,
    sampling_strategy: str,
    sampling_seed: int,
    force_download: bool,
    save_raw_batches: bool,
) -> Dict[str, object]:
    required_shots = max([reference_shots, *fixed_baselines, *(config.max_shots for config in configs)])
    load_start = now_seconds()
    raw_batches = load_qsimbench_batches(
        spec=spec,
        total_shots=required_shots,
        source_batch_size=source_batch_size,
        sampling_strategy=sampling_strategy,
        sampling_seed=sampling_seed,
        force=force_download,
    )
    materialization_time = now_seconds() - load_start

    reference = prefix_counts(raw_batches, reference_shots)
    fixed_payload: Dict[str, object] = {}
    for budget in sorted(int(b) for b in fixed_baselines):
        eval_start = now_seconds()
        counts = prefix_counts(raw_batches, budget)
        eval_time = now_seconds() - eval_start
        fixed_payload[str(budget)] = {
            "shots": budget,
            "counts": counts_to_json(counts),
            "tvd_to_reference": tvd_counts(counts, reference),
            "saving_vs_reference": 1.0 - budget / reference_shots,
            "eval_time_seconds": eval_time,
        }

    stable_payload: Dict[str, object] = {}
    for config in configs:
        eval_start = now_seconds()
        counts, shots, last_delta, stop_reason, checks = run_stable_shots(raw_batches, config)
        eval_time = now_seconds() - eval_start
        stable_payload[config.config_id] = {
            **asdict(config),
            "shots": shots,
            "counts": counts_to_json(counts),
            "tvd_to_reference": tvd_counts(counts, reference),
            "saving_vs_reference": 1.0 - shots / reference_shots,
            "last_stability_tvd": last_delta,
            "stop_reason": stop_reason,
            "checks_performed": checks,
            "eval_time_seconds": eval_time,
        }

    payload: Dict[str, object] = {
        "trace_id": spec.trace_id,
        "circuit_key": spec.circuit_key,
        "algorithm": spec.algorithm,
        "size": spec.size,
        "backend": spec.backend,
        "circuit_kind": spec.circuit_kind,
        "reference_shots": reference_shots,
        "source_batch_size": source_batch_size,
        "sampling_strategy": sampling_strategy,
        "sampling_seed": sampling_seed,
        "required_shots": required_shots,
        "qsimbench_materialization_time_seconds": materialization_time,
        "reference_counts": counts_to_json(reference),
        "fixed": fixed_payload,
        "stable": stable_payload,
        "raw_batches_saved": save_raw_batches,
    }
    if save_raw_batches:
        payload["raw_batches"] = raw_batches_to_json(raw_batches)
    return payload


def run_materialize(args: argparse.Namespace, configs: Sequence[StableShotsConfig]) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "trace_payloads.jsonl.gz"

    algorithms = parse_csv_strings(args.algorithms)
    sizes = parse_csv_ints(args.sizes)
    backends = parse_csv_strings(args.backends)
    fixed_baselines = parse_csv_ints(args.fixed_baselines)

    specs = [
        TraceSpec(algorithm=algorithm, size=size, backend=backend, circuit_kind=args.circuit_kind)
        for algorithm in algorithms
        for size in sizes
        for backend in backends
    ]

    payloads: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    run_start = now_seconds()
    for index, spec in enumerate(specs, start=1):
        print(f"[{index}/{len(specs)}] materialize {spec.trace_id}")
        try:
            payload = materialize_one_trace(
                spec=spec,
                configs=configs,
                fixed_baselines=fixed_baselines,
                reference_shots=args.reference_shots,
                source_batch_size=args.source_batch_size,
                sampling_strategy=args.sampling_strategy,
                sampling_seed=args.sampling_seed,
                force_download=args.force_download,
                save_raw_batches=not args.no_save_raw_batches,
            )
            payloads.append(payload)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append({"trace_id": spec.trace_id, "error": message})
            print(f"  skipped: {message}", file=sys.stderr)

    if not payloads:
        raise RuntimeError("no traces were materialized")

    write_payloads_jsonl(payload_path, payloads)
    trace_rows: List[Dict[str, object]] = []
    for payload in payloads:
        trace_rows.extend(trace_payload_to_metric_rows(payload))
    trace_metrics = pd.DataFrame(trace_rows)
    trace_metrics.to_csv(output_dir / "materialized_trace_metrics.csv", index=False)

    pd.DataFrame(
        [
            {
                "stage": "materialize",
                "payload_path": str(payload_path),
                "trace_count": len(payloads),
                "failed_trace_count": len(failures),
                "run_time_seconds": now_seconds() - run_start,
                "reference_shots": args.reference_shots,
                "source_batch_size": args.source_batch_size,
                "sampling_strategy": args.sampling_strategy,
                "sampling_seed": args.sampling_seed,
                "fixed_baselines": args.fixed_baselines,
                "stable_config_ids": ",".join(config.config_id for config in configs),
                "raw_batches_saved": not args.no_save_raw_batches,
            }
        ]
    ).to_csv(output_dir / "materialization_metadata.csv", index=False)

    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "materialization_failures.csv", index=False)
        if args.fail_on_missing_traces:
            raise RuntimeError(f"{len(failures)} requested traces failed during materialization")

    print(f"Wrote materialized payloads to {payload_path.resolve()}")


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def get_reference_counts(payload: Mapping[str, object]) -> Counts:
    return counts_from_json(payload["reference_counts"])  # type: ignore[arg-type]


def get_raw_batches(payload: Mapping[str, object]) -> List[Batch]:
    trace_id = str(payload.get("trace_id", ""))
    if trace_id in _RAW_BATCH_CACHE:
        return _RAW_BATCH_CACHE[trace_id]
    if "raw_batches" not in payload:
        raise KeyError(
            f"payload {payload.get('trace_id')} does not contain raw_batches; "
            "aggregate_only_sync and hierarchical_local_aggregate require materialization without --no-save-raw-batches"
        )
    batches = raw_batches_from_json(payload["raw_batches"])  # type: ignore[arg-type]
    if trace_id:
        _RAW_BATCH_CACHE[trace_id] = batches
    return batches


def get_stable_entry(payload: Mapping[str, object], config_id: str) -> Mapping[str, object]:
    stable = payload.get("stable", {})
    if not isinstance(stable, Mapping) or config_id not in stable:
        raise KeyError(f"config {config_id!r} not available in payload {payload.get('trace_id')}")
    entry = stable[config_id]
    if not isinstance(entry, Mapping):
        raise TypeError(f"stable entry for {config_id!r} has invalid type")
    return entry


def get_fixed_entry(payload: Mapping[str, object], budget: int) -> Mapping[str, object]:
    fixed = payload.get("fixed", {})
    key = str(int(budget))
    if not isinstance(fixed, Mapping) or key not in fixed:
        raise KeyError(f"fixed budget {budget} not available in payload {payload.get('trace_id')}")
    entry = fixed[key]
    if not isinstance(entry, Mapping):
        raise TypeError(f"fixed entry for {budget} has invalid type")
    return entry


def normalize_weights(raw: Sequence[float]) -> List[float]:
    values = [float(v) for v in raw]
    clean = [v if math.isfinite(v) and v > 0 else 0.0 for v in values]
    total = sum(clean)
    if total <= 0:
        n = len(values)
        if n == 0:
            raise ValueError("cannot normalize empty weights")
        return [1.0 / n] * n
    return [v / total for v in clean]


def stable_weights(entries: Sequence[Mapping[str, object]], scheme: str, confidence_floor: float) -> List[float]:
    if scheme == "uniform":
        return [1.0 / len(entries)] * len(entries)
    if scheme == "shot_proportional":
        return normalize_weights([float(entry["shots"]) for entry in entries])
    if scheme == "confidence":
        raw: List[float] = []
        for entry in entries:
            value = float(entry.get("last_stability_tvd", float("nan")))
            if math.isfinite(value) and value >= 0:
                raw.append(1.0 / max(value, confidence_floor))
            else:
                raw.append(0.0)
        return normalize_weights(raw)
    raise ValueError(f"unsupported weighting scheme {scheme!r}")


def subset_key(backends: Sequence[str]) -> str:
    return "+".join(backends)


def backend_label(backend: str) -> str:
    return backend.replace("fake_", "")


def enumerate_available_subsets(backends: Sequence[str]) -> Iterator[Tuple[str, ...]]:
    sorted_backends = tuple(sorted(backends))
    for k in range(1, len(sorted_backends) + 1):
        for subset in combinations(sorted_backends, k):
            yield subset


def compute_diversity_matrix(payloads: Sequence[Mapping[str, object]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    by_circuit_backend: Dict[Tuple[str, str], Mapping[str, object]] = {}
    backends = sorted({str(p["backend"]) for p in payloads})
    circuit_keys = sorted({str(p["circuit_key"]) for p in payloads})
    for payload in payloads:
        by_circuit_backend[(str(payload["circuit_key"]), str(payload["backend"]))] = payload

    long_rows: List[Dict[str, object]] = []
    matrix = pd.DataFrame(np.nan, index=backends, columns=backends)
    for backend_i in backends:
        for backend_j in backends:
            if backend_i == backend_j:
                value = 0.0
                n = len(circuit_keys)
            else:
                values: List[float] = []
                for circuit_key in circuit_keys:
                    pi = by_circuit_backend.get((circuit_key, backend_i))
                    pj = by_circuit_backend.get((circuit_key, backend_j))
                    if pi is None or pj is None:
                        continue
                    values.append(tvd_counts(get_reference_counts(pi), get_reference_counts(pj)))
                value = float(np.mean(values)) if values else float("nan")
                n = len(values)
            matrix.loc[backend_i, backend_j] = value
            long_rows.append(
                {
                    "backend_i": backend_i,
                    "backend_j": backend_j,
                    "backend_i_label": backend_label(backend_i),
                    "backend_j_label": backend_label(backend_j),
                    "mean_reference_tvd": value,
                    "circuit_count": n,
                }
            )
    return matrix, pd.DataFrame(long_rows)


def subset_diversity_global(backends: Sequence[str], diversity_matrix: pd.DataFrame) -> float:
    if len(backends) < 2:
        return float("nan")
    values = []
    for b1, b2 in combinations(backends, 2):
        values.append(float(diversity_matrix.loc[b1, b2]))
    return float(np.mean(values)) if values else float("nan")


def subset_diversity_on_circuit(payloads: Sequence[Mapping[str, object]]) -> float:
    if len(payloads) < 2:
        return float("nan")
    values = []
    for p1, p2 in combinations(payloads, 2):
        values.append(tvd_counts(get_reference_counts(p1), get_reference_counts(p2)))
    return float(np.mean(values)) if values else float("nan")


# ---------------------------------------------------------------------------
# New online aggregate policies
# ---------------------------------------------------------------------------


def uniform_subset_weights(n: int) -> List[float]:
    if n <= 0:
        raise ValueError("n must be positive")
    return [1.0 / n] * n


def current_aggregate_distribution(states: Sequence[LocalState], weights: Sequence[float]) -> Dict[str, float]:
    if len(states) != len(weights):
        raise ValueError("states and weights must have equal length")
    out: Dict[str, float] = {}
    for state, weight in zip(states, weights):
        if state.shots <= 0:
            return {}
        scale = float(weight) / float(state.shots)
        for bitstring, count in state.cumulative.items():
            out[bitstring] = out.get(bitstring, 0.0) + float(count) * scale
    return out


def update_aggregate_state_with_distribution(
    agg_state: AggregateState,
    aggregate_dist: Mapping[str, float],
    config: StableShotsConfig,
    round_index: int,
) -> None:
    agg_state.snapshots.append(dict(aggregate_dist))
    if len(agg_state.snapshots) > config.lookback_batches:
        previous = agg_state.snapshots[-1 - config.lookback_batches]
        agg_state.last_delta = tvd_probs(aggregate_dist, previous)
        agg_state.checks_performed += 1
        agg_state.stable_checks = agg_state.stable_checks + 1 if agg_state.last_delta <= config.epsilon else 0
        if agg_state.stable_checks >= config.stability:
            agg_state.stopped = True
            agg_state.stop_reason = "aggregate_stable"
            agg_state.stop_round = round_index


def initialize_local_states(backends: Sequence[str]) -> Dict[str, LocalState]:
    return {
        backend: LocalState(
            backend=backend,
            cumulative=Counter(),
            snapshots=[],
            stable_checks=0,
            shots=0,
            last_delta=float("nan"),
            checks_performed=0,
            stopped=False,
            stop_reason="running",
            stop_round=None,
        )
        for backend in backends
    }


def initialize_aggregate_state() -> AggregateState:
    return AggregateState(
        snapshots=[],
        stable_checks=0,
        last_delta=float("nan"),
        checks_performed=0,
        stopped=False,
        stop_reason="running",
        stop_round=None,
    )


def make_prefix_batches_by_backend(
    subset_payloads: Sequence[Mapping[str, object]],
    config: StableShotsConfig,
) -> Dict[str, List[Batch]]:
    out: Dict[str, List[Batch]] = {}
    for payload in subset_payloads:
        backend = str(payload["backend"])
        trace_id = str(payload.get("trace_id", backend))
        cache_key = (trace_id, int(config.batch_size), int(config.max_shots))
        if cache_key not in _PREFIX_BATCH_CACHE:
            raw = get_raw_batches(payload)
            _PREFIX_BATCH_CACHE[cache_key] = make_batch_prefixes(raw, config.batch_size, config.max_shots)
        out[backend] = _PREFIX_BATCH_CACHE[cache_key]
    return out


def simulate_aggregate_only_sync(
    subset_payloads: Sequence[Mapping[str, object]],
    config: StableShotsConfig,
) -> Dict[str, object]:
    backends = [str(p["backend"]) for p in subset_payloads]
    batch_lists = make_prefix_batches_by_backend(subset_payloads, config)
    states_by_backend = initialize_local_states(backends)
    agg_state = initialize_aggregate_state()
    weights = uniform_subset_weights(len(backends))
    max_rounds = min(len(batch_lists[b]) for b in backends)
    aggregate_dist: Dict[str, float] = {}

    for round_index in range(1, max_rounds + 1):
        for backend in backends:
            batch_shots, batch_counts = batch_lists[backend][round_index - 1]
            state = states_by_backend[backend]
            merge_counts(state.cumulative, batch_counts)
            state.shots += int(batch_shots)
            state.snapshots.append(Counter(state.cumulative))
            if state.shots >= config.max_shots:
                state.stopped = True
                state.stop_reason = "max_budget"
                state.stop_round = round_index
        states = [states_by_backend[b] for b in backends]
        aggregate_dist = current_aggregate_distribution(states, weights)
        update_aggregate_state_with_distribution(agg_state, aggregate_dist, config, round_index)
        if agg_state.stopped:
            break

    if not agg_state.stopped:
        agg_state.stopped = True
        agg_state.stop_reason = "max_budget"
        agg_state.stop_round = max_rounds

    states = [states_by_backend[b] for b in backends]
    final_counts = [state.cumulative for state in states]
    return {
        "policy": "aggregate_only_sync",
        "online_weight_scheme": "uniform_fixed",
        "weights": weights,
        "aggregate_dist": aggregate_dist if aggregate_dist else current_aggregate_distribution(states, weights),
        "final_counts": final_counts,
        "per_backend_shots": {state.backend: state.shots for state in states},
        "per_backend_stop_reason": {state.backend: state.stop_reason for state in states},
        "per_backend_stop_round": {state.backend: state.stop_round for state in states},
        "per_backend_last_stability_tvd": {state.backend: state.last_delta for state in states},
        "per_backend_checks_performed": {state.backend: state.checks_performed for state in states},
        "aggregate_stop_reason": agg_state.stop_reason,
        "aggregate_stop_round": agg_state.stop_round,
        "aggregate_last_stability_tvd": agg_state.last_delta,
        "aggregate_checks_performed": agg_state.checks_performed,
        "aggregate_stable_checks": agg_state.stable_checks,
        "local_stopped_before_global_count": 0,
        "local_stopped_before_global_rate": 0.0,
        "active_backend_count_final": len(backends),
    }


def simulate_hierarchical_local_aggregate(
    subset_payloads: Sequence[Mapping[str, object]],
    config: StableShotsConfig,
) -> Dict[str, object]:
    backends = [str(p["backend"]) for p in subset_payloads]
    batch_lists = make_prefix_batches_by_backend(subset_payloads, config)
    states_by_backend = initialize_local_states(backends)
    agg_state = initialize_aggregate_state()
    weights = uniform_subset_weights(len(backends))
    max_rounds = max(len(batch_lists[b]) for b in backends)
    aggregate_dist: Dict[str, float] = {}
    active_counts_by_round: List[int] = []

    for round_index in range(1, max_rounds + 1):
        active_before = [b for b in backends if not states_by_backend[b].stopped]
        if not active_before:
            break
        active_counts_by_round.append(len(active_before))
        for backend in active_before:
            batches = batch_lists[backend]
            state = states_by_backend[backend]
            if round_index <= len(batches):
                batch_shots, batch_counts = batches[round_index - 1]
                update_local_state_with_batch(state, batch_counts, batch_shots, config, round_index)
            else:
                state.stopped = True
                state.stop_reason = "max_budget"
                state.stop_round = round_index - 1
        states = [states_by_backend[b] for b in backends]
        if any(state.shots <= 0 for state in states):
            continue
        aggregate_dist = current_aggregate_distribution(states, weights)
        update_aggregate_state_with_distribution(agg_state, aggregate_dist, config, round_index)
        if agg_state.stopped:
            break

    states = [states_by_backend[b] for b in backends]
    if not agg_state.stopped:
        if all(state.stopped for state in states):
            agg_state.stopped = True
            agg_state.stop_reason = "all_backends_inactive"
            agg_state.stop_round = max(state.stop_round or 0 for state in states)
        else:
            agg_state.stopped = True
            agg_state.stop_reason = "max_budget"
            agg_state.stop_round = max_rounds
            for state in states:
                if not state.stopped:
                    state.stopped = True
                    state.stop_reason = "max_budget"
                    state.stop_round = max_rounds

    global_round = int(agg_state.stop_round or 0)
    local_before = sum(1 for state in states if state.stop_round is not None and state.stop_round < global_round)
    final_counts = [state.cumulative for state in states]
    return {
        "policy": "hierarchical_local_aggregate",
        "online_weight_scheme": "uniform_fixed",
        "weights": weights,
        "aggregate_dist": aggregate_dist if aggregate_dist else current_aggregate_distribution(states, weights),
        "final_counts": final_counts,
        "per_backend_shots": {state.backend: state.shots for state in states},
        "per_backend_stop_reason": {state.backend: state.stop_reason for state in states},
        "per_backend_stop_round": {state.backend: state.stop_round for state in states},
        "per_backend_last_stability_tvd": {state.backend: state.last_delta for state in states},
        "per_backend_checks_performed": {state.backend: state.checks_performed for state in states},
        "aggregate_stop_reason": agg_state.stop_reason,
        "aggregate_stop_round": agg_state.stop_round,
        "aggregate_last_stability_tvd": agg_state.last_delta,
        "aggregate_checks_performed": agg_state.checks_performed,
        "aggregate_stable_checks": agg_state.stable_checks,
        "local_stopped_before_global_count": local_before,
        "local_stopped_before_global_rate": local_before / len(states) if states else float("nan"),
        "active_backend_count_final": sum(1 for state in states if not state.stopped),
        "active_backend_count_by_round": active_counts_by_round,
    }


# ---------------------------------------------------------------------------
# Aggregation policies
# ---------------------------------------------------------------------------


def get_or_compute_stable_entry(payload: Mapping[str, object], config: StableShotsConfig) -> Mapping[str, object]:
    """Return local StableShots result for payload/config, computing it from raw batches if needed.

    Materialization may intentionally store only one selected StableShots profile to keep
    the first phase cheap. The grid-search phase must still be able to evaluate arbitrary
    StableShots configurations; when a config is absent from payload["stable"], compute it
    lazily from raw_batches and cache it process-locally.
    """
    stable = payload.get("stable", {})
    if isinstance(stable, Mapping) and config.config_id in stable:
        entry = stable[config.config_id]
        if not isinstance(entry, Mapping):
            raise TypeError(f"stable entry for {config.config_id!r} has invalid type")
        return entry

    trace_id = str(payload.get("trace_id", payload.get("backend", "unknown")))
    key = (trace_id, config.config_id)
    if key not in _LOCAL_STABLE_CACHE:
        raw = get_raw_batches(payload)
        counts, shots, last_delta, stop_reason, checks = run_stable_shots(raw, config)
        _LOCAL_STABLE_CACHE[key] = {
            **asdict(config),
            "shots": int(shots),
            "counts": counts_to_json(counts),
            "tvd_to_reference": tvd_counts(counts, get_reference_counts(payload)),
            "saving_vs_reference": 1.0 - int(shots) / int(payload["reference_shots"]),
            "last_stability_tvd": float(last_delta),
            "stop_reason": stop_reason,
            "checks_performed": int(checks),
            "computed_in_grid_search": True,
        }
    return _LOCAL_STABLE_CACHE[key]


def aggregate_local_only_payloads(
    subset_payloads: Sequence[Mapping[str, object]],
    config: StableShotsConfig,
    weighting_schemes: Sequence[str],
    confidence_floor: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    backends = [str(p["backend"]) for p in subset_payloads]
    stable_entries = [get_or_compute_stable_entry(p, config) for p in subset_payloads]
    stable_counts = [counts_from_json(e["counts"]) for e in stable_entries]  # type: ignore[arg-type]
    reference_counts = [get_reference_counts(p) for p in subset_payloads]
    reference_shots = int(subset_payloads[0]["reference_shots"])
    shots_used = [int(entry["shots"]) for entry in stable_entries]
    last_deltas = [float(entry.get("last_stability_tvd", float("nan"))) for entry in stable_entries]
    stop_reasons = [str(entry.get("stop_reason", "")) for entry in stable_entries]
    checks_performed = [int(entry.get("checks_performed", 0)) for entry in stable_entries]

    for scheme in weighting_schemes:
        weights = stable_weights(stable_entries, scheme, confidence_floor)
        adaptive_dist = weighted_distribution(stable_counts, weights)
        reference_dist = weighted_distribution(reference_counts, weights)
        rows.append(
            {
                "policy": "local_only_offline_aggregate",
                "scheme": scheme,
                "online_weight_scheme": "not_applicable",
                "weights": weights,
                "aggregate_dist": adaptive_dist,
                "reference_dist": reference_dist,
                "per_backend_shots": dict(zip(backends, shots_used)),
                "per_backend_last_stability_tvd": dict(zip(backends, last_deltas)),
                "per_backend_stop_reason": dict(zip(backends, stop_reasons)),
                "per_backend_checks_performed": dict(zip(backends, checks_performed)),
                "per_backend_stop_round": {},
                "aggregate_stop_reason": "not_applicable",
                "aggregate_stop_round": None,
                "aggregate_last_stability_tvd": float("nan"),
                "aggregate_checks_performed": 0,
                "aggregate_stable_checks": 0,
                "local_stopped_before_global_count": 0,
                "local_stopped_before_global_rate": 0.0,
                "active_backend_count_final": 0,
                "total_reference_shots": len(backends) * reference_shots,
                "total_adaptive_shots": int(sum(shots_used)),
                "max_backend_shots": int(max(shots_used)) if shots_used else 0,
                "tvd": tvd_probs(adaptive_dist, reference_dist),
            }
        )
    return rows


def final_policy_weights(
    result: Mapping[str, object],
    scheme: str,
    confidence_floor: float,
) -> List[float]:
    per_backend_shots = result.get("per_backend_shots", {})
    if not isinstance(per_backend_shots, Mapping):
        raise TypeError("per_backend_shots must be a mapping")
    backends = list(per_backend_shots.keys())
    if scheme == "uniform":
        return uniform_subset_weights(len(backends))
    if scheme == "shot_proportional":
        return normalize_weights([float(per_backend_shots[b]) for b in backends])
    if scheme == "confidence":
        deltas = result.get("per_backend_last_stability_tvd", {})
        if not isinstance(deltas, Mapping):
            return uniform_subset_weights(len(backends))
        raw: List[float] = []
        for backend in backends:
            value = float(deltas.get(backend, float("nan")))
            raw.append(1.0 / max(value, confidence_floor) if math.isfinite(value) and value >= 0 else 0.0)
        return normalize_weights(raw)
    raise ValueError(f"unsupported weighting scheme {scheme!r}")


def aggregate_new_policy_payloads(
    subset_payloads: Sequence[Mapping[str, object]],
    config: StableShotsConfig,
    policy: str,
    weighting_schemes: Sequence[str],
    confidence_floor: float,
) -> List[Dict[str, object]]:
    if policy == "aggregate_only_sync":
        base_result = simulate_aggregate_only_sync(subset_payloads, config)
    elif policy == "hierarchical_local_aggregate":
        base_result = simulate_hierarchical_local_aggregate(subset_payloads, config)
    else:
        raise ValueError(f"unsupported new policy {policy!r}")

    reference_counts = [get_reference_counts(p) for p in subset_payloads]
    reference_shots = int(subset_payloads[0]["reference_shots"])
    final_counts = base_result.get("final_counts")
    if not isinstance(final_counts, Sequence):
        raise TypeError("new policy result must contain final_counts")
    per_backend_shots = base_result["per_backend_shots"]
    results: List[Dict[str, object]] = []
    for scheme in weighting_schemes:
        result = dict(base_result)
        weights = final_policy_weights(result, scheme, confidence_floor)
        aggregate_dist = weighted_distribution(final_counts, weights)  # type: ignore[arg-type]
        reference_dist = weighted_distribution(reference_counts, weights)
        result["scheme"] = scheme
        result["weights"] = weights
        result["aggregate_dist"] = aggregate_dist
        result["reference_dist"] = reference_dist
        result["tvd"] = tvd_probs(aggregate_dist, reference_dist)
        result["total_reference_shots"] = len(subset_payloads) * reference_shots
        result["total_adaptive_shots"] = int(sum(int(v) for v in per_backend_shots.values()))  # type: ignore[union-attr]
        result["max_backend_shots"] = int(max(int(v) for v in per_backend_shots.values()))  # type: ignore[union-attr]
        results.append(result)
    return results


def common_policy_row_fields(
    subset_payloads: Sequence[Mapping[str, object]],
    subset: Sequence[str],
    config_id: str,
    diversity_matrix: pd.DataFrame,
) -> Dict[str, object]:
    k = len(subset)
    return {
        "circuit_key": subset_payloads[0]["circuit_key"],
        "algorithm": subset_payloads[0]["algorithm"],
        "size": int(subset_payloads[0]["size"]),
        "circuit_kind": subset_payloads[0]["circuit_kind"],
        "config_id": config_id,
        "k": k,
        "subset_key": subset_key(subset),
        "backends": json.dumps(list(subset)),
        "backend_labels": json.dumps([backend_label(b) for b in subset]),
        "diversity_global": subset_diversity_global(subset, diversity_matrix),
        "diversity_on_circuit": subset_diversity_on_circuit(subset_payloads),
    }


def policy_result_to_row(
    result: Mapping[str, object],
    base: Mapping[str, object],
) -> Dict[str, object]:
    reference_total = int(result["total_reference_shots"])
    adaptive_total = int(result["total_adaptive_shots"])
    max_backend_shots = int(result["max_backend_shots"])
    return {
        **base,
        "policy": result["policy"],
        "scheme": result["scheme"],
        "online_weight_scheme": result.get("online_weight_scheme", ""),
        "weights": json.dumps(result["weights"]),
        "per_backend_shots": json.dumps(result["per_backend_shots"]),
        "per_backend_last_stability_tvd": json.dumps(result.get("per_backend_last_stability_tvd", {})),
        "per_backend_stop_reason": json.dumps(result.get("per_backend_stop_reason", {})),
        "per_backend_stop_round": json.dumps(result.get("per_backend_stop_round", {})),
        "per_backend_checks_performed": json.dumps(result.get("per_backend_checks_performed", {})),
        "adaptive_shots_total": adaptive_total,
        "reference_shots_total": reference_total,
        "shots_saved": reference_total - adaptive_total,
        "ssr": 1.0 - adaptive_total / reference_total,
        "max_backend_shots": max_backend_shots,
        "parallel_ssr": 1.0 - max_backend_shots / (reference_total / int(base["k"])),
        "tvd": float(result["tvd"]),
        "aggregate_stop_reason": result.get("aggregate_stop_reason", ""),
        "aggregate_stop_round": result.get("aggregate_stop_round", ""),
        "aggregate_last_stability_tvd": result.get("aggregate_last_stability_tvd", float("nan")),
        "aggregate_checks_performed": result.get("aggregate_checks_performed", 0),
        "aggregate_stable_checks": result.get("aggregate_stable_checks", 0),
        "local_stopped_before_global_count": result.get("local_stopped_before_global_count", 0),
        "local_stopped_before_global_rate": result.get("local_stopped_before_global_rate", 0.0),
        "active_backend_count_final": result.get("active_backend_count_final", 0),
        "active_backend_count_by_round": json.dumps(result.get("active_backend_count_by_round", [])),
    }


def aggregate_policy_payloads(
    payloads: Sequence[Mapping[str, object]],
    config: StableShotsConfig,
    weighting_schemes: Sequence[str],
    policies: Sequence[str],
    confidence_floor: float,
    diversity_matrix: pd.DataFrame,
) -> pd.DataFrame:
    by_circuit: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for payload in payloads:
        by_circuit.setdefault(str(payload["circuit_key"]), {})[str(payload["backend"])] = payload

    rows: List[Dict[str, object]] = []
    requested_policies = set(policies)
    for circuit_key, backend_payloads in sorted(by_circuit.items()):
        backends = sorted(backend_payloads)
        for subset in enumerate_available_subsets(backends):
            subset_payloads = [backend_payloads[b] for b in subset]
            base = common_policy_row_fields(subset_payloads, subset, config.config_id, diversity_matrix)
            if "local_only_offline_aggregate" in requested_policies:
                for result in aggregate_local_only_payloads(
                    subset_payloads,
                    config,
                    weighting_schemes,
                    confidence_floor,
                ):
                    rows.append(policy_result_to_row(result, base))
            for policy in ["aggregate_only_sync", "hierarchical_local_aggregate"]:
                if policy in requested_policies:
                    for result in aggregate_new_policy_payloads(
                        subset_payloads,
                        config,
                        policy,
                        weighting_schemes,
                        confidence_floor,
                    ):
                        rows.append(policy_result_to_row(result, base))
    return pd.DataFrame(rows)


def aggregate_stable_payloads_backward_compatible(policy_df: pd.DataFrame) -> pd.DataFrame:
    if policy_df.empty:
        return pd.DataFrame()
    cols = [
        "circuit_key",
        "algorithm",
        "size",
        "circuit_kind",
        "config_id",
        "scheme",
        "k",
        "subset_key",
        "backends",
        "backend_labels",
        "weights",
        "per_backend_shots",
        "per_backend_last_stability_tvd",
        "per_backend_stop_reason",
        "per_backend_checks_performed",
        "adaptive_shots_total",
        "reference_shots_total",
        "shots_saved",
        "ssr",
        "parallel_ssr",
        "tvd",
        "diversity_global",
        "diversity_on_circuit",
    ]
    out = policy_df[policy_df["policy"] == "local_only_offline_aggregate"].copy()
    return out[[c for c in cols if c in out.columns]]


def aggregate_fixed_payloads(
    payloads: Sequence[Mapping[str, object]],
    fixed_baselines: Sequence[int],
    diversity_matrix: pd.DataFrame,
) -> pd.DataFrame:
    by_circuit: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for payload in payloads:
        by_circuit.setdefault(str(payload["circuit_key"]), {})[str(payload["backend"])] = payload

    rows: List[Dict[str, object]] = []
    for circuit_key, backend_payloads in sorted(by_circuit.items()):
        backends = sorted(backend_payloads)
        for subset in enumerate_available_subsets(backends):
            subset_payloads = [backend_payloads[b] for b in subset]
            reference_counts = [get_reference_counts(p) for p in subset_payloads]
            reference_shots = int(subset_payloads[0]["reference_shots"])
            k = len(subset)
            weights = [1.0 / k] * k
            for budget in sorted(int(b) for b in fixed_baselines):
                fixed_entries = [get_fixed_entry(p, budget) for p in subset_payloads]
                fixed_counts = [counts_from_json(e["counts"]) for e in fixed_entries]  # type: ignore[arg-type]
                aggregate_dist = weighted_distribution(fixed_counts, weights)
                reference_dist = weighted_distribution(reference_counts, weights)
                total_fixed_shots = budget * k
                total_reference_shots = reference_shots * k
                rows.append(
                    {
                        "circuit_key": circuit_key,
                        "algorithm": subset_payloads[0]["algorithm"],
                        "size": int(subset_payloads[0]["size"]),
                        "circuit_kind": subset_payloads[0]["circuit_kind"],
                        "strategy": f"fixed_{budget}_uniform",
                        "fixed_budget": budget,
                        "scheme": "uniform",
                        "k": k,
                        "subset_key": subset_key(subset),
                        "backends": json.dumps(list(subset)),
                        "backend_labels": json.dumps([backend_label(b) for b in subset]),
                        "weights": json.dumps(weights),
                        "shots_total": total_fixed_shots,
                        "reference_shots_total": total_reference_shots,
                        "shots_saved": total_reference_shots - total_fixed_shots,
                        "ssr": 1.0 - total_fixed_shots / total_reference_shots,
                        "parallel_ssr": 1.0 - budget / reference_shots,
                        "tvd": tvd_probs(aggregate_dist, reference_dist),
                        "diversity_global": subset_diversity_global(subset, diversity_matrix),
                        "diversity_on_circuit": subset_diversity_on_circuit(subset_payloads),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def bootstrap_median_ci(values: Sequence[float], reps: int, seed: int, alpha: float = 0.05) -> Tuple[float, float]:
    clean = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if clean.size == 0:
        return float("nan"), float("nan")
    if clean.size == 1 or reps <= 0:
        med = float(np.median(clean))
        return med, med
    rng = np.random.default_rng(seed)
    boot = np.empty(reps, dtype=float)
    for i in range(reps):
        sample = rng.choice(clean, size=clean.size, replace=True)
        boot[i] = np.median(sample)
    return (float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2)))


def summarize_policy_by_k(df: pd.DataFrame, bootstrap_reps: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    group_cols = ["policy", "scheme", "k"]
    for (policy, scheme, k), group in df.groupby(group_cols, sort=True):
        tvd_ci = bootstrap_median_ci(group["tvd"].values, bootstrap_reps, deterministic_seed(seed, policy, scheme, k, "tvd"))
        ssr_ci = bootstrap_median_ci(group["ssr"].values, bootstrap_reps, deterministic_seed(seed, policy, scheme, k, "ssr"))
        pssr_ci = bootstrap_median_ci(group["parallel_ssr"].values, bootstrap_reps, deterministic_seed(seed, policy, scheme, k, "parallel_ssr"))
        q25_tvd = float(group["tvd"].quantile(0.25))
        q75_tvd = float(group["tvd"].quantile(0.75))
        q25_ssr = float(group["ssr"].quantile(0.25))
        q75_ssr = float(group["ssr"].quantile(0.75))
        rows.append(
            {
                "policy": policy,
                "scheme": scheme,
                "k": int(k),
                "experiments": int(len(group)),
                "circuit_configs": int(group["circuit_key"].nunique()),
                "subsets": int(group["subset_key"].nunique()),
                "median_tvd": float(group["tvd"].median()),
                "q25_tvd": q25_tvd,
                "q75_tvd": q75_tvd,
                "iqr_tvd": q75_tvd - q25_tvd,
                "mean_tvd": float(group["tvd"].mean()),
                "max_tvd": float(group["tvd"].max()),
                "median_ssr": float(group["ssr"].median()),
                "q25_ssr": q25_ssr,
                "q75_ssr": q75_ssr,
                "iqr_ssr": q75_ssr - q25_ssr,
                "mean_ssr": float(group["ssr"].mean()),
                "median_parallel_ssr": float(group["parallel_ssr"].median()),
                "mean_parallel_ssr": float(group["parallel_ssr"].mean()),
                "median_tvd_ci_low": tvd_ci[0],
                "median_tvd_ci_high": tvd_ci[1],
                "median_ssr_ci_low": ssr_ci[0],
                "median_ssr_ci_high": ssr_ci[1],
                "median_parallel_ssr_ci_low": pssr_ci[0],
                "median_parallel_ssr_ci_high": pssr_ci[1],
            }
        )
    return pd.DataFrame(rows)


def circuit_level_policy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(["policy", "scheme", "k", "circuit_key", "algorithm", "size", "circuit_kind"], sort=True)
        .agg(
            circuit_median_tvd=("tvd", "median"),
            circuit_mean_tvd=("tvd", "mean"),
            circuit_max_tvd=("tvd", "max"),
            circuit_median_ssr=("ssr", "median"),
            circuit_mean_ssr=("ssr", "mean"),
            circuit_median_parallel_ssr=("parallel_ssr", "median"),
            circuit_mean_parallel_ssr=("parallel_ssr", "mean"),
            subset_rows=("subset_key", "count"),
            subsets=("subset_key", "nunique"),
        )
        .reset_index()
    )


def summarize_policy_clustered_by_circuit(
    df: pd.DataFrame,
    bootstrap_reps: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    circuit_df = circuit_level_policy_metrics(df)
    if circuit_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows: List[Dict[str, object]] = []
    row_groups = df.groupby(["policy", "scheme", "k"], sort=True)
    for (policy, scheme, k), group in circuit_df.groupby(["policy", "scheme", "k"], sort=True):
        raw_group = row_groups.get_group((policy, scheme, k))
        tvd_values = group["circuit_median_tvd"].values
        ssr_values = group["circuit_median_ssr"].values
        pssr_values = group["circuit_median_parallel_ssr"].values
        tvd_ci = bootstrap_median_ci(tvd_values, bootstrap_reps, deterministic_seed(seed, "cluster", policy, scheme, k, "tvd"))
        ssr_ci = bootstrap_median_ci(ssr_values, bootstrap_reps, deterministic_seed(seed, "cluster", policy, scheme, k, "ssr"))
        pssr_ci = bootstrap_median_ci(pssr_values, bootstrap_reps, deterministic_seed(seed, "cluster", policy, scheme, k, "parallel_ssr"))
        subset_counts = group["subsets"].astype(int)
        rows.append(
            {
                "policy": policy,
                "scheme": scheme,
                "k": int(k),
                "bootstrap_unit": "circuit_key",
                "bootstrap_reps": int(bootstrap_reps),
                "row_experiments": int(len(raw_group)),
                "circuit_configs": int(group["circuit_key"].nunique()),
                "subset_rows_per_circuit_min": int(subset_counts.min()),
                "subset_rows_per_circuit_median": float(subset_counts.median()),
                "subset_rows_per_circuit_max": int(subset_counts.max()),
                "row_level_median_tvd": float(raw_group["tvd"].median()),
                "median_tvd": float(np.median(tvd_values)),
                "median_tvd_ci_low": tvd_ci[0],
                "median_tvd_ci_high": tvd_ci[1],
                "q25_tvd": float(group["circuit_median_tvd"].quantile(0.25)),
                "q75_tvd": float(group["circuit_median_tvd"].quantile(0.75)),
                "row_level_median_ssr": float(raw_group["ssr"].median()),
                "median_ssr": float(np.median(ssr_values)),
                "median_ssr_ci_low": ssr_ci[0],
                "median_ssr_ci_high": ssr_ci[1],
                "q25_ssr": float(group["circuit_median_ssr"].quantile(0.25)),
                "q75_ssr": float(group["circuit_median_ssr"].quantile(0.75)),
                "row_level_median_parallel_ssr": float(raw_group["parallel_ssr"].median()),
                "median_parallel_ssr": float(np.median(pssr_values)),
                "median_parallel_ssr_ci_low": pssr_ci[0],
                "median_parallel_ssr_ci_high": pssr_ci[1],
                "non_independence_correction": (
                    "Subsets are collapsed within each circuit before bootstrapping; "
                    "CIs resample circuit_key-level medians, not overlapping subset rows."
                ),
            }
        )
    return pd.DataFrame(rows), circuit_df


def summarize_fixed_by_k(df: pd.DataFrame, bootstrap_reps: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for (budget, k), group in df.groupby(["fixed_budget", "k"], sort=True):
        tvd_ci = bootstrap_median_ci(group["tvd"].values, bootstrap_reps, deterministic_seed(seed, "fixed", budget, k, "tvd"))
        q25_tvd = float(group["tvd"].quantile(0.25))
        q75_tvd = float(group["tvd"].quantile(0.75))
        rows.append(
            {
                "fixed_budget": int(budget),
                "k": int(k),
                "experiments": int(len(group)),
                "circuit_configs": int(group["circuit_key"].nunique()),
                "subsets": int(group["subset_key"].nunique()),
                "median_tvd": float(group["tvd"].median()),
                "q25_tvd": q25_tvd,
                "q75_tvd": q75_tvd,
                "iqr_tvd": q75_tvd - q25_tvd,
                "mean_tvd": float(group["tvd"].mean()),
                "max_tvd": float(group["tvd"].max()),
                "median_ssr": float(group["ssr"].median()),
                "mean_ssr": float(group["ssr"].mean()),
                "median_parallel_ssr": float(group["parallel_ssr"].median()) if "parallel_ssr" in group else float("nan"),
                "median_tvd_ci_low": tvd_ci[0],
                "median_tvd_ci_high": tvd_ci[1],
            }
        )
    return pd.DataFrame(rows)


def rank_corr_spearman(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean) < 3:
        return float("nan"), float("nan")
    try:
        from scipy.stats import spearmanr  # type: ignore

        result = spearmanr(clean["x"].values, clean["y"].values)
        return float(result.statistic), float(result.pvalue)
    except Exception:
        return float(clean["x"].rank().corr(clean["y"].rank())), float("nan")


def complementarity_analysis(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    subset_level = (
        df.groupby(["policy", "scheme", "k", "subset_key", "backends", "backend_labels"], sort=True)
        .agg(
            median_tvd=("tvd", "median"),
            mean_tvd=("tvd", "mean"),
            max_tvd=("tvd", "max"),
            median_ssr=("ssr", "median"),
            mean_ssr=("ssr", "mean"),
            median_parallel_ssr=("parallel_ssr", "median"),
            mean_parallel_ssr=("parallel_ssr", "mean"),
            diversity_global=("diversity_global", "first"),
            median_diversity_on_circuit=("diversity_on_circuit", "median"),
            circuit_configs=("circuit_key", "nunique"),
        )
        .reset_index()
    )

    corr_rows: List[Dict[str, object]] = []
    for (policy, scheme, k), group in subset_level.groupby(["policy", "scheme", "k"], sort=True):
        if int(k) < 2 or len(group) < 3:
            continue
        rho_global, p_global = rank_corr_spearman(group["diversity_global"], group["median_tvd"])
        rho_local, p_local = rank_corr_spearman(group["median_diversity_on_circuit"], group["median_tvd"])
        corr_rows.append(
            {
                "policy": policy,
                "scheme": scheme,
                "k": int(k),
                "subsets": int(len(group)),
                "spearman_rho_diversity_global_vs_median_tvd": rho_global,
                "spearman_p_diversity_global_vs_median_tvd": p_global,
                "spearman_rho_diversity_on_circuit_vs_median_tvd": rho_local,
                "spearman_p_diversity_on_circuit_vs_median_tvd": p_local,
                "interpretation": "negative rho is consistent with complementarity/noise averaging",
            }
        )
    corr_df = pd.DataFrame(corr_rows)

    rank_frames: List[pd.DataFrame] = []
    for (policy, scheme, k), group in subset_level.groupby(["policy", "scheme", "k"], sort=True):
        best = group.sort_values(["median_tvd", "mean_tvd"], ascending=[True, True]).head(5).copy()
        best["rank_type"] = "best_low_tvd"
        best["rank"] = range(1, len(best) + 1)
        worst = group.sort_values(["median_tvd", "mean_tvd"], ascending=[False, False]).head(5).copy()
        worst["rank_type"] = "worst_high_tvd"
        worst["rank"] = range(1, len(worst) + 1)
        rank_frames.extend([best, worst])
    rank_df = pd.concat(rank_frames, ignore_index=True) if rank_frames else pd.DataFrame()
    return subset_level, corr_df, rank_df


def policy_tvd_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    table = df.groupby(["policy", "scheme", "k"])["tvd"].median().reset_index()
    return table.pivot_table(index="k", columns=["policy", "scheme"], values="tvd").reset_index()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def make_aggregate_plots(
    policy_df: pd.DataFrame,
    diversity_matrix: pd.DataFrame,
    subset_level: pd.DataFrame,
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"could not import matplotlib; skipping plots: {exc}", file=sys.stderr)
        return

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not policy_df.empty:
        summary, _ = summarize_policy_clustered_by_circuit(policy_df, bootstrap_reps=0, seed=0)
        for metric, ylabel, filename_metric in [
            ("median_tvd", "Circuit-clustered median TVD", "clustered_median_tvd"),
            ("median_ssr", "Circuit-clustered median SSR", "clustered_median_ssr"),
            ("median_parallel_ssr", "Circuit-clustered median parallel SSR", "clustered_median_parallel_ssr"),
        ]:
            plt.figure(figsize=(8, 5))
            for (policy, scheme), group in summary.groupby(["policy", "scheme"], sort=True):
                group = group.sort_values("k")
                label = f"{policy}:{scheme}"
                plt.plot(group["k"], group[metric], marker="o", label=label)
            plt.xlabel("Number of backends k")
            plt.ylabel(ylabel)
            plt.title(ylabel + " by policy")
            plt.xticks(sorted(policy_df["k"].unique()))
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(plot_dir / f"{filename_metric}_by_policy.png", dpi=200)
            plt.close()

    if not diversity_matrix.empty:
        plt.figure(figsize=(6, 5))
        plt.imshow(diversity_matrix.values, aspect="auto")
        plt.colorbar(label="Mean reference TVD")
        labels = [backend_label(str(v)) for v in diversity_matrix.index]
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
        plt.title("Backend diversity matrix")
        plt.tight_layout()
        plt.savefig(plot_dir / "backend_diversity_matrix.png", dpi=200)
        plt.close()

    if not subset_level.empty:
        for (policy, scheme), group_scheme in subset_level.groupby(["policy", "scheme"], sort=True):
            for k, group in group_scheme.groupby("k", sort=True):
                if int(k) < 2 or len(group) < 2:
                    continue
                plt.figure(figsize=(6, 4))
                plt.scatter(group["diversity_global"], group["median_tvd"], s=32)
                for _, row in group.iterrows():
                    labels = json.loads(row["backend_labels"])
                    label = "+".join(labels)
                    plt.annotate(label, (row["diversity_global"], row["median_tvd"]), fontsize=6)
                plt.xlabel("Subset diversity, global D")
                plt.ylabel("Median aggregate TVD")
                plt.title(f"{policy}:{scheme}, diversity vs TVD, k={k}")
                plt.tight_layout()
                safe = f"{policy}_{scheme}_k{int(k)}".replace("/", "_")
                plt.savefig(plot_dir / f"diversity_vs_tvd_{safe}.png", dpi=200)
                plt.close()


# ---------------------------------------------------------------------------
# Aggregate stage
# ---------------------------------------------------------------------------


def run_aggregate(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir or args.output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = input_dir / "trace_payloads.jsonl.gz"
    if not payload_path.exists():
        raise FileNotFoundError(f"materialized payload file not found: {payload_path}")

    payloads = read_payloads_jsonl(payload_path)
    if not payloads:
        raise RuntimeError(f"no payloads found in {payload_path}")

    aggregate_config_id = args.aggregate_config_id
    config = config_from_id_or_args(aggregate_config_id, args.reference_shots)
    weighting_schemes = parse_csv_strings(args.weighting_schemes)
    fixed_baselines = parse_csv_ints(args.fixed_baselines)
    policies = parse_csv_strings(args.policies)

    for policy in policies:
        if policy not in set(DEFAULT_POLICIES):
            raise ValueError(f"unsupported policy {policy!r}; allowed {DEFAULT_POLICIES}")

    run_start = now_seconds()
    diversity_matrix, diversity_long = compute_diversity_matrix(payloads)
    policy_df = aggregate_policy_payloads(
        payloads=payloads,
        config=config,
        weighting_schemes=weighting_schemes,
        policies=policies,
        confidence_floor=args.confidence_floor,
        diversity_matrix=diversity_matrix,
    )
    local_compat_df = aggregate_stable_payloads_backward_compatible(policy_df)
    fixed_aggregate_df = aggregate_fixed_payloads(
        payloads=payloads,
        fixed_baselines=fixed_baselines,
        diversity_matrix=diversity_matrix,
    )

    policy_summary_by_k = summarize_policy_by_k(policy_df, args.bootstrap_reps, args.bootstrap_seed)
    policy_clustered_summary_by_k, policy_circuit_level = summarize_policy_clustered_by_circuit(
        policy_df, args.bootstrap_reps, args.bootstrap_seed
    )
    fixed_summary_by_k = summarize_fixed_by_k(fixed_aggregate_df, args.bootstrap_reps, args.bootstrap_seed)
    subset_level, complementarity_corr, subset_rankings = complementarity_analysis(policy_df)
    tvd_table = policy_tvd_table(policy_df)

    policy_df.to_csv(output_dir / "aggregate_policy_subset_metrics.csv", index=False)
    local_compat_df.to_csv(output_dir / "aggregate_stableshots_subset_metrics.csv", index=False)
    fixed_aggregate_df.to_csv(output_dir / "aggregate_fixed_subset_metrics.csv", index=False)
    policy_summary_by_k.to_csv(output_dir / "aggregate_policy_summary_by_k.csv", index=False)
    policy_clustered_summary_by_k.to_csv(output_dir / "aggregate_policy_rq_clustered_summary_by_k.csv", index=False)
    policy_circuit_level.to_csv(output_dir / "aggregate_policy_circuit_level_metrics.csv", index=False)
    # Backward-compatible names for local-only rows only.
    summarize_policy_by_k(local_compat_df.assign(policy="local_only_offline_aggregate"), args.bootstrap_reps, args.bootstrap_seed).to_csv(
        output_dir / "aggregate_summary_by_k.csv", index=False
    ) if not local_compat_df.empty else pd.DataFrame().to_csv(output_dir / "aggregate_summary_by_k.csv", index=False)
    local_clustered_summary, local_circuit_level = summarize_policy_clustered_by_circuit(
        local_compat_df.assign(policy="local_only_offline_aggregate"), args.bootstrap_reps, args.bootstrap_seed
    ) if not local_compat_df.empty else (pd.DataFrame(), pd.DataFrame())
    local_clustered_summary.to_csv(output_dir / "aggregate_rq1_rq2_clustered_summary_by_k.csv", index=False)
    local_circuit_level.to_csv(output_dir / "aggregate_rq1_rq2_circuit_level_metrics.csv", index=False)

    fixed_summary_by_k.to_csv(output_dir / "aggregate_fixed_summary_by_k.csv", index=False)
    subset_level.to_csv(output_dir / "aggregate_subset_level_metrics.csv", index=False)
    complementarity_corr.to_csv(output_dir / "aggregate_complementarity_correlations.csv", index=False)
    subset_rankings.to_csv(output_dir / "aggregate_subset_rankings.csv", index=False)
    tvd_table.to_csv(output_dir / "aggregate_policy_tvd_table.csv", index=False)
    diversity_long.to_csv(output_dir / "backend_diversity_matrix_long.csv", index=False)
    diversity_matrix.to_csv(output_dir / "backend_diversity_matrix.csv")

    pd.DataFrame(
        [
            {
                "stage": "aggregate",
                "payload_path": str(payload_path),
                "trace_count": len(payloads),
                "circuit_config_count": policy_df["circuit_key"].nunique() if not policy_df.empty else 0,
                "backend_count": len({str(p["backend"]) for p in payloads}),
                "selected_config_id": aggregate_config_id,
                "policies": args.policies,
                "weighting_schemes": args.weighting_schemes,
                "fixed_baselines": args.fixed_baselines,
                "bootstrap_reps": args.bootstrap_reps,
                "confidence_floor": args.confidence_floor,
                "run_time_seconds": now_seconds() - run_start,
                "non_independence_note": (
                    "aggregate_policy_summary_by_k.csv uses row-level descriptive bootstrap CIs. "
                    "aggregate_policy_rq_clustered_summary_by_k.csv is the corrected policy table: "
                    "it collapses overlapping subset rows within each circuit_key before bootstrapping."
                ),
            }
        ]
    ).to_csv(output_dir / "aggregation_metadata.csv", index=False)

    if args.make_plots:
        make_aggregate_plots(policy_df, diversity_matrix, subset_level, output_dir)

    print(f"Wrote aggregate results to {output_dir.resolve()}")
    print("\nPolicy clustered summary by k:")
    print(policy_clustered_summary_by_k.to_string(index=False))
    print("\nComplementarity correlations:")
    if complementarity_corr.empty:
        print("  No correlations available.")
    else:
        print(complementarity_corr.to_string(index=False))



# ---------------------------------------------------------------------------
# QSimBench availability probing
# ---------------------------------------------------------------------------


def requested_trace_specs(args: argparse.Namespace) -> List[TraceSpec]:
    return [
        TraceSpec(algorithm=algorithm, size=size, backend=backend, circuit_kind=args.circuit_kind)
        for algorithm in parse_csv_strings(args.algorithms)
        for size in parse_csv_ints(args.sizes)
        for backend in parse_csv_strings(args.backends)
    ]


def probe_qsimbench_availability(args: argparse.Namespace) -> pd.DataFrame:
    """Probe local QSimBench availability for requested traces.

    This function uses qsimbench.get_index when possible and falls back to a
    one-shot get_outcomes probe. It is intended to confirm that expanded size
    grids such as 4..14 are available in the user's local QSimBench install and
    cache/network environment.
    """
    specs = requested_trace_specs(args)
    rows: List[Dict[str, object]] = []
    index = None
    index_error = ""
    try:
        from qsimbench import get_index  # type: ignore
        try:
            index = get_index(circuit_kind=args.circuit_kind)
        except TypeError:
            index = get_index()
    except Exception as exc:
        index_error = f"{type(exc).__name__}: {exc}"

    def index_contains(spec: TraceSpec) -> Optional[bool]:
        if index is None:
            return None
        try:
            alg_entry = index.get(spec.algorithm) if isinstance(index, Mapping) else None
            if alg_entry is None:
                return False
            size_entry = None
            if isinstance(alg_entry, Mapping):
                size_entry = alg_entry.get(spec.size, alg_entry.get(str(spec.size)))
            if size_entry is None:
                return False
            if isinstance(size_entry, Mapping):
                return spec.backend in size_entry or spec.backend in size_entry.values()
            if isinstance(size_entry, (list, tuple, set)):
                return spec.backend in size_entry
            return False
        except Exception:
            return None

    for spec in specs:
        available = index_contains(spec)
        method = "get_index"
        error = index_error
        if available is None or args.probe_sample:
            try:
                from qsimbench import get_outcomes  # type: ignore
                counts = get_outcomes(
                    algorithm=spec.algorithm,
                    size=spec.size,
                    backend=spec.backend,
                    shots=1,
                    circuit_kind=spec.circuit_kind,
                    exact=True,
                    strategy=args.sampling_strategy,
                    seed=stable_sampling_seed(args.sampling_seed, spec, 0),
                    force=False,
                )
                available = total_count(Counter({str(k): int(v) for k, v in counts.items()})) == 1
                method = "get_outcomes_1shot"
                error = ""
            except Exception as exc:
                available = False
                method = "get_outcomes_1shot"
                error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "trace_id": spec.trace_id,
                "circuit_key": spec.circuit_key,
                "algorithm": spec.algorithm,
                "size": spec.size,
                "backend": spec.backend,
                "circuit_kind": spec.circuit_kind,
                "available": bool(available),
                "probe_method": method,
                "error": error,
            }
        )
    return pd.DataFrame(rows)


def run_probe(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_df = probe_qsimbench_availability(args)
    probe_df.to_csv(output_dir / "qsimbench_probe.csv", index=False)
    summary = (
        probe_df.groupby(["algorithm", "size"], sort=True)
        .agg(requested=("available", "size"), available=("available", "sum"))
        .reset_index()
    )
    summary["complete"] = summary["requested"] == summary["available"]
    summary.to_csv(output_dir / "qsimbench_probe_summary_by_algorithm_size.csv", index=False)
    missing = probe_df[~probe_df["available"]]
    print(f"Wrote QSimBench probe to {output_dir.resolve()}")
    print(summary.to_string(index=False))
    if not missing.empty:
        print("\nMissing requested traces:")
        print(missing[["trace_id", "error"]].to_string(index=False))
        if args.fail_on_missing_traces:
            raise RuntimeError(f"{len(missing)} requested QSimBench traces are missing")


# ---------------------------------------------------------------------------
# Grid-search selection over policies, schemes, and StableShots configs
# ---------------------------------------------------------------------------


def available_config_ids(payloads: Sequence[Mapping[str, object]]) -> List[str]:
    config_sets: List[set[str]] = []
    for payload in payloads:
        stable = payload.get("stable", {})
        if isinstance(stable, Mapping):
            config_sets.append(set(str(k) for k in stable.keys()))
    if not config_sets:
        return []
    common = set.intersection(*config_sets)
    return sorted(common)


def filter_payloads_for_requested_grid(payloads: Sequence[Mapping[str, object]], args: argparse.Namespace) -> List[Mapping[str, object]]:
    algorithms = set(parse_csv_strings(args.algorithms))
    sizes = set(parse_csv_ints(args.sizes))
    backends = set(parse_csv_strings(args.backends))
    return [
        p
        for p in payloads
        if str(p.get("algorithm")) in algorithms
        and int(p.get("size")) in sizes
        and str(p.get("backend")) in backends
        and str(p.get("circuit_kind")) == args.circuit_kind
    ]


def make_grid_splits(
    circuit_df: pd.DataFrame,
    mode: str,
    test_fraction: float,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    if mode not in GRID_SPLIT_MODES:
        raise ValueError(f"unsupported grid split mode {mode!r}; allowed {sorted(GRID_SPLIT_MODES)}")
    circuits = circuit_df[["circuit_key", "algorithm", "size", "circuit_kind"]].drop_duplicates().copy()
    rows: List[Dict[str, object]] = []
    for rep in range(repetitions):
        rng = np.random.default_rng(seed + rep)
        test_keys: set[str] = set()
        if mode == "stratified_size":
            for size, group in circuits.groupby("size", sort=True):
                keys = sorted(group["circuit_key"].tolist())
                n_test = max(1, min(len(keys) - 1, int(round(len(keys) * test_fraction))))
                chosen = rng.choice(keys, size=n_test, replace=False)
                test_keys.update(str(k) for k in chosen)
        elif mode == "grouped_circuit_random":
            keys = sorted(circuits["circuit_key"].tolist())
            n_test = max(1, min(len(keys) - 1, int(round(len(keys) * test_fraction))))
            chosen = rng.choice(keys, size=n_test, replace=False)
            test_keys.update(str(k) for k in chosen)
        split_id = f"{mode}_rep{rep:03d}"
        for _, row in circuits.iterrows():
            key = str(row["circuit_key"])
            rows.append(
                {
                    "split_id": split_id,
                    "split_mode": mode,
                    "repetition": rep,
                    "circuit_key": key,
                    "algorithm": row["algorithm"],
                    "size": int(row["size"]),
                    "circuit_kind": row["circuit_kind"],
                    "split_role": "test" if key in test_keys else "train",
                }
            )
    return pd.DataFrame(rows)


def aggregate_policy_grid_payloads(
    payloads: Sequence[Mapping[str, object]],
    configs: Sequence[StableShotsConfig],
    weighting_schemes: Sequence[str],
    policies: Sequence[str],
    confidence_floor: float,
    diversity_matrix: pd.DataFrame,
    parts_dir: Optional[Path] = None,
    resume: bool = False,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if parts_dir is not None:
        parts_dir.mkdir(parents=True, exist_ok=True)
    for index, config in enumerate(configs, start=1):
        part_path = parts_dir / f"grid_policy_subset_metrics__{config.config_id}.csv.gz" if parts_dir is not None else None
        if resume and part_path is not None and part_path.exists():
            print(f"[{index}/{len(configs)}] reuse cached grid config {config.config_id}", flush=True)
            frame = pd.read_csv(part_path)
            frames.append(frame)
            continue
        start = now_seconds()
        print(f"[{index}/{len(configs)}] aggregate policy grid config {config.config_id}", flush=True)
        frame = aggregate_policy_payloads(
            payloads=payloads,
            config=config,
            weighting_schemes=weighting_schemes,
            policies=policies,
            confidence_floor=confidence_floor,
            diversity_matrix=diversity_matrix,
        )
        elapsed = now_seconds() - start
        print(f"    done {config.config_id}: {len(frame)} rows in {elapsed:.2f}s", flush=True)
        if part_path is not None:
            frame.to_csv(part_path, index=False, compression="gzip")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def grid_circuit_level_metrics(policy_df: pd.DataFrame) -> pd.DataFrame:
    if policy_df.empty:
        return pd.DataFrame()
    return (
        policy_df.groupby(
            ["config_id", "policy", "scheme", "k", "circuit_key", "algorithm", "size", "circuit_kind"],
            sort=True,
        )
        .agg(
            circuit_median_tvd=("tvd", "median"),
            circuit_q95_tvd=("tvd", lambda s: float(s.quantile(0.95))),
            circuit_max_tvd=("tvd", "max"),
            circuit_median_ssr=("ssr", "median"),
            circuit_median_parallel_ssr=("parallel_ssr", "median"),
            circuit_median_total_shots=("adaptive_shots_total", "median"),
            circuit_median_max_backend_shots=("max_backend_shots", "median"),
            subset_rows=("subset_key", "nunique"),
        )
        .reset_index()
    )


def summarize_grid_candidate_frame(frame: pd.DataFrame, prefix: str = "") -> Dict[str, object]:
    if frame.empty:
        return {}
    risk_q95 = float(frame["circuit_median_tvd"].quantile(0.95))
    risk_max = float(frame["circuit_median_tvd"].max())
    return {
        f"{prefix}rows": int(len(frame)),
        f"{prefix}circuit_configs": int(frame["circuit_key"].nunique()),
        f"{prefix}k_values": int(frame["k"].nunique()),
        f"{prefix}median_tvd": float(frame["circuit_median_tvd"].median()),
        f"{prefix}q75_tvd": float(frame["circuit_median_tvd"].quantile(0.75)),
        f"{prefix}q95_tvd": risk_q95,
        f"{prefix}max_tvd": risk_max,
        f"{prefix}valid_rate_tvd_le_0p01": float((frame["circuit_median_tvd"] <= 0.01).mean()),
        f"{prefix}valid_rate_tvd_le_0p05": float((frame["circuit_median_tvd"] <= 0.05).mean()),
        f"{prefix}valid_rate_tvd_le_0p10": float((frame["circuit_median_tvd"] <= 0.10).mean()),
        f"{prefix}median_ssr": float(frame["circuit_median_ssr"].median()),
        f"{prefix}q25_ssr": float(frame["circuit_median_ssr"].quantile(0.25)),
        f"{prefix}median_parallel_ssr": float(frame["circuit_median_parallel_ssr"].median()),
        f"{prefix}median_total_shots": float(frame["circuit_median_total_shots"].median()),
        f"{prefix}median_max_backend_shots": float(frame["circuit_median_max_backend_shots"].median()),
        f"{prefix}selection_risk_q95": risk_q95,
        f"{prefix}selection_risk_max": risk_max,
    }


def build_grid_candidate_train_metrics(
    circuit_level: pd.DataFrame,
    split_assignments: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for split_id, split in split_assignments.groupby("split_id", sort=True):
        train_keys = set(split.loc[split["split_role"] == "train", "circuit_key"])
        train = circuit_level[circuit_level["circuit_key"].isin(train_keys)]
        for (config_id, policy, scheme), group in train.groupby(["config_id", "policy", "scheme"], sort=True):
            row = {
                "split_id": split_id,
                "split_mode": split["split_mode"].iloc[0],
                "repetition": int(split["repetition"].iloc[0]),
                "config_id": config_id,
                "policy": policy,
                "scheme": scheme,
            }
            row.update(summarize_grid_candidate_frame(group, prefix="train_"))
            rows.append(row)
    return pd.DataFrame(rows)


def select_grid_configs(train_metrics: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    risk_col = "train_selection_risk_q95" if args.selection_risk_stat == "q95" else "train_selection_risk_max"
    for (split_id, policy, scheme), group in train_metrics.groupby(["split_id", "policy", "scheme"], sort=True):
        eligible = group[group[risk_col] <= args.selection_target_tvd].copy()
        relaxed = False
        if eligible.empty:
            eligible = group.copy()
            relaxed = True
        if args.selection_objective == "max_ssr":
            sort_cols = ["train_median_ssr", risk_col, "train_median_tvd"]
            ascending = [False, True, True]
        elif args.selection_objective == "max_parallel_ssr":
            sort_cols = ["train_median_parallel_ssr", risk_col, "train_median_tvd"]
            ascending = [False, True, True]
        elif args.selection_objective == "min_tvd":
            sort_cols = [risk_col, "train_median_tvd", "train_median_ssr"]
            ascending = [True, True, False]
        else:
            raise ValueError(f"unsupported selection objective {args.selection_objective!r}")
        selected = eligible.sort_values(sort_cols, ascending=ascending).iloc[0].to_dict()
        selected.update(
            {
                "selected_config_id": selected["config_id"],
                "selection_relaxed": relaxed,
                "selection_target_tvd": args.selection_target_tvd,
                "selection_risk_stat": args.selection_risk_stat,
                "selection_objective": args.selection_objective,
                "selection_risk_value": selected[risk_col],
            }
        )
        rows.append(selected)
    return pd.DataFrame(rows)


def evaluate_grid_selected_configs(
    circuit_level: pd.DataFrame,
    split_assignments: pd.DataFrame,
    selected_configs: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    detail_frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []
    for _, sel in selected_configs.iterrows():
        split_id = str(sel["split_id"])
        test_keys = set(
            split_assignments.loc[
                (split_assignments["split_id"] == split_id) & (split_assignments["split_role"] == "test"),
                "circuit_key",
            ]
        )
        mask = (
            (circuit_level["config_id"] == sel["selected_config_id"])
            & (circuit_level["policy"] == sel["policy"])
            & (circuit_level["scheme"] == sel["scheme"])
            & (circuit_level["circuit_key"].isin(test_keys))
        )
        detail = circuit_level[mask].copy()
        detail.insert(0, "split_id", split_id)
        detail.insert(1, "split_mode", sel["split_mode"])
        detail.insert(2, "repetition", int(sel["repetition"]))
        detail.insert(3, "selected_config_id", sel["selected_config_id"])
        detail_frames.append(detail)
        row = {
            "split_id": split_id,
            "split_mode": sel["split_mode"],
            "repetition": int(sel["repetition"]),
            "policy": sel["policy"],
            "scheme": sel["scheme"],
            "selected_config_id": sel["selected_config_id"],
            "selection_relaxed": bool(sel["selection_relaxed"]),
            "selection_target_tvd": float(sel["selection_target_tvd"]),
            "selection_risk_stat": sel["selection_risk_stat"],
            "selection_objective": sel["selection_objective"],
            "selection_risk_value": float(sel["selection_risk_value"]),
        }
        row.update(summarize_grid_candidate_frame(detail, prefix="test_"))
        summary_rows.append(row)
    detail_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    return detail_df, summary_df


def summarize_grid_selected_repetitions(selected_test_summary: pd.DataFrame) -> pd.DataFrame:
    if selected_test_summary.empty:
        return pd.DataFrame()
    return (
        selected_test_summary.groupby(["policy", "scheme"], sort=True)
        .agg(
            repetitions=("split_id", "nunique"),
            median_test_median_tvd=("test_median_tvd", "median"),
            q75_test_median_tvd=("test_median_tvd", lambda s: float(s.quantile(0.75))),
            median_test_q95_tvd=("test_q95_tvd", "median"),
            median_test_max_tvd=("test_max_tvd", "median"),
            median_test_valid_rate_tvd_le_0p05=("test_valid_rate_tvd_le_0p05", "median"),
            median_test_median_ssr=("test_median_ssr", "median"),
            median_test_median_parallel_ssr=("test_median_parallel_ssr", "median"),
            relaxed_selection_rate=("selection_relaxed", "mean"),
        )
        .reset_index()
    )


def summarize_grid_config_frequency(selected_configs: pd.DataFrame) -> pd.DataFrame:
    if selected_configs.empty:
        return pd.DataFrame()
    out = (
        selected_configs.groupby(["policy", "scheme", "selected_config_id"], sort=True)
        .agg(selections=("split_id", "count"), relaxed_rate=("selection_relaxed", "mean"))
        .reset_index()
    )
    totals = selected_configs.groupby(["policy", "scheme"])["split_id"].count().rename("total")
    out = out.merge(totals, on=["policy", "scheme"], how="left")
    out["selection_frequency"] = out["selections"] / out["total"]
    return out.sort_values(["policy", "scheme", "selections"], ascending=[True, True, False])


def run_grid_search(args: argparse.Namespace, configs: Sequence[StableShotsConfig]) -> None:
    input_dir = Path(args.input_dir or args.output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = input_dir / "trace_payloads.jsonl.gz"
    if not payload_path.exists():
        raise FileNotFoundError(f"materialized payload file not found: {payload_path}")
    payloads_all = read_payloads_jsonl(payload_path)
    payloads = filter_payloads_for_requested_grid(payloads_all, args)
    if not payloads:
        raise RuntimeError("no payloads match requested algorithms/sizes/backends/circuit-kind")
    expected = len(parse_csv_strings(args.algorithms)) * len(parse_csv_ints(args.sizes)) * len(parse_csv_strings(args.backends))
    if args.fail_on_missing_traces and len(payloads) != expected:
        raise RuntimeError(f"expected {expected} payloads but found {len(payloads)} after filtering")

    materialized_configs = set(available_config_ids(payloads))
    requested_configs = configs
    if str(getattr(args, "grid_config_ids", "")).strip():
        wanted = set(parse_csv_strings(args.grid_config_ids))
        requested_configs = [c for c in requested_configs if c.config_id in wanted]
        if not requested_configs:
            raise RuntimeError("--grid-config-ids did not match any requested StableShots configuration")

    # Grid search is allowed to evaluate configurations that were not precomputed during
    # materialization. Those local StableShots results are computed lazily from raw_batches.
    # This is essential for the DGX workflow: materialize raw traces once, then evaluate
    # the 75-configuration grid in parallel without re-querying QSimBench.
    missing_configs = [c.config_id for c in requested_configs if c.config_id not in materialized_configs]
    if missing_configs:
        raw_missing = [str(p.get("trace_id", "unknown")) for p in payloads if "raw_batches" not in p]
        if raw_missing:
            raise RuntimeError(
                "materialized payload is missing requested StableShots configs and raw_batches are unavailable; "
                "rerun materialization without --no-save-raw-batches. First affected traces: "
                + ",".join(raw_missing[:10])
            )
        print(
            "Grid search will compute missing local StableShots configs from raw_batches: "
            + ",".join(missing_configs[:10])
            + ("..." if len(missing_configs) > 10 else ""),
            flush=True,
        )

    policies = parse_csv_strings(args.policies)
    weighting_schemes = parse_csv_strings(args.weighting_schemes)
    diversity_matrix, diversity_long = compute_diversity_matrix(payloads)
    run_start = now_seconds()
    policy_grid_df = aggregate_policy_grid_payloads(
        payloads=payloads,
        configs=requested_configs,
        weighting_schemes=weighting_schemes,
        policies=policies,
        confidence_floor=args.confidence_floor,
        diversity_matrix=diversity_matrix,
        parts_dir=output_dir / "grid_parts",
        resume=args.resume_grid,
    )
    policy_grid_df.to_csv(output_dir / "grid_policy_subset_metrics.csv", index=False)
    diversity_long.to_csv(output_dir / "grid_backend_diversity_matrix_long.csv", index=False)
    diversity_matrix.to_csv(output_dir / "grid_backend_diversity_matrix.csv")

    circuit_level = grid_circuit_level_metrics(policy_grid_df)
    circuit_level.to_csv(output_dir / "grid_candidate_circuit_level_metrics.csv", index=False)
    split_assignments = make_grid_splits(
        circuit_level[["circuit_key", "algorithm", "size", "circuit_kind"]].drop_duplicates(),
        mode=args.split_mode,
        test_fraction=args.test_fraction,
        repetitions=args.split_repetitions,
        seed=args.split_seed,
    )
    split_assignments.to_csv(output_dir / "grid_split_assignments.csv", index=False)
    candidate_train = build_grid_candidate_train_metrics(circuit_level, split_assignments)
    candidate_train.to_csv(output_dir / "grid_candidate_train_metrics.csv", index=False)
    selected_configs = select_grid_configs(candidate_train, args)
    selected_configs.to_csv(output_dir / "grid_selected_configs.csv", index=False)
    selected_detail, selected_test_summary = evaluate_grid_selected_configs(circuit_level, split_assignments, selected_configs)
    selected_detail.to_csv(output_dir / "grid_selected_test_metrics.csv", index=False)
    selected_test_summary.to_csv(output_dir / "grid_selected_test_summary.csv", index=False)
    repeated_summary = summarize_grid_selected_repetitions(selected_test_summary)
    repeated_summary.to_csv(output_dir / "grid_repeated_selected_summary.csv", index=False)
    config_frequency = summarize_grid_config_frequency(selected_configs)
    config_frequency.to_csv(output_dir / "grid_config_frequency.csv", index=False)

    pd.DataFrame(
        [
            {
                "stage": "grid_search",
                "payload_path": str(payload_path),
                "payload_count": len(payloads),
                "expected_payload_count": expected,
                "circuit_config_count": circuit_level["circuit_key"].nunique(),
                "config_count": len(requested_configs),
                "grid_config_ids_filter": args.grid_config_ids,
                "resume_grid": bool(args.resume_grid),
                "policies": args.policies,
                "weighting_schemes": args.weighting_schemes,
                "sizes": args.sizes,
                "split_mode": args.split_mode,
                "test_fraction": args.test_fraction,
                "split_repetitions": args.split_repetitions,
                "selection_target_tvd": args.selection_target_tvd,
                "selection_risk_stat": args.selection_risk_stat,
                "selection_objective": args.selection_objective,
                "run_time_seconds": now_seconds() - run_start,
                "note": "Grid search collapses overlapping subset rows to circuit-level metrics before train/test selection.",
            }
        ]
    ).to_csv(output_dir / "grid_search_metadata.csv", index=False)

    if args.make_plots:
        make_aggregate_plots(policy_grid_df, diversity_matrix, pd.DataFrame(), output_dir)

    print(f"Wrote grid-search results to {output_dir.resolve()}")
    print("\nRepeated selected test summary:")
    print(repeated_summary.to_string(index=False))
    print("\nConfig frequency:")
    print(config_frequency.head(50).to_string(index=False))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def config_from_id_or_args(config_id: str, max_shots: int) -> StableShotsConfig:
    # Expected: b50_lb3_k5_eps0p005
    try:
        parts = config_id.split("_")
        batch_size = int(parts[0].lstrip("b"))
        lookback = int(parts[1].lstrip("lb"))
        stability = int(parts[2].lstrip("k"))
        eps = float(parts[3].lstrip("eps").replace("p", "."))
        return StableShotsConfig(batch_size=batch_size, lookback_batches=lookback, stability=stability, epsilon=eps, max_shots=max_shots)
    except Exception:
        raise ValueError(f"could not parse aggregate config id {config_id!r}; expected format b50_lb3_k5_eps0p005")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize QSimBench traces, evaluate multi-QPU StableShots policies, and run grouped grid search over StableShots configurations."
    )
    parser.add_argument("--mode", choices=["probe", "materialize", "aggregate", "grid_search", "all", "all_grid"], default="all")
    parser.add_argument("--output-dir", default="results/stable_shots_multi_qpu")
    parser.add_argument("--input-dir", default="")

    parser.add_argument("--algorithms", default=",".join(DEFAULT_ALGORITHMS))
    parser.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)))
    parser.add_argument("--backends", default=",".join(DEFAULT_BACKENDS))
    parser.add_argument("--circuit-kind", choices=["circuit", "mirror"], default="circuit")
    parser.add_argument("--reference-shots", type=int, default=DEFAULT_REFERENCE_SHOTS)
    parser.add_argument("--source-batch-size", type=int, default=DEFAULT_SOURCE_BATCH_SIZE)
    parser.add_argument("--fixed-baselines", default=",".join(map(str, DEFAULT_FIXED_BASELINES)))
    parser.add_argument("--sampling-strategy", choices=["sequential", "random"], default="sequential")
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-save-raw-batches", action="store_true")

    parser.add_argument("--batch-size", default="50")
    parser.add_argument("--lookback-batches", default="3")
    parser.add_argument("--stability", default="5")
    parser.add_argument("--epsilon", default="0.005")

    parser.add_argument("--aggregate-config-id", default=DEFAULT_SELECTED_CONFIG_ID)
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--weighting-schemes", default=",".join(DEFAULT_WEIGHTING_SCHEMES))
    parser.add_argument(
        "--grid-config-ids",
        default="",
        help="Optional comma-separated config IDs to evaluate in grid_search. Useful for smoke tests or resuming a subset.",
    )
    parser.add_argument(
        "--resume-grid",
        action="store_true",
        help="Cache each grid config as CSV.GZ and skip configs whose part file already exists.",
    )
    parser.add_argument("--confidence-floor", type=float, default=1e-12)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--make-plots", action="store_true")

    parser.add_argument("--probe-sample", action="store_true", help="During --mode probe, call get_outcomes(..., shots=1) even if get_index is available.")
    parser.add_argument("--fail-on-missing-traces", action="store_true", help="Fail if requested QSimBench traces or materialized payloads are missing.")

    parser.add_argument("--split-mode", choices=sorted(GRID_SPLIT_MODES), default=DEFAULT_GRID_SPLIT_MODE)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--split-repetitions", type=int, default=1)
    parser.add_argument("--selection-target-tvd", type=float, default=0.05)
    parser.add_argument("--selection-risk-stat", choices=sorted(GRID_SELECTION_RISK_STATS), default="q95")
    parser.add_argument("--selection-objective", choices=sorted(GRID_SELECTION_OBJECTIVES), default="max_ssr")
    return parser


def validate_args(args: argparse.Namespace, configs: Sequence[StableShotsConfig]) -> None:
    require_non_empty("algorithms", parse_csv_strings(args.algorithms))
    require_non_empty("sizes", parse_csv_ints(args.sizes))
    require_non_empty("backends", parse_csv_strings(args.backends))
    require_non_empty("fixed_baselines", parse_csv_ints(args.fixed_baselines))
    require_non_empty("configs", list(configs))
    require_non_empty("weighting_schemes", parse_csv_strings(args.weighting_schemes))
    require_non_empty("policies", parse_csv_strings(args.policies))
    if args.reference_shots <= 0:
        raise ValueError("--reference-shots must be positive")
    if args.source_batch_size <= 0:
        raise ValueError("--source-batch-size must be positive")
    if args.confidence_floor <= 0:
        raise ValueError("--confidence-floor must be positive")
    if args.bootstrap_reps < 0:
        raise ValueError("--bootstrap-reps must be non-negative")
    for scheme in parse_csv_strings(args.weighting_schemes):
        if scheme not in set(DEFAULT_WEIGHTING_SCHEMES):
            raise ValueError(f"unsupported weighting scheme {scheme!r}")
    for policy in parse_csv_strings(args.policies):
        if policy not in set(DEFAULT_POLICIES):
            raise ValueError(f"unsupported policy {policy!r}")
    max_fixed = max(parse_csv_ints(args.fixed_baselines))
    max_config = max(config.max_shots for config in configs)
    if max(max_fixed, max_config) > args.reference_shots:
        raise ValueError("fixed baselines and config max shots must not exceed --reference-shots")
    raw_required = any(p in {"aggregate_only_sync", "hierarchical_local_aggregate"} for p in parse_csv_strings(args.policies))
    if raw_required and args.no_save_raw_batches and args.mode in {"materialize", "all", "all_grid"}:
        raise ValueError("new aggregate policies require raw batches; do not use --no-save-raw-batches")
    if not (0 < args.test_fraction < 1):
        raise ValueError("--test-fraction must be in (0, 1)")
    if args.split_repetitions <= 0:
        raise ValueError("--split-repetitions must be positive")
    if args.selection_target_tvd <= 0:
        raise ValueError("--selection-target-tvd must be positive")


def print_header(args: argparse.Namespace, configs: Sequence[StableShotsConfig]) -> None:
    print("StableShots multi-QPU experiment")
    print(f"  mode                  = {args.mode}")
    print(f"  output_dir            = {args.output_dir}")
    print(f"  input_dir             = {args.input_dir or args.output_dir}")
    print(f"  algorithms            = {args.algorithms}")
    print(f"  sizes                 = {args.sizes}")
    print(f"  backends              = {args.backends}")
    print(f"  reference_shots       = {args.reference_shots}")
    print(f"  fixed_baselines       = {args.fixed_baselines}")
    print(f"  source_batch_size     = {args.source_batch_size}")
    print(f"  stable_config_ids     = {','.join(config.config_id for config in configs)}")
    print(f"  aggregate_config_id   = {args.aggregate_config_id}")
    print(f"  policies              = {args.policies}")
    print(f"  weighting_schemes     = {args.weighting_schemes}")
    print(f"  save_raw_batches      = {not args.no_save_raw_batches}")
    print(f"  split_mode            = {args.split_mode}")
    print(f"  test_fraction         = {args.test_fraction}")
    print(f"  split_repetitions     = {args.split_repetitions}")
    print(f"  selection_target_tvd  = {args.selection_target_tvd}")
    print(f"  selection_risk_stat   = {args.selection_risk_stat}")
    print(f"  selection_objective   = {args.selection_objective}")
    print()


def main() -> None:
    args = build_arg_parser().parse_args()
    grid = StableShotsGrid(
        batch_sizes=parse_csv_ints(args.batch_size),
        lookback_batches=parse_csv_ints(args.lookback_batches),
        stabilities=parse_csv_ints(args.stability),
        epsilons=parse_csv_floats(args.epsilon),
        max_shots=args.reference_shots,
    )
    configs = grid.expand()
    validate_args(args, configs)
    print_header(args, configs)

    if args.mode == "probe":
        run_probe(args)
    if args.mode in {"materialize", "all", "all_grid"}:
        run_materialize(args, configs)
    if args.mode in {"aggregate", "all", "all_grid"}:
        run_aggregate(args)
    if args.mode in {"grid_search", "all_grid"}:
        run_grid_search(args, configs)


if __name__ == "__main__":
    main()
