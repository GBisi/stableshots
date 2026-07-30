#!/usr/bin/env python3
"""
StableShots experiments with fixed baselines, Hoeffding/Weissman bound analysis,
and validation/test configuration-selection analysis.

This script supports:
  * fixed-shot baselines,
  * a grid of StableShots configurations,
  * pairwise StableShots-vs-fixed comparisons,
  * empirical validity rates at multiple TVD thresholds,
  * Hoeffding and Weissman bound comparisons at multiple TVD targets,
  * global scaled-bound calibration,
  * leave-one-group-out scaled-bound calibration,
  * structured validation/test splits for selecting StableShots configurations,
  * validation-trained/test-evaluated scaled-bound calibration,
  * repeated structured split robustness summaries.

Main outputs
------------
  trace_metrics.csv
  pairwise_metrics.csv
  summary_by_strategy.csv
  summary_pairwise.csv
  run_metadata.csv
  failures.csv, if needed

Bound outputs
-------------
  bound_trace_table.csv
  bound_trace_scaling_factors.csv
  bound_alpha_summary.csv
  bound_global_scaled_summary.csv
  bound_global_scaled_trace_metrics.csv
  bound_leave_one_group_out_summary.csv
  bound_leave_one_group_out_trace_metrics.csv

Validation/test outputs, if --validation-test-mode is not none
--------------------------------------------------------------
  split_assignments.csv
  validation_test_strategy_summary.csv
  validation_test_selected_summary.csv
  validation_test_selected_trace_metrics.csv
  validation_test_selected_pairwise_metrics.csv
  validation_test_selected_pairwise_summary.csv
  validation_test_config_frequency.csv
  bound_validation_test_summary.csv, if bound analysis is enabled
  bound_validation_test_trace_metrics.csv, if bound analysis is enabled

Bound parameters
----------------
  --bound-taus accepts a comma-separated list of TVD targets, e.g. 0.01,0.05,0.10.
  If omitted, it defaults to the values passed through --validity-deltas.

  --bound-delta is intentionally a single value. tau is the accuracy target being
  swept, while delta is the confidence/failure-probability level. Keeping one delta
  fixes the confidence level, e.g. delta=0.05 means 95% confidence, while comparing
  several TVD thresholds. Sweeping both tau and delta would answer a different
  question and would multiply the result tables unnecessarily.

Validation/test protocol
------------------------
  The recommended mode is:

      --validation-test-mode backend_holdout --test-fraction 0.2

  With the default 5 backends, this holds out one backend per algorithm-size cell,
  giving 36 test traces and 144 validation traces for the default 6x6x5 design.
  Configuration selection is performed on validation traces only; selected
  configurations are then evaluated on held-out test traces.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from stableshot.audit import AuditPolicy, StableShotsAudit
except ModuleNotFoundError:  # Allows direct execution as src/stableshot/main.py.
    from audit import AuditPolicy, StableShotsAudit

DEFAULT_ALGORITHMS = ["dj", "qaoa", "qft", "qnn", "random", "vqe"]
DEFAULT_SIZES = [4, 6, 8, 10, 12, 14]
DEFAULT_BACKENDS = ["fake_fez", "fake_kyiv", "fake_marrakesh", "fake_sherbrooke", "fake_torino"]
FIXED_BASELINES = [1000, 2500, 5000, 10000, 15000, 18000]
REFERENCE_SHOTS = 20000
QSIMBENCH_SOURCE_BATCH_SIZE = 50
ALLOWED_BOUND_CALIBRATIONS = {"median", "p75", "p90", "max"}
BOUND_GROUPS = ("size", "algorithm", "backend")
ALLOWED_VALIDATION_TEST_MODES = {"none", "backend_holdout", "random"}

Counts = Counter[str]
Batch = Tuple[int, Counts]


@dataclass(frozen=True)
class StableShotsConfig:
    batch_size: int = 50
    lookback_batches: int = 3
    stability: int = 3
    epsilon: float = 0.005
    max_shots: int = REFERENCE_SHOTS

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
    max_shots: int = REFERENCE_SHOTS

    def expand(self) -> List[StableShotsConfig]:
        configs: List[StableShotsConfig] = []
        for batch_size, lookback, stability, epsilon in product(
            self.batch_sizes, self.lookback_batches, self.stabilities, self.epsilons
        ):
            configs.append(
                StableShotsConfig(
                    batch_size=batch_size,
                    lookback_batches=lookback,
                    stability=stability,
                    epsilon=epsilon,
                    max_shots=self.max_shots,
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


@dataclass(frozen=True)
class SplitDefinition:
    split_id: str
    mode: str
    repetition: int
    validation_trace_ids: Tuple[str, ...]
    test_trace_ids: Tuple[str, ...]


def parse_csv_strings(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_csv_ints(value: str) -> List[int]:
    return [int(item) for item in parse_csv_strings(value)]


def parse_csv_floats(value: str) -> List[float]:
    return [float(item) for item in parse_csv_strings(value)]


def format_float_for_column(value: float) -> str:
    return f"{value:g}"


def valid_rate_column(delta: float) -> str:
    return f"valid_rate_tvd_to_reference_le_{format_float_for_column(delta)}"


def require_non_empty(name: str, values: Sequence[object]) -> None:
    if not values:
        raise ValueError(f"{name} must contain at least one value")


def require_positive_ints(name: str, values: Sequence[int]) -> None:
    for value in values:
        if value <= 0:
            raise ValueError(f"{name} values must be positive integers; got {value}")


def require_positive_floats(name: str, values: Sequence[float]) -> None:
    for value in values:
        if value <= 0:
            raise ValueError(f"{name} values must be positive; got {value}")


def validate_args(
    algorithms: Sequence[str],
    sizes: Sequence[int],
    backends: Sequence[str],
    fixed_baselines: Sequence[int],
    validity_deltas: Sequence[float],
    grid: StableShotsGrid,
    reference_shots: int,
    source_batch_size: int,
    fidelity_margin: float,
    bound_taus: Sequence[float],
    bound_delta: float,
    bound_calibrations: Sequence[str],
    validation_test_mode: str,
    test_fraction: float,
    split_repetitions: int,
    selection_target_tvd: float,
    selection_min_valid_rate: float,
) -> None:
    require_non_empty("algorithms", algorithms)
    require_non_empty("sizes", sizes)
    require_non_empty("backends", backends)
    require_non_empty("fixed_baselines", fixed_baselines)
    require_non_empty("batch_sizes", grid.batch_sizes)
    require_non_empty("lookback_batches", grid.lookback_batches)
    require_non_empty("stabilities", grid.stabilities)
    require_non_empty("epsilons", grid.epsilons)
    require_non_empty("validity_deltas", validity_deltas)
    require_non_empty("bound_taus", bound_taus)
    require_non_empty("bound_calibrations", bound_calibrations)

    require_positive_ints("sizes", sizes)
    require_positive_ints("fixed_baselines", fixed_baselines)
    require_positive_ints("batch_sizes", grid.batch_sizes)
    require_positive_ints("lookback_batches", grid.lookback_batches)
    require_positive_ints("stabilities", grid.stabilities)
    require_positive_floats("epsilons", grid.epsilons)
    require_positive_floats("validity_deltas", validity_deltas)
    require_positive_floats("bound_taus", bound_taus)

    if reference_shots <= 0:
        raise ValueError("reference_shots must be positive")
    if source_batch_size <= 0:
        raise ValueError("source_batch_size must be positive")
    if fidelity_margin < 0:
        raise ValueError("fidelity_margin must be non-negative")
    if max(fixed_baselines) > reference_shots:
        raise ValueError("all fixed baselines must be <= reference_shots")
    if not (0 < bound_delta < 1):
        raise ValueError("--bound-delta must be in (0, 1)")
    for calibration in bound_calibrations:
        if calibration not in ALLOWED_BOUND_CALIBRATIONS:
            raise ValueError(
                f"unsupported bound calibration {calibration!r}; "
                f"allowed values: {sorted(ALLOWED_BOUND_CALIBRATIONS)}"
            )
    if validation_test_mode not in ALLOWED_VALIDATION_TEST_MODES:
        raise ValueError(
            f"unsupported validation/test mode {validation_test_mode!r}; "
            f"allowed values: {sorted(ALLOWED_VALIDATION_TEST_MODES)}"
        )
    if not (0 < test_fraction < 1):
        raise ValueError("--test-fraction must be in (0, 1)")
    if split_repetitions <= 0:
        raise ValueError("--split-repetitions must be positive")
    if selection_target_tvd <= 0:
        raise ValueError("--selection-target-tvd must be positive")
    if not (0 < selection_min_valid_rate <= 1):
        raise ValueError("--selection-min-valid-rate must be in (0, 1]")
    if validation_test_mode == "backend_holdout" and len(backends) < 2:
        raise ValueError("backend_holdout validation/test mode requires at least two backends")


def now_seconds() -> float:
    return time.perf_counter()


def total_count(counts: Counts) -> int:
    return int(sum(counts.values()))


def merge_counts(target: Counts, source: Counts) -> None:
    for bitstring, count in source.items():
        target[bitstring] += int(count)


def deterministic_subcount(counts: Counts, target_shots: int) -> Counts:
    source_total = total_count(counts)
    if target_shots > source_total:
        raise ValueError("cannot take more shots than available")
    if target_shots == source_total:
        return Counter(counts)
    if target_shots == 0:
        return Counter()

    quotas = []
    for key, value in counts.items():
        exact = value * target_shots / source_total
        base = int(math.floor(exact))
        quotas.append((key, base, exact - base))

    result: Counts = Counter({key: base for key, base, _ in quotas if base > 0})
    missing = target_shots - total_count(result)
    quotas.sort(key=lambda item: (-item[2], item[0]))
    for key, _, _ in quotas[:missing]:
        result[key] += 1
    return result


def tvd(left: Counts, right: Counts) -> float:
    n_left = total_count(left)
    n_right = total_count(right)
    if n_left == 0 or n_right == 0:
        raise ValueError("TVD requires non-empty distributions")
    support = set(left.keys()) | set(right.keys())
    return 0.5 * sum(abs(left.get(x, 0) / n_left - right.get(x, 0) / n_right) for x in support)


def stable_sampling_seed(base_seed: int, spec: TraceSpec, batch_index: int) -> int:
    payload = f"{base_seed}|{spec.trace_id}|{spec.circuit_kind}|{batch_index}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


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

    if not batches:
        raise RuntimeError(f"no batches found for {spec.trace_id}")
    return batches


def iter_execution_batches(raw_batches: Sequence[Batch], batch_size: int, max_shots: int) -> Iterable[Batch]:
    emitted = 0
    buffer: Counts = Counter()
    buffer_shots = 0
    for shots, counts in raw_batches:
        if emitted >= max_shots:
            break
        remaining_record = Counter(counts)
        remaining_shots = shots
        while remaining_shots > 0 and emitted < max_shots:
            needed_batch = batch_size - buffer_shots
            needed_budget = max_shots - emitted - buffer_shots
            take = min(remaining_shots, needed_batch, needed_budget)
            if take == remaining_shots:
                take_counts = remaining_record
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


def run_stable_shots(
    raw_batches: Sequence[Batch],
    config: StableShotsConfig,
    audit: Optional[StableShotsAudit] = None,
) -> Tuple[Counts, int, float, str]:
    """Run StableShots, optionally emitting a complete decision audit trail.

    The return shape is intentionally unchanged for backward compatibility.
    Call ``run_stable_shots_audited`` when the caller also needs the audit object.
    """
    cumulative: Counts = Counter()
    snapshots: List[Counts] = []
    stable_checks = 0
    checks_performed = 0
    shots_used = 0
    last_delta = float("nan")
    last_round = 0
    for round_index, (batch_shots, batch_counts) in enumerate(
        iter_execution_batches(raw_batches, config.batch_size, config.max_shots), start=1
    ):
        last_round = round_index
        merge_counts(cumulative, batch_counts)
        shots_used += batch_shots
        snapshots.append(Counter(cumulative))
        if audit is not None:
            audit.record_batch(
                round_index=round_index,
                batch_shots=batch_shots,
                total_shots=shots_used,
                batch_counts=batch_counts,
                cumulative_counts=cumulative,
            )
        if len(snapshots) > config.lookback_batches:
            previous = snapshots[-1 - config.lookback_batches]
            last_delta = tvd(cumulative, previous)
            checks_performed += 1
            streak_before = stable_checks
            stable_checks = stable_checks + 1 if last_delta <= config.epsilon else 0
            if audit is not None:
                audit.record_check(
                    round_index=round_index,
                    lookback_round_index=round_index - config.lookback_batches,
                    total_shots=shots_used,
                    current_counts=cumulative,
                    previous_counts=previous,
                    delta=last_delta,
                    epsilon=config.epsilon,
                    streak_before=streak_before,
                    streak_after=stable_checks,
                    required_stability=config.stability,
                )
            if stable_checks >= config.stability:
                if audit is not None:
                    audit.record_stop(
                        reason="stable",
                        total_shots=shots_used,
                        round_index=round_index,
                        stable_streak=stable_checks,
                        checks_performed=checks_performed,
                        last_delta=last_delta,
                    )
                return Counter(cumulative), shots_used, last_delta, "stable"

    stop_reason = "max_budget" if shots_used >= config.max_shots else "input_exhausted"
    if audit is not None:
        audit.record_stop(
            reason=stop_reason,
            total_shots=shots_used,
            round_index=last_round,
            stable_streak=stable_checks,
            checks_performed=checks_performed,
            last_delta=None if math.isnan(last_delta) else last_delta,
        )
    return Counter(cumulative), shots_used, last_delta, stop_reason


def run_stable_shots_audited(
    raw_batches: Sequence[Batch],
    config: StableShotsConfig,
    *,
    context: Optional[Mapping[str, object]] = None,
    policy: Optional[AuditPolicy] = None,
) -> Tuple[Counts, int, float, str, StableShotsAudit]:
    """Convenience API returning both the StableShots result and its audit trail."""
    audit = StableShotsAudit(config=asdict(config), context=context, policy=policy)
    counts, shots, last_delta, reason = run_stable_shots(raw_batches, config, audit=audit)
    return counts, shots, last_delta, reason, audit


def stable_config_columns(config: StableShotsConfig) -> Dict[str, object]:
    return {
        "config_id": config.config_id,
        "batch_size": config.batch_size,
        "lookback_batches": config.lookback_batches,
        "stability": config.stability,
        "epsilon": config.epsilon,
    }


def empty_config_columns() -> Dict[str, object]:
    return {"config_id": "", "batch_size": "", "lookback_batches": "", "stability": "", "epsilon": ""}


def timing_columns(materialization: float, eval_time: float, adaptive_overhead: float) -> Dict[str, float]:
    return {
        "qsimbench_materialization_time_seconds": materialization,
        "strategy_eval_time_seconds": eval_time,
        "strategy_total_wall_time_seconds": materialization + eval_time,
        "adaptive_time_overhead_seconds": adaptive_overhead,
    }


def evaluate_trace(
    spec: TraceSpec,
    configs: Sequence[StableShotsConfig],
    fixed_baselines: Sequence[int],
    reference_shots: int,
    fidelity_margin: float,
    source_batch_size: int,
    sampling_strategy: str,
    sampling_seed: int,
    force: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    required_shots = max([reference_shots, *fixed_baselines, *(config.max_shots for config in configs)])
    load_start = now_seconds()
    raw_batches = load_qsimbench_batches(
        spec, required_shots, source_batch_size, sampling_strategy, sampling_seed, force
    )
    materialization_time = now_seconds() - load_start

    ref_start = now_seconds()
    reference = prefix_counts(raw_batches, reference_shots)
    ref_eval_time = now_seconds() - ref_start

    strategy_rows: List[Dict[str, object]] = []
    pairwise_rows: List[Dict[str, object]] = []
    fixed_results: Dict[int, Dict[str, float]] = {}

    for budget in fixed_baselines:
        budget = int(budget)
        start = now_seconds()
        counts = prefix_counts(raw_batches, budget)
        value_tvd = tvd(counts, reference)
        eval_time = now_seconds() - start
        fixed_results[budget] = {"tvd": value_tvd, "eval_time_seconds": eval_time}
        strategy_rows.append(
            {
                "trace_id": spec.trace_id,
                "algorithm": spec.algorithm,
                "size": spec.size,
                "backend": spec.backend,
                "strategy": f"fixed_{budget}",
                "strategy_family": "fixed",
                "shots": budget,
                "reference_shots": reference_shots,
                "tvd_to_reference": value_tvd,
                "saving_vs_reference": 1.0 - budget / reference_shots,
                "stop_reason": "fixed_budget",
                "last_stability_tvd": float("nan"),
                **timing_columns(materialization_time, eval_time, 0.0),
                **empty_config_columns(),
            }
        )

    for config in configs:
        start = now_seconds()
        stable_counts, stable_shots, last_delta, stop_reason = run_stable_shots(raw_batches, config)
        stable_tvd = tvd(stable_counts, reference)
        eval_time = now_seconds() - start
        config_cols = stable_config_columns(config)
        strategy_rows.append(
            {
                "trace_id": spec.trace_id,
                "algorithm": spec.algorithm,
                "size": spec.size,
                "backend": spec.backend,
                "strategy": config.strategy_name,
                "strategy_family": "stable_shots",
                "shots": stable_shots,
                "reference_shots": reference_shots,
                "tvd_to_reference": stable_tvd,
                "saving_vs_reference": 1.0 - stable_shots / reference_shots,
                "stop_reason": stop_reason,
                "last_stability_tvd": last_delta,
                **timing_columns(materialization_time, eval_time, eval_time),
                **config_cols,
            }
        )
        for budget, fixed_info in fixed_results.items():
            fixed_tvd = fixed_info["tvd"]
            baseline_eval_time = fixed_info["eval_time_seconds"]
            delta_shots = stable_shots - budget
            delta_tvd = stable_tvd - fixed_tvd
            comparable = stable_tvd <= fixed_tvd + fidelity_margin
            fewer_or_equal = stable_shots <= budget
            delta_time = eval_time - baseline_eval_time
            pairwise_rows.append(
                {
                    "trace_id": spec.trace_id,
                    "algorithm": spec.algorithm,
                    "size": spec.size,
                    "backend": spec.backend,
                    "strategy": config.strategy_name,
                    "baseline": f"fixed_{budget}",
                    "baseline_shots": budget,
                    "stable_shots": stable_shots,
                    "reference_shots": reference_shots,
                    "baseline_tvd_to_reference": fixed_tvd,
                    "stable_tvd_to_reference": stable_tvd,
                    "delta_shots_stable_minus_fixed": delta_shots,
                    "shots_saved_by_stable": budget - stable_shots,
                    "delta_tvd_stable_minus_fixed": delta_tvd,
                    "stable_comparable_fidelity": comparable,
                    "stable_fewer_or_equal_shots": fewer_or_equal,
                    "stable_dominates": fewer_or_equal and comparable,
                    "qsimbench_materialization_time_seconds": materialization_time,
                    "baseline_eval_time_seconds": baseline_eval_time,
                    "stable_eval_time_seconds": eval_time,
                    "delta_time_stable_minus_fixed_seconds": delta_time,
                    "absolute_time_overhead_seconds": abs(delta_time),
                    **config_cols,
                }
            )

    strategy_rows.append(
        {
            "trace_id": spec.trace_id,
            "algorithm": spec.algorithm,
            "size": spec.size,
            "backend": spec.backend,
            "strategy": f"reference_{reference_shots}",
            "strategy_family": "reference",
            "shots": reference_shots,
            "reference_shots": reference_shots,
            "tvd_to_reference": 0.0,
            "saving_vs_reference": 0.0,
            "stop_reason": "reference_only",
            "last_stability_tvd": float("nan"),
            **timing_columns(materialization_time, ref_eval_time, 0.0),
            **empty_config_columns(),
        }
    )
    return strategy_rows, pairwise_rows


def summarize_strategy_metrics(trace_df: pd.DataFrame, validity_deltas: Sequence[float]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if trace_df.empty:
        return pd.DataFrame()
    for strategy, group in trace_df.groupby("strategy", sort=False):
        first = group.iloc[0]
        row: Dict[str, object] = {
            "strategy": strategy,
            "strategy_family": first.get("strategy_family", ""),
            "config_id": first.get("config_id", ""),
            "batch_size": first.get("batch_size", ""),
            "lookback_batches": first.get("lookback_batches", ""),
            "stability": first.get("stability", ""),
            "epsilon": first.get("epsilon", ""),
            "traces": len(group),
            "reference_shots": group["reference_shots"].median(),
            "median_shots": group["shots"].median(),
            "mean_shots": group["shots"].mean(),
            "min_shots": group["shots"].min(),
            "max_shots": group["shots"].max(),
            "median_tvd_to_reference": group["tvd_to_reference"].median(),
            "mean_tvd_to_reference": group["tvd_to_reference"].mean(),
            "max_tvd_to_reference": group["tvd_to_reference"].max(),
            "median_saving_vs_reference": group["saving_vs_reference"].median(),
            "mean_saving_vs_reference": group["saving_vs_reference"].mean(),
            "stable_stop_rate": (group["stop_reason"] == "stable").mean(),
            "max_budget_rate": (group["stop_reason"] == "max_budget").mean(),
            "median_qsimbench_materialization_time_seconds": group[
                "qsimbench_materialization_time_seconds"
            ].median(),
            "mean_qsimbench_materialization_time_seconds": group[
                "qsimbench_materialization_time_seconds"
            ].mean(),
            "median_strategy_eval_time_seconds": group["strategy_eval_time_seconds"].median(),
            "mean_strategy_eval_time_seconds": group["strategy_eval_time_seconds"].mean(),
            "median_strategy_total_wall_time_seconds": group["strategy_total_wall_time_seconds"].median(),
            "mean_strategy_total_wall_time_seconds": group["strategy_total_wall_time_seconds"].mean(),
            "median_adaptive_time_overhead_seconds": group["adaptive_time_overhead_seconds"].median(),
            "mean_adaptive_time_overhead_seconds": group["adaptive_time_overhead_seconds"].mean(),
        }
        for delta in validity_deltas:
            row[valid_rate_column(delta)] = (group["tvd_to_reference"] <= delta).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_pairwise_metrics(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    if pairwise_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for (strategy, baseline), group in pairwise_df.groupby(["strategy", "baseline"], sort=False):
        first = group.iloc[0]
        rows.append(
            {
                "comparison": f"{strategy}_vs_{baseline}",
                "strategy": strategy,
                "baseline": baseline,
                "config_id": first["config_id"],
                "batch_size": first["batch_size"],
                "lookback_batches": first["lookback_batches"],
                "stability": first["stability"],
                "epsilon": first["epsilon"],
                "traces": len(group),
                "reference_shots": group["reference_shots"].median(),
                "median_shots_saved_by_stable": group["shots_saved_by_stable"].median(),
                "mean_shots_saved_by_stable": group["shots_saved_by_stable"].mean(),
                "median_delta_shots_stable_minus_fixed": group["delta_shots_stable_minus_fixed"].median(),
                "median_delta_tvd_stable_minus_fixed": group["delta_tvd_stable_minus_fixed"].median(),
                "mean_delta_tvd_stable_minus_fixed": group["delta_tvd_stable_minus_fixed"].mean(),
                "comparable_fidelity_rate": group["stable_comparable_fidelity"].mean(),
                "fewer_or_equal_shots_rate": group["stable_fewer_or_equal_shots"].mean(),
                "dominance_rate": group["stable_dominates"].mean(),
                "median_baseline_eval_time_seconds": group["baseline_eval_time_seconds"].median(),
                "mean_baseline_eval_time_seconds": group["baseline_eval_time_seconds"].mean(),
                "median_stable_eval_time_seconds": group["stable_eval_time_seconds"].median(),
                "mean_stable_eval_time_seconds": group["stable_eval_time_seconds"].mean(),
                "median_delta_time_stable_minus_fixed_seconds": group[
                    "delta_time_stable_minus_fixed_seconds"
                ].median(),
                "mean_delta_time_stable_minus_fixed_seconds": group[
                    "delta_time_stable_minus_fixed_seconds"
                ].mean(),
                "median_absolute_time_overhead_seconds": group["absolute_time_overhead_seconds"].median(),
                "mean_absolute_time_overhead_seconds": group["absolute_time_overhead_seconds"].mean(),
                "median_qsimbench_materialization_time_seconds": group[
                    "qsimbench_materialization_time_seconds"
                ].median(),
                "mean_qsimbench_materialization_time_seconds": group[
                    "qsimbench_materialization_time_seconds"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def tvd_column_name_for_budget(budget: int) -> str:
    return f"tvd_{int(budget)}"


def log_two_power_M_minus_2(M: int) -> float:
    if M <= 1:
        raise ValueError("M must be at least 2")
    if M < 50:
        return math.log((2 ** M) - 2)
    return M * math.log(2)


def hoeffding_shots_for_tvd(num_qubits: int, tau: float, delta: float) -> int:
    M = 2 ** int(num_qubits)
    n = (M**2 / (8 * tau**2)) * math.log((2 * M) / delta)
    return int(math.ceil(n))


def weissman_shots_for_tvd(num_qubits: int, tau: float, delta: float) -> int:
    M = 2 ** int(num_qubits)
    n = (log_two_power_M_minus_2(M) + math.log(1 / delta)) / (2 * tau**2)
    return int(math.ceil(n))


def build_bound_analysis_trace_table(trace_df: pd.DataFrame, fixed_baselines: Sequence[int]) -> pd.DataFrame:
    fixed_names = {f"fixed_{int(b)}" for b in fixed_baselines}
    fixed_df = trace_df[trace_df["strategy"].isin(fixed_names)].copy()
    if fixed_df.empty:
        raise ValueError("no fixed-shot rows available for bound analysis")
    fixed_df["shots"] = fixed_df["shots"].astype(int)
    index_cols = ["trace_id", "algorithm", "size", "backend", "reference_shots"]
    wide = fixed_df.pivot_table(
        index=index_cols,
        columns="shots",
        values="tvd_to_reference",
        aggfunc="first",
    ).reset_index()
    wide.columns = [
        tvd_column_name_for_budget(c) if isinstance(c, (int, np.integer)) else c for c in wide.columns
    ]
    for budget in fixed_baselines:
        col = tvd_column_name_for_budget(int(budget))
        if col not in wide.columns:
            wide[col] = np.nan
    return wide


def empirical_sufficient_budget(row: pd.Series, tau: float, fixed_baselines: Sequence[int]):
    for budget in sorted(int(b) for b in fixed_baselines):
        value = row.get(tvd_column_name_for_budget(budget), np.nan)
        if pd.notna(value) and value <= tau:
            return budget
    return np.nan


def round_up_to_available_budget(n, fixed_baselines: Sequence[int]):
    if pd.isna(n):
        return np.nan
    for budget in sorted(int(b) for b in fixed_baselines):
        if n <= budget:
            return budget
    return np.nan


def tvd_at_available_budget(row: pd.Series, budget):
    if pd.isna(budget):
        return np.nan
    return row.get(tvd_column_name_for_budget(int(budget)), np.nan)


def add_bound_and_scaling_columns(
    bound_df: pd.DataFrame,
    fixed_baselines: Sequence[int],
    tau: float,
    delta: float,
) -> pd.DataFrame:
    out = bound_df.copy()
    out["bound_tau"] = tau
    out["bound_delta"] = delta
    out["n_empirical_tau"] = out.apply(lambda r: empirical_sufficient_budget(r, tau, fixed_baselines), axis=1)
    out["n_hoeffding"] = out["size"].apply(lambda q: hoeffding_shots_for_tvd(int(q), tau, delta))
    out["n_weissman"] = out["size"].apply(lambda q: weissman_shots_for_tvd(int(q), tau, delta))
    out["alpha_hoeffding"] = out["n_empirical_tau"] / out["n_hoeffding"]
    out["alpha_weissman"] = out["n_empirical_tau"] / out["n_weissman"]
    out["empirical_target_reached_by_fixed_budget"] = out["n_empirical_tau"].notna()
    return out


def choose_alpha(values: pd.Series, calibration: str):
    clean = values.dropna()
    if clean.empty:
        return np.nan
    if calibration == "median":
        return clean.median()
    if calibration == "p75":
        return clean.quantile(0.75)
    if calibration == "p90":
        return clean.quantile(0.90)
    if calibration == "max":
        return clean.max()
    raise ValueError(f"unsupported calibration {calibration!r}")


def summarize_alpha_values(
    frame: pd.DataFrame,
    alpha_col: str,
    bound: str,
    group_col: str,
    group_value: object,
) -> Dict[str, object]:
    values = frame[alpha_col].dropna()
    bound_tau = frame["bound_tau"].iloc[0] if "bound_tau" in frame.columns and len(frame) else np.nan
    bound_delta = frame["bound_delta"].iloc[0] if "bound_delta" in frame.columns and len(frame) else np.nan
    row: Dict[str, object] = {
        "bound_tau": bound_tau,
        "bound_delta": bound_delta,
        "bound": bound,
        "alpha_col": alpha_col,
        "group_col": group_col,
        "group_value": group_value,
        "count": int(values.count()),
    }
    if values.empty:
        row.update(
            {
                "min": np.nan,
                "p25": np.nan,
                "median": np.nan,
                "p75": np.nan,
                "p90": np.nan,
                "max": np.nan,
                "iqr": np.nan,
            }
        )
        return row
    p25 = values.quantile(0.25)
    p75 = values.quantile(0.75)
    row.update(
        {
            "min": values.min(),
            "p25": p25,
            "median": values.median(),
            "p75": p75,
            "p90": values.quantile(0.90),
            "max": values.max(),
            "iqr": p75 - p25,
        }
    )
    return row


def summarize_bound_alpha_variability(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    specs = [("hoeffding", "alpha_hoeffding"), ("weissman", "alpha_weissman")]
    rows: List[Dict[str, object]] = []
    for bound, alpha_col in specs:
        rows.append(summarize_alpha_values(df, alpha_col, bound, "overall", "overall"))
        for group_col in group_cols:
            for group_value, group in df.groupby(group_col, dropna=False):
                rows.append(summarize_alpha_values(group, alpha_col, bound, group_col, group_value))
    return pd.DataFrame(rows)


def apply_scaled_bound_policy(
    df: pd.DataFrame,
    fixed_baselines: Sequence[int],
    bound_col: str,
    tau: float,
    alpha,
) -> pd.DataFrame:
    out = df.copy()
    out["n_scaled_raw"] = np.nan if pd.isna(alpha) else np.ceil(alpha * out[bound_col])
    out["n_scaled_budget"] = out["n_scaled_raw"].apply(lambda n: round_up_to_available_budget(n, fixed_baselines))
    out["tvd_scaled"] = out.apply(lambda r: tvd_at_available_budget(r, r["n_scaled_budget"]), axis=1)
    out["scaled_uncovered"] = out["n_scaled_budget"].isna()
    out["scaled_success"] = (out["tvd_scaled"] <= tau).fillna(False)
    return out


def scaled_bound_summary_from_detail(
    detail: pd.DataFrame,
    mode: str,
    tau: float,
    delta: float,
    bound: str,
    bound_col: str,
    alpha_col: str,
    calibration: str,
    alpha,
    calibration_trace_count: int,
    extra: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    evaluable = detail[~detail["scaled_uncovered"]]
    summary: Dict[str, object] = {
        "mode": mode,
        "bound_tau": tau,
        "bound_delta": delta,
        "bound": bound,
        "bound_col": bound_col,
        "alpha_col": alpha_col,
        "calibration": calibration,
        "alpha": alpha,
        "calibration_trace_count": calibration_trace_count,
        "test_trace_count": int(len(detail)),
        "success_rate_all": detail["scaled_success"].mean() if len(detail) else np.nan,
        "uncovered_rate": detail["scaled_uncovered"].mean() if len(detail) else np.nan,
        "success_rate_evaluable": evaluable["scaled_success"].mean() if not evaluable.empty else np.nan,
        "median_scaled_budget": detail["n_scaled_budget"].median(),
        "mean_scaled_budget": detail["n_scaled_budget"].mean(),
        "median_scaled_tvd": detail["tvd_scaled"].median(),
        "mean_scaled_tvd": detail["tvd_scaled"].mean(),
    }
    if extra:
        summary.update(extra)
    return summary


def evaluate_global_scaled_bound(
    df: pd.DataFrame,
    fixed_baselines: Sequence[int],
    bound: str,
    bound_col: str,
    alpha_col: str,
    tau: float,
    delta: float,
    calibration: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    alpha = choose_alpha(df[alpha_col], calibration)
    out = apply_scaled_bound_policy(df, fixed_baselines, bound_col, tau, alpha)
    out["mode"] = "global"
    out["bound"] = bound
    out["bound_col"] = bound_col
    out["alpha_col"] = alpha_col
    out["calibration"] = calibration
    out["alpha_global"] = alpha
    out["alpha_train"] = alpha
    summary = scaled_bound_summary_from_detail(
        out,
        mode="global",
        tau=tau,
        delta=delta,
        bound=bound,
        bound_col=bound_col,
        alpha_col=alpha_col,
        calibration=calibration,
        alpha=alpha,
        calibration_trace_count=int(out[alpha_col].dropna().count()),
    )
    return out, summary


def evaluate_leave_one_group_out(
    df: pd.DataFrame,
    fixed_baselines: Sequence[int],
    group_col: str,
    bound: str,
    bound_col: str,
    alpha_col: str,
    tau: float,
    delta: float,
    calibration: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    details: List[pd.DataFrame] = []
    summaries: List[Dict[str, object]] = []
    group_values = sorted(df[group_col].dropna().unique(), key=lambda value: str(value))
    for group_value in group_values:
        train = df[df[group_col] != group_value].copy()
        test = df[df[group_col] == group_value].copy()
        alpha = choose_alpha(train[alpha_col], calibration)
        test = apply_scaled_bound_policy(test, fixed_baselines, bound_col, tau, alpha)
        test["mode"] = "leave_one_group_out"
        test["held_out_group_col"] = group_col
        test["held_out_group"] = group_value
        test["bound"] = bound
        test["bound_col"] = bound_col
        test["alpha_col"] = alpha_col
        test["calibration"] = calibration
        test["alpha_train"] = alpha
        summaries.append(
            scaled_bound_summary_from_detail(
                test,
                mode="leave_one_group_out",
                tau=tau,
                delta=delta,
                bound=bound,
                bound_col=bound_col,
                alpha_col=alpha_col,
                calibration=calibration,
                alpha=alpha,
                calibration_trace_count=int(train[alpha_col].dropna().count()),
                extra={"held_out_group_col": group_col, "held_out_group": group_value, "alpha_train": alpha},
            )
        )
        details.append(test)
    return pd.concat(details, ignore_index=True) if details else pd.DataFrame(), pd.DataFrame(summaries)


def run_scaled_bound_analysis_for_tau(
    trace_table: pd.DataFrame,
    fixed_baselines: Sequence[int],
    tau: float,
    delta: float,
    calibrations: Sequence[str],
    group_cols: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    scaling = add_bound_and_scaling_columns(trace_table, fixed_baselines, tau, delta)
    alpha_summary = summarize_bound_alpha_variability(scaling, group_cols)
    bound_specs = [("hoeffding", "n_hoeffding", "alpha_hoeffding"), ("weissman", "n_weissman", "alpha_weissman")]
    global_detail_frames: List[pd.DataFrame] = []
    global_summary_rows: List[Dict[str, object]] = []
    loo_detail_frames: List[pd.DataFrame] = []
    loo_summary_frames: List[pd.DataFrame] = []
    for bound, bound_col, alpha_col in bound_specs:
        for calibration in calibrations:
            detail, summary = evaluate_global_scaled_bound(
                scaling, fixed_baselines, bound, bound_col, alpha_col, tau, delta, calibration
            )
            global_detail_frames.append(detail)
            global_summary_rows.append(summary)
            for group_col in group_cols:
                loo_detail, loo_summary = evaluate_leave_one_group_out(
                    scaling, fixed_baselines, group_col, bound, bound_col, alpha_col, tau, delta, calibration
                )
                loo_detail_frames.append(loo_detail)
                loo_summary_frames.append(loo_summary)
    return {
        "bound_trace_scaling_factors": scaling,
        "bound_alpha_summary": alpha_summary,
        "bound_global_scaled_summary": pd.DataFrame(global_summary_rows),
        "bound_global_scaled_trace_metrics": pd.concat(global_detail_frames, ignore_index=True)
        if global_detail_frames
        else pd.DataFrame(),
        "bound_leave_one_group_out_summary": pd.concat(loo_summary_frames, ignore_index=True)
        if loo_summary_frames
        else pd.DataFrame(),
        "bound_leave_one_group_out_trace_metrics": pd.concat(loo_detail_frames, ignore_index=True)
        if loo_detail_frames
        else pd.DataFrame(),
    }


def run_scaled_bound_analysis(
    trace_df: pd.DataFrame,
    fixed_baselines: Sequence[int],
    taus: Sequence[float],
    delta: float,
    calibrations: Sequence[str],
    group_cols: Sequence[str] = BOUND_GROUPS,
) -> Dict[str, pd.DataFrame]:
    trace_table = build_bound_analysis_trace_table(trace_df, fixed_baselines)
    per_tau_results: List[Dict[str, pd.DataFrame]] = []
    for tau in taus:
        per_tau_results.append(
            run_scaled_bound_analysis_for_tau(
                trace_table=trace_table,
                fixed_baselines=fixed_baselines,
                tau=tau,
                delta=delta,
                calibrations=calibrations,
                group_cols=group_cols,
            )
        )

    combined: Dict[str, pd.DataFrame] = {"bound_trace_table": trace_table}
    table_names = [
        "bound_trace_scaling_factors",
        "bound_alpha_summary",
        "bound_global_scaled_summary",
        "bound_global_scaled_trace_metrics",
        "bound_leave_one_group_out_summary",
        "bound_leave_one_group_out_trace_metrics",
    ]
    for name in table_names:
        frames = [result[name] for result in per_tau_results if name in result and not result[name].empty]
        combined[name] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined


def make_split_id(mode: str, repetition: int) -> str:
    return f"{mode}_rep{repetition:03d}"


def make_validation_test_splits(
    specs: Sequence[TraceSpec],
    mode: str,
    test_fraction: float,
    repetitions: int,
    seed: int,
) -> List[SplitDefinition]:
    if mode == "none":
        return []
    if not specs:
        return []

    trace_by_id = {spec.trace_id: spec for spec in specs}
    trace_ids = sorted(trace_by_id)
    splits: List[SplitDefinition] = []

    if mode == "random":
        for rep in range(repetitions):
            rng = np.random.default_rng(seed + rep)
            permuted = list(rng.permutation(trace_ids))
            n_test = max(1, min(len(trace_ids) - 1, int(round(len(trace_ids) * test_fraction))))
            test_ids = tuple(sorted(permuted[:n_test]))
            validation_ids = tuple(sorted(set(trace_ids) - set(test_ids)))
            splits.append(
                SplitDefinition(
                    split_id=make_split_id(mode, rep),
                    mode=mode,
                    repetition=rep,
                    validation_trace_ids=validation_ids,
                    test_trace_ids=test_ids,
                )
            )
        return splits

    if mode == "backend_holdout":
        cells: Dict[Tuple[str, int], List[TraceSpec]] = {}
        for spec in specs:
            cells.setdefault((spec.algorithm, spec.size), []).append(spec)
        for rep in range(repetitions):
            rng = np.random.default_rng(seed + rep)
            test_ids_set = set()
            for cell_key in sorted(cells, key=lambda x: (x[0], x[1])):
                cell_specs = sorted(cells[cell_key], key=lambda spec: spec.backend)
                if len(cell_specs) < 2:
                    raise ValueError(f"cell {cell_key} has fewer than two backends; cannot hold out test backend")
                n_test = max(1, min(len(cell_specs) - 1, int(round(len(cell_specs) * test_fraction))))
                indices = rng.choice(len(cell_specs), size=n_test, replace=False)
                for index in indices:
                    test_ids_set.add(cell_specs[int(index)].trace_id)
            test_ids = tuple(sorted(test_ids_set))
            validation_ids = tuple(sorted(set(trace_ids) - set(test_ids)))
            splits.append(
                SplitDefinition(
                    split_id=make_split_id(mode, rep),
                    mode=mode,
                    repetition=rep,
                    validation_trace_ids=validation_ids,
                    test_trace_ids=test_ids,
                )
            )
        return splits

    raise ValueError(f"unsupported validation/test mode {mode!r}")


def build_split_assignments(splits: Sequence[SplitDefinition], specs: Sequence[TraceSpec]) -> pd.DataFrame:
    spec_by_trace = {spec.trace_id: spec for spec in specs}
    rows: List[Dict[str, object]] = []
    for split in splits:
        for role, trace_ids in [("validation", split.validation_trace_ids), ("test", split.test_trace_ids)]:
            for trace_id in trace_ids:
                spec = spec_by_trace[trace_id]
                rows.append(
                    {
                        "split_id": split.split_id,
                        "split_mode": split.mode,
                        "repetition": split.repetition,
                        "trace_id": trace_id,
                        "split_role": role,
                        "algorithm": spec.algorithm,
                        "size": spec.size,
                        "backend": spec.backend,
                    }
                )
    return pd.DataFrame(rows)


def select_stable_strategy(
    validation_summary: pd.DataFrame,
    target_tvd: float,
    min_valid_rate: float,
    selected_config_id: str = "",
) -> Dict[str, object]:
    if validation_summary.empty:
        raise ValueError("validation summary is empty")
    stable = validation_summary[validation_summary["strategy_family"] == "stable_shots"].copy()
    if stable.empty:
        raise ValueError("no stable_shots strategies available for selection")

    target_col = valid_rate_column(target_tvd)
    if target_col not in stable.columns:
        raise ValueError(f"selection target column {target_col!r} is missing from validation summary")

    if selected_config_id:
        selected = stable[
            (stable["config_id"].astype(str) == selected_config_id)
            | (stable["strategy"].astype(str) == selected_config_id)
        ]
        if selected.empty:
            raise ValueError(f"selected config {selected_config_id!r} not found among stable_shots strategies")
        row = selected.iloc[0].to_dict()
        row["selection_relaxed"] = False
        row["selection_rule"] = "forced_config_id"
        row["selection_target_tvd"] = target_tvd
        row["selection_min_valid_rate"] = min_valid_rate
        row["selection_valid_rate_col"] = target_col
        return row

    eligible = stable[stable[target_col] >= min_valid_rate].copy()
    relaxed = False
    if eligible.empty:
        eligible = stable.copy()
        relaxed = True

    sort_cols = [target_col, "median_shots", "mean_shots", "max_tvd_to_reference", "median_tvd_to_reference"]
    ascending = [False, True, True, True, True]
    eligible = eligible.sort_values(sort_cols, ascending=ascending)
    row = eligible.iloc[0].to_dict()
    row["selection_relaxed"] = relaxed
    row["selection_rule"] = (
        f"valid_rate>={min_valid_rate:g}_then_min_median_shots" if not relaxed else "relaxed_max_valid_rate_then_min_median_shots"
    )
    row["selection_target_tvd"] = target_tvd
    row["selection_min_valid_rate"] = min_valid_rate
    row["selection_valid_rate_col"] = target_col
    return row


def prefix_columns(frame: pd.DataFrame, prefix: str, exclude: Optional[Sequence[str]] = None) -> pd.DataFrame:
    exclude_set = set(exclude or [])
    return frame.rename(columns={col: f"{prefix}{col}" for col in frame.columns if col not in exclude_set})


def run_validation_test_analysis(
    trace_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    splits: Sequence[SplitDefinition],
    validity_deltas: Sequence[float],
    selection_target_tvd: float,
    selection_min_valid_rate: float,
    selected_config_id: str,
) -> Dict[str, pd.DataFrame]:
    strategy_summary_frames: List[pd.DataFrame] = []
    selected_summary_rows: List[Dict[str, object]] = []
    selected_trace_frames: List[pd.DataFrame] = []
    selected_pairwise_frames: List[pd.DataFrame] = []
    selected_pairwise_summary_frames: List[pd.DataFrame] = []

    for split in splits:
        validation_ids = set(split.validation_trace_ids)
        test_ids = set(split.test_trace_ids)
        validation_trace_df = trace_df[trace_df["trace_id"].isin(validation_ids)].copy()
        test_trace_df = trace_df[trace_df["trace_id"].isin(test_ids)].copy()

        validation_summary = summarize_strategy_metrics(validation_trace_df, validity_deltas)
        test_summary_all = summarize_strategy_metrics(test_trace_df, validity_deltas)
        selected = select_stable_strategy(
            validation_summary,
            target_tvd=selection_target_tvd,
            min_valid_rate=selection_min_valid_rate,
            selected_config_id=selected_config_id,
        )
        selected_strategy = str(selected["strategy"])

        for role, summary in [("validation", validation_summary), ("test", test_summary_all)]:
            enriched = summary.copy()
            enriched.insert(0, "split_role", role)
            enriched.insert(0, "repetition", split.repetition)
            enriched.insert(0, "split_mode", split.mode)
            enriched.insert(0, "split_id", split.split_id)
            strategy_summary_frames.append(enriched)

        selected_test_trace = test_trace_df[test_trace_df["strategy"] == selected_strategy].copy()
        selected_validation_trace = validation_trace_df[validation_trace_df["strategy"] == selected_strategy].copy()
        selected_test_summary = summarize_strategy_metrics(selected_test_trace, validity_deltas)
        selected_validation_summary = summarize_strategy_metrics(selected_validation_trace, validity_deltas)

        if selected_test_summary.empty or selected_validation_summary.empty:
            raise RuntimeError(f"selected strategy {selected_strategy} produced no validation or test summary")

        validation_metrics = prefix_columns(selected_validation_summary.iloc[[0]].reset_index(drop=True), "validation_")
        test_metrics = prefix_columns(selected_test_summary.iloc[[0]].reset_index(drop=True), "test_")
        row: Dict[str, object] = {
            "split_id": split.split_id,
            "split_mode": split.mode,
            "repetition": split.repetition,
            "validation_trace_count": len(validation_ids),
            "test_trace_count": len(test_ids),
            "selected_strategy": selected_strategy,
            "selected_config_id": selected.get("config_id", ""),
            "selected_batch_size": selected.get("batch_size", ""),
            "selected_lookback_batches": selected.get("lookback_batches", ""),
            "selected_stability": selected.get("stability", ""),
            "selected_epsilon": selected.get("epsilon", ""),
            "selection_rule": selected.get("selection_rule", ""),
            "selection_relaxed": selected.get("selection_relaxed", False),
            "selection_target_tvd": selection_target_tvd,
            "selection_min_valid_rate": selection_min_valid_rate,
        }
        row.update(validation_metrics.iloc[0].to_dict())
        row.update(test_metrics.iloc[0].to_dict())
        selected_summary_rows.append(row)

        selected_test_trace.insert(0, "split_role", "test")
        selected_test_trace.insert(0, "repetition", split.repetition)
        selected_test_trace.insert(0, "split_mode", split.mode)
        selected_test_trace.insert(0, "split_id", split.split_id)
        selected_trace_frames.append(selected_test_trace)

        if not pairwise_df.empty:
            selected_pairwise = pairwise_df[
                pairwise_df["trace_id"].isin(test_ids) & (pairwise_df["strategy"] == selected_strategy)
            ].copy()
            if not selected_pairwise.empty:
                selected_pairwise.insert(0, "split_role", "test")
                selected_pairwise.insert(0, "repetition", split.repetition)
                selected_pairwise.insert(0, "split_mode", split.mode)
                selected_pairwise.insert(0, "split_id", split.split_id)
                selected_pairwise_frames.append(selected_pairwise)
                pairwise_summary = summarize_pairwise_metrics(selected_pairwise)
                pairwise_summary.insert(0, "split_role", "test")
                pairwise_summary.insert(0, "repetition", split.repetition)
                pairwise_summary.insert(0, "split_mode", split.mode)
                pairwise_summary.insert(0, "split_id", split.split_id)
                selected_pairwise_summary_frames.append(pairwise_summary)

    selected_summary = pd.DataFrame(selected_summary_rows)
    config_frequency = summarize_selected_config_frequency(selected_summary, selection_target_tvd)
    return {
        "validation_test_strategy_summary": pd.concat(strategy_summary_frames, ignore_index=True)
        if strategy_summary_frames
        else pd.DataFrame(),
        "validation_test_selected_summary": selected_summary,
        "validation_test_selected_trace_metrics": pd.concat(selected_trace_frames, ignore_index=True)
        if selected_trace_frames
        else pd.DataFrame(),
        "validation_test_selected_pairwise_metrics": pd.concat(selected_pairwise_frames, ignore_index=True)
        if selected_pairwise_frames
        else pd.DataFrame(),
        "validation_test_selected_pairwise_summary": pd.concat(selected_pairwise_summary_frames, ignore_index=True)
        if selected_pairwise_summary_frames
        else pd.DataFrame(),
        "validation_test_config_frequency": config_frequency,
    }


def summarize_selected_config_frequency(selected_summary: pd.DataFrame, selection_target_tvd: float) -> pd.DataFrame:
    if selected_summary.empty:
        return pd.DataFrame()
    target_col = f"test_{valid_rate_column(selection_target_tvd)}"
    agg: Dict[str, Tuple[str, str]] = {
        "splits_selected": ("split_id", "count"),
        "median_test_shots": ("test_median_shots", "median"),
        "mean_test_shots": ("test_mean_shots", "mean"),
        "median_test_tvd": ("test_median_tvd_to_reference", "median"),
        "mean_test_tvd": ("test_mean_tvd_to_reference", "mean"),
        "max_test_tvd_seen": ("test_max_tvd_to_reference", "max"),
        "median_test_stable_stop_rate": ("test_stable_stop_rate", "median"),
        "selection_relaxed_rate": ("selection_relaxed", "mean"),
    }
    if target_col in selected_summary.columns:
        agg[f"median_test_valid_rate_le_{format_float_for_column(selection_target_tvd)}"] = (target_col, "median")
        agg[f"min_test_valid_rate_le_{format_float_for_column(selection_target_tvd)}"] = (target_col, "min")
    out = selected_summary.groupby("selected_config_id", dropna=False).agg(**agg).reset_index()
    out["selection_frequency"] = out["splits_selected"] / selected_summary["split_id"].nunique()
    return out.sort_values(["splits_selected", "median_test_shots"], ascending=[False, True])


def evaluate_validation_test_scaled_bounds(
    trace_df: pd.DataFrame,
    fixed_baselines: Sequence[int],
    splits: Sequence[SplitDefinition],
    taus: Sequence[float],
    delta: float,
    calibrations: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    trace_table = build_bound_analysis_trace_table(trace_df, fixed_baselines)
    summary_rows: List[Dict[str, object]] = []
    detail_frames: List[pd.DataFrame] = []
    bound_specs = [("hoeffding", "n_hoeffding", "alpha_hoeffding"), ("weissman", "n_weissman", "alpha_weissman")]

    for split in splits:
        validation_ids = set(split.validation_trace_ids)
        test_ids = set(split.test_trace_ids)
        validation_table = trace_table[trace_table["trace_id"].isin(validation_ids)].copy()
        test_table = trace_table[trace_table["trace_id"].isin(test_ids)].copy()
        for tau in taus:
            validation_scaling = add_bound_and_scaling_columns(validation_table, fixed_baselines, tau, delta)
            test_scaling = add_bound_and_scaling_columns(test_table, fixed_baselines, tau, delta)
            for bound, bound_col, alpha_col in bound_specs:
                for calibration in calibrations:
                    alpha = choose_alpha(validation_scaling[alpha_col], calibration)
                    detail = apply_scaled_bound_policy(test_scaling, fixed_baselines, bound_col, tau, alpha)
                    detail["mode"] = "validation_test"
                    detail["split_id"] = split.split_id
                    detail["split_mode"] = split.mode
                    detail["repetition"] = split.repetition
                    detail["bound"] = bound
                    detail["bound_col"] = bound_col
                    detail["alpha_col"] = alpha_col
                    detail["calibration"] = calibration
                    detail["alpha_train"] = alpha
                    detail_frames.append(detail)
                    summary_rows.append(
                        scaled_bound_summary_from_detail(
                            detail,
                            mode="validation_test",
                            tau=tau,
                            delta=delta,
                            bound=bound,
                            bound_col=bound_col,
                            alpha_col=alpha_col,
                            calibration=calibration,
                            alpha=alpha,
                            calibration_trace_count=int(validation_scaling[alpha_col].dropna().count()),
                            extra={
                                "split_id": split.split_id,
                                "split_mode": split.mode,
                                "repetition": split.repetition,
                                "alpha_train": alpha,
                                "validation_trace_count": len(validation_table),
                            },
                        )
                    )
    return {
        "bound_validation_test_summary": pd.DataFrame(summary_rows),
        "bound_validation_test_trace_metrics": pd.concat(detail_frames, ignore_index=True)
        if detail_frames
        else pd.DataFrame(),
    }


def make_plots(
    trace_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    bound_results: Mapping[str, pd.DataFrame],
    validation_test_results: Mapping[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"could not import matplotlib; skipping plots: {exc}", file=sys.stderr)
        return

    plot_df = trace_df[trace_df["strategy_family"] != "reference"].copy()
    strategies = list(plot_df["strategy"].drop_duplicates())
    if strategies:
        for metric, ylabel, filename in [
            ("tvd_to_reference", "TVD to reference", "tvd_to_reference_by_strategy.png"),
            ("shots", "Shots used", "shots_by_strategy.png"),
            ("strategy_eval_time_seconds", "Strategy eval time (seconds)", "strategy_eval_time_by_strategy.png"),
        ]:
            values = [plot_df.loc[plot_df["strategy"] == strategy, metric].values for strategy in strategies]
            plt.figure(figsize=(max(8, len(strategies) * 0.55), 4))
            plt.boxplot(values, labels=strategies, showmeans=True)
            plt.ylabel(ylabel)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(output_dir / filename, dpi=200)
            plt.close()
    if not pairwise_df.empty:
        pairwise_df.groupby(["strategy", "baseline"])["stable_dominates"].mean().reset_index().to_csv(
            output_dir / "dominance_rates_for_plot.csv", index=False
        )
        pairwise_df.groupby(["strategy", "baseline"])["absolute_time_overhead_seconds"].median().reset_index().to_csv(
            output_dir / "median_absolute_time_overhead_for_plot.csv", index=False
        )
    if bound_results:
        if "bound_alpha_summary" in bound_results and not bound_results["bound_alpha_summary"].empty:
            bound_results["bound_alpha_summary"].to_csv(output_dir / "bound_alpha_summary_for_plot.csv", index=False)
        if "bound_leave_one_group_out_summary" in bound_results and not bound_results["bound_leave_one_group_out_summary"].empty:
            bound_results["bound_leave_one_group_out_summary"].to_csv(output_dir / "bound_loo_summary_for_plot.csv", index=False)
    if validation_test_results:
        freq = validation_test_results.get("validation_test_config_frequency", pd.DataFrame())
        if not freq.empty:
            freq.to_csv(output_dir / "validation_test_config_frequency_for_plot.csv", index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run StableShots and bound-scaling experiments on QSimBench")
    parser.add_argument("--algorithms", default=",".join(DEFAULT_ALGORITHMS))
    parser.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)))
    parser.add_argument("--backends", default=",".join(DEFAULT_BACKENDS))
    parser.add_argument("--circuit-kind", default="circuit", choices=["circuit", "mirror"])
    parser.add_argument("--fixed-baselines", default=",".join(map(str, FIXED_BASELINES)))
    parser.add_argument("--reference-shots", type=int, default=REFERENCE_SHOTS)
    parser.add_argument("--batch-size", default=str(StableShotsConfig().batch_size))
    parser.add_argument("--lookback-batches", default=str(StableShotsConfig().lookback_batches))
    parser.add_argument("--stability", default=str(StableShotsConfig().stability))
    parser.add_argument("--epsilon", default=str(StableShotsConfig().epsilon))
    parser.add_argument("--validity-deltas", default="0.01,0.05,0.10")
    parser.add_argument("--fidelity-margin", type=float, default=0.0)
    parser.add_argument("--source-batch-size", type=int, default=QSIMBENCH_SOURCE_BATCH_SIZE)
    parser.add_argument("--sampling-strategy", choices=["sequential", "random"], default="sequential")
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--skip-bound-analysis", action="store_true")
    parser.add_argument(
        "--bound-taus",
        default=None,
        help=(
            "Comma-separated TVD targets for Hoeffding/Weissman bound analysis. "
            "If omitted, defaults to the same values as --validity-deltas."
        ),
    )
    parser.add_argument(
        "--bound-delta",
        type=float,
        default=0.05,
        help=(
            "Failure probability for concentration bounds. Use one value to keep the "
            "confidence level fixed while sweeping multiple bound taus; 0.05 means 95%% confidence."
        ),
    )
    parser.add_argument("--bound-calibrations", default="median")
    parser.add_argument(
        "--validation-test-mode",
        choices=sorted(ALLOWED_VALIDATION_TEST_MODES),
        default="backend_holdout",
        help=(
            "Validation/test protocol. backend_holdout holds out a backend per algorithm-size cell; "
            "random uses a global random trace split; none disables split analysis."
        ),
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--split-repetitions", type=int, default=1)
    parser.add_argument("--selection-target-tvd", type=float, default=0.05)
    parser.add_argument("--selection-min-valid-rate", type=float, default=1.0)
    parser.add_argument(
        "--selected-config-id",
        default="",
        help=(
            "Optional fixed StableShots config_id or strategy name to evaluate in validation/test mode. "
            "If omitted, the script selects a config on validation traces."
        ),
    )
    parser.add_argument("--output-dir", default="results/stable_shots_grid")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--make-plots", action="store_true")
    return parser


def print_run_header(
    grid: StableShotsGrid,
    configs: Sequence[StableShotsConfig],
    args: argparse.Namespace,
    bound_taus: Sequence[float],
    calibrations: Sequence[str],
) -> None:
    print("StableShots grid:")
    print(f"  batch_size              = {list(grid.batch_sizes)}")
    print(f"  lookback_batches        = {list(grid.lookback_batches)}")
    print(f"  stability               = {list(grid.stabilities)}")
    print(f"  epsilon                 = {list(grid.epsilons)}")
    print(f"  max/reference           = {grid.max_shots}")
    print(f"  configurations          = {len(configs)}")
    print()
    print("QSimBench sampling:")
    print("  public API              = qsimbench.get_outcomes")
    print(f"  source_batch_size       = {args.source_batch_size}")
    print(f"  sampling_strategy       = {args.sampling_strategy}")
    print(f"  sampling_seed           = {args.sampling_seed}")
    print()
    print("Bound-scaling analysis:")
    print(f"  enabled                 = {not args.skip_bound_analysis}")
    print(f"  taus                    = {list(bound_taus)}")
    print(f"  delta                   = {args.bound_delta}")
    print(f"  calibrations            = {list(calibrations)}")
    print("  note                    = one delta fixes confidence while taus sweep TVD targets")
    print()
    print("Validation/test analysis:")
    print(f"  mode                    = {args.validation_test_mode}")
    print(f"  test_fraction           = {args.test_fraction}")
    print(f"  split_seed              = {args.split_seed}")
    print(f"  split_repetitions       = {args.split_repetitions}")
    print(f"  selection_target_tvd    = {args.selection_target_tvd}")
    print(f"  selection_min_valid_rate= {args.selection_min_valid_rate}")
    print(f"  selected_config_id      = {args.selected_config_id or '<selected from validation>'}")
    print()


def write_frame(frame: pd.DataFrame, output_dir: Path, filename: str) -> None:
    if frame is not None and not frame.empty:
        frame.to_csv(output_dir / filename, index=False)


def main() -> None:
    args = build_arg_parser().parse_args()
    algorithms = parse_csv_strings(args.algorithms)
    sizes = parse_csv_ints(args.sizes)
    backends = parse_csv_strings(args.backends)
    fixed_baselines = parse_csv_ints(args.fixed_baselines)
    validity_deltas = parse_csv_floats(args.validity_deltas)
    if args.selection_target_tvd not in validity_deltas:
        validity_deltas = sorted(set([*validity_deltas, args.selection_target_tvd]))
    bound_taus = parse_csv_floats(args.bound_taus) if args.bound_taus else list(validity_deltas)
    bound_calibrations = parse_csv_strings(args.bound_calibrations)
    grid = StableShotsGrid(
        batch_sizes=parse_csv_ints(args.batch_size),
        lookback_batches=parse_csv_ints(args.lookback_batches),
        stabilities=parse_csv_ints(args.stability),
        epsilons=parse_csv_floats(args.epsilon),
        max_shots=args.reference_shots,
    )
    validate_args(
        algorithms,
        sizes,
        backends,
        fixed_baselines,
        validity_deltas,
        grid,
        args.reference_shots,
        args.source_batch_size,
        args.fidelity_margin,
        bound_taus,
        args.bound_delta,
        bound_calibrations,
        args.validation_test_mode,
        args.test_fraction,
        args.split_repetitions,
        args.selection_target_tvd,
        args.selection_min_valid_rate,
    )
    configs = grid.expand()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print_run_header(grid, configs, args, bound_taus, bound_calibrations)

    specs = [
        TraceSpec(algorithm=algorithm, size=size, backend=backend, circuit_kind=args.circuit_kind)
        for algorithm in algorithms
        for size in sizes
        for backend in backends
    ]

    all_strategy_rows: List[Dict[str, object]] = []
    all_pairwise_rows: List[Dict[str, object]] = []
    failures: List[Tuple[str, str]] = []
    run_start = now_seconds()
    for index, spec in enumerate(specs, start=1):
        print(f"[{index}/{len(specs)}] {spec.trace_id}")
        try:
            strategy_rows, pairwise_rows = evaluate_trace(
                spec,
                configs,
                fixed_baselines,
                args.reference_shots,
                args.fidelity_margin,
                args.source_batch_size,
                args.sampling_strategy,
                args.sampling_seed,
                args.force_download,
            )
            all_strategy_rows.extend(strategy_rows)
            all_pairwise_rows.extend(pairwise_rows)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append((spec.trace_id, message))
            print(f"  skipped: {message}", file=sys.stderr)
    total_run_time_seconds = now_seconds() - run_start

    if not all_strategy_rows:
        raise RuntimeError("no traces were successfully evaluated")

    trace_df = pd.DataFrame(all_strategy_rows)
    pairwise_df = pd.DataFrame(all_pairwise_rows)
    summary_strategy_df = summarize_strategy_metrics(trace_df, validity_deltas)
    summary_pairwise_df = summarize_pairwise_metrics(pairwise_df)

    bound_results: Dict[str, pd.DataFrame] = {}
    if not args.skip_bound_analysis:
        bound_results = run_scaled_bound_analysis(
            trace_df=trace_df,
            fixed_baselines=fixed_baselines,
            taus=bound_taus,
            delta=args.bound_delta,
            calibrations=bound_calibrations,
            group_cols=BOUND_GROUPS,
        )

    splits = make_validation_test_splits(
        specs=specs,
        mode=args.validation_test_mode,
        test_fraction=args.test_fraction,
        repetitions=args.split_repetitions,
        seed=args.split_seed,
    )
    split_assignments_df = build_split_assignments(splits, specs) if splits else pd.DataFrame()
    validation_test_results: Dict[str, pd.DataFrame] = {}
    if splits:
        validation_test_results = run_validation_test_analysis(
            trace_df=trace_df,
            pairwise_df=pairwise_df,
            splits=splits,
            validity_deltas=validity_deltas,
            selection_target_tvd=args.selection_target_tvd,
            selection_min_valid_rate=args.selection_min_valid_rate,
            selected_config_id=args.selected_config_id,
        )
        if not args.skip_bound_analysis:
            validation_test_results.update(
                evaluate_validation_test_scaled_bounds(
                    trace_df=trace_df,
                    fixed_baselines=fixed_baselines,
                    splits=splits,
                    taus=bound_taus,
                    delta=args.bound_delta,
                    calibrations=bound_calibrations,
                )
            )

    trace_df.to_csv(output_dir / "trace_metrics.csv", index=False)
    pairwise_df.to_csv(output_dir / "pairwise_metrics.csv", index=False)
    summary_strategy_df.to_csv(output_dir / "summary_by_strategy.csv", index=False)
    summary_pairwise_df.to_csv(output_dir / "summary_pairwise.csv", index=False)
    for name, frame in bound_results.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    if not split_assignments_df.empty:
        split_assignments_df.to_csv(output_dir / "split_assignments.csv", index=False)
    for name, frame in validation_test_results.items():
        write_frame(frame, output_dir, f"{name}.csv")

    pd.DataFrame(
        [
            {
                "total_run_time_seconds": total_run_time_seconds,
                "successful_trace_count": trace_df["trace_id"].nunique(),
                "failed_trace_count": len(failures),
                "strategy_row_count": len(trace_df),
                "pairwise_row_count": len(pairwise_df),
                "reference_shots": args.reference_shots,
                "source_batch_size": args.source_batch_size,
                "sampling_strategy": args.sampling_strategy,
                "sampling_seed": args.sampling_seed,
                "fixed_baselines": ",".join(str(v) for v in fixed_baselines),
                "validity_deltas": ",".join(str(v) for v in validity_deltas),
                "bound_analysis_enabled": not args.skip_bound_analysis,
                "bound_taus": ",".join(str(v) for v in bound_taus),
                "bound_delta": args.bound_delta,
                "bound_calibrations": ",".join(bound_calibrations),
                "validation_test_mode": args.validation_test_mode,
                "test_fraction": args.test_fraction,
                "split_seed": args.split_seed,
                "split_repetitions": args.split_repetitions,
                "selection_target_tvd": args.selection_target_tvd,
                "selection_min_valid_rate": args.selection_min_valid_rate,
                "selected_config_id": args.selected_config_id,
            }
        ]
    ).to_csv(output_dir / "run_metadata.csv", index=False)

    if failures:
        pd.DataFrame(failures, columns=["trace_id", "error"]).to_csv(output_dir / "failures.csv", index=False)
    if args.make_plots:
        make_plots(trace_df, pairwise_df, bound_results, validation_test_results, output_dir)

    print()
    print("Wrote results to:")
    print(f"  {output_dir.resolve()}")
    print()
    print("Total run time:")
    print(f"  {total_run_time_seconds:.6f} seconds")
    print()
    print("Main strategy summary:")
    print(summary_strategy_df.to_string(index=False))
    print()
    print("Pairwise StableShots-vs-fixed summary:")
    print("  No pairwise rows produced." if summary_pairwise_df.empty else summary_pairwise_df.to_string(index=False))
    if bound_results:
        print()
        print("Bound alpha variability summary:")
        print(bound_results["bound_alpha_summary"].to_string(index=False))
        print()
        print("Global scaled-bound summary:")
        print(bound_results["bound_global_scaled_summary"].to_string(index=False))
        print()
        print("Leave-one-group-out scaled-bound summary:")
        print(bound_results["bound_leave_one_group_out_summary"].to_string(index=False))
    if validation_test_results:
        print()
        print("Validation/test selected configuration summary:")
        print(validation_test_results["validation_test_selected_summary"].to_string(index=False))
        freq = validation_test_results.get("validation_test_config_frequency", pd.DataFrame())
        if not freq.empty:
            print()
            print("Validation/test selected configuration frequency:")
            print(freq.to_string(index=False))
        bound_vt = validation_test_results.get("bound_validation_test_summary", pd.DataFrame())
        if not bound_vt.empty:
            print()
            print("Validation-trained/test-evaluated scaled-bound summary:")
            print(bound_vt.to_string(index=False))
    if failures:
        print()
        print(f"Skipped {len(failures)} trace(s). See failures.csv for details.")


if __name__ == "__main__":
    main()
