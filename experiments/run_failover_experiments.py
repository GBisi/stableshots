#!/usr/bin/env python3
"""
Single-entry failover experiment runner for StableShots.

The runner materializes QSimBench traces once, constructs a high-shot Aer ideal
reference, ranks QPUs only for oracle-controlled experimental conditions, and
runs:
  1. no-failure and fixed-shot baselines,
  2. exhaustive directed single-failure handoffs,
  3. deterministic two-failure reliability-order stress tests,
  4. stochastic per-batch failure stress tests.

All experiment choices live in experiments/failover_config.json.  The runner
does not implement a production QPU scheduler: oracle reliability order is an
experimental control derived from TVD to the Aer reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from stableshot.main import (
    StableShotsConfig,
    TraceSpec,
    iter_execution_batches,
    load_qsimbench_batches,
    prefix_counts,
    run_stable_shots,
)

Counts = Counter[str]
WeightedCounts = Counter[str]
Batch = Tuple[int, Counts]


def load_config(path: Path) -> Dict[str, object]:
    cfg = json.loads(path.read_text())
    required = [
        "experiment_name",
        "output_dir",
        "algorithms",
        "sizes",
        "backends",
        "ideal_backend",
        "ideal_reference_shots",
        "backend_reference_shots",
        "source_batch_size",
        "sampling_strategy",
        "sampling_seed",
        "reference_seed",
        "stableshots",
        "single_failure",
        "analysis",
    ]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"missing configuration keys: {missing}")
    return cfg


def auto_publish_results(
    cfg: Mapping[str, object],
    paths: Sequence[Path],
    stage: str,
) -> None:
    """Commit and push only generated result paths after a successful stage."""
    publish = cfg.get("git_publish", {})
    if not isinstance(publish, Mapping) or not bool(publish.get("enabled", False)):
        return

    disabled = os.getenv("STABLESHOTS_DISABLE_AUTO_PUSH", "").strip().lower()
    if disabled in {"1", "true", "yes", "on"}:
        print("automatic result push disabled by STABLESHOTS_DISABLE_AUTO_PUSH", flush=True)
        return

    expected_branch = str(publish.get("branch", "")).strip()
    remote = str(publish.get("remote", "origin")).strip() or "origin"
    if not expected_branch:
        raise ValueError("git_publish.branch must be set when automatic publishing is enabled")

    try:
        root_text = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("automatic result push requires execution inside a Git repository") from exc

    repo_root = Path(root_text).resolve()
    current_branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_branch != expected_branch:
        raise RuntimeError(
            f"refusing to auto-push results from branch {current_branch!r}; "
            f"configured branch is {expected_branch!r}"
        )

    relative_paths: List[str] = []
    for path in paths:
        resolved = path.resolve()
        try:
            relative_paths.append(str(resolved.relative_to(repo_root)))
        except ValueError as exc:
            raise RuntimeError(f"result path {resolved} is outside repository {repo_root}") from exc

    subprocess.run(["git", "add", "--", *relative_paths], cwd=repo_root, check=True)
    diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", *relative_paths],
        cwd=repo_root,
        check=False,
    )
    if diff.returncode == 0:
        print(f"no new {stage} result changes to publish", flush=True)
        return
    if diff.returncode != 1:
        raise RuntimeError(f"git diff failed while preparing automatic {stage} result publication")

    message_key = f"{stage}_commit_message"
    default_message = f"Add failover {stage} results"
    message = str(publish.get(message_key, default_message)).strip() or default_message
    subprocess.run(
        ["git", "commit", "-m", message, "--", *relative_paths],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(
        ["git", "push", remote, f"HEAD:refs/heads/{expected_branch}"],
        cwd=repo_root,
        check=True,
    )
    print(
        f"published {stage} results to {remote}/{expected_branch}",
        flush=True,
    )


def deterministic_seed(base: int, *parts: object) -> int:
    payload = "|".join([str(base), *(str(p) for p in parts)])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def total_mass(counts: Mapping[str, float]) -> float:
    return float(sum(float(v) for v in counts.values()))


def merge_weighted(target: MutableMapping[str, float], source: Mapping[str, float], weight: float = 1.0) -> None:
    for outcome, value in source.items():
        target[str(outcome)] = float(target.get(str(outcome), 0.0)) + weight * float(value)


def scale_counts(counts: Mapping[str, float], factor: float) -> WeightedCounts:
    out: WeightedCounts = Counter()
    for outcome, value in counts.items():
        weighted = factor * float(value)
        if weighted > 0:
            out[str(outcome)] = weighted
    return out


def tvd_weighted(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    nl = total_mass(left)
    nr = total_mass(right)
    if nl <= 0 or nr <= 0:
        raise ValueError("TVD requires non-empty distributions")
    support = set(left) | set(right)
    return 0.5 * sum(abs(float(left.get(x, 0.0)) / nl - float(right.get(x, 0.0)) / nr) for x in support)


def counts_to_probs(counts: Mapping[str, float]) -> Dict[str, float]:
    n = total_mass(counts)
    if n <= 0:
        return {}
    return {str(k): float(v) / n for k, v in counts.items() if float(v) > 0}


def sample_from_probs(probs: Mapping[str, float], shots: int, rng: np.random.Generator) -> Counts:
    if shots <= 0:
        return Counter()
    keys = sorted(probs)
    if not keys:
        return Counter()
    p = np.asarray([float(probs[k]) for k in keys], dtype=float)
    p = p / p.sum()
    draw = rng.multinomial(shots, p)
    return Counter({k: int(v) for k, v in zip(keys, draw) if int(v) > 0})


def execution_batches(raw_batches: Sequence[Batch], config: StableShotsConfig, max_shots: int) -> List[Batch]:
    return list(iter_execution_batches(raw_batches, config.batch_size, max_shots))


@dataclass
class Controller:
    config: StableShotsConfig
    counts: WeightedCounts = field(default_factory=Counter)
    snapshots: List[WeightedCounts] = field(default_factory=list)
    stable_checks: int = 0
    checks_performed: int = 0
    last_delta: float = float("nan")
    stopped: bool = False
    stop_reason: str = ""
    processed_new_shots: int = 0

    def reset_decision_state(self) -> None:
        self.snapshots = []
        self.stable_checks = 0
        self.checks_performed = 0
        self.last_delta = float("nan")
        self.stopped = False
        self.stop_reason = ""
        self.processed_new_shots = 0

    def clone(self) -> "Controller":
        return Controller(
            config=self.config,
            counts=Counter(self.counts),
            snapshots=[Counter(s) for s in self.snapshots],
            stable_checks=self.stable_checks,
            checks_performed=self.checks_performed,
            last_delta=self.last_delta,
            stopped=self.stopped,
            stop_reason=self.stop_reason,
            processed_new_shots=self.processed_new_shots,
        )

    def apply_batch(self, batch_counts: Mapping[str, int], batch_shots: int, enforce_budget: Optional[int] = None) -> bool:
        if self.stopped:
            return True
        merge_weighted(self.counts, batch_counts)
        self.processed_new_shots += int(batch_shots)
        self.snapshots.append(Counter(self.counts))
        if len(self.snapshots) > self.config.lookback_batches:
            previous = self.snapshots[-1 - self.config.lookback_batches]
            self.last_delta = tvd_weighted(self.counts, previous)
            self.checks_performed += 1
            self.stable_checks = self.stable_checks + 1 if self.last_delta <= self.config.epsilon else 0
            if self.stable_checks >= self.config.stability:
                self.stopped = True
                self.stop_reason = "stable"
                return True
        if enforce_budget is not None and self.processed_new_shots >= enforce_budget:
            self.stopped = True
            self.stop_reason = "recovery_budget"
            return True
        return False


def run_controller(
    controller: Controller,
    batches: Sequence[Batch],
    max_new_shots: int,
    start_index: int = 0,
) -> Tuple[int, int]:
    consumed = 0
    used_batches = 0
    for idx in range(start_index, len(batches)):
        if consumed >= max_new_shots:
            break
        batch_shots, batch_counts = batches[idx]
        if consumed + batch_shots > max_new_shots:
            break
        consumed += batch_shots
        used_batches += 1
        if controller.apply_batch(batch_counts, batch_shots, enforce_budget=max_new_shots):
            break
    if not controller.stopped:
        controller.stopped = True
        controller.stop_reason = "recovery_budget" if consumed >= max_new_shots else "input_exhausted"
    return consumed, used_batches


def replay_prefix_controller(batches: Sequence[Batch], config: StableShotsConfig, failure_shots: int) -> Tuple[Controller, Counts, int]:
    controller = Controller(config=config)
    physical: Counts = Counter()
    consumed = 0
    for batch_shots, batch_counts in batches:
        if consumed + batch_shots > failure_shots:
            break
        merge_weighted(physical, batch_counts)
        controller.apply_batch(batch_counts, batch_shots)
        consumed += batch_shots
        if consumed >= failure_shots:
            break
    controller.stopped = False
    controller.stop_reason = ""
    return controller, physical, consumed


def rounded_failure_shots(no_failure_shots: int, fraction: float, batch_size: int) -> int:
    if not (0 < fraction < 1):
        raise ValueError(f"failure fraction must be in (0,1), got {fraction}")
    raw = int(math.floor(no_failure_shots * fraction / batch_size)) * batch_size
    raw = max(batch_size, raw)
    if no_failure_shots > batch_size:
        raw = min(raw, no_failure_shots - batch_size)
    return raw


def bootstrap_shift(
    old_counts: Mapping[str, float],
    probe_counts: Mapping[str, int],
    repetitions: int,
    seed: int,
    minimum_null_tvd: float,
) -> Dict[str, float]:
    probe_shots = int(sum(int(v) for v in probe_counts.values()))
    if probe_shots <= 0:
        return {
            "shift_handoff_mean_tvd": float("nan"),
            "shift_null_mean_tvd": float("nan"),
            "shift_corrected_tvd": float("nan"),
            "shift_null_percentile": float("nan"),
            "adaptive_lambda": 0.0,
        }
    probs = counts_to_probs(old_counts)
    rng = np.random.default_rng(seed)
    handoff: List[float] = []
    null: List[float] = []
    for _ in range(repetitions):
        old_a = sample_from_probs(probs, probe_shots, rng)
        old_b = sample_from_probs(probs, probe_shots, rng)
        handoff.append(tvd_weighted(old_a, probe_counts))
        null.append(tvd_weighted(old_a, old_b))
    handoff_mean = float(np.mean(handoff))
    null_mean = float(np.mean(null))
    corrected = max(0.0, handoff_mean - null_mean)
    percentile = float(np.mean(np.asarray(null) <= handoff_mean))
    denom = max(handoff_mean, float(minimum_null_tvd))
    adaptive_lambda = float(np.clip(null_mean / denom, 0.0, 1.0))
    return {
        "shift_handoff_mean_tvd": handoff_mean,
        "shift_null_mean_tvd": null_mean,
        "shift_corrected_tvd": corrected,
        "shift_null_percentile": percentile,
        "adaptive_lambda": adaptive_lambda,
    }


def process_probe(
    old_counts: Mapping[str, float],
    target_batches: Sequence[Batch],
    config: StableShotsConfig,
    probe_batches: int,
    bootstrap_repetitions: int,
    seed: int,
    minimum_null_tvd: float,
) -> Tuple[Controller, Counts, int, Dict[str, float]]:
    actual_probe = min(probe_batches, len(target_batches))
    probe_counts: Counts = Counter()
    probe_shots = 0
    for batch_shots, batch_counts in target_batches[:actual_probe]:
        merge_weighted(probe_counts, batch_counts)
        probe_shots += batch_shots
    shift = bootstrap_shift(
        old_counts=old_counts,
        probe_counts=probe_counts,
        repetitions=bootstrap_repetitions,
        seed=seed,
        minimum_null_tvd=minimum_null_tvd,
    )
    lam = float(shift["adaptive_lambda"])
    controller = Controller(config=config, counts=scale_counts(old_counts, lam))
    for batch_shots, batch_counts in target_batches[:actual_probe]:
        controller.apply_batch(batch_counts, batch_shots)
    controller.stopped = False if not controller.stop_reason else controller.stopped
    return controller, probe_counts, probe_shots, shift


def make_policy_names(fixed_lambdas: Sequence[float]) -> List[str]:
    names = ["full_restart", "naive_continuation", "controller_reset", "epoch_local"]
    for lam in fixed_lambdas:
        names.append(f"fixed_decay_{float(lam):g}")
    names.append("shift_aware_decay")
    return names


def parse_fixed_lambda(policy: str) -> Optional[float]:
    prefix = "fixed_decay_"
    if not policy.startswith(prefix):
        return None
    return float(policy[len(prefix):])


def final_counts_for_policy(
    policy: str,
    source_physical: Counts,
    target_physical: Counts,
    controller: Controller,
) -> WeightedCounts:
    if policy == "full_restart":
        return Counter(target_physical)
    if policy == "epoch_local":
        out: WeightedCounts = Counter()
        merge_weighted(out, source_physical)
        merge_weighted(out, target_physical)
        return out
    return Counter(controller.counts)


def simulate_single_handoff(
    *,
    circuit_key: str,
    algorithm: str,
    size: int,
    source_backend: str,
    target_backend: str,
    source_batches: Sequence[Batch],
    target_batches: Sequence[Batch],
    source_reference: Counts,
    target_reference: Counts,
    ideal_reference: Counts,
    no_failure_counts: Counts,
    no_failure_shots: int,
    no_failure_tvd: float,
    failure_fraction: float,
    policy: str,
    config: StableShotsConfig,
    max_new_shots: int,
    shift_cfg: Mapping[str, object],
    seed_base: int,
) -> Dict[str, object]:
    failure_shots = rounded_failure_shots(no_failure_shots, failure_fraction, config.batch_size)
    source_controller, source_physical, consumed_source = replay_prefix_controller(
        source_batches, config, failure_shots
    )
    if consumed_source != failure_shots:
        raise RuntimeError(f"could not materialize failure prefix {failure_shots} for {circuit_key}/{source_backend}")
    target_physical: Counts = Counter()
    target_shots = 0
    target_batch_count = 0
    shift = {
        "shift_handoff_mean_tvd": float("nan"),
        "shift_null_mean_tvd": float("nan"),
        "shift_corrected_tvd": float("nan"),
        "shift_null_percentile": float("nan"),
        "adaptive_lambda": float("nan"),
    }

    lam = parse_fixed_lambda(policy)
    if policy == "naive_continuation":
        controller = source_controller.clone()
        controller.processed_new_shots = 0
        consumed, used = run_controller(controller, target_batches, max_new_shots)
        for batch_shots, batch_counts in target_batches[:used]:
            merge_weighted(target_physical, batch_counts)
            target_shots += batch_shots
        target_batch_count = used
    elif policy == "controller_reset":
        controller = Controller(config=config, counts=Counter(source_physical))
        consumed, used = run_controller(controller, target_batches, max_new_shots)
        for batch_shots, batch_counts in target_batches[:used]:
            merge_weighted(target_physical, batch_counts)
            target_shots += batch_shots
        target_batch_count = used
    elif policy == "epoch_local":
        controller = Controller(config=config)
        consumed, used = run_controller(controller, target_batches, max_new_shots)
        for batch_shots, batch_counts in target_batches[:used]:
            merge_weighted(target_physical, batch_counts)
            target_shots += batch_shots
        target_batch_count = used
    elif policy == "full_restart":
        controller = Controller(config=config)
        consumed, used = run_controller(controller, target_batches, max_new_shots)
        for batch_shots, batch_counts in target_batches[:used]:
            merge_weighted(target_physical, batch_counts)
            target_shots += batch_shots
        target_batch_count = used
    elif lam is not None:
        controller = Controller(config=config, counts=scale_counts(source_physical, lam))
        consumed, used = run_controller(controller, target_batches, max_new_shots)
        for batch_shots, batch_counts in target_batches[:used]:
            merge_weighted(target_physical, batch_counts)
            target_shots += batch_shots
        target_batch_count = used
        shift["adaptive_lambda"] = lam
    elif policy == "shift_aware_decay":
        probe_batches = int(shift_cfg["probe_batches"])
        repetitions = int(shift_cfg["bootstrap_repetitions"])
        minimum_null = float(shift_cfg.get("minimum_null_tvd", 1e-6))
        seed = deterministic_seed(seed_base, circuit_key, source_backend, target_backend, failure_fraction, policy)
        controller, probe_counts, probe_shots, shift = process_probe(
            source_physical,
            target_batches,
            config,
            probe_batches,
            repetitions,
            seed,
            minimum_null,
        )
        actual_probe_batches = min(probe_batches, len(target_batches))
        merge_weighted(target_physical, probe_counts)
        target_shots += probe_shots
        target_batch_count += actual_probe_batches
        remaining_budget = max(0, max_new_shots - probe_shots)
        if not controller.stopped and remaining_budget > 0:
            # The probe is already accounted for in target_shots. Reset only the
            # recovery-budget counter so run_controller may consume exactly the
            # remaining post-probe budget while retaining reconstructed snapshots.
            controller.processed_new_shots = 0
            consumed, used = run_controller(
                controller,
                target_batches,
                remaining_budget,
                start_index=actual_probe_batches,
            )
            for batch_shots, batch_counts in target_batches[actual_probe_batches:actual_probe_batches + used]:
                merge_weighted(target_physical, batch_counts)
                target_shots += batch_shots
            target_batch_count += used
    else:
        raise ValueError(f"unsupported policy {policy}")

    final_counts = final_counts_for_policy(policy, source_physical, target_physical, controller)
    total_physical = consumed_source + target_shots
    effective_mass = total_mass(final_counts)
    final_tvd = tvd_weighted(final_counts, ideal_reference)
    source_error = tvd_weighted(source_reference, ideal_reference)
    target_error = tvd_weighted(target_reference, ideal_reference)
    pair_shift = tvd_weighted(source_reference, target_reference)
    return {
        "experiment": "single_failure",
        "circuit_key": circuit_key,
        "algorithm": algorithm,
        "size": int(size),
        "source_backend": source_backend,
        "target_backend": target_backend,
        "policy": policy,
        "failure_fraction": float(failure_fraction),
        "failure_shots": int(consumed_source),
        "post_failure_shots": int(target_shots),
        "physical_shots_total": int(total_physical),
        "effective_retained_shots": float(effective_mass),
        "discarded_or_downweighted_shots": float(total_physical - effective_mass),
        "target_batches_consumed": int(target_batch_count),
        "stop_reason": controller.stop_reason,
        "stability_checks": int(controller.checks_performed),
        "last_stability_tvd": float(controller.last_delta),
        "final_tvd_to_aer": float(final_tvd),
        "no_failure_tvd_to_aer": float(no_failure_tvd),
        "delta_tvd_vs_no_failure": float(final_tvd - no_failure_tvd),
        "no_failure_shots": int(no_failure_shots),
        "shot_overhead_vs_no_failure": int(total_physical - no_failure_shots),
        "shot_ratio_vs_no_failure": float(total_physical / no_failure_shots),
        "target_violation": bool(final_tvd > 0.0),  # overwritten by caller using configured threshold
        "source_tvd_to_aer": float(source_error),
        "target_tvd_to_aer": float(target_error),
        "quality_change": float(target_error - source_error),
        "pair_reference_tvd": float(pair_shift),
        **shift,
    }


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
    return Counter({str(k): int(v) for k, v in counts.items() if int(v) > 0})


def materialize_circuit(
    algorithm: str,
    size: int,
    cfg: Mapping[str, object],
    stable_cfg: StableShotsConfig,
) -> Tuple[Counts, Dict[str, List[Batch]], Dict[str, Counts]]:
    circuit_kind = str(cfg["circuit_kind"])
    force = bool(cfg.get("force_download", False))
    ideal_seed = deterministic_seed(int(cfg["reference_seed"]), algorithm, size, cfg["ideal_backend"])
    ideal = get_ideal_reference(
        algorithm,
        size,
        str(cfg["ideal_backend"]),
        int(cfg["ideal_reference_shots"]),
        circuit_kind,
        ideal_seed,
        force,
    )
    required = max(
        int(cfg["backend_reference_shots"]),
        stable_cfg.max_shots,
        int(cfg["single_failure"]["recovery_max_new_shots"]),  # type: ignore[index]
    )
    streams: Dict[str, List[Batch]] = {}
    references: Dict[str, Counts] = {}
    for backend in cfg["backends"]:  # type: ignore[assignment]
        spec = TraceSpec(algorithm=algorithm, size=size, backend=str(backend), circuit_kind=circuit_kind)
        raw = load_qsimbench_batches(
            spec=spec,
            total_shots=required,
            source_batch_size=int(cfg["source_batch_size"]),
            sampling_strategy=str(cfg["sampling_strategy"]),
            sampling_seed=int(cfg["sampling_seed"]),
            force=force,
        )
        streams[str(backend)] = execution_batches(raw, stable_cfg, required)
        references[str(backend)] = prefix_counts(raw, int(cfg["backend_reference_shots"]))
    return ideal, streams, references


def run_no_failure(
    algorithm: str,
    size: int,
    backend: str,
    batches: Sequence[Batch],
    ideal: Counts,
    config: StableShotsConfig,
    target_tvd: float,
) -> Tuple[Dict[str, object], Counts]:
    raw = [(shots, Counter(counts)) for shots, counts in batches]
    counts, shots, last_delta, reason = run_stable_shots(raw, config)
    value = tvd_weighted(counts, ideal)
    return {
        "experiment": "no_failure",
        "circuit_key": f"{algorithm}_{size}",
        "algorithm": algorithm,
        "size": int(size),
        "backend": backend,
        "shots": int(shots),
        "final_tvd_to_aer": float(value),
        "target_violation": bool(value > target_tvd),
        "stop_reason": reason,
        "last_stability_tvd": float(last_delta),
    }, counts


def fixed_shot_rows(
    algorithm: str,
    size: int,
    backend: str,
    batches: Sequence[Batch],
    ideal: Counts,
    budgets: Sequence[int],
    target_tvd: float,
) -> List[Dict[str, object]]:
    rows = []
    flat_raw = [(shots, Counter(counts)) for shots, counts in batches]
    for budget in budgets:
        counts = prefix_counts(flat_raw, int(budget))
        value = tvd_weighted(counts, ideal)
        rows.append({
            "experiment": "fixed_shot",
            "circuit_key": f"{algorithm}_{size}",
            "algorithm": algorithm,
            "size": int(size),
            "backend": backend,
            "shots": int(budget),
            "final_tvd_to_aer": float(value),
            "target_violation": bool(value > target_tvd),
        })
    return rows


def reference_rows(
    algorithm: str,
    size: int,
    references: Mapping[str, Counts],
    ideal: Counts,
) -> List[Dict[str, object]]:
    rows = []
    for backend, counts in references.items():
        rows.append({
            "circuit_key": f"{algorithm}_{size}",
            "algorithm": algorithm,
            "size": int(size),
            "backend": backend,
            "reference_tvd_to_aer": tvd_weighted(counts, ideal),
        })
    return rows


def build_sequence(
    references: Mapping[str, Counts],
    ideal: Counts,
    condition: str,
    seed: int,
) -> List[str]:
    backends = list(references)
    errors = {b: tvd_weighted(references[b], ideal) for b in backends}
    if condition == "ascending_reliability":
        return sorted(backends, key=lambda b: errors[b], reverse=True)  # worst -> best
    if condition == "descending_reliability":
        return sorted(backends, key=lambda b: errors[b])  # best -> worst
    if condition == "random":
        rng = random.Random(seed)
        order = backends[:]
        rng.shuffle(order)
        return order
    raise ValueError(f"unknown sequence condition {condition}")


def policy_after_handoff(
    policy: str,
    config: StableShotsConfig,
    previous_controller: Controller,
    accumulated_output: WeightedCounts,
    old_effective_counts: WeightedCounts,
    new_batches: Sequence[Batch],
    shift_cfg: Mapping[str, object],
    seed: int,
) -> Tuple[Controller, WeightedCounts, int, int, Dict[str, float]]:
    shift = {
        "shift_handoff_mean_tvd": float("nan"),
        "shift_null_mean_tvd": float("nan"),
        "shift_corrected_tvd": float("nan"),
        "shift_null_percentile": float("nan"),
        "adaptive_lambda": float("nan"),
    }
    if policy == "naive_continuation":
        controller = previous_controller.clone()
        controller.stopped = False
        controller.stop_reason = ""
        controller.processed_new_shots = 0
        return controller, accumulated_output, 0, 0, shift
    if policy == "controller_reset":
        return Controller(config=config, counts=Counter(old_effective_counts)), accumulated_output, 0, 0, shift
    if policy == "full_restart":
        return Controller(config=config), Counter(), 0, 0, shift
    lam = parse_fixed_lambda(policy)
    if lam is not None:
        shift["adaptive_lambda"] = lam
        return Controller(config=config, counts=scale_counts(old_effective_counts, lam)), scale_counts(accumulated_output, lam), 0, 0, shift
    if policy == "shift_aware_decay":
        probe_batches = int(shift_cfg["probe_batches"])
        actual_probe = min(probe_batches, len(new_batches))
        controller, probe_counts, probe_shots, shift = process_probe(
            old_effective_counts,
            new_batches,
            config,
            actual_probe,
            int(shift_cfg["bootstrap_repetitions"]),
            seed,
            float(shift_cfg.get("minimum_null_tvd", 1e-6)),
        )
        output = scale_counts(accumulated_output, float(shift["adaptive_lambda"]))
        merge_weighted(output, probe_counts)
        return controller, output, probe_shots, actual_probe, shift
    raise ValueError(f"policy {policy!r} unsupported in repeated-failure experiment")


def simulate_deterministic_sequence(
    *,
    algorithm: str,
    size: int,
    sequence: Sequence[str],
    streams: Mapping[str, Sequence[Batch]],
    references: Mapping[str, Counts],
    ideal: Counts,
    no_failure_stop_by_backend: Mapping[str, int],
    config: StableShotsConfig,
    policy: str,
    condition: str,
    thresholds: Sequence[float],
    max_new_shots: int,
    shift_cfg: Mapping[str, object],
    seed: int,
    target_tvd: float,
) -> Dict[str, object]:
    start = sequence[0]
    n0 = int(no_failure_stop_by_backend[start])
    failure_targets = [rounded_failure_shots(n0, float(f), config.batch_size) for f in thresholds]
    max_failures = min(len(failure_targets), len(sequence) - 1)
    controller = Controller(config=config)
    output: WeightedCounts = Counter()
    physical_total = 0
    actual_failures = 0
    epoch_shots: List[int] = []
    handoff_lambdas: List[float] = []
    handoff_shifts: List[float] = []
    backend_index = 0
    batch_index = 0
    epoch_physical = 0
    source_stream = streams[start]

    while backend_index < len(sequence):
        if controller.stopped:
            break
        backend = sequence[backend_index]
        batches = streams[backend]
        if batch_index >= len(batches):
            break
        batch_shots, batch_counts = batches[batch_index]
        controller.apply_batch(batch_counts, batch_shots)
        merge_weighted(output, batch_counts)
        physical_total += batch_shots
        epoch_physical += batch_shots
        batch_index += 1

        next_failure_target = failure_targets[actual_failures] if actual_failures < max_failures else None
        should_fail = next_failure_target is not None and physical_total >= next_failure_target

        if controller.stopped and not should_fail:
            break
        if should_fail:
            epoch_shots.append(epoch_physical)
            old_effective = Counter(controller.counts)
            actual_failures += 1
            backend_index += 1
            if backend_index >= len(sequence):
                break
            next_backend = sequence[backend_index]
            next_batches = streams[next_backend]
            controller, output, probe_shots, probe_batches, shift = policy_after_handoff(
                policy,
                config,
                controller,
                output,
                old_effective,
                next_batches,
                shift_cfg,
                deterministic_seed(seed, algorithm, size, condition, policy, actual_failures),
            )
            physical_total += probe_shots
            epoch_physical = probe_shots
            batch_index = probe_batches
            if math.isfinite(float(shift["adaptive_lambda"])):
                handoff_lambdas.append(float(shift["adaptive_lambda"]))
            if math.isfinite(float(shift["shift_corrected_tvd"])):
                handoff_shifts.append(float(shift["shift_corrected_tvd"]))
            continue

        if epoch_physical >= max_new_shots and actual_failures >= max_failures:
            controller.stopped = True
            controller.stop_reason = "recovery_budget"
            break

    epoch_shots.append(epoch_physical)
    final_counts = Counter(output if policy != "naive_continuation" else controller.counts)
    final_tvd = tvd_weighted(final_counts, ideal)
    return {
        "experiment": "multi_failure",
        "circuit_key": f"{algorithm}_{size}",
        "algorithm": algorithm,
        "size": int(size),
        "sequence_condition": condition,
        "sequence": json.dumps(list(sequence)),
        "policy": policy,
        "planned_failures": int(max_failures),
        "actual_failures": int(actual_failures),
        "failure_targets": json.dumps(failure_targets[:max_failures]),
        "epoch_physical_shots": json.dumps(epoch_shots),
        "physical_shots_total": int(physical_total),
        "effective_retained_shots": float(total_mass(final_counts)),
        "discarded_or_downweighted_shots": float(physical_total - total_mass(final_counts)),
        "final_tvd_to_aer": float(final_tvd),
        "target_violation": bool(final_tvd > target_tvd),
        "stop_reason": controller.stop_reason,
        "handoff_lambdas": json.dumps(handoff_lambdas),
        "handoff_corrected_shifts": json.dumps(handoff_shifts),
    }


def simulate_stochastic_sequence(
    *,
    algorithm: str,
    size: int,
    sequence: Sequence[str],
    streams: Mapping[str, Sequence[Batch]],
    ideal: Counts,
    config: StableShotsConfig,
    policy: str,
    condition: str,
    p_failure: float,
    max_failures: int,
    max_total_shots: int,
    shift_cfg: Mapping[str, object],
    seed: int,
    target_tvd: float,
) -> Dict[str, object]:
    rng = random.Random(seed)
    controller = Controller(config=config)
    output: WeightedCounts = Counter()
    physical_total = 0
    actual_failures = 0
    backend_index = 0
    batch_index = 0
    handoff_lambdas: List[float] = []
    handoff_shifts: List[float] = []

    while physical_total < max_total_shots and backend_index < len(sequence):
        if controller.stopped:
            break
        backend = sequence[backend_index]
        batches = streams[backend]
        if batch_index >= len(batches):
            break
        batch_shots, batch_counts = batches[batch_index]
        controller.apply_batch(batch_counts, batch_shots)
        merge_weighted(output, batch_counts)
        physical_total += batch_shots
        batch_index += 1
        if controller.stopped:
            break

        can_fail = actual_failures < max_failures and backend_index + 1 < len(sequence)
        if can_fail and rng.random() < p_failure:
            old_effective = Counter(controller.counts)
            actual_failures += 1
            backend_index += 1
            next_backend = sequence[backend_index]
            next_batches = streams[next_backend]
            controller, output, probe_shots, probe_batches, shift = policy_after_handoff(
                policy,
                config,
                controller,
                output,
                old_effective,
                next_batches,
                shift_cfg,
                deterministic_seed(seed, "handoff", actual_failures),
            )
            physical_total += probe_shots
            batch_index = probe_batches
            if math.isfinite(float(shift["adaptive_lambda"])):
                handoff_lambdas.append(float(shift["adaptive_lambda"]))
            if math.isfinite(float(shift["shift_corrected_tvd"])):
                handoff_shifts.append(float(shift["shift_corrected_tvd"]))

    final_counts = Counter(output if policy != "naive_continuation" else controller.counts)
    final_tvd = tvd_weighted(final_counts, ideal)
    return {
        "experiment": "stochastic_failure",
        "circuit_key": f"{algorithm}_{size}",
        "algorithm": algorithm,
        "size": int(size),
        "sequence_condition": condition,
        "sequence": json.dumps(list(sequence)),
        "policy": policy,
        "failure_probability_per_batch": float(p_failure),
        "actual_failures": int(actual_failures),
        "physical_shots_total": int(physical_total),
        "effective_retained_shots": float(total_mass(final_counts)),
        "discarded_or_downweighted_shots": float(physical_total - total_mass(final_counts)),
        "final_tvd_to_aer": float(final_tvd),
        "target_violation": bool(final_tvd > target_tvd),
        "stop_reason": controller.stop_reason,
        "handoff_lambdas": json.dumps(handoff_lambdas),
        "handoff_corrected_shifts": json.dumps(handoff_shifts),
    }


def append_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    write_header = not path.exists()
    frame.to_csv(path, mode="a", header=write_header, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run StableShots QPU-failover experiments")
    parser.add_argument("--config", default="experiments/failover_config.json")
    parser.add_argument("--only", choices=["all", "single", "multi", "stochastic"], default="all")
    parser.add_argument("--resume", action="store_true", help="Append to an existing result directory instead of refusing")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)
    output_dir = Path(str(cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    marker = output_dir / "run_manifest.json"
    if marker.exists() and not args.resume:
        raise RuntimeError(
            f"{output_dir} already contains a run_manifest.json. Use --resume only when intentionally appending."
        )

    stable_raw = cfg["stableshots"]  # type: ignore[assignment]
    stable_cfg = StableShotsConfig(
        batch_size=int(stable_raw["batch_size"]),  # type: ignore[index]
        lookback_batches=int(stable_raw["lookback_batches"]),  # type: ignore[index]
        stability=int(stable_raw["stability"]),  # type: ignore[index]
        epsilon=float(stable_raw["epsilon"]),  # type: ignore[index]
        max_shots=int(stable_raw["max_shots"]),  # type: ignore[index]
    )
    single_cfg = cfg["single_failure"]  # type: ignore[assignment]
    shift_cfg = single_cfg["shift_aware"]  # type: ignore[index]
    target_tvd = float(cfg["target_tvd"])
    fixed_lambdas = [float(x) for x in single_cfg["fixed_decay_lambdas"]]  # type: ignore[index]
    policy_names = make_policy_names(fixed_lambdas)

    manifest = {
        "experiment_name": cfg["experiment_name"],
        "config_path": str(config_path),
        "started_unix": time.time(),
        "stableshots_config_id": stable_cfg.config_id,
        "selected_mode": args.only,
        "config": cfg,
    }
    marker.write_text(json.dumps(manifest, indent=2))

    failure_rows: List[Dict[str, object]] = []
    for algorithm in cfg["algorithms"]:  # type: ignore[assignment]
        for size_raw in cfg["sizes"]:  # type: ignore[assignment]
            size = int(size_raw)
            circuit_key = f"{algorithm}_{size}"
            print(f"[materialize] {circuit_key}", flush=True)
            try:
                ideal, streams, references = materialize_circuit(str(algorithm), size, cfg, stable_cfg)
            except Exception as exc:
                failure_rows.append({
                    "stage": "materialize",
                    "circuit_key": circuit_key,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"  skipped: {exc}", flush=True)
                continue

            refs = reference_rows(str(algorithm), size, references, ideal)
            append_csv(raw_dir / "qpu_references.csv", refs)

            no_failure_by_backend: Dict[str, Dict[str, object]] = {}
            no_failure_counts_by_backend: Dict[str, Counts] = {}
            no_failure_stop_by_backend: Dict[str, int] = {}
            baseline_rows: List[Dict[str, object]] = []
            fixed_rows: List[Dict[str, object]] = []
            for backend in cfg["backends"]:  # type: ignore[assignment]
                row, counts = run_no_failure(
                    str(algorithm), size, str(backend), streams[str(backend)], ideal, stable_cfg, target_tvd
                )
                baseline_rows.append(row)
                no_failure_by_backend[str(backend)] = row
                no_failure_counts_by_backend[str(backend)] = counts
                no_failure_stop_by_backend[str(backend)] = int(row["shots"])
                fixed_rows.extend(
                    fixed_shot_rows(
                        str(algorithm),
                        size,
                        str(backend),
                        streams[str(backend)],
                        ideal,
                        [int(x) for x in cfg["fixed_shot_budgets"]],  # type: ignore[index]
                        target_tvd,
                    )
                )
            append_csv(raw_dir / "no_failure_runs.csv", baseline_rows)
            append_csv(raw_dir / "fixed_shot_runs.csv", fixed_rows)

            if args.only in {"all", "single"} and bool(single_cfg.get("enabled", True)):
                rows: List[Dict[str, object]] = []
                for source in cfg["backends"]:  # type: ignore[assignment]
                    for target in cfg["backends"]:  # type: ignore[assignment]
                        source = str(source)
                        target = str(target)
                        if source == target:
                            continue
                        base = no_failure_by_backend[source]
                        for fraction in single_cfg["failure_fractions"]:  # type: ignore[index]
                            for policy in policy_names:
                                row = simulate_single_handoff(
                                    circuit_key=circuit_key,
                                    algorithm=str(algorithm),
                                    size=size,
                                    source_backend=source,
                                    target_backend=target,
                                    source_batches=streams[source],
                                    target_batches=streams[target],
                                    source_reference=references[source],
                                    target_reference=references[target],
                                    ideal_reference=ideal,
                                    no_failure_counts=no_failure_counts_by_backend[source],
                                    no_failure_shots=int(base["shots"]),
                                    no_failure_tvd=float(base["final_tvd_to_aer"]),
                                    failure_fraction=float(fraction),
                                    policy=policy,
                                    config=stable_cfg,
                                    max_new_shots=int(single_cfg["recovery_max_new_shots"]),
                                    shift_cfg=shift_cfg,
                                    seed_base=int(cfg["sampling_seed"]),
                                )
                                row["target_violation"] = bool(float(row["final_tvd_to_aer"]) > target_tvd)
                                rows.append(row)
                append_csv(raw_dir / "single_failure_runs.csv", rows)

            multi_cfg = cfg.get("multi_failure", {})
            if args.only in {"all", "multi"} and bool(multi_cfg.get("enabled", False)):
                rows = []
                conditions = [str(x) for x in multi_cfg["sequence_conditions"]]
                for condition in conditions:
                    repetitions = int(multi_cfg["random_repetitions"]) if condition == "random" else 1
                    for rep in range(repetitions):
                        sequence = build_sequence(
                            references,
                            ideal,
                            condition,
                            deterministic_seed(int(multi_cfg["random_seed"]), circuit_key, condition, rep),
                        )
                        for policy in [str(x) for x in multi_cfg["policies"]]:
                            row = simulate_deterministic_sequence(
                                algorithm=str(algorithm),
                                size=size,
                                sequence=sequence,
                                streams=streams,
                                references=references,
                                ideal=ideal,
                                no_failure_stop_by_backend=no_failure_stop_by_backend,
                                config=stable_cfg,
                                policy=policy,
                                condition=condition,
                                thresholds=[float(x) for x in multi_cfg["failure_fractions_of_original_stop"]],
                                max_new_shots=int(single_cfg["recovery_max_new_shots"]),
                                shift_cfg=shift_cfg,
                                seed=deterministic_seed(int(multi_cfg["random_seed"]), circuit_key, condition, rep, policy),
                                target_tvd=target_tvd,
                            )
                            row["repetition"] = rep
                            rows.append(row)
                append_csv(raw_dir / "multi_failure_runs.csv", rows)

            stochastic_cfg = cfg.get("stochastic_failure", {})
            if args.only in {"all", "stochastic"} and bool(stochastic_cfg.get("enabled", False)):
                rows = []
                conditions = [str(x) for x in stochastic_cfg["sequence_conditions"]]
                for p_failure in stochastic_cfg["per_batch_probabilities"]:
                    for condition in conditions:
                        for rep in range(int(stochastic_cfg["repetitions"])):
                            sequence = build_sequence(
                                references,
                                ideal,
                                condition,
                                deterministic_seed(
                                    int(stochastic_cfg["random_seed"]),
                                    circuit_key,
                                    condition,
                                    p_failure,
                                    rep,
                                ),
                            )
                            for policy in [str(x) for x in stochastic_cfg["policies"]]:
                                row = simulate_stochastic_sequence(
                                    algorithm=str(algorithm),
                                    size=size,
                                    sequence=sequence,
                                    streams=streams,
                                    ideal=ideal,
                                    config=stable_cfg,
                                    policy=policy,
                                    condition=condition,
                                    p_failure=float(p_failure),
                                    max_failures=int(stochastic_cfg["max_failures"]),
                                    max_total_shots=int(stable_cfg.max_shots + single_cfg["recovery_max_new_shots"]),
                                    shift_cfg=shift_cfg,
                                    seed=deterministic_seed(
                                        int(stochastic_cfg["random_seed"]),
                                        circuit_key,
                                        condition,
                                        p_failure,
                                        rep,
                                    ),
                                    target_tvd=target_tvd,
                                )
                                row["repetition"] = rep
                                rows.append(row)
                append_csv(raw_dir / "stochastic_failure_runs.csv", rows)

    if failure_rows:
        append_csv(raw_dir / "failures.csv", failure_rows)
    manifest["finished_unix"] = time.time()
    marker.write_text(json.dumps(manifest, indent=2))
    print(f"results written under {output_dir}", flush=True)
    auto_publish_results(cfg, [marker, raw_dir], "experiment")


if __name__ == "__main__":
    main()
