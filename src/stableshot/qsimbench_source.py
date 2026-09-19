from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Mapping, Sequence

from qsimbench import get_index

from stableshot.main import (
    FIXED_BASELINES,
    QSIMBENCH_SOURCE_BATCH_SIZE,
    REFERENCE_SHOTS,
    TraceSpec,
    load_qsimbench_batches,
    prefix_counts,
)
from stableshot.replay import ReplayScenario


def qsimbench_catalog(circuit_kind: str = "circuit") -> Dict[str, Dict[int, tuple[str, ...]]]:
    """Return the current QSimBench trace catalog in a stable, sorted shape."""
    raw = get_index(circuit_kind=circuit_kind)
    catalog: Dict[str, Dict[int, tuple[str, ...]]] = {}
    for algorithm, sizes in raw.items():
        normalized_sizes: Dict[int, tuple[str, ...]] = {}
        for size, backends in sizes.items():
            normalized_sizes[int(size)] = tuple(sorted(str(backend) for backend in backends))
        catalog[str(algorithm)] = dict(sorted(normalized_sizes.items()))
    return dict(sorted(catalog.items()))


def materialize_qsimbench_scenario(
    *,
    algorithm: str,
    size: int,
    backend: str,
    circuit_kind: str = "circuit",
    materialization_shots: int = REFERENCE_SHOTS,
    source_batch_size: int = QSIMBENCH_SOURCE_BATCH_SIZE,
    sampling_strategy: str = "sequential",
    sampling_seed: int = 0,
    force: bool = False,
    fixed_budgets: Sequence[int] = FIXED_BASELINES,
) -> ReplayScenario:
    """Materialize one QSimBench trace into the same replay object used by the demo.

    The final materialized prefix is retained as post-hoc reference counts. StableShots
    never receives that reference object; it only receives the batch sequence.
    """
    if materialization_shots <= 0:
        raise ValueError("materialization_shots must be positive")
    if source_batch_size <= 0:
        raise ValueError("source_batch_size must be positive")
    if sampling_strategy not in {"sequential", "random"}:
        raise ValueError("sampling_strategy must be 'sequential' or 'random'")

    spec = TraceSpec(
        algorithm=str(algorithm),
        size=int(size),
        backend=str(backend),
        circuit_kind=str(circuit_kind),
    )
    raw_batches = load_qsimbench_batches(
        spec=spec,
        total_shots=int(materialization_shots),
        source_batch_size=int(source_batch_size),
        sampling_strategy=sampling_strategy,
        sampling_seed=int(sampling_seed),
        force=bool(force),
    )
    reference = prefix_counts(raw_batches, int(materialization_shots))
    budgets = tuple(
        int(budget)
        for budget in fixed_budgets
        if 0 < int(budget) <= int(materialization_shots)
    )

    scenario_id = (
        f"qsimbench:{circuit_kind}:{algorithm}:{int(size)}:{backend}:"
        f"{sampling_strategy}:seed{int(sampling_seed)}:{int(materialization_shots)}"
    )
    return ReplayScenario(
        scenario_id=scenario_id,
        title=f"{algorithm} / {int(size)} qubits / {backend}",
        description=(
            f"QSimBench {circuit_kind} trace materialized as {len(raw_batches)} batches "
            f"({int(materialization_shots):,} shots)."
        ),
        source="QSimBench",
        batches=tuple((shots, Counter(counts)) for shots, counts in raw_batches),
        reference_counts=Counter(reference),
        recommended_config={
            "batch_size": 50,
            "lookback_batches": 3,
            "stability": 5,
            "epsilon": 0.005,
            "max_shots": int(materialization_shots),
        },
        fixed_budgets=budgets,
        context={
            "source_type": "qsimbench",
            "algorithm": str(algorithm),
            "size": int(size),
            "backend": str(backend),
            "circuit_kind": str(circuit_kind),
            "sampling_strategy": sampling_strategy,
            "sampling_seed": int(sampling_seed),
            "source_batch_size": int(source_batch_size),
            "materialization_shots": int(materialization_shots),
            "reference_shots": int(materialization_shots),
        },
    )
