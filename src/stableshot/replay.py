from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

Counts = Counter[str]
Batch = Tuple[int, Counts]
REPLAY_SCHEMA = "stableshot-replay-v1"


@dataclass(frozen=True)
class ReplayScenario:
    scenario_id: str
    title: str
    description: str
    source: str
    batches: Tuple[Batch, ...]
    reference_counts: Counts | None
    recommended_config: Mapping[str, int | float]
    fixed_budgets: Tuple[int, ...]
    expected_stop_reason: str | None = None
    expected_shots: int | None = None

    @property
    def total_shots(self) -> int:
        return sum(shots for shots, _ in self.batches)


class ReplayFormatError(ValueError):
    pass


def _normalize_counts(raw: Mapping[str, Any], *, field: str) -> Counts:
    counts: Counts = Counter()
    for outcome, value in raw.items():
        count = int(value)
        if count < 0:
            raise ReplayFormatError(f"{field} contains a negative count for {outcome!r}")
        if count:
            counts[str(outcome)] = count
    if not counts:
        raise ReplayFormatError(f"{field} must contain at least one positive count")
    return counts


def _metadata_record(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not records or records[0].get("type") != "metadata":
        raise ReplayFormatError("first JSONL record must be a metadata record")
    metadata = records[0]
    if metadata.get("schema_version") != REPLAY_SCHEMA:
        raise ReplayFormatError(
            f"unsupported replay schema {metadata.get('schema_version')!r}; expected {REPLAY_SCHEMA!r}"
        )
    return metadata


def load_replay(path: Path | str) -> ReplayScenario:
    replay_path = Path(path)
    records: list[Dict[str, Any]] = []
    with replay_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReplayFormatError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ReplayFormatError(f"line {line_number} must contain a JSON object")
            records.append(record)

    metadata = _metadata_record(records)
    batches: list[Batch] = []
    for index, record in enumerate(records[1:], start=2):
        if record.get("type") != "batch":
            raise ReplayFormatError(f"line {index} has unsupported record type {record.get('type')!r}")
        shots = int(record.get("shots", 0))
        repeat = int(record.get("repeat", 1))
        if shots <= 0:
            raise ReplayFormatError(f"line {index} must have a positive shots value")
        if repeat <= 0:
            raise ReplayFormatError(f"line {index} must have repeat >= 1")
        raw_counts = record.get("counts")
        if not isinstance(raw_counts, dict):
            raise ReplayFormatError(f"line {index} must contain a counts object")
        counts = _normalize_counts(raw_counts, field=f"line {index} counts")
        if sum(counts.values()) != shots:
            raise ReplayFormatError(
                f"line {index} counts sum to {sum(counts.values())}, expected shots={shots}"
            )
        for _ in range(repeat):
            batches.append((shots, Counter(counts)))

    if not batches:
        raise ReplayFormatError("replay must contain at least one batch")

    reference_raw = metadata.get("reference_counts")
    reference_counts = None
    if reference_raw is not None:
        if not isinstance(reference_raw, dict):
            raise ReplayFormatError("reference_counts must be a JSON object")
        reference_counts = _normalize_counts(reference_raw, field="reference_counts")

    recommended = metadata.get("recommended_config", {})
    if not isinstance(recommended, dict):
        raise ReplayFormatError("recommended_config must be a JSON object")
    fixed_budgets_raw = metadata.get("fixed_budgets", [])
    if not isinstance(fixed_budgets_raw, list):
        raise ReplayFormatError("fixed_budgets must be a JSON array")
    fixed_budgets = tuple(int(value) for value in fixed_budgets_raw)
    if any(value <= 0 for value in fixed_budgets):
        raise ReplayFormatError("fixed_budgets values must be positive")
    if any(value > sum(shots for shots, _ in batches) for value in fixed_budgets):
        raise ReplayFormatError("fixed_budgets cannot exceed replay length")

    return ReplayScenario(
        scenario_id=str(metadata.get("scenario_id", replay_path.stem)),
        title=str(metadata.get("title", replay_path.stem)),
        description=str(metadata.get("description", "")),
        source=str(metadata.get("source", "bundled replay")),
        batches=tuple(batches),
        reference_counts=reference_counts,
        recommended_config={str(key): value for key, value in recommended.items()},
        fixed_budgets=fixed_budgets,
        expected_stop_reason=(
            str(metadata["expected_stop_reason"]) if metadata.get("expected_stop_reason") is not None else None
        ),
        expected_shots=(int(metadata["expected_shots"]) if metadata.get("expected_shots") is not None else None),
    )


def bundled_data_dir() -> Path:
    return Path(__file__).with_name("demo_data")


def available_scenarios() -> Dict[str, Path]:
    scenarios: Dict[str, Path] = {}
    for path in sorted(bundled_data_dir().glob("*.jsonl")):
        scenario = load_replay(path)
        scenarios[scenario.scenario_id] = path
    return scenarios


def load_bundled_scenario(scenario_id: str) -> ReplayScenario:
    scenarios = available_scenarios()
    try:
        path = scenarios[scenario_id]
    except KeyError as exc:
        raise KeyError(f"unknown bundled scenario {scenario_id!r}; choices={sorted(scenarios)}") from exc
    return load_replay(path)


def iter_batches(scenario: ReplayScenario) -> Iterable[Batch]:
    for shots, counts in scenario.batches:
        yield shots, Counter(counts)
