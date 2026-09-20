from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from stableshot.audit import AuditPolicy, StableShotsAudit, verify_events
from stableshot.main import StableShotsConfig, prefix_counts, run_stable_shots_audited, tvd
from stableshot.replay import ReplayScenario, available_scenarios, load_bundled_scenario


@dataclass(frozen=True)
class DemoExecution:
    scenario_id: str
    counts: Counter[str]
    shots: int
    last_delta: float
    stop_reason: str
    audit: StableShotsAudit


@dataclass(frozen=True)
class DistributionSnapshot:
    round_index: int
    total_shots: int
    counts: Counter[str]

    @property
    def frequencies(self) -> Dict[str, float]:
        if self.total_shots <= 0:
            return {}
        return {
            outcome: int(count) / self.total_shots
            for outcome, count in self.counts.items()
        }


def config_for_scenario(scenario: ReplayScenario) -> StableShotsConfig:
    defaults: Dict[str, int | float] = {
        "batch_size": 50,
        "lookback_batches": 3,
        "stability": 5,
        "epsilon": 0.005,
        "max_shots": scenario.total_shots,
    }
    defaults.update(scenario.recommended_config)
    return StableShotsConfig(
        batch_size=int(defaults["batch_size"]),
        lookback_batches=int(defaults["lookback_batches"]),
        stability=int(defaults["stability"]),
        epsilon=float(defaults["epsilon"]),
        max_shots=int(defaults["max_shots"]),
    )


def execute_demo(scenario: ReplayScenario, config: StableShotsConfig) -> DemoExecution:
    context = {
        "scenario_id": scenario.scenario_id,
        "source": scenario.source,
        "source_batch_count": len(scenario.batches),
        "source_shots": scenario.total_shots,
        **dict(scenario.context),
        "decision_scope": "online controller receives batch counts only",
    }
    counts, shots, last_delta, reason, audit = run_stable_shots_audited(
        scenario.batches,
        config,
        context=context,
        policy=AuditPolicy(
            include_batch_counts=True,
            include_check_counts=False,
            top_k_contributions=10,
        ),
    )
    return DemoExecution(
        scenario_id=scenario.scenario_id,
        counts=Counter(counts),
        shots=int(shots),
        last_delta=float(last_delta),
        stop_reason=str(reason),
        audit=audit,
    )


def distribution_iterations(audit: StableShotsAudit) -> List[Dict[str, int]]:
    """Return the available audited iterations without copying cumulative maps."""
    iterations: List[Dict[str, int]] = []
    for event in audit.events:
        if event.get("event_type") != "batch_accepted":
            continue
        payload = event.get("payload", {})
        iterations.append(
            {
                "round_index": int(payload["round_index"]),
                "total_shots": int(payload["total_shots"]),
            }
        )
    return iterations


def cumulative_distribution_snapshot(
    audit: StableShotsAudit,
    round_index: int,
) -> DistributionSnapshot:
    """Reconstruct one cumulative empirical distribution from audited batches.

    This is intentionally on-demand rather than materializing a cumulative count
    map for every iteration, which keeps the UI practical for large-support traces.
    """
    target_round = int(round_index)
    cumulative: Counter[str] = Counter()
    for event in audit.events:
        if event.get("event_type") != "batch_accepted":
            continue
        payload = event.get("payload", {})
        current_round = int(payload["round_index"])
        batch_counts = payload.get("batch_counts")
        if not isinstance(batch_counts, Mapping):
            raise ValueError(
                "cumulative distribution reconstruction requires batch_counts in the audit log"
            )
        for outcome, count in batch_counts.items():
            cumulative[str(outcome)] += int(count)

        total_shots = int(payload.get("total_shots", sum(cumulative.values())))
        if sum(cumulative.values()) != total_shots:
            raise ValueError(
                "audit batch counts do not match the recorded cumulative shot count"
            )
        if current_round == target_round:
            return DistributionSnapshot(
                round_index=current_round,
                total_shots=total_shots,
                counts=Counter(cumulative),
            )
        if current_round > target_round:
            break
    raise KeyError(f"no audited batch exists for round {target_round}")


def select_distribution_outcomes(
    current_counts: Mapping[str, int],
    *,
    comparison_counts: Mapping[str, int] | None = None,
    mode: str = "top_n",
    top_n: int = 20,
) -> List[str]:
    """Choose outcomes for visualization using a top-N or frequency threshold rule.

    When a comparison distribution is supplied, threshold filters use the union of
    outcomes passing in either distribution and top-N ranks outcomes by the larger
    of the two frequencies. This avoids hiding an outcome that is important in the
    look-back snapshot but not yet prominent in the current snapshot.
    """
    current_total = int(sum(int(value) for value in current_counts.values()))
    comparison_total = (
        int(sum(int(value) for value in comparison_counts.values()))
        if comparison_counts is not None
        else 0
    )
    if current_total <= 0:
        return []

    support = set(str(key) for key in current_counts)
    if comparison_counts is not None:
        support |= set(str(key) for key in comparison_counts)

    def current_frequency(outcome: str) -> float:
        return int(current_counts.get(outcome, 0)) / current_total

    def comparison_frequency(outcome: str) -> float:
        if comparison_counts is None or comparison_total <= 0:
            return 0.0
        return int(comparison_counts.get(outcome, 0)) / comparison_total

    score = {
        outcome: max(current_frequency(outcome), comparison_frequency(outcome))
        for outcome in support
    }

    if mode == "top_n":
        limit = min(max(int(top_n), 1), 100)
        selected = sorted(support, key=lambda outcome: (-score[outcome], outcome))[:limit]
    else:
        thresholds = {
            "freq_1pct": 0.01,
            "freq_5pct": 0.05,
            "freq_10pct": 0.10,
        }
        if mode not in thresholds:
            raise ValueError(
                f"unsupported distribution filter {mode!r}; "
                f"expected one of {['top_n', *thresholds]}"
            )
        threshold = thresholds[mode]
        selected = [
            outcome
            for outcome in support
            if current_frequency(outcome) > threshold
            or comparison_frequency(outcome) > threshold
        ]
        selected.sort(key=lambda outcome: (-score[outcome], outcome))
    return selected


def distribution_table_rows(
    current: DistributionSnapshot,
    outcomes: Sequence[str],
    *,
    comparison: DistributionSnapshot | None = None,
) -> List[Dict[str, Any]]:
    """Build display rows containing both cumulative counts and frequencies."""
    current_freq = current.frequencies
    comparison_freq = comparison.frequencies if comparison is not None else {}
    rows: List[Dict[str, Any]] = []
    for outcome in outcomes:
        row: Dict[str, Any] = {
            "outcome": str(outcome),
            "current_count": int(current.counts.get(outcome, 0)),
            "current_frequency": float(current_freq.get(outcome, 0.0)),
        }
        if comparison is not None:
            row.update(
                {
                    "lookback_count": int(comparison.counts.get(outcome, 0)),
                    "lookback_frequency": float(comparison_freq.get(outcome, 0.0)),
                    "frequency_change": float(
                        current_freq.get(outcome, 0.0)
                        - comparison_freq.get(outcome, 0.0)
                    ),
                }
            )
        rows.append(row)
    return rows


def posthoc_tvd(execution: DemoExecution, scenario: ReplayScenario) -> float | None:
    if scenario.reference_counts is None:
        return None
    return tvd(execution.counts, scenario.reference_counts)


def compare_policies(
    scenario: ReplayScenario,
    config: StableShotsConfig,
    fixed_budgets: Sequence[int] | None = None,
) -> List[Dict[str, Any]]:
    budgets = tuple(fixed_budgets if fixed_budgets is not None else scenario.fixed_budgets)
    rows: List[Dict[str, Any]] = []
    for budget in budgets:
        counts = prefix_counts(scenario.batches, int(budget))
        rows.append(
            {
                "policy": f"Fixed-{int(budget)}",
                "shots": int(budget),
                "stop_reason": "fixed_budget",
                "posthoc_tvd_to_reference": (
                    tvd(counts, scenario.reference_counts) if scenario.reference_counts is not None else None
                ),
            }
        )
    execution = execute_demo(scenario, config)
    rows.append(
        {
            "policy": "StableShots",
            "shots": execution.shots,
            "stop_reason": execution.stop_reason,
            "posthoc_tvd_to_reference": posthoc_tvd(execution, scenario),
        }
    )
    return rows


def decisive_checks(audit: StableShotsAudit) -> List[Mapping[str, Any]]:
    checks = [event["payload"] for event in audit.events if event["event_type"] == "stability_check"]
    if not checks:
        return []
    required = int(audit.config.get("stability", 1))
    return checks[-required:]


def tamper_events(events: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    tampered: List[Dict[str, Any]] = json.loads(json.dumps(list(events)))
    for event in tampered:
        if event.get("event_type") == "batch_accepted":
            payload = event["payload"]
            payload["batch_shots"] = max(0, int(payload["batch_shots"]) - 1)
            return tampered
    raise ValueError("audit contains no batch_accepted event to tamper with")


def audit_jsonl(audit: StableShotsAudit) -> str:
    return "".join(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n" for event in audit.events)


def self_check() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for scenario_id in sorted(available_scenarios()):
        scenario = load_bundled_scenario(scenario_id)
        config = config_for_scenario(scenario)
        execution = execute_demo(scenario, config)
        expected_reason_ok = (
            scenario.expected_stop_reason is None or execution.stop_reason == scenario.expected_stop_reason
        )
        expected_shots_ok = scenario.expected_shots is None or execution.shots == scenario.expected_shots
        audit_ok = bool(verify_events(execution.audit.events).get("valid"))
        tamper_ok = not bool(verify_events(tamper_events(execution.audit.events)).get("valid"))
        ok = expected_reason_ok and expected_shots_ok and audit_ok and tamper_ok
        detail = (
            f"reason={execution.stop_reason}, shots={execution.shots}, "
            f"audit={'valid' if audit_ok else 'invalid'}, tamper={'detected' if tamper_ok else 'missed'}"
        )
        rows.append({"check": scenario_id, "status": "PASS" if ok else "FAIL", "detail": detail})
    return rows


def format_delta(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:.6f}"
