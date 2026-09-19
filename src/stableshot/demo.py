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
