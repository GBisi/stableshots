"""Tamper-evident audit records for StableShots decisions.

The audit trail is deliberately independent of QSimBench.  It records only the
configuration, execution context, batch/cumulative count fingerprints, each
stability check, and the final stop decision.  Full counts are optional because
large quantum output spaces can make a complete event log expensive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


ZERO_HASH = "0" * 64


def _format_metric(value: float | int) -> str:
    """Format a numeric decision metric without binary floating-point noise."""
    return f"{float(value):.6g}"


def _format_metric_list(values: Sequence[float | int]) -> str:
    return "[" + ", ".join(_format_metric(value) for value in values) + "]"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def counts_fingerprint(counts: Mapping[str, int]) -> Dict[str, Any]:
    normalized = {str(key): int(value) for key, value in counts.items() if int(value) > 0}
    return {
        "shots": int(sum(normalized.values())),
        "support_size": len(normalized),
        "sha256": _sha256_json(normalized),
    }


def tvd_contributions(
    current: Mapping[str, int],
    previous: Mapping[str, int],
    top_k: int,
) -> List[Dict[str, Any]]:
    current_total = int(sum(int(v) for v in current.values()))
    previous_total = int(sum(int(v) for v in previous.values()))
    if current_total <= 0 or previous_total <= 0:
        return []

    rows: List[Dict[str, Any]] = []
    for outcome in set(current) | set(previous):
        current_probability = int(current.get(outcome, 0)) / current_total
        previous_probability = int(previous.get(outcome, 0)) / previous_total
        rows.append(
            {
                "outcome": str(outcome),
                "current_probability": current_probability,
                "previous_probability": previous_probability,
                "tvd_contribution": 0.5 * abs(current_probability - previous_probability),
            }
        )
    rows.sort(key=lambda row: (-float(row["tvd_contribution"]), str(row["outcome"])))
    return rows[: max(0, int(top_k))]


@dataclass(frozen=True)
class AuditPolicy:
    """Controls the size/detail trade-off of the audit log."""

    include_batch_counts: bool = True
    include_check_counts: bool = False
    top_k_contributions: int = 10


class StableShotsAudit:
    """Append-only, hash-chained record of one StableShots controller run."""

    def __init__(
        self,
        config: Mapping[str, Any],
        context: Optional[Mapping[str, Any]] = None,
        policy: Optional[AuditPolicy] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self.run_id = run_id or str(uuid.uuid4())
        self.config = dict(config)
        self.context = dict(context or {})
        self.policy = policy or AuditPolicy()
        self.events: List[Dict[str, Any]] = []
        self._head_hash = ZERO_HASH
        self.append(
            "run_started",
            {
                "config": self.config,
                "context": self.context,
                "audit_policy": asdict(self.policy),
                "decision_scope": (
                    "The online decision uses batch measurement counts and cumulative-history TVD only; "
                    "reference-distribution TVD is not a decision input."
                ),
            },
        )

    def append(self, event_type: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        event_without_hash: Dict[str, Any] = {
            "schema_version": "stableshot-audit-v1",
            "run_id": self.run_id,
            "sequence": len(self.events),
            "recorded_at_utc": _utc_now(),
            "event_type": str(event_type),
            "previous_event_hash": self._head_hash,
            "payload": dict(payload),
        }
        event_hash = _sha256_json(event_without_hash)
        event = {**event_without_hash, "event_hash": event_hash}
        self.events.append(event)
        self._head_hash = event_hash
        return event

    def record_batch(
        self,
        *,
        round_index: int,
        batch_shots: int,
        total_shots: int,
        batch_counts: Mapping[str, int],
        cumulative_counts: Mapping[str, int],
    ) -> None:
        payload: Dict[str, Any] = {
            "round_index": int(round_index),
            "batch_shots": int(batch_shots),
            "total_shots": int(total_shots),
            "batch_fingerprint": counts_fingerprint(batch_counts),
            "cumulative_fingerprint": counts_fingerprint(cumulative_counts),
        }
        if self.policy.include_batch_counts:
            payload["batch_counts"] = {
                str(key): int(value) for key, value in sorted(batch_counts.items()) if int(value) > 0
            }
        self.append("batch_accepted", payload)

    def record_check(
        self,
        *,
        round_index: int,
        lookback_round_index: int,
        total_shots: int,
        current_counts: Mapping[str, int],
        previous_counts: Mapping[str, int],
        delta: float,
        epsilon: float,
        streak_before: int,
        streak_after: int,
        required_stability: int,
    ) -> None:
        passed = float(delta) <= float(epsilon)
        payload: Dict[str, Any] = {
            "round_index": int(round_index),
            "lookback_round_index": int(lookback_round_index),
            "total_shots": int(total_shots),
            "metric": "total_variation_distance",
            "delta": float(delta),
            "epsilon": float(epsilon),
            "comparison_passed": bool(passed),
            "streak_before": int(streak_before),
            "streak_after": int(streak_after),
            "required_stability": int(required_stability),
            "current_fingerprint": counts_fingerprint(current_counts),
            "lookback_fingerprint": counts_fingerprint(previous_counts),
            "top_outcome_contributions": tvd_contributions(
                current_counts, previous_counts, self.policy.top_k_contributions
            ),
        }
        payload["decision_input_sha256"] = _sha256_json(
            {
                "current_fingerprint": payload["current_fingerprint"],
                "lookback_fingerprint": payload["lookback_fingerprint"],
                "delta": payload["delta"],
                "epsilon": payload["epsilon"],
                "streak_before": payload["streak_before"],
                "required_stability": payload["required_stability"],
            }
        )
        if self.policy.include_check_counts:
            payload["current_counts"] = {
                str(key): int(value) for key, value in sorted(current_counts.items()) if int(value) > 0
            }
            payload["lookback_counts"] = {
                str(key): int(value) for key, value in sorted(previous_counts.items()) if int(value) > 0
            }
        self.append("stability_check", payload)

    def record_stop(
        self,
        *,
        reason: str,
        total_shots: int,
        round_index: int,
        stable_streak: int,
        checks_performed: int,
        last_delta: Optional[float],
    ) -> None:
        reason_text = {
            "stable": "The required number of consecutive TVD checks passed.",
            "max_budget": "The configured maximum shot budget was reached before the stability rule passed.",
            "input_exhausted": "The supplied batch stream ended before stability or the configured budget was reached.",
        }.get(reason, "The controller stopped for an implementation-specific reason.")
        self.append(
            "stop_decision",
            {
                "reason_code": str(reason),
                "reason_text": reason_text,
                "total_shots": int(total_shots),
                "round_index": int(round_index),
                "stable_streak": int(stable_streak),
                "checks_performed": int(checks_performed),
                "last_delta": None if last_delta is None else float(last_delta),
                "audit_head_hash_before_stop": self._head_hash,
            },
        )

    def write_jsonl(self, path: Path | str) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        return output

    def explain(self) -> Dict[str, Any]:
        stop = next((event for event in reversed(self.events) if event["event_type"] == "stop_decision"), None)
        checks = [event["payload"] for event in self.events if event["event_type"] == "stability_check"]
        if stop is None:
            return {"run_id": self.run_id, "status": "running", "explanation": "No stop decision is recorded."}

        payload = stop["payload"]
        reason = str(payload["reason_code"])
        if reason == "stable":
            required = int(self.config["stability"])
            decisive_checks = checks[-required:]
            deltas = [float(check["delta"]) for check in decisive_checks]
            explanation = (
                f"Stopped after {payload['total_shots']} shots because {required} consecutive checks "
                f"had TVD <= {_format_metric(self.config['epsilon'])}. "
                f"Decisive TVDs: {_format_metric_list(deltas)}."
            )
        elif reason == "max_budget":
            explanation = (
                f"Stopped after {payload['total_shots']} shots because max_shots={self.config['max_shots']} "
                f"was reached. Final stable streak was {payload['stable_streak']} of {self.config['stability']}."
            )
        elif reason == "input_exhausted":
            explanation = (
                f"Stopped after {payload['total_shots']} shots because the input batches ended before "
                "the stability criterion or maximum budget was reached."
            )
        else:
            explanation = str(payload["reason_text"])

        return {
            "run_id": self.run_id,
            "status": "stopped",
            "reason_code": reason,
            "explanation": explanation,
            "config": self.config,
            "context": self.context,
            "checks_performed": int(payload["checks_performed"]),
            "final_event_hash": stop["event_hash"],
            "decision_inputs": "batch counts, cumulative counts, lookback cumulative counts, TVD threshold, and streak history",
            "not_decision_inputs": "reference counts and post-hoc tvd_to_reference",
        }


def read_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_events(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    previous_hash = ZERO_HASH
    run_id: Optional[str] = None
    for index, event in enumerate(events):
        current = dict(event)
        event_hash = str(current.pop("event_hash", ""))
        if int(current.get("sequence", -1)) != index:
            return {"valid": False, "failed_sequence": index, "reason": "sequence_mismatch"}
        if current.get("previous_event_hash") != previous_hash:
            return {"valid": False, "failed_sequence": index, "reason": "previous_hash_mismatch"}
        if run_id is None:
            run_id = str(current.get("run_id"))
        elif str(current.get("run_id")) != run_id:
            return {"valid": False, "failed_sequence": index, "reason": "run_id_mismatch"}
        expected = _sha256_json(current)
        if event_hash != expected:
            return {"valid": False, "failed_sequence": index, "reason": "event_hash_mismatch"}
        previous_hash = event_hash
    return {"valid": True, "events": len(events), "run_id": run_id, "head_hash": previous_hash}


def explain_events(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not events:
        return {"status": "empty", "explanation": "No audit events found."}
    start = next(event for event in events if event["event_type"] == "run_started")
    stop = next((event for event in reversed(events) if event["event_type"] == "stop_decision"), None)
    checks = [event["payload"] for event in events if event["event_type"] == "stability_check"]
    config = start["payload"]["config"]
    context = start["payload"].get("context", {})
    if stop is None:
        return {"run_id": start["run_id"], "status": "running", "config": config, "context": context}
    payload = stop["payload"]
    reason = payload["reason_code"]
    if reason == "stable":
        decisive = checks[-int(config["stability"]):]
        decisive_values = [float(row["delta"]) for row in decisive]
        explanation = (
            f"Stopped after {payload['total_shots']} shots: {config['stability']} consecutive TVD checks "
            f"were <= {_format_metric(config['epsilon'])}; "
            f"values={_format_metric_list(decisive_values)}."
        )
    else:
        explanation = str(payload["reason_text"])
    return {
        "run_id": start["run_id"],
        "status": "stopped",
        "reason_code": reason,
        "explanation": explanation,
        "config": config,
        "context": context,
        "head_hash": events[-1]["event_hash"],
    }


def plot_events(events: Sequence[Mapping[str, Any]], output_path: Path | str) -> Path:
    import matplotlib.pyplot as plt

    checks = [event["payload"] for event in events if event["event_type"] == "stability_check"]
    if not checks:
        raise ValueError("audit log contains no stability checks to plot")
    shots = [int(check["total_shots"]) for check in checks]
    deltas = [float(check["delta"]) for check in checks]
    epsilon = float(checks[0]["epsilon"])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(shots, deltas, marker="o", label="marginal TVD")
    plt.axhline(epsilon, linestyle="--", label=f"epsilon={epsilon:g}")
    passed_shots = [int(check["total_shots"]) for check in checks if check["comparison_passed"]]
    passed_deltas = [float(check["delta"]) for check in checks if check["comparison_passed"]]
    if passed_shots:
        plt.scatter(passed_shots, passed_deltas, label="passed checks")

    stop = next((event for event in reversed(events) if event["event_type"] == "stop_decision"), None)
    if stop is not None:
        stop_shots = int(stop["payload"]["total_shots"])
        stop_reason = str(stop["payload"].get("reason_code", "stop"))
        plt.axvline(stop_shots, linestyle=":", label=f"STOP: {stop_reason} ({stop_shots} shots)")

    plt.xlabel("Cumulative shots")
    plt.ylabel("TVD to lookback snapshot")
    plt.title("StableShots decision history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    return output


def _main() -> None:
    parser = argparse.ArgumentParser(description="Inspect StableShots audit JSONL files")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("verify", "explain"):
        sub = subparsers.add_parser(command)
        sub.add_argument("audit_jsonl")
    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("audit_jsonl")
    plot_parser.add_argument("output_png")
    args = parser.parse_args()

    events = read_jsonl(args.audit_jsonl)
    if args.command == "verify":
        print(json.dumps(verify_events(events), indent=2, sort_keys=True))
    elif args.command == "explain":
        print(json.dumps(explain_events(events), indent=2, sort_keys=True))
    else:
        output = plot_events(events, args.output_png)
        print(output)


if __name__ == "__main__":
    _main()
