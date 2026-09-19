from __future__ import annotations

import json
import math

import pandas as pd
import streamlit as st

from stableshot.audit import verify_events
from stableshot.demo import (
    audit_jsonl,
    compare_policies,
    config_for_scenario,
    decisive_checks,
    execute_demo,
    posthoc_tvd,
    tamper_events,
)
from stableshot.main import StableShotsConfig
from stableshot.replay import available_scenarios, load_bundled_scenario


def _check_rows(audit) -> pd.DataFrame:
    rows = []
    for event in audit.events:
        if event["event_type"] != "stability_check":
            continue
        payload = event["payload"]
        rows.append(
            {
                "shots": int(payload["total_shots"]),
                "marginal_tvd": float(payload["delta"]),
                "epsilon": float(payload["epsilon"]),
                "passed": bool(payload["comparison_passed"]),
                "streak": int(payload["streak_after"]),
            }
        )
    return pd.DataFrame(rows)


def _configuration_controls(scenario):
    recommended = config_for_scenario(scenario)
    st.sidebar.subheader("StableShots policy")
    batch_size = st.sidebar.number_input(
        "Batch size", min_value=1, value=recommended.batch_size, step=1, key=f"batch-{scenario.scenario_id}"
    )
    lookback = st.sidebar.number_input(
        "Look-back batches",
        min_value=1,
        value=recommended.lookback_batches,
        step=1,
        key=f"lookback-{scenario.scenario_id}",
    )
    stability = st.sidebar.number_input(
        "Consecutive passes",
        min_value=1,
        value=recommended.stability,
        step=1,
        key=f"stability-{scenario.scenario_id}",
    )
    epsilon = st.sidebar.number_input(
        "Epsilon",
        min_value=0.0,
        value=recommended.epsilon,
        step=0.001,
        format="%.4f",
        key=f"epsilon-{scenario.scenario_id}",
    )
    max_shots = st.sidebar.number_input(
        "Maximum shots",
        min_value=int(batch_size),
        max_value=scenario.total_shots,
        value=min(recommended.max_shots, scenario.total_shots),
        step=int(batch_size),
        key=f"maxshots-{scenario.scenario_id}",
    )
    return StableShotsConfig(
        batch_size=int(batch_size),
        lookback_batches=int(lookback),
        stability=int(stability),
        epsilon=float(epsilon),
        max_shots=int(max_shots),
    )


def _render_execute(execution, scenario) -> None:
    posthoc = posthoc_tvd(execution, scenario)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Shots used", f"{execution.shots:,}")
    col2.metric("Stop reason", execution.stop_reason)
    col3.metric(
        "Last marginal TVD",
        "-" if math.isnan(execution.last_delta) else f"{execution.last_delta:.5f}",
    )
    col4.metric(
        "Post-hoc TVD to reference",
        "n/a" if posthoc is None else f"{posthoc:.5f}",
        help="Evaluation only. The reference distribution is not available to the online stopping controller.",
    )

    checks = _check_rows(execution.audit)
    if not checks.empty:
        st.subheader("Decision history")
        st.line_chart(checks.set_index("shots")[["marginal_tvd", "epsilon"]])
        st.dataframe(checks, use_container_width=True, hide_index=True)
    else:
        st.info("This run stopped before a stability check became eligible.")

    with st.expander("Online decision boundary"):
        st.write(
            "The controller receives only replayed measurement batches, cumulative count history, "
            "the configured threshold, the stability streak, and the shot cap. The reference counts "
            "shown above are evaluated only after the run finishes."
        )


def _render_explain(execution) -> None:
    explanation = execution.audit.explain()
    st.subheader("Why did the controller stop?")
    st.write(explanation.get("explanation", "No explanation available."))
    decisive = decisive_checks(execution.audit)
    if decisive:
        table = pd.DataFrame(
            [
                {
                    "shots": int(check["total_shots"]),
                    "TVD": float(check["delta"]),
                    "epsilon": float(check["epsilon"]),
                    "pass": bool(check["comparison_passed"]),
                    "streak": int(check["streak_after"]),
                }
                for check in decisive
            ]
        )
        st.dataframe(table, use_container_width=True, hide_index=True)

    checks = [event["payload"] for event in execution.audit.events if event["event_type"] == "stability_check"]
    if checks:
        contributions = checks[-1].get("top_outcome_contributions", [])
        if contributions:
            st.subheader("Largest contributors to the last TVD change")
            st.dataframe(pd.DataFrame(contributions), use_container_width=True, hide_index=True)


def _render_compare(scenario, config) -> None:
    st.subheader("Same replay, different execution policies")
    rows = compare_policies(scenario, config)
    frame = pd.DataFrame(rows)
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.caption(
        "posthoc_tvd_to_reference is an evaluation metric. It is never passed to the StableShots online controller."
    )


def _render_audit(execution) -> None:
    st.subheader("Tamper-evident execution record")
    verification = verify_events(execution.audit.events)
    if verification.get("valid"):
        st.success(
            f"Audit valid: {verification.get('events')} events, head {str(verification.get('head_hash'))[:16]}..."
        )
    else:
        st.error(f"Audit invalid: {verification}")

    st.download_button(
        "Download audit JSONL",
        data=audit_jsonl(execution.audit),
        file_name=f"{execution.scenario_id}-audit.jsonl",
        mime="application/x-ndjson",
    )

    if st.button("Create tampered copy", key="tamper-button"):
        st.session_state["tampered_events"] = tamper_events(execution.audit.events)
    tampered = st.session_state.get("tampered_events")
    if tampered:
        st.code(json.dumps(tampered[1], indent=2)[:1800], language="json")
        tampered_verification = verify_events(tampered)
        if tampered_verification.get("valid"):
            st.warning("Tampered copy still verified; this should not happen for the bundled demo mutation.")
        else:
            st.error(
                "Tamper detected: "
                f"sequence={tampered_verification.get('failed_sequence')}, "
                f"reason={tampered_verification.get('reason')}"
            )


def main() -> None:
    st.set_page_config(page_title="StableShots Demo", page_icon="SS", layout="wide")
    st.title("StableShots: adaptive and auditable shot control")
    st.write(
        "Run a deterministic replay, inspect the stopping evidence, compare the same measurement stream "
        "with fixed-shot policies, and verify the execution audit."
    )

    scenario_ids = sorted(available_scenarios())
    selected_id = st.sidebar.selectbox("Replay scenario", scenario_ids)
    scenario = load_bundled_scenario(selected_id)
    st.sidebar.caption(scenario.description)
    st.sidebar.write(f"Source: {scenario.source}")
    st.sidebar.write(f"Replay length: {scenario.total_shots:,} shots")
    config = _configuration_controls(scenario)

    if st.sidebar.button("Run execution", type="primary", use_container_width=True):
        st.session_state["execution"] = execute_demo(scenario, config)
        st.session_state["execution_scenario_id"] = scenario.scenario_id
        st.session_state["execution_config"] = config
        st.session_state.pop("tampered_events", None)

    execution = st.session_state.get("execution")
    execution_scenario_id = st.session_state.get("execution_scenario_id")
    if execution is None or execution_scenario_id != scenario.scenario_id:
        st.info("Choose a scenario and run it from the sidebar.")
        return

    effective_config = st.session_state.get("execution_config", config)
    execute_tab, explain_tab, compare_tab, audit_tab = st.tabs(["Execute", "Explain", "Compare", "Audit"])
    with execute_tab:
        _render_execute(execution, scenario)
    with explain_tab:
        _render_explain(execution)
    with compare_tab:
        _render_compare(scenario, effective_config)
    with audit_tab:
        _render_audit(execution)


if __name__ == "__main__":
    main()
