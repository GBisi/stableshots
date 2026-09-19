from __future__ import annotations

import json
import math

import altair as alt
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
from stableshot.qsimbench_source import materialize_qsimbench_scenario, qsimbench_catalog
from stableshot.replay import available_scenarios, load_bundled_scenario


@st.cache_data(ttl=600, show_spinner=False)
def _cached_qsimbench_catalog(circuit_kind: str):
    return qsimbench_catalog(circuit_kind)


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


def _decision_history_chart(checks: pd.DataFrame, execution):
    shot_axis = alt.Axis(
        title="Cumulative shots",
        format=",d",
        tickCount=12,
        labelOverlap=True,
    )
    shot_x = alt.X("shots:Q", axis=shot_axis)

    marginal = (
        alt.Chart(checks)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=shot_x,
            y=alt.Y(
                "marginal_tvd:Q",
                title="TVD to look-back snapshot",
                axis=alt.Axis(format=".3f"),
            ),
            tooltip=[
                alt.Tooltip("shots:Q", title="Shots", format=","),
                alt.Tooltip("marginal_tvd:Q", title="Marginal TVD", format=".6f"),
                alt.Tooltip("epsilon:Q", title="Epsilon", format=".6f"),
                alt.Tooltip("passed:N", title="Passed"),
                alt.Tooltip("streak:Q", title="Stable streak"),
            ],
        )
    )

    epsilon = (
        alt.Chart(checks)
        .mark_line(strokeDash=[7, 5], strokeWidth=2)
        .encode(
            x=shot_x,
            y=alt.Y("epsilon:Q"),
        )
    )

    passed_checks = checks[checks["passed"]]
    passed = (
        alt.Chart(passed_checks)
        .mark_point(shape="diamond", filled=True, size=80)
        .encode(
            x=shot_x,
            y="marginal_tvd:Q",
            tooltip=[
                alt.Tooltip("shots:Q", title="Shots", format=","),
                alt.Tooltip("marginal_tvd:Q", title="Marginal TVD", format=".6f"),
                alt.Tooltip("streak:Q", title="Stable streak"),
            ],
        )
    )

    stop_frame = pd.DataFrame(
        {
            "shots": [execution.shots],
            "label": [
                f"STOP · {execution.stop_reason} · {execution.shots:,} shots"
            ],
        }
    )
    stop_rule = (
        alt.Chart(stop_frame)
        .mark_rule(strokeDash=[4, 4], strokeWidth=2, color="#ff4b4b")
        .encode(x=shot_x)
    )
    stop_label = (
        alt.Chart(stop_frame)
        .mark_text(
            align="right",
            baseline="top",
            dx=-6,
            dy=8,
            fontWeight="bold",
            color="#ff4b4b",
        )
        .encode(
            x=shot_x,
            y=alt.value(0),
            text="label:N",
        )
    )

    layers = marginal + epsilon + passed + stop_rule + stop_label
    if not math.isnan(execution.last_delta):
        stop_point = pd.DataFrame(
            {
                "shots": [execution.shots],
                "marginal_tvd": [execution.last_delta],
            }
        )
        layers = layers + (
            alt.Chart(stop_point)
            .mark_point(filled=True, size=130, color="#ff4b4b")
            .encode(x=shot_x, y="marginal_tvd:Q")
        )

    return layers.properties(height=390).interactive()


def _audit_event_rows(events) -> pd.DataFrame:
    rows = []
    for event in events:
        payload = event.get("payload", {})
        event_type = str(event.get("event_type", ""))
        status = ""
        if event_type == "stability_check":
            status = "PASS" if payload.get("comparison_passed") else "FAIL"
        elif event_type == "stop_decision":
            status = str(payload.get("reason_code", ""))
        rows.append(
            {
                "sequence": int(event.get("sequence", -1)),
                "event_type": event_type,
                "recorded_at_utc": event.get("recorded_at_utc", ""),
                "round": payload.get("round_index"),
                "total_shots": payload.get("total_shots"),
                "TVD": (
                    round(float(payload["delta"]), 6)
                    if payload.get("delta") is not None
                    else None
                ),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def _compact_numbers(value):
    """Return a display-only copy with floating values rounded to useful precision."""
    if isinstance(value, float):
        return float(f"{value:.8g}")
    if isinstance(value, dict):
        return {key: _compact_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact_numbers(item) for item in value]
    return value


def _move_audit_event(nav_key: str, sequences: list[int], offset: int) -> None:
    if not sequences:
        return
    current = int(st.session_state.get(nav_key, sequences[0]))
    try:
        index = sequences.index(current)
    except ValueError:
        index = 0
    next_index = min(max(index + offset, 0), len(sequences) - 1)
    st.session_state[nav_key] = sequences[next_index]


def _configuration_controls(scenario):
    recommended = config_for_scenario(scenario)
    st.sidebar.subheader("StableShots policy")
    batch_size = st.sidebar.number_input(
        "Batch size",
        min_value=1,
        value=recommended.batch_size,
        step=1,
        key=f"batch-{scenario.scenario_id}",
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


def _bundled_source_controls():
    scenario_ids = sorted(available_scenarios())
    selected_id = st.sidebar.selectbox("Replay scenario", scenario_ids)
    return load_bundled_scenario(selected_id)


def _qsimbench_source_controls():
    st.sidebar.caption(
        "QSimBench mode browses the live dataset index. Internet access is needed for the index "
        "and for traces that are not already in the QSimBench cache."
    )
    circuit_kind = st.sidebar.selectbox("Circuit kind", ["circuit", "mirror"])
    try:
        catalog = _cached_qsimbench_catalog(circuit_kind)
    except Exception as exc:
        st.sidebar.error(f"Could not load QSimBench index: {exc}")
        return None

    if not catalog:
        st.sidebar.warning("QSimBench returned an empty trace index.")
        return None

    algorithm = st.sidebar.selectbox("Algorithm", sorted(catalog))
    sizes = sorted(catalog[algorithm])
    size = int(st.sidebar.selectbox("Size (qubits)", sizes))
    backends = list(catalog[algorithm][size])
    backend = st.sidebar.selectbox("Backend", backends)

    st.sidebar.subheader("QSimBench materialization")
    sampling_strategy = st.sidebar.selectbox(
        "Sampling strategy",
        ["sequential", "random"],
        help=(
            "Sequential follows the QSimBench trace cursor at materialization time. "
            "Random uses the supplied seed and is convenient for repeatable resampling."
        ),
    )
    sampling_seed = int(
        st.sidebar.number_input("Sampling seed", min_value=0, value=0, step=1)
    )
    materialization_shots = int(
        st.sidebar.number_input(
            "Materialization shots",
            min_value=1000,
            max_value=20000,
            value=20000,
            step=50,
            help="The final materialized prefix is used only as a post-hoc reference distribution.",
        )
    )
    force = st.sidebar.checkbox(
        "Refresh QSimBench cache",
        value=False,
        help="Ignore cached QSimBench history files and download them again.",
    )

    selection_key = (
        circuit_kind,
        algorithm,
        size,
        backend,
        sampling_strategy,
        sampling_seed,
        materialization_shots,
    )
    if st.sidebar.button("Load QSimBench trace", use_container_width=True):
        try:
            with st.spinner(
                f"Materializing {algorithm}/{size}/{backend} from QSimBench..."
            ):
                scenario = materialize_qsimbench_scenario(
                    algorithm=algorithm,
                    size=size,
                    backend=backend,
                    circuit_kind=circuit_kind,
                    materialization_shots=materialization_shots,
                    sampling_strategy=sampling_strategy,
                    sampling_seed=sampling_seed,
                    force=force,
                )
            st.session_state["qsimbench_scenario"] = scenario
            st.session_state["qsimbench_selection_key"] = selection_key
            st.session_state.pop("execution", None)
            st.session_state.pop("tampered_events", None)
        except Exception as exc:
            st.sidebar.error(f"QSimBench materialization failed: {exc}")

    loaded = st.session_state.get("qsimbench_scenario")
    loaded_key = st.session_state.get("qsimbench_selection_key")
    if loaded is None or loaded_key != selection_key:
        st.info(
            "Choose a QSimBench algorithm, size, and backend in the sidebar, then click "
            "**Load QSimBench trace**. The materialized batches are kept in this session so "
            "you can rerun StableShots with different policy parameters without downloading again."
        )
        return None
    return loaded


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
        st.altair_chart(
            _decision_history_chart(checks, execution),
            use_container_width=True,
        )
        st.caption(
            "Solid line: marginal TVD · dashed horizontal line: epsilon · "
            "diamonds: passing checks · red dashed vertical line: controller stop."
        )
        st.dataframe(
            checks.style.format(
                {
                    "marginal_tvd": "{:.6f}",
                    "epsilon": "{:.6f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("This run stopped before a stability check became eligible.")

    with st.expander("Online decision boundary"):
        st.write(
            "The controller receives only measurement batches, cumulative count history, "
            "the configured threshold, the stability streak, and the shot cap. The reference "
            "counts shown above are evaluated only after the run finishes."
        )
        if scenario.context:
            st.json(dict(scenario.context))


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
        st.dataframe(
            table.style.format({"TVD": "{:.6g}", "epsilon": "{:.6g}"}),
            use_container_width=True,
            hide_index=True,
        )

    checks = [
        event["payload"]
        for event in execution.audit.events
        if event["event_type"] == "stability_check"
    ]
    if checks:
        contributions = checks[-1].get("top_outcome_contributions", [])
        if contributions:
            st.subheader("Largest contributors to the last TVD change")
            contribution_frame = pd.DataFrame(contributions)
            st.dataframe(
                contribution_frame.style.format(
                    {
                        "current_probability": "{:.6g}",
                        "previous_probability": "{:.6g}",
                        "tvd_contribution": "{:.6g}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )


def _render_compare(scenario, config) -> None:
    st.subheader("Same measurement stream, different execution policies")
    rows = compare_policies(scenario, config)
    frame = pd.DataFrame(rows)
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.caption(
        "posthoc_tvd_to_reference is an evaluation metric. It is never passed to the StableShots online controller."
    )


def _render_audit(execution) -> None:
    st.subheader("Tamper-evident execution record")
    events = execution.audit.events
    verification = verify_events(events)
    if verification.get("valid"):
        st.success(
            f"Audit valid: {verification.get('events')} events, "
            f"head {str(verification.get('head_hash'))[:16]}..."
        )
    else:
        st.error(f"Audit invalid: {verification}")

    all_event_types = sorted({str(event["event_type"]) for event in events})
    selected_types = st.multiselect(
        "Event types",
        options=all_event_types,
        default=all_event_types,
        help="Filter the timeline without changing the underlying audit log.",
    )
    filtered_events = [
        event for event in events if event["event_type"] in selected_types
    ]

    st.markdown("#### Event timeline")
    if filtered_events:
        st.dataframe(
            _audit_event_rows(filtered_events),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No events match the current filter.")
        return

    st.markdown("#### Event navigator")
    sequences = [int(event["sequence"]) for event in filtered_events]
    by_sequence = {int(event["sequence"]): event for event in filtered_events}
    nav_key = f"audit-event-sequence-{execution.audit.run_id}"
    if nav_key not in st.session_state or st.session_state[nav_key] not in sequences:
        st.session_state[nav_key] = sequences[0]

    previous_col, next_col, position_col = st.columns([1, 1, 3])
    current_index = sequences.index(int(st.session_state[nav_key]))
    previous_col.button(
        "← Previous",
        key=f"audit-prev-{execution.audit.run_id}",
        disabled=current_index == 0,
        on_click=_move_audit_event,
        args=(nav_key, sequences, -1),
        use_container_width=True,
    )
    next_col.button(
        "Next →",
        key=f"audit-next-{execution.audit.run_id}",
        disabled=current_index == len(sequences) - 1,
        on_click=_move_audit_event,
        args=(nav_key, sequences, 1),
        use_container_width=True,
    )

    selected_sequence = position_col.selectbox(
        "Jump to event",
        options=sequences,
        format_func=lambda sequence: (
            f"#{sequence} · {by_sequence[sequence]['event_type']}"
        ),
        key=nav_key,
        label_visibility="collapsed",
    )
    selected_event = by_sequence[int(selected_sequence)]
    selected_payload = selected_event.get("payload", {})

    detail_left, detail_middle, detail_right = st.columns(3)
    detail_left.metric("Sequence", int(selected_event["sequence"]))
    detail_middle.metric("Event type", str(selected_event["event_type"]))
    detail_right.metric(
        "Shots",
        (
            f"{int(selected_payload['total_shots']):,}"
            if selected_payload.get("total_shots") is not None
            else "—"
        ),
    )
    st.caption(str(selected_event.get("recorded_at_utc", "")))

    st.markdown("**Payload (readable view)**")
    st.json(_compact_numbers(selected_payload), expanded=True)
    with st.expander("Hash-chain metadata and full event"):
        st.code(
            "\n".join(
                [
                    f"previous_event_hash: {selected_event.get('previous_event_hash', '')}",
                    f"event_hash:          {selected_event.get('event_hash', '')}",
                ]
            )
        )
        st.json(selected_event, expanded=False)

    st.download_button(
        "Download complete audit JSONL",
        data=audit_jsonl(execution.audit),
        file_name=f"{execution.scenario_id.replace(':', '-')}-audit.jsonl",
        mime="application/x-ndjson",
        use_container_width=True,
    )

    st.divider()
    st.markdown("#### Tamper test")
    st.caption(
        "Create an in-memory copy with one batch event modified, then verify the hash chain again."
    )
    if st.button("Create tampered copy", key="tamper-button"):
        st.session_state["tampered_events"] = tamper_events(events)
    tampered = st.session_state.get("tampered_events")
    if tampered:
        tampered_verification = verify_events(tampered)
        failed_sequence = tampered_verification.get("failed_sequence")
        if failed_sequence is not None and 0 <= int(failed_sequence) < len(tampered):
            st.json(tampered[int(failed_sequence)], expanded=False)
        if tampered_verification.get("valid"):
            st.warning(
                "Tampered copy still verified; this should not happen for the demo mutation."
            )
        else:
            st.error(
                "Tamper detected: "
                f"sequence={failed_sequence}, "
                f"reason={tampered_verification.get('reason')}"
            )


def main() -> None:
    st.set_page_config(page_title="StableShots Demo", page_icon="SS", layout="wide")
    st.title("StableShots: adaptive and auditable shot control")
    st.write(
        "Use a bundled offline replay or browse QSimBench traces, then inspect the stopping "
        "evidence, compare fixed-shot policies, and verify the execution audit."
    )

    source_mode = st.sidebar.radio(
        "Trace source",
        ["Bundled replay", "QSimBench"],
        help="Bundled replays are offline. QSimBench exposes the live benchmark catalog.",
    )
    scenario = (
        _bundled_source_controls()
        if source_mode == "Bundled replay"
        else _qsimbench_source_controls()
    )
    if scenario is None:
        return

    st.sidebar.caption(scenario.description)
    st.sidebar.write(f"Source: {scenario.source}")
    st.sidebar.write(f"Materialized length: {scenario.total_shots:,} shots")
    config = _configuration_controls(scenario)

    if st.sidebar.button("Run execution", type="primary", use_container_width=True):
        st.session_state["execution"] = execute_demo(scenario, config)
        st.session_state["execution_scenario_id"] = scenario.scenario_id
        st.session_state["execution_config"] = config
        st.session_state.pop("tampered_events", None)

    execution = st.session_state.get("execution")
    execution_scenario_id = st.session_state.get("execution_scenario_id")
    if execution is None or execution_scenario_id != scenario.scenario_id:
        st.info("The trace is ready. Configure StableShots in the sidebar and click **Run execution**.")
        return

    effective_config = st.session_state.get("execution_config", config)
    execute_tab, explain_tab, compare_tab, audit_tab = st.tabs(
        ["Execute", "Explain", "Compare", "Audit"]
    )
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
