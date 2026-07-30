# StableShots

StableShots is an online shot-stopping rule for static quantum circuit
execution. Instead of choosing a fixed shot budget before execution, it runs the
same circuit in small batches and stops when the cumulative empirical output
distribution has become stable.

This repository contains the experimental code and result artifacts for the
paper:

> StableShots: Online Shot Stopping for Quantum Circuit Execution

The method is black-box: it only needs batch-level measurement counts. It does
not inspect the circuit, backend calibration data, or a noise model.

## Method

For a fixed circuit and backend, StableShots accumulates measurement counts over
batches of `b` shots. After each batch, it compares the current cumulative
empirical distribution with the cumulative distribution from `lookback_batches`
batches earlier using Total Variation Distance (TVD):

```text
TVD(P, Q) = 1/2 * sum_x |P(x) - Q(x)|
```

Execution stops when this marginal TVD is at most `epsilon` for `stability`
consecutive checks, or when the maximum budget is reached.

The selected paper configuration is:

```text
b50_lb3_k5_eps0p005
```

That means:

- `batch_size = 50`
- `lookback_batches = 3`
- `stability = 5`
- `epsilon = 0.005`
- `max_shots = 20000`

This configuration compares the current cumulative distribution with the one
150 shots earlier and requires five consecutive passing comparisons.

StableShots is a diminishing-returns heuristic. It does not prove closeness to
the unknown backend-induced distribution; it stops when additional batches have
small observed effect on the cumulative empirical distribution.

## Auditable StableShots Decisions

StableShots can optionally produce an append-only audit trail for each stopping
decision. The audit answers four questions:

1. **Why did execution stop?** It records whether the controller became stable,
   reached `max_shots`, or exhausted the supplied batch stream.
2. **Which data was used?** It records each accepted batch, cumulative-count
   fingerprints, and optionally the full counts.
3. **Which history was compared?** Every stability check references the current
   round and the exact lookback round used in the TVD calculation.
4. **How did the decision evolve?** It records every TVD value, threshold result,
   and stable-streak transition, and can render a decision-history plot.

Auditing is optional. The original `run_stable_shots()` return type is unchanged.
Use `run_stable_shots_audited()` when the caller also needs the audit object.

### Minimal audited run

```python
from collections import Counter

from stableshot.audit import AuditPolicy
from stableshot.main import StableShotsConfig, run_stable_shots_audited

raw_batches = [
    (50, Counter({"00": 31, "11": 19})),
    (50, Counter({"00": 30, "11": 20})),
    (50, Counter({"00": 31, "11": 19})),
    (50, Counter({"00": 30, "11": 20})),
]

config = StableShotsConfig(
    batch_size=50,
    lookback_batches=1,
    stability=2,
    epsilon=0.01,
    max_shots=200,
)

counts, shots, last_delta, reason, audit = run_stable_shots_audited(
    raw_batches,
    config,
    context={
        "trace_id": "example-circuit",
        "backend": "fake_kyiv",
        "sampling_strategy": "sequential",
        "sampling_seed": 0,
        "source_batch_size": 50,
        # Add a Git commit, container image digest, job ID, or dataset version
        # here when those identifiers are available.
        "code_revision": "<git-commit>",
    },
    policy=AuditPolicy(
        include_batch_counts=True,
        include_check_counts=False,
        top_k_contributions=10,
    ),
)

print(reason, shots, last_delta)
print(audit.explain()["explanation"])
audit.write_jsonl("results/audit/example-circuit.jsonl")
```

`context` is caller-defined provenance. It should contain enough information to
reconstruct where the input batches came from, such as the circuit or trace ID,
backend, sampling policy, random seed, source batch size, dataset version, code
revision, and execution job ID.

### Audit event model

Each JSONL line is one event with a monotonically increasing sequence number,
timestamp, run ID, previous-event hash, payload, and event hash.

| Event | Purpose |
| --- | --- |
| `run_started` | Records the full StableShots configuration, execution context, retention policy, and decision scope. |
| `batch_accepted` | Records the execution round, batch size, cumulative shots, batch fingerprint, cumulative fingerprint, and optionally batch counts. |
| `stability_check` | Records the current and lookback rounds, TVD, `epsilon`, pass/fail result, streak transition, count fingerprints, and largest per-outcome TVD contributions. |
| `stop_decision` | Records the final reason code, shot count, round, stable streak, checks performed, and final TVD evidence. |

The supported stop reason codes are:

- `stable`: the required number of consecutive TVD checks passed.
- `max_budget`: `max_shots` was reached before the stability criterion passed.
- `input_exhausted`: the supplied batch stream ended before either condition was
  reached.

Separating `input_exhausted` from `max_budget` prevents an incomplete input trace
from being reported as an intentional budget stop.

### Explain, verify, and plot an audit

The audit module includes a small command-line interface:

```bash
uv run python -m stableshot.audit verify \
  results/audit/example-circuit.jsonl

uv run python -m stableshot.audit explain \
  results/audit/example-circuit.jsonl

uv run python -m stableshot.audit plot \
  results/audit/example-circuit.jsonl \
  results/audit/example-circuit.png
```

`verify` recomputes the hash chain and reports the first invalid event if an
event was edited, removed, inserted, or reordered. `explain` generates a
structured summary of the stop decision. `plot` renders marginal TVD against
cumulative shots, the `epsilon` threshold, and passing checks.

The same operations are available from Python:

```python
from stableshot.audit import (
    explain_events,
    plot_events,
    read_jsonl,
    verify_events,
)

events = read_jsonl("results/audit/example-circuit.jsonl")
print(verify_events(events))
print(explain_events(events))
plot_events(events, "results/audit/example-circuit.png")
```

Example artifacts are checked in under `example_audit/`:

```text
example_audit/audit_demo.jsonl
example_audit/audit_demo.png
example_audit/audit_demo_explanation.json
```

### What the explanation uses

An online StableShots decision uses only:

- Batch measurement counts.
- Cumulative measurement-count history.
- The cumulative snapshot selected by `lookback_batches`.
- TVD, `epsilon`, and the consecutive stable-streak history.
- The configured shot budget.

The 20,000-shot reference distribution and `tvd_to_reference` are **not**
decision inputs. They are post-hoc evaluation data used by the experiments to
measure the quality of a completed run. Store those values separately from the
online decision audit so an audit consumer cannot mistake evaluation evidence
for information available to the stopping controller.

### Per-outcome evidence

Each `stability_check` includes the largest per-outcome TVD contributions. For an
outcome `x`, the recorded contribution is:

```text
0.5 * |P_current(x) - P_lookback(x)|
```

These rows identify which measured outcomes caused the empirical distribution
to move. `top_k_contributions` controls how many are retained.

### Audit retention policy

`AuditPolicy` controls the detail-versus-size trade-off:

```python
AuditPolicy(
    include_batch_counts=True,
    include_check_counts=False,
    top_k_contributions=10,
)
```

- `include_batch_counts=True` stores the raw count map for every accepted batch.
  This provides direct replay evidence but can produce large logs.
- `include_check_counts=True` stores both cumulative count maps used by every TVD
  check. This is the most self-contained mode, but duplicates substantial data.
- When a count map is not retained, the audit still stores its shot count,
  support size, and SHA-256 fingerprint.
- `top_k_contributions=0` disables per-outcome contribution rows.

For large output spaces, a practical default is to retain raw batch counts,
omit repeated cumulative check counts, and archive the source batches in an
immutable object store. The count fingerprints then bind the audit events to the
archived source data.

### Tamper evidence and trust boundary

Every event is chained to the preceding event with SHA-256. This detects local
modification, insertion, deletion, and reordering within an exported audit log.
It does not by itself provide non-repudiation: an actor that can replace the
whole file can replace the complete chain.

For stronger assurance, persist the final head hash outside the audit file, for
example by:

- Signing it with an organizational key.
- Writing it to an append-only database or transparency log.
- Attaching it to an experiment-tracking record.
- Publishing it with the corresponding result artifact.

The audit establishes what the instrumented controller observed and decided. It
does not prove that an external backend honestly produced the supplied counts.
Backend job identifiers, provider receipts, or signed acquisition records should
therefore be included in `context` when available.

### Current audit scope

The prototype instruments the single-controller `run_stable_shots()` path in
`src/stableshot/main.py`. The multi-QPU controllers in
`src/stableshot/multi_qpu_main.py` are not yet instrumented. A multi-QPU audit
should use the same event format with a `controller_id` for each local backend
and the aggregate controller, and should additionally record backend weights,
active/frozen status, and the distributions included in every aggregate check.

See `AUDIT_DESIGN.md` for the design rationale and the proposed multi-QPU
extension.

## Paper Results

The evaluation uses 180 QSimBench traces:

- 6 circuit families: `dj`, `qaoa`, `qft`, `qnn`, `random`, `vqe`
- 6 sizes: 4, 6, 8, 10, 12, and 14 qubits
- 5 noisy IBM simulated backends
- 20,000-shot empirical references

StableShots configurations are selected through a 75-point grid:

- `batch_size`: `50`
- `lookback_batches`: `1,2,3,5,10`
- `stability`: `1,2,3,5,10`
- `epsilon`: `0.001,0.0025,0.005`

The paper uses 100 backend-holdout repetitions. In each repetition, one backend
per algorithm-size cell is held out for test evaluation and the remaining
backends are used for validation.

For the TVD `<= 0.05` target, the selected configuration reaches the target on
all held-out test evaluations with median 7,650 shots. On the full 180-trace
benchmark, it reaches the same target with median 7,700 shots.

## Repository Layout

```text
src/stableshot/main.py                         Main StableShots grid and baseline experiments
src/stableshot/audit.py                        Audit events, explanations, verification, and plotting
src/stableshot/multi_qpu_main.py               Multi-QPU experiment implementation
src/stableshot/select_conf.py                  Strategy ranking and Pareto-frontier selection
src/stableshot/q3.py                           Hoeffding/Weissman scaling analysis
src/stableshot/grid_size_robust_selection.py   Size-aware robust selection over aggregate outputs
tests/test_audit.py                            Audit behavior and tamper-detection tests
example_audit/                                 Example JSONL, explanation, and plot
AUDIT_DESIGN.md                                Audit architecture and extension notes
results/                                       Materialized experiment outputs
strategy_selection/                            Strategy-selection summaries
select.sh                                      Example strategy-selection command
q3.sh                                          Example RQ3 bound-scaling command
```

Key result files include:

```text
results/grid_selected_test_summary.csv
results/grid_selected_test_metrics.csv
results/grid_repeated_selected_summary.csv
results/grid_config_frequency.csv
results/grid_size_robust_selected_test_summary.csv
results/rq3_grouped_scaling_sensitivity/table1_style_comparison.csv
strategy_selection/recommended_strategies.csv
strategy_selection/pareto_frontier.csv
```

## Installation

The project is configured with `uv` and requires Python 3.12 or newer.

```bash
uv sync
```

The main dependencies are `pandas`, `matplotlib`, and `qsimbench`.

## Running the Tests

Run the audit tests with:

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests -v
```

The tests cover stable-decision explanation, distinction between input
exhaustion and budget exhaustion, JSONL round-tripping, and hash-chain tamper
detection.

## Running the Experiments

Run a StableShots grid over QSimBench traces:

```bash
uv run python src/stableshot/main.py \
  --lookback-batches 1,2,3,5,10 \
  --stability 1,2,3,5,10 \
  --epsilon 0.001,0.0025,0.005 \
  --validation-test-mode backend_holdout \
  --split-repetitions 100 \
  --selection-target-tvd 0.05 \
  --output-dir results/stable_shots_grid
```

Select and rank StableShots configurations from trace metrics:

```bash
uv run python src/stableshot/select_conf.py \
  --trace-metrics results/stable_shots_grid/trace_metrics.csv \
  --target-tvd 0.05 \
  --min-success-rate 1.0 \
  --risk-metric max \
  --objective min_median_shots
```

Run the grouped Hoeffding/Weissman scaling analysis:

```bash
uv run python src/stableshot/q3.py \
  --fixed-subset-metrics results/aggregate_fixed_subset_metrics.csv \
  --stableshots-trace-metrics results/aggregate_stableshots_subset_metrics.csv \
  --output-dir results/rq3_grouped_scaling_sensitivity \
  --taus 0.05 \
  --bound-delta 0.05 \
  --calibrations median,p75,p90 \
  --group-modes global,size,size_algorithm \
  --split-repetitions 100
```

The helper scripts `select.sh` and `q3.sh` contain shorter versions of these
commands for the current checked-in result layout.

## Reusing StableShots

The core stopping rule is represented by `StableShotsConfig` in
`src/stableshot/main.py`. A configuration is identified by:

```text
b{batch_size}_lb{lookback_batches}_k{stability}_eps{epsilon}
```

For example, `b50_lb3_k5_eps0p005` compares the current cumulative distribution
against the one 150 shots earlier and requires five consecutive stable checks.

To apply the rule outside QSimBench, provide repeated batches of counts for the
same static circuit and backend, preserve cumulative counts, and evaluate the
same marginal TVD stopping condition after each batch. Use the audited API when
that external system also needs data provenance, decision explanations, or a
verifiable execution record.

## Scope and Limitations

StableShots targets static circuits: the circuit structure and parameters must
remain fixed while shots are collected. It is not a replacement for
observable-specific measurement allocation in variational algorithms, although
it can be used locally inside a single fixed-parameter iteration.

The reported TVD values are measured against finite 20,000-shot noisy-backend
empirical references, not against the unknown true backend-induced
distributions. The experiments use noisy simulated QSimBench backends; live QPU
behavior, queueing overhead, and backend drift require separate evaluation.

An audit trail improves traceability and reproducibility, but it does not turn
the stopping heuristic into a statistical guarantee. It explains the data and
history used by the rule and why the implemented rule stopped.

## Citation

If you use this repository, cite the related paper:

```bibtex
@inproceedings{Bisicchia2026Stableshots,
  author       = {Bisicchia, G. and Bocci, A. and Pimentel, E. and Brogi, A.},
  title        = {{StableShots: Adaptive Shot Control for Quantum Circuits}},
  booktitle    = {IEEE International Conference on Quantum Computing and Engineering (QCE 2026)},
  year         = {2026},
  note         = {In Press}
}
```
