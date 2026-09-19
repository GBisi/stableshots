# StableShots

StableShots is an online shot-stopping rule for static quantum-circuit execution. It consumes measurement counts in batches and stops after repeated evidence that the cumulative empirical output distribution has stabilized.

This repository includes an interactive demo artifact for the ICSOC 2026 Demonstrations and Resources track. The demo exposes the existing StableShots controller, its decision history, fixed-budget comparisons, and the tamper-evident audit log through a browser interface.

## Try the interactive demo

The fastest path is Docker:

```bash
docker compose up --build
```

Open `http://localhost:8501`.

The UI supports two trace sources:

- **Bundled replay**: three deterministic synthetic scenarios that work offline;
- **QSimBench**: browse the current QSimBench catalog by circuit kind, algorithm, size, and backend, materialize the selected trace, and run StableShots on those measurement batches.

The UI has four tabs:

1. **Execute** shows consumed shots, the stop reason, marginal TVD, the threshold, and the decision history. The chart marks the exact controller stop with a vertical stop indicator.
2. **Explain** reconstructs the decisive checks and the outcomes contributing most to the last TVD change. Decision metrics are formatted compactly rather than exposing binary floating-point artifacts.
3. **Compare** uses the same materialized measurement stream for fixed-shot budgets and StableShots. TVD to the materialized reference is shown only as a post-hoc evaluation metric and is never passed to the online controller.
4. **Audit** shows the complete event timeline, supports filtering and previous/next or direct event navigation, exposes each event payload and hash-chain metadata, downloads the full JSONL record, and demonstrates tamper detection.

### Bundled offline scenarios

The conference-safe path does not need a QPU, GPU, provider account, or network access after the image has been built:

- `early-stability`: a stationary stream that stops early;
- `late-stability`: an alternating stream that reaches the configured cap;
- `audit-walkthrough`: the compact sequence used to explain and tamper with an audited decision.

### Browse and run QSimBench traces

Choose **QSimBench** under **Trace source** in the sidebar. The application calls QSimBench's `get_index()` API and exposes the available hierarchy:

```text
circuit kind -> algorithm -> size -> backend
```

For the selected trace you can also choose:

- sequential or random QSimBench sampling;
- the sampling seed;
- the number of shots to materialize, up to 20,000;
- whether to force a refresh of the QSimBench cache.

Click **Load QSimBench trace** to materialize the selected execution history. The resulting batches are kept in the Streamlit session, so StableShots parameters can be changed and rerun without downloading or resampling the trace again.

QSimBench mode needs Internet access to browse the dataset index and to fetch history files that are not already cached. The Docker Compose setup keeps the QSimBench cache in a named volume. GitHub allows a much larger API quota when a token is supplied, so for repeated browsing you can optionally run:

```bash
export GITHUB_TOKEN=ghp_your_token
docker compose up --build
```

Do not commit the token. The bundled replay mode remains available when Internet access is unavailable.

The final materialized QSimBench prefix is used as the post-hoc empirical reference, matching the evaluation pattern used by the research code: StableShots itself receives only the batch sequence. In sequential mode, materialization follows QSimBench's process-local sequential cursor at load time. Once loaded, the demo reuses that fixed materialized stream.

### Local installation

The project uses Python 3.12 or newer and `uv`.

```bash
uv sync --extra demo
uv run stableshot self-check
uv run stableshot demo
```

Then open `http://127.0.0.1:8501`.

The deterministic self-check exercises the bundled scenarios, verifies each generated audit, and confirms that a controlled mutation is detected. It intentionally does not make network calls to QSimBench.

## Replay format

Bundled scenarios live in `src/stableshot/demo_data/` and use JSON Lines with schema identifier `stableshot-replay-v1`.

A replay begins with one metadata object:

```json
{"type":"metadata","schema_version":"stableshot-replay-v1","scenario_id":"example","reference_counts":{"00":60,"11":40},"recommended_config":{"batch_size":50,"lookback_batches":1,"stability":2,"epsilon":0.01,"max_shots":100}}
```

It is followed by batch records:

```json
{"type":"batch","shots":50,"counts":{"00":30,"11":20}}
{"type":"batch","shots":50,"counts":{"00":30,"11":20}}
```

A batch may include `"repeat": N` to compact repeated deterministic batches. The loader validates that every count map sums to the declared number of shots. The optional `reference_counts` field is used only after execution for evaluation.

The QSimBench adapter in `src/stableshot/qsimbench_source.py` materializes a selected QSimBench trace directly into the same `ReplayScenario` abstraction used by bundled files. This keeps the controller, comparison, explanation, and audit paths identical for both sources.

## Command-line interface

```bash
stableshot demo [--host HOST] [--port PORT]
stableshot self-check
stableshot audit verify AUDIT.jsonl
stableshot audit explain AUDIT.jsonl
stableshot audit plot AUDIT.jsonl OUTPUT.png
```

The package entry point resolves to `stableshot.cli:main`.

## Method

For a fixed circuit and backend, StableShots accumulates measurement counts over batches of `b` shots. After each batch it compares the current cumulative empirical distribution with the cumulative distribution from `lookback_batches` batches earlier using Total Variation Distance:

```text
TVD(P, Q) = 1/2 * sum_x |P(x) - Q(x)|
```

Execution stops when this marginal TVD is at most `epsilon` for `stability` consecutive checks, or when `max_shots` is reached.

The selected paper configuration is:

```text
b50_lb3_k5_eps0p005
```

StableShots is a diminishing-returns heuristic. It does not certify closeness to the unknown backend-induced distribution.

## Auditable decisions

Use `run_stable_shots_audited()` when the caller also needs provenance and a reconstructable stopping decision. The audit records:

| Event | Purpose |
| --- | --- |
| `run_started` | configuration, execution context, retention policy, decision scope |
| `batch_accepted` | batch and cumulative-count fingerprints, optional raw counts |
| `stability_check` | compared rounds, TVD, threshold result, streak transition, top outcome contributions |
| `stop_decision` | final reason, shot count, stable streak, and final evidence |

Every event includes the previous event hash and its own SHA-256 hash. This detects local modification, insertion, deletion, and reordering inside an exported log. It does not provide non-repudiation if an actor can replace the complete file.

For a QSimBench execution, the audit context records the algorithm, size, backend, circuit kind, sampling strategy, seed, source batch size, and materialized shot count. Reference counts are deliberately not included in the online audit context.

## Repository layout

```text
src/stableshot/main.py              StableShots controller and paper experiments
src/stableshot/audit.py             audit events, explanation, verification, plotting
src/stableshot/replay.py            deterministic replay format and loader
src/stableshot/qsimbench_source.py  QSimBench catalog and trace materialization adapter
src/stableshot/demo.py              demo orchestration and artifact self-check
src/stableshot/webapp.py            Streamlit application
src/stableshot/cli.py               package command-line interface
src/stableshot/demo_data/           bundled offline replay scenarios
demo/app.py                         direct Streamlit launcher
Dockerfile                          containerized demo
compose.yaml                        one-command conference setup
tests/test_audit.py                 existing audit tests
tests/test_replay.py                replay-format tests
tests/test_demo.py                  deterministic artifact tests
tests/test_qsimbench_source.py       QSimBench adapter tests with mocked network API
```

## Running tests

```bash
uv sync --extra demo
PYTHONPATH=src uv run python -m unittest discover -s tests -v
```

The QSimBench adapter tests mock QSimBench network access, so the unit suite remains deterministic and can run offline.

## Running the paper experiments

The original experiment commands remain available. For example:

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

## Scope and limitations

StableShots targets static circuits. The paper reports TVD against finite 20,000-shot noisy-backend empirical references, not against unknown true backend-induced distributions. The experiments use noisy simulated QSimBench backends; live QPU behavior, queueing overhead, and backend drift require separate evaluation.

The bundled conference scenarios are deterministic synthetic replays. QSimBench mode gives the attendee access to the real benchmark trace catalog but introduces the expected network/cache dependency during materialization.

## Citation

```bibtex
@inproceedings{Bisicchia2026Stableshots,
  author       = {Bisicchia, G. and Bocci, A. and Pimentel, E. and Brogi, A.},
  title        = {{StableShots: Adaptive Shot Control for Quantum Circuits}},
  booktitle    = {IEEE International Conference on Quantum Computing and Engineering (QCE 2026)},
  year         = {2026},
  note         = {In Press}
}
```
