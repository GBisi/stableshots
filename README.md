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

This is a diminishing-returns heuristic. It does not prove closeness to the
unknown backend-induced distribution; it stops when additional batches have
small observed effect on the cumulative empirical distribution.

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
src/stableshot/select_conf.py                  Strategy ranking and Pareto-frontier selection
src/stableshot/q3.py                           Hoeffding/Weissman scaling analysis
src/stableshot/grid_size_robust_selection.py   Size-aware robust selection over aggregate outputs
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
same marginal TVD stopping condition after each batch.

## Scope and Limitations

StableShots targets static circuits: the circuit structure and parameters must
remain fixed while shots are collected. It is not a replacement for
observable-specific measurement allocation in variational algorithms, although
it can be used locally inside a single fixed-parameter iteration.

The reported TVD values are measured against finite 20,000-shot noisy-backend
empirical references, not against the unknown true backend-induced
distributions. The experiments use noisy simulated QSimBench backends; live QPU
behavior, queueing overhead, and backend drift require separate evaluation.

## Citation

If you use this repository, cite the related paper:

```bibtex
@inproceedings{bisicchia2026stableshots,
  title = {StableShots: Online Shot Stopping for Quantum Circuit Execution},
  author = {Bisicchia, Giuseppe and Bocci, Alessandro and Pimentel, Ernesto and Brogi, Antonio},
  year = {2026}
}
```
