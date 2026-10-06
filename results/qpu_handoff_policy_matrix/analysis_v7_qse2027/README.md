# Simplified Q-SE 2027 analysis

This revision keeps the raw experiment data unchanged and changes the paper-facing aggregation and visualization.

Run:

```bash
python experiments/analyze_qse2027_simplified.py
python experiments/plot_qse2027_simplified.py
```

The analysis writes derived CSVs to this directory and figures to
`results/qpu_handoff_policy_matrix/plots_v7_qse2027/`.

## RQ1

RQ1 asks what changes when a failure forces execution to move to another QPU.
For each source and failure point, `source_failure_prefix.csv` records the
median source TVD and completed source shots at the failure boundary. Each
paper trajectory then connects that point to the final StableShots+keep result
at 100% execution progress. Dotted source-colored lines show the matched
no-failure source endpoint. TVD is always shown before physical shots.

The corollary compares StableShots+keep/restart and Fixed20k+keep/restart to
isolate the measurement cost of discarding completed work. In the current
matrix, 2,610/3,600 StableShots+keep handoffs (72.5%) satisfy the stability
criterion within the 20k budget; 990/3,600 reach the cap without stabilizing.

## RQ2

`BEST` and `WORST` are retrospective quality oracles over the four realized
StableShots+keep candidates for each circuit/source/failure event. `BEST`
minimizes final TVD to the 200k Aer reference and `WORST` maximizes it. They
are descriptive bounds, not deployable selectors.

The offline-20k diagnostic chooses the available QPU with the smallest
standalone Fixed20k TVD and counts whether it belongs to the retrospective
BEST set. Ties in the retrospective BEST set count as agreement.
