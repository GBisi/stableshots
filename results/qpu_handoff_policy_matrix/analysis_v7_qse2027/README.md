# Simplified Q-SE 2027 analysis

This analysis revision keeps the existing raw experiment data unchanged and changes only the paper-facing aggregation.

Run:

```bash
python experiments/analyze_qse2027_simplified.py
python experiments/plot_qse2027_simplified.py
```

The analysis writes derived CSVs to this directory and the figures to
`results/qpu_handoff_policy_matrix/plots_v7_qse2027/`.

## RQ1

RQ1 reports StableShots+keep trajectories for every ordered source/destination
QPU pair over the five failure fractions. TVD is reported before physical
shots. StableShots+restart and Fixed20k keep/restart remain as a corollary that
isolates the effect of retaining previous measurement/controller state.

## RQ2

`BEST` and `WORST` are retrospective quality oracles over the four realised
StableShots+keep candidate handoffs for each circuit/source/failure event.
`BEST` minimizes final TVD to the 200k Aer reference; `WORST` maximizes it.
They are descriptive bounds and are not deployable selectors.

The separate offline-20k diagnostic chooses the available QPU with the smallest
standalone Fixed20k TVD and counts whether it belongs to the retrospective BEST
set. Ties in the retrospective BEST set count as agreement.
