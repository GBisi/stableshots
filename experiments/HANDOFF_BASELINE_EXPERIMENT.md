# Clean directed QPU handoff experiment

## Goal

Measure the statistical effect of a single backend handoff without mixing in
recovery-policy comparisons.

The experiment answers:

1. How many physical shots does StableShots use when execution moves from backend
   A to backend B?
2. What is the estimator's TVD to a common high-shot Aer reference at the exact
   failure point?
3. What is the TVD of the target-only evidence collected after migration?
4. What is the TVD of the final cumulative estimator after the handoff?
5. How do those quantities change as the handoff occurs later?

## Design

The default design uses 6 algorithms x 6 circuit sizes = 36 circuits and 5 fake
QPU backends.

For every circuit:

- run StableShots with **no failure** on each of the 5 backends;
- enumerate every directed source-target pair with source != target:
  5 x 4 = 20 directed pairs;
- inject one handoff at each of:
  0.10, 0.25, 0.50, 0.75, 0.90.

This produces:

- 36 x 5 = **180 no-failure runs**;
- 36 x 20 x 5 = **3,600 handoff runs**.

No backend pairs are collapsed into ascending/descending/random scenarios in this
experiment. Each directed pair remains identifiable in the raw data and primary
summary.

## Failure location

For a source backend A, first obtain its matched no-failure StableShots stopping
count S_A.

For requested fraction f, the handoff occurs at the nearest lower complete
StableShots batch:

    S_fail = floor(f * S_A / batch_size) * batch_size

with guards ensuring at least one source batch and that the handoff precedes the
matched no-failure stopping point.

Thus "50% failure" means approximately halfway through the source backend's own
no-failure StableShots execution, not halfway through an arbitrary fixed budget.

## Handoff semantics

There is one policy only:

**naive cumulative continuation**

At the handoff, all StableShots state is preserved:

- cumulative measurement counts;
- lookback snapshots;
- stability streak;
- stability-check history.

The next batch comes from the target backend. No reset, restart, decay, or
quality-aware adaptation is performed.

The total physical-shot budget remains capped at **20,000** shots. A handoff
therefore does not receive an extra post-failure budget.

## Common ground truth

Every measurement is compared with the same per-circuit high-shot Aer simulator
reference. The default reference uses **200,000 shots**.

The backend traces are never used to define an oracle ordering in this
experiment.

## Recorded metrics

### No-failure run

For each circuit-backend pair:

- shots;
- final_tvd_to_aer;
- stop_reason;
- last_stability_tvd.

### Handoff run

For each circuit, directed source-target pair, and failure fraction:

- source_no_failure_shots;
- source_no_failure_final_tvd_to_aer;
- target_no_failure_shots;
- target_no_failure_final_tvd_to_aer;
- failure_shots;
- actual_failure_fraction;
- **failure_point_tvd_to_aer**;
- stable_streak_at_failure;
- last_stability_tvd_at_failure;
- post_failure_shots;
- **post_failure_only_tvd_to_aer**;
- total_shots;
- source_evidence_share;
- target_evidence_share;
- **final_aggregated_tvd_to_aer**;
- delta_tvd_failure_to_final;
- delta_tvd_vs_source_no_failure;
- delta_tvd_vs_target_no_failure;
- shot_delta_vs_source_no_failure;
- stop_reason.

The two "partial" TVD quantities are intentionally distinct:

- failure_point_tvd_to_aer measures the cumulative source estimator at migration;
- post_failure_only_tvd_to_aer measures only the evidence collected from the
  replacement backend.

The final_aggregated_tvd_to_aer then measures the actual cumulative estimator
that StableShots returns.

## Primary analysis

Primary summaries are grouped by:

- source_backend;
- target_backend;
- failure_fraction.

This preserves direction. In particular, A->B and B->A are never merged in the
primary result.

For each pair/fraction the analysis reports medians, IQRs and bootstrap 95%
confidence intervals for shot counts and TVD metrics.

Heatmaps are generated separately for each failure fraction for:

- final aggregated TVD;
- total physical shots;
- TVD at failure;
- target-only post-failure TVD;
- final TVD minus failure-point TVD.

The analysis deliberately avoids a headline signed TVD statistic pooled across
opposite directed backend pairs.
