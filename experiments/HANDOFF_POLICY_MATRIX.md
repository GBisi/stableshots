# QPU handoff policy matrix

This experiment restarts the failover analysis around a 2x2 policy design:

- stopping rule: StableShots vs fixed shot budget;
- failure history: keep accumulated evidence vs discard and restart.

Fixed-shot policies are evaluated at both 10,000 and 20,000 shots, yielding six
concrete policies. The proposal is `stableshots_keep`; the other five concrete
configurations instantiate the three baseline concepts with the two fixed
budgets.

## Failure timing

Failure fraction is measured relative to the matched no-failure stopping count
of the same stopping rule on the source backend. This avoids comparing a 50%
failure in a 10k fixed run with 50% of an unrelated StableShots trace.

## Random replacement

For every event there are exactly four alternative target QPUs. Uniform-random
replacement is therefore evaluated exactly as the mean over the four observed
target outcomes. No Monte Carlo target draw is used.

## Primary outputs

1. No-failure TVD to Aer and shot count by QPU/stopping rule.
2. Handoff TVD and physical-shot distributions by policy, source QPU and
   failure fraction.
3. Per-event TVD-optimal and shot-optimal target, with exact random-vs-best
   regret in each objective and cross-objective cost.
4. Failure impact relative to the matched no-failure source execution,
   separately for random and informed target choice.
