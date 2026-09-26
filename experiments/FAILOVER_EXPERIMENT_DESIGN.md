# StableShots QPU-Failover Experiments

## Scope

This branch studies **fault resilience of a single active QPU execution**. At every instant exactly one QPU supplies measurement shots. If that QPU becomes unavailable, execution migrates to one replacement QPU and continues.

The study intentionally does **not** propose a production QPU-selection algorithm. Real selection can depend on queue length, availability, calibration, monetary cost, provider constraints, topology, or multi-objective policies. Instead, replacement quality is controlled experimentally using oracle rankings computed from high-shot references, plus random replacement. This isolates the statistical effect of migration on StableShots.

The main target is the high-shot QSimBench **aer_simulator** distribution for the same circuit. The experiment therefore asks whether the final empirical distribution remains close to an ideal/noiseless simulator reference, not merely whether it matches the noisy distribution of the active backend.

All parameters are stored in **experiments/failover_config.json**.

The experiment launcher is **experiments/run_failover_experiments.py**.

The analysis script is **experiments/analyze_failover_results.py**.

The configured output root is **results/qpu_failover/**.

## Reproducible execution

From the repository root:

~~~bash
uv sync
uv run python experiments/run_failover_experiments.py \
  --config experiments/failover_config.json

uv run python experiments/analyze_failover_results.py \
  --config experiments/failover_config.json
~~~

The runner refuses to overwrite an existing result directory containing a run manifest. The **--resume** option is available only for deliberate continuation/appending.

For debugging or staged execution:

~~~bash
uv run python experiments/run_failover_experiments.py --only single
uv run python experiments/run_failover_experiments.py --only multi
uv run python experiments/run_failover_experiments.py --only stochastic
~~~

## Circuits and backend traces

The default configuration uses QSimBench circuit traces for:

- algorithms: dj, qaoa, qft, qnn, random, vqe;
- sizes: 4, 6, 8, 10, 12, 14;
- candidate QPUs:
  - fake_fez
  - fake_kyiv
  - fake_marrakesh
  - fake_sherbrooke
  - fake_torino;
- ideal backend: aer_simulator.

Each noisy backend receives a 20,000-shot high-shot reference. The ideal reference is sampled at 200,000 shots by default. Both values are configurable.

QSimBench random sampling is seeded deterministically. The runner materializes a backend stream once for a circuit and reuses that stream across recovery policies so policy comparisons do not accidentally compare different random measurement streams.

## StableShots configuration

The default selected StableShots configuration is:

~~~text
batch_size       = 50
lookback_batches = 3
stability        = 5
epsilon          = 0.005
max_shots        = 20000
~~~

The configuration is read from the JSON file rather than hard-coded into the experiment logic.

## Reference quantities

For circuit c, let the high-shot ideal reference be

\[
P_{\mathrm{Aer}}^{(c)}.
\]

For noisy backend q, let its high-shot reference be

\[
P_q^{(c)}.
\]

The oracle backend error is

\[
E(q,c)=
\operatorname{TVD}
\left(
P_q^{(c)},P_{\mathrm{Aer}}^{(c)}
\right).
\]

Lower values correspond to a backend whose observed high-shot output is closer to the Aer reference for that circuit.

For a directed handoff \(A\rightarrow B\), the experiment also records

\[
S_{A,B}=
\operatorname{TVD}(P_A^{(c)},P_B^{(c)})
\]

and

\[
\Delta E_{A\rightarrow B}
=
E(B,c)-E(A,c).
\]

A negative quality change means migration moved toward a backend closer to Aer; a positive quality change means migration moved toward a less accurate backend. Pair TVD is symmetric, while handoff quality change is directional.

## Baselines

### B0 — No failure

StableShots executes normally on one QPU until it stops or reaches its normal shot cap.

This baseline supplies the counterfactual no-failure shot count

\[
N_0(c,q)
\]

and no-failure final TVD for every circuit/backend pair.

### B1 — Full restart

At failure:

1. all pre-failure measurement evidence is discarded from the output;
2. StableShots decision state is reset;
3. execution starts from zero on the replacement QPU.

Previously executed shots still count toward **physical resource cost**. They are not hidden from shot-overhead metrics.

This is the conservative, wasteful recovery baseline.

### B2 — Naive continuation

At failure the source of future batches changes, but the following are all retained:

- cumulative counts;
- look-back snapshots;
- stability streak;
- previous stopping evidence.

This models a transparent failover implementation that assumes the QPU change does not alter the semantics of the measurement stream.

## Recovery policies

### R1 — Controller reset

Pre-failure counts are retained, but StableShots decision history is reset:

- stability streak = 0;
- look-back snapshot history is cleared;
- future checks are reconstructed from the post-handoff batches.

The cumulative distribution used by the controller still includes all old counts.

This isolates whether stale StableShots state itself is the problem while preserving complete measurement history.

### R2 — Epoch-local StableShots

At handoff, a fresh StableShots controller is started using only shots from the replacement QPU.

The old shots do **not** affect the post-handoff stopping decision, but they remain in the final reported empirical distribution.

This deliberately separates:

- evidence used to decide that the current QPU has stabilized;
- evidence retained in the final measurement result.

It is primarily an ablation for understanding those two roles.

### R3 — Fixed decay

At failure, old measurement evidence is downweighted by a fixed retention factor

\[
0\le\lambda\le1.
\]

If pre-failure weighted counts are \(C_{\text{old}}\), the post-handoff effective counts are

\[
C_{\text{effective}}
=
\lambda C_{\text{old}}+C_{\text{new}}.
\]

Default fixed values are

\[
\lambda\in\{0.25,0.5,0.75\}.
\]

The StableShots decision state is reset at every handoff. Only the measurement evidence is partially retained.

Physical shot accounting is never discounted. If 8,000 shots were physically executed and decay leaves 5,500 effective retained shots, resource cost remains 8,000.

### R4 — Shift-aware decay

This policy adapts \(\lambda\) to the observed handoff discontinuity. It does not use Aer ground truth or oracle QPU rankings.

Suppose the current effective cumulative pre-handoff distribution is \(\hat P_A\). After switching to B, the policy collects a probe of w ordinary execution shots. These probe shots are not wasted: they become the first shots of the new epoch.

The policy uses equal-size bootstrap comparisons.

For each bootstrap repetition k:

1. sample w synthetic shots from the current cumulative distribution:
   \[
   \hat P^{(k)}_{A,w}\sim\hat P_A;
   \]
2. compare this sample with the actual w-shot probe:
   \[
   D_k^{\mathrm{handoff}}
   =
   \operatorname{TVD}
   \left(
   \hat P^{(k)}_{A,w},
   \hat P_{B,w}
   \right);
   \]
3. independently sample another w-shot distribution from \(\hat P_A\):
   \[
   \hat P^{(k,2)}_{A,w}\sim\hat P_A;
   \]
4. estimate ordinary finite-shot disagreement under no shift:
   \[
   D_k^{\mathrm{null}}
   =
   \operatorname{TVD}
   \left(
   \hat P^{(k)}_{A,w},
   \hat P^{(k,2)}_{A,w}
   \right).
   \]

The implementation records

\[
\bar D_{\mathrm{handoff}}
=
\frac1K\sum_kD_k^{\mathrm{handoff}}
\]

and

\[
\bar D_{\mathrm{null}}
=
\frac1K\sum_kD_k^{\mathrm{null}}.
\]

The bootstrap-corrected shift diagnostic is

\[
S_{\mathrm{corrected}}
=
\max
\left(
0,
\bar D_{\mathrm{handoff}}
-
\bar D_{\mathrm{null}}
\right).
\]

The default parameter-free retention rule is

\[
\lambda
=
\min
\left(
1,
\frac{\bar D_{\mathrm{null}}}
{\max(\bar D_{\mathrm{handoff}},\varepsilon_{\mathrm{null}})}
\right).
\]

Thus:

- handoff disagreement comparable to ordinary finite-shot variation gives \(\lambda\approx1\);
- a handoff roughly twice the ordinary disagreement gives \(\lambda\approx0.5\);
- a very large handoff discontinuity pushes \(\lambda\) toward zero.

The script also records the empirical percentile of the handoff mean relative to the null bootstrap distribution.

The default probe is five StableShots batches, i.e. w = 250 shots, with 200 bootstrap repetitions. Both are configurable.

## Main experiment: exhaustive single failure

The main controlled experiment contains exactly one failure.

For each circuit c and every source QPU A, StableShots is first run without failure to determine its counterfactual stopping point \(N_0(c,A)\).

Failures are then injected at:

\[
\tau\in\{0.10,0.25,0.50,0.75,0.90\}
\]

of that source-specific no-failure stopping point:

\[
N_f
=
\left\lfloor
\frac{\tau N_0}{b}
\right\rfloor b,
\]

where b is the batch size. The failure location is constrained to remain before the no-failure stop.

This definition is preferable to taking percentages of the global 20,000-shot cap because it guarantees that the injected failure occurs during an execution that would otherwise still be running.

Every directed source/target pair is simulated:

\[
A\rightarrow B,\qquad A\neq B.
\]

With five QPUs there are 20 directed handoffs per circuit.

All baselines and recovery policies consume the same source and target materialized streams for a given handoff.

## Oracle replacement conditions

Production replacement selection is outside the study scope.

The analysis derives three controlled replacement conditions from the exhaustive pair experiment.

### Most reliable remaining

For a failed source A, select

\[
B=
\arg\min_{q\neq A}E(q,c).
\]

### Least reliable remaining

Select

\[
B=
\arg\max_{q\neq A}E(q,c).
\]

### Random remaining

Uniformly select a remaining backend. The default analysis uses 30 deterministic random repetitions per circuit/source pair.

Because these conditions are derived after exhaustive pair simulation, no experiment has to be rerun to change the oracle aggregation logic.

## Directed QPU-pair handoff analysis

Every directed pair \(A\rightarrow B\) is analyzed independently.

The analysis produces pair-level summaries and heatmaps for:

- final TVD to Aer;
- TVD change relative to no failure;
- physical shot overhead;
- post-failure shots;
- target-violation rate.

It also computes Spearman correlations between outcomes and:

- pair reference TVD \(S_{A,B}\);
- target backend error \(E(B,c)\);
- quality change \(\Delta E\);
- source backend error \(E(A,c)\).

This distinguishes two effects that oracle reliability ranking alone cannot separate:

1. whether the replacement QPU is better or worse in absolute ideal accuracy;
2. how discontinuous the handoff is relative to the old QPU.

## Multiple-failure robustness experiment

The primary scientific analysis remains the single-failure experiment because it provides a clean causal object:

\[
(A,\tau,B,\text{policy}).
\]

A secondary robustness experiment tests whether recovery logic composes across **two scheduled failures**.

The configured sequence conditions are:

- ascending_reliability: worst-to-best oracle order;
- descending_reliability: best-to-worst oracle order;
- random: random backend permutation.

The default planned failures occur at 25% and 50% of the initial backend's no-failure stopping count. A run may stop before a later scheduled failure; therefore the raw table records both planned_failures and actual_failures.

The multi-failure experiment intentionally uses a smaller policy subset configured in JSON. The default includes:

- full restart;
- naive continuation;
- controller reset;
- fixed decay with \(\lambda=0.5\);
- shift-aware decay.

## Stochastic-failure robustness experiment

A final robustness experiment samples an independent failure event after each completed batch with probability p.

The defaults are:

\[
p\in\{0.0025,0.005,0.01\}.
\]

For 50-shot batches these correspond to expected pre-failure run lengths of approximately:

- p = 0.0025: 400 batches, about 20,000 shots;
- p = 0.005: 200 batches, about 10,000 shots;
- p = 0.01: 100 batches, about 5,000 shots.

The number of failures is capped by configuration (default: two). Each condition is repeated with deterministic random seeds.

This experiment is not used to explain the mechanism; it tests whether conclusions from the controlled experiment survive random failure timing.

## Metrics

### Primary accuracy

\[
\operatorname{TVD}
\left(
\hat P_{\mathrm{final}},
P_{\mathrm{Aer}}
\right).
\]

### Failure-induced TVD change

For single failures:

\[
\Delta\mathrm{TVD}
=
\mathrm{TVD}_{\mathrm{failure}}
-
\mathrm{TVD}_{\mathrm{no\ failure}}.
\]

Positive values indicate degradation relative to the corresponding source-QPU counterfactual. Negative values indicate that migration improved closeness to Aer.

### Target violation

The default accuracy target is:

\[
\delta=0.05.
\]

A run is marked target_violation=true when

\[
\mathrm{TVD}_{\mathrm{final}}>\delta.
\]

### Physical shots

Every shot sent to any QPU is counted:

\[
N_{\mathrm{physical}}.
\]

Discarded or downweighted shots remain part of this cost.

### Post-failure shots

For the single-failure experiment:

\[
N_{\mathrm{post}}
=
N_{\mathrm{physical}}-N_f.
\]

This is the recovery-shot requirement.

### Effective retained shots

Weighted recovery produces an effective measurement mass

\[
N_{\mathrm{effective}}
=
\sum_x C_{\mathrm{effective}}(x).
\]

This is not a resource-cost metric. It describes the amount of evidence retained by the estimator.

### Discarded/downweighted evidence

\[
N_{\mathrm{lost}}
=
N_{\mathrm{physical}}
-
N_{\mathrm{effective}}.
\]

For a restart, this includes all discarded pre-failure shots. For decay, it includes the fraction removed by weighting.

### Shot overhead relative to no failure

\[
\Delta N
=
N_{\mathrm{physical,failure}}
-
N_0.
\]

The raw tables also include the ratio of failure-run physical shots to no-failure shots.

### Shift-aware diagnostics

For shift-aware runs the raw output includes:

- shift_handoff_mean_tvd;
- shift_null_mean_tvd;
- shift_corrected_tvd;
- shift_null_percentile;
- adaptive_lambda.

These fields allow later sensitivity analysis without rerunning the experiment.

## Interpretation discipline

The oracle reliability conditions must not be described as production scheduling methods. They are controlled experimental treatments.

A result such as "most reliable remaining performs better" establishes sensitivity to replacement quality; it does not establish that an online system can know that ranking for arbitrary circuits.

Likewise, pair reference TVD is a post-hoc explanatory variable. The shift-aware recovery method does not receive the high-shot pair TVD; it estimates handoff change only from previously observed cumulative evidence and the replacement-QPU probe.

## Expected output tree

~~~text
results/qpu_failover/
├── run_manifest.json
├── raw/
│   ├── qpu_references.csv
│   ├── no_failure_runs.csv
│   ├── fixed_shot_runs.csv
│   ├── single_failure_runs.csv
│   ├── multi_failure_runs.csv
│   ├── stochastic_failure_runs.csv
│   └── failures.csv
├── analysis/
│   ├── no_failure_summary.csv
│   ├── fixed_shot_summary.csv
│   ├── single_failure_summary.csv
│   ├── handoff_pair_summary.csv
│   ├── handoff_predictor_correlations.csv
│   ├── decay_sensitivity.csv
│   ├── oracle_replacement_runs.csv
│   ├── oracle_replacement_summary.csv
│   ├── multi_failure_summary.csv
│   └── stochastic_failure_summary.csv
└── plots/
    ├── failure_fraction_accuracy.png
    ├── failure_fraction_shot_overhead.png
    ├── oracle_replacement_accuracy.png
    ├── oracle_replacement_shot_overhead.png
    ├── shift_aware_lambda_response.png
    ├── multi_failure_accuracy.png
    ├── stochastic_failure_accuracy.png
    └── handoff_heatmaps/
~~~

See **experiments/FAILOVER_RESULTS_GUIDE.md** for the exact meaning and intended use of each output.
