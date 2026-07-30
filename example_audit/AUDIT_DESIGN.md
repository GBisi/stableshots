# StableShots audit prototype

This prototype adds an optional, append-only decision audit without changing the existing
`run_stable_shots` return type.

```python
from stableshot.audit import AuditPolicy
from stableshot.main import StableShotsConfig, run_stable_shots_audited

counts, shots, delta, reason, audit = run_stable_shots_audited(
    raw_batches,
    StableShotsConfig(),
    context={
        "trace_id": "qaoa_10_fake_kyiv",
        "sampling_strategy": "sequential",
        "sampling_seed": 0,
        "source_batch_size": 50,
    },
    policy=AuditPolicy(
        include_batch_counts=True,
        include_check_counts=False,
        top_k_contributions=10,
    ),
)
audit.write_jsonl("results/audit/qaoa_10_fake_kyiv.jsonl")
print(audit.explain())
```

Inspect the resulting file:

```bash
python -m stableshot.audit verify results/audit/qaoa_10_fake_kyiv.jsonl
python -m stableshot.audit explain results/audit/qaoa_10_fake_kyiv.jsonl
python -m stableshot.audit plot results/audit/qaoa_10_fake_kyiv.jsonl results/audit/qaoa_10_fake_kyiv.png
```

The log contains:

1. `run_started`: complete StableShots configuration, input context, and retention policy.
2. `batch_accepted`: batch and cumulative count fingerprints, plus optional batch counts.
3. `stability_check`: current/lookback fingerprints, TVD, epsilon, pass/fail, streak transition,
   and the largest per-outcome TVD contributions.
4. `stop_decision`: stable, max-budget, or input-exhausted reason with final evidence.

Every event includes the previous event hash and its own SHA-256 hash. This detects edits,
reordering, deletion, and insertion inside an exported audit trail. For external
non-repudiation, periodically sign or publish the final head hash; a local hash chain alone
cannot prevent wholesale replacement of the entire file.

The online decision does **not** use reference counts or `tvd_to_reference`; those are
post-hoc evaluation data and should be stored in a separate `evaluation_result` record if
needed.
