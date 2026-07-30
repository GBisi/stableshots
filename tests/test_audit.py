import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from stableshot.audit import AuditPolicy, read_jsonl, verify_events
from stableshot.main import StableShotsConfig, run_stable_shots_audited


class StableShotsAuditTests(unittest.TestCase):
    def test_stable_decision_is_explained_and_hash_chain_verifies(self):
        raw_batches = [(50, Counter({"0": 50})) for _ in range(5)]
        config = StableShotsConfig(batch_size=50, lookback_batches=1, stability=2, epsilon=0.0, max_shots=250)

        counts, shots, last_delta, reason, audit = run_stable_shots_audited(
            raw_batches,
            config,
            context={"trace_id": "demo", "backend": "fake"},
            policy=AuditPolicy(include_batch_counts=True, include_check_counts=True),
        )

        self.assertEqual(reason, "stable")
        self.assertEqual(shots, 150)
        self.assertEqual(last_delta, 0.0)
        self.assertEqual(sum(counts.values()), 150)
        explanation = audit.explain()
        self.assertEqual(explanation["reason_code"], "stable")
        self.assertIn("2 consecutive checks", explanation["explanation"])
        self.assertTrue(verify_events(audit.events)["valid"])

    def test_input_exhaustion_is_not_mislabeled_as_budget_stop(self):
        raw_batches = [(50, Counter({"0": 50})), (50, Counter({"1": 50}))]
        config = StableShotsConfig(batch_size=50, lookback_batches=1, stability=3, epsilon=0.0, max_shots=200)
        _, shots, _, reason, audit = run_stable_shots_audited(raw_batches, config)
        self.assertEqual(shots, 100)
        self.assertEqual(reason, "input_exhausted")
        self.assertEqual(audit.explain()["reason_code"], "input_exhausted")

    def test_tampering_is_detected(self):
        raw_batches = [(50, Counter({"0": 50})) for _ in range(3)]
        config = StableShotsConfig(batch_size=50, lookback_batches=1, stability=1, epsilon=0.0, max_shots=150)
        _, _, _, _, audit = run_stable_shots_audited(raw_batches, config)
        tampered = json.loads(json.dumps(audit.events))
        tampered[1]["payload"]["batch_shots"] = 49
        result = verify_events(tampered)
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "event_hash_mismatch")

    def test_jsonl_round_trip(self):
        raw_batches = [(50, Counter({"0": 25, "1": 25})) for _ in range(3)]
        config = StableShotsConfig(batch_size=50, lookback_batches=1, stability=1, epsilon=0.0, max_shots=150)
        _, _, _, _, audit = run_stable_shots_audited(raw_batches, config)
        with tempfile.TemporaryDirectory() as tmp:
            path = audit.write_jsonl(Path(tmp) / "audit.jsonl")
            self.assertTrue(verify_events(read_jsonl(path))["valid"])


if __name__ == "__main__":
    unittest.main()
