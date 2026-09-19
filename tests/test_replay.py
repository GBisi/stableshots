import tempfile
import unittest
from pathlib import Path

from stableshot.replay import ReplayFormatError, load_bundled_scenario, load_replay


class ReplayTests(unittest.TestCase):
    def test_repeat_expands_to_batches(self):
        scenario = load_bundled_scenario("early-stability")
        self.assertEqual(len(scenario.batches), 20)
        self.assertEqual(scenario.total_shots, 1000)
        self.assertEqual(sum(scenario.reference_counts.values()), 1000)

    def test_invalid_shot_total_is_rejected(self):
        content = "\n".join(
            [
                '{"type":"metadata","schema_version":"stableshot-replay-v1","scenario_id":"bad"}',
                '{"type":"batch","shots":50,"counts":{"0":49}}',
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text(content, encoding="utf-8")
            with self.assertRaises(ReplayFormatError):
                load_replay(path)


if __name__ == "__main__":
    unittest.main()
