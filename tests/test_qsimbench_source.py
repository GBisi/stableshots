import unittest
from collections import Counter
from unittest.mock import patch

from stableshot.qsimbench_source import (
    materialize_qsimbench_scenario,
    qsimbench_catalog,
)


class QSimBenchSourceTests(unittest.TestCase):
    @patch(
        "stableshot.qsimbench_source.get_index",
        return_value={"qaoa": {8: ["fake_kyiv", "aer_simulator"]}, "dj": {4: ["fake_kyiv"]}},
    )
    def test_catalog_is_normalized_and_sorted(self, get_index):
        catalog = qsimbench_catalog("circuit")
        self.assertEqual(list(catalog), ["dj", "qaoa"])
        self.assertEqual(catalog["qaoa"][8], ("aer_simulator", "fake_kyiv"))
        get_index.assert_called_once_with(circuit_kind="circuit")

    @patch("stableshot.qsimbench_source.load_qsimbench_batches")
    def test_materialization_reuses_demo_replay_contract(self, load_batches):
        load_batches.return_value = [
            (50, Counter({"00": 30, "11": 20})),
            (50, Counter({"00": 29, "11": 21})),
            (50, Counter({"00": 31, "11": 19})),
            (50, Counter({"00": 30, "11": 20})),
        ]
        scenario = materialize_qsimbench_scenario(
            algorithm="qaoa",
            size=8,
            backend="fake_kyiv",
            materialization_shots=200,
            source_batch_size=50,
            sampling_strategy="random",
            sampling_seed=7,
            fixed_budgets=(50, 100, 200, 500),
        )

        self.assertEqual(scenario.source, "QSimBench")
        self.assertEqual(scenario.total_shots, 200)
        self.assertEqual(sum(scenario.reference_counts.values()), 200)
        self.assertEqual(scenario.fixed_budgets, (50, 100, 200))
        self.assertEqual(scenario.context["algorithm"], "qaoa")
        self.assertEqual(scenario.context["backend"], "fake_kyiv")
        self.assertEqual(scenario.context["sampling_seed"], 7)
        self.assertEqual(scenario.recommended_config["max_shots"], 200)
        load_batches.assert_called_once()


if __name__ == "__main__":
    unittest.main()
