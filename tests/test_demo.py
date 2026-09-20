import unittest

from stableshot.audit import verify_events
from stableshot.demo import (
    compare_policies,
    config_for_scenario,
    cumulative_distribution_snapshot,
    distribution_iterations,
    distribution_table_rows,
    execute_demo,
    select_distribution_outcomes,
    self_check,
    tamper_events,
)
from stableshot.replay import load_bundled_scenario


class DemoTests(unittest.TestCase):
    def test_early_scenario_stops_at_expected_point(self):
        scenario = load_bundled_scenario("early-stability")
        result = execute_demo(scenario, config_for_scenario(scenario))
        self.assertEqual(result.stop_reason, "stable")
        self.assertEqual(result.shots, 400)
        self.assertTrue(verify_events(result.audit.events)["valid"])

    def test_late_scenario_reaches_cap(self):
        scenario = load_bundled_scenario("late-stability")
        result = execute_demo(scenario, config_for_scenario(scenario))
        self.assertEqual(result.stop_reason, "max_budget")
        self.assertEqual(result.shots, 1000)

    def test_reference_is_not_written_into_online_context(self):
        scenario = load_bundled_scenario("early-stability")
        result = execute_demo(scenario, config_for_scenario(scenario))
        start = result.audit.events[0]["payload"]
        self.assertNotIn("reference_counts", start.get("context", {}))
        self.assertIn("reference-distribution TVD is not a decision input", start["decision_scope"])

    def test_tampering_is_detected(self):
        scenario = load_bundled_scenario("audit-walkthrough")
        result = execute_demo(scenario, config_for_scenario(scenario))
        tampered = tamper_events(result.audit.events)
        self.assertFalse(verify_events(tampered)["valid"])

    def test_comparison_uses_bundled_fixed_budgets(self):
        scenario = load_bundled_scenario("early-stability")
        rows = compare_policies(scenario, config_for_scenario(scenario))
        self.assertEqual([row["policy"] for row in rows[:-1]], ["Fixed-200", "Fixed-400", "Fixed-800"])
        self.assertEqual(rows[-1]["policy"], "StableShots")

    def test_cumulative_distribution_snapshot_reconstructs_any_iteration(self):
        scenario = load_bundled_scenario("audit-walkthrough")
        result = execute_demo(scenario, config_for_scenario(scenario))
        iterations = distribution_iterations(result.audit)

        self.assertEqual(len(iterations), result.shots // 50)
        self.assertEqual(iterations[0], {"round_index": 1, "total_shots": 50})
        self.assertEqual(iterations[-1]["total_shots"], result.shots)

        first = cumulative_distribution_snapshot(result.audit, 1)
        final = cumulative_distribution_snapshot(
            result.audit,
            iterations[-1]["round_index"],
        )
        self.assertEqual(first.total_shots, 50)
        self.assertEqual(final.counts, result.counts)
        self.assertAlmostEqual(sum(final.frequencies.values()), 1.0)

    def test_distribution_filters_support_top_n_and_frequency_thresholds(self):
        current = {"a": 50, "b": 20, "c": 15, "d": 10, "e": 5}
        lookback = {"a": 40, "b": 20, "c": 10, "d": 5, "e": 25}

        self.assertEqual(
            select_distribution_outcomes(current, mode="top_n", top_n=3),
            ["a", "b", "c"],
        )
        self.assertEqual(
            select_distribution_outcomes(current, mode="top_n", top_n=500),
            ["a", "b", "c", "d", "e"],
        )
        many = {f"o{i:03d}": 1 for i in range(150)}
        self.assertEqual(
            len(select_distribution_outcomes(many, mode="top_n", top_n=500)),
            100,
        )
        self.assertEqual(
            select_distribution_outcomes(current, mode="freq_1pct"),
            ["a", "b", "c", "d", "e"],
        )
        self.assertEqual(
            select_distribution_outcomes(current, mode="freq_5pct"),
            ["a", "b", "c", "d"],
        )
        self.assertEqual(
            select_distribution_outcomes(current, mode="freq_10pct"),
            ["a", "b", "c"],
        )
        self.assertEqual(
            select_distribution_outcomes(
                current,
                comparison_counts=lookback,
                mode="freq_10pct",
            ),
            ["a", "e", "b", "c"],
        )

    def test_distribution_table_contains_counts_and_frequencies(self):
        scenario = load_bundled_scenario("early-stability")
        result = execute_demo(scenario, config_for_scenario(scenario))
        iterations = distribution_iterations(result.audit)
        final = cumulative_distribution_snapshot(
            result.audit,
            iterations[-1]["round_index"],
        )
        rows = distribution_table_rows(final, ["00", "11"])

        by_outcome = {row["outcome"]: row for row in rows}
        self.assertEqual(by_outcome["00"]["current_count"], 240)
        self.assertAlmostEqual(by_outcome["00"]["current_frequency"], 0.6)
        self.assertEqual(by_outcome["11"]["current_count"], 160)
        self.assertAlmostEqual(by_outcome["11"]["current_frequency"], 0.4)

    def test_self_check_passes(self):
        rows = self_check()
        self.assertTrue(rows)
        self.assertTrue(all(row["status"] == "PASS" for row in rows))


if __name__ == "__main__":
    unittest.main()
