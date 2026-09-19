import unittest

from stableshot.audit import verify_events
from stableshot.demo import (
    compare_policies,
    config_for_scenario,
    execute_demo,
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

    def test_self_check_passes(self):
        rows = self_check()
        self.assertTrue(rows)
        self.assertTrue(all(row["status"] == "PASS" for row in rows))


if __name__ == "__main__":
    unittest.main()
