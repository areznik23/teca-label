"""Tests for steps: per-step units for trajectory questions. Deterministic, no API."""
import unittest

from teca_label.units import unitize, steps

TRACE = {"id": "t1", "steps": [
    {"tool": "read_file", "input": "plan.md"},
    {"tool": "search", "input": "pricing"},
    {"tool": "search", "input": "pricing"},
    {"tool": "answer", "input": "..."}]}


class TestSteps(unittest.TestCase):
    def test_one_unit_per_step_with_position(self):
        units = steps()(TRACE)
        self.assertEqual(len(units), 4)
        self.assertEqual([u["step_index"] for u in units], [0, 1, 2, 3])
        self.assertTrue(all(u["n_steps"] == 4 for u in units))
        self.assertEqual(units[1]["tool"], "search")

    def test_context_window_carries_prior_steps(self):
        units = steps(context=2)(TRACE)
        self.assertNotIn("context_before", units[0])          # nothing before step 0
        self.assertEqual(len(units[1]["context_before"]), 1)
        self.assertEqual(len(units[3]["context_before"]), 2)  # capped at the window
        self.assertEqual(units[2]["context_before"][-1]["tool"], "search")  # the loop is visible

    def test_steps_accessor_can_be_a_field_name_or_callable(self):
        by_field = steps(steps="steps")(TRACE)
        by_call = steps(steps=lambda r: r["steps"])(TRACE)
        self.assertEqual([u["tool"] for u in by_field], [u["tool"] for u in by_call])
        self.assertEqual(steps(steps="missing")({"id": "x"}), [])  # no steps -> no units

    def test_non_dict_steps_are_wrapped(self):
        units = steps(steps="events")({"events": ["read", "write"]})
        self.assertEqual(units[0]["step"], "read")
        self.assertEqual(units[0]["step_index"], 0)

    def test_flows_through_unitize_with_provenance(self):
        u = steps()
        self.assertEqual(u.model, "deterministic")   # units-cache provenance: no model read this
        triples, failed = unitize([("t1", TRACE)], u)
        self.assertEqual(failed, {})
        self.assertEqual([(rid, idx) for rid, idx, _ in triples],
                         [("t1", 0), ("t1", 1), ("t1", 2), ("t1", 3)])


if __name__ == "__main__":
    unittest.main()
