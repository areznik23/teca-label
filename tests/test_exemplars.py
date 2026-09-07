"""Tests for exemplars(): measured typicality via test-retest agreement. No API."""
import unittest

from teca_label.core import Category, Codebook


class RetestCodebook(Codebook):
    """classify() answers from each trace's scripted 'retest' field — simulating
    rows that hold their label vs rows that flip on re-ask."""
    def classify(self, traces, workers=8, model=None, retries=1):
        return [t.get("retest") for t in traces]


def make_cb():
    return RetestCodebook("why do sessions fail?", [
        Category(name="tool_loop", definition="repeats the same call"),
        Category(name="missing_context", definition="lacked available info")], version=1)


TRACES = [{"id": k, "retest": r} for k, r in [
    ("a", "tool_loop"), ("b", "tool_loop"), ("c", "missing_context"),  # c flips on re-ask
    ("d", "tool_loop"), ("e", None),                                   # e's re-ask fails
    ("f", "missing_context")]]
LABELS = ["tool_loop", "tool_loop", "tool_loop", "tool_loop", "tool_loop", "missing_context"]


class TestExemplars(unittest.TestCase):
    def test_only_rows_that_hold_their_label_are_exemplars(self):
        r = make_cb().exemplars("tool_loop", TRACES, LABELS, n=5)
        ids = [e["trace"]["id"] for e in r["exemplars"]]
        self.assertEqual(ids, ["a", "b", "d"])   # c flipped, e failed — excluded
        self.assertEqual(r["n_labeled"], 5)

    def test_agreement_excludes_failed_calls_and_counts_flips(self):
        r = make_cb().exemplars("tool_loop", TRACES, LABELS, n=5)
        self.assertEqual(r["n_tested"], 4)                 # e's None call dropped
        self.assertEqual(r["agreement"], 0.75)             # 3 of 4 held
    def test_n_caps_the_result_but_never_pads_with_borderline_rows(self):
        r = make_cb().exemplars("tool_loop", TRACES, LABELS, n=2)
        self.assertEqual(len(r["exemplars"]), 2)
        many_flips = [{"id": "x", "retest": "missing_context"}] * 4
        r2 = make_cb().exemplars("tool_loop", many_flips, ["tool_loop"] * 4, n=3)
        self.assertEqual(r2["exemplars"], [])              # nothing stable -> nothing shown
        self.assertEqual(r2["agreement"], 0.0)

    def test_unknown_or_unused_category_raises(self):
        with self.assertRaises(ValueError):
            make_cb().exemplars("ghost", TRACES, LABELS)

    def test_retest_caps_the_spend(self):
        cb = make_cb()
        seen = []
        original = cb.classify
        cb.classify = lambda ts, **kw: (seen.append(len(ts)), original(ts))[1]
        cb.exemplars("tool_loop", TRACES, LABELS, retest=2)
        self.assertLessEqual(seen[0], 3)   # strided sample never exceeds ~retest rows


if __name__ == "__main__":
    unittest.main()
