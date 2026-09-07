"""Tests for the unit layer — pure logic, no API calls (semantic extraction is stubbed)."""
import unittest

from teca_label.units import unitize, identity


class TestUnitFunctions(unittest.TestCase):
    def test_identity_passes_record_through_as_single_unit(self):
        out, failed = unitize([("r1", {"text": "hello"})], identity)
        self.assertEqual(out, [("r1", 0, {"text": "hello"})])
        self.assertEqual(failed, {})

    def test_structural_units_are_any_callable(self):
        split_turns = lambda record: [{"turn": t} for t in record["turns"]]
        out, _ = unitize([("r1", {"turns": ["hi", "bye"]})], split_turns)
        self.assertEqual(out, [("r1", 0, {"turn": "hi"}), ("r1", 1, {"turn": "bye"})])

    def test_empty_extraction_yields_no_units_for_that_record(self):
        stub_semantic = lambda record: [] if record["topic"] == "weather" else [{"quote": "price!"}]
        out, failed = unitize([("dull", {"topic": "weather"}), ("hot", {"topic": "pricing"})],
                                     stub_semantic)
        self.assertEqual(out, [("hot", 0, {"quote": "price!"})])
        self.assertEqual(failed, {})  # empty is a result, not a failure

    def test_unit_indexes_are_per_record(self):
        two_each = lambda record: [{"n": 1}, {"n": 2}]
        out, _ = unitize([("a", {}), ("b", {})], two_each)
        self.assertEqual([(rid, i) for rid, i, _ in out],
                         [("a", 0), ("a", 1), ("b", 0), ("b", 1)])

    def test_persistent_failure_is_reported_not_raised(self):
        attempts = {"n": 0}

        def flaky(record):
            attempts["n"] += 1
            if record.get("bad"):
                raise RuntimeError("truncated JSON")
            return [{"quote": "fine"}]

        out, failed = unitize([("ok", {}), ("boom", {"bad": True})], flaky, retries=1)
        self.assertEqual(out, [("ok", 0, {"quote": "fine"})])
        self.assertEqual(list(failed), ["boom"])
        self.assertEqual(failed["boom"], "RuntimeError: truncated JSON")
        self.assertEqual(attempts["n"], 3)  # ok once + boom twice (retry exhausted)

    def test_transient_failure_recovers_via_retry(self):
        state = {"failed_once": False}

        def flaky_once(record):
            if not state["failed_once"]:
                state["failed_once"] = True
                raise RuntimeError("blip")
            return [{"quote": "recovered"}]

        out, failed = unitize([("r", {})], flaky_once, retries=1)
        self.assertEqual(out, [("r", 0, {"quote": "recovered"})])
        self.assertEqual(failed, {})


if __name__ == "__main__":
    unittest.main()
