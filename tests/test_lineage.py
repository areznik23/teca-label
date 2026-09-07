"""Tests for lineage and accounting: usage tracking, client
injection, revision-source logging, pending-file safety, units-function model provenance.
Model calls are mocked — no API."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from teca_label.core import Category, Codebook, Op, Revision, _sample
from teca_label.units import excerpts


class FakeLabel:
    def __init__(self, label, evidence=""):
        self.label = label
        self.evidence = evidence


class FakeProvider:
    def parse(self, model, system, content, schema, max_tokens, timeout):
        raise AssertionError("never called: _parse is patched")


class TestSample(unittest.TestCase):
    def test_spreads_across_sequence_and_caps_at_n(self):
        self.assertEqual(_sample(list(range(100)), 4), [0, 25, 50, 75])
        self.assertEqual(_sample([1, 2], 10), [1, 2])  # fewer than n: all of them
        self.assertEqual(len(_sample(list(range(61)), 60)), 60)


class TestUsageAndClient(unittest.TestCase):
    def test_classify_accumulates_usage_across_calls(self):
        cb = Codebook("q", [Category(name="a", definition="d")], version=1)
        with patch("teca_label.core._parse",
                   return_value=(FakeLabel("a"), {"input_tokens": 10, "output_tokens": 2})):
            cb.classify([{"t": "one"}, {"t": "two"}], workers=2)
            cb.classify([{"t": "three"}], workers=1)
        self.assertEqual(cb.usage["calls"], 3)
        self.assertEqual(cb.usage["input_tokens"], 30)
        self.assertEqual(cb.usage["output_tokens"], 6)

    def test_injected_provider_reaches_every_model_call(self):
        marker = FakeProvider()
        cb = Codebook("q", [Category(name="a", definition="d")], version=1, client=marker)
        with patch("teca_label.core._parse",
                   return_value=(FakeLabel("a"), {"input_tokens": 0, "output_tokens": 0})) as m:
            cb.classify([{"t": "one"}], workers=1)
        self.assertIs(m.call_args.kwargs["provider"], marker)

    def test_drill_child_inherits_provider(self):
        marker = FakeProvider()
        cb = Codebook("q", [Category(name="a", definition="d")], version=1, client=marker)
        rev = Revision(ops=[Op(op="add", name="x", definition="d", evidence=[0])])
        with patch("teca_label.core._parse",
                   return_value=(rev, {"input_tokens": 0, "output_tokens": 0})):
            child = cb.drill("a", [{"t": i} for i in range(5)], ["a"] * 5, path=None)
        self.assertIs(child.provider, marker)


class TestRevisionProvenance(unittest.TestCase):
    def test_log_records_pending_vs_direct_source(self):
        with tempfile.TemporaryDirectory() as d:
            cb = Codebook("q", version=0, path=Path(d) / "x.codebook.json")
            direct = Revision(ops=[Op(op="add", name="a", definition="d")])
            cb.apply(direct)
            cb.pending_path.write_text(
                Revision(ops=[Op(op="add", name="b", definition="d")]).model_dump_json())
            cb.apply()  # from pending
            sources = [json.loads(l)["source"] for l in cb.log_path.read_text().splitlines()
                       if json.loads(l)["event"] == "revision"]
            self.assertEqual(sources, ["direct", "pending"])

    def test_direct_apply_preserves_unrelated_pending_proposal(self):
        with tempfile.TemporaryDirectory() as d:
            cb = Codebook("q", [Category(name="a", definition="d")], version=1,
                          path=Path(d) / "x.codebook.json")
            unreviewed = Revision(ops=[Op(op="add", name="reviewme", definition="d")])
            cb.pending_path.write_text(unreviewed.model_dump_json())
            cb.apply(Revision(ops=[Op(op="redefine", name="a", definition="sharper")]))
            self.assertTrue(cb.pending_path.exists())  # someone's review is not discarded
            self.assertEqual(Revision.model_validate_json(cb.pending_path.read_text()), unreviewed)

    def test_direct_apply_of_the_pending_revision_consumes_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            cb = Codebook("q", [Category(name="a", definition="d")], version=1,
                          path=Path(d) / "x.codebook.json")
            rev = Revision(ops=[Op(op="add", name="b", definition="d")])
            cb.pending_path.write_text(rev.model_dump_json(indent=2))
            cb.apply(rev)  # the in-memory propose()->apply() path (build)
            self.assertFalse(cb.pending_path.exists())


class TestUnitsProvenance(unittest.TestCase):
    def test_excerpts_carries_its_model(self):
        unitize = excerpts("q?", model="claude-haiku-4-5")
        self.assertEqual(unitize.model, "claude-haiku-4-5")
        self.assertEqual(unitize.usage["calls"], 0)


if __name__ == "__main__":
    unittest.main()
