"""Tests for tags: single-category detectors where 'other' means "no", not drift.
Model calls are stubbed — no API."""
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from teca_label.core import Category, Codebook, Op, Revision, drift_signals

POLICY = {"other_threshold": 0.1, "decay_periods": 3, "decay_min_share": 0.01}


def make_tag(tmp, **kw) -> Codebook:
    return Codebook.tag("refund_request",
                        "User asks for their money back",
                        path=Path(tmp) / "refund-request.codebook.json", **kw)


class TestTagBirth(unittest.TestCase):
    def test_tag_is_a_one_category_codebook_of_kind_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = make_tag(tmp)
            self.assertEqual(t.kind, "tag")
            self.assertEqual([c.name for c in t.active()], ["refund_request"])
            self.assertEqual(t.version, 1)
            self.assertIn("money back", t.question)

    def test_kind_survives_save_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = make_tag(tmp)
            loaded = Codebook.load(t.path)
            self.assertEqual(loaded.kind, "tag")
            self.assertEqual(len(loaded.categories), 1)

    def test_files_without_kind_load_as_partitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = Codebook("q", [Category(name="a", definition="x")], version=1,
                          path=Path(tmp) / "old.codebook.json")
            cb.save()
            self.assertEqual(Codebook.load(cb.path).kind, "partition")

    def test_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            Codebook("q", kind="taxonomy")

    def test_label_contract_is_a_two_value_enum(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, Label = make_tag(tmp)._label_schema()
            values = Label.model_fields["label"].annotation.__args__
            self.assertEqual(set(values), {"refund_request", "other"})


class TestTagAlarms(unittest.TestCase):
    """'other' is the tag's normal negative answer — no alarm may treat it as drift."""

    def test_a_mostly_no_tag_emits_no_emergence(self):
        story = [("P1", Counter({"refund_request": 1, "other": 9}))]
        self.assertEqual(drift_signals(story, ["refund_request"], POLICY, kind="tag"), [])
        # the same story read as a partition is a screaming alarm
        partition = drift_signals(story, ["refund_request"], POLICY)
        self.assertIn("emergence", [s["kind"] for s in partition])

    def test_a_share_jump_is_the_reading_not_absorption(self):
        story = [("P1", Counter({"refund_request": 1, "other": 9})),
                 ("P2", Counter({"refund_request": 5, "other": 5}))]
        self.assertEqual(drift_signals(story, ["refund_request"], POLICY, kind="tag"), [])

    def test_decay_still_fires_for_a_dead_tag(self):
        story = [("P%d" % i, Counter({"other": 10})) for i in range(4)]
        signals = drift_signals(story, ["refund_request"], POLICY, kind="tag")
        self.assertEqual([s["kind"] for s in signals], ["decay"])
        self.assertEqual(signals[0]["category"], "refund_request")


class TestTagRevisions(unittest.TestCase):
    def test_growth_ops_are_stripped_a_tag_never_gains_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = make_tag(tmp)
            rev = t._enforce_policy(Revision(ops=[
                Op(op="add", name="new_category", definition="x"),
                Op(op="merge", names=["refund_request"], new_name="m", definition="x"),
                Op(op="split", name="refund_request", into=[Category(name="s1", definition="x")]),
                Op(op="redefine", name="refund_request", definition="sharper boundary")]))
            self.assertEqual([o.op for o in rev.ops], ["redefine"])

    def test_no_rows_are_misfit_evidence_so_propose_never_calls_the_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = make_tag(tmp)
            t._call = lambda *a, **k: self.fail("a tag must not draft from its 'no' rows")
            rev = t.propose([{"x": 1}] * 10, labels=["other"] * 10)  # 100% "no"
            self.assertEqual(rev.ops, [])

    def test_decay_signal_yields_a_deterministic_deprecate(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = make_tag(tmp)
            t._call = lambda *a, **k: self.fail("decay retirement needs no model")
            rev = t.propose([{"x": 1}] * 5, labels=["other"] * 5,
                            signals=[{"kind": "decay", "category": "refund_request",
                                      "period": "P4", "shares": [0, 0, 0]}])
            self.assertEqual([(o.op, o.name) for o in rev.ops],
                             [("deprecate", "refund_request")])


class DeadTag(Codebook):
    """A tag whose behavior vanished: classify always answers 'no'."""
    def classify(self, traces, workers=8, model=None, retries=1):
        return ["other"] * len(traces)


class TestTagLifecycle(unittest.TestCase):
    def test_decay_retires_a_flatlined_tag_through_the_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = DeadTag.tag("refund_request", "User asks for a refund",
                            path=Path(tmp) / "dead.codebook.json",
                            policy={"min_batch": 1, "decay_periods": 3})
            t._call = lambda *a, **k: self.fail("no model call should happen")
            rows = [{"x": i} for i in range(3)]
            labels = t.classify(rows)
            story = [(f"2026-0{m}", Counter(labels)) for m in (1, 2, 3, 4)]
            signals = drift_signals(story, [c.name for c in t.active()], t.policy, kind=t.kind)
            t.apply(t.propose(rows, labels=labels, signals=signals))
            self.assertEqual(t.active(), [])  # retired by decay, through the same gate
            self.assertEqual(len(t.categories), 1)  # deprecated, never deleted
if __name__ == "__main__":
    unittest.main()
