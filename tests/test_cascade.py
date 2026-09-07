"""Tests for the drift cascade: decays, drift_signals, signal-aware propose.
The key scenario: a new theme absorbing into an old
category with other-rate at zero — invisible to the escape valve, caught by the cascade."""
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from teca_label.core import Category, Codebook, Revision, decays, drift_signals


def C(**kv):
    return Counter(kv)


class TestDecays(unittest.TestCase):
    def test_flatlined_category_is_flagged(self):
        story = [("W1", C(a=10, b=10)), ("W2", C(a=20)), ("W3", C(a=15)), ("W4", C(a=25))]
        out = decays(story, ["a", "b"], periods=3)
        self.assertEqual([f["category"] for f in out], ["b"])
        self.assertEqual(out[0]["shares"], [0, 0, 0])

    def test_needs_full_quiet_stretch(self):
        story = [("W1", C(a=20)), ("W2", C(a=20, b=1)), ("W3", C(a=20))]  # b blipped in W2
        self.assertEqual(decays(story, ["a", "b"], periods=3, min_share=0.01), [])

    def test_short_story_never_flags(self):
        self.assertEqual(decays([("W1", C(a=10))], ["a", "b"], periods=3), [])


class TestDriftSignals(unittest.TestCase):
    POLICY = {"other_threshold": 0.1, "decay_periods": 3, "decay_min_share": 0.01}

    def test_stable_story_is_silent(self):
        story = [("W1", C(a=30, b=30)), ("W2", C(a=31, b=29)), ("W3", C(a=29, b=31))]
        self.assertEqual(drift_signals(story, ["a", "b"], self.POLICY), [])

    def test_emergence_from_other_share(self):
        story = [("W1", C(a=50)), ("W2", C(a=40, other=10))]
        kinds = [s["kind"] for s in drift_signals(story, ["a"], self.POLICY)]
        self.assertEqual(kinds, ["emergence"])

    def test_absorption_fires_with_zero_other(self):
        # the measured absorption failure: 'a' swells, nothing lands in other
        story = [("W1", C(a=30, b=30)), ("W2", C(a=30, b=30)), ("W3", C(a=55, b=30))]
        signals = drift_signals(story, ["a", "b"], self.POLICY)
        self.assertEqual([(s["kind"], s["category"]) for s in signals], [("absorption", "a")])

    def test_settled_jumps_age_out_of_the_baseline(self):
        # a jumped long ago and stabilized; the prior-mean baseline absorbs the new
        # normal, so the alarm stops ringing once the level is old news
        story = [("W1", C(a=10, b=30))] + [(f"W{i}", C(a=40, b=30)) for i in range(2, 6)]
        self.assertEqual(drift_signals(story, ["a", "b"], self.POLICY), [])


class TestSignalAwarePropose(unittest.TestCase):
    def make_cb(self, tmp):
        return Codebook("q", [Category(name="a", definition="d"), Category(name="dead", definition="d")],
                        version=1, path=Path(tmp) / "x.codebook.json")

    def test_decay_signals_become_deterministic_deprecates(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = self.make_cb(tmp)
            rev = cb.propose([{"t": 1}], labels=["a"],
                             signals=[{"kind": "decay", "category": "dead", "period": "W9", "shares": [0, 0, 0]}])
            self.assertEqual([(o.op, o.name) for o in rev.ops], [("deprecate", "dead")])
            self.assertTrue(cb.pending_path.exists())  # reviewable like any proposal
            cb.apply(rev)
            self.assertEqual({c.name for c in cb.active()}, {"a"})

    def test_absorption_widens_evidence_to_flagged_categories_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = self.make_cb(tmp)
            seen = {}

            def fake_call(role, system, content, schema, *a, **kw):
                seen["system"], seen["content"] = system, content
                return Revision(ops=[])

            cb._call = fake_call
            cb.propose([{"t": i} for i in range(6)], labels=["a"] * 6,
                       signals=[{"kind": "absorption", "category": "a", "period": "W3",
                                 "share": 0.6, "prev_share": 0.3}])
            self.assertIn("hiding inside", seen["system"])   # signal context reached the prompt
            self.assertIn('"current_label": "a"', seen["content"])  # a's own rows are the evidence

    def test_no_signals_no_misfits_stays_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb = self.make_cb(tmp)
            self.assertEqual(cb.propose([{"t": 1}], labels=["a"]).ops, [])


class TestPolicyValidation(unittest.TestCase):
    def test_unknown_policy_key_raises(self):
        with self.assertRaises(ValueError) as ctx:
            Codebook("q", policy={"other_treshold": 0.2})  # the classic typo
        self.assertIn("other_treshold", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()




class TestBaselinesStartWhereTheCategoryDoes(unittest.TestCase):
    POLICY = {"other_threshold": 0.1, "decay_periods": 3, "decay_min_share": 0.01}

    def test_a_category_a_revision_just_created_raises_no_absorption(self):
        # 'c' was added last period and relabeling landed rows in it: a jump from a
        # baseline of nothing is the revision's doing, not a hidden theme
        story = [("W1", C(a=30, b=30)), ("W2", C(a=30, b=30)), ("W3", C(a=20, b=30, c=20))]
        kinds = [(s["kind"], s["category"]) for s in drift_signals(story, ["a", "b", "c"], self.POLICY)]
        self.assertNotIn(("absorption", "c"), kinds)

    def test_leading_zero_periods_do_not_dilute_the_baseline(self):
        # 'c' appeared in W3 at 20% and holds at 20%: no alarm, even though W1-W2 were 0
        story = [("W1", C(a=40, b=40)), ("W2", C(a=40, b=40)), ("W3", C(a=32, b=32, c=16)),
                 ("W4", C(a=32, b=32, c=16))]
        self.assertEqual(drift_signals(story, ["a", "b", "c"], self.POLICY), [])
