"""Tests for child codebooks: drill, the parent record, parent-revision
projection (op -> consequence routing), and reparent(). Model calls are stubbed — no API."""
import json
import tempfile
import unittest
from pathlib import Path

from teca_label.core import Category, Codebook, Op, Revision


class StubbedCodebook(Codebook):
    """propose() invents one child category per distinct 'kind' field in the traces it sees."""
    def propose(self, traces, labels=None, min_evidence=3):
        kinds = sorted({t["kind"] for t in traces})
        return Revision(ops=[Op(op="add", name=k, definition=f"kind {k}") for k in kinds])

    def classify(self, traces, workers=8, model=None, retries=1):
        return [t["kind"] for t in traces]


def make_parent(tmp) -> StubbedCodebook:
    cb = StubbedCodebook("what are users doing?", [
        Category(name="pricing", definition="about money"),
        Category(name="bugs", definition="about defects")], version=1,
        path=Path(tmp) / "support.codebook.json")
    cb.save()
    return cb


TRACES = [{"kind": "sticker_shock"}, {"kind": "sticker_shock"}, {"kind": "sticker_shock"},
          {"kind": "seat_math"}, {"kind": "seat_math"}, {"kind": "seat_math"},
          {"kind": "crash"}]
LABELS = ["pricing", "pricing", "pricing", "pricing", "pricing", "pricing", "bugs"]


def make_child(tmp):
    parent = make_parent(tmp)
    return parent, parent.drill("pricing", TRACES, LABELS)


class TestDrill(unittest.TestCase):
    def test_drill_narrows_to_the_category_and_records_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, child = make_child(tmp)
            self.assertEqual({c.name for c in child.active()}, {"sticker_shock", "seat_math"})
            self.assertNotIn("crash", {c.name for c in child.categories})  # bugs rows never seen
            ref = child.parent
            self.assertEqual(ref["categories"], ["pricing"])
            self.assertEqual(ref["version"], 1)
            self.assertIn("pinned_at", child.parent)
            self.assertEqual(child.path.name, "support.pricing.codebook.json")
            self.assertIn("about money", child.question)  # auto-question carries the definition

    def test_parent_survives_save_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, child = make_child(tmp)
            loaded = Codebook.load(child.path)
            self.assertEqual(loaded.parent, child.parent)
            self.assertEqual(loaded.models, child.models)  # inherited config persisted

    def test_drill_rejects_unknown_or_thin_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = make_parent(tmp)
            with self.assertRaises(ValueError):
                parent.drill("ghost", TRACES, LABELS)
            with self.assertRaises(ValueError):
                parent.drill("bugs", TRACES, LABELS)  # only 1 row — under min_evidence


class TestParentProjection(unittest.TestCase):
    """Each parent op type routes to its consequence for the child."""

    def project(self, tmp, ops):
        parent, child = make_child(tmp)
        parent.apply(Revision(ops=ops))
        report = child.staleness()
        self.assertTrue(report["parent_stale"])
        return child, report["parent"]["ops"]

    def test_untouched_child_is_not_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, child = make_child(tmp)
            self.assertFalse(child.staleness()["parent_stale"])

    def test_op_on_an_unfollowed_category_moves_version_but_projects_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            child, ops = self.project(tmp, [Op(op="redefine", name="bugs", definition="defects")])
            self.assertEqual(ops, [])  # moved, but nothing touched MY slice

    def test_rename_routes_to_auto_follow(self):
        with tempfile.TemporaryDirectory() as tmp:
            child, ops = self.project(tmp, [Op(op="rename", name="pricing",
                                               new_name="pricing_objections")])
            self.assertEqual(ops[0]["action"], "follow")
            self.assertTrue(ops[0]["auto"])
            self.assertEqual(ops[0]["to"], "pricing_objections")
            self.assertIn("renames only", child.staleness()["note"])

    def test_redefine_routes_to_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            child, ops = self.project(tmp, [Op(op="redefine", name="pricing",
                                               definition="money AND billing")])
            self.assertEqual(ops[0]["action"], "review")
            self.assertIn("decision", child.staleness()["note"])

    def test_split_routes_to_choose_with_the_menu(self):
        with tempfile.TemporaryDirectory() as tmp:
            child, ops = self.project(tmp, [Op(op="split", name="pricing", into=[
                Category(name="pricing_value", definition="worth it?"),
                Category(name="pricing_process", definition="billing mechanics")])])
            self.assertEqual(ops[0]["action"], "choose")
            self.assertEqual(ops[0]["options"], ["pricing_value", "pricing_process"])

    def test_merge_routes_to_review_with_the_widened_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            child, ops = self.project(tmp, [Op(op="merge", names=["pricing", "bugs"],
                                               new_name="complaints", definition="any gripe")])
            self.assertEqual(ops[0]["action"], "review")
            self.assertEqual(ops[0]["widened_to"], "complaints")

    def test_deprecate_routes_to_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            child, ops = self.project(tmp, [Op(op="deprecate", name="pricing")])
            self.assertEqual(ops[0]["action"], "archive")


class TestReparent(unittest.TestCase):
    def test_rename_absorbs_automatically_and_repins(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, child = make_child(tmp)
            parent.apply(Revision(ops=[Op(op="rename", name="pricing",
                                          new_name="pricing_objections")]))
            child.reparent()
            ref = child.parent
            self.assertEqual(ref["categories"], ["pricing_objections"])
            self.assertEqual(ref["version"], 2)
            self.assertFalse(child.staleness()["parent_stale"])  # fully absorbed

    def test_refuses_to_repin_past_an_undecided_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, child = make_child(tmp)
            parent.apply(Revision(ops=[Op(op="split", name="pricing", into=[
                Category(name="pricing_value", definition="a"),
                Category(name="pricing_process", definition="b")])]))
            with self.assertRaises(ValueError):
                child.reparent()
            self.assertEqual(child.parent["version"], 1)  # still pinned, still flagged
            self.assertTrue(child.staleness()["parent_stale"])

    def test_split_decision_follows_the_chosen_half(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, child = make_child(tmp)
            parent.apply(Revision(ops=[Op(op="split", name="pricing", into=[
                Category(name="pricing_value", definition="a"),
                Category(name="pricing_process", definition="b")])]))
            child.reparent(decisions={"pricing": ["pricing_value"]})
            self.assertEqual(child.parent["categories"], ["pricing_value"])
            self.assertFalse(child.staleness()["parent_stale"])

    def test_redefine_decision_reaffirms_the_same_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, child = make_child(tmp)
            parent.apply(Revision(ops=[Op(op="redefine", name="pricing",
                                          definition="money AND billing")]))
            child.reparent(decisions={"pricing": ["pricing"]})
            self.assertEqual(child.parent["categories"], ["pricing"])
            self.assertEqual(child.parent["version"], 2)

    def test_deprecate_decision_can_retire_the_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, child = make_child(tmp)
            parent.apply(Revision(ops=[Op(op="deprecate", name="pricing")]))
            child.reparent(decisions={"pricing": []})
            self.assertEqual(child.parent["categories"], [])  # feed ended; history intact

    def test_rename_then_split_keys_the_decision_by_the_new_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, child = make_child(tmp)
            parent.apply(Revision(ops=[Op(op="rename", name="pricing", new_name="px")]))
            parent.apply(Revision(ops=[Op(op="split", name="px", into=[
                Category(name="px_value", definition="a"), Category(name="px_process", definition="b")])]))
            menu = child.staleness()["parent"]["ops"]
            self.assertEqual([o["action"] for o in menu], ["follow", "choose"])
            self.assertEqual(menu[1]["name"], "px")  # projection followed the rename
            child.reparent(decisions={"px": ["px_process"]})
            self.assertEqual(child.parent["categories"], ["px_process"])
            self.assertEqual(child.parent["version"], 3)

    def test_reparent_is_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, child = make_child(tmp)
            parent.apply(Revision(ops=[Op(op="rename", name="pricing", new_name="p2")]))
            child.reparent()
            events = [json.loads(l)["event"] for l in child.log_path.read_text().splitlines()]
            self.assertIn("reparent", events)



class TestSmallPopulationGuidance(unittest.TestCase):
    """The band scales with the sample; drill signposts the harvest move."""

    def test_bootstrap_band_scales_with_evidence_rows(self):
        import re
        cb = Codebook("q")
        band = lambda n: re.search(r"ops: (\d+)-(\d+) mutually",
                                   cb._reviser_instructions(3, "", n_rows=n)).groups()
        self.assertEqual(band(34), ("3", "5"))     # a small population stays coarse
        self.assertEqual(band(60), ("6", "10"))    # the classic default, unchanged
        self.assertEqual(band(600), ("6", "10"))   # capped — codebooks don't sprawl

    def test_drill_refusal_names_the_fix(self):
        cb = Codebook("q", [Category(name="a", definition="d")], version=1)
        with self.assertRaisesRegex(ValueError, "classify strided batches"):
            cb.drill("a", [{"t": 1}], ["a"])

    def test_thin_drill_warns_but_proceeds(self):
        import warnings as w
        cb = StubbedCodebook("q", [Category(name="a", definition="d")], version=1)
        traces = [{"kind": "retry"} for _ in range(10)]
        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            child = cb.drill("a", traces, ["a"] * 10)
        self.assertTrue(any("thin for a standing child" in str(x.message) for x in caught))
        self.assertEqual([c.name for c in child.active()], ["retry"])

if __name__ == "__main__":
    unittest.main()
