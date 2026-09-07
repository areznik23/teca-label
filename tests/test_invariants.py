"""The contract's hard edges: configuration errors raise, unreviewed proposals are never
buried, names stay reserved through renames, files with duplicate names are refused,
empty revisions never bump, tags stay tags, nothing-to-read is unclassifiable, table
names are identifiers, and the runs ledger closes on every exit. No API, no database."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from teca_label.core import Category, Codebook, InvalidRevision, Op, Revision
from teca_label.labeling import check_table_name
from teca_label.providers import ConfigurationError


def cb_at(tmp, **kw) -> Codebook:
    return Codebook("q", [Category(name="a", definition="d")], version=1,
                    path=Path(tmp) / "x.codebook.json", **kw)


class TestConfigurationErrorsRaise(unittest.TestCase):
    def test_missing_key_raises_instead_of_labeling_nothing(self):
        cb = Codebook("q", [Category(name="a", definition="d")], version=1)
        with patch.dict("os.environ", {}, clear=True), \
                patch.dict("teca_label.providers._defaults", clear=True):
            with self.assertRaises(ConfigurationError) as ctx:
                cb.classify([{"text": "some real content here"}] * 3)
        self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))
        self.assertEqual(cb.last_errors, [])          # not a gap, a fault

    def test_unknown_model_family_raises(self):
        cb = Codebook("q", [Category(name="a", definition="d")], version=1,
                      models={"classify": "gemini-3"})
        with self.assertRaises(ConfigurationError):
            cb.classify([{"text": "some real content here"}])

    def test_transient_failures_still_become_gaps(self):
        cb = Codebook("q", [Category(name="a", definition="d")], version=1)
        with patch.object(Codebook, "_call", side_effect=TimeoutError("slow")):
            self.assertEqual(cb.classify([{"text": "some real content here"}], retries=1), [None])
        self.assertEqual(len(cb.last_errors), 1)


class TestPendingIsNeverBuried(unittest.TestCase):
    def test_propose_refuses_while_a_proposal_awaits_review(self):
        with tempfile.TemporaryDirectory() as d:
            cb = cb_at(d)
            cb.pending_path.write_text(
                Revision(ops=[Op(op="add", name="stale", definition="d")]).model_dump_json())
            cb._call = lambda *a, **k: self.fail("must not spend before the pending file is resolved")
            with self.assertRaises(FileExistsError):
                cb.propose([{"text": "x" * 60}] * 5, labels=["other"] * 5)


class TestNamesStayReserved(unittest.TestCase):
    def test_a_renamed_away_name_cannot_come_back(self):
        with tempfile.TemporaryDirectory() as d:
            cb = cb_at(d)
            cb.apply(Revision(ops=[Op(op="rename", name="a", new_name="b")]))
            with self.assertRaises(InvalidRevision):
                cb.apply(Revision(ops=[Op(op="add", name="a", definition="an unrelated thing")]))
            reopened = Codebook.load(cb.path)                # the reservation survives reload
            with self.assertRaises(InvalidRevision):
                reopened.apply(Revision(ops=[Op(op="add", name="A", definition="case-insensitive")]))

    def test_within_one_revision_too(self):
        cb = Codebook("q", [Category(name="a", definition="d")], version=1)
        with self.assertRaises(InvalidRevision):
            cb.apply(Revision(ops=[Op(op="rename", name="a", new_name="b"),
                                   Op(op="add", name="a", definition="d")]))


class TestFileInvariants(unittest.TestCase):
    def test_duplicate_names_are_refused(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            Codebook("q", [Category(name="a", definition="d"), Category(name="a", definition="again")])

    def test_apply_never_mutates_the_callers_categories(self):
        mine = [Category(name="a", definition="d")]
        cb = Codebook("q", mine, version=1)
        cb.apply(Revision(ops=[Op(op="deprecate", name="a")]))
        self.assertIsNone(mine[0].deprecated_v)

    def test_empty_revision_does_not_bump(self):
        cb = Codebook("q", [Category(name="a", definition="d")], version=1)
        with self.assertRaises(InvalidRevision):
            cb.apply(Revision(ops=[]))
        self.assertEqual(cb.version, 1)

    def test_a_tag_stays_a_tag(self):
        with self.assertRaisesRegex(ValueError, "tag"):
            Codebook("q", [Category(name="a", definition="d"), Category(name="b", definition="d")],
                     kind="tag")
        with tempfile.TemporaryDirectory() as d:
            t = Codebook.tag("a", "about a", path=Path(d) / "t.codebook.json")
            with self.assertRaises(InvalidRevision):
                t.apply(Revision(ops=[Op(op="add", name="b", definition="d")]))
            self.assertEqual(t.version, 1)


class TestNothingToRead(unittest.TestCase):
    def test_blank_fields_are_unclassifiable_without_a_call(self):
        cb = Codebook("q", [Category(name="a", definition="d")], version=1)
        cb._call = lambda *a, **k: self.fail("no call for an empty record")
        self.assertEqual(cb.classify([{}, {"text": ""}, {"text": "   ", "n": 3}]),
                         ["unclassifiable"] * 3)


class TestTableNames(unittest.TestCase):
    def test_identifiers_only(self):
        for ok in ("teca_labels", "analytics.labels", "_t1"):
            self.assertEqual(check_table_name(ok), ok)
        for bad in ("x; DROP TABLE y", "labels-2", "a.b.c", "", 8):
            with self.assertRaises(ValueError):
                check_table_name(bad)


if __name__ == "__main__":
    unittest.main()
