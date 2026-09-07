"""Tests for revision discipline (add cap, pending-file review gate) and the audit pass.
Model calls are mocked — no API."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from teca_label.core import Category, Codebook, Op, Revision


def add(name, evidence=()):
    return Op(op="add", name=name, definition=f"def {name}", evidence=list(evidence))


class TestAddCap(unittest.TestCase):
    def test_excess_adds_trimmed_keeping_best_evidenced(self):
        cb = Codebook("q", [Category(name="existing", definition="d")])
        rev = Revision(ops=[add("weak", [1]), add("strong", [1, 2, 3, 4]),
                            add("mid", [1, 2]), add("ok", [1, 2, 3]),
                            Op(op="redefine", name="existing", definition="sharper")])
        out = cb._enforce_policy(rev)
        names = [o.name for o in out.ops if o.op == "add"]
        self.assertEqual(set(names), {"strong", "ok", "mid"})  # cap 3, by evidence
        self.assertEqual(len([o for o in out.ops if o.op == "redefine"]), 1)  # non-adds untouched

    def test_bootstrap_draft_is_exempt_from_cap(self):
        cb = Codebook("q")  # no categories yet
        rev = Revision(ops=[add(f"c{i}") for i in range(9)])
        self.assertEqual(len(cb._enforce_policy(rev).ops), 9)

    def test_cap_is_policy_configurable(self):
        cb = Codebook("q", [Category(name="x", definition="d")], policy={"max_adds_per_revision": 1})
        out = cb._enforce_policy(Revision(ops=[add("a", [1]), add("b", [1, 2])]))
        self.assertEqual([o.name for o in out.ops], ["b"])


class TestPendingReviewGate(unittest.TestCase):
    def run_propose(self, cb, canned: Revision):
        with patch("teca_label.core._parse", return_value=(canned, {"input_tokens": 0, "output_tokens": 0})):
            return cb.propose([{"t": "x"}], labels=["other"])

    def test_propose_writes_pending_and_bare_apply_consumes_it(self):
        with tempfile.TemporaryDirectory() as d:
            cb = Codebook("q", [Category(name="a", definition="d")], version=1,
                          path=Path(d) / "x.codebook.json")
            self.run_propose(cb, Revision(ops=[add("b", [0])]))
            self.assertTrue(cb.pending_path.exists())  # the review artifact
            cb.apply()  # no argument: accept the reviewed pending file
            self.assertEqual({c.name for c in cb.active()}, {"a", "b"})
            self.assertFalse(cb.pending_path.exists())  # consumed on accept

    def test_pending_file_is_editable_before_apply(self):
        with tempfile.TemporaryDirectory() as d:
            cb = Codebook("q", [Category(name="a", definition="d")], version=1,
                          path=Path(d) / "x.codebook.json")
            self.run_propose(cb, Revision(ops=[add("good", [0]), add("junk", [1])]))
            pending = json.loads(cb.pending_path.read_text())  # human rejects one op
            pending["ops"] = [o for o in pending["ops"] if o["name"] != "junk"]
            cb.pending_path.write_text(json.dumps(pending))
            cb.apply()
            self.assertEqual({c.name for c in cb.active()}, {"a", "good"})

    def test_bare_apply_without_pending_raises(self):
        with tempfile.TemporaryDirectory() as d:
            cb = Codebook("q", path=Path(d) / "x.codebook.json")
            with self.assertRaises(FileNotFoundError):
                cb.apply()


class TestAdoptPolicy(unittest.TestCase):
    def test_adopt_carries_policy_into_first_save(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.codebook.json"
            Codebook.adopt("q", [Category(name="a", definition="d")], path,
                           policy={"other_threshold": 1.0})
            saved = json.loads(path.read_text())
            self.assertEqual(saved["policy"]["other_threshold"], 1.0)
            self.assertEqual(Codebook.load(path).policy["other_threshold"], 1.0)

    def test_adopt_rejects_unknown_policy_keys(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                Codebook.adopt("q", [Category(name="a", definition="d")],
                               Path(d) / "x.codebook.json",
                               policy={"other_treshold": 1.0})  # the typo the kwarg exists to catch


class TestAudit(unittest.TestCase):
    def test_audit_samples_per_category_and_flags_hidden_themes(self):
        with tempfile.TemporaryDirectory() as d:
            cb = Codebook("q", [Category(name="fatigue", definition="d"), Category(name="tiny", definition="d")],
                          version=1, path=Path(d) / "x.codebook.json")
            traces = [{"quote": f"q{i}"} for i in range(12)]
            labels = ["fatigue"] * 10 + ["tiny", "other"]

            class Finding:  # what the mocked model returns
                def model_dump(self):
                    return {"coherent": False, "hidden_theme": "competitor mentions", "note": ""}

            with patch("teca_label.core._parse",
                       return_value=(Finding(), {"input_tokens": 0, "output_tokens": 0})) as mocked:
                findings = cb.audit(traces, labels, per_category=5)
            self.assertEqual(mocked.call_count, 1)  # 'tiny' skipped (<5 units), 'other' never audited
            self.assertEqual(findings[0]["category"], "fatigue")
            self.assertEqual(findings[0]["hidden_theme"], "competitor mentions")
            self.assertIn("audit", [json.loads(l)["event"]
                                    for l in cb.log_path.read_text().splitlines()])


if __name__ == "__main__":
    unittest.main()
