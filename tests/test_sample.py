"""The codebook owns its sample: build() keeps the rows it drafted from with their
labels, show() renders and checks them, exemplars() reads them without
arguments. No API — classify() answers from each trace's 'true' field."""
import tempfile
import unittest
from pathlib import Path

from teca_label.core import Category, Codebook, Op, Revision


def trace(true: str, k: int = 0) -> dict:
    return {"true": true, "text": f"row {k}: a long enough line of prose about {true} to pass the thin check"}


class Scripted(Codebook):
    """classify() by the trace's 'true' field when that category is active, else
    'other'; the draft adds one category per theme with >= min_evidence rows."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classify_calls = 0

    def classify(self, traces, workers=8, model=None, retries=1):
        self.classify_calls += 1
        active = {c.name for c in self.active()}
        flip = {"flip": "other"}   # a row scripted to change its answer on re-ask
        return [flip.get(t["true"], t["true"] if t["true"] in active else "other") for t in traces]

    def _call(self, role, system, content, schema, max_tokens=2048, timeout=60.0, model=None):
        import json
        rows = json.loads(content)
        from collections import Counter
        themes = Counter(r["true"] for r in rows)
        return Revision(ops=[Op(op="add", name=name, definition=f"about {name}", evidence=[0, 1, 2])
                             for name, n in themes.items() if n >= 3])


ROWS = [trace("a", k) for k in range(6)] + [trace("b", k) for k in range(4)] + [trace("zzz", 9)] * 2


class TestBuildKeepsItsSample(unittest.TestCase):
    def test_sample_and_labels_persist_beside_the_codebook(self):
        with tempfile.TemporaryDirectory() as d:
            cb = Scripted.plan(ROWS, "q", path=Path(d) / "x.codebook.json", sample=12).build()
            self.assertEqual(len(cb.sample), 12)
            self.assertEqual(cb.sample_labels.count("a"), 6)
            self.assertEqual(cb.sample_labels.count("other"), 2)
            self.assertEqual(cb.sample_version, 1)
            self.assertTrue((Path(d) / "x.codebook.sample.jsonl").exists())
            loaded = Codebook.load(cb.path)
            self.assertEqual(loaded.sample, cb.sample)
            self.assertEqual(loaded.sample_labels, cb.sample_labels)
            self.assertEqual(loaded.sample_version, 1)

    def test_label_sample_false_skips_the_pass(self):
        cb = Scripted.plan(ROWS, "q", path=None, label_sample=False).build()
        self.assertEqual(cb.sample, [])
        with self.assertRaisesRegex(ValueError, "no sample"):
            cb.show()

    def test_adopted_codebooks_take_a_sample_explicitly(self):
        cb = Scripted("q", [Category(name="a", definition="d")], version=1)
        with self.assertRaisesRegex(ValueError, "label_sample"):
            cb.exemplars("a")
        cb.label_sample(ROWS)
        self.assertEqual(len(cb.sample_labels), 12)
        self.assertEqual(cb.exemplars("a", n=2)["n_labeled"], 6)


class TestShow(unittest.TestCase):
    def test_table_then_summary_then_checks(self):
        cb = Scripted.plan(ROWS, "q", path=None, sample=12).build()
        out = cb.show()
        self.assertIn("| category | definition | n (of 12) |", out)
        self.assertIn("| `a` | about a | 6 |", out)
        self.assertIn("| `other` | — | 2 |", out)
        self.assertIn("v1 · 12 rows · other 17% (threshold 10%)", out)
        self.assertIn("over the 10% threshold", out)
        self.assertIn("2 categories on 12 rows (1:6, under 1:15)", out)
        self.assertNotIn("0 of", out)

    def test_zero_row_categories_are_called_out_as_the_wrong_shape(self):
        cb = Scripted.plan(ROWS, "q", path=None, sample=12).build()
        cb.apply(Revision(ops=[Op(op="add", name="ghost", definition="never a row's main thing")]))
        out = cb.show()
        self.assertIn("`ghost`: 0 of 12 rows", out)
        self.assertIn("Codebook.tag", out)

    def test_show_relabels_only_when_the_version_moved(self):
        cb = Scripted.plan(ROWS, "q", path=None, sample=12).build()
        calls = cb.classify_calls
        cb.show()
        self.assertEqual(cb.classify_calls, calls)              # labels are current: no spend
        cb.apply(Revision(ops=[Op(op="rename", name="a", new_name="alpha")]))
        out = cb.show()
        self.assertEqual(cb.classify_calls, calls + 1)          # one pass to catch up
        self.assertEqual(cb.sample_version, 2)
        self.assertIn("| `alpha` | about a | 0 |", out)         # scripted classify knows only 'a'

    def test_tags_skip_the_partition_checks(self):
        cb = Scripted.tag("a", "about a", path=None)
        cb.label_sample(ROWS)
        out = cb.show()
        self.assertIn("v1 · 12 rows · other 50%", out)
        self.assertNotIn("⚠", out)


class TestSampleBackedIntrospection(unittest.TestCase):
    def test_exemplars_read_the_sample_by_default(self):
        cb = Scripted.plan(ROWS + [trace("flip")] * 3, "q", path=None, sample=15).build()
        report = cb.exemplars("a", n=3)
        self.assertEqual(report["n_labeled"], 6)
        self.assertEqual(len(report["exemplars"]), 3)
        with self.assertRaisesRegex(ValueError, "pass labels"):
            cb.exemplars("a", ROWS)

if __name__ == "__main__":
    unittest.main()
