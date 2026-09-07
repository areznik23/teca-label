"""Model governance: a human-labeled benchmark beside the codebook, measured agreement,
and a model change that is refused below policy and logged with its measurement."""
import json
import tempfile
import unittest
from pathlib import Path

from teca_label.core import Category, Codebook


class Scripted(Codebook):
    """classify() answers from each trace's 'true' field, except under the 'bad' model,
    which answers 'a' for everything."""
    def classify(self, traces, workers=8, model=None, retries=1):
        model = model or self.models["classify"]
        return ["a" if model == "claude-bad" else t["true"] for t in traces]


def make(tmp):
    cb = Scripted("q", [Category(name="a", definition="d"), Category(name="b", definition="d")],
                  version=1, path=Path(tmp) / "x.codebook.json")
    cb.save()
    rows = [{"true": "a"}] * 6 + [{"true": "b"}] * 4
    cb.bench(rows, [r["true"] for r in rows])
    return cb


class TestBenchmark(unittest.TestCase):
    def test_bench_writes_the_file_and_validates_labels(self):
        with tempfile.TemporaryDirectory() as d:
            cb = make(d)
            self.assertEqual(len(cb.bench_path.read_text().splitlines()), 10)
            with self.assertRaisesRegex(ValueError, "active categories"):
                cb.bench([{"true": "a"}], ["zzz"])
            with self.assertRaisesRegex(ValueError, "labels"):
                cb.bench([{"true": "a"}], [])

    def test_measure_reports_agreement_and_disagreements(self):
        with tempfile.TemporaryDirectory() as d:
            cb = make(d)
            self.assertEqual(cb.measure()["agreement"], 1.0)
            bad = cb.measure("claude-bad")
            self.assertEqual((bad["n"], bad["agreement"]), (10, 0.6))
            self.assertEqual(bad["disagreements"], {"b ~ a": 4})

    def test_set_model_refuses_below_policy_and_logs_above(self):
        with tempfile.TemporaryDirectory() as d:
            cb = make(d)
            with self.assertRaisesRegex(ValueError, "60%"):
                cb.set_model("classify", "claude-bad")
            self.assertEqual(cb.models["classify"], "claude-opus-5")   # unchanged
            report = cb.set_model("classify", "claude-fine")
            self.assertEqual(report["agreement"], 1.0)
            self.assertEqual(Codebook.load(cb.path).models["classify"], "claude-fine")
            events = [json.loads(l) for l in cb.log_path.read_text().splitlines()]
            change = [e for e in events if e["event"] == "model_change"][-1]
            self.assertEqual((change["old"], change["new"], change["measurement"]["n"]),
                             ("claude-opus-5", "claude-fine", 10))
            self.assertIn("library", change)
            self.assertTrue(any("claude-fine" in line and "agreement 100%" in line
                                for line in cb.history()))

    def test_no_benchmark_means_no_classify_model_change(self):
        with tempfile.TemporaryDirectory() as d:
            cb = Scripted("q", [Category(name="a", definition="d")], version=1,
                          path=Path(d) / "x.codebook.json")
            cb.save()
            with self.assertRaisesRegex(FileNotFoundError, "bench"):
                cb.set_model("classify", "claude-fine")
            cb.set_model("draft", "claude-fine")                    # other roles need no benchmark
            self.assertEqual(cb.models["draft"], "claude-fine")


if __name__ == "__main__":
    unittest.main()
