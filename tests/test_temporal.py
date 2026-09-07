"""Tests for the temporal machinery: periods, policy round-trip, staleness, draft
windows. Stubbed model — no API calls."""
import tempfile
import unittest
from pathlib import Path

from teca_label.core import Category, Codebook, Op, Revision, _period_of


def t(ts: str, **fields) -> dict:
    return {"ts": ts, **fields}


class TestPeriods(unittest.TestCase):
    def test_week_quarter_and_day_periods(self):
        from datetime import datetime
        dt = datetime(2026, 4, 22)
        self.assertEqual(_period_of(dt, "day"), "2026-04-22")
        self.assertEqual(_period_of(dt, "week"), "2026-W17")
        self.assertEqual(_period_of(dt, "quarter"), "2026-Q2")

    def test_bad_cadence_rejected(self):
        from datetime import datetime
        with self.assertRaises(ValueError):
            _period_of(datetime(2026, 1, 1), "fortnight")


class TestDriftAndConfig(unittest.TestCase):
    def test_policy_survives_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.codebook.json"
            cb = Codebook("q", [Category(name="a", definition="d")], version=1, path=path,
                          policy={"other_threshold": 0.2})
            cb.save()
            loaded = Codebook.load(path)
            self.assertEqual(loaded.policy["other_threshold"], 0.2)
            self.assertEqual(loaded.policy["min_batch"], 20)  # defaults still merged in

class StubbedCodebook(Codebook):
    """A model-free codebook: classify() labels by the trace's own 'true' field if that
    category is active (else 'other'); propose() adds a category per theme with >=3 misfits."""
    api_calls = 0

    def classify(self, traces, workers=8, model=None, retries=1):
        active = {c.name for c in self.active()}
        return [tr["true"] if tr["true"] in active else "other" for tr in traces]

    def propose(self, traces, labels=None, min_evidence=3, signals=None):
        from collections import Counter
        themes = Counter(tr["true"] for tr in traces)
        ops = [Op(op="add", name=name, definition=f"emergent: {name}")
               for name, n in themes.items() if n >= min_evidence
               and name not in {c.name for c in self.categories}]
        return Revision(ops=ops)


class TestStaleness(unittest.TestCase):
    def test_staleness_reports_drift_and_revisions(self):
        with tempfile.TemporaryDirectory() as d:
            cb = StubbedCodebook("q", path=Path(d) / "x.codebook.json")
            cb.apply(Revision(ops=[Op(op="add", name="a", definition="d")]))
            report = cb.staleness(recent_labels=["a"] * 8 + ["other"] * 2)
            self.assertEqual(report["revisions"], 1)
            self.assertIsNotNone(report["last_revision_at"])
            self.assertEqual(report["recent_other_rate"], 0.2)
            self.assertTrue(report["drifting"])


if __name__ == "__main__":
    unittest.main()


class TestDraftWindow(unittest.TestCase):
    def test_window_filters_sample_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            traces = ([t(f"2026-03-{i:02d}", true="old_problem") for i in range(1, 20)]
                      + [t(f"2026-08-{i:02d}", true="current_problem") for i in range(1, 20)])
            cb = StubbedCodebook.plan(traces, "q", path=Path(d) / "x.codebook.json",
                                       window=("2026-07-01", None), time_key="ts",
                                       allow_empty=True).build()
            self.assertEqual({c.name for c in cb.active()}, {"current_problem"})  # stale era excluded
            self.assertEqual(cb.draft_window["start"], "2026-07-01")
            loaded = Codebook.load(cb.path)
            self.assertEqual(loaded.draft_window["n_sampled"], cb.draft_window["n_sampled"])

    def test_window_requires_time_key_and_enough_traces(self):
        with self.assertRaises(ValueError):
            StubbedCodebook.plan([t("2026-08-01", true="a")], "q", path=None,
                                  window=("2026-07-01", None)).build()
        with self.assertRaises(ValueError):
            StubbedCodebook.plan([t("2026-01-01", true="a")] * 5, "q", path=None,
                                  window=("2026-07-01", None), time_key="ts").build()

    def test_no_window_records_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            cb = StubbedCodebook.plan([t(f"2026-08-{i:02d}", true="a") for i in range(1, 9)],
                                       "q", path=Path(d) / "x.codebook.json", allow_empty=True).build()
            self.assertIsNone(cb.draft_window)
            self.assertNotIn("draft_window", cb.path.read_text())


class TestOpenWindowBounds(unittest.TestCase):
    def test_an_open_end_is_null_not_the_string_none(self):
        with tempfile.TemporaryDirectory() as d:
            traces = [t(f"2026-08-{i:02d}", true="a") for i in range(1, 20)]
            cb = StubbedCodebook.plan(traces, "q", path=Path(d) / "x.codebook.json",
                                       window=("2026-07-01", None), time_key="ts",
                                       allow_empty=True).build()
            self.assertIsNone(cb.draft_window["end"])
            self.assertIsNone(Codebook.load(cb.path).draft_window["end"])
