"""Codebook.plan(): what a build will do, printed before any call is spent. No API."""
import unittest
from unittest.mock import patch

from teca_label import Codebook, Plan
from teca_label.plan import PRICES, Estimate, parse_window, price, tokens


def traces(n=40, ts=None, text="the user asked about billing and the agent looped on search "):
    return [{"id": str(i), "text": text * 3, "tools": ["search"] if i % 2 else [],
             **({"ts": ts(i)} if ts else {})} for i in range(n)]


class TestPlanIsFree(unittest.TestCase):
    def test_no_model_call_until_build(self):
        with patch.object(Codebook, "_call", side_effect=AssertionError("called")):
            plan = Codebook.plan(traces(), "q", path=None)
            str(plan)
        self.assertIsInstance(plan, Plan)
        self.assertEqual(plan.n_seen, 40)
        self.assertEqual(len(plan.traces), 40)

    def test_build_is_the_only_way(self):
        self.assertFalse(hasattr(Codebook, "build"))


class TestWhatItSees(unittest.TestCase):
    def test_fields_report_fill_and_size_text_first(self):
        plan = Codebook.plan(traces(), "q", path=None)
        names = [name for name, _, _ in plan.fields]
        self.assertEqual(names[0], "text")
        by_name = {name: (share, chars) for name, share, chars in plan.fields}
        self.assertEqual(by_name["text"][0], 1.0)
        self.assertGreater(by_name["text"][1], 100)
        self.assertEqual(by_name["tools"][0], 0.5)          # half the rows have an empty list

    def test_thin_traces_show_in_the_plan_and_refuse_the_build(self):
        rows = traces(30) + [{"id": "t", "text": ""}] * 5
        plan = Codebook.plan(rows, "q", path=None)
        self.assertIn("5 of 35", plan.thin)
        self.assertIn("build() refuses them", str(plan))
        with self.assertRaisesRegex(ValueError, "allow_empty=True"):
            plan.build()
        relaxed = Codebook.plan(rows, "q", path=None, allow_empty=True)
        self.assertIn("included (allow_empty=True)", str(relaxed))

    def test_too_few_traces_refuse_the_build_not_the_plan(self):
        plan = Codebook.plan(traces(2), "q", path=None)
        with self.assertRaisesRegex(ValueError, "not enough"):
            plan.build()

    def test_path_names_the_labels(self):
        plan = Codebook.plan(traces(), "q", path="codebooks/failure-modes.codebook.json")
        self.assertIn("labels named 'failure-modes'", str(plan))
        self.assertIn("in memory (path=None)", str(Codebook.plan(traces(), "q", path=None)))


class TestWindow(unittest.TestCase):
    def dated(self):
        return traces(30, ts=lambda i: f"2026-05-{1 + i:02d}")

    def test_string_window_reads_ts_by_default(self):
        plan = Codebook.plan(self.dated(), "q", path=None, window="2026-05-21..")
        self.assertEqual(plan.window, ("2026-05-21", None))
        self.assertEqual(plan.time_key, "ts")
        self.assertEqual(len(plan.traces), 10)
        self.assertEqual(plan.n_seen, 30)
        self.assertIn("30 seen · 10 in window 2026-05-21.. (ts)", str(plan))

    def test_closed_window_and_tuple_form(self):
        plan = Codebook.plan(self.dated(), "q", path=None, window=("2026-05-05", "2026-05-10"))
        self.assertEqual(len(plan.traces), 6)
        self.assertIn("in window 2026-05-05..2026-05-10", str(plan))

    def test_window_without_ts_needs_a_time_key(self):
        with self.assertRaisesRegex(ValueError, "time_key"):
            Codebook.plan(traces(), "q", path=None, window="2026-05-01..")
        plan = Codebook.plan(traces(30, ts=lambda i: f"2026-05-{1 + i:02d}"), "q", path=None,
                             window="2026-05-25..", time_key=lambda t: t["ts"])
        self.assertEqual(len(plan.traces), 6)
        self.assertIn("(time_key)", str(plan))

    def test_window_too_narrow_is_refused_at_plan_time(self):
        with self.assertRaisesRegex(ValueError, "widen it"):
            Codebook.plan(self.dated(), "q", path=None, window="2026-05-29..")

    def test_malformed_window(self):
        with self.assertRaisesRegex(ValueError, "start..end"):
            parse_window("2026-05-01")
        self.assertEqual(parse_window("..2026-06-01"), (None, "2026-06-01"))
        self.assertIsNone(parse_window(None))


class TestCost(unittest.TestCase):
    def test_price_matches_the_longest_prefix(self):
        self.assertEqual(price("claude-haiku-4-5-20251001"), PRICES["claude-haiku-4"])
        self.assertEqual(price("gpt-5-mini-2026-01-01"), PRICES["gpt-5-mini"])
        self.assertEqual(price("gpt-5.4-mini"), PRICES["gpt-5.4-mini"])
        self.assertIsNone(price("llama-9"))
        self.assertEqual(price("llama-9", {"llama-9": (0.1, 0.2)}), (0.1, 0.2))

    def test_tokens_fold_in_the_denser_tokenizer(self):
        self.assertEqual(tokens(4000, "claude-haiku-4-5"), 1000)
        self.assertEqual(tokens(4000, "claude-fable-5-1"), 1300)

    def test_estimate_arithmetic_with_a_known_price(self):
        plan = Codebook.plan(traces(), "q", path=None, sample=10,
                             models={"draft": "claude-haiku-4-5", "classify": "claude-haiku-4-5"})
        draft, sample, full = (plan.estimate[k] for k in ("draft", "sample_labels", "labeling"))
        self.assertEqual(draft.calls, 1)
        self.assertEqual(sample.calls, 10)
        self.assertEqual(full.calls, 40)                      # every row in scope
        in_rate, out_rate = PRICES["claude-haiku-4"]
        self.assertAlmostEqual(draft.usd, (draft.input_tokens * in_rate
                                           + draft.output_tokens * out_rate) / 1e6)
        self.assertAlmostEqual(full.usd, sample.usd * 4)
        self.assertAlmostEqual(plan.build_usd, draft.usd + sample.usd)
        for line in ("draft", "sample labels", "build", "labeling all"):
            self.assertIn(line, str(plan))

    def test_rows_sizes_the_full_labeling_estimate_only(self):
        plan = Codebook.plan(traces(), "q", path=None, sample=10, rows=48_000)
        self.assertEqual(plan.labeling_rows, 48_000)
        self.assertEqual(plan.estimate["labeling"].calls, 48_000)
        self.assertEqual(plan.estimate["sample_labels"].calls, 10)
        self.assertIn("48,000 rows", str(plan))
        self.assertIn("(rows=)", str(plan))

    def test_unknown_model_prints_price_unknown_not_zero(self):
        plan = Codebook.plan(traces(), "q", path=None, models={"draft": "llama-9", "classify": "llama-9"})
        self.assertIsNone(plan.estimate["draft"].usd)
        self.assertIsNone(plan.build_usd)
        self.assertIn("price unknown", str(plan))
        priced = Codebook.plan(traces(), "q", path=None, models={"draft": "llama-9", "classify": "llama-9"},
                               prices={"llama-9": (1.0, 1.0)})
        self.assertIsNotNone(priced.build_usd)

    def test_label_sample_false_drops_the_sample_stage(self):
        plan = Codebook.plan(traces(), "q", path=None, label_sample=False)
        self.assertEqual(plan.estimate["sample_labels"].calls, 0)
        self.assertEqual(plan.build_usd, plan.estimate["draft"].usd)
        self.assertNotIn("sample labels", str(plan))

    def test_estimate_prints_its_parts(self):
        self.assertEqual(str(Estimate("m", 1, 45_088, 2_000, 0.28)),
                         "≈ $0.28 · 1 call · ~45k tokens in, 2,000 out · m")
        self.assertEqual(str(Estimate("m", 2, 500, 20, None)),
                         "price unknown · 2 calls · ~500 tokens in, 20 out · m")


class TestSampleLine(unittest.TestCase):
    def test_reach_and_stride(self):
        plan = Codebook.plan(traces(600), "q", path=None, sample=150)
        self.assertIn("150 of 600, spread evenly (1 in 4) → catches themes above ~4%", str(plan))
        self.assertEqual(len(plan.sampled), 150)
        small = Codebook.plan(traces(20), "q", path=None, sample=60)
        self.assertIn("20 of 20, spread evenly → catches themes above ~30%", str(small))


if __name__ == "__main__":
    unittest.main()


class TestBuildNeverOverwrites(unittest.TestCase):
    def test_an_existing_codebook_path_is_refused_before_any_call(self):
        import tempfile
        from pathlib import Path
        from teca_label.core import Category, Codebook
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.codebook.json"
            Codebook("q", [Category(name="a", definition="d")], version=1, path=path).save()
            plan = Codebook.plan([{"text": "x" * 60}] * 5, "q", path=path)
            with self.assertRaisesRegex(FileExistsError, "never overwrites"):
                plan.build()

    def test_policy_travels_from_plan_to_codebook(self):
        from teca_label.core import Codebook
        with self.assertRaisesRegex(ValueError, "unknown policy"):
            Codebook.plan([{"text": "x" * 60}] * 5, "q", path=None, policy={"other_treshold": 0.2})
        plan = Codebook.plan([{"text": "x" * 60}] * 5, "q", path=None, policy={"min_batch": 5})
        self.assertEqual(plan.policy, {"min_batch": 5})
