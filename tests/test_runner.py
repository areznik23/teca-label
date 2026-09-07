"""The standing loop: project config, the relabel fold, the story from durable
state, proposal rendering, and one whole tick. Model calls stubbed, no database —
sources is patched at the seams the tick composes."""
import tempfile
import unittest
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import teca_label.sources as sources
from teca_label.core import Category, Codebook, Op, Revision
from teca_label.labeling import LabelRun
from teca_label.runner import (load_project, pending_md_path, plan_defaults,
                            relabel_targets, render_proposal, story_from, tick)


def make_project(root: Path, codebook_body: str = "", project_body: str = "") -> Path:
    Codebook("q", [Category(name="a", definition="d"), Category(name="dead", definition="d")],
             version=1, path=root / "x.codebook.json",
             policy={"min_batch": 2, "decay_periods": 2, "audit_every": 0}).save()
    (root / "teca-label.toml").write_text(
        (f"[project]\n{project_body}\n" if project_body else "")
        + "[codebooks.support]\n"
          'path = "x.codebook.json"\n'
          'query = "SELECT id, ts, trace FROM t ORDER BY ts"\n'
        + codebook_body)
    return root


class TestLoadProject(unittest.TestCase):
    def test_defaults_merge_and_name_derives(self):
        with tempfile.TemporaryDirectory() as d:
            specs = load_project(make_project(Path(d)))
            spec = specs["support"]
            self.assertEqual(spec["name"], "support")
            self.assertEqual(spec["cadence"], "week")
            self.assertEqual(spec["labels_table"], "teca_labels")
            self.assertTrue(spec["path"].exists())

    def test_project_table_overrides_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            specs = load_project(make_project(Path(d), project_body='cadence = "month"\n'))
            self.assertEqual(specs["support"]["cadence"], "month")

    def test_unknown_keys_and_missing_pieces_fail_loudly(self):
        with tempfile.TemporaryDirectory() as d:
            root = make_project(Path(d), codebook_body='surprise = 1\n')
            with self.assertRaisesRegex(ValueError, "unknown keys.*surprise"):
                load_project(root)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "teca-label.toml").write_text('[codebooks.x]\nquery = "SELECT 1"\n')
            with self.assertRaisesRegex(ValueError, "path.*required"):
                load_project(d)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                load_project(d)

    def test_plan_choices_live_under_project_and_stay_out_of_the_tick(self):
        body = ('window = "2026-04-20.."\nsample = 150\n'
                '[project.models]\ndraft = "claude-opus-5"\nclassify = "claude-fable-5-1"\n')
        with tempfile.TemporaryDirectory() as d:
            root = make_project(Path(d), project_body=body)
            self.assertEqual(plan_defaults(root), {
                "models": {"draft": "claude-opus-5", "classify": "claude-fable-5-1"},
                "window": "2026-04-20..", "sample": 150})
            spec = load_project(root)["support"]
            self.assertNotIn("window", spec)
            self.assertNotIn("models", spec)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(plan_defaults(d), {})              # no file: no choices yet
            root = make_project(Path(d), project_body='sample = "lots"\n')
            with self.assertRaisesRegex(ValueError, "sample must be int"):
                plan_defaults(root)
            make_project(Path(d), project_body='models = "claude-opus-5"\n')
            with self.assertRaisesRegex(ValueError, "models must be dict"):
                plan_defaults(root)

    def test_missing_codebook_file_fails_before_any_tick(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "teca-label.toml").write_text(
                '[codebooks.x]\npath = "nope.codebook.json"\nquery = "SELECT 1"\n')
            with self.assertRaisesRegex(ValueError, "not found"):
                load_project(d)


class TestRelabelTargets(unittest.TestCase):
    def targets(self, *ops):
        return relabel_targets([{"ops": [op for op in ops]}])

    def test_boundary_changes_move_everything_local_ops_move_their_rows(self):
        for boundary in ({"op": "add", "name": "n"}, {"op": "redefine", "name": "c"},
                         {"op": "split", "name": "c", "into": []},
                         {"op": "merge", "names": ["x", "y"], "new_name": "z", "name": ""}):
            self.assertIsNone(self.targets(boundary), boundary["op"])   # None = every row
        self.assertEqual(self.targets({"op": "rename", "name": "old", "new_name": "new"}), {"old"})
        self.assertEqual(self.targets({"op": "deprecate", "name": "c"}), {"c"})

    def test_events_accumulate_and_no_events_means_nothing_moves(self):
        self.assertEqual(relabel_targets([]), set())
        local = relabel_targets([{"ops": [{"op": "deprecate", "name": "c"}]},
                                 {"ops": [{"op": "rename", "name": "d", "new_name": "e"}]}])
        self.assertEqual(local, {"c", "d"})
        self.assertIsNone(relabel_targets([{"ops": [{"op": "deprecate", "name": "c"}]},
                                           {"ops": [{"op": "add", "name": "n"}]}]))


class TestStoryFrom(unittest.TestCase):
    NOW = datetime(2026, 9, 2, tzinfo=timezone.utc)   # ISO week 2026-W36

    def test_buckets_by_latest_label_sorted_by_period(self):
        times = {"r1": "2026-08-03", "r2": "2026-08-04", "r3": "2026-08-11"}
        latest = {("r1", 0): ("a", 1), ("r2", 0): ("other", 1), ("r3", 0): ("a", 2)}
        story = story_from(times, latest, "week", min_current=1, now=self.NOW)
        self.assertEqual(story, [("2026-W32", Counter({"a": 1, "other": 1})),
                                 ("2026-W33", Counter({"a": 1}))])

    def test_partial_current_period_is_dropped(self):
        times = {"r1": "2026-08-31", "r2": "2026-09-01"}      # both in 2026-W36 = now
        latest = {("r1", 0): ("a", 1), ("r2", 0): ("a", 1)}
        self.assertEqual(story_from(times, latest, "week", min_current=3, now=self.NOW), [])
        kept = story_from(times, latest, "week", min_current=2, now=self.NOW)
        self.assertEqual(kept, [("2026-W36", Counter({"a": 2}))])
        self.assertEqual(story_from(times, latest, "week", min_current=None, now=self.NOW), [])

    def test_rows_without_timestamps_are_skipped(self):
        story = story_from({}, {("r1", 0): ("a", 1)}, "week", now=self.NOW)
        self.assertEqual(story, [])


class TestRenderProposal(unittest.TestCase):
    def test_ops_evidence_and_review_instructions(self):
        cb = Codebook("why?", [Category(name="a", definition="d")], version=3,
                      path=Path("/tmp/support.codebook.json"))
        revision = Revision(ops=[
            Op(op="add", name="refund_status", definition="asks where a refund is",
               evidence=[1]),
            Op(op="deprecate", name="a")])
        md = render_proposal("support", cb, revision,
                             [{"kind": "decay", "category": "a", "period": "2026-W35"}],
                             [{"text": "t0"}, {"text": "still waiting on my refund"}],
                             ["other", "other"])
        self.assertIn("(v3 → v4)", md)
        self.assertIn("add `refund_status`", md)
        self.assertIn("deprecate `a`", md)
        self.assertIn("still waiting on my refund", md)
        self.assertIn("**decay**", md)
        self.assertIn("teca-label apply --codebook support", md)


class TestTick(unittest.TestCase):
    def test_signals_produce_a_reviewable_proposal_and_never_an_apply(self):
        with tempfile.TemporaryDirectory() as d:
            root = make_project(Path(d))
            spec = load_project(root)["support"]
            # two complete weeks where 'dead' never appears -> decay fires (periods=2)
            rows = {f"r{i}": (f"2026-08-{3 + 7 * (i % 2):02d}", {"text": f"t{i}"})
                    for i in range(6)}
            latest = {(row_id, 0): ("a", 1) for row_id in rows}
            canned = Revision(ops=[Op(op="deprecate", name="dead")])
            with patch.dict("os.environ", {"TECA_LABEL_DSN": "postgres://fake"}), \
                 patch("teca_label.runner._fetch", return_value=rows), \
                 patch.object(sources, "run_start", return_value="rid1") as start_call, \
                 patch.object(sources, "last_proposal", return_value=None), \
                 patch.object(sources, "run_finish") as finish_call, \
                 patch.object(sources, "latest_labels", return_value=latest), \
                 patch.object(sources, "label_postgres",
                              return_value=LabelRun(counts=Counter({"a": 6}))) as label_call, \
                 patch.object(Codebook, "propose", return_value=canned) as propose_call, \
                 patch.object(Codebook, "apply") as apply_call:
                summary = tick("support", spec)
            self.assertEqual(label_call.call_args.kwargs["on_version_change"], "relabel_touched")
            # the tick is bracketed by its ledger row: opened first, closed 'ok'
            # exactly once, and the labels written carry the run's id
            start_call.assert_called_once()
            self.assertEqual(label_call.call_args.kwargs["run_id"], "rid1")
            self.assertEqual(finish_call.call_args.args[1:3], ("rid1", "ok"))
            self.assertEqual(finish_call.call_args.kwargs["rows_labeled"], 6)
            self.assertEqual(summary["run_id"], "rid1")
            self.assertEqual(summary["labeled"], {"a": 6})
            self.assertEqual(summary["failed"], [])
            self.assertTrue(summary["proposed"])
            self.assertEqual([s["kind"] for s in summary["signals"]], ["decay"])
            self.assertEqual([s["category"] for s in summary["signals"]], ["dead"])
            apply_call.assert_not_called()                     # review is the whole point
            md = pending_md_path(Codebook.load(spec["path"])).read_text()
            self.assertIn("deprecate `dead`", md)
            signals_passed = propose_call.call_args.kwargs["signals"]
            self.assertEqual(signals_passed, summary["signals"])

    def test_quiet_tick_labels_and_proposes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            root = make_project(Path(d))
            spec = load_project(root)["support"]
            rows = {f"r{i}": (f"2026-08-{3 + 7 * (i % 2):02d}", {"text": f"t{i}"})
                    for i in range(6)}
            # 'dead' alive in both weeks -> no decay; nothing else can fire
            latest = {(f"r{i}", 0): ("dead" if i < 4 else "a", 1) for i in range(6)}
            with patch.dict("os.environ", {"TECA_LABEL_DSN": "postgres://fake"}), \
                 patch("teca_label.runner._fetch", return_value=rows), \
                 patch.object(sources, "run_start", return_value="rid1"), \
                 patch.object(sources, "last_proposal", return_value=None), \
                 patch.object(sources, "run_finish"), \
                 patch.object(sources, "latest_labels", return_value=latest), \
                 patch.object(sources, "label_postgres", return_value=LabelRun()), \
                 patch.object(Codebook, "propose") as propose_call:
                summary = tick("support", spec)
            self.assertFalse(summary["proposed"])
            self.assertEqual(summary["signals"], [])
            propose_call.assert_not_called()

    def test_a_crashing_tick_closes_its_ledger_row_as_failed(self):
        with tempfile.TemporaryDirectory() as d:
            root = make_project(Path(d))
            spec = load_project(root)["support"]
            with patch.dict("os.environ", {"TECA_LABEL_DSN": "postgres://fake"}), \
                 patch.object(sources, "run_start", return_value="rid1"), \
                 patch.object(sources, "last_proposal", return_value=None), \
                 patch.object(sources, "run_finish") as finish_call, \
                 patch("teca_label.runner._fetch", side_effect=RuntimeError("db down")):
                with self.assertRaises(RuntimeError):
                    tick("support", spec)
            self.assertEqual(finish_call.call_args.args[1:3], ("rid1", "failed"))
            self.assertIn("db down", finish_call.call_args.kwargs["error"])

    def test_the_project_policy_reaches_the_labeler(self):
        with tempfile.TemporaryDirectory() as d:
            root = make_project(Path(d), codebook_body='on_version_change = "keep"\n')
            spec = load_project(root)["support"]
            rows = {"r0": ("2026-08-03", {"text": "t"})}
            with patch.dict("os.environ", {"TECA_LABEL_DSN": "postgres://fake"}), \
                 patch("teca_label.runner._fetch", return_value=rows), \
                 patch.object(sources, "run_start", return_value="rid1"), \
                 patch.object(sources, "last_proposal", return_value=None), \
                 patch.object(sources, "run_finish"), \
                 patch.object(sources, "latest_labels", return_value={}), \
                 patch.object(sources, "label_postgres", return_value=LabelRun()) as label_call:
                tick("support", spec)
            self.assertEqual(label_call.call_args.kwargs["on_version_change"], "keep")


class TestLatestLabelsFirstTick(unittest.TestCase):
    def test_missing_labels_table_reads_as_no_labels_yet(self):
        # found live: the first tick reads latest_labels before any label has
        # created the table — that is a normal state, not an error
        class NoTableCursor:
            def execute(self, sql, params=None):
                raise sources.psycopg2.errors.UndefinedTable("no such table")

        conn = type("Conn", (), {"cursor": lambda self_: NoTableCursor(),
                                 "close": lambda self_: None, "autocommit": False})()
        with patch.object(sources.psycopg2, "connect", return_value=conn):
            self.assertEqual(sources.latest_labels("dsn", "support"), {})

if __name__ == "__main__":
    unittest.main()


class TestLedgerClosesOnEveryPath(unittest.TestCase):
    def test_a_failure_after_labeling_still_closes_the_row(self):
        from teca_label.labeling import LabelRun
        with tempfile.TemporaryDirectory() as d:
            root = make_project(Path(d))
            spec = load_project(root)["support"]
            rows = {"r0": ("2026-08-03", {"text": "t"})}
            with patch.dict("os.environ", {"TECA_LABEL_DSN": "postgres://fake"}), \
                 patch("teca_label.runner._fetch", return_value=rows), \
                 patch.object(sources, "run_start", return_value="rid1"), \
                 patch.object(sources, "last_proposal", return_value=None), \
                 patch.object(sources, "run_finish") as finish_call, \
                 patch.object(sources, "label_postgres", return_value=LabelRun()), \
                 patch.object(sources, "latest_labels", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    tick("support", spec)
            self.assertEqual(finish_call.call_args.args[1:3], ("rid1", "failed"))
            self.assertIn("KeyboardInterrupt", finish_call.call_args.kwargs["error"])

    def test_a_pending_proposal_is_reported_and_never_overwritten(self):
        from teca_label.labeling import LabelRun
        with tempfile.TemporaryDirectory() as d:
            root = make_project(Path(d))
            spec = load_project(root)["support"]
            Path(spec["path"]).with_suffix(".pending.json").write_text('{"ops": []}')
            rows = {f"r{i}": (f"2026-08-{3 + 7 * (i % 2):02d}", {"text": f"t{i}"}) for i in range(6)}
            latest = {(f"r{i}", 0): ("other", 1) for i in range(6)}      # other at 100%: emergence
            with patch.dict("os.environ", {"TECA_LABEL_DSN": "postgres://fake"}), \
                 patch("teca_label.runner._fetch", return_value=rows), \
                 patch.object(sources, "run_start", return_value="rid1"), \
                 patch.object(sources, "last_proposal", return_value=None), \
                 patch.object(sources, "run_finish"), \
                 patch.object(sources, "latest_labels", return_value=latest), \
                 patch.object(sources, "label_postgres", return_value=LabelRun()), \
                 patch.object(Codebook, "propose") as propose_call:
                summary = tick("support", spec)
            self.assertTrue(summary["proposed"])
            propose_call.assert_not_called()

    def test_a_period_legislates_once(self):
        from teca_label.labeling import LabelRun
        with tempfile.TemporaryDirectory() as d:
            root = make_project(Path(d))
            spec = load_project(root)["support"]
            cb = Codebook.load(spec["path"])
            cb.apply(Revision(ops=[Op(op="add", name="fresh", definition="d")], periods=["2026-W33"]))
            rows = {f"r{i}": (f"2026-08-{3 + 7 * (i % 2):02d}", {"text": f"t{i}"}) for i in range(40)}
            latest = {(f"r{i}", 0): ("other" if i % 2 else "a", 2) for i in range(40)}   # W33 (the latest period) is half other
            with patch.dict("os.environ", {"TECA_LABEL_DSN": "postgres://fake"}), \
                 patch("teca_label.runner._fetch", return_value=rows), \
                 patch.object(sources, "run_start", return_value="rid1"), \
                 patch.object(sources, "last_proposal", return_value=None), \
                 patch.object(sources, "run_finish"), \
                 patch.object(sources, "latest_labels", return_value=latest), \
                 patch.object(sources, "label_postgres", return_value=LabelRun()), \
                 patch.object(Codebook, "propose") as propose_call:
                summary = tick("support", spec)
            self.assertTrue(summary["signals"])          # the alarm still rings
            self.assertFalse(summary["proposed"])         # but W33 already legislated
            propose_call.assert_not_called()

    def test_an_open_proposal_in_the_ledger_blocks_a_second_draft(self):
        from teca_label.labeling import LabelRun
        with tempfile.TemporaryDirectory() as d:
            root = make_project(Path(d))
            spec = load_project(root)["support"]
            rows = {f"r{i}": (f"2026-08-{3 + 7 * (i % 2):02d}", {"text": f"t{i}"}) for i in range(40)}
            latest = {(f"r{i}", 0): ("other" if i % 2 else "a", 1) for i in range(40)}   # W33 half other
            with patch.dict("os.environ", {"TECA_LABEL_DSN": "postgres://fake"}), \
                 patch("teca_label.runner._fetch", return_value=rows), \
                 patch.object(sources, "run_start", return_value="rid1"), \
                 patch.object(sources, "last_proposal", return_value=("2026-W33", 1)), \
                 patch.object(sources, "run_finish") as finish_call, \
                 patch.object(sources, "latest_labels", return_value=latest), \
                 patch.object(sources, "label_postgres", return_value=LabelRun()), \
                 patch.object(Codebook, "propose") as propose_call:
                summary = tick("support", spec)
            self.assertTrue(summary["signals"])
            self.assertTrue(summary["proposed"])          # reported as open, from the ledger alone
            propose_call.assert_not_called()               # no pending file needed, no re-spend
            self.assertIsNone(finish_call.call_args.kwargs["proposed_for"])
