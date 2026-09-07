"""label_postgres against an in-memory stand-in for the two tables: a failed call
writes nothing and is pending on the next run; on_version_change is a stated policy;
the LabelRun names what happened. Model calls scripted, no database."""
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import teca_label.sources as sources
from teca_label.core import Category, Codebook, Op, Revision
from teca_label.labeling import LabelRun, eligibility


class Store:
    """Just enough of Postgres for _label: the query's rows, a labels table and a units
    table as lists of tuples with their primary keys honored, and the four reads
    _label makes, dispatched on the SQL."""
    def __init__(self, rows):
        self.rows = rows
        self.labels = []   # (row_id, unit_index, category, evidence, codebook, version, model)
        self.units = []    # (row_id, codebook, unit_index, content, model)
        self.result = []
        self.queries = 0

    def execute(self, sql, params=None):
        self.result = []
        if sql.startswith(("CREATE", "ALTER")):
            return
        if "information_schema" in sql:
            self.result = [(1,)]          # every column present: no ALTER
            return
        if "DISTINCT ON" in sql:
            latest = {}
            for row_id, index, category, _, _, version, *_ in sorted(self.labels, key=lambda r: r[5]):
                latest[(row_id, index)] = (row_id, index, category, version)
            self.result = list(latest.values())
        elif "DISTINCT row_id" in sql:
            self.result = [(u[0],) for u in self.units]
        elif "unit_index >= 0" in sql:
            self.result = [(u[0], u[2], u[3]) for u in self.units if u[2] >= 0]
        else:
            self.queries += 1
            self.result = self.rows

    def fetchall(self):
        return self.result

    def fetchone(self):
        return self.result[0] if self.result else None

    def insert(self, cur, sql, values):
        table, key = (self.units, (0, 1, 2)) if "teca_units" in sql else (self.labels, (0, 1, 4, 5))
        seen = {tuple(r[i] for i in key) for r in table}
        table.extend(v for v in values if tuple(v[i] for i in key) not in seen)

    def run(self, cb, classify, **kwargs):
        conn = type("Conn", (), {"cursor": lambda _: self, "close": lambda _: None,
                                 "autocommit": False})()
        with patch.object(sources.psycopg2, "connect", return_value=conn), \
                patch.object(sources.psycopg2.extras, "execute_values", self.insert), \
                patch.object(cb, "judge", side_effect=lambda traces, **kw: [
                    None if label is None else (label, None) for label in classify(traces, **kw)]):
            return sources.label_postgres(cb, "dsn", "SELECT id, t FROM x",
                                          to_trace=lambda t: {"t": t}, name="support", **kwargs)

    def labeled(self):
        return {(r[0], r[5]): r[2] for r in self.labels}


def codebook(version=1, *names):
    return Codebook("q", [Category(name=n, definition="d") for n in names or ("a",)], version=version)


class TestFailedCallsStayPending(unittest.TestCase):
    def test_a_failed_row_is_labeled_by_the_next_run(self):
        store = Store([("1", "x"), ("2", "y"), ("3", "z")])
        cb = codebook()
        asked = []

        def flaky(units, workers):   # row "2" fails the first time it is asked
            asked.append([u["t"] for u in units])
            if len(asked) == 1:
                cb.last_errors = ["RuntimeError: boom"]
                return ["a", None, "a"]
            cb.last_errors = []
            return ["a"] * len(units)

        first = store.run(cb, flaky)
        self.assertIsInstance(first, LabelRun)
        self.assertEqual(first.counts, Counter({"a": 2}))
        self.assertEqual(first.failed_row_ids, ["2"])
        self.assertEqual(first.errors, ["RuntimeError: boom"])
        self.assertEqual(store.labeled(), {("1", 1): "a", ("3", 1): "a"})   # nothing written for "2"

        second = store.run(cb, flaky)
        self.assertEqual(asked[1], ["y"])                    # only the pending row is judged
        self.assertEqual(second.failed_row_ids, [])
        self.assertEqual(second.counts, Counter({"a": 1}))
        self.assertEqual(second.versions, {1: 3})

        third = store.run(cb, flaky)
        self.assertEqual(len(asked), 2)                      # nothing pending: no call
        self.assertEqual(third.counts, Counter())
        self.assertEqual(str(third), "labeled 0 (other 0%) · rows by version v1: 3")

    def test_run_summary_names_other_share_and_failures(self):
        store = Store([("1", "x"), ("2", "y"), ("3", "z"), ("4", "w")])
        run = store.run(codebook(), lambda units, workers: ["a", "other", "unclassifiable", None])
        self.assertEqual(run.other_share, 0.5)               # of judged labels, not of rows
        self.assertEqual(str(run), "labeled 3 (other 50%) · 1 failed (pending next run) "
                                   "· rows by version v1: 3")


class TestOnVersionChange(unittest.TestCase):
    def seeded(self):
        """Rows A and B labeled under v1, C never labeled; the codebook now at v2."""
        store = Store([("A", "x"), ("B", "y"), ("C", "z")])
        store.labels = [("A", 0, "a", None, "support", 1, "m"), ("B", 0, "b", None, "support", 1, "m")]
        return store

    def judged(self, store, cb, **kwargs):
        asked = []
        store.run(cb, lambda units, workers: asked.extend(u["t"] for u in units) or ["a"] * len(units),
                  **kwargs)
        return sorted(asked)

    def test_keep_is_the_default_and_judges_only_unlabeled_rows(self):
        store = self.seeded()
        self.assertEqual(self.judged(store, codebook(2, "a", "b")), ["z"])
        run = store.run(codebook(2, "a", "b"), lambda units, workers: [])
        self.assertEqual(run.versions, {1: 2, 2: 1})

    def test_relabel_sweeps_every_row_below_the_live_version(self):
        self.assertEqual(self.judged(self.seeded(), codebook(2, "a", "b"),
                                     on_version_change="relabel"), ["x", "y", "z"])

    def test_relabel_touched_follows_the_revision_log(self):
        with tempfile.TemporaryDirectory() as d:
            cb = Codebook("q", [Category(name="a", definition="d"), Category(name="b", definition="d")],
                          version=1, path=Path(d) / "support.codebook.json")
            cb.save()
            cb.apply(Revision(ops=[Op(op="rename", name="a", new_name="alpha")]))   # v2: touches a
            self.assertEqual(self.judged(self.seeded(), cb, on_version_change="relabel_touched"),
                             ["x", "z"])                     # A (was 'a') and C; B's 'b' stands

    def test_relabel_touched_needs_a_revision_log(self):
        with self.assertRaisesRegex(ValueError, "no path"):
            eligibility(codebook(2), {("A", 0): ("a", 1)}, "relabel_touched")(("A", 0))

    def test_unknown_policies_are_refused_before_any_call(self):
        with self.assertRaisesRegex(ValueError, "on_version_change must be one of"):
            self.seeded().run(codebook(2), None, on_version_change="overwrite")

    def test_a_live_version_label_never_moves_under_any_policy(self):
        for policy in ("keep", "relabel", "relabel_touched"):
            store = Store([("A", "x")])
            store.labels = [("A", 0, "a", None, "support", 3, "m")]
            self.assertEqual(self.judged(store, codebook(2), on_version_change=policy), [], policy)


class TestUnitsPath(unittest.TestCase):
    def test_extraction_failures_are_named_and_retried_next_run(self):
        store = Store([("1", "x"), ("2", "y")])
        attempts = Counter()

        def units(record):
            attempts[record["t"]] += 1
            if record["t"] == "y" and attempts["y"] <= 2:   # both tries of the first run
                raise RuntimeError("extract boom")
            return [{"quote": record["t"]}]

        cb = codebook()
        first = store.run(cb, lambda us, workers: ["a"] * len(us), units=units)
        self.assertEqual(first.failed_row_ids, ["2"])
        self.assertEqual(first.errors, ["RuntimeError: extract boom"])
        self.assertEqual(first.counts, Counter({"a": 1}))
        second = store.run(cb, lambda us, workers: ["a"] * len(us), units=units)
        self.assertEqual(second.failed_row_ids, [])
        self.assertEqual(second.counts, Counter({"a": 1}))
        self.assertEqual(attempts, Counter({"x": 1, "y": 3}))


if __name__ == "__main__":
    unittest.main()


class TestBoundaryChangesRelabelEverything(unittest.TestCase):
    def seeded(self):
        store = Store([("A", "x"), ("B", "y"), ("C", "z")])
        store.labels = [("A", 0, "a", None, "support", 1, "m"), ("B", 0, "b", None, "support", 1, "m")]
        return store

    def judged(self, store, cb):
        asked = []
        store.run(cb, lambda units, workers: asked.extend(u["t"] for u in units) or ["a"] * len(units),
                  on_version_change="relabel_touched")
        return sorted(asked)

    def test_an_add_revisits_every_old_row_not_just_other(self):
        # rows of a new theme were sitting under an old category; adding the new one must
        # re-judge them too, or old rows keep the old label while new rows get the new one
        with tempfile.TemporaryDirectory() as d:
            cb = Codebook("q", [Category(name="a", definition="d"), Category(name="b", definition="d")],
                          version=1, path=Path(d) / "support.codebook.json")
            cb.save()
            cb.apply(Revision(ops=[Op(op="add", name="c", definition="d")]))
            self.assertEqual(self.judged(self.seeded(), cb), ["x", "y", "z"])

    def test_a_deprecate_stays_local(self):
        with tempfile.TemporaryDirectory() as d:
            cb = Codebook("q", [Category(name="a", definition="d"), Category(name="b", definition="d")],
                          version=1, path=Path(d) / "support.codebook.json")
            cb.save()
            cb.apply(Revision(ops=[Op(op="deprecate", name="b")]))
            self.assertEqual(self.judged(self.seeded(), cb), ["y", "z"])   # B (was 'b') and C


class TestRelabelTouchedNeedsTheLog(unittest.TestCase):
    def test_a_codebook_with_a_path_but_no_log_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            cb = Codebook("q", [Category(name="a", definition="d")], version=3,
                          path=Path(d) / "support.codebook.json")
            cb.save()                                        # file, no log beside it
            with self.assertRaisesRegex(ValueError, "log"):
                eligibility(cb, {("A", 0): ("a", 1)}, "relabel_touched")(("A", 0))
