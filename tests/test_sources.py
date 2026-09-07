"""fetch() and the shared select path: id-first convention, SQL-side stride, and the
same query serving both fetch and label_postgres. No database — the cursor is a stand-in
that records SQL and replays canned rows."""
import unittest
from unittest.mock import patch

import teca_label.sources as sources
from teca_label.core import Category, Codebook, _sample


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.executed = []
        self.params = []

    def execute(self, sql, params=None):
        self.executed.append(sql)
        self.params.append(params)

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class FakeConn:
    def __init__(self, rows):
        self.cursor_ = FakeCursor(rows)
        self.closed = False
        self.autocommit = False

    def cursor(self):
        return self.cursor_

    def close(self):
        self.closed = True


def connected(rows):
    conn = FakeConn(rows)
    return patch.object(sources.psycopg2, "connect", return_value=conn), conn


class TestFetch(unittest.TestCase):
    def test_id_first_rest_to_trace_in_query_order(self):
        patcher, conn = connected([(7, "a", 1), (3, "b", 2)])
        with patcher, self.assertWarnsRegex(UserWarning, "2 of 2 traces have under 50 chars"):
            traces = sources.fetch("dsn", "SELECT id, body, n FROM t ORDER BY ts",
                                   lambda body, n: {"body": body, "n": n})
        self.assertEqual(traces, [{"body": "a", "n": 1}, {"body": "b", "n": 2}])
        self.assertEqual(conn.cursor_.executed, ["SELECT id, body, n FROM t ORDER BY ts"])
        self.assertTrue(conn.closed)

    def test_sample_wraps_the_query_and_strips_the_window_columns(self):
        patcher, conn = connected([(7, "a", 1, 200), (3, "b", 4, 200)])
        with patcher, self.assertWarns(UserWarning):
            traces = sources.fetch("dsn", "SELECT id, body FROM t ORDER BY ts",
                                   lambda body: body, sample=60)
        self.assertEqual(traces, ["a", "b"])
        sql = conn.cursor_.executed[0]
        self.assertIn("FROM (SELECT id, body FROM t ORDER BY ts) q", sql)
        self.assertIn("DISTINCT ON (((teca_rn - 1) * 60) / teca_n)", sql)   # first of 60 even buckets

    def test_sample_is_an_int_or_nothing_reaches_sql(self):
        patcher, conn = connected([])
        with patcher, self.assertRaises(ValueError):
            sources.fetch("dsn", "SELECT id FROM t", dict, sample="60; DROP TABLE t")
        self.assertEqual(conn.cursor_.executed, [])
        self.assertTrue(conn.closed)

    def test_connection_is_closed_when_to_trace_fails(self):
        patcher, conn = connected([(1, "a")])

        def boom(body):
            raise KeyError("missing")

        with patcher, self.assertRaises(KeyError):
            sources.fetch("dsn", "SELECT id, body FROM t", boom)
        self.assertTrue(conn.closed)


class TestStrideMatchesInMemorySample(unittest.TestCase):
    """The SQL predicate mirrors _sample: step = max(1, n // k); keep indexes 0, step, 2·step…"""
    def stride(self, total, k):
        step = max(1, total // k)
        return [i for i in range(total) if i % step == 0][:k]

    def test_same_indexes_on_every_edge(self):
        for total, k in [(1000, 60), (100, 60), (7, 60), (61, 60), (0, 60), (250, 1), (59, 60)]:
            expected = [i for i, _ in enumerate(_sample(list(range(total)), k))]
            self.assertEqual(self.stride(total, k), [i * max(1, total // k) for i in expected])


class TestLabelPostgresSharesTheSelectPath(unittest.TestCase):
    def test_labeling_runs_the_query_verbatim_and_skips_labeled_rows(self):
        cb = Codebook("q", [Category(name="a", definition="d")], version=2)
        cursor_rows = iter([
            [(1, "x"), (2, "y")],     # the query
            [("1", 0, "a", 2)],       # latest labels: row 1 judged at v2 (row_id is text)
        ])
        conn = FakeConn([])
        conn.cursor_.fetchall = lambda: next(cursor_rows, [])
        with patch.object(sources.psycopg2, "connect", return_value=conn), \
                patch.object(sources.psycopg2.extras, "execute_values"), \
                patch.object(Codebook, "judge", return_value=[("a", "quoted")]) as judge:
            written = sources.label_postgres(cb, "dsn", "SELECT id, body FROM t",
                                             to_trace=lambda body: {"body": body}, name="l")
        self.assertIn("SELECT id, body FROM t", conn.cursor_.executed)   # verbatim, after the DDL/column checks
        judge.assert_called_once_with([{"body": "y"}], workers=8)
        self.assertEqual(written.counts, {"a": 1})
        self.assertEqual(written.versions, {2: 2})


class TestRunsLedger(unittest.TestCase):
    def test_run_start_opens_a_running_row_and_returns_its_id(self):
        patcher, conn = connected([])
        with patcher:
            run_id = sources.run_start("dsn", "support", 3, "claude-opus-5")
        self.assertEqual(len(run_id), 32)                       # uuid4 hex
        self.assertIn("CREATE TABLE IF NOT EXISTS teca_runs", conn.cursor_.executed[0])
        insert_at = next(i for i, sql in enumerate(conn.cursor_.executed) if "INSERT INTO teca_runs" in sql)
        self.assertEqual(conn.cursor_.params[insert_at], (run_id, "support", 3, "claude-opus-5"))
        self.assertTrue(conn.closed)

    def test_run_finish_updates_only_the_still_running_row(self):
        patcher, conn = connected([])
        with patcher:
            sources.run_finish("dsn", "rid", "ok", rows_labeled=6, gaps=1, other_share=0.03)
        sql = conn.cursor_.executed[0]
        self.assertIn("WHERE run_id = %s AND status = 'running'", sql)
        self.assertIn("rows_labeled = %s", sql)
        self.assertEqual(conn.cursor_.params[0], ("ok", 6, 1, 0.03, "rid"))

    def test_run_finish_rejects_fields_outside_the_ledger(self):
        with self.assertRaises(ValueError):                    # before any connect
            sources.run_finish("dsn", "rid", "ok", vibes=1)

    def test_last_run_is_none_before_any_ledger_exists(self):
        class NoTableCursor(FakeCursor):
            def execute(self, sql, params=None):
                raise sources.psycopg2.errors.UndefinedTable("no teca_runs")
        conn = FakeConn([])
        conn.cursor_ = NoTableCursor([])
        with patch.object(sources.psycopg2, "connect", return_value=conn):
            self.assertIsNone(sources.last_run("dsn", "support"))

    def test_labels_written_by_a_run_carry_its_run_id(self):
        cb = Codebook("q", [Category(name="a", definition="d")], version=1)
        cursor_rows = iter([
            [(1, "x")],               # the query
            [],                       # no labels yet
        ])
        conn = FakeConn([])
        conn.cursor_.fetchall = lambda: next(cursor_rows, [])
        with patch.object(sources.psycopg2, "connect", return_value=conn), \
                patch.object(sources.psycopg2.extras, "execute_values") as inserts, \
                patch.object(Codebook, "judge", return_value=[("a", None)]):
            sources.label_postgres(cb, "dsn", "SELECT id, body FROM t",
                                   to_trace=lambda body: {"body": body}, name="l",
                                   run_id="rid7")
        insert_sql, values = inserts.call_args.args[1], inserts.call_args.args[2]
        self.assertIn("run_id", insert_sql)
        self.assertEqual(values[0][-1], "rid7")


if __name__ == "__main__":
    unittest.main()
