"""Tests for the envelope contract and the message-array adapter. No API."""
import json
import tempfile
import unittest
from pathlib import Path

from teca_label.ingest import (envelope, events, from_messages, read_jsonl,
                            revision_loops, tool_firings, write_jsonl)


class TestEnvelope(unittest.TestCase):
    def test_normalizes_ts_and_coerces_id(self):
        row = envelope(42, "2026-07-24T13:09:24Z", {"text": "hi"})
        self.assertEqual(row["id"], "42")
        self.assertEqual(row["ts"], "2026-07-24T13:09:24+00:00")

    def test_naive_ts_assumed_utc(self):
        self.assertEqual(envelope("a", "2026-01-01T00:00:00", "x")["ts"],
                         "2026-01-01T00:00:00+00:00")

    def test_summary_derived_from_str_and_dict_traces(self):
        self.assertEqual(envelope("a", "2026-01-01", "  two\n words  ")["summary"], "two words")
        self.assertIn("hello", envelope("a", "2026-01-01", {"text": "hello"})["summary"])
        long = envelope("a", "2026-01-01", "x" * 500)["summary"]
        self.assertLessEqual(len(long), 141)  # clipped with an ellipsis

    def test_explicit_summary_wins(self):
        self.assertEqual(envelope("a", "2026-01-01", "t", summary="mine")["summary"], "mine")

    def test_apps_omitted_unless_given(self):
        self.assertNotIn("apps", envelope("a", "2026-01-01", "t"))
        self.assertEqual(envelope("a", "2026-01-01", "t", apps=["x"])["apps"], ["x"])

    def test_loud_failures(self):
        with self.assertRaises(ValueError):
            envelope("", "2026-01-01", "t")            # empty id
        with self.assertRaises(ValueError):
            envelope("a", "not a date", "t")           # garbage ts
        with self.assertRaises(ValueError):
            envelope("a", "2026-01-01", None)          # no trace
        with self.assertRaises(ValueError):
            envelope("a", "2026-01-01", "")            # empty trace


class TestFromMessages(unittest.TestCase):
    OPENAI = [
        {"role": "system", "content": "You are a helpful agent."},
        {"role": "user", "content": "book me a flight"},
        {"role": "assistant", "content": "Searching now.",
         "tool_calls": [{"function": {"name": "search_flights",
                                      "arguments": '{"dest": "SFO"}'}}]},
        {"role": "tool", "content": "FLIGHT DATA " * 100},
        {"role": "assistant", "content": "Booked UA 512."},
    ]

    def test_openai_shape_keeps_intent_drops_bulk(self):
        t = from_messages(self.OPENAI)
        kinds = [next(iter(l)) for l in t["messages"]]
        self.assertEqual(kinds, ["user", "assistant", "tool_call", "assistant"])
        self.assertEqual(t["messages"][2]["tool_call"], "search_flights")
        self.assertEqual(t["n_messages"], 5)  # counts the originals, not the survivors

    def test_tool_results_kept_and_clipped_on_request(self):
        t = from_messages(self.OPENAI, keep_tool_results=True, max_result_chars=50)
        results = [l for l in t["messages"] if "tool_result" in l]
        self.assertEqual(len(results), 1)
        self.assertLessEqual(len(results[0]["tool_result"]), 51)

    def test_anthropic_shape(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "fix the bug"}]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Reading the file."},
                {"type": "tool_use", "name": "read_file", "input": {"path": "a.py"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "content": [{"type": "text", "text": "def f(): pass"}]}]},
            {"role": "assistant", "content": "Fixed."},
        ]
        t = from_messages(msgs)
        kinds = [next(iter(l)) for l in t["messages"]]
        self.assertEqual(kinds, ["user", "assistant", "tool_call", "assistant"])
        self.assertEqual(t["messages"][2]["tool_call"], "read_file")
        kept = from_messages(msgs, keep_tool_results=True)
        self.assertIn({"tool_result": "def f(): pass"}, kept["messages"])

    def test_long_traces_keep_head_and_tail(self):
        msgs = [{"role": "user", "content": f"message number {i} " + "pad " * 30}
                for i in range(200)]
        t = from_messages(msgs, max_chars=3_000)
        rendered = json.dumps(t["messages"])
        self.assertLess(len(rendered), 3_500)
        self.assertIn("message number 0", rendered)          # the ask survives
        self.assertIn("message number 199", rendered)        # the outcome survives
        marker = [l for l in t["messages"] if "omitted" in l]
        self.assertEqual(len(marker), 1)
        self.assertIn("messages", marker[0]["omitted"])

    def test_under_budget_is_untouched(self):
        t = from_messages([{"role": "user", "content": "hi"}], max_chars=12_000)
        self.assertEqual(t["messages"], [{"user": "hi"}])


class TestJsonl(unittest.TestCase):
    def test_round_trip_and_blank_lines(self):
        rows = [envelope(i, "2026-01-01", f"trace {i}") for i in range(3)]
        with tempfile.TemporaryDirectory() as d:
            p = write_jsonl(Path(d) / "t.jsonl", rows)
            p.write_text(p.read_text() + "\n\n")  # trailing blanks happen
            self.assertEqual(read_jsonl(p), rows)


class TestSequences(unittest.TestCase):
    SESSION = [  # user asks, tool fires, user revises, same tool fires again
        {"role": "user", "content": "find last quarter's churned accounts"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "drafting"},
            {"type": "tool_use", "name": "search_docs", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "content": "ok"},
                                     {"type": "text", "text": "too many, just the top three"}]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "name": "search_docs", "input": {}},
            {"type": "tool_use", "name": "run_query", "input": {}}]},
        {"role": "assistant", "content": [  # re-fires with NO user text between
            {"type": "tool_use", "name": "run_query", "input": {}}]},
    ]

    def test_events_stream_order_and_shapes(self):
        self.assertEqual(events(self.SESSION), [
            ("user", "find last quarter's churned accounts"),
            ("tool", "search_docs"),
            ("user", "too many, just the top three"),
            ("tool", "search_docs"),
            ("tool", "run_query"),
            ("tool", "run_query")])

    def test_events_openai_tool_calls(self):
        msgs = [{"role": "assistant",
                 "tool_calls": [{"function": {"name": "search", "arguments": "{}"}}]}]
        self.assertEqual(events(msgs), [("tool", "search")])

    def test_tool_firings_carry_the_live_ask(self):
        rows = tool_firings(self.SESSION, "search_docs",
                            session_id="s1", ts="2026-07-01T00:00:00Z")
        self.assertEqual([r["id"] for r in rows], ["s1:1", "s1:2"])
        self.assertEqual(rows[0]["trace"]["preceding_ask"], "find last quarter's churned accounts")
        self.assertEqual(rows[1]["trace"]["preceding_ask"], "too many, just the top three")

    def test_firings_before_any_user_text_are_skipped(self):
        msgs = [{"role": "assistant",
                 "content": [{"type": "tool_use", "name": "x", "input": {}}]}]
        self.assertEqual(tool_firings(msgs, "x", session_id="s", ts="2026-01-01"), [])

    def test_revision_loops_need_user_text_between(self):
        rows = revision_loops(self.SESSION, session_id="s1", ts="2026-07-01T00:00:00Z")
        self.assertEqual(len(rows), 1)   # run_query's silent re-fire makes no row
        self.assertEqual(rows[0]["id"], "s1:search_docs:1")
        self.assertEqual(rows[0]["trace"]["user_reply"], "too many, just the top three")

    def test_revision_loops_tool_filter(self):
        rows = revision_loops(self.SESSION, tools=["run_query"],
                              session_id="s1", ts="2026-07-01T00:00:00Z")
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
