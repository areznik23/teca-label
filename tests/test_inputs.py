"""Input checks that fire before a model call is spent: empty corpora, non-dict
traces. No API."""
import unittest

from teca_label.core import Category, Codebook, Op, Revision, text_chars, thin_traces


class TestBuildInputs(unittest.TestCase):
    def test_too_few_traces_refuses_to_draft(self):
        with self.assertRaisesRegex(ValueError, "not enough to draft"):
            Codebook.plan([], "q", path=None).build()
        with self.assertRaisesRegex(ValueError, "2 traces"):
            Codebook.plan([{"text": "a"}, {"text": "b"}], "q", path=None, min_evidence=3).build()


class TestThinTraces(unittest.TestCase):
    PROSE = {"title": "Export flow stalls before the pricing page goes live.",
             "body": "User asked twice, agent looped on the same search call."}

    def test_counts_prose_across_nested_fields(self):
        self.assertEqual(text_chars({"a": " hi ", "b": {"c": ["there", 3]}}), 7)
        self.assertEqual(text_chars({"n": 42, "flag": True}), 0)

    def test_names_the_fields_blank_in_every_thin_trace(self):
        traces = [self.PROSE] * 5 + [{"title": "", "body": None, "id": "x"}] * 2
        self.assertEqual(thin_traces(traces),
                         "2 of 7 traces have under 50 chars of text (title, body all empty)")
        self.assertIsNone(thin_traces([self.PROSE] * 3))

    def test_build_refuses_thin_traces_unless_allowed(self):
        class Drafting(Codebook):
            def _call(self, role, system, content, schema, max_tokens=2048, timeout=60.0, model=None):
                return Revision(ops=[Op(op="add", name="a", definition="d", evidence=[0, 1, 2])])
        traces = [self.PROSE] * 4 + [{"title": "", "body": ""}] * 2
        with self.assertRaisesRegex(ValueError, "2 of 6 traces have under 50 chars.*allow_empty=True"):
            Drafting.plan(traces, "q", path=None).build()
        self.assertEqual(Drafting.plan(traces, "q", path=None, allow_empty=True).build().version, 1)
        self.assertEqual(Drafting.plan(traces, "q", path=None, min_chars=0).build().version, 1)


class TestTraceShape(unittest.TestCase):
    def test_non_dict_traces_fail_before_any_call(self):
        cb = Codebook("q", [Category(name="a", definition="d")], version=1)
        calls = []
        cb._call = lambda *a, **k: calls.append(1)
        with self.assertRaisesRegex(TypeError, r"traces\[1\] is str"):
            cb.classify([{"text": "fine"}, "not a dict"])
        with self.assertRaisesRegex(TypeError, "not a dict"):
            cb.propose(["plain"], labels=["other"])
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()


class TestEnvelopesAreUnwrapped(unittest.TestCase):
    def test_the_model_reads_only_the_trace_of_an_envelope(self):
        from teca_label import envelope
        from teca_label.core import content, text_chars
        row = envelope("s1", "2026-01-01", {"text": "hello there"}, summary="a long summary here")
        self.assertEqual(content(row), {"text": "hello there"})
        self.assertEqual(text_chars(row), len("hello there"))     # summary/id/ts don't count
        self.assertEqual(content(envelope("s2", "2026-01-01", "plain")), {"text": "plain"})
        self.assertEqual(content({"text": "not an envelope"}), {"text": "not an envelope"})

    def test_classify_sends_the_trace_not_the_envelope(self):
        from unittest.mock import patch
        from teca_label import envelope
        from teca_label.core import Category, Codebook
        cb = Codebook("q", [Category(name="a", definition="d")], version=1)
        sent = []
        def fake_call(role, system, body, schema, *a, **k):
            sent.append(body)
            return type("L", (), {"label": "a", "evidence": ""})()
        with patch.object(Codebook, "_call", side_effect=fake_call):
            cb.classify([envelope("s1", "2026-01-01", {"text": "hello there"}, summary="SUMMARY")])
        self.assertNotIn("SUMMARY", sent[0])
        self.assertNotIn("s1", sent[0])
