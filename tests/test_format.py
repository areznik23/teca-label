"""The on-disk format (FORMAT.md): every artifact names its schema; load() reads
schema 1, treats a missing field as schema 1, and refuses a newer one."""
import json
import tempfile
import unittest
from pathlib import Path

from teca_label import Category, Codebook, Op, Revision
from teca_label.core import SCHEMA


def saved(path: Path, **overrides) -> Path:
    body = {"question": "q", "version": 1,
            "categories": [{"name": "a", "definition": "A", "created_v": 1, "deprecated_v": None}]}
    body.update(overrides)
    path.write_text(json.dumps(body))
    return path


class TestSchemaField(unittest.TestCase):
    def test_every_artifact_names_the_schema(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.codebook.json"
            cb = Codebook.adopt("q", [Category(name="a", definition="A")], path=path)
            self.assertEqual(json.loads(path.read_text())["schema"], SCHEMA)
            log_line = json.loads(cb.log_path.read_text().splitlines()[0])
            self.assertEqual(log_line["schema"], SCHEMA)
            cb._finish_proposal(Revision(ops=[Op(op="add", name="b", definition="B")]), [])
            self.assertEqual(json.loads(cb.pending_path.read_text())["schema"], SCHEMA)
            cb.apply()                                    # the schema key doesn't break the revision
            self.assertEqual([c.name for c in cb.active()], ["a", "b"])

    def test_hand_written_minimum_loads(self):
        with tempfile.TemporaryDirectory() as d:
            cb = Codebook.load(saved(Path(d) / "x.codebook.json"))
            self.assertEqual(cb.version, 1)
            self.assertEqual(cb.kind, "partition")
            self.assertEqual([c.name for c in cb.active()], ["a"])

    def test_newer_schema_refuses_with_upgrade_hint(self):
        with tempfile.TemporaryDirectory() as d:
            path = saved(Path(d) / "x.codebook.json", schema=SCHEMA + 1)
            with self.assertRaisesRegex(ValueError, "upgrade teca-label"):
                Codebook.load(path)

    def test_not_a_codebook_says_so(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.codebook.json"
            path.write_text(json.dumps({"ops": []}))
            with self.assertRaisesRegex(ValueError, "not a codebook"):
                Codebook.load(path)


if __name__ == "__main__":
    unittest.main()


class TestLoadIsForgivingWhereFormatPromises(unittest.TestCase):
    def test_missing_required_keys_are_named(self):
        import json
        import tempfile
        from pathlib import Path
        from teca_label.core import Codebook
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.codebook.json"
            p.write_text(json.dumps({"question": "q", "categories": []}))
            with self.assertRaisesRegex(ValueError, "version"):
                Codebook.load(p)

    def test_unknown_policy_keys_from_a_later_library_are_ignored_with_a_warning(self):
        import json
        import tempfile
        from pathlib import Path
        from teca_label.core import Codebook
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.codebook.json"
            p.write_text(json.dumps({"question": "q", "version": 1, "categories": [],
                                     "policy": {"min_batch": 7, "future_knob": 3}}))
            with self.assertWarns(UserWarning):
                cb = Codebook.load(p)
            self.assertEqual(cb.policy["min_batch"], 7)
            self.assertNotIn("future_knob", cb.policy)
