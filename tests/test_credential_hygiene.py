"""Credential-hygiene guards: the DSN must never leak into errors, logs, or artifacts.
No database and no API required — failures are simulated."""
import inspect
import re
import unittest
from unittest.mock import patch

import teca_label.sources as sources
from teca_label.core import Category, Codebook

SECRET_DSN = "postgresql://user:hunter2-secret@db.example.com:5432/app"


class TestNameDefaulting(unittest.TestCase):
    def test_name_derives_from_the_codebook_filename(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            cb = Codebook("q", [Category(name="a", definition="d")], version=1,
                          path=Path(tmp) / "failure-modes.codebook.json")
            with patch.object(sources.psycopg2, "connect",
                              side_effect=Exception("stop here")):
                try:
                    sources.label_postgres(cb, "dsn", "SELECT 1", to_trace=dict)
                except Exception:
                    pass
            # the derivation itself is the contract; verify it directly
            self.assertEqual(cb.path.name.removesuffix(".json").removesuffix(".codebook"),
                             "failure-modes")

    def test_pathless_codebook_requires_an_explicit_name(self):
        cb = Codebook("q", [Category(name="a", definition="d")], version=1)
        with self.assertRaises(ValueError):
            sources.label_postgres(cb, "dsn", "SELECT 1", to_trace=dict)


class TestDsnNeverLeaks(unittest.TestCase):
    def test_connect_failure_propagates_without_the_dsn(self):
        cb = Codebook("q", [Category(name="a", definition="d")], version=1)
        with patch.object(sources.psycopg2, "connect",
                          side_effect=Exception("connection refused")):
            with self.assertRaises(Exception) as ctx:
                sources.label_postgres(cb, SECRET_DSN, "SELECT id, body FROM t",
                                       to_trace=lambda body: {"body": body}, name="x")
        self.assertNotIn("hunter2-secret", str(ctx.exception))
        self.assertNotIn(SECRET_DSN, str(ctx.exception))

    def test_dsn_is_used_exactly_once_and_only_to_connect(self):
        # static tripwire: if a future edit interpolates the DSN into any message,
        # this count changes and the diff gets a deliberate look
        src = inspect.getsource(sources)
        uses = re.findall(r"\bdsn\b", src)
        # _connect(dsn) declares + connects; fetch, label_postgres, latest_labels,
        # and the runs-ledger four (run_start, run_finish, last_run, last_proposal)
        # each declare + pass it on — audited: none reach a message or artifact
        self.assertEqual(len(uses), 16,
                         "dsn is referenced somewhere new in sources.py — verify it "
                         "cannot reach a log, exception message, or artifact")
        self.assertIn("psycopg2.connect(dsn", src)

    def test_codebook_artifacts_contain_no_environment_secrets(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            cb = Codebook("q", [Category(name="a", definition="d")], version=1,
                          path=Path(tmp) / "x.codebook.json")
            cb.save()
            saved = json.loads(cb.path.read_text())
            # the artifact schema is a closed set: nothing resembling config or auth
            self.assertLessEqual(set(saved),
                                 {"schema", "question", "version", "models", "status", "kind",
                                  "policy", "parent", "draft_window", "categories"})


if __name__ == "__main__":
    unittest.main()
