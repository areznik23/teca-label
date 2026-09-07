"""Category names are identifiers: snake_case ASCII, at most 40 chars, unique
case-insensitively, never a label-contract value. Model drafts are normalized; human
ops are refused with the fix spelled out. No API."""
import unittest

from teca_label.core import (Category, Codebook, InvalidRevision, Op, Revision,
                          _normalize_names, _validate_ops, slug)


def categories(*names: str) -> list[Category]:
    return [Category(name=n, definition=f"def of {n}") for n in names]


class TestSlug(unittest.TestCase):
    def test_prose_titles_become_identifiers(self):
        self.assertEqual(slug("Rate-limit cascade"), "rate_limit_cascade")
        self.assertEqual(slug("Tool loop with no new information"),
                         "tool_loop_with_no_new_information")
        self.assertEqual(slug("  Rate/limit  (429) storm! "), "rate_limit_429_storm")
        self.assertEqual(slug("Café résumé"), "cafe_resume")

    def test_truncates_at_forty_without_a_dangling_underscore(self):
        long = slug("a very long category name that keeps going and going and going")
        self.assertLessEqual(len(long), 40)
        self.assertFalse(long.endswith("_"))


class TestHumanOpsAreRefused(unittest.TestCase):
    def test_bad_names_name_the_fix(self):
        for bad in ("Tool loop", "Rate-limit", "x" * 41, "9lives"):
            with self.assertRaisesRegex(InvalidRevision, "snake_case", msg=bad):
                _validate_ops([], [Op(op="add", name=bad, definition="d")])
        with self.assertRaisesRegex(InvalidRevision, "try 'tool_loop'"):
            _validate_ops([], [Op(op="add", name="Tool loop", definition="d")])

    def test_reserved_labels_cannot_become_categories(self):
        with self.assertRaisesRegex(InvalidRevision, "reserved"):
            _validate_ops([], [Op(op="add", name="other", definition="d")])
        with self.assertRaisesRegex(ValueError, "reserved"):
            Codebook("q", categories("unclassifiable"))

    def test_collisions_are_case_insensitive_across_every_new_name(self):
        base = categories("tool_loop")
        with self.assertRaisesRegex(InvalidRevision, "already exists"):
            _validate_ops(base, [Op(op="add", name="Tool_Loop", definition="d")])
        with self.assertRaisesRegex(InvalidRevision, "already exists"):
            _validate_ops(categories("a", "b"), [Op(op="rename", name="a", new_name="B")])
        with self.assertRaisesRegex(InvalidRevision, "already exists"):
            _validate_ops(categories("a", "b", "c"),
                          [Op(op="merge", names=["a", "b"], new_name="C", definition="d")])
        with self.assertRaisesRegex(InvalidRevision, "already exists"):
            _validate_ops(categories("a", "b"), [Op(op="split", name="a", into=categories("B"))])

    def test_adopt_and_load_check_names_too(self):
        with self.assertRaisesRegex(ValueError, "snake_case"):
            Codebook("q", [Category(name="Has Spaces", definition="d")])


class TestDraftsAreNormalized(unittest.TestCase):
    def test_every_new_name_in_a_model_revision_is_slugged(self):
        revision = Revision(ops=[
            Op(op="add", name="Rate-limit cascade", definition="d"),
            Op(op="rename", name="a", new_name="New Name"),
            Op(op="merge", names=["b", "c"], new_name="Merged Thing", definition="d"),
            Op(op="split", name="d", into=[Category(name="Part One", definition="d"),
                                            Category(name="Part Two", definition="d")])])
        ops = _normalize_names(revision).ops
        self.assertEqual(ops[0].name, "rate_limit_cascade")
        self.assertEqual(ops[1].new_name, "new_name")
        self.assertEqual(ops[2].new_name, "merged_thing")
        self.assertEqual([c.name for c in ops[3].into], ["part_one", "part_two"])
        self.assertEqual(ops[1].name, "a")          # targets of ops are untouched

    def test_propose_normalizes_before_validating_and_applying(self):
        class Drafting(Codebook):
            def _call(self, role, system, content, schema, max_tokens=2048, timeout=60.0, model=None):
                return Revision(ops=[Op(op="add", name="Tool loop with no new information",
                                        definition="d", evidence=[0, 1, 2])])
        cb = Drafting("q", path=None)
        cb.apply(cb.propose([{"text": "x"}] * 3))
        self.assertEqual([c.name for c in cb.active()],
                         ["tool_loop_with_no_new_information"])


if __name__ == "__main__":
    unittest.main()
