"""Table-driven tests for the deterministic core: the op fold and its validation.
Run: python3 -m unittest discover -s tests"""
import unittest

from teca_label.core import Category, InvalidRevision, Op, _apply_ops, _validate_ops


def categories(*names: str) -> list[Category]:
    return [Category(name=n, definition=f"def of {n}") for n in names]


def active_names(cs: list[Category]) -> set[str]:
    return {c.name for c in cs if c.deprecated_v is None}


class TestApplyOps(unittest.TestCase):
    def test_add_creates_category_at_version(self):
        out = _apply_ops([], [Op(op="add", name="a", definition="d")], version=3)
        self.assertEqual(out[0].name, "a")
        self.assertEqual(out[0].created_v, 3)
        self.assertIsNone(out[0].deprecated_v)

    def test_rename_keeps_lineage_fields(self):
        base = _apply_ops([], [Op(op="add", name="a", definition="d")], 1)
        out = _apply_ops(base, [Op(op="rename", name="a", new_name="b")], 2)
        self.assertEqual(active_names(out), {"b"})
        self.assertEqual(out[0].created_v, 1)  # rename is identity-preserving

    def test_redefine_changes_definition_only(self):
        out = _apply_ops(categories("a"), [Op(op="redefine", name="a", definition="sharper")], 2)
        self.assertEqual(out[0].definition, "sharper")
        self.assertEqual(active_names(out), {"a"})

    def test_deprecate_never_deletes(self):
        out = _apply_ops(categories("a", "b"), [Op(op="deprecate", name="a")], 2)
        self.assertEqual(len(out), 2)  # still present in history
        self.assertEqual(active_names(out), {"b"})
        self.assertEqual(next(c for c in out if c.name == "a").deprecated_v, 2)

    def test_merge_deprecates_sources_and_births_successor(self):
        out = _apply_ops(categories("a", "b"),
                         [Op(op="merge", names=["a", "b"], new_name="ab", definition="d")], 2)
        self.assertEqual(active_names(out), {"ab"})
        self.assertEqual(len(out), 3)
        self.assertEqual(next(c for c in out if c.name == "ab").created_v, 2)

    def test_split_deprecates_parent_and_births_children(self):
        out = _apply_ops(categories("a"),
                         [Op(op="split", name="a", into=categories("a1", "a2"))], 2)
        self.assertEqual(active_names(out), {"a1", "a2"})
        self.assertEqual(len(out), 3)

    def test_replay_is_deterministic(self):
        ops1 = [Op(op="add", name="a", definition="d"), Op(op="add", name="b", definition="d")]
        ops2 = [Op(op="merge", names=["a", "b"], new_name="ab", definition="d")]
        ops3 = [Op(op="rename", name="ab", new_name="c")]
        replay = lambda: _apply_ops(_apply_ops(_apply_ops([], ops1, 1), ops2, 2), ops3, 3)
        self.assertEqual([c.model_dump() for c in replay()],
                         [c.model_dump() for c in replay()])
        self.assertEqual(active_names(replay()), {"c"})


class TestValidateOps(unittest.TestCase):
    def assertRejected(self, base, ops, fragment):
        with self.assertRaises(InvalidRevision) as ctx:
            _validate_ops(base, ops)
        self.assertIn(fragment, str(ctx.exception))

    def test_add_duplicate_rejected(self):
        self.assertRejected(categories("a"), [Op(op="add", name="a", definition="d")], "already exists")

    def test_add_without_definition_rejected(self):
        self.assertRejected([], [Op(op="add", name="a")], "missing definition")

    def test_rename_missing_target_rejected(self):
        self.assertRejected(categories("a"), [Op(op="rename", name="ghost", new_name="b")], "does not exist")

    def test_rename_onto_deprecated_name_rejected(self):
        base = _apply_ops(categories("a", "b"), [Op(op="deprecate", name="a")], 2)
        self.assertRejected(base, [Op(op="rename", name="b", new_name="a")], "already exists")

    def test_deprecate_twice_rejected(self):
        base = _apply_ops(categories("a", "b"), [Op(op="deprecate", name="a")], 2)
        self.assertRejected(base, [Op(op="deprecate", name="a")], "already deprecated")

    def test_ops_on_deprecated_category_rejected(self):
        base = _apply_ops(categories("a", "b"), [Op(op="deprecate", name="a")], 2)
        self.assertRejected(base, [Op(op="redefine", name="a", definition="d")], "already deprecated")
        self.assertRejected(base, [Op(op="rename", name="a", new_name="c")], "already deprecated")
        self.assertRejected(base, [Op(op="split", name="a", into=categories("c"))], "already deprecated")
        self.assertRejected(base, [Op(op="merge", names=["a", "b"], new_name="m", definition="d")],
                            "already deprecated")

    def test_deprecate_twice_within_one_revision_rejected(self):
        self.assertRejected(categories("a"), [Op(op="deprecate", name="a"), Op(op="deprecate", name="a")],
                            "already deprecated")

    def test_merge_with_missing_source_rejected(self):
        self.assertRejected(categories("a"),
                            [Op(op="merge", names=["a", "ghost"], new_name="m", definition="d")],
                            "does not exist")

    def test_split_into_colliding_name_rejected(self):
        self.assertRejected(categories("a", "b"), [Op(op="split", name="a", into=categories("b"))], "already exists")

    def test_ops_validate_in_sequence(self):
        ok = [Op(op="add", name="a", definition="d"), Op(op="rename", name="a", new_name="b")]
        _validate_ops([], ok)  # rename sees the add before it — no raise

    def test_valid_revision_passes(self):
        _validate_ops(categories("a", "b"),
                      [Op(op="deprecate", name="a"),
                       Op(op="add", name="c", definition="d"),
                       Op(op="redefine", name="b", definition="sharper")])


if __name__ == "__main__":
    unittest.main()
