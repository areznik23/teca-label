"""The store-agnostic half of labeling a table: what a version bump does to rows
already labeled, and what one run reports back. `sources.label_postgres` is the
Postgres end of this; another store would reuse everything here.

Labels are kept honest across versions by policy, never by accident:

- "keep"            label only rows no version has judged. Old labels stand as history
                    (the conservative default — a redefinition never rewrites the past
                    without being asked).
- "relabel"         also relabel every row whose latest label is below the codebook's
                    version — the full sweep.
- "relabel_touched" also relabel the rows a revision could have moved, read from the
                    codebook's log: after a rename or deprecate, only that category's
                    rows; after any op that changes a boundary (add, redefine, split,
                    merge), every row below the live version — a new category can pull
                    rows out of any old one, and relabeling less would manufacture a
                    trend. The standing loop's default.

Labels written under older versions are never deleted; the labels table is history,
and a latest-per-unit view is the present."""
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from .core import Codebook, _revision_events, relabel_targets

VERSION_POLICIES = ("keep", "relabel", "relabel_touched")

Latest = dict[tuple[str, int], tuple[str, int]]   # (row_id, unit_index) -> (category, version)


@dataclass
class LabelRun:
    """What one labeling run wrote, what it could not, and where the table stands after it.
    Rows in `failed_row_ids` got no label, so they are pending on the next call — always."""
    counts: Counter = field(default_factory=Counter)   # labels written this run, by category
    failed_row_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)    # distinct failure messages, in order seen
    other_share: float = 0.0                           # 'other' among this run's judged labels
    versions: dict[int, int] = field(default_factory=dict)   # units by latest-label version, after

    def __str__(self) -> str:
        written = sum(self.counts.values())
        line = f"labeled {written:,} (other {self.other_share:.0%})"
        if self.failed_row_ids:
            line += f" · {len(self.failed_row_ids):,} failed (pending next run)"
        if self.versions:
            versions = ", ".join(f"v{v}: {n:,}" for v, n in sorted(self.versions.items()))
            line += f" · rows by version {versions}"
        return line

    def finish(self, latest: Latest) -> "LabelRun":
        """Derive the summary fields once the writes are in."""
        self.errors = list(dict.fromkeys(self.errors))
        self.failed_row_ids = list(dict.fromkeys(self.failed_row_ids))
        judged = sum(n for category, n in self.counts.items() if category != "unclassifiable")
        self.other_share = self.counts.get("other", 0) / judged if judged else 0.0
        self.versions = dict(sorted(Counter(version for _, version in latest.values()).items()))
        return self


def eligibility(codebook: Codebook, latest: Latest,
                on_version_change: str) -> Callable[[tuple[str, int]], bool]:
    """The predicate for 'does this unit need a label this run', per policy. A unit
    nobody has judged always does; one already judged under the live version never
    does; between those, the policy decides."""
    if on_version_change not in VERSION_POLICIES:
        raise ValueError(f"on_version_change must be one of {VERSION_POLICIES}, "
                         f"got '{on_version_change}'")
    targets_by_version: dict[int, set[str] | None] = {}

    def targets(version: int) -> set[str] | None:
        if version not in targets_by_version:
            if codebook.path is None or not codebook.log_path.exists():
                raise ValueError("on_version_change='relabel_touched' reads the codebook's "
                                 "revision log — this codebook has no path or no "
                                 "<name>.codebook.log.jsonl beside it; use 'relabel' or 'keep'")
            targets_by_version[version] = relabel_targets(
                _revision_events(codebook.path, version, codebook.version))
        return targets_by_version[version]

    def eligible(unit_key: tuple[str, int]) -> bool:
        current = latest.get(unit_key)
        if current is None:
            return True
        category, version = current
        if version >= codebook.version or on_version_change == "keep":
            return False
        if on_version_change == "relabel":
            return True
        touched = targets(version)
        return touched is None or category in touched

    return eligible


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")


def check_table_name(name: str) -> str:
    """Table names come from config and are spliced into SQL as identifiers, so they
    must look like one: `name` or `schema.name`, letters/digits/underscores only."""
    if not isinstance(name, str) or not _IDENTIFIER.match(name):
        raise ValueError(f"table name {name!r} is not a plain identifier (letters, digits, "
                         f"underscores; optionally schema.name)")
    return name
