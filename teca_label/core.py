"""Teca Label: stable, versioned semantic classification over unstructured data."""
import json
import re
import threading
import unicodedata
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, create_model

from ._version import __version__
from .providers import Provider, as_provider, is_configuration_error
from .providers import parse as _parse

DEFAULT_MODELS = {"draft": "claude-opus-5", "classify": "claude-opus-5", "extract": "claude-opus-5"}
DEFAULT_POLICY = {"other_threshold": 0.1, "min_batch": 20, "max_adds_per_revision": 3,
                  "decay_periods": 3, "decay_min_share": 0.01, "audit_every": 4,
                  "min_agreement": 0.8}   # a classify model must reach this on the benchmark
SCHEMA = 1   # the on-disk format (FORMAT.md); bumped only for changes an older reader can't ignore
KINDS = ("partition", "tag")
RESERVED_LABELS = ("other", "unclassifiable")   # the label contract's own values
MIN_TRACE_CHARS = 50   # under this much prose a trace can't carry a theme
NAME_MAX_CHARS = 40
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def slug(name: str) -> str:
    """A category name's canonical form: snake_case ASCII, at most NAME_MAX_CHARS.
    'Rate-limit cascade' -> 'rate_limit_cascade'. Names become SQL column
    values and dict keys everywhere downstream, so a model's prose title is normalized
    here, at draft time, and never accepted as-is."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    parts = re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")
    return parts[:NAME_MAX_CHARS].rstrip("_")


def _check_name(name: str) -> None:
    if name in RESERVED_LABELS:
        raise ValueError(f"'{name}' is reserved by the label contract and cannot be a category")
    if not _NAME.match(name) or len(name) > NAME_MAX_CHARS:
        hint = slug(name)
        raise ValueError(f"category names are snake_case ASCII, at most {NAME_MAX_CHARS} chars "
                         f"— got '{name}'" + (f"; try '{hint}'" if hint else ""))


class Category(BaseModel):
    name: str
    definition: str
    created_v: int = 1
    deprecated_v: int | None = None


class Op(BaseModel):
    """One typed change to a codebook. Field use per op kind:
    `name` is the target category (the source for rename/split, the new category for add);
    `new_name` is the result of a rename/merge; `definition` serves add/redefine/merge;
    `names` lists merge sources; `into` lists split results; `evidence` cites the
    trace indices that motivated the op."""
    op: Literal["add", "rename", "redefine", "merge", "split", "deprecate"]
    name: str = ""
    new_name: str = ""
    definition: str = ""
    names: list[str] = []
    into: list[Category] = []
    evidence: list[int] = []


class Revision(BaseModel):
    ops: list[Op]
    periods: list[str] = []   # the story periods whose signals motivated it, when known


class InvalidRevision(ValueError):
    """A proposed revision references categories that don't exist or would collide."""


def _validate_ops(categories: list[Category], ops: list[Op],
                  retired_names: set[str] = frozenset()) -> None:
    """Refuse a revision whole if any op is malformed. Every name a codebook has
    ever used — active, deprecated, or renamed away (`retired_names`, read from
    the log) — stays reserved forever, so historical labels never become ambiguous."""
    reserved_names = {category.name for category in categories} | set(retired_names)
    active_names = {category.name for category in categories if category.deprecated_v is None}
    problems: list[str] = []
    for position, op in enumerate(ops):
        def need(condition: bool, message: str):
            if not condition:
                problems.append(f"op[{position}] {op.op}: {message}")

        def need_new(name: str):
            # A new name must be well-formed and must not collide with any name ever used —
            # compared case-insensitively, since labels end up in SQL and in people's mouths.
            try:
                _check_name(name)
            except ValueError as bad:
                need(False, str(bad))
            taken = {existing.lower() for existing in reserved_names}
            need(name.lower() not in taken, f"'{name}' already exists")

        def need_active(name: str, role: str = ""):
            # An op can only touch a live category: acting on a deprecated one would bump
            # the version and log a change that changes nothing.
            if name not in reserved_names:
                need(False, f"{role}'{name}' does not exist")
            elif name not in active_names:
                need(False, f"{role}'{name}' is already deprecated")
        if op.op == "add":
            need(bool(op.name), "missing name")
            need(bool(op.definition), "missing definition")
            need_new(op.name)
            reserved_names.add(op.name)
            active_names.add(op.name)
        elif op.op == "rename":
            need_active(op.name)
            need(bool(op.new_name), "missing new_name")
            need_new(op.new_name)
            active_names.discard(op.name)      # the old name stays reserved forever
            reserved_names.add(op.new_name)
            active_names.add(op.new_name)
        elif op.op == "redefine":
            need_active(op.name)
            need(bool(op.definition), "missing definition")
        elif op.op == "deprecate":
            need_active(op.name)
            active_names.discard(op.name)
        elif op.op == "merge":
            need(bool(op.names), "missing source names")
            need(bool(op.new_name), "missing new_name")
            need(bool(op.definition), "missing definition")
            for source_name in op.names:
                need_active(source_name, role="source ")
                active_names.discard(source_name)
            need_new(op.new_name)
            reserved_names.add(op.new_name)
            active_names.add(op.new_name)
        elif op.op == "split":
            need_active(op.name)
            active_names.discard(op.name)
            need(bool(op.into), "missing 'into' categories")
            for result_category in op.into:
                need_new(result_category.name)
                reserved_names.add(result_category.name)
                active_names.add(result_category.name)
    if problems:
        raise InvalidRevision("; ".join(problems))


def read_log(log_path: Path | None) -> list[dict]:
    """The events in a codebook log, oldest first; blank lines tolerated; [] when absent."""
    if not (log_path and Path(log_path).exists()):
        return []
    return [json.loads(line) for line in Path(log_path).read_text().splitlines() if line.strip()]


def _revision_events(codebook_path: Path, after_version: int, upto_version: int) -> list[dict]:
    """A parent codebook's revision log entries between a child's pinned version and the live one."""
    return [event for event in read_log(codebook_path.with_suffix(".log.jsonl"))
            if event["event"] == "revision" and after_version < event["version"] <= upto_version]


def relabel_targets(revision_events: list[dict]) -> set[str] | None:
    """Which current labels a revision could move, given the revision log entries
    between a row's version and the live one. None means every label.

    Any op that changes a boundary (add, redefine, split, merge — each carries a
    definition) can pull rows out of ANY category, not just 'other': a new
    'rate_limit_cascade' category takes rows that were sitting under 'tool_loop'.
    Relabeling only the escape valve after such a change would leave old rows
    under the old label while new rows get the new one — an artificial trend. So
    a boundary change relabels everything below the live version. Only rename
    and deprecate are local: rename is a mechanical migration of one category's
    rows, and deprecating a category can only move the rows that were in it."""
    targets: set[str] = set()
    for event in revision_events:
        for op in event.get("ops", []):
            if op["op"] in ("add", "redefine", "split", "merge"):
                return None
            targets.add(op["name"])   # rename, deprecate
    return targets


def _project_parent(revision_events: list[dict], categories: list[str]) -> tuple[list[dict], list[str]]:
    """Project parent revision ops onto the categories a child codebook reads from.

    Each op that touched a followed category routes to its consequence for the child:
    rename -> follow (auto: same population, new name), redefine -> review (the
    population's boundary moved), split -> choose (the population was cut up),
    merge -> review (it widened), deprecate -> archive (the feed dries up).
    Returns (touched ops, the followed names with renames already applied)."""
    tracked_categories, touched_ops = list(categories), []
    for event in revision_events:
        for op in event.get("ops", []):
            hits = []
            if op["op"] == "rename" and op["name"] in tracked_categories:
                hits.append({"name": op["name"], "action": "follow", "auto": True,
                             "to": op["new_name"]})
                tracked_categories[tracked_categories.index(op["name"])] = op["new_name"]
            elif op["op"] == "redefine" and op["name"] in tracked_categories:
                hits.append({"name": op["name"], "action": "review"})
            elif op["op"] == "split" and op["name"] in tracked_categories:
                hits.append({"name": op["name"], "action": "choose",
                             "options": [result["name"] for result in op.get("into", [])]})
            elif op["op"] == "merge":
                hits += [{"name": source, "action": "review", "widened_to": op["new_name"]}
                         for source in op.get("names", []) if source in tracked_categories]
            elif op["op"] == "deprecate" and op["name"] in tracked_categories:
                hits.append({"name": op["name"], "action": "archive"})
            touched_ops += [{"version": event["version"], "op": op["op"], **hit} for hit in hits]
    return touched_ops, tracked_categories


def content(trace: dict) -> dict:
    """What the model reads. An envelope ({"id", "ts", "trace", ...} — see
    ingest.envelope) contributes only its `trace`; any other dict is read whole. The
    id, timestamp, and summary are for joining and scanning, never for judging."""
    if isinstance(trace, dict) and "trace" in trace and "id" in trace and "ts" in trace:
        inner = trace["trace"]
        return inner if isinstance(inner, dict) else {"text": str(inner)}
    return trace


def text_chars(value) -> int:
    """How much prose a trace holds: the characters of its string leaves, recursively.
    The model reads text; a record whose fields are all blank has nothing to be labeled."""
    if isinstance(value, dict) and "trace" in value and "id" in value and "ts" in value:
        value = content(value)
    if isinstance(value, str):
        return len(value.strip())
    if isinstance(value, dict):
        return sum(text_chars(field) for field in value.values())
    if isinstance(value, (list, tuple)):
        return sum(text_chars(item) for item in value)
    return 0


def thin_traces(traces: list[dict], min_chars: int = MIN_TRACE_CHARS) -> str | None:
    """The one-line warning for traces too thin to carry a theme, or None when every
    trace has at least `min_chars` of prose. Names the fields that are blank in all of
    them — the fastest way to spot a format change upstream."""
    thin = [content(trace) for trace in traces if text_chars(trace) < min_chars]
    if not thin:
        return None
    blank_fields = [key for key in thin[0] if isinstance(thin[0], dict)
                    and all(isinstance(t, dict) and text_chars(t.get(key)) == 0 for t in thin)]
    detail = f" ({', '.join(blank_fields)} all empty)" if blank_fields else ""
    return f"{len(thin)} of {len(traces)} traces have under {min_chars} chars of text{detail}"


def _check_schema(saved: dict, path) -> None:
    """A codebook file names the format it was written in. Missing means schema 1 (files
    from before the field existed); newer than this library means upgrade, not guess."""
    if not isinstance(saved, dict) or "categories" not in saved or "question" not in saved:
        raise ValueError(f"{path} is not a codebook: expected a JSON object with 'question' "
                         f"and 'categories' (see FORMAT.md)")
    found = saved.get("schema", 1)
    if found != SCHEMA:
        raise ValueError(f"{path} is codebook schema {found}; this teca-label reads schema {SCHEMA} "
                         f"— upgrade teca-label" if found > SCHEMA else
                         f"{path} is codebook schema {found}; this teca-label reads schema {SCHEMA}")


def _check_traces(traces) -> None:
    """A trace is a dict — the model reads its fields. Anything else fails here, before a
    call is spent, instead of as a TypeError deep inside propose()."""
    for position, trace in enumerate(traces):
        if not isinstance(trace, dict):
            raise TypeError(f"traces[{position}] is {type(trace).__name__}, not a dict — wrap "
                            f"plain text as {{'text': ...}} (to_trace=lambda body: {{'text': body}})")


def _sample(seq: list, n: int) -> list:
    """n items spread evenly across a (time-ordered) sequence: the first of each of n
    equal buckets, so 60 of 100 reach the end of the era rather than stopping at row 60."""
    if n <= 0 or len(seq) <= n:
        return list(seq)
    return [seq[i * len(seq) // n] for i in range(n)]


def _normalize_names(revision: Revision) -> Revision:
    """Slug every NEW name a model proposed (adds, rename/merge targets, split results),
    so a draft's prose titles land as column-safe identifiers before validation."""
    for op in revision.ops:
        if op.op == "add":
            op.name = slug(op.name)
        elif op.op in ("rename", "merge"):
            op.new_name = slug(op.new_name)
        elif op.op == "split":
            for result in op.into:
                result.name = slug(result.name)
    return revision


def _apply_ops(categories: list[Category], ops: list[Op], version: int) -> list[Category]:
    """Fold validated ops into a new category list. Pure: the input objects are never
    touched; deprecation marks, never deletes."""
    by_name = {category.name: category.model_copy() for category in categories}
    for op in ops:
        if op.op == "add":
            by_name[op.name] = Category(name=op.name, definition=op.definition, created_v=version)
        elif op.op == "rename":   # in place: the list order is the order the model sees
            by_name = {(op.new_name if name == op.name else name):
                       (category.model_copy(update={"name": op.new_name}) if name == op.name else category)
                       for name, category in by_name.items()}
        elif op.op == "redefine":
            by_name[op.name].definition = op.definition
        elif op.op == "deprecate":
            by_name[op.name].deprecated_v = version
        elif op.op == "merge":
            for source_name in op.names:
                by_name[source_name].deprecated_v = version
            by_name[op.new_name] = Category(name=op.new_name, definition=op.definition,
                                        created_v=version)
        elif op.op == "split":
            by_name[op.name].deprecated_v = version
            for result in op.into:
                by_name[result.name] = Category(name=result.name, definition=result.definition,
                                            created_v=version)
    return list(by_name.values())


def _coerce_dt(value) -> datetime:
    """Coerce to an aware UTC datetime; naive timestamps are assumed to already be UTC."""
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _period_of(dt: datetime, cadence: str) -> str:
    if cadence == "day":
        return dt.strftime("%Y-%m-%d")
    if cadence == "week":
        return dt.strftime("%G-W%V")
    if cadence == "month":
        return dt.strftime("%Y-%m")
    if cadence == "quarter":
        return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
    raise ValueError(f"cadence must be day|week|month|quarter, got '{cadence}'")


def decays(story: list[tuple[str, Counter]], active_categories: list[str],
           periods: int = 3, min_share: float = 0.01) -> list[dict]:
    """The free drift detector for death. Every other detector fires on arrival; nothing
    notices a category quietly going to zero and sitting as dead weight. Flags active categories whose share
    stayed under min_share for the last `periods` consecutive periods. Zero API calls."""
    if len(story) < periods:
        return []
    tail = story[-periods:]
    findings = []
    for category in active_categories:
        shares = []
        for _, counts in tail:
            total = sum(count for name, count in counts.items()
                        if name and name != "unclassifiable") or 1
            shares.append(counts.get(category, 0) / total)
        if all(share <= min_share for share in shares):
            findings.append({"category": category, "period": tail[-1][0],
                             "shares": [round(share, 3) for share in shares]})
    return findings


def drift_signals(story: list[tuple[str, Counter]], active_categories: list[str],
                  policy: dict, kind: str = "partition") -> list[dict]:
    """The free tier of the drift cascade, as one pure seam: reads the period story and
    emits typed signals. 'emergence' = the latest period's other-share cleared the
    threshold (themes fitting nothing); 'absorption' = a category's share jumped in the
    latest period (a theme hiding inside it); 'decay' = a category flatlined (retirement
    candidate). Detection is automatic; every signal still goes through propose/review.

    For kind='tag', emergence and absorption are off: 'other' is the tag's normal
    "no" answer, and a share move is the tag's *reading*, not a symptom. Decay stays.

    Absorption compares the latest share against the MEAN of prior periods — a
    single-period ratio misses gradual creep, which is how absorption actually
    arrives: a new theme seeps into an old category a few rows per period."""
    if not story:
        return []
    if kind == "tag":
        return [{"kind": "decay", **finding} for finding in decays(
            story, active_categories, policy["decay_periods"], policy["decay_min_share"])]

    def judged_total(counts: Counter) -> int:
        return sum(count for name, count in counts.items()
                   if name and name != "unclassifiable") or 1

    signals = []
    latest_period, latest_counts = story[-1]
    latest_total = judged_total(latest_counts)
    other_share = latest_counts.get("other", 0) / latest_total
    if other_share > policy["other_threshold"]:
        signals.append({"kind": "emergence", "period": latest_period,
                        "other_share": round(other_share, 3)})
    if len(story) >= 2:
        prior_shares_by_category: dict[str, list[float]] = {}
        for _, counts in story[:-1]:
            period_total = judged_total(counts)
            for category in active_categories:
                prior_shares_by_category.setdefault(category, []).append(
                    counts.get(category, 0) / period_total)
        for category in active_categories:
            share = latest_counts.get(category, 0) / latest_total
            # the baseline starts where the category does: periods before it first
            # appeared say nothing about what could be hiding inside it, and a category
            # a revision just created has no baseline at all
            prior = prior_shares_by_category[category]
            first_seen = next((i for i, prior_share in enumerate(prior) if prior_share > 0), None)
            if first_seen is None:
                continue
            prior = prior[first_seen:]
            baseline = sum(prior) / len(prior)
            delta = share - baseline
            if delta >= 0.10 or (baseline > 0 and share / baseline >= 1.5 and delta >= 0.03):
                signals.append({"kind": "absorption", "category": category, "period": latest_period,
                                "share": round(share, 3), "prev_share": round(baseline, 3)})
    for finding in decays(story, active_categories, policy["decay_periods"],
                          policy["decay_min_share"]):
        signals.append({"kind": "decay", **finding})
    return signals


class Codebook:
    """A versioned instrument: question + categories + per-role models, persisted with an
    append-only log.

    `parent` states this codebook's population when it isn't the whole corpus:
    {"path", "categories", "version", "pinned_at"} — the parent codebook, the
    categories whose rows this one reads, pinned at the parent version that defined
    them; None means root. `draft_window`
    ({"start", "end", "n_sampled"}) records the era the categories were drafted from.
    `client` is BYO: a teca-label Provider, or an anthropic.Anthropic / openai.OpenAI client
    (key, base_url, proxy); None routes each call by its model's family. `usage`
    accumulates tokens/calls across this instrument's operations."""

    def __init__(self, question: str, categories: list[Category] | None = None, version: int = 0,
                 models: dict | None = None, path: str | Path | None = None,
                 policy: dict | None = None,
                 parent: dict | None = None, client: Provider | None = None,
                 draft_window: dict | None = None, kind: str = "partition"):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        self.kind = kind
        self.question = question
        self.categories = list(categories or [])
        for category in self.categories:
            _check_name(category.name)
        lowered = [category.name.lower() for category in self.categories]
        duplicates = sorted({name for name in lowered if lowered.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate category names {duplicates} — every name in a "
                             f"codebook is unique (compared case-insensitively)")
        if kind == "tag" and len([c for c in self.categories if c.deprecated_v is None]) > 1:
            raise ValueError("a tag has exactly one active category")
        self.version = version
        self.models = {**DEFAULT_MODELS, **(models or {})}
        unknown = set(policy or {}) - set(DEFAULT_POLICY)
        if unknown:
            raise ValueError(f"unknown policy keys {sorted(unknown)} — known: {sorted(DEFAULT_POLICY)}")
        self.policy = {**DEFAULT_POLICY, **(policy or {})}
        self.parent = parent
        self.draft_window = draft_window
        self.path = Path(path) if path else None
        self.provider = as_provider(client) if client is not None else None
        self.usage = Counter()
        self.last_errors: list[str] = []
        self._usage_lock = threading.Lock()
        # the rows this codebook was drafted from, with their labels — see label_sample()
        self.sample: list[dict] = []
        self.sample_labels: list[str | None] = []
        self.sample_version: int | None = None

    # ---------- persistence ----------
    @property
    def log_path(self) -> Path | None:
        return self.path.with_suffix(".log.jsonl") if self.path else None

    @property
    def sample_path(self) -> Path | None:
        return self.path.with_suffix(".sample.jsonl") if self.path else None

    @property
    def bench_path(self) -> Path | None:
        return self.path.with_suffix(".bench.jsonl") if self.path else None

    def _save_sample(self) -> None:
        if self.sample_path:
            self.sample_path.write_text("".join(
                json.dumps({"label": label, "version": self.sample_version, "trace": trace},
                           default=str) + "\n"
                for trace, label in zip(self.sample, self.sample_labels)))

    def _load_sample(self) -> None:
        if not (self.sample_path and self.sample_path.exists()):
            return
        rows = [json.loads(line) for line in self.sample_path.read_text().splitlines() if line]
        self.sample = [row["trace"] for row in rows]
        self.sample_labels = [row["label"] for row in rows]
        self.sample_version = rows[0]["version"] if rows else None

    def save(self) -> "Codebook":
        if self.path:
            self.path.write_text(json.dumps({
                "schema": SCHEMA,
                "question": self.question, "version": self.version, "models": self.models,
                "kind": self.kind, "policy": self.policy,
                **({"parent": self.parent} if self.parent else {}),
                **({"draft_window": self.draft_window} if self.draft_window else {}),
                "categories": [c.model_dump() for c in self.categories]}, indent=2))
        return self

    @classmethod
    def load(cls, path: str | Path, client: Provider | None = None) -> "Codebook":
        """Reopen a saved instrument."""
        saved = json.loads(Path(path).read_text())
        _check_schema(saved, path)
        missing = [key for key in ("question", "version", "categories") if key not in saved]
        if missing:
            raise ValueError(f"{path} is missing {missing} — a codebook file needs question, "
                             f"version, and categories (see FORMAT.md)")
        policy = saved.get("policy") or {}
        unknown = sorted(set(policy) - set(DEFAULT_POLICY))
        if unknown:   # a later library's knobs: read the file anyway, as the format promises
            warnings.warn(f"{path}: ignoring unknown policy keys {unknown}", stacklevel=2)
            policy = {key: value for key, value in policy.items() if key in DEFAULT_POLICY}
        cb = cls(saved["question"], [Category(**category) for category in saved["categories"]],
                 saved["version"], saved.get("models"), path,
                 policy, saved.get("parent"),
                 client=client, draft_window=saved.get("draft_window"),
                 kind=saved.get("kind", "partition"))
        cb._load_sample()
        return cb

    def _log(self, event: str, **extra):
        if self.log_path:
            with self.log_path.open("a") as f:
                f.write(json.dumps({"schema": SCHEMA, "event": event, "version": self.version,
                                    "time": datetime.now(timezone.utc).isoformat(),
                                    "models": self.models, "library": __version__,
                                    "categories": [c.model_dump() for c in self.categories], **extra},
                                   default=str) + "\n")

    def _call(self, role: str, system: str, content: str, schema: type,
              max_tokens: int = 2048, timeout: float = 60.0, model: str | None = None):
        """Every model call goes through here: role -> model, BYO provider, usage accounting."""
        parsed, usage = _parse(model or self.models[role], system, content, schema,
                               max_tokens, timeout, provider=self.provider)
        with self._usage_lock:
            self.usage.update({**usage, "calls": 1})
        return parsed

    # ---------- lifecycle ----------
    def active(self) -> list[Category]:
        return [c for c in self.categories if c.deprecated_v is None]

    @classmethod
    def plan(cls, traces: list[dict], question: str, models: dict | None = None,
             sample: int = 60, min_evidence: int = 3,
             path: str | Path | None = "teca-label.codebook.json",
             client: Provider | None = None,
             window=None, time_key=None, rows: int | None = None,
             parent: dict | None = None, min_chars: int = MIN_TRACE_CHARS,
             allow_empty: bool = False, label_sample: bool = True,
             prices: dict | None = None, policy: dict | None = None):
        """What a build would do, before it does it — no API call. Print the plan to
        see the rows in scope, the fields the model will read, the sample, the model
        per role, where the codebook lands, and the estimated cost of the draft, of
        labeling the sample, and of labeling every row. `plan.build()` is the only
        way to draft a codebook from a corpus. drill() drafts a child inside one
        category through the same propose/apply path, without this gate.

        The codebook keeps the sample it drafted from: it labels those rows under v1
        (`sample` classify calls) and saves them beside the file as
        `<name>.codebook.sample.jsonl`, so show() and exemplars() run
        against the same rows every time without re-fetching or re-labeling.
        `label_sample=False` skips that pass.

        Traces with under `min_chars` of prose refuse the build — blank records drafted
        from silently become 'other', and a format change upstream hides behind them.
        Pass allow_empty=True to draft with them anyway, or filter them out first.

        `window` scopes the draft to an era — "2026-04-20.." / "..2026-06-01" /
        "2026-04-20..2026-06-01", or (start, end) with either bound None — read from
        `time_key` (defaults to the envelope's 'ts'). Categories describe current
        problems, not stale ones, and the window is recorded in the codebook file as
        provenance; older records classified against it land in 'other' honestly.

        `rows` is how many rows the codebook will eventually label when `traces` is
        only a sample of them (fetch(sample=60) hands you 60; the table holds 48,000)
        — it sizes the full-labeling estimate and nothing else.

        `policy` sets the codebook's thresholds at birth (other_threshold, min_batch,
        max_adds_per_revision, decay_periods, decay_min_share, audit_every,
        min_agreement); unknown keys are refused here, before any call.

        `parent` records the population this codebook studies when it isn't everything:
        {"path", "categories", "version"} — another codebook's labels, pinned at the
        version that defined them. The caller filters the rows; the parent record is
        the durable statement of what the filter was."""
        from .plan import Plan, scope
        in_scope, bounds, time_key = scope(traces, window, time_key, min_evidence)
        return Plan(cls, question, in_scope, sample, models, path, min_evidence=min_evidence,
                    min_chars=min_chars, allow_empty=allow_empty, label_sample=label_sample,
                    parent=parent, window=bounds, time_key=time_key, client=client, rows=rows,
                    n_seen=len(traces), prices=prices, policy=policy)

    def label_sample(self, traces: list[dict] | None = None, workers: int = 8) -> list[str | None]:
        """Label the codebook's sample under the current version and keep it — the
        rows show() and exemplars() read. build() does this itself; call it
        with traces after adopt()/tag(), or with none to refresh after a revision.
        Persisted as `<name>.codebook.sample.jsonl` when the codebook has a path."""
        if traces is not None:
            _check_traces(traces)
            self.sample = list(traces)
        if not self.sample:
            raise ValueError("no sample to label — pass traces (build() keeps its own)")
        self.sample_labels = self.classify(self.sample, workers=workers)
        self.sample_version = self.version
        self._save_sample()
        return self.sample_labels

    def _fresh_sample(self, workers: int = 8) -> tuple[list[dict], list[str | None]]:
        """The sample with labels under the current version, relabeling if a revision
        moved past them."""
        if not self.sample:
            raise ValueError("this codebook has no sample — build() keeps one; after "
                             "adopt()/tag(), call label_sample(traces) first")
        if self.sample_version != self.version:
            self.label_sample(workers=workers)
        return self.sample, self.sample_labels

    def show(self, workers: int = 8) -> str:
        """The codebook as a markdown table over its sample — category, definition, rows —
        with the checks a reviewer runs every time, spelled out beneath it: categories
        with no rows (usually the wrong shape: a tag, not a partition category), too many
        categories for the rows (tail categories rest on 2-3 examples), and an other-share
        over the policy threshold. Relabels the sample first if a revision moved past it."""
        traces, labels = self._fresh_sample(workers)
        counts = Counter(labels)
        n = len(traces)
        lines = [f"| category | definition | n (of {n}) |", "|---|---|---|"]
        lines += [f"| `{c.name}` | {c.definition} | {counts.get(c.name, 0)} |" for c in self.active()]
        lines.append(f"| `other` | — | {counts.get('other', 0)} |")
        judged = [label for label in labels if label not in (None, "unclassifiable")]
        other_share = counts.get("other", 0) / max(len(judged), 1)
        summary = f"v{self.version} · {n} rows · other {other_share:.0%}"
        if self.kind == "partition":
            summary += f" (threshold {self.policy['other_threshold']:.0%})"
        lines += ["", summary]
        for note in self._sample_checks(counts, len(judged), other_share):
            lines.append(f"- ⚠ {note}")
        if counts.get(None):
            lines.append(f"- {counts[None]} calls failed — see last_errors; label_sample() retries")
        return "\n".join(lines)

    def _sample_checks(self, counts: Counter, judged: int, other_share: float) -> list[str]:
        notes = []
        empty = [c.name for c in self.active() if counts.get(c.name, 0) == 0]
        if empty and self.kind == "partition":
            names = ", ".join(f"`{name}`" for name in empty)
            notes.append(f"{names}: 0 of {judged} rows — real but never a row's main thing? "
                         f"that's a tag, not a partition category (Codebook.tag), or deprecate it")
        active = len(self.active())
        if self.kind == "partition" and active > 1 and judged and judged / active < 15:
            notes.append(f"{active} categories on {judged} rows (1:{judged // active}, under 1:15) — "
                         f"tail categories rest on a few examples; merge some, or build with more rows")
        if self.kind == "partition" and other_share > self.policy["other_threshold"]:
            notes.append(f"other {other_share:.0%} is over the {self.policy['other_threshold']:.0%} "
                         f"threshold — either the query includes rows the question isn't about (filter "
                         f"them out in SQL) or the codebook doesn't fit; propose() to revise")
        return notes

    def _label_schema(self):
        defs = json.dumps([c.model_dump(include={"name", "definition"}) for c in self.active()])
        prompt = (f"Assign the single best-fitting category from this codebook ('other' if none "
                  f"fit): {defs}\nQuote the shortest verbatim passage of the record that "
                  f"supports your label as evidence (under 200 characters); leave evidence "
                  f"empty for 'other'.")
        return prompt, create_model(
            "Label",
            label=(Literal[tuple(c.name for c in self.active()) + ("other",)], ...),
            evidence=(str, Field(description="verbatim quote from the record supporting the "
                                             "label, under 200 characters; empty for 'other'")))

    def judge(self, traces: list[dict], workers: int = 8, model: str | None = None,
              retries: int = 1) -> list[tuple[str, str | None] | None]:
        """Label each trace and quote the evidence: one (label, evidence) pair per trace,
        or None when every attempt failed (a recorded gap, not a crash — errors land in
        self.last_errors and a rerun fills the gap). The label follows the contract:
        an active category name | 'other' (fits nothing — the drift signal) |
        'unclassifiable' (nothing to read — no API call). Evidence is a short verbatim
        quote, or None. A configuration error (no key, unknown model, rejected
        credential) raises immediately instead of becoming a gap per row."""
        model = model or self.models["classify"]
        prompt, Label = self._label_schema()
        self.last_errors = []
        _check_traces(traces)

        def judge_one(trace):
            if not trace or text_chars(trace) == 0:
                return ("unclassifiable", None)
            for _ in range(max(retries, 0) + 1):
                try:
                    answer = self._call("classify", prompt, json.dumps(content(trace), default=str),
                                        Label, model=model)
                    return (answer.label, answer.evidence.strip() or None)
                except Exception as exc:
                    if is_configuration_error(exc):
                        raise
                    last_error = exc
            self.last_errors.append(f"{type(last_error).__name__}: {last_error}")
            return None

        with ThreadPoolExecutor(workers) as pool:
            return list(pool.map(judge_one, traces))

    def classify(self, traces: list[dict], workers: int = 8, model: str | None = None,
                 retries: int = 1) -> list[str | None]:
        """judge(), labels only: one label per trace (see judge() for the contract)."""
        return [judgment[0] if judgment else None
                for judgment in self.judge(traces, workers=workers, model=model, retries=retries)]

    @property
    def pending_path(self) -> Path | None:
        return self.path.with_suffix(".pending.json") if self.path else None

    def propose(self, traces: list[dict], labels: list[str] | None = None,
                min_evidence: int = 3, signals: list[dict] | None = None) -> Revision:
        """Propose a policy-capped revision from the drift evidence.

        Without `signals`, evidence is the misfit ('other') traces — except for
        tags, whose 'other' means "no" and is never misfit evidence. With `signals`
        (from drift_signals()), the evidence widens to match each signal's nature:
        absorption signals put the flagged category's own rows in front of the reviser
        (the rows a hidden theme is disguised among — misfits alone can't reveal it),
        and decay signals become deterministic deprecate ops with no model involved.
        When the codebook has a path, the proposal is also written to
        `<name>.codebook.pending.json` — the review gate: inspect or edit it, then call
        apply() with no argument to accept it. An unreviewed proposal is never
        overwritten: apply or delete it before proposing again."""
        _check_traces(traces)
        if self.pending_path and self.pending_path.exists():
            raise FileExistsError(f"{self.pending_path} already holds an unreviewed proposal — "
                                  f"apply() it or delete it before proposing again")
        if labels is None:
            labels = self.classify(traces) if self.categories else ["other"] * len(traces)
        signals = signals or []
        active_names = {category.name for category in self.active()}
        decayed_categories = [signal["category"] for signal in signals
                         if signal["kind"] == "decay" and signal["category"] in active_names]
        misfit_indexes = ([] if self.kind == "tag"
                          else [i for i, label in enumerate(labels) if label == "other"])
        signal_indexes, signal_text = self._signal_evidence(signals, labels, min_evidence)
        evidence_indexes = list(dict.fromkeys(misfit_indexes + signal_indexes))
        if not evidence_indexes:
            return self._finish_proposal(Revision(ops=[]), decayed_categories, signals)
        revision = self._call(
            "draft", self._reviser_instructions(min_evidence, signal_text,
                                                n_rows=len(evidence_indexes)),
            json.dumps([{"i": i, "current_label": labels[i], **content(traces[i])}
                        for i in evidence_indexes], default=str), Revision, 8192,
            timeout=600.0)
        revision = self._enforce_policy(_normalize_names(revision))
        return self._finish_proposal(revision, decayed_categories, signals)

    def _reviser_instructions(self, min_evidence: int, signal_text: str,
                              n_rows: int = 60) -> str:
        """The reviser's system prompt, by situation: maintaining a tag (never grows),
        bootstrapping an empty book (only grows), or revising a partition (rationed)."""
        active_defs = json.dumps([category.model_dump(include={"name", "definition"})
                                  for category in self.active()])
        if self.kind == "tag":
            return (
                f"You maintain a single-category yes/no detector for the question: {self.question}\n"
                f"The category: {active_defs}\n"
                f"Propose the MINIMAL ops to keep the detector sharp, using ONLY rename, "
                f"redefine, or deprecate — a tag never gains categories. Prefer zero ops; redefine "
                f"only when the evidence rows show the definition's boundary is wrong."
                + signal_text)
        if not self.active():
            # the category band scales with what the sample can certify (n ≈ 6/prevalence):
            # 60 rows -> the classic 6-10; 34 rows -> 3-5, so a small population can't
            # be fragmented into a taxonomy of instances
            low = max(3, min(6, n_rows // 10))
            high = max(low + 2, min(10, n_rows // 6))
            return (
                f"You are drafting the initial codebook for the question: {self.question}\n"
                f"Propose ONLY 'add' ops: {low}-{high} mutually exclusive categories with sharp one-sentence "
                f"definitions, each backed by >={min_evidence} traces. Names are snake_case identifiers "
                f"(e.g. tool_loop), at most {NAME_MAX_CHARS} chars. Cite trace indices as evidence. "
                f"If a meaningful share of the records is irrelevant to the question, do NOT spend "
                f"categories on irrelevant material — leave it to 'other' and keep every category inside the "
                f"question's universe; a single category cannot gate and discriminate at once.")
        add_cap = self.policy["max_adds_per_revision"]
        return (
            f"You maintain the evolving codebook for the question: {self.question}\n"
            f"Current active categories: {active_defs}\n"
            f"Propose the MINIMAL ops (add/rename/redefine/merge/split/deprecate) so the codebook "
            f"fits the evidence rows below. STRONGLY prefer consolidating — redefine or merge "
            f"existing categories — over adding new ones; propose at most {add_cap} adds, and only for a "
            f"theme >={min_evidence} traces share that is clearly disjoint from EVERY active category. "
            f"Prefer zero ops. Keep categories mutually exclusive with sharp one-sentence definitions; "
            f"names are snake_case identifiers, at most {NAME_MAX_CHARS} chars. "
            f"Cite trace indices as evidence. Every name in an add or split must be NEW — never "
            f"reuse an existing category's name (to narrow a category while carving a theme out of it, "
            f"use redefine + add instead of split). Quality bar: records belong under the same "
            f"category only if they would lead to the SAME decision or fix — and never merge groups "
            f"that would route to different owners or trigger different responses, however small."
            + signal_text)

    _SIGNAL_FORMATTERS = {
        "absorption": lambda signal: (
            f"'{signal['category']}' share {signal['prev_share']:.0%} -> {signal['share']:.0%} — "
            f"a NEW theme may be hiding inside it in disguise"),
        "audit": lambda signal: (
            f"an audit of '{signal['category']}' suspects a hidden sub-theme: "
            f"\"{signal['hidden_theme']}\""),
    }

    def _signal_evidence(self, signals: list[dict], labels: list[str | None],
                         min_evidence: int) -> tuple[list[int], str]:
        """Map drift signals to (extra evidence row indices, reviser context text).

        Every signal kind shares one shape — include the flagged category's own rows
        and say why. New detector kinds join by adding a formatter — propose() itself
        never grows."""
        evidence_indexes, context_lines = [], []
        for signal in signals:
            formatter = self._SIGNAL_FORMATTERS.get(signal["kind"])
            if formatter is None:
                continue
            context_lines.append(formatter(signal))
            flagged_rows = [i for i, label in enumerate(labels)
                            if label == signal.get("category")]
            evidence_indexes += _sample(flagged_rows, 20)
        if not context_lines:
            return [], ""
        return evidence_indexes, (
            "\nDrift signals this period: " + "; ".join(context_lines) + ". Rows currently "
            "labeled with flagged categories are included below (see current_label); if "
            f">={min_evidence} of them share a distinct theme, propose a split or add for it.")

    def _finish_proposal(self, revision: Revision, decayed_categories: list[str],
                         signals: list[dict] = ()) -> Revision:
        """Append deterministic decay retirements (skipping categories other ops already touch),
        then persist the whole proposal to the pending file for review."""
        touched_names = ({op.name for op in revision.ops}
                         | {source for op in revision.ops for source in op.names})
        revision = Revision(
            ops=revision.ops + [Op(op="deprecate", name=category)
                                for category in decayed_categories if category not in touched_names],
            periods=sorted({signal["period"] for signal in signals if signal.get("period")}))
        if self.pending_path and revision.ops:
            self.pending_path.write_text(json.dumps(
                {"schema": SCHEMA, **revision.model_dump(exclude_defaults=True)}, indent=2))
        return revision

    def _enforce_policy(self, revision: Revision) -> Revision:
        """Hard-enforce the add cap: keep the best-evidenced adds, drop the rest.
        Tags lose every growth op — a tag that gains a category isn't a tag. The
        bootstrap draft (no active categories yet) is exempt from the cap."""
        if self.kind == "tag":
            return Revision(ops=[op for op in revision.ops
                                 if op.op in ("rename", "redefine", "deprecate")])
        if not self.active():
            return revision
        add_ops = [op for op in revision.ops if op.op == "add"]
        add_cap = self.policy["max_adds_per_revision"]
        if len(add_ops) <= add_cap:
            return revision
        best_evidenced = sorted(add_ops, key=lambda op: len(op.evidence), reverse=True)
        kept_add_ids = {id(op) for op in best_evidenced[:add_cap]}
        return Revision(ops=[op for op in revision.ops
                             if op.op != "add" or id(op) in kept_add_ids])

    def apply(self, revision: Revision | None = None) -> "Codebook":
        """Apply a revision. With no argument, applies (and consumes) the pending file —
        the reviewed-proposal path. Validates every op before mutating anything, and
        logs the provenance: 'pending' (passed the review gate) or 'direct'. The
        pending file is consumed only when it IS the revision just applied — a direct
        apply must never silently discard someone else's unreviewed proposal."""
        source = "direct"
        if revision is None:
            if not (self.pending_path and self.pending_path.exists()):
                raise FileNotFoundError("no pending revision to apply — propose() (or a "
                                        "`teca-label run` tick) writes one")
            revision = Revision.model_validate_json(self.pending_path.read_text())
            source = "pending"
        if not revision.ops:
            raise InvalidRevision("the revision has no ops — nothing to apply, so no version bump")
        _validate_ops(self.categories, revision.ops, self._retired_names())
        categories = _apply_ops(self.categories, revision.ops, self.version + 1)
        if self.kind == "tag" and len([c for c in categories if c.deprecated_v is None]) > 1:
            raise InvalidRevision("a tag has exactly one active category — a tag never gains one")
        self.version += 1
        self.categories = categories
        self.save()
        self._log("revision", ops=[op.model_dump() for op in revision.ops], source=source,
                  periods=revision.periods)
        if self.pending_path and self.pending_path.exists():
            try:
                pending = Revision.model_validate_json(self.pending_path.read_text())
            except Exception:
                pending = None
            if source == "pending" or pending == revision:
                self.pending_path.unlink()
        return self

    def _retired_names(self) -> set[str]:
        """Names renamed away in this codebook's history: gone from the file, still reserved."""
        if not (self.log_path and self.log_path.exists()):
            return set()
        return {op["name"] for event in read_log(self.log_path)
                if event["event"] == "revision" for op in event.get("ops", [])
                if op["op"] == "rename"}

    @classmethod
    def adopt(cls, question: str, categories: list[Category], path: str | Path,
              models: dict | None = None, version: int = 1,
              client: Provider | None = None,
              policy: dict | None = None) -> "Codebook":
        """Bring an externally drafted set of categories under library management.

        `policy` overrides land at birth (validated like any policy), so the first
        saved file and 'adopted' log entry reflect the instrument as configured —
        e.g. a single-category detector adopts with {"other_threshold": 1.0}, since
        'other' is its expected negative class, not drift."""
        cb = cls(question, categories, version, models, path, client=client, policy=policy)
        cb.save()
        cb._log("adopted")
        return cb

    @classmethod
    def tag(cls, name: str, definition: str, path: str | Path,
            question: str | None = None, models: dict | None = None,
            policy: dict | None = None,
            client: Provider | None = None) -> "Codebook":
        """A tag: a single-category yes/no detector. Same instrument, inverted escape valve —
        'other' here means "no", a normal answer, not drift. Emergence and absorption
        alarms are off by kind (there is nothing to emerge into); decay stays on (a tag
        at zero share for decay_periods proposes its own retirement); audit stays on
        (are the yes-rows really yeses). Its label column is a WHERE clause like any
        other codebook's — tags are the cheap population selectors of the DAG."""
        question = question or f"Does the record contain this? {definition}"
        cb = cls(question, [Category(name=name, definition=definition)], version=1,
                 models=models, path=path, client=client, policy=policy, kind="tag")
        cb.save()
        cb._log("adopted")
        return cb

    # ---------- temporal ----------
    def audit(self, traces: list[dict], labels: list[str | None], per_category: int = 15,
              workers: int = 8) -> list[dict]:
        """The second drift detector: absorption. The other-rate only catches units that fit
        NOTHING; a new theme that loosely resembles an old category hides inside it. audit() samples
        each category's own units and asks whether a distinct sub-theme is lurking. Findings with a
        hidden_theme are candidates for the next propose(). One call per active category."""
        class Finding(BaseModel):
            coherent: bool
            hidden_theme: str = ""
            note: str = ""

        rows_by_category: dict[str, list[dict]] = {}
        for trace, label in zip(traces, labels):
            if label and label not in ("other", "unclassifiable"):
                rows_by_category.setdefault(label, []).append(trace)

        def audit_one(category: Category) -> dict | None:
            own_rows = rows_by_category.get(category.name, [])
            if len(own_rows) < 5:
                return None
            sampled_rows = [content(row) for row in _sample(own_rows, per_category)]
            finding = self._call(
                "draft",
                f"You audit one category of a codebook for the question: {self.question}\n"
                f"Category '{category.name}': {category.definition}\n"
                f"Below are units currently labeled with this category. Judge: (1) do they cohere "
                f"under the definition? (2) is a DISTINCT recurring sub-theme hiding among them "
                f"that deserves its own category — would some of these units lead to a DIFFERENT "
                f"decision or fix than the rest? Be conservative — most categories are fine; name a "
                f"hidden_theme only if several units clearly share it.",
                json.dumps(sampled_rows, default=str), Finding, timeout=600.0)
            return {"category": category.name, "n_sampled": len(sampled_rows), **finding.model_dump()}

        with ThreadPoolExecutor(workers) as pool:
            findings = [finding for finding in pool.map(audit_one, self.active()) if finding]
        flagged = [finding for finding in findings
                   if not finding["coherent"] or finding["hidden_theme"]]
        self._log("audit", findings=flagged)
        return findings

    def staleness(self, recent_labels: list[str | None] | None = None) -> dict:
        """How stale is this instrument? Pass labels from a recent classify() for the drift gauge."""
        report = {"version": self.version,
                  "active_categories": len(self.active()), "last_revision_at": None, "revisions": 0}
        for event in read_log(self.log_path):
            if event["event"] == "revision":
                report["revisions"] += 1
                report["last_revision_at"] = event["time"]
        if recent_labels is not None:
            judged = [label for label in recent_labels if label not in (None, "unclassifiable")]
            rate = judged.count("other") / max(len(judged), 1)
            report["recent_other_rate"] = round(rate, 3)
            report["drifting"] = rate > self.policy["other_threshold"]
        if self.parent:
            report["parent"] = self._parent_report()
            report["parent_stale"] = report["parent"]["moved"]
            if report["parent_stale"]:
                undecided = [op for op in report["parent"].get("ops", []) if not op.get("auto")]
                report["note"] = ("parent moved: " + (
                    f"{len(undecided)} decision(s) needed — see 'parent' ops, then "
                    f"reparent(decisions=...)"
                    if undecided else "renames only — reparent() absorbs them"))
        return report

    def _parent_report(self) -> dict:
        """The parent's staleness: has it moved past the pinned version, and which of
        its ops touched the categories this codebook follows."""
        ref = self.parent
        entry = {"path": ref.get("path"), "categories": list(ref.get("categories", [])),
                 "pinned_version": ref.get("version"), "moved": False}
        parent_path = ref.get("path")
        if parent_path and Path(parent_path).exists():
            live_version = json.loads(Path(parent_path).read_text())["version"]
            entry["live_version"] = live_version
            entry["moved"] = live_version > ref["version"]
            if entry["moved"]:
                entry["ops"], _ = _project_parent(
                    _revision_events(Path(parent_path), ref["version"], live_version),
                    entry["categories"])
        return entry

    def reparent(self, decisions: dict[str, list[str]] | None = None) -> "Codebook":
        """Absorb the parent's movement and re-pin to its live version.

        Renames follow automatically. Every other op that touched a followed category is a
        human ruling — pass decisions={category: [categories to follow now]} ([] stops following
        it; [category] re-affirms it after a redefine). Refuses to re-pin past an undecided
        op, so a stale child stays flagged until someone rules. The ruling is logged."""
        decisions = decisions or {}
        ref = self.parent
        if not ref:
            return self
        parent_path = ref.get("path")
        if not (parent_path and Path(parent_path).exists()):
            return self
        live_version = json.loads(Path(parent_path).read_text())["version"]
        if live_version <= ref["version"]:
            return self
        ops, tracked_categories = _project_parent(
            _revision_events(Path(parent_path), ref["version"], live_version),
            ref.get("categories", []))
        undecided = sorted({op["name"] for op in ops if not op.get("auto")}
                           - set(decisions))
        if undecided:
            raise ValueError(f"parent ops need a ruling on {undecided} — pass "
                             f"decisions={{category: [categories to follow]}}; staleness() shows the menu")
        for op in ops:
            if op.get("auto") or op["name"] not in tracked_categories:
                continue
            position = tracked_categories.index(op["name"])
            other_categories = [category for category in tracked_categories if category != op["name"]]
            tracked_categories[position:position + 1] = [
                category for category in decisions[op["name"]] if category not in other_categories]
        ref["categories"], ref["version"] = tracked_categories, live_version
        ref["pinned_at"] = datetime.now(timezone.utc).isoformat()
        self.save()
        self._log("reparent", parent=self.parent, decisions=decisions)
        return self

    def drill(self, category: str, traces: list[dict], labels: list[str | None],
              question: str | None = None, path: str | Path | None = None,
              sample: int = 60, min_evidence: int = 3) -> "Codebook":
        """Go a level deeper inside one category: build a child codebook over only the rows
        labeled with it (the router pattern, as a first-class move).

        The child is a full instrument — its own file, versions, log — and records its
        parent: which codebook it reads from, which category, pinned at the version
        that defined it. A later parent revision changes the child's universe, so
        staleness() projects the parent's ops onto the followed categories and flags
        what each demands; reparent() absorbs the movement once ruled on."""
        by_name = {c.name: c for c in self.active()}
        if category not in by_name:
            raise ValueError(f"'{category}' is not an active category of this codebook")
        subset = [t for t, l in zip(traces, labels) if l == category]
        if len(subset) < min_evidence:
            raise ValueError(
                f"only {len(subset)} rows labeled '{category}' — not enough to drill into. "
                f"Bank more first: classify strided batches of the corpus until ~60 rows land "
                f"on '{category}' (rows to classify ≈ 60 / the category's share), then drill "
                f"from those. Labels written along the way persist — a later full sweep "
                f"skips them.")
        if len(subset) < 30:
            warnings.warn(
                f"{len(subset)} rows labeled '{category}' — enough to summarize, thin for a "
                f"standing child (the draft will be capped to a few categories). For a "
                f"trustworthy instrument, bank ~60+ rows of '{category}' first.",
                stacklevel=2)
        question = question or (
            f"Within '{category}' — {by_name[category].definition} — what distinct kinds are there? "
            f"(Parent question: {self.question})")
        if path is None and self.path:
            stem = self.path.name.removesuffix(".codebook.json")
            path = self.path.with_name(f"{stem}.{category}.codebook.json")
        child = type(self)(question, models=dict(self.models), path=path, policy=dict(self.policy),
                           parent={"path": str(self.path) if self.path else None,
                                   "categories": [category], "version": self.version,
                                   "pinned_at": datetime.now(timezone.utc).isoformat()},
                           client=self.provider)
        draft = child.propose(_sample(subset, sample), min_evidence=min_evidence)
        if not draft.ops:
            raise ValueError(f"the draft inside '{category}' produced no categories — the rows "
                             f"may not vary on the question; drill with more rows or a sharper question")
        child.apply(draft)
        return child

    # ---------- introspection ----------
    def exemplars(self, category: str, traces: list[dict] | None = None,
                  labels: list[str | None] | None = None,
                  n: int = 3, retest: int = 20, workers: int = 8) -> dict:
        """The clearest members of one category: rows whose label survives a repeat
        classification. Typicality is measured (test-retest agreement), never
        self-reported — a row that flips on re-ask is borderline by definition,
        whatever confidence the model would claim. Reads the codebook's sample unless
        `traces` and `labels` are given. Costs at most `retest` enum-locked classify
        calls, only when asked; nothing is persisted.

        Returns {"category", "n_labeled", "n_tested", "agreement", "exemplars"} —
        each exemplar {"index", "trace"}, in corpus order. Fewer than n stable
        rows come back as fewer exemplars, never padded with borderline ones."""
        if traces is None:
            traces, labels = self._fresh_sample(workers)
        elif labels is None:
            raise ValueError("pass labels with traces (or neither, to use the sample)")
        labeled_rows = [(i, trace) for i, (trace, label) in enumerate(zip(traces, labels))
                        if label == category]
        if not labeled_rows:
            raise ValueError(f"no rows labeled '{category}' — nothing to exemplify")
        retested = _sample(labeled_rows, retest)
        relabels = self.classify([trace for _, trace in retested], workers=workers)
        judged = [(row, relabel) for row, relabel in zip(retested, relabels)
                  if relabel is not None]
        held = [row for row, relabel in judged if relabel == category]
        return {"category": category, "n_labeled": len(labeled_rows), "n_tested": len(judged),
                "agreement": round(len(held) / max(len(judged), 1), 3),
                "exemplars": [{"index": i, "trace": trace} for i, trace in held[:n]]}

    # ---------- the instrument beyond the categories: model governance ----------
    def bench(self, traces: list[dict], labels: list[str]) -> "Codebook":
        """Keep a small human-labeled benchmark beside the codebook. Fixed definitions
        alone don't make two runs comparable — the model can change, by your choice or
        the vendor's — so every classify-model change is measured against these rows
        first (set_model), and the measurement goes in the log. Labels must be active
        category names or 'other'. Overwrites any earlier benchmark."""
        _check_traces(traces)
        if len(traces) != len(labels):
            raise ValueError(f"{len(traces)} traces but {len(labels)} labels")
        allowed = {category.name for category in self.active()} | {"other"}
        bad = sorted({label for label in labels if label not in allowed})
        if bad:
            raise ValueError(f"benchmark labels must be active categories or 'other' — got {bad}")
        if not self.bench_path:
            raise ValueError("the codebook needs a path to keep a benchmark")
        self.bench_path.write_text("".join(
            json.dumps({"trace": trace, "label": label}, default=str) + "\n"
            for trace, label in zip(traces, labels)))
        self._log("bench", n=len(traces))
        return self

    def _load_bench(self) -> tuple[list[dict], list[str]]:
        if not (self.bench_path and self.bench_path.exists()):
            raise FileNotFoundError(
                "no benchmark — call bench(traces, labels) with human-confirmed labels first "
                "(a reviewed sample from show() is the usual source)")
        rows = [json.loads(line) for line in self.bench_path.read_text().splitlines() if line]
        return [row["trace"] for row in rows], [row["label"] for row in rows]

    def measure(self, model: str | None = None, workers: int = 8) -> dict:
        """Agreement between a classify model (the current one by default) and the human
        benchmark: {model, n, agreement, disagreements}. One classify call per benchmark
        row; the result is logged. Failed calls are excluded from n."""
        traces, human = self._load_bench()
        model = model or self.models["classify"]
        answers = self.classify(traces, workers=workers, model=model)
        pairs = [(h, a) for h, a in zip(human, answers) if a is not None]
        disagreements = Counter(f"{h} ~ {a}" for h, a in pairs if h != a)
        report = {"model": model, "n": len(pairs),
                  "agreement": round(sum(h == a for h, a in pairs) / max(len(pairs), 1), 3),
                  "disagreements": dict(disagreements.most_common(10))}
        self._log("measure", **report)
        return report

    def set_model(self, role: str, model: str, workers: int = 8) -> dict | None:
        """Change the model behind a role, on the record. For 'classify' the candidate is
        measured against the benchmark first and refused below policy.min_agreement;
        the change is logged with its measurement, so a shift in the numbers after a
        model change is a change you can point to. Returns the measurement."""
        if role not in self.models:
            raise ValueError(f"role must be one of {sorted(self.models)}")
        measurement = None
        if role == "classify":
            measurement = self.measure(model, workers=workers)
            floor = self.policy["min_agreement"]
            if measurement["agreement"] < floor:
                raise ValueError(
                    f"{model} agrees with the benchmark on {measurement['agreement']:.0%} of "
                    f"{measurement['n']} rows, under policy.min_agreement={floor:.0%}; top "
                    f"disagreements: {measurement['disagreements']}")
        old, self.models[role] = self.models[role], model
        self.save()
        self._log("model_change", role=role, old=old, new=model, measurement=measurement)
        return measurement

    def history(self) -> list[str]:
        if not (self.log_path and self.log_path.exists()):
            return []
        lines = []
        for event in read_log(self.log_path):
            if event["event"] == "revision":
                summary = "; ".join(f"{op['op']} {op['name'] or '+'.join(op['names'])}"
                                    for op in event["ops"]) or "no change"
            elif event["event"] == "model_change":
                summary = (f"{event['role']}: {event['old']} -> {event['new']}"
                           + (f" (agreement {event['measurement']['agreement']:.0%} on "
                              f"{event['measurement']['n']})" if event.get("measurement") else ""))
            elif event["event"] == "measure":
                summary = f"{event['model']}: agreement {event['agreement']:.0%} on {event['n']}"
            elif event["event"] == "bench":
                summary = f"{event['n']} human-labeled rows"
            elif event["event"] == "audit":
                summary = ("flagged: " + "; ".join(
                    finding["category"] + (f" (hidden: {finding['hidden_theme']})"
                                       if finding["hidden_theme"] else "")
                    for finding in event["findings"])) if event["findings"] else "all categories coherent"
            else:
                summary = event["event"]
            lines.append(f"v{event['version']} {event['time'][:16]} [{event['event']}] {summary}")
        return lines

    def __repr__(self):
        categories = "\n".join(f"  {c.name}: {c.definition}" for c in self.active())
        return (f"Codebook(v{self.version}, {len(self.active())} categories, "
                f"models={self.models})\n{categories}")
