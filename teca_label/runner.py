"""The standing loop: one stateless tick per schedule, review by diff.

A Teca Label project is a directory (usually its own repo, like a dbt project):
`teca-label.toml` names each codebook — its file, warehouse query, cadence —
and a scheduler runs `teca-label run` on cron. A tick holds no state of its own:
the codebook file, its log, and the labels table are the memory, so any tick
can be re-run, and a codebook promoted after months of neglect simply catches up.

One convention makes codebooks declarable in config alone: **the query
returns exactly (id, ts, trace)** — id first (the labels-table join key),
ts second (data time: the story clock), trace third (a JSON object or plain
text). Shaping belongs in SQL, where warehouse users already do it. Codebooks
that need a Python units function or to_trace stay in scripts; config covers
the common record-grain case.

A tick never applies a revision. It labels eagerly, reads the drift signals,
and when they clear policy it writes the proposal: `<name>.codebook.pending.json` (the
machine diff, core's review gate) and `<name>.codebook.pending.md` — the ops with their
evidence quoted, frozen at propose time, ready to be a PR body. Acceptance is
`teca-label apply` — locally, or in the branch a scheduler opened."""
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .labeling import VERSION_POLICIES, check_table_name

try:
    import tomllib
except ModuleNotFoundError:                      # Python 3.10
    import tomli as tomllib                      # declared for <3.11 in pyproject

from .core import (Codebook, Revision, _coerce_dt, _period_of, drift_signals, read_log,
                   relabel_targets)

__all__ = ["TICK_PROPOSED", "load_project", "plan_defaults", "relabel_targets", "story_from",
           "pending_md_path", "render_proposal", "tick"]

TICK_PROPOSED = 10   # `teca-label run` exit code: a proposal awaits review

_DEFAULTS = {"dsn_env": "TECA_LABEL_DSN", "cadence": "week", "labels_table": "teca_labels",
             "workers": 8, "on_version_change": "relabel_touched",
             "runs_table": "teca_runs"}   # the tick's own ledger (observability)
_KEYS = set(_DEFAULTS) | {"path", "query", "name"}
# [project] keys that belong to Codebook.plan(), not to the tick: the models, window
# and sample size chosen once for every build in this project.
_PLAN_KEYS = {"models": dict, "window": str, "sample": int}


def _read_toml(root: str | Path) -> dict:
    path = Path(root) / "teca-label.toml"
    if not path.exists():
        raise FileNotFoundError(f"no teca-label.toml in {Path(root).resolve()} — a Teca Label "
                                f"project needs one: a [codebooks.<name>] table with "
                                f"'path' and 'query'")
    config = tomllib.loads(path.read_text())
    unknown = set(config.get("project", {})) - set(_DEFAULTS) - set(_PLAN_KEYS)
    if unknown:
        raise ValueError(f"unknown [project] keys {sorted(unknown)} — "
                         f"known: {sorted(set(_DEFAULTS) | set(_PLAN_KEYS))}")
    return config


def plan_defaults(root: str | Path = ".") -> dict:
    """The plan-time choices a project made once, on the record: the `models`,
    `window` and `sample` under [project] in teca-label.toml, as keyword arguments for
    Codebook.plan(). Only the keys the file sets; {} when there is no file — a
    project without one has made no choices yet."""
    if not (Path(root) / "teca-label.toml").exists():
        return {}
    project = _read_toml(root).get("project", {})
    chosen = {}
    for key, kind in _PLAN_KEYS.items():
        if key in project:
            if not isinstance(project[key], kind) or isinstance(project[key], bool):
                raise ValueError(f"[project] {key} must be {kind.__name__}, "
                                 f"got {project[key]!r}")
            chosen[key] = project[key]
    if "models" in chosen:
        bad = {role for role, model in chosen["models"].items() if not isinstance(model, str)}
        if bad:
            raise ValueError(f"[project.models] must map role to model name; {sorted(bad)} are not")
    return chosen


def load_project(root: str | Path = ".") -> dict[str, dict]:
    """Read teca-label.toml: {codebook name: fully-resolved spec}. Bad config fails
    here, loudly and specifically — a scheduler's first tick is nobody's
    debugging session. DSNs are never in the file, only env var names."""
    config = _read_toml(root)
    project_defaults = {k: v for k, v in config.get("project", {}).items() if k in _DEFAULTS}
    codebooks = config.get("codebooks")
    if not codebooks:
        raise ValueError("teca-label.toml has no [codebooks.<name>] tables")
    resolved = {}
    for name, spec in codebooks.items():
        unknown = set(spec) - _KEYS
        if unknown:
            raise ValueError(f"[codebooks.{name}]: unknown keys {sorted(unknown)} — "
                             f"known: {sorted(_KEYS)}")
        merged = {**_DEFAULTS, **project_defaults, **spec}
        for required in ("path", "query"):
            if not merged.get(required):
                raise ValueError(f"[codebooks.{name}]: '{required}' is required")
        merged["name"] = merged.get("name") or name
        merged["path"] = Path(root) / merged["path"]
        if not merged["path"].exists():
            raise ValueError(f"[codebooks.{name}]: codebook {merged['path']} not found — "
                             f"draft it first (Codebook.plan(...).build()) or fix the path")
        _period_of(datetime.now(timezone.utc), merged["cadence"])  # validates cadence
        if merged["on_version_change"] not in VERSION_POLICIES:
            raise ValueError(f"[codebooks.{name}]: on_version_change must be one of "
                             f"{VERSION_POLICIES}, got '{merged['on_version_change']}'")
        for key in ("labels_table", "runs_table"):
            try:
                check_table_name(merged[key])
            except ValueError as bad:
                raise ValueError(f"[codebooks.{name}]: {key}: {bad}") from None
        resolved[name] = merged
    return resolved


def story_from(times: dict[str, object], latest: dict[tuple[str, int], tuple[str, int]],
               cadence: str, min_current: int | None = None,
               now: datetime | None = None) -> list[tuple[str, Counter]]:
    """The period story, rebuilt from durable state: each unit counted under its
    latest label, bucketed by data time — recovered from the warehouse each tick,
    never held in memory: the tick's statelessness.

    The wall-clock-current period is partial by definition (a Monday tick sees
    a week of forty rows and would trip false alarms), so it is dropped unless
    it already holds >= min_current rows; min_current=None always drops it."""
    buckets: dict[str, Counter] = {}
    for (row_id, _unit), (category, _version) in latest.items():
        ts = times.get(row_id)
        if ts is None or category is None:
            continue
        buckets.setdefault(_period_of(_coerce_dt(ts), cadence), Counter())[category] += 1
    current = _period_of(now or datetime.now(timezone.utc), cadence)
    partial = current in buckets and (
        min_current is None or sum(buckets[current].values()) < min_current)
    if partial:
        del buckets[current]
    return sorted(buckets.items())


# ---------- the proposal, rendered for review ----------

def pending_md_path(cb: Codebook) -> Path:
    return cb.path.with_suffix(".pending.md")


def _op_line(op) -> str:
    if op.op == "add":
        return f"**add `{op.name}`** — {op.definition} ({len(op.evidence)} evidence rows)"
    if op.op == "redefine":
        return f"**redefine `{op.name}`** — {op.definition}"
    if op.op == "rename":
        return f"**rename `{op.name}` → `{op.new_name}`**"
    if op.op == "merge":
        merged = ", ".join(f"`{name}`" for name in op.names)
        return f"**merge {merged} → `{op.new_name}`** — {op.definition}"
    if op.op == "split":
        results = ", ".join(f"`{category.name}`" for category in op.into)
        return f"**split `{op.name}` → {results}**"
    return f"**deprecate `{op.name}`**"


def _signal_line(signal: dict) -> str:
    detail = ", ".join(f"{key}={value}" for key, value in signal.items() if key != "kind")
    return f"**{signal['kind']}** ({detail})"


def render_proposal(name: str, cb: Codebook, revision: Revision,
                    signals: list[dict], traces: list[dict],
                    labels: list[str | None]) -> str:
    """The pending revision as markdown — a PR body, a review email, a message.
    Evidence is quoted here because the op indices point into a batch that is
    gone by review time; this file freezes what the reviser actually saw."""
    lines = [f"## `{name}`: proposed revision (v{cb.version} → v{cb.version + 1})", "",
             f"**Question:** {cb.question}", ""]
    if signals:
        lines += ["**Signals:**"] + [f"- {_signal_line(s)}" for s in signals] + [""]
    lines += ["### Ops", ""] + [f"- {_op_line(op)}" for op in revision.ops] + [""]
    cited = list(dict.fromkeys(i for op in revision.ops for i in op.evidence))
    if cited:
        lines += ["### Evidence", ""]
        for i in cited:
            if 0 <= i < len(traces):
                clipped = json.dumps(traces[i], default=str)[:300]
                lines.append(f"- `[{i}]` (current: {labels[i]}) {clipped}")
        lines.append("")
    lines += ["### Review", "",
              f"Amend `{cb.pending_path.name}` if needed, "
              f"then `teca-label apply --codebook {name}` — or merge the branch carrying this file."]
    return "\n".join(lines)


# ---------- the tick ----------

def _parse_trace(raw) -> dict:
    if isinstance(raw, dict):        # jsonb arrives already parsed
        return raw
    if isinstance(raw, str):
        if raw.lstrip()[:1] in ("{", "["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except ValueError:
                pass
        return {"text": raw}
    return {"text": str(raw)}


def _shape(fetched: dict[str, tuple]) -> dict[str, tuple]:
    """Check fetched rows against the (id, ts, trace) convention:
    {row_id: (ts, trace dict)}."""
    rows = {}
    for row_id, columns in fetched.items():
        if len(columns) != 2:
            raise ValueError(f"the query must return exactly (id, ts, trace) — "
                             f"got {1 + len(columns)} columns; shape the trace in SQL "
                             f"(e.g. jsonb_build_object(...) / struct_pack(...))")
        ts, raw = columns
        rows[row_id] = (ts, _parse_trace(raw))
    return rows


def _fetch(spec: dict, dsn: str) -> dict[str, tuple]:
    """The codebook's query against Postgres, id-first."""
    from . import sources
    conn = sources._connect(dsn)
    try:
        return sources._select(conn.cursor(), query=spec["query"])
    finally:
        conn.close()


def _revised_periods(cb: Codebook) -> set[str]:
    """Periods whose signals have already produced an applied revision (from the log)."""
    return {period for event in read_log(cb.log_path)
            if event["event"] == "revision" for period in event.get("periods", [])}


def tick(name: str, spec: dict, provider=None) -> dict:
    """One cycle of the standing loop, composed from durable artifacts only:
    label what's new (plus, per on_version_change, what the last accepted revision
    could have moved), rebuild the story from the warehouse, read the drift signals, and — when
    they clear policy — write the proposal for review. Never applies.

    Returns the run summary: what was labeled, the periods seen, the signals,
    whether a proposal awaits, and the tokens spent."""
    from . import sources
    cb = Codebook.load(spec["path"], client=provider)
    dsn = os.environ.get(spec["dsn_env"], "")
    if not dsn:
        raise ValueError(f"[codebooks.{name}]: env var {spec['dsn_env']} is not set")

    # the tick's own ledger row: opened before any work, closed exactly once —
    # 'ok' or 'failed' — on every exit path, including an interrupt. A row left
    # 'running' with no process behind it is the crash alarm. Labels written this
    # tick carry this run_id (lineage).
    started = time.monotonic()
    run_id = sources.run_start(dsn, spec["name"], cb.version, cb.models["classify"],
                               runs_table=spec["runs_table"])
    try:
        return _tick(name, spec, cb, dsn, run_id, started)
    except BaseException as failure:
        sources.run_finish(dsn, run_id, "failed", runs_table=spec["runs_table"],
                           seconds=round(time.monotonic() - started, 1),
                           error=f"{type(failure).__name__}: {failure}"[:500])
        raise


def _tick(name: str, spec: dict, cb: Codebook, dsn: str, run_id: str, started: float) -> dict:
    from . import sources
    fetched = _fetch(spec, dsn)
    rows = _shape(fetched)
    run = sources.label_postgres(
        cb, dsn, spec["query"], to_trace=lambda ts, raw: _parse_trace(raw),
        name=spec["name"], labels_table=spec["labels_table"], workers=spec["workers"],
        on_version_change=spec["on_version_change"],
        rows=fetched, run_id=run_id)

    times = {row_id: ts for row_id, (ts, _) in rows.items()}
    latest = sources.latest_labels(dsn, spec["name"], spec["labels_table"])
    story = story_from(times, latest, spec["cadence"], min_current=cb.policy["min_batch"])
    active_names = [category.name for category in cb.active()]
    signals = drift_signals(story, active_names, cb.policy, kind=cb.kind)

    # the audit cadence, derived statelessly: every Nth period of the story
    audit_every = cb.policy["audit_every"]
    if audit_every and story and len(story) % audit_every == 0 and active_names:
        window = sorted((row_id for row_id in rows if (row_id, 0) in latest),
                        key=lambda row_id: str(times[row_id]))[-500:]
        if len(window) >= 25:
            findings = cb.audit([rows[row_id][1] for row_id in window],
                                [latest[(row_id, 0)][0] for row_id in window],
                                workers=spec["workers"])
            signals += [{"kind": "audit", "category": finding["category"], "period": story[-1][0],
                         "hidden_theme": finding["hidden_theme"]}
                        for finding in findings if finding.get("hidden_theme")]

    # a proposal already awaiting review stands: never re-spend, never overwrite it.
    # A period legislates once: the revision that came out of this period's signals
    # is applied, so the same alarms must not draft it again on tomorrow's tick.
    # An open proposal is known two ways: the pending file beside the codebook, and the
    # ledger row that wrote it — the file may live only in a review branch, the ledger is
    # always here. Either one stands until the version moves.
    proposed = bool(cb.pending_path and cb.pending_path.exists())
    last_period = story[-1][0] if story else None
    open_proposal = sources.last_proposal(dsn, spec["name"], spec["runs_table"])
    if open_proposal and open_proposal == (last_period, cb.version):
        proposed = True
    already_revised = bool(story) and last_period in _revised_periods(cb)
    proposed_for = None
    if signals and story and not proposed and not already_revised:
        batch_ids = [row_id for row_id in rows if (row_id, 0) in latest
                     and _period_of(_coerce_dt(times[row_id]), spec["cadence"]) == last_period]
        if len(batch_ids) >= cb.policy["min_batch"]:
            batch = [rows[row_id][1] for row_id in batch_ids]
            batch_labels = [latest[(row_id, 0)][0] for row_id in batch_ids]
            revision = cb.propose(batch, labels=batch_labels, signals=signals)
            if revision.ops:
                pending_md_path(cb).write_text(render_proposal(
                    name, cb, revision, signals, batch, batch_labels))
                proposed = True
                proposed_for = last_period

    sources.run_finish(
        dsn, run_id, "ok", runs_table=spec["runs_table"],
        rows_seen=len(fetched), rows_labeled=sum(run.counts.values()),
        gaps=len(run.failed_row_ids), unclassifiable=run.counts.get("unclassifiable", 0),
        other_share=round(run.other_share, 4),
        seconds=round(time.monotonic() - started, 1), proposed_for=proposed_for)
    return {"codebook": name, "version": cb.version, "labeled": dict(run.counts),
            "failed": run.failed_row_ids, "errors": run.errors, "versions": run.versions,
            "periods": [period for period, _ in story], "signals": signals,
            "proposed": proposed, "usage": dict(cb.usage), "run_id": run_id}
