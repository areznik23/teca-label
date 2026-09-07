"""Postgres as the trace source: fetch a (sampled) subset to draft or audit from, and label
every row a query returns, incrementally, writing back next to the data.

One convention throughout: the query selects the row's unique id as its FIRST column and
`to_trace` maps the remaining columns to a dict. The same query drives both movements —
`fetch(..., sample=60)` to draft a codebook, `label_postgres(...)` to apply it exhaustively.

Two persistent artifacts make every labeling run resumable and every dollar spent once:
- the units table: extraction results per (row, codebook) — the expensive full-document read
  is paid one time; codebook revisions re-classify cached units without re-reading rows.
  A sentinel row (unit_index = -1) records "extracted, nothing relevant" so empty
  documents aren't re-extracted either.
- the labels table: one row per (unit, codebook, codebook version), carrying the evidence
  quote and the model that judged it. A call that fails writes nothing, so the row is
  pending on the next run — always; what a version bump does to rows already labeled is
  a stated policy (`on_version_change`, see teca_label.labeling), never an accident."""
import json
import uuid
import warnings
from typing import Callable

try:
    import psycopg2
    import psycopg2.extras
except ImportError as missing:
    raise ImportError("pip install 'teca-label[postgres]' to read from or label into Postgres") from missing

from .core import Codebook, thin_traces
from .labeling import check_table_name, VERSION_POLICIES, LabelRun, Latest, eligibility
from .units import Units, unitize

_LABELS_DDL = """CREATE TABLE IF NOT EXISTS {table} (
    row_id text NOT NULL,
    unit_index int NOT NULL DEFAULT 0,
    category text NOT NULL,
    evidence text,
    codebook text NOT NULL,
    codebook_version int NOT NULL,
    model text NOT NULL,
    labeled_at timestamptz NOT NULL DEFAULT now(),
    run_id text,
    PRIMARY KEY (row_id, unit_index, codebook, codebook_version))"""

# The pipeline's own ledger: one row per tick per codebook, inserted at start
# (status 'running') and updated exactly once on landing — mutable in flight,
# frozen after. A row stuck in 'running' with no process behind it IS the
# crash alarm. Labels carry run_id, so every label joins back to the exact
# instrument state (model, version, health) that wrote it.
_RUNS_DDL = """CREATE TABLE IF NOT EXISTS {table} (
    run_id text PRIMARY KEY,
    run_at timestamptz NOT NULL DEFAULT now(),
    codebook text NOT NULL,
    codebook_version int NOT NULL,
    model text NOT NULL,
    status text NOT NULL DEFAULT 'running',
    rows_seen int,
    rows_labeled int,
    gaps int,
    unclassifiable int,
    other_share real,
    seconds real,
    error text,
    proposed_for text)"""

_RUN_FIELDS = ("rows_seen", "rows_labeled", "gaps", "unclassifiable", "other_share",
               "seconds", "error", "proposed_for")   # proposed_for: the period a proposal covered

_UNITS_DDL = """CREATE TABLE IF NOT EXISTS {table} (
    row_id text NOT NULL,
    codebook text NOT NULL,
    unit_index int NOT NULL,
    content text,
    model text NOT NULL,
    extracted_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (row_id, codebook, unit_index))"""

_LATEST_SQL = """SELECT DISTINCT ON (row_id, unit_index) row_id, unit_index, category, codebook_version
    FROM {table} WHERE codebook = %s
    ORDER BY row_id, unit_index, codebook_version DESC"""

# the first row of each of n equal buckets: the same even spread core._sample makes
_STRIDED_SAMPLE = """SELECT DISTINCT ON (((teca_rn - 1) * {n}) / teca_n) * FROM (
    SELECT q.*, row_number() OVER () AS teca_rn, count(*) OVER () AS teca_n
    FROM ({query}) q) s
ORDER BY ((teca_rn - 1) * {n}) / teca_n, teca_rn"""


def _ensure_column(cur, table: str, column: str, kind: str) -> None:
    """Add a column a table created by an earlier version lacks. Checked first, altered
    only when missing: an ALTER takes an exclusive lock and would wait behind any open
    read transaction on the table (a BI tool, an idle session), stalling the run."""
    schema, _, name = table.rpartition(".")
    cur.execute("""SELECT 1 FROM information_schema.columns
                   WHERE table_name = %s AND column_name = %s
                   AND (%s = '' OR table_schema = %s)""", (name, column, schema, schema))
    if cur.fetchone() is None:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {kind}")


def _connect(dsn: str):
    """Connect, or fail with one readable line: the DSN never appears in the message."""
    try:
        conn = psycopg2.connect(dsn, connect_timeout=15)
    except psycopg2.OperationalError as unreachable:
        reason = str(unreachable).strip().splitlines()[0]
        raise ConnectionError(f"cannot reach the database: {reason}") from None
    conn.autocommit = True
    return conn


def _select(cur, query: str, sample: int | None = None) -> dict[str, tuple]:
    """Rows keyed by their id column. With `sample`, the stride happens in the database:
    n rows spread evenly across the query's own ordering, so only those n leave the store."""
    if sample is None:
        cur.execute(query)
        return {str(row[0]): row[1:] for row in cur.fetchall()}
    cur.execute(_STRIDED_SAMPLE.format(query=query, n=int(sample)))
    return {str(row[0]): row[1:-2] for row in cur.fetchall()}


def fetch(dsn: str, query: str, to_trace: Callable[..., dict],
          sample: int | None = None) -> list[dict]:
    """Traces straight from Postgres, in the query's order, ready for Codebook.plan(),
    audit(), or drill().

    `sample=n` takes n rows spread evenly across the result — the same stride
    a plan would apply in memory, done in SQL so the rest never leaves the
    database. Give the query an ORDER BY (usually a timestamp) so the spread covers the
    era rather than whatever order the table happens to return."""
    conn = _connect(dsn)
    try:
        traces = [to_trace(*columns) for columns in _select(conn.cursor(), query, sample).values()]
    finally:
        conn.close()
    thin = thin_traces(traces)
    if thin:
        warnings.warn(f"{thin} — check to_trace and the query", stacklevel=2)
    return traces


def latest_labels(dsn: str, name: str,
                  labels_table: str = "teca_labels") -> dict[tuple[str, int], tuple[str, int]]:
    """Each unit's current label: {(row_id, unit_index): (category, codebook_version)} —
    'current' meaning the highest codebook version that has judged the unit, the same
    row a latest-per-unit view exposes to BI. The standing loop reads its story from
    this; labels under older versions stay in the table as honest history."""
    check_table_name(labels_table)
    conn = _connect(dsn)
    try:
        cur = conn.cursor()
        try:
            cur.execute(_LATEST_SQL.format(table=labels_table), (name,))
        except psycopg2.errors.UndefinedTable:
            return {}   # the very first tick: the table is born when the first label lands
        return {(row[0], row[1]): (row[2], row[3]) for row in cur.fetchall()}
    finally:
        conn.close()


def run_start(dsn: str, name: str, version: int, model: str,
              runs_table: str = "teca_runs") -> str:
    """Open a run in the ledger and return its run_id. The row lands as
    status='running'; run_finish() closes it exactly once."""
    check_table_name(runs_table)
    run_id = uuid.uuid4().hex
    conn = _connect(dsn)
    try:
        cur = conn.cursor()
        cur.execute(_RUNS_DDL.format(table=runs_table))
        _ensure_column(cur, runs_table, "proposed_for", "text")
        cur.execute(f"""INSERT INTO {runs_table} (run_id, codebook, codebook_version, model)
                        VALUES (%s, %s, %s, %s)""", (run_id, name, version, model))
    finally:
        conn.close()
    return run_id


def run_finish(dsn: str, run_id: str, status: str,
               runs_table: str = "teca_runs", **fields) -> None:
    """Close a run: set its final status ('ok' | 'failed')
    and outcome fields (any of _RUN_FIELDS). Only a 'running' row can be
    closed — a landed row is history and never changes again."""
    check_table_name(runs_table)
    unknown = set(fields) - set(_RUN_FIELDS)
    if unknown:
        raise ValueError(f"unknown run fields: {sorted(unknown)}")
    sets = ", ".join(["status = %s"] + [f"{key} = %s" for key in fields])
    conn = _connect(dsn)
    try:
        conn.cursor().execute(
            f"UPDATE {runs_table} SET {sets} WHERE run_id = %s AND status = 'running'",
            (status, *fields.values(), run_id))
    finally:
        conn.close()


def last_run(dsn: str, name: str, runs_table: str = "teca_runs") -> dict | None:
    """The most recent ledger row for a codebook, as a dict — or None when the
    ledger doesn't exist yet (no tick has ever run against this database)."""
    check_table_name(runs_table)
    conn = _connect(dsn)
    try:
        cur = conn.cursor()
        try:
            cur.execute(f"""SELECT run_id, run_at, codebook_version, model, status, rows_seen,
                                   rows_labeled, gaps, unclassifiable, other_share,
                                   seconds, error, proposed_for
                            FROM {runs_table} WHERE codebook = %s
                            ORDER BY run_at DESC LIMIT 1""", (name,))
        except psycopg2.errors.UndefinedTable:
            return None
        row = cur.fetchone()
        if row is None:
            return None
        keys = ("run_id", "run_at", "codebook_version", "model", "status", "rows_seen",
                "rows_labeled", "gaps", "unclassifiable", "other_share",
                "seconds", "error", "proposed_for")
        return dict(zip(keys, row))
    finally:
        conn.close()


def last_proposal(dsn: str, name: str, runs_table: str = "teca_runs") -> tuple[str, int] | None:
    """(period, codebook_version) of the most recent run that wrote a proposal for this
    codebook, or None. The tick reads it so an open proposal is never drafted twice —
    the pending file lives in a branch, but the ledger is always there."""
    check_table_name(runs_table)
    conn = _connect(dsn)
    try:
        cur = conn.cursor()
        try:
            cur.execute(f"""SELECT proposed_for, codebook_version FROM {runs_table}
                            WHERE codebook = %s AND proposed_for IS NOT NULL
                            ORDER BY run_at DESC LIMIT 1""", (name,))
        except (psycopg2.errors.UndefinedTable, psycopg2.errors.UndefinedColumn):
            return None
        row = cur.fetchone()
        return None if row is None else (row[0], row[1])
    finally:
        conn.close()


def label_postgres(codebook: Codebook, dsn: str, query: str, to_trace: Callable[..., dict],
                   name: str | None = None, units: Units | None = None,
                   labels_table: str = "teca_labels", units_table: str = "teca_units",
                   batch_size: int = 100, workers: int = 8, max_rows: int | None = None,
                   on_progress: Callable[[int, int], None] | None = None,
                   on_version_change: str = "keep",
                   rows: dict[str, tuple] | None = None,
                   run_id: str | None = None) -> LabelRun:
    """Run the codebook over every row the query returns that still needs a label.

    `query` selects the row's unique id as its FIRST column; `to_trace` maps the remaining
    columns to a dict. With `units`, extraction runs first (stage 1, resumable: only
    rows never extracted under this codebook; a failed row gets no sentinel so the next run
    retries it), then every cached unit that needs a label is labeled (stage 2). Unit
    provenance comes from the units function itself — it knows which model it ran, which
    may differ from the codebook's configured 'extract' role. Without `units`, the record
    itself is the single unit.

    Returns a LabelRun: `counts` written this run by category, `failed_row_ids` (calls that
    failed — nothing was written for them, so they are pending on the next call, always),
    `errors`, `other_share`, and `versions` (units by latest-label version, after).

    `name` names this codebook's rows in the labels table; it defaults to the
    codebook's own filename stem (failure-modes.codebook.json -> 'failure-modes').

    `on_version_change` says what happens to rows labeled under an older codebook version:
    "keep" (default — old labels stand, only never-labeled rows are judged), "relabel"
    (every row below the live version is judged again), or "relabel_touched" (only rows
    whose latest label an op since their version could have moved, read from the revision
    log). See teca_label.labeling.

    `max_rows` caps the rows extracted this call on the units path, and the rows
    labeled this call otherwise. `rows` supplies the query's result already fetched
    ({row_id: remaining columns}, id-first) so a caller that has just read the table
    doesn't read it twice; the query is then only the record of what was read."""
    if name is None:
        if not codebook.path:
            raise ValueError("pass name= — the codebook has no path to derive a name from")
        name = codebook.path.name.removesuffix(".json").removesuffix(".codebook")
    if on_version_change not in VERSION_POLICIES:
        raise ValueError(f"on_version_change must be one of {VERSION_POLICIES}, "
                         f"got '{on_version_change}'")
    check_table_name(labels_table)
    check_table_name(units_table)
    conn = _connect(dsn)
    try:
        return _label(codebook, conn.cursor(), query, to_trace, name, units, labels_table,
                      units_table, batch_size, workers, max_rows, on_progress,
                      on_version_change, rows, run_id)
    finally:
        conn.close()


def _label(codebook, cur, query, to_trace, name, units, labels_table, units_table,
           batch_size, workers, max_rows, on_progress, on_version_change,
           rows=None, run_id=None) -> LabelRun:
    cur.execute(_LABELS_DDL.format(table=labels_table))
    # tables created by an earlier version gain the newer columns in place
    _ensure_column(cur, labels_table, "evidence", "text")
    _ensure_column(cur, labels_table, "run_id", "text")
    fetched = rows if rows is not None else _select(cur, query)
    cur.execute(_LATEST_SQL.format(table=labels_table), (name,))
    latest: Latest = {(row[0], row[1]): (row[2], row[3]) for row in cur.fetchall()}
    eligible = eligibility(codebook, latest, on_version_change)
    run = LabelRun()

    if units is not None:
        if getattr(units, "model", None) is None:
            units.model = codebook.models["extract"]   # the 'extract' role governs extraction
        cur.execute(_UNITS_DDL.format(table=units_table))
        cur.execute(f"SELECT DISTINCT row_id FROM {units_table} WHERE codebook = %s", (name,))
        already_extracted = {row[0] for row in cur.fetchall()}
        pending = [(row_id, to_trace(*columns)) for row_id, columns in fetched.items()
                   if row_id not in already_extracted]
        if max_rows is not None:
            pending = pending[:max_rows]
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            triples, failed = unitize(batch, units, workers=workers)
            run.failed_row_ids.extend(failed)
            run.errors.extend(failed.values())
            extract_model = getattr(units, "model", None) or codebook.models["extract"]
            rows_with_units = {row_id for row_id, _, _ in triples} | set(failed)
            values = [(row_id, name, index, json.dumps(unit, default=str), extract_model)
                      for row_id, index, unit in triples]
            values += [(row_id, name, -1, None, extract_model)
                       for row_id, _ in batch if row_id not in rows_with_units]
            psycopg2.extras.execute_values(cur, f"""INSERT INTO {units_table}
                (row_id, codebook, unit_index, content, model) VALUES %s ON CONFLICT DO NOTHING""", values)
        cur.execute(f"""SELECT row_id, unit_index, content FROM {units_table}
                        WHERE codebook = %s AND unit_index >= 0 ORDER BY row_id, unit_index""", (name,))
        to_label = [(row[0], row[1], json.loads(row[2])) for row in cur.fetchall()
                    if row[0] in fetched and eligible((row[0], row[1]))]   # only rows the query returns
    else:
        to_label = [(row_id, 0, to_trace(*columns)) for row_id, columns in fetched.items()
                    if eligible((row_id, 0))]
        if max_rows is not None:
            to_label = to_label[:max_rows]

    for start in range(0, len(to_label), batch_size):
        batch = to_label[start:start + batch_size]
        judgments = codebook.judge([unit for _, _, unit in batch], workers=workers)
        # evidence: the excerpt's own quote when the unit is an excerpt, else the
        # quote the classifier picked from the record
        values = [(row_id, index, label, unit.get("quote") or evidence, name, codebook.version,
                   codebook.models["classify"], run_id)
                  for (row_id, index, unit), judgment in zip(batch, judgments)
                  if judgment is not None for label, evidence in [judgment]]
        run.failed_row_ids.extend(row_id for (row_id, _, _), judgment in zip(batch, judgments)
                                  if judgment is None)
        run.errors.extend(codebook.last_errors)
        psycopg2.extras.execute_values(cur, f"""INSERT INTO {labels_table}
            (row_id, unit_index, category, evidence, codebook, codebook_version, model, run_id)
            VALUES %s ON CONFLICT DO NOTHING""", values)
        run.counts.update(v[2] for v in values)
        for row_id, index, label, *_ in values:
            latest[(row_id, index)] = (label, codebook.version)
        if on_progress:
            on_progress(start + len(batch), len(to_label))
    return run.finish(latest)
