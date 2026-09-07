# The Teca Label format

Teca Label is a file and a table. The file is a codebook: one question and the fixed list of
answers to it. The table is the labels it produces. The Python library is one way to write
the file and fill the table; anything that reads or writes these shapes is compatible with it.

This page is the contract. Schema **1**.

## The rules

1. A label means what its codebook meant at the version that wrote it. `(codebook,
   codebook_version)` on a label row is enough to look that up, forever.
2. A codebook changes only by a revision: a list of typed ops, applied all at once, bumping
   `version` by exactly one. Nothing edits a category in place.
3. A category is never deleted. Deprecation marks it with the version that retired it; its
   name stays in the file so old labels stay readable.
4. `other` and `unclassifiable` are reserved. `other` is a label that fits no category and is
   always a legal answer; `unclassifiable` is a record with nothing to read — no text in any
   field — and is written without a model call. Neither can be a category name.
5. Every artifact carries `"schema"`. Changes to this format are additive; a schema 1 reader
   can always read a schema 1 file written later. A file that omits `schema` is schema 1.

## The codebook file — `<name>.codebook.json`

```json
{
  "schema": 1,
  "question": "Why do my agent's sessions fail?",
  "version": 2,
  "kind": "partition",
  "categories": [
    {"name": "tool_loop",       "definition": "The agent repeats a failing tool call without changing inputs.", "created_v": 1, "deprecated_v": null},
    {"name": "missing_context", "definition": "The agent lacked information present elsewhere in the account.", "created_v": 1, "deprecated_v": null},
    {"name": "sandbox_timeout", "definition": "The run hit the execution time limit.",                          "created_v": 1, "deprecated_v": 2},
    {"name": "rate_limit_cascade", "definition": "An upstream 429 propagates into repeated whole-task retries.", "created_v": 2, "deprecated_v": null}
  ]
}
```

The name before `.codebook.json` is the codebook's name; it is the `codebook` value on every
label row it writes.

| Field | Required | Meaning |
|---|---|---|
| `schema` | no (default 1) | The format version of this file. |
| `question` | yes | The one question every label answers. |
| `version` | yes | Current revision number. `0` means drafted but never applied; `1` is the first usable codebook. |
| `kind` | no (default `partition`) | `partition`: every record gets exactly one category or `other`. `tag`: one category; a label is either that category or `other` (meaning "no"). |
| `categories` | yes | The answers. Order is the order they are presented to a model. |
| `categories[].name` | yes | `^[a-z][a-z0-9_]*$`, at most 40 characters. It is a value in a SQL column, a dict key, and a word people say. |
| `categories[].definition` | yes | One or two sentences a labeler can apply. Two records share a category only if they would lead to the same decision. |
| `categories[].created_v` | no (default 1) | The version that introduced it. |
| `categories[].deprecated_v` | no (default null) | The version that retired it; `null` means active. |
| `parent` | no | For a child codebook: `{"path", "categories", "version", "pinned_at"}` — the parent file, the parent categories whose records this codebook reads, the parent version that defined them, and when the pin was taken. Absent means the codebook covers the whole corpus. |
| `draft_window` | no | `{"start", "end", "n_sampled"}` — the era of records the categories were drafted from. Provenance; older records labeled against it may fall to `other` honestly. |
| `models`, `policy` | no | Library configuration the Teca Label library stores beside the codebook (which model fills which role, drift thresholds). Another tool may ignore or omit them. |

The minimum hand-written codebook is `question`, `version` (use `1`), and `categories`,
each category a `name` and a `definition`. Everything else has a default.

`policy` holds the thresholds the library reads, all optional:

| Key | Default | Meaning |
|---|---|---|
| `other_threshold` | `0.1` | `other` share in the latest period above this is an emergence signal. |
| `min_batch` | `20` | A period needs this many rows before it can be judged or legislated. |
| `max_adds_per_revision` | `3` | A proposal keeps at most this many `add` ops (bootstrap drafts exempt). |
| `decay_periods` | `3` | A category under `decay_min_share` for this many periods proposes its retirement. |
| `decay_min_share` | `0.01` | The share below which a category counts as flat. |
| `audit_every` | `4` | Every Nth period the runner audits each category for a hidden sub-theme; `0` disables. |
| `min_agreement` | `0.8` | A classify model must reach this agreement with the benchmark to be adopted. |

**Active categories** are those with `deprecated_v` null. A labeler presents only active
categories, plus `other`.

## The log — `<name>.codebook.log.jsonl`

Append-only, one JSON object per line, one line per event. Every line carries the full
`categories` list as it stood after the event, so any past version is reconstructable from
the log alone.

```json
{"schema": 1, "event": "revision", "version": 2, "time": "2026-07-28T09:15:02+00:00",
 "models": {"draft": "claude-opus-5", "classify": "claude-opus-5", "extract": "claude-opus-5"},
 "categories": [ ...the list above... ],
 "ops": [ ...the revision's ops... ], "source": "pending"}
```

| Field | Meaning |
|---|---|
| `schema` | Format version. |
| `event` | What happened. `adopted` (categories were written by a person or another tool, not drafted), `revision` (ops applied; `source` is `pending` if it passed review or `direct`), `audit`, `reparent`, `bench` (a human-labeled benchmark was recorded), `measure` (a model was scored against it), `model_change` (a role's model changed; carries the measurement for `classify`). New event kinds may appear; a reader skips kinds it doesn't know. |
| `version` | The codebook version after the event. |
| `time` | ISO 8601, UTC. |
| `models` | The role → model map at the time. Provenance. |
| `library` | The library version that wrote the event. The classify prompt is fixed per library version, so this plus `models` pins the instrument. |
| `categories` | The full category list after the event. |
| *(other keys)* | Event-specific detail — `ops`, `source`, and `periods` on a revision, findings on an audit. |

The last `revision` line and the codebook file agree; if they don't, the log is the record.

## A revision — `<name>.codebook.pending.json`

A proposed change awaiting review. It's the same shape a person would write by hand, and
deleting it rejects the proposal.

```json
{"schema": 1, "ops": [
  {"op": "add", "name": "rate_limit_cascade",
   "definition": "An upstream 429 propagates into repeated whole-task retries.", "evidence": [3, 17, 41]},
  {"op": "redefine", "name": "missing_context",
   "definition": "The agent lacked information present elsewhere in the account, EXCLUDING rate-limit retries."},
  {"op": "deprecate", "name": "sandbox_timeout"}
]}
```

Six ops. A revision is validated whole before any of it applies; one bad op rejects all of it.
An optional `periods` list names the story periods whose drift signals motivated the proposal;
the runner uses it so a period legislates once.

| `op` | Fields | Effect at version *v* |
|---|---|---|
| `add` | `name`, `definition` | New active category, `created_v = v`. |
| `redefine` | `name`, `definition` | Replaces the definition in place. Same name, same `created_v`. |
| `rename` | `name`, `new_name` | The category continues under `new_name`, keeping its `created_v`. Labels written under the old name are read through the log. |
| `deprecate` | `name` | Sets `deprecated_v = v`. |
| `merge` | `names`, `new_name`, `definition` | Deprecates every source at *v*, adds `new_name` with `created_v = v`. |
| `split` | `name`, `into[]` (each a `name` + `definition`) | Deprecates `name` at *v*, adds each result with `created_v = v`. |

`evidence` on any op is optional: indices into whatever batch of records motivated it, for
the reviewer. It has no meaning after the revision is applied.

Constraints, checked on apply: an op targets an active category (or, for a new name, one
that has never been used, compared case-insensitively); a new name is well-formed and not
reserved; `add`, `redefine`, and `merge` carry a non-empty definition.

## The labels table — `teca_labels`

One row per (record, unit, codebook, codebook version). The default table name is
`teca_labels`; a project may choose another.

```sql
CREATE TABLE IF NOT EXISTS teca_labels (
    row_id           text        NOT NULL,
    unit_index       int         NOT NULL DEFAULT 0,
    category         text        NOT NULL,
    evidence         text,
    codebook         text        NOT NULL,
    codebook_version int         NOT NULL,
    model            text        NOT NULL,
    labeled_at       timestamptz NOT NULL DEFAULT now(),
    run_id           text,
    PRIMARY KEY (row_id, unit_index, codebook, codebook_version)
);
```

| Column | Meaning |
|---|---|
| `row_id` | The record's id in your table, as text. Cast on join: `sessions.id::text = row_id`. |
| `unit_index` | `0` when the whole record was labeled. When a record was split into units (steps, excerpts), the unit's position. |
| `category` | An active category name of that codebook version, or `other`, or `unclassifiable`. |
| `evidence` | A short verbatim quote from the record supporting the label, chosen by the classifier; when the record was split into excerpts, the excerpt's own quote. `NULL` when there is none, usually for `other` and always for `unclassifiable`. |
| `codebook` | The codebook's name (the file name before `.codebook.json`). |
| `codebook_version` | The version whose categories this label was chosen from. |
| `model` | The model that chose it. |
| `labeled_at` | When. |
| `run_id` | Optional. The `teca_runs` row of the scheduled run that wrote it; `NULL` for labels written outside a run. |

The primary key makes labeling idempotent: relabeling a record under the same version is a
no-op, and a new version adds a row rather than overwriting one. **The latest label** for
a record is the row with the highest `codebook_version`:

```sql
SELECT DISTINCT ON (row_id, unit_index) row_id, unit_index, category, codebook_version
FROM teca_labels WHERE codebook = 'failure-modes'
ORDER BY row_id, unit_index, codebook_version DESC;
```

A failed call writes nothing, so the record is picked up on the next run. Any tool that
writes rows of this shape is a Teca Label-compatible labeler; any query over this shape works
regardless of what wrote it.

## The units table — `teca_units`

Written only when records are split into units (excerpts, steps) before labeling. One row
per unit: the extracted content the classifier read, so a label's `unit_index` can be
resolved back to text. Extraction is cached here — a codebook revision relabels cached
units without re-reading the record.

```sql
CREATE TABLE IF NOT EXISTS teca_units (
    row_id       text        NOT NULL,
    codebook     text        NOT NULL,
    unit_index   int         NOT NULL,
    content      text,
    model        text        NOT NULL,
    extracted_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (row_id, codebook, unit_index)
);
```

| Column | Meaning |
|---|---|
| `row_id`, `codebook` | As in `teca_labels`. |
| `unit_index` | The unit's position, matching `teca_labels.unit_index`. `-1` is a sentinel: the record was read and yielded no units, so it is not re-extracted. |
| `content` | The unit as JSON; for excerpts, `{"quote", ...}`. `NULL` on the sentinel row. |
| `model` | The model that extracted it. |
| `extracted_at` | When. |

## The runs table — `teca_runs`

One row per scheduled run of one codebook: the pipeline's own ledger. The runner opens the
row before any labeling (`status = 'running'`) and closes it exactly once at the end. A row
that stays `running` with no process behind it is a crashed run.

```sql
CREATE TABLE IF NOT EXISTS teca_runs (
    run_id                 text PRIMARY KEY,
    run_at                 timestamptz NOT NULL DEFAULT now(),
    codebook               text NOT NULL,
    codebook_version       int  NOT NULL,
    model                  text NOT NULL,
    status                 text NOT NULL DEFAULT 'running',
    rows_seen              int,
    rows_labeled           int,
    gaps                   int,
    unclassifiable         int,
    other_share            real,
    seconds                real,
    error                  text,
    proposed_for           text
);
```

| Column | Meaning |
|---|---|
| `run_id` | Opaque id; labels written by this run carry it in `teca_labels.run_id`. |
| `run_at` | When the run started. |
| `codebook`, `codebook_version`, `model` | The instrument state that ran. |
| `status` | `running`, then `ok` or `failed`. |
| `rows_seen`, `rows_labeled` | Rows the query returned; rows that received a label. |
| `gaps` | Rows the model failed on after retries (picked up next run). |
| `unclassifiable`, `other_share` | Escape-valve counts for the run. |
| `seconds` | Wall time. |
| `error` | The failure, truncated to 500 characters, on `failed`. |
| `proposed_for` | The story period a proposal written by this run covered; `NULL` otherwise. The runner reads the latest one so an open proposal is never drafted twice. |

Every label joins to the run that wrote it: `teca_labels.run_id = teca_runs.run_id`.

## The benchmark — `<name>.codebook.bench.jsonl`

Optional. One JSON object per line, `{"trace": {...}, "label": "..."}`, with labels a person
confirmed (active category names or `other`). A codebook's categories are frozen, but the
model that applies them is not: a vendor update or a deliberate switch can move the numbers
with no change to a definition. The benchmark is what a model change is measured against
before it is allowed, and the measurement is logged with the change. Keep it small (tens of
rows), keep it committed, and refresh it when the categories change.

## Not part of the format

`<name>.codebook.sample.jsonl` is a cache of the records a codebook was drafted from, with
their labels, kept so review tools can show examples without re-fetching. If it is lost,
`label_sample(traces)` rebuilds it from any rows.
`<name>.codebook.pending.md` is a rendered, human-readable copy of the pending revision.
Neither is needed to read a codebook or its labels.
