# The Teca Label project: standing codebooks on a schedule, revisions by diff

A **Teca Label project** is to Teca Label what a dbt project is to dbt: a directory —
usually its own repo — holding the instruments, their config, and the schedule
that runs them. The app that produces the traces never changes; it reads labels
with a JOIN. Taxonomy changes happen here, as reviewable diffs.

```
acme-codebooks/
  codebooks/
    support-drivers.codebook.json       # the instruments (committed)
    support-drivers.codebook.log.jsonl  # their append-only histories (committed)
  teca-label.toml                          # which codebook reads what, how often
  .github/workflows/run-codebooks.yml   # the scheduler
```

`*.codebook.pending.json` / `*.codebook.pending.md` are scratch — gitignore them; a proposal's
durable form is the branch the scheduler opens. The tick also records the period it proposed for
in `teca_runs`, so a fresh checkout that lacks the pending file still knows a proposal is open
and does not draft it again.

## teca-label.toml

```toml
[project]                 # optional defaults for every codebook
dsn_env = "WAREHOUSE_DSN" # env var NAME — the DSN itself never goes in a file
cadence = "week"          # day | week | month | quarter
window = "2026-04-20.."   # the choices a build makes, made once — every
sample = 150              # plan in this project reads them (see below)
[project.models]
draft = "claude-opus-5"
classify = "claude-fable-5-1"

[codebooks.support-drivers]
path = "codebooks/support-drivers.codebook.json"
query = """
  SELECT id, created_at,
         jsonb_build_object('title', title, 'body', body) AS trace
  FROM tickets ORDER BY created_at
"""
```

| key | default | meaning |
|---|---|---|
| `path` | required | path to the codebook file (draft it once with `Codebook.plan(...).build()`) |
| `query` | required | returns exactly **(id, ts, trace)** — see below |
| `dsn_env` | `TECA_LABEL_DSN` | env var holding the Postgres DSN (labels always land here) |
| `cadence` | `week` | the story's period size for drift detection |
| `labels_table` | `teca_labels` | where labels land |
| `runs_table` | `teca_runs` | the run ledger `teca-label status` reads |
| `name` | the table key | this codebook's name inside the labels table |
| `workers` | `8` | classify concurrency |
| `on_version_change` | `relabel_touched` | what a tick does with rows labeled under an older version: `keep`, `relabel`, or `relabel_touched` (every row after a boundary change; only the named category's rows after a rename or deprecate) |

Three more `[project]` keys are not for the tick at all. `models`, `window`
and `sample` are the choices a build makes — which model drafts and which
labels, which era the categories describe, how many rows to draft from — kept
in the file so nobody has to make them again, and nothing decides them silently:

```python
from teca_label import Codebook
from teca_label.runner import plan_defaults

plan = Codebook.plan(traces, question, path="codebooks/x.codebook.json", **plan_defaults())
print(plan)     # rows, fields, sample, models and estimated cost — before any call
cb = plan.build()
```

`plan_defaults()` returns only the keys the file sets (`{}` with no file), so the
call reads the same whether or not a project has made its choices yet. After the
build the codebook file owns its models. Changing the classify model later is
`cb.set_model("classify", ...)`, which measures the candidate against the codebook's
human-labeled benchmark (`cb.bench(traces, labels)`) and refuses below
`policy.min_agreement`; the change and its measurement land in the log.

**The (id, ts, trace) convention** is what makes a codebook declarable in config
alone: id first (the labels-table join key), ts second (data time — drift
buckets on it), trace third (a JSON object via `jsonb_build_object`/`to_jsonb`,
or plain text). Shape the trace in SQL. Codebooks needing a Python `to_trace` or
`excerpts()` stay as scripts calling `label_postgres` directly.

## What a tick does (`teca-label run`)

1. **Label eagerly** — every row the query returns that no version has judged.
   A call that fails writes nothing, so the row is pending on the next tick. After
   an accepted revision, `on_version_change` says what happens to the rest; the
   default `relabel_touched` re-judges the rows the revision could have moved
   (`teca_label.core.relabel_targets`, read from the log): every row below the
   live version after a boundary change (add, redefine, split, merge — a new
   category can pull rows out of any old one), only the named category's rows
   after a rename or deprecate.
2. **Rebuild the story from the warehouse** — each unit under its latest label,
   bucketed by data time. The wall-clock-current period is skipped until it
   holds `policy.min_batch` rows (a Monday tick must not read a 40-row week as
   drift). Ticks are stateless; re-running one is always safe.
3. **Read the signals** — emergence / absorption / decay (`drift_signals`),
   plus the audit pass every `policy.audit_every` periods.
4. **Propose, never apply** — when signals clear policy, the tick writes
   `<name>.codebook.pending.json` (the machine diff) and `<name>.codebook.pending.md` (ops with
   their evidence quoted, frozen at propose time — the PR body), and `teca-label
   run` exits **10** — also whenever an earlier proposal still awaits review. Otherwise
   exit 0. The codebook on disk never moves inside a tick.

Accepting is `teca-label apply --codebook <name>` — after editing the pending file, or
not. `teca-label status` shows every codebook's version, revision count, and whether a
proposal awaits.

## The scheduler (revisions as PRs)

```yaml
# .github/workflows/run-codebooks.yml
name: run codebooks
on:
  schedule: [{cron: "0 6 * * *"}]
  workflow_dispatch:
concurrency: {group: codebooks}          # ticks are single-writer per project
permissions: {contents: write, pull-requests: write}

jobs:
  tick:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: pip install 'teca-label[postgres] @ git+https://github.com/areznik23/teca-label'
      - name: run codebooks
        id: run
        env:
          WAREHOUSE_DSN: ${{ secrets.WAREHOUSE_DSN }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          set +e
          teca-label run
          code=$?
          [ $code -eq 10 ] && echo "proposed=true" >> "$GITHUB_OUTPUT" && exit 0
          exit $code
      - name: open a revision PR
        if: steps.run.outputs.proposed == 'true'
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          for pending in codebooks/*.codebook.pending.json; do
            [ -e "$pending" ] || continue
            name=$(basename "$pending" .codebook.pending.json)
            branch="codebook/$name-$(date +%Y%m%d)"
            body=$(cat "codebooks/$name.codebook.pending.md")
            git checkout -b "$branch"
            teca-label apply --codebook "$name"
            git add "codebooks/$name.codebook.json" "codebooks/$name.codebook.log.jsonl"
            git -c user.name=teca-label -c user.email=bot@teca-label commit -m "$name: proposed revision"
            git push -u origin "$branch"
            gh pr create --title "$name: codebook revision" --body "$body"
            git checkout -
          done
```

**Merge = accept**: the codebook at main moves to v+1, and the next scheduled
tick relabels what the revision could have moved (every row after a boundary change;
only the named category's rows after a rename or deprecate). **Close = reject**: main never moved.
Review lives entirely in the PR — the diff of the codebook file, the evidence
in the body, your team's normal approval rules.

## Reading labels

Labels append per codebook version — honest history, never rewritten. Give BI
the current view:

```sql
CREATE VIEW teca_labels_current AS
SELECT DISTINCT ON (codebook, row_id, unit_index) *
FROM teca_labels
ORDER BY codebook, row_id, unit_index, codebook_version DESC;
```

Then everything is one JOIN: `... JOIN teca_labels_current f ON f.row_id =
t.id AND f.codebook = 'support-drivers'`.
