---
name: teca-label
description: Drive the Teca Label library from chat — build a codebook from traces, show it, revise it on the record, label a table, watch for drift. Use whenever the user asks to categorize, label, classify, or "find the kinds of" anything in their traces, sessions, tickets, or transcripts.
---

# Teca Label from Claude Code

Teca Label turns a question over a pile of text records into a frozen, versioned set of categories (a **codebook**), then labels every record with one of them. You are the interface: the user asks in prose, you write a short Python script, run it, and show the result as a table. Never make them read raw JSON.

Every step is one script of ~10 lines. Run with `python`. `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) must be in the environment — check `env | grep -i api_key` before the first model call and stop with a clear message if absent.

## 0. Find the traces

Before asking the user anything, look: `*.jsonl`, a `traces`/`sessions` table mentioned in the repo, a Postgres DSN in `.env`. Then load with whichever fits:

```python
from teca_label import read_jsonl, envelope, from_messages
from teca_label.sources import fetch

traces = read_jsonl("sessions.jsonl")                                   # one dict per line
traces = fetch(dsn, "SELECT id, body FROM traces ORDER BY created_at",  # Postgres, sampled in SQL
               to_trace=lambda body: {"text": body}, sample=60)
traces = [envelope(s["id"], s["ts"], from_messages(s["messages"])) for s in sessions]  # chat arrays
```

A trace is any dict; the model sees all of it. `plan()` and `fetch()` refuse or warn on traces with under 50 chars of text and name the empty fields — that's usually the wrong column, not the wrong data; fix `to_trace` before passing `allow_empty=True`. If records are huge (long conversations, generated reports) reach for `excerpts()` — see §6. Only ask the user when you genuinely can't tell which column/file holds the text.

## 0.5. Is the question about a process or an outcome?

A codebook can only label what is *in the text*. If the question contains an outcome word — success, fail, convert, churn, retain, abandon, satisfied — stop and ask: **"what column or event tells you that happened?"** Then:

- Build the codebook on the **process** side ("how do sessions open?", "where does the agent ask for something it already has?").
- Get the outcome as a **column** — a status field, a tool that fired, an export event, a return visit. If the trace already has a proxy (a list of tools called, a final state), derive it from that.
- Finish with the JOIN: process category × outcome column. That table is the answer; a codebook named after an outcome is not.

Never name a codebook after an outcome. "What kinds of interactions are successful?" produces a taxonomy of interaction styles with "success" stapled on — the labels look like an answer but no success was observed.

## 0.7. The design sheet — required, before the first model call

Every choice a build makes is one you would otherwise make by hand after the money is spent. Make them all first, on paper, and get them confirmed. `Codebook.plan()` makes no API call; it prints the sheet:

```python
from teca_label import Codebook
from teca_label.runner import plan_defaults          # models/window/sample from teca-label.toml, if any

plan = Codebook.plan(traces, question="Why do sessions fail?",
                     path="codebooks/failure-modes.codebook.json",
                     rows=48_000,                                # how many rows the table holds, when traces is a sample
                     **plan_defaults())
print(plan)
```

```
plan · Why do sessions fail?
  rows      3,400 seen · 2,100 in window 2026-06-01.. (ts)
  fields    text (98%, ~900 chars) · ts (100%, ~10 chars) · tools (70%, ~8 chars)
  sample    150 of 2,100, spread evenly (1 in 14) → catches themes above ~4%
  models    draft claude-opus-5 · classify claude-fable-5-1
  codebook  codebooks/failure-modes.codebook.json → labels named 'failure-modes'
  cost      draft          ≈ $0.27 · 1 call · ~44k tokens in, 2,000 out · claude-opus-5
            sample labels  ≈ $1.18 · 150 calls · ~103k tokens in, 3,000 out · claude-fable-5-1
            build          ≈ $1.45
            labeling all   ≈ $377.76 · 48,000 rows at ~$0.0079/row (rows=) · claude-fable-5-1
  no API call has been made — plan.build() makes them (prices as of 2026-09-02; ±30% is normal)
```

Show it to the user as a table and ask for a yes. One row per decision, and every row is a decision — say what you chose and why, not just what the print says:

| decision | choice |
|---|---|
| question framing | "Why do sessions fail?" — process side; outcome column is `status` (§0.5) |
| text fields | `text` (~900 chars, 98% filled); `tools` is mostly empty — is that the right column? |
| lookback / row scope | `window="2026-06-01.."` — 2,100 of 3,400 rows; older rows predate the current agent |
| sample size | 150 → themes above ~4%; 60 would only reach ~10% |
| models | draft `claude-opus-5`, classify `claude-fable-5-1` |
| labels table name | `failure-modes` in `teca_labels` |
| cost | build ≈ $1.45 · labeling all 48,000 rows ≈ $378 |

Things to fix on the sheet, not after: a field that is mostly empty (wrong column — go back to `to_trace`), a `⚠` line about thin traces (`build()` will refuse them), a window with too few rows, a full-labeling estimate the user did not expect (a smaller model for `classify` is a `set_model` after a benchmark, §7), an unknown model ("price unknown" — name one from the table in `teca_label.plan.PRICES` or pass `prices=`). If `teca-label.toml` has a `[project]` with `models` / `window` / `sample`, those are the project's standing choices: use them, show them, and don't re-decide them.

Do not call `plan.build()` until the user has confirmed the sheet. Then:

## 1. Build

One question → one codebook file. Name the file after the question, kebab-case, under `codebooks/`:

```python
cb = plan.build()       # the only way to build — one drafting call, then one classify call per sampled row
```

Two shapes:

- **Partition** (`plan().build()`) — every row gets exactly one category; misfits land in `other`.
- **Tag** (`Codebook.tag(name, definition, path)`) — a yes/no detector for something real but rarely a row's main thing ("user asks for a refund").

Pick partition unless the user's question is obviously yes/no. For the recurring case, put the confirmed `models`, `window` and `sample` under `[project]` in `teca-label.toml` so the next build reads them instead of deciding.

## 2. Show it — always, right after any build or revision

The codebook keeps the sample it was drafted from, already labeled. Render it — paste the output **as a markdown table in your reply**:

```python
print(cb.show())
```

| category | definition | n (of 60) |
|---|---|---|
| `tool_loop` | The agent repeats the same tool call ≥3 times without new information. | 14 |
| `other` | — | 4 |

v1 · 60 rows · other 7% (threshold 10%)

`show()` re-labels the sample only when the version moved, so calling it after every revision is free until then. Under the table it prints the checks a reviewer runs every time — repeat them out loud when they fire, and say what to do:

- **A category with zero or near-zero rows** is usually the wrong shape, not wrong. Things that are real but never a row's *main* thing starve in a partition. Say: "`sandbox_timeout` got 0 of 150 — this looks like a tag, not a partition category. Deprecate it here and make it `Codebook.tag(...)`?"
- **Too many categories for the rows.** If categories / rows is above 1:15 (e.g. 8 categories on 60 rows), the tail categories rest on 2–3 examples and won't be stable. Say so; suggest fewer categories (merge) or more rows (`sample=` higher).
- **`other` over threshold** — the draft missed a theme; `cb.exemplars("other", n=8)` shows what.

Offer exemplars for any category the user squints at:

```python
ex = cb.exemplars("tool_loop", n=3)   # reads the sample; rows whose label survives a re-ask
ex["exemplars"]                         # the rows; ex["n_labeled"], ex["agreement"] alongside
```

It returns a dict, not a list. Quote the exemplar text back (trimmed). This is how the user decides a category is real. A codebook that was adopted rather than built has no sample yet — `cb.label_sample(traces)` gives it one.

## 3. Revise on the record — never hand-edit the JSON

When the user says "merge these", "rename that", "this definition is too broad", author a typed revision. The version bumps and the log records it:

```python
from teca_label.core import Revision, Op
cb.apply(Revision(ops=[
    Op(op="merge", names=["retry_storm", "rate_limit_loop"], new_name="rate_limit_cascade",
       definition="An upstream 429 propagates into repeated whole-task retries."),
    Op(op="redefine", name="missing_context", definition="... EXCLUDING rate-limit retries."),
    Op(op="rename", name="misc_fail", new_name="unhandled_error"),
    Op(op="deprecate", name="sandbox_timeout"),
]))
```

Ops: `add`, `rename`, `redefine`, `merge`, `split`, `deprecate`. State the ops in one line before running; re-show the table (§2) after. Deprecated categories stay in the file so old labels remain readable.

Editing `<name>.codebook.json` directly bypasses the log — don't, and say so if the user asks you to.

## 4. Label everything

**In memory** — `cb.classify(traces)` returns a list of strings aligned to the input. `None` means the call failed — rerun fills it; never write it as a label.

**In Postgres** — incremental; only unlabeled rows cost anything:

```python
from teca_label.sources import label_postgres
run = label_postgres(cb, dsn, "SELECT id, body FROM traces", to_trace=lambda body: {"text": body})
print(run)          # labeled 2,097 (other 7%) · 3 failed (pending next run) · rows by version v1: 2,097
run.failed_row_ids  # nothing was written for these; the next call retries them
run.errors          # why, deduplicated
```

Writes `teca_labels(row_id, unit_index, category, evidence, codebook, codebook_version, model, labeled_at, run_id)` next to their data. Report the `LabelRun` line to the user every time, and the errors if any. After a revision, rows labeled under the old version keep their labels by default; pass `on_version_change="relabel_touched"` to re-judge the rows the revision could have moved — every row after a boundary change (add/redefine/split/merge), only the named category's rows after a rename or deprecate — or `"relabel"` for everything. Say which you chose. Show them the JOIN they came for:

```sql
SELECT category, count(*) FROM teca_labels JOIN traces ON traces.id::text = row_id
WHERE codebook = 'failure-modes' GROUP BY category ORDER BY 2 DESC;
```

Then suggest putting the `label_postgres` script on a cron — that's the point.

## 5. Drift

`other` above `other_threshold` (default 10%) means the world moved. Don't rebuild — propose:

```python
rev = cb.propose(new_traces)        # writes codebooks/failure-modes.codebook.pending.json
```

Show the ops as a list ("add `rate_limit_cascade` — evidence rows 3, 17, 41"), let the user strike any, then `cb.apply()` with no argument consumes the pending file. `cb.history()` prints the whole story — show it when they ask "what changed".

Deeper, still propose-only: `cb.audit(traces, labels)` (is a category hiding two themes?), `cb.drill(category, traces, labels)` (child codebook inside one category).

**Digging into one category (a child codebook).** Count the category's labeled rows first — the child can only be as good as its population:

- **≥ 100 rows labeled** with the category → drill directly: `cb.drill(category, traces, labels)`.
- **Fewer, and the corpus is big** → harvest to quota first. Classify strided batches and keep the hits until ~60 rows carry the category, then drill from those. Rows to classify ≈ 60 / the category's share; write a small loop:

```python
from teca_label.sources import fetch, label_postgres
hits, wave = [], 240
while len(hits) < 60:                      # cap the loop at a sane budget, e.g. 4,000 rows
    batch = fetch(dsn, query, to_trace, sample=wave)
    labels = cb.classify(batch)
    hits += [t for t, l in zip(batch, labels) if l == category]
    wave *= 2
child = cb.drill(category, hits, [category] * len(hits))
```

  Better: run the wave through `label_postgres` instead of bare `classify` so every label persists — a later full sweep skips those rows automatically. Nothing is classified twice.
- **Share under ~1%** → a drill is the wrong tool; suggest a tag gate (cheap yes/no over everything) and drill the yeses later.

Tell the user what the harvest measured (share, rows classified) — it's a tighter prevalence estimate than the draft sample gave.

## 6. Long records

A 200-turn session labeled whole smashes everything together. Extract question-relevant excerpts first; labels attach to excerpts with the quote as evidence:

```python
from teca_label.units import excerpts
label_postgres(cb, dsn, "SELECT id, conversation FROM sessions",
               to_trace=lambda c: {"conversation": c},
               units=excerpts("Where did the user hit friction?"))
```

Extractions are cached — revising the codebook never re-reads a session.

## 7. Changing the model

Fixed categories don't make two runs comparable on their own: the model can change, by choice or by the vendor. Every label records its model; every model change is measured. Keep a small human-labeled benchmark beside the codebook and switch through it:

```python
traces, labels = cb.sample, cb.sample_labels       # after the user has reviewed show() and corrected any labels
cb.bench(traces, labels)                            # <name>.codebook.bench.jsonl, logged
cb.measure()                                        # agreement of the current model with the humans
cb.set_model("classify", "claude-haiku-4-5")        # measures the candidate; refuses under policy.min_agreement (80%)
```

Report the agreement and the top disagreements. Never switch the classify model without the benchmark — `set_model` won't let you, and neither should you.

## Conventions

- The file and table shapes are specified in `FORMAT.md` at the repo root — read it when you need to hand-write a codebook or a pending revision, or query the labels table directly.
- Commit `*.codebook.json` and `*.codebook.log.jsonl`; `*.pending.json` is a review queue, not history; `*.sample.jsonl` is the build sample with its labels (what `show()` reads) — fine either way.
- Category names are identifiers: snake_case ASCII under 40 chars. Drafts are normalized; a human `Op(op="add", name="Tool loop")` is refused with the slug to use instead.
- The DSN is a plain argument — read it from the environment, never print it, never write it to a file.
- One codebook = one question. A second question is a second file, not more categories in the first.
- Tell the user what a step will cost before running it: the plan prints the build and full-labeling estimate in dollars (§0.7); labeling is one small call per new record; `exemplars` is up to 20 re-asks.
- Don't offer confidence scores — there are none by design. Offer exemplars.
