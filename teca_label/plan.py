"""The plan a build requires: what it will read, how it will sample, which models it
will run, and what that costs — printed before a single call is spent.

There is no silent default to charge ahead with. `Codebook.plan(...)` inspects the
traces (no API call) and returns a Plan; `plan.build()` drafts the codebook. Whether the driver is a person at a REPL, an agent, or a script, the
choices are visible on the way in: question, rows in scope, the fields the model
will read, sample size, model per role, the file the codebook lands in, and an
estimate for the draft, for labeling the sample, and for labeling every row.

Estimates are estimates: tokens at ~4 characters each (the models in
_DENSER_TOKENIZER tokenize ~30% more tokens per character, folded in), prices as of
PRICES_AS_OF. Pass `prices=` to override or extend the table."""
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from .core import (DEFAULT_MODELS, DEFAULT_POLICY, MIN_TRACE_CHARS, _check_traces, _coerce_dt, _sample, content,
                   text_chars, thin_traces)

PRICES_AS_OF = "2026-09-02"

# USD per million tokens (input, output), standard tier. Matched by longest prefix,
# so dated ids (claude-haiku-4-5-20251001) find their family.
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.0, 50.0),
    "claude-mythos-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-haiku-4": (1.0, 5.0),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-pro": (15.0, 120.0),
    "gpt-5": (1.25, 10.0),
    "gpt-5.1": (1.25, 10.0),
    "gpt-5.2": (1.75, 14.0),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4": (2.50, 15.0),
    "gpt-5.5": (5.0, 30.0),
}
_DENSER_TOKENIZER = ("claude-fable", "claude-mythos", "claude-opus-5", "claude-opus-4-7",
                     "claude-opus-4-8", "claude-sonnet-5")
_CHARS_PER_TOKEN = 4
_DRAFT_PROMPT_TOKENS = 700      # the reviser's instructions around the sample
_DRAFT_OUTPUT_TOKENS = 2_000    # 6-10 categories with definitions and evidence
_LABEL_PROMPT_TOKENS = 400      # the codebook's definitions in the classify prompt
_LABEL_OUTPUT_TOKENS = 60       # one enum value plus a short evidence quote


def price(model: str, prices: dict[str, tuple[float, float]] | None = None
          ) -> tuple[float, float] | None:
    table = {**PRICES, **(prices or {})}
    hits = [prefix for prefix in table if model.startswith(prefix)]
    return table[max(hits, key=len)] if hits else None


def tokens(chars: int, model: str) -> int:
    factor = 1.3 if model.startswith(_DENSER_TOKENIZER) else 1.0
    return int(chars / _CHARS_PER_TOKEN * factor)


def parse_window(window) -> tuple | None:
    """A window is (start, end) with either bound None, or the string form
    "2026-04-20..", "..2026-06-01", "2026-04-20..2026-06-01"."""
    if window is None or isinstance(window, tuple):
        return window
    if isinstance(window, str) and ".." in window:
        start, end = window.split("..", 1)
        return (start or None, end or None)
    raise ValueError(f"window must be (start, end) or 'start..end' (either side may be "
                     f"empty), got {window!r}")


@dataclass
class Estimate:
    """One stage's cost: calls, tokens in and out, and dollars when the model's price is known."""
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    usd: float | None

    def __str__(self) -> str:
        dollars = f"≈ ${self.usd:,.2f}" if self.usd is not None else "price unknown"
        return (f"{dollars} · {self.calls:,} call{'s' if self.calls != 1 else ''} · "
                f"~{_k(self.input_tokens)} tokens in, {_k(self.output_tokens)} out · {self.model}")


def _k(n: int) -> str:
    return f"{n:,}" if n < 10_000 else (f"{n / 1_000:,.0f}k" if n < 10_000_000 else f"{n / 1e6:,.1f}M")


@dataclass
class Plan:
    """Everything build() is about to do, decided and visible. Made by Codebook.plan()."""
    codebook_cls: Any
    question: str
    traces: list[dict]                  # the rows in scope (inside the window, if any)
    sample: int
    models: dict[str, str]
    path: Path | None
    min_evidence: int = 3
    min_chars: int = MIN_TRACE_CHARS
    allow_empty: bool = False
    label_sample: bool = True
    parent: dict | None = None
    window: tuple | None = None
    time_key: Any = None
    client: Any = None
    rows: int | None = None             # rows the codebook will label in full, if not len(traces)
    n_seen: int = 0                     # rows handed in, before the window
    prices: dict | None = None
    policy: dict | None = None
    fields: list[tuple[str, float, int]] = field(default_factory=list)   # (name, filled share, mean chars)
    thin: str | None = None
    estimate: dict[str, Estimate] = field(default_factory=dict)

    def __post_init__(self):
        self.models = {**DEFAULT_MODELS, **(self.models or {})}
        unknown = set(self.policy or {}) - set(DEFAULT_POLICY)
        if unknown:
            raise ValueError(f"unknown policy keys {sorted(unknown)} — known: {sorted(DEFAULT_POLICY)}")
        self.path = Path(self.path) if self.path else None
        self.thin = thin_traces(self.traces, self.min_chars)
        self.fields = _fields(self.traces)
        self.estimate = self._estimate()

    # ---- what it costs ----

    @property
    def sampled(self) -> list[dict]:
        return _sample(self.traces, self.sample)

    @property
    def labeling_rows(self) -> int:
        return self.rows if self.rows is not None else len(self.traces)

    def _estimate(self) -> dict[str, Estimate]:
        sampled = self.sampled
        chars = [len(json.dumps(t, default=str)) for t in sampled]
        mean_chars = int(mean(chars)) if chars else 0
        draft_model, label_model = self.models["draft"], self.models["classify"]

        def stage(model, calls, input_tokens, output_tokens):
            rate = price(model, self.prices)
            usd = None if rate is None else (input_tokens * rate[0] + output_tokens * rate[1]) / 1e6
            return Estimate(model, calls, input_tokens, output_tokens, usd)

        draft = stage(draft_model, 1, tokens(sum(chars), draft_model) + _DRAFT_PROMPT_TOKENS,
                      _DRAFT_OUTPUT_TOKENS)
        per_label_in = tokens(mean_chars, label_model) + _LABEL_PROMPT_TOKENS
        sample_labels = stage(label_model, len(sampled) if self.label_sample else 0,
                              per_label_in * (len(sampled) if self.label_sample else 0),
                              _LABEL_OUTPUT_TOKENS * (len(sampled) if self.label_sample else 0))
        full = stage(label_model, self.labeling_rows, per_label_in * self.labeling_rows,
                     _LABEL_OUTPUT_TOKENS * self.labeling_rows)
        return {"draft": draft, "sample_labels": sample_labels, "labeling": full}

    @property
    def build_usd(self) -> float | None:
        parts = [self.estimate["draft"].usd, self.estimate["sample_labels"].usd]
        return None if any(p is None for p in parts) else sum(parts)

    # ---- how it reads ----

    def __str__(self) -> str:
        n = len(self.traces)
        rows = f"{self.n_seen:,} seen"
        if self.window is not None:
            start, end = self.window
            span = f"{start or ''}..{end or ''}"
            rows += f" · {n:,} in window {span} ({self.time_key if not callable(self.time_key) else 'time_key'})"
        if self.thin:
            rows += (f"\n              ⚠ {self.thin} — build() refuses them; fix to_trace, filter them, "
                     f"or pass allow_empty=True" if not self.allow_empty
                     else f"\n              {self.thin} — included (allow_empty=True)")
        fields = " · ".join(f"{name} ({share:.0%}{f', ~{chars:,} chars' if chars else ''})"
                            for name, share, chars in self.fields[:8]) or "—"
        if len(self.fields) > 8:
            fields += f" · +{len(self.fields) - 8} more"
        k = min(self.sample, n)
        stride = max(1, n // self.sample) if self.sample else 1
        reach = 2 * self.min_evidence / k if k else 1
        sample = (f"{k:,} of {n:,}, spread evenly" + (f" (1 in {stride})" if stride > 1 else "")
                  + f" → catches themes above ~{reach:.0%}")
        models = " · ".join(f"{role} {model}" for role, model in self.models.items()
                            if role in ("draft", "classify"))
        if self.path:
            name = self.path.name.removesuffix(".json").removesuffix(".codebook")
            where = f"{self.path} → labels named '{name}'"
        else:
            where = "in memory (path=None) — nothing saved"
        est = self.estimate
        cost = [f"draft          {est['draft']}"]
        if self.label_sample:
            cost.append(f"sample labels  {est['sample_labels']}")
        build = f"≈ ${self.build_usd:,.2f}" if self.build_usd is not None else "price unknown"
        cost.append(f"build          {build}")
        full = est["labeling"]
        per_row = (f" at ~${full.usd / full.calls:,.4f}/row" if full.usd is not None and full.calls
                   else "")
        rows_note = f"{self.labeling_rows:,} rows{per_row}" + ("" if self.rows is None else " (rows=)")
        dollars = f"≈ ${full.usd:,.2f}" if full.usd is not None else "price unknown"
        cost.append(f"labeling all   {dollars} · {rows_note} · {full.model}")
        lines = [f"plan · {self.question}",
                 f"  rows      {rows}",
                 f"  fields    {fields}",
                 f"  sample    {sample}",
                 f"  models    {models}",
                 f"  codebook  {where}",
                 "  cost      " + "\n            ".join(cost),
                 f"  no API call has been made — plan.build() makes them "
                 f"(prices as of {PRICES_AS_OF}; ±30% is normal)"]
        return "\n".join(lines)

    # ---- the only way to build ----

    def build(self):
        """Draft the codebook this plan describes: one drafting call over the sample,
        then (unless label_sample=False) one classify call per sampled row so the
        codebook keeps its labeled sample. Returns the Codebook."""
        if self.thin and not self.allow_empty:
            raise ValueError(f"{self.thin}. Pass allow_empty=True to include them, "
                             f"or filter them out first")
        if len(self.traces) < self.min_evidence:
            raise ValueError(f"{len(self.traces)} traces is not enough to draft from — "
                             f"check the query or file before building")
        if self.path and self.path.exists():
            raise FileExistsError(f"{self.path} already exists — a build never overwrites a "
                                  f"codebook (its labels are pinned to it). Codebook.load() it "
                                  f"and revise, or choose another path")
        parent = self.parent
        if parent is not None and "pinned_at" not in parent:
            parent = {**parent, "pinned_at": datetime.now(timezone.utc).isoformat()}
        cb = self.codebook_cls(self.question, models=self.models, path=self.path,
                               client=self.client, parent=parent, policy=self.policy)
        sampled = self.sampled
        if self.window is not None:
            start, end = self.window
            cb.draft_window = {"start": None if start is None else str(start),
                               "end": None if end is None else str(end),
                               "n_sampled": len(sampled)}
        draft = cb.propose(sampled, min_evidence=self.min_evidence)
        if not draft.ops:
            raise ValueError("the draft produced no categories — the sample may not bear on the "
                             "question; check the fields the plan shows and the question, or "
                             "build with more rows")
        cb.apply(draft)
        if self.label_sample:
            cb.label_sample(sampled)
        return cb


def _fields(traces: list[dict]) -> list[tuple[str, float, int]]:
    """Top-level fields across the traces: how often each is filled with text, and how
    much — the model reads all of it, so this is what 'the text' means for this corpus."""
    if not traces:
        return []
    chars: dict[str, list[int]] = {}
    for trace in traces:
        for key, value in content(trace).items():
            chars.setdefault(key, []).append(text_chars(value))
    out = []
    for key, counts in chars.items():
        filled = [c for c in counts if c > 0]
        out.append((key, len(filled) / len(traces), int(mean(filled)) if filled else 0))
    return sorted(out, key=lambda f: (-f[1] * f[2], f[0]))


def scope(traces: list[dict], window, time_key, min_evidence: int) -> tuple[list[dict], tuple | None, Any]:
    """The rows a plan is over: all of them, or those inside the window. A window with
    no time_key reads the envelope's 'ts' when every trace has one."""
    _check_traces(traces)
    bounds = parse_window(window)
    if bounds is None:
        return list(traces), None, time_key
    if time_key is None:
        if traces and all("ts" in t for t in traces):
            time_key = "ts"
        else:
            raise ValueError("window requires time_key (field name or callable) — "
                             "traces have no 'ts' field to read")
    get_time = time_key if callable(time_key) else (lambda trace: trace[time_key])
    start = _coerce_dt(bounds[0]) if bounds[0] else None
    end = _coerce_dt(bounds[1]) if bounds[1] else None
    inside = [trace for trace in traces
              if (start is None or _coerce_dt(get_time(trace)) >= start)
              and (end is None or _coerce_dt(get_time(trace)) <= end)]
    if len(inside) < min_evidence:
        raise ValueError(f"only {len(inside)} traces inside the window — widen it")
    return inside, bounds, time_key
