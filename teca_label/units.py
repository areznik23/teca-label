"""The unit layer: the unit of labeling is the smallest thing that means one thing.

Three strategies, chosen per codebook:
- identity: the record already means one thing (blocks, tickets, verbatims). Default.
- structural: split on seams the data already has (conversation turns, OTel spans,
  tool calls) — any plain callable record -> list of unit dicts.
- semantic: an LLM extracts question-relevant excerpts (call transcripts, long docs).
  Question-scoped: a pricing codebook and a competitor codebook extract different units from
  the same document. A record with no relevant content yields [] — itself a signal
  (topic penetration rate).

A units function that calls a model must carry a `.model` attribute naming it — that
attribute, not configuration elsewhere, is the provenance recorded with the units
(deterministic ones carry the literal "deterministic")."""
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable

from pydantic import BaseModel

from .providers import ConfigurationError, Provider, is_configuration_error, as_provider, parse

Units = Callable[[dict], list[dict]]


class Excerpt(BaseModel):
    quote: str
    speaker: str = ""


class _Excerpts(BaseModel):
    excerpts: list[Excerpt]


def identity(record: dict) -> list[dict]:
    """The record is already the unit."""
    return [record]


def excerpts(question: str, model: str | None = None,
                      max_tokens: int = 8192, client: Provider | None = None) -> Units:
    """Build a units function that extracts question-relevant excerpts from long documents.

    Extraction is the expensive read (the whole document passes through the model),
    so pair it with a persistent units cache — see sources.label_postgres(units=...).
    `client` is BYO as on Codebook. The returned callable carries `.model` (its
    provenance) and `.usage` (a Counter of tokens/calls it has spent). With
    model=None, label_postgres fills in the codebook's 'extract' role at run time,
    so set_model("extract", ...) governs extraction; a bare call needs model=."""
    provider = as_provider(client) if client is not None else None
    system = (
        f"Extract every segment of this document that is relevant to the question: {question}\n"
        "Quote only the essential sentences of each segment — trim aggressively; never quote "
        "more than ~50 words per excerpt. Return at most the 12 most significant excerpts. "
        "Attribute the speaker when the document identifies one. If nothing in the document "
        "is relevant, return an empty list — do not stretch.")
    lock = threading.Lock()

    def unitize(record: dict) -> list[dict]:
        if unitize.model is None:
            raise ConfigurationError("excerpts() has no model — pass model=, or run it through "
                                     "label_postgres, which supplies the codebook's 'extract' role")
        parsed, usage = parse(unitize.model, system, json.dumps(record, default=str), _Excerpts,
                              max_tokens, timeout=600.0, provider=provider)
        with lock:
            unitize.usage.update({**usage, "calls": 1})
        return [{"quote": e.quote, **({"speaker": e.speaker} if e.speaker else {})}
                for e in parsed.excerpts]

    unitize.model = model
    unitize.usage = Counter()
    return unitize


def steps(steps: str | Callable[[dict], list] = "steps",
                  context: int = 1) -> Units:
    """Cut an agent trace into per-step units — the structural units function for trajectory
    questions ("where does the agent loop / skip / choose the wrong tool?").

    `steps` is the field holding the trace's step list, or a callable returning it.
    Each unit is one step plus a window of the `context` steps before it, because a
    step judged in isolation is unjudgeable ("called search" — fine, or redundant?).
    Deterministic: no model call, no cost — labels land per step with unit_index as
    the step's position."""
    get_steps = steps if callable(steps) else (lambda record: record.get(steps) or [])

    def unitize(record: dict) -> list[dict]:
        sequence = list(get_steps(record))
        units = []
        for position, step in enumerate(sequence):
            unit = dict(step) if isinstance(step, dict) else {"step": step}
            unit["step_index"], unit["n_steps"] = position, len(sequence)
            if context and position:
                unit["context_before"] = sequence[max(0, position - context):position]
            units.append(unit)
        return units

    unitize.model = "deterministic"
    return unitize


def unitize(records: Iterable[tuple[str, dict]], units: Units = identity,
                   workers: int = 8, retries: int = 1) -> tuple[list[tuple[str, int, dict]], dict[str, str]]:
    """Fan (record_id, record) pairs out into (record_id, unit_index, unit) triples.

    Returns (triples, failed): a record whose extraction keeps failing is reported
    ({record_id: "ExcType: message"}), not raised — the caller decides whether to retry
    it on a later run."""
    pairs = list(records)

    def unitize_one(pair):
        for _ in range(max(retries, 0) + 1):
            try:
                return units(pair[1])
            except Exception as exc:
                if is_configuration_error(exc):
                    raise
                last_error = exc
        return last_error

    with ThreadPoolExecutor(workers) as pool:
        per_record = list(pool.map(unitize_one, pairs))
    triples, failed = [], {}
    for (record_id, _), units in zip(pairs, per_record):
        if isinstance(units, Exception):
            failed[record_id] = f"{type(units).__name__}: {units}"
        else:
            triples.extend((record_id, index, unit) for index, unit in enumerate(units))
    return triples, failed
