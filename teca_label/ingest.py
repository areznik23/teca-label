"""Bringing traces in: the envelope contract and the one adapter everyone needs.

Every source — chat sessions, call transcripts, agent event streams — reduces to
one envelope per record:

    {"id": str, "ts": ISO-8601, "summary": str, "trace": dict | str, "apps": [str]?}

`trace` is what the model reads; `summary` is what a human scans; `id` and `ts`
are what a labels table joins and trends on. The engine itself stays
format-agnostic (Codebook takes plain dicts), and an envelope handed to it
contributes only its `trace` to the model (core.content) — id, ts, and summary
are for joining and scanning. JSONL files of envelopes are the file-mode input.

from_messages() is the only format adapter that lives in the library, because it
encodes a judgment users shouldn't have to make twice: in a provider message
array, intent lives in the user messages, the assistant's words, and the NAMES
of the tools it reached for — while tool results dominate the tokens and carry
almost none of it. Third-party export schemas (Langfuse, LangSmith, OTel) churn;
their recipes are one short mapping each, kept out of the library.
"""
import json
from pathlib import Path
from typing import Iterable

from .core import _coerce_dt


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "…"


def envelope(id, ts, trace, summary: str | None = None,
             apps: list[str] | None = None) -> dict:
    """Normalize one record to the envelope. Bad inputs fail here, loudly —
    an unparseable timestamp should die at ingest, not deep inside a tick."""
    if trace is None or trace == "":
        raise ValueError("trace is required — it is the content the model reads")
    rid = str(id).strip()
    if not rid:
        raise ValueError("id is required and must be non-empty")
    try:
        iso = _coerce_dt(ts).isoformat()
    except Exception:
        raise ValueError(f"unparseable ts {ts!r} — pass ISO-8601 or a datetime") from None
    if summary is None:
        blob = trace if isinstance(trace, str) else json.dumps(trace, default=str, ensure_ascii=False)
        summary = _clip(" ".join(blob.split()), 140)
    row = {"id": rid, "ts": iso, "summary": summary, "trace": trace}
    if apps:
        row["apps"] = list(apps)
    return row


def _text_of(content) -> str:
    """Plain text of a message body: a string, or a provider content-part list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content
                        if isinstance(part, dict) and part.get("type") in (None, "text")).strip()
    return ""


def from_messages(messages: list[dict], keep_tool_results: bool = False,
                  max_result_chars: int = 200, max_chars: int = 12_000) -> dict:
    """Flatten a provider message array (OpenAI or Anthropic shape) into a trace.

    Handles both wire shapes: OpenAI puts tool results in role="tool" messages and
    tool calls on the assistant message; Anthropic nests tool_use blocks in
    assistant content and tool_result blocks inside user turns.

    What survives: user text, assistant text, tool CALLS (name + clipped args).
    What doesn't, by default: system prompts (identical across a corpus — pure
    noise to a classifier) and tool results (the token bulk; the call line
    already records that the tool ran). Over max_chars, the head and tail are
    kept and the middle elided — the ask opens a trace, the outcome closes it.
    Returns {"messages": [...], "n_messages": N}; wrap it with envelope()."""
    lines: list[dict] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        content_blocks = content if isinstance(content, list) else []
        if role == "system":
            continue
        if role == "tool":
            if keep_tool_results:
                lines.append({"tool_result": _clip(_text_of(content), max_result_chars)})
            continue
        if role == "user":
            for block in content_blocks:
                if (isinstance(block, dict) and block.get("type") == "tool_result"
                        and keep_tool_results):
                    lines.append({"tool_result": _clip(_text_of(block.get("content")),
                                                       max_result_chars)})
            text = _text_of(content)
            if text:
                lines.append({"user": text})
        elif role == "assistant":
            text = _text_of(content)
            if text:
                lines.append({"assistant": text})
            for call in message.get("tool_calls") or []:
                function_call = call.get("function", {})
                lines.append({"tool_call": function_call.get("name", ""),
                              "args": _clip(str(function_call.get("arguments", "")),
                                            max_result_chars)})
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    lines.append({"tool_call": block.get("name", ""),
                                  "args": _clip(json.dumps(block.get("input", {}), default=str),
                                                max_result_chars)})
    return {"messages": _fit(lines, max_chars), "n_messages": len(messages)}


def _fit(lines: list[dict], max_chars: int) -> list[dict]:
    """Head + tail under budget, middle elided with a marker that says how much."""
    sizes = [len(json.dumps(line, ensure_ascii=False)) for line in lines]
    if sum(sizes) <= max_chars:
        return lines
    head_budget, tail_budget = int(max_chars * .6), int(max_chars * .4)
    head: list[dict] = []
    used = 0
    for line, size in zip(lines, sizes):
        if used + size > head_budget:
            break
        head.append(line)
        used += size
    tail: list[dict] = []
    used = 0
    for line, size in zip(reversed(lines[len(head):]), reversed(sizes[len(head):])):
        if used + size > tail_budget:
            break
        tail.append(line)
        used += size
    tail.reverse()
    omitted = len(lines) - len(head) - len(tail)
    marker = [{"omitted": f"{omitted} messages"}] if omitted else []
    return head + marker + tail


def events(messages: list[dict]) -> list[tuple[str, str]]:
    """A session as an ordered event stream: ("user", text) and ("tool", name).

    The primitive under sequence questions — "what preceded tool X", "what made
    the agent redo X". Both wire shapes (OpenAI tool_calls, Anthropic tool_use);
    system prompts and tool results don't make events."""
    stream: list[tuple[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "user":
            text = _text_of(content)
            if text:
                stream.append(("user", text))
        elif role == "assistant":
            for call in message.get("tool_calls") or []:
                stream.append(("tool", call.get("function", {}).get("name", "")))
            for block in (content if isinstance(content, list) else []):
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    stream.append(("tool", block.get("name", "")))
    return stream


def tool_firings(messages: list[dict], tool: str, *, session_id, ts,
                 max_chars: int = 2500) -> list[dict]:
    """One envelope per firing of `tool`, carrying the user ask that was live
    when it fired. The trigger-analysis slice: classify these and you know what
    people are asking for at the moment the agent reaches for the tool.
    Firings before any user text are skipped. Ids are deterministic
    (session:n), so reruns never double-label."""
    rows, last_ask, n = [], None, 0
    for kind, value in events(messages):
        if kind == "user":
            last_ask = value
        elif value == tool and last_ask:
            n += 1
            rows.append(envelope(f"{session_id}:{n}", ts,
                                 {"tool": tool, "preceding_ask": _clip(last_ask, max_chars)}))
    return rows


def revision_loops(messages: list[dict], tools: list[str] | None = None, *,
                   session_id, ts, max_chars: int = 2500) -> list[dict]:
    """One envelope per redo: the same tool fired again after the user spoke,
    carrying what the user said in between. The revision slice: classify these
    and you know whether a tool's re-runs are dialogue or corrections.
    A tool re-firing with no user text between is the agent's own doing and
    makes no row. Ids are deterministic (session:tool:k)."""
    rows, last_fired_at, k = [], {}, 0
    stream = events(messages)
    for i, (kind, value) in enumerate(stream):
        if kind != "tool" or (tools is not None and value not in tools):
            continue
        if value in last_fired_at:
            between = [v for kk, v in stream[last_fired_at[value] + 1:i] if kk == "user"]
            if between:
                k += 1
                rows.append(envelope(f"{session_id}:{value}:{k}", ts,
                                     {"tool_rerun": value,
                                      "user_reply": _clip("\n".join(between), max_chars)}))
        last_fired_at[value] = i
    return rows


def read_jsonl(path: str | Path) -> list[dict]:
    """One envelope (or any dict) per line; blank lines tolerated."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> Path:
    p = Path(path)
    p.write_text("".join(json.dumps(row, default=str, ensure_ascii=False) + "\n" for row in rows))
    return p
