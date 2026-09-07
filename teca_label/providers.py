"""The model seam. Every call Teca Label makes is one structured-output request:
(model, system, content, schema) -> (an instance of schema, token usage). A Provider is
anything that answers that request; the rest of the library never sees a vendor SDK.

Resolution is explicit: a client you pass wins; with none, the model name picks the
family (claude-* -> Anthropic, gpt-*/o* -> OpenAI) and the SDK reads its own env var.
Any OpenAI-compatible endpoint is `OpenAIProvider(base_url=..., api_key=...)`."""
import os
import threading
from typing import Any, Protocol

from pydantic import BaseModel

Usage = dict[str, int]


class ConfigurationError(ValueError):
    """The call could never succeed as configured: no key, no provider for the model,
    a rejected credential. Raised straight through — never retried, never a gap."""


def is_configuration_error(exc: BaseException) -> bool:
    """Errors a retry cannot fix: our own ConfigurationError, a missing SDK, and a
    vendor 401/403 (both SDKs put the HTTP status on the exception)."""
    return (isinstance(exc, (ConfigurationError, ImportError))
            or getattr(exc, "status_code", None) in (401, 403))


class Provider(Protocol):
    def parse(self, model: str, system: str, content: str, schema: type[BaseModel],
              max_tokens: int, timeout: float) -> tuple[BaseModel, Usage]: ...


def _require_key(var: str, client_options: dict) -> None:
    """Fail with the variable's name before the SDK does, deep inside the first call."""
    if not client_options.get("api_key") and not os.environ.get(var):
        raise ConfigurationError(f"{var} is not set — export it, or pass client= (a provider "
                                 f"or SDK client) to route model calls elsewhere")


class AnthropicProvider:
    def __init__(self, client: Any = None, **client_options):
        if client is None:
            import anthropic
            _require_key("ANTHROPIC_API_KEY", client_options)
            client = anthropic.Anthropic(**client_options)
        self.client = client

    def parse(self, model, system, content, schema, max_tokens, timeout):
        response = self.client.with_options(timeout=timeout).messages.parse(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": content}],
            output_format=schema)
        if response.stop_reason == "refusal":
            raise RuntimeError(f"{model} refused the request")
        if response.parsed_output is None:
            raise RuntimeError(f"{model} returned no structured output (stop_reason="
                               f"{response.stop_reason}); if it hit max_tokens, raise max_tokens")
        return response.parsed_output, {"input_tokens": response.usage.input_tokens,
                                        "output_tokens": response.usage.output_tokens}


class OpenAIProvider:
    """OpenAI, or anything that speaks its Chat Completions API (Cerebras, Groq, Together,
    vLLM, Ollama...). Structured output rides on `response_format=json_schema, strict=True`
    with a schema derived here from the pydantic model, so every schema the library uses —
    including ones with defaults — is accepted uniformly across endpoints."""

    def __init__(self, client: Any = None, **client_options):
        if client is None:
            try:
                import openai
            except ImportError as missing:
                raise ImportError("pip install 'teca-label[openai]' to use OpenAI models") from missing
            _require_key("OPENAI_API_KEY", client_options)
            client = openai.OpenAI(**client_options)
        self.client = client

    def parse(self, model, system, content, schema, max_tokens, timeout):
        response = self.client.with_options(timeout=timeout).chat.completions.create(
            model=model, max_completion_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": content}],
            response_format={"type": "json_schema", "json_schema": {
                "name": schema.__name__, "strict": True, "schema": strict_schema(schema)}})
        choice = response.choices[0]
        if getattr(choice.message, "refusal", None):
            raise RuntimeError(f"{model} refused: {choice.message.refusal}")
        if not choice.message.content:
            raise RuntimeError(f"{model} returned no content (finish_reason={choice.finish_reason}); "
                               "if the model reasons before answering, raise max_tokens")
        return schema.model_validate_json(choice.message.content), {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens}


def strict_schema(schema: type[BaseModel]) -> dict:
    """The pydantic JSON schema, tightened to what strict structured output requires:
    every object closed, every property required, defaults dropped (the model must
    always answer every field, so a default has nothing to fill)."""
    def tighten(node):
        if isinstance(node, dict):
            node = {key: tighten(value) for key, value in node.items()
                    if key not in ("default", "title")}
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            return node
        if isinstance(node, list):
            return [tighten(item) for item in node]
        return node
    return tighten(schema.model_json_schema())


_FAMILY_PREFIXES = (("claude", "anthropic"), ("gpt-", "openai"), ("o1", "openai"),
                    ("o3", "openai"), ("o4", "openai"))
_FACTORIES = {"anthropic": AnthropicProvider, "openai": OpenAIProvider}
_defaults: dict[str, Provider] = {}
_defaults_lock = threading.Lock()


def family_of(model: str) -> str:
    for prefix, family in _FAMILY_PREFIXES:
        if model.startswith(prefix):
            return family
    raise ConfigurationError(f"no default provider for model '{model}' — pass client=, e.g. "
                             f"OpenAIProvider(base_url=...) for an OpenAI-compatible endpoint")


def default_provider(model: str) -> Provider:
    family = family_of(model)
    with _defaults_lock:
        if family not in _defaults:
            _defaults[family] = _FACTORIES[family]()
        return _defaults[family]


def as_provider(client: Any) -> Provider:
    """Accept a Provider as-is; wrap a raw SDK client in the provider that drives it."""
    sdk = type(client).__module__.split(".")[0]
    if sdk == "anthropic":
        return AnthropicProvider(client)
    if sdk == "openai":
        return OpenAIProvider(client)
    if callable(getattr(client, "parse", None)):
        return client
    raise TypeError(f"client must be a teca-label Provider, an anthropic.Anthropic, or an "
                    f"openai.OpenAI — got {type(client).__name__}")


def parse(model: str, system: str, content: str, schema: type[BaseModel],
          max_tokens: int = 2048, timeout: float = 60.0,
          provider: Provider | None = None) -> tuple[BaseModel, Usage]:
    """One structured-output call, routed to `provider` or the model's default family.
    The default timeout is short on purpose: one slow request must not jam a parallel
    batch (retries cover it); synthesis and audit calls pass a longer timeout explicitly."""
    return (provider or default_provider(model)).parse(model, system, content, schema,
                                                       max_tokens, timeout)
