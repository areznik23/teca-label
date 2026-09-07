"""The model seam: schema tightening, both providers' request/response contracts,
client resolution, and per-family defaults. No network — SDK clients are stand-ins
that record what they were asked."""
import sys
import unittest
from types import SimpleNamespace
from typing import Literal
from unittest.mock import patch

from pydantic import BaseModel, create_model

import teca_label.providers as providers
from teca_label.core import Category, Codebook, Revision
from teca_label.providers import (AnthropicProvider, OpenAIProvider, as_provider,
                               default_provider, family_of, strict_schema)
from teca_label.units import _Excerpts


def every_object(node):
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield node
        for value in node.values():
            yield from every_object(value)
    elif isinstance(node, list):
        for item in node:
            yield from every_object(item)


def every_key(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from every_key(value)
    elif isinstance(node, list):
        for item in node:
            yield from every_key(item)


class TestStrictSchema(unittest.TestCase):
    def check(self, schema):
        tightened = strict_schema(schema)
        for obj in every_object(tightened):
            self.assertIs(obj["additionalProperties"], False)
            self.assertEqual(obj["required"], list(obj["properties"]))
        self.assertNotIn("default", set(every_key(tightened)))
        return tightened

    def test_revision_with_nested_defaults_and_refs(self):
        tightened = self.check(Revision)
        self.assertIn("Op", tightened["$defs"])
        self.assertIn("Category", tightened["$defs"])
        self.assertEqual(tightened["$defs"]["Op"]["required"],
                         ["op", "name", "new_name", "definition", "names", "into", "evidence"])

    def test_dynamic_label_literal_becomes_enum(self):
        label = create_model("Label", label=(Literal["a", "b", "other"], ...))
        tightened = self.check(label)
        self.assertEqual(tightened["properties"]["label"]["enum"], ["a", "b", "other"])

    def test_optional_int_survives_as_nullable(self):
        tightened = self.check(Category)
        self.assertEqual(tightened["properties"]["deprecated_v"]["anyOf"],
                         [{"type": "integer"}, {"type": "null"}])

    def test_excerpts(self):
        self.check(_Excerpts)

    def test_descriptions_survive_titles_do_not(self):
        tightened = strict_schema(Revision)
        self.assertIn("description", tightened["$defs"]["Op"])
        self.assertNotIn("title", set(every_key(tightened)))


class Answer(BaseModel):
    label: str


class FakeOpenAI:
    """Records the chat.completions.create call; replies with a canned message."""
    def __init__(self, content='{"label": "a", "evidence": ""}', refusal=None, finish_reason="stop"):
        self.calls = []
        self.timeouts = []
        self.reply = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason=finish_reason,
                                     message=SimpleNamespace(content=content, refusal=refusal))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3))

    def with_options(self, timeout):
        self.timeouts.append(timeout)
        return self

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.calls.append(request)
        return self.reply


class TestOpenAIProvider(unittest.TestCase):
    def test_request_shape_and_normalized_usage(self):
        client = FakeOpenAI()
        parsed, usage = OpenAIProvider(client).parse("gpt-5", "sys", "hello", Answer, 512, 7.5)
        self.assertEqual(parsed.label, "a")
        self.assertEqual(usage, {"input_tokens": 11, "output_tokens": 3})
        self.assertEqual(client.timeouts, [7.5])
        request = client.calls[0]
        self.assertEqual(request["model"], "gpt-5")
        self.assertEqual(request["max_completion_tokens"], 512)
        self.assertEqual(request["messages"], [{"role": "system", "content": "sys"},
                                               {"role": "user", "content": "hello"}])
        json_schema = request["response_format"]["json_schema"]
        self.assertIs(json_schema["strict"], True)
        self.assertEqual(json_schema["name"], "Answer")
        self.assertEqual(json_schema["schema"], strict_schema(Answer))

    def test_refusal_raises(self):
        provider = OpenAIProvider(FakeOpenAI(content=None, refusal="no"))
        with self.assertRaisesRegex(RuntimeError, "refused"):
            provider.parse("gpt-5", "s", "c", Answer, 10, 1)

    def test_empty_content_names_the_finish_reason(self):
        provider = OpenAIProvider(FakeOpenAI(content="", finish_reason="length"))
        with self.assertRaisesRegex(RuntimeError, "finish_reason=length"):
            provider.parse("gpt-5", "s", "c", Answer, 10, 1)

    def test_malformed_content_surfaces_as_validation_error(self):
        provider = OpenAIProvider(FakeOpenAI(content='{"wrong": 1}'))
        with self.assertRaises(ValueError):
            provider.parse("gpt-5", "s", "c", Answer, 10, 1)

    def test_missing_sdk_points_at_the_extra(self):
        with patch.dict(sys.modules, {"openai": None}):
            with self.assertRaisesRegex(ImportError, r"teca-label\[openai\]"):
                OpenAIProvider()


class FakeAnthropic:
    def __init__(self, stop_reason="end_turn", parsed_output=Answer(label="b")):
        self.calls = []
        self.timeouts = []
        self.stop_reason, self.parsed_output = stop_reason, parsed_output

    def with_options(self, timeout):
        self.timeouts.append(timeout)
        return self

    @property
    def messages(self):
        return SimpleNamespace(parse=self._parse)

    def _parse(self, **request):
        self.calls.append(request)
        return SimpleNamespace(parsed_output=self.parsed_output, stop_reason=self.stop_reason,
                               usage=SimpleNamespace(input_tokens=5, output_tokens=2))


class TestAnthropicProvider(unittest.TestCase):
    def test_request_shape_and_normalized_usage(self):
        client = FakeAnthropic()
        parsed, usage = AnthropicProvider(client).parse("claude-opus-5", "sys", "hi", Answer, 256, 3.0)
        self.assertEqual(parsed.label, "b")
        self.assertEqual(usage, {"input_tokens": 5, "output_tokens": 2})
        self.assertEqual(client.timeouts, [3.0])
        request = client.calls[0]
        self.assertEqual(request["model"], "claude-opus-5")
        self.assertEqual(request["max_tokens"], 256)
        self.assertEqual(request["system"], "sys")
        self.assertIs(request["output_format"], Answer)

    def test_refusal_and_truncation_raise_instead_of_returning_none(self):
        with self.assertRaisesRegex(RuntimeError, "refused"):
            AnthropicProvider(FakeAnthropic(stop_reason="refusal", parsed_output=None)).parse(
                "claude-opus-5", "sys", "hi", Answer, 256, 3.0)
        with self.assertRaisesRegex(RuntimeError, "max_tokens"):
            AnthropicProvider(FakeAnthropic(stop_reason="max_tokens", parsed_output=None)).parse(
                "claude-opus-5", "sys", "hi", Answer, 256, 3.0)


class TestResolution(unittest.TestCase):
    def test_family_by_model_prefix(self):
        self.assertEqual(family_of("claude-opus-5"), "anthropic")
        self.assertEqual(family_of("gpt-5-mini"), "openai")
        self.assertEqual(family_of("o3-mini"), "openai")

    def test_unknown_family_says_how_to_fix_it(self):
        with self.assertRaisesRegex(ValueError, "client="):
            family_of("gemini-2.5-pro")

    def test_sdk_clients_are_wrapped_and_providers_pass_through(self):
        import anthropic
        import openai
        raw_anthropic = anthropic.Anthropic(api_key="x")
        wrapped = as_provider(raw_anthropic)
        self.assertIsInstance(wrapped, AnthropicProvider)
        self.assertIs(wrapped.client, raw_anthropic)
        raw_openai = openai.OpenAI(api_key="x")
        wrapped = as_provider(raw_openai)
        self.assertIsInstance(wrapped, OpenAIProvider)
        self.assertIs(wrapped.client, raw_openai)
        provider = OpenAIProvider(FakeOpenAI())
        self.assertIs(as_provider(provider), provider)

    def test_codebook_rejects_an_unusable_client_at_construction(self):
        with self.assertRaises(TypeError):
            Codebook("q", client=object())

    def test_default_provider_is_lazy_per_family_and_cached(self):
        built = []

        def factory(family):
            def make():
                built.append(family)
                return OpenAIProvider(FakeOpenAI()) if family == "openai" \
                    else AnthropicProvider(FakeAnthropic())
            return make

        with patch.dict(providers._FACTORIES, {"anthropic": factory("anthropic"),
                                               "openai": factory("openai")}), \
                patch.dict(providers._defaults, clear=True):
            first = default_provider("gpt-5")
            second = default_provider("gpt-5-mini")
            self.assertIs(first, second)
            self.assertEqual(built, ["openai"])

    def test_codebook_routes_by_model_family_when_no_client_is_given(self):
        openai_client = FakeOpenAI(content='{"label": "a", "evidence": "t"}')
        with patch.dict(providers._FACTORIES, {"openai": lambda: OpenAIProvider(openai_client)}), \
                patch.dict(providers._defaults, clear=True):
            cb = Codebook("q", [Category(name="a", definition="d")], version=1,
                          models={"classify": "gpt-5-mini"})
            labels = cb.classify([{"t": "one"}], workers=1)
        self.assertEqual(labels, ["a"])
        self.assertEqual(openai_client.calls[0]["model"], "gpt-5-mini")
        self.assertEqual(cb.usage["input_tokens"], 11)


if __name__ == "__main__":
    unittest.main()
