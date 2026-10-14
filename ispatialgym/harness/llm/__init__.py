"""Model clients over plain HTTP (no provider SDK in the env): ``make_client("provider/model", **options)``.

* ``openai/<model>``: OpenAI Responses API (GPT-6 Astra, GPT-5.5)
* ``anthropic/<model>``: Anthropic Messages API
* ``compat/<model>``: OpenAI-compatible Chat Completions (needs option ``base_url``; ``api_key_env`` names the key)
* ``scripted:<steps.json>``: a fixed list of tool calls (tests, smoke runs)
"""

from __future__ import annotations

from typing import Any

from ispatialgym.harness.llm.base import ModelClient


def make_client(spec: str, **options: Any) -> ModelClient:
    if spec.startswith("scripted:"):
        from ispatialgym.harness.llm.scripted import ScriptedClient

        return ScriptedClient(spec.split(":", 1)[1], **options)
    provider, sep, model = spec.partition("/")
    if not sep or not model:
        raise ValueError(f"model spec {spec!r}: expected provider/model or scripted:<file>")
    if provider == "openai":
        from ispatialgym.harness.llm.openai_responses import OpenAIResponsesClient

        return OpenAIResponsesClient(model, **options)
    if provider == "anthropic":
        from ispatialgym.harness.llm.anthropic_messages import AnthropicMessagesClient

        return AnthropicMessagesClient(model, **options)
    if provider == "compat":
        from ispatialgym.harness.llm.openai_chat import OpenAIChatClient

        return OpenAIChatClient(model, **options)
    raise ValueError(f"unknown model provider {provider!r} (openai, anthropic, compat, scripted)")
