"""Anthropic Messages API client: tool use, observations as user turns; assistant content blocks (thinking with
signatures, text, tool_use) are replayed unchanged."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence

from ispatialgym.harness.core.types import CapabilityFailure, InfrastructureFailure
from ispatialgym.harness.llm.base import (AssistantEntry, AssistantTurn, Entry, ImagePart, ModelClient, TextPart, ToolCall, ToolEntry, ToolSpec,
                                          UserEntry, post_json)


def _blocks(content) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in content:
        if isinstance(p, TextPart):
            out.append({"type": "text", "text": p.text})
        elif isinstance(p, ImagePart):
            out.append({"type": "text", "text": f"{p.label}:"})
            out.append({"type": "image", "source": {"type": "base64", "media_type": p.media_type, "data": p.data_b64}})
    return out


class AnthropicMessagesClient(ModelClient):
    """Options: ``max_tokens`` (default 8192), ``thinking_budget`` (extended thinking), ``effort`` (output effort,
    passed through when set), ``temperature``, ``base_url``, ``api_key_env`` (default ANTHROPIC_API_KEY),
    ``anthropic_version``, ``timeout_s`` (60), ``hard_timeout_s`` (90), ``max_retries`` (12), ``extra_body``."""

    name = "anthropic_messages"

    def build_messages(self, transcript: Sequence[Entry]) -> List[Dict[str, Any]]:
        msgs: List[Dict[str, Any]] = []

        def add(role: str, blocks: List[Dict[str, Any]]):
            if msgs and msgs[-1]["role"] == role:  # the API wants alternating roles: merge consecutive user entries
                msgs[-1]["content"].extend(blocks)
            else:
                msgs.append({"role": role, "content": list(blocks)})

        for e in transcript:
            if isinstance(e, UserEntry):
                add("user", _blocks(e.content))
            elif isinstance(e, AssistantEntry):
                raw = e.turn.raw if isinstance(e.turn.raw, list) else None
                if raw:
                    add("assistant", raw)
                else:
                    blocks = ([{"type": "text", "text": e.turn.text}] if e.turn.text else [])
                    blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments or {}} for c in e.turn.tool_calls]
                    add("assistant", blocks or [{"type": "text", "text": "(no output)"}])
            elif isinstance(e, ToolEntry):
                add("user", [{"type": "tool_result", "tool_use_id": e.call_id, "content": _blocks(e.content) or [{"type": "text", "text": "ok"}],
                              "is_error": bool(e.is_error)}])
        return msgs

    def build_request(self, system: str, transcript: Sequence[Entry], tools: Sequence[ToolSpec], tool_choice: Optional[str]) -> Dict[str, Any]:
        o = self.options
        thinking = o.get("thinking_budget")
        kind = "any" if (tool_choice == "required" and not thinking) else "auto"
        body: Dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": self.build_messages(transcript),
            "tools": [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools],
            "tool_choice": {"type": kind, "disable_parallel_tool_use": True},
            "max_tokens": int(o.get("max_tokens", 8192)),
        }
        if thinking:
            body["thinking"] = {"type": "enabled", "budget_tokens": int(thinking)}
        if o.get("effort"):
            body["output_config"] = {"effort": o["effort"]}
        if o.get("temperature") is not None and not thinking:
            body["temperature"] = float(o["temperature"])
        body.update(o.get("extra_body") or {})
        return body

    @staticmethod
    def parse_response(data: Dict[str, Any]) -> AssistantTurn:
        content = data.get("content")
        if not isinstance(content, list):
            raise InfrastructureFailure(f"Messages API reply without content: {str(data)[:300]}")
        if data.get("stop_reason") == "refusal":
            raise CapabilityFailure("model refusal: stop_reason=refusal")
        texts, calls = [], []
        for b in content:
            if b.get("type") == "text" and b.get("text"):
                texts.append(b["text"])
            elif b.get("type") == "tool_use":
                inp = b.get("input")
                calls.append(ToolCall(id=b.get("id") or "", name=b.get("name") or "", arguments=inp if isinstance(inp, dict) else None, raw_arguments=""))
        usage = data.get("usage") or {}
        u = {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"), "cached_tokens": usage.get("cache_read_input_tokens")}
        return AssistantTurn(text="\n".join(texts), tool_calls=calls, usage=u, raw=content, stop_reason=str(data.get("stop_reason") or ""))

    def generate(self, system, transcript, tools, tool_choice: Optional[str] = None) -> AssistantTurn:
        o = self.options
        key_env = o.get("api_key_env", "ANTHROPIC_API_KEY")
        key = os.environ.get(key_env, "")
        if not key:
            raise InfrastructureFailure(f"{key_env} is not set")
        url = o.get("base_url", "https://api.anthropic.com/v1").rstrip("/") + "/messages"
        headers = {"x-api-key": key, "anthropic-version": o.get("anthropic_version", "2023-06-01"), "content-type": "application/json"}
        t0 = time.time()
        data = post_json(url, headers, self.build_request(system, transcript, tools, tool_choice), timeout_s=float(o.get("timeout_s", 60)),
                         hard_timeout_s=float(o.get("hard_timeout_s", 90)), retries=int(o.get("max_retries", 12)))
        turn = self.parse_response(data)
        turn.latency_s = time.time() - t0
        return turn
