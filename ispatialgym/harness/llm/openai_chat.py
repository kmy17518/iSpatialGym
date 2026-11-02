"""OpenAI-compatible Chat Completions client (DeepSeek, vLLM / SGLang servers, other gateways). Tool messages carry
text only."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence

from ispatialgym.harness.core.types import CapabilityFailure, InfrastructureFailure
from ispatialgym.harness.llm.base import (AssistantEntry, AssistantTurn, Entry, ImagePart, ModelClient, TextPart, ToolCall, ToolEntry, ToolSpec,
                                          UserEntry, parse_arguments, post_json)


def _parts(content) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in content:
        if isinstance(p, TextPart):
            out.append({"type": "text", "text": p.text})
        elif isinstance(p, ImagePart):
            out.append({"type": "text", "text": f"{p.label}:"})
            out.append({"type": "image_url", "image_url": {"url": f"data:{p.media_type};base64,{p.data_b64}"}})
    return out


class OpenAIChatClient(ModelClient):
    """Options: ``base_url`` (required), ``api_key_env`` (default OPENAI_API_KEY; empty string = no key),
    ``tool_choice`` (sent only when set), ``parallel_tool_calls`` (default false), ``reasoning_effort``, ``max_tokens``,
    ``temperature``, ``timeout_s`` (60), ``hard_timeout_s`` (90), ``max_retries`` (12), ``extra_body``."""

    name = "openai_chat"

    def build_messages(self, system: str, transcript: Sequence[Entry]) -> List[Dict[str, Any]]:
        msgs: List[Dict[str, Any]] = [{"role": "system", "content": system}]
        for e in transcript:
            if isinstance(e, UserEntry):
                msgs.append({"role": "user", "content": _parts(e.content)})
            elif isinstance(e, AssistantEntry):
                m: Dict[str, Any] = {"role": "assistant", "content": e.turn.text or None}
                if e.turn.tool_calls:
                    m["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.raw_arguments or "{}"}}
                                       for c in e.turn.tool_calls]
                msgs.append(m)
            elif isinstance(e, ToolEntry):
                text = "\n".join(p.text for p in e.content if isinstance(p, TextPart)) or ("error" if e.is_error else "ok")
                msgs.append({"role": "tool", "tool_call_id": e.call_id, "content": text})
                images = [p for p in e.content if isinstance(p, ImagePart)]
                if images:
                    msgs.append({"role": "user", "content": _parts(images)})
        return msgs

    def build_request(self, system: str, transcript: Sequence[Entry], tools: Sequence[ToolSpec], tool_choice: Optional[str]) -> Dict[str, Any]:
        o = self.options
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": self.build_messages(system, transcript),
            "tools": [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}} for t in tools],
            "parallel_tool_calls": bool(o.get("parallel_tool_calls", False)),
        }
        choice = o.get("tool_choice", tool_choice)
        if choice:
            body["tool_choice"] = choice
        if o.get("reasoning_effort"):
            body["reasoning_effort"] = o["reasoning_effort"]
        if o.get("max_tokens"):
            body["max_tokens"] = int(o["max_tokens"])
        if o.get("temperature") is not None:
            body["temperature"] = float(o["temperature"])
        body.update(o.get("extra_body") or {})
        return body

    @staticmethod
    def parse_response(data: Dict[str, Any]) -> AssistantTurn:
        choices = data.get("choices")
        if not choices:
            raise InfrastructureFailure(f"chat completion without choices: {str(data)[:300]}")
        finish = str(choices[0].get("finish_reason") or "")
        if finish in ("content_filter", "content_policy"):
            raise CapabilityFailure(f"model refusal: finish_reason={finish}")
        msg = choices[0].get("message") or {}
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function") or {}
            raw = fn.get("arguments") or ""
            calls.append(ToolCall(id=c.get("id") or "", name=fn.get("name") or "", arguments=parse_arguments(raw), raw_arguments=raw if isinstance(raw, str) else ""))
        usage = data.get("usage") or {}
        u = {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
             "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")}
        return AssistantTurn(text=msg.get("content") or "", tool_calls=calls, usage=u, raw=None, stop_reason=finish)

    def generate(self, system, transcript, tools, tool_choice: Optional[str] = None) -> AssistantTurn:
        o = self.options
        if not o.get("base_url"):
            raise InfrastructureFailure("OpenAI-compatible client needs model option base_url")
        env = o.get("api_key_env", "OPENAI_API_KEY")
        headers = {"Content-Type": "application/json"}
        if env:
            key = os.environ.get(env, "")
            if not key:
                raise InfrastructureFailure(f"{env} is not set")
            headers["Authorization"] = f"Bearer {key}"
        t0 = time.time()
        data = post_json(o["base_url"].rstrip("/") + "/chat/completions", headers, self.build_request(system, transcript, tools, tool_choice),
                         timeout_s=float(o.get("timeout_s", 60)), hard_timeout_s=float(o.get("hard_timeout_s", 90)), retries=int(o.get("max_retries", 12)))
        turn = self.parse_response(data)
        turn.latency_s = time.time() - t0
        return turn
