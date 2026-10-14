"""OpenAI Responses API client (GPT-6 Astra, GPT-5.5): stateless requests (``store: false``) that replay the model's own
output items, encrypted reasoning included, so a reasoning model keeps its chain of thought across tool calls."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence

from ispatialgym.harness.core.types import CapabilityFailure, InfrastructureFailure
from ispatialgym.harness.llm.base import (AssistantEntry, AssistantTurn, Entry, ImagePart, ModelClient, TextPart, ToolCall, ToolEntry, ToolSpec,
                                          UserEntry, parse_arguments, post_json)


def _content_parts(content, detail: Optional[str]) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    for p in content:
        if isinstance(p, TextPart):
            parts.append({"type": "input_text", "text": p.text})
        elif isinstance(p, ImagePart):
            parts.append({"type": "input_text", "text": f"{p.label}:"})
            img: Dict[str, Any] = {"type": "input_image", "image_url": f"data:{p.media_type};base64,{p.data_b64}"}
            if detail:
                img["detail"] = detail
            parts.append(img)
    return parts


class OpenAIResponsesClient(ModelClient):
    """Options: ``effort`` (reasoning effort, default medium as RoboProbe; null sends none), ``reasoning_summary``,
    ``max_output_tokens``, ``temperature``, ``image_detail`` (auto | low | high; default unset), ``base_url``,
    ``api_key_env`` (default OPENAI_API_KEY), ``timeout_s`` (60), ``hard_timeout_s`` (90), ``max_retries`` (12),
    ``extra_body`` (merged into the request)."""

    name = "openai_responses"

    def build_input(self, transcript: Sequence[Entry]) -> List[Dict[str, Any]]:
        detail = self.options.get("image_detail")
        items: List[Dict[str, Any]] = []
        for e in transcript:
            if isinstance(e, UserEntry):
                items.append({"role": "user", "content": _content_parts(e.content, detail)})
            elif isinstance(e, AssistantEntry):
                raw = e.turn.raw if isinstance(e.turn.raw, list) else None
                if raw:
                    items.extend(raw)
                else:  # a turn without provider items (scripted / converted): a plain assistant message + calls
                    if e.turn.text:
                        items.append({"role": "assistant", "content": [{"type": "output_text", "text": e.turn.text}]})
                    for c in e.turn.tool_calls:
                        items.append({"type": "function_call", "call_id": c.id, "name": c.name, "arguments": c.raw_arguments or "{}"})
            elif isinstance(e, ToolEntry):
                text = "\n".join(p.text for p in e.content if isinstance(p, TextPart)) or ("error" if e.is_error else "ok")
                items.append({"type": "function_call_output", "call_id": e.call_id, "output": text})
                images = [p for p in e.content if isinstance(p, ImagePart)]
                if images:  # function_call_output carries text only
                    items.append({"role": "user", "content": _content_parts(images, detail)})
        return items

    def build_request(self, system: str, transcript: Sequence[Entry], tools: Sequence[ToolSpec], tool_choice: Optional[str]) -> Dict[str, Any]:
        o = self.options
        body: Dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": self.build_input(transcript),
            "tools": [{"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters, "strict": False} for t in tools],
            "parallel_tool_calls": False,
            "store": False,
        }
        if tool_choice:
            body["tool_choice"] = tool_choice
        effort = o.get("effort", "medium")
        if effort:
            body["reasoning"] = {"effort": effort}
            if o.get("reasoning_summary"):
                body["reasoning"]["summary"] = o["reasoning_summary"]
            body["include"] = ["reasoning.encrypted_content"]
        if o.get("max_output_tokens"):
            body["max_output_tokens"] = int(o["max_output_tokens"])
        if o.get("temperature") is not None:
            body["temperature"] = float(o["temperature"])
        body.update(o.get("extra_body") or {})
        return body

    @staticmethod
    def parse_response(data: Dict[str, Any]) -> AssistantTurn:
        output = data.get("output")
        if not isinstance(output, list):
            raise InfrastructureFailure(f"Responses API reply without output: {str(data)[:300]}")
        reason = str((data.get("incomplete_details") or {}).get("reason") or "")
        if data.get("status") == "incomplete" and reason in ("content_filter", "content_policy"):
            raise CapabilityFailure(f"model refusal: incomplete ({reason})")
        texts, calls = [], []
        for item in output:
            t = item.get("type")
            if t == "message":
                for c in item.get("content") or []:
                    if c.get("type") in ("output_text", "text") and c.get("text"):
                        texts.append(c["text"])
                    elif c.get("type") == "refusal" and c.get("refusal"):
                        texts.append(c["refusal"])
            elif t == "function_call":
                raw = item.get("arguments") or ""
                calls.append(ToolCall(id=item.get("call_id") or item.get("id") or "", name=item.get("name") or "",
                                      arguments=parse_arguments(raw), raw_arguments=raw if isinstance(raw, str) else ""))
        usage = data.get("usage") or {}
        u = {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
             "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
             "cached_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens")}
        return AssistantTurn(text="\n".join(texts), tool_calls=calls, usage=u, raw=output, stop_reason=str(data.get("status") or ""))

    def generate(self, system, transcript, tools, tool_choice: Optional[str] = None) -> AssistantTurn:
        o = self.options
        key_env = o.get("api_key_env", "OPENAI_API_KEY")
        key = os.environ.get(key_env, "")
        if not key:
            raise InfrastructureFailure(f"{key_env} is not set")
        url = o.get("base_url", "https://api.openai.com/v1").rstrip("/") + "/responses"
        t0 = time.time()
        data = post_json(url, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, self.build_request(system, transcript, tools, tool_choice),
                         timeout_s=float(o.get("timeout_s", 60)), hard_timeout_s=float(o.get("hard_timeout_s", 90)), retries=int(o.get("max_retries", 12)))
        turn = self.parse_response(data)
        turn.latency_s = time.time() - t0
        return turn
