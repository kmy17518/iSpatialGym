"""Scripted client: replays a fixed list of model turns (tests, simulator smoke runs, no API key). After the script it
calls ``give_up``."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ispatialgym.harness.llm.base import AssistantTurn, ModelClient, ToolCall


class ScriptedClient(ModelClient):
    """``model``: path of a JSON list of turns, or ``steps=[...]`` given directly. A turn is ``{"name": tool, "arguments":
    {...}}`` (one call; ``"raw_arguments": "..."`` instead sends unparsable arguments), ``{"calls": [turn, ...]}``
    (parallel calls), ``{"text": "..."}`` (a reply without a tool call) or ``{}`` (an empty reply)."""

    name = "scripted"

    def __init__(self, model: str = "", steps: Optional[List[Dict[str, Any]]] = None, **options: Any):
        super().__init__(model, **options)
        if steps is None:
            with open(model) as f:
                steps = json.load(f)
        self.steps = list(steps or [])
        self.calls = 0
        self.seen: List[Any] = []  # (system, transcript length) per call, for tests

    def generate(self, system, transcript, tools, tool_choice: Optional[str] = None) -> AssistantTurn:
        self.seen.append((system, len(transcript)))
        i = self.calls
        self.calls += 1
        s = self.steps[i] if i < len(self.steps) else {"name": "give_up", "arguments": {"reason": "end of the scripted steps", "hindsight": ""}}
        turns = s.get("calls") or ([s] if "name" in s else [])
        calls = []
        for j, c in enumerate(turns):
            if "raw_arguments" in c:
                raw, args = str(c["raw_arguments"]), None
            else:
                args = c.get("arguments", {})
                raw, args = json.dumps(args), (args if isinstance(args, dict) else None)
            calls.append(ToolCall(id=f"call_{i}_{j}", name=c["name"], arguments=args, raw_arguments=raw))
        return AssistantTurn(text=str(s.get("text", "")), tool_calls=calls, usage={"input_tokens": 0, "output_tokens": 0})
