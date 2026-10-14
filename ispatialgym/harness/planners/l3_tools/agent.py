"""The L3 tool agent (RoboProbe Inspect EEF's turn loop): every observation is a user turn; the model is called until it
returns one valid motion call, whose tool result (``Accepted: playing N waypoints ...``) is written before it plays; a
refused motion gets a text-only result and the model answers again on the same observation. Malformed calls are repaired
up to ``max_repairs`` times per observation, after which (or on a refusal, or several tool calls that do not fold into
one) the episode ends as a capability failure. Provider failures propagate as :class:`InfrastructureFailure`."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from ispatialgym.harness.config import HarnessConfig
from ispatialgym.harness.core.observation import ObservationEncoder
from ispatialgym.harness.core.trace import EpisodeTrace
from ispatialgym.harness.core.types import CapabilityFailure, Decision, MotionOutcome, MotionPlan, Observation, arrival_lines
from ispatialgym.harness.llm.base import AssistantEntry, Entry, ImagePart, ModelClient, TextPart, ToolCall, ToolEntry, UserEntry, apply_image_horizon
from ispatialgym.harness.planners.base import EpisodeContext, Planner
from ispatialgym.harness.planners.l3_tools.docs import build_r1pro_docs
from ispatialgym.harness.planners.l3_tools.prompts import goal_text, system_message
from ispatialgym.harness.planners.l3_tools.recipes import task_recipe
from ispatialgym.harness.planners.l3_tools.tools import MOTION_TOOLS, ToolArgError, merge_parallel_calls, to_decision, tool_specs

REPLY_WITH_TOOL = "Reply with exactly one tool call from the provided tools."
EMPTY_REPLY = "The model returned an empty response. Reply with exactly one tool call from the provided tools."


class L3ToolAgent(Planner):
    name = "l3_tools"

    def __init__(self, client: ModelClient, cfg: HarnessConfig, encoder: ObservationEncoder, trace: Optional[EpisodeTrace] = None):
        self.client = client
        self.cfg = cfg
        self.enc = encoder
        self.trace = trace or EpisodeTrace(None)
        self.tools = tool_specs(cfg)
        self.tool_names = {t.name for t in self.tools}
        self.transcript: List[Entry] = []
        self.system = ""
        self.goal = ""
        self.recipe: Optional[str] = None
        self.ctx: Optional[EpisodeContext] = None
        self.calls = 0
        self.turns = 0  # observation turns sent; the camera labels' step
        self.repairs = 0  # malformed calls on the current observation
        self.repairs_total = 0
        self.unreachable = 0
        self.pending: Optional[ToolCall] = None
        self.usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0}
        self.latency_s = 0.0
        self.stop_reason = ""
        self.hindsight: Optional[str] = None

    # -- episode ----------------------------------------------------------------------------------------------------
    def start(self, ctx: EpisodeContext) -> None:
        self.ctx = ctx
        self.system = system_message(self.cfg, build_r1pro_docs(ctx.mounting))
        recipe = task_recipe(ctx.alias) if self.cfg.use_recipe else None
        self.recipe = recipe[0] if recipe else None
        self.goal = goal_text(ctx.instruction, recipe[1] if recipe else None)
        content: List[Any] = [TextPart(self.goal)]
        content.append(self.enc.goal_part(ctx.goal_rgb) if ctx.goal_rgb is not None else TextPart("(No goal image is available for this task.)"))
        self.transcript.append(UserEntry(content))
        self.trace.event("system", text=self.system, tools=[asdict(t) for t in self.tools], model=self.client.model, client=self.client.name,
                         client_options={k: v for k, v in self.client.options.items() if "key" not in k.lower()})
        self.trace.event("goal", text=self.goal, recipe=self.recipe, images=self._save_frames(content, "goal"))

    def decide(self, obs: Observation, outcome: Optional[MotionOutcome]) -> Decision:
        assert self.ctx is not None, "start() first"
        self.repairs = 0
        content = self.enc.observation_content(obs, self.ctx.instruction, self.turns, arrival_lines(outcome.details if outcome else None))
        self.transcript.append(UserEntry(content))
        self.trace.event("user", turn=self.turns, control_step=obs.step, text=self._text_of(content), images=self._save_frames(content, f"{self.turns:04d}"))
        self.turns += 1
        return self._ask()

    def accepted(self, plan: MotionPlan) -> None:
        if self.pending is not None:
            self._tool_result(self.pending, plan.accepted_text())
            self.pending = None

    def rejected(self, reason: str, repairable: bool) -> Decision:
        if self.pending is not None:
            self._tool_result(self.pending, reason, is_error=True)
            self.pending = None
        if repairable:
            stop = self._repair("Model kept failing to produce a valid tool call")
            if stop is not None:
                return stop
        else:
            self.unreachable += 1
        return self._ask()

    # -- the turn loop ----------------------------------------------------------------------------------------------
    def _ask(self) -> Decision:
        cfg = self.cfg
        while True:
            if self.calls >= cfg.max_calls:
                return self._stop("call_budget", detail="LLM call budget exhausted")
            request = apply_image_horizon(self.transcript, cfg.image_horizon)
            self.calls += 1
            try:
                turn = self.client.generate(self.system, request, self.tools)
            except CapabilityFailure as e:
                self.trace.event("model_refusal", call=self.calls, error=str(e))
                return self._stop("capability_failure", error=str(e))
            for k in self.usage:
                self.usage[k] += int(turn.usage.get(k) or 0)
            self.latency_s += float(turn.latency_s or 0.0)
            self.trace.event("assistant", call=self.calls, turn=self.turns - 1, repair=self.repairs, text=turn.text, stop_reason=turn.stop_reason,
                             usage=turn.usage, latency_s=round(turn.latency_s, 2),
                             tool_calls=[{"id": c.id, "name": c.name, "arguments": c.arguments, "raw": None if c.arguments is not None else c.raw_arguments}
                                         for c in turn.tool_calls])
            if not turn.tool_calls:
                if turn.text.strip():
                    self.transcript.append(AssistantEntry(turn))
                    self.transcript.append(UserEntry([TextPart(REPLY_WITH_TOOL)]))
                    stop = self._repair("Model kept failing: no tool call")
                else:
                    self.transcript.append(UserEntry([TextPart(EMPTY_REPLY)]))
                    stop = self._repair("Model kept failing: empty response")
                if stop is not None:
                    return stop
                continue
            self.transcript.append(AssistantEntry(turn))
            folded = merge_parallel_calls(turn.tool_calls)
            if len(folded) != 1:
                return self._stop("capability_failure", error=f"model returned {len(turn.tool_calls)} tool calls; expected exactly one")
            call = folded[0]
            for extra in turn.tool_calls[1:]:  # every call id needs a result on every provider
                self._tool_result(extra, "Merged into the first move_eef call of this turn.")
            if call.arguments is None:
                self._tool_result(call, f"invalid JSON arguments: {self._json_error(call.raw_arguments)}", is_error=True)
                stop = self._repair("Model kept failing to produce a valid tool call")
                if stop is not None:
                    return stop
                continue
            if call.name in ("give_up", "done") and call.name in self.tool_names:
                self._tool_result(call, f"Acknowledged {call.name}.")
                self.hindsight = str(call.arguments.get("hindsight") or "") or None
                detail = call.arguments.get("summary" if call.name == "done" else "reason")
                return self._stop(call.name, detail=str(detail or ""), hindsight=self.hindsight)
            if call.name not in MOTION_TOOLS:
                self._tool_result(call, f"unknown tool {call.name!r}", is_error=True)
                stop = self._repair("Model kept failing to produce a valid tool call")
                if stop is not None:
                    return stop
                continue
            try:
                d = to_decision(call, cfg)
            except ToolArgError as e:
                self._tool_result(call, str(e), is_error=True)
                stop = self._repair("Model kept failing to produce a valid tool call")
                if stop is not None:
                    return stop
                continue
            self.pending = call
            return d

    def _repair(self, message: str) -> Optional[Decision]:
        self.repairs += 1
        self.repairs_total += 1
        if self.repairs >= self.cfg.max_repairs:
            return self._stop("capability_failure", error=message)
        return None

    def _stop(self, why: str, /, **details: Any) -> Decision:
        self.stop_reason = why
        self.trace.event("planner_stop", stop=why, **details)
        return Decision("stop", reason=why, details=details)

    def _tool_result(self, call: ToolCall, text: str, is_error: bool = False) -> None:
        self.transcript.append(ToolEntry(call.id, call.name, [TextPart(text)], is_error=is_error))
        self.trace.event("tool_result", call_id=call.id, name=call.name, text=text, is_error=is_error)

    @staticmethod
    def _json_error(raw: str) -> str:
        try:
            json.loads(raw)
        except (TypeError, ValueError) as e:
            return str(e)
        return "the arguments are not a JSON object"

    def _save_frames(self, content, prefix: str) -> List[Dict[str, Any]]:
        refs = []
        for i, p in enumerate(q for q in content if isinstance(q, ImagePart)):
            slug = re.sub(r"[^a-z0-9]+", "_", p.label.lower()).strip("_")[:48]
            refs.append({"label": p.label, "file": self.trace.frame(f"{prefix}_{i}_{slug}.jpg", base64.b64decode(p.data_b64))})
        return refs

    @staticmethod
    def _text_of(content) -> str:
        return "\n".join(p.text for p in content if isinstance(p, TextPart))

    def summary(self) -> Dict[str, Any]:
        return {"planner": self.name, "model": self.client.model, "calls": self.calls, "turns": self.turns, "repairs": self.repairs_total,
                "unreachable": self.unreachable, "stop_reason": self.stop_reason, "hindsight": self.hindsight, "recipe": self.recipe,
                "usage": dict(self.usage), "model_latency_s": round(self.latency_s, 1)}
