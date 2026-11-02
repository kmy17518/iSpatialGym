"""Provider-neutral conversation types and the client interface.

A transcript is a list of entries:

* ``UserEntry(content)``: text and images from the harness (Goal turn, observations, repair prompts);
* ``AssistantEntry(turn)``: one model response; ``turn.raw`` keeps the provider's own output items (reasoning items,
  thinking blocks with signatures) so a stateless request can replay them unchanged;
* ``ToolEntry(call_id, name, content, is_error)``: the result of one tool call (text).

Each provider client converts the transcript into its wire format after :func:`apply_image_horizon`.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Union

from ispatialgym.harness.core.types import CapabilityFailure, InfrastructureFailure

log = logging.getLogger(__name__)

IMAGE_PLACEHOLDER = "[earlier camera image omitted to save context]"
#: RoboProbe's ladder: a sweep runs every rollout against one account, so it has to outlast a throttling window
RETRY_BACKOFF_S = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0)
_REFUSAL_TOKENS = ("content_filter", "content_policy", "refusal")


@dataclass
class TextPart:
    text: str


@dataclass
class ImagePart:
    label: str  # e.g. "camera 'head' (step 3)"; precedes the image as a text line on every wire
    data_b64: str
    media_type: str = "image/jpeg"
    pinned: bool = False  # never evicted by the image horizon (the goal image)


Content = List[Union[TextPart, ImagePart]]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON schema of the arguments object


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Optional[Dict[str, Any]]  # None when the arguments are not valid JSON
    raw_arguments: str = ""


@dataclass
class AssistantTurn:
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None  # provider output items, replayed as is
    stop_reason: str = ""
    latency_s: float = 0.0


@dataclass
class UserEntry:
    content: Content
    role: str = "user"


@dataclass
class AssistantEntry:
    turn: AssistantTurn
    role: str = "assistant"


@dataclass
class ToolEntry:
    call_id: str
    name: str
    content: Content
    is_error: bool = False
    role: str = "tool"


Entry = Union[UserEntry, AssistantEntry, ToolEntry]


class ModelClient:
    """``generate(system, transcript, tools, tool_choice)`` -> :class:`AssistantTurn`. ``tool_choice=None`` sends none
    (the provider's default, auto), as RoboProbe does."""

    name = "base"

    def __init__(self, model: str, **options: Any):
        self.model = model
        self.options = dict(options)

    def generate(self, system: str, transcript: Sequence[Entry], tools: Sequence[ToolSpec], tool_choice: Optional[str] = None) -> AssistantTurn:
        raise NotImplementedError


def image_bearing(entry: Entry) -> bool:
    content = getattr(entry, "content", None) or []
    return any(isinstance(p, ImagePart) and not p.pinned for p in content)


def apply_image_horizon(transcript: Sequence[Entry], horizon: Optional[int]) -> List[Entry]:
    """Keep the camera images of the newest ``horizon`` observation turns; older images become a text placeholder
    (pinned images, i.e. the goal image, always stay). ``None`` keeps every image."""
    if horizon is None:
        return list(transcript)
    bearing = [i for i, e in enumerate(transcript) if image_bearing(e)]
    keep = set(bearing[-int(horizon):]) if horizon > 0 else set()
    out: List[Entry] = []
    for i, e in enumerate(transcript):
        if i in keep or not image_bearing(e):
            out.append(e)
            continue
        content: Content = []
        for p in e.content:  # type: ignore[union-attr]
            content.append(TextPart(IMAGE_PLACEHOLDER) if isinstance(p, ImagePart) and not p.pinned else p)
        out.append(replace(e, content=content))  # type: ignore[arg-type]
    return out


def parse_arguments(raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {} if raw in (None, "") else None
    try:
        v = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return v if isinstance(v, dict) else None


def _post_once(url: str, headers: Dict[str, str], body: Dict[str, Any], timeout_s: float, hard_timeout_s: float):
    """One POST; ``hard_timeout_s`` bounds the whole exchange (httpx's timeout bounds each read, not the total), so a
    stalled provider cannot hold the simulator."""
    import httpx

    if not hard_timeout_s or hard_timeout_s <= 0:
        return httpx.post(url, headers=headers, json=body, timeout=timeout_s)
    box: Dict[str, Any] = {}

    def run() -> None:
        try:
            box["r"] = httpx.post(url, headers=headers, json=body, timeout=timeout_s)
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller's thread
            box["e"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(hard_timeout_s)
    if t.is_alive():
        raise TimeoutError(f"no response within the {hard_timeout_s:g} s hard deadline")
    if "e" in box:
        raise box["e"]
    return box["r"]


def _error_detail(r, secrets: Sequence[str]) -> str:
    try:
        err = r.json().get("error", {})
    except ValueError:
        err = {"message": r.text[:300]}
    if not isinstance(err, dict):
        err = {"message": str(err)[:300]}
    fields = [f"{k}={err[k]}" for k in ("type", "code", "param") if err.get(k)]
    msg = str(err.get("message") or "")[:300]
    for s in secrets:
        if s and len(s) >= 4:
            msg = msg.replace(s, "[redacted]")
    return ", ".join(fields + ([msg] if msg else []))


def post_json(url: str, headers: Dict[str, str], body: Dict[str, Any], timeout_s: float = 60.0, hard_timeout_s: float = 90.0,
              retries: int = 12) -> Dict[str, Any]:
    """POST with RoboProbe's retry policy: connection errors, timeouts, 408 / 409 / 429 and 5xx are retried up to ``retries``
    times on the jittered :data:`RETRY_BACKOFF_S` ladder (``retry-after`` wins); a refusal raises :class:`CapabilityFailure`,
    anything else :class:`InfrastructureFailure` (the episode is then not scored)."""
    secrets = [v.split(" ", 1)[-1] for k, v in headers.items() if k.lower() in ("authorization", "x-api-key", "api-key")]
    attempt = 0
    while True:
        try:
            r = _post_once(url, headers, body, timeout_s, hard_timeout_s)
        except Exception as e:  # noqa: BLE001 - connection errors and deadlines are transient
            failure = InfrastructureFailure(f"{type(e).__name__}: {e}", retryable=True)
        else:
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    failure = InfrastructureFailure(f"non-JSON response from {url}", retryable=True)
            else:
                detail = _error_detail(r, secrets)
                if any(t in detail.lower() for t in _REFUSAL_TOKENS):
                    raise CapabilityFailure(f"model refusal: HTTP {r.status_code} ({detail})")
                retry_after = None
                ra = r.headers.get("retry-after")
                if ra:
                    try:
                        retry_after = float(ra)
                    except ValueError:
                        retry_after = None
                failure = InfrastructureFailure(f"HTTP {r.status_code} ({detail})", retryable=r.status_code in (408, 409, 429) or r.status_code >= 500,
                                                retry_after_s=retry_after, status=r.status_code)
        if not failure.retryable or attempt >= retries:
            if failure.retryable:
                raise InfrastructureFailure(f"request to {url} failed after {attempt + 1} attempts: {failure}", status=failure.status)
            raise failure
        if failure.retry_after_s and failure.retry_after_s > 0:
            delay = min(float(failure.retry_after_s), 120.0)
        else:
            delay = RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)] * (1.0 + random.random())
        log.warning("model request failed (%s); retry %d/%d in %.1fs", failure, attempt + 1, retries, delay)
        time.sleep(delay)
        attempt += 1
