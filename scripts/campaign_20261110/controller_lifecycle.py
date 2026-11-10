"""Keep the owner's reporting deadline separate from collection shutdown."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

from scripts.campaign_20261110.contracts import file_digest, require


REQUEST_SHA256 = "1cf3922b7ff69f546009e4a3c083f0e0940b4c0af1dd00d3ee25acc2ec1d5bb2"
OWNER_QUOTE = "But don’t automatically shut it down at that time, I will tell you."
SCHEMA = "isg-controller-operating-policy-v1"


def operating_policy(cfg):
    root = Path(cfg["root"]).resolve()
    path = root / "manifests/controller_operating_policy.json"
    if not path.exists():
        return None
    require(path.resolve().is_relative_to(root / "manifests"), "operating policy outside campaign manifests")
    policy = json.loads(path.read_text())
    require(policy.get("schema") == SCHEMA and policy.get("version") == 1, "unknown controller operating policy")
    require(policy.get("campaign_id") == cfg["campaign_id"], "operating policy campaign differs")
    require(policy.get("mode") == "owner_controlled" and policy.get("budget_and_quota_limits_unchanged") is True,
            "operating policy changes budget/quota authorization")
    target = policy.get("target_deadline_at")
    require(isinstance(target, (int, float)) and math.isfinite(target) and target == cfg["stop_at"],
            "operating policy reporting deadline differs")
    source = policy.get("owner_instruction", {})
    require(source.get("attachment_sha256") == REQUEST_SHA256 and source.get("quote") == OWNER_QUOTE,
            "operating policy lacks the explicit owner instruction")
    return {"path": str(path), "sha256": file_digest(path), "policy": policy}


def continue_running(cfg, *, now=None):
    """No policy preserves legacy limits; the explicit owner policy removes only the time cutoff."""
    return operating_policy(cfg) is not None or (time.time() if now is None else now) < cfg["stop_at"]


def bounded_group_count(cfg, requested, group_seconds, *, now=None):
    require(isinstance(requested, int) and requested >= 0 and group_seconds > 0, "invalid bounded queue request")
    if operating_policy(cfg) is not None:
        return requested
    remaining = cfg["stop_at"] - (time.time() if now is None else now) - 3600
    return min(requested, max(0, int(remaining // group_seconds)))


def sleep_seconds(cfg, *, now=None):
    if operating_policy(cfg) is not None:
        return 60
    return min(60, max(1, cfg["stop_at"] - (time.time() if now is None else now)))


def record_policy(cfg, state):
    """Pin applied operational provenance without changing immutable config or queue identities."""
    current = operating_policy(cfg)
    previous = state.get("operating_policy")
    require(previous is None or previous == current, "applied controller operating policy changed or disappeared")
    if current is not None:
        state["operating_policy"] = current

