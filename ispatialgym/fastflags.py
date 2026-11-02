"""Switches of the train-fast branch (2026-10-31): every speed-up that changes how the sampler / oracle computes (never what the
outputs mean) is behind ``ISPATIALGYM_FAST_<NAME>``, **on by default on this branch**, off with ``=0`` (``false`` / ``no`` / ``off``).
``ISPATIALGYM_FAST=0`` turns every one of them off at once (the A side of an A/B on the same code).

Usage::

    from ispatialgym.fastflags import fast_on
    if fast_on("EARLY_REPEAT_FILTER"):
        ...

``fast_value(name, default)`` reads a tunable (``ISPATIALGYM_FAST_<NAME>=<value>``) the same way. Every switch logs the first time it
is consulted (one line per process) so a job log shows which paths were live.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

log = logging.getLogger("ispatialgym.fast")

_OFF = ("0", "false", "no", "off")
_seen: Dict[str, Any] = {}


def fast_master() -> bool:
    return os.environ.get("ISPATIALGYM_FAST", "1").strip().lower() not in _OFF


def fast_on(name: str, default: bool = True) -> bool:
    """``ISPATIALGYM_FAST_<NAME>`` (default: ``default``, normally on); always off when ``ISPATIALGYM_FAST=0``."""
    key = f"ISPATIALGYM_FAST_{name.upper()}"
    raw = os.environ.get(key)
    if not fast_master():
        on = False
    elif raw is None or raw.strip() == "":
        on = bool(default)
    else:
        on = raw.strip().lower() not in _OFF
    if key not in _seen:
        _seen[key] = on
        log.info("fast switch %s = %s%s", key, "on" if on else "off", "" if raw is not None else " (default)")
    return on


def fast_value(name: str, default: Any, cast=None) -> Any:
    """A tunable ``ISPATIALGYM_FAST_<NAME>=<value>`` (``default`` when unset or when ``ISPATIALGYM_FAST=0``)."""
    key = f"ISPATIALGYM_FAST_{name.upper()}"
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "" or not fast_master():
        val = default
    else:
        try:
            val = (cast or type(default))(raw.strip()) if (cast or (default is not None)) else raw.strip()
        except (TypeError, ValueError):
            log.warning("fast tunable %s=%r is not a %s: using %r", key, raw, (cast or type(default)).__name__, default)
            val = default
    if key not in _seen:
        _seen[key] = val
        log.info("fast tunable %s = %r%s", key, val, "" if raw is not None else " (default)")
    return val


def fast_flags_seen() -> Dict[str, Any]:
    """The switches / tunables this process consulted so far (for summaries and metas)."""
    return dict(_seen)
