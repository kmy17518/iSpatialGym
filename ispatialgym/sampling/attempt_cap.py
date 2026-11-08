"""i7-stancememo (owner 2026-11-08, 5x.3): the adaptive per-attempt time cap, ``ISPATIALGYM_ATTEMPT_CAP``: DEFAULT ON since
the lead's decision #33 (2026-11-08, owner away; A/B $OFH/stancememo_20261108: open_small_02 2.74 -> 1.49 GPU-h, under_large_03 2.12 -> 0.98,
the control inside_large_01 0.30 / success in both, no instance lost); ``ISPATIALGYM_ATTEMPT_CAP=0`` = no cap (the former behaviour).

Rule (owner): per alias, an attempt is capped at max(300 s, 5 x the median written-attempt time of that alias), from the existing runs.
The medians are the table ``attempt_cap_medians.tsv`` next to this module (``scripts/oracle_fix/time_cap_count.py table``: the i7-diag
audit's primary set of round-4 runs, upper median as the audit computed it); ``ISPATIALGYM_ATTEMPT_CAP_TABLE`` names another table.

- ``ISPATIALGYM_ATTEMPT_CAP=1`` (the default): the cap of the job's alias (``ISPATIALGYM_ATTEMPT_CAP_ALIAS``, else ``ISPATIALGYM_ALIAS_MEMORY_ALIAS``,
  which submit_fast.py sets for every job); an alias without a row is not capped. ``=<seconds>``: that cap for every alias (tests).
- The cap runs while the attempt SAMPLES (``SamplerV2.run_instances`` around ``sample``; the start draw is exempt - it restores its
  state only on a ``Reject``): the expensive kit queries (``RobotKit.ik`` / ``ik_all`` / ``ik_batch_solve`` / ``ik_path`` /
  ``collision_free``) and the samplers' own deadline checks call :func:`check`, which raises :class:`AttemptCapStop` once the cap
  is spent. It derives from ``BaseException`` on purpose: the search's ``except Reject`` / ``except Exception`` handlers neither
  swallow it nor record it as a stance's failure (memos, persisted caches); ``run_instances`` turns it into ``Reject("time_cap")`` -
  a rejection with its own reason, never a silent loss. The post-sample stages are short and bounded and are not capped. The fixed
  budgets (``generation.attempt_budget_s``: 360 / 1500 / 2400 / 3000 s) stay as ceilings: whichever comes first ends the attempt.

Kit counters: ``time_cap.cut`` (and ``time_cap.cut_at.<where>``: the first query that found the cap spent), ``time_cap.armed`` once
per attempt with a cap."""
from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

ENV = "ISPATIALGYM_ATTEMPT_CAP"
ENV_TABLE = "ISPATIALGYM_ATTEMPT_CAP_TABLE"
ENV_ALIAS = "ISPATIALGYM_ATTEMPT_CAP_ALIAS"
FLOOR_S, MULT = 300.0, 5.0
DEFAULT_TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2", "attempt_cap_medians.tsv")


class AttemptCapStop(BaseException):
    """The attempt's cap is spent (see the module doc): caught only by ``SamplerV2.run_instances``."""

    def __init__(self, cap_s: float, spent_s: float, where: str):
        super().__init__(f"attempt cap of {cap_s:.0f} s spent ({spent_s:.0f} s) at {where}")
        self.cap_s, self.spent_s, self.where = float(cap_s), float(spent_s), str(where)


_STATE: Dict[str, Optional[float]] = {"t0": None, "deadline": None, "cap": None}
_SUSPENDED = [0]
_TABLE_CACHE: Dict[str, Dict[str, float]] = {}


def on() -> bool:
    return os.environ.get(ENV, "1") not in ("", "0")


def load_table(path: str) -> Dict[str, float]:
    """``{alias_key: median written-attempt s}`` of a ``time_cap_count.py table`` file (cached per path)."""
    if path not in _TABLE_CACHE:
        out: Dict[str, float] = {}
        with open(path) as f:
            for line in f:
                if line.startswith("#") or line.startswith("alias_key") or not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                out[parts[0]] = float(parts[2])
        _TABLE_CACHE[path] = out
    return _TABLE_CACHE[path]


def cap_of_median(median_s: float) -> float:
    return max(FLOOR_S, MULT * float(median_s))


def cap_for(alias: Optional[str] = None) -> Tuple[Optional[float], str]:
    """(cap seconds or None, provenance) for the job's alias under the switch; (None, "off") when off."""
    v = os.environ.get(ENV, "1").strip()  # default on (decision #33)
    if v in ("", "0"):
        return None, "off"
    if v != "1":
        try:
            return float(v), "fixed"
        except ValueError:
            return None, f"bad value {v!r}"
    alias = alias or os.environ.get(ENV_ALIAS) or os.environ.get("ISPATIALGYM_ALIAS_MEMORY_ALIAS")
    if not alias:
        return None, "no alias"
    table = os.environ.get(ENV_TABLE) or DEFAULT_TABLE
    m = load_table(table).get(alias)
    if m is None:
        return None, f"no row for {alias}"
    return cap_of_median(m), f"median {m:.1f} s of {alias} ({os.path.basename(table)})"


def begin(t0: float, cap_s: Optional[float]) -> None:
    """Arm the cap for the attempt that started at ``t0`` (None = no cap)."""
    _STATE.update({"t0": float(t0), "cap": None if cap_s is None else float(cap_s),
                   "deadline": None if cap_s is None else float(t0) + float(cap_s)})


def end() -> None:
    _STATE.update({"t0": None, "deadline": None, "cap": None})


class suspended:
    """Context: no cap inside (the start draw, which restores its state only on a ``Reject``)."""

    def __enter__(self):
        _SUSPENDED[0] += 1

    def __exit__(self, *exc):
        _SUSPENDED[0] -= 1


def check(where: str = "") -> None:
    """Raise :class:`AttemptCapStop` when the armed cap is spent (no-op when none is armed or inside :class:`suspended`)."""
    dl = _STATE["deadline"]
    if dl is None or _SUSPENDED[0]:
        return
    now = time.time()
    if now > dl:
        raise AttemptCapStop(_STATE["cap"] or 0.0, now - float(_STATE["t0"] or now), where or "a kit query")
