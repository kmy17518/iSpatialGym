"""i7-stancememo (owner 2026-11-08, 5x.1): the per-INSTANCE memo of failing stance B / C candidates of an IORS large floor init, and the
redraw-after-2 rule. Two switches, default off, so their effects can be separated:

- ``ISPATIALGYM_STANCE_MEMO=1`` (a): each B / C stance that fails in the B / C search is recorded with its reason against the instance's
  drawn floor start (the anchor cell c_t of the target, the stance-B pool, the partner and its stance-C pool: the layout); a later
  attempt of the same instance on the same start skips it. The memo is cleared when the sampler moves to the next instance and whenever
  the floor start is redrawn; it is never shared across instances.
- ``ISPATIALGYM_FLOOR_REDRAW=1`` (b) (or ``=<n>``: after n, default 2): when 2 attempts on one floor start failed in the B / C search, the
  next attempt redraws the floor start through the existing floor-init draw (``IORSLarge.try_surface("floor")``: free floor cell, stances
  B / A collision-free and framed, the A -> B leg, the stance-C existence test; the attempt's early eval-repeat check as always) instead
  of repeating it. A redraw that finds no start keeps the old one (kit ``redraw.failed.<reason>``).

Why per instance and keyed this way (audit i7-diag item 2, verified on i7diag_floor 330003): the floor init draws no new layout per
attempt, so an instance's attempts re-search the same B / C. Since i7-startdraw (1ba40aae, default on) the robot's stance A is drawn per
attempt, so a failure that depends on A (the A -> B leg, the A -> B drive past the placed target) is keyed with A's pose and matches
only that A; the others (F_init / exact IK at B, the push chain, F_goal, the B -> C carry leg at C) are keyed with B (and C) and A's
trunk translate (the carry's travel trunk). ``ISPATIALGYM_STANCE_MEMO_RECHECK=<k>`` (default 4): every k-th memo hit is NOT skipped but
searched fresh, ``memo.would_have_passed.<b|c>`` when it then passes (must be ~0), ``memo.recheck_failed.<b|c>`` when it fails again.

Kit counters (``kit.stats``, per attempt in the ``kit:`` log line): ``memo.recorded.<b|c>``, ``memo.skipped.<b|c>.<reason>``,
``memo.recheck.<b|c>``, ``memo.would_have_passed.<b|c>``, ``memo.recheck_failed.<b|c>``, ``memo.cleared.<instance|redraw>``;
``redraw.search_fail``, ``redraw.fired``, ``redraw.failed.<reason>``, ``redraw.would_fire`` (b off) and the log-only
``redraw.third_attempt_would_have_succeeded.<written|search>`` / ``redraw.third_attempt_failed.<search|post>`` (b off: the attempt after
the second search failure on one start, to judge the count of 2)."""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Tuple

ENV_MEMO = "ISPATIALGYM_STANCE_MEMO"
ENV_REDRAW = "ISPATIALGYM_FLOOR_REDRAW"
ENV_RECHECK = "ISPATIALGYM_STANCE_MEMO_RECHECK"

#: the C failure texts of ``IORSLarge._couple`` -> the memo reason class (first match wins)
C_CLASSES = (("no push chain", "push_chain"), ("push chain unreachable", "push_chain"), ("∩ F_goal", "empty_F"), ("carry leg", "carry_leg"),
             ("no room for the carried sweep", "carry_margin"), ("no (yaw, grasp) of F", "no_pair"), ("under clearance", "under_clearance"),
             ("drop trunk", "drop_trunk"), ("yaw lock", "yaw_lock"), ("drop -> push", "drop_push_transition"), ("B -> pick", "pick_transition"),
             ("c_skip_dead_b", "dead_at_b"), ("PUSH_CHAIN_LIVE_B", "dead_at_b"))
#: B failure reasons that depend on stance A (keyed with A's pose)
A_DEPENDENT = ("init -> manipulation", "A -> B drive passes", "not in the init image")  # the last: stance A's camera (sm_a 2026-11-08)
#: rejections that end the attempt, not the stance: never recorded
ATTEMPT_STOPS = ("time_budget", "time_cap")


def memo_on() -> bool:
    return os.environ.get(ENV_MEMO, "0") == "1"


def redraw_after() -> int:
    """0 = (b) off; else the number of B / C search failures on one floor start after which the next attempt redraws it (``=1`` -> 2)."""
    v = os.environ.get(ENV_REDRAW, "0").strip()
    if v in ("", "0"):
        return 0
    if v == "1":
        return 2
    try:
        return max(1, int(v))
    except ValueError:
        return 2


def active() -> bool:
    return memo_on() or redraw_after() > 0


def recheck_every() -> int:
    try:
        return max(0, int(os.environ.get(ENV_RECHECK, "4")))
    except ValueError:
        return 4


def stance_key(st: Any) -> Tuple[float, ...]:
    """A stance's identity: base pose (x, y, yaw) and trunk, rounded to 1 mm / 1 mrad."""
    return tuple(round(float(v), 3) for v in st.base_pose2d) + (round(float(st.translate), 3), round(float(st.tilt), 3))


def c_class(text: str) -> str:
    for needle, cls in C_CLASSES:
        if needle in text:
            return cls
    return "other"


def a_dependent(text: str) -> bool:
    return any(n in text for n in A_DEPENDENT)


class StanceMemo:
    """The memo of one instance's floor start. ``count`` bumps a kit counter (``tabletop._count``)."""

    def __init__(self, count: Callable[[str], None] = lambda k: None):
        self.count = count
        self.instance: Optional[int] = None
        self.start_id = 0  # the floor start of this instance (0 = the template's; +1 per redraw)
        self.entries: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self.hits = 0
        self.search_fails = 0  # B / C search failures on the current start
        self.pending_third = False  # (b) off: the next attempt is the "third" on a start with ``redraw_after`` search failures

    # ------------------------------------------------------------------ scope

    def enter_instance(self, index: int) -> bool:
        """The sampler is at instance ``index``: a new instance starts with an empty memo (True when that cleared one)."""
        if self.instance == int(index):
            return False
        had = bool(self.entries)
        self.instance, self.start_id = int(index), 0
        self._reset("instance" if had else None)
        return True

    def new_start(self) -> None:
        """The floor start was redrawn: nothing recorded against the old start holds."""
        self.start_id += 1
        self._reset("redraw")

    def _reset(self, why: Optional[str]) -> None:
        if why is not None:
            self.count(f"memo.cleared.{why}")
        self.entries.clear()
        self.hits = 0
        self.search_fails = 0
        self.pending_third = False

    # ------------------------------------------------------------------ B

    def b_failed(self, b: Tuple[float, ...], a: Tuple[float, ...], reason: str, text: str, a_dep: bool = False) -> None:
        """B failed under stance A ``a``: keyed with A when the failure depends on it (``a_dep``, or the text names the A -> B leg / drive)."""
        if reason in ATTEMPT_STOPS:
            return
        key = ("b", b, a if (a_dep or a_dependent(text)) else None)
        if key not in self.entries:
            self.entries[key] = {"reason": reason, "text": text[:300]}
            self.count("memo.recorded.b")

    def lookup_b(self, b: Tuple[float, ...], a: Tuple[float, ...]) -> Tuple[Optional[Dict[str, Any]], bool]:
        """(entry, recheck): the entry that fails B under this A (None: search it); ``recheck`` = search it anyway (the k-th hit)."""
        for key in (("b", b, None), ("b", b, a)):
            e = self.entries.get(key)
            if e is not None:
                return e, self._hit("b", e["reason"])
        return None, False

    def b_passed(self, b: Tuple[float, ...], a: Tuple[float, ...]) -> None:
        """A re-checked B coupled: its entries go (and the counter that must stay ~0)."""
        self.count("memo.would_have_passed.b")
        self.entries.pop(("b", b, None), None)
        self.entries.pop(("b", b, a), None)

    # ------------------------------------------------------------------ C

    def c_key(self, b: Tuple[float, ...], c: Tuple[float, ...], arm: str, a_translate: float) -> Tuple[Any, ...]:
        """C at B for the arm; A's trunk translate is in the key (the carry's travel trunks: ``IORSLarge.travel_trunks``)."""
        return ("c", b, c, str(arm), round(float(a_translate), 3))

    def c_failed(self, key: Tuple[Any, ...], text: str, a: Optional[Tuple[float, ...]] = None, a_dep: bool = False) -> None:
        """The C try failed; keyed with stance A's pose ``a`` too when it may depend on A (``a_dep``: a draw at this B met the A -> B drive past
        the placed target - later draws of that pair count as "(cached)" - or the text names it)."""
        k = key + ((a,) if a is not None and (a_dep or a_dependent(text)) else (None,))
        if k not in self.entries:
            self.entries[k] = {"reason": c_class(text), "text": text[:300]}
            self.count("memo.recorded.c")

    def lookup_c(self, key: Tuple[Any, ...], a: Optional[Tuple[float, ...]] = None) -> Tuple[Optional[Dict[str, Any]], bool]:
        for k in (key + (None,), key + (a,)):
            e = self.entries.get(k)
            if e is not None:
                return e, self._hit("c", e["reason"])
        return None, False

    def c_passed(self, key: Tuple[Any, ...], a: Optional[Tuple[float, ...]] = None) -> None:
        self.count("memo.would_have_passed.c")
        self.entries.pop(key + (None,), None)
        self.entries.pop(key + (a,), None)

    # ------------------------------------------------------------------ hits

    def _hit(self, kind: str, reason: str) -> bool:
        """A memo hit: True = re-check it (searched fresh), False = skipped."""
        self.hits += 1
        k = recheck_every()
        if k and self.hits % k == 0:
            self.count(f"memo.recheck.{kind}")
            return True
        self.count(f"memo.skipped.{kind}.{reason}")
        return False

    def recheck_failed(self, kind: str) -> None:
        self.count(f"memo.recheck_failed.{kind}")
