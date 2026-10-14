"""Episode scoring (owner 2026-10-14): symbolic and geometric success, under both settle-eval modes, and the error metric.

Per step ``t`` the task reads three facts:

* ``G(t)``: every goal atom except ``pose_settled`` holds;
* ``M(t)``: every ``(:measure ...)`` atom is within its tolerance (``pose_reached`` by its pose report's ``within_tolerance``,
  ``layout_reached`` by its layout report's) - :func:`measure_within`;
* ``S(t)``: every ``pose_settled`` atom holds: its window of ``settle_steps`` steps ending at ``t`` is full and within the drift
  thresholds (vacuously true without such atoms).

**Symbolic** success is the BDDL ``:goal``; **geometric** success is symbolic success plus every ``:measure`` atom within
tolerance at the same step(s). The settle-eval switch defines both:

* off: symbolic = the first ``t`` with ``G(t)``; geometric = the first ``t`` with ``G(t)`` and ``M(t)``;
* on: symbolic = the first ``t`` with ``S(t)`` where ``G`` held at every step of the window (the ``settle_steps`` steps ending at
  ``t``); geometric = the first such ``t`` where ``G`` and ``M`` held at every step of the window.

:class:`SuccessTracker` follows both modes at once (the run's mode decides, the other is reported); a value the episode ended
before reaching is ``None``. Steps are the task's 0-based episode step indices, consecutive (a checkpoint rollback restores the
tracker with the step counter, :meth:`SuccessTracker.state` / :meth:`SuccessTracker.load`).

:func:`error_metric` reduces the error reports of the ``:measure`` atoms to one position and one orientation error.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

MODES = ("off", "on")
KINDS = ("symbolic", "geometric")
#: ``--success-metric``: what run_isg's success, status and exit code follow
SUCCESS_METRICS = ("geometric", "symbolic")
ERROR_METRIC_CASES = ("poses", "layout", "pose", "layout_only")


class ErrorMetricError(ValueError):
    """A ``(:measure ...)`` combination :func:`error_metric` has no case for."""


def mode_name(settle_eval: bool) -> str:
    return "on" if settle_eval else "off"


def measure_atoms(problem) -> List[Tuple[str, str]]:
    """``[(predicate, instance)]`` of a problem's ``(:measure ...)`` atoms (``?`` stripped)."""
    return [(str(lit[0]), str(lit[1]).lstrip("?")) for lit in getattr(problem, "measure", None) or []]


def _report(errors: Dict[str, Any], predicate: str, inst: str) -> Optional[Dict[str, Any]]:
    reports = (errors or {}).get("pose_errors" if predicate == "pose_reached" else "layout_errors") or {}
    rep = reports.get(inst)
    return rep if isinstance(rep, dict) else None


def measure_within(errors: Dict[str, Any], atoms: Sequence[Tuple[str, str]]) -> bool:
    """``M``: every ``(:measure ...)`` atom's report says ``within_tolerance`` (an atom without a report is not within)."""
    return all(bool((_report(errors, p, a) or {}).get("within_tolerance")) for p, a in atoms)


def measure_outside(errors: Dict[str, Any], atoms: Sequence[Tuple[str, str]]) -> List[str]:
    """The ``(:measure ...)`` atoms (``"pose_reached ?x"``) not within tolerance."""
    return [f"{p} ?{a}" for p, a in atoms if not bool((_report(errors, p, a) or {}).get("within_tolerance"))]


def _max(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return max(vals) if vals else None


def error_metric_case(atoms: Sequence[Tuple[str, str]], where: str = "") -> str:
    """The case of a ``(:measure ...)`` block: ``poses`` (two or more ``pose_reached``, no layout: IORS small-scale), ``layout``
    (``layout_reached`` and ``pose_reached`` on the same entity: articulation small-scale, block design, attachment), ``pose`` (one
    ``pose_reached``), ``layout_only`` (one ``layout_reached``, no pose: articulation large-scale); anything else raises."""
    poses = [a for p, a in atoms if p == "pose_reached"]
    layouts = [a for p, a in atoms if p == "layout_reached"]
    other = [p for p, _a in atoms if p not in ("pose_reached", "layout_reached")]
    if not other:
        if not layouts and len(poses) >= 2:
            return "poses"
        if len(layouts) == 1 and len(poses) == 1 and layouts[0] == poses[0]:
            return "layout"
        if not layouts and len(poses) == 1:
            return "pose"
        if len(layouts) == 1 and not poses:
            return "layout_only"
    atoms_s = ", ".join(f"({p} ?{a})" for p, a in atoms) or "none"
    raise ErrorMetricError(f"{where or 'problem'}: no error_metric case for the (:measure ...) atoms {atoms_s} (cases: two or more pose_reached; "
                           "layout_reached and pose_reached on one entity; one pose_reached; one layout_reached)")


def error_metric(errors: Dict[str, Any], atoms: Sequence[Tuple[str, str]], where: str = "") -> Dict[str, Any]:
    """``{"position_error", "orientation_error", "case"}`` from the error reports of the ``(:measure ...)`` atoms
    (:func:`error_metric_case`): ``poses`` the max over the pose reports, ``layout`` / ``layout_only`` the layout report's errors,
    ``pose`` the pose report's. A missing component (an orientation-free target, a revolute-only layout) is ``None``."""
    case = error_metric_case(atoms, where)
    if case == "poses":
        reps = [_report(errors, "pose_reached", a) or {} for p, a in atoms if p == "pose_reached"]
        pos, ori = _max(r.get("position_error") for r in reps), _max(r.get("orientation_error") for r in reps)
    else:
        pred = "pose_reached" if case == "pose" else "layout_reached"
        inst = next(a for p, a in atoms if p == pred)
        rep = _report(errors, pred, inst) or {}
        pos, ori = rep.get("position_error"), rep.get("orientation_error")
    return {"position_error": None if pos is None else float(pos), "orientation_error": None if ori is None else float(ori), "case": case}


class SuccessTracker:
    """Symbolic and geometric success of both settle-eval modes, step by step (module docstring)."""

    def __init__(self, settle_eval: bool = False, settle_steps: int = 45):
        if int(settle_steps) < 1:
            raise ValueError(f"settle_steps must be >= 1, got {settle_steps}")
        self.settle_eval = bool(settle_eval)
        self.settle_steps = int(settle_steps)
        self.reset()

    def reset(self) -> None:
        #: consecutive steps up to the last update where G (G and M) held
        self.g_run = 0
        self.gm_run = 0
        self.first: Dict[str, Dict[str, Optional[int]]] = {m: {k: None for k in KINDS} for m in MODES}
        self.errors_at: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = {m: {k: None for k in KINDS} for m in MODES}
        self.last: Optional[Tuple[int, bool, bool, bool]] = None

    @property
    def mode(self) -> str:
        return mode_name(self.settle_eval)

    @property
    def other_mode(self) -> str:
        return mode_name(not self.settle_eval)

    def update(self, t: int, goal: bool, within: bool, settled: bool, errors: Optional[Dict[str, Any]] = None) -> None:
        """The facts of step ``t`` (``G``, ``M``, ``S``) and its errors (kept at every first success)."""
        goal, within, settled = bool(goal), bool(within), bool(settled)
        self.g_run = self.g_run + 1 if goal else 0
        self.gm_run = self.gm_run + 1 if (goal and within) else 0
        n = self.settle_steps
        hits = {("off", "symbolic"): goal, ("off", "geometric"): goal and within,
                ("on", "symbolic"): settled and self.g_run >= n, ("on", "geometric"): settled and self.gm_run >= n}
        for (mode, kind), hit in hits.items():
            if hit and self.first[mode][kind] is None:
                self.first[mode][kind] = int(t)
                self.errors_at[mode][kind] = errors
        self.last = (int(t), goal, within, settled)

    def step_of(self, kind: str, mode: Optional[str] = None) -> Optional[int]:
        return self.first[mode or self.mode][kind]

    def reached(self, kind: str, mode: Optional[str] = None) -> bool:
        return self.step_of(kind, mode) is not None

    @property
    def done(self) -> bool:
        """The run mode's geometric success (the episode's termination)."""
        return self.reached("geometric")

    @property
    def goal_first_step(self) -> Optional[int]:
        """The first step ``G`` held (the off mode's symbolic success)."""
        return self.first["off"]["symbolic"]

    def evaluation_errors(self, current: Dict[str, Any], mode: Optional[str] = None) -> Dict[str, Any]:
        """The errors a mode is judged by: at its geometric success, else ``current`` (the cutoff)."""
        snap = self.errors_at[mode or self.mode]["geometric"]
        return snap if snap is not None else current

    def mode_report(self, mode: str, current: Dict[str, Any], atoms: Sequence[Tuple[str, str]], where: str = "") -> Dict[str, Any]:
        errs = self.evaluation_errors(current, mode)
        sym, geo = self.first[mode]["symbolic"], self.first[mode]["geometric"]
        return {"settle_eval": mode == "on", "symbolic_success": sym is not None, "symbolic_success_step": sym, "geometric_success": geo is not None,
                "geometric_success_step": geo, "errors": errs, "error_metric": error_metric(errs, atoms, where) if atoms else None}

    def report(self, current: Dict[str, Any], atoms: Sequence[Tuple[str, str]], where: str = "") -> Dict[str, Any]:
        """The run mode's report (its ``errors`` / ``error_metric`` at its geometric success, else at ``current``) with
        ``other_mode`` (the same for the other mode, without its errors) and the settle window."""
        out = self.mode_report(self.mode, current, atoms, where)
        other = self.mode_report(self.other_mode, current, atoms, where)
        other.pop("errors")
        out.update({"settle_steps": self.settle_steps, "goal_first_step": self.goal_first_step, "other_mode": other})
        return out

    def state(self) -> Dict[str, Any]:
        """A copy for a checkpoint (:meth:`load` restores it)."""
        return {"settle_eval": self.settle_eval, "settle_steps": self.settle_steps, "g_run": self.g_run, "gm_run": self.gm_run,
                "first": copy.deepcopy(self.first), "errors_at": {m: dict(v) for m, v in self.errors_at.items()}, "last": self.last}

    def load(self, state: Dict[str, Any]) -> None:
        self.settle_eval, self.settle_steps = bool(state["settle_eval"]), int(state["settle_steps"])
        self.g_run, self.gm_run = int(state["g_run"]), int(state["gm_run"])
        self.first = copy.deepcopy(state["first"])
        self.errors_at = {m: dict(v) for m, v in state["errors_at"].items()}
        self.last = state["last"]


__all__ = ["MODES", "KINDS", "SUCCESS_METRICS", "ERROR_METRIC_CASES", "ErrorMetricError", "SuccessTracker", "mode_name", "measure_atoms",
           "measure_within", "measure_outside", "error_metric_case", "error_metric"]
