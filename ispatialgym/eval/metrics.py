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

:class:`SuccessTracker` follows the run's mode, which decides. A settle-eval-on run also follows the off mode and reports it as
``other_mode``; a settle-eval-off run follows no other mode (``other_mode`` None: a policy run ends shortly after its first success,
before an on-mode window could fill). A value the episode ended before reaching is ``None``. Steps are the task's 0-based episode
step indices, consecutive (a checkpoint rollback restores the tracker with the step counter, :meth:`SuccessTracker.state` /
:meth:`SuccessTracker.load`).

**The evaluated success step** (owner 2026-11-01). The step a kind's condition first holds is its *first* step (``first``, reported
as ``*_first_step``); the step the episode is judged at is the *chosen* step (``best``, reported as ``*_success_step``):

* off mode (an unsettled success): among the steps ``t`` in ``[first, first + W - 1]`` (``W`` = ``settle_steps``) where the kind's
  condition (``G``; ``G`` and ``M``) holds, the one with the lowest :func:`selection_score`, ties to the earlier step; with no
  computable score (no ``:measure`` atoms) the first step;
* on mode (a settled success): the first step, as before - the settled window already certifies the state.

The errors, error metric and settled flag a mode is judged by are those of its chosen geometric step, else its chosen symbolic
step, else the current step. The run mode's geometric selection is *final* (:attr:`SuccessTracker.final`, the episode's
termination and a stopping run's end) at the first settled geometric success in the on mode, and at ``first + W - 1`` in the off
mode (``W`` = 1 gives the earlier first-success behaviour).

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


def _ratio(error, tolerance) -> Optional[float]:
    """``error / tolerance`` (None when either is missing; a zero tolerance gives 0 for a zero error, else infinity)."""
    if error is None or tolerance is None:
        return None
    error, tolerance = abs(float(error)), float(tolerance)
    if tolerance == 0.0:
        return 0.0 if error == 0.0 else float("inf")
    return error / tolerance


def selection_score(errors: Dict[str, Any], atoms: Sequence[Tuple[str, str]]) -> Optional[float]:
    """The score the evaluated success step is chosen by (lower is better): the largest error-to-tolerance ratio over the
    ``(:measure ...)`` atoms' reports in ``errors``. A pose or group report contributes ``position_error`` / its position tolerance
    and ``orientation_error`` / its orientation tolerance (the report's ``tolerances``); an articulation report contributes every
    joint's ``error`` / that joint's ``tolerance``. Missing components are skipped; None when no ratio can be formed (no atoms, no
    reports)."""
    ratios: List[float] = []
    for predicate, inst in atoms:
        rep = _report(errors, predicate, inst)
        if rep is None:
            continue
        joints = rep.get("joints")
        if isinstance(joints, dict) and joints:
            for e in joints.values():
                if isinstance(e, dict):
                    r = _ratio(e.get("error"), e.get("tolerance"))
                    if r is not None:
                        ratios.append(r)
            continue
        tol = rep.get("tolerances") or {}
        for err_key, tol_key in (("position_error", "position"), ("orientation_error", "orientation")):
            r = _ratio(rep.get(err_key), tol.get(tol_key))
            if r is not None:
                ratios.append(r)
    return max(ratios) if ratios else None


def _better(score: Optional[float], best: Optional[float]) -> bool:
    """``score`` strictly beats ``best`` (a computable score beats none; ties keep the earlier step)."""
    if score is None:
        return False
    return best is None or score < best


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
    """Symbolic and geometric success of the run's settle-eval mode, and of the off mode alongside settle-eval on, step by step,
    with the evaluated (chosen) success step of each (module docstring). ``atoms``: the ``(:measure ...)`` atoms the off mode's
    selection score reads (none: every chosen step is the first step)."""

    def __init__(self, settle_eval: bool = False, settle_steps: int = 45, atoms: Optional[Sequence[Tuple[str, str]]] = None):
        if int(settle_steps) < 1:
            raise ValueError(f"settle_steps must be >= 1, got {settle_steps}")
        self.settle_eval = bool(settle_eval)
        self.settle_steps = int(settle_steps)
        self.atoms = [(str(p), str(a)) for p, a in (atoms or [])]
        self.reset()

    def reset(self) -> None:
        #: consecutive steps up to the last update where G (G and M) held
        self.g_run = 0
        self.gm_run = 0
        #: the first step each kind's condition held, per mode
        self.first: Dict[str, Dict[str, Optional[int]]] = {m: {k: None for k in KINDS} for m in self.modes}
        #: the chosen (evaluated) step, its score, its errors and its settled flag, per mode and kind
        self.best: Dict[str, Dict[str, Optional[int]]] = {m: {k: None for k in KINDS} for m in self.modes}
        self.best_score: Dict[str, Dict[str, Optional[float]]] = {m: {k: None for k in KINDS} for m in self.modes}
        self.errors_at: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = {m: {k: None for k in KINDS} for m in self.modes}
        self.settled_at: Dict[str, Dict[str, Optional[bool]]] = {m: {k: None for k in KINDS} for m in self.modes}
        #: the errors at the first step (kept for comparison; the evaluation uses ``errors_at``)
        self.errors_first: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = {m: {k: None for k in KINDS} for m in self.modes}
        self.last: Optional[Tuple[int, bool, bool, bool]] = None

    @property
    def mode(self) -> str:
        return mode_name(self.settle_eval)

    @property
    def other_mode(self) -> Optional[str]:
        """The mode reported alongside the run's: the off mode of a settle-eval-on run, None with settle-eval off."""
        return "off" if self.settle_eval else None

    @property
    def modes(self) -> Tuple[str, ...]:
        """The modes followed: both with settle-eval on, only the off mode with settle-eval off."""
        return MODES if self.settle_eval else ("off",)

    def _choose(self, mode: str, kind: str, t: int, score: Optional[float], errors, settled: bool) -> None:
        self.best[mode][kind] = int(t)
        self.best_score[mode][kind] = score
        self.errors_at[mode][kind] = errors
        self.settled_at[mode][kind] = bool(settled)

    def update(self, t: int, goal: bool, within: bool, settled: bool, errors: Optional[Dict[str, Any]] = None) -> None:
        """The facts of step ``t`` (``G``, ``M``, ``S``) and its errors."""
        goal, within, settled = bool(goal), bool(within), bool(settled)
        t = int(t)
        self.g_run = self.g_run + 1 if goal else 0
        self.gm_run = self.gm_run + 1 if (goal and within) else 0
        n = self.settle_steps
        score = selection_score(errors, self.atoms) if self.atoms else None
        # off mode: the first step, then the better candidates of [first, first + W - 1]
        for kind, hit in (("symbolic", goal), ("geometric", goal and within)):
            first = self.first["off"][kind]
            if not hit:
                continue
            if first is None:
                self.first["off"][kind] = t
                self.errors_first["off"][kind] = errors
                self._choose("off", kind, t, score, errors, settled)
            elif t <= first + n - 1 and _better(score, self.best_score["off"][kind]):
                self._choose("off", kind, t, score, errors, settled)
        # on mode (a settled success): the first settled step is the evaluated step, as before
        if self.settle_eval:
            for kind, hit in (("symbolic", settled and self.g_run >= n), ("geometric", settled and self.gm_run >= n)):
                if not hit or self.first["on"][kind] is not None:
                    continue
                self.first["on"][kind] = t
                self.errors_first["on"][kind] = errors
                self._choose("on", kind, t, score, errors, settled)
        self.last = (t, goal, within, settled)

    def step_of(self, kind: str, mode: Optional[str] = None) -> Optional[int]:
        """The chosen (evaluated) step of ``kind`` in ``mode`` (default the run's)."""
        return self.best[mode or self.mode][kind]

    def first_step_of(self, kind: str, mode: Optional[str] = None) -> Optional[int]:
        """The first step ``kind``'s condition held in ``mode``."""
        return self.first[mode or self.mode][kind]

    def reached(self, kind: str, mode: Optional[str] = None) -> bool:
        return self.first[mode or self.mode][kind] is not None

    @property
    def done(self) -> bool:
        """The run mode's geometric success reached (``info["success"]``)."""
        return self.reached("geometric")

    def selection_final(self, kind: str = "geometric", mode: Optional[str] = None) -> bool:
        """The chosen step of ``kind`` can no longer change: reached, and in the off mode the step ``first + W - 1`` is updated."""
        mode = mode or self.mode
        first = self.first[mode][kind]
        if first is None:
            return False
        if mode == "on":
            return True
        return self.last is not None and self.last[0] >= first + self.settle_steps - 1

    @property
    def final(self) -> bool:
        """The run mode's geometric selection is final (the episode's termination; a stopping run ends here)."""
        return self.selection_final("geometric")

    @property
    def goal_first_step(self) -> Optional[int]:
        """The first step ``G`` held (the off mode's first symbolic step)."""
        return self.first["off"]["symbolic"]

    def _evaluation_kind(self, mode: str) -> Optional[str]:
        if self.best[mode]["geometric"] is not None:
            return "geometric"
        if self.best[mode]["symbolic"] is not None:
            return "symbolic"
        return None

    def evaluation_errors(self, current: Dict[str, Any], mode: Optional[str] = None) -> Dict[str, Any]:
        """The errors a mode is judged by: at its chosen geometric step, else its chosen symbolic step, else ``current``."""
        mode = mode or self.mode
        kind = self._evaluation_kind(mode)
        return self.errors_at[mode][kind] if kind is not None else current

    def mode_report(self, mode: str, current: Dict[str, Any], atoms: Sequence[Tuple[str, str]], where: str = "") -> Dict[str, Any]:
        errs = self.evaluation_errors(current, mode)
        kind = self._evaluation_kind(mode)
        sym, geo = self.best[mode]["symbolic"], self.best[mode]["geometric"]
        return {"settle_eval": mode == "on", "symbolic_success": self.first[mode]["symbolic"] is not None, "symbolic_success_step": sym,
                "symbolic_first_step": self.first[mode]["symbolic"], "geometric_success": self.first[mode]["geometric"] is not None,
                "geometric_success_step": geo, "geometric_first_step": self.first[mode]["geometric"],
                "evaluation_step": None if kind is None else self.best[mode][kind], "evaluation_kind": kind,
                "evaluation_settled": None if kind is None else self.settled_at[mode][kind],
                "success_final": self.selection_final("geometric", mode),
                "errors": errs, "error_metric": error_metric(errs, atoms, where) if atoms else None}

    def report(self, current: Dict[str, Any], atoms: Sequence[Tuple[str, str]], where: str = "") -> Dict[str, Any]:
        """The run mode's report (its ``errors`` / ``error_metric`` at its chosen geometric step, else its chosen symbolic step,
        else at ``current``) with the settle window and ``other_mode``: the same for the off mode, without its errors, with
        settle-eval on; None with settle-eval off."""
        out = self.mode_report(self.mode, current, atoms, where)
        other = None
        if self.other_mode is not None:
            other = self.mode_report(self.other_mode, current, atoms, where)
            other.pop("errors")
        out.update({"settle_steps": self.settle_steps, "goal_first_step": self.goal_first_step, "other_mode": other})
        return out

    def state(self) -> Dict[str, Any]:
        """A copy for a checkpoint (:meth:`load` restores it)."""
        return {"settle_eval": self.settle_eval, "settle_steps": self.settle_steps, "atoms": list(self.atoms), "g_run": self.g_run,
                "gm_run": self.gm_run, "first": copy.deepcopy(self.first), "best": copy.deepcopy(self.best),
                "best_score": copy.deepcopy(self.best_score), "errors_at": {m: dict(v) for m, v in self.errors_at.items()},
                "settled_at": copy.deepcopy(self.settled_at), "errors_first": {m: dict(v) for m, v in self.errors_first.items()},
                "last": self.last}

    def load(self, state: Dict[str, Any]) -> None:
        self.settle_eval, self.settle_steps = bool(state["settle_eval"]), int(state["settle_steps"])
        self.atoms = list(state.get("atoms", self.atoms))
        self.g_run, self.gm_run = int(state["g_run"]), int(state["gm_run"])
        self.first = copy.deepcopy(state["first"])
        self.best = copy.deepcopy(state["best"])
        self.best_score = copy.deepcopy(state["best_score"])
        self.errors_at = {m: dict(v) for m, v in state["errors_at"].items()}
        self.settled_at = copy.deepcopy(state["settled_at"])
        self.errors_first = {m: dict(v) for m, v in state["errors_first"].items()}
        self.last = state["last"]


__all__ = ["MODES", "KINDS", "SUCCESS_METRICS", "ERROR_METRIC_CASES", "ErrorMetricError", "SuccessTracker", "mode_name", "measure_atoms",
           "measure_within", "measure_outside", "error_metric_case", "error_metric", "selection_score"]
