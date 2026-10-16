"""Episode scoring: symbolic and geometric success under both settle-eval modes, and the error metric.

Per step ``t`` the task reads three facts:

* ``G(t)``: every goal atom except ``pose_settled`` holds;
* ``M(t)``: every ``(:measure ...)`` atom is within its tolerance;
* ``S(t)``: every ``pose_settled`` atom holds (its window of ``settle_steps`` steps ending at ``t`` is full and within the
  drift thresholds; vacuously true without such atoms).

**Symbolic** success is the BDDL goal; **geometric** success is symbolic success with every measured atom within tolerance
at the same step(s):

* settle eval off: symbolic = the first ``t`` with ``G(t)``; geometric = the first ``t`` with ``G(t)`` and ``M(t)``;
* settle eval on: symbolic = the first ``t`` with ``S(t)`` where ``G`` held at every step of the window ending at ``t``;
  geometric = the first such ``t`` where ``G`` and ``M`` held at every step of the window.

A settle-eval-on run also reports the off mode as ``other_mode``. Steps are the task's 0-based episode steps; a value the
episode never reached is ``None``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

MODES = ("off", "on")
KINDS = ("symbolic", "geometric")
SUCCESS_METRICS = ("geometric", "symbolic")


class ErrorMetricError(ValueError):
    """A ``(:measure ...)`` combination :func:`error_metric` has no case for."""


def mode_name(settle_eval: bool) -> str:
    return "on" if settle_eval else "off"


def measure_atoms(problem) -> list[tuple[str, str]]:
    """``[(predicate, instance)]`` of a problem's ``(:measure ...)`` atoms."""
    return [(str(lit[0]), str(lit[1]).lstrip("?")) for lit in getattr(problem, "measure", None) or []]


def _report(errors: dict[str, Any], predicate: str, inst: str) -> dict[str, Any] | None:
    reports = (errors or {}).get("pose_errors" if predicate == "pose_reached" else "layout_errors") or {}
    rep = reports.get(inst)
    return rep if isinstance(rep, dict) else None


def measure_within(errors: dict[str, Any], atoms: Sequence[tuple[str, str]]) -> bool:
    """``M``: every measured atom's report is within tolerance (an atom without a report is not)."""
    return all(bool((_report(errors, p, a) or {}).get("within_tolerance")) for p, a in atoms)


def measure_outside(errors: dict[str, Any], atoms: Sequence[tuple[str, str]]) -> list[str]:
    """The measured atoms (``"pose_reached ?x"``) not within tolerance."""
    return [f"{p} ?{a}" for p, a in atoms if not bool((_report(errors, p, a) or {}).get("within_tolerance"))]


def _max(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return max(vals) if vals else None


def error_metric_case(atoms: Sequence[tuple[str, str]], where: str = "") -> str:
    """The case of a ``(:measure ...)`` block.

    ``poses``: two or more ``pose_reached`` and no layout; ``layout``: ``layout_reached`` and ``pose_reached`` on the same
    entity; ``pose``: one ``pose_reached``; ``layout_only``: one ``layout_reached`` and no pose. Anything else raises.
    """
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
    raise ErrorMetricError(f"{where or 'problem'}: no error_metric case for the (:measure ...) atoms {atoms_s}")


def error_metric(errors: dict[str, Any], atoms: Sequence[tuple[str, str]], where: str = "") -> dict[str, Any]:
    """``{"position_error", "orientation_error", "case"}`` of the measured atoms (:func:`error_metric_case`).

    ``poses`` takes the max over the pose reports, ``layout`` / ``layout_only`` the layout report, ``pose`` the pose
    report. A component a target does not compare is ``None``.
    """
    case = error_metric_case(atoms, where)
    if case == "poses":
        reps = [_report(errors, "pose_reached", a) or {} for p, a in atoms if p == "pose_reached"]
        pos, ori = _max(r.get("position_error") for r in reps), _max(r.get("orientation_error") for r in reps)
    else:
        pred = "pose_reached" if case == "pose" else "layout_reached"
        rep = _report(errors, pred, next(a for p, a in atoms if p == pred)) or {}
        pos, ori = rep.get("position_error"), rep.get("orientation_error")
    return {
        "position_error": None if pos is None else float(pos),
        "orientation_error": None if ori is None else float(ori),
        "case": case,
    }


class SuccessTracker:
    """Symbolic and geometric success of the run's settle-eval mode (and of the off mode alongside settle eval on)."""

    def __init__(self, settle_eval: bool = False, settle_steps: int = 45):
        if int(settle_steps) < 1:
            raise ValueError(f"settle_steps must be >= 1, got {settle_steps}")
        self.settle_eval = bool(settle_eval)
        self.settle_steps = int(settle_steps)
        self.reset()

    def reset(self) -> None:
        self.g_run = 0  # consecutive steps up to the last update where G held
        self.gm_run = 0  # ... where G and M held
        self.first: dict[str, dict[str, int | None]] = {m: {k: None for k in KINDS} for m in self.modes}
        self.errors_at: dict[str, dict[str, dict[str, Any] | None]] = {m: {k: None for k in KINDS} for m in self.modes}
        self.last: tuple[int, bool, bool, bool] | None = None

    @property
    def mode(self) -> str:
        return mode_name(self.settle_eval)

    @property
    def other_mode(self) -> str | None:
        return "off" if self.settle_eval else None

    @property
    def modes(self) -> tuple[str, ...]:
        return MODES if self.settle_eval else ("off",)

    def update(self, t: int, goal: bool, within: bool, settled: bool, errors: dict[str, Any] | None = None) -> None:
        """The facts ``G``, ``M``, ``S`` of step ``t`` and its errors (kept at every first success)."""
        goal, within, settled = bool(goal), bool(within), bool(settled)
        self.g_run = self.g_run + 1 if goal else 0
        self.gm_run = self.gm_run + 1 if (goal and within) else 0
        hits = {("off", "symbolic"): goal, ("off", "geometric"): goal and within}
        if self.settle_eval:
            n = self.settle_steps
            hits.update(
                {("on", "symbolic"): settled and self.g_run >= n, ("on", "geometric"): settled and self.gm_run >= n}
            )
        for (mode, kind), hit in hits.items():
            if hit and self.first[mode][kind] is None:
                self.first[mode][kind] = int(t)
                self.errors_at[mode][kind] = errors
        self.last = (int(t), goal, within, settled)

    def step_of(self, kind: str, mode: str | None = None) -> int | None:
        return self.first[mode or self.mode][kind]

    @property
    def done(self) -> bool:
        """The run mode's geometric success: the episode's termination."""
        return self.step_of("geometric") is not None

    @property
    def goal_first_step(self) -> int | None:
        return self.first["off"]["symbolic"]

    def evaluation_errors(self, current: dict[str, Any], mode: str | None = None) -> dict[str, Any]:
        """The errors a mode is judged by: at its geometric success, else its symbolic success, else ``current``."""
        snaps = self.errors_at[mode or self.mode]
        snap = snaps["geometric"] if snaps["geometric"] is not None else snaps["symbolic"]
        return snap if snap is not None else current

    def mode_report(
        self, mode: str, current: dict[str, Any], atoms: Sequence[tuple[str, str]], where: str = ""
    ) -> dict[str, Any]:
        errs = self.evaluation_errors(current, mode)
        sym, geo = self.first[mode]["symbolic"], self.first[mode]["geometric"]
        return {
            "settle_eval": mode == "on",
            "symbolic_success": sym is not None,
            "symbolic_success_step": sym,
            "geometric_success": geo is not None,
            "geometric_success_step": geo,
            "errors": errs,
            "error_metric": error_metric(errs, atoms, where) if atoms else None,
        }

    def report(self, current: dict[str, Any], atoms: Sequence[tuple[str, str]], where: str = "") -> dict[str, Any]:
        """The run mode's report with the settle window and ``other_mode`` (the off mode without its errors)."""
        out = self.mode_report(self.mode, current, atoms, where)
        other = None
        if self.other_mode is not None:
            other = self.mode_report(self.other_mode, current, atoms, where)
            other.pop("errors")
        out.update({"settle_steps": self.settle_steps, "goal_first_step": self.goal_first_step, "other_mode": other})
        return out
