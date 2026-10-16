import pytest

from ispatialgym.eval import metrics as MET
from ispatialgym.eval.executor import verdict

POSE, LAYOUT = "pose_reached", "layout_reached"


def report(pos, ori, within=True):
    return {"position_error": pos, "orientation_error": ori, "within_tolerance": within}


def errors(poses=None, layouts=None):
    return {"pose_errors": dict(poses or {}), "layout_errors": dict(layouts or {})}


def run(tracker, facts):
    """``facts``: (G, M, S) per step from step 0; the errors of step t are ``{"t": t}``."""
    for t, (g, m, s) in enumerate(facts):
        tracker.update(t, g, m, s, {"t": t})
    return tracker


def test_error_metric_cases():
    poses = errors({"t": report(0.01, None), "p": report(0.03, 0.2)})
    assert MET.error_metric(poses, [(POSE, "t"), (POSE, "p")]) == {
        "position_error": 0.03,
        "orientation_error": 0.2,
        "case": "poses",
    }
    layout = errors({"g": report(0.02, 0.3)}, {"g": report(None, 0.1)})
    assert MET.error_metric(layout, [(POSE, "g"), (LAYOUT, "g")]) == {
        "position_error": None,
        "orientation_error": 0.1,
        "case": "layout",
    }
    assert MET.error_metric(errors(), [(POSE, "a")]) == {
        "position_error": None,
        "orientation_error": None,
        "case": "pose",
    }
    with pytest.raises(MET.ErrorMetricError):
        MET.error_metric_case([(LAYOUT, "a"), (LAYOUT, "b")], "problem")


def test_measure_within_requires_every_report():
    atoms = [(POSE, "t"), (LAYOUT, "d")]
    assert MET.measure_within(errors({"t": report(0, 0)}, {"d": report(0, 0)}), atoms)
    outside = errors({"t": report(0.1, 0, within=False)}, {"d": report(0, 0)})
    assert not MET.measure_within(outside, atoms) and MET.measure_outside(outside, atoms) == ["pose_reached ?t"]
    assert not MET.measure_within(errors({"t": report(0, 0)}), atoms)


def test_settle_eval_on_needs_the_goal_through_the_window():
    facts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 1, 1), (1, 1, 1)]
    tracker = run(MET.SuccessTracker(settle_eval=True, settle_steps=3), facts)
    assert tracker.first["off"] == {"symbolic": 1, "geometric": 2}
    assert tracker.first["on"] == {"symbolic": 3, "geometric": 4}
    assert tracker.done and tracker.errors_at["on"]["geometric"] == {"t": 4}


def test_settle_eval_off_succeeds_at_the_first_step_and_has_no_other_mode():
    tracker = run(MET.SuccessTracker(settle_eval=False, settle_steps=3), [(1, 0, 0), (1, 1, 0)])
    assert tracker.step_of("geometric") == 1 and tracker.goal_first_step == 0
    rep = tracker.report({"t": "now"}, [(POSE, "t")])
    assert rep["other_mode"] is None and rep["errors"] == {"t": 1}


def test_errors_are_taken_at_the_cutoff_without_success():
    tracker = run(MET.SuccessTracker(settle_eval=True, settle_steps=2), [(1, 1, 0)] * 4)
    current = {"t": "cutoff"}
    rep = tracker.report(current, [(POSE, "t")])
    assert not rep["geometric_success"] and not rep["symbolic_success"] and rep["errors"] is current
    assert rep["other_mode"]["geometric_success_step"] == 0


def test_errors_fall_back_to_symbolic_success_when_geometric_not_reached():
    tracker = run(MET.SuccessTracker(settle_eval=False, settle_steps=2), [(0, 0, 0), (1, 0, 0), (0, 0, 0)])
    rep = tracker.report({"t": "cutoff"}, [(POSE, "t")])
    assert rep["symbolic_success"] and rep["symbolic_success_step"] == 1
    assert not rep["geometric_success"] and rep["errors"] == {"t": 1}


def test_verdict_reasons():
    assert verdict({"geometric_success": True, "symbolic_success": True}) is None
    assert verdict({"symbolic_success": True}, success_metric="symbolic") is None
    measured = [{"bddl": "(pose_reached ?a)", "within_tolerance": False}]
    assert verdict({"symbolic_success": True, "evaluation": {"measured": measured}})["reason"] == "symbolic_only"
    assert verdict({"settle_eval": True, "goal_first_step": 3})["reason"] == "not_settled"
    unsatisfied = verdict({"conditions": [{"bddl": "(ontop ?a ?b)", "satisfied": False}]})
    assert unsatisfied["reason"] == "not_reached" and "(ontop ?a ?b)" in unsatisfied["message"]
