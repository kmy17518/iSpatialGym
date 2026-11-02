"""Episode scoring (``ispatialgym.eval.metrics``, owner 2026-10-14): the error metric of the ``(:measure ...)`` atoms and the
symbolic / geometric success of both settle-eval modes on synthetic step sequences."""

import pytest

from ispatialgym.eval import metrics as MET

POSE = "pose_reached"
LAYOUT = "layout_reached"


def pose(pos, ori, within=True):
    return {"position_error": pos, "orientation_error": ori, "within_tolerance": within}


def errors(poses=None, layouts=None):
    return {"pose_errors": dict(poses or {}), "layout_errors": dict(layouts or {})}


def test_error_metric_poses_takes_the_max_over_the_pose_reports():
    atoms = [(POSE, "t"), (POSE, "p")]
    e = errors({"t": pose(0.01, None), "p": pose(0.03, 0.2)})
    assert MET.error_metric(e, atoms) == {"position_error": 0.03, "orientation_error": 0.2, "case": "poses"}
    e = errors({"t": pose(0.01, None), "p": pose(0.02, None)})  # inside: no orientation compared
    assert MET.error_metric(e, atoms) == {"position_error": 0.02, "orientation_error": None, "case": "poses"}


def test_error_metric_layout_uses_the_layout_report_of_the_entity():
    atoms = [(POSE, "g"), (LAYOUT, "g")]
    e = errors({"g": pose(0.02, 0.3)}, {"g": pose(None, 0.1)})  # a revolute-only articulation: no position component
    assert MET.error_metric(e, atoms) == {"position_error": None, "orientation_error": 0.1, "case": "layout"}


def test_error_metric_pose_and_layout_only():
    assert MET.error_metric(errors({"a": pose(0.004, 0.06)}), [(POSE, "a")]) == {"position_error": 0.004, "orientation_error": 0.06, "case": "pose"}
    assert MET.error_metric(errors(layouts={"d": pose(0.05, None)}), [(LAYOUT, "d")]) == {"position_error": 0.05, "orientation_error": None,
                                                                                          "case": "layout_only"}
    assert MET.error_metric(errors(), [(POSE, "a")]) == {"position_error": None, "orientation_error": None, "case": "pose"}  # no report: null


@pytest.mark.parametrize("atoms", [[], [(LAYOUT, "a"), (LAYOUT, "b")], [(POSE, "a"), (LAYOUT, "b")], [(POSE, "a"), (POSE, "b"), (LAYOUT, "a")],
                                   [("ontop", "a")]])
def test_error_metric_unknown_combination_raises_naming_the_problem(atoms):
    with pytest.raises(MET.ErrorMetricError, match="alignment-3 .*no error_metric case"):
        MET.error_metric_case(atoms, "alignment-3 (alignment-axial-x-y)")


def test_measure_within_and_outside():
    atoms = [(POSE, "t"), (LAYOUT, "d")]
    assert MET.measure_within(errors({"t": pose(0.0, 0.0)}, {"d": pose(0.0, 0.0)}), atoms)
    e = errors({"t": pose(0.1, 0.0, within=False)}, {"d": pose(0.0, 0.0)})
    assert not MET.measure_within(e, atoms) and MET.measure_outside(e, atoms) == ["pose_reached ?t"]
    assert not MET.measure_within(errors({"t": pose(0.0, 0.0)}), atoms)  # a missing report is not within
    assert MET.measure_within(errors(), [])


def run(tracker, seq):
    """``seq``: (G, M, S) per step from step 0; the errors of step t are ``{"t": t}``."""
    for t, (g, m, s) in enumerate(seq):
        tracker.update(t, g, m, s, {"t": t})
    return tracker


def test_pass_through_both_modes():
    seq = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)]
    tr = run(MET.SuccessTracker(settle_eval=True, settle_steps=3), seq)
    assert tr.first["off"] == {"symbolic": 1, "geometric": 2}
    # on: S first at 3, G held at 1..3 (window of 3 ending at 3); G and M held at 2..4 -> geometric at 4
    assert tr.first["on"] == {"symbolic": 3, "geometric": 4}
    assert tr.done and tr.goal_first_step == 1 and tr.errors_at["on"]["geometric"] == {"t": 4}
    off = run(MET.SuccessTracker(settle_eval=False, settle_steps=3), seq)
    assert off.done and off.step_of("geometric") == 2 and off.mode == "off" and off.other_mode is None


def test_goal_lost_inside_the_window():
    seq = [(1, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)]
    tr = run(MET.SuccessTracker(settle_eval=True, settle_steps=3), seq)
    assert tr.first["off"] == {"symbolic": 0, "geometric": 0}
    assert tr.first["on"] == {"symbolic": 5, "geometric": 5}  # S held from step 2, but G broke at 2: the window must restart


def test_measure_lost_inside_the_window_delays_geometric_only():
    seq = [(1, 1, 1), (1, 0, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)]
    tr = run(MET.SuccessTracker(settle_eval=True, settle_steps=2), seq)
    assert tr.first["on"] == {"symbolic": 1, "geometric": 3}


def test_never_settled_and_errors_at_the_cutoff():
    seq = [(1, 1, 0)] * 5
    tr = run(MET.SuccessTracker(settle_eval=True, settle_steps=2), seq)
    assert tr.first["on"] == {"symbolic": None, "geometric": None} and not tr.done
    current = {"t": "cutoff"}
    rep = tr.report(current, [(POSE, "t")])
    assert rep["geometric_success"] is False and rep["geometric_success_step"] is None and rep["errors"] is current
    other = rep["other_mode"]
    assert {k: other[k] for k in ("settle_eval", "symbolic_success", "symbolic_success_step", "geometric_success", "geometric_success_step",
                                  "error_metric")} == {"settle_eval": False, "symbolic_success": True, "symbolic_success_step": 0, "geometric_success": True,
                                                       "geometric_success_step": 0, "error_metric": {"position_error": None, "orientation_error": None, "case": "pose"}}
    # no score computable (the errors carry no reports): the chosen steps are the first steps
    assert (other["symbolic_first_step"], other["geometric_first_step"], other["evaluation_step"], other["evaluation_kind"]) == (0, 0, 0, "geometric")
    assert other["success_final"] is True and "errors" not in other
    assert tr.evaluation_errors(current, "off") == {"t": 0}


def test_settle_eval_off_follows_no_other_mode():
    tr = run(MET.SuccessTracker(settle_eval=False, settle_steps=3), [(1, 1, 1)] * 6)  # an on-mode window would fill at step 2
    assert set(tr.first) == set(tr.errors_at) == {"off"}
    rep = tr.report({"t": "cutoff"}, [(POSE, "t")])
    assert rep["other_mode"] is None and rep["geometric_success_step"] == 0 and rep["goal_first_step"] == 0
    with pytest.raises(KeyError):
        tr.step_of("geometric", "on")  # not followed, rather than never reached


def test_errors_fall_back_to_symbolic_success_when_geometric_not_reached():
    tr = run(MET.SuccessTracker(settle_eval=False, settle_steps=2), [(0, 0, 0), (1, 0, 0), (0, 0, 0)])
    rep = tr.report({"t": "cutoff"}, [(POSE, "t")])
    assert rep["symbolic_success"] and rep["symbolic_success_step"] == 1
    assert not rep["geometric_success"] and rep["errors"] == {"t": 1}


def test_no_goal_ever():
    tr = run(MET.SuccessTracker(settle_eval=False, settle_steps=1), [(0, 1, 1)] * 4)
    assert tr.first == {"off": {"symbolic": None, "geometric": None}} and tr.goal_first_step is None
    tr = run(MET.SuccessTracker(settle_eval=True, settle_steps=1), [(0, 1, 1)] * 4)
    assert tr.first == {"off": {"symbolic": None, "geometric": None}, "on": {"symbolic": None, "geometric": None}} and tr.goal_first_step is None


def test_state_and_load_restore_a_rollback():
    tr = run(MET.SuccessTracker(settle_eval=True, settle_steps=3), [(1, 1, 0), (1, 1, 0)])
    cp = tr.state()
    run_after = [(0, 0, 0), (0, 0, 0)]
    for t, (g, m, s) in enumerate(run_after, start=2):
        tr.update(t, g, m, s, {"t": t})
    assert tr.g_run == 0
    tr.load(cp)
    assert (tr.g_run, tr.gm_run, tr.last) == (2, 2, (1, True, True, False))
    tr.update(2, 1, 1, 1, {"t": 2})  # the retried attempt continues the window
    assert tr.first["on"] == {"symbolic": 2, "geometric": 2}
    cp["first"]["off"]["symbolic"] = 99  # the checkpoint is a copy
    assert tr.first["off"]["symbolic"] == 0


def test_settle_steps_must_be_positive():
    with pytest.raises(ValueError):
        MET.SuccessTracker(settle_steps=0)


# ---------------------------------------------------------------------------------------------- the evaluated success step (2026-11-01)

ATOM = [(POSE, "x")]


def perr(pos, ori, within=True, tol=(0.02, 0.1)):
    """Errors with one pose report on ``x`` (position / orientation tolerances ``tol``)."""
    return errors(poses={"x": {"position_error": pos, "orientation_error": ori, "within_tolerance": within,
                               "tolerances": {"position": tol[0], "orientation": tol[1]}}})


def feed(tracker, steps):
    """``steps``: (G, M, S, errors) per step from step 0."""
    for t, (g, m, s, e) in enumerate(steps):
        tracker.update(t, g, m, s, e)
    return tracker


def test_selection_score_is_the_largest_error_to_tolerance_ratio():
    assert MET.selection_score(perr(0.01, 0.02), ATOM) == pytest.approx(0.5)  # max(0.01/0.02, 0.02/0.1)
    assert MET.selection_score(perr(0.001, 0.08), ATOM) == pytest.approx(0.8)
    # an orientation-free target: only its position ratio
    assert MET.selection_score(perr(0.01, None, tol=(0.02, None)), ATOM) == pytest.approx(0.5)
    # an articulation: every joint against its own tolerance
    joints = {"j1": {"error": 0.05, "tolerance": 0.1}, "j2": {"error": 0.03, "tolerance": 0.05}}
    art = errors(layouts={"a": {"position_error": None, "orientation_error": 0.05, "within_tolerance": True, "joints": joints,
                                "tolerances": {"joints": {"j1": 0.1, "j2": 0.05}}}})
    assert MET.selection_score(art, [(LAYOUT, "a")]) == pytest.approx(0.6)
    # a zero tolerance: 0 for a zero error, else infinity; nothing to score: None
    assert MET.selection_score(perr(0.0, 0.0, tol=(0.0, 0.0)), ATOM) == 0.0
    assert MET.selection_score(perr(0.001, 0.0, tol=(0.0, 0.1)), ATOM) == float("inf")
    assert MET.selection_score(errors(), ATOM) is None and MET.selection_score(perr(0.01, 0.02), []) is None


def test_off_mode_chooses_the_lowest_score_within_the_window_after_the_first_success():
    # W = 3: candidates are the steps 1..3 where G and M hold; step 2 is best; step 4 (better still) is outside the window
    steps = [(0, 0, 0, perr(0.0, 0.0)), (1, 1, 0, perr(0.015, 0.0)), (1, 1, 0, perr(0.002, 0.0)), (1, 1, 0, perr(0.01, 0.0)),
             (1, 1, 0, perr(0.0, 0.0))]
    tr = feed(MET.SuccessTracker(settle_eval=False, settle_steps=3, atoms=ATOM), steps)
    assert tr.first["off"] == {"symbolic": 1, "geometric": 1}
    assert tr.best["off"] == {"symbolic": 2, "geometric": 2}
    rep = tr.report(perr(9, 9), ATOM)
    assert (rep["geometric_success_step"], rep["geometric_first_step"], rep["evaluation_step"]) == (2, 1, 2)
    assert rep["error_metric"]["position_error"] == pytest.approx(0.002)


def test_off_mode_candidates_need_the_kinds_condition_and_ties_keep_the_earlier_step():
    # step 2 has the lowest error but M fails there: no geometric candidate; symbolic (G only) may take it
    steps = [(1, 1, 0, perr(0.01, 0.0)), (1, 1, 0, perr(0.01, 0.0)), (1, 0, 0, perr(0.001, 0.0, within=False)), (1, 1, 0, perr(0.01, 0.0))]
    tr = feed(MET.SuccessTracker(settle_eval=False, settle_steps=4, atoms=ATOM), steps)
    assert tr.best["off"]["geometric"] == 0  # 0, 1, 3 tie: the earliest
    assert tr.best["off"]["symbolic"] == 2


def test_off_mode_selection_is_final_at_the_window_end_and_a_one_step_window_is_the_first_success():
    steps = [(0, 0, 0, perr(0.0, 0.0))] + [(1, 1, 0, perr(0.01 - 0.001 * k, 0.0)) for k in range(6)]
    tr = MET.SuccessTracker(settle_eval=False, settle_steps=3, atoms=ATOM)
    finals = []
    for t, (g, m, s, e) in enumerate(steps):
        tr.update(t, g, m, s, e)
        finals.append(tr.final)
    assert finals == [False, False, False, True, True, True, True]  # first at 1, window 1..3
    assert tr.done and tr.best["off"]["geometric"] == 3
    one = feed(MET.SuccessTracker(settle_eval=False, settle_steps=1, atoms=ATOM), steps)
    assert one.best["off"]["geometric"] == one.first["off"]["geometric"] == 1 and one.final


def test_on_mode_keeps_the_first_settled_step_and_the_off_mode_alongside_chooses():
    # W = 3: G and M from step 1, S from step 3 -> first settled geometric success at 3. A settled success is judged at that step as
    # before (no selection), even though step 2 has a lower error; the off mode reported alongside (an unsettled success) chooses 2
    steps = [(0, 0, 0, perr(0.0, 0.0)), (1, 1, 0, perr(0.01, 0.0)), (1, 1, 0, perr(0.003, 0.0)), (1, 1, 1, perr(0.008, 0.0)),
             (1, 1, 1, perr(0.0, 0.0))]
    tr = feed(MET.SuccessTracker(settle_eval=True, settle_steps=3, atoms=ATOM), steps)
    assert tr.first["on"]["geometric"] == tr.best["on"]["geometric"] == 3
    assert tr.settled_at["on"]["geometric"] is True and tr.final
    rep = tr.report(perr(9, 9), ATOM)
    assert (rep["geometric_success_step"], rep["geometric_first_step"], rep["evaluation_step"], rep["evaluation_settled"]) == (3, 3, 3, True)
    assert rep["error_metric"]["position_error"] == pytest.approx(0.008)  # the errors at the first settled step
    assert (rep["other_mode"]["geometric_first_step"], rep["other_mode"]["geometric_success_step"]) == (1, 2)


def test_state_and_load_carry_the_selection():
    steps = [(1, 1, 0, perr(0.01, 0.0)), (1, 1, 0, perr(0.002, 0.0))]
    tr = feed(MET.SuccessTracker(settle_eval=False, settle_steps=4, atoms=ATOM), steps)
    cp = tr.state()
    tr.update(2, 1, 1, 0, perr(0.0, 0.0))
    assert tr.best["off"]["geometric"] == 2
    tr.load(cp)
    assert tr.best["off"]["geometric"] == 1 and tr.best_score["off"]["geometric"] == pytest.approx(0.1)
    tr.update(2, 1, 0, 0, perr(0.0, 0.0, within=False))  # the retried step fails M: the selection stays
    assert tr.best["off"]["geometric"] == 1
