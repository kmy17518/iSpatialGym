"""Scoring-only tests ported from of-v2maps2a33c696 / bundle8dab09c9."""
import math
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.eval import round_symmetry as RS
from ispatialgym.eval.geometry import orientation_error
from ispatialgym.eval.targets import PoseTarget
from ispatialgym.sampling import geometry as G


# ------------------------------------------------------------------ A4: the round-axis table and the success measure


def test_yaw_free_switch_default_on(monkeypatch):
    monkeypatch.delenv(RS.SUCCESS_YAW_FREE_ENV, raising=False)
    assert RS.yaw_free_on()  # default on since lead #100
    monkeypatch.setenv(RS.SUCCESS_YAW_FREE_ENV, "0")
    assert not RS.yaw_free_on()
    monkeypatch.setenv(RS.SUCCESS_YAW_FREE_ENV, "1")
    assert RS.yaw_free_on()


def test_round_table_is_the_icl_visual_symmetry():
    from ispatialgym.oracle.executor import revolution_axes

    tab = RS.round_axes()
    assert tab["trash_can/jlawet"]["axis"] == "z" and tab["plate/luhkiz"]["axis"] == "z"
    for k in ("briefcase/lwbctq", "wall_clock/jerval", "marker/xakoic"):  # asymmetric / a visible face or print: not free
        assert k not in tab, k
    cyl = revolution_axes()
    for k, v in tab.items():  # where both tables hold a model they agree on its axis (the QA frees what the measure frees)
        if k in cyl:
            assert cyl[k]["axis"] == v["axis"], k


def test_with_axis_composes_and_is_idempotent():
    eqs, n_new = RS.with_axis([], "z")
    assert len(eqs) == 71 and n_new == 71
    again, n2 = RS.with_axis(eqs, "z")
    assert n2 == 0 and len(again) == 71
    half_x = [[1.0, 0.0, 0.0, 0.0]]  # a x2 model gets the product group (both flips x every spin)
    both, _ = RS.with_axis(half_x, "z")
    assert len(both) == 143


def _target(goal_q):
    return PoseTarget.from_dict("ashcan.n.01_1", {"frame": "world", "position": [0.0, 0.0, 0.1], "orientation": [float(v) for v in goal_q],
                                                   "position_tolerance": 0.03, "orientation_tolerance": 0.2})


def test_apply_yaw_free_frees_the_spin_only(monkeypatch):
    goal = G.quat_from_yaw(0.3)
    meta = SimpleNamespace(pose_targets={"ashcan.n.01_1": _target(goal), "box.n.01_1": _target(goal)})
    scope = {"ashcan.n.01_1": SimpleNamespace(category="trash_can", model="jlawet", name="target_trash_can_jlawet_0"),
             "box.n.01_1": SimpleNamespace(category="briefcase", model="lwbctq", name="briefcase_lwbctq_0")}
    monkeypatch.setenv(RS.SUCCESS_YAW_FREE_ENV, "0")
    assert RS.apply_yaw_free(meta, scope) == {}
    assert meta.pose_targets["ashcan.n.01_1"].orientation_equivalences == []
    monkeypatch.setenv(RS.SUCCESS_YAW_FREE_ENV, "1")
    rep = RS.apply_yaw_free(meta, scope)
    assert rep == {"ashcan.n.01_1": {"model": "trash_can/jlawet", "axis": "z", "acted": True, "n_equivalences": 71}}
    can, box = meta.pose_targets["ashcan.n.01_1"], meta.pose_targets["box.n.01_1"]
    assert box.orientation_equivalences == []
    spun = G.qmul(G.quat_from_yaw(0.3 + 0.62), G.IDENTITY_QUAT)  # the can turned 0.62 rad about the vertical
    tilted = G.qmul(G.quat_about_axis(th.tensor([1.0, 0.0, 0.0]), 0.25), goal)
    assert orientation_error(spun, can.orientation) > 0.6
    assert orientation_error(spun, can.orientation, can.orientation_equivalences) <= math.radians(2.5) + 1e-4
    assert orientation_error(tilted, can.orientation, can.orientation_equivalences) == pytest.approx(0.25, abs=1e-3)  # the tilt stays gated
    again = RS.apply_yaw_free(meta, scope)["ashcan.n.01_1"]  # idempotent: nothing added twice, the report stays "acted"
    assert again["acted"] is True and again["n_equivalences"] == 71


def test_entity_model_from_name():
    assert RS.entity_model(SimpleNamespace(name="trash_can_jlawet_0")) == ("trash_can", "jlawet")
    assert RS.entity_model(None) == (None, None)


def test_scoring_provenance_records_both_effective_switches(monkeypatch):
    monkeypatch.delenv(RS.SUCCESS_YAW_FREE_ENV, raising=False)
    monkeypatch.delenv('ISPATIALGYM_GROUP_ASSIGN_POSE_TIEBREAK', raising=False)
    before = RS.scoring_provenance()
    assert before['yaw_free_symmetric'] and before['group_pose_tiebreak']
    assert len(before['round_axes_sha256']) == 64
    monkeypatch.setenv(RS.SUCCESS_YAW_FREE_ENV, '0')
    after = RS.scoring_provenance()
    assert not after['yaw_free_symmetric'] and after['group_pose_tiebreak']
    assert before['round_axes_sha256'] == after['round_axes_sha256']
