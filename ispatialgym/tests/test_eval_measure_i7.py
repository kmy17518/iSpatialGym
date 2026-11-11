"""i7-measure 2026-11-09, the one eval-measure change (owner A2 + A4): under_large_02's briefcase dropped and pushed lying flat
(``ISPATIALGYM_BRIEFCASE_LYING``, iors_large) and a round object's rotation about its axis free in the success measure
(``ISPATIALGYM_SUCCESS_YAW_FREE_SYMMETRIC``, eval.round_symmetry), both default off."""
import math
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.eval import round_symmetry as RS
from ispatialgym.eval.geometry import orientation_error
from ispatialgym.eval.targets import PoseTarget
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.v2.tasks import iors_large as IL


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


# ------------------------------------------------------------------ A2: the lying briefcase


def _briefcase_hull():
    lo, hi = (-0.119, -0.164, -0.059), (0.151, 0.166, 0.060)  # lwbctq's collision box (object frame: x height, y long, z thin)
    return th.tensor([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], dtype=th.float32)


def test_switch_default_off(monkeypatch):
    monkeypatch.delenv(IL.BRIEFCASE_LYING_ENV, raising=False)
    assert not IL.briefcase_lying_on()
    monkeypatch.setenv(IL.BRIEFCASE_LYING_ENV, "1")
    assert IL.briefcase_lying_on()


def test_lying_rest_turns_the_thin_axis_up():
    rest = G.quat_about_axis(th.tensor([0.0, 1.0, 0.0]), -math.pi / 2)  # the library's r0: standing, the object x up
    hull = _briefcase_hull()
    q, facts = IL.lying_rest(hull, rest)
    assert facts["thin_axis"] == "z" and facts["tilt_deg"] == pytest.approx(90.0, abs=0.01)
    assert float(G.rotate(q, th.tensor([0.0, 0.0, 1.0]))[2]) == pytest.approx(1.0, abs=1e-5)
    pts = G.local_points_world(hull, (th.zeros(3), q))
    assert float(pts[:, 2].max() - pts[:, 2].min()) == pytest.approx(0.119, abs=1e-3)  # 0.12 tall lying
    qf, _ = IL.lying_rest(hull, rest, flip=True)
    assert float(G.rotate(qf, th.tensor([0.0, 0.0, 1.0]))[2]) == pytest.approx(-1.0, abs=1e-5)
    q0, f0 = IL.lying_rest(hull, G.IDENTITY_QUAT)  # already lying: nothing to tilt
    assert f0["tilt_deg"] == 0.0 and th.allclose(q0, G.IDENTITY_QUAT)


def test_lying_contact_cap():
    # lying briefcase pushed along its 0.27 depth: the back face (0.12 - 0.02) binds, far below the no-tip bound
    assert IL.lying_contact_cap(0.12, 0.135) == pytest.approx(0.10)
    # a tall narrow box: the no-tip bound binds
    assert IL.lying_contact_cap(0.30, 0.06) == pytest.approx(0.75 * 0.06 / 0.8)


class _Gp:
    def __init__(self, gen):
        self._gen, self.variant = dict(gen), "v"

    def gen(self, k, d=None):
        return self._gen.get(k, d)


def _stub(gen, monkeypatch, on=True):
    if on:
        monkeypatch.setenv(IL.BRIEFCASE_LYING_ENV, "1")
    else:
        monkeypatch.delenv(IL.BRIEFCASE_LYING_ENV, raising=False)
    s = _Stub.__new__(_Stub)
    s.gp, s.relation, s.hull, s.template_facts = _Gp(gen), "under", _briefcase_hull(), {}
    s.rest = G.quat_about_axis(th.tensor([0.0, 1.0, 0.0]), -math.pi / 2)
    return s


class _Stub(IL.IORSLarge):
    target = SimpleNamespace(name="briefcase_lwbctq_0")


def test_scope_and_release_quat(monkeypatch):
    monkeypatch.setattr(IL.TT, "_count", lambda s, k: None)
    pitched = {"push_under_hand": "pitched_far", IL.PUSH_UNDER_GOAL_REST_KEY: IL.LYING}
    s = _stub(pitched, monkeypatch, on=False)
    assert not s.lying and th.allclose(s.c_rest(), s.rest)  # switch off: today's standing drop
    level = G.qmul(G.quat_from_yaw(0.7), s.rest)
    assert th.allclose(s.release_quat(level, None, 0.0), level)
    s = _stub(pitched, monkeypatch)
    assert s.lying
    lying = s.release_quat(level, None, 0.0)
    assert th.allclose(lying, s.c_quat(0.7), atol=1e-5)  # the carried yaw kept, lying flat
    assert float(G.rotate(lying, th.tensor([0.0, 0.0, 1.0]))[2]) == pytest.approx(1.0, abs=1e-5)
    assert s.template_facts["briefcase_lying"]["tilt_deg"] == pytest.approx(90.0, abs=0.01)
    assert not _stub({"push_under_hand": "pitched_far"}, monkeypatch).lying  # no customs key: the switch does not act
    assert not _stub({IL.PUSH_UNDER_GOAL_REST_KEY: IL.LYING}, monkeypatch).lying  # the blade hand: not scoped
    with pytest.raises(IL.Reject):
        _stub({"push_under_hand": "pitched_far", IL.PUSH_UNDER_GOAL_REST_KEY: "sideways"}, monkeypatch).lying


def test_contact_cap_lying_vs_com(monkeypatch):
    monkeypatch.setattr(IL.TT, "_count", lambda s, k: None)
    s = _stub({"push_under_hand": "pitched_far", IL.PUSH_UNDER_GOAL_REST_KEY: IL.LYING}, monkeypatch)
    s._com_local = (s.target.name, th.tensor([0.016, 0.001, 0.0]))
    q = s.c_quat(0.0)
    com_h = s.com_height(q)
    assert com_h == pytest.approx(0.059, abs=2e-3)  # lying: the centre of mass 6 cm up - today's rule would cap the tips there
    assert s.push_contact_cap(q, (1.0, 0.0)) == pytest.approx(0.0999, abs=2e-3)  # the back face's 10 cm instead
    s_off = _stub({"push_under_hand": "pitched_far", IL.PUSH_UNDER_GOAL_REST_KEY: IL.LYING}, monkeypatch, on=False)
    s_off._com_local = s._com_local
    assert s_off.push_contact_cap(G.qmul(G.quat_from_yaw(0.0), s_off.rest), (1.0, 0.0)) == pytest.approx(s_off.com_height(s_off.rest))


def test_ul02_customs_scope():
    import json
    import os

    from ispatialgym import REPO_ROOT

    with open(os.path.join(REPO_ROOT, "ispatialgym", "assets", "task_specs", "isg_customs.json")) as f:
        data = json.load(f)
    lying = [k for k, v in data.items() if isinstance(v, dict) and (v.get("generation") or {}).get(IL.PUSH_UNDER_GOAL_REST_KEY) == IL.LYING]
    assert lying == ["inter_object_relation_satisfaction-under_large_scale-briefcase-desk_mdhelw_0-floor_pitched"]


def test_lying_drop_height_ladder(monkeypatch):
    monkeypatch.setattr(IL.TT, "_count", lambda s, k: None)
    s = _stub({"push_under_hand": "pitched_far", IL.PUSH_UNDER_GOAL_REST_KEY: IL.LYING}, monkeypatch)
    s.num = lambda k, d: d
    assert s.lying_drop_heights() == list(IL.LYING_DROP_HEIGHTS_M)
    assert s.drop_height() == IL.PUSH_UNDER_DROP_HEIGHT_M  # before the C loop sets a rung
    s._lying_drop_h = 0.35
    assert s.drop_height() == 0.35
    off = _stub({"push_under_hand": "pitched_far", IL.PUSH_UNDER_GOAL_REST_KEY: IL.LYING}, monkeypatch, on=False)
    off.num, off._lying_drop_h = (lambda k, d: d), 0.35
    assert off.drop_height() == IL.PUSH_UNDER_DROP_HEIGHT_M  # switch off: the rung is never read


def test_place_retreat_up_for_the_lying_drop():
    from ispatialgym.oracle.actions import RETREAT_DIST, Place, Pose

    pl = Pose([1.0, 2.0, 0.3], [0.0, 0.70710678, 0.0, 0.70710678])  # a horizontal hand
    today = Place(kind="drop", arm="right", object="b", place=pl)
    up = Place(kind="drop", arm="right", object="b", place=pl, retreat_up_m=0.10)
    e_today, e_up = today.end_pose(None, None), up.end_pose(None, None)
    assert abs(float(e_up.position[2]) - 0.40) < 1e-6 and abs(float(e_up.position[0]) - 1.0) < 1e-6
    assert abs(((float(e_today.position[0]) - 1.0) ** 2 + (float(e_today.position[2]) - 0.3) ** 2) ** 0.5 - RETREAT_DIST) < 1e-5


def test_lying_push_strokes(monkeypatch):
    monkeypatch.setattr(IL.TT, "_count", lambda s, k: None)
    monkeypatch.delenv(IL.FLOORPUSH_STROKE_ENV, raising=False)
    gen = {"push_under_hand": "pitched_far", IL.PUSH_UNDER_GOAL_REST_KEY: IL.LYING}
    s = _stub(gen, monkeypatch)
    s.num = lambda k, d: s.gp.gen(k, d)
    st = s.push_strokes_record(0.31)["strokes"]
    assert st["n"] == 7 and st["reaim_max_m"] == IL.LYING_REAIM_MAX_M and st["yaw_gain_m_per_rad"] == IL.LYING_YAW_GAIN_M_PER_RAD
    off = _stub(gen, monkeypatch, on=False)
    off.num = lambda k, d: off.gp.gen(k, d)
    st0 = off.push_strokes_record(0.31)["strokes"]  # today's default strokes: 0.10 m, 3 cm re-aim
    assert st0["n"] == 4 and st0["reaim_max_m"] == 0.03


def test_partial_tilt_release_lands_flat_where_planned(monkeypatch):
    # lead #92d (a): released at 70 deg with its lowest edge above the floor, the briefcase falls the last 20 deg about that edge onto the lying pose
    monkeypatch.setenv(IL.BRIEFCASE_LYING_TILT_ENV, "70, 60,x,95")
    assert IL.briefcase_lying_tilts() == [70.0, 60.0]
    monkeypatch.delenv(IL.BRIEFCASE_LYING_TILT_ENV)
    assert IL.briefcase_lying_tilts() == []
    hull = _briefcase_hull()
    rest = G.quat_about_axis(th.tensor([0.0, 1.0, 0.0]), -math.pi / 2)
    q_ly, facts = IL.lying_rest(hull, rest)
    a = th.tensor(facts["tilt_axis_world"])
    for t in (60.0, 70.0, 75.0):
        q_r, off, shift = IL.partial_landing(hull, q_ly, a, t)
        up_r = float(G.rotate(q_r, th.tensor([0.0, 0.0, 1.0]))[2])  # the thin axis: lying = 1, standing = 0
        assert up_r == pytest.approx(math.sin(math.radians(t + facts["tilt_deg"] - 90.0)), abs=0.02)
        # the root released at -off from a pivot at the origin; the rigid fall about the pivot ends at the lying orientation, the lowest point on the pivot's height
        root_r = off
        rot = G.quat_about_axis(a / th.linalg.norm(a), math.radians(90.0 - t))
        root_l = G.rotate(rot, root_r)
        pts_l = G.local_points_world(hull, (root_l, q_ly))
        assert float(pts_l[:, 2].min()) == pytest.approx(0.0, abs=0.006)
        assert th.allclose(root_l[:2] - root_r[:2], shift, atol=1e-5)
