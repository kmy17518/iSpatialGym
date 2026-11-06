"""oracle-fix 2026-11-06 (owner feedback on the inside_large#02 demo, the jar of sauce carried into the fridge: "does this use the same
navigation module as camera relocalization? the path doesn't seem smooth ... make iors large tasks use the smooth navigation module"):
the CARRIED legs of a family that opts in (``generation.carried_base_path_style``; the IORS large-scale defaults) are checked and recorded
as the ``robot.base_path`` face_travel schedule camera relocalization drives - with the object attached, the carry arms and the carry
margin - and fall back to the polyline when the schedule collides (``generation.face_travel_fallback_cfgs`` tried in between, recorded as
``paths[].face_travel_cfg`` and built alike by the oracle) or turns the body more than the polyline does
(``generation.face_travel_max_extra_rotation_deg``); every re-sweep of a face_travel record sweeps the schedule. No simulator."""

from __future__ import annotations

import math
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.robot import base_path as BP
from ispatialgym.sampling.v2 import navigation as NAV
from ispatialgym.sampling.v2.stances import PathRecord, SidecarDraft, Stance

TIGHT = {"simplify_tol_m": 0.03, "fillet_deviation_m": 0.03}
#: the jar6_s070_s3 0000 carried leg manipulation -> placement (global, 3.60 m) as the sampler recorded it
JAR6_CARRY = [(3.868, 2.733, -1.571), (3.85, 2.65, -0.077), (5.15, 2.55, -1.476), (5.35, 0.45, 0.539), (5.439, 0.503, -1.571)]


def _l_path(monkeypatch):
    """An L-shaped grid path (0,0) -> (0,20) -> (20,20) on a 5 cm grid (as test_v2_core's face_travel test)."""
    monkeypatch.setenv("ISG_ATTACHED_CHECKS", "off")
    monkeypatch.setattr(NAV, "local_map", lambda kit, rooms=None, blocked=(): th.ones(30, 30, dtype=th.bool))
    monkeypatch.setattr(NAV, "_to_map", lambda kit, xy: (int(round(float(xy[1]) / 0.05)), int(round(float(xy[0]) / 0.05))))
    monkeypatch.setattr(NAV, "_to_world", lambda kit, ij: th.tensor([ij[1] * 0.05, ij[0] * 0.05]))
    monkeypatch.setattr(NAV, "nearest_free_cell", lambda free, ij, r=15: ij)
    monkeypatch.setattr(NAV, "same_region", lambda free, a, b: True)
    monkeypatch.setattr(NAV, "grid_shortest_path", lambda free, a, b: [(0, 0), (0, 20), (20, 20)])
    monkeypatch.setattr(NAV, "simplify_path", lambda cells, free: list(cells))
    st = lambda name, x, y, yaw: SimpleNamespace(name=name, base_xy=th.tensor([x, y]), yaw=yaw, base_pose2d=(x, y, yaw), translate=0.0, tilt=0.0)  # noqa: E731
    return st("manipulation", 0.0, 0.0, 0.0), st("placement", 1.0, 1.0, 1.5708)


def test_carried_leg_is_recorded_as_the_face_travel_schedule_when_it_clears(monkeypatch):
    a, b = _l_path(monkeypatch)
    calls, attached = [], []
    verdicts = []  # per check, in order; missing = clear

    def fake_free(kit, poses, trunk, posture="untucked", margin=0.01, q_arms=None, fingers=None):
        calls.append((list(poses), margin, q_arms))
        ok = verdicts[len(calls) - 1] if len(calls) <= len(verdicts) else True
        return ok, (-1 if ok else 2)

    monkeypatch.setattr(NAV, "path_collision_free", fake_free)
    monkeypatch.setattr(NAV, "attached", lambda kit, arm, obj, eef_in_object=None: (attached.append((arm, obj, eef_in_object)), nullcontext())[1])
    obj, eio, q_carry = object(), ("T_obj_eef",), {"right": th.full((7,), 3.0)}
    kw = dict(arm_q=q_carry, carried="jar.n.01_1", carried_obj=obj, carried_arm="right", eef_in_object=eio, margin=0.03, style="face_travel")
    before = dict(NAV.FACE_TRAVEL_CARRIED_PROBE)
    # the schedule clears: recorded face_travel after ONE check - the schedule with its yaw-lag envelope, the jar attached to the carrying
    # arm per its grasp record, the carry arms, the carry margin; no probe
    rec = NAV.plan_path(None, a, b, **kw)
    assert rec.style == "face_travel" and rec.face_travel_cfg is None and len(calls) == 1
    assert calls[0][0] == NAV.yaw_lag_envelope(NAV.face_travel_poses(rec.waypoints)) and calls[0][1] == 0.03 and calls[0][2] is q_carry
    assert attached[0] == ("right", obj, eio)
    assert rec.to_sidecar()["style"] == "face_travel" and "face_travel_cfg" not in rec.to_sidecar() and rec.carried == "jar.n.01_1"
    assert NAV.FACE_TRAVEL_CARRIED_PROBE == before
    # the schedule collides: the polyline decides (strides + envelope at the carry margin), the record says recorded; still no probe
    calls.clear()
    verdicts[:] = [False]
    rec = NAV.plan_path(None, a, b, **kw)
    assert rec.style == "recorded" and len(calls) == 2 and calls[1][0] == NAV.stride_poses(rec.waypoints, 0.1, yaw_envelope=True) and calls[1][1] == 0.03
    assert NAV.FACE_TRAVEL_CARRIED_PROBE == before
    # both collide: the same path_collision rejection as a recorded leg
    calls.clear()
    verdicts[:] = [False, False]
    with pytest.raises(NAV.NavigationError) as e:
        NAV.plan_path(None, a, b, **kw)
    assert e.value.reason == "path_collision"


def test_fallback_schedule_configuration_is_tried_before_the_polyline_and_recorded(monkeypatch):
    a, b = _l_path(monkeypatch)
    calls, verdicts = [], []

    def fake_free(kit, poses, trunk, posture="untucked", margin=0.01, q_arms=None, fingers=None):
        calls.append(list(poses))
        ok = verdicts[len(calls) - 1] if len(calls) <= len(verdicts) else True
        return ok, (-1 if ok else 0)

    monkeypatch.setattr(NAV, "path_collision_free", fake_free)
    kw = dict(arm_q={}, carried="jar.n.01_1", carried_obj=object(), style="face_travel", face_travel_cfgs=[TIGHT])
    # the default schedule collides, the tight one clears: recorded face_travel WITH its configuration, swept as that schedule
    verdicts[:] = [False]
    rec = NAV.plan_path(None, a, b, **kw)
    assert rec.style == "face_travel" and rec.face_travel_cfg == TIGHT and len(calls) == 2
    assert calls[0] == NAV.yaw_lag_envelope(NAV.face_travel_poses(rec.waypoints))
    assert calls[1] == NAV.yaw_lag_envelope(NAV.face_travel_poses(rec.waypoints, cfg=TIGHT)) and calls[1] != calls[0]
    assert rec.to_sidecar()["face_travel_cfg"] == TIGHT
    # every schedule collides: the polyline (third check), recorded, no configuration
    calls.clear()
    verdicts[:] = [False, False]
    rec = NAV.plan_path(None, a, b, **kw)
    assert rec.style == "recorded" and rec.face_travel_cfg is None and len(calls) == 3 and "face_travel_cfg" not in rec.to_sidecar()
    # the default schedule clears: the configurations are never tried
    calls.clear()
    verdicts[:] = []
    rec = NAV.plan_path(None, a, b, **kw)
    assert rec.style == "face_travel" and rec.face_travel_cfg is None and len(calls) == 1
    # a recorded leg ignores them
    calls.clear()
    rec = NAV.plan_path(None, a, b, **{**kw, "style": "recorded", "carried_obj": None, "carried": None})
    assert rec.style is None and len(calls) == 1


def test_face_travel_is_not_used_where_it_turns_the_body_more_than_the_polyline(monkeypatch):
    """``face_travel_max_extra_rotation_deg``: a straight 10 cm back-up as face_travel spins 180 deg at the start and 180 deg at the end
    (11.5 s against 0.8 s as recorded) - not checked, the polyline decides; the jar6 carry (402 deg as recorded, ~100 as the schedule) and
    the L-shaped corner (90 / 90) keep the schedule."""
    assert math.degrees(NAV.total_rotation(JAR6_CARRY)) == pytest.approx(402, abs=2)
    assert math.degrees(NAV.total_rotation(NAV.face_travel_poses(JAR6_CARRY, max_yaw_step=None))) == pytest.approx(101, abs=3)
    a, b = _l_path(monkeypatch)
    calls = []
    monkeypatch.setattr(NAV, "path_collision_free", lambda kit, poses, trunk, posture="untucked", margin=0.01, q_arms=None, fingers=None:
                        (calls.append(list(poses)), (True, -1))[1])
    kw = dict(arm_q={}, carried="jar.n.01_1", carried_obj=object(), style="face_travel", face_travel_max_extra_rotation_deg=10.0)
    rec = NAV.plan_path(None, a, b, **kw)
    assert rec.style == "face_travel" and len(calls) == 1
    # the back-up: 0.10 m along +y with the body facing -y
    monkeypatch.setattr(NAV, "grid_shortest_path", lambda free, a, b: [(0, 0), (2, 0)])
    st = lambda name, x, y, yaw: SimpleNamespace(name=name, base_xy=th.tensor([x, y]), yaw=yaw, base_pose2d=(x, y, yaw), translate=0.0, tilt=0.0)  # noqa: E731
    a, b = st("init", 0.0, 0.0, -math.pi / 2), st("manipulation", 0.0, 0.1, -math.pi / 2)
    calls.clear()
    rec = NAV.plan_path(None, a, b, **kw)
    assert rec.style == "recorded" and len(calls) == 1 and calls[0] == NAV.stride_poses(rec.waypoints, 0.1, yaw_envelope=True)
    assert math.degrees(NAV.total_rotation(NAV.face_travel_poses(rec.waypoints, max_yaw_step=None))) == pytest.approx(360, abs=1)
    # without the rule (the 2026-09-26 families): the schedule is checked and kept
    calls.clear()
    rec = NAV.plan_path(None, a, b, **{**kw, "face_travel_max_extra_rotation_deg": None})
    assert rec.style == "face_travel" and len(calls) == 1


def test_base_path_config_with():
    assert BP.config_with(None) == BP.BasePathConfig() and BP.config_with({}) == BP.BasePathConfig()
    c = BP.config_with(TIGHT)
    assert c.simplify_tol_m == 0.03 and c.fillet_deviation_m == 0.03 and c.blend_m == BP.BasePathConfig().blend_m
    base = BP.BasePathConfig(blend_m=1.0)
    assert BP.config_with({"simplify_tol_m": 0.05}, base).blend_m == 1.0
    with pytest.raises(ValueError):
        BP.config_with({"simplify_tol": 0.03})


# ----------------------------------------------------------------------------------------------------------------
# record_path: the knobs, the generation_defaults fallback
# ----------------------------------------------------------------------------------------------------------------


def _record_path_sampler(monkeypatch, generation=None, subtask=None):
    from ispatialgym.sampling.v2 import core as CORE

    seen = {}
    monkeypatch.setattr(NAV, "plan_path", lambda kit, start, to, **kw: (seen.clear(), seen.update(kw), PathRecord(start.name, to.name, "local", [(0, 0, 0), (1, 0, 0)], 1.0))[2])
    s = CORE.SamplerV2.__new__(CORE.SamplerV2)
    gen = dict(generation or {})
    s.gp = SimpleNamespace(gen=lambda k, d=None: gen.get(k, d), rooms=lambda scene: ["kitchen_0"], subtask=subtask, variant="v")
    s.scene, s.world, s.kit = "scene", SimpleNamespace(rooms=["kitchen_0"]), None  # carry_arm_q: the property's empty default
    s.sidecar = SimpleNamespace(add_path=lambda p: p)
    s.path_blocked = lambda: []
    s.path_keep_off_footprints = lambda: []
    obj = object()
    s.carried_attachment = lambda carried, arm=None: ((obj, "right", None) if carried else (None, arm or "right", None))
    st = lambda name, carried=None: SimpleNamespace(name=name, posture="untucked", carried=carried)  # noqa: E731
    return s, seen, st


def test_record_path_carried_leg_style_knob(monkeypatch):
    # default (no knob): carried legs stay recorded with the log-only probe; empty-handed legs read base_path_style
    s, seen, st = _record_path_sampler(monkeypatch)
    s.record_path(st("manipulation"), st("placement", "jar"))
    assert seen["style"] == "recorded" and seen["probe_face_travel"] is True and "face_travel_cfgs" not in seen
    assert "face_travel_max_extra_rotation_deg" not in seen
    s.record_path(st("init"), st("manipulation"))
    assert seen["style"] == "recorded" and seen["probe_face_travel"] is False
    # carried_base_path_style face_travel: the carried leg is checked as the schedule, no probe; the empty-handed leg unchanged
    s, seen, st = _record_path_sampler(monkeypatch, {"carried_base_path_style": "face_travel", "face_travel_fallback_cfgs": [TIGHT]})
    s.record_path(st("manipulation"), st("placement", "jar"))
    assert seen["style"] == "face_travel" and seen["probe_face_travel"] is False and seen["face_travel_cfgs"] == [TIGHT]
    s.record_path(st("init"), st("manipulation"))
    assert seen["style"] == "recorded" and "face_travel_cfgs" not in seen
    # base_path_style alone (the 4 aliases of 2026-09-26): carried legs stay recorded
    s, seen, st = _record_path_sampler(monkeypatch, {"base_path_style": "face_travel"})
    s.record_path(st("manipulation"), st("placement", "jar"))
    assert seen["style"] == "recorded" and seen["probe_face_travel"] is True
    s.record_path(st("init"), st("manipulation"))
    assert seen["style"] == "face_travel"


def test_iors_large_defaults_reach_entries_without_the_keys(monkeypatch):
    """The counted templates read their own customs copy, seeded before the keys existed: ``gen_seeded`` falls back to the subtask's
    generation_defaults block; an entry's own value wins; other subtasks keep "recorded"."""
    from ispatialgym.sampling import generation_defaults as GD

    for sub in ("ontop", "under", "inside", "touching", "nextto"):
        d = GD.defaults_for(f"inter_object_relation_satisfaction-{sub}_large_scale")
        assert d["base_path_style"] == d["carried_base_path_style"] == "face_travel" and "face_travel_fallback_cfgs" not in d
        assert d["face_travel_max_extra_rotation_deg"] == 10.0
    s, seen, st = _record_path_sampler(monkeypatch, subtask="inter_object_relation_satisfaction-inside_large_scale")
    s.record_path(st("manipulation"), st("placement", "jar"))
    assert seen["style"] == "face_travel" and seen["probe_face_travel"] is False and "face_travel_cfgs" not in seen
    assert seen["face_travel_max_extra_rotation_deg"] == 10.0
    s.record_path(st("init"), st("manipulation"))
    assert seen["style"] == "face_travel" and seen["face_travel_max_extra_rotation_deg"] == 10.0
    # the entry's fallback configurations (opt-in) reach plan_path with the subtask's style default
    s, seen, st = _record_path_sampler(monkeypatch, {"face_travel_fallback_cfgs": [TIGHT]}, subtask="inter_object_relation_satisfaction-inside_large_scale")
    s.record_path(st("manipulation"), st("placement", "jar"))
    assert seen["style"] == "face_travel" and seen["face_travel_cfgs"] == [TIGHT]
    s, seen, st = _record_path_sampler(monkeypatch, {"carried_base_path_style": "recorded"}, subtask="inter_object_relation_satisfaction-under_large_scale")
    s.record_path(st("manipulation"), st("placement", "jar"))
    assert seen["style"] == "recorded" and seen["probe_face_travel"] is True
    for sub in ("inter_object_relation_satisfaction-ontop_small_scale", "camera_relocalization", None):
        s, seen, st = _record_path_sampler(monkeypatch, subtask=sub)
        s.record_path(st("manipulation"), st("placement", "jar"))
        assert seen["style"] == "recorded", sub


# ----------------------------------------------------------------------------------------------------------------
# the re-sweeps of a record
# ----------------------------------------------------------------------------------------------------------------


def test_record_sweep_poses_follow_the_record_style():
    rec = PathRecord("manipulation", "placement", "global", list(JAR6_CARRY), 3.6, carried="jar", stride_m=0.1)
    poly = NAV.stride_poses(rec.waypoints, 0.1, yaw_envelope=True)
    assert NAV.record_sweep_poses(rec) == poly and NAV.record_sweep_poses(rec, yaw_envelope=False) == NAV.stride_poses(rec.waypoints, 0.1)
    rec.style = "recorded"
    assert NAV.record_sweep_poses(rec) == poly
    rec.style = "face_travel"
    ft = NAV.record_sweep_poses(rec)
    assert ft == NAV.yaw_lag_envelope(NAV.face_travel_poses(rec.waypoints, 0.1)) and ft != poly
    # the schedule leaves the polyline (simplified within 10 cm, filleted): poses the polyline sweep never checks
    far = max(min(math.hypot(p[0] - q[0], p[1] - q[1]) for q in poly) for p in ft)
    assert far > 0.03
    rec.face_travel_cfg = dict(TIGHT)
    assert NAV.record_sweep_poses(rec) == NAV.yaw_lag_envelope(NAV.face_travel_poses(rec.waypoints, 0.1, cfg=TIGHT)) != ft


def _face_travel_sidecar():
    sc = SidecarDraft("a", "f")
    sc.add_stance(Stance("init", JAR6_CARRY[0][:2] + (0.0,), 0.5, 0.0))
    sc.add_stance(Stance("manipulation", JAR6_CARRY[0], 0.5, 0.0))
    sc.add_stance(Stance("placement", JAR6_CARRY[-1], 0.5, 0.0))
    p = sc.add_path(PathRecord("manipulation", "placement", "global", list(JAR6_CARRY), 3.6, carried="t", trunk=(0.5, 0.0), stride_m=0.1, style="face_travel"))
    return sc, p


def test_recheck_carried_travel_sweeps_the_face_travel_schedule(monkeypatch):
    from ispatialgym.sampling.v2 import ik as IKmod
    from ispatialgym.sampling.v2.core import SamplerV2

    seen = []
    monkeypatch.setattr(NAV, "attached", lambda kit, arm, obj, eef_in_object=None: nullcontext())
    monkeypatch.setattr(NAV, "path_collision_free", lambda kit, poses, trunk, posture="untucked", margin=0.01, q_arms=None, fingers=None:
                        (seen.append(list(poses)), (True, -1))[1])
    monkeypatch.setattr(IKmod, "stance_arm_vectors", lambda kit, st, arms=None, **kw: (th.zeros(7), th.zeros(7), 0.0))
    sc, p = _face_travel_sidecar()
    s = SimpleNamespace(sidecar=sc, kit=SimpleNamespace(), carry_arm_q={}, gp=SimpleNamespace(gen=lambda k, d=None: d),
                        carried_attachment=lambda carried, arm=None: ("OBJ", "right", None), object_joints_of=lambda st: nullcontext(),
                        ikc=SimpleNamespace(ignoring=lambda objs: nullcontext(), stance_collision_free=lambda st, **kw: True))
    assert SamplerV2.recheck_carried_travel(s, "t") is None
    assert seen == [NAV.record_sweep_poses(p)] and seen[0] != NAV.stride_poses(p.waypoints, 0.1, yaw_envelope=True)


def test_leg_ab_clear_of_target_sweeps_the_face_travel_schedule(monkeypatch):
    from ispatialgym.sampling.v2 import ik as IKmod
    from ispatialgym.sampling.v2.tasks.iors_large import IORSLarge

    seen = []
    monkeypatch.setattr(NAV, "path_collision_free", lambda kit, poses, trunk, posture="untucked", margin=0.01, q_arms=None, fingers=None:
                        (seen.append(list(poses)), (False, 1))[1])
    monkeypatch.setattr(IKmod, "stance_arm_vectors", lambda kit, st, arms=None, **kw: (th.zeros(7), th.zeros(7), 0.0))
    sc = SidecarDraft("a", "f")
    init = sc.add_stance(Stance("init", (0.0, 0.0, 0.0), 0.5, 0.0))
    manip = sc.add_stance(Stance("manipulation", (1.0, 1.0, 1.5708), 0.5, 0.0))
    p = sc.add_path(PathRecord("init", "manipulation", "local", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 1.5708)], 2.0, trunk=(0.5, 0.0), style="face_travel"))
    tgt = SimpleNamespace(get_position_orientation=lambda: (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])), set_position_orientation=lambda *a: None,
                          keep_still=lambda: None)
    s = SimpleNamespace(sidecar=sc, kit=SimpleNamespace(sync_world=lambda: None), target=tgt, _floor_relaxed=lambda: nullcontext())
    why = IORSLarge._leg_ab_clear_of_target(s, init, manip, (th.tensor([0.5, 0.5, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0])))
    assert seen == [NAV.record_sweep_poses(p)] and seen[0] != NAV.stride_poses(p.waypoints, 0.1, yaw_envelope=True)
    assert why.startswith("B: the A -> B drive passes the target")


def test_program_check_departure_arms_sweep_the_face_travel_schedule(monkeypatch):
    from ispatialgym.sampling.v2 import ik as IKmod
    from ispatialgym.sampling.v2 import program_check as PC

    seen = []
    monkeypatch.setattr(NAV, "path_collision_free", lambda kit, poses, trunk, posture="untucked", margin=0.01, q_arms=None, fingers=None:
                        (seen.append((list(poses), q_arms)), (False, 4))[1])
    monkeypatch.setattr(IKmod, "stance_arm_vectors", lambda kit, st, arms=None, **kw: (th.zeros(7), th.zeros(7), 0.0))
    sc, p = _face_travel_sidecar()
    p.carried = None  # an empty-handed face_travel leg re-swept with the departure arms (the 2026-09-26 families)
    action = SimpleNamespace(name="navigate", path=SimpleNamespace(frm="manipulation", to="placement"))
    override = {"right": th.ones(7)}
    ok, why, q_arms = PC._sweep_travel_record(sc, SimpleNamespace(kit=SimpleNamespace()), action, override)
    assert not ok and seen[0][0] == NAV.record_sweep_poses(p) and th.equal(seen[0][1]["right"], th.ones(7))
    assert "face_travel schedule + yaw-lag envelope" in why
    p.style = None
    seen.clear()
    PC._sweep_travel_record(sc, SimpleNamespace(kit=SimpleNamespace()), action, override)
    assert seen[0][0] == NAV.stride_poses(p.waypoints, 0.1, yaw_envelope=True)


# ----------------------------------------------------------------------------------------------------------------
# the oracle: the record's style and configuration reach track_path
# ----------------------------------------------------------------------------------------------------------------


def _oracle_path(extra):
    from ispatialgym.oracle.sidecar import Path

    return Path("manipulation", "placement", "global", [list(w) for w in JAR6_CARRY], 3.6, "untucked", carried="jar", extra=dict(extra))


def test_navigate_targets_carry_the_record_style_and_configuration():
    from ispatialgym.oracle.actions import Navigate

    t = Navigate(path=_oracle_path({"style": "face_travel"}), objects=("jar",)).targets(None, None)
    assert t["style"] == "face_travel" and t["face_travel_cfg"] is None and t["carried"] == "jar"
    t = Navigate(path=_oracle_path({"style": "face_travel", "face_travel_cfg": TIGHT}), objects=("jar",)).targets(None, None)
    assert t["style"] == "face_travel" and t["face_travel_cfg"] == TIGHT
    # a recorded leg drives the polyline: no configuration; the program's style wins
    assert Navigate(path=_oracle_path({"face_travel_cfg": TIGHT})).targets(None, None)["face_travel_cfg"] is None
    assert Navigate(path=_oracle_path({}), style=BP.FACE_TRAVEL).targets(None, None)["face_travel_cfg"] is None
    assert Navigate(path=_oracle_path({})).targets(None, None)["style"] == "recorded"
    with pytest.raises(ValueError):
        Navigate(path=_oracle_path({"style": "face_travel", "face_travel_cfg": {"simplify": 0.03}}))


def test_execute_navigate_passes_the_configuration_to_track_path():
    from ispatialgym.oracle import execute as EX
    from ispatialgym.oracle.actions import Navigate

    seen = []

    def track_path(waypoints, posture=None, style=None, **kw):
        seen.append((style, kw))
        yield th.zeros(1)

    motions = SimpleNamespace(track_path=track_path)
    for extra, want in (({"style": "face_travel", "face_travel_cfg": TIGHT}, {"face_travel_cfg": TIGHT}), ({"style": "face_travel"}, {}), ({}, {})):
        nav = Navigate(path=_oracle_path(extra), objects=("jar",))
        targets = nav.targets(None, None)
        list(EX.execute_navigate(nav, motions, None, targets, {}))
        assert seen[-1] == (extra.get("style", "recorded"), want)


def _reference(style, cfg=None, wps=JAR6_CARRY):
    from ispatialgym.oracle.motions import Motions, MotionsConfig

    mc = MotionsConfig()
    tc = mc.base_tracker
    fake = SimpleNamespace(cfg=mc, dt=1.0 / 30.0, base_pose2d=lambda: tuple(wps[0]))
    v_lin, v_ang = mc.path_speed_fraction * min(tc.v_max[0], tc.v_max[1]), mc.path_speed_fraction * tc.v_max[2]
    kw = {"face_travel_cfg": cfg} if cfg else {}
    return Motions._base_reference(fake, list(wps[1:]), v_lin, v_ang, a_max=min(tc.a_max[0], tc.a_max[1]), style=style, **kw), fake


def test_oracle_reference_of_the_jar6_carry_face_travel_vs_recorded():
    """The owner's jar6 demo: the carried leg as recorded turns the body ~400 deg for a net -1 deg in ~17.6 s; the face_travel
    schedule of the same polyline turns it < 150 deg and takes ~9 s."""
    def stats(ref):
        r = ref.double()
        dyaw = (r[1:, 2] - r[:-1, 2] + math.pi) % (2 * math.pi) - math.pi
        return len(r) / 30.0, math.degrees(float(dyaw.abs().sum()))

    t_rec, rot_rec = stats(_reference(BP.RECORDED)[0])
    t_ft, rot_ft = stats(_reference(BP.FACE_TRAVEL)[0])
    assert rot_rec > 350 and t_rec > 15
    assert rot_ft < 150 and 8.0 < t_ft < 10.5
    # why the IORS large-scale defaults set no face_travel_fallback_cfgs: simplify / fillet 3 cm keeps the global path's 10 cm end hook as
    # a corner + an end spin - more rotation than the polyline as recorded
    assert stats(_reference(BP.FACE_TRAVEL, TIGHT)[0])[1] > rot_rec
    # the record's own configuration builds the oracle's schedule exactly as the sampler's check built it
    from ispatialgym.oracle.motions import as_pose2d

    ref_tight, fake = _reference(BP.FACE_TRAVEL, TIGHT)
    path = BP.BasePath.build([fake.base_pose2d()] + [as_pose2d(w) for w in JAR6_CARRY[1:]], BP.config_with(TIGHT, fake.cfg.face_travel))
    tc = fake.cfg.base_tracker
    want = BP.time_parameterise(path, fake.dt, fake.cfg.path_speed_fraction * min(tc.v_max[0], tc.v_max[1]), fake.cfg.path_speed_fraction * tc.v_max[2],
                                a_lin=min(tc.a_max[0], tc.a_max[1]), a_yaw=float(tc.a_max[2]), ds=float(fake.cfg.path_profile_ds))
    assert th.allclose(ref_tight, want) and not (ref_tight.shape == _reference(BP.FACE_TRAVEL)[0].shape and th.allclose(ref_tight, _reference(BP.FACE_TRAVEL)[0]))


def test_oracle_path_sweep_follows_the_face_travel_schedule(monkeypatch):
    """``keep_arm_branch``'s sweep of an empty-handed leg (``execute.path_sweep``): the schedule the base drives for a face_travel record."""
    from ispatialgym.oracle import execute as EX
    from ispatialgym.oracle.motions import MotionsConfig

    monkeypatch.setattr(EX, "_q_at_base", lambda motions, q, x, y, yaw: th.tensor([x, y, yaw], dtype=th.float64))
    motions = SimpleNamespace(backend=SimpleNamespace(q_full=lambda: th.zeros(3)), cfg=MotionsConfig())
    state = SimpleNamespace(base_pose2d=lambda: JAR6_CARRY[0])
    wps = [list(w) for w in JAR6_CARRY]
    poly = EX.path_sweep(motions, state, wps)
    for cfg in (None, TIGHT):
        ft = EX.path_sweep(motions, state, wps, style=BP.FACE_TRAVEL, face_travel_cfg=cfg)
        sched = BP.BasePath.build([JAR6_CARRY[0]] + list(JAR6_CARRY), BP.config_with(cfg))
        want = th.tensor(sched.stride_poses(0.05, dwell_step_rad=0.1, max_yaw_step=0.1)[1:], dtype=th.float64)
        assert len(ft) - 1 == len(want) and th.allclose(th.stack(ft[1:]), want, atol=1e-4)  # the oracle's waypoints pass through float32
    far = max(min(math.hypot(float(p[0] - q[0]), float(p[1] - q[1])) for q in poly[1:]) for p in EX.path_sweep(motions, state, wps, style=BP.FACE_TRAVEL)[1:])
    assert far > 0.03, "the polyline sweep missed where the schedule drives"
