"""i7-ul03 2026-11-09: the carry's floor clearance (ISPATIALGYM_CARRY_FLOOR_CLEARANCE) and the early frame rules (ISPATIALGYM_LARGE_EARLY_FRAME)."""
import pytest
import torch as th

from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2 import program_check as PCK
from ispatialgym.sampling.v2.tasks import iors_large as IL


class _GP:
    def __init__(self, gen=None):
        self._g = dict(gen or {})
        self.variant = "test"

    def gen(self, key, default=None):
        return self._g.get(key, default)


class _St:
    def __init__(self, translate, tilt, base=(0.0, 0.0, 0.0)):
        self.translate, self.tilt, self.base_pose2d = translate, tilt, base


def _sampler(hull):
    s = IL.IORSLarge.__new__(IL.IORSLarge)
    s.gp = _GP()
    s.kit = type("K", (), {"stats": {}})()
    s.hull = hull
    s.floor_z = 0.0
    return s


def _box(half):
    return th.tensor([[sx * half, sy * half, sz * half] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=th.float32)


def test_both_switches_are_off_by_default(monkeypatch):
    monkeypatch.delenv(IL.CARRY_FLOOR_CLEARANCE_ENV, raising=False)
    monkeypatch.delenv(IL.LARGE_EARLY_FRAME_ENV, raising=False)
    monkeypatch.delenv(IL.CARRY_FLOOR_CLEARANCE_M_ENV, raising=False)
    assert IL.carry_floor_mode() == "off" and IL.large_early_frame_mode() == "off"
    assert IL.carry_floor_clearance_m() == 0.0
    for v, m in (("shadow", "shadow"), ("1", "on"), ("on", "on"), ("0", "off")):
        monkeypatch.setenv(IL.CARRY_FLOOR_CLEARANCE_ENV, v)
        monkeypatch.setenv(IL.LARGE_EARLY_FRAME_ENV, v)
        assert IL.carry_floor_mode() == m and IL.large_early_frame_mode() == m


def _hand_at(st, height_above_floor):
    """A hand pose straight above the torso's front at B, and T_obj->eef with the object 0.1 m below the hand."""
    t4 = PCK.torso_link4_world(st.base_pose2d, st.translate, st.tilt)
    eef = (th.tensor([float(t4[0][0]) + 0.5, float(t4[0][1]), height_above_floor]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    eio = (th.tensor([0.0, 0.0, 0.1]), th.tensor([0.0, 0.0, 0.0, 1.0]))  # the eef 0.1 m above the object's root
    return eef, eio


def test_the_clearance_follows_the_torso_and_is_zero_change_without_a_trunk_move():
    s = _sampler(_box(0.05))
    st = _St(0.75, -1.2)
    eef, eio = _hand_at(st, 0.35)
    s._cf_hand = (eef, eio)
    # no trunk change: the object's lowest point = hand z - 0.1 (eef above the root) - 0.05 (half box)
    v = s.carry_floor_clearance(st, [("dep", st.base_pose2d, (0.75, -1.2), (0.75, -1.2))])
    assert abs(v["min_m"] - 0.20) < 1e-3
    # a deeper translate lowers the torso: the held object goes lower, the minimum is reported per leg
    v2 = s.carry_floor_clearance(st, [("dep", st.base_pose2d, (0.75, -1.2), (2.0, 0.0)), ("arr", st.base_pose2d, (2.0, 0.0), (0.95, -1.2))])
    assert set(v2["per_leg"]) == {"dep", "arr"} and v2["min_m"] == min(v2["per_leg"].values())
    # the hand rides with torso_link4: the same as composing the torso move on the object pose
    A = PCK.torso_link4_world(st.base_pose2d, 0.75, -1.2)
    B = PCK.torso_link4_world(st.base_pose2d, 2.0, 0.0)
    obj0 = IL.G.compose(eef, IL.G.invert(eio))
    obj1 = IL.G.compose(B, IL.G.compose(IL.G.invert(A), obj0))
    z_end = float(IL.G.local_points_world(s.hull, obj1)[:, 2].min())
    assert v2["per_leg"]["dep"] <= z_end + 1e-4


def test_the_gate_refuses_only_when_on_and_below(monkeypatch):
    s = _sampler(_box(0.05))
    st = _St(0.75, -1.2)
    s._cf_hand = _hand_at(st, 0.16)  # lowest point 1 cm above the floor at B
    legs_low = [("dep", st.base_pose2d, (0.75, -1.2), (2.0, 0.0))]
    v = s.carry_floor_clearance(st, legs_low)
    assert v["min_m"] < 0.0  # the deep travel trunk takes it into the floor
    monkeypatch.setenv(IL.CARRY_FLOOR_CLEARANCE_ENV, "0")
    s.carry_floor_gate(st, "obj", legs_low)
    assert s.kit.stats == {}
    monkeypatch.setenv(IL.CARRY_FLOOR_CLEARANCE_ENV, "shadow")
    s.carry_floor_gate(st, "obj", legs_low)
    assert s.kit.stats["carry_floor.shadow_low"] == 1 and "carry_floor.rejected" not in s.kit.stats
    monkeypatch.setenv(IL.CARRY_FLOOR_CLEARANCE_ENV, "1")
    with pytest.raises(Reject) as e:
        s.carry_floor_gate(st, "obj", legs_low)
    assert e.value.reason == "no_transition" and s.kit.stats["carry_floor.rejected"] == 1
    # no trunk change: clear -> no reject
    s.carry_floor_gate(st, "obj", [("arr", st.base_pose2d, (0.75, -1.2), (0.75, -1.2))])
    assert s.kit.stats["carry_floor.checked"] == 3


class _Ikc:
    def __init__(self, inside):
        self.inside = inside

    def eef_in_frame(self, pose, cam):
        return self.inside


def test_the_goal_check_rule_is_one_function(monkeypatch):
    s = _sampler(_box(0.05))
    monkeypatch.setattr(IL.IORSLarge, "camera_pose_at", lambda self, st: (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])))
    s.ikc = _Ikc(False)
    ok, why = s.grasp_frame_verdict(_St(0.95, -1.2), (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])))
    assert not ok and why == "grasp eef pose outside the manipulation image (5 %)"
    s.ikc = _Ikc(True)
    assert s.grasp_frame_verdict(_St(0.95, -1.2), (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0]))) == (True, None)


def test_early_grasp_and_preplace_use_the_program_check_rule(monkeypatch):
    s = _sampler(_box(0.05))
    s.ikc = _Ikc(True)
    monkeypatch.setattr(IL.IORSLarge, "camera_pose_at", lambda self, st: (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])))
    seen = []

    def fake_fif(sampler, stance, name, items, *a, **k):
        seen.append((name, [key for key, _ in items], [len(p) for _, p in items]))
        return None if name == "grasp" else {"key": "pre_place", "index": 0, "n": 1, "u": 0, "v": 0, "depth_m": 1.0, "slack_px": 0}

    monkeypatch.setattr(PCK, "fingertips_in_frame", fake_fif)
    eef = (th.tensor([1.0, 0.0, 0.3]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    v = s.early_grasp_frame(_St(0.95, -1.2), eef)
    assert v == {"box": True, "tip": True, "ok": True}
    assert not s.early_preplace_frame(_St(0.95, -1.2), eef, (th.tensor([1.0, 0.0, 0.2]), th.tensor([0.0, 0.0, 0.0, 1.0])))
    assert seen[0] == ("grasp", ["pre_grasp", "approach"], [1, 3]) and seen[1] == ("place", ["pre_place", "descent"], [1, 3])
    assert s.kit.stats["early_frame.preplace.failed"] == 1 and s.kit.stats["early_frame.grasp.checked"] == 1


def test_agreement_counters(monkeypatch):
    s = _sampler(_box(0.05))
    s._ef_agree("grasp_box", True, True)
    s._ef_agree("grasp_box", False, False)
    s._ef_agree("preplace", True, False)
    s._ef_agree("preplace", None, False)
    assert s.kit.stats == {"early_frame.agree.grasp_box.pass_pass": 1, "early_frame.agree.grasp_box.fail_fail": 1,
                           "early_frame.agree.preplace.pass_fail": 1}
