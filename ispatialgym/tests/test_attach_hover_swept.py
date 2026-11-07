"""The attach touchdown's hover from the SWEPT FIRST TOUCH (owner 2026-11-07, configuration_matching-attachment_04 option B):
the hover = first touch + ``attach_hover_clear_m`` under ``ISPATIALGYM_ATTACH_HOVER=swept`` (default ``box``, today's rule, with
the swept first touch recorded beside it), the fingertips touching first, the box fallback of a sweep that never meets (or
raises), the cache by the pose relative to the partner, the touchdown from a swept hover, and the A/B records (the touchdown's
hover source, the drop at the release, the summary). The sweep itself: ``test_convex_sweep.py``."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.oracle import execute as EX
from ispatialgym.sampling import predicates as PRED
from ispatialgym.tests.test_convex_sweep import BASE_Z, HANDLE_TOP, SEAT_BOTTOM, _box, _fingers, _lid, _pot


# ---------------------------------------------------------------------------------------------------------------------- the hover


class _Obj:
    def __init__(self, name, pos, links):
        self.name, self.links = name, links
        self.pos, self.quat = th.tensor(pos, dtype=th.float32), th.tensor([0.0, 0.0, 0.0, 1.0])

    def get_position_orientation(self):
        return self.pos.clone(), self.quat.clone()


def _meshes(arrs, dz=0.0, dx=0.0):
    return {f"m{i}": SimpleNamespace(points_in_world_frame=th.tensor(a + [dx, 0.0, dz], dtype=th.float32)) for i, a in enumerate(arrs)}


def _scene(monkeypatch, lift=0.10, dx=0.0, part_dx=0.0, partner=None, finger_pieces=None):
    """The lid in the hand ``lift`` above its seated pose (the hand at the place pose + lift), the pot under it; ``dx`` moves the
    whole scene, ``part_dx`` the part (and the hand) alone; ``finger_pieces`` (at the seated pose) replace the two fingertips."""
    monkeypatch.setattr(EX, "_seat_axis", lambda state, action: th.tensor([0.0, 0.0, 1.0]))
    lid = _lid(dz=lift, dx=0.0)
    lid_link = SimpleNamespace(collision_meshes=_meshes(lid, dx=dx + part_dx),
                               collision_boundary_points_world=th.tensor(np.concatenate(lid) + [dx + part_dx, 0.0, 0.0], dtype=th.float32))
    part = _Obj("lid_0", [dx + part_dx, 0.0, SEAT_BOTTOM + lift], {"base": lid_link})
    pot = _Obj("stockpot_0", [dx, 0.0, BASE_Z], {"base": SimpleNamespace(collision_meshes=_meshes(partner if partner is not None else _pot(), dx=dx))})
    fingers = [SimpleNamespace(collision_meshes=_meshes([f], dz=lift, dx=dx + part_dx),
                               collision_boundary_points_world=th.tensor(f + [dx + part_dx, 0.0, lift], dtype=th.float32))
               for f in (finger_pieces if finger_pieces is not None else _fingers())]
    eef_seat = [dx + part_dx, 0.0, SEAT_BOTTOM + 0.08]
    state = SimpleNamespace(object=lambda n: part if n == "lid_0" else pot, attachment_links=lambda c, p: (None, None),
                            eef_pose=lambda arm: (th.tensor([eef_seat[0], 0.0, eef_seat[2] + lift]), th.tensor([0.0, 0.0, 0.0, 1.0])))
    motions = SimpleNamespace(robot=SimpleNamespace(finger_links={"right": fingers}))
    action = SimpleNamespace(object="lid_0", on="stockpot_0", arm="right")
    targets = {"place": EX.Pose(position=list(eef_seat), orientation=[0.0, 0.0, 0.0, 1.0]), "descent": [EX.Pose(position=list(eef_seat), orientation=[0.0, 0.0, 0.0, 1.0])],
               "retreat": [EX.Pose(position=[eef_seat[0], 0.0, eef_seat[2] + 0.05], orientation=[0.0, 0.0, 0.0, 1.0])]}
    return action, motions, state, targets


def test_the_switch_defaults_to_the_box_rule(monkeypatch, caplog):
    monkeypatch.delenv("ISPATIALGYM_ATTACH_HOVER", raising=False)
    assert EX.ExecuteConfig().attach_hover_rule == "box" and EX.attach_hover_rule(EX.ExecuteConfig()) == "box"
    monkeypatch.setenv("ISPATIALGYM_ATTACH_HOVER", "swept")
    assert EX.attach_hover_rule(EX.ExecuteConfig()) == "swept"
    monkeypatch.setenv("ISPATIALGYM_ATTACH_HOVER", " Swept ")
    assert EX.attach_hover_rule(EX.ExecuteConfig()) == "swept"
    monkeypatch.setenv("ISPATIALGYM_ATTACH_HOVER", "sweep")
    with caplog.at_level(logging.WARNING, logger=EX.log.name):
        assert EX.attach_hover_rule(EX.ExecuteConfig()) == "box"
    assert "not one of box/swept" in caplog.text


def test_swept_hover_is_the_first_touch_plus_the_clearance_and_box_stays_todays(monkeypatch):
    monkeypatch.setattr(EX, "_SWEEP_CACHE", {})
    action, motions, state, targets = _scene(monkeypatch)
    ctx = {}
    out = EX.attach_hover_targets(action, motions, state, targets, ctx, EX.ExecuteConfig(attach_hover_rule="swept"))
    rec = ctx["attach"]["hover"]
    assert rec["rule"] == rec["source"] == "swept" and rec["lowest"] == "part"
    assert rec["first_touch_m"] == pytest.approx(0.0123, abs=1e-5) and rec["height_above_seat_m"] == pytest.approx(0.0153, abs=1e-4)
    assert rec["part_clearance_m"] == pytest.approx(0.003, abs=1e-4) and rec["gap_to_first_touch_m"] == pytest.approx(0.003, abs=1e-5)
    assert rec["first_touch_fingers_m"] < rec["first_touch_m"]  # the fingertips ride above the lid: they would meet the pot's bottom
    assert rec["box_height_above_seat_m"] == pytest.approx(HANDLE_TOP + 0.003 - SEAT_BOTTOM, abs=1e-4)  # the handles' top + 3 mm
    assert out["place"].position[2] == pytest.approx(targets["place"].position[2] + 0.0153, abs=1e-4) and out["descent"] == [out["place"]]
    assert out["retreat"][0].position[2] == pytest.approx(targets["retreat"][0].position[2] + 0.0153, abs=1e-4)
    assert rec["sweep"]["cached"] is False and rec["sweep"]["pieces"] == {"part": 2, "fingers": 2, "partner": 21}
    # the default: today's box rule, the swept first touch recorded beside it (the gap the touchdown would have to travel)
    ctx = {}
    out = EX.attach_hover_targets(action, motions, state, targets, ctx, EX.ExecuteConfig(attach_hover_rule="box"))
    rec = ctx["attach"]["hover"]
    h_box = HANDLE_TOP + 0.003 - SEAT_BOTTOM
    assert rec["rule"] == rec["source"] == "box" and rec["height_above_seat_m"] == pytest.approx(h_box, abs=1e-4)
    assert rec["part_clearance_m"] == pytest.approx(0.003, abs=1e-4) and rec["partner_top_z"] == pytest.approx(HANDLE_TOP, abs=1e-4)
    assert rec["first_touch_m"] == pytest.approx(0.0123, abs=1e-5) and rec["gap_to_first_touch_m"] == pytest.approx(h_box - 0.0123, abs=1e-4)
    assert rec["gap_to_first_touch_m"] > EX.ExecuteConfig().attach_hover_clear_m + EX.ExecuteConfig().attach_touchdown_extra_m  # never touches
    assert out["place"].position[2] == pytest.approx(targets["place"].position[2] + h_box, abs=1e-4)
    assert rec["sweep"]["cached"] is True  # the same part / hand / pot as the swept call


def test_the_fingertips_touching_first_set_the_swept_hover(monkeypatch):
    monkeypatch.setattr(EX, "_SWEEP_CACHE", {})
    # a post under the right fingertip, beside the knob, higher than the rim: the fingers meet it before the lid meets the rim
    post = _box(0.013, 0.019, -0.004, 0.004, BASE_Z, SEAT_BOTTOM + 0.04)
    action, motions, state, targets = _scene(monkeypatch, partner=_pot() + [post])
    ctx = {}
    EX.attach_hover_targets(action, motions, state, targets, ctx, EX.ExecuteConfig(attach_hover_rule="swept"))
    rec = ctx["attach"]["hover"]
    # the post goes through the lid's disc (the pieces collide only with the partner's): the lid meets the post's top first too
    assert rec["first_touch_m"] == pytest.approx(0.04, abs=1e-5) and rec["first_touch_fingers_m"] == pytest.approx(0.04 - 0.012, abs=1e-5)
    # a fingertip reaching past the lid's edge (its bottom 4 mm above the lid's) over a post outside the pot's wall, 30 mm above the
    # lid's seated bottom: the fingertip meets the post 26 mm above the seat, the lid the rim at 12.3 mm - the fingertip sets the hover
    tip = _box(0.117, 0.125, -0.005, 0.005, SEAT_BOTTOM + 0.004, SEAT_BOTTOM + 0.05)
    post = _box(0.118, 0.124, -0.004, 0.004, BASE_Z, SEAT_BOTTOM + 0.03)
    action, motions, state, targets = _scene(monkeypatch, partner=_pot() + [post], finger_pieces=[tip, _fingers()[1]])
    ctx = {}
    EX.attach_hover_targets(action, motions, state, targets, ctx, EX.ExecuteConfig(attach_hover_rule="swept"))
    rec = ctx["attach"]["hover"]
    assert rec["first_touch_m"] == pytest.approx(0.0123, abs=1e-5) and rec["first_touch_fingers_m"] == pytest.approx(0.026, abs=1e-5)
    assert rec["lowest"] == "fingertips" and rec["height_above_seat_m"] == pytest.approx(0.029, abs=1e-4)
    assert rec["finger_clearance_m"] == pytest.approx(0.003, abs=1e-4) and rec["part_clearance_m"] == pytest.approx(0.029 - 0.0123, abs=1e-4)
    assert rec["gap_to_first_touch_m"] == pytest.approx(0.029 - 0.0123, abs=1e-4)


def test_a_sweep_that_never_meets_falls_back_to_the_box_rule(monkeypatch, caplog):
    monkeypatch.setattr(EX, "_SWEEP_CACHE", {})
    # the pot 30 cm aside: no piece of the lid meets one of the pot along the axis
    far = [p + [0.30, 0.0, 0.0] for p in _pot()]
    action, motions, state, targets = _scene(monkeypatch, partner=far)
    ctx = {}
    with caplog.at_level(logging.WARNING, logger=EX.log.name):
        EX.attach_hover_targets(action, motions, state, targets, ctx, EX.ExecuteConfig(attach_hover_rule="swept"))
    rec = ctx["attach"]["hover"]
    assert rec["rule"] == "swept" and rec["source"] == "box" and rec["first_touch_m"] is None and rec["gap_to_first_touch_m"] is None
    assert rec["swept_fallback"].startswith("no piece of the part meets") and "the box rule" in caplog.text
    assert rec["height_above_seat_m"] == pytest.approx(rec["box_height_above_seat_m"])
    # a sweep that raises: the box rule too
    monkeypatch.setattr(EX, "swept_first_touch", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("qhull")))
    ctx = {}
    action, motions, state, targets = _scene(monkeypatch)
    EX.attach_hover_targets(action, motions, state, targets, ctx, EX.ExecuteConfig(attach_hover_rule="swept"))
    rec = ctx["attach"]["hover"]
    assert rec["source"] == "box" and "qhull" in rec["swept_fallback"] and "qhull" in rec["sweep"]["error"]


def test_the_sweep_is_cached_by_the_pose_relative_to_the_partner(monkeypatch):
    monkeypatch.setattr(EX, "_SWEEP_CACHE", {})
    cfg = EX.ExecuteConfig(attach_hover_rule="swept")
    first = {}
    EX.attach_hover_targets(*_scene(monkeypatch), first, cfg)
    moved = {}
    EX.attach_hover_targets(*_scene(monkeypatch, dx=0.5), moved, cfg)  # the whole scene 50 cm aside: the same relative geometry
    assert moved["attach"]["hover"]["sweep"]["cached"] is True
    assert moved["attach"]["hover"]["first_touch_m"] == first["attach"]["hover"]["first_touch_m"]
    shifted = {}
    EX.attach_hover_targets(*_scene(monkeypatch, part_dx=0.005), shifted, cfg)  # the lid 5 mm across the pot: swept again
    assert shifted["attach"]["hover"]["sweep"]["cached"] is False and len(EX._SWEEP_CACHE) == 2


def test_the_touchdown_from_a_swept_hover_reaches_the_first_touch_with_the_extra_to_spare(monkeypatch):
    monkeypatch.setattr(EX, "_SWEEP_CACHE", {})
    action, motions, state, targets = _scene(monkeypatch)
    ctx = {}
    cfg = EX.ExecuteConfig(attach_hover_rule="swept")
    EX.attach_hover_targets(action, motions, state, targets, ctx, cfg)
    legs = []
    eef = {"p": th.tensor([0.0, 0.0, 1.0])}

    def arm_along(arm, poses, **kw):
        legs.append(poses)
        for _ in range(20):
            eef["p"] = eef["p"] - th.tensor([0.0, 0.0, 0.0005])
            yield th.zeros(1)

    k = {"n": 0}

    def contact(c, p):
        k["n"] += 1
        return k["n"] > 6  # the check before the leg, then 3 mm at 0.5 mm per step

    monkeypatch.setattr(PRED, "pair_in_contact", contact)
    tmotions = SimpleNamespace(arm_along=arm_along, hold=lambda n=1: iter([th.zeros(1)] * n))
    tstate = SimpleNamespace(object=lambda n: SimpleNamespace(name=n), eef_pose=lambda arm: (eef["p"].clone(), th.tensor([0.0, 0.0, 0.0, 1.0])))
    list(EX._touch_down(action, tmotions, tstate, ctx, cfg))
    td = ctx["attach"]["touchdown"]
    assert td["depth_max_m"] == pytest.approx(0.003 + cfg.attach_touchdown_extra_m, abs=1e-4)
    assert td["hover_source"] == "swept" and td["gap_to_first_touch_m"] == pytest.approx(0.003, abs=1e-5)
    assert td["touched"] and td["fallback"] is None and td["lowered_m"] == pytest.approx(0.003, abs=1e-5)


def test_the_release_records_the_drop_and_the_ab_summary(monkeypatch):
    def run(contact_at, join_at, vertical, join_vertical=None):
        k = {"n": 0}
        monkeypatch.setattr(PRED, "JOINS", [])

        def attached(child, parent):
            if k["n"] >= join_at and not PRED.JOINS:
                PRED.JOINS.append({"trigger": "per_step", "mode": "snap", "contact": True, "vertical_m": join_vertical})
            return k["n"] >= join_at

        def contact(child, parent):
            k["n"] += 1
            return k["n"] >= contact_at

        monkeypatch.setattr(PRED, "pair_in_contact", contact)
        monkeypatch.setattr(EX, "_hover_offset", lambda state, action: {"vertical_m": vertical(k["n"]), "lateral_m": 0.0005, "rot_rad": 0.004})
        motions = SimpleNamespace(gripper=lambda arm, close: iter([th.zeros(1)]), hold=lambda n=1: iter([th.zeros(1)] * n),
                                  backend=SimpleNamespace(detach_object=lambda arm: None))
        state = SimpleNamespace(object=lambda n: SimpleNamespace(name=n), held_object=lambda arm: None, attached=attached)
        ctx = {"attach": {"hover": {"rule": "box", "source": "box", "first_touch_m": 0.0123, "height_above_seat_m": 0.0416, "gap_to_first_touch_m": 0.0293},
                          "touchdown": {"touched": False, "lowered_m": 0.007}}}
        list(EX._release_into_join(SimpleNamespace(arm="right", object="lid_0", on="stockpot_0"), motions, state, ctx, EX.ExecuteConfig()))
        return ctx

    # attachment_04's box arm: released 34.7 mm above the seat, falling; the first contact and the join in the same step at 13.1 mm
    ctx = run(3, 3, lambda n: 0.0347 - 0.008 * n, join_vertical=0.0131)
    rel = ctx["attach"]["release"]
    assert rel["vertical_at_release_m"] == pytest.approx(0.0347) and rel["drop_m"] == pytest.approx(0.0347 - 0.0131, abs=1e-6)
    s = ctx["attach"]["summary"]
    assert s == {"hover_rule": "box", "hover_source": "box", "first_touch_m": 0.0123, "hover_m": 0.0416, "gap_to_first_touch_m": 0.0293, "touched": False,
                 "lowered_m": 0.007, "drop_m": rel["drop_m"], "attached": True}
    # a touched-down part: in contact at the first step after the open, joined a step later - no drop
    ctx = run(1, 2, lambda n: 0.0050)
    assert ctx["attach"]["release"]["drop_m"] == pytest.approx(0.0, abs=1e-9)
