"""first-demo v3.0 (owner findings on v2.2): each fix switches on only in its situation, detected at run time.

F2 (door open 0928-042225): a release / move / re-grasp keeps the arm where the back-off left it when the recorded retreat witness is
another IK solution and the drive and the self-clearance check out with the arm as it is.
F4 (cork 0928-041525): an attach place keeps the holding arm out of the jam probe from the descent until the attachment exists.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.oracle import execute as EX
from ispatialgym.oracle import r1pro_model as M


# ------------------------------------------------------------------------------------------------ F2


def _follow_setup(gap=3.4, collide_at=None, pairs=()):
    from ispatialgym.tests.test_v2_oracle import FakeRobot

    robot = FakeRobot()
    idx = robot.arm_control_idx["right"]
    q_ret = robot.q[idx].clone()
    q_ret[0] += gap
    swept = []

    def check_collision(qs, sync_world=False, margin=0.0):
        swept.append((len(qs), margin))
        hits = th.zeros(len(qs), dtype=th.bool)
        if collide_at is not None:
            hits[collide_at] = True
        return hits

    backend = SimpleNamespace(_jidx={n: i for i, n in enumerate(M.ALL_JOINTS)}, q_full=robot.get_joint_positions, world_to_root=lambda pose: pose,
                              check_collision=check_collision, self_collision_pairs=lambda q, emb, margin=0.0: list(pairs))
    motions = SimpleNamespace(robot=robot, backend=backend)
    state = SimpleNamespace(base_pose2d=lambda: (0.0, 0.0, 0.0))
    seg = {"arm_q_retreat": q_ret.tolist(), "base_path": [[0.14, 0.0, 0.0]], "hold_posture": "retreat"}
    return motions, state, seg, swept


def test_f2_far_retreat_witness_keeps_the_arm_when_the_drive_and_the_self_clearance_check_out():
    motions, state, seg, swept = _follow_setup()
    cfg = EX.ExecuteConfig()
    f = EX.keep_retreat_arm(motions, state, "right", seg, cfg)
    assert f["fired"] and f["gap_rad"] == pytest.approx(3.4, abs=1e-3) and f["threshold_rad"] == cfg.follow_retreat_ramp_max_rad
    n, margin = swept[0]
    assert n == f["drive_configurations"] and n >= 3 and margin == cfg.follow_keep_drive_margin_m  # the start + the 14 cm drive in 5 cm steps


def test_f2_near_witness_or_failed_checks_ramp_as_before():
    cfg = EX.ExecuteConfig()
    motions, state, seg, swept = _follow_setup(gap=0.55)  # null-space drift (the later re-grasps of 0928-042225): the recorded ramp
    f = EX.keep_retreat_arm(motions, state, "right", seg, cfg)
    assert not f["fired"] and "near" in f["why"] and swept == []  # nothing checked, nothing synced
    motions, state, seg, _ = _follow_setup(collide_at=2)
    f = EX.keep_retreat_arm(motions, state, "right", seg, cfg)
    assert not f["fired"] and "collides on the drive" in f["why"]
    motions, state, seg, _ = _follow_setup(pairs=[{"links": ("right_arm_link5", "torso_link4"), "penetration": 0.004}])
    f = EX.keep_retreat_arm(motions, state, "right", seg, cfg)
    assert not f["fired"] and "self-collision padding" in f["why"] and "right_arm_link5-torso_link4" in f["why"]
    motions, state, seg, swept = _follow_setup()
    seg["hold_posture"] = "idle"  # the sampler drove in the idle posture: the arm goes there anyway
    f = EX.keep_retreat_arm(motions, state, "right", seg, cfg)
    assert not f["fired"] and "idle posture" in f["why"] and swept == []


# ------------------------------------------------------------------------------------------------ F4


def _attach_setup(kind="attach", on="wine_bottle.n.01_1", held=True, kept=()):
    motions = SimpleNamespace(keep_hold_arms=set(kept))
    part = object()
    state = SimpleNamespace(held_object=lambda arm: part if held else None)
    action = SimpleNamespace(kind=kind, on=on, arm="right", object="cork.n.04_1")
    return motions, state, action


def test_f4_attach_place_keeps_the_holding_arm_out_of_the_jam_probe_until_the_attachment():
    motions, state, action = _attach_setup()
    ctx = {}
    with EX.attach_hold_kept(action, motions, state, ctx):
        assert motions.keep_hold_arms == {"right"}
    assert motions.keep_hold_arms == set()
    assert ctx["attach_hold"]["fired"] and ctx["attach_hold"]["arm"] == "right"
    # an arm the executor already keeps (a final hold) stays kept after the place window
    motions, state, action = _attach_setup(kept=("right",))
    with EX.attach_hold_kept(action, motions, state, {}):
        pass
    assert motions.keep_hold_arms == {"right"}


def test_f4_attach_hold_released_when_the_place_fails_inside_the_window():
    motions, state, action = _attach_setup()

    def place():
        with EX.attach_hold_kept(action, motions, state, {}):
            yield 1
            raise RuntimeError("attach refused")

    gen = place()
    next(gen)
    assert motions.keep_hold_arms == {"right"}
    gen.close()  # the executor's rollback closes the action's generator mid-place
    assert motions.keep_hold_arms == set()


def test_f4_every_other_place_is_left_alone():
    for kw, why in (({"kind": "support"}, "not an attach place"), ({"on": None}, "not an attach place"), ({"held": False}, "the part is not in hand")):
        motions, state, action = _attach_setup(**kw)
        ctx = {}
        with EX.attach_hold_kept(action, motions, state, ctx):
            assert motions.keep_hold_arms == set()
        assert ctx["attach_hold"] == {"fired": False, "why": why}
