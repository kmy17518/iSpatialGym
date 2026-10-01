"""first-demo v3.0 (owner findings on v2.2): each fix switches on only in its situation, detected at run time.

F1 (world vertical 0928-043205): a transition keeps a free arm as it is when its next action at the arriving stance is a brace / push
and the sweep, its route to the contact and the later arm's hold are collision-free.
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


# ------------------------------------------------------------------------------------------------ F1


def _transition_setup(collide=lambda q: False, contacts=("right", "left")):
    from ispatialgym.oracle.motions import MotionsConfig
    from ispatialgym.tests.test_v2_oracle import FakeRobot

    robot = FakeRobot()
    ridx, lidx = robot.arm_control_idx["right"], robot.arm_control_idx["left"]
    checked = []

    def check_collision(qs, sync_world=False, margin=0.0):
        checked.append(len(qs))
        return th.tensor([bool(collide(q)) for q in qs])

    def solve_ik_path(emb, waypoints, q_start=None, **kw):
        return SimpleNamespace(success=True, q_traj=th.stack([q_start.clone() for _ in waypoints]), status="ok")

    backend = SimpleNamespace(_jidx={n: i for i, n in enumerate(M.ALL_JOINTS)}, q_full=robot.get_joint_positions, world_to_root=lambda pose: pose,
                              check_collision=check_collision, solve_ik_path=solve_ik_path)
    motions = SimpleNamespace(robot=robot, backend=backend, cfg=MotionsConfig(), _densify=lambda start, wps: list(wps))
    state = SimpleNamespace(base_pose2d=lambda: (0.0, 0.0, 0.0), held_object=lambda arm: None, contact_supports=lambda name: [], object=lambda name: None)
    q_w = robot.q.clone()
    q_w[ridx] = th.tensor([0.9, -3.1, -0.7, -1.6, 1.5, -0.7, 0.4])  # the brace witness (right) and the push witness (left)
    q_w[lidx] = th.tensor([-1.7, 0.1, -0.2, -0.2, 2.3, 1.0, 1.1])
    pre = SimpleNamespace(position=[0.5, 0.0, 1.4], orientation=[0.0, 0.0, 0.0, 1.0])
    stance_contacts = {"right": {"index": 2, "name": "brace", "object": "painting.n.01_1", "targets": {"pre": pre}, "witnesses": {"pre": q_w, "contact": q_w}},
                       "left": {"index": 3, "name": "push", "object": "painting.n.01_1", "targets": {"pre": pre}, "witnesses": {"pre": q_w, "waypoints": q_w}}}
    ctx = {"stance_contacts": {a: (stance_contacts[a] if a in contacts else None) for a in M.ARMS}, "protect": {"goal": []}}
    targets = {"trunk": (0.0, 0.0), "drive_in": [0.3, 0.0, 0.0]}
    move = {"right": th.zeros(7), "left": th.zeros(7)}  # the wall stance's recorded posture: tucked
    _WALL_STANCE[0] = SimpleNamespace(extra={"arm_posture": "tucked"}, working_arms=["left", "right"])
    return motions, state, targets, ctx, move, checked, (ridx, lidx)


_WALL_STANCE = [None]


def _keep(motions, state, targets, ctx, move, record=None):
    return EX.keep_contact_arms(motions, state, targets, ctx, move, record=_WALL_STANCE[0] if record is None else record)


def test_f1_both_arms_stay_untucked_when_the_sweep_the_routes_and_the_hold_are_clear():
    motions, state, targets, ctx, move, checked, _ = _transition_setup()
    left = _keep(motions, state, targets, ctx, move)
    rec = ctx["contact_arms_kept"]
    assert left == {} and rec["fired"] and rec["arms"]["right"]["kept"] and rec["arms"]["left"]["kept"]
    assert rec["arms"]["left"]["held_beside"] == "right brace" and rec["arms"]["left"]["hold_configurations"] == 2  # the brace's pre + contact
    assert rec["arms"]["right"]["route_to"].startswith("the brace's pre witness")
    assert len(checked) == 4  # the sweep, the right route, the left route, the left arm's hold beside the brace


def test_f1_a_colliding_sweep_keeps_the_recorded_posture_for_every_arm():
    motions, state, targets, ctx, move, _, _ = _transition_setup(collide=lambda q: float(q[M.ALL_JOINTS.index("base_footprint_x_joint")]) > 0.2)
    assert _keep(motions, state, targets, ctx, move) is move
    rec = ctx["contact_arms_kept"]
    assert not rec["fired"] and "sweep" in rec["why"] and not rec["arms"]["right"]["kept"] and not rec["arms"]["left"]["kept"]


def test_f1_per_arm_route_and_hold_checks():
    motions, state, targets, ctx, move, _, (ridx, lidx) = _transition_setup()
    # the left arm's straight route to its push witness collides (half way there): it is tucked, the right arm still goes straight
    motions, state, targets, ctx, move, _, (ridx, lidx) = _transition_setup(collide=lambda q: -1.0 < float(q[lidx[0]]) < -0.5)
    left = _keep(motions, state, targets, ctx, move)
    rec = ctx["contact_arms_kept"]
    assert sorted(left) == ["left"] and rec["fired"] and rec["arms"]["right"]["kept"] and "route" in rec["arms"]["left"]["why"]
    # the untucked left arm (as it is) meets the right arm on its brace approach (a witness between the pre-pose and the contact):
    # the left arm is tucked, the right arm still goes straight to its brace
    q0 = motions.robot.q.clone()
    motions, state, targets, ctx, move, _, (ridx, lidx) = _transition_setup(
        collide=lambda q: bool(th.allclose(q[lidx], q0[lidx])) and float(q[ridx[0]]) > 1.4)
    wr = ctx["stance_contacts"]["right"]["witnesses"]
    q_app, q_con = wr["pre"].clone(), wr["pre"].clone()
    q_app[ridx[0]], q_con[ridx[0]] = 1.5, 1.2
    ctx["stance_contacts"]["right"]["witnesses"] = {"pre": wr["pre"], "approach": q_app, "contact": q_con}
    left = _keep(motions, state, targets, ctx, move)
    rec = ctx["contact_arms_kept"]
    assert sorted(left) == ["left"] and rec["arms"]["right"]["kept"] and "collides with the right arm's brace" in rec["arms"]["left"]["why"]


def test_f1_outside_its_situation_nothing_is_checked():
    motions, state, targets, ctx, move, checked, _ = _transition_setup(contacts=())
    assert _keep(motions, state, targets, ctx, move) is move
    rec = ctx["contact_arms_kept"]
    assert not rec["fired"] and checked == [] and "not a brace / push" in rec["arms"]["right"]["why"]
    motions, state, targets, ctx, _, checked, _ = _transition_setup()
    assert _keep(motions, state, targets, ctx, {}) == {} and ctx["contact_arms_kept"]["why"] == "no arm to re-pose" and checked == []


def test_f1_only_working_arms_put_into_the_stances_named_posture():
    # a stance that records the arm's own configuration (blender / carton close: the push's posture), not a named posture
    motions, state, targets, ctx, move, checked, _ = _transition_setup()
    assert _keep(motions, state, targets, ctx, move, record=SimpleNamespace(extra={}, working_arms=["left", "right"])) is move
    assert not ctx["contact_arms_kept"]["fired"] and "names no arm posture" in ctx["contact_arms_kept"]["why"] and checked == []
    # the transition's own swept joints (the sandal's push stance: the arm back to where the previous motion left it)
    motions, state, targets, ctx, move, checked, _ = _transition_setup()
    targets["arms_q"] = {"right": [0.0] * 7}
    assert _keep(motions, state, targets, ctx, move) is move and "swept with" in ctx["contact_arms_kept"]["why"] and checked == []
    # the stance's idle arm goes to the posture as before
    motions, state, targets, ctx, move, checked, _ = _transition_setup()
    left = _keep(motions, state, targets, ctx, move, record=SimpleNamespace(extra={"arm_posture": "tucked"}, working_arms=["right"]))
    assert sorted(left) == ["left"] and "not a working arm" in ctx["contact_arms_kept"]["arms"]["left"]["why"]
    assert ctx["contact_arms_kept"]["arms"]["right"]["kept"]


def test_f1_executor_finds_each_arms_contact_at_the_arriving_stance():
    from ispatialgym.oracle.actions import Brace, Push
    from ispatialgym.tests.test_v2_oracle import iors_world, make_executor

    sc, env = iors_world()
    ex, motions, backend, rec = make_executor(env, sc)
    p = SimpleNamespace(position=[0.5, 0.0, 1.0], orientation=[0.0, 0.0, 0.0, 1.0])

    class _A(SimpleNamespace):
        def targets(self, sidecar, state):
            return {"pre": p}

    brace = Brace.__new__(Brace)
    push = Push.__new__(Push)
    for a, arm, name in ((brace, "right", "brace"), (push, "left", "push")):
        a.__dict__.update({"name": name, "arms": (arm,), "contact": SimpleNamespace(object="painting.n.01_1", arm=arm)})
        a.targets = (lambda sidecar, state: {"pre": p})
    from ispatialgym.oracle.actions import Transition

    tr = Transition.__new__(Transition)
    tr.__dict__.update({"name": "transition", "arms": ()})
    ex.program_actions = [tr, brace, push]
    sc.extra["ik_witnesses"] = [{"action": 1, "arm": "right", "targets": {"pre": {"q": [0.0] * 28}}}]
    r, l = ex.stance_contact(0, "right"), ex.stance_contact(0, "left")
    assert r["index"] == 1 and r["name"] == "brace" and sorted(r["witnesses"]) == ["pre"]
    assert l["index"] == 2 and l["name"] == "push" and l["witnesses"] == {}
    ex.program_actions = [tr, brace]
    assert ex.stance_contact(0, "left") is None  # the left arm does nothing at this stance


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
