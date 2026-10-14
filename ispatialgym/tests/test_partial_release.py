"""The partial-opening release of a next-to place (owner 2026-10-14 ~00:43, COMPLAINTS.md nextto#03: "open only as far as the clearance
the sampler already records, and let go there"): ``generation.partial_release`` (default unset = off), copied by the sampler into
``sidecar.extra["partial_release"]``; the oracle loosens the pads with the weld holding, commands +1 until the weld is gone and then
the instance's release opening through the retreat - gripper commands only. The sampler judges the swept fingers at that peak."""
import os
from types import SimpleNamespace

import math
import numpy as np
import pytest
import torch as th

from ispatialgym.oracle import execute as EX
from ispatialgym.oracle.programs import expand
from ispatialgym.oracle.sidecar import Sidecar
from ispatialgym.robot import conventions as C
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2.tasks import iors_small as IS
from ispatialgym.tests.test_v2_oracle import SIDECARS, FakeEnv, FakeObject, FakeRobot, make_executor


def nextto_world(partial=None, plan=None):
    """The nextto test sidecar (salt shaker next to the pepper shaker) as a fake world; ``partial`` -> ``sidecar.extra``, ``plan`` -> the
    grasp's ``release_sweep``."""
    sc = Sidecar.load(os.path.join(SIDECARS, "inter_object_relation_satisfaction-nextto.json"))
    if partial is not None:
        sc.extra["partial_release"] = partial
    if plan is not None:
        sc.grasps[0].extra["release_sweep"] = dict(plan)
    robot = FakeRobot(base=tuple(sc.stance("init").base_pose2d))
    g = sc.grasps[0]
    salt = FakeObject("salt_shaker_fonexa_0", [g.grasp.position[0], g.grasp.position[1], g.grasp.position[2] - 0.03], category="salt_shaker",
                      extent=(0.03, 0.03, 0.08))
    pepper = FakeObject("pepper_shaker_xyzabc_0", [g.place.position[0] + 0.3, g.place.position[1], g.place.position[2] - 0.03], category="pepper_shaker",
                        extent=(0.03, 0.03, 0.08))
    pepper.graspable = False
    env = FakeEnv(robot, {"salt_shaker.n.01_1": salt, "pepper_shaker.n.01_1": pepper})

    def evaluate(env):
        d = float(th.linalg.norm(salt.pos[:2] - th.tensor(g.place.position[:2])))
        ok = d < 0.08 and robot._ag_obj_in_hand["right"] is None
        return {"binary_success": ok, "settled": float(robot.dq.abs().max()) < 1e-6, "errors": {}}

    env.evaluate = evaluate
    return sc, env


def _run(tmp_path, partial=None, plan=None):
    sc, env = nextto_world(partial, plan)
    robot = env.robot
    cmds, held = [], []
    idx = robot.controller_action_idx["gripper_right"]
    env.after_step = lambda a: (cmds.append(float(a[idx])), held.append(robot._ag_obj_in_hand["right"] is not None))
    ex, motions, *_ = make_executor(env, sc, tmp_path)
    result = ex.run_program(expand(sc.alias, sc))
    place = next(a for a in result["actions"] if a["name"] == "place")
    return result, place, cmds, held, robot


def test_regime_is_off_unless_set_and_merges_the_defaults():
    assert C.partial_release_regime(None) is None and C.partial_release_regime(False) is None
    assert C.partial_release_regime(True) == C.PARTIAL_RELEASE_DEFAULTS
    reg = C.partial_release_regime({"loosen_m": 0.003, "unknown": 1})
    assert reg["loosen_m"] == 0.003 and reg["overshoot_m"] == C.PARTIAL_RELEASE_DEFAULTS["overshoot_m"] and "unknown" not in reg
    with pytest.raises(ValueError):
        C.partial_release_regime("yes")
    assert EX.partial_release_of({"sidecar": SimpleNamespace(extra={})}) is None
    assert EX.partial_release_of({"sidecar": SimpleNamespace(extra={"partial_release": "bad"})}) is None  # a bad entry: the full opening
    assert EX.partial_release_of({"sidecar": SimpleNamespace(extra={"partial_release": {"pad_clear_m": 0.005}})})["pad_clear_m"] == 0.005


def test_without_the_key_the_place_opens_to_plus_one_at_once(tmp_path):
    """Owner 2026-10-10: every release is the open command (+1) in one step - unchanged for an instance without the key."""
    result, place, cmds, held, robot = _run(tmp_path)
    assert result["success"], result["verdict"]
    assert "partial_release" not in place["notes"] and [e["event"] for e in place["notes"]["release"]] == ["instant_open"]
    first_open = next(i for i, c in enumerate(cmds) if c == 1.0 and i > 0 and cmds[i - 1] < 1.0 and not held[i])
    assert all(c == 1.0 for c in cmds[first_open:])


def test_partial_release_loosens_lets_go_at_plus_one_then_holds_the_release_opening(tmp_path):
    """With the key: the pads loosen (a command below +1 with the weld still holding), the open command (+1) lets go, then the
    instance's release opening (``release_sweep.release_opening_m``, capped by ``release_opening_max_m``) is commanded through the
    release window and the retreat; the gripper opens fully only after the retreat. Every command that let go was +1."""
    plan = {"release_opening_m": 0.05, "release_opening_max_m": 0.06}
    result, place, cmds, held, robot = _run(tmp_path, partial={"loosen_m": 0.004, "pad_clear_m": 0.004}, plan=plan)
    assert result["success"], result["verdict"]
    rec = place["notes"]["partial_release"]
    assert rec["commanded_q_m"] == pytest.approx(0.025, abs=1e-4) and rec["commanded_opening_m"] == pytest.approx(0.05, abs=2e-4)
    assert rec["open_steps"] == 1 and rec["pads_clear_before_open"] and rec["opened_after_retreat"]
    assert rec["finger_max_m"] >= rec["commanded_q_m"] - 1e-4 and rec["overshoot_m"] == pytest.approx(rec["finger_max_m"] - rec["commanded_q_m"], abs=1e-4)
    assert rec["partner_moved_release_m"] == 0.0 and rec["partner_moved_retreat_m"] == 0.0
    assert [e["event"] for e in place["notes"]["release"]] == ["instant_open"] and place["notes"]["release"][0]["partial"]
    # the commands: hold (closed) -> loosen (< +1, still held) -> +1 (let go) -> the release opening (< +1) ... -> +1 after the retreat
    release_at = next(i for i in range(1, len(held)) if held[i - 1] and not held[i])
    assert cmds[release_at] == 1.0 and cmds[release_at - 1] < 1.0
    q_cmd = EX.Motions.gripper_command(SimpleNamespace(_gripper_limits=lambda arm: (0.0, 0.05)), "right", rec["commanded_q_m"])
    after = cmds[release_at + 1:]
    n_rel = next(i for i, c in enumerate(after) if c == 1.0)
    assert n_rel > 10 and all(c == pytest.approx(q_cmd, abs=1e-6) for c in after[:n_rel])  # the release opening through the retreat
    assert all(c == 1.0 for c in after[n_rel:])
    assert float(robot.q[robot.gripper_control_idx["right"]].max()) > 0.049


def test_partial_release_opening_is_bounded_by_the_part_and_the_recorded_clearance(tmp_path):
    """The release opening clears the held part by ``pad_clear_m`` even when the plan asks for less; a plan wider than the recorded
    ``release_opening_max_m`` is capped there; without a plan the least opening is used."""
    _, place, *_ = _run(tmp_path / "a", partial=True, plan={"release_opening_m": 0.0, "release_opening_max_m": 0.06})
    rec = place["notes"]["partial_release"]
    assert rec["commanded_q_m"] == pytest.approx(rec["touch_m"] + C.PARTIAL_RELEASE_DEFAULTS["pad_clear_m"], abs=1e-4)
    _, place, *_ = _run(tmp_path / "b", partial=True, plan={"release_opening_m": 0.08, "release_opening_max_m": 0.06})
    assert place["notes"]["partial_release"]["commanded_q_m"] == pytest.approx(0.03, abs=1e-4)
    _, place, *_ = _run(tmp_path / "c", partial=True)
    rec = place["notes"]["partial_release"]
    assert rec["planned_opening_m"] is None and rec["commanded_q_m"] == pytest.approx(rec["touch_m"] + C.PARTIAL_RELEASE_DEFAULTS["pad_clear_m"], abs=1e-4)


def _stick(near, table=0.81, r=0.0025):
    """A 5 mm round stick along x lying on the table, its near side ``near`` m from the pinch centre along +y."""
    pts = [[x, near + r + r * math.cos(a), table + r + r * math.sin(a)] for x in (-0.12, -0.01, 0.10) for a in [2 * math.pi * k / 12 for k in range(12)]]
    return th.tensor(pts, dtype=th.float32)


def test_swept_fingers_judge_the_partial_release_where_the_open_jaw_rejects():
    """The fingers' meshes swept from the touch to the partial release's peak (22.2 mm per finger) clear a stick whose near side is 35 mm
    from the pinch (a spoon 38 mm wide beside it at the 16 mm nextto gap) by more than the 5 mm margin, where the fully open jaw's box and
    its largest release opening (26 mm < pinch + 1 cm) reject it; the widest release opening keeping 5 mm grows 1:1 with the distance."""
    s = IS.IORSSmall.__new__(IS.IORSSmall)
    table = 0.81
    place = (th.tensor([0.0, 0.0, table + C.FINGERTIP_LENGTH + 0.002]), C.eef_quat(th.tensor([0.0, 0.0, -1.0]), th.tensor([0.0, 1.0, 0.0])))
    touch, peak, least = 0.00765, 0.02215, 0.01165
    gaps, q_max = {}, {}
    for near in (0.025, 0.035, 0.045):
        probe = IS.HullProbe(_stick(near, table).double().numpy())
        gaps[near] = s.partial_release_gap(place, probe, "right", touch, peak, least)
        q_max[near] = s.partial_release_opening_max(place, probe, "right", touch, 0.005)
        assert s.partial_release_gap(place, probe, "right", touch, C.FINGER_OPEN, C.FINGER_OPEN) < 0  # the fully open fingers sweep it
    assert gaps[0.025] < 0.005 < gaps[0.035] < gaps[0.045]
    assert s.open_jaw_clearance(place, _stick(0.035, table)) < 0 and s.open_jaw_opening_max(place, _stick(0.035, table)) < 0.0382 + 0.01
    assert q_max[0.035] - q_max[0.025] == pytest.approx(0.01, abs=0.001) and q_max[0.045] - q_max[0.035] == pytest.approx(0.01, abs=0.001)
    far = IS.HullProbe(_stick(0.10, table).double().numpy())
    assert s.partial_release_opening_max(place, far, "right", touch, 0.005) == C.FINGER_OPEN
    # the hull gap is a lower bound of the true distance: two unit cubes 0.3 apart
    cube = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
    assert IS.convex_gap(IS.HullProbe(cube), IS.HullProbe(cube + [1.3, 0.0, 0.0])) == pytest.approx(0.3, abs=1e-9)
    assert IS.convex_gap(IS.HullProbe(cube), IS.HullProbe(cube + [0.5, 0.0, 0.0])) < 0


def _spoon_grasp(width=0.0083):
    """A top-down pinch across the spoon's width (eef c = its local y) on the handle, fingertips 2 mm under its bottom (the box model)."""
    from ispatialgym.sampling.v2.stances import GraspRecord

    eio = (th.tensor([0.045, 0.0, -0.0115 + C.FINGERTIP_LENGTH - 0.002]), C.eef_quat(th.tensor([0.0, 0.0, -1.0]), th.tensor([0.0, 1.0, 0.0])))
    return GraspRecord("target", "right", "tablespoon__yxxfdj__st", "grasp_2", 0.0, eef_in_object=eio)


def test_regime_is_nextto_only_and_the_plan_uses_the_library_pinch(monkeypatch):
    from ispatialgym.tests.test_iors_start_region import _sampler

    monkeypatch.setattr(C, "grasp_entry", lambda key, name: SimpleNamespace(meta={"width": 0.0083}))
    assert _sampler("nextto", gen={}).partial_release() is None
    assert _sampler("touching", gen={"partial_release": True}).partial_release() is None
    s = _sampler("nextto", gen={"partial_release": {"touch_extra_m": 0.003}})
    reg = s.partial_release()
    plan = s.partial_release_plan(_spoon_grasp(), reg)
    assert plan["pinch_source"] == "library" and plan["pinch_held_m"] == 0.0083 and plan["touch_q_m"] == pytest.approx(0.00415 + 0.003, abs=1e-4)
    assert plan["peak_q_m"] == pytest.approx(plan["touch_q_m"] + IS.CONTACT_SQUEEZE + reg["loosen_m"] + reg["overshoot_m"], abs=2e-4)
    assert plan["least_q_m"] == pytest.approx(plan["touch_q_m"] + reg["pad_clear_m"], abs=2e-4)
    monkeypatch.setattr(C, "grasp_entry", lambda key, name: (_ for _ in ()).throw(KeyError(name)))
    s.hull = {"target": th.tensor([[x, y, z] for x in (-0.05, 0.05) for y in (-0.006, 0.006) for z in (-0.004, 0.004)]), "partner": s.hull["partner"]}
    eio = (th.tensor([0.0, 0.0, -0.004 + C.FINGERTIP_LENGTH]), C.eef_quat(th.tensor([0.0, 0.0, -1.0]), th.tensor([0.0, 1.0, 0.0])))
    plan = s.partial_release_plan(SimpleNamespace(grasp_key="k", grasp_name="g", eef_in_object=eio), reg)
    assert plan["pinch_source"] == "hull_chord" and plan["pinch_held_m"] == pytest.approx(0.012, abs=1e-4)  # no library row: the chord


def test_nextto03_side_by_side_goal_passes_with_the_partial_release_and_not_with_the_open_jaw(monkeypatch):
    """nextto#03's pair goal (spoon beside the chopstick, the 16 mm nextto gap): the fully open jaw rejects every admissible place
    (release_sweep), the partial release keeps its margin at the plain gap and records the opening the oracle commands."""
    from ispatialgym.tests.test_iors_pair_goal_align import _goal_sampler, _pose

    monkeypatch.setattr(C, "grasp_entry", lambda key, name: SimpleNamespace(meta={"width": 0.0083}))
    outcomes = {}
    for key in (None, True):
        s, manip = _goal_sampler(monkeypatch, "nextto03", g_nt=0.026, gen={} if key is None else {"partial_release": key})
        # the chopstick's hull with points along it, as its collision hull has a ring near the middle (the probe's 8 corners would sit
        # outside the jaw along its length)
        s.hull["partner"] = th.tensor([[x, y, z] for x in np.linspace(-0.123, 0.099, 12) for y in (-0.00435, 0.00435) for z in (-0.0026, 0.0026)],
                                      dtype=th.float32)
        grasp = _spoon_grasp()
        s.sidecar.grasp_of = lambda name: grasp
        t_pose = _pose(s, "target", s.pair_target_yaws(manip)[0], (1.31, -0.24))
        p_pose = s.place_partner(np.random.default_rng(3), t_pose, manip, None)
        try:
            s.goal_pose(t_pose, p_pose)
            outcomes[key] = dict(s.release_sweep)
        except Reject as e:
            outcomes[key] = e.reason
    assert outcomes[None] == "release_sweep"
    rec = outcomes[True]
    assert rec["partial_release"] and rec["remedy"] is None and rec["clearance_peak_m"] >= C.PARTIAL_RELEASE_DEFAULTS["partner_margin_m"]
    assert rec["release_opening_m"] == pytest.approx(2.0 * rec["least_q_m"], abs=2e-4) and rec["release_opening_max_m"] >= rec["release_opening_m"]
    assert rec["regime"] == C.PARTIAL_RELEASE_DEFAULTS and rec["peak_q_m"] < 0.025


def test_nextto03_alias_carries_the_key_alone():
    import json

    path = os.path.join(os.path.dirname(__file__), "..", "assets", "task_specs", "isg_customs.json")
    customs = json.load(open(path))
    keyed = sorted(a for a, e in customs.items() if isinstance(e, dict) and (e.get("generation") or {}).get("partial_release") is not None)
    assert keyed == ["inter_object_relation_satisfaction-nextto-tablespoon-chopstick-breakfast_table"]
    reg = C.partial_release_regime(customs[keyed[0]]["generation"]["partial_release"])
    assert set(reg) == set(C.PARTIAL_RELEASE_DEFAULTS) and all(v > 0 for v in reg.values())


def test_release_clearance_sweeps_the_fingers_only_to_the_partial_peak(monkeypatch):
    """``clear_release_of_partner`` sweeps the fingers from their hold to the partial release's peak (touch + loosen + overshoot), not
    to the open limit."""
    seen = []
    monkeypatch.setattr(EX, "_open_fingers_meet", lambda arm, q, q_open, place, partner, cfg: (seen.append(q_open), (0.0, 1, [], None))[1])
    robot = FakeRobot()
    robot.q[robot.gripper_control_idx["right"]] = 0.006
    motions = SimpleNamespace(robot=robot, cfg=SimpleNamespace(gripper_contact_squeeze_m=0.001), _gripper_limits=lambda arm: (0.0, 0.05))
    state = SimpleNamespace(object=lambda name: SimpleNamespace(aabb=(th.zeros(3), th.ones(3))))
    action = SimpleNamespace(arm="right", on="pepper", object="salt")
    targets = {"place": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])}
    ctx = {"sidecar": SimpleNamespace(extra={"partial_release": {"loosen_m": 0.002, "overshoot_m": 0.0115}})}
    EX.clear_release_of_partner(action, motions, state, targets, ctx, EX.CONFIG)
    assert seen == [pytest.approx(0.006 + 0.001 + 0.002 + 0.0115)] and ctx["partner_release_clearance"]["partial_release_peak_m"] == pytest.approx(0.0205, abs=1e-4)
    ctx = {"sidecar": SimpleNamespace(extra={})}
    EX.clear_release_of_partner(action, motions, state, targets, ctx, EX.CONFIG)
    assert seen[-1] == 0.05 and "partial_release_peak_m" not in ctx["partner_release_clearance"]
