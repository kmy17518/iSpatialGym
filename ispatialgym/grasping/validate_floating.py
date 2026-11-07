"""Physics validation of FLOATING grasp candidates (:mod:`floating`; i7-floating 2026-11-07): the held object in hand, under gravity, then
through the held motions of its task - in the persistent validation world of :mod:`validate_tabletop` (the same simulator settings: 30 Hz
control / 120 Hz physics, the JoyLo R1Pro with assisted grasping, the evaluator's base mass, the task's AG rules - FixedJoint, pad-clear
release - the rigid sleep threshold and the sleep assist). Its slab stays parked: a floating object has no support.

Per job (``run_model``) and per hold of its held tasks (``job["held_tasks"]``: ``mental_rotation`` roll 15 deg, ``object_scaling`` roll
90 deg - ``generation_defaults``' ``hold``; default both):

1. **hold**: the robot at the base origin, trunk ``hold.trunk``, the idle arm at the JoyLo reset (zeros), the holding arm at a cuRobo IK
   solution of the hold eef pose (``held_object.hold_eef_pose``: the pad centre on the head camera's optical axis at ``hold.depth_m``,
   approach base +y, closing = world down rolled ``roll_deg`` about it); a snapshot of that state;
2. **grasp** (per family representative, from the snapshot, verified): the object put into the fully open hand at the candidate's
   ``T_object_eef`` with gravity off, and the HELD SAMPLER's closure (:func:`close_like_held_sampler` = ``HeldObjectV2.establish_grasp``:
   the fingers advanced ``grasp_finger_advance_m_per_step`` per control step until both pads touch, ``grasp_press_m`` past the contact
   through the assisted-grasp window - the object re-posed every step until the joint exists - gravity on for ``GRASP_GRAVITY_STEPS``);
3. **held motions** (the object attached, the gripper target at the contact like the recorded start's controller targets): the oracle's
   own arm tracking (``Motions.arm_along``, the ``Reorient`` executor's call, default speeds) through the task's key poses -
   ``mental_rotation``: orbits about the object root (``conventions.orbit_waypoints``, 3 intermediate + the goal) by +a, -2a, +a about
   the base x and the base y axes (a = ``rotation_deg``: the object's weight turned through four directions relative to the grasp);
   ``object_scaling``: straight lines along the optical axis (``conventions.line_waypoints``) by ``line_offsets_m`` (toward the camera
   and back) - each leg followed by ``leg_settle_steps`` of hold;
4. **acceptance**: the assisted grasp formed and holds after gravity; the achieved grasp within ``on_target_*`` of the candidate; after
   gravity and after every executed leg the object still attached with its in-hand pose within ``retain_*`` of the post-gravity pose;
   at least ``min_legs`` executed legs per held task (a leg the arm cannot track - ``plan_failed`` - is recorded as unreachable: a reach
   limit of the test pose, not a property of the grasp).

Fidelity notes: the hold is the sampler's fixed hold at a base at the origin (the instance's base pose does not change the in-hand
physics); the oracle's ``finish`` fallback of a stalled leg is not used (a stalled leg counts as unreachable). Thresholds are the
tabletop validator's (PROVISIONAL, uncalibrated)."""

from __future__ import annotations

import math
import os
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ispatialgym.grasping import contact_policy as CP
from ispatialgym.grasping import schema as S
from ispatialgym.grasping import validate_tabletop as VT

VALIDATOR_VERSION = "isg-floating-validator/1"
HELD_TASKS = ("mental_rotation", "object_scaling")


@dataclass
class FloatingValidateConfig:
    arm: str = "right"
    max_trials: int = 10
    max_valid: int = 6
    # the held sampler's closure (HeldObjectV2.establish_grasp; generation_defaults _HELD_OBJECT_COMMON; held_object constants)
    init_steps: int = 8
    advance_m_per_step: float = 0.001
    press_m: float = 0.004
    attach_steps: int = 90
    gravity_steps: int = 30
    # acceptance (validate_tabletop.ValidateConfig's values)
    retain_pos_m: float = 0.01
    retain_rot_rad: float = 0.10
    on_target_pos_m: float = 0.015
    on_target_rot_rad: float = math.radians(15.0)
    # held motions
    rotation_deg: float = 90.0
    rotation_axes: Tuple[str, ...] = ("x", "y")
    line_offsets_m: Tuple[float, ...] = (-0.25, 0.25)
    leg_settle_steps: int = 10
    min_legs: int = 1
    ik_seeds: int = 16
    sleep_assist: bool = True
    #: the arm at the hold: the eef within this of the hold pose (diagnostic of the IK / drive; the object is placed at the actual eef)
    hold_tol_m: float = 0.01


def held_tasks_of(job: Dict[str, Any]) -> List[str]:
    t = [x for x in (job.get("held_tasks") or []) if x in HELD_TASKS]
    return sorted(set(t)) or list(HELD_TASKS)


def hold_of(task: str):
    """The task's fixed hold (``generation_defaults``)."""
    from ispatialgym.sampling import generation_defaults as GD
    from ispatialgym.sampling.v2.tasks import held_object as HO

    return HO.HoldConstants.from_generation(GD.defaults_for(task)["hold"])


def rotation_legs(axes: Sequence[str], deg: float) -> List[Tuple[str, np.ndarray, float]]:
    """``(name, world axis, angle rad)`` of the rotation legs: +a, -2a, +a about each axis (back at the hold after each axis)."""
    out = []
    a = math.radians(float(deg))
    for ax in axes:
        v = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}[ax]
        out += [(f"rot_{ax}+{deg:g}", v, a), (f"rot_{ax}-{2 * deg:g}", v, -2 * a), (f"rot_{ax}+{deg:g}b", v, a)]
    return out


def line_legs(offsets: Sequence[float]) -> List[Tuple[str, float]]:
    """``(name, displacement m along the optical axis, + = away from the camera)`` of the line legs."""
    return [(f"line{d:+.2f}", float(d)) for d in offsets]


def _pose_err(Ta: np.ndarray, Tb: np.ndarray) -> Tuple[float, float]:
    return float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])), VT._Rang(Ta[:3, :3], Tb[:3, :3])


class FloatingRunner:
    """The floating trials in a :class:`validate_tabletop.TabletopWorld` (one world serves tabletop and floating jobs)."""

    def __init__(self, world: VT.TabletopWorld, cfg: Optional[FloatingValidateConfig] = None):
        self.world, self.cfg = world, (FloatingValidateConfig() if cfg is None else cfg)
        self.th = world.th
        self.robot = world.robot
        self.holds: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ the hold

    def park_slab(self):
        th = self.th
        self.world.slab.set_position_orientation(th.tensor([3.0, 3.0, -1.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))

    def hold_q(self, task: str) -> Dict[str, Any]:
        """The robot configuration of ``task``'s hold (cached per task): IK of the hold eef pose from the default posture."""
        if task in self.holds:
            return self.holds[task]
        th, w, cfg = self.th, self.world, self.cfg
        from ispatialgym.oracle import r1pro_model as M
        from ispatialgym.robot import conventions as C
        from ispatialgym.sampling.v2.tasks import held_object as HO

        hold = hold_of(task)
        q = th.zeros(28)
        q[self.robot.trunk_control_idx] = C.torso_q(hold.translate, hold.tilt, w.trunk_limits)
        q[self.robot.gripper_control_idx["left"]] = 0.05
        q[self.robot.gripper_control_idx[cfg.arm]] = 0.05
        rp, rq = w.root
        root2d = (float(rp[0]), float(rp[1]), C.yaw_of(rq.float()))
        q = C.q_with_base_pose(q, 0.0, 0.0, 0.0, root_pose2d=root2d, base_z=C.NOMINAL_BASE_Z, root_z=float(rp[2]), joint_index=w.jidx)
        cam = C.camera_pose(q[self.robot.trunk_control_idx], base_pose=(0.0, 0.0, 0.0))
        (pos, quat), pad = HO.hold_eef_pose(cam, 0.0, hold, cfg.arm)
        succ, qs = w.backend.solve_ik_batch(f"{cfg.arm}_arm", M.EEF_LINK[cfg.arm], [(pos.float(), quat.float())], q, batch_size=8, seeds=cfg.ik_seeds)
        if not bool(succ[0]):
            raise RuntimeError(f"{task}: no IK solution for the hold eef pose {pos.tolist()}")
        qh = qs[0].clone()
        axis = C.quat_rotate(C.as_pose(cam)[1], th.tensor([0.0, 0.0, -1.0]))
        rec = {"task": task, "hold": hold.to_json(), "q": qh, "eef": (pos.float(), quat.float()), "pad": pad.float(), "camera": cam,
               "optical_axis": axis.float(), "arm_q": [round(float(v), 5) for v in qh[self.robot.arm_control_idx[cfg.arm]].tolist()]}
        self.holds[task] = rec
        return rec

    def set_hold(self, rec: Dict[str, Any], obj, park_xyz=(3.0, -3.0, 1.0)):
        th = self.th
        for arm in ("left", "right"):
            try:
                self.robot.release_grasp_immediately(arm)
            except Exception:  # noqa: BLE001
                pass
        obj.disable_gravity()
        obj.set_position_orientation(th.tensor(park_xyz, dtype=th.float32), th.tensor([0.0, 0.0, 0.0, 1.0]))
        obj.keep_still()
        self.robot.set_joint_positions(rec["q"])
        self.robot.keep_still()
        a = self.world.hold_action()
        for _ in range(4):
            self.world.step(a)

    # ------------------------------------------------------------------ the held sampler's closure

    def finger_pairs(self) -> List[Tuple[int, str]]:
        """``HeldObjectV2.finger_joints``: gripper joint indices paired with the finger link each one moves."""
        arm = self.cfg.arm
        gidx = [int(i) for i in self.robot.gripper_control_idx[arm].tolist()]
        names = list(self.robot.joints.keys())
        finger_paths = {l.prim_path for l in self.robot.finger_links[arm]}
        pairs = []
        for i in gidx:
            body1 = str(self.robot.joints[names[i]].body1)
            link = next((p for p in finger_paths if p == body1 or p.endswith(body1.split("/")[-1])), None)
            pairs.append((i, link))
        if any(l is None for _, l in pairs) or len({l for _, l in pairs}) != len(pairs):
            pairs = [(i, p) for i, p in zip(gidx, sorted(finger_paths))]
        return pairs

    def close_like_held_sampler(self, obj, pose, step) -> Dict[str, Any]:
        """``HeldObjectV2.establish_grasp`` (v1 §5.2) with ``step(action)`` as the control step: fully open, the object held at ``pose`` for
        ``init_steps``; each finger advanced ``advance_m_per_step`` per step until both pads touch; the drives ``press_m`` past the contact
        while the object is re-posed every step until the assisted grasp attaches (``attach_steps``); the fingers back at the contact,
        gravity on for ``gravity_steps``. Returns the outcome (``attached``, ``held_after_gravity``, ``fail``, ``q_contact``)."""
        from ispatialgym.oracle.action import ActionState

        th, cfg, arm, robot = self.th, self.cfg, self.cfg.arm, self.robot
        state = ActionState(robot)
        state.hold_current()
        pairs = self.finger_pairs()
        gidx = th.tensor([i for i, _ in pairs], dtype=th.long)
        lo, hi = robot.joint_lower_limits[gidx], robot.joint_upper_limits[gidx]
        target_paths = {l.prim_path for l in obj.links.values()}
        p, q = (th.as_tensor(pose[0], dtype=th.float32), th.as_tensor(pose[1], dtype=th.float32))

        def cmd_for(qq):
            return float(th.clamp(2.0 * (qq.mean() - lo.mean()) / (hi.mean() - lo.mean()) - 1.0, -1.0, 1.0))

        def touching():
            _, links = robot._find_gripper_contacts(arm)
            touched = set()
            for tp in target_paths:
                touched |= set(links.get(tp, ()))
            return [path in touched for _, path in pairs]

        qf = hi.clone()
        state.gripper_cmd[arm] = cmd_for(qf)
        for _ in range(cfg.init_steps):
            obj.set_position_orientation(p, q)
            obj.keep_still()
            robot.set_joint_positions(qf, indices=gidx, drive=False)
            step(state.action())
        out: Dict[str, Any] = {"attached": False, "held_after_gravity": False, "fail": None}
        contact = touching()
        for _ in range(int(float((hi - lo).max()) / cfg.advance_m_per_step) + 10):
            if all(contact):
                break
            moved = False
            for k in range(len(pairs)):
                if not contact[k] and float(qf[k]) > float(lo[k]) + 1e-4:
                    qf[k] = max(float(lo[k]), float(qf[k]) - cfg.advance_m_per_step)
                    moved = True
            if not moved:
                out["fail"] = f"fingers closed without both pads touching the object (contact {contact})"
                return out
            robot.set_joint_positions(qf, indices=gidx, drive=False)
            state.gripper_cmd[arm] = cmd_for(qf)
            obj.keep_still()
            step(state.action())
            contact = touching()
        if not all(contact):
            out["fail"] = f"pads not both in contact after closing (contact {contact})"
            return out
        q_contact = qf.clone()
        out["q_contact"] = [round(float(v), 5) for v in q_contact.tolist()]
        state.gripper_cmd[arm] = cmd_for(th.clamp(q_contact - cfg.press_m, min=lo))
        for i in range(cfg.attach_steps):
            if robot._ag_obj_in_hand[arm] is None:
                obj.set_position_orientation(p, q)
                obj.keep_still()
            step(state.action())
            if robot._ag_obj_in_hand[arm] is obj:
                out["attached"] = True
                out["attached_after_steps"] = i + 1
                break
        if not out["attached"]:
            out["fail"] = "assisted grasping did not attach the object (both pads in contact)"
            return out
        out["T_object_eef_at_attach"] = self._T_oe(obj)
        robot.set_joint_positions(q_contact, indices=gidx, drive=False)
        state.gripper_cmd[arm] = cmd_for(q_contact)
        obj.enable_gravity()
        for _ in range(cfg.gravity_steps):
            step(state.action())
        out["held_after_gravity"] = robot._ag_obj_in_hand[arm] is obj
        if not out["held_after_gravity"]:
            out["fail"] = "object dropped after gravity was re-enabled"
        out["gripper_cmd"] = cmd_for(q_contact)
        return out

    def _T_oe(self, obj) -> np.ndarray:
        p, q = obj.get_position_orientation()
        ep, eq = self.robot.eef_links[self.cfg.arm].get_position_orientation()
        return np.linalg.inv(S.pose_mat(p.numpy(), q.numpy())) @ S.pose_mat(ep.numpy(), eq.numpy())

    # ------------------------------------------------------------------ the held motions

    def legs_for(self, task: str, rec: Dict[str, Any], obj) -> List[Dict[str, Any]]:
        """The legs of ``task`` as functions of the current eef pose / object root (computed when the leg starts)."""
        if task == "mental_rotation":
            return [{"name": n, "kind": "orbit", "axis": v, "angle": a} for n, v, a in rotation_legs(self.cfg.rotation_axes, self.cfg.rotation_deg)]
        return [{"name": n, "kind": "line", "d": d} for n, d in line_legs(self.cfg.line_offsets_m)]

    def run_leg(self, leg: Dict[str, Any], rec: Dict[str, Any], obj, motions, run) -> Dict[str, Any]:
        from ispatialgym.oracle.motions import SkillError
        from ispatialgym.robot import conventions as C

        th, arm = self.th, self.cfg.arm
        start = motions.eef_pose(arm)
        if leg["kind"] == "orbit":
            pivot = obj.get_position_orientation()[0].float().clone()
            dq = C.quat_from_axis_angle(th.as_tensor(leg["axis"], dtype=th.float32), float(leg["angle"]))
            wps = C.orbit_waypoints(start, pivot, dq, n=4)
        else:
            end = (start[0] + rec["optical_axis"] * float(leg["d"]), start[1].clone())
            wps = C.line_waypoints(start, end, 4)
        t0 = time.time()
        out: Dict[str, Any] = {"name": leg["name"], "kind": leg["kind"]}
        try:
            run(motions.arm_along(arm, wps), leg["name"])
            out["executed"] = True
        except SkillError as e:
            out["executed"] = False
            out["unreachable"] = f"{getattr(e, 'reason', 'skill_error')}: {str(e)[:160]}"
        except Exception as e:  # noqa: BLE001 - a crash of the tracking is recorded (the trial is invalid: leg_error), the next trial runs
            out["executed"] = False
            out["error"] = repr(e)[:300]
            out["traceback"] = traceback.format_exc()[-1200:]
        run(motions.hold(self.cfg.leg_settle_steps), leg["name"] + ":settle")
        goal = wps[-1]
        now = motions.eef_pose(arm)
        out["eef_err_m"] = round(float(th.linalg.norm(now[0] - goal[0])), 5)
        out["eef_err_rad"] = round(float(C.quat_angle(now[1], goal[1])), 5)
        out["wall_s"] = round(time.time() - t0, 2)
        return out

    # ------------------------------------------------------------------ one trial

    def trial(self, obj, cand: Dict[str, Any], task: str) -> Dict[str, Any]:
        from ispatialgym.oracle.motions import Motions, MotionsConfig

        cfg, w, arm = self.cfg, self.world, self.cfg.arm
        t0 = time.time()
        rec = self.hold_q(task)
        self.set_hold(rec, obj)
        ep, eq = self.robot.eef_links[arm].get_position_orientation()
        T_we = S.pose_mat(ep.numpy(), eq.numpy())
        hold_err = _pose_err(T_we, S.pose_mat(rec["eef"][0].numpy(), rec["eef"][1].numpy()))
        T_c = S.pose_of(cand["T_object_eef"])
        T_wo = T_we @ np.linalg.inv(T_c)
        pose = (T_wo[:3, 3].astype(np.float32), S.mat_to_quat(T_wo[:3, :3]).astype(np.float32))
        tr: Dict[str, Any] = {"schema": S.TRIAL, "candidate": cand["name"], "family_id": None, "rest_id": None, "mode": "floating", "task": task,
                              "hold": rec["hold"], "hold_eef_err": [round(hold_err[0], 5), round(hold_err[1], 5)], "hold_arm_q": rec["arm_q"]}
        if hold_err[0] > cfg.hold_tol_m:
            tr.update({"verdict": "invalid", "reasons": ["hold_not_reached"]})
            return tr
        obj.wake()
        sa = w.SA.SleepAssist(self.robot, {obj.name: obj}, "gbf") if cfg.sleep_assist else None
        k = [0]
        fo: List[float] = []

        def step(a, phase=""):
            f = w.step(a, sa, k[0])
            k[0] += 1
            fo.append(f.get("finger_obj", 0.0))

        def run(gen, phase):
            for a in gen:
                step(a, phase)

        cl = self.close_like_held_sampler(obj, pose, step)
        reasons: List[str] = []
        if cl.get("fail"):
            reasons.append("ag_not_formed" if not cl["attached"] else "dropped_under_gravity")
        legs: List[Dict[str, Any]] = []
        if cl["attached"]:
            off = _pose_err(cl["T_object_eef_at_attach"], T_c)
            tr["achieved"] = {"T_object_eef_at_attach": S.mat_pose(cl["T_object_eef_at_attach"]), "off_target_m": round(off[0], 5), "off_target_rad": round(off[1], 5)}
            if off[0] > cfg.on_target_pos_m or off[1] > cfg.on_target_rot_rad:
                reasons.append("grasp_off_target")
        if cl["held_after_gravity"]:
            T_g = self._T_oe(obj)
            dg = _pose_err(T_g, cl["T_object_eef_at_attach"])
            tr["gravity_drift"] = [round(dg[0], 5), round(dg[1], 5)]
            if dg[0] > cfg.retain_pos_m or dg[1] > cfg.retain_rot_rad:
                reasons.append("slipped_under_gravity")
            motions = Motions(w.env, w.backend, cfg=MotionsConfig(), robot=self.robot)
            motions.state.hold_current()
            motions.state.gripper_cmd[arm] = cl["gripper_cmd"]
            for leg in self.legs_for(task, rec, obj):
                r = self.run_leg(leg, rec, obj, motions, run)
                r["held"] = self.robot._ag_obj_in_hand.get(arm) is obj
                if r["held"]:
                    d = _pose_err(self._T_oe(obj), T_g)
                    r["drift"] = [round(d[0], 5), round(d[1], 5)]
                    r["slipped"] = d[0] > cfg.retain_pos_m or d[1] > cfg.retain_rot_rad
                legs.append(r)
                if r.get("error"):
                    reasons.append(f"leg_error:{r['name']}")
                    break
                if not r["held"]:
                    reasons.append(f"dropped_in_motion:{r['name']}")
                    break
                if r.get("slipped"):
                    reasons.append(f"slipped_in_motion:{r['name']}")
                    break
            n_exec = sum(1 for r in legs if r.get("executed"))
            if not reasons and n_exec < cfg.min_legs:
                reasons.append("held_motion_unreachable")
        fingers = self.robot.get_joint_positions()[self.robot.gripper_control_idx[arm]].tolist()
        tr.update({"verdict": "valid" if not reasons else "invalid", "reasons": reasons,
                   "ag": {"formed": bool(cl["attached"]), "attached_after_steps": cl.get("attached_after_steps"), "held_after_gravity": bool(cl["held_after_gravity"]),
                          "fail": cl.get("fail"), "q_contact": cl.get("q_contact")},
                   "closure": {"held_sampler": True, "advance_m_per_step": cfg.advance_m_per_step, "press_m": cfg.press_m, "fingers_final_m": [round(float(v), 5) for v in fingers]},
                   "legs": legs, "legs_executed": sum(1 for r in legs if r.get("executed")), "legs_unreachable": [r["name"] for r in legs if not r.get("executed")],
                   "forces": {"finger_obj": VT._summ(fo)}, "steps": k[0], "wall_s": round(time.time() - t0, 2)})
        if sa is not None:
            try:
                tr["sleep_assist"] = {"sleeps": sa.sleeps, "wakes": sa.wakes}
            except Exception:  # noqa: BLE001
                pass
        return tr


def run_model(world: VT.TabletopWorld, job: Dict[str, Any], model_dir: str, cfg: Optional[FloatingValidateConfig] = None,
              runner: Optional[FloatingRunner] = None) -> Dict[str, Any]:
    """Trials of one floating model at every hold of its held tasks; writes ``validation.json``, ``rests.json`` (no rests) and
    ``library.json`` (the grasps valid at every hold; ``rest_id`` None)."""
    cfg = FloatingValidateConfig() if cfg is None else cfg
    t0 = time.time()
    cand_doc = S.load_json(os.path.join(model_dir, "candidates.json"))
    if cand_doc is None:
        raise FileNotFoundError(f"{model_dir}: no candidates.json")
    by_name = {c["name"]: c for c in cand_doc["candidates"]}
    reps = [by_name[n] for n in cand_doc.get("representatives", []) if n in by_name][: cfg.max_trials]
    runner = runner if runner is not None else getattr(world, "_floating_runner", None)
    if runner is None or runner.cfg != cfg:  # one runner per world: the holds' IK is solved once for all the world's models
        runner = FloatingRunner(world, cfg)
        world._floating_runner = runner
    obj = world.load(job, model_dir)
    tasks = held_tasks_of(job)
    trials: List[Dict[str, Any]] = []
    holds: Dict[str, Any] = {}
    n_valid = 0
    try:
        runner.park_slab()
        for t in tasks:
            h = runner.hold_q(t)
            holds[t] = {"hold": h["hold"], "arm_q": h["arm_q"], "eef_in_world": {"position": h["eef"][0].tolist(), "orientation_xyzw": h["eef"][1].tolist()}}
        for c in reps:
            if n_valid >= cfg.max_valid:
                break
            per = []
            for t in tasks:
                try:
                    per.append(runner.trial(obj, c, t))
                except Exception as e:  # noqa: BLE001 - one trial's crash is that trial's verdict; the next candidate runs
                    per.append({"schema": S.TRIAL, "candidate": c["name"], "task": t, "verdict": "invalid", "reasons": ["trial_error"], "error": repr(e)[:300],
                                "traceback": traceback.format_exc()[-1500:]})
            trials += per
            n_valid += all(x.get("verdict") == "valid" for x in per)
    except Exception as e:  # noqa: BLE001
        trials.append({"error": repr(e), "traceback": traceback.format_exc()[-2000:]})
    finally:
        world.unload(obj)
    timing = {"model_s": round(time.time() - t0, 2), "trial_s": round(sum(t.get("wall_s", 0.0) for t in trials), 2)}
    S.save_json(os.path.join(model_dir, "rests.json"), {"schema": S.REST, "rests": {}, "note": "floating: no support, no rest"})
    S.save_json(os.path.join(model_dir, "validation.json"), {"schema": S.TRIAL, "validator": VALIDATOR_VERSION, "config": asdict(cfg), "tasks": tasks, "holds": holds,
                                                             "policy_key": world.policy.key(cfg.arm), "trials": trials, "timing": timing, "created": S.now()})
    lib = build_library(job, cand_doc, trials, tasks, world.policy, {**world.physics_profile, "slab": "parked (floating: no support)"})
    S.save_json(os.path.join(model_dir, "library.json"), lib)
    return {"id": job.get("id"), "n_reps": len(reps), "n_valid": len(lib["grasps"]), "tasks": tasks, "timing": timing,
            "reasons": [t.get("reasons") or t.get("error") for t in trials if t.get("verdict") != "valid"]}


def build_library(job, cand_doc, trials, tasks: Sequence[str], policy: CP.ContactPolicy, physics_profile) -> Dict[str, Any]:
    """The library record of a floating entry: the representatives valid at EVERY hold of its tasks, in candidate order, ``rest_id``
    None, ``rests`` empty, ``constraint`` floating."""
    by_name = {c["name"]: c for c in cand_doc["candidates"]}
    per: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for t in trials:
        if "candidate" in t:
            per.setdefault(t["candidate"], {})[t.get("task")] = t
    grasps = []
    for name, by_task in per.items():
        if set(by_task) != set(tasks) or any(x.get("verdict") != "valid" for x in by_task.values()):
            continue
        c = by_name[name]
        first = by_task[sorted(by_task)[0]]
        grasps.append({"name": c["name"], "rest_id": None, "family_id": None, "mode": "floating", "floating_mode": c.get("floating_mode"),
                       "T_object_eef": c["T_object_eef"], "T_object_eef_at_attach": (first.get("achieved") or {}).get("T_object_eef_at_attach"),
                       "width_m": c["width_m"], "preshape_q": c["preshape_q"], "finger_q": first["closure"]["fingers_final_m"],
                       "graspgen_conf": c.get("graspgen_conf"), "physics": "tested",
                       "held_motions": {t: {"legs_executed": x["legs_executed"], "legs_unreachable": x["legs_unreachable"]} for t, x in by_task.items()}})
    arm = cand_doc.get("arm", "right")
    return {"schema": S.LIBRARY, "category": job["category"], "model": job["model"], "scale": list(S.canonical_scale(job.get("scale"))),
            "dataset": job.get("dataset"), "constraint": "floating", "legacy_key": job.get("legacy_key"), "arm": arm,
            "policy_key": policy.key(arm), "policy": policy.meta(arm), "physics_profile": physics_profile, "rests": {}, "families": {},
            "grasps": grasps, "held_tasks": list(tasks), "floating_mode": cand_doc.get("floating_mode"),
            "twins": "finger-swapped twins (eef turned pi about +z) are NOT stored",
            "provenance": {"candidates": cand_doc.get("pipeline"), "validator": VALIDATOR_VERSION, "graspgen": cand_doc.get("graspgen"), "created": S.now()}}
