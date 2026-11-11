"""Physics validation of FLOATING grasp candidates (:mod:`floating`; i7-floating 2026-11-07): the held object in hand, under gravity, then
through the held motions of its task - in the persistent validation world of :mod:`validate_tabletop` (the same simulator settings: 30 Hz
control / 120 Hz physics, the JoyLo R1Pro with assisted grasping, the evaluator's base mass, the task's AG rules - FixedJoint, pad-clear
release - the rigid sleep threshold and the sleep assist). Its slab stays parked: a floating object has no support.

Per job (``run_model``), per candidate and per held task of the job (``job["held_tasks"]``), the MOTIONS THAT TASK'S SAMPLER DRAWS
(owner 2026-11-07 item 4.3: match the task, not stricter), each its own closure:

1. **the task's hold** (``generation_defaults`` ``hold``: mental_rotation roll 15 deg, object_scaling roll 90 deg): the robot at the base
   origin, trunk ``hold.trunk``, the idle arm at the JoyLo reset (zeros), the holding arm at a cuRobo IK solution of the hold eef pose
   (``held_object.hold_eef_pose``);
2. **the drawn motions** (:meth:`FloatingRunner.draw_motions`, the sampler's own draws and its screening IK, base at the origin - the IK
   places every drawn motion where the arm reaches it): mental_rotation - starts = the hold's object turned about its root by
   ``draw_rotation_starts`` (axis uniform, ``init_angle_range_deg``), the start configuration by batch IK from the hold; goals = the start
   turned by ``draw_rotation_goals`` (``goal_angle_range_deg``, signed), kept when the batch IK reaches its 3 intermediate key poses + the
   goal (``orbit_key_poses``) from the start; object_scaling - the grasp's depth window (``depth_grid`` over ``depth_search_range_m``,
   the screening IK of the slid hand poses, ``depth_window`` with ``depth_margin_m``), then ``draw_start_goal`` (``min_depth_change_m``),
   the start configuration by IK, the line key poses (``line_key_poses``). ``motion_pairs`` motions per (candidate, task), seeded by the
   entry, the candidate and the task. Not applied: the sampler's in-frame / visibility checks (rendering) and its whole-body collision
   check at a scene base pose;
3. per motion: the robot at the START configuration, the object put into the fully open hand at the candidate's ``T_object_eef`` (gravity
   off), the HELD SAMPLER's closure (:func:`close_like_held_sampler` = ``HeldObjectV2.establish_grasp``), gravity, then the oracle's own
   tracking of the recorded key poses (``Motions.arm_along``, the ``Reorient`` executor's call) and ``leg_settle_steps`` of hold;
4. **acceptance** (the task's own rules): the object still attached after gravity and after the motion, its in-hand drift (from the
   post-gravity pose) within the TASK's goal tolerance - mental_rotation rotation <= ``tolerances.orientation`` (0.15 rad), object_scaling
   position <= ``depth_tolerance(dd)`` = max(0.02 m, 0.20 |dd|) - a drift the goal check would absorb is not a failure. Calibration on the
   collected held-object demos (``$OFH/floating_20261107/drift_demos.json``: 63 successful demos, largest in-hand drift 0.24 mm /
   4.4 mrad): every successful demo passes. A grasp is valid when at least ``min_executed`` drawn motions ran and none dropped or drifted
   past its tolerance; a closure that does not attach is recorded per motion (the sampler then tries another row); a motion the arm cannot
   track (``plan_failed``) is recorded as unreachable. The achieved-vs-candidate offset is recorded, not judged (the sampler records the
   realised grasp and the goal follows it).

Fidelity notes: base at the origin (the instance's base pose does not change the in-hand physics); the oracle's ``finish`` fallback of a
stalled leg is not used."""

from __future__ import annotations

import os
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ispatialgym.grasping import contact_policy as CP
from ispatialgym.grasping import schema as S
from ispatialgym.grasping import validate_tabletop as VT

VALIDATOR_VERSION = "isg-floating-validator/2"
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
    # the drawn motions (owner 2026-11-07 item 4.3): per (candidate, task) this many motions the task's sampler draws, from these many draws
    motion_pairs: int = 3
    start_draws: int = 64
    goal_draws: int = 48
    min_executed: int = 1
    leg_settle_steps: int = 10
    seed: int = 0
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


def drift_limits(task: str, delta_depth: float = 0.0) -> Dict[str, Optional[float]]:
    """The in-hand drift a held motion may leave: the task's goal tolerance (``generation_defaults`` ``tolerances``) - mental_rotation
    rotation <= ``orientation`` (position unconstrained: the target is the orientation), object_scaling position <= max(``depth_min_m``,
    ``depth_fraction`` |dd|) (rotation unconstrained: the target is the depth)."""
    from ispatialgym.sampling import generation_defaults as GD
    from ispatialgym.sampling.v2.tasks import held_object as HO

    tol = dict(GD.defaults_for(task)["tolerances"])
    if task == "mental_rotation":
        return {"pos_m": None, "rot_rad": float(tol.get("orientation", 0.15))}
    return {"pos_m": HO.depth_tolerance(delta_depth, tol), "rot_rad": None}


def drift_ok(drift: Tuple[float, float], limits: Dict[str, Optional[float]]) -> bool:
    return (limits["pos_m"] is None or drift[0] <= limits["pos_m"]) and (limits["rot_rad"] is None or drift[1] <= limits["rot_rad"])


def motion_rng(entry: str, cand: str, task: str, seed: int = 0) -> np.random.Generator:
    import hashlib

    h = int(hashlib.sha1(f"{entry}|{cand}|{task}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(np.random.SeedSequence([int(seed), h]))


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
        rec = {"task": task, "hold": hold.to_json(), "q": qh, "q_default": q.clone(), "eef": (pos.float(), quat.float()), "pad": pad.float(), "camera": cam,
               "optical_axis": axis.float(), "arm_q": [round(float(v), 5) for v in qh[self.robot.arm_control_idx[cfg.arm]].tolist()]}
        self.holds[task] = rec
        return rec

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

    def ik(self, poses, q_seed):
        """Batch IK of the holding arm (cuRobo, the robot alone): ``(ok list, q list)``."""
        from ispatialgym.oracle import r1pro_model as M

        if not poses:
            return [], []
        succ, qs = self.world.backend.solve_ik_batch(f"{self.cfg.arm}_arm", M.EEF_LINK[self.cfg.arm], [(p.float(), q.float()) for p, q in poses],
                                                     q_seed, batch_size=64, seeds=self.cfg.ik_seeds)
        return [bool(v) for v in succ.tolist()], [qs[i].clone() for i in range(len(poses))]

    def draw_motions(self, task: str, rec: Dict[str, Any], eef_in_object, rng: np.random.Generator) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Up to ``motion_pairs`` motions of ``task`` as its sampler draws them for this grasp (module docstring, 2.): ``[{"q0", "eef0",
        "obj0", "wps", "info"}]`` and the draw statistics."""
        from ispatialgym.robot import conventions as C
        from ispatialgym.sampling import generation_defaults as GD
        from ispatialgym.sampling.v2.tasks import held_object as HO

        cfg, th = self.cfg, self.th
        g = GD.defaults_for(task)
        n_wp = int(g.get("waypoints", 3))
        eef_hold = rec["eef"]
        obj_hold = C.compose(eef_hold, C.invert(eef_in_object))
        out: List[Dict[str, Any]] = []
        st: Dict[str, Any] = {}
        if task == "mental_rotation":
            starts = HO.draw_rotation_starts(rng, g["init_angle_range_deg"], cfg.start_draws)
            obj0s = [HO.turned_about_root(obj_hold, 0.0, ax, ang) for ax, ang in starts]
            eef0s = [C.compose(o, eef_in_object) for o in obj0s]
            ok, qs = self.ik(eef0s, rec["q"])
            st.update(starts=len(starts), starts_reached=sum(ok), goals_screened=0, goals_reached=0)
            for i in [k for k, v in enumerate(ok) if v]:
                goals = HO.draw_rotation_goals(rng, g["goal_angle_range_deg"], cfg.goal_draws)
                cands = []
                for ax, ang in goals:
                    eef1 = C.compose(HO.turned_about_root(obj0s[i], 0.0, ax, ang), eef_in_object)
                    dq = C.qmul(C.as_tensor(eef1[1]), C.qinv(C.as_tensor(eef0s[i][1])))
                    cands.append((ax, ang, HO.orbit_key_poses(eef0s[i], C.as_tensor(obj0s[i][0]), dq, n_wp)))
                okw, _ = self.ik([p for _, _, w in cands for p in w], qs[i])  # one batch, as the sampler's screening IK
                st["goals_screened"] += len(cands)
                for j, (ax, ang, wps) in enumerate(cands):
                    if all(okw[j * (n_wp + 1):(j + 1) * (n_wp + 1)]):
                        st["goals_reached"] += 1
                        out.append({"q0": qs[i], "eef0": eef0s[i], "obj0": obj0s[i], "wps": wps,
                                    "info": {"start": {"axis_base": [round(float(v), 4) for v in starts[i][0].tolist()], "angle_deg": round(float(starts[i][1]), 2)},
                                             "goal": {"axis_base": [round(float(v), 4) for v in ax.tolist()], "angle_deg": round(float(ang), 2)}}, "delta_depth": 0.0})
                        break
                if len(out) >= cfg.motion_pairs:
                    break
            return out, st
        cam = rec["camera"]
        axis = rec["optical_axis"]
        cam_p = C.as_tensor(cam[0]).float()
        grid = HO.depth_grid(g["depth_search_range_m"], float(g["depth_search_grid_m"]))
        d_ref = float(th.dot(C.as_tensor(obj_hold[0]).float() - cam_p, axis))
        slid = [HO.slide_to_depth(eef_hold, eef_in_object, cam_p, axis, d) for d in grid]
        ok, _ = self.ik([e for e, _ in slid], rec["q_default"])
        window, run = HO.depth_window(grid, ok, d_ref, float(g["depth_margin_m"]))
        st.update(depths=len(grid), depths_reached=sum(ok), reference_depth=round(d_ref, 4), window=window, run=run)
        min_change = float(g["min_depth_change_m"])
        if window is None or window[1] - window[0] <= min_change:
            return out, st
        for _ in range(cfg.motion_pairs * 4):
            d0, d1 = HO.draw_start_goal(rng, window, min_change)
            eef0, obj0 = HO.slide_to_depth(eef_hold, eef_in_object, cam_p, axis, d0)
            eef1, _ = HO.slide_to_depth(eef_hold, eef_in_object, cam_p, axis, d1)
            ok0, q0 = self.ik([eef0], rec["q_default"])
            if not ok0[0]:
                continue
            out.append({"q0": q0[0], "eef0": eef0, "obj0": obj0, "wps": HO.line_key_poses(eef0, eef1, n_wp),
                        "info": {"start_depth_m": round(d0, 4), "goal_depth_m": round(d1, 4)}, "delta_depth": d1 - d0})
            if len(out) >= cfg.motion_pairs:
                break
        return out, st

    def set_start(self, q0, obj):
        """The robot at a motion's start configuration (fingers fully open), the object parked, gravity off."""
        for arm in ("left", "right"):
            try:
                self.robot.release_grasp_immediately(arm)
            except Exception:  # noqa: BLE001
                pass
        th = self.th
        obj.disable_gravity()
        obj.set_position_orientation(th.tensor([3.0, -3.0, 1.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
        obj.keep_still()
        q = q0.clone()
        q[self.robot.gripper_control_idx[self.cfg.arm]] = 0.05
        self.robot.set_joint_positions(q)
        self.robot.keep_still()
        a = self.world.hold_action()
        for _ in range(4):
            self.world.step(a)

    def run_motion(self, obj, cand: Dict[str, Any], m: Dict[str, Any], task: str) -> Dict[str, Any]:
        """One drawn motion: the start, the held sampler's closure there, gravity, the oracle's tracking of its key poses, the judgement."""
        from ispatialgym.oracle.motions import Motions, MotionsConfig, SkillError
        from ispatialgym.robot import conventions as C

        cfg, w, arm, th = self.cfg, self.world, self.cfg.arm, self.th
        t0 = time.time()
        self.set_start(m["q0"], obj)
        ep, eq = self.robot.eef_links[arm].get_position_orientation()
        T_we = S.pose_mat(ep.numpy(), eq.numpy())
        start_err = _pose_err(T_we, S.pose_mat(C.as_tensor(m["eef0"][0]).numpy(), C.as_tensor(m["eef0"][1]).numpy()))
        T_c = S.pose_of(cand["T_object_eef"])
        T_wo = T_we @ np.linalg.inv(T_c)
        pose = (T_wo[:3, 3].astype(np.float32), S.mat_to_quat(T_wo[:3, :3]).astype(np.float32))
        r: Dict[str, Any] = {**m["info"], "start_eef_err": [round(start_err[0], 5), round(start_err[1], 5)]}
        limits = drift_limits(task, m.get("delta_depth", 0.0))
        r["drift_limits"] = limits
        obj.wake()
        sa = w.SA.SleepAssist(self.robot, {obj.name: obj}, "gbf") if cfg.sleep_assist else None
        k = [0]
        fo: List[float] = []

        def step(a):
            f = w.step(a, sa, k[0])
            k[0] += 1
            fo.append(f.get("finger_obj", 0.0))

        cl = self.close_like_held_sampler(obj, pose, step)
        r["ag"] = {"formed": bool(cl["attached"]), "attached_after_steps": cl.get("attached_after_steps"), "held_after_gravity": bool(cl["held_after_gravity"]),
                   "fail": cl.get("fail"), "q_contact": cl.get("q_contact")}
        if cl["attached"]:
            off = _pose_err(cl["T_object_eef_at_attach"], T_c)
            r["achieved_off_target"] = [round(off[0], 5), round(off[1], 5)]
        if not cl["attached"]:
            r["outcome"] = "ag_not_formed"
        elif not cl["held_after_gravity"]:
            r["outcome"] = "dropped_under_gravity"
        else:
            T_g = self._T_oe(obj)
            motions = Motions(w.env, w.backend, cfg=MotionsConfig(), robot=self.robot)
            motions.state.hold_current()
            motions.state.gripper_cmd[arm] = cl["gripper_cmd"]
            try:
                for a in motions.arm_along(arm, m["wps"]):
                    step(a)
                r["executed"] = True
            except SkillError as e:
                r["executed"] = False
                r["unreachable"] = f"{getattr(e, 'reason', 'skill_error')}: {str(e)[:160]}"
            except Exception as e:  # noqa: BLE001
                r["executed"] = False
                r["error"] = repr(e)[:300]
                r["traceback"] = traceback.format_exc()[-1200:]
            for a in motions.hold(cfg.leg_settle_steps):
                step(a)
            now = motions.eef_pose(arm)
            goal = m["wps"][-1]
            r["eef_err_m"] = round(float(th.linalg.norm(now[0] - C.as_tensor(goal[0]))), 5)
            r["eef_err_rad"] = round(float(C.quat_angle(now[1], C.as_tensor(goal[1]))), 5)
            held = self.robot._ag_obj_in_hand.get(arm) is obj
            r["held"] = held
            if held:
                d = _pose_err(self._T_oe(obj), T_g)
                r["drift"] = [round(d[0], 6), round(d[1], 6)]
            if r.get("error"):
                r["outcome"] = "error"
            elif not held:
                r["outcome"] = "dropped_in_motion"
            elif not drift_ok(tuple(r["drift"]), limits):
                r["outcome"] = "drift_beyond_task_tolerance"
            else:
                r["outcome"] = "ok" if r["executed"] else "unreachable"
        r["forces"] = {"finger_obj": VT._summ(fo)}
        r["steps"] = k[0]
        r["wall_s"] = round(time.time() - t0, 2)
        if sa is not None:
            try:
                r["sleep_assist"] = {"sleeps": sa.sleeps, "wakes": sa.wakes}
            except Exception:  # noqa: BLE001
                pass
        return r

    # ------------------------------------------------------------------ one trial

    def trial(self, obj, cand: Dict[str, Any], task: str, entry: str = "") -> Dict[str, Any]:
        """The candidate under ``task``'s drawn motions (module docstring); the verdict from the motions' outcomes (:func:`verdict_of`)."""
        from ispatialgym.robot import conventions as C

        t0 = time.time()
        rec = self.hold_q(task)
        T_c = S.pose_of(cand["T_object_eef"])
        pose = S.mat_pose(T_c)
        eef_in_object = (C.as_tensor(pose["position"]).float(), C.as_tensor(pose["orientation_xyzw"]).float())
        rng = motion_rng(entry, cand["name"], task, self.cfg.seed)
        drawn, st = self.draw_motions(task, rec, eef_in_object, rng)
        motions = [self.run_motion(obj, cand, m, task) for m in drawn]
        verdict, reasons = verdict_of(motions, self.cfg.min_executed)
        fingers = [m.get("ag", {}).get("q_contact") for m in motions]
        return {"schema": S.TRIAL, "candidate": cand["name"], "family_id": None, "rest_id": None, "mode": "floating", "task": task, "hold": rec["hold"],
                "hold_arm_q": rec["arm_q"], "draws": st, "motions": motions, "motions_tested": [m.get("info") for m in drawn],
                "motions_executed": sum(1 for m in motions if m.get("outcome") == "ok"), "verdict": verdict, "reasons": reasons,
                "closure": {"held_sampler": True, "advance_m_per_step": self.cfg.advance_m_per_step, "press_m": self.cfg.press_m,
                            "fingers_final_m": next((f for f in fingers if f), [])}, "wall_s": round(time.time() - t0, 2)}


def verdict_of(motions: Sequence[Dict[str, Any]], min_executed: int = 1) -> Tuple[str, List[str]]:
    """``("valid" | "invalid", reasons)``: invalid when any motion dropped the object, drifted past its task tolerance or crashed, or fewer
    than ``min_executed`` ran (``no_drawn_motion`` when the sampler's draws found none, ``ag_not_formed`` when no closure attached,
    ``held_motion_unreachable`` when the arm tracked none); a closure that does not attach in some of the motions is not a reason (the
    sampler retries another row - recorded per motion)."""
    bad = [m["outcome"] for m in motions if m.get("outcome") in ("dropped_under_gravity", "dropped_in_motion", "drift_beyond_task_tolerance", "error")]
    if bad:
        return "invalid", sorted(set(bad))
    n_ok = sum(1 for m in motions if m.get("outcome") == "ok")
    if n_ok >= min_executed:
        return "valid", []
    if not motions:
        return "invalid", ["no_drawn_motion"]
    if all(m.get("outcome") == "ag_not_formed" for m in motions):
        return "invalid", ["ag_not_formed"]
    return "invalid", ["held_motion_unreachable"]


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
                    per.append(runner.trial(obj, c, t, entry=job.get("id", "")))
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
                       "held_motions": {t: {"motions_executed": x.get("motions_executed"), "motions_tested": x.get("motions_tested")}
                                        for t, x in by_task.items()}})
    arm = cand_doc.get("arm", "right")
    return {"schema": S.LIBRARY, "category": job["category"], "model": job["model"], "scale": list(S.canonical_scale(job.get("scale"))),
            "dataset": job.get("dataset"), "constraint": "floating", "legacy_key": job.get("legacy_key"), "arm": arm,
            "policy_key": policy.key(arm), "policy": policy.meta(arm), "physics_profile": physics_profile, "rests": {}, "families": {},
            "grasps": grasps, "held_tasks": list(tasks), "floating_mode": cand_doc.get("floating_mode"),
            **({"candidate_switches": S.candidate_switches(cand_doc.get("config"))}
               if cand_doc.get("config", {}).get("outer_pinch") is True else {}),
            "twins": "finger-swapped twins (eef turned pi about +z) are NOT stored",
            "provenance": {"candidates": cand_doc.get("pipeline"), "validator": VALIDATOR_VERSION, "graspgen": cand_doc.get("graspgen"), "created": S.now()}}
