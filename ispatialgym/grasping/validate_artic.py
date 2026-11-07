"""Tabletop articulation validation in the persistent validation world (:class:`validate_tabletop.TabletopWorld`: robot, slab, cuRobo
backend; objects come and go - no scene or simulator reload per model or trial).

Per (model, joint, handle candidate):

1. **handles**: the handle records (``runlogs/grasp_library_20260905/artic/handles/<key>.json``: contact, approach, pinch axis, width, in
   the object frame with the joint closed; manual overrides of ``manual_handles.json``) whose pinch width is within the selection range
   ``PINCH_RANGE_M`` = 3-80 mm (``articulation_large.PINCH_WIDTH_RANGE``); every distinct handle / edge is a candidate - an AG that
   formed on a trial whose pull then failed does not end the search;
2. **physics**: the joint's profile (:mod:`artic_profile`: recorded demo settings, else a bounded brake calibration ONCE per model /
   scale / joint, cached) applied before the rest snapshot; the body free-standing as in the demos (``fixed_base`` only for the
   labelled diagnostic mode); the assisted grasp's FixedJoint rule on moving links (the task's environment rules);
3. **stance**: the demo convention (``articulation_small`` ``lid_facing: hinge_right``: the hinge line along the robot's line of sight
   on its right) ranks the base yaws; the grasp configuration by batched IK, the whole robot under the contact policy against the slab;
   the hand starts at the handle pose with the fingers pre-shaped (no approach executed); the demo closure (``Motions.gripper``);
4. **pull**: the demo's arm-mode follow - the hand turned rigidly about the LIVE hinge (the body may be free-standing) in
   ``follow_arc_step_rad`` steps, tracked by ``Motions.arm_along`` in chunks of ``follow_reaim_chunk`` (``execute._follow_handle_poses``);
   stop on the joint limit (within ``limit_tol_rad``), a stall (joint progress below ``stall_rad`` over ``stall_steps`` while
   commanded), robot wobble (torso error above ``wobble_rad`` for ``wobble_steps``), the grasp lost, or the tracking failing;
5. **sequential hybrid** when the arm alone stops short of the limit for reach (tracking failure / stall at the reach boundary):
   release, retreat along the approach, a new base pose from the stance search for the remaining arc, regrasp at the handle's current
   pose, pull again - SEQUENTIAL segments, never a coordinated base-and-arm pull; every segment and transition is labelled.

Records (schema ``ARTIC_TRIAL``): the eef trajectory (t, world pose, the object-relative pose anchored to the initial settled root),
the joint angle over time, base pose, torso error, segment labels, the achieved range (specific to this grasp / stance / controller -
not the object's global limit), candidate, stance, profile, stop reason.

    python -m ispatialgym.grasping.validate_artic --keys blender__eyedvd blender__cwkvib --root ROOT

i7-artic 2026-11-07: ``rest`` / ``motions`` per ALIAS - the oracle's own small-articulation motions (``artic_motions``: the pull-open
prefix, v2 / v4's main pinch at q_ajar, the push-close tip contact), physics-validated, written as ``artic_library__<alias>.json`` for the
articulation prefix maps (``reach_maps prefix``); see the section above :func:`main`::

    python -m ispatialgym.grasping.validate_artic rest    --aliases ALIAS [...] --root ROOT
    python -m ispatialgym.grasping.validate_artic motions --aliases ALIAS [...] --root ROOT
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ispatialgym import REPO_ROOT
from ispatialgym.grasping import artic_profile as AP
from ispatialgym.grasping import contact_policy as CP
from ispatialgym.grasping import schema as S

HANDLES = os.path.join(REPO_ROOT, "runlogs", "grasp_library_20260905", "artic", "handles")
MANUAL = os.path.join(REPO_ROOT, "runlogs", "grasp_library_20260905", "artic", "manual_handles.json")
VALIDATOR_VERSION = "isg-artic-validator/1"


@dataclass
class ArticConfig:
    pinch_range_m: Tuple[float, float] = (0.003, 0.080)
    follow_arc_step_rad: float = 0.03   # execute.ExecuteConfig.follow_arc_step_rad
    follow_reaim_chunk: int = 4         # execute.ExecuteConfig.follow_reaim_chunk
    limit_tol_rad: float = 0.05
    stall_rad: float = 0.01
    stall_steps: int = 30
    wobble_rad: float = math.radians(1.0)
    wobble_steps: int = 10
    max_segments: int = 3
    retreat_m: float = 0.08
    calib_fracs: Tuple[float, ...] = (0.25, 0.5, 0.75)
    calib_hold_steps: int = 30
    calib_drift_rad: float = 0.02
    fixed_base: bool = False            # diagnostic only: results are labelled "fixed_base"


def handle_candidates(key: str, cfg: ArticConfig) -> List[Dict[str, Any]]:
    d = S.load_json(os.path.join(HANDLES, f"{key}.json")) or {}
    manual = (S.load_json(MANUAL) or {}).get(key, {})
    out = []
    for j in d.get("joints", []):
        m = manual.get(j["joint"], {})
        if m.get("skip"):
            continue
        for i, h in enumerate(j.get("handles") or []):
            w = h.get("width")
            if w is None or not (cfg.pinch_range_m[0] <= float(w) <= cfg.pinch_range_m[1]):
                continue
            out.append({"key": key, "joint": j, "handle": h, "id": f"{j['joint']}#{i}", "manual": m or None})
    return out


def handle_eef_obj(h: Dict[str, Any]) -> np.ndarray:
    """The eef pose (object frame) pinching the handle: ``conventions.handle_eef`` (pad centre on the contact, approach, closing axis)."""
    import torch as th

    from ispatialgym.robot import conventions as C

    p, q = C.handle_eef(th.tensor(h["contact"], dtype=th.float32), th.tensor(h["approach_dir"], dtype=th.float32), th.tensor(h["pinch_axis"], dtype=th.float32))
    return S.pose_mat(p.numpy(), q.numpy())


class ArticWorld:
    """Articulation trials on a :class:`TabletopWorld` (shared simulator, robot, slab, backend)."""

    def __init__(self, world, cfg: ArticConfig = ArticConfig()):
        self.w, self.cfg = world, cfg

    def joint_q(self, obj, joint: str) -> float:
        j = obj.joints[joint]
        return float(j.get_state()[0][0])

    def hinge_world(self, obj, jrec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        p, q = obj.get_position_orientation()
        T = S.pose_mat(p.numpy(), q.numpy())
        c = T[:3, :3] @ np.asarray(jrec["hinge_point_obj"]) + T[:3, 3]
        a = T[:3, :3] @ np.asarray(jrec["axis_dir_obj"], dtype=float)
        return c, a / np.linalg.norm(a)

    def calibrate_brake(self, obj, jrec: Dict[str, Any], state):
        """The smallest ladder brake that holds every probe angle (``calib_fracs`` of the range) within ``calib_drift_rad`` over
        ``calib_hold_steps`` with no robot contact - the sampler's stable-angle criterion (``articulation_small._stable_angles``)."""
        from ispatialgym.sampling import constraints as CN

        cfg = self.cfg
        lo, hi = float(jrec["lower"]), float(jrec["upper"])
        j = obj.joints[jrec["joint"]]

        def run(ladder):
            probe = []
            for spec in ladder:
                CN.apply_joint_brake(j, CN.joint_brake_spec(float(spec.split(":", 1)[1])))
                ok_all = True
                for f in cfg.calib_fracs:
                    self.w.scene.load_state(state, serialized=False)
                    q0 = lo + f * (hi - lo)
                    j.set_pos(q0)
                    a = self.w.hold_action()
                    for _ in range(cfg.calib_hold_steps):
                        self.w.step(a)
                    drift = abs(self.joint_q(obj, jrec["joint"]) - q0)
                    probe.append({"spec": spec, "frac": f, "drift_rad": round(drift, 4)})
                    ok_all &= drift <= cfg.calib_drift_rad
                if ok_all:
                    return spec, probe
            return ladder[-1], probe + [{"note": "no ladder value held every probe angle: the strongest is used"}]
        return run

    def trial(self, obj, cand: Dict[str, Any], state, rest: Dict[str, Any], geom) -> Dict[str, Any]:
        import torch as th

        from ispatialgym.oracle import execute as EX
        from ispatialgym.oracle.motions import Motions, MotionsConfig
        from ispatialgym.grasping import validate_tabletop as VT

        w, cfg, arm = self.w, self.cfg, self.w.cfg.arm
        t0 = time.time()
        jrec, h = cand["joint"], cand["handle"]
        rec: Dict[str, Any] = {"schema": S.ARTIC_TRIAL, "candidate": cand["id"], "joint": jrec["joint"], "handle_kind": h.get("kind"),
                               "width_m": h.get("width"), "fixed_base": cfg.fixed_base, "segments": [], "trajectory": []}
        rec.update(w.restore(obj, state, rest))
        p0, q0 = obj.get_position_orientation()
        T_wo0 = S.pose_mat(p0.numpy(), q0.numpy())
        T_oe = handle_eef_obj(h)
        cand_g = {"name": cand["id"], "T_object_eef": S.mat_pose(T_oe), "preshape_q": min(0.05, 0.5 * float(h["width"]) + 0.010),
                  "contact_q": 0.5 * float(h["width"]) + 0.0009, "width_m": float(h["width"])}
        plans = w.plan(obj, geom, [cand_g], (p0.numpy(), q0.numpy()))
        pl = plans.get(cand["id"])
        if pl is None or pl.get("q") is None:
            rec.update({"stop": "no_placement", "fail": None if pl is None else pl.get("fail")})
            return rec
        rec["stance"] = {"base": pl["base"], "trunk": pl["trunk"]}
        q = pl["q"].clone()
        q[w.robot.gripper_control_idx[arm]] = cand_g["preshape_q"]
        w.robot.set_joint_positions(q)
        w.robot.keep_still()
        mcfg = MotionsConfig()
        motions = Motions(w.env, w.backend, cfg=mcfg, robot=w.robot)
        motions.state.hold_current()
        motions.state.gripper_cmd[arm] = motions.gripper_command(arm, cand_g["preshape_q"])
        obj.wake()
        k = [0]
        jq0 = self.joint_q(obj, jrec["joint"])
        upper = float(jrec["upper"]) if int(jrec.get("open_sign", 1)) > 0 else float(jrec["lower"])

        def step_all(gen, label):
            for a in gen:
                w.step(a)
                k[0] += 1
                ep, eq = (x.numpy() for x in w.robot.eef_links[arm].get_position_orientation())
                Te = S.pose_mat(ep, eq)
                bp, bq = w.robot.get_position_orientation()
                rec["trajectory"].append({"step": k[0], "t_s": round(k[0] / 30.0, 4), "segment": label, "q": round(self.joint_q(obj, jrec["joint"]), 5),
                                          "eef_world": S.mat_pose(Te), "eef_in_initial_object": S.mat_pose(np.linalg.inv(T_wo0) @ Te),
                                          "base": [round(float(bp[0]), 4), round(float(bp[1]), 4), round(S.yaw_of_quat(bq.numpy()), 4)],
                                          "torso_err_rad": round(float(motions.state.trunk_error()), 5) if hasattr(motions.state, "trunk_error") else None,
                                          "held": w.robot._ag_obj_in_hand.get(arm) is obj})
        rate, squeeze, _ = EX.closure_for_width(float(h["width"]), mcfg, library=False)
        step_all(motions.hold(2), "init")
        step_all(motions.gripper(arm, True, rate=rate, squeeze=squeeze), "close")
        if w.robot._ag_obj_in_hand.get(arm) is not obj:
            rec.update({"stop": "ag_not_formed", "achieved_rad": 0.0, "wall_s": round(time.time() - t0, 2)})
            return rec
        stop = None
        for seg in range(cfg.max_segments):
            stop = None
            start_q = self.joint_q(obj, jrec["joint"])
            rec["segments"].append({"kind": "arm_pull", "index": seg, "start_step": k[0], "start_q": round(start_q, 4)})
            centre, axis = self.hinge_world(obj, jrec)
            ep, eq = (x.numpy() for x in w.robot.eef_links[arm].get_position_orientation())
            remaining = upper - start_q
            n = max(1, int(math.ceil(abs(remaining) / cfg.follow_arc_step_rad)))
            sense = float(np.sign(remaining)) * float(jrec.get("open_sign", 1))
            T0 = S.pose_mat(ep, eq)
            dense = []
            for i in range(1, n + 1):
                R = _axis_angle(axis, sense * abs(remaining) * i / n)
                Ti = np.eye(4)
                Ti[:3, :3] = R @ T0[:3, :3]
                Ti[:3, 3] = centre + R @ (T0[:3, 3] - centre)
                dense.append((th.as_tensor(Ti[:3, 3], dtype=th.float32), th.as_tensor(S.mat_to_quat(Ti[:3, :3]), dtype=th.float32)))
            stalled = wob = 0
            last_q = start_q
            for c0 in range(0, len(dense), cfg.follow_reaim_chunk):
                try:
                    step_all(motions.arm_along(arm, dense[c0:c0 + cfg.follow_reaim_chunk], converge=False), f"pull{seg}")
                except Exception as e:  # noqa: BLE001 - the demo tracker gave up (reach / IK)
                    stop = f"tracking_failed: {type(e).__name__}: {str(e)[:160]}"
                    break
                qn = self.joint_q(obj, jrec["joint"])
                stalled = stalled + cfg.follow_reaim_chunk if abs(qn - last_q) < cfg.stall_rad else 0
                last_q = qn
                te = rec["trajectory"][-1].get("torso_err_rad") or 0.0
                wob = wob + 1 if te > cfg.wobble_rad else 0
                if w.robot._ag_obj_in_hand.get(arm) is not obj:
                    stop = "grasp_lost"
                    break
                if abs(upper - qn) <= cfg.limit_tol_rad:
                    stop = "limit_reached"
                    break
                if stalled >= cfg.stall_steps:
                    stop = "stalled"
                    break
                if wob >= cfg.wobble_steps:
                    stop = "robot_wobble"
                    break
            rec["segments"][-1].update({"end_step": k[0], "end_q": round(self.joint_q(obj, jrec["joint"]), 4), "stop": stop or "arc_end"})
            if abs(upper - self.joint_q(obj, jrec["joint"])) <= cfg.limit_tol_rad:
                stop = "limit_reached"
                break
            if stop in ("grasp_lost", "robot_wobble"):
                break
            if seg + 1 >= cfg.max_segments:
                break
            # sequential hybrid: release, retreat, re-stance for the remaining arc, regrasp, pull again
            rec["segments"].append({"kind": "release", "start_step": k[0]})
            step_all(motions.gripper(arm, False), "release")
            EXr = [0]
            for _ in range(30):
                if w.robot._ag_obj_in_hand.get(arm) is None:
                    break
                step_all(motions.hold(1), "release")
                EXr[0] += 1
            ep, eq = (x.numpy() for x in w.robot.eef_links[arm].get_position_orientation())
            Te = S.pose_mat(ep, eq)
            back = Te.copy()
            back[:3, 3] -= cfg.retreat_m * Te[:3, 2]
            rec["segments"].append({"kind": "retreat", "start_step": k[0]})
            try:
                step_all(motions.arm_along(arm, [(th.as_tensor(back[:3, 3], dtype=th.float32), th.as_tensor(S.mat_to_quat(back[:3, :3]), dtype=th.float32))], converge=False), "retreat")
            except Exception:  # noqa: BLE001
                pass
            # the handle's current pose -> a new stance by the same batched-IK stance search, then the base moved there
            p1, q1 = obj.get_position_orientation()
            T_wo1 = S.pose_mat(p1.numpy(), q1.numpy())
            cur = dict(cand_g)
            # the handle on the moved link: rotate the closed-pose handle eef about the hinge by the joint's displacement
            dq = self.joint_q(obj, jrec["joint"]) - jq0
            R = _axis_angle(np.asarray(jrec["axis_dir_obj"], dtype=float), dq * float(jrec.get("open_sign", 1)))
            hp = np.asarray(jrec["hinge_point_obj"], dtype=float)
            T_oe1 = np.eye(4)
            T_oe1[:3, :3] = R @ T_oe[:3, :3]
            T_oe1[:3, 3] = hp + R @ (T_oe[:3, 3] - hp)
            cur["T_object_eef"] = S.mat_pose(T_oe1)
            plans = w.plan(obj, geom, [cur], (p1.numpy(), q1.numpy()))
            pl2 = plans.get(cand["id"])
            if pl2 is None or pl2.get("q") is None:
                stop = "no_regrasp_stance"
                break
            rec["segments"].append({"kind": "base_move", "start_step": k[0], "to": pl2["base"], "method": "teleport_between_segments",
                                    "note": "the base pose is set between pull segments (the object released, the hand retreated); the base drive itself is not simulated"})
            qq = pl2["q"].clone()
            qq[w.robot.gripper_control_idx[arm]] = cand_g["preshape_q"]
            w.robot.set_joint_positions(qq)
            w.robot.keep_still()
            motions = Motions(w.env, w.backend, cfg=mcfg, robot=w.robot)
            motions.state.hold_current()
            motions.state.gripper_cmd[arm] = motions.gripper_command(arm, cand_g["preshape_q"])
            rec["segments"].append({"kind": "regrasp", "start_step": k[0]})
            step_all(motions.hold(2), "regrasp")
            step_all(motions.gripper(arm, True, rate=rate, squeeze=squeeze), "regrasp")
            if w.robot._ag_obj_in_hand.get(arm) is not obj:
                stop = "regrasp_failed"
                break
        jq = self.joint_q(obj, jrec["joint"])
        rec.update({"stop": stop or "arc_end", "start_q": round(jq0, 5), "final_q": round(jq, 5), "achieved_rad": round(abs(jq - jq0), 5),
                    "range_rad": round(abs(upper - jq0), 5), "achieved_fraction": round(abs(jq - jq0) / max(abs(upper - jq0), 1e-6), 4),
                    "achieved_scope": "this grasp / stance / controller only - not the object's global limit",
                    "steps": k[0], "wall_s": round(time.time() - t0, 2),
                    "frames": {"eef_world": "world", "eef_in_initial_object": "the object root as settled before the trial (rest snapshot)"}})
        return rec


def _axis_angle(axis, ang: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def run_key(world, aw: ArticWorld, key: str, root: str) -> Dict[str, Any]:
    from ispatialgym.grasping import object_geometry as OG

    cat, model = key.split("__")[:2]
    job = {"category": cat, "model": model, "scale": None, "dataset": "behavior-1k-assets", "id": S.entry_id(cat, model, None, "articulation")}
    d = S.model_dir(cat, model, None, root, "articulation")
    os.makedirs(d, exist_ok=True)
    cands = handle_candidates(key, aw.cfg)
    out: Dict[str, Any] = {"key": key, "candidates": [c["id"] for c in cands], "trials": [], "profiles": {}}
    if not cands:
        out["error"] = "no handle within the 3-80 mm pinch range"
        return out
    t0 = time.time()
    obj = world.load(job, d)
    try:
        geom = OG.load(d)
        rest, state = world.settle(obj, geom, np.array([0.0, 0.0, 0.0, 1.0]))
        rest["rest_id"] = "r0"
        for c in cands:
            jrec = c["joint"]
            if jrec["joint"] not in out["profiles"]:
                prof = AP.resolve(cat, model, None, jrec, fixed_base=aw.cfg.fixed_base, aabb_extent=[float(v) for v in obj.aabb_extent], root=root,
                                  calibrate=aw.calibrate_brake(obj, jrec, state))
                out["profiles"][jrec["joint"]] = {"applied": AP.apply(obj, prof), "source": prof.get("source_used"), "kind": prof["kind"],
                                                  "brake": prof.get("brake"), "provenance": prof.get("provenance")}
                rest, state = world.settle(obj, geom, np.array([0.0, 0.0, 0.0, 1.0]))  # the snapshot after the USD edits
                rest["rest_id"] = "r0"
            try:
                tr = aw.trial(obj, c, state, rest, geom)
            except Exception as e:  # noqa: BLE001
                tr = {"candidate": c["id"], "stop": "error", "error": repr(e), "traceback": traceback.format_exc()[-1500:]}
            out["trials"].append(tr)
    finally:
        world.unload(obj)
    out["wall_s"] = round(time.time() - t0, 2)
    S.save_json(os.path.join(d, "artic_validation.json"), {"schema": S.ARTIC_TRIAL, "validator": VALIDATOR_VERSION, "config": asdict(aw.cfg), **out})
    # what the articulation prefix maps consume (reach_maps.compute_prefix_map): the measured rest and, per trial, its FIRST fixed-stance
    # pull segment as object-relative eef waypoints with the joint value (a hybrid demonstration is never one fixed-base trajectory)
    segs = []
    for tr in out["trials"]:
        traj = tr.get("trajectory") or []
        first = [x for x in traj if x.get("segment") == "pull0"]
        if not first:
            continue
        segs.append({"candidate": tr["candidate"], "joint": tr.get("joint"), "stop": tr.get("stop"), "profile": out["profiles"].get(tr.get("joint")),
                     "waypoints": [{"q": x["q"], "eef_in_object": x["eef_in_initial_object"]} for x in first],
                     "start_q": tr.get("start_q"), "achieved_rad_trial": tr.get("achieved_rad"), "segments": [g.get("kind") for g in tr.get("segments", [])]})
    S.save_json(os.path.join(d, "artic_library.json"), {"schema": "isg-artic-library/1", "category": cat, "model": model, "scale": [1.0, 1.0, 1.0],
                                                        "rests": {"r0": rest} if "rest" in locals() else {}, "segments": segs,
                                                        "frames": "eef_in_object: the eef in the object root frame as settled before the trial"})
    return out


# ------------------------------------------------------------------------------------------------ the oracle's motions per alias
#
# i7-artic 2026-11-07 (owner decision: handles stay with the ARTICULATION library; validate_artic extended, physics-validated, to the motions
# the oracle executes for articulation_open_small_01 / 02 and articulation_close_small_01 / 02): ``rest`` and ``motions`` (subcommands).
#
#   python -m ispatialgym.grasping.validate_artic rest    --aliases ALIAS [...] --root ROOT
#   python -m ispatialgym.grasping.validate_artic motions --aliases ALIAS [...] --root ROOT [--max-trials N] [--any-yaw]
#
# Per alias (``artic_motions.build``: the sampler's own handle code on the alias's customs entry):
#
# 1. **profile** - the sampler's rule (``articulation_small.sample_template``): the alias's ``joint_friction`` candidates in order, each applied
#    as the brake and probed for the stable set S (``open_fraction_range`` fractions + the upper limit, held ``stable_hold_steps`` physics steps,
#    drift <= ``stable_drift_tolerance_fraction`` x the probe tolerance; the object standing on the slab, the robot parked), the first that buys an
#    open value below the upper limit kept; then the alias's ``joint_drives`` of the OTHER joints (the carton's left-flap spring, its extended
#    lower limit with ``restrict_openable_joints``);
# 2. **rest** - the object dropped from the sampler's placement rest (identity, turned to the task's face yaw) at ``rest_drops_m`` (the 3 mm
#    sampler drop first) with the profile applied: stable when every outcome is still with support contact (``TabletopWorld.settle``), within
#    ``rest_tilt_tol_rad`` of the intended rest and the same rest as the first (``schema.same_rest``); recorded yaw-free;
# 3. **trials** - per handle variant (both finger assignments / fist turns): the start state (the target joint at q_start, every other joint
#    closed; held ``start_hold_steps`` with the robot parked: the brake must hold it within ``start_drift_rad``), the stance by the validation
#    world's batched IK with the object at the task's face yaw relative to the base (``ValidateConfig(base_yaws=1)``; ``--any-yaw``: 8 base
#    yaws), the hand at the PRE-grasp / pre-contact (``conventions.handle_waypoints`` 10 cm back; else at the contact, labelled), the oracle's
#    approach (``Motions.arm_along``), then
#    * a pull: the oracle's pre-shape (pinch / 2 + 1 cm; v2: + ``v2_second_preshape_margin_m``), the closure (``closure_for_width``), the
#      assisted grasp must form, the hand turned rigidly about the LIVE hinge toward the upper limit (``follow_arc_step_rad`` steps, streamed
#      in ``follow_reaim_chunk`` chunks - the ``BaseFollow`` arm mode); one fixed-stance segment;
#    * a push: the closed fist, the approach ``loaded`` into the tip, the fist turned about the live hinge toward the lower limit + the
#      alias's ``close_push_overshoot_rad`` (``loaded`` legs, as ``execute._push_leg`` tracks an articulated push); one fixed-stance segment;
#    stopped on the goal (within ``limit_tol_rad``), a stall, robot wobble, tracking failure, and - ending the VALIDATED part - the grasp lost
#    (pull), the fist ``contact_lost_m`` off its contact pose at the live joint value (push), or the body moved (root ``body_shift_m`` /
#    ``body_rot_rad`` from the start: the free-standing body is never held by anything but its own weight here);
# 4. **library** - ``artic_library__<alias>.json`` (schema ``isg-artic-library/2``): per passing trial (validated travel >= ``min_travel_rad``)
#    one segment = the nominal contact pose at the held start q, then the measured eef poses of the motion segment up to the last validated
#    step (object-relative to the root at the motion start; resampled every ``wp_step_rad`` of joint travel / ``wp_step_m`` of hand travel),
#    with its joint values, finger openings, rest, face yaw, offered arms, handle signature and provenance - what
#    ``reach_maps prefix`` reads. The physics runs with the validation world's right arm; the segments are object-relative, so the prefix
#    maps of the left arm read the same segments (the map identity names the validated arm).

MOTIONS_VERSION = "isg-artic-motions/1"
LIBRARY_V2 = "isg-artic-library/2"
REST_SCHEMA = "isg-artic-rest/1"
#: trial stops that leave no validated motion at all
HARD_STOPS = ("error", "no_placement", "ag_not_formed", "start_not_held", "approach_failed")


@dataclass
class MotionConfig:
    start_hold_steps: int = 30
    start_drift_rad: float = 0.03
    body_shift_m: float = 0.02
    body_rot_rad: float = 0.10
    contact_lost_m: float = 0.03
    min_travel_rad: float = 0.0
    wp_step_rad: float = 0.03
    wp_step_m: float = 0.01
    rest_drops_m: Tuple[float, ...] = (0.003, 0.02, 0.05)
    max_trials: int = 48
    pre_ik_seeds: int = 12
    #: no placement at the task's face yaw (``ValidateConfig(base_yaws=1)``): retry over this many base yaws (0 = no fallback); labelled
    yaw_fallback_base_yaws: int = 8
    #: the oracle's approach from the pre-grasp failed to track: rerun the trial from the contact pose (labelled ``start_at`` contact_retry)
    contact_retry: bool = True


def alias_slug(alias: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in alias)


def load_annotation(hkey: str) -> Dict[str, Any]:
    """The sampler's handle annotation (``articulation_small._load_handle``: ``<grasp library>/artic/handles/<key>.json``)."""
    from ispatialgym.sampling.grasp_library import library_root

    path = os.path.join(library_root(), "artic", "handles", f"{hkey}.json")
    with open(path) as f:
        return json.load(f)


def local_hull(obj):
    """``sim_utils.World.local_hull``: every link's collision boundary points in the object's root frame (PhysX synced to Fabric first)."""
    import omnigibson as og
    import torch as th

    try:
        og.sim.sync_physx_to_fabric()
    except Exception:  # noqa: BLE001 - no fabric interface in this build: one render frame does the same
        og.sim.render()
    p, q = obj.get_position_orientation()
    pts = [link.collision_boundary_points_world for link in obj.links.values()]
    pts = [x for x in pts if x is not None and len(x)]
    if not pts:
        return th.zeros(0, 3)
    R = th.as_tensor(S.quat_to_mat(q.numpy()), dtype=th.float32)
    return (th.cat(pts, dim=0).float() - p.float()) @ R


def resample_waypoints(first: Dict[str, Any], samples: Sequence[Dict[str, Any]], sense: float, step_rad: float, step_m: float) -> List[Dict[str, Any]]:
    """``first`` (the nominal contact pose at the start q) + the samples whose joint travel (in the motion's ``sense``) passed the last
    kept one's by ``step_rad`` or whose hand moved ``step_m`` from it - monotone in travel (a sample that went back is never kept) - and
    the last sample when it advanced at all."""
    q0 = float(first["q"])
    kept = [{"q": round(q0, 5), "eef_in_object": first["eef_in_object"]}]
    trav_last = 0.0
    p_last = np.asarray(first["eef_in_object"]["position"], dtype=float)
    best = None
    for x in samples:
        trav = sense * (float(x["q"]) - q0)
        p = np.asarray(x["eef_in_initial_object"]["position"], dtype=float)
        if trav < trav_last - 1e-9:
            continue
        if trav >= trav_last + step_rad or (trav >= trav_last and float(np.linalg.norm(p - p_last)) >= step_m):
            kept.append({"q": round(float(x["q"]), 5), "eef_in_object": x["eef_in_initial_object"]})
            trav_last, p_last, best = trav, p, None
        elif trav > trav_last + 1e-6:
            best = x
    if best is not None:
        kept.append({"q": round(float(best["q"]), 5), "eef_in_object": best["eef_in_initial_object"]})
    return kept


def segment_from_trial(tr: Dict[str, Any], motion: Dict[str, Any], rest: Dict[str, Any], mcfg: MotionConfig, source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The library segment of a passing trial (None otherwise): the validated part of its motion segment (module section above)."""
    if tr.get("stop") in HARD_STOPS or tr.get("validated_until_step") is None or not tr.get("T_object_eef_used"):
        return None
    label = tr.get("motion_segment")
    until = int(tr["validated_until_step"])
    samples = [x for x in tr.get("trajectory") or [] if x.get("segment") == label and int(x["step"]) <= until]
    sense = 1.0 if motion["motion"] != "push_close" else -1.0
    first = {"q": float(tr["start"]["q_held"]), "eef_in_object": tr["T_object_eef_used"]}
    wps = resample_waypoints(first, samples, sense, mcfg.wp_step_rad, mcfg.wp_step_m)
    travel = max(0.0, sense * (float(wps[-1]["q"]) - float(wps[0]["q"])))
    if travel < mcfg.min_travel_rad - 1e-9:
        return None
    var = next((v for v in motion["variants"] if v["name"] == tr.get("variant")), {})
    seg = {"alias": motion["alias"], "motion": motion["motion"], "variant": tr.get("variant"), "assignment": var.get("assignment"), "joint": motion["joint"],
           "q_start": round(float(wps[0]["q"]), 5), "q_list": [w["q"] for w in wps], "waypoints": wps, "validated_travel_rad": round(travel, 5),
           "trial_stop": tr.get("stop"), "rest_id": rest.get("rest_id", "r0"), "fingers": {"pre": var.get("preshape_q"), "path": var.get("contact_q")},
           "face_yaw_offset": motion["face_yaw_offset"], "arms_offered": list(motion["arms_offered"]), "handle_sig": motion["handle_sig"],
           "arm_validated": tr.get("arm_validated"), "stance": tr.get("stance"), "start": tr.get("start"),
           "source": dict(source, trial=tr.get("candidate"), validated_until_step=until, motion_segment=label)}
    seg["id"] = S.digest({k: seg[k] for k in ("alias", "motion", "variant", "joint", "q_list", "handle_sig")} | {"src": source.get("validator")}, 12)
    return seg


def build_motion_library(motion: Dict[str, Any], rest: Dict[str, Any], profile: Dict[str, Any], trials: Sequence[Dict[str, Any]], mcfg: MotionConfig,
                         source: Dict[str, Any]) -> Dict[str, Any]:
    segs = [s for s in (segment_from_trial(t, motion, rest, mcfg, source) for t in trials) if s is not None]
    return {"schema": LIBRARY_V2, "category": motion["category"], "model": motion["model"], "scale": motion["scale"], "alias": motion["alias"],
            "motion": motion["motion"], "joint": motion["joint"], "q_start": motion["q_start"], "handle_sig": motion["handle_sig"],
            "face_yaw_offset": motion["face_yaw_offset"], "arms_offered": list(motion["arms_offered"]), "handle_arm": motion["handle_arm"],
            "rests": {rest.get("rest_id", "r0"): rest}, "profile": profile, "segments": segs,
            "n_trials": len(trials), "n_segments": len(segs), "validator": VALIDATOR_VERSION, "motions_version": MOTIONS_VERSION, "config": asdict(mcfg),
            "frames": "eef_in_object: the eef in the object root frame at the motion start (the settled rest, the joint at q_start)", "created": S.now()}


class MotionWorld:
    """The oracle's motions (module section above) on an :class:`ArticWorld` (its config: follow step / chunk, limits, stall, wobble)."""

    def __init__(self, aw: ArticWorld, mcfg: MotionConfig = MotionConfig()):
        self.aw, self.w, self.mcfg = aw, aw.w, mcfg

    # -- the alias's object, motion, profile and rest

    def setup(self, alias: str, root: str, customs_path: Optional[str] = None) -> Dict[str, Any]:
        from ispatialgym.grasping import artic_motions as AM
        from ispatialgym.grasping import object_geometry as OG

        entry = AM.load_alias(alias, customs_path)
        cat, model, joint, hkey = AM.target_model(entry)
        job = {"category": cat, "model": model, "scale": None, "dataset": "behavior-1k-assets", "id": S.entry_id(cat, model, None, "articulation")}
        d = S.model_dir(cat, model, None, root, "articulation")
        os.makedirs(d, exist_ok=True)
        obj = self.w.load(job, d)
        try:
            geom = OG.load(d)
            hull = local_hull(obj)
            jn = joint or next(iter(obj.joints))
            jl = (float(obj.joints[jn].lower_limit), float(obj.joints[jn].upper_limit))
            motion = AM.build(alias, entry, load_annotation(hkey), hull, jl, scale=getattr(obj, "scale", None).tolist() if getattr(obj, "scale", None) is not None else None)
            quat0 = S.quat_yaw(float(motion["face_yaw_offset"]))
            rest0, state0 = self.w.settle(obj, geom, quat0)
            profile = self.profile(obj, motion, state0, rest0)
            profile["force_probe"] = self.refresh_probe(obj)
            rest, state, outcomes = self.rest(obj, geom, quat0, motion)
        except Exception:
            self.w.unload(obj)
            raise
        return {"entry": entry, "dir": d, "obj": obj, "geom": geom, "motion": motion, "profile": profile, "rest": rest, "state": state, "outcomes": outcomes}

    def refresh_probe(self, obj) -> Dict[str, Any]:
        """A new :class:`validate_tabletop.ForceProbe` after the profile's USD edits. ``constraints.apply_hinge_drive`` ends with
        ``og.sim.update_handles()``, which creates a NEW physics simulation view: the probe made at ``load`` keeps the old one, its force
        matrix raises on every read and ``ForceProbe.read`` swallows that - every support load read 0 N, so ``settle``'s contact test (awake
        load > 0.5 x weight) failed and no rest was stable (artic_prefix_20261107: 4 / 4 aliases, rest_stable false, 0 trials). The new
        probe is read once here; a probe that still cannot read raises instead of reporting no contact."""
        import omnigibson as og

        from ispatialgym.grasping import validate_tabletop as VT

        try:
            self.w.probe.uninstall()
        except Exception:  # noqa: BLE001
            pass
        self.w.probe = VT.ForceProbe(self.w.robot, self.w.slab, obj, self.w.cfg.arm)
        self.w.probe.install()
        try:
            F = self.w.probe.view.get_contact_force_matrix(dt=og.sim.get_physics_dt())
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"the force probe cannot read after the profile's USD edits ({e!r}): the rest's contact test would read 0 N") from e
        return {"refreshed": True, "shape": [int(v) for v in F.shape], "why": "og.sim.update_handles() after the joint drive edits replaces the physics view"}

    def set_start(self, obj, motion: Dict[str, Any], q_target: float) -> None:
        """The target joint at ``q_target``, every other joint at its closed value (``articulation_small._close_push_flow``'s rule)."""
        import torch as th

        names = list(obj.joints)
        qs = obj.get_joint_positions().clone()
        for i, n in enumerate(names):
            if n == motion["joint"]:
                qs[i] = float(q_target)
            else:
                ji = (motion.get("all_jinfo") or {}).get(n) or {}
                qs[i] = float(ji.get("closed_q", float(obj.joints[n].lower_limit)))
        obj.set_joint_positions(qs)
        obj.set_joint_velocities(th.zeros_like(qs))

    def stable_set(self, obj, motion: Dict[str, Any], state, rest) -> Tuple[List[float], List[Dict[str, Any]]]:
        """The sampler's stable set S (``_stable_angles``) under the brake applied now, each probe from the rest snapshot."""
        ps = motion["profile_spec"]
        lo_f, hi_f = ps["open_fraction_range"]
        fracs = [float(f) for f in np.linspace(float(lo_f), float(hi_f), int(ps["stable_set_fractions"])).tolist()] + [1.0]
        lo, hi = float(motion["lower"]), float(motion["upper"])
        n_env = max(1, int(math.ceil(int(ps["stable_hold_steps"]) / 4.0)))  # physics steps at 120 Hz -> 30 Hz validation steps
        tol = float(ps["stable_drift_tolerance_fraction"]) * float(ps["stable_probe_tolerance"])
        out, logs = [], []
        for f in fracs:
            self.w.restore(obj, state, rest)
            q = lo + f * (hi - lo)
            self.set_start(obj, motion, q)
            a = self.w.hold_action()
            for _ in range(n_env):
                self.w.step(a)
            drift = abs(self.aw.joint_q(obj, motion["joint"]) - q)
            logs.append({"frac": round(f, 4), "q": round(q, 4), "drift_rad": round(drift, 4)})
            if drift <= tol:
                out.append(q)
        return sorted(set(round(v, 4) for v in out)), logs

    def profile(self, obj, motion: Dict[str, Any], state, rest) -> Dict[str, Any]:
        """Step 1 of the module section: the alias's brake by the sampler's rule, then its other joints' drives."""
        from ispatialgym.grasping import artic_motions as AM
        from ispatialgym.sampling import constraints as CN

        ps = motion["profile_spec"]
        j = obj.joints[motion["joint"]]
        chosen, applied, stable, probes = None, None, [], []
        for c in AM.brake_candidates(ps.get("joint_friction")):
            spec = CN.joint_brake_spec(float(str(c).split(":", 1)[1])) if isinstance(c, str) and str(c).startswith("brake:") else c
            if spec is not None:
                applied = CN.apply_joint_brake(j, spec)
                chosen = c
            stable, logs = self.stable_set(obj, motion, state, rest)
            probes.append({"candidate": c, "stable": stable, "probe": logs})
            if stable and (spec is None or any(q < float(motion["upper"]) - 1e-6 for q in stable)):
                break
        drives = {}
        for jn, spec in (ps.get("joint_drives") or {}).items():
            if jn == motion["joint"]:
                continue
            if jn not in obj.joints:
                raise ValueError(f"generation.joint_drives names {jn!r}, not a joint of the model ({list(obj.joints)})")
            drives[jn] = CN.apply_joint_brake(obj.joints[jn], spec)
        if any((spec or {}).get("lower_limit_rad") is not None for spec in (ps.get("joint_drives") or {}).values()):
            CN.restrict_openable_joints(obj, [motion["joint"]])
        return {"joint": motion["joint"], "brake_candidate": chosen, "brake_applied": AM._jsonable(applied), "stable_set": stable, "probes": probes,
                "other_joint_drives": AM._jsonable(drives), "rule": "articulation_small.sample_template (joint_friction candidates, then joint_drives)"}

    def rest(self, obj, geom, quat0, motion: Dict[str, Any]) -> Tuple[Dict[str, Any], Any, List[Dict[str, Any]]]:
        """Step 2: ``(rest record, (snapshot of the first drop, its simulator record), outcomes)``; the record's ``stable`` is the stable-rest
        rule. Every joint closed before each drop (the stable-set probes leave the target joint open)."""
        from ispatialgym.grasping.settle_first import yaw_free

        outs, first, state = [], None, None
        q_closed = float((motion.get("jinfo") or {}).get("closed_q", motion["lower"]))
        for drop in self.mcfg.rest_drops_m:
            self.set_start(obj, motion, q_closed)
            rec, st = self.w.settle(obj, geom, quat0, drop_m=float(drop))
            if first is None:
                first, state = rec, st
            outs.append(rec)
        tol = float(self.w.cfg.rest_tilt_tol_rad)
        stable = all(r["stable"] for r in outs) and all(float(r["tilt_from_intended_rad"]) <= tol for r in outs) and all(S.same_rest(first, r) for r in outs)
        rest = dict(first, rest_id="r0", stable=bool(stable), sim_quat_xyzw=first["quat_xyzw"],
                    quat_xyzw=[round(float(v), 7) for v in yaw_free(first["quat_xyzw"])], intended_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
                    settled_at_yaw=round(float(S.yaw_of_quat(np.asarray(quat0, dtype=float))), 6),
                    rule=f"stable at every drop {list(self.mcfg.rest_drops_m)} m, tilt <= {tol} rad from the intended (identity) rest, the same rest as the first drop",
                    outcomes=[{k: r.get(k) for k in ("drop_m", "stable", "pose_still", "contact", "tilt_from_intended_rad", "h_root", "quat_xyzw", "settle_steps")} for r in outs])
        # the snapshot is restored with the simulator's own (yawed) quaternion: TabletopWorld.restore compares against rest["quat_xyzw"]
        rest_sim = dict(first)
        rest["restore"] = {"quat_xyzw": first["quat_xyzw"]}
        return rest, (state, rest_sim), outs

    # -- one trial

    def trial(self, obj, motion: Dict[str, Any], variant: Dict[str, Any], state, rest_sim: Dict[str, Any], geom) -> Dict[str, Any]:
        """One variant (module section above); when the oracle's approach from the pre-grasp does not track, the trial is rerun from the
        contact pose (``contact_retry``; the first attempt's stop is kept as ``retry_of``)."""
        rec = self._trial(obj, motion, variant, state, rest_sim, geom, from_pre=True)
        if self.mcfg.contact_retry and rec.get("stop") == "approach_failed" and (rec.get("stance") or {}).get("start_at") == "pre":
            first = {k: rec.get(k) for k in ("stop", "error", "stance")}
            rec = self._trial(obj, motion, variant, state, rest_sim, geom, from_pre=False)
            rec["retry_of"] = first
            if (rec.get("stance") or {}).get("start_at") == "contact":
                rec["stance"]["start_at"] = "contact_retry"
        return rec

    def _trial(self, obj, motion: Dict[str, Any], variant: Dict[str, Any], state, rest_sim: Dict[str, Any], geom, from_pre: bool = True) -> Dict[str, Any]:
        import torch as th

        from ispatialgym.grasping import validate_tabletop as VT
        from ispatialgym.oracle import execute as EX
        from ispatialgym.oracle.motions import Motions, MotionsConfig
        from ispatialgym.robot import conventions as C

        w, cfg, mcfg, arm = self.w, self.aw.cfg, self.mcfg, self.w.cfg.arm
        t0 = time.time()
        push = motion["motion"] == "push_close"
        jn, jinfo, h = motion["joint"], motion["jinfo"], variant["handle"]
        label = "push0" if push else "pull0"
        rec: Dict[str, Any] = {"schema": S.ARTIC_TRIAL, "alias": motion["alias"], "motion": motion["motion"], "variant": variant["name"],
                               "candidate": f"{motion['alias']}:{variant['name']}", "joint": jn, "arm_validated": arm, "fixed_base": cfg.fixed_base,
                               "motion_segment": label, "segments": [], "trajectory": [], "validated_until_step": None}
        rec.update(w.restore(obj, state, rest_sim))
        q_set = float(motion["q_start"])
        self.set_start(obj, motion, q_set)
        a = w.hold_action()
        for _ in range(mcfg.start_hold_steps):
            w.step(a)
        q_held = self.aw.joint_q(obj, jn)
        rec["start"] = {"q_set": round(q_set, 5), "q_held": round(q_held, 5), "drift_rad": round(abs(q_held - q_set), 5), "hold_steps": mcfg.start_hold_steps}
        if abs(q_held - q_set) > mcfg.start_drift_rad:
            rec.update({"stop": "start_not_held", "wall_s": round(time.time() - t0, 2)})
            return rec
        p0, q0 = obj.get_position_orientation()
        T_wo0 = S.pose_mat(p0.numpy(), q0.numpy())
        wp = C.handle_waypoints(h, jinfo, [q_held])[0]  # the object frame
        T_oe = S.pose_mat(wp.eef[0].numpy(), wp.eef[1].numpy())
        pre_q, con_q = float(variant["preshape_q"] or 0.0), float(variant["contact_q"] or 0.0)
        cand_g = {"name": variant["name"], "T_object_eef": S.mat_pose(T_oe), "preshape_q": pre_q, "contact_q": con_q, "width_m": float(variant.get("width_m") or 0.0)}
        plans = w.plan(obj, geom, [cand_g], (p0.numpy(), q0.numpy()))
        pl = plans.get(variant["name"])
        yaw_fallback = False
        if (pl is None or pl.get("q") is None) and mcfg.yaw_fallback_base_yaws > w.cfg.base_yaws:
            # no stance at the face yaw: the object-relative motion is what the map reads, so the stance may stand at another base yaw
            n0 = w.cfg.base_yaws
            w.cfg.base_yaws = int(mcfg.yaw_fallback_base_yaws)
            try:
                pl = w.plan(obj, geom, [cand_g], (p0.numpy(), q0.numpy())).get(variant["name"])
            finally:
                w.cfg.base_yaws = n0
            yaw_fallback = True
        if pl is None or pl.get("q") is None:
            rec.update({"stop": "no_placement", "fail": None if pl is None else pl.get("fail"), "yaw_fallback": yaw_fallback, "wall_s": round(time.time() - t0, 2)})
            return rec
        T_oe_used = S.pose_of(pl["T_object_eef_used"])
        rec["T_object_eef_used"] = pl["T_object_eef_used"]
        rec["stance"] = {"base": pl["base"], "trunk": pl["trunk"], "start_lift_m": pl.get("start_lift_m"), "yaw_fallback": yaw_fallback}
        T_g = T_wo0 @ T_oe_used
        T_pre = T_g.copy()
        T_pre[:3, 3] -= C.PRE_DIST * T_g[:3, 2]

        def pose_t(T):
            return (th.as_tensor(T[:3, 3], dtype=th.float32), th.as_tensor(S.mat_to_quat(T[:3, :3]), dtype=th.float32))

        q_start = pl["q"].clone()
        start_at = "contact"
        try:
            if from_pre:
                succ, qq = w.backend.solve_ik_batch(f"{arm}_arm", w.M.EEF_LINK[arm], [pose_t(T_pre)], pl["q"], batch_size=8, seeds=mcfg.pre_ik_seeds)
                if bool(succ[0]):
                    q_start, start_at = qq[0].clone(), "pre"
        except Exception as e:  # noqa: BLE001 - the pre-grasp IK is a convenience: the trial then starts at the contact (labelled)
            rec["pre_ik_error"] = repr(e)[:200]
        rec["stance"]["start_at"] = start_at
        fingers0 = C.FINGER_CLOSED if push else pre_q
        q_start[w.robot.gripper_control_idx[arm]] = fingers0
        w.robot.set_joint_positions(q_start)
        w.robot.keep_still()
        mcfg_m = MotionsConfig()
        motions = Motions(w.env, w.backend, cfg=mcfg_m, robot=w.robot)
        motions.state.hold_current()
        motions.state.gripper_cmd[arm] = motions.gripper_command(arm, fingers0)
        obj.wake()
        k = [0]
        bad = {"why": None}
        hc = {"h": h, "jinfo": jinfo}

        def step_all(gen, seg_label):
            for act in gen:
                w.step(act)
                k[0] += 1
                ep, eq = (x.numpy() for x in w.robot.eef_links[arm].get_position_orientation())
                Te = S.pose_mat(ep, eq)
                bp, bq = w.robot.get_position_orientation()
                po, qo = obj.get_position_orientation()
                qj = self.aw.joint_q(obj, jn)
                shift = float(np.linalg.norm(po.numpy() - p0.numpy()))
                rot = float(VT._qang(qo.numpy(), q0.numpy()))
                held = w.robot._ag_obj_in_hand.get(arm) is obj
                x = {"step": k[0], "t_s": round(k[0] / 30.0, 4), "segment": seg_label, "q": round(qj, 5), "eef_world": S.mat_pose(Te),
                     "eef_in_initial_object": S.mat_pose(np.linalg.inv(T_wo0) @ Te),
                     "base": [round(float(bp[0]), 4), round(float(bp[1]), 4), round(S.yaw_of_quat(bq.numpy()), 4)],
                     "torso_err_rad": round(float(motions.state.trunk_error()), 5) if hasattr(motions.state, "trunk_error") else None,
                     "held": held, "body_shift_m": round(shift, 5), "body_rot_rad": round(rot, 5)}
                if push:
                    wn = C.handle_waypoints(hc["h"], hc["jinfo"], [qj], (po, qo))[0]
                    x["contact_dist_m"] = round(float(np.linalg.norm(ep - wn.eef[0].numpy())), 5)
                rec["trajectory"].append(x)
                if seg_label == label and bad["why"] is None:
                    why = None
                    if shift > mcfg.body_shift_m or rot > mcfg.body_rot_rad:
                        why = "body_moved"
                    elif push and x["contact_dist_m"] > mcfg.contact_lost_m:
                        why = "contact_lost"
                    elif not push and not held:
                        why = "grasp_lost"
                    if why is None:
                        rec["validated_until_step"] = k[0]
                    else:
                        bad["why"] = why

        try:
            step_all(motions.hold(2), "init")
            if push:
                step_all(motions.gripper(arm, True), "close")
                if start_at == "pre":
                    step_all(motions.arm_along(arm, [pose_t(T_g)], loaded=True, converge=False), "approach")
            else:
                if start_at == "pre":
                    step_all(motions.arm_along(arm, [pose_t(T_g)], converge=True), "approach")
                rate, squeeze, _ = EX.closure_for_width(float(variant.get("width_m") or 0.0), mcfg_m, library=False)
                step_all(motions.gripper(arm, True, rate=rate, squeeze=squeeze), "close")
                if w.robot._ag_obj_in_hand.get(arm) is not obj:
                    rec.update({"stop": "ag_not_formed", "achieved_rad": 0.0, "wall_s": round(time.time() - t0, 2)})
                    return rec
        except Exception as e:  # noqa: BLE001 - the oracle's approach did not track
            rec.update({"stop": "approach_failed", "error": f"{type(e).__name__}: {str(e)[:200]}", "wall_s": round(time.time() - t0, 2)})
            return rec
        # the motion: one fixed-stance segment, the hand turned rigidly about the LIVE hinge toward the arc end
        start_q = self.aw.joint_q(obj, jn)
        q_end = float(motion["q_arc_end"])
        rec["segments"].append({"kind": "arm_push" if push else "arm_pull", "index": 0, "start_step": k[0], "start_q": round(start_q, 4), "q_arc_end": round(q_end, 4)})
        centre, axis = self.aw.hinge_world(obj, jinfo)
        ep, eq = (x.numpy() for x in w.robot.eef_links[arm].get_position_orientation())
        T0 = S.pose_mat(ep, eq)
        remaining = q_end - start_q
        n = max(1, int(math.ceil(abs(remaining) / cfg.follow_arc_step_rad)))
        dense = []
        for i in range(1, n + 1):
            R = _axis_angle(axis, remaining * i / n)  # conventions.handle_waypoints: +dq turns the link (and the hand on it) +dq about the axis
            Ti = np.eye(4)
            Ti[:3, :3] = R @ T0[:3, :3]
            Ti[:3, 3] = centre + R @ (T0[:3, 3] - centre)
            dense.append(pose_t(Ti))
        goal_q = float(motion["q_goal"]) if push else float(motion["upper"])
        stop, stalled, wob, last_q = None, 0, 0, start_q
        for c0 in range(0, len(dense), cfg.follow_reaim_chunk):
            try:
                step_all(motions.arm_along(arm, dense[c0:c0 + cfg.follow_reaim_chunk], converge=False, loaded=push), label)
            except Exception as e:  # noqa: BLE001 - the demo tracker gave up (reach / IK)
                stop = f"tracking_failed: {type(e).__name__}: {str(e)[:160]}"
                break
            qn = self.aw.joint_q(obj, jn)
            stalled = stalled + cfg.follow_reaim_chunk if abs(qn - last_q) < cfg.stall_rad else 0
            last_q = qn
            te = rec["trajectory"][-1].get("torso_err_rad") or 0.0
            wob = wob + 1 if te > cfg.wobble_rad else 0
            if bad["why"] is not None:
                stop = bad["why"]
                break
            if abs(goal_q - qn) <= cfg.limit_tol_rad:
                stop = "goal_reached" if push else "limit_reached"
                break
            if stalled >= cfg.stall_steps:
                stop = "stalled"
                break
            if wob >= cfg.wobble_steps:
                stop = "robot_wobble"
                break
        rec["segments"][-1].update({"end_step": k[0], "end_q": round(self.aw.joint_q(obj, jn), 4), "stop": stop or "arc_end"})
        jq = self.aw.joint_q(obj, jn)
        until = rec.get("validated_until_step")
        q_until = next((x["q"] for x in reversed(rec["trajectory"]) if until is not None and x["step"] == until), None)
        rec.update({"stop": stop or "arc_end", "start_q": round(start_q, 5), "final_q": round(jq, 5), "achieved_rad": round(abs(jq - q_held), 5),
                    "validated_travel_rad": round(abs(q_until - q_held), 5) if q_until is not None else None, "range_rad": round(abs(q_end - q_held), 5),
                    "achieved_scope": "this handle variant / stance / controller only - not the object's global limit",
                    "steps": k[0], "wall_s": round(time.time() - t0, 2),
                    "frames": {"eef_world": "world", "eef_in_initial_object": "the object root at the motion start (rest, joint at q_start)"}})
        return rec


def run_alias(mw: MotionWorld, alias: str, root: str, *, rest_only: bool = False, customs_path: Optional[str] = None) -> Dict[str, Any]:
    """One alias: setup (motion, profile, rest), the trials (unless ``rest_only``), ``motions/<alias>.json`` and the library."""
    t0 = time.time()
    ctx = mw.setup(alias, root, customs_path)
    d, obj, motion, rest = ctx["dir"], ctx["obj"], ctx["motion"], ctx["rest"]
    state, rest_sim = ctx["state"]
    mdir = os.path.join(d, "motions")
    os.makedirs(mdir, exist_ok=True)
    out: Dict[str, Any] = {"schema": "isg-artic-motion-validation/1", "alias": alias, "validator": VALIDATOR_VERSION, "motions_version": MOTIONS_VERSION,
                           "motion": motion, "profile": ctx["profile"], "rest": rest, "config": {"artic": asdict(mw.aw.cfg), "motion": asdict(mw.mcfg)},
                           "validation_world": {"arm": mw.w.cfg.arm, "base_yaws": mw.w.cfg.base_yaws, "slab_top_z": mw.w.cfg.slab_top_z}, "trials": []}
    try:
        S.save_json(os.path.join(mdir, f"{alias_slug(alias)}__rest.json"), {"schema": REST_SCHEMA, "alias": alias, "rest": rest, "profile": ctx["profile"],
                                                                           "motion": {k: motion[k] for k in ("motion", "joint", "q_start", "face_yaw_offset", "handle_sig")}})
        if rest_only:
            out["wall_s"] = round(time.time() - t0, 2)
            return out
        if not rest.get("stable"):
            out["error"] = "no stable rest (validate_artic rest): no trial is run"
        else:
            for v in motion["variants"][: mw.mcfg.max_trials]:
                try:
                    tr = mw.trial(obj, motion, v, state, rest_sim, ctx["geom"])
                except Exception as e:  # noqa: BLE001
                    tr = {"candidate": f"{alias}:{v['name']}", "variant": v["name"], "stop": "error", "error": repr(e), "traceback": traceback.format_exc()[-1500:]}
                out["trials"].append(tr)
        out["wall_s"] = round(time.time() - t0, 2)
        vpath = os.path.join(mdir, f"{alias_slug(alias)}.json")
        S.save_json(vpath, out)
        lib = build_motion_library(motion, {k: v for k, v in rest.items() if k not in ("physics_profile",)}, ctx["profile"], out["trials"], mw.mcfg,
                                   {"validator": f"{VALIDATOR_VERSION}+{MOTIONS_VERSION}", "validation": os.path.relpath(vpath, d)})
        lib_path = os.path.join(d, f"artic_library__{alias_slug(alias)}.json")
        S.save_json(lib_path, lib)
        out["library"] = lib_path
        out["n_segments"] = lib["n_segments"]
    finally:
        mw.w.unload(obj)
    return out


def _motions_main(argv) -> None:
    from collections import Counter

    ap = argparse.ArgumentParser(description="validate_artic rest / motions: the oracle's small-articulation motions per alias (artic_motions)")
    ap.add_argument("cmd", choices=["rest", "motions"])
    ap.add_argument("--aliases", nargs="+", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--customs", default=None, help="the customs file (default the code's isg_customs.json)")
    ap.add_argument("--max-trials", type=int, default=MotionConfig.max_trials)
    ap.add_argument("--any-yaw", action="store_true", help="the stance search over 8 base yaws (default: the task's face yaw only)")
    a = ap.parse_args(argv)
    from ispatialgym.grasping import validate_tabletop as VT

    world = VT.TabletopWorld(VT.ValidateConfig() if a.any_yaw else VT.ValidateConfig(base_yaws=1))
    mw = MotionWorld(ArticWorld(world, ArticConfig(max_segments=1)), MotionConfig(max_trials=a.max_trials))
    for alias in a.aliases:
        try:
            r = run_alias(mw, alias, a.root, rest_only=a.cmd == "rest", customs_path=a.customs)
            m = r["motion"]
            print("GB_ARTIC_MOTION " + json.dumps({"alias": alias, "cmd": a.cmd, "motion": m["motion"], "joint": m["joint"], "q_start": m["q_start"],
                                                    "face_yaw_offset": m["face_yaw_offset"], "arms_offered": m["arms_offered"], "n_variants": len(m["variants"]),
                                                    "rest_stable": r["rest"].get("stable"), "rest_quat": r["rest"].get("quat_xyzw"), "h_root": r["rest"].get("h_root"),
                                                    "brake": r["profile"].get("brake_candidate"), "stable_set": r["profile"].get("stable_set"), "error": r.get("error"),
                                                    "rest_outcomes": [{k: o.get(k) for k in ("drop_m", "stable", "pose_still", "contact", "tilt_from_intended_rad")} for o in r["rest"].get("outcomes") or []],
                                                    "support_load_N": r["rest"].get("support_load_N"), "weight_N": r["rest"].get("weight_N"),
                                                    "stops": dict(Counter(str(t.get("stop")).split(":")[0] for t in r["trials"])),
                                                    "trials": [{k: t.get(k) for k in ("variant", "stop", "validated_travel_rad", "achieved_rad", "steps")} | {"start_at": (t.get("stance") or {}).get("start_at")} for t in r["trials"]],
                                                    "n_segments": r.get("n_segments"), "library": r.get("library"), "wall_s": r.get("wall_s")}, default=str), flush=True)
        except Exception as e:  # noqa: BLE001
            print("GB_ARTIC_MOTION " + json.dumps({"alias": alias, "cmd": a.cmd, "error": repr(e), "tb": traceback.format_exc()[-1500:]}), flush=True)
    print("GB_ARTIC_DONE", flush=True)
    import omnigibson as og

    og.shutdown()


def main(argv=None):
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("rest", "motions"):
        return _motions_main(argv)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--keys", nargs="+", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--fixed-base", action="store_true")
    a = ap.parse_args(argv)
    from ispatialgym.grasping import validate_tabletop as VT

    world = VT.TabletopWorld(VT.ValidateConfig())
    aw = ArticWorld(world, ArticConfig(fixed_base=a.fixed_base))
    for key in a.keys:
        try:
            r = run_key(world, aw, key, a.root)
            print("GB_ARTIC " + json.dumps({"key": key, "wall_s": r.get("wall_s"), "error": r.get("error"),
                                            "trials": [{k: t.get(k) for k in ("candidate", "stop", "achieved_rad", "range_rad", "achieved_fraction", "steps", "error")} for t in r["trials"]],
                                            "profiles": {k: {kk: v.get(kk) for kk in ("source", "kind", "brake")} for k, v in r["profiles"].items()}}), flush=True)
        except Exception as e:  # noqa: BLE001
            print("GB_ARTIC " + json.dumps({"key": key, "error": repr(e), "tb": traceback.format_exc()[-1500:]}), flush=True)
    import omnigibson as og

    og.shutdown()


if __name__ == "__main__":
    main()
