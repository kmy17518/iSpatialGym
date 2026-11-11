"""Program check (spec §3.5, principle 4; oracle doc §3.7): the last stage of every attempt.

``run(variant, sidecar, sampler)`` expands the oracle's program for the variant from the completed draft sidecar
(``ispatialgym.oracle.programs.expand(variant, sidecar_dict)``, F6), asks every action for its ``targets()`` and IK- /
collision-checks every pose and waypoint it returns — in the recorded posture, trunk and stance, with the carried
object attached where the action carries one (screening IK at waypoints, exact IK at end poses, §3.6 / §3.9). A failing
pose is a bug in the rule that recorded it: the attempt is rejected with ``Reject("program_check_failed", action=<name>)``
and nothing is patched. ``validate_instances.py`` (D5) runs the same expansion on written instances.

Until F6 lands (``git log`` shows ``oracle: sidecar schema ... (F6)``), :func:`run` falls back to :func:`check_blocks`,
a structural stub that validates the sidecar blocks the variant's program will consume exist and are well-formed
(stances, transitions, paths, grasps, contacts), so that the pipeline stage is exercised end to end.

``targets()`` contract we code against (oracle doc §3.6): a dict whose values are eef poses ``(pos, quat)`` /
``{"position", "orientation"}`` dicts, lists of such (waypoint lists), ``(x, y, yaw)`` base poses under keys containing
``base`` / ``path``, or nested dicts; the action carries ``name``, ``arms``, ``objects`` and optional attributes
``stance`` (stance name used), ``carried`` (bddl instance held during the action) and ``ignore`` (object names removed
from the collision world for the final pose).
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch as th

from ispatialgym.fastflags import fast_on  # train-fast switches (2026-10-31): EARLY_FRAME_GATE, SYNC_ONLY_OBJECTS, ATTACH_REUSE
from ispatialgym.sampling.v2.ik import IKResult
from ispatialgym.sampling.v2.stances import PathRecord, SidecarDraft, Stance, pose_from_sidecar

#: a push whose recorded pre-contact witness lies this far (rad, any arm joint) from the raw IK solved from the stance posture
#: is a FLIPPED branch (owner 2026-09-26, axial 0926-043254 / 051919 / 061151: 3.10 / 2.99 rad, wrist flips of about pi); world
#: vertical's green instances keep 0.73 rad (wave 15 0926-061803 attempt 7 was rejected at 0.6), so only a flip is judged
FLIPPED_PUSH_WITNESS_RAD = 2.0


def _flipped_push_chain_check(sampler, chain, stance, arm: str, q_first, ignore_last, ignore_all, fingers, approached) -> Optional[Tuple[bool, str]]:
    """Owner 2026-09-26 (axial 0926-043254 wave 10 and 0926-051919 wave 12, one instance: the push pre-contact witness 3.10 rad
    from the branch the IK reaches from the stance posture; the oracle's closed-loop arc then hit a joint limit at q = -0.44 from
    EITHER branch - 0926-050248 reached the witness via the point above, 0926-053900 kept cuRobo's branch - while every pose of the
    chain had an IK solution of its own: the chain check solves each pose, it does not follow the arc).
    ``None`` unless that situation holds (the recorded witness more than :data:`FLIPPED_PUSH_WITNESS_RAD` from the raw IK of the
    pre-contact seeded at the stance posture - the solver without the joint-limit re-solves, whose rescue is what can land in the
    other branch); else ``(ok, why)``: the chain tracked densely (``track_chain`` at the oracle's densification) from the recorded
    witness, else from the stance posture's branch (the oracle keeps that branch for a push: ``arm_to(accept_other_branch=True)``);
    when neither tracks, the instance is rejected."""
    idx = sampler.kit.robot.arm_control_idx[arm]
    q_seed = sampler.ikc.q_locked(stance, fingers=fingers).clone()
    try:
        q_raw = sampler.kit.ik(chain[0], q_seed, arm=arm)
    except Exception:  # noqa: BLE001 - no judgement without the raw solution
        return None
    if q_raw is None:
        return None
    q_raw = th.as_tensor(q_raw).reshape(-1)
    gap = float(th.max(th.abs(q_raw[idx].float() - th.as_tensor(q_first).reshape(-1)[idx].float())))
    if gap <= FLIPPED_PUSH_WITNESS_RAD:
        return None
    # rejected outright (wave 14 verification 0926-060408 / 060411: the sampler's dense track of that chain passed from the flipped
    # witness while the oracle's closed-loop arc fails from it and from cuRobo's branch - the dense track is no proxy for it; trajopt
    # never reached such a witness, 5 of 5): the sampler draws another candidate (axial's other instance passed, 0926-051915 / 051917)
    log.info("program check: the push pre-contact witness lies %.2f rad from the stance posture's branch - rejected", gap)
    return False, (f"approach_untrackable: push pre-contact witness {gap:.2f} rad from the raw IK from the stance posture (another branch): "
                   f"trajopt reaches such a witness only through the pushed object and the closed-loop arc fails from either branch")

def _flipped_push_witness(sampler, chain, stance, arm: str, q_first, fingers) -> Optional[Tuple[float, th.Tensor]]:
    """``(gap, q_raw)`` when the recorded pre-contact witness ``q_first`` lies more than :data:`FLIPPED_PUSH_WITNESS_RAD` (any arm joint)
    from ``q_raw``, the raw IK of the pre-contact seeded at the stance posture (the branch the oracle's planner reaches from the stance
    without going through the pushed object); ``None`` when it is on that branch or the raw IK is unavailable."""
    idx = sampler.kit.robot.arm_control_idx[arm]
    q_seed = sampler.ikc.q_locked(stance, fingers=fingers).clone()
    try:
        q_raw = sampler.kit.ik(chain[0], q_seed, arm=arm)
    except Exception:  # noqa: BLE001 - no judgement without the raw solution
        return None
    if q_raw is None:
        return None
    q_raw = th.as_tensor(q_raw).reshape(-1).float()
    gap = float(th.max(th.abs(q_raw[idx] - th.as_tensor(q_first).reshape(-1)[idx].float())))
    return None if gap <= FLIPPED_PUSH_WITNESS_RAD else (gap, q_raw)


def _rewitness_flipped_push(sampler, chain, stance, arm: str, q_first, fingers, replay, witnesses: Dict[str, Any], single_q: Dict[str, Any],
                            key: str, ignore_all) -> Optional[Tuple[bool, str]]:
    """oracle-fix 2026-11-03 (axial#02 never sampled for eval; alignment-axial_04 lost 44 of ~300 train attempts and axial_05 its share to
    :func:`_flipped_push_chain_check`): a free-object push whose recorded pre-contact witness sits in another IK branch than the stance
    posture's is no longer rejected outright. The 2026-09-26 rule assumed the oracle lands on the stance branch; since 2026-10-12 the
    oracle plans straight to the recorded witness (``execute.reach_pre_pose``), and the 2026-09-27 arc replay (:func:`replay_arc_push`)
    simulates the closed-loop arc the rule was guarding against. Trajopt still reaches a flipped witness only through the pushed object
    (owner 2026-09-26, 5 of 5), so the witness is moved to the stance posture's branch instead: the arc is replayed from the raw IK of
    the pre-contact seeded at the stance posture (``replay(q_raw)``, the same replay as from the recorded witness) and, when it follows,
    the ``pre`` witness (``witnesses[key]``, ``single_q[key]``: what the oracle plans to and what ``push_pre_standoff`` reads) becomes
    that configuration, polished onto the recorded pose under the pre-pose residual limit. When the arc does not replay from that branch
    either, the instance is rejected as before (both reasons kept). ``None`` when the witness is not flipped.
    oracle-fix 2026-11-05 (hb21_desk_s55: a straight push moved to the stance branch with nothing replayed, the oracle's push stalled there):
    a straight push is tracked from that branch too (``replay(q_raw, line=True)``, :func:`track_line_push`); ``ISG_PUSH_REWITNESS_LINE=off``
    restores the unchecked move."""
    flipped = _flipped_push_witness(sampler, chain, stance, arm, q_first, fingers)
    if flipped is None:
        return None
    gap, q_raw = flipped
    why_old = (f"push pre-contact witness {gap:.2f} rad from the raw IK from the stance posture (another branch): trajopt reaches such a "
               f"witness only through the pushed object")
    ok_r, why_r, rep_r = replay(q_raw, line=os.environ.get("ISG_PUSH_REWITNESS_LINE", "on").lower() not in ("off", "0", "false"))
    if rep_r:
        log.info("program check: push replay from the stance posture's branch (the recorded witness is %.2f rad away): %s %s", gap, why_r[:200], rep_r)
    if not ok_r and "line" in rep_r and bool(sampler.gp.gen("push_flipped_keep_recorded") or False):
        # oracle-fix 2026-11-05 (opt-in generation.push_flipped_keep_recorded; hb21_desk_s55 / hb22_desk_s13: the stance branch of the hardback's
        # overhang push sits at left/right_arm_joint2's limit (margin 0.003 rad) and its straight push stalls, while the recorded branch tracks):
        # a straight push that tracks from the recorded witness keeps it - the oracle plans to the recorded witness (execute.reach_pre_pose)
        ok_k, why_k, _ = track_line_push(sampler, arm, q_first, chain[1:])
        if ok_k:
            log.info("program check: the push pre-contact witness lies %.2f rad from the stance posture's branch, whose straight push fails (%s); the "
                     "recorded witness is kept: %s", gap, why_r[:160], why_k)
            return True, f"recorded witness kept ({gap:.2f} rad from the stance posture's branch, whose straight push fails: {why_r[:100]}); {why_k}"
        why_r = f"{why_r}; from the recorded witness too: {why_k}"
    if not ok_r:
        log.info("program check: the push pre-contact witness lies %.2f rad from the stance posture's branch and the arc does not replay from that branch - rejected", gap)
        return False, f"approach_untrackable: {why_old}; the arc replay from the stance posture's branch fails too: {why_r}"
    q_pre, res_pre = sampler.ikc.polish_witness(q_raw, chain[0], arm, ignore=list(ignore_all))
    limit = float(sampler.gp.gen("witness_residual_pre_m", sampler.gp.gen("witness_residual_max_m", sampler.ikc.WITNESS_RESIDUAL_MAX)))
    if res_pre > limit:
        return False, (f"approach_untrackable: {why_old}; the stance-branch pre-contact polishes to {res_pre * 1000:.1f} mm > {limit * 1000:.0f} mm "
                       f"from the recorded pose")
    m_pre, j_pre = sampler.ikc.joint_limit_margin(q_pre, arm)
    note = f"re-witnessed on the stance posture's branch (the recorded witness was {gap:.2f} rad away, another branch); arc replay from it: {why_r}"
    if key in witnesses:
        witnesses[key].update({"q": [round(float(v), 5) for v in sampler.ikc.canonical_q(q_pre, stance).tolist()], "fk_residual_m": round(res_pre, 5),
                               "joint_limit_margin_rad": round(m_pre, 4), "joint_limit_nearest": j_pre, "branch": note[:200]})
    single_q[key] = q_pre.clone()
    log.info("program check: the %s witness of the push re-witnessed on the stance posture's branch (%.2f rad from the recorded one); the arc replays from it", key, gap)
    return True, note


def track_line_push(sampler, arm: str, q_from, waypoints) -> Tuple[bool, str, Dict[str, Any]]:
    """oracle-fix 2026-11-05 (hardback hb21_desk_s55, the first instance through the program check: the straight overhang push's pre-contact
    witness was re-witnessed on the stance posture's branch 3.13 rad from the recorded one and nothing was replayed - 'not an arc' - and
    the oracle's ``arm_along`` from that branch stalled at waypoint 21/28, pos_err 11.5 mm, plan_failed): a straight push tracked from
    ``q_from`` as ``Motions.along_reaches`` / ``arm_along`` runs it - ``Motions._densify`` from the FK pose through ``waypoints`` at the
    arm_along solver settings, collisions not judged (the push's contact legs run without a veto). ``(ok, why, report)``."""
    import math

    from ispatialgym.oracle import motions as MO
    from ispatialgym.oracle import r1pro_model as M
    from ispatialgym.robot import conventions as C

    mcfg, kit = MO.MotionsConfig(), sampler.kit
    frame = M.EEF_LINK[arm]
    kit._ensure_synced()
    q = th.as_tensor(q_from).reshape(-1).float().clone()
    p0, o0 = kit.fk([frame], q)[frame]
    prev = (th.as_tensor(p0).float(), th.as_tensor(o0).float())
    dense = []
    for wp in waypoints:
        p1, o1 = th.as_tensor(wp[0]).float(), th.as_tensor(wp[1]).float()
        n_d = max(1, int(math.ceil(max(float(th.linalg.norm(p1 - prev[0])) / mcfg.cartesian_step, float(C.quat_angle(prev[1], o1)) / mcfg.cartesian_rot_step))))
        dense.extend(tuple(w) for w in C.line_waypoints(prev, (p1, o1), n_d))
        prev = (p1, o1)
    res = kit.backend.solve_ik_path(f"{arm}_arm", [{frame: w} for w in dense], q_start=q, pos_tol=mcfg.diff_ik_pos_tol, rot_tol=mcfg.diff_ik_rot_tol,
                                    max_iters=mcfg.diff_ik_max_iters, check_collisions=False, sync_world=False, soft_pos_tol=mcfg.diff_ik_soft_pos_tol,
                                    soft_rot_tol=mcfg.diff_ik_soft_rot_tol)
    rep = {"dense": len(dense)}
    if not res.success or res.q_traj is None:
        return False, f"approach_untrackable: straight push replay ({len(dense)} dense poses): {res.status}", rep
    return True, f"straight push tracked ({len(dense)} dense poses)", rep


def _push_stop_frac_arc() -> float:
    """The oracle's ``ExecuteConfig.push_stop_frac_arc``: an arc push stops at this fraction of the orientation tolerance."""
    from ispatialgym.oracle import execute as E

    return float(E.ExecuteConfig().push_stop_frac_arc)


def replay_arc_push(sampler, arm: str, q_pre: th.Tensor, pre_pose, waypoints, direction, obj, supports: Sequence[Any], ignore: Sequence[Any],
                    band_rad: float, retreat_distance: float, speed_m_s: Optional[float] = None, line: bool = False,
                    lag_allowance_rad: float = 0.0) -> Tuple[bool, str, Dict[str, Any]]:
    """A free-object arc push replayed kinematically as the oracle executes it (owner 2026-09-27, first-demo v2 A8; the object
    follows the pad): from the pre-contact witness ``q_pre`` the straight leg to the first contact pose (``execute_push`` ->
    ``_push_arc_closed_loop``: ``Motions.arm_along`` at the arm_along solver settings), then the TRUE arc of ``push_geometry``
    (``Motions.arm_arc_closed_loop``: every control step the FK pose rotated by ``push_speed_arc / radius · dt`` about the arc's
    axis, one differential-IK step at the loop's tolerances with its joint-speed bound, 10 consecutive misses fail) until the
    object reaches its goal band, then the contact retreat's escape leg from the arc's end (``execute_retreat``: radial off the
    object, then off its supports, the end probed for reach and shortened like ``_retreat_reach``; tracked with the retreat's
    joint-limit margin; a stall away from a joint limit is finished by IK, a partial retreat accepted as the oracle accepts it).
    Every setting is read from the oracle's ``ExecuteConfig`` / ``MotionsConfig``; ``speed_m_s`` (the push's recorded
    ``push_speed_m_s``, ``execute.push_speed_of``) replaces ``push_speed_arc``. Returns ``(ok, why, report)``; a replay that
    cannot follow the push (``ok`` False) is an ``approach_untrackable`` instance. Collisions are not judged (the oracle's contact
    legs run without a veto).
    ``lag_allowance_rad`` (oracle-fix 2026-11-06, ``generation.push_arc_replay_lag_rad``; axial#01 / #05): the arc must track this much
    beyond the goal band (the replay's object follows the pad exactly; the oracle's free object lags the hand by 0.07-0.16 rad near
    the arc's end - 8 / 26 axial#05 and 6 / 31 axial#01 demos failed on 10 consecutive misses with the elbow straight there), and the
    retreat leaves from that arc end."""
    import math

    from ispatialgym.oracle import execute as E
    from ispatialgym.oracle import motions as MO
    from ispatialgym.oracle import r1pro_model as M
    from ispatialgym.robot import conventions as C

    ecfg, mcfg = E.ExecuteConfig(), MO.MotionsConfig()
    backend, kit = sampler.kit.backend, sampler.kit
    frame = M.EEF_LINK[arm]
    idx = kit.robot.arm_control_idx[arm]
    dt = float(getattr(backend, "control_dt", 1.0 / 30.0))
    rep: Dict[str, Any] = {}
    lag = max(0.0, float(lag_allowance_rad or 0.0))
    geom = E.push_geometry(list(waypoints), direction)
    if geom.kind != "arc":
        if not line:
            return True, "not an arc", rep
        # ``line`` (a re-witnessed straight push, :func:`_rewitness_flipped_push`): the straight push from ``q_pre``, as arm_along runs it
        ok_l, why_l, rep["line"] = track_line_push(sampler, arm, q_pre, waypoints)
        return ok_l, why_l, rep

    def fk(q):
        p, qq = kit.fk([frame], q)[frame]
        return th.as_tensor(p).float(), th.as_tensor(qq).float()

    kit._ensure_synced()
    q = th.as_tensor(q_pre).reshape(-1).float().clone()
    # 1. the straight leg pre -> the first contact pose, densified as Motions._densify does
    start = fk(q)
    end = (th.as_tensor(waypoints[0][0]).float(), th.as_tensor(waypoints[0][1]).float())
    n_d = max(1, int(math.ceil(max(float(th.linalg.norm(end[0] - start[0])) / mcfg.cartesian_step, float(C.quat_angle(start[1], end[1])) / mcfg.cartesian_rot_step))))
    dense = [tuple(w) for w in C.line_waypoints(start, end, n_d)]
    res = backend.solve_ik_path(f"{arm}_arm", [{frame: w} for w in dense], q_start=q, pos_tol=mcfg.diff_ik_pos_tol, rot_tol=mcfg.diff_ik_rot_tol,
                                max_iters=mcfg.diff_ik_max_iters, check_collisions=False, sync_world=False, soft_pos_tol=mcfg.diff_ik_soft_pos_tol,
                                soft_rot_tol=mcfg.diff_ik_soft_rot_tol)
    if not res.success or res.q_traj is None:
        return False, f"approach_untrackable: push replay - the straight leg to the first contact pose: {res.status}", rep
    q = res.q_traj[-1].reshape(-1).float().clone()
    # 2. the closed-loop arc (the object follows the pad: its swept angle is the commanded one)
    ax = C.unit(geom.axis.float())
    cen = geom.pivot.float()

    def radial(p):
        r = p - cen
        return r - ax * float(r @ ax)

    def signed_angle(ra, rb) -> float:
        return math.atan2(float(th.cross(ra, rb, dim=0) @ ax), float(ra @ rb))

    p_t = th.as_tensor(waypoints[-1][0]).float()
    q_to = float(geom.angle)
    sgn_joint = 1.0 if q_to >= 0 else -1.0
    q_goal = q_to + sgn_joint * (lag + band_rad) if lag > 0.0 else q_to  # oracle-fix 2026-11-06: the lag allowance past the goal band
    r_now = radial(fk(q)[0])
    radius = max(float(th.linalg.norm(r_now)), 0.05)
    step_ang = (float(speed_m_s) if speed_m_s else ecfg.push_speed_arc) / radius * dt
    arc_dir = signed_angle(r_now, radial(p_t))
    sgn = 1.0 if arc_dir >= 0 else -1.0
    max_steps = int(max(abs(q_goal), abs(arc_dir)) / max(step_ang, 1e-6) * 3) + 60
    swept, commanded, consecutive, steps, misses = 0.0, 0.0, 0, 0, 0
    stop = None
    worst = sampler.ikc.joint_limit_margin(q, arm)  # (margin, joint): the arc's closest approach to a joint limit
    while True:
        if sgn_joint * (swept - q_goal) >= -band_rad:
            stop = "joint_at_target"
            break
        if commanded >= ecfg.push_arc_lag_factor * max(abs(q_goal), 1e-3) + 2 * step_ang:
            stop = "door_lags"
            break
        if steps >= max_steps:
            stop = "max_steps"
            break
        pose = fk(q)
        remaining = signed_angle(radial(pose[0]), radial(p_t))
        d_ang = sgn * min(step_ang, max(abs(remaining), 0.25 * step_ang))
        wp = C.orbit_waypoints(pose, cen, C.quat_from_axis_angle(ax, d_ang), n=1, frame="world")[-1]
        step_m = abs(d_ang) * radius
        r_ = backend.solve_ik_path(f"{arm}_arm", [{frame: (th.as_tensor(wp[0]).float(), th.as_tensor(wp[1]).float())}], q_start=q, pos_tol=min(1e-4, 0.05 * step_m),
                                   rot_tol=5e-4, max_iters=20, check_collisions=False, sync_world=False, soft_pos_tol=0.25 * step_m, soft_rot_tol=2e-3)
        if r_.success and r_.q_traj is not None and len(r_.q_traj) > 0:
            q_cmd = r_.q_traj[-1].reshape(-1).float()
            dq = float(th.max(th.abs(q_cmd[idx] - q[idx])))
            if dq > mcfg.diff_ik_joint_speed * dt:
                q_cmd = q.clone()
                q_cmd[idx] = q[idx] + (r_.q_traj[-1].reshape(-1).float()[idx] - q[idx]) * (mcfg.diff_ik_joint_speed * dt / dq)
            q = q_cmd.clone()
            commanded += abs(d_ang)
            swept += d_ang
            consecutive = 0
            m_ = sampler.ikc.joint_limit_margin(q, arm)
            if m_[0] < worst[0]:
                worst = m_
        else:
            consecutive += 1
            misses += 1
            if consecutive >= 10:
                margin, jn = sampler.ikc.joint_limit_margin(q, arm)
                return False, (f"approach_untrackable: push replay - the closed-loop arc misses 10 differential-IK steps in a row at swept {swept:+.3f} of {q_to:+.3f} rad "
                               f"{f'(+{lag:.2f} lag allowance) ' if lag > 0.0 else ''}"
                               f"(nearest joint limit {margin:.3f} rad at {jn}; {getattr(r_, 'status', '')[:100]})"), rep
        steps += 1
    rep["arc"] = {"steps": steps, "swept_rad": round(swept, 4), "q_to_rad": round(q_to, 4), "stop": stop, "ik_misses": misses,
                  "min_limit_margin_rad": round(float(worst[0]), 4), "nearest_joint": worst[1]}
    if lag > 0.0:
        rep["arc"]["lag_allowance_rad"] = round(lag, 4)
    if stop != "joint_at_target":
        return False, f"approach_untrackable: push replay - the closed-loop arc stops at {stop} (swept {swept:+.3f} of {q_to:+.3f} rad)", rep
    # 3. the contact retreat's escape leg from the arc's end (execute_retreat)
    status, why, rep_r, _ = replay_contact_escape(sampler, arm, q, obj, supports, ignore, retreat_distance)
    if rep_r is not None:
        rep["retreat"] = rep_r
    if status == "none":
        return True, "arc replayed (no escape direction)", rep
    if status == "replayed":
        return True, "arc + retreat replayed", rep
    if status == "finished":
        return True, "arc + retreat replayed (stall finished)", rep
    if status == "partial":
        return True, "arc replayed, retreat partial", rep
    return False, why, rep


def replay_contact_escape(sampler, arm: str, q: th.Tensor, obj, supports: Sequence[Any], ignore: Sequence[Any], retreat_distance: float,
                          start: str = "the arc's end", escape_up_m: Optional[float] = None) -> Tuple[str, str, Optional[Dict[str, Any]], th.Tensor]:
    """The contact retreat's escape leg as ``execute_retreat`` runs it after a push / brace, from the configuration ``q`` the push left
    the arm in: radial off ``obj``'s centre in the support plane, then off its ``supports`` (``escape_direction``), the end probed for
    reach and shortened like ``_retreat_reach``; tracked with the retreat's joint-limit margin; a stall away from a joint limit finished
    by IK, a partial retreat accepted as the oracle accepts it. Collisions are not judged (a colliding leg is finished around by the
    oracle's planned finish, only the kinematics can strand it). Returns ``(status, why, report, q_end)``: ``status`` "none" (no escape
    direction - the oracle runs the recorded leg), "replayed", "finished", "partial" or "failed" (``why`` says where the leg stalls,
    leaving ``start``); ``q_end`` is where the leg leaves the arm. Step 3 of :func:`replay_arc_push`, shared with the retreat after the
    overhang push (oracle-fix 2026-11-05)."""
    import math

    from ispatialgym.oracle import execute as E
    from ispatialgym.oracle import motions as MO
    from ispatialgym.oracle import r1pro_model as M
    from ispatialgym.robot import conventions as C

    ecfg, mcfg = E.ExecuteConfig(), MO.MotionsConfig()
    backend, kit = sampler.kit.backend, sampler.kit
    frame = M.EEF_LINK[arm]
    idx = kit.robot.arm_control_idx[arm]

    def fk(q_):
        p, qq = kit.fk([frame], q_)[frame]
        return th.as_tensor(p).float(), th.as_tensor(qq).float()

    kit._ensure_synced()
    q = th.as_tensor(q).reshape(-1).float().clone()
    here = fk(q)
    n = E.escape_direction(None, here, supports)
    radial_d = None
    try:
        centre = th.as_tensor(obj.get_position_orientation()[0]).reshape(-1)[:3].float()
        radial_d = here[0] - centre
        if n is not None:
            radial_d = radial_d - n * float(th.dot(radial_d, n))
        radial_d = radial_d / float(th.linalg.norm(radial_d)) if float(th.linalg.norm(radial_d)) > 1e-3 else None
    except Exception:  # noqa: BLE001
        radial_d = None
    if radial_d is None and n is None:
        return "none", "", None, q
    d1 = radial_d if radial_d is not None else n
    out = ecfg.contact_retreat_escape_m
    far = max(float(retreat_distance), 2 * out)
    up = float(escape_up_m) if escape_up_m else out  # Retreat.escape_up_m, as execute_retreat reads it
    normal_out = n * up if (n is not None and radial_d is not None) else th.zeros(3)
    far_used, tried = far, []
    with sampler.ikc.ignoring(list(ignore)):
        for f in tuple(ecfg.retreat_reach_factors) or (1.0,):
            d = max(far * float(f), out)
            if tried and abs(d - tried[-1][0]) < 1e-6:
                break
            ok = kit.ik((here[0] + d1 * d + normal_out, here[1]), q, arm=arm) is not None
            tried.append((round(d, 4), ok))
            if ok:
                far_used = d
                break
        else:
            far_used = tried[-1][0]
    leg = [(here[0] + d1 * out, here[1]), (here[0] + d1 * far_used + normal_out, here[1])]
    rep = {"distance_m": far_used, "probe": tried}
    dense_r: List[Any] = []
    prev = here
    for w in leg:
        n_r = max(1, int(math.ceil(float(th.linalg.norm(w[0] - prev[0])) / mcfg.cartesian_step)))
        dense_r.extend(tuple(x) for x in C.line_waypoints(prev, w, n_r))
        prev = w
    # the retreat leg as arm_along runs it (free-space soft tolerance, the retreat's joint-limit margin); its collision veto is
    # left out - a colliding leg is finished around by the oracle's planned finish, only the kinematics can strand it
    r_leg = backend.solve_ik_path(f"{arm}_arm", [{frame: w} for w in dense_r], q_start=q, pos_tol=mcfg.diff_ik_pos_tol, rot_tol=mcfg.diff_ik_rot_tol,
                                  max_iters=mcfg.diff_ik_max_iters, check_collisions=False, sync_world=False, soft_pos_tol=mcfg.diff_ik_soft_pos_tol_free,
                                  soft_rot_tol=mcfg.diff_ik_soft_rot_tol, limit_margin=ecfg.retreat_limit_margin_rad)
    if r_leg.success:
        q_end = r_leg.q_traj[-1].reshape(-1).float() if r_leg.q_traj is not None and len(r_leg.q_traj) > 0 else q
        return "replayed", "", rep, q_end
    diag = dict(getattr(r_leg, "diagnostics", None) or {})
    k, stall = diag.get("fail_index"), diag.get("pos_err")
    at_limit = bool(diag.get("at_lower") or diag.get("at_upper"))
    q_stall = r_leg.q_traj[-1].reshape(-1).float() if r_leg.q_traj is not None and len(r_leg.q_traj) > 0 else q
    remaining = float(stall or 0.0) + sum(float(th.linalg.norm(th.as_tensor(dense_r[i][0]) - th.as_tensor(dense_r[i - 1][0]))) for i in range(max(int(k or 1), 1), len(dense_r)))
    rep.update({"stall_m": None if stall is None else round(float(stall), 4), "at_limit": at_limit, "fail_index": k, "n_dense": len(dense_r)})
    if diag.get("stage") == "diff_ik" and stall is not None and not at_limit and float(stall) <= ecfg.free_leg_finish_max_stall and remaining <= ecfg.free_leg_finish_max_remaining:
        # ApproachFinish without a witness: cuRobo's IK of the end pose from the stall, within the finish's joint excursion
        with sampler.ikc.ignoring(list(ignore)):
            q_fin = kit.ik(leg[-1], q_stall, arm=arm)
        if q_fin is not None and float(th.max(th.abs(th.as_tensor(q_fin).reshape(-1).float()[idx] - q_stall[idx]))) <= MO.ApproachFinish(None).max_excursion:
            rep["finished"] = True
            q_end = q_stall.clone()
            q_end[idx] = th.as_tensor(q_fin).reshape(-1).float()[idx]
            return "finished", "", rep, q_end
    moved = float(th.linalg.norm(fk(q_stall)[0] - here[0]))
    end_gap = float(th.linalg.norm(fk(q_stall)[0] - leg[-1][0]))
    if end_gap <= ecfg.retreat_partial_accept_m or moved >= out - 0.005:
        rep["partial"] = {"short_m": round(end_gap, 4), "moved_m": round(moved, 4)}
        return "partial", "", rep, q_stall
    return "failed", (f"approach_untrackable: push replay - the contact retreat's escape leg from {start} stalls after {moved * 1000:.0f} mm "
                      f"({end_gap * 1000:.0f} mm short; {r_leg.status[:160]})"), rep, q_stall


def push_keeps_recorded_pre(contact, pre_pose, first_contact_pose) -> bool:
    """Whether the oracle reaches a push's RECORDED pre-contact with its witness (``execute_push``): an arc push (``mid``), a pitched
    blade class (``push_blade_30`` and steeper) or a recorded pre-contact within ``push_pre_dist_m`` + 5 mm of the first contact pose
    (the recorded ``contact.init``) along the approach. A straight push's longer pre-contact is lowered to ``push_pre_dist_m``
    before it (``execute.push_lowered_pre``) and planned without a witness."""
    from ispatialgym.oracle import execute as E

    return E.push_lowered_pre(contact, pre_pose, first_contact_pose, E.ExecuteConfig()) is None


def push_pre_standoff(sampler, arm: str, q_pre: th.Tensor, pre_pose, supports: Sequence[Any], through: Sequence[Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """The oracle's reach of a push pre-contact next to its supports (``execute.reach_pre_pose``, owner 2026-10-12 item 7c; axial#02
    c3_k0 passed this check and failed the demo there). A pre-contact closer than ``pre_pose_standoff_skip_m`` to the supports (their
    AABBs, ``escape_direction`` / ``support_clearance``) goes through the standoff ``pre_pose_standoff_m`` off them when the standoff
    tracks from the witness ``q_pre`` (``standoff_configuration``: the oracle's densification and differential-IK settings, no
    collision check). When it does not, the oracle plans straight to the pre-contact with the witness, the supports present: accepted
    when the witness - or its IK refinement within ``witness_refine_max_jump`` (``Motions.arm_to``) - is collision-free with only
    ``through`` removed. Else the bare standoff: accepted when its pose has an IK solution with the supports present. Neither ->
    ``approach_untrackable``. Returns ``(ok, why, report)``; ``report["branch"]`` is the oracle's branch (witness / direct / planner)."""
    import math

    from ispatialgym.oracle import execute as E
    from ispatialgym.oracle import motions as MO
    from ispatialgym.oracle import r1pro_model as M
    from ispatialgym.robot import conventions as C

    ecfg, mcfg = E.ExecuteConfig(), MO.MotionsConfig()
    rep: Dict[str, Any] = {}
    pre = (th.as_tensor(pre_pose[0]).float(), th.as_tensor(pre_pose[1]).float())
    n = E.escape_direction(None, pre, supports)
    if n is None:
        return True, "no support near the pre-contact", rep
    clearance = E.support_clearance(None, pre, supports)
    rep["clearance_m"] = None if clearance is None else round(clearance, 4)
    if clearance is not None and clearance >= ecfg.pre_pose_standoff_skip_m:
        rep["branch"] = "skipped"
        return True, f"the pre-contact stands {clearance:.3f} m clear of its supports (planned directly)", rep
    above = E.height_above_supports(pre, supports) if ecfg.pre_pose_standoff_skip_above_m is not None else None
    if above is not None and above >= ecfg.pre_pose_standoff_skip_above_m:
        rep.update({"branch": "skipped", "above_m": round(above, 4)})
        return True, f"the pre-contact stands {above:.3f} m above its supports (planned directly)", rep
    kit = sampler.kit
    frame, idx = M.EEF_LINK[arm], kit.robot.arm_control_idx[arm]
    q_w = th.as_tensor(q_pre).reshape(-1).float().clone()
    standoff = (pre[0] + n * ecfg.pre_pose_standoff_m, pre[1])
    kit._ensure_synced()
    n_d = max(1, int(math.ceil(float(th.linalg.norm(standoff[0] - pre[0])) / mcfg.cartesian_step)))
    dense = [tuple(w) for w in C.line_waypoints(pre, standoff, n_d)]
    res = kit.backend.solve_ik_path(f"{arm}_arm", [{frame: w} for w in dense], q_start=q_w, pos_tol=mcfg.diff_ik_pos_tol, rot_tol=mcfg.diff_ik_rot_tol,
                                    max_iters=mcfg.diff_ik_max_iters, check_collisions=False, sync_world=False, soft_pos_tol=mcfg.diff_ik_soft_pos_tol_free)
    if res.success and res.q_traj is not None and len(res.q_traj) > 0:
        rep["branch"] = "witness"
        return True, f"the {ecfg.pre_pose_standoff_m:.2f} m standoff tracks from the pre-contact witness", rep
    rep["standoff"] = str(res.status)[:160]
    if sampler.ikc.collision_free_at(q_w, pre, arm, ignore=list(through)):
        rep["branch"] = "direct"
        return True, "the standoff does not track from the witness; the witness is a collision-free goal with the supports present", rep
    with sampler.ikc.ignoring(list(through)):
        q_ik = kit.ik(pre, q_w, arm=arm)
    if q_ik is not None:
        jump = float(th.max(th.abs(th.as_tensor(q_ik).reshape(-1).float()[idx] - q_w[idx])))
        rep["refined_jump_rad"] = round(jump, 3)
        if jump <= mcfg.witness_refine_max_jump:
            rep["branch"] = "direct"
            return True, f"the standoff does not track from the witness; its IK refinement ({jump:.2f} rad) is a collision-free goal with the supports present", rep
    with sampler.ikc.ignoring(list(through)):
        q_s = kit.ik(standoff, q_w, arm=arm)
    if q_s is not None:
        rep["branch"] = "planner"
        return True, "the standoff does not track from the witness and the witness collides with the supports present; the bare standoff has an IK solution", rep
    try:
        blame = sampler.ikc._collision_blame(q_w, list(through))
    except Exception as e:  # noqa: BLE001 - diagnostic only
        blame = f"no blame ({e!r})"[:120]
    return False, (f"approach_untrackable: push pre-contact {clearance:.3f} m from its supports - the {ecfg.pre_pose_standoff_m:.2f} m standoff does not track "
                   f"from the witness ({rep['standoff'][:100]}), the witness is no collision-free goal with the supports present ({blame[:120]}"
                   + (f"; its IK refinement lies {rep['refined_jump_rad']:.2f} rad away" if "refined_jump_rad" in rep else "")
                   + ") and the standoff pose has no IK solution"), rep


log = logging.getLogger("ispatialgym.sampling.v2")

# ----------------------------------------------------------------------------------------------------------------
# train-fast (2026-10-31): run statistics, partial world syncs, attachment reuse
# ----------------------------------------------------------------------------------------------------------------

#: the statistics of the last :func:`run` in this process (``core`` reads them into the attempt's timing summary): ``program_check_s``,
#: ``n_actions``, ``n_poses``, ``n_ik_single``, ``n_chains``, ``n_sync``, ``sync_s``, ``n_attach``, ``early_gate_poses``, ``rejected``
last_stats: Optional[Dict[str, Any]] = None
#: the counters of the run in progress (None outside :func:`run`: the helpers below then count nothing)
_STATS: Optional[Dict[str, Any]] = None
_FAST_LOGGED: set = set()


def _fast_note(name: str, msg: str) -> None:
    """One INFO line the first time a train-fast path is taken in this process."""
    if name not in _FAST_LOGGED:
        _FAST_LOGGED.add(name)
        log.info("program check (train-fast %s): %s", name, msg)


def _count(key: str, n: int = 1) -> None:
    if _STATS is not None:
        _STATS[key] = _STATS.get(key, 0) + n


def _partial_sync_ok(kit, objects: Sequence[Any]) -> bool:
    """Whether ``RobotKit.sync_world(only_objects=objects)`` is equivalent to a full sync for the step that moved ``objects``: the
    cuRobo world is loaded and every collision mesh of every object is already in it (``R1ProCuRobo.update_world``'s partial path only
    re-poses meshes it holds - a mesh it does not hold would be skipped silently where the full sync adds it). A fake backend without
    these attributes (tests) gets the full sync, as before."""
    backend = getattr(kit, "backend", None)
    names = getattr(backend, "_world_mesh_names", None)
    if not objects or not getattr(backend, "_world_loaded", False) or not isinstance(names, (set, frozenset)):
        return False
    # a mesh the backend disabled (``_disabled_obstacles``: attached to a hand, or unseen by the last full pass) is re-enabled by a FULL
    # sync and left alone by the partial one: such an object gets the full sync, as before
    disabled = getattr(backend, "_disabled_obstacles", None) or set()
    for obj in objects:
        if obj is None:
            return False
        for link in (getattr(obj, "links", None) or {}).values():
            for cm in (getattr(link, "collision_meshes", None) or {}).values():
                name = getattr(cm, "prim_path", None)
                if name not in names or name in disabled:
                    return False
    return True


def _reset_graph_buffers(kit) -> None:
    """What the full ``update_world`` does after a pose change and its partial path does not: drop the PRM graph planners' cached graphs,
    so a joint-space plan after the step (``check_repose`` with ``sync_world=False``) does not re-use seed paths validated against the
    old poses."""
    entries = getattr(getattr(kit, "backend", None), "_entries", None)
    if not isinstance(entries, dict):
        return
    for e in entries.values():
        gp = getattr(getattr(e, "planner", None), "graph_planner", None)
        if gp is not None:
            try:
                gp.reset_buffer()
            except Exception:  # noqa: BLE001 - a planner without a buffer: nothing to drop
                pass


def _sync(kit, objects: Optional[Sequence[Any]] = None) -> None:
    """``kit.sync_world()`` for the program check's kinematic advances. ``objects`` = the sim objects whose poses / joints the step just
    changed; with ``SYNC_ONLY_OBJECTS`` (train-fast 2026-10-31) their collision meshes alone are re-posed (``RobotKit.sync_world(
    only_objects=...)``: pose updates of meshes the loaded world already holds, nothing enabled, disabled or added) instead of the full
    pass over the scene's meshes (0.5-13 s) - the robot root never moves during the check (the kit falls back to a full sync by itself
    when it drifted), no ignore / attach context is open where the advances run (nothing a full sync would re-enable), and the graph
    buffers are reset as the full sync resets them. The full sync stays when ``objects`` is None (the moved set is unknown), when the
    switch is off, when the world is not loaded or when one of the objects' meshes is not in it (:func:`_partial_sync_ok`). Counted
    and timed into :data:`last_stats`."""
    t0 = time.time()
    try:
        if objects is not None and fast_on("SYNC_ONLY_OBJECTS") and _partial_sync_ok(kit, objects):
            _fast_note("SYNC_ONLY_OBJECTS", "the advances re-pose the moved objects' meshes only (kit.sync_world(only_objects=...))")
            kit.sync_world(only_objects=list(objects))
            _reset_graph_buffers(kit)
        else:
            kit.sync_world()
    finally:
        _count("n_sync")
        if _STATS is not None:
            _STATS["sync_s"] = _STATS.get("sync_s", 0.0) + (time.time() - t0)


def _attach_ctx(kit, arm: Optional[str], obj: Any, eef_in_object=None, nested: bool = False):
    """``navigation.attached`` for the checks inside (counted into :data:`last_stats`). ``nested``: the same (arm, object, eef_in_object)
    an enclosing context of this action already attached - the carried base poses of a navigate / transition inside the action's own
    attached context (the lift-end solve and the drive check, per base pose). With ``ATTACH_REUSE`` (train-fast 2026-10-31) that
    attachment is kept when the backend holds this object on this arm: no second teleport into the hand (a no-op there), ``og.sim.
    step_physics()`` and sphere fit of the same geometry (the fit is seeded: the same spheres), and no detach at the nested exit (which
    today drops the enclosing attachment too, until the next nested enter re-fits it). Off, or with nothing attached: the plain context,
    as before. ``navigation.attached`` itself does not no-op on a nested identical attach - it re-fits and its exit detaches."""
    from ispatialgym.sampling.v2.navigation import attached, attached_checks_enabled

    if nested and obj is not None and arm is not None and fast_on("ATTACH_REUSE"):
        cur = (getattr(getattr(kit, "backend", None), "_attached", None) or {}).get(arm)
        if isinstance(cur, dict) and cur.get("obj") is obj:
            _fast_note("ATTACH_REUSE", "a nested attach of the object already attached to the arm is skipped (the enclosing context's fit is kept)")
            return nullcontext()
    if obj is not None and arm is not None and attached_checks_enabled():
        _count("n_attach")
    return attached(kit, arm, obj, eef_in_object=eef_in_object)


#: which sidecar blocks each subtask's program consumes (oracle doc §6.2; used by the stub and as a sanity list)
PROGRAM_BLOCKS: Dict[str, Tuple[str, ...]] = {
    "camera_relocalization": ("stances.init", "goal", "paths"),
    "object_scaling": ("stances.init", "grasps"),
    "mental_rotation": ("stances.init", "grasps"),
    "disembedding": ("stances.init", "stances.manipulation", "stances.placement", "transitions", "grasps", "paths"),
    "configuration_matching-block_design": ("stances.init", "stances.manipulation", "transitions", "blocks"),
    "configuration_matching-articulation_large_scale": ("stances.init", "stances.manipulation", "articulation"),
    "configuration_matching-articulation_small_scale": ("stances.init", "stances.manipulation", "articulation"),
    # brace optional only without a joint cell after the second base pose (owner 2026-09-16 17:10 item 2; oracle program_attachment)
    "configuration_matching-attachment": ("stances.init", "stances.manipulation", "transitions", "grasps"),
    "alignment-world_vertical": ("stances.init", "stances.manipulation", "contacts.push"),  # brace optional (owner 2026-10-11: round targets push alone)
    "alignment-axial": ("stances.init", "stances.manipulation", "transitions", "contacts.push"),  # brace optional (owner 2026-09-16 08:10; F6 9b03aa4)
    "inter_object_relation_satisfaction-ontop_small_scale": ("stances.init", "stances.manipulation", "transitions", "grasps"),
    "inter_object_relation_satisfaction-under_small_scale": ("stances.init", "stances.manipulation", "transitions", "grasps"),
    "inter_object_relation_satisfaction-inside_small_scale": ("stances.init", "stances.manipulation", "transitions", "grasps"),
    "inter_object_relation_satisfaction-nextto_small_scale": ("stances.init", "stances.manipulation", "transitions", "grasps"),
    "inter_object_relation_satisfaction-touching_small_scale": ("stances.init", "stances.manipulation", "transitions", "grasps", "contacts.push"),  # brace optional (owner 2026-09-16 17:10; same rule as axial)
    "inter_object_relation_satisfaction-ontop_large_scale": ("stances.init", "stances.manipulation", "stances.placement", "transitions", "grasps", "paths"),
    "inter_object_relation_satisfaction-under_large_scale": ("stances.init", "stances.manipulation", "stances.placement", "transitions", "grasps", "paths"),
    # registered 2026-10-12 ahead of their modes (each mode may adjust its list): inside as ontop; touching as ontop + the push record
    # of under's push-under; nextto has no grasp (the base pushes the floor-standing target)
    "inter_object_relation_satisfaction-inside_large_scale": ("stances.init", "stances.manipulation", "stances.placement", "transitions", "grasps", "paths"),
    "inter_object_relation_satisfaction-touching_large_scale": ("stances.init", "stances.manipulation", "stances.placement", "transitions", "grasps", "paths",
                                                                "contacts.push"),
    "inter_object_relation_satisfaction-nextto_large_scale": ("stances.init", "stances.manipulation", "stances.placement", "transitions", "paths", "contacts.push"),
}


class ProgramCheckError(Exception):
    def __init__(self, action: str, message: str):
        super().__init__(f"{action}: {message}")
        self.action = action


def _expand():
    """``oracle.programs.expand`` when F6 has landed, else None."""
    try:
        from ispatialgym.oracle import programs as P  # type: ignore

        return getattr(P, "expand", None)
    except Exception:
        return None


def lift_away(g) -> Optional[th.Tensor]:
    """The grasp record's ``lift_away`` (world xyz, m; oracle-fix 2026-11-04 round 4, touching#03's shelf tip release): the oracle's ``Lift`` adds it to
    the straight-up segment (the book leaning on a side board is lifted away from it). None when the record has none."""
    ex = getattr(g, "extras", None)
    if ex is None:
        ex = getattr(g, "extra", None)
    v = (ex or {}).get("lift_away") if isinstance(ex, dict) else None
    return None if v is None else th.as_tensor([float(x) for x in v], dtype=th.float32)


def lift_end_pose(g, carry_height: Optional[float] = None):
    """The lift's end pose of a grasp record: ``carry_height`` (default the record's) straight above the grasp pose, plus :func:`lift_away`."""
    h = float(g.carry_height or 0.0) if carry_height is None else float(carry_height)
    end = th.as_tensor(g.grasp[0], dtype=th.float32) + th.tensor([0.0, 0.0, h])
    away = lift_away(g)
    if away is not None:
        end = end + away
    quat = th.as_tensor(g.grasp[1], dtype=th.float32)
    turn = lift_turn(g)
    if turn is not None:  # i7-t03 round 2 (d): the straightening lift turns the hand (and the book) by this world rotation
        from ispatialgym.sampling import geometry as G

        quat = G.qmul(turn, quat)
    return end, quat


def lift_turn(g) -> Optional[th.Tensor]:
    """The grasp record's ``lift_turn`` (world rotation xyzw; i7-t03 round 2 (d), touching_03's straightening lift, the oracle's ``Lift.turn``):
    applied to the hand over the lift so the leaning book ends upright. None when the record has none."""
    ex = getattr(g, "extras", None)
    if ex is None:
        ex = getattr(g, "extra", None)
    v = (ex or {}).get("lift_turn") if isinstance(ex, dict) else None
    return None if v is None else th.as_tensor([float(x) for x in v], dtype=th.float32)


def subtask_blocks(subtask: str) -> Tuple[str, ...]:
    if subtask in PROGRAM_BLOCKS:  # ``...-inside_large_scale`` also starts with ``...-inside``
        return PROGRAM_BLOCKS[subtask]
    for k, v in PROGRAM_BLOCKS.items():
        if subtask == k or subtask.startswith(k):
            return v
    return ("stances.init",)


# ----------------------------------------------------------------------------------------------------------------
# structural stub (until F6)
# ----------------------------------------------------------------------------------------------------------------


def _has_block(d: Dict[str, Any], dotted: str) -> bool:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur or cur[part] in (None, [], {}):
            return False
        cur = cur[part]
    return True


def check_blocks(subtask: str, sidecar: Dict[str, Any]) -> List[str]:
    """Structural validation of a sidecar dict for ``subtask``: the blocks its program consumes exist, every stance has a
    base pose and trunk, transitions / paths reference existing stances and paths satisfy ``PathRecord.schema_ok``,
    every grasp has its four eef poses. Returns the list of problems (empty = ok)."""
    problems: List[str] = []
    blocks = subtask_blocks(subtask)
    if subtask.startswith("inter_object_relation_satisfaction-touching_large_scale") and not sidecar.get("grasps"):
        # oracle-fix 2026-11-03: the sidecar's own relaxation (``oracle.sidecar``): a floor start has nothing to pick - the push chain
        # alone (push_touching: no grasp / placement / path) or the base push against the partner's face (base_push: no grasp)
        kind = ((sidecar.get("contacts") or {}).get("push") or {}).get("kind")
        if kind == "push_touching":
            blocks = tuple(b for b in blocks if b not in ("stances.placement", "grasps", "paths"))
        elif kind == "base_push":
            blocks = tuple(b for b in blocks if b != "grasps")
    elif subtask.startswith("inter_object_relation_satisfaction-touching_large_scale") and str((sidecar.get("grasps") or [{}])[0].get("place_kind") or "") in ("lean", "surface"):
        # oracle-fix 2026-11-04 round 4 (touching_large#02): the sidecar's lean relaxation - the lean release has no push; oracle-fix 2026-11-06:
        # nor the release on the partner's board (``generation.touching_place: "surface"``)
        blocks = tuple(b for b in blocks if b != "contacts.push")
    elif subtask.startswith("inter_object_relation_satisfaction-touching_small_scale") and str((sidecar.get("grasps") or [{}])[0].get("place_kind") or "") == "shelf_tip":
        # oracle-fix 2026-11-04 round 4 (touching#03): the sidecar's shelf tip relaxation - the book tips onto the partner, no push
        blocks = tuple(b for b in blocks if b != "contacts.push")
    for blk in blocks:
        if not _has_block(sidecar, blk):
            problems.append(f"missing block {blk}")
    stances = sidecar.get("stances") or {}
    for name, st in stances.items():
        if not (isinstance(st.get("base_pose2d"), list) and len(st["base_pose2d"]) == 3):
            problems.append(f"stance {name}: base_pose2d")
        if "translate" not in st or "tilt" not in st:
            problems.append(f"stance {name}: trunk")
    for tr in sidecar.get("transitions") or []:
        for k in ("from", "to"):
            if tr.get(k) not in stances:
                problems.append(f"transition {tr.get('from')}->{tr.get('to')}: unknown stance {tr.get(k)!r}")
        if not isinstance(tr.get("stand_off"), (int, float)):
            problems.append("transition without stand_off")
        if list(tr.get("order") or []) != ["drive-back", "trunk", "drive-in"]:
            problems.append(f"transition order {tr.get('order')}")
    for p in sidecar.get("paths") or []:
        if not PathRecord.schema_ok(p):
            problems.append(f"path {p.get('from')}->{p.get('to')}: schema")
        elif p["from"] not in stances or p["to"] not in stances:
            problems.append(f"path {p['from']}->{p['to']}: unknown stance")
    for g in sidecar.get("grasps") or []:
        for k in ("pre_grasp", "grasp"):
            if not g.get(k):
                problems.append(f"grasp of {g.get('object')}: missing {k}")
        if g.get("arm") not in ("left", "right"):
            problems.append(f"grasp of {g.get('object')}: arm {g.get('arm')!r}")
    for kind, c in (sidecar.get("contacts") or {}).items():
        if not c.get("pre"):
            problems.append(f"contact {kind}: missing pre")
        if kind == "push" and not (c.get("init") and c.get("goal")):
            problems.append("push contact: missing init / goal")
        if kind == "brace" and not c.get("contact"):
            problems.append("brace contact: missing contact")
    return problems


# ----------------------------------------------------------------------------------------------------------------
# target extraction
# ----------------------------------------------------------------------------------------------------------------


def _is_pose(v) -> bool:
    if isinstance(v, dict):
        return "position" in v and "orientation" in v
    if hasattr(v, "position") and hasattr(v, "orientation"):  # oracle.sidecar.Pose
        return True
    if isinstance(v, (tuple, list)) and len(v) == 2:
        try:
            return len(v[0]) == 3 and len(v[1]) == 4
        except TypeError:
            return False
    return False


def _as_pose(v):
    if isinstance(v, dict):
        return pose_from_sidecar(v)
    if hasattr(v, "position") and hasattr(v, "orientation"):
        return th.as_tensor(list(v.position), dtype=th.float32), th.as_tensor(list(v.orientation), dtype=th.float32)
    return th.as_tensor(v[0], dtype=th.float32), th.as_tensor(v[1], dtype=th.float32)


def iter_pose_targets(targets: Any, prefix: str = "") -> Iterable[Tuple[str, List[Any]]]:
    """Flatten an action's ``targets()`` into ``(key, [poses])``: a single pose -> one-element list, a list of poses ->
    the waypoint list (checked with the waypoint policy). Keys containing ``base`` / ``path`` / ``trunk`` / ``joint`` /
    ``q`` are not eef poses and are skipped here."""
    skip = ("base", "path", "trunk", "joint", "gripper", "finger")
    if _is_pose(targets):
        yield prefix or "pose", [_as_pose(targets)]
        return
    if isinstance(targets, dict):
        for k, v in targets.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if any(s in str(k).lower() for s in skip):
                continue
            yield from iter_pose_targets(v, key)
    elif isinstance(targets, (list, tuple)) and targets and all(_is_pose(x) for x in targets):
        yield prefix or "waypoints", [_as_pose(x) for x in targets]


# ----------------------------------------------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------------------------------------------


class _SamplerState:
    """The ``State`` of oracle doc §3.6 backed by the sampler's world at init: ``object_pose(name)`` -> (pos, quat);
    ``eef_pose(arm)`` -> where the previous arm action left the hand (owner 2026-09-25), so ``targets()`` of a ``Place``
    builds the carry leg the oracle tracks (``Place._carry_leg``) and a ``Lift`` / ``Retreat`` leaves from that pose (within
    the witness residual of the recorded one); it raises when the arm's configuration is unknown (after a travel, after a
    push / brace) so ``targets()`` keeps its recorded-pose fallbacks."""

    def __init__(self, sampler, tracker=None):
        self.sampler = sampler
        self.tracker = tracker

    def object_pose(self, name: str):
        obj = self.sampler.objects[name]
        p, q = obj.get_position_orientation()
        return p.float().clone(), q.float().clone()

    def eef_pose(self, arm: str):
        q = (getattr(self.tracker, "arm_end_q", None) or {}).get(arm) if self.tracker is not None else None
        if q is None:
            raise KeyError(f"no configuration recorded for the {arm} arm (the previous action was an empty-handed travel / a contact)")
        # the arm joints where the last motion left them, on the CURRENT stance's base / trunk (a carried travel keeps the arm
        # and moves the base: the configuration was recorded at the departing stance)
        stance = self.tracker.stance_obj if getattr(self.tracker, "stance_obj", None) is not None else getattr(self.tracker.sidecar, "stances", {}).get(self.tracker.stance)
        if stance is not None:
            q_here = self.sampler.ikc.q_locked(stance).clone()
            idx = self.sampler.kit.robot.arm_control_idx[arm]
            q_here[idx] = th.as_tensor(q, dtype=th.float32)[idx]
            q = q_here
        return arm_eef_pose(self.sampler, q, arm)


def arm_eef_pose(sampler, q: th.Tensor, arm: str):
    """FK eef pose ``(pos, quat)`` of ``arm`` at the sampler-frame configuration ``q``."""
    from ispatialgym.oracle import r1pro_model as M
    from ispatialgym.sampling import geometry as G

    eef = sampler.kit.fk([M.EEF_LINK[arm]], G.t(q))[M.EEF_LINK[arm]]
    return G.t(eef[0]).float().clone(), G.t(eef[1]).float().clone()


def _sidecar_object(sd: Dict[str, Any]):
    """``oracle.sidecar.Sidecar`` built from the draft dict (``tro_sha256`` placeholder before the TRO exists)."""
    from ispatialgym.oracle import sidecar as SC  # type: ignore

    d = dict(sd)
    if not isinstance(d.get("tro_sha256"), str):
        d["tro_sha256"] = "0" * 64
    return SC.Sidecar.from_dict(d)


#: the hand is tried lifted by these (m) above where the last motion left it when the travel at that configuration collides
DEPARTURE_LIFTS_M = (0.05, 0.10)
#: whole-body margin of the travel sweeps at the departure configuration (the recorded paths / transitions were swept at 1 cm)
DEPARTURE_SWEEP_MARGIN = 0.01


def _travel_group(actions: Sequence[Any], i: int) -> List[int]:
    """Indices of the consecutive ``navigate`` / ``transition`` actions starting at ``i`` (one travel between two arm actions)."""
    out: List[int] = []
    j = i
    while j < len(actions) and getattr(actions[j], "name", "") in ("navigate", "transition"):
        out.append(j)
        j += 1
    return out


def _draft_record(sidecar: SidecarDraft, action):
    """The sampler's own record behind an oracle-side ``Navigate`` / ``Transition`` (matched by stance pair and kind)."""
    name = getattr(action, "name", "")
    if name == "navigate" and getattr(action, "path", None) is not None:
        return next((p for p in sidecar.paths if str(p.frm) == str(action.path.frm) and str(p.to) == str(action.path.to)), None)
    if name == "transition" and getattr(action, "transition", None) is not None:
        tr = action.transition
        dep = bool(getattr(tr, "departure", False))
        return next((t for t in sidecar.transitions if str(t.frm) == str(tr.frm) and str(t.to) == str(tr.to) and bool(t.departure) == dep), None)
    return None


def _start_contacts(action, sampler, name: str, key: str) -> List[Any]:
    """Objects a leg is in contact with where it starts, beyond the ignore sets of its action: the object a retreat leaves (the
    ``Retreat`` action's ``object``; a ``Place``'s retreat key leaves the placed object)."""
    out: List[Any] = []
    obj_name = getattr(action, "object", None) if (name == "retreat" or (name == "place" and str(key) == "retreat")) else None
    if obj_name and obj_name in getattr(sampler, "objects", {}):
        out.append(sampler.objects[obj_name])
    return out


def _record_arms_q(action) -> Dict[str, th.Tensor]:
    """``arms_q`` (arm -> 7 joints) an oracle-side path / transition record carries (written by :func:`choose_departure_arms`)."""
    rec = getattr(action, "path", None) if getattr(action, "name", "") == "navigate" else getattr(action, "transition", None)
    aq = ((getattr(rec, "extra", None) or {}).get("arms_q") if rec is not None else None) or {}
    out: Dict[str, th.Tensor] = {}
    for a, v in aq.items():
        try:
            t_ = th.as_tensor([float(x) for x in v], dtype=th.float32)
        except (TypeError, ValueError):
            continue
        if t_.numel() == 7:
            out[a] = t_
    return out


def _record_preshape(action) -> Optional[float]:
    """oracle-fix 2026-11-06 (the shelf back-off): the finger opening an oracle-side transition record pre-shapes to before its arms move
    (``preshape_m``: arm -> m, the widest; written by ``IORSSmall.record_shelf_backoff``), None without it."""
    rec = getattr(action, "transition", None)
    shape = ((getattr(rec, "extra", None) or {}).get("preshape_m") if rec is not None else None) or {}
    vals = []
    for v in shape.values() if isinstance(shape, dict) else ():
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    return max(vals) if vals else None


class _released_placed:
    """Context: every object a ``Place`` before ``upto`` set down (and the tracker did not advance already, i.e. not a block) stands
    at its release pose - ``place ∘ inv(T_obj→eef)`` of its grasp record - for the travel sweeps, so the hand hovering where it
    let go is checked against the object it let go of (the sampler's world has it at its init pose otherwise). Restored on exit."""

    def __init__(self, sidecar: SidecarDraft, sampler, actions: Sequence[Any], upto: int, tracker: "_Tracker"):
        self.sidecar, self.sampler, self.actions, self.upto, self.tracker = sidecar, sampler, actions, upto, tracker
        self.saved: Dict[str, Tuple[th.Tensor, th.Tensor]] = {}

    def __enter__(self):
        from ispatialgym.sampling import geometry as G

        placed: Dict[str, Any] = {}
        for a in self.actions[: self.upto]:
            if getattr(a, "name", "") == "place" and getattr(a, "object", None):
                placed[a.object] = a
            elif getattr(a, "name", "") == "grasp" and getattr(a, "object", None) in placed:
                placed.pop(a.object, None)  # picked up again later (a re-grasp): the last place decides
        for obj_name, a in placed.items():
            if obj_name in self.tracker.moved or obj_name == self.tracker.carried:
                continue
            obj = self.sampler.objects.get(obj_name)
            g = self.sidecar.grasp_of(obj_name)
            if obj is None or g is None or getattr(g, "eef_in_object", None) is None or getattr(a, "place", None) is None:
                continue
            try:
                eef = (th.as_tensor(list(a.place.position), dtype=th.float32), th.as_tensor(list(a.place.orientation), dtype=th.float32))
                pose = G.compose(eef, G.invert((G.t(g.eef_in_object[0]), G.t(g.eef_in_object[1]))))
                p0, q0 = obj.get_position_orientation()
                self.saved[obj_name] = (p0.clone(), q0.clone())
                obj.set_position_orientation(pose[0], pose[1])
                obj.keep_still()
            except Exception as e:  # noqa: BLE001 - the sweep then runs with the object at its init pose, as before
                log.debug("departure arms: %s not moved to its release pose (%r)", obj_name, e)
                self.saved.pop(obj_name, None)
        if self.saved:
            # train-fast 2026-10-31: the released objects moved, nothing else (the robot stands still) - re-pose their meshes alone
            _sync(self.sampler.kit, [self.sampler.objects[n] for n in self.saved if self.sampler.objects.get(n) is not None])
        return self

    def __exit__(self, *exc):
        for obj_name, (p0, q0) in self.saved.items():
            obj = self.sampler.objects.get(obj_name)
            if obj is not None:
                obj.set_position_orientation(p0, q0)
                obj.keep_still()
        if self.saved:
            _sync(self.sampler.kit, [self.sampler.objects[n] for n in self.saved if self.sampler.objects.get(n) is not None])  # train-fast 2026-10-31
        self.saved.clear()
        return False


def _sweep_travel_record(sidecar: SidecarDraft, sampler, action, override: Dict[str, th.Tensor]) -> Tuple[bool, str, Dict[str, th.Tensor]]:
    """Re-check one recorded travel leg with the acting arm(s) at ``override`` (arm -> 7 joints) instead of the stance posture:
    a path at its recorded trunk over its strides (a face_travel record: its schedule, ``NAV.record_sweep_poses``), an arrival transition's sweep / drive-in at its recorded s_T, a departure
    transition's drive-back / sweep at its recorded s_T (``NAV.transition`` / ``NAV.departure_transition`` asked for exactly the
    recorded stand-off). Returns ``(ok, why, arms swept with)``."""
    from ispatialgym.sampling.v2 import navigation as NAV
    from ispatialgym.sampling.v2.ik import stance_arm_vectors

    kit = sampler.kit
    name = getattr(action, "name", "")
    rec = _draft_record(sidecar, action)
    if rec is None:
        return False, f"{name}: no sampler record", {}
    frm_st, to_st = sidecar.stances.get(str(rec.frm)), sidecar.stances.get(str(rec.to))
    if frm_st is None or to_st is None:
        return False, f"{name}: stance missing", {}
    if name == "navigate":
        r, l, f = stance_arm_vectors(kit, frm_st)
        q_arms = {"right": r, "left": l}
        q_arms.update(override)
        trunk = tuple(float(v) for v in rec.trunk) if rec.trunk is not None else (float(frm_st.translate), float(frm_st.tilt))
        # the strides + the tracker's yaw lag on the turns; oracle-fix 2026-11-06: a face_travel record as its schedule (what the oracle drives -
        # the polyline was swept here for an empty-handed face_travel leg too, up to 0.15 m off the drive)
        poses = NAV.record_sweep_poses(rec)
        ok, bad = NAV.path_collision_free(kit, poses, trunk, q_arms=q_arms, fingers=f, margin=DEPARTURE_SWEEP_MARGIN)
        what = "face_travel schedule" if getattr(rec, "style", None) == "face_travel" else "strides"
        return ok, "" if ok else f"path {rec.frm} -> {rec.to} collides at swept pose {bad}/{len(poses)} ({what} + yaw-lag envelope)", q_arms
    # a transition: the arriving stance's arms with the override on top (what NAV.transition sweeps with)
    r, l, f = stance_arm_vectors(kit, to_st)
    q_arms = {"right": r, "left": l}
    q_arms.update(override)
    if rec.departure:
        if rec.trunk_to is None:
            return False, f"departure {rec.frm} -> {rec.to}: no travel trunk", q_arms
        res = NAV.departure_transition(kit, frm_st, to_st, (float(rec.trunk_to[0]), float(rec.trunk_to[1])), arm_q_override=override, min_standoff=float(rec.stand_off),
                                       margin=DEPARTURE_SWEEP_MARGIN, arms_stance=to_st)
    else:
        dep = next((t for t in sidecar.transitions if str(t.frm) == str(rec.frm) and str(t.to) == str(rec.to) and t.departure), None)
        trunk_from = (float(dep.trunk_to[0]), float(dep.trunk_to[1])) if dep is not None and dep.trunk_to is not None else None
        s = float(to_st.stand_off if to_st.stand_off is not None else 0.0)
        extra = max(NAV.TRANSITION_EXTRA_MAX, float(rec.stand_off) - s + 0.05)
        res = NAV.transition(kit, frm_st, to_st, arm_q_override=override, trunk_from=trunk_from, min_standoff=float(rec.stand_off), margin=DEPARTURE_SWEEP_MARGIN, extra_max=extra)
    if res is None:
        return False, f"{'departure' if rec.departure else 'transition'} {rec.frm} -> {rec.to}: no s_T >= the recorded {float(rec.stand_off):.2f} is free", q_arms
    if abs(float(res.stand_off) - float(rec.stand_off)) > 1e-6:
        return False, f"{'departure' if rec.departure else 'transition'} {rec.frm} -> {rec.to}: the recorded s_T {float(rec.stand_off):.2f} collides (first free {float(res.stand_off):.2f})", q_arms
    return True, "", q_arms


def _arrival_holds_next_contact(sidecar: SidecarDraft, sampler, actions: Sequence[Any], group: Sequence[int]) -> Optional[Tuple[str, str]]:
    """``(arm, action name)`` when the travel ``group`` arrives at a stance whose recorded configuration already puts ``arm`` at the
    pre-contact of the push / brace right after the travel (FK of the stance's arms at the action's ``pre``), else None - the stance
    records the arm at its next contact and the transition was swept that way (the push-close re-stance, ``articulation_small``)."""
    follow: List[Any] = []
    for act in actions[group[-1] + 1:] if group else []:
        if getattr(act, "name", "") not in ("push", "brace"):
            break
        follow.append(act)
    if not follow:
        return None
    rec = _draft_record(sidecar, actions[group[-1]])
    to_st = sidecar.stances.get(str(rec.to)) if rec is not None else None
    if to_st is None:
        return None
    q_to = sampler.ikc.q_locked(to_st)
    for act in follow:
        c = getattr(act, "contact", None)
        pre = getattr(c, "pre", None)
        if pre is None or getattr(c, "arm", None) is None:
            continue
        pose = (th.tensor([float(v) for v in pre.position], dtype=th.float32), th.tensor([float(v) for v in pre.orientation], dtype=th.float32))
        if sampler.ikc.reaches(q_to, pose, str(c.arm)):
            return str(c.arm), str(act.name)
    return None


def choose_departure_arms(sidecar: SidecarDraft, sampler, actions: Sequence[Any], group: Sequence[int], tracker: "_Tracker") -> Optional[Dict[str, Any]]:
    """Owner 2026-09-24 ("a teleoperator lifts the arm only as much as needed, then drives"): the arms an empty-handed travel
    (the consecutive ``navigate`` / ``transition`` actions ``group``) is driven with. Candidates, in order: the acting arm(s)
    exactly where the previous motion left them (``_Tracker.arm_end_q``, the tracked retreat end), then lifted straight up by
    :data:`DEPARTURE_LIFTS_M`; the first that keeps every leg of the travel whole-body collision-free (``_sweep_travel_record``,
    with the released objects at their release poses) is recorded as ``arms_q`` on each leg's record - the oracle then re-poses
    an arm only when it deviates from that - and the tracker keeps the arm's configuration through the travel. None when no
    candidate passes: the records stay as swept (the stance postures), the oracle re-poses as before."""
    from ispatialgym.oracle import r1pro_model as M

    kit = sampler.kit
    ends = {a: q for a, q in tracker.arm_end_q.items() if a in ("left", "right")}
    if not ends or tracker.arm_q:  # nothing left where it is / a braced arm: the canonical postures
        return None
    held = _arrival_holds_next_contact(sidecar, sampler, actions, group)
    if held is not None:
        # the arriving stance's recorded arms are the next contact's pre-pose, not a posture: the arm goes straight there
        log.info("program check: travel at indices %s keeps the arriving stance's recorded arms (the %s arm there is at the pre-contact of the %s that follows)",
                 list(group), held[0], held[1])
        return {"arms": sorted(ends), "choice": f"arriving stance's arms ({held[0]} at the {held[1]} pre-contact)", "legs": [int(j) for j in group], "tried": []}
    candidates: List[Tuple[str, Dict[str, th.Tensor]]] = []
    base = {a: q[kit.robot.arm_control_idx[a]].clone().float() for a, q in ends.items()}
    candidates.append(("where the arm is", base))
    for dz in DEPARTURE_LIFTS_M:
        lifted: Dict[str, th.Tensor] = {}
        for a, q in ends.items():
            try:
                eef = kit.fk([M.EEF_LINK[a]], q)[M.EEF_LINK[a]]
                p_from = (th.as_tensor(eef[0], dtype=th.float32), th.as_tensor(eef[1], dtype=th.float32))
                p_to = (p_from[0] + th.tensor([0.0, 0.0, float(dz)]), p_from[1])
                q_up, why = sampler.ikc.track_segment(q, p_from, p_to, a)
            except Exception as e:  # noqa: BLE001
                q_up, why = None, repr(e)
            if q_up is None:
                log.debug("departure arms: %s lifted %.2f m not trackable (%s)", a, dz, why)
                lifted = {}
                break
            lifted[a] = q_up[kit.robot.arm_control_idx[a]].clone().float()
        if lifted:
            candidates.append((f"lifted {dz:.2f} m", lifted))
    whys: List[str] = []
    with _released_placed(sidecar, sampler, actions, group[0], tracker):
        for label, override in candidates:
            swept: Dict[int, Dict[str, th.Tensor]] = {}
            failed = None
            for j in group:
                ok, why, q_arms = _sweep_travel_record(sidecar, sampler, actions[j], override)
                if not ok:
                    failed = f"{label}: {why}"
                    break
                swept[j] = q_arms
            if failed is not None:
                whys.append(failed)
                continue
            # record the swept arms on every leg (the sampler's record -> the sidecar; the oracle-side object -> this run's base checks)
            for j in group:
                arms_q = {a: [round(float(v), 5) for v in q.tolist()] for a, q in swept[j].items()}
                rec = _draft_record(sidecar, actions[j])
                act = actions[j]
                if rec is not None:
                    if getattr(act, "name", "") == "navigate":
                        rec.arms_q = arms_q
                    else:
                        rec.extras["arms_q"] = arms_q
                        rec.extras["arms_source"] = label
                side = getattr(act, "path", None) if getattr(act, "name", "") == "navigate" else getattr(act, "transition", None)
                if side is not None:
                    if getattr(side, "extra", None) is None:
                        side.extra = {}
                    side.extra["arms_q"] = arms_q
                act._keeps_arms = True
            legs = ", ".join(f"{getattr(actions[j], 'name', '')} {getattr(_draft_record(sidecar, actions[j]), 'frm', '?')}->{getattr(_draft_record(sidecar, actions[j]), 'to', '?')}" for j in group)
            log.info("program check: travel [%s] driven with the %s arm %s (owner 2026-09-24: no re-pose to the stance posture)", legs, "/".join(sorted(ends)), label)
            return {"arms": sorted(ends), "choice": label, "legs": [int(j) for j in group], "tried": whys}
    log.info("program check: travel at indices %s keeps the stance postures (the arm as left / lifted collides: %s)", list(group), "; ".join(whys)[:400])
    return {"arms": sorted(ends), "choice": "stance posture (re-pose)", "legs": [int(j) for j in group], "tried": whys}


def run(variant: str, sidecar: SidecarDraft, sampler, subtask: Optional[str] = None) -> Dict[str, Any]:
    """§3.5 program check. ``sampler`` supplies ``ikc`` (``IKChecker``), ``kit``, ``objects`` (bddl instance -> sim
    object), ``support_objects()`` (ignored for final grasp / contact poses) and ``subtask``. Returns a report
    ``{"mode": "expand"|"stub", "actions": [{"name", "stance", "carried", "n_poses"}], ...}``; raises
    ``Reject("program_check_failed", action=<name>)`` (``sim_utils.Reject``) on the first failing target."""
    from ispatialgym.sampling.sim_utils import Reject

    import os

    global _STATS, last_stats

    subtask = subtask or getattr(sampler, "subtask", variant)
    sd = sidecar.to_dict()
    problems = check_blocks(subtask, sd)
    if problems:
        raise Reject("program_check_failed", "; ".join(problems), action="blocks")
    # ``ISG_PROGRAM_CHECK=stub`` (development switch while F6 is a draft): structural check only, recorded as such
    expand = None if os.environ.get("ISG_PROGRAM_CHECK", "").lower() == "stub" else _expand()
    if expand is None:
        return {"mode": "stub", "problems": [], "blocks": list(subtask_blocks(subtask))}
    try:
        sc = _sidecar_object(sd)
        actions = expand(variant, sc)
    except Exception as e:  # a program that cannot expand is a sampler-side bug too
        raise Reject("program_check_failed", f"expand failed: {e!r}", action="expand")
    report: Dict[str, Any] = {"mode": "expand", "actions": []}
    # train-fast 2026-10-31 (PC_TIMING, always on): the run's counters - IK singles / chains, world syncs and their time, attachments -
    # collected by the helpers while it runs, published as ``program_check.last_stats`` (and ``report["stats"]``) and one INFO line
    t_run = time.time()
    prev_stats, _STATS = _STATS, {"program_check_s": 0.0, "n_actions": len(actions), "n_actions_checked": 0, "n_poses": 0, "n_ik_single": 0, "n_chains": 0,
                                  "n_sync": 0, "sync_s": 0.0, "n_attach": 0, "early_gate_poses": 0, "rejected": True}
    tracker = _Tracker(sidecar)
    state = _SamplerState(sampler, tracker)
    # joint state carried through the program (2026-09-19, two-flap cartons): a push with an ``articulation`` record leaves its
    # joint at q_to until that arm's retreat (the pressed flap), a base-follow leaves its joint where the follow ends - the
    # following actions are checked in that world (0919-204505: the second pinch's approach was checked against the flat flap)
    joint_saved: Dict[str, Any] = {}

    def _set_joint(obj_name, joint, q):
        obj = sampler.objects.get(obj_name) if obj_name else None
        if obj is None or joint not in getattr(obj, "joints", {}):
            return
        if obj_name not in joint_saved:
            joint_saved[obj_name] = obj.get_joint_positions().clone()
        qs = obj.get_joint_positions().clone()
        qs[list(obj.joints).index(joint)] = float(q)
        obj.set_joint_positions(qs)
        obj.keep_still()
        _sync(sampler.kit, [obj])  # train-fast 2026-10-31: only this object's links moved

    def _carry_joint_state(action):
        nm = getattr(action, "name", "")
        if nm == "push":
            a = (getattr(getattr(action, "contact", None), "extra", None) or {}).get("articulation")
            if a and a.get("q_to") is not None:
                _set_joint(getattr(action.contact, "object", None), a.get("joint"), a["q_to"])
        elif nm == "retreat":
            after = getattr(action, "after", None)
            if getattr(after, "name", "") == "push":
                a = (getattr(getattr(after, "contact", None), "extra", None) or {}).get("articulation")
                # the pressed link springs back when the fist leaves - unless the other hand holds the object by then (the lever's
                # lift, two-flap v4: the main hand has pinched the raised flap before the fist slides out) or the record says the
                # brake HOLDS the link where the fist leaves it (``holds``: the push-close segments - 0920-171442: the re-stance
                # transition was checked with the flap sprung back open and the arriving fist ran into it)
                if a and a.get("q_from") is not None and not a.get("holds") and tracker.carried != getattr(after.contact, "object", None):
                    _set_joint(getattr(after.contact, "object", None), a.get("joint"), a["q_from"])
        elif nm == "base_follow":
            art = getattr(action, "articulation", None)
            if art is not None:
                stop = (getattr(art, "extra", None) or {}).get("follow_to")
                k_end = float(stop) if stop is not None else 1.0
                _set_joint(art.object, art.joint, float(art.q_init) + k_end * (float(art.q_goal) - float(art.q_init)))

    travel_handled: set = set()
    try:
        if fast_on("EARLY_FRAME_GATE") and os.environ.get("ISG_FINGERTIP_FRAME_CHECK", "on").lower() not in ("off", "0", "false"):
            # train-fast 2026-10-31: the camera-frame gate of every action whose poses are known now, before the first IK (the per-action
            # gate below stays and simply passes for them)
            _fast_note("EARLY_FRAME_GATE", "the fingertip frame gate runs over the static targets of every action before any IK")
            _STATS["early_gate_poses"] = _early_frame_gate(sidecar, sc, sampler, actions)
        # oracle-fix 2026-11-05 (ISG_PC_DIAG_ALL=1, diagnostic, off by default): a rejected action is logged and the check goes on with the
        # next one (the tracker advanced past it as if it had passed), the first rejection raised at the end - one long attempt (the
        # hardback's 3500 s) then names every action the program check would reject, not only the first
        diag_all = os.environ.get("ISG_PC_DIAG_ALL", "").lower() in ("1", "on", "true")
        diag_fails: List[Tuple[int, str, Any]] = []
        for i, action in enumerate(actions):
            try:
                name = getattr(action, "name", type(action).__name__)
                if name in ("navigate", "transition") and i not in travel_handled:
                    # owner 2026-09-24: an EMPTY-HANDED travel after an arm action is driven with the arm where the last motion left it
                    # (or lifted a little) when that sweeps free - recorded as ``arms_q`` on its legs (``choose_departure_arms``)
                    grp = _travel_group(actions, i)
                    travel_handled.update(grp)
                    carries = tracker.carried is not None or any(getattr(getattr(actions[j], "path", None), "carried", None) or getattr(actions[j], "carried", None) for j in grp)
                    if not carries and tracker.arm_end_q and os.environ.get("ISG_DEPARTURE_ARMS", "on").lower() not in ("off", "0", "false"):
                        try:
                            dep = choose_departure_arms(sidecar, sampler, actions, grp, tracker)
                        except Reject:
                            raise
                        except Exception as e:  # noqa: BLE001 - a failed selection keeps the canonical postures (the records as swept)
                            log.warning("program check: departure arms not chosen for the travel at %s (%r): the stance postures stay", grp, e)
                            dep = None
                        if dep is not None:
                            report.setdefault("departure_arms", []).append(dep)
                # i7-cork: the hand-over (the other hand's grasp of the carried male, the giver's release retreat) - the male posed at its
                # recorded hand-over pose in the world, both hands' chains with it ignored, the other hand locked (_Tracker.handover_phase)
                ho_phase = tracker.handover_phase(action)
                if ho_phase == "grasp":
                    tracker.before_handover_grasp(action, sampler)
                try:
                    targets = action.targets(sc, state)
                except Exception as e:
                    raise Reject("program_check_failed", f"targets() failed: {e!r}", action=name)
                stance_name, carried = tracker.before(action)
                carried_arm = tracker.carrying_arm()
                if ho_phase is not None:
                    carried, carried_arm = None, None
                locked = tracker.locked_arms(action)
                # block design (owner 2026-09-19): the acting arm's IK is seeded from where the previous action left it - the lift
                # tracked from the grasp witness, the reorient from the lift's end, the carry / place from the reorient's end - as
                # the oracle's arm_along runs them (pod runs 0919-140558 / 151544: the reorient checked from a fresh exact IK passed
                # while the oracle, tracking from the grasp's branch, jumped 0.23 rad / stalled 14 mm)
                q_prev = tracker.arm_end_q.get((getattr(action, "arms", None) or ["right"])[0]) if getattr(sidecar, "blocks", None) else None
                if q_prev is None and name == "place" and carried and carried in (getattr(sampler, "carry_arm_q", None) or {}):
                    # a place after a CARRY (IORS large-scale, owner 2026-09-20): the object arrives in the hand at the carry configuration the
                    # sampler recorded (``carry_arm_q``), so the pre-place / place witnesses are solved in that branch - the oracle reaches the
                    # pre-place from there with a bounded joint swing (0920-135625: a witness 2.09 rad away made the drop 'plan_failed')
                    st_seed = tracker.stance_obj if tracker.stance_obj is not None else sidecar.stances.get(stance_name)
                    if st_seed is not None:
                        arm_c = (getattr(action, "arms", None) or ["right"])[0]
                        q_prev = sampler.ikc.q_locked(st_seed).clone()
                        # the tracked end of the straight carry leg when the task recorded it (``carry_end_q``), else the carry configuration
                        src = (getattr(sampler, "carry_end_q", None) or {}).get(carried, sampler.carry_arm_q[carried])
                        q_prev[sampler.kit.robot.arm_control_idx[arm_c]] = th.as_tensor(src, dtype=th.float32)
                if name == "push" and (getattr(getattr(action, "contact", None), "extra", None) or {}).get("q_hint") is not None:
                    # a press recorded with the branch that leaves room for the other hand's pinch (two-flap v2, 2026-09-19): the IK seed
                    q_prev = th.as_tensor(action.contact.extra["q_hint"], dtype=th.float32)
                # owner 2026-09-24: where the previous arm action left THIS arm (every family) - the leg that starts there is tracked from it
                q_end_prev = tracker.arm_end_q.get((getattr(action, "arms", None) or ["right"])[0])
                from ispatialgym.sampling.v2 import tabletop as TT

                # of-v2maps 2026-11-07 (v2open2: 'approach: support_policy: finger_penetration' 21 / 20 / 6 program-check rejections): a grasp
                # action's approach and grasp are policed at the oracle's pre-shape of its recorded grasp (tabletop.record_policy_opening)
                ho_ignore = sampler.ikc.ignoring([sampler.objects[action.object]]) if ho_phase is not None and getattr(action, "object", None) in sampler.objects else nullcontext()
                with TT.record_policy_opening(sampler, getattr(action, "grasp", None) if name == "grasp" else None), ho_ignore:
                    n, witnesses = check_action_targets(sidecar, sampler, action, targets, stance_name, carried, locked_arms=locked, stance_obj=tracker.stance_obj,
                                                        q_seed_prev=q_prev, q_end_prev=q_end_prev,
                                                        held_since_init=tracker.carried is not None and tracker.carried_arm is None,
                                                        locked_fingers=tracker.locked_fingers(action), carried_arm=carried_arm)
                end_q = witnesses.pop("__arm_end_q__", None)
                partial_leg = witnesses.pop("__partial__", None)
                repose_note = witnesses.pop("__repose__", None)
                giver = tracker.handover_giver
                overpath_note = witnesses.pop("__overpath__", None)
                tracker.after(action, witnesses, sampler)
                tracker.note_arm_end(action, end_q)
                if ho_phase is not None:
                    tracker.after_handover(action, ho_phase, giver, witnesses, sampler)
                _carry_joint_state(action)
                entry = {"index": i, "name": name, "arms": list(getattr(action, "arms", ()) or []), "objects": list(getattr(action, "objects", ()) or []),
                         "stance": stance_name, "carried": carried, "n_poses": n, "witnesses": witnesses}
                if locked:
                    entry["locked_arms"] = sorted(locked)  # the other arm(s) held at their contact configuration while these poses were solved
                if partial_leg is not None:
                    entry["partial_leg"] = partial_leg
                if repose_note is not None:
                    entry["repose"] = repose_note
                if overpath_note is not None:
                    entry["overpath"] = overpath_note
                report["actions"].append(entry)
                _STATS["n_actions_checked"] += 1
                _STATS["n_poses"] += int(n)
            except Reject as e_diag:
                if not diag_all:
                    raise
                name_d = getattr(action, "name", type(action).__name__)
                diag_fails.append((i, name_d, e_diag))
                log.warning("program check DIAG (ISG_PC_DIAG_ALL): action %d %s rejected: %s - going on with the next action", i, name_d, str(e_diag)[:400])
                try:
                    tracker.after(action, {}, sampler)
                    tracker.note_arm_end(action, None)
                except Exception as e_adv:  # noqa: BLE001 - a diagnostic run keeps going
                    log.warning("program check DIAG: the tracker could not advance past action %d (%r)", i, e_adv)
        if diag_fails:
            log.warning("program check DIAG: %d of %d actions rejected: %s", len(diag_fails), len(actions),
                        "; ".join(f"{i_} {n_}: {str(e_)[:200]}" for i_, n_, e_ in diag_fails))
            raise diag_fails[0][2]
        _STATS["rejected"] = False
    finally:
        tracker.restore_moved(sampler)  # block records advanced through the program go back to their init poses (the TRO is written from the live world)
        restored_joints: List[Any] = []
        for obj_name, qs in joint_saved.items():
            obj = sampler.objects.get(obj_name)
            if obj is not None:
                obj.set_joint_positions(qs)
                obj.keep_still()
                restored_joints.append(obj)
        if joint_saved:
            _sync(sampler.kit, restored_joints)  # train-fast 2026-10-31: the restored articulated objects alone
        # PC_TIMING: the run's statistics (``last_stats`` for core, ``report["stats"]``) and one INFO line
        _STATS["program_check_s"] = round(time.time() - t_run, 3)
        _STATS["sync_s"] = round(float(_STATS["sync_s"]), 3)
        st, _STATS = _STATS, prev_stats
        last_stats = st
        report["stats"] = dict(st)
        try:
            sampler.last_stats = dict(st)  # core.py reads ``self.last_stats`` on the sampler when present (its agent's contract, 2026-10-31)
        except Exception:  # noqa: BLE001 - a read-only sampler stand-in (tests)
            pass
        log.info("program_check: %d actions, %d poses, %d singles, %d chains, %d syncs (%.1fs), %d attach, total %.1fs%s", st["n_actions"], st["n_poses"],
                 st["n_ik_single"], st["n_chains"], st["n_sync"], st["sync_s"], st["n_attach"], st["program_check_s"],
                 "" if not st["rejected"] else f", rejected after {st['n_actions_checked']} actions")
    # the recorded joint solutions (28-D, arm joints + trunk + fingers meaningful) per checked end pose: D5 verifies them as
    # witnesses (FK + collision) instead of re-solving the randomised IK
    sidecar.ik_witnesses = [{"action": e["index"], "name": e["name"], "stance": e["stance"], "arm": (e["arms"] or ["right"])[0], "carried": e["carried"],
                            "targets": e["witnesses"]} for e in report["actions"] if e["witnesses"]]
    return report


#: witness keys of the chain a library pinch decides the arm configuration of (``tabletop.held_turns``): the pick and the held chain
#: this check tracks from the grasp witness - lift, in-hand reorient, carry, pre-place, descent, place. The place's release retreat is
#: not one of them: the oracle's retreat keeps its own joint-limit box (``execute.retreat_limit_margin_rad``, never past its start).
TURN_KEYS = {"grasp": ("pre_grasp", "approach", "grasp"), "lift": ("lift",), "reorient": ("waypoints",), "place": ("carry", "pre_place", "descent", "place")}


def turn_chain(report: Optional[Dict[str, Any]], obj: str) -> Dict[str, Any]:
    """The pick and held chain of ``obj`` in a :func:`run` report: ``margin`` (the smallest ``joint_limit_margin_rad`` of its
    :data:`TURN_KEYS` witnesses from its grasp to its place; None without them), ``joint`` / ``where`` (``<action>.<key>``) of that
    witness, ``q_pre`` (the pre-grasp witness, 28 values) and ``actions`` (the actions while ``obj`` is held)."""
    out: Dict[str, Any] = {"margin": None, "joint": None, "where": None, "q_pre": None, "actions": []}
    holding = False
    for e in (report or {}).get("actions") or []:
        name = e.get("name")
        if name == "grasp" and obj in (e.get("objects") or []):
            holding = True
        elif not holding:
            continue
        elif e.get("carried") != obj:
            break
        else:
            out["actions"].append(name)
        wit = e.get("witnesses") or {}
        if name == "grasp" and isinstance(wit.get("pre_grasp"), dict):
            out["q_pre"] = wit["pre_grasp"].get("q")
        for key in TURN_KEYS.get(name, ()):
            w = wit.get(key)
            m = w.get("joint_limit_margin_rad") if isinstance(w, dict) else None
            if m is not None and (out["margin"] is None or float(m) < out["margin"]):
                out.update(margin=float(m), joint=w.get("joint_limit_nearest"), where=f"{name}.{key}")
    return out


class _Tracker:
    """Where the robot is and what it holds while the program runs: ``Navigate`` -> ``path.to``, ``Transition`` ->
    ``transition.to``; ``Grasp`` picks up ``object`` (held until ``Place``); ``Brace`` parks its arm at the contact
    configuration until that arm's ``Retreat`` (``arm_q``: the witnesses of the other arm's actions in between are
    solved with the bracing arm locked there — owner 2026‑09‑16 22:55 item 3: the horizontal push witnesses solved
    with the brace arm at the stance posture put the push arm in a branch that runs through the braced arm,
    ``left_realsense_link`` vs ``right_arm_link7`` 13 mm along every path, demo 0916‑212037)."""

    def __init__(self, sidecar: SidecarDraft):
        self.stance = "init"
        init = sidecar.stances.get("init")
        # the held-object tasks start with the object in the hand (§6.3): the init stance records ``carried``
        self.carried: Optional[str] = init.carried if init is not None else None
        #: the arm holding ``carried`` once a grasp action set it (a held-object init: looked up in the sidecar's grasps)
        self.carried_arm: Optional[str] = None
        self.sidecar = sidecar
        #: arm -> its 7 arm-joint values while it braces (from the brace action's ``contact`` witness)
        self.arm_q: Dict[str, th.Tensor] = {}
        #: arm -> the finger opening a WIDE brace holds while locked (its ``fingers_opening_m``; i7-braceopen 2026-11-07) - a closed
        #: brace has no entry (its hand keeps the acting arm's opening, as before)
        self.arm_fingers: Dict[str, float] = {}
        #: arm -> the (sampler-frame, 28-D) configuration the previous arm action ended in (block design chaining); reset by
        #: every base motion
        self.arm_end_q: Dict[str, th.Tensor] = {}
        #: where the robot stands after a hybrid ``base_follow`` (the base moved off the manipulation stance): the in-memory
        #: stance the following actions (the retreat) are checked from, until the next navigate / transition
        self.stance_obj: Optional[Stance] = None
        #: sim objects moved by :meth:`advance_block` (name -> init (pos, quat)), restored by :meth:`restore_moved`
        self.moved: Dict[str, Tuple[th.Tensor, th.Tensor]] = {}
        #: i7-cork: the arm that handed the carried object over (until its release retreat)
        self.handover_giver: Optional[str] = None

    # -- i7-cork: the hand-over (owner decision 2026-11-08 4.2 (i)) -------------------------------------------------------------------
    def handover_phase(self, action) -> Optional[str]:
        """``"grasp"``: a grasp of the object the OTHER arm carries (the hand-over); ``"release"``: the giving arm's retreat after it;
        None otherwise."""
        name = getattr(action, "name", "")
        arms = list(getattr(action, "arms", ()) or [])
        if name == "grasp" and arms and self.carried is not None and getattr(action, "object", None) == self.carried \
                and self.carried_arm is not None and self.carried_arm != arms[0]:
            return "grasp"
        if name == "retreat" and arms and self.handover_giver is not None and arms[0] == self.handover_giver:
            return "release"
        return None

    def before_handover_grasp(self, action, sampler) -> None:
        """The sim object at its recorded hand-over pose (``grasps[*].handover.object_pose``; restored by :meth:`restore_moved`) and the
        giving arm locked at the configuration its last action left it in (its carry to the hand-over)."""
        g = getattr(action, "grasp", None)
        ex = getattr(g, "extra", None) if isinstance(getattr(g, "extra", None), dict) else (getattr(g, "extras", None) or {})
        pose = ((ex or {}).get("handover") or {}).get("object_pose")
        name = getattr(action, "object", None)
        if sampler is not None and pose and name in getattr(sampler, "objects", {}):
            obj = sampler.objects[name]
            if name not in self.moved:
                p0, q0 = obj.get_position_orientation()
                self.moved[name] = (p0.clone(), q0.clone())
            obj.set_position_orientation(th.tensor([float(v) for v in pose[0]]), th.tensor([float(v) for v in pose[1]]))
            obj.keep_still()
            _sync(sampler.kit, [obj])
        giver = self.carried_arm
        q_end = self.arm_end_q.get(giver) if giver else None
        if sampler is not None and giver and q_end is not None:
            self.arm_q[giver] = q_end[sampler.kit.robot.arm_control_idx[giver]].clone()
        self.handover_giver = giver

    def after_handover(self, action, phase: str, giver: Optional[str], witnesses: Optional[Dict[str, Any]], sampler) -> None:
        """After the hand-over grasp: the giver unlocked, the taker locked at its grasp witness for the giver's release retreat; after that
        retreat: the taker unlocked (it carries the object from there)."""
        arms = list(getattr(action, "arms", ()) or [])
        if phase == "grasp":
            if giver:
                self.arm_q.pop(giver, None)
                self.handover_giver = giver
            w = (witnesses or {}).get("grasp")
            if arms and sampler is not None and w is not None and w.get("q") is not None:
                self.arm_q[arms[0]] = th.tensor(w["q"], dtype=th.float32)[sampler.kit.robot.arm_control_idx[arms[0]]].clone()
        elif phase == "release":
            if self.carried_arm:
                self.arm_q.pop(self.carried_arm, None)
            self.handover_giver = None

    def advance_block(self, action, sampler) -> None:
        """Block design (owner 2026-09-19): after the intermediate ``Place`` of a flip-in-place block (``table_flip.place``) the
        sim object is moved to ``table_flip.block_pose_after`` (turned, on its own cell), so the re-grasp — ``Grasp[cube]``
        derives its pose from the live block — is checked on the block the program leaves; after the final ``Place`` to the
        record's ``goal_pose`` (the world the later blocks are checked in). Restored by :meth:`restore_moved`."""
        name = getattr(action, "object", None)
        blocks = getattr(self.sidecar, "blocks", None) or []
        rec = next((b for b in blocks if isinstance(b, dict) and b.get("instance") == name), None)
        if rec is None or sampler is None or name not in getattr(sampler, "objects", {}):
            return
        place = getattr(action, "place", None)
        target = None
        tf = rec.get("table_flip") if isinstance(rec.get("table_flip"), dict) else None
        if tf is not None and place is not None and tf.get("place") and tf.get("block_pose_after"):
            p_mid = tf["place"].get("position") if isinstance(tf["place"], dict) else None
            if p_mid is not None and float(th.linalg.norm(th.tensor([float(v) for v in place.position]) - th.tensor([float(v) for v in p_mid]))) < 1e-3:
                target = tf["block_pose_after"]
        # the FINAL place advances the block to its recorded goal pose (``blocks[].goal_pose``, the sampler's world after the
        # place): the later blocks' grasps / lifts / reorients see the placed blocks where they are, not at their emptied init cells
        # (0919-164732 attempt 0: a set-down retreat 'had no solution' with a placed block still standing 11 cm away at its init
        # cell). The adjacency artefact of the coarse cube sphere fit at the next block's place - the reason the final place was
        # not advanced before - is handled by ``blocks[].ignore_place`` (the sampler's placed-neighbour rule, ``_ignore_for``)
        if target is None and place is not None and isinstance(rec.get("goal_pose"), dict) and isinstance(rec.get("place"), dict):
            p_fin = rec["place"].get("position")
            if p_fin is not None and float(th.linalg.norm(th.tensor([float(v) for v in place.position]) - th.tensor([float(v) for v in p_fin]))) < 1e-3:
                target = rec["goal_pose"]
        if not (isinstance(target, dict) and target.get("position") and target.get("orientation")):
            return
        obj = sampler.objects[name]
        if name not in self.moved:
            p0, q0 = obj.get_position_orientation()
            self.moved[name] = (p0.clone(), q0.clone())
        obj.set_position_orientation(th.tensor([float(v) for v in target["position"]]), th.tensor([float(v) for v in target["orientation"]]))
        obj.keep_still()
        _sync(sampler.kit, [obj])  # train-fast 2026-10-31: the advanced block alone

    def advance_pick_push(self, action, sampler) -> None:
        """The overhang pick's push (``contacts.push_pick``, kind ``pick_overhang``; owner 2026-10-13 ~03:15 item 4): the pushed object is
        moved to the record's ``post_push_pose`` (where the sampler settled it), so the grasp is checked at the overhang as the oracle
        takes it (at the object's current pose). Restored by :meth:`restore_moved`."""
        contact = getattr(action, "contact", None)
        name = getattr(contact, "object", None)
        post = (getattr(contact, "extra", None) or {}).get("post_push_pose")
        if sampler is None or name not in getattr(sampler, "objects", {}) or not isinstance(post, dict) or post.get("position") is None:
            log.warning("program check: the pick push of %s has no post_push_pose - the world is not advanced", name)
            return
        obj = sampler.objects[name]
        if name not in self.moved:
            p0, q0 = obj.get_position_orientation()
            self.moved[name] = (p0.clone(), q0.clone())
        obj.set_position_orientation(th.tensor([float(v) for v in post["position"]]), th.tensor([float(v) for v in post["orientation"]]))
        obj.keep_still()
        _sync(sampler.kit, [obj])  # train-fast 2026-10-31: the pushed object alone

    def advance_base_push(self, action, sampler) -> None:
        """A base push (IORS nextto large-scale): the robot ends at the push's recorded ``end_stance`` and the pushed object
        ``distance_m`` further along the push direction, so the retreat is checked from there against the object where the push
        leaves it (restored by :meth:`restore_moved`)."""
        end = getattr(action, "end_stance", None)
        if end in self.sidecar.stances:
            self.stance = end
            self.stance_obj = None
        contact = getattr(action, "contact", None)
        name = getattr(contact, "object", None)
        if contact is None or sampler is None or name not in getattr(sampler, "objects", {}):
            return
        d = th.tensor([float(v) for v in contact.direction])
        d[2] = 0.0
        if float(th.linalg.norm(d)) < 1e-6:
            return
        # the tracker's STANCE changed above, not the sim robot (the checks pose the base through the virtual base joints): the pushed
        # object is the only mesh that moved (train-fast 2026-10-31)
        _sync(sampler.kit, [self._shift(name, sampler.objects[name], d / th.linalg.norm(d) * float(contact.distance_m), sampler)])

    def _shift(self, name: str, obj, delta: th.Tensor, sampler):
        """Moves ``obj`` by ``delta`` (its init pose remembered for :meth:`restore_moved`); returns ``obj`` (the caller syncs)."""
        p0, q0 = obj.get_position_orientation()
        if name not in self.moved:
            self.moved[name] = (p0.clone(), q0.clone())
        obj.set_position_orientation(p0 + delta, q0)
        obj.keep_still()
        return obj

    def restore_moved(self, sampler) -> None:
        if not self.moved or sampler is None:
            return
        restored: List[Any] = []
        for name, (p0, q0) in self.moved.items():
            obj = sampler.objects.get(name)
            if obj is not None:
                obj.set_position_orientation(p0, q0)
                obj.keep_still()
                restored.append(obj)
        self.moved.clear()
        _sync(sampler.kit, restored)  # train-fast 2026-10-31: the objects put back alone

    def note_arm_end(self, action, end_q: Optional[th.Tensor]) -> None:
        """Remember where an arm action left the acting arm (``check_action_targets``' raw end witness); a base motion forgets it."""
        name = getattr(action, "name", "")
        if name in ("navigate", "transition", "base_follow", "set_torso"):
            if getattr(action, "_keeps_arms", False):
                return  # owner 2026-09-24: the record carries the arms the drive was swept at - the arm stays where it is
            # owner 2026-09-25: a carried navigate / transition keeps the configuration of the arm HOLDING the object - the oracle
            # freezes that arm through carried legs (execute_navigate re-poses empty arms only, move_idle_arms skips holding arms) -
            # so the carry -> pre-place -> descent -> place -> retreat chain is tracked from it for IORS large-scale, touching and
            # disembedding too, not seeded; the other arms are dropped
            keep = None
            if self.carried is not None and name in ("navigate", "transition"):
                arm_c = self.carrying_arm()
                keep = (arm_c, self.arm_end_q[arm_c]) if arm_c is not None and arm_c in self.arm_end_q else None
            self.arm_end_q.clear()
            if keep is not None:
                self.arm_end_q[keep[0]] = keep[1]
            return
        arms = list(getattr(action, "arms", ()) or [])
        if not arms:
            return
        if end_q is not None:
            self.arm_end_q[arms[0]] = end_q.clone()
        else:
            self.arm_end_q.pop(arms[0], None)  # a push / brace leaves the arm at its contact, not at a recorded end: nothing to continue from

    def carrying_arm(self) -> Optional[str]:
        """The arm holding ``carried``: the grasp action's arm, else the sidecar's grasp record of the object (a held-object init)."""
        if self.carried is None:
            return None
        if self.carried_arm is not None:
            return self.carried_arm
        for g in getattr(self.sidecar, "grasps", None) or []:
            if getattr(g, "object", None) == self.carried and getattr(g, "arm", None):
                return str(g.arm)
        return None

    def locked_arms(self, action) -> Dict[str, th.Tensor]:
        """The arms held at a contact configuration while ``action`` runs (never the acting arm itself)."""
        acting = set(getattr(action, "arms", ()) or [])
        return {a: q for a, q in self.arm_q.items() if a not in acting}

    def locked_fingers(self, action) -> Dict[str, float]:
        """The finger openings of the arms :meth:`locked_arms` holds while ``action`` runs - a wide brace's ``fingers_opening_m`` (the
        oracle keeps the stop open through the other arm's push and retreat); closed braces are not listed."""
        locked = self.locked_arms(action)
        return {a: f for a, f in self.arm_fingers.items() if a in locked}

    def before(self, action) -> Tuple[str, Optional[str]]:
        name = getattr(action, "name", "")
        carried = self.carried
        if name == "navigate":
            carried = getattr(getattr(action, "path", None), "carried", None) or self.carried
        elif name == "transition":
            carried = getattr(action, "carried", None) or self.carried
        stance = self.stance
        if name == "navigate" and getattr(action, "path", None) is not None:
            stance = action.path.to
        elif name == "transition" and getattr(action, "transition", None) is not None:
            stance = action.transition.to
        elif name == "base_follow" and getattr(action, "stance", None) is not None:
            # the action's stance by NAME when it is a recorded one (two-flap v2: "manipulation" and "manipulation2" may share the
            # base pose and differ in trunk only - the base-pose match picked the wrong one, 0919-235259); else by base pose
            nm = getattr(action.stance, "name", None)
            if nm in self.sidecar.stances:
                stance = nm
            else:
                best = None
                for k, v in self.sidecar.stances.items():
                    if list(v.base_pose2d) == list(action.stance.base_pose2d):
                        same_trunk = abs(float(v.translate) - float(action.stance.translate)) < 1e-6 and abs(float(v.tilt) - float(action.stance.tilt)) < 1e-6
                        if best is None or same_trunk:
                            best = k
                            if same_trunk:
                                break
                if best is not None:
                    stance = best
        return stance, carried

    def after(self, action, witnesses: Optional[Dict[str, Any]] = None, sampler=None) -> None:
        name = getattr(action, "name", "")
        if name == "navigate" and getattr(action, "path", None) is not None:
            self.stance = action.path.to
            self.stance_obj = None
        elif name == "transition" and getattr(action, "transition", None) is not None:
            self.stance = action.transition.to
            self.stance_obj = None
        elif name == "base_follow" and witnesses and witnesses.get("follow_end"):
            # hybrid follow: the base ends where the recorded path ends (``articulation_hybrid.check_program_follow``)
            from ispatialgym.sampling.v2.tasks import articulation_hybrid as AH

            self.stance_obj = AH.end_stance_from_witness(witnesses["follow_end"])
        elif name == "base_push":
            self.advance_base_push(action, sampler)
            second = getattr(action, "second", None)
            w2 = (witnesses or {}).get("second.contact")
            if second is not None and sampler is not None and w2 and w2.get("q") is not None:
                # two hands: the second hand stays on the object until its own retreat, as a brace does
                self.arm_q[second.arm] = th.tensor(w2["q"], dtype=th.float32)[sampler.kit.robot.arm_control_idx[second.arm]].clone()
        if name == "grasp":
            # a handle grasp does not carry anything: the articulated object stays in the world and only its handle link
            # moves with the hand along the recorded joint trajectory (large articulation, Rs_int drawer bamfsz_1: the
            # whole cabinet was attached to the gripper and every base-follow handle pose failed)
            self.carried = getattr(action, "object", None) if getattr(action, "kind", "library") != "handle" else None
            self.carried_arm = (list(getattr(action, "arms", ()) or []) or [None])[0] if self.carried is not None else None
        elif name == "place":
            self.carried = None
            self.carried_arm = None
            self.advance_block(action, sampler)
        elif name == "push":
            if str((getattr(getattr(action, "contact", None), "extra", None) or {}).get("kind") or "") == "pick_overhang":
                self.advance_pick_push(action, sampler)
        arms = list(getattr(action, "arms", ()) or [])
        if name == "brace" and arms and witnesses and sampler is not None:
            # the brace hand stays at its contact until its own retreat: its solved contact configuration locks the arm for
            # every action of the other arm in between (the push and the push arm's retreat)
            w = witnesses.get("contact") or witnesses.get("approach")
            if w is not None and w.get("q") is not None:
                q = th.tensor(w["q"], dtype=th.float32)
                self.arm_q[arms[0]] = q[sampler.kit.robot.arm_control_idx[arms[0]]].clone()
                wide = brace_open_fingers(action, name, None)
                if wide is not None:
                    self.arm_fingers[arms[0]] = wide
                else:
                    self.arm_fingers.pop(arms[0], None)
        elif name == "retreat" and arms:
            self.arm_q.pop(arms[0], None)
            self.arm_fingers.pop(arms[0], None)


def check_base_follow_handles(sidecar: SidecarDraft, sampler, action, targets: Dict[str, Any], stance: Stance, arm: str, witnesses: Dict[str, Any]) -> int:
    """Arm-mode base-follow: every handle pose h(q_k) is IK-checked from the stance with the articulated object's joint set
    to q_k (its link where the hand is at that step; the object itself removed at the pose like a grasped object, the
    supports too), the joint restored afterwards. Returns the number of poses checked; witnesses per k."""
    from ispatialgym.sampling.sim_utils import Reject

    art = getattr(action, "articulation", None)
    poses = targets.get("handle_poses") or []
    qs = targets.get("q_values") or []
    obj = sampler.objects.get(getattr(art, "object", None)) if art is not None else None
    ignore = list(sampler.support_objects()) + ([obj] if obj is not None else [])
    joint_idx = None
    q_saved = None
    if obj is not None and getattr(art, "joint", None) in getattr(obj, "joints", {}):
        joint_idx = list(obj.joints.keys()).index(art.joint)
        q_saved = obj.get_joint_positions().clone()
    n = 0
    try:
        for k, pose in enumerate(poses):
            if joint_idx is not None and k < len(qs):
                q_all = q_saved.clone()
                q_all[joint_idx] = float(qs[k])
                obj.set_joint_positions(q_all)
                obj.keep_still()
                _sync(sampler.kit, [obj])  # train-fast 2026-10-31: one joint of one object per handle pose (was a full pass per k)
            p = _as_pose(pose)
            _count("n_ik_single")
            res = sampler.ikc.exact([p], stance, arm, ignore_for_last=ignore, cache=False, approached=[obj] if obj is not None else [])
            n += 1
            if not res.ok:
                raise Reject(res.code or "program_check_failed", f"handle_poses[{k}] (q = {qs[k] if k < len(qs) else '?'}): {res.reason} (stance {stance.name}, arm {arm})",
                             action=getattr(action, "name", "base_follow"))
            if res.q is not None:
                q_pol, residual = sampler.ikc.polish_witness(res.q, p, arm, ignore=ignore)
                margin, joint = sampler.ikc.joint_limit_margin(q_pol, arm)
                witnesses[f"handle_poses[{k}]"] = {"pose": {"position": [round(float(v), 5) for v in p[0].tolist()], "orientation": [round(float(v), 6) for v in p[1].tolist()]},
                                                  "q": [round(float(v), 5) for v in sampler.ikc.canonical_q(q_pol, stance).tolist()],
                                                  "q_frame": "root at the world origin (identity pose, z = 0)", "base_pose2d": [float(v) for v in stance.base_pose2d],
                                                  "fk_residual_m": round(residual, 5), "joint_limit_margin_rad": round(margin, 4), "joint_limit_nearest": joint,
                                                  "joint_q": None if k >= len(qs) else float(qs[k])}
    finally:
        if joint_idx is not None and q_saved is not None:
            obj.set_joint_positions(q_saved)
            obj.keep_still()
            _sync(sampler.kit, [obj])  # train-fast 2026-10-31
    return n


def _ignore_for(action, sampler, carried: Optional[str], sidecar=None) -> Tuple[List[Any], List[Any]]:
    """(objects removed for the final pose only, objects removed for every waypoint) per §3.6: the support for grasp /
    place / contact end poses; the pushed object along a push; the braced object at the brace contact; the object placed
    on at the place pose (the carried object rests on it 1 cm above). A push-under leg after the drop (:func:`_push_under_floor`) removes
    the pushed / released object and the floor only - never the supports, which on a bed-top init are the partner.

    Block design (``sidecar.blocks[]``, owner 2026-09-19): a block record's ``ignore_place`` (instance names) is removed for
    the block's final ``place`` and the ``retreat`` after it - the blocks already at their slots beside it, which the sampler
    ignored with the finger clearance guaranteed geometrically (block_design ``placed_neighbours``); the checker's world
    keeps them at their init cells, where the anchor's cell IS its slot (0919-093852: 'place: exact IK failed at pose 1 of 1'
    on the second block, 3 cm from the anchor)."""
    name = getattr(action, "name", "")
    supports = list(sampler.support_objects())
    objs = sampler.objects
    last: List[Any] = []
    every: List[Any] = []
    if sidecar is not None and name in ("place", "retreat"):
        place_action = action if name == "place" else getattr(action, "after", None)
        if getattr(place_action, "name", "") == "place":
            obj_name = getattr(action, "object", None)
            rec = next((b for b in (getattr(sidecar, "blocks", None) or []) if isinstance(b, dict) and b.get("instance") == obj_name), None)
            final = (rec or {}).get("place") if isinstance((rec or {}).get("place"), dict) else None
            p_act = getattr(getattr(place_action, "place", None), "position", None)
            if final is not None and p_act is not None:
                d = th.linalg.norm(th.tensor([float(v) for v in final["position"]]) - th.tensor([float(v) for v in p_act]))
                if float(d) < 0.002:  # the FINAL place (not the flip-in-place set-down on the block's own cell)
                    for n_ in rec.get("ignore_place") or []:
                        if n_ in objs:
                            every.append(objs[n_])
    # a module may record exactly what it ignored while solving a chain (names of bddl instances or "supports") on the
    # sidecar record the action carries: ``ignore_every`` / ``ignore_last`` in ``contacts.*`` / ``grasps[]`` extras
    rec = getattr(action, "contact", None) or getattr(action, "grasp", None)
    explicit = _explicit_ignores(rec, sampler)
    if explicit is not None:
        return explicit
    floor = _push_under_floor(action)
    if floor is not None:
        # owner 2026-10-13 ~04:30 item 1b: a push-under leg after the drop works on the floor in front of the partner - the pushed / released
        # object and the floor the blade slides on leave the world, the init surface does not (on a bed-top init it IS the partner: the push
        # and the retreat were checked with the bed ignored and the wrist camera's contact with it went unseen, under_large#01 c3_k0)
        obj = getattr(getattr(action, "contact", None), "object", None) if name == "push" else getattr(action, "object", None)
        return [], [objs[n] for n in (obj, floor) if n in objs]
    # the surface the grasped object rests on: a module may name it (``grasp_ignore_objects``: IORS large-scale's floor class rests
    # the target on the FLOOR, which ``support_objects`` leaves out - its AABB would forbid every parking cell); the lift tracked
    # from the real grasp configuration (2026-09-24) brushed the floor mesh at margin 0 in 1 of 11 configurations with the floor
    # present (0925-082016: 7/8 attempts 'lift: 1/11 tracked chain configurations collide with the scene'), as the module's own
    # lift check never did (it ignores the floor)
    grasp_supports = supports
    try:
        if hasattr(sampler, "grasp_ignore_objects"):
            grasp_supports = list(sampler.grasp_ignore_objects()) or supports
    except Exception:  # noqa: BLE001
        grasp_supports = supports
    if name == "grasp":
        last += grasp_supports + _may_touch(getattr(action, "grasp", None), objs)  # the final grasp pose only
        every += _may_touch(getattr(action, "grasp", None), objs, "pre_grasp")  # the puzzle's fixed-layout parts: the pre-grasp too
    if name == "lift":
        # the lift starts at the grasp pose (object resting on the support, the fingers among the parts it may touch) and goes up
        every += grasp_supports + _may_touch(getattr(getattr(action, "after", None), "grasp", None), objs)
    if name == "place":
        every += supports  # the descent ends 1 cm above the support / partner with the object attached
        # F6 ``Place.on`` = the instance placed on / into (``grasps[].place_on``); older drafts carried the grasp record
        place_on = getattr(action, "on", None) or getattr(getattr(action, "grasp", None), "place_on", None)
        if place_on in objs:
            every.append(objs[place_on])
    if name == "push":
        c_ = getattr(action, "contact", None)
        obj = getattr(c_, "object", None)
        if obj in objs:
            every.append(objs[obj])  # the pushed object moves with the pads
        # a pack push drives the cubes in line ahead of the pushed one along (``driven`` on the record): out of the blade's way
        for n_ in ((getattr(c_, "extra", None) or {}).get("driven") or []):
            if n_ in objs and all(objs[n_] is not o for o in every):
                every.append(objs[n_])
        every += supports
    if name == "brace":
        # the support(s) are removed along the *whole* approach chain, as for a push: a wall / stop brace approaches
        # along the wall 2 cm off it (the sampler solves the chain with the wall ignored — alignment_wall._solve_from);
        # the braced object stays in the world except at the contact pose
        every += supports
        obj = getattr(getattr(action, "contact", None), "object", None)
        if obj in objs:
            last.append(objs[obj])
    if name == "retreat":
        obj = getattr(action, "object", None)
        if obj in objs:
            every.append(objs[obj])  # the retreat starts with the pads around the released object
        every += supports
        # a retreat from a wide brace / pack push starts with a finger side on EVERY braced cube (``braced`` on the contact record)
        c_after = getattr(getattr(action, "after", None), "contact", None)
        for n_ in ((getattr(c_after, "extra", None) or {}).get("braced") or []):
            if n_ in objs and all(objs[n_] is not o for o in every):
                every.append(objs[n_])
    return last, every


def _push_kind(action) -> Optional[str]:
    """``contacts.push.kind`` of a push action (``Push.kind``), None for anything else."""
    if getattr(action, "name", "") != "push":
        return None
    return str((getattr(getattr(action, "contact", None), "extra", None) or {}).get("kind") or "push")


def _push_under_floor(action) -> Optional[str]:
    """The floor instance a push-under leg after the drop slides along: the ``push_under`` push (``Push.touches``) and the retreats after
    it and after the push-under drop (``Place`` of kind ``drop`` under the partner: ``Place.on``); None for every other action."""
    name = getattr(action, "name", "")
    after = getattr(action, "after", None) if name == "retreat" else None
    push = action if name == "push" else after
    if _push_kind(push) == "push_under":
        return next(iter(getattr(push, "touches", None) or ()), "")
    if after is not None and getattr(after, "name", "") == "place" and getattr(after, "kind", None) == "drop" and getattr(after, "relation", None) == "under":
        return getattr(after, "on", None) or ""
    return None


def _slab_off(action) -> bool:
    """Whether ``action``'s targets are checked with the cuRobo ground slab off: a floor-level push whose blade runs below the slab's padding
    (``contacts.push.ground_ignored``, IORS push-under 2026-09-20: the sampler solved the chain with the slab off; the oracle tracks the legs
    without a veto), and the retreat after it that backs out at the push height before it lifts (``Retreat.lift_after_m``)."""
    name = getattr(action, "name", "")
    push = action if name == "push" else (getattr(action, "after", None) if name == "retreat" and getattr(action, "lift_after_m", None) is not None else None)
    return bool(_push_kind(push) is not None and (getattr(getattr(push, "contact", None), "extra", None) or {}).get("ground_ignored"))


def push_open_fingers(action, name: str, after_name: Optional[str]) -> Optional[float]:
    """The finger joints an open-hand push and the retreat after it are checked with: the push record's ``fingers_opening_m`` - the opening
    ``execute_push`` holds the gripper at instead of closing it (the overhang pick's open hand; oracle-fix 2026-11-06, touching's
    ``generation.push_gripper_open``, whose sampler checks the chain, the retreat clear pose and the two-arm brace with it); None otherwise."""
    if name == "push":
        rec = getattr(action, "contact", None)
    elif name == "retreat" and after_name == "push":
        rec = getattr(getattr(action, "after", None), "contact", None)
    else:
        return None
    v = (getattr(rec, "extra", None) or {}).get("fingers_opening_m") if rec is not None else None
    return None if v is None else float(v)


def brace_open_fingers(action, name: str, after_name: Optional[str]) -> Optional[float]:
    """The finger joints a WIDE brace (owner 2026-09-19 00:49) and the retreat after it are checked with: the brace record's
    ``fingers_opening_m`` - ``execute_brace`` pre-shapes the gripper to it, and ``execute_retreat`` leaves a brace's gripper as it is
    (i7-braceopen 2026-11-07: the retreat was checked with the stance's open fingers); None for anything else (a brace record without the
    key). The value is :func:`tabletop.brace_fingers`'s - the one rule the samplers check a brace with (the oracle's ``actions.brace_is_open``:
    a non-positive opening is the closed gripper)."""
    if name == "brace":
        rec = getattr(action, "contact", None)
    elif name == "retreat" and after_name == "brace":
        rec = getattr(getattr(action, "after", None), "contact", None)
    else:
        return None
    extras = (getattr(rec, "extra", None) or {}) if rec is not None else {}
    if extras.get("fingers_opening_m") is None:
        return None
    from ispatialgym.sampling.v2 import tabletop as TT

    return TT.brace_fingers(extras)


def locked_seed(sampler, stance: Stance, fingers: Optional[float], locked: Dict[str, th.Tensor],
                locked_fingers: Optional[Dict[str, float]] = None) -> th.Tensor:
    """The IK seed / locked-joint source of an action while other arms hold their contacts: the stance configuration at the acting
    arm's ``fingers``, every arm of ``locked`` put at its contact configuration and - i7-braceopen 2026-11-07 - its finger joints at
    ``locked_fingers`` (a wide brace held open at its ``fingers_opening_m`` through the other arm's push and retreat; the acting arm's
    opening was put on both hands before)."""
    q_seed = sampler.ikc.q_locked(stance, fingers=fingers).clone()
    for a, qa in locked.items():
        q_seed[sampler.kit.robot.arm_control_idx[a]] = th.as_tensor(qa, dtype=th.float32)
    for a, f in (locked_fingers or {}).items():
        if a in locked and f is not None:
            q_seed[sampler.kit.robot.gripper_control_idx[a]] = float(f)
    return q_seed


def _two_point_fingers(action, name: str, after_name: Optional[str]) -> Optional[float]:
    """The finger joints a two-point push-under (``contacts.push.push_contact_form`` = ``"two_point"``, owner 2026-10-13 ~04:30 item 2) and the
    retreat after it are checked with: the record's ``fingers_opening_m`` (the oracle opens the gripper to it); None for anything else."""
    if name == "push":
        rec = getattr(action, "contact", None)
    elif name == "retreat" and after_name == "push":
        rec = getattr(getattr(action, "after", None), "contact", None)
    else:
        return None
    ex = getattr(rec, "extra", None) or {}
    return float(ex["fingers_opening_m"]) if ex.get("push_contact_form") == "two_point" and ex.get("fingers_opening_m") is not None else None


def _approached_for(action, sampler) -> Optional[List[Any]]:
    """The object an action's approach closes in on (removed along the whole tracked segment, owner 2026-09-16 10:15 A):
    the grasped / lifted / released object, the placed-on partner, the pushed / braced object; a grasp's ``grasps[].may_touch``
    objects too (owner 2026-10-12 A2b: the fingers may meet disembedding's distractors on their way down, no spacing keeps them
    out of the finger path; the pre-grasp itself keeps them). None = let ``IKChecker`` infer it from the end pose."""
    name = getattr(action, "name", "")
    objs = sampler.objects
    cand = None
    if name in ("grasp", "lift", "retreat", "reorient"):
        cand = getattr(action, "object", None) or getattr(getattr(action, "grasp", None), "object", None)
    elif name == "place":
        cand = getattr(action, "on", None) or getattr(getattr(action, "grasp", None), "place_on", None)
    elif name in ("push", "brace"):
        cand = getattr(getattr(action, "contact", None), "object", None)
    if cand is None:
        return None
    out = [objs[cand]] if cand in objs else []
    if name == "grasp":
        out += [o for o in _may_touch(getattr(action, "grasp", None), objs) if all(o is not x for x in out)]
    if name == "place":
        # i7-split 2026-11-07: a REGISTERED support placed on (the destination tray: ``support_split.registered_names``) is not removed along
        # the whole descent - it is the place's support (its split: the floor off at the last pose, the rims obstacles), as
        # ``IKChecker.approached_objects`` infers it for the sampler's own place checks
        from ispatialgym.sampling.v2 import support_split as SS

        out = SS.drop_registered(getattr(sampler, "ikc", None), out)
    return out


def _block_eef_in_object(sidecar, sampler, carried: str) -> Optional[Tuple[th.Tensor, th.Tensor]]:
    """``T_block->eef`` of the cube grasp that holds block ``carried`` (block design, ``blocks[]`` records): the record's
    ``pick`` eef pose relative to the block's *current* root pose - the init pose for the first leg, the turned set-down pose
    (``table_flip.block_pose_after``, where :meth:`_Tracker.advance_block` moved it) with the ``table_flip.regrasp.pick`` for
    the second leg. None without a record."""
    from ispatialgym.sampling import geometry as G

    blocks = getattr(sidecar, "blocks", None) or []
    rec = next((b for b in blocks if isinstance(b, dict) and b.get("instance") == carried), None)
    obj = sampler.objects.get(carried)
    if rec is None or obj is None or not isinstance(rec.get("pick"), dict):
        return None
    p, q = obj.get_position_orientation()
    pick = rec["pick"]
    tf = rec.get("table_flip") or {}
    after = tf.get("block_pose_after") if isinstance(tf, dict) else None
    regrasp_pick = (tf.get("regrasp") or {}).get("pick") if isinstance(tf, dict) else None
    if isinstance(after, dict) and isinstance(regrasp_pick, dict) and after.get("position") is not None:
        if float(th.linalg.norm(th.tensor([float(v) for v in after["position"]]) - p.float())) < 0.02:
            pick = regrasp_pick
    eef = (th.tensor([float(v) for v in pick["position"]]), th.tensor([float(v) for v in pick["orientation"]]))
    rel = G.relative(eef, (p.float(), q.float()))
    return (th.as_tensor(rel[0], dtype=th.float32), th.as_tensor(rel[1], dtype=th.float32))


def _block_pick_q(sidecar, sampler, name: str) -> Optional[List[float]]:
    """The sampler's right-arm pick configuration (7 joints) of block ``name``: ``blocks[].pick_q`` for the first grasp,
    ``table_flip.regrasp.pick_q`` once the block lies turned on its own cell (``advance_block`` moved it there). The grasp IK is
    seeded there so the witnesses - and the oracle's whole chain from them - run in the branch the sampler tracked."""
    blocks = getattr(sidecar, "blocks", None) or []
    rec = next((b for b in blocks if isinstance(b, dict) and b.get("instance") == name), None)
    obj = sampler.objects.get(name)
    if rec is None or obj is None:
        return None
    q = rec.get("pick_q")
    tf = rec.get("table_flip") or {}
    after = tf.get("block_pose_after") if isinstance(tf, dict) else None
    rq = (tf.get("regrasp") or {}).get("pick_q") if isinstance(tf, dict) else None
    if isinstance(after, dict) and rq and after.get("position") is not None:
        p, _ = obj.get_position_orientation()
        if float(th.linalg.norm(th.tensor([float(v) for v in after["position"]]) - p.float())) < 0.02:
            q = rq
    return [float(v) for v in q] if isinstance(q, list) and len(q) == 7 else None


def _may_touch(grasp_rec, objs: Dict[str, Any], key: str = "objects") -> List[Any]:
    """The objects of ``grasps[].may_touch.<key>`` (owner 2026-10-12 A2b: ``objects`` = disembedding's distractors and fixed-layout
    parts, which its grasp may touch or displace as long as they stay on the source tray - the oracle checks that after the grasp;
    ``pre_grasp`` = the fixed-layout parts, which the pre-grasp ignores too)."""
    mt = (getattr(grasp_rec, "extra", None) or {}).get("may_touch") if grasp_rec is not None else None
    names = mt.get(key) or [] if isinstance(mt, dict) else []
    return [objs[n] for n in names if n in objs]


def _explicit_ignores(rec, sampler) -> Optional[Tuple[List[Any], List[Any]]]:
    """``(last, every)`` from a record's explicit ``ignore_last`` / ``ignore_every`` lists (bddl instance names, or the
    token ``"supports"``), when the module recorded them; None otherwise."""
    if rec is None:
        return None
    src = getattr(rec, "extra", None) or getattr(rec, "extras", None) or {}
    if not isinstance(src, dict) or not ("ignore_every" in src or "ignore_last" in src):
        return None
    objs = sampler.objects
    supports = list(sampler.support_objects())

    def _resolve(names) -> List[Any]:
        out: List[Any] = []
        for n in names or []:
            if n == "supports":
                out += supports
            elif n in objs:
                out.append(objs[n])
        return out

    return _resolve(src.get("ignore_last")), _resolve(src.get("ignore_every"))


#: the target key whose witness is where an action leaves the acting arm (the retreat leg of a place, the chain end of a lift)
_END_KEY = {"grasp": "grasp", "lift": "lift", "reorient": "waypoints", "place": "retreat", "retreat": "retreat"}

#: the oracle's ``ExecuteConfig.retreat_partial_accept_m``: a ``Retreat`` action's free-space leg that stalls within this (m) of its
#: end pose is accepted as a partial retreat (``execute_retreat``); the check mirrors it (no witness for the finish to aim at)
RETREAT_PARTIAL_ACCEPT_M = 0.04
#: the oracle's ``ExecuteConfig.retreat_limit_margin_rad``: a Retreat action's leg keeps this far from the joint limits
RETREAT_LIMIT_MARGIN_RAD = 0.05
#: the joint-space re-pose of a block grasp after a base move (``Grasp.repose``) is planned as the oracle plans it (``execute.repose_arm``:
#: 1 + ``MotionsConfig.witness_plan_retries`` rounds of ``MotionsConfig.plan_attempts`` attempts)
REPOSE_PLAN_ROUNDS = 5
REPOSE_PLAN_ATTEMPTS = 4


def check_repose(sampler, stance: Stance, arm: str, repose: Dict[str, Any], pre_pose, q_from: Optional[th.Tensor], fingers: Optional[float] = None,
                 where: str = "") -> Tuple[th.Tensor, Dict[str, Any]]:
    """Block design after a between-block base move (``Grasp.repose``, owner 2026-10-12 item 2): the recorded re-pose of the grasp's arm
    is validated and planned as the oracle runs it. The target - the stance's configuration with the arm at ``repose["q"]`` - must be
    this grasp's pre-grasp (the recorded pose matches ``pre_pose`` and the configuration reaches it) and collision-free in the world the
    program leaves (the blocks advanced, the board, the table), and cuRobo must plan a joint-space path to it from ``q_from`` (where the
    travel left the arm, on this stance; the stance posture when the travel re-posed it). Returns ``(target configuration, note)``;
    ``Reject`` otherwise. The grasp's pre-grasp witness is the target (the chain is tracked from it)."""
    from ispatialgym.sampling import geometry as G
    from ispatialgym.sampling.sim_utils import Reject

    idx = sampler.kit.robot.arm_control_idx[arm]
    q_goal = sampler.ikc.q_locked(stance, fingers=fingers).clone()
    q_goal[idx] = th.as_tensor([float(v) for v in repose["q"]], dtype=th.float32)
    rp = repose["pose"]  # programs.block_repose: an F6 Pose
    rp_pose = (th.tensor([float(v) for v in rp.position], dtype=th.float32), th.tensor([float(v) for v in rp.orientation], dtype=th.float32))
    pre = (G.t(pre_pose[0]).float(), G.t(pre_pose[1]).float())
    if not _same_pose(rp_pose, pre, pos_tol=sampler.ikc.WITNESS_POS_TOL, ang_tol=sampler.ikc.WITNESS_ANG_TOL):
        raise Reject("program_check_failed", f"repose: the recorded pre-grasp {[round(float(v), 3) for v in rp_pose[0].tolist()]} is not this grasp's pre-grasp "
                                             f"{[round(float(v), 3) for v in pre[0].tolist()]} ({where})", action="grasp")
    if not sampler.ikc.reaches(q_goal, pre, arm):
        raise Reject("program_check_failed", f"repose: the recorded configuration misses the pre-grasp pose ({where})", action="grasp")
    if not bool(sampler.kit.collision_free(q_goal, margin=0.0)[0]):
        raise Reject("program_check_failed", f"repose: the recorded configuration collides ({sampler.ikc._collision_blame(q_goal, [])}; {where})", action="grasp")
    q_start = sampler.ikc.q_locked(stance, fingers=fingers).clone()
    label = "the stance posture"
    if q_from is not None:
        q_start[idx] = th.as_tensor(q_from, dtype=th.float32)[idx]
        label = "where the travel left the arm"
    res = None
    rounds = 0
    for rounds in range(1, REPOSE_PLAN_ROUNDS + 1):
        res = sampler.kit.backend.plan_joint(f"{arm}_arm", q_goal, q_start=q_start, max_attempts=REPOSE_PLAN_ATTEMPTS, sync_world=False)
        if res.success:
            break
    if res is None or not res.success:
        raise Reject("repose_unplannable", f"repose: no joint-space plan from {label} to the recorded pre-grasp configuration in {rounds} rounds "
                                           f"({getattr(res, 'status', '?')}; {where})", action="grasp")
    note = {"target": repose.get("target"), "from_stance": repose.get("from_stance"), "planned_from": label, "rounds": rounds,
            "joint_distance_rad": round(float(th.max(th.abs(q_goal[idx] - q_start[idx]))), 3)}
    log.info("program check: grasp re-poses from %s to the recorded pre-grasp configuration after the base move from %s (%d rounds, max joint change %.2f rad)",
             label, repose.get("from_stance"), rounds, note["joint_distance_rad"])
    return q_goal, note


#: i7-overpath: the finger opening the recorded transit runs at (``execute_grasp`` opens the hand fully before its transit)
OPATH_FINGERS = 0.05


def check_overpath(sampler, stance: Stance, arm: str, rec: Dict[str, Any], pre_pose, q_from: Optional[th.Tensor], obj=None,
                   where: str = "") -> Tuple[th.Tensor, Dict[str, Any]]:
    """i7-overpath 2026-11-08 (owner 3(a), ``Grasp.overpath``): the overhang pick's recorded transit over the book replayed as the oracle
    runs it (``execute.follow_overpath`` -> ``Motions.recorded_joint_path``): from where the previous arm action left the arm (the escape
    after the push; the stance posture after the drive to the grasp stance) - within ``overpath.START_GAP_MAX_RAD`` of the record's first
    configuration - the piecewise-linear joint path through the recorded waypoints (:func:`overpath.densify`) must be collision-free in the
    world the program leaves (the pushed object at its post-push pose, the support, the scene; margin 0), keep the arm's spheres
    ``protection.INFLATION_M`` from ``obj``'s box (the oracle's protected-object check) and end at this grasp's pre-grasp. Returns
    ``(the end configuration - the grasp's pre-grasp witness, note)``; ``(None, note)`` when the arm starts farther than the gap from it
    (the oracle then plans the transit, checked as before); ``Reject`` otherwise."""
    from ispatialgym.oracle import overpath as OPATH
    from ispatialgym.oracle import protection as PROT
    from ispatialgym.sampling import geometry as G
    from ispatialgym.sampling.sim_utils import Reject

    try:
        qs = OPATH.parse(rec)
    except ValueError as e:
        raise Reject("program_check_failed", f"overpath: {e} ({where})", action="grasp")
    idx = sampler.kit.robot.arm_control_idx[arm]
    q0 = sampler.ikc.q_locked(stance, fingers=OPATH_FINGERS).clone()
    label = "the stance posture"
    if q_from is not None:
        q0[idx] = th.as_tensor(q_from, dtype=th.float32)[idx]
        label = "where the previous action left the arm"
    gap = float((q0[idx] - qs[0]).abs().max())
    if gap > OPATH.START_GAP_MAX_RAD:
        # the oracle refuses the record from there too (``Motions.recorded_joint_path``) and plans the transit as before: checked as before
        log.warning("program check: the recorded overpath starts %.2f rad from %s (> %.2f): the oracle plans this transit (not followed)", gap, label,
                    OPATH.START_GAP_MAX_RAD)
        return None, {"from": label, "start_gap_rad": round(gap, 4), "followed": False, "why": f"start {gap:.2f} rad off (> {OPATH.START_GAP_MAX_RAD:.2f})"}
    rows = []
    for q in qs[1:]:
        r = q0.clone()
        r[idx] = q
        rows.append(r)
    dense = OPATH.densify(q0, rows, OPATH.STEP_RAD)
    free = sampler.kit.collision_free(dense, margin=0.0)
    if not bool(free[1:].all()):
        k = int(th.nonzero(~free[1:]).reshape(-1)[0]) + 1
        raise Reject("program_check_failed", f"overpath: configuration {k}/{len(dense) - 1} of the recorded transit collides ({sampler.ikc._collision_blame(dense[k], [])}; "
                                             f"{where})", action="grasp")
    min_clear = None
    if obj is not None:
        box = PROT.object_box(obj)
        fn = getattr(sampler.kit.backend, "robot_spheres_world", None)
        if box is not None and fn is not None:
            spheres, links = fn(dense, "whole_body")
            val, _ = PROT.sphere_box_clearance(spheres[:, PROT.moving_sphere_mask(links, [arm])], box)
            viol = PROT.transit_violations(val, PROT.INFLATION_M)
            if viol is not None:
                raise Reject("program_check_failed", f"overpath: the recorded transit passes {getattr(obj, 'name', obj)} at {viol['clearance_m']} m (threshold "
                                                     f"{viol['threshold_m']} m; {where})", action="grasp")
            min_clear = round(float(val[1:-1].min()) if len(val) > 2 else float(val.min()), 4)
    pre = (G.t(pre_pose[0]).float(), G.t(pre_pose[1]).float())
    if not sampler.ikc.reaches(dense[-1], pre, arm):
        raise Reject("program_check_failed", f"overpath: the recorded transit does not end at this grasp's pre-grasp ({where})", action="grasp")
    note = {"from": label, "start_gap_rad": round(gap, 4), "waypoints": len(qs), "steps": len(dense) - 1, "min_book_clearance_m": min_clear, "followed": True}
    log.info("program check: grasp follows the recorded overpath from %s (start %.3f rad off, %d waypoints, %d steps, book clearance %s m)", label, gap, len(qs),
             len(dense) - 1, min_clear)
    return dense[-1].clone(), note


def _same_pose(a, b, pos_tol: float = 0.002, ang_tol: float = 0.01) -> bool:
    """Two eef poses within 2 mm / 0.01 rad (the same recorded key pose reached by two target keys of one action)."""
    from ispatialgym.sampling import geometry as G

    return float(th.linalg.norm(G.t(a[0]) - G.t(b[0]))) <= pos_tol and float(G.quat_angle(G.t(a[1]), G.t(b[1]))) <= ang_tol


def _chain_starts(poses: Sequence[Any]) -> List[Tuple[Any, bool]]:
    """The poses a chain may leave from, with whether that pose must be PREPENDED to the chain: ``poses[0]`` itself (a chain
    that lists its start), and the start ``conventions.line_waypoints`` left out - the straight legs (approach, descent, lift,
    retreat) are ``n`` poses at fractions k/n *after* the start, so the arm leaves from ``2 poses[0] - poses[1]`` (the same
    orientation), 3.3 cm before the first listed pose for a 10 cm leg. It is that unlisted start the previous motion's end
    configuration reaches; solving the first listed pose afresh instead (what the check did until 2026-09-24) picked its IK
    branch at random - the pre witness 3.3 cm away is outside the 3 cm remembered-solution radius - and the oracle, tracking the
    leg from where its arm is, met witnesses 2-3 rad away."""
    from ispatialgym.sampling import geometry as G

    out: List[Tuple[Any, bool]] = [(poses[0], False)]
    if len(poses) >= 2:
        p0, p1 = G.t(poses[0][0]), G.t(poses[1][0])
        if float(th.linalg.norm(p1 - p0)) > 1e-4 and float(G.quat_angle(G.t(poses[0][1]), G.t(poses[1][1]))) <= 0.01:
            out.append(((2.0 * p0 - p1, G.t(poses[0][1])), True))
    return out


#: target keys whose poses are CONTACT poses: the task modules' box check (``IKChecker.eef_in_frame``) covers them with slack for
#: execution error; every other pose inside the QA's watched span is checked on the fingertip point (owner 2026-09-25)
CONTACT_POSE_KEYS = {"grasp", "place", "contact"}
#: actions outside the QA's watched span (the span opens at the pre-grasp / pre-contact and closes at the release / the contact leg)
UNWATCHED_ACTIONS = {"retreat", "navigate", "transition", "set_torso", "base_follow"}
FINGERTIP_FRAME_MARGIN = 0.05
#: execution slack at the in-between poses, in metres at the fingertip (projected through the depth): the cap's recorded pre-attach
#: fingertip sat 11 px inside the 5 % bound at 0.20 m and the demo's, moved 11 mm by the in-hand re-aim, 12 px outside (0925-191101)
FINGERTIP_FRAME_SLACK_M = 0.02


def fingertip_frame_margin(model, cam, pose, margin: float = FINGERTIP_FRAME_MARGIN, slack_m: float = FINGERTIP_FRAME_SLACK_M) -> Dict[str, Any]:
    """The program check's per-pose frame criterion (:func:`fingertips_in_frame`): the fingertip point ``C.fingertip(pose)`` projected into
    the camera ``cam`` of ``model``, inside the image less QA's ``margin`` (5 %) plus ``slack_m`` at the fingertip projected through its
    depth. ``{inside, margin_px (signed: the smallest distance to that border, negative outside; -inf behind the camera), u, v, depth_m,
    slack_px}``. i7-plateqtf2 2026-11-08 (owner item 3): the draws' pre-place frame check uses it too."""
    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling import geometry as G

    tip = C.fingertip((G.t(pose[0]).float(), G.t(pose[1]).float()))
    uv, depth = model.project(cam, G.t(tip).reshape(1, 3))
    u, v, d = float(uv[0, 0]), float(uv[0, 1]), float(depth[0])
    px = float(model.fx) * slack_m / max(d, 0.05) if slack_m > 0 else 0.0
    mu, mv = margin * float(model.width) + px, margin * float(model.height) + px
    m = min(u - mu, float(model.width) - mu - u, v - mv, float(model.height) - mv - v) if d > 0.05 else float("-inf")
    return {"inside": bool(d > 0.05 and m >= 0.0), "margin_px": m, "u": round(u, 1), "v": round(v, 1), "depth_m": round(d, 3), "slack_px": round(px, 1)}


def fingertips_in_frame(sampler, stance: Stance, name: str, pose_targets: Sequence[Tuple[str, Sequence[Any]]], margin: float = FINGERTIP_FRAME_MARGIN,
                        slack_m: float = FINGERTIP_FRAME_SLACK_M) -> Optional[Dict[str, Any]]:
    """Owner 2026-09-25 (cap 0925-191101: ``gripper_out_of_fov`` twice at the place - the pre-attach 20 cm from the head camera put
    the fingertips over the frame's top edge, and the attachment check had only looked at the grasp, the attach and the brace):
    QA's own criterion at every IN-BETWEEN pose of an arm action inside the watched span - the fingertip point ``C.fingertip(eef)``
    inside the stance camera's image with a 5 % margin (``IKChecker.camera_pose``, the world's camera model) - for every recorded
    pose but the contact poses (``CONTACT_POSE_KEYS``: the box check of the task modules) and a place's post-release retreat leg.
    The oracle moves straight between the recorded poses, so the endpoints cover the motion (the carry's SLERP waypoints are all
    listed). Returns ``None`` when every pose passes, else ``{key, index, u, v, depth}`` of the first pose outside."""
    if name in UNWATCHED_ACTIONS or stance is None:
        return None
    model = getattr(getattr(sampler.kit, "world", None), "camera_model", None)
    if model is None:
        return None
    cam = sampler.ikc.camera_pose(stance)
    for key, poses in pose_targets:
        k0 = key.split(".")[-1]
        if k0 in CONTACT_POSE_KEYS or (name == "place" and k0 == "retreat"):
            continue
        for i, p in enumerate(poses):
            # QA's 5 % of the image, plus ``slack_m`` at the fingertip projected through its depth (execution error)
            fm = fingertip_frame_margin(model, cam, p, margin, slack_m)
            if not fm["inside"]:
                return {"key": key, "index": i, "n": len(poses), "u": fm["u"], "v": fm["v"], "depth_m": fm["depth_m"], "slack_px": fm["slack_px"]}
    return None


# ------------------------------------------------------------------ i7-framing (owner decision 2026-11-08 5.1): the lifted carry

#: the oracle's protective via-above of the CARRY (``oracle/execute.py``, the place's straight carry leg when its tracked path passes a
#: protected object too closely): the hand rises straight to ``z_top = max(lift end z, pre-place z) + protect_above_m``, crosses at
#: ``z_top`` to above the pre-place and descends onto it. ``MotionsConfig.protect_above_m`` (0.12; a test keeps them equal).
ORACLE_PROTECT_ABOVE_M = 0.12
#: waypoints per leg of the via-above the frame check projects (the oracle tracks the legs straight, orientation SLERPed)
CARRY_FRAME_LEG_WAYPOINTS = 8
#: ``1``: the IORS small sampler checks the lifted carry's fingertip framing (default off); kit ``framing.carry.checked`` / ``.failed``
CARRY_FRAME_CHECK_ENV = "ISPATIALGYM_CARRY_FRAME_CHECK"
#: ``1``: where only the via-above's arm lift leaves the frame, the sampler records a TORSO lift for the carry (``grasps[].carry_torso``)
#: and the oracle executes it on the protective via-above (default off); kit ``framing.carry.torso_lift_used``
CARRY_TORSO_LIFT_ENV = "ISPATIALGYM_CARRY_TORSO_LIFT"
#: the torso lift: the hand rises at least this much at both carry ends (the via-above's rise), moves at most CARRY_TORSO_MAX_SHIFT_M
#: sideways and turns at most CARRY_TORSO_MAX_ROT_RAD (torso_link4: the JoyLo translate keeps its orientation, the tilt turns it)
CARRY_TORSO_MIN_RISE_M = ORACLE_PROTECT_ABOVE_M
CARRY_TORSO_MAX_SHIFT_M = 0.04
CARRY_TORSO_MAX_ROT_RAD = 0.1
CARRY_TORSO_STEP = 0.05


def _env_on(name: str) -> bool:
    return (os.environ.get(name, "") or "0").strip().lower() in ("1", "on", "true", "yes")


#: the middle option (owner 2026-11-08 5d.3; DEFAULT ON since 5s.1, ``0`` = the rule before: no lifted-carry check): the straight
#: carry's nominal protect margin picks the band - below
#: CARRY_BAND_BELOW_M: the lifted-carry framing check (out of frame -> reject); above CARRY_BAND_ABOVE_M: no check; between: the torso
#: lift when possible, else NO check (owner 2026-11-08 5s.1: the near band never rejects, the lift fires in ~5 % of its carries)
CARRY_BAND_ENV = "ISPATIALGYM_CARRY_BAND"
#: the band edges (m of nominal margin, carry_margin.carry_protect_margin; from the replay of the written carries,
#: $OFH/framing_20261108/margin_replay.json + margin_stats.py): above 9 mm no carry of 896 demos fired (the 42 fired all lie at
#: 1.3-8.5 mm); below 2 mm 20 of 131 fired (15 %, the most any band reaches - the lift is never "almost sure"), 2-9 mm 22 of 396
CARRY_BAND_BELOW_M = 0.002
CARRY_BAND_ABOVE_M = 0.009


def carry_band_on() -> bool:
    """DEFAULT ON (owner 2026-11-08 5s.1); ``ISPATIALGYM_CARRY_BAND=0`` turns it off."""
    return (os.environ.get(CARRY_BAND_ENV, "") or "1").strip().lower() in ("1", "on", "true", "yes")


def carry_band(margin_m: Optional[float], below: float = CARRY_BAND_BELOW_M, above: float = CARRY_BAND_ABOVE_M) -> str:
    """``below`` / ``near`` / ``above`` of a nominal protect margin (``carry_margin.carry_protect_margin``; None / inf = nothing to fire on)."""
    if margin_m is None or margin_m != margin_m or margin_m == float("inf") or margin_m > above:
        return "above"
    return "below" if margin_m < below else "near"


def carry_frame_check_on() -> bool:
    return _env_on(CARRY_FRAME_CHECK_ENV)


def carry_torso_lift_on() -> bool:
    return _env_on(CARRY_TORSO_LIFT_ENV)


def via_above_carry_poses(lift_end, pre_place, above_m: float = ORACLE_PROTECT_ABOVE_M, n: int = CARRY_FRAME_LEG_WAYPOINTS) -> Dict[str, List[Any]]:
    """The oracle's protective via-above of the carry from the lift end to the pre-place (``oracle/execute.py``): ``{"rise": [...],
    "cross": [...], "descend": [...]}`` - the lift end raised straight to ``z_top`` (its orientation), across at ``z_top`` to above the
    pre-place (orientation SLERPed to the pre-place's), straight down onto the pre-place; ``n`` waypoints per leg, each leg's start
    excluded (the lift end itself is checked by the caller's other checks)."""
    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling import geometry as G

    a = (G.t(lift_end[0]).float(), G.t(lift_end[1]).float())
    b = (G.t(pre_place[0]).float(), G.t(pre_place[1]).float())
    z_top = max(float(a[0][2]), float(b[0][2])) + float(above_m)
    up = (th.tensor([float(a[0][0]), float(a[0][1]), z_top]), a[1])
    over = (th.tensor([float(b[0][0]), float(b[0][1]), z_top]), b[1])
    return {"rise": list(C.line_waypoints(a, up, n)), "cross": list(C.line_waypoints(up, over, n)), "descend": list(C.line_waypoints(over, b, n))}


def first_out_of_frame(model, cam, legs: Dict[str, Sequence[Any]], margin: float = FINGERTIP_FRAME_MARGIN, slack_m: float = FINGERTIP_FRAME_SLACK_M) -> Optional[Dict[str, Any]]:
    """The first pose of ``legs`` ({name: poses}) whose fingertip fails :func:`fingertip_frame_margin` (the pre-place's rule), as
    ``{key, index, n, u, v, depth_m, margin_px}``; None when all pass."""
    for key, poses in legs.items():
        for i, p in enumerate(poses):
            fm = fingertip_frame_margin(model, cam, p, margin, slack_m)
            if not fm["inside"]:
                return {"key": key, "index": i, "n": len(poses), "u": fm["u"], "v": fm["v"], "depth_m": fm["depth_m"], "margin_px": round(fm["margin_px"], 1)}
    return None


def torso_link4_world(base_pose2d, translate: float, tilt: float, floor_z: float = 0.0):
    """World pose of ``torso_link4`` (the link the head camera and both arms hang from) at a base pose and JoyLo trunk."""
    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling import geometry as G

    x, y, yaw = (float(v) for v in base_pose2d)
    base = (th.tensor([x, y, float(floor_z) + C.NOMINAL_BASE_Z]), G.quat_from_yaw(yaw))
    return C.compose(base, C.torso_fk(C.torso_q(translate, tilt, C.TORSO_JOINT_LIMITS))["torso_link4"])


def torso_move(base_pose2d, trunk_from: Tuple[float, float], trunk_to: Tuple[float, float], floor_z: float = 0.0):
    """The world transform ``Mt`` a trunk change applies to everything on ``torso_link4`` with the arm joints held (the hand, the head
    camera): a world pose ``p`` at ``trunk_from`` is ``Mt ∘ p`` at ``trunk_to``. The hand's image position is unchanged by construction."""
    from ispatialgym.sampling import geometry as G

    A = torso_link4_world(base_pose2d, *trunk_from, floor_z=floor_z)
    B = torso_link4_world(base_pose2d, *trunk_to, floor_z=floor_z)
    return G.compose(B, G.invert(A))


def carry_torso_choice(base_pose2d, trunk: Tuple[float, float], points: Sequence[Any], floor_z: float = 0.0, min_rise: float = CARRY_TORSO_MIN_RISE_M,
                       max_shift: float = CARRY_TORSO_MAX_SHIFT_M, max_rot: float = CARRY_TORSO_MAX_ROT_RAD, step: float = CARRY_TORSO_STEP) -> Optional[Dict[str, Any]]:
    """The smallest trunk change (the torso straightened along the JoyLo translate alone first - it keeps torso_link4's orientation -,
    then with the tilt raised too) that lifts every one of ``points`` (world positions on the hand, arm joints held) by >= ``min_rise`` with <=
    ``max_shift`` sideways and <= ``max_rot`` turn: ``{"translate", "tilt", "from": [translate, tilt], "rise_m": [...], "shift_m",
    "rot_rad"}``, or None. Within each group in order of the change ``|d translate| + |d tilt|``, both within their ranges / joint limits."""
    from ispatialgym.oracle import r1pro_model as M
    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling import geometry as G

    t0, k0 = float(trunk[0]), float(trunk[1])
    lo_t = float(M.TRUNK_TRANSLATE_RANGE[0])
    j = M.TRUNK_TILT_JOINT_INDEX
    cands = []
    for i in range(0, int(round((t0 - lo_t) / step)) + 1):
        for k in range(0, int(round(max_rot / step)) + 1):
            if i == 0 and k == 0:
                continue
            tr, kt = max(lo_t, t0 - i * step), k0 + k * step
            q = C.translate_to_torso_q(tr)[j] + kt
            if not (float(C.TORSO_JOINT_LIMITS[0][j]) <= float(q) <= float(C.TORSO_JOINT_LIMITS[1][j])):
                continue
            cands.append((k > 0, abs(t0 - tr) + abs(kt - k0), tr, kt))  # the translate alone first (no turn of the hand), then with the tilt
    A_q = torso_link4_world(base_pose2d, t0, k0, floor_z)[1]
    for _, _, tr, kt in sorted(cands):
        Mt = torso_move(base_pose2d, (t0, k0), (tr, kt), floor_z)
        rot = float(G.quat_angle(A_q, torso_link4_world(base_pose2d, tr, kt, floor_z)[1]))
        if rot > max_rot + 1e-6:
            continue
        rises, shifts = [], []
        for p in points:
            p = G.t(p).float().reshape(3)
            p2 = G.compose(Mt, (p, th.tensor([0.0, 0.0, 0.0, 1.0])))[0].float()
            rises.append(float(p2[2] - p[2]))
            shifts.append(float(th.linalg.norm(p2[:2] - p[:2])))
        if min(rises) >= min_rise and max(shifts) <= max_shift:
            return {"translate": round(tr, 4), "tilt": round(kt, 4), "from": [round(t0, 4), round(k0, 4)], "rise_m": [round(r, 4) for r in rises],
                    "shift_m": round(max(shifts), 4), "rot_rad": round(rot, 4)}
    return None


def _carry_start_ignores(sampler) -> List[Any]:
    """What the carry leg may still touch where it starts (the lift end): the grasp's support objects (the floor class's floor
    included, ``grasp_ignore_objects``) - exempted for the first ``APPROACH_WAYPOINT_M`` of the leg only."""
    out = list(sampler.support_objects())
    try:
        if hasattr(sampler, "grasp_ignore_objects"):
            out = list(sampler.grasp_ignore_objects()) or out
    except Exception:  # noqa: BLE001
        pass
    return out


def _place_order(items: List[Tuple[str, List[Any]]], q_prev_end: Optional[th.Tensor], sampler, arm: str) -> List[Tuple[str, List[Any]]]:
    """A ``Place`` checked in MOTION order (owner 2026-09-25; teacup 0925-141122 retreat[4]: the pre-place witness a fresh solve
    4.3 rad from the lift's end while the oracle tracks the carry from there and stays in the lift's branch): ``carry`` first -
    the leg ``Place._carry_leg`` built from ``state.eef_pose`` (where the lift left the hand), with that START pose prepended
    (``line_waypoints`` leaves it out and the orientation SLERP defeats ``_chain_starts``) so it is a continuation from the
    arm's configuration - then ``pre_place`` (= the carry's end), ``descent`` (from it), ``place`` (= the descent's end), the
    ``retreat`` leg, then anything else. ``carry_via`` is part of ``carry`` and not checked on its own. Without a carry leg
    (no configuration for the arm: the place after a carried travel, IORS large-scale's own ``carry_end_q``) the order is the
    generic ``_chain_order`` one, as before."""
    items = [(k, p) for k, p in items if k != "carry_via" and p]
    keyed = {k: p for k, p in items}
    carry = keyed.get("carry")
    if not carry or q_prev_end is None:
        return _chain_order(items)
    start = arm_eef_pose(sampler, q_prev_end, arm)
    keyed["carry"] = [start] + list(carry)
    order = ["carry", "pre_place", "descent", "place", "retreat"]
    out = [(k, keyed[k]) for k in order if k in keyed]
    out += [(k, p) for k, p in items if k not in order]
    return out


def _chain_order(items: List[Tuple[str, List[Any]]]) -> List[Tuple[str, List[Any]]]:
    """Owner 2026-09-24: a single-pose key that is the END of a chain of the same action (``place`` = the end of ``descent``,
    ``grasp`` = the end of ``approach``, a brace ``contact``) is checked right after that chain and takes the chain's tracked
    end as its witness, so the oracle's finish (``ApproachFinish(place, q_hint=witness(place))``) aims into the branch the
    descent actually ends in. A single that a chain leaves from (``pre_place`` -> descent, ``pre_grasp`` -> approach; the
    start ``line_waypoints`` leaves unlisted, ``_chain_starts``) keeps its place before that chain even when another, later
    chain ends at its pose (the place's retreat ends at the pre-place pose)."""
    singles = [(k, p) for k, p in items if len(p) == 1]
    chains = [(k, p) for k, p in items if len(p) > 1]
    if not singles or not chains:
        return list(items)
    order = {k: i for i, (k, _) in enumerate(items)}

    def _starts(sp, cp) -> bool:
        return any(_same_pose(s, sp[0]) for s, _ in _chain_starts(cp))

    ends_of: Dict[str, Optional[str]] = {}
    for sk, sp in singles:
        end_chain = next((ck for ck, cp in chains if _same_pose(cp[-1], sp[0]) and not _starts(sp, cp)), None)
        if end_chain is not None and any(_starts(sp, cp) and order[ck] < order[end_chain] for ck, cp in chains):
            end_chain = None  # it starts an earlier chain: stays where it is (the chain leaves from its witness)
        ends_of[sk] = end_chain
    out: List[Tuple[str, List[Any]]] = []
    placed = set()
    for k, p in items:
        if k in placed:
            continue
        if len(p) == 1 and ends_of.get(k) is not None and ends_of[k] not in placed:
            continue  # deferred until its chain has been checked
        out.append((k, p))
        placed.add(k)
        if len(p) > 1:
            for sk, sp in singles:
                if sk not in placed and ends_of.get(sk) == k:
                    out.append((sk, sp))
                    placed.add(sk)
    for k, p in items:
        if k not in placed:
            out.append((k, p))
            placed.add(k)
    return out


def _skipped_target_keys(name: str, action, targets: Any) -> set:
    """Target keys of ``action`` the program check does not IK-check.

    ``base_follow.handle_poses`` are the handle's poses along the base path (checked by the base-follow module itself).
    ``reorient.derived`` when the action carries **recorded** ``waypoints``: the oracle tracks the recorded key poses
    (``Reorient.targets``), ``derived`` is the same motion re-derived from the start pose and is never executed - checked
    as a second, independent chain from a fresh IK seed it rejected the floor-class shampoo carry whose recorded chain had
    just been tracked from the lift's end (0925-210939: 'derived: exact IK failed at pose 1 of 4')."""
    skip: set = set()
    if name == "base_follow" and isinstance(targets, dict):
        skip.add("handle_poses")
    if name == "reorient" and isinstance(targets, dict) and getattr(action, "waypoints", None):
        skip.add("derived")
    if name == "base_push" and isinstance(targets, dict):
        skip.add("second")  # the other arm's chain: :func:`check_base_push_second_hand`
    return skip


def second_hand_chain(sampler, stance: Stance, arm1: str, q_pre1: th.Tensor, q_c1: th.Tensor, arm2: str, pre, approach: Sequence[Any], target,
                      fingers: float = 0.0, margin: float = 0.01) -> Dict[str, Any]:
    """The second closed hand of a two-hand base push (owner 2026-10-12 ~20:17 item C), solved as the second-arm rule solves a brace
    (``tabletop.brace_second_arm``) - the same steps for the sampler's draw and this check: its pre-contact ``pre`` with the first arm at its
    pre-contact configuration ``q_pre1`` (the target present: the oracle brings both hands to their pre-contacts before either approaches),
    the straight ``approach`` from that witness with the first arm locked at its contact configuration ``q_c1`` (the target out of the world:
    the first hand touches it), then the configuration with both hands at contact whole-body collision-free at ``margin`` (the target out).
    Only the arm joints of ``q_pre1`` / ``q_c1`` are read. Returns ``{"ok", "why", "q_pre", "q"}`` (28-D; ``q`` = both hands at contact)."""
    ikc, kit = sampler.ikc, sampler.kit
    idx1, idx2 = kit.robot.arm_control_idx[arm1], kit.robot.arm_control_idx[arm2]
    q_seed = ikc.q_locked(stance, fingers=fingers).clone()
    q_seed[idx1] = th.as_tensor(q_pre1, dtype=th.float32).reshape(-1)[idx1]
    _count("n_ik_single")
    r_pre = ikc.exact([pre], stance, arm2, cache=False, fingers=fingers, q_seed=q_seed)
    if not r_pre.ok or r_pre.q is None:
        return {"ok": False, "why": f"pre-contact with the {arm1} hand at its pre-contact: {r_pre.reason}"}
    q_lock = ikc.q_locked(stance, fingers=fingers).clone()
    q_lock[idx1] = th.as_tensor(q_c1, dtype=th.float32).reshape(-1)[idx1]
    q_first = q_lock.clone()
    q_first[idx2] = r_pre.q[idx2]
    with ikc.ignoring([target] if target is not None else []):
        _count("n_chains")
        r = ikc.waypoints_ok([pre] + list(approach), stance, arm2, cache=False, fingers=fingers, q_seed=q_lock, q_first=q_first)
        if not r.ok or r.q is None:
            return {"ok": False, "why": f"approach with the {arm1} hand at its contact: {r.reason}"}
        free = bool(kit.collision_free(r.q, margin=margin)[0])
    if not free:
        return {"ok": False, "why": f"both hands at contact: the configuration collides ({ikc._collision_blame(r.q, [target] if target is not None else [], margin=margin)[:160]})"}
    return {"ok": True, "why": "", "q_pre": r_pre.q.clone(), "q": r.q.clone()}


def check_base_push_second_hand(sidecar: SidecarDraft, sampler, action, targets: Dict[str, Any], stance: Stance, witnesses: Dict[str, Any], where: str) -> int:
    """The second hand of a two-hand base push (``BasePush.second``; targets ``second``), after the first hand's ``pre`` / ``approach`` /
    ``contact`` were checked: :func:`second_hand_chain` from the first hand's witnesses, the second hand's fingertips in the camera frame,
    its witnesses ``second.pre``, ``second.approach`` and ``second.contact`` (both hands at contact) recorded as the others. Returns the
    number of poses checked; ``Reject`` otherwise."""
    from ispatialgym.sampling.sim_utils import Reject

    sec = targets["second"]
    arm1, arm2 = action.contact.arm, str(sec["arm"])
    w_pre, w_c = witnesses.get("pre"), witnesses.get("contact") or witnesses.get("approach")
    if not (w_pre and w_c and w_pre.get("q") is not None and w_c.get("q") is not None):
        raise Reject("program_check_failed", f"second hand: the {arm1} hand's pre-contact / contact witnesses are missing ({where})", action="base_push")
    pre = _as_pose(sec["pre"])
    approach = [_as_pose(p) for p in sec["approach"]]
    contact = _as_pose(sec["contact"])
    if os.environ.get("ISG_FINGERTIP_FRAME_CHECK", "on").lower() not in ("off", "0", "false"):
        out = fingertips_in_frame(sampler, stance, "base_push", [("second.pre", [pre]), ("second.approach", approach)])
        if out is not None:
            raise Reject("gripper_not_in_frame", f"base_push.{out['key']}[{out['index']}/{out['n']}]: the fingertip point projects outside the stance camera's image "
                                                 f"(5 %): u={out['u']}, v={out['v']}, depth={out['depth_m']} m ({where})", action="base_push")
    target = sampler.objects.get(action.contact.object)
    r = second_hand_chain(sampler, stance, arm1, th.tensor(w_pre["q"], dtype=th.float32), th.tensor(w_c["q"], dtype=th.float32), arm2, pre, approach, target)
    if not r["ok"]:
        raise Reject("program_check_failed", f"second hand ({arm2}): {r['why']} ({where})", action="base_push")
    limit = float(sampler.gp.gen("witness_residual_max_m", sampler.ikc.WITNESS_RESIDUAL_MAX))
    for key, q, pose, ign, n_wp in (("second.pre", r["q_pre"], pre, [], 1), ("second.approach", r["q"], contact, [target], len(approach)),
                                    ("second.contact", r["q"], contact, [target], 1)):
        lim = float(sampler.gp.gen("witness_residual_pre_m", limit)) if key == "second.pre" else limit
        q_pol, residual = sampler.ikc.polish_witness(q, pose, arm2, ignore=[o for o in ign if o is not None])
        if residual > lim:
            raise Reject("program_check_failed", f"{key}: witness FK residual {residual * 1000:.1f} mm > {lim * 1000:.0f} mm after polishing ({where})", action="base_push")
        margin, joint = sampler.ikc.joint_limit_margin(q_pol, arm2)
        witnesses[key] = {"pose": {"position": [round(float(v), 5) for v in pose[0].tolist()], "orientation": [round(float(v), 6) for v in pose[1].tolist()]},
                          "q": [round(float(v), 5) for v in sampler.ikc.canonical_q(q_pol, stance).tolist()],
                          "q_frame": "root at the world origin (identity pose, z = 0): FK(q) puts base_link at base_pose2d on the measured floor",
                          "base_pose2d": [float(v) for v in stance.base_pose2d], "floor_z": round(float(sampler.kit.floor_z), 4), "arm": arm2,
                          "fk_residual_m": round(residual, 5), "joint_limit_margin_rad": round(margin, 4), "joint_limit_nearest": joint, "n_waypoints": n_wp,
                          "chain": f"the {arm1} hand locked at its {'pre-contact' if key == 'second.pre' else 'contact'}"}
    return 1 + len(approach) + 1


def _held_start_q(sampler, stance, grasp, carried: Optional[str], arm: str, fingers: Optional[float], held_since_init: bool, stance_obj,
                  q_seed_prev, q_end_prev) -> Optional[th.Tensor]:
    """The configuration the acting arm starts the program in when it already holds ``carried`` (a held-object init,
    §6.3): the object has been in the hand since the init stance (no grasp action before this one), the recorded stance
    names it as carried, the grasp record puts it in this arm and the arm has not moved yet. The in-hand chain leaves
    from it (:func:`_with_held_start`), as the sampler's draw check (``waypoints_ok([hold] + waypoints, q_seed=<hold>)``)
    and the oracle do; checked from its first waypoint instead, that waypoint was solved afresh and held to a chosen
    witness's joint-limit margin, which next to a hold at a joint limit (the mental-rotation hold's wrist joint 6 sits
    3 mrad from its limit) depends on the solver finding another branch. None otherwise."""
    if not held_since_init or carried is None or grasp is None or getattr(grasp, "arm", None) != arm or stance_obj is not None:
        return None
    if q_seed_prev is not None or q_end_prev is not None or getattr(stance, "carried", None) != carried:
        return None
    return sampler.ikc.q_locked(stance, fingers=fingers).clone()


def _with_held_start(chain: List[Any], q_held: Optional[th.Tensor], sampler, arm: str) -> Tuple[List[Any], bool]:
    """``chain`` leaving from the hand's recorded start pose (FK of the held start ``q_held``, :func:`_held_start_q`),
    which the action's waypoints leave out - an orbit (a reorient) has no straight-line start to recover - unless the
    chain already starts there. With ``q_seed = q_held`` the start is then pinned (``IKChecker.exact``) and the chain is
    checked as the sampler's draw check checks ``[hold] + waypoints``. Returns ``(chain, prepended)``."""
    if q_held is None or not chain or sampler.ikc.reaches(q_held, chain[0], arm):
        return list(chain), False
    return [sampler.kit.eef_pose(q_held, arm)] + list(chain), True


def _where_of(action, name: str, stance_name: str, arm: str, carried: Optional[str], locked) -> str:
    """The ``(...)`` tail of every rejection message of :func:`check_action_targets` (the early frame gate builds the same one)."""
    where = f"stance {stance_name}, arm {arm}, carried {carried}" + (f", {'/'.join(sorted(locked))} arm locked at its contact" if locked else "")
    if name == "retreat" and getattr(action, "after", None) is not None:
        where += f", after {getattr(action.after, 'name', type(action.after).__name__)}"
    if name in ("push", "brace") and getattr(action, "contact", None) is not None:
        c_ = action.contact
        where += f", {name} of {getattr(c_, 'object', None)} {(getattr(c_, 'extra', None) or {}).get('pose_class') or ''}".rstrip()
    return where


#: actions whose ``targets()`` are recorded poses or poses of the objects' init poses - known before the program moved anything - and so
#: are judged by the early frame gate (train-fast 2026-10-31); lift / reorient (the hand's position), retreat (unwatched) are not
_EARLY_GATE_ACTIONS = ("grasp", "place", "push", "brace", "base_push")


def overhang_escape_backoff_on() -> bool:
    """i7-startdraw2 (``ISPATIALGYM_OVERHANG_ESCAPE_BACKOFF``, the same switch as ``iors_large.overhang_escape_backoff_on``): the overhang
    push's contact escape is replayed from the pad backed off ``push_backoff_m`` along the reversed push, as ``execute_push`` runs it -
    in the sampler's transit start (``_overpath_start``) and in the program check's retreat (whose end the drive's departure arm records)."""
    return os.environ.get("ISPATIALGYM_OVERHANG_ESCAPE_BACKOFF", "1") != "0"  # DEFAULT ON (lead #73; r3 A/B: fix no worse than off)


def push_pick_pre_from_chain_on() -> bool:
    """i7-startdraw2 r2 (lead #71, the transit-start gap): ``ISPATIALGYM_PUSH_PICK_PRE_FROM_CHAIN=1`` - the overhang pick's push pre-contact
    witness is the SAMPLED chain's pre-contact configuration (``contacts.push_pick.chain_arm_q[0]``, the branch the push chain, the
    escape replay and every transit prediction were computed in) whenever it still reaches the pre-contact collision-free here; else the
    fresh solve as before. Off (default): the fresh solve - which landed ~3 rad away (another IK branch) in 8 of 32 inside_large_01 demos,
    so the oracle pushed, retreated and started the grasp transit in a branch no sampler check had seen."""
    return os.environ.get("ISPATIALGYM_PUSH_PICK_PRE_FROM_CHAIN", "1") != "0"  # DEFAULT ON (lead #73; r3 A/B: fix no worse than off)


def _push_pick_pre_from_chain(sampler, action, stance, arm: str, pose, fingers, ignore) -> Optional[IKResult]:
    """:func:`push_pick_pre_from_chain_on`: the chain's pre-contact configuration as the ``pre`` witness, or None (no chain configuration
    recorded, or it no longer reaches the pose collision-free with ``ignore`` out). Kit ``push_pick_pre.chain`` / ``.fresh``."""
    extra = getattr(getattr(action, "contact", None), "extra", None) or {}
    arm_q = (extra.get("chain_arm_q") or [None])[0]
    if arm_q is None:
        _kit(sampler, "push_pick_pre.no_chain")
        return None
    q = sampler.ikc.q_locked(stance, fingers=fingers).clone()
    q[sampler.kit.robot.arm_control_idx[arm]] = th.as_tensor(arm_q, dtype=q.dtype)
    if not sampler.ikc.collision_free_at(q, pose, arm, ignore=list(ignore)):
        _kit(sampler, "push_pick_pre.fresh")
        return None
    _kit(sampler, "push_pick_pre.chain")
    return IKResult(True, q, "the push chain's pre-contact configuration", 1)


def _kit(sampler, key: str) -> None:
    st = getattr(getattr(sampler, "kit", None), "stats", None)
    if isinstance(st, dict):
        st[key] = st.get(key, 0) + 1


def _pick_overhang_push(action) -> bool:
    """``action`` is the overhang pick's push (``contacts.push_pick``, kind ``pick_overhang``)."""
    return getattr(action, "name", "") == "push" and str((getattr(getattr(action, "contact", None), "extra", None) or {}).get("kind") or "") == "pick_overhang"


def _campaign_axial_push(sampler, action) -> bool:
    """Only a fixed-goal axial group's free-object arc push."""
    plan = getattr(sampler, "_campaign_surface_active", None)
    contact = getattr(action, "contact", None)
    return (getattr(plan, "task", None) == "alignment-axial"
            and getattr(action, "name", None) == "push"
            and getattr(contact, "mid", None) is not None
            and not (getattr(contact, "extra", None) or {}).get("articulation"))


def _is_handover(sidecar) -> bool:
    """i7-cork: a hand-over sidecar (``grasps[*].attach_route == "handover"``; the sampler's draft records ``extras``, the F6 sidecar ``extra``)."""
    for g in getattr(sidecar, "grasps", None) or []:
        ex = getattr(g, "extras", None) if isinstance(getattr(g, "extras", None), dict) else getattr(g, "extra", None)
        if isinstance(ex, dict) and ex.get("attach_route") == "handover":
            return True
    return False


def _world_advances(sidecar, action) -> bool:
    """Whether the check's world changes AFTER ``action`` (``_Tracker.after`` / ``run._carry_joint_state``), so later actions' targets
    are not the static ones: a place while block records exist (``advance_block``), a pick-overhang push (``advance_pick_push``), a push - or the retreat after one - with an articulation record (its joint), a base
    push (the object and the stance), a base follow (its joint; a hybrid end stance)."""
    name = getattr(action, "name", "")
    if name in ("base_push", "base_follow"):
        return True
    if name == "lift" and _is_handover(sidecar):
        return True  # i7-cork: the hand-over poses the male in the air before the other hand's grasp (_Tracker.before_targets)
    if name == "place":
        return bool(getattr(sidecar, "blocks", None))
    if name == "push":
        extra = getattr(getattr(action, "contact", None), "extra", None) or {}
        return bool(extra.get("articulation")) or str(extra.get("kind") or "") == "pick_overhang"
    if name == "retreat":
        after = getattr(action, "after", None)
        if getattr(after, "name", "") == "push":
            return bool((getattr(getattr(after, "contact", None), "extra", None) or {}).get("articulation"))
    return False


def _gate_pose_targets(sidecar, sampler, action, name: str, arm: str, targets: Any) -> List[Tuple[str, List[Any]]]:
    """The ``(key, poses)`` list the frame gate of :func:`check_action_targets` sees for an action whose chains leave from no earlier
    configuration: ``iter_pose_targets``, ``_place_order`` / ``_chain_order``, the skipped
    keys out - the same keys, indices and order, so the early gate's message is the per-action gate's."""
    pose_targets = list(iter_pose_targets(targets))
    pose_targets = _place_order(pose_targets, None, sampler, arm) if name == "place" else _chain_order(pose_targets)
    skip = _skipped_target_keys(name, action, targets)
    return [(k_, p_) for k_, p_ in pose_targets if k_.split(".")[0] not in skip]


def _early_frame_gate(sidecar: SidecarDraft, sc, sampler, actions: Sequence[Any]) -> int:
    """train-fast 2026-10-31 (``EARLY_FRAME_GATE``): the fingertip frame gate of :func:`check_action_targets` (:func:`fingertips_in_frame`,
    a camera projection of the recorded poses, no IK) over every action whose targets are known before the program has moved anything,
    BEFORE the first IK of the run - the per-action gate judges an action only when its turn comes, after the IK of every action before
    it. Covered: ``grasp`` (pre_grasp + approach from the object's init pose), ``place`` (pre_place + descent from the recorded place;
    the carry leg needs the lift's end and stays with the per-action gate), ``push`` (pre + waypoints), ``brace`` (pre + approach),
    ``base_push`` (pre + approach; the second hand has its own gate). A shadow :class:`_Tracker` follows the stance (navigate /
    transition), the carried object (grasp / place) and the braced arms (brace / retreat) for the stance and the ``where`` of the message;
    the walk stops after the first action that changes the check's world (:func:`_world_advances`) - from there the per-action gate
    judges, as before. Raises the per-action gate's ``Reject("gripper_not_in_frame", ...)`` with its message; returns the number of
    poses projected."""
    from ispatialgym.sampling.sim_utils import Reject

    shadow = _Tracker(sidecar)
    state = _SamplerState(sampler, None)  # no arm configuration: ``targets()`` keep their recorded-pose fallbacks (a place has no carry leg)
    braced: set = set()
    n_checked = 0
    for i, action in enumerate(actions):
        name = getattr(action, "name", type(action).__name__)
        stance_name, carried = shadow.before(action)
        if name in _EARLY_GATE_ACTIONS:
            try:
                targets = action.targets(sc, state)
                stance = sidecar.stance(stance_name)
            except Exception as e:  # noqa: BLE001 - the per-action path raises its own Reject for this action, in program order
                log.debug("early frame gate: %s skipped (%r)", name, e)
                targets, stance = None, None
            if targets is not None:
                arm = (list(getattr(action, "arms", ()) or ["right"]))[0]
                g = sidecar.grasp_of(carried) if carried else None
                if carried and sampler.objects.get(carried) is not None and g is not None and g.arm and name in ("lift", "reorient", "place", "navigate", "transition", "set_torso"):
                    arm = g.arm
                items = _gate_pose_targets(sidecar, sampler, action, name, arm, targets)
                out = fingertips_in_frame(sampler, stance, name, items)
                n_checked += sum(len(p_) for _, p_ in items)
                if out is not None:
                    where = _where_of(action, name, stance_name, arm, carried, {a for a in braced if a != arm})
                    log.info("program check: the early frame gate rejects %s.%s (the per-action gate would have, after the IK of %d actions)", name, out["key"], i)
                    raise Reject("gripper_not_in_frame", f"{name}.{out['key']}[{out['index']}/{out['n']}]: the fingertip point projects outside the stance "
                                                         f"camera's image (5 %): u={out['u']}, v={out['v']}, depth={out['depth_m']} m ({where})", action=name)
        if _world_advances(sidecar, action):
            break
        arms_ = list(getattr(action, "arms", ()) or [])
        if name == "brace" and arms_:
            braced.add(arms_[0])
        elif name == "retreat" and arms_:
            braced.discard(arms_[0])
        shadow.after(action, None, None)
    return n_checked


def check_action_targets(sidecar: SidecarDraft, sampler, action, targets: Any, stance_name: str, carried: Optional[str],
                         locked_arms: Optional[Dict[str, th.Tensor]] = None, stance_obj: Optional[Stance] = None,
                         q_seed_prev: Optional[th.Tensor] = None, q_end_prev: Optional[th.Tensor] = None,
                         held_since_init: bool = False, locked_fingers: Optional[Dict[str, float]] = None,
                         carried_arm: Optional[str] = None) -> Tuple[int, Dict[str, Any]]:
    """IK / collision check of every eef pose / waypoint list in ``targets`` from the recorded stance ``stance_name``
    (its recorded configuration: working arm untucked, idle arm in its posture, trunk of the stance) with the ``carried``
    object attached to the carrying arm and posed in the hand. Base-pose targets (paths, transitions, base-follow) are
    whole-body collision-checked at the stance trunk. Returns ``(number of poses checked, witnesses)`` where
    ``witnesses`` maps each target key to ``{"pose": {position, orientation}, "q": [28 joints]}`` of the end pose.

    ``locked_arms`` (owner 2026‑09‑16 22:55 item 3): arm -> 7 arm-joint values of an arm that holds a contact while this
    action runs (the brace arm at its contact during the push and the push arm's retreat): those arms are **locked at
    that configuration** for every IK / collision check here — a two-arm exact IK as the wall / axial modules solve it —
    instead of the stance's idle posture; a pre-pose without a solution in that world is a rejection
    (``program_check_failed``), never a witness the oracle has to plan through the braced arm. ``locked_fingers``
    (i7-braceopen 2026-11-07, ``_Tracker.locked_fingers``): arm -> the finger opening a locked arm holds - a wide brace's
    ``fingers_opening_m`` (the acting arm's opening was put on both hands before, closing the open stop for the push).

    ``stance_obj``: an in-memory stance that replaces the recorded ``stance_name`` — where a hybrid ``base_follow`` left
    the base (``_Tracker.stance_obj``), so the retreat after it is checked from there.

    ``q_seed_prev``: the configuration the acting arm's IK is *seeded* from (block design / IORS carry chaining).
    ``q_end_prev`` (owner 2026-09-24, every family): where the previous arm action left the acting arm — a chain whose first
    pose is there (the retreat off a place, the lift off a grasp) is tracked from it instead of being re-solved.
    ``held_since_init``: ``carried`` has been in the hand since the init stance (no grasp action before this one) - with the
    arm not moved yet, its chain leaves from the stance's recorded configuration (:func:`_held_start_q`)."""
    from ispatialgym.sampling.sim_utils import Reject
    from ispatialgym.sampling.v2 import navigation as NAV
    from ispatialgym.sampling.v2.navigation import attached

    name = getattr(action, "name", type(action).__name__)
    arms = list(getattr(action, "arms", ()) or ["right"])
    arm = arms[0]
    stance: Stance = stance_obj if stance_obj is not None else sidecar.stance(stance_name)
    if stance_obj is not None:
        stance_name = f"{stance_name} -> {stance_obj.name} {[round(float(v), 3) for v in stance_obj.base_pose2d]}"
    locked = {a: q for a, q in (locked_arms or {}).items() if a != arm}
    carried_obj = sampler.objects.get(carried) if carried else None
    # i7-cork: ``carried_arm`` (the tracker's carrying arm) picks the record of that hand when the object has two (the hand-over)
    g = sidecar.grasp_of(carried, arm=carried_arm) if carried else None
    eef_in_object = g.eef_in_object if g is not None else None
    if g is None and carried_obj is not None:
        # a block (``blocks[]`` carries no grasps[] record): the cube grasp's eef in the block frame from the record's pick and
        # the block where it lies now. Without it ``attached`` used the CURRENT object-eef offset - the block on the table
        # against the idle hand - and every attached pose was checked with a phantom cube a hand-to-block vector away
        # (0919-093852 attempt 1: 'place: exact IK failed at pose 1 of 1 ... carried toy_dice.n.01_4' where the sampler's
        # own check of that place passed)
        eef_in_object = _block_eef_in_object(sidecar, sampler, carried)
    if carried_obj is not None and g is not None and g.arm and name in ("lift", "reorient", "place", "navigate", "transition", "set_torso"):
        arm = g.arm
    ignore_last, ignore_all = _ignore_for(action, sampler, carried, sidecar)
    approached = _approached_for(action, sampler)
    # push / brace contacts are made with closed pads (§3.6 contact eef poses; the gripper closes before the pre-contact); the
    # push arm retreats still closed (the brace arm releases first: ``_brace_push_retreat``) - checked open, the spread pads
    # at 3 cm above the push goal reached past the narrow box into the partner it now touches ('retreat: exact IK failed at
    # pose 1 of 3 ... after push' on 7/16 attempts of touching 0918-181010)
    after_name = getattr(getattr(action, "after", None), "name", None)
    fingers = 0.0 if name in ("push", "brace", "base_push") or (name == "retreat" and after_name in ("push", "base_push")) else None
    push_open = push_open_fingers(action, name, after_name)
    if push_open is not None:
        fingers = push_open  # an open-hand push (the overhang pick's, touching's open push): pushed and left with the fingers at the recorded opening
    wide = brace_open_fingers(action, name, after_name)
    if wide is not None:
        fingers = wide  # a wide stop braces with the fingers pre-shaped open (owner 2026-09-19 00:49) and lifts off with them so
    two_point = _two_point_fingers(action, name, after_name)
    if two_point is not None:
        fingers = two_point
    # base-follow in base mode: the handle poses h(q_k) are reached by driving the base, not by the arm from the stance;
    # in arm mode they are checked one by one with the articulated object at q_k (``check_base_follow_handles``), not as a
    # chain against the object frozen at q_init (closing: the hand moves into where the open front was)
    skip_keys = _skipped_target_keys(name, action, targets)
    n = 0
    witnesses: Dict[str, Any] = {}
    if name == "base_follow" and isinstance(targets, dict) and targets.get("mode") == "hybrid":
        # hybrid follow: the recorded dense base + arm path is verified pose by pose (``articulation_hybrid``)
        from ispatialgym.sampling.v2.tasks import articulation_hybrid as AH

        n += AH.check_program_follow(sidecar, sampler, action, targets, stance, arm, witnesses)
    elif name == "base_follow" and isinstance(targets, dict) and targets.get("mode") != "base":
        n += check_base_follow_handles(sidecar, sampler, action, targets, stance, arm, witnesses)
    attach_arm = (g.arm if g is not None and g.arm else arm) if carried_obj is not None else None
    # the IK seed / locked-joint source: the stance configuration, with every arm that holds a contact put at it
    q_seed = None
    q_held: Optional[th.Tensor] = None
    if locked:
        q_seed = locked_seed(sampler, stance, fingers, locked, locked_fingers)
    elif q_seed_prev is not None and (name in _END_KEY or name == "push"):
        # the acting arm continues from the previous action's end (block design chaining) / a press's recorded branch (q_hint);
        # trunk / other arm from the stance
        q_seed = sampler.ikc.q_locked(stance, fingers=fingers).clone()
        idx = sampler.kit.robot.arm_control_idx[arm]
        q_seed[idx] = th.as_tensor(q_seed_prev, dtype=th.float32)[idx]
    else:
        q_held = q_seed = _held_start_q(sampler, stance, g, carried, arm, fingers, held_since_init, stance_obj, q_seed_prev, q_end_prev)
    if name == "grasp" and getattr(action, "kind", None) == "cube" and getattr(sidecar, "blocks", None):
        # a block's grasp IK is seeded with the sampler's pick configuration (``blocks[].pick_q``): the branch its tracked legs passed in
        pq = _block_pick_q(sidecar, sampler, getattr(action, "object", None))
        if pq is not None:
            q_seed = sampler.ikc.q_locked(stance, fingers=fingers).clone()
            q_seed[sampler.kit.robot.arm_control_idx[arm]] = th.tensor(pq, dtype=th.float32)
    end_q_raw: Optional[th.Tensor] = None
    end_key = _END_KEY.get(name)
    if end_key is None and (_pick_overhang_push(action) or _campaign_axial_push(sampler, action)):
        end_key = "waypoints"  # oracle-fix 2026-11-05: the retreat after the overhang push leaves from where the push leaves the arm
    if name == "retreat" and after_name == "base_follow" and isinstance(targets, dict) and targets.get("retreat"):
        # the retreat after a hybrid / sequential follow: the search recorded the retreat configuration it found at the final
        # pose (``base_follow_path.retreat_arm_q``); verify THAT witness (FK residual, collision, margin) instead of re-solving
        # the retreat as a tracked chain - the two solvers disagreed by 8-17 mrad of margin (0919-213438, 0921-022826)
        art_ = getattr(action.after, "articulation", None)
        path_ = (getattr(art_, "extra", None) or {}).get("base_follow_path") if art_ is not None else None
        aq = (path_ or {}).get("retreat_arm_q")
        if aq is not None and stance_obj is not None:
            aq_t = th.tensor([float(v) for v in aq], dtype=th.float32)
            q_w = sampler.ikc.q_locked(stance, right_arm=aq_t if arm == "right" else None, left_arm=aq_t if arm == "left" else None, fingers=fingers)
            end_pose = targets["retreat"][-1]
            end_pose = (th.as_tensor(end_pose[0] if not hasattr(end_pose, "position") else end_pose.position, dtype=th.float32),
                        th.as_tensor(end_pose[1] if not hasattr(end_pose, "orientation") else end_pose.orientation, dtype=th.float32))
            q_pol, residual = sampler.ikc.polish_witness(q_w, end_pose, arm, ignore=())
            from ispatialgym.sampling.v2 import prompt0 as _P0
            free = bool(sampler.kit.collision_free(q_pol, margin=_P0.COLLISION_MARGIN)[0])
            margin, jn = sampler.ikc.joint_limit_margin(q_pol, arm)
            limit = float(sampler.gp.gen("witness_residual_follow_m", 5e-3))
            if residual <= limit and free and margin >= sampler.ikc.JOINT_LIMIT_MARGIN * 0.5:
                witnesses["retreat"] = {"pose": {"position": [round(float(v), 5) for v in end_pose[0].tolist()], "orientation": [round(float(v), 6) for v in end_pose[1].tolist()]},
                                        "q": [round(float(v), 5) for v in sampler.ikc.canonical_q(q_pol, stance).tolist()], "q_frame": "root at the world origin (identity pose, z = 0)",
                                        "fk_residual_m": round(residual, 5), "joint_limit_margin_rad": round(margin, 4), "joint_limit_nearest": jn, "source": "search retreat witness"}
                log.info("program check: retreat after the follow verified from the search's witness (residual %.1f mm, margin %.3f rad at %s)", residual * 1000, margin, jn)
                return 1, witnesses
            log.info("program check: the search's retreat witness fails (residual %.1f mm, free %s, margin %.3f at %s): re-solving the retreat", residual * 1000, free, margin, jn)
    axial_escape = name == "retreat" and _campaign_axial_push(sampler, getattr(action, "after", None))
    if name == "retreat" and (_pick_overhang_push(getattr(action, "after", None)) or axial_escape):
        # oracle-fix 2026-11-05 (hardback inside_large#01, hb14_desk_s3 attempt 0 after 3480 s of sampling: 'retreat: ik_joint_limit:
        # left_arm_joint7 within 3 mrad of its limit at the pose (every solution found ...) (... after push)'): the oracle never runs the
        # recorded leg after the overhang push (10 cm along -a: toward the torso for the open hand that drew the object to the edge) -
        # execute_retreat's contact escape leaves radially off the object (over its far rim, back where the push came from) and up off
        # the support, from where the push left the arm. That leg is replayed here (replay_contact_escape, the arc pushes' escape) from
        # the push's tracked end configuration. Fixed-goal axial pushes use the
        # same oracle escape. Their old generic retreat check re-solved a different
        # approach-axis leg after the arc replay had already passed this escape.
        c_ = action.after.contact
        goal_ = c_.goal
        goal_t = (th.as_tensor(goal_.position if hasattr(goal_, "position") else goal_[0], dtype=th.float32),
                  th.as_tensor(goal_.orientation if hasattr(goal_, "orientation") else goal_[1], dtype=th.float32))
        obj_ = sampler.objects.get(c_.object)
        sup_ = list(sampler.support_objects()) if hasattr(sampler, "support_objects") else []
        ign_ = ([obj_] if obj_ is not None else []) + sup_
        where_ = _where_of(action, name, stance_name, arm, carried, locked)
        q_from = None
        if q_end_prev is not None:
            q_from = (q_seed if axial_escape and q_seed is not None else sampler.ikc.q_locked(stance, fingers=fingers)).clone()
            idx_ = sampler.kit.robot.arm_control_idx[arm]
            q_from[idx_] = th.as_tensor(q_end_prev, dtype=th.float32)[idx_]
            if not sampler.ikc.reaches(q_from, goal_t, arm):
                q_from = None
        src_ = "the push's tracked end configuration"
        if q_from is None and axial_escape:
            raise Reject("program_check_failed", "axial retreat lacks the push's checked end configuration", action=name)
        if q_from is None:
            src_ = "a fresh IK of the push's end pose"
            _count("n_ik_single")
            r_ = sampler.ikc.exact([goal_t], stance, arm, ignore_for_last=ign_, cache=False, fingers=fingers)
            if not r_.ok or r_.q is None:
                raise Reject(r_.code or "program_check_failed", f"retreat: the overhang push's end pose: {r_.reason} ({where_})", action=name)
            q_from = r_.q
        from ispatialgym.oracle.actions import RETREAT_DIST

        if not axial_escape and overhang_escape_backoff_on():
            # i7-startdraw2 r3 (lead #73): execute_push backs the pad off push_backoff_m along the reversed push (the init -> goal line) before
            # this retreat; the escape is replayed from there, tracked from the push's end as the oracle's arm_along runs it
            from ispatialgym.oracle.execute import CONFIG as _ECFG

            back_m = float(getattr(_ECFG, "push_backoff_m", 0.0) or 0.0)
            init_ = c_.init
            p0 = th.as_tensor(init_.position if hasattr(init_, "position") else init_[0], dtype=th.float32).reshape(-1)[:3]
            d_ = goal_t[0].reshape(-1)[:3] - p0
            if back_m > 0 and float(th.linalg.norm(d_)) > 1e-6:
                back_t = (goal_t[0].reshape(-1)[:3] - d_ / th.linalg.norm(d_) * back_m, goal_t[1])
                r_b = sampler.ikc.continue_chain(q_from, [goal_t, back_t], arm, ignore=ign_, ignore_end=ign_, ignore_start=ign_)
                if r_b.ok and r_b.q is not None:
                    q_from = r_b.q
                    src_ += f", backed off {back_m * 100:.0f} cm along the reversed push"
                    _kit(sampler, "overhang_escape_backoff.program_check_tracked")
                else:
                    _kit(sampler, "overhang_escape_backoff.program_check_failed")
        push_kind = "fixed-goal axial" if axial_escape else "overhang"
        status_, why_, rep_, q_end_ = replay_contact_escape(sampler, arm, q_from, obj_, sup_, ign_, float(getattr(action, "distance", None) or RETREAT_DIST),
                                                            start=f"the {push_kind} push's end", escape_up_m=getattr(action, "escape_up_m", None))
        if status_ == "failed":
            raise Reject("approach_untrackable", f"retreat: {why_} ({where_})", action=name)
        if status_ != "none":
            log.info("program check: the retreat after the %s push of %s replayed as the oracle's contact escape from %s: %s %s",
                     push_kind, c_.object, src_, status_, rep_)
            witnesses["__arm_end_q__"] = q_end_
            return 1, witnesses
    where = _where_of(action, name, stance_name, arm, carried, locked)
    pose_targets = list(iter_pose_targets(targets))
    slab_off = _slab_off(action)
    # owner 2026-09-24 (block design 0921-050358 action 15: 'the sidecar's pre-pose and end-pose witnesses are in different IK
    # branches', finish rejected 3.2 rad away): the legs are checked as ONE chain the way the oracle runs them -
    #   * a chain whose first pose is where the arm already IS (the previous action's end configuration, or the end of a chain
    #     checked earlier in this action: the retreat off the place configuration, the lift off the grasp) is TRACKED from that
    #     configuration (``IKChecker.continue_chain``: no re-solve of the first pose, a stall finished within the oracle's 0.6 rad
    #     bound, else the leg fails);
    #   * a chain whose first pose is a recorded single witness the oracle plans to (``pre_place`` -> ``descent``, ``pre_grasp`` ->
    #     ``approach``) leaves from that witness (``q_first``); when the chain re-tracks backward from another branch, the
    #     recorded pre witness follows the chain's start;
    #   * a single pose that is the END of a chain (``place``, ``grasp``) takes the chain's tracked end (``_chain_order``).
    idx_arm = sampler.kit.robot.arm_control_idx[arm]
    q_prev_end: Optional[th.Tensor] = None
    prev_src = q_end_prev if q_end_prev is not None else q_seed_prev
    if prev_src is not None and name in _END_KEY:  # the arm actions whose legs continue from where the arm is; pushes / braces plan to their pre pose
        q_prev_end = (q_seed if q_seed is not None else sampler.ikc.q_locked(stance, fingers=fingers)).clone()
        q_prev_end[idx_arm] = th.as_tensor(prev_src, dtype=th.float32)[idx_arm]
    chain_notes: Dict[str, str] = {}  # key -> how its start was chosen (the report)
    if name == "place":
        pose_targets = _place_order(pose_targets, q_prev_end, sampler, arm)
    else:
        pose_targets = _chain_order(pose_targets)
    if os.environ.get("ISG_FINGERTIP_FRAME_CHECK", "on").lower() not in ("off", "0", "false"):
        out_of_frame = fingertips_in_frame(sampler, stance_obj if stance_obj is not None else stance, name, [(k_, p_) for k_, p_ in pose_targets if k_.split(".")[0] not in skip_keys])
        if out_of_frame is not None:
            raise Reject("gripper_not_in_frame", f"{name}.{out_of_frame['key']}[{out_of_frame['index']}/{out_of_frame['n']}]: the fingertip point projects outside the stance "
                                                 f"camera's image (5 %): u={out_of_frame['u']}, v={out_of_frame['v']}, depth={out_of_frame['depth_m']} m ({where})", action=name)
    repose_q: Optional[th.Tensor] = None
    repose_note: Optional[Dict[str, Any]] = None
    if name == "grasp" and getattr(action, "repose", None) is not None:
        # block design after a base move (``Grasp.repose``): the recorded re-pose planned from where the travel left the arm; its target
        # is the pre-grasp witness, so the approach and the legs after it are tracked in the branch the sampler tracked them in
        pre_l = next((p_ for k_, p_ in pose_targets if k_ == "pre_grasp"), None)
        if not pre_l:
            raise Reject("program_check_failed", f"repose: the grasp has no pre_grasp target ({where})", action=name)
        repose_q, repose_note = check_repose(sampler, stance, arm, action.repose, pre_l[0], q_end_prev, fingers=fingers, where=where)
    overpath_q: Optional[th.Tensor] = None
    overpath_note: Optional[Dict[str, Any]] = None
    if name == "grasp" and getattr(action, "overpath", None) is not None:
        # i7-overpath 2026-11-08 (owner 3(a)): the overhang pick's recorded transit over the book, replayed; its end is the pre-grasp witness
        pre_l = next((p_ for k_, p_ in pose_targets if k_ == "pre_grasp"), None)
        if not pre_l:
            raise Reject("program_check_failed", f"overpath: the grasp has no pre_grasp target ({where})", action=name)
        overpath_q, overpath_note = check_overpath(sampler, stance, arm, action.overpath, pre_l[0], q_end_prev, obj=sampler.objects.get(getattr(action, "object", None)),
                                                   where=where)
    motion_ends: List[th.Tensor] = []  # tracked chain ends of this action, most recent first
    single_q: Dict[str, th.Tensor] = {}  # single-pose key -> its witness configuration (this frame)
    partial: Optional[Dict[str, Any]] = None

    def _motion_end_at(first_pose) -> Optional[th.Tensor]:
        for q_ in motion_ends:
            if sampler.ikc.reaches(q_, first_pose, arm):
                return q_
        if q_prev_end is not None and sampler.ikc.reaches(q_prev_end, first_pose, arm):
            return q_prev_end
        return None

    def _single_at(first_pose) -> Tuple[Optional[str], Optional[th.Tensor]]:
        for k_, q_ in single_q.items():
            if sampler.ikc.reaches(q_, first_pose, arm):
                return k_, q_
        return None, None

    with _attach_ctx(sampler.kit, attach_arm, carried_obj, eef_in_object=eef_in_object), sampler.ikc.ignoring([], ground=not slab_off):
        for key, poses in pose_targets:
            if key.split(".")[0] in skip_keys:
                continue
            q_start = None
            if len(poses) == 1:
                q_end_here = next((q_ for q_ in motion_ends if sampler.ikc.reaches(q_, poses[0], arm)), None)
                if q_end_here is not None:
                    # the end of a chain checked just before (the place after the descent): the arm is there, no re-solve
                    res = IKResult(True, q_end_here.clone(), "chain end", 1)
                    chain_notes[key] = "end of the tracked chain"
                elif repose_q is not None and key == "pre_grasp":
                    res = IKResult(True, repose_q.clone(), "re-pose target", 1)
                    chain_notes[key] = f"the recorded re-pose target, planned from {repose_note['planned_from']}"
                elif overpath_q is not None and key == "pre_grasp":
                    res = IKResult(True, overpath_q.clone(), "overpath end", 1)
                    chain_notes[key] = f"the recorded overpath's end, followed from {overpath_note['from']}"
                elif key == "pre" and _pick_overhang_push(action) and push_pick_pre_from_chain_on() and (pinned := _push_pick_pre_from_chain(
                        sampler, action, stance, arm, poses[0], fingers, ignore_last + ignore_all + list(approached or []))) is not None:
                    res = pinned
                    chain_notes[key] = "the sampled push chain's pre-contact configuration (contacts.push_pick.chain_arm_q[0])"
                else:
                    # a single end pose (``grasp``, ``place``, ``contact``) is the end of its approach: the approached object is
                    # removed for it as along the tracked chain (owner A) - with it in the world the open pads straddling the
                    # target pass or fail at margin 0 by millimetres (teacup 18:41: 'grasp: exact IK failed at pose 1 of 1')
                    _count("n_ik_single")
                    res = sampler.ikc.exact(poses, stance, arm, ignore_for_last=ignore_last + ignore_all + list(approached or []), cache=False, fingers=fingers,
                                            approached=approached, q_seed=q_seed)
            else:
                # block design (owner 2026-09-19): the reorient chains (yaw-align orbit, in-hand flip) are tracked at the oracle's
                # densification as the sampler tracked them (``IKChecker.track_chain``; demo 0919-140558 'joint jump 0.23 rad')
                track = name == "reorient" and bool(getattr(sidecar, "blocks", None))
                # where the chain leaves from: its first listed pose, or the unlisted start 1/n before it (``_chain_starts``) -
                # whichever the arm's configuration / a recorded witness reaches; an unlisted start is prepended to the chain
                q_start, k_first, q_first, prepend = None, None, None, False
                if name not in ("push", "brace"):
                    for sp, pre_ in _chain_starts(poses):
                        q_start = _motion_end_at(sp)
                        if q_start is not None:
                            prepend = pre_
                            break
                if q_start is None:
                    for sp, pre_ in _chain_starts(poses):
                        k_first, q_first = _single_at(sp)
                        if q_first is not None:
                            prepend = pre_
                            break
                chain = ([_chain_starts(poses)[1][0]] if prepend else []) + list(poses)
                if q_start is None and q_first is None:
                    chain, held_start = _with_held_start(chain, q_held, sampler, arm)
                    if held_start:
                        chain_notes[key] = "leaves from the held object's recorded start configuration"
                if q_start is not None:
                    # the arm is at the chain's start: tracked from there, as the oracle's arm_along runs it (a reorient has no
                    # finish in the oracle; the free legs are finished within free_leg_finish_max_stall)
                    # what the leg leaves at its start: the object's support under a lift, the released object + its support behind a retreat
                    # an ATTACH place's carry / descent chains are tracked with the retreat's joint-limit margin too (owner 2026-09-25, cork
                    # 0925-195316: the insertion legs at a joint limit swung the correction residual by +-10 mm / +-3 deg between passes;
                    # the oracle tracks those legs with the same margin, execute.attach_leg_limit_margin_rad)
                    if name == "place" and key == "carry":
                        # the carry leg is checked against the world QA sees - partner and neighbours present, the destination support
                        # too (only the descent ends 1 cm above it); the first APPROACH_WAYPOINT_M off the lift end leave the grasp's
                        # support (and the floor class's floor) out, as the lift itself does
                        ign_all, ign_end, ign_start = [], [], _carry_start_ignores(sampler)
                    else:
                        ign_all, ign_end = list(ignore_all) + list(approached or []), list(ignore_last)
                        ign_start = list(ignore_last) + list(ignore_all) + _start_contacts(action, sampler, name, key)
                        if name == "lift" and callable(getattr(sampler, "lift_touch_ignores", None)):
                            # i7-t03 round 2 (d): the board the lifted book leaned on (its touch allowance, checked by the sampler's lift test)
                            ign_all = ign_all + list(sampler.lift_touch_ignores(g))
                    # a FLOOR-class lift (the grasp's support is the floor: IORS large-scale's floor init) keeps the ground slab off along
                    # the whole leg - the carried object's coarse sphere fit reaches below its mesh into the slab for the first
                    # centimetres (shampoo dispenser 0925-171050); the floor mesh is already in the ignore set
                    floor_lift = name == "lift" and any("floor" in str(getattr(o, "name", "")).lower() for o in ign_all)
                    _count("n_chains")
                    # oracle-fix 2026-11-05 (opt-in generation.carry_finish_max_remaining_m): the place's carry stall is finished only this close to its end
                    fin_rem = sampler.gp.gen("carry_finish_max_remaining_m") if (name == "place" and key == "carry" and hasattr(sampler, "gp")) else None
                    res = sampler.ikc.continue_chain(q_start, chain, arm, ignore=ign_all, ignore_end=ign_end, finish_max_remaining_m=None if fin_rem is None else float(fin_rem),
                                                     soft_stall_m=None if name == "reorient" else sampler.ikc.FREE_LEG_STALL_M,
                                                     limit_margin=RETREAT_LIMIT_MARGIN_RAD if (name == "retreat" or (name == "place" and getattr(action, "kind", None) == "attach" and key in ("carry", "descent"))) else 0.0,
                                                     ignore_start=ign_start, ground_along=not floor_lift, ground_meshes_along=not floor_lift)
                    chain_notes[key] = "tracked from the arm's configuration (" + ("previous action" if q_start is q_prev_end else "chain end") + ")"
                    if not res.ok:
                        stall = sampler.ikc.last_track_stall
                        if name == "retreat" and stall is not None and stall.get("free") and float(stall["remaining_m"]) <= RETREAT_PARTIAL_ACCEPT_M:
                            # the oracle accepts a Retreat action's leg that stalls this close to its end as a partial retreat
                            # (execute_retreat, retreat_partial_accept_m): the check does too - no witness is recorded for the key (nothing
                            # for a finish to aim at: the oracle's own fallback lands where this tracking did), the arm ends where it stalled
                            log.warning("program check: %s of %s stalls %.1f mm short of its end from the arm's configuration (%s); accepted as the oracle's partial "
                                        "retreat, no witness recorded", key, name, float(stall["remaining_m"]) * 1000, res.reason[:160])
                            n += len(poses)
                            end_q_raw = stall["q"].clone()
                            partial = {"key": key, "remaining_m": round(float(stall["remaining_m"]), 4), "pos_err_m": round(float(stall["pos_err_m"]), 4),
                                       "at_limit": bool(stall.get("at_limit")), "reason": res.reason[:200]}
                            continue
                        raise Reject(res.code or "program_check_failed", f"{key}: {res.reason} ({where}; the leg is tracked from where the previous motion leaves the arm, "
                                                                          f"as the oracle runs it)", action=name)
                else:
                    _count("n_chains")
                    res = sampler.ikc.waypoints_ok(chain, stance, arm, ignore_for_last=ignore_last, ignore_all=ignore_all, cache=False, fingers=fingers, approached=approached,
                                                   q_seed=q_seed, track_dense=track, q_first=q_first)
                    if k_first is not None:
                        chain_notes[key] = f"leaves from the {k_first} witness"
                    if not res.ok and name == "push" and q_seed_prev is not None and not locked:
                        # a push's pre-contact pose is reached by the oracle's planned ``arm_to`` from wherever the arm is, so the
                        # branch the previous retreat ended in need not hold it: re-solve the chain from the stance posture (the
                        # sampler's seed) before failing (0920-080331: a pack push the sampler's identical chain check had passed)
                        _count("n_chains")
                        res = sampler.ikc.waypoints_ok(chain, stance, arm, ignore_for_last=ignore_last, ignore_all=ignore_all, cache=False, fingers=fingers,
                                                       approached=approached, q_seed=sampler.ikc.q_locked(stance, fingers=fingers).clone(), track_dense=track, q_first=q_first)
                    if (res.ok and name == "push" and not track
                            and not (getattr(getattr(action, "contact", None), "extra", None) or {}).get("articulation")):
                        # owner 2026-09-26 (axial 0926-043254 / 051919): a free-object push whose pre-contact witness sits in another branch
                        # than the stance posture's IK is followed densely, as the oracle runs it (arm_to the pre-contact, then the legs
                        # from there) - only in that situation. The pre-contact is its own single target ("pre", solved before the
                        # chain): the chain checked is pre + the waypoints from its witness
                        q_ref, chain_ref = q_first, list(chain)
                        if q_ref is None and "pre" in single_q:
                            pre_poses = next((p_ for k_, p_ in pose_targets if k_ == "pre"), None)
                            if pre_poses:
                                q_ref, chain_ref = single_q["pre"], list(pre_poses) + list(chain)
                        chk = None
                        replay = None  # the arc replay from a given pre-contact configuration (set when the replay runs)
                        if (q_ref is not None and len(chain_ref) >= 4
                                and os.environ.get("ISG_PUSH_REPLAY", "on").lower() not in ("off", "0", "false")):
                            # owner 2026-09-27 (A8): a free-object arc push is replayed as the oracle runs it - the straight leg, the true
                            # arc of the closed loop, the contact retreat's escape leg - and rejected when the replay cannot follow it
                            from ispatialgym.oracle.actions import RETREAT_DIST

                            tol_o = (sampler.gp.gen("tolerances") or {}).get("orientation") if hasattr(sampler, "gp") else None
                            band = float(tol_o) * _push_stop_frac_arc() if tol_o else 0.005
                            obj_ = sampler.objects.get(getattr(getattr(action, "contact", None), "object", None))
                            from ispatialgym.oracle.execute import push_speed_of

                            # oracle-fix 2026-11-06 (axial#01 / #05, opt-in generation.push_arc_replay_lag_rad): the arc tracked that far past the band
                            lag_rad = float((sampler.gp.gen("push_arc_replay_lag_rad") if hasattr(sampler, "gp") else 0.0) or 0.0)

                            def replay(q_from, _chain=chain_ref, _dir=targets.get("direction") if isinstance(targets, dict) else None, _obj=obj_, _band=band,
                                       _ign=list(ignore_all) + list(ignore_last), _speed=push_speed_of(getattr(action, "contact", None)), line=False, _lag=lag_rad):
                                return replay_arc_push(sampler, arm, q_from, _chain[0], _chain[1:], _dir, _obj, list(sampler.support_objects()), _ign, _band,
                                                       RETREAT_DIST, speed_m_s=_speed, line=line, lag_allowance_rad=_lag)

                            ok_r, why_r, rep_r = replay(q_ref)
                            if rep_r:
                                log.info("program check: push replay of %s: %s %s", getattr(getattr(action, "contact", None), "object", None), why_r[:200], rep_r)
                            chain_notes[key] = why_r[:160]
                            if not ok_r:
                                chk = (False, why_r)
                        flipped_mode = os.environ.get("ISG_FLIPPED_PUSH_CHECK", "on").lower()
                        if chk is None and q_ref is not None and _pick_overhang_push(action) and push_pick_pre_from_chain_on():
                            # i7-startdraw2 r2: the overhang push keeps the sampled chain's branch (its pre witness, the push replayed from it
                            # above) - re-witnessing it on the stance posture's branch moved the oracle's push, retreat and grasp transit 3 rad
                            # away from the branch the escape replay and the transit checks were computed in
                            _kit(sampler, "push_pick_pre.rewitness_skipped")
                        elif chk is None and q_ref is not None and flipped_mode not in ("off", "0", "false"):
                            if flipped_mode == "strict" or replay is None:
                                chk = _flipped_push_chain_check(sampler, chain_ref, stance, arm, q_ref, ignore_last, ignore_all, fingers, approached)
                            else:
                                # oracle-fix 2026-11-03: a flipped witness is re-witnessed on the stance posture's branch when the arc replays
                                # from there (ISG_FLIPPED_PUSH_CHECK=strict restores the 2026-09-26 outright rejection)
                                chk = _rewitness_flipped_push(sampler, chain_ref, stance, arm, q_ref, fingers, replay, witnesses, single_q,
                                                              k_first if q_first is not None else "pre", ignore_all)
                        if chk is not None:
                            chain_notes[key] = chk[1][:160]
                            if not chk[0]:
                                res = IKResult(False, None, chk[1], len(chain))
                    if res.ok and k_first is not None and q_first is not None:
                        # the chain may have re-tracked backward from the end's branch (``track_approach``): the recorded pre witness
                        # must be the configuration the chain leaves from, or the oracle plans to one branch and tracks in another
                        q_used = sampler.ikc.solutions.get((arm, sampler.ikc._pose_key(chain[0])))
                        if q_used is not None and float(th.max(th.abs(q_used[idx_arm].float() - q_first[idx_arm].float()))) > 1e-3 and k_first in witnesses:
                            q_pre, res_pre = sampler.ikc.polish_witness(q_used, chain[0], arm, ignore=list(ignore_all))
                            if res_pre <= float(sampler.gp.gen("witness_residual_pre_m", sampler.gp.gen("witness_residual_max_m", sampler.ikc.WITNESS_RESIDUAL_MAX))):
                                m_pre, j_pre = sampler.ikc.joint_limit_margin(q_pre, arm)
                                witnesses[k_first].update({"q": [round(float(v), 5) for v in sampler.ikc.canonical_q(q_pre, stance).tolist()], "fk_residual_m": round(res_pre, 5),
                                                           "joint_limit_margin_rad": round(m_pre, 4), "joint_limit_nearest": j_pre, "branch": "re-tracked from the chain's end"})
                                single_q[k_first] = q_pre.clone()
                                log.info("program check: the %s witness of %s follows the %s chain's start (re-tracked from the end's branch)", k_first, name, key)
            n += len(poses)
            if not res.ok:
                # a straight approach without a configuration in the pre pose's IK branch is its own reason (the oracle
                # cannot track it: plan_failed); everything else is program_check_failed
                extra_why = ""
                if name == "push" and len(poses) >= 1:
                    # diagnostic (0920-083827: a pack push chain the sampler's identical check passed): is the first pose reachable with
                    # every spawned object removed, and if so what does its configuration collide with in this world?
                    try:
                        spawned = list(getattr(sampler.kit.world, "spawned", {}).values())
                        with sampler.ikc.ignoring(spawned):
                            r_free = sampler.ikc.exact([poses[0]], stance, arm, cache=False, fingers=fingers, q_seed=sampler.ikc.q_locked(stance, fingers=fingers).clone())
                        if r_free.ok and r_free.q is not None:
                            extra_why = "; the first pose is reachable in an empty world and there " + sampler.ikc._collision_blame(r_free.q, ignore_all + ignore_last)
                        else:
                            extra_why = "; the first pose has no IK solution even in an empty world"
                        b_ = stance.world_to_base(poses[0][0][:2]) if hasattr(stance, "world_to_base") else None
                        if b_ is not None:
                            extra_why += f" (first pose at base depth {float(b_[0]):.2f}, lateral {float(b_[1]):+.2f}, z {float(poses[0][0][2]):.3f})"
                        # the world the check ran in: every block's position (base frame, mm) - compared with the sampler's pack world log
                        blocks_ = [b.get("instance") for b in (getattr(sidecar, "blocks", None) or []) if isinstance(b, dict)]
                        cells_ = []
                        for n_ in blocks_:
                            o_ = sampler.objects.get(n_)
                            if o_ is not None:
                                p_o, _ = o_.get_position_orientation()
                                bb = stance.world_to_base(p_o[:2])
                                cells_.append(f"{n_[-1]}:({float(bb[0]) * 1000:.0f},{float(bb[1]) * 1000:+.0f},{float(p_o[2]) * 1000:.0f})")
                        extra_why += "; check world blocks (base mm) " + " ".join(cells_)
                    except Exception as exc:  # noqa: BLE001
                        extra_why = f"; (diagnostic failed: {exc})"
                raise Reject(res.code or "program_check_failed", f"{key}: {res.reason} ({where}){extra_why}", action=name)
            if res.q is not None:
                last = poses[-1]
                # the witness the oracle re-uses (primary oracle 2026-09-16 23:06): polished onto the recorded pose (FK residual
                # <= 2 mm, else program_check_failed), joint-limit margin recorded, and its virtual base joints re-expressed for
                # a robot root at the world origin so FK(q) with an identity root lands base_link on the recorded base_pose2d
                # (the sampler's root stood elsewhere: a constant per-sidecar world offset in the oracle's FK check)
                ign_w = ignore_last + ignore_all + list(approached or [])
                q_pol, residual = sampler.ikc.polish_witness(res.q, last, arm, ignore=ign_w)
                # a pre-pose (pre_grasp / pre_place / pre-contact) is a free-space pose 10 cm off its contact: the oracle's
                # approach finish closes a residual there, so a problem may allow more (generation.witness_residual_pre_m;
                # owner 2026-09-18 01:01 touching: witnesses on joint 2's limit sit 9-12 mm short of the pre_grasp); the end
                # poses keep WITNESS_RESIDUAL_MAX
                limit = float(sampler.gp.gen("witness_residual_max_m", sampler.ikc.WITNESS_RESIDUAL_MAX))  # per-problem override (test settings)
                if str(key).startswith("pre"):
                    limit = float(sampler.gp.gen("witness_residual_pre_m", limit))
                if residual > limit:
                    raise Reject("program_check_failed", f"{key}: witness FK residual {residual * 1000:.1f} mm > {limit * 1000:.0f} mm "
                                                         f"after polishing ({where})", action=name)
                margin, joint = sampler.ikc.joint_limit_margin(q_pol, arm)
                if end_key is not None and (str(key) == end_key or end_q_raw is None):
                    end_q_raw = q_pol.clone()
                witnesses[key] = {"pose": {"position": [round(float(v), 5) for v in last[0].tolist()], "orientation": [round(float(v), 6) for v in last[1].tolist()]},
                                  "q": [round(float(v), 5) for v in sampler.ikc.canonical_q(q_pol, stance).tolist()],
                                  "q_frame": "root at the world origin (identity pose, z = 0): FK(q) puts base_link at base_pose2d on the measured floor",
                                  "base_pose2d": [float(v) for v in stance.base_pose2d], "floor_z": round(float(sampler.kit.floor_z), 4),
                                  "fk_residual_m": round(residual, 5), "joint_limit_margin_rad": round(margin, 4), "joint_limit_nearest": joint,
                                  "n_waypoints": len(poses)}
                if key in chain_notes:
                    witnesses[key]["chain"] = chain_notes[key]
                if name == "retreat" or (name == "place" and str(key) == "retreat"):
                    # i7-t03 round 7 (ISPATIALGYM_T03_PAIR_MESH, default off): the arm_link5 / torso_link2 collision-mesh check the sphere model
                    # misses, at the retreat witnesses (the sampler's own hook; None when off or for samplers without it)
                    pm = getattr(sampler, "pair_mesh_why", None)
                    touch = pm(q_pol, arm, f"program_{name}") if callable(pm) else None
                    if touch is not None:
                        raise Reject("self_mesh_contact", f"{key}: {touch} ({where})", action=name)
                if len(poses) == 1:
                    single_q[key] = q_pol.clone()
                else:
                    motion_ends.insert(0, q_pol.clone())  # the arm is here after this chain: the next chain of the action leaves from it
                    if q_start is not None:
                        witnesses[key]["branch"] = "tracked from the arm's configuration"
        if name == "push" and "pre" in single_q and not locked and os.environ.get("ISG_PUSH_STANDOFF_CHECK", "on").lower() not in ("off", "0", "false"):
            # owner 2026-10-12 (item 7c): the oracle's standoff decision at a push pre-contact it reaches with the recorded witness - not
            # an articulated push (its moving links leave the world instead) or a lever slide (no transit); a pack push has no separate
            # pre-contact. With another arm locked at its contact the oracle relaxes that arm's start collisions (not modelled here)
            c_ = getattr(action, "contact", None)
            extra_ = getattr(c_, "extra", None) or {}
            pre_poses = next((p_ for k_, p_ in pose_targets if k_ == "pre"), None)
            wp_poses = next((p_ for k_, p_ in pose_targets if k_ == "waypoints"), None)
            first_ = getattr(c_, "init", None)
            if (pre_poses and wp_poses and not isinstance(extra_.get("articulation"), dict) and not extra_.get("approach_is_slide")
                    and push_keeps_recorded_pre(c_, pre_poses[0], first_ if first_ is not None else wp_poses[0])):
                sup_ = list(sampler.support_objects())
                through_ = [o for o in list(ignore_all) + list(ignore_last) if not any(o is s for s in sup_)]
                ok_s, why_s, rep_s = push_pre_standoff(sampler, arm, single_q["pre"], pre_poses[0], sup_, through_)
                if rep_s:
                    log.info("program check: push pre-contact of %s: %s %s", getattr(c_, "object", None), why_s[:200], rep_s)
                    if "pre" in witnesses:
                        witnesses["pre"]["standoff"] = {k_: v_ for k_, v_ in rep_s.items() if k_ in ("clearance_m", "branch")}
                if not ok_s:
                    raise Reject("approach_untrackable", f"pre: {why_s} ({where})", action=name)
        if name == "base_push" and isinstance(targets, dict) and targets.get("second"):
            n += check_base_push_second_hand(sidecar, sampler, action, targets, stance, witnesses, where)
        # base poses: a transition's drive-back pose is visited with the departing trunk (arrival) and the arriving trunk
        # (after the sweep), the drive-in pose with the arriving trunk only; a path's goal with the stance trunk
        checks: List[Tuple[Tuple[float, float, float], Tuple[float, float]]] = []
        to_trunk = (stance.translate, stance.tilt)
        if isinstance(targets, dict):

            def _p2d(v):
                return (float(v[0]), float(v[1]), float(v[2])) if isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v) else None

            back, drive_in, goal = _p2d(targets.get("drive_back")), _p2d(targets.get("drive_in")), _p2d(targets.get("goal"))
            if name == "transition":
                frm = sidecar.stances.get(getattr(action.transition, "frm", ""))
                if getattr(action.transition, "departure", False):
                    # a departure transition (owner 2026-09-19 21:56): the drive-back pose on the DEPARTING stance's axis is visited
                    # with the departing trunk and, after the sweep, with the travel trunk; there is no drive-in
                    travel = getattr(action.transition, "trunk_to", None) or to_trunk
                    if back is not None:
                        if frm is not None:
                            checks.append((back, (frm.translate, frm.tilt)))
                        checks.append((back, (float(travel[0]), float(travel[1]))))
                else:
                    # the trunk the robot ARRIVES with: the departing stance's, or the travel trunk when a departure transition
                    # of the same pair straightened it
                    dep = next((t for t in sidecar.transitions if t.frm == action.transition.frm and t.to == action.transition.to and getattr(t, "departure", False)), None)
                    from_trunk = (frm.translate, frm.tilt) if frm is not None else None
                    if dep is not None and dep.trunk_to is not None:
                        from_trunk = (float(dep.trunk_to[0]), float(dep.trunk_to[1]))
                    if back is not None:
                        checks.append((back, to_trunk))
                        if from_trunk is not None:
                            checks.append((back, from_trunk))
                    if drive_in is not None:
                        checks.append((drive_in, to_trunk))
            elif goal is not None:
                checks.append((goal, to_trunk))
        rec_arms = _record_arms_q(action) if name in ("navigate", "transition") else {}
        # oracle-fix 2026-11-06 (the shelf back-off): a transition whose record pre-shapes the fingers before the arms move (``preshape_m``)
        # visits its poses with them so
        rec_fingers = _record_preshape(action) if name == "transition" else None
        for pose2d, trunk in checks:
            r_arm, l_arm, fingers_st = sampler.ikc.arm_vectors(stance)
            if rec_fingers is not None:
                fingers_st = rec_fingers
            q_arms = {"right": r_arm, "left": l_arm}
            q_arms.update(rec_arms)  # owner 2026-09-24: the arms the record was swept with (``arms_q``) when it carries them
            drive_ctx = attached(sampler.kit, None, None)
            if carried_obj is not None and g is not None and attach_arm and name in ("navigate", "transition"):
                # a carried drive (touching / attachment second base pose, owner 2026-09-18 16:32): the object rides in the hand
                # at the LIFT configuration of the departing stance (as the oracle drives it) - not at its init pose beside the
                # untucked arm of the arriving stance (0918-171641: 'base pose ... collides' on 4/12 attempts)
                frm_name = getattr(getattr(action, "path", None), "frm", None) or getattr(getattr(action, "transition", None), "frm", None)
                frm_st = sidecar.stances.get(frm_name) if frm_name else None
                if frm_st is not None and getattr(g, "grasp", None) is not None:
                    lift_end = lift_end_pose(g)
                    wps = getattr(g, "waypoints", None)
                    if wps:
                        # an in-hand move after the lift (``Reorient`` to the CARRY pose, owner 2026-09-20 05:20): the object rides at
                        # the waypoint end, not at the lift
                        end = wps[-1]
                        lift_end = (th.as_tensor(getattr(end, "position", end[0]), dtype=th.float32), th.as_tensor(getattr(end, "orientation", end[1]), dtype=th.float32))
                    supports = list(sampler.support_objects()) if hasattr(sampler, "support_objects") else []
                    # train-fast 2026-10-31 (ATTACH_REUSE): the action's own context above holds this very attachment - the nested
                    # contexts (one lift-end solve + one drive check per base pose) re-use it instead of re-fitting it twice per pose
                    with _attach_ctx(sampler.kit, attach_arm, carried_obj, eef_in_object=eef_in_object, nested=True):
                        _count("n_ik_single")
                        r_lift = sampler.ikc.exact([lift_end], frm_st, attach_arm, ignore_for_last=supports, cache=False)
                    if r_lift.ok and r_lift.q is not None:
                        q_arms[attach_arm] = r_lift.q[sampler.kit.robot.arm_control_idx[attach_arm]].clone()
                drive_ctx = _attach_ctx(sampler.kit, attach_arm, carried_obj, eef_in_object=eef_in_object, nested=True)
            # oracle-fix 2026-11-04 round 3: the sampler's stance_check_ignores of the leg's arriving stance (the target at its INIT pose is not where
            # the robot meets it after the pick: the briefcase's push-under crouch "base pose ... collides" 5-8 x per run)
            to_name = getattr(getattr(action, "path", None), "to", None) or getattr(getattr(action, "transition", None), "to", None)
            to_st = sidecar.stances.get(to_name) if to_name else None
            ign_extra = list(sampler.stance_check_ignores(to_st)) if (to_st is not None and callable(getattr(sampler, "stance_check_ignores", None))) else []
            with drive_ctx, (sampler.ikc.ignoring(ign_extra) if ign_extra else contextlib.nullcontext()):
                # the stance's recorded finger opening too (a push-close re-stance records the closed fist at its pre-contact: checked
                # open, the spread pads reached the flap - 0920-191145 'base pose ... collides at trunk')
                ok, _ = NAV.path_collision_free(sampler.kit, [pose2d], trunk, q_arms=q_arms, fingers=fingers_st)
            n += 1
            if not ok:
                raise Reject("program_check_failed", f"base pose {tuple(round(v, 3) for v in pose2d)} collides at trunk {trunk}"
                                                     f"{' (carried ' + str(carried) + ' in the lifted hand)' if carried_obj is not None and name in ('navigate', 'transition') else ''}",
                             action=name)
    if end_q_raw is not None:
        witnesses["__arm_end_q__"] = end_q_raw  # popped by ``run`` (not a recorded witness)
    if partial is not None:
        witnesses["__partial__"] = partial  # popped by ``run``: a Retreat leg accepted short, as the oracle accepts it
    if repose_note is not None:
        witnesses["__repose__"] = repose_note  # popped by ``run``: the re-pose planned before the grasp
    if overpath_note is not None:
        witnesses["__overpath__"] = overpath_note  # popped by ``run``: the recorded transit replayed before the grasp (i7-overpath)
    return n, witnesses
