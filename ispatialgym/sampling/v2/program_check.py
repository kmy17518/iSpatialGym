"""Program check (spec §3.5, principle 4; oracle doc §3.7): the last stage of every attempt.

``run(alias, sidecar, sampler)`` expands the oracle's program for the alias from the completed draft sidecar
(``ispatialgym.oracle.programs.expand(alias, sidecar_dict)``, F6), asks every action for its ``targets()`` and IK- /
collision-checks every pose and waypoint it returns — in the recorded posture, trunk and stance, with the carried
object attached where the action carries one (screening IK at waypoints, exact IK at end poses, §3.6 / §3.9). A failing
pose is a bug in the rule that recorded it: the attempt is rejected with ``Reject("program_check_failed", action=<name>)``
and nothing is patched. ``validate_instances.py`` (D5) runs the same expansion on written instances.

Until F6 lands (``git log`` shows ``oracle: sidecar schema ... (F6)``), :func:`run` falls back to :func:`check_blocks`,
a structural stub that validates the sidecar blocks the alias's program will consume exist and are well-formed
(stances, transitions, paths, grasps, contacts), so that the pipeline stage is exercised end to end.

``targets()`` contract we code against (oracle doc §3.6): a dict whose values are eef poses ``(pos, quat)`` /
``{"position", "orientation"}`` dicts, lists of such (waypoint lists), ``(x, y, yaw)`` base poses under keys containing
``base`` / ``path``, or nested dicts; the action carries ``name``, ``arms``, ``objects`` and optional attributes
``stance`` (stance name used), ``carried`` (bddl instance held during the action) and ``ignore`` (object names removed
from the collision world for the final pose).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch as th

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

def _push_stop_frac_arc() -> float:
    """The oracle's ``ExecuteConfig.push_stop_frac_arc``: an arc push stops at this fraction of the orientation tolerance."""
    from ispatialgym.oracle import execute as E

    return float(E.ExecuteConfig().push_stop_frac_arc)


def replay_arc_push(sampler, arm: str, q_pre: th.Tensor, pre_pose, waypoints, direction, obj, supports: Sequence[Any], ignore: Sequence[Any],
                    band_rad: float, retreat_distance: float) -> Tuple[bool, str, Dict[str, Any]]:
    """A free-object arc push replayed kinematically as the oracle executes it (owner 2026-09-27, first-demo v2 A8; the object
    follows the pad): from the pre-contact witness ``q_pre`` the straight leg to the first contact pose (``execute_push`` ->
    ``_push_arc_closed_loop``: ``Motions.arm_along`` at the arm_along solver settings), then the TRUE arc of ``push_geometry``
    (``Motions.arm_arc_closed_loop``: every control step the FK pose rotated by ``push_speed_arc / radius · dt`` about the arc's
    axis, one differential-IK step at the loop's tolerances with its joint-speed bound, 10 consecutive misses fail) until the
    object reaches its goal band, then the contact retreat's escape leg from the arc's end (``execute_retreat``: radial off the
    object, then off its supports, the end probed for reach and shortened like ``_retreat_reach``; tracked with the retreat's
    joint-limit margin; a stall away from a joint limit is finished by IK, a partial retreat accepted as the oracle accepts it).
    Every setting is read from the oracle's ``ExecuteConfig`` / ``MotionsConfig``. Returns ``(ok, why, report)``; a replay that
    cannot follow the push (``ok`` False) is an ``approach_untrackable`` instance. Collisions are not judged (the oracle's contact
    legs run without a veto)."""
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
    geom = E.push_geometry(list(waypoints), direction)
    if geom.kind != "arc":
        return True, "not an arc", rep

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
    r_now = radial(fk(q)[0])
    radius = max(float(th.linalg.norm(r_now)), 0.05)
    step_ang = ecfg.push_speed_arc / radius * dt
    arc_dir = signed_angle(r_now, radial(p_t))
    sgn = 1.0 if arc_dir >= 0 else -1.0
    max_steps = int(max(abs(q_to), abs(arc_dir)) / max(step_ang, 1e-6) * 3) + 60
    swept, commanded, consecutive, steps, misses = 0.0, 0.0, 0, 0, 0
    stop = None
    worst = sampler.ikc.joint_limit_margin(q, arm)  # (margin, joint): the arc's closest approach to a joint limit
    while True:
        if sgn_joint * (swept - q_to) >= -band_rad:
            stop = "joint_at_target"
            break
        if commanded >= ecfg.push_arc_lag_factor * max(abs(q_to), 1e-3) + 2 * step_ang:
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
                               f"(nearest joint limit {margin:.3f} rad at {jn}; {getattr(r_, 'status', '')[:100]})"), rep
        steps += 1
    rep["arc"] = {"steps": steps, "swept_rad": round(swept, 4), "q_to_rad": round(q_to, 4), "stop": stop, "ik_misses": misses,
                  "min_limit_margin_rad": round(float(worst[0]), 4), "nearest_joint": worst[1]}
    if stop != "joint_at_target":
        return False, f"approach_untrackable: push replay - the closed-loop arc stops at {stop} (swept {swept:+.3f} of {q_to:+.3f} rad)", rep
    # 3. the contact retreat's escape leg from the arc's end (execute_retreat)
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
        return True, "arc replayed (no escape direction)", rep
    d1 = radial_d if radial_d is not None else n
    out = ecfg.contact_retreat_escape_m
    far = max(float(retreat_distance), 2 * out)
    normal_out = n * out if (n is not None and radial_d is not None) else th.zeros(3)
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
    rep["retreat"] = {"distance_m": far_used, "probe": tried}
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
        return True, "arc + retreat replayed", rep
    diag = dict(getattr(r_leg, "diagnostics", None) or {})
    k, stall = diag.get("fail_index"), diag.get("pos_err")
    at_limit = bool(diag.get("at_lower") or diag.get("at_upper"))
    q_stall = r_leg.q_traj[-1].reshape(-1).float() if r_leg.q_traj is not None and len(r_leg.q_traj) > 0 else q
    remaining = float(stall or 0.0) + sum(float(th.linalg.norm(th.as_tensor(dense_r[i][0]) - th.as_tensor(dense_r[i - 1][0]))) for i in range(max(int(k or 1), 1), len(dense_r)))
    rep["retreat"].update({"stall_m": None if stall is None else round(float(stall), 4), "at_limit": at_limit, "fail_index": k, "n_dense": len(dense_r)})
    if diag.get("stage") == "diff_ik" and stall is not None and not at_limit and float(stall) <= ecfg.free_leg_finish_max_stall and remaining <= ecfg.free_leg_finish_max_remaining:
        # ApproachFinish without a witness: cuRobo's IK of the end pose from the stall, within the finish's joint excursion
        with sampler.ikc.ignoring(list(ignore)):
            q_fin = kit.ik(leg[-1], q_stall, arm=arm)
        if q_fin is not None and float(th.max(th.abs(th.as_tensor(q_fin).reshape(-1).float()[idx] - q_stall[idx]))) <= MO.ApproachFinish(None).max_excursion:
            rep["retreat"]["finished"] = True
            return True, "arc + retreat replayed (stall finished)", rep
    moved = float(th.linalg.norm(fk(q_stall)[0] - here[0]))
    end_gap = float(th.linalg.norm(fk(q_stall)[0] - leg[-1][0]))
    if end_gap <= ecfg.retreat_partial_accept_m or moved >= out - 0.005:
        rep["retreat"]["partial"] = {"short_m": round(end_gap, 4), "moved_m": round(moved, 4)}
        return True, "arc replayed, retreat partial", rep
    return False, (f"approach_untrackable: push replay - the contact retreat's escape leg from the arc's end stalls after {moved * 1000:.0f} mm "
                   f"({end_gap * 1000:.0f} mm short; {r_leg.status[:160]})"), rep


log = logging.getLogger("ispatialgym.sampling.v2")

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
    "inter_object_relation_satisfaction-inside": ("stances.init", "stances.manipulation", "transitions", "grasps"),
    "inter_object_relation_satisfaction-nextto": ("stances.init", "stances.manipulation", "transitions", "grasps"),
    "inter_object_relation_satisfaction-touching": ("stances.init", "stances.manipulation", "transitions", "grasps", "contacts.push"),  # brace optional (owner 2026-09-16 17:10; same rule as axial)
    "inter_object_relation_satisfaction-ontop_large_scale": ("stances.init", "stances.manipulation", "stances.placement", "transitions", "grasps", "paths"),
    "inter_object_relation_satisfaction-under_large_scale": ("stances.init", "stances.manipulation", "stances.placement", "transitions", "grasps", "paths"),
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


def subtask_blocks(subtask: str) -> Tuple[str, ...]:
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
    for blk in subtask_blocks(subtask):
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
            self.sampler.kit.sync_world()
        return self

    def __exit__(self, *exc):
        for obj_name, (p0, q0) in self.saved.items():
            obj = self.sampler.objects.get(obj_name)
            if obj is not None:
                obj.set_position_orientation(p0, q0)
                obj.keep_still()
        if self.saved:
            self.sampler.kit.sync_world()
        self.saved.clear()
        return False


def _sweep_travel_record(sidecar: SidecarDraft, sampler, action, override: Dict[str, th.Tensor]) -> Tuple[bool, str, Dict[str, th.Tensor]]:
    """Re-check one recorded travel leg with the acting arm(s) at ``override`` (arm -> 7 joints) instead of the stance posture:
    a path at its recorded trunk over its strides, an arrival transition's sweep / drive-in at its recorded s_T, a departure
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
        poses = NAV.stride_poses(rec.waypoints, float(rec.stride_m or NAV.STRIDE_M), yaw_envelope=True)  # + the tracker's yaw lag on the turns
        ok, bad = NAV.path_collision_free(kit, poses, trunk, q_arms=q_arms, fingers=f, margin=DEPARTURE_SWEEP_MARGIN)
        return ok, "" if ok else f"path {rec.frm} -> {rec.to} collides at swept pose {bad}/{len(poses)} (strides + yaw-lag envelope)", q_arms
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


def run(alias: str, sidecar: SidecarDraft, sampler, subtask: Optional[str] = None) -> Dict[str, Any]:
    """§3.5 program check. ``sampler`` supplies ``ikc`` (``IKChecker``), ``kit``, ``objects`` (bddl instance -> sim
    object), ``support_objects()`` (ignored for final grasp / contact poses) and ``subtask``. Returns a report
    ``{"mode": "expand"|"stub", "actions": [{"name", "stance", "carried", "n_poses"}], ...}``; raises
    ``Reject("program_check_failed", action=<name>)`` (``sim_utils.Reject``) on the first failing target."""
    from ispatialgym.sampling.sim_utils import Reject

    import os

    subtask = subtask or getattr(sampler, "subtask", alias)
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
        actions = expand(alias, sc)
    except Exception as e:  # a program that cannot expand is a sampler-side bug too
        raise Reject("program_check_failed", f"expand failed: {e!r}", action="expand")
    report: Dict[str, Any] = {"mode": "expand", "actions": []}
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
        sampler.kit.sync_world()

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
        for i, action in enumerate(actions):
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
            try:
                targets = action.targets(sc, state)
            except Exception as e:
                raise Reject("program_check_failed", f"targets() failed: {e!r}", action=name)
            stance_name, carried = tracker.before(action)
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
            n, witnesses = check_action_targets(sidecar, sampler, action, targets, stance_name, carried, locked_arms=locked, stance_obj=tracker.stance_obj, q_seed_prev=q_prev,
                                                q_end_prev=q_end_prev, held_since_init=tracker.carried is not None and tracker.carried_arm is None)
            end_q = witnesses.pop("__arm_end_q__", None)
            partial_leg = witnesses.pop("__partial__", None)
            tracker.after(action, witnesses, sampler)
            tracker.note_arm_end(action, end_q)
            _carry_joint_state(action)
            entry = {"index": i, "name": name, "arms": list(getattr(action, "arms", ()) or []), "objects": list(getattr(action, "objects", ()) or []),
                     "stance": stance_name, "carried": carried, "n_poses": n, "witnesses": witnesses}
            if locked:
                entry["locked_arms"] = sorted(locked)  # the other arm(s) held at their contact configuration while these poses were solved
            if partial_leg is not None:
                entry["partial_leg"] = partial_leg
            report["actions"].append(entry)
    finally:
        tracker.restore_moved(sampler)  # block records advanced through the program go back to their init poses (the TRO is written from the live world)
        for obj_name, qs in joint_saved.items():
            obj = sampler.objects.get(obj_name)
            if obj is not None:
                obj.set_joint_positions(qs)
                obj.keep_still()
        if joint_saved:
            sampler.kit.sync_world()
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
        #: arm -> the (sampler-frame, 28-D) configuration the previous arm action ended in (block design chaining); reset by
        #: every base motion
        self.arm_end_q: Dict[str, th.Tensor] = {}
        #: where the robot stands after a hybrid ``base_follow`` (the base moved off the manipulation stance): the in-memory
        #: stance the following actions (the retreat) are checked from, until the next navigate / transition
        self.stance_obj: Optional[Stance] = None
        #: sim objects moved by :meth:`advance_block` (name -> init (pos, quat)), restored by :meth:`restore_moved`
        self.moved: Dict[str, Tuple[th.Tensor, th.Tensor]] = {}

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
        sampler.kit.sync_world()

    def advance_push(self, action, sampler) -> None:
        """Block design packing tail (owner 2026-09-19 21:47): a push of ``sidecar.pack`` moves its cube (and, for a wall push, the
        cube it drives ahead of it) by the recorded travel minus the blade's stand-off, so the next push's blade is checked
        against the cubes where the program leaves them (a pack push of 18 mm otherwise overlaps the un-moved neighbour in this
        static world). Restored by :meth:`restore_moved`."""
        contact = getattr(action, "contact", None)
        name = getattr(action, "object", None) or getattr(contact, "object", None)
        # the tracker holds the sampler's SidecarDraft (``pack`` attribute); the F6 Sidecar keeps it under ``extra``
        pack = getattr(self.sidecar, "pack", None)
        if not isinstance(pack, dict):
            pack = (self.sidecar.extra or {}).get("pack") if isinstance(getattr(self.sidecar, "extra", None), dict) else None
        if not isinstance(pack, dict) or sampler is None or name not in getattr(sampler, "objects", {}) or contact is None:
            if isinstance(pack, dict):
                log.warning("program check: pack push of %s not advanced (object unknown or no contact)", name)
            return
        rec, best = None, 1e9
        try:
            p_init = th.tensor([float(v) for v in contact.init.position])
        except Exception:  # noqa: BLE001
            return
        for stage in pack.get("stages") or []:
            for p_ in stage.get("pushes") or []:
                if p_.get("object") != name or not isinstance(p_.get("init"), dict):
                    continue
                d_ = float(th.linalg.norm(th.tensor([float(v) for v in p_["init"].get("position", [0, 0, 0])]) - p_init))
                if d_ < best:
                    rec, best = p_, d_
        if rec is None or best > 1e-3:
            log.warning("program check: no pack push record matches push of %s (nearest %.4f m) - the world is not advanced", name, best)
            return
        travel = float(rec.get("distance_m", 0.0)) - float(rec.get("standoff_m", 0.0))
        d = th.tensor([float(v) for v in rec.get("direction", [0, 0, 0])])
        if travel <= 0.0 or float(th.linalg.norm(d)) < 1e-6:
            return
        d = d / th.linalg.norm(d)
        d2 = d[:2] / max(float(th.linalg.norm(d[:2])), 1e-9)
        # the cubes ahead of the pushed one along the direction (the wall pushes drive the neighbour column / row along) - in the
        # table plane: a turned cube's ROOT sits an edge higher than an unturned one's (0920-173048: the driven right column was
        # missed on the 25 mm root-z difference and stayed 9 mm short of the sampler's world)
        pushed = sampler.objects[name]
        p_c, _ = pushed.get_position_orientation()
        edge = float(getattr(sampler, "edge", 0.0254))
        for other, obj in sampler.objects.items():
            if other == name or not any(other == b.get("instance") for b in (getattr(self.sidecar, "blocks", None) or [])):
                continue
            p_o, _ = obj.get_position_orientation()
            rel = (p_o - p_c)[:2]
            along = float(th.dot(rel, d2))
            across = float(th.linalg.norm(rel - along * d2))
            if 0.0 < along < travel + edge and across < 0.6 * edge:
                # it moves by what the pushed cube's travel leaves after closing the gap between them down to the sampler's packed
                # gap (its world model keeps ``block_gap_m`` between packed cubes; 0920-093727: 1 mm off put a hand 2.7 mm from
                # the neighbour where the sampler had 3.7)
                g_pack = float(sampler.g("block_gap_m")) if hasattr(sampler, "g") else 0.0
                gap = max(0.0, along - edge - g_pack)
                move = max(0.0, travel - gap)
                if move > 1e-4:
                    self._shift(other, obj, d * move, sampler)
        self._shift(name, pushed, d * travel, sampler)
        sampler.kit.sync_world()

    def _shift(self, name: str, obj, delta: th.Tensor, sampler) -> None:
        p0, q0 = obj.get_position_orientation()
        if name not in self.moved:
            self.moved[name] = (p0.clone(), q0.clone())
        obj.set_position_orientation(p0 + delta, q0)
        obj.keep_still()

    def restore_moved(self, sampler) -> None:
        if not self.moved or sampler is None:
            return
        for name, (p0, q0) in self.moved.items():
            obj = sampler.objects.get(name)
            if obj is not None:
                obj.set_position_orientation(p0, q0)
                obj.keep_still()
        self.moved.clear()
        sampler.kit.sync_world()

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
            self.advance_push(action, sampler)
        arms = list(getattr(action, "arms", ()) or [])
        if name == "brace" and arms and witnesses and sampler is not None:
            # the brace hand stays at its contact until its own retreat: its solved contact configuration locks the arm for
            # every action of the other arm in between (the push and the push arm's retreat)
            w = witnesses.get("contact") or witnesses.get("approach")
            if w is not None and w.get("q") is not None:
                q = th.tensor(w["q"], dtype=th.float32)
                self.arm_q[arms[0]] = q[sampler.kit.robot.arm_control_idx[arms[0]]].clone()
        elif name == "retreat" and arms:
            self.arm_q.pop(arms[0], None)


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
                sampler.kit.sync_world()
            p = _as_pose(pose)
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
            sampler.kit.sync_world()
    return n


def _ignore_for(action, sampler, carried: Optional[str], sidecar=None) -> Tuple[List[Any], List[Any]]:
    """(objects removed for the final pose only, objects removed for every waypoint) per §3.6: the support for grasp /
    place / contact end poses; the pushed object along a push; the braced object at the brace contact; the object placed
    on at the place pose (the carried object rests on it 1 cm above).

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


def _is_pack_push(sidecar, action) -> bool:
    """A push of the block-design packing tail (``sidecar.pack`` stages; the draft's attribute or the F6 ``extra``)."""
    if getattr(action, "name", "") != "push":
        return False
    pack = getattr(sidecar, "pack", None)
    if not isinstance(pack, dict):
        pack = (sidecar.extra or {}).get("pack") if isinstance(getattr(sidecar, "extra", None), dict) else None
    if not isinstance(pack, dict):
        return False
    obj = getattr(getattr(action, "contact", None), "object", None)
    return any(p_.get("object") == obj for st_ in (pack.get("stages") or []) for p_ in (st_.get("pushes") or []))


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


def fingertips_in_frame(sampler, stance: Stance, name: str, pose_targets: Sequence[Tuple[str, Sequence[Any]]], margin: float = FINGERTIP_FRAME_MARGIN,
                        slack_m: float = FINGERTIP_FRAME_SLACK_M) -> Optional[Dict[str, Any]]:
    """Owner 2026-09-25 (cap 0925-191101: ``gripper_out_of_fov`` twice at the place - the pre-attach 20 cm from the head camera put
    the fingertips over the frame's top edge, and the attachment check had only looked at the grasp, the attach and the brace):
    QA's own criterion at every IN-BETWEEN pose of an arm action inside the watched span - the fingertip point ``C.fingertip(eef)``
    inside the stance camera's image with a 5 % margin (``IKChecker.camera_pose``, the world's camera model) - for every recorded
    pose but the contact poses (``CONTACT_POSE_KEYS``: the box check of the task modules) and a place's post-release retreat leg.
    The oracle moves straight between the recorded poses, so the endpoints cover the motion (the carry's SLERP waypoints are all
    listed). Returns ``None`` when every pose passes, else ``{key, index, u, v, depth}`` of the first pose outside."""
    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling import geometry as G

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
            tip = C.fingertip((G.t(p[0]).float(), G.t(p[1]).float()))
            uv, depth = model.project(cam, G.t(tip).reshape(1, 3))
            u, v, d = float(uv[0, 0]), float(uv[0, 1]), float(depth[0])
            # QA's 5 % of the image, plus ``slack_m`` at the fingertip projected through its depth (execution error)
            px = float(model.fx) * slack_m / max(d, 0.05) if slack_m > 0 else 0.0
            mu, mv = margin * float(model.width) + px, margin * float(model.height) + px
            inside = d > 0.05 and mu <= u <= float(model.width) - mu and mv <= v <= float(model.height) - mv
            if not inside:
                return {"key": key, "index": i, "n": len(poses), "u": round(u, 1), "v": round(v, 1), "depth_m": round(d, 3), "slack_px": round(px, 1)}
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
    return skip


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


def check_action_targets(sidecar: SidecarDraft, sampler, action, targets: Any, stance_name: str, carried: Optional[str],
                         locked_arms: Optional[Dict[str, th.Tensor]] = None, stance_obj: Optional[Stance] = None,
                         q_seed_prev: Optional[th.Tensor] = None, q_end_prev: Optional[th.Tensor] = None,
                         held_since_init: bool = False) -> Tuple[int, Dict[str, Any]]:
    """IK / collision check of every eef pose / waypoint list in ``targets`` from the recorded stance ``stance_name``
    (its recorded configuration: working arm untucked, idle arm in its posture, trunk of the stance) with the ``carried``
    object attached to the carrying arm and posed in the hand. Base-pose targets (paths, transitions, base-follow) are
    whole-body collision-checked at the stance trunk. Returns ``(number of poses checked, witnesses)`` where
    ``witnesses`` maps each target key to ``{"pose": {position, orientation}, "q": [28 joints]}`` of the end pose.

    ``locked_arms`` (owner 2026‑09‑16 22:55 item 3): arm -> 7 arm-joint values of an arm that holds a contact while this
    action runs (the brace arm at its contact during the push and the push arm's retreat): those arms are **locked at
    that configuration** for every IK / collision check here — a two-arm exact IK as the wall / axial modules solve it —
    instead of the stance's idle posture; a pre-pose without a solution in that world is a rejection
    (``program_check_failed``), never a witness the oracle has to plan through the braced arm.

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
    g = sidecar.grasp_of(carried) if carried else None
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
    fingers = 0.0 if name in ("push", "brace") or (name == "retreat" and after_name == "push") else None
    if name == "brace":  # a wide stop braces with the fingers pre-shaped open (owner 2026-09-19 00:49)
        wide = (getattr(getattr(action, "contact", None), "extra", None) or {}).get("fingers_opening_m")
        if wide is not None:
            fingers = float(wide)
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
        q_seed = sampler.ikc.q_locked(stance, fingers=fingers).clone()
        for a, qa in locked.items():
            q_seed[sampler.kit.robot.arm_control_idx[a]] = th.as_tensor(qa, dtype=th.float32)
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
    where = f"stance {stance_name}, arm {arm}, carried {carried}" + (f", {'/'.join(sorted(locked))} arm locked at its contact" if locked else "")
    if name == "retreat" and getattr(action, "after", None) is not None:
        where += f", after {getattr(action.after, 'name', type(action.after).__name__)}"
    if name in ("push", "brace") and getattr(action, "contact", None) is not None:
        c_ = action.contact
        where += f", {name} of {getattr(c_, 'object', None)} {(getattr(c_, 'extra', None) or {}).get('pose_class') or ''}".rstrip()
    pose_targets = list(iter_pose_targets(targets))
    if name == "push" and isinstance(targets, dict) and _is_pack_push(sidecar, action):
        # a packing push (block design, owner 2026-09-19 21:47) is verified as the oracle executes it: ``arm_to(pre)`` then
        # ``arm_along(init -> goal)`` from the pre's configuration - ONE chain pre -> init -> ... -> goal tracked from the pre's
        # branch (the sampler's ``_pack_pushes`` checks this very chain). Checked as "pre" alone + "waypoints" from a fresh IK at
        # init, the init pose landed in a branch that collides with the neighbour cube 2-4 mm from the blade (0920-101139: 7/8
        # attempts 'waypoints: exact IK failed at pose 1 of 3 ... collides with <neighbour>')
        pre_l = [(k, p_) for k, p_ in pose_targets if k == "pre"]
        wps_l = [(k, p_) for k, p_ in pose_targets if k == "waypoints"]
        if pre_l and wps_l:
            rest = [(k, p_) for k, p_ in pose_targets if k not in ("pre", "waypoints")]
            pose_targets = [("waypoints", list(pre_l[0][1]) + list(wps_l[0][1]))] + rest
    # a floor-level push whose blade runs below the cuRobo ground slab's padding (``contacts.push.ground_ignored``, IORS push-under
    # 2026-09-20): the sampler solved the chain with the slab off, so is it checked; the oracle tracks the legs without a veto
    slab_off = bool(name == "push" and ((getattr(getattr(action, "contact", None), "extra", None) or {}).get("ground_ignored")))
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

    with attached(sampler.kit, attach_arm, carried_obj, eef_in_object=eef_in_object), sampler.ikc.ignoring([], ground=not slab_off):
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
                else:
                    # a single end pose (``grasp``, ``place``, ``contact``) is the end of its approach: the approached object is
                    # removed for it as along the tracked chain (owner A) - with it in the world the open pads straddling the
                    # target pass or fail at margin 0 by millimetres (teacup 18:41: 'grasp: exact IK failed at pose 1 of 1')
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
                    # a FLOOR-class lift (the grasp's support is the floor: IORS large-scale's floor init) keeps the ground slab off along
                    # the whole leg - the carried object's coarse sphere fit reaches below its mesh into the slab for the first
                    # centimetres (shampoo dispenser 0925-171050); the floor mesh is already in the ignore set
                    floor_lift = name == "lift" and any("floor" in str(getattr(o, "name", "")).lower() for o in ign_all)
                    res = sampler.ikc.continue_chain(q_start, chain, arm, ignore=ign_all, ignore_end=ign_end,
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
                    res = sampler.ikc.waypoints_ok(chain, stance, arm, ignore_for_last=ignore_last, ignore_all=ignore_all, cache=False, fingers=fingers, approached=approached,
                                                   q_seed=q_seed, track_dense=track, q_first=q_first)
                    if k_first is not None:
                        chain_notes[key] = f"leaves from the {k_first} witness"
                    if not res.ok and name == "push" and q_seed_prev is not None and not locked:
                        # a push's pre-contact pose is reached by the oracle's planned ``arm_to`` from wherever the arm is, so the
                        # branch the previous retreat ended in need not hold it: re-solve the chain from the stance posture (the
                        # sampler's seed) before failing (0920-080331: a pack push the sampler's identical chain check had passed)
                        res = sampler.ikc.waypoints_ok(chain, stance, arm, ignore_for_last=ignore_last, ignore_all=ignore_all, cache=False, fingers=fingers,
                                                       approached=approached, q_seed=sampler.ikc.q_locked(stance, fingers=fingers).clone(), track_dense=track, q_first=q_first)
                    if (res.ok and name == "push" and not track
                            and not (getattr(getattr(action, "contact", None), "extra", None) or {}).get("articulation")):
                        # owner 2026-09-26 (axial 0926-043254 / 051919): a free-object push whose pre-contact witness sits in another branch
                        # than the stance posture's IK is followed densely, as the oracle runs it (arm_to the pre-contact, then the legs
                        # from there) - only in that situation. The pre-contact is its own single target ("pre", solved before the chain;
                        # a pack push merges it into the chain and has q_first): the chain checked is pre + the waypoints from its witness
                        q_ref, chain_ref = q_first, list(chain)
                        if q_ref is None and "pre" in single_q:
                            pre_poses = next((p_ for k_, p_ in pose_targets if k_ == "pre"), None)
                            if pre_poses:
                                q_ref, chain_ref = single_q["pre"], list(pre_poses) + list(chain)
                        chk = None
                        if (q_ref is not None and len(chain_ref) >= 4 and not _is_pack_push(sidecar, action)
                                and os.environ.get("ISG_PUSH_REPLAY", "on").lower() not in ("off", "0", "false")):
                            # owner 2026-09-27 (A8): a free-object arc push is replayed as the oracle runs it - the straight leg, the true
                            # arc of the closed loop, the contact retreat's escape leg - and rejected when the replay cannot follow it
                            from ispatialgym.oracle.actions import RETREAT_DIST

                            tol_o = (sampler.gp.gen("tolerances") or {}).get("orientation") if hasattr(sampler, "gp") else None
                            band = float(tol_o) * _push_stop_frac_arc() if tol_o else 0.005
                            obj_ = sampler.objects.get(getattr(getattr(action, "contact", None), "object", None))
                            ok_r, why_r, rep_r = replay_arc_push(sampler, arm, q_ref, chain_ref[0], chain_ref[1:], targets.get("direction") if isinstance(targets, dict) else None,
                                                                 obj_, list(sampler.support_objects()), list(ignore_all) + list(ignore_last), band, RETREAT_DIST)
                            if rep_r:
                                log.info("program check: push replay of %s: %s %s", getattr(getattr(action, "contact", None), "object", None), why_r[:200], rep_r)
                            chain_notes[key] = why_r[:160]
                            if not ok_r:
                                chk = (False, why_r)
                        if chk is None and q_ref is not None and os.environ.get("ISG_FLIPPED_PUSH_CHECK", "on").lower() not in ("off", "0", "false"):
                            chk = _flipped_push_chain_check(sampler, chain_ref, stance, arm, q_ref, ignore_last, ignore_all, fingers, approached)
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
                if len(poses) == 1:
                    single_q[key] = q_pol.clone()
                else:
                    motion_ends.insert(0, q_pol.clone())  # the arm is here after this chain: the next chain of the action leaves from it
                    if q_start is not None:
                        witnesses[key]["branch"] = "tracked from the arm's configuration"
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
        for pose2d, trunk in checks:
            r_arm, l_arm, fingers_st = sampler.ikc.arm_vectors(stance)
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
                    lift_end = (th.as_tensor(g.grasp[0], dtype=th.float32) + th.tensor([0.0, 0.0, float(g.carry_height or 0.0)]), th.as_tensor(g.grasp[1], dtype=th.float32))
                    wps = getattr(g, "waypoints", None)
                    if wps:
                        # an in-hand move after the lift (``Reorient`` to the CARRY pose, owner 2026-09-20 05:20): the object rides at
                        # the waypoint end, not at the lift
                        end = wps[-1]
                        lift_end = (th.as_tensor(getattr(end, "position", end[0]), dtype=th.float32), th.as_tensor(getattr(end, "orientation", end[1]), dtype=th.float32))
                    supports = list(sampler.support_objects()) if hasattr(sampler, "support_objects") else []
                    with attached(sampler.kit, attach_arm, carried_obj, eef_in_object=eef_in_object):
                        r_lift = sampler.ikc.exact([lift_end], frm_st, attach_arm, ignore_for_last=supports, cache=False)
                    if r_lift.ok and r_lift.q is not None:
                        q_arms[attach_arm] = r_lift.q[sampler.kit.robot.arm_control_idx[attach_arm]].clone()
                drive_ctx = attached(sampler.kit, attach_arm, carried_obj, eef_in_object=eef_in_object)
            with drive_ctx:
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
    return n, witnesses
