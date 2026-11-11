"""Bounded, checked carry detours for explicitly enabled campaign recoveries.

The selector never moves the robot. It rejects incomplete IK, collisions,
protected-object near passes and paths outside the camera before returning a
candidate. The executor retains its normal motion and hard QA checks.
"""
from contextlib import nullcontext
import math

import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.oracle import protection as PROT
from ispatialgym.oracle import r1pro_model as M


def candidates(start, end, above_m):
    """At most nine paths; the recorded endpoints and orientations stay fixed."""
    p0, q0 = start
    p1, q1 = end
    delta = p1 - p0
    side = th.stack((-delta[1], delta[0], th.zeros_like(delta[0])))
    norm = float(th.linalg.norm(side))
    if norm > 1e-6:
        side = side / norm
    heights = list(dict.fromkeys((float(above_m), min(float(above_m), .08), min(float(above_m), .04))))
    for lateral in (0., .08, -.08):
        if lateral and norm <= 1e-6:
            continue
        for height in heights:
            z = max(float(p0[2]), float(p1[2])) + height
            a, b = p0.clone(), p1.clone()
            a[2] = b[2] = z
            a += side * lateral
            b += side * lateral
            yield f'rise_{height:.3f}_side_{lateral:.3f}', [(a, q0), (b, q1), (p1, q1)]


def choose(motions, arm, end, *, ignore, monitor):
    facts = []
    protection = getattr(motions, '_protect', None)
    backend = motions.backend
    camera = getattr(monitor, 'camera', None)
    if (not protection or not callable(getattr(backend, 'robot_spheres_world', None))
            or camera is None or not callable(getattr(monitor, 'head_camera_pose', None))):
        return None, [{'reason': 'required protection or camera observation unavailable'}]
    start = motions.eef_pose(arm)
    camera_pose = monitor.head_camera_pose()
    margin = float(monitor.cfg.fov_margin)
    q0 = backend.q_full()
    emb = f'{arm}_arm'
    frame = M.EEF_LINK[arm]
    for label, path in candidates(start, end, motions.cfg.protect_above_m):
        row = {'candidate': label, 'accepted': False}
        facts.append(row)
        try:
            dense = motions._densify(start, path)
            # Use the actual QA frame margin, with no extra tolerance.
            visible = True
            for pose in [start] + dense:
                u, v, depth = camera.project(camera_pose, C.fingertip(pose))
                if not (all(math.isfinite(float(x)) for x in (u, v, depth)) and depth > .05
                        and margin*camera.width <= u <= (1-margin)*camera.width
                        and margin*camera.height <= v <= (1-margin)*camera.height):
                    visible = False
                    break
            if not visible:
                row['reason'] = 'fingertip_out_of_frame'
                continue
            backend.update_world(ignore_objects=list(ignore))
            relax = getattr(backend, 'start_collisions_relaxed', None)
            with relax(emb, q0, [arm]) if relax else nullcontext([]):
                result = backend.solve_ik_path(emb, [{frame: pose} for pose in dense], q_start=q0,
                    pos_tol=motions.cfg.diff_ik_pos_tol, rot_tol=motions.cfg.diff_ik_rot_tol,
                    max_iters=motions.cfg.diff_ik_max_iters, check_collisions=True, sync_world=False,
                    soft_pos_tol=motions.cfg.diff_ik_soft_pos_tol_free,
                    soft_rot_tol=motions.cfg.diff_ik_soft_rot_tol)
            if not result.success or result.q_traj is None or len(result.q_traj) < 3:
                row['reason'] = 'incomplete_ik_or_collision: ' + str(result.status)[:150]
                continue
            spheres, links = backend.robot_spheres_world(result.q_traj, emb)
            violations = PROT.check_trajectory(spheres, links, [arm], protection['objects'], protection['inflation'])
            if violations:
                row.update(reason='protected_object_clearance', violations=violations)
                continue
            row.update(accepted=True, reason='complete_ik_collision_clear_protected_clear_in_frame')
            return path, facts
        except Exception as exc:
            # Failure to prove a candidate is never a pass.
            row['reason'] = 'validation_unavailable: ' + repr(exc)[:200]
    return None, facts
