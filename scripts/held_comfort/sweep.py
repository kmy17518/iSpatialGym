"""Comfort sweep of the held-object hold for mental rotation (owner 2026-09-26): how comfortably the robot holds the
object at a pad depth along the head camera's optical axis, for the object-scaling gripper orientation (roll 90) and the
current mental-rotation hold as the reference (roll 15, 0.50 m).

Every (roll, depth) hold is solved as ``HeldObjectV2.solve_hold`` solves it (hold-cache hit verified by FK, else one cuRobo
IK of the hold eef pose from the default posture - arms zero, trunk (0.5, 0) - with the scene ignored, free of self-collision)
in an empty scene; the cache is only read. Each hold is scored:

* ``clearance``: the smallest distance of a right-arm joint to its limit (rad, and the joint); the pipeline's own witness
  margin is 0.05 rad (``IKChecker.JOINT_LIMIT_MARGIN``). ``best_alt`` = the largest clearance over ``--alt-seeds`` extra IK
  solves of the same hand pose from random arm seeds: whether another arm configuration holds it more comfortably;
* ``dexterity``: the smallest singular value of the hand's rotational Jacobian (rad of hand rotation per rad of joint motion,
  finite differences of the planner FK; near 0 the hand cannot turn about some axis) and the translational manipulability
  sqrt(det(J J^T)) (m^3);
* ``rotations``: a fixed grid of rotations screened from the hold the way the sampler screens its continuous draws
  (13 axes x the signed 15 deg grid of ``goal_angle_range_deg``, ``held_object.rotation_goal_set``; 3 SLERP waypoints + the
  goal per candidate, right arm only with the object attached, robot + object only, the whole object in frame with
  ``in_frame_margin``): feasible (grasp, rotation) pairs of all candidates, and per axis the share of signed angles some
  grasp reaches;
* ``visibility``: the share of the object's silhouette the robot leaves visible at the hold, per library grasp (segmentation
  renders with / without the robot, as ``scripts/held_visibility/survey.py``; the sampler needs ``min_visible_fraction``).

A depth is *comfortable* when the hold solves, the object is in frame for every grasp, the clearance is at least 0.05 rad,
at least one grasp leaves ``min_visible_fraction`` visible and the feasible rotations reach 90 % of the sweep's best depth.

Output: ``<out>/comfort.json`` (every number, the holds' arm configurations), ``<out>/README.md`` (the table, the
comfortable range) and ``<out>/comfort.png``.

Usage (local 4090, one process, about 2 min per configuration)::

    OMNIGIBSON_HEADLESS=1 python scripts/held_comfort/sweep.py --out runlogs/held_comfort_20260926 \\
        [--variant mental_rotation-banana] [--model banana/znakxm] [--roll 90] [--depths 0.23:0.53:0.03] [--reference 15:0.50] \\
        [--alt-seeds 16]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch as th  # noqa: E402

from scripts.held_visibility import survey as SV  # noqa: E402  (the empty world; imports omnigibson after the warp pinning)

import omnigibson as og  # noqa: E402

from ispatialgym.oracle import r1pro_model as M  # noqa: E402
from ispatialgym.robot import conventions as C  # noqa: E402
from ispatialgym.sampling import generation_defaults as GD  # noqa: E402
from ispatialgym.sampling import geometry as G  # noqa: E402
from ispatialgym.sampling import grasp_library as GL  # noqa: E402
from ispatialgym.sampling.customs import DEFAULT_CUSTOMS_PATH  # noqa: E402
from ispatialgym.sampling.robot_utils import RobotKit  # noqa: E402
from ispatialgym.sampling.v2.navigation import attached  # noqa: E402
from ispatialgym.sampling.v2.tasks import held_object as HO  # noqa: E402

log = logging.getLogger("held_comfort")
SCHEMA = "held-comfort-v1"
#: the joint-limit margin the pipeline asks of a chosen witness (``IKChecker.JOINT_LIMIT_MARGIN``)
CLEARANCE_MIN_RAD = 0.05
#: comfortable depths reach this share of the best depth's feasible rotations
NEAR_BEST = 0.90
FD_EPS = 1e-3


def generation(variant: str, subtask: str) -> Dict[str, Any]:
    """The variant's customs ``generation`` over the subtask's ``generation_defaults`` (as ``HeldObjectV2.g``)."""
    with open(DEFAULT_CUSTOMS_PATH) as f:
        entry = json.load(f).get(variant) or {}
    return {**GD.defaults_for(subtask), **(entry.get("generation") or {})}


def others(world) -> List[Any]:
    return [o for o in world.scene.objects if o is not world.robot]


def arm_clearance(robot, q: th.Tensor) -> Tuple[float, str, List[float]]:
    idx = robot.arm_control_idx[HO.ARM]
    lo, hi = robot.joint_lower_limits[idx].float(), robot.joint_upper_limits[idx].float()
    qa = G.t(q)[idx].float()
    m = th.minimum(qa - lo, hi - qa)
    k = int(th.argmin(m))
    return float(m[k]), M.ARM_JOINTS[HO.ARM][k], [round(float(v), 4) for v in m.tolist()]


def dexterity(kit, robot, q: th.Tensor) -> Dict[str, float]:
    """Singular values of the hand's rotational / translational Jacobian w.r.t. the 7 right-arm joints (finite differences)."""
    link = M.EEF_LINK[HO.ARM]
    p0, q0 = (G.t(x).float() for x in kit.fk([link], q)[link])
    cols_p, cols_r = [], []
    for i in robot.arm_control_idx[HO.ARM].tolist():
        qq = G.t(q).clone()
        qq[i] += FD_EPS
        p1, q1 = (G.t(x).float() for x in kit.fk([link], qq)[link])
        dq = C.qmul(q1, C.qinv(q0))
        if float(dq[3]) < 0.0:
            dq = -dq
        cols_p.append((p1 - p0) / FD_EPS)
        cols_r.append(2.0 * dq[:3] / FD_EPS)
    sp, sr = th.linalg.svdvals(th.stack(cols_p, 1)), th.linalg.svdvals(th.stack(cols_r, 1))
    return {"rot_sigma_min": round(float(sr.min()), 4), "rot_sigma_max": round(float(sr.max()), 4), "pos_sigma_min": round(float(sp.min()), 4),
            "pos_manipulability": round(float(th.prod(sp)), 6)}


def hold_frames(kit, hc: HO.HoldConstants):
    x, y, yaw = kit.world_base_pose2d()
    q0 = kit.q_at_base_pose(kit.q_config(arms=HO.IDLE_ARMS, translate=hc.translate, tilt=hc.tilt), x, y, yaw)
    cam = kit.camera_pose(q0)
    eef_w, _pad = HO.hold_eef_pose(cam, yaw, hc)
    return q0, cam, eef_w, (x, y, yaw)


def solve_hold(world, kit, hc: HO.HoldConstants, q0: th.Tensor, eef_w) -> Tuple[Optional[th.Tensor], str]:
    """``HeldObjectV2.solve_hold`` without writing the cache: a cache hit verified by FK, else one IK from the default posture."""
    arm_idx = world.robot.arm_control_idx[HO.ARM]
    hit = HO._read_json(HO.HOLD_CACHE_PATH).get(HO.hold_cache_key("r1pro", hc, HO.hold_cache_extras(kit)))
    with kit.ignoring(others(world)):
        if hit is not None:
            qc = q0.clone()
            qc[arm_idx] = th.tensor(hit["right_arm_q"], dtype=th.float32)
            eef = kit.eef_pose(qc, HO.ARM)
            if float(th.linalg.norm(G.t(eef[0]) - G.t(eef_w[0]))) < 0.002 and C.quat_angle(eef[1], eef_w[1]) < math.radians(1.0):
                return qc, "cache"
        q = kit.ik(eef_w, q0, arm=HO.ARM)
        if q is None or not bool(kit.collision_free(q, margin=0.01)[0]):
            return None, "no IK solution" if q is None else "self-collides"
    return q, "solved"


def alternatives(world, kit, q0: th.Tensor, eef_w, n: int, gen: th.Generator) -> List[th.Tensor]:
    """IK solves of the same hand pose from random right-arm seeds (self-collision free)."""
    idx = world.robot.arm_control_idx[HO.ARM]
    lo, hi = world.robot.joint_lower_limits[idx].float(), world.robot.joint_upper_limits[idx].float()
    out = []
    with kit.ignoring(others(world)):
        for _ in range(int(n)):
            seed = q0.clone()
            seed[idx] = lo + (hi - lo) * th.rand(len(idx), generator=gen)
            q = kit.ik(eef_w, seed, arm=HO.ARM)
            if q is not None and bool(kit.collision_free(q, margin=0.01)[0]):
                out.append(q)
    return out


def in_frame(world, cam, hull, pose, margin: float) -> bool:
    pts = world.hull_at(hull, pose)
    return bool(world.camera_model.all_inside(cam, G.aabb_corners(pts.min(dim=0).values, pts.max(dim=0).values), margin))


def screen_rotations(world, kit, obj, hull, grasps, q_hold: th.Tensor, eef_w, cam, base, gp: Dict[str, Any]) -> Dict[str, Any]:
    """The rotation grid (``HO.rotation_goal_set``) screened at this hold: feasible (grasp, rotation) pairs and per-axis
    coverage."""
    margin = float(gp["in_frame_margin"])
    cands = HO.rotation_goal_set(gp["goal_angle_range_deg"], HO.ROTATION_ANGLE_STEP_DEG)
    n_wp = int(gp["waypoints"])
    base_q = C.quat_from_yaw(base[2])
    axes = HO.rotation_axes()
    reached = [set() for _ in range(len(axes))]  # per axis: the signed angles some grasp reaches
    stats = {"grasps": len(grasps), "candidates": 0, "held_out_of_view": 0, "goal_out_of_view": 0, "ik_failed": 0, "feasible": 0}
    per_grasp: Dict[str, Optional[int]] = {}
    for g in grasps:
        obj0 = C.compose(eef_w, C.invert(g.pose))
        if not in_frame(world, cam, hull, obj0, margin):
            stats["held_out_of_view"] += 1
            per_grasp[g.name] = None
            continue
        entries = []
        for axis_base, angle in cands:
            stats["candidates"] += 1
            dq = C.quat_from_axis_angle(C.quat_rotate(base_q, axis_base), math.radians(angle))
            goal = (G.t(obj0[0]).clone(), C.qmul(dq, G.t(obj0[1])))
            if not in_frame(world, cam, hull, goal, margin):
                stats["goal_out_of_view"] += 1
                continue
            eef1 = C.compose(goal, g.pose)
            wps = HO.orbit_key_poses(eef_w, G.t(obj0[0]), C.qmul(G.t(eef1[1]), C.qinv(G.t(eef_w[1]))), n_wp)
            k = int(th.argmin(th.linalg.norm(axes - axis_base.unsqueeze(0), dim=1)))
            entries.append((k, float(angle), wps))
        per_grasp[g.name] = 0
        if not entries:
            continue
        kit.apply(q_hold, base)
        obj.set_position_orientation(obj0[0], obj0[1])
        obj.keep_still()
        kit.sync_world()
        with kit.ignoring(others(world)), attached(kit, HO.ARM, obj):
            ok = kit.ik_batch([p for _, _, w in entries for p in w], q_hold, arm=HO.ARM)
        ok = ok.view(len(entries), len(entries[0][2])).all(dim=1)
        for (k, angle, _), s in zip(entries, ok.tolist()):
            if s:
                stats["feasible"] += 1
                per_grasp[g.name] += 1
                reached[k].add(angle)
            else:
                stats["ik_failed"] += 1
    n_angles = len(cands) // len(axes)
    coverage = [len(r) / n_angles for r in reached]
    return {**stats, "feasible_fraction": round(stats["feasible"] / max(1, len(grasps) * len(cands)), 4), "per_grasp": per_grasp,
            "axis_coverage": [round(c, 3) for c in coverage], "axis_coverage_mean": round(sum(coverage) / len(coverage), 4),
            "axes_full": sum(1 for c in coverage if c >= 1.0 - 1e-9), "max_angle_per_axis_deg": [max((abs(a) for a in r), default=0.0) for r in reached]}


def visibility(world, kit, obj, grasps, q_hold: th.Tensor, eef_w, base) -> Dict[str, Optional[float]]:
    gidx = world.robot.gripper_control_idx[HO.ARM]
    out: Dict[str, Optional[float]] = {}
    for g in grasps:
        obj0 = C.compose(eef_w, C.invert(g.pose))
        kit.apply(q_hold, base)
        if g.finger_q and len(g.finger_q) == len(gidx):
            world.robot.set_joint_positions(th.tensor(g.finger_q, dtype=th.float32), indices=gidx, drive=False)
        obj.set_position_orientation(obj0[0], obj0[1])
        obj.disable_gravity()
        obj.wake()
        obj.keep_still()
        world.robot.keep_still()
        world.render(2)
        try:
            vis, ref = SV.object_pixels_settled(world, obj)
            out[g.name] = round(HO.visible_fraction(vis, ref), 4) if ref > 0 else None
        except Exception as e:  # noqa: BLE001 - one render must not stop the sweep
            log.warning("visibility render failed for %s: %r", g.name, e)
            out[g.name] = None
    return out


def score(world, kit, obj, hull, grasps, hc: HO.HoldConstants, gp: Dict[str, Any], n_alt: int, gen: th.Generator) -> Dict[str, Any]:
    t0 = time.time()
    q0, cam, eef_w, base = hold_frames(kit, hc)
    arm_idx = world.robot.arm_control_idx[HO.ARM]
    rec: Dict[str, Any] = {"roll_deg": hc.roll_deg, "depth_m": hc.depth_m, "trunk": [hc.translate, hc.tilt]}
    q, source = solve_hold(world, kit, hc, q0, eef_w)
    rec["hold"] = source
    if q is None:
        rec["seconds"] = round(time.time() - t0, 1)
        return rec
    c, cj, per = arm_clearance(world.robot, q)
    rec.update({"right_arm_q": [round(float(v), 5) for v in q[arm_idx].tolist()], "clearance_rad": round(c, 4), "clearance_joint": cj, "clearance_per_joint_rad": per,
                "dexterity": dexterity(kit, world.robot, q)})
    alts = alternatives(world, kit, q0, eef_w, n_alt, gen)
    if alts:
        best = max(alts, key=lambda x: arm_clearance(world.robot, x)[0])
        cb, cbj, _ = arm_clearance(world.robot, best)
        rec["best_alt"] = {"solutions": len(alts), "clearance_rad": round(cb, 4), "clearance_joint": cbj, "right_arm_q": [round(float(v), 5) for v in best[arm_idx].tolist()],
                           "dexterity": dexterity(kit, world.robot, best)}
    rec["rotations"] = screen_rotations(world, kit, obj, hull, grasps, q, eef_w, cam, base, gp)
    if alts and c < CLEARANCE_MIN_RAD <= rec["best_alt"]["clearance_rad"]:
        rec["best_alt"]["rotations"] = screen_rotations(world, kit, obj, hull, grasps, best, eef_w, cam, base, gp)
    rec["visibility"] = visibility(world, kit, obj, grasps, q, eef_w, base)
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def classify(records: List[Dict[str, Any]], roll: float, min_vis: float) -> Dict[str, Any]:
    """The comfortable depths of the ``roll`` sweep (module docstring) and the most comfortable one."""
    sweep = [r for r in records if r["roll_deg"] == roll and not r.get("reference")]
    best = max((r["rotations"]["feasible"] for r in sweep if "rotations" in r), default=0)
    for r in records:
        why = []
        if "rotations" not in r:
            why.append(f"hold: {r['hold']}")
        else:
            if r["rotations"]["held_out_of_view"]:
                why.append(f"{r['rotations']['held_out_of_view']} grasp(s) out of frame at the hold")
            if r["clearance_rad"] < CLEARANCE_MIN_RAD:
                why.append(f"{r['clearance_joint']} {math.degrees(r['clearance_rad']):.2f} deg from its limit")
            if not any(v is not None and v >= min_vis for v in (r.get("visibility") or {}).values()):
                why.append(f"no grasp leaves {min_vis:.0%} visible")
            if best and r["rotations"]["feasible"] < NEAR_BEST * best:
                why.append(f"feasible rotations {r['rotations']['feasible']} < {NEAR_BEST:.0%} of the best ({best})")
        r["comfortable"] = not why
        r["not_comfortable_because"] = why
    ok = [r for r in sweep if r["comfortable"]]
    top = max(ok, key=lambda r: (r["rotations"]["feasible"], r["clearance_rad"])) if ok else None
    depths = sorted(r["depth_m"] for r in sweep)
    rng = None
    if top is not None:
        comfortable = {r["depth_m"] for r in ok}
        i = depths.index(top["depth_m"])
        lo = hi = i
        while lo > 0 and depths[lo - 1] in comfortable:
            lo -= 1
        while hi < len(depths) - 1 and depths[hi + 1] in comfortable:
            hi += 1
        rng = [depths[lo], depths[hi]]
    return {"roll_deg": roll, "best_feasible": best, "most_comfortable_depth_m": top["depth_m"] if top else None, "comfortable_range_m": rng,
            "comfortable_depths_m": sorted(r["depth_m"] for r in ok)}


def write_readme(path: str, data: Dict[str, Any]) -> None:
    s = data["summary"]
    lines = [f"# Held-object comfort sweep ({data['date'][:10]})", "",
             f"`{data['variant']}` ({data['model']}, {data['n_grasps']} library grasps), hold solved as the sampler solves it, empty scene. "
             f"Generated by `scripts/held_comfort/sweep.py` (commit {data['commit']}); every number is in `comfort.json`.", "",
             f"**Roll {s['roll_deg']:.0f} deg sweep:** most comfortable depth **{_cm(s['most_comfortable_depth_m'])}**, comfortable range "
             f"**{_cm_range(s['comfortable_range_m'])}** (comfortable = hold solved, object in frame for every grasp, joint clearance >= "
             f"{math.degrees(CLEARANCE_MIN_RAD):.1f} deg, some grasp >= {data['min_visible_fraction']:.0%} visible, feasible rotations >= "
             f"{NEAR_BEST:.0%} of the best depth).", "",
             "| roll | depth | hold | nearest joint limit | best other config | hand rotation dexterity | feasible rotations | axis coverage | visible grasps | comfortable |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in data["records"]:
        tag = " (current MR hold)" if r.get("reference") else ""
        if "rotations" not in r:
            lines.append(f"| {r['roll_deg']:.0f}{tag} | {_cm(r['depth_m'])} | {r['hold']} | - | - | - | - | - | - | no |")
            continue
        alt = r.get("best_alt") or {}
        vis = r.get("visibility") or {}
        n_vis = sum(1 for v in vis.values() if v is not None and v >= data["min_visible_fraction"])
        rot = r["rotations"]
        lines.append(f"| {r['roll_deg']:.0f}{tag} | {_cm(r['depth_m'])} | {r['hold']} | {math.degrees(r['clearance_rad']):.1f} deg ({r['clearance_joint']}) | "
                     f"{(str(round(math.degrees(alt['clearance_rad']), 1)) + ' deg') if alt else '-'} | {r['dexterity']['rot_sigma_min']:.2f} | "
                     f"{rot['feasible']} / {rot['grasps'] * len(HO.rotation_goal_set(data['generation']['goal_angle_range_deg']))} ({rot['feasible_fraction']:.0%}) | "
                     f"{rot['axis_coverage_mean']:.0%} ({rot['axes_full']} full) | {n_vis} / {len(vis)} | {'yes' if r['comfortable'] else 'no: ' + '; '.join(r['not_comfortable_because'])} |")
    lines += ["", "Columns: *nearest joint limit* = the smallest distance of a right-arm joint to its limit at the solved hold; *best other config* = "
              "the largest such distance over IK solves of the same hand pose from random arm seeds; *hand rotation dexterity* = the smallest "
              "singular value of the hand's rotational Jacobian (0 = a direction the wrist cannot turn); *feasible rotations* = (grasp, rotation) "
              "pairs of the mental-rotation goal set that pass the sampler's screening from the hold; *axis coverage* = mean share of the "
              "signed angles per rotation axis some grasp reaches (full = all of them); *visible grasps* = grasps leaving at least "
              f"{data['min_visible_fraction']:.0%} of the object visible.", "",
              "To move mental rotation's start there, set `generation.hold` of the mental-rotation variants (`depth_m`, `roll_deg`); the "
              "solved arm configuration of every hold is in `comfort.json` (`right_arm_q`) should the sampler's own IK land elsewhere."]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _cm(d: Optional[float]) -> str:
    return "-" if d is None else f"{100 * d:.0f} cm"


def _cm_range(r: Optional[List[float]]) -> str:
    return "none" if not r else (_cm(r[0]) if r[0] == r[1] else f"{_cm(r[0])} - {_cm(r[1])}")


def write_plot(path: str, data: Dict[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available: no plot")
        return
    roll = data["summary"]["roll_deg"]
    rs = sorted((r for r in data["records"] if r["roll_deg"] == roll and not r.get("reference") and "rotations" in r), key=lambda r: r["depth_m"])
    ref = [r for r in data["records"] if r.get("reference") and "rotations" in r]
    d = [100 * r["depth_m"] for r in rs]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(d, [100 * r["rotations"]["feasible_fraction"] for r in rs], "o-", label=f"roll {roll:.0f}")
    ax[1].plot(d, [math.degrees(r["clearance_rad"]) for r in rs], "o-", label="solved hold")
    ax[1].plot(d, [math.degrees((r.get("best_alt") or {}).get("clearance_rad", float("nan"))) for r in rs], "x--", label="best other configuration")
    ax[1].axhline(math.degrees(CLEARANCE_MIN_RAD), color="grey", lw=0.8, ls=":", label="pipeline margin")
    ax[2].plot(d, [r["dexterity"]["rot_sigma_min"] for r in rs], "o-", label=f"roll {roll:.0f}")
    for r in ref:
        ax[0].plot([100 * r["depth_m"]], [100 * r["rotations"]["feasible_fraction"]], "r*", ms=12, label=f"current MR hold (roll {r['roll_deg']:.0f})")
        ax[1].plot([100 * r["depth_m"]], [math.degrees(r["clearance_rad"])], "r*", ms=12, label="current MR hold")
        ax[2].plot([100 * r["depth_m"]], [r["dexterity"]["rot_sigma_min"]], "r*", ms=12, label="current MR hold")
    rng = data["summary"]["comfortable_range_m"]
    for a in ax:
        if rng:
            a.axvspan(100 * rng[0] - 1, 100 * rng[1] + 1, color="green", alpha=0.08)
        a.set_xlabel("pad depth along the camera axis (cm)")
        a.legend(fontsize=8)
    ax[0].set_ylabel("feasible rotations (% of candidates)")
    ax[1].set_ylabel("nearest joint limit (deg)")
    ax[2].set_ylabel("hand rotation dexterity (sigma min)")
    fig.suptitle(f"{data['variant']}: hold comfort vs depth (green = comfortable range)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _depths(text: str) -> List[float]:
    lo, hi, step = (float(v) for v in text.split(":"))
    n = int(round((hi - lo) / step))
    return [round(lo + k * step, 4) for k in range(n + 1)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(REPO, "runlogs", "held_comfort_20260926"))
    parser.add_argument("--variant", default="mental_rotation-banana")
    parser.add_argument("--model", default="banana/znakxm")
    parser.add_argument("--roll", type=float, default=90.0)
    parser.add_argument("--depths", default="0.23:0.53:0.03", help="lo:hi:step (m)")
    parser.add_argument("--reference", default="15:0.50", help="roll:depth of the reference hold ('' = none)")
    parser.add_argument("--alt-seeds", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="[%(levelname)s %(name)s] %(message)s", stream=sys.stdout, force=True)
    os.makedirs(args.out, exist_ok=True)
    subtask = args.variant.split("-")[0]
    gp = generation(args.variant, subtask)
    hold0 = HO.HoldConstants.from_generation(gp["hold"])
    configs = [(args.roll, d, False) for d in _depths(args.depths)]
    if args.reference:
        r_roll, r_depth = (float(v) for v in args.reference.split(":"))
        configs.append((r_roll, r_depth, True))
    picks = SV.target_picks((subtask,), ("representative", "variation_geometry", "heldout", "variation_texture"), [args.model])
    if not picks:
        raise SystemExit(f"{args.model} is not a {subtask} target pick")
    pick = picks[0]
    grasps = GL.load_grasps(HO.floating_key(pick["category"], pick["model"], pick.get("grasp_key")))
    started = _dt.datetime.now().isoformat(timespec="seconds")
    commit = os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip()
    world = SV.EmptyWorld()
    kit = RobotKit(world, prebuild=("right_arm",))
    gen = th.Generator().manual_seed(int(args.seed))
    records: List[Dict[str, Any]] = []
    obj = world.spawn(f"comfort_{pick['category']}_{pick['model']}", pick["category"], pick["model"], scale=pick.get("scale"))
    try:
        hull = world.local_hull(obj)
        data: Dict[str, Any] = {}
        for i, (roll, depth, ref) in enumerate(configs):
            hc = HO.HoldConstants(float(depth), hold0.translate, hold0.tilt, float(roll))
            rec = score(world, kit, obj, hull, grasps, hc, gp, args.alt_seeds, gen)
            rec["reference"] = ref
            records.append(rec)
            rot = rec.get("rotations") or {}
            log.info("[%d/%d] roll %.0f depth %.2f: hold %s, clearance %s, feasible %s/%s, %.0fs", i + 1, len(configs), roll, depth, rec["hold"],
                     f"{math.degrees(rec['clearance_rad']):.2f} deg ({rec['clearance_joint']})" if "clearance_rad" in rec else "-", rot.get("feasible"),
                     rot.get("candidates"), rec["seconds"])
            data = {"schema": SCHEMA, "date": started, "finished": _dt.datetime.now().isoformat(timespec="seconds"), "commit": commit, "variant": args.variant,
                    "model": args.model, "n_grasps": len(grasps), "grasp_key": HO.floating_key(pick["category"], pick["model"], pick.get("grasp_key")),
                    "generation": {k: gp.get(k) for k in ("hold", "goal_angle_range_deg", "waypoints", "in_frame_margin", "min_visible_fraction")},
                    "min_visible_fraction": float(gp["min_visible_fraction"]), "clearance_min_rad": CLEARANCE_MIN_RAD, "near_best": NEAR_BEST,
                    "alt_seeds": args.alt_seeds, "records": records}
            data["summary"] = classify(records, args.roll, float(gp["min_visible_fraction"]))
            with open(os.path.join(args.out, "comfort.json"), "w") as f:
                json.dump(data, f, indent=1)
        write_readme(os.path.join(args.out, "README.md"), data)
        write_plot(os.path.join(args.out, "comfort.png"), data)
        print("HELD_COMFORT_REPORT " + json.dumps(data["summary"]), flush=True)
    finally:
        world.remove(obj)
        og.shutdown()


if __name__ == "__main__":
    main()
