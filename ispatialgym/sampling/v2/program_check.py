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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch as th

from ispatialgym.sampling.v2.stances import PathRecord, SidecarDraft, Stance, pose_from_sidecar

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
    "alignment-world_vertical": ("stances.init", "stances.manipulation", "contacts.push", "contacts.brace"),
    "alignment-world_horizontal": ("stances.init", "stances.manipulation", "contacts.push", "contacts.brace"),
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
    """The ``State`` of oracle doc §3.6 backed by the sampler's world at init: ``object_pose(name)`` -> (pos, quat)."""

    def __init__(self, sampler):
        self.sampler = sampler

    def object_pose(self, name: str):
        obj = self.sampler.objects[name]
        p, q = obj.get_position_orientation()
        return p.float().clone(), q.float().clone()


def _sidecar_object(sd: Dict[str, Any]):
    """``oracle.sidecar.Sidecar`` built from the draft dict (``tro_sha256`` placeholder before the TRO exists)."""
    from ispatialgym.oracle import sidecar as SC  # type: ignore

    d = dict(sd)
    if not isinstance(d.get("tro_sha256"), str):
        d["tro_sha256"] = "0" * 64
    return SC.Sidecar.from_dict(d)


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
    state = _SamplerState(sampler)
    tracker = _Tracker(sidecar)
    for i, action in enumerate(actions):
        name = getattr(action, "name", type(action).__name__)
        try:
            targets = action.targets(sc, state)
        except Exception as e:
            raise Reject("program_check_failed", f"targets() failed: {e!r}", action=name)
        stance_name, carried = tracker.before(action)
        n, witnesses = check_action_targets(sidecar, sampler, action, targets, stance_name, carried)
        tracker.after(action)
        entry = {"index": i, "name": name, "arms": list(getattr(action, "arms", ()) or []), "objects": list(getattr(action, "objects", ()) or []),
                 "stance": stance_name, "carried": carried, "n_poses": n, "witnesses": witnesses}
        report["actions"].append(entry)
    # the recorded joint solutions (28-D, arm joints + trunk + fingers meaningful) per checked end pose: D5 verifies them as
    # witnesses (FK + collision) instead of re-solving the randomised IK
    sidecar.ik_witnesses = [{"action": e["index"], "name": e["name"], "stance": e["stance"], "arm": (e["arms"] or ["right"])[0], "carried": e["carried"],
                            "targets": e["witnesses"]} for e in report["actions"] if e["witnesses"]]
    return report


class _Tracker:
    """Where the robot is and what it holds while the program runs: ``Navigate`` -> ``path.to``, ``Transition`` ->
    ``transition.to``; ``Grasp`` picks up ``object`` (held until ``Place``)."""

    def __init__(self, sidecar: SidecarDraft):
        self.stance = "init"
        init = sidecar.stances.get("init")
        # the held-object tasks start with the object in the hand (§6.3): the init stance records ``carried``
        self.carried: Optional[str] = init.carried if init is not None else None
        self.sidecar = sidecar

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
            for k, v in self.sidecar.stances.items():
                if list(v.base_pose2d) == list(action.stance.base_pose2d):
                    stance = k
        return stance, carried

    def after(self, action) -> None:
        name = getattr(action, "name", "")
        if name == "navigate" and getattr(action, "path", None) is not None:
            self.stance = action.path.to
        elif name == "transition" and getattr(action, "transition", None) is not None:
            self.stance = action.transition.to
        if name == "grasp":
            self.carried = getattr(action, "object", None)
        elif name == "place":
            self.carried = None


def _ignore_for(action, sampler, carried: Optional[str]) -> Tuple[List[Any], List[Any]]:
    """(objects removed for the final pose only, objects removed for every waypoint) per §3.6: the support for grasp /
    place / contact end poses; the pushed object along a push; the braced object at the brace contact; the object placed
    on at the place pose (the carried object rests on it 1 cm above)."""
    name = getattr(action, "name", "")
    supports = list(sampler.support_objects())
    objs = sampler.objects
    last: List[Any] = []
    every: List[Any] = []
    # a module may record exactly what it ignored while solving a chain (names of bddl instances or "supports") on the
    # sidecar record the action carries: ``ignore_every`` / ``ignore_last`` in ``contacts.*`` / ``grasps[]`` extras
    rec = getattr(action, "contact", None) or getattr(action, "grasp", None)
    explicit = _explicit_ignores(rec, sampler)
    if explicit is not None:
        return explicit
    if name == "grasp":
        last += supports  # the final grasp pose only
    if name == "lift":
        every += supports  # the lift starts at the grasp pose (object resting on the support) and goes up
    if name == "place":
        every += supports  # the descent ends 1 cm above the support / partner with the object attached
        # F6 ``Place.on`` = the instance placed on / into (``grasps[].place_on``); older drafts carried the grasp record
        place_on = getattr(action, "on", None) or getattr(getattr(action, "grasp", None), "place_on", None)
        if place_on in objs:
            every.append(objs[place_on])
    if name == "push":
        obj = getattr(getattr(action, "contact", None), "object", None)
        if obj in objs:
            every.append(objs[obj])  # the pushed object moves with the pads
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
    return last, every


def _approached_for(action, sampler) -> Optional[List[Any]]:
    """The object an action's approach closes in on (removed along the whole tracked segment, owner 2026-09-16 10:15 A):
    the grasped / lifted / released object, the placed-on partner, the pushed / braced object. None = let ``IKChecker``
    infer it from the end pose."""
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
    return [objs[cand]] if cand in objs else []


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


def check_action_targets(sidecar: SidecarDraft, sampler, action, targets: Any, stance_name: str, carried: Optional[str]) -> Tuple[int, Dict[str, Any]]:
    """IK / collision check of every eef pose / waypoint list in ``targets`` from the recorded stance ``stance_name``
    (its recorded configuration: working arm untucked, idle arm in its posture, trunk of the stance) with the ``carried``
    object attached to the carrying arm and posed in the hand. Base-pose targets (paths, transitions, base-follow) are
    whole-body collision-checked at the stance trunk. Returns ``(number of poses checked, witnesses)`` where
    ``witnesses`` maps each target key to ``{"pose": {position, orientation}, "q": [28 joints]}`` of the end pose."""
    from ispatialgym.sampling.sim_utils import Reject
    from ispatialgym.sampling.v2 import navigation as NAV
    from ispatialgym.sampling.v2.navigation import attached

    name = getattr(action, "name", type(action).__name__)
    arms = list(getattr(action, "arms", ()) or ["right"])
    arm = arms[0]
    stance: Stance = sidecar.stance(stance_name)
    carried_obj = sampler.objects.get(carried) if carried else None
    g = sidecar.grasp_of(carried) if carried else None
    eef_in_object = g.eef_in_object if g is not None else None
    if carried_obj is not None and g is not None and g.arm and name in ("lift", "reorient", "place", "navigate", "transition", "set_torso"):
        arm = g.arm
    ignore_last, ignore_all = _ignore_for(action, sampler, carried)
    approached = _approached_for(action, sampler)
    # push / brace contacts are made with closed pads (§3.6 contact eef poses; the gripper closes before the pre-contact)
    fingers = 0.0 if name in ("push", "brace") else None
    # base-follow in base mode: the handle poses h(q_k) are reached by driving the base, not by the arm from the stance
    skip_keys = set()
    if name == "base_follow" and isinstance(targets, dict) and targets.get("mode") == "base":
        skip_keys.add("handle_poses")
    n = 0
    witnesses: Dict[str, Any] = {}
    attach_arm = (g.arm if g is not None and g.arm else arm) if carried_obj is not None else None
    with attached(sampler.kit, attach_arm, carried_obj, eef_in_object=eef_in_object):
        for key, poses in iter_pose_targets(targets):
            if key.split(".")[0] in skip_keys:
                continue
            if len(poses) == 1:
                # a single end pose (``grasp``, ``place``, ``contact``) is the end of its approach: the approached object is
                # removed for it as along the tracked chain (owner A) - with it in the world the open pads straddling the
                # target pass or fail at margin 0 by millimetres (teacup 18:41: 'grasp: exact IK failed at pose 1 of 1')
                res = sampler.ikc.exact(poses, stance, arm, ignore_for_last=ignore_last + ignore_all + list(approached or []), cache=False, fingers=fingers,
                                        approached=approached)
            else:
                res = sampler.ikc.waypoints_ok(poses, stance, arm, ignore_for_last=ignore_last, ignore_all=ignore_all, cache=False, fingers=fingers, approached=approached)
            n += len(poses)
            if not res.ok:
                # a straight approach without a configuration in the pre pose's IK branch is its own reason (the oracle
                # cannot track it: plan_failed); everything else is program_check_failed
                raise Reject(res.code or "program_check_failed", f"{key}: {res.reason} (stance {stance_name}, arm {arm}, carried {carried})", action=name)
            if res.q is not None:
                last = poses[-1]
                witnesses[key] = {"pose": {"position": [round(float(v), 5) for v in last[0].tolist()], "orientation": [round(float(v), 6) for v in last[1].tolist()]},
                                  "q": [round(float(v), 5) for v in res.q.tolist()], "n_waypoints": len(poses)}
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
                if back is not None:
                    checks.append((back, to_trunk))
                    if frm is not None:
                        checks.append((back, (frm.translate, frm.tilt)))
                if drive_in is not None:
                    checks.append((drive_in, to_trunk))
            elif goal is not None:
                checks.append((goal, to_trunk))
        for pose2d, trunk in checks:
            r_arm, l_arm, _ = sampler.ikc.arm_vectors(stance)
            ok, _ = NAV.path_collision_free(sampler.kit, [pose2d], trunk, q_arms={"right": r_arm, "left": l_arm})
            n += 1
            if not ok:
                raise Reject("program_check_failed", f"base pose {tuple(round(v, 3) for v in pose2d)} collides at trunk {trunk}", action=name)
    return n, witnesses
