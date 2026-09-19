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
        locked = tracker.locked_arms(action)
        n, witnesses = check_action_targets(sidecar, sampler, action, targets, stance_name, carried, locked_arms=locked, stance_obj=tracker.stance_obj)
        tracker.after(action, witnesses, sampler)
        entry = {"index": i, "name": name, "arms": list(getattr(action, "arms", ()) or []), "objects": list(getattr(action, "objects", ()) or []),
                 "stance": stance_name, "carried": carried, "n_poses": n, "witnesses": witnesses}
        if locked:
            entry["locked_arms"] = sorted(locked)  # the other arm(s) held at their contact configuration while these poses were solved
        report["actions"].append(entry)
    # the recorded joint solutions (28-D, arm joints + trunk + fingers meaningful) per checked end pose: D5 verifies them as
    # witnesses (FK + collision) instead of re-solving the randomised IK
    sidecar.ik_witnesses = [{"action": e["index"], "name": e["name"], "stance": e["stance"], "arm": (e["arms"] or ["right"])[0], "carried": e["carried"],
                            "targets": e["witnesses"]} for e in report["actions"] if e["witnesses"]]
    return report


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
        self.sidecar = sidecar
        #: arm -> its 7 arm-joint values while it braces (from the brace action's ``contact`` witness)
        self.arm_q: Dict[str, th.Tensor] = {}
        #: where the robot stands after a hybrid ``base_follow`` (the base moved off the manipulation stance): the in-memory
        #: stance the following actions (the retreat) are checked from, until the next navigate / transition
        self.stance_obj: Optional[Stance] = None

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
            for k, v in self.sidecar.stances.items():
                if list(v.base_pose2d) == list(action.stance.base_pose2d):
                    stance = k
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
        elif name == "place":
            self.carried = None
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


def check_action_targets(sidecar: SidecarDraft, sampler, action, targets: Any, stance_name: str, carried: Optional[str],
                         locked_arms: Optional[Dict[str, th.Tensor]] = None, stance_obj: Optional[Stance] = None) -> Tuple[int, Dict[str, Any]]:
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
    the base (``_Tracker.stance_obj``), so the retreat after it is checked from there."""
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
    if carried_obj is not None and g is not None and g.arm and name in ("lift", "reorient", "place", "navigate", "transition", "set_torso"):
        arm = g.arm
    ignore_last, ignore_all = _ignore_for(action, sampler, carried)
    approached = _approached_for(action, sampler)
    # push / brace contacts are made with closed pads (§3.6 contact eef poses; the gripper closes before the pre-contact)
    fingers = 0.0 if name in ("push", "brace") else None
    # base-follow in base mode: the handle poses h(q_k) are reached by driving the base, not by the arm from the stance;
    # in arm mode they are checked one by one with the articulated object at q_k (``check_base_follow_handles``), not as a
    # chain against the object frozen at q_init (closing: the hand moves into where the open front was)
    skip_keys = set()
    if name == "base_follow" and isinstance(targets, dict):
        skip_keys.add("handle_poses")
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
    if locked:
        q_seed = sampler.ikc.q_locked(stance, fingers=fingers).clone()
        for a, qa in locked.items():
            q_seed[sampler.kit.robot.arm_control_idx[a]] = th.as_tensor(qa, dtype=th.float32)
    where = f"stance {stance_name}, arm {arm}, carried {carried}" + (f", {'/'.join(sorted(locked))} arm locked at its contact" if locked else "")
    with attached(sampler.kit, attach_arm, carried_obj, eef_in_object=eef_in_object):
        for key, poses in iter_pose_targets(targets):
            if key.split(".")[0] in skip_keys:
                continue
            if len(poses) == 1:
                # a single end pose (``grasp``, ``place``, ``contact``) is the end of its approach: the approached object is
                # removed for it as along the tracked chain (owner A) - with it in the world the open pads straddling the
                # target pass or fail at margin 0 by millimetres (teacup 18:41: 'grasp: exact IK failed at pose 1 of 1')
                res = sampler.ikc.exact(poses, stance, arm, ignore_for_last=ignore_last + ignore_all + list(approached or []), cache=False, fingers=fingers,
                                        approached=approached, q_seed=q_seed)
            else:
                res = sampler.ikc.waypoints_ok(poses, stance, arm, ignore_for_last=ignore_last, ignore_all=ignore_all, cache=False, fingers=fingers, approached=approached,
                                               q_seed=q_seed)
            n += len(poses)
            if not res.ok:
                # a straight approach without a configuration in the pre pose's IK branch is its own reason (the oracle
                # cannot track it: plan_failed); everything else is program_check_failed
                raise Reject(res.code or "program_check_failed", f"{key}: {res.reason} ({where})", action=name)
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
                                                         f"after polishing (stance {stance_name}, arm {arm})", action=name)
                margin, joint = sampler.ikc.joint_limit_margin(q_pol, arm)
                witnesses[key] = {"pose": {"position": [round(float(v), 5) for v in last[0].tolist()], "orientation": [round(float(v), 6) for v in last[1].tolist()]},
                                  "q": [round(float(v), 5) for v in sampler.ikc.canonical_q(q_pol, stance).tolist()],
                                  "q_frame": "root at the world origin (identity pose, z = 0): FK(q) puts base_link at base_pose2d on the measured floor",
                                  "base_pose2d": [float(v) for v in stance.base_pose2d], "floor_z": round(float(sampler.kit.floor_z), 4),
                                  "fk_residual_m": round(residual, 5), "joint_limit_margin_rad": round(margin, 4), "joint_limit_nearest": joint,
                                  "n_waypoints": len(poses)}
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
            q_arms = {"right": r_arm, "left": l_arm}
            drive_ctx = attached(sampler.kit, None, None)
            if carried_obj is not None and g is not None and attach_arm and name in ("navigate", "transition"):
                # a carried drive (touching / attachment second base pose, owner 2026-09-18 16:32): the object rides in the hand
                # at the LIFT configuration of the departing stance (as the oracle drives it) - not at its init pose beside the
                # untucked arm of the arriving stance (0918-171641: 'base pose ... collides' on 4/12 attempts)
                frm_name = getattr(getattr(action, "path", None), "frm", None) or getattr(getattr(action, "transition", None), "frm", None)
                frm_st = sidecar.stances.get(frm_name) if frm_name else None
                if frm_st is not None and getattr(g, "grasp", None) is not None:
                    lift_end = (th.as_tensor(g.grasp[0], dtype=th.float32) + th.tensor([0.0, 0.0, float(g.carry_height or 0.0)]), th.as_tensor(g.grasp[1], dtype=th.float32))
                    supports = list(sampler.support_objects()) if hasattr(sampler, "support_objects") else []
                    with attached(sampler.kit, attach_arm, carried_obj, eef_in_object=eef_in_object):
                        r_lift = sampler.ikc.exact([lift_end], frm_st, attach_arm, ignore_for_last=supports, cache=False)
                    if r_lift.ok and r_lift.q is not None:
                        q_arms[attach_arm] = r_lift.q[sampler.kit.robot.arm_control_idx[attach_arm]].clone()
                drive_ctx = attached(sampler.kit, attach_arm, carried_obj, eef_in_object=eef_in_object)
            with drive_ctx:
                ok, _ = NAV.path_collision_free(sampler.kit, [pose2d], trunk, q_arms=q_arms)
            n += 1
            if not ok:
                raise Reject("program_check_failed", f"base pose {tuple(round(v, 3) for v in pose2d)} collides at trunk {trunk}"
                                                     f"{' (carried ' + str(carried) + ' in the lifted hand)' if carried_obj is not None and name in ('navigate', 'transition') else ''}",
                             action=name)
    return n, witnesses
