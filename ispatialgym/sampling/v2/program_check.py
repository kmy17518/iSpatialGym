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

    try:
        for i, action in enumerate(actions):
            name = getattr(action, "name", type(action).__name__)
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
            n, witnesses = check_action_targets(sidecar, sampler, action, targets, stance_name, carried, locked_arms=locked, stance_obj=tracker.stance_obj, q_seed_prev=q_prev)
            end_q = witnesses.pop("__arm_end_q__", None)
            tracker.after(action, witnesses, sampler)
            tracker.note_arm_end(action, end_q)
            _carry_joint_state(action)
            entry = {"index": i, "name": name, "arms": list(getattr(action, "arms", ()) or []), "objects": list(getattr(action, "objects", ()) or []),
                     "stance": stance_name, "carried": carried, "n_poses": n, "witnesses": witnesses}
            if locked:
                entry["locked_arms"] = sorted(locked)  # the other arm(s) held at their contact configuration while these poses were solved
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
            self.arm_end_q.clear()
            return
        arms = list(getattr(action, "arms", ()) or [])
        if end_q is not None and arms:
            self.arm_end_q[arms[0]] = end_q.clone()

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
        elif name == "place":
            self.carried = None
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


def check_action_targets(sidecar: SidecarDraft, sampler, action, targets: Any, stance_name: str, carried: Optional[str],
                         locked_arms: Optional[Dict[str, th.Tensor]] = None, stance_obj: Optional[Stance] = None,
                         q_seed_prev: Optional[th.Tensor] = None) -> Tuple[int, Dict[str, Any]]:
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
    elif q_seed_prev is not None and (name in _END_KEY or name == "push"):
        # the acting arm continues from the previous action's end (block design chaining) / a press's recorded branch (q_hint);
        # trunk / other arm from the stance
        q_seed = sampler.ikc.q_locked(stance, fingers=fingers).clone()
        idx = sampler.kit.robot.arm_control_idx[arm]
        q_seed[idx] = th.as_tensor(q_seed_prev, dtype=th.float32)[idx]
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
    with attached(sampler.kit, attach_arm, carried_obj, eef_in_object=eef_in_object), sampler.ikc.ignoring([], ground=not slab_off):
        for key, poses in pose_targets:
            if key.split(".")[0] in skip_keys:
                continue
            if len(poses) == 1:
                # a single end pose (``grasp``, ``place``, ``contact``) is the end of its approach: the approached object is
                # removed for it as along the tracked chain (owner A) - with it in the world the open pads straddling the
                # target pass or fail at margin 0 by millimetres (teacup 18:41: 'grasp: exact IK failed at pose 1 of 1')
                res = sampler.ikc.exact(poses, stance, arm, ignore_for_last=ignore_last + ignore_all + list(approached or []), cache=False, fingers=fingers,
                                        approached=approached, q_seed=q_seed)
            else:
                # block design (owner 2026-09-19): the reorient chains (yaw-align orbit, in-hand flip) are tracked at the oracle's
                # densification as the sampler tracked them (``IKChecker.track_chain``; demo 0919-140558 'joint jump 0.23 rad')
                track = name == "reorient" and bool(getattr(sidecar, "blocks", None))
                res = sampler.ikc.waypoints_ok(poses, stance, arm, ignore_for_last=ignore_last, ignore_all=ignore_all, cache=False, fingers=fingers, approached=approached,
                                               q_seed=q_seed, track_dense=track)
                if not res.ok and name == "push" and q_seed_prev is not None and not locked:
                    # a push's pre-contact pose is reached by the oracle's planned ``arm_to`` from wherever the arm is, so the
                    # branch the previous retreat ended in need not hold it: re-solve the chain from the stance posture (the
                    # sampler's seed) before failing (0920-080331: a pack push the sampler's identical chain check had passed)
                    res = sampler.ikc.waypoints_ok(poses, stance, arm, ignore_for_last=ignore_last, ignore_all=ignore_all, cache=False, fingers=fingers,
                                                   approached=approached, q_seed=sampler.ikc.q_locked(stance, fingers=fingers).clone(), track_dense=track)
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
        for pose2d, trunk in checks:
            r_arm, l_arm, fingers_st = sampler.ikc.arm_vectors(stance)
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
    return n, witnesses
