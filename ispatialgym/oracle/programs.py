"""Layer 3 of the v2 oracle: the per-subtask programs and ``expand`` (``oracle_primitives_design.md`` §3.7, §6.2).

``expand(alias, sidecar) -> list[Action]`` dispatches on the alias's subtask to one ``program_<row>(sidecar)`` per
row of §6.2 and returns the action list with the row's loops and conditionals resolved from the sidecar facts
(block order, ``brace_row`` / ``brace_col`` present or not, ``passive_from``, ``follow_mode``, the placement stance,
``goal.arms``, whether a path and / or a transition is recorded between two stances). The sidecar carries facts, not
the sequence (spec §3.11, §7); this module is the one readable place the sequence lives, next to the table it must
match.

Pure: no simulator, no torch at module level. Callers: the executor (``run_isg``), the sampler's program check
(spec §3.5) and ``validate_instances.py`` (TODO D5). A fact a program needs but the sidecar lacks raises
:class:`~ispatialgym.oracle.sidecar.SidecarError` (spec principle 2: nothing is substituted).

Stance travel rule (used by every row that says "init → manipulation stance (transition)", "drive ... to the
placement stance (recorded path)", "navigate_to placement C (recorded path, retreat transition ...)"): between two
consecutive stances the program emits ``Navigate`` when ``paths[]`` records the pair and ``Transition`` when
``transitions[]`` records it, in that order (drive the validated path, then drive back to s_T, move the trunk, drive
in — spec §3.7 / §3.8). Two stances that differ in trunk without a recorded transition, or in base pose without a
path or transition, are an error.

The 14 rows of §6.2 map to 13 functions: camera relocalization standard / obstructed share
``program_camera_relocalization`` (flag ``obstructed``), articulation open / close share the large- and the
small-scale programs, the two wall alignments share ``program_alignment_wall``, IORS ontop / under (large) share
``program_iors_large``, IORS ontop / under / inside / nextto (small) share ``program_iors_small``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from ispatialgym.oracle.actions import (
    RETREAT_LITTLE,
    Action,
    BaseFollow,
    Brace,
    Grasp,
    Lift,
    Navigate,
    Place,
    Push,
    Reorient,
    Retreat,
    SetTorso,
    Transition,
)
from ispatialgym.oracle.sidecar import ARMS, GraspRecord, Sidecar, SidecarError, subtask_of_alias

# ----------------------------------------------------------------------------------------------------------------
# Shared pieces
# ----------------------------------------------------------------------------------------------------------------


def _other_arm(arm: str) -> str:
    return "left" if arm == "right" else "right"


def _travel(sc: Sidecar, frm: str, to: str, carried: Optional[str] = None) -> List[Action]:
    """``Navigate`` (recorded path) and / or ``Transition`` (recorded transition) from stance ``frm`` to ``to``."""
    s_from, s_to = sc.stance(frm), sc.stance(to)
    path, transition = sc.path(frm, to), sc.transition(frm, to)
    actions: List[Action] = []
    if path is not None:
        if (path.carried or None) != carried:
            raise SidecarError(f"paths {frm!r} -> {to!r}: recorded carried={path.carried!r}, the program carries {carried!r}")
        actions.append(Navigate(path=path, objects=(carried,) if carried else ()))
    if transition is not None:
        actions.append(Transition(transition=transition, stance_from=s_from, stance_to=s_to, carried=carried, objects=(carried,) if carried else ()))
    trunk_differs = abs(s_from.translate - s_to.translate) > 1e-6 or abs(s_from.tilt - s_to.tilt) > 1e-6
    base_differs = any(abs(a - b) > 1e-6 for a, b in zip(s_from.base_pose2d, s_to.base_pose2d))
    if trunk_differs and transition is None:
        raise SidecarError(f"stances {frm!r} and {to!r} differ in trunk but no transition is recorded (spec §3.8)")
    if base_differs and not actions:
        raise SidecarError(f"stances {frm!r} and {to!r} differ in base pose but neither a path nor a transition is recorded")
    return actions


def _single_grasp(sc: Sidecar, *needs: str) -> GraspRecord:
    if len(sc.grasps) != 1:
        raise SidecarError(f"grasps: expected exactly one record, got {len(sc.grasps)}")
    g = sc.grasps[0]
    for attr in needs:
        if getattr(g, attr) is None:
            raise SidecarError(f"grasps[0]: '{attr}' is required for this subtask")
    return g


def _working_arm(sc: Sidecar, stance: str = "manipulation") -> str:
    arms = sc.stance(stance).working_arms
    if len(arms) != 1:
        raise SidecarError(f"stances[{stance!r}].arms: expected exactly one working arm, got {arms!r}")
    return arms[0]


def _push_contact(sc: Sidecar):
    if sc.contacts is None or sc.contacts.push is None:
        raise SidecarError("contacts.push: required for this subtask")
    return sc.contacts.push


def _brace_contact(sc: Sidecar):
    if sc.contacts is None or sc.contacts.brace is None:
        raise SidecarError("contacts.brace: required for this subtask")
    return sc.contacts.brace


def _pick(g: GraspRecord) -> Tuple[Grasp, Lift]:
    grasp = Grasp(kind="library", arm=g.arm, object=g.object, grasp=g)
    return grasp, Lift(arm=g.arm, object=g.object, carry_height=g.carry_height, after=grasp)


def _place_and_retreat(g: GraspRecord, kind: str, relation: str = "on") -> List[Action]:
    place = Place(kind=kind, arm=g.arm, object=g.object, place=g.place, pre_place=g.pre_place, on=g.place_on, relation=relation)
    return [place, Retreat(arm=g.arm, object=g.object, after=place, distance=RETREAT_LITTLE)]


# annotation relation word of the IORS place per relation (the partner ends under the target for under_small)
_IORS_RELATION_WORDS = {"ontop": "on", "under": "under", "under_small": "on", "inside": "into", "nextto": "next to", "touching": "next to"}


def _brace_push_retreat(sc: Sidecar, brace, push, base_yaw: float) -> List[Action]:
    """Brace first (held), push with the other arm, retreat the push arm, release and retreat the brace arm."""
    if brace.arm == push.arm:
        raise SidecarError(f"brace and push both use the {brace.arm} arm")
    b = Brace(contact=brace, base_yaw=base_yaw)
    p = Push(contact=push, base_yaw=base_yaw)
    return [
        b,
        p,
        Retreat(arm=push.arm, object=push.object, after=p),
        Retreat(arm=brace.arm, object=brace.object, after=b, release=True),
    ]


# ----------------------------------------------------------------------------------------------------------------
# Programs (one per row of §6.2)
# ----------------------------------------------------------------------------------------------------------------


def program_camera_relocalization(sc: Sidecar, obstructed: bool = False) -> List[Action]:
    """navigate_to the goal base (recorded path, bucket length) → set_torso to the goal (translate, tilt), lifting the
    arms to untucked first where ``goal.arms`` says so. Obstructed: the same on the map with the obstacle blocked; the
    obstacles (every object the sidecar names) go into the annotation, not into the contact whitelist."""
    if sc.goal is None:
        raise SidecarError("goal: required for camera relocalization")
    path = sc.path("init", "goal")
    if path is None:
        raise SidecarError("paths: no recorded path from 'init' to 'goal'")
    if path.carried:
        raise SidecarError("paths init -> goal: camera relocalization carries nothing")
    obstacles = tuple(sorted(sc.language.objects)) if obstructed else ()
    if obstructed and not obstacles:
        raise SidecarError("language.objects: the obstructed variant must name its obstacle(s)")
    return [Navigate(path=path, obstacles=obstacles), SetTorso(goal=sc.goal)]


def program_held_object_scaling(sc: Sidecar) -> List[Action]:
    """move the held object along the camera centre ray to the recorded depth (arm only; 3 recorded waypoints), object
    orientation unchanged."""
    g = _single_grasp(sc, "waypoints")
    return [Reorient(mode="line", arm=g.arm, object=g.object, start_pose=g.grasp, waypoints=g.waypoints)]


def program_held_object_rotation(sc: Sidecar) -> List[Action]:
    """rotate the held object to the goal orientation about its own centre (arm only; SLERP waypoints), position
    unchanged."""
    g = _single_grasp(sc, "waypoints")
    return [Reorient(mode="orbit", arm=g.arm, object=g.object, start_pose=g.grasp, waypoints=g.waypoints)]


def program_disembedding(sc: Sidecar) -> List[Action]:
    """init → manipulation stance (transition) → grasp at the recorded pose → lift to the carry height (orientation
    kept) → drive sideways along the edge to the placement stance (recorded path) → place at the destination centre
    (orientation kept) → retreat."""
    g = _single_grasp(sc, "pre_grasp", "carry_height", "place", "place_on")
    grasp, lift = _pick(g)
    return (
        _travel(sc, "init", "manipulation")
        + [grasp, lift]
        + _travel(sc, "manipulation", "placement", carried=g.object)
        + _place_and_retreat(g, "support")
    )


def program_block_design(sc: Sidecar) -> List[Action]:
    """init → manipulation stance → for each block in the recorded order (anchor first): grasp (recorded cube grasp)
    → lift → reorient_in_hand (smallest yaw to the table axes, then one 90° flip about the pinch axis; skipped for a
    block that needs neither) → place at its slot → push it left against the placed row with the working hand while
    the closed other gripper braces the row-left blocks (where ``brace_row`` / ``push_row`` are recorded) → push it up
    while the other gripper braces the column-top blocks (where recorded) → retreat both grippers."""
    if not sc.blocks:
        raise SidecarError("blocks: required for block design")
    arm = _working_arm(sc)
    base_yaw = sc.stance("manipulation").base_pose2d[2]
    actions = _travel(sc, "init", "manipulation")
    for b in sorted(sc.blocks, key=lambda blk: blk.order):
        grasp = Grasp(kind="cube", arm=arm, object=b.instance, block=b)
        lift = Lift(arm=arm, object=b.instance, carry_height=b.carry_height, after=grasp)
        actions += [grasp, lift]
        last: Action = lift
        if b.reorients:
            last = Reorient(mode="flip", arm=arm, object=b.instance, after=lift, block=b, waypoints=b.waypoints)
            actions.append(last)
        slot = "at its initial position (the anchor)" if b.order == 0 else "at its slot in the goal grid"
        place = Place(kind="support", arm=arm, object=b.instance, place=b.place, relation=slot)
        actions += [place, Retreat(arm=arm, object=b.instance, after=place, distance=RETREAT_LITTLE)]
        for kind in ("row", "col"):
            brace, push = getattr(b, f"brace_{kind}"), getattr(b, f"push_{kind}")
            if brace is None and push is None:
                continue
            if brace is None or push is None:
                raise SidecarError(f"blocks[{b.order}]: brace_{kind} and push_{kind} must be recorded together")
            if push.arm != arm:
                raise SidecarError(f"blocks[{b.order}].push_{kind}: the working arm is {arm!r}, the push is recorded for {push.arm!r}")
            actions += _brace_push_retreat(sc, brace, push, base_yaw)
    return actions


def program_articulation_large(sc: Sidecar, direction: str = "move") -> List[Action]:
    """init → manipulation stance (transition) → grasp the handle at h(q_init) → move the joint: base_follow along the
    recorded b_k (``follow_mode`` base) or the arm along h(q_k) (``follow_mode`` arm), up to ``passive_from`` or q_goal
    → release → retreat."""
    art = sc.articulation
    if art is None:
        raise SidecarError("articulation: required for articulation tasks")
    arm = _working_arm(sc)
    grasp = Grasp(kind="handle", arm=arm, object=art.object, articulation=art)
    follow = BaseFollow(arm=arm, articulation=art, stance=sc.stance("manipulation"), grasp_action=grasp, direction=direction)
    return _travel(sc, "init", "manipulation") + [grasp, follow, Retreat(arm=arm, object=art.object, after=follow, release=True)]


def program_articulation_small(sc: Sidecar, direction: str = "move") -> List[Action]:
    """init → manipulation stance → brace the base link (recorded kind) with the brace arm → grasp the handle with the
    handle arm → move the arm along h(q_k) to the goal (or ``passive_from``) → release → retreat both."""
    art = sc.articulation
    if art is None:
        raise SidecarError("articulation: required for articulation tasks")
    if art.follow_mode != "arm":
        raise SidecarError(f"articulation.follow_mode: small-scale articulation moves the arm, got {art.follow_mode!r}")
    brace = _brace_contact(sc)
    arm = _working_arm(sc)
    if brace.arm != _other_arm(arm):
        raise SidecarError(f"contacts.brace.arm: the handle arm is {arm!r}, the brace must use {_other_arm(arm)!r}")
    if brace.object != art.object:
        raise SidecarError("contacts.brace.object: the brace must act on the articulated object's base link")
    base_yaw = sc.stance("manipulation").base_pose2d[2]
    b = Brace(contact=brace, base_yaw=base_yaw)
    grasp = Grasp(kind="handle", arm=arm, object=art.object, articulation=art)
    follow = BaseFollow(arm=arm, articulation=art, stance=sc.stance("manipulation"), grasp_action=grasp, direction=direction)
    return _travel(sc, "init", "manipulation") + [
        b,
        grasp,
        follow,
        Retreat(arm=arm, object=art.object, after=follow, release=True),
        Retreat(arm=brace.arm, object=brace.object, after=b, release=True),
    ]


def program_attachment(sc: Sidecar) -> List[Action]:
    """init → manipulation stance → grasp the target → lift → reorient_in_hand toward the attach orientation while
    moving to the pre-attach pose (recorded SLERP waypoints, end = pre-attach 10 cm back along the female insertion
    axis) → brace the partner on its face opposite to the insertion with the other gripper → insert along the axis
    until attached → release → retreat both."""
    g = _single_grasp(sc, "pre_grasp", "carry_height", "waypoints", "pre_place", "place", "place_on")
    brace = _brace_contact(sc)
    if brace.arm != _other_arm(g.arm):
        raise SidecarError(f"contacts.brace.arm: the target is grasped with the {g.arm} arm, the brace must use the other")
    if brace.object != g.place_on:
        raise SidecarError("contacts.brace.object: the brace must hold the attachment partner (grasps[0].place_on)")
    base_yaw = sc.stance("manipulation").base_pose2d[2]
    grasp, lift = _pick(g)
    reorient = Reorient(mode="slerp", arm=g.arm, object=g.object, after=lift, waypoints=g.waypoints)
    b = Brace(contact=brace, base_yaw=base_yaw)
    place = Place(kind="attach", arm=g.arm, object=g.object, place=g.place, pre_place=g.pre_place, on=g.place_on)
    return _travel(sc, "init", "manipulation") + [
        grasp,
        lift,
        reorient,
        b,
        place,
        Retreat(arm=g.arm, object=g.object, after=place, distance=RETREAT_LITTLE),
        Retreat(arm=brace.arm, object=brace.object, after=b, release=True),
    ]


def program_alignment_wall(sc: Sidecar, axis: str = "vertical") -> List[Action]:
    """init → manipulation stance → close both grippers → brace at the recorded stop (below_high / above_low /
    side_high) → push at E_low along the arc (δ/2, goal) with the arm only until the upright (vertical) / level
    (horizontal) orientation → retreat both."""
    push, brace = _push_contact(sc), _brace_contact(sc)
    if push.mid is None:
        raise SidecarError("contacts.push.mid: the wall alignment push is an arc (δ/2 waypoint required)")
    if push.object != brace.object:
        raise SidecarError("contacts: push and brace must act on the same target")
    base_yaw = sc.stance("manipulation").base_pose2d[2]
    return _travel(sc, "init", "manipulation") + _brace_push_retreat(sc, brace, push, base_yaw)


def program_alignment_axial(sc: Sidecar) -> List[Action]:
    """init → manipulation stance → close both grippers → left gripper brace at the recorded stop edge → right gripper
    push at the long-axis end along the arc to the aligned yaw → retreat both."""
    push, brace = _push_contact(sc), _brace_contact(sc)
    if push.arm != "right" or brace.arm != "left":
        raise SidecarError("alignment-axial: the right gripper pushes and the left gripper braces (spec §6.10)")
    if push.mid is None:
        raise SidecarError("contacts.push.mid: the axial push is an arc (δ/2 waypoint required)")
    if push.object != brace.object:
        raise SidecarError("contacts: push and brace must act on the same target")
    base_yaw = sc.stance("manipulation").base_pose2d[2]
    return _travel(sc, "init", "manipulation") + _brace_push_retreat(sc, brace, push, base_yaw)


def program_iors_large(sc: Sidecar, relation: str = "ontop") -> List[Action]:
    """init A → manipulation B (transition) → grasp → lift (orientation kept) → navigate_to placement C with the object
    attached (recorded path; the retreat transition before lowering the trunk toward the floor band for under, where
    recorded) → place at the goal cell → retreat."""
    g = _single_grasp(sc, "pre_grasp", "carry_height", "place", "place_on")
    if sc.path("manipulation", "placement") is None:
        raise SidecarError("paths: IORS large-scale needs the recorded path from 'manipulation' to 'placement'")
    grasp, lift = _pick(g)
    return (
        _travel(sc, "init", "manipulation")
        + [grasp, lift]
        + _travel(sc, "manipulation", "placement", carried=g.object)
        + _place_and_retreat(g, "support", _IORS_RELATION_WORDS[relation])
    )


def program_iors_small(sc: Sidecar, relation: str = "ontop") -> List[Action]:
    """init → manipulation stance → grasp → lift (orientation kept) → place at the recorded goal pose (centre of the
    partner; nearest satisfying position for nextto; inside: drop into the fillable volume) → retreat."""
    g = _single_grasp(sc, "pre_grasp", "carry_height", "place", "place_on")
    grasp, lift = _pick(g)
    kind = "drop" if relation == "inside" else "support"
    word = _IORS_RELATION_WORDS["under_small" if relation == "under" else relation]
    return _travel(sc, "init", "manipulation") + [grasp, lift] + _place_and_retreat(g, kind, word)


def program_iors_touching(sc: Sidecar) -> List[Action]:
    """as nextto to the recorded nextto position, then brace the partner's far side with the left gripper and push the
    target toward the partner with the right gripper until touching → retreat both."""
    g = _single_grasp(sc, "pre_grasp", "carry_height", "place", "place_on")
    push, brace = _push_contact(sc), _brace_contact(sc)
    if push.arm != "right" or brace.arm != "left":
        raise SidecarError("touching: the right gripper pushes the target and the left gripper braces the partner (spec §6.11)")
    if push.object != g.object:
        raise SidecarError("contacts.push.object: the push must act on the placed target")
    if brace.object != g.place_on:
        raise SidecarError("contacts.brace.object: the brace must hold the partner (grasps[0].place_on)")
    base_yaw = sc.stance("manipulation").base_pose2d[2]
    return program_iors_small(sc, "nextto") + _brace_push_retreat(sc, brace, push, base_yaw)


# ----------------------------------------------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------------------------------------------

PROGRAMS: Dict[str, Tuple[Callable[..., List[Action]], Dict[str, object]]] = {
    "camera_relocalization-standard": (program_camera_relocalization, {"obstructed": False}),
    "camera_relocalization-obstructed": (program_camera_relocalization, {"obstructed": True}),
    "object_scaling": (program_held_object_scaling, {}),
    "mental_rotation": (program_held_object_rotation, {}),
    "disembedding": (program_disembedding, {}),
    "configuration_matching-block_design": (program_block_design, {}),
    "configuration_matching-articulation_open_large_scale": (program_articulation_large, {"direction": "open"}),
    "configuration_matching-articulation_close_large_scale": (program_articulation_large, {"direction": "close"}),
    "configuration_matching-articulation_open_small_scale": (program_articulation_small, {"direction": "open"}),
    "configuration_matching-articulation_close_small_scale": (program_articulation_small, {"direction": "close"}),
    "configuration_matching-attachment": (program_attachment, {}),
    "alignment-world_vertical": (program_alignment_wall, {"axis": "vertical"}),
    "alignment-world_horizontal": (program_alignment_wall, {"axis": "horizontal"}),
    "alignment-axial": (program_alignment_axial, {}),
    "inter_object_relation_satisfaction-ontop_large_scale": (program_iors_large, {"relation": "ontop"}),
    "inter_object_relation_satisfaction-under_large_scale": (program_iors_large, {"relation": "under"}),
    "inter_object_relation_satisfaction-ontop_small_scale": (program_iors_small, {"relation": "ontop"}),
    "inter_object_relation_satisfaction-under_small_scale": (program_iors_small, {"relation": "under"}),
    "inter_object_relation_satisfaction-inside": (program_iors_small, {"relation": "inside"}),
    "inter_object_relation_satisfaction-nextto": (program_iors_small, {"relation": "nextto"}),
    "inter_object_relation_satisfaction-touching": (program_iors_touching, {}),
}


def program_for(subtask: str) -> Tuple[Callable[..., List[Action]], Dict[str, object]]:
    try:
        return PROGRAMS[subtask]
    except KeyError:
        raise SidecarError(f"no oracle program for subtask {subtask!r}") from None


def expand(alias: str, sidecar: Sidecar) -> List[Action]:
    """The action list of ``alias``'s program for ``sidecar`` (§3.7). ``alias`` must be the sidecar's own alias (or
    resolve to the same subtask); the executor's epilogue (§3.8) is not an action."""
    subtask = subtask_of_alias(alias)
    if alias != sidecar.alias and subtask_of_alias(sidecar.alias) != subtask:
        raise SidecarError(f"alias {alias!r} is not the sidecar's alias {sidecar.alias!r}")
    fn, kwargs = program_for(subtask)
    actions = fn(sidecar, **kwargs)
    for action in actions:
        for arm in action.arms:
            if arm not in ARMS:
                raise SidecarError(f"{action.name}: unknown arm {arm!r}")
    return actions


__all__ = ["PROGRAMS", "expand", "program_for"] + sorted(
    name for name in globals() if name.startswith("program_") and name != "program_for"
)
