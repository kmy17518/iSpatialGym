"""Layer 3 of the v2 oracle: the per-subtask programs and ``expand`` (``oracle_primitives_design.md`` §3.7, §6.2).

``expand(variant, sidecar) -> list[Action]`` dispatches on the variant's subtask to one ``program_<row>(sidecar)`` per
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

The rows of §6.2 map to 13 functions: camera relocalization standard / obstructed share
``program_camera_relocalization`` (flag ``obstructed``), articulation open / close share the large- and the
small-scale programs, world vertical is ``program_alignment_wall``, IORS ontop / under (large) share
``program_iors_large``, IORS ontop / under / inside / nextto (small) share ``program_iors_small``.
"""

from __future__ import annotations

import math
import os
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ispatialgym.oracle.actions import (
    BASE_PUSH_KIND,
    RETREAT_DIST,
    RETREAT_LITTLE,
    Action,
    BaseFollow,
    BasePush,
    Brace,
    Grasp,
    Lift,
    Navigate,
    Place,
    Push,
    PushClose,
    Reorient,
    Retreat,
    SetTorso,
    Transition,
    block_reorients,
)
from ispatialgym.oracle.sidecar import ARMS, BraceContact, GraspRecord, Pose, PushContact, Sidecar, SidecarError, subtask_of_variant
from ispatialgym.robot import base_path as BP

# ----------------------------------------------------------------------------------------------------------------
# Shared pieces
# ----------------------------------------------------------------------------------------------------------------


def _other_arm(arm: str) -> str:
    return "left" if arm == "right" else "right"


def _travel(sc: Sidecar, frm: str, to: str, carried: Optional[str] = None, axis_leg: bool = False) -> List[Action]:
    """``Navigate`` (recorded path) and / or ``Transition`` (recorded transition) from stance ``frm`` to ``to``.

    ``carried`` = the bddl instance in the hand during the travel: a recorded path must carry exactly it
    (``paths[].carried``) and a recorded transition that names what it was swept with (``transitions[].holding``,
    core ``bd10c72``) must name it too — the sampler validated the drive-back / trunk / drive-in with that object
    attached to the frozen arm, the oracle drives it with the arms as they are (no re-planning of the stance).
    ``axis_leg`` (owner 2026-10-13): the recorded path must keep the heading and move only forward / backward or left /
    right (``robot.base_path.axis_leg_violation``, else ``SidecarError``) and is driven as such (``Navigate.axis_leg``)."""
    s_from, s_to = sc.stance(frm), sc.stance(to)
    path, transition, departure = sc.path(frm, to), sc.transition(frm, to), sc.departure(frm, to)
    actions: List[Action] = []
    if departure is not None:
        # owner 2026-09-19 21:56 (IORS large-scale carry): straighten the torso FIRST - drive back along the departing stance's
        # own axis, sweep to the travel trunk, then navigate at that trunk; the arrival transition (below) sets the arriving
        # stance's trunk at its pre-place stand-off
        if departure.holding is not None and departure.holding != carried:
            raise SidecarError(f"transitions {frm!r} -> {to!r} (departure): recorded holding={departure.holding!r}, the program carries {carried!r}")
        actions.append(Transition(transition=departure, stance_from=s_from, stance_to=s_to, carried=carried, objects=(carried,) if carried else ()))
    if path is not None:
        if (path.carried or None) != carried:
            raise SidecarError(f"paths {frm!r} -> {to!r}: recorded carried={path.carried!r}, the program carries {carried!r}")
        if axis_leg:
            why = BP.axis_leg_violation(path.waypoints)
            if why is not None:
                raise SidecarError(f"paths {frm!r} -> {to!r}: {why} - the leg into the place pose keeps the robot's heading and moves only "
                                   "forward / backward or left / right (owner 2026-10-13)")
        actions.append(Navigate(path=path, objects=(carried,) if carried else (), axis_leg=axis_leg))
    if transition is not None:
        if transition.holding is not None and transition.holding != carried:
            raise SidecarError(f"transitions {frm!r} -> {to!r}: recorded holding={transition.holding!r}, the program carries {carried!r}")
        actions.append(Transition(transition=transition, stance_from=s_from, stance_to=s_to, carried=carried, objects=(carried,) if carried else ()))
    trunk_differs = abs(s_from.translate - s_to.translate) > 1e-6 or abs(s_from.tilt - s_to.tilt) > 1e-6
    base_differs = any(abs(a - b) > 1e-6 for a, b in zip(s_from.base_pose2d, s_to.base_pose2d))
    if departure is not None and transition is None:
        # the departure sweep must land on the arriving stance's trunk when no arrival transition follows
        tt = departure.trunk_to
        if tt is None or abs(tt[0] - s_to.translate) > 1e-6 or abs(tt[1] - s_to.tilt) > 1e-6:
            raise SidecarError(f"stances {frm!r} -> {to!r}: the departure transition ends at trunk {tt}, the arriving stance has "
                               f"{(s_to.translate, s_to.tilt)} and no arrival transition is recorded")
    elif trunk_differs and transition is None:
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


def _block_working_arm(sc: Sidecar, stance: str) -> str:
    """The arm that picks / places the blocks at ``stance`` (block design): the stance's single working arm, or - at a stance
    whose arms list the brace arm too (``stances[].arms == ["right", "left"]`` where a block carries a brace) - the arm
    that is not any recorded brace's arm (the sampler lists the working arm first; pod run 0919-144520: the first
    two-arm stance broke the single-arm rule)."""
    arms = sc.stance(stance).working_arms
    if len(arms) == 1:
        return arms[0]
    if not arms:
        raise SidecarError(f"stances[{stance!r}].arms: no working arm recorded")
    brace_arms = {getattr(getattr(b, k, None), "arm", None) for b in sc.blocks for k in ("brace_row", "brace_col")} - {None}
    rest = [a for a in arms if a not in brace_arms]
    if len(rest) == 1:
        return rest[0]
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
    grasp = Grasp(kind="library", arm=g.arm, object=g.object, grasp=g, touches=g.may_touch[0])
    return grasp, Lift(arm=g.arm, object=g.object, carry_height=g.carry_height, after=grasp)


#: ``contacts.push_pick``: the push before an overhang pick (owner 2026-10-13 ~03:15 item 4; sampler ``iors_large._overhang_pick``)
PICK_PUSH_SLOT = "push_pick"
PICK_PUSH_KIND = "pick_overhang"


def _pick_push(sc: Sidecar, g: GraspRecord) -> Optional[PushContact]:
    """``contacts.push_pick`` (kind ``pick_overhang``: the object pushed along its support until it overhangs the edge, then grasped at
    the overhang), checked against the grasp it prepares - the same object, the same hand; None when the sidecar records none."""
    rec = (sc.contacts.extra or {}).get(PICK_PUSH_SLOT) if sc.contacts is not None else None
    if rec is None:
        return None
    push = PushContact.from_dict(rec, f"contacts.{PICK_PUSH_SLOT}")
    if push.kind != PICK_PUSH_KIND:
        raise SidecarError(f"contacts.{PICK_PUSH_SLOT}.kind {push.kind!r}: expected {PICK_PUSH_KIND!r}")
    if push.object != g.object:
        raise SidecarError(f"contacts.{PICK_PUSH_SLOT}.object {push.object!r} is not the grasped object {g.object!r}")
    if push.arm != g.arm:
        raise SidecarError(f"contacts.{PICK_PUSH_SLOT}.arm {push.arm!r} differs from the grasp arm {g.arm!r} (the hand that pushes grasps)")
    return push


def _grasp_stance(g: GraspRecord) -> str:
    """The stance a grasp is taken from (``grasps[].stance``): ``manipulation``, or the overhang pick's grasp stance ``pick`` (B backed off
    along its axis); the carry to the placement stance leaves from it."""
    st = g.extra.get("stance") if isinstance(g.extra, dict) else None
    return st if isinstance(st, str) and st else "manipulation"


def _pick_actions(sc: Sidecar, g: GraspRecord, stance: str = "manipulation") -> List[Action]:
    """grasp -> lift, preceded by the overhang push and its retreat when ``contacts.push_pick`` is recorded: push along the support (it may
    touch the support: ``touches``) -> retreat -> [the recorded travel to the grasp stance ``pick``] -> grasp at the object's current
    (pushed) pose -> lift."""
    grasp, lift = _pick(g)
    push = _pick_push(sc, g)
    if push is None:
        return [grasp, lift]
    support = push.extra.get("support")
    p = Push(contact=push, base_yaw=sc.stance(stance).base_pose2d[2], touches=(support,) if isinstance(support, str) and support else ())
    to_grasp = _travel(sc, stance, _grasp_stance(g)) if _grasp_stance(g) != stance else []
    return [p, Retreat(arm=push.arm, object=push.object, after=p)] + to_grasp + [grasp, lift]


def _place_and_retreat(g: GraspRecord, kind: str, relation: str = "on", avoid: Tuple[str, ...] = ()) -> List[Action]:
    place = Place(kind=kind, arm=g.arm, object=g.object, place=g.place, pre_place=g.pre_place, on=g.place_on, relation=relation, avoid=avoid)
    return [place, Retreat(arm=g.arm, object=g.object, after=place, distance=RETREAT_LITTLE)]


# annotation relation word of the IORS place per relation (the partner ends under the target for under_small)
_IORS_RELATION_WORDS = {"ontop": "on", "under": "under", "under_small": "on", "inside": "into", "nextto": "next to", "touching": "next to"}

#: ``contacts.push.layout`` of the touching#03 shelf layout (owner 2026-10-11 11:46): the target starts leaning against one side
#: board of an interior shelf and the partner against the other; the hand that carried the target pushes it toward the partner
#: (the left hand rightward for a partner on the right board) and nothing braces the partner
TOUCHING_SHELF_LAYOUT = "upright_shelf"


def _pack_stage_actions(sc: Sidecar, brace, pushes, base_yaw: float) -> List[Action]:
    """Block design packing stage (owner 2026-09-19 21:47): the wide brace (fingers pre-shaped to ``fingers_opening_m`` so a
    finger side rests on each of the two braced cubes) is placed and held, every listed push closes one cube's gap against it
    (the push arm retreats a little between pushes), then the brace arm releases and retreats."""
    out: List[Action] = []
    b = None
    if brace is not None:
        b = Brace(contact=brace, base_yaw=base_yaw)
        out.append(b)
    for push in pushes:
        if brace is not None and push.arm == brace.arm:
            raise SidecarError(f"pack: brace and push both use the {brace.arm} arm")
        p = Push(contact=push, base_yaw=base_yaw)
        # the pad leaves a pack push 5 cm opposite the push and 5 cm up (Retreat.escape_far; the sampler validates that pose)
        out += [p, Retreat(arm=push.arm, object=push.object, after=p, distance=0.05, escape_far=0.05)]
    if b is not None:
        # a wide brace holds nothing: its fingers stay at the recorded opening while the arm lifts off (a release would slide the
        # finger sides along the cubes' faces first); a closed-blade brace releases as before
        wide = (getattr(brace, "extra", None) or {}).get("fingers_opening_m") is not None
        out.append(Retreat(arm=brace.arm, object=brace.object, after=b, release=not wide))
    return out


def _brace_push_retreat(sc: Sidecar, brace, push, base_yaw: float, follow_braced: bool = False) -> List[Action]:
    """Brace first (held), push with the other arm, retreat the push arm, release and retreat the brace arm.
    ``follow_braced``: the braced object is untouched by the program until the brace (``Brace.follow``), and the push's goal
    was sampled against it (``Push.follow``)."""
    if brace.arm == push.arm:
        raise SidecarError(f"brace and push both use the {brace.arm} arm")
    b = Brace(contact=brace, base_yaw=base_yaw, follow=follow_braced)
    p = Push(contact=push, base_yaw=base_yaw, follow=brace.object if follow_braced else None)
    if os.environ.get("ISG_BRACE_RETREATS_FIRST", "0") == "1":
        # rush 2026-09-17: the brace pad leaves first while the push pad still holds the target - the brace caught the
        # painting with ~1 cm of penetration and its departure let the painting relax +0.05 rad toward the short side
        # (0917-220043 / 0917-214548); with the push pad still in contact that relaxation is blocked
        return [
            b,
            p,
            Retreat(arm=brace.arm, object=brace.object, after=b, release=True),
            Retreat(arm=push.arm, object=push.object, after=p),
        ]
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
    obstacles (every object the sidecar names) go into the annotation, not into the contact whitelist. The drive is
    ``face_travel`` (owner 2026-09-22): the long paths of this family are driven as the shared ``robot.base_path``
    schedule — also for the pilot's dense grid records, which it simplifies within one map cell."""
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
    return [Navigate(path=path, obstacles=obstacles, style=BP.FACE_TRAVEL), SetTorso(goal=sc.goal)]


def program_held_object_scaling(sc: Sidecar) -> List[Action]:
    """move the held object along the camera centre ray to the recorded depth (arm only), object orientation
    unchanged — through the recorded key poses ``grasps[0].waypoints`` (3 intermediate + the goal, spec §3.9 / §6.3;
    the sampler's ``line_waypoints(hold, goal, 4)``), which ``Reorient`` tracks as recorded."""
    g = _single_grasp(sc, "waypoints")
    return [Reorient(mode="line", arm=g.arm, object=g.object, start_pose=g.grasp, waypoints=g.waypoints)]


def program_held_object_rotation(sc: Sidecar) -> List[Action]:
    """rotate the held object to the goal orientation about its own centre (arm only), position unchanged — through
    the recorded key poses ``grasps[0].waypoints`` (3 intermediate SLERP poses + the goal; the sampler's
    ``orbit_waypoints(hold, object root, Δq, 4)``), which ``Reorient`` tracks as recorded."""
    g = _single_grasp(sc, "waypoints")
    return [Reorient(mode="orbit", arm=g.arm, object=g.object, start_pose=g.grasp, waypoints=g.waypoints)]


def program_disembedding(sc: Sidecar) -> List[Action]:
    """init → manipulation stance (transition) → grasp at the recorded pose → lift to the carry height (orientation
    kept) → drive sideways along the edge to the placement stance (recorded path; owner 2026-10-13: the heading kept, only
    forward / backward or left / right moves - ``_travel(axis_leg=True)``) → place at the destination centre (orientation
    kept) → retreat. The grasp may touch the parts / distractors of ``grasps[0].may_touch`` (owner 2026-10-12 A2b); after
    the grasp and after the lift each must still rest on the source tray (``execute.may_touch_verdict``)."""
    g = _single_grasp(sc, "pre_grasp", "carry_height", "place", "place_on")
    grasp, lift = _pick(g)
    return (
        _travel(sc, "init", "manipulation")
        + [grasp, lift]
        + _travel(sc, "manipulation", "placement", carried=g.object, axis_leg=True)
        + _place_and_retreat(g, "support")
    )


def table_flip_record(b) -> Optional[Dict[str, object]]:
    """The block's ``table_flip`` sub-chain (``blocks[k].table_flip``, carried in the record's ``extra`` until the
    sidecar schema names it): ``{"place": Pose, "pre_place": Pose | None, "block": BlockRecord}`` where ``block`` is
    the re-grasp as a block record (the other pitch class: ``face_pair`` / ``tilt_pick_deg`` / ``yaw_deg`` /
    ``carry_height`` / ``pick`` replaced, no flip); None when the block has none."""
    raw = (getattr(b, "extra", None) or {}).get("table_flip")
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise SidecarError(f"blocks[{b.order}].table_flip: expected an object, got {type(raw).__name__}")
    where = f"blocks[{b.order}].table_flip"
    for key in ("place", "regrasp"):
        if key not in raw:
            raise SidecarError(f"{where}.{key}: required")
    rg = raw["regrasp"]
    if not isinstance(rg, dict):
        raise SidecarError(f"{where}.regrasp: expected an object")
    for key in ("face_pair", "tilt_pick_deg", "yaw_deg", "pick"):
        if key not in rg:
            raise SidecarError(f"{where}.regrasp.{key}: required")
    extra = {k: v for k, v in (b.extra or {}).items() if k not in ("table_flip", "repose")}
    # the re-grasp's own pinch depth (its tip clearance) replaces leg 1's; the yaw-align is done (leg 2 keeps the orientation)
    for key in ("tip_clearance_m", "grasp_class", "stance"):
        if rg.get(key) is not None:
            extra[key] = rg[key]
    extra["yaw_align_deg"] = 0.0
    block = replace(b, face_pair=str(rg["face_pair"]), tilt_pick_deg=float(rg["tilt_pick_deg"]), yaw_deg=float(rg["yaw_deg"]),
                    carry_height=float(rg.get("carry_height", b.carry_height)), pick=Pose.from_dict(rg["pick"], f"{where}.regrasp.pick"),
                    flip_sign=0, flip_axis=None, waypoints=None, extra=extra)
    return {"place": Pose.from_dict(raw["place"], f"{where}.place"),
            "pre_place": Pose.from_dict(raw["pre_place"], f"{where}.pre_place") if raw.get("pre_place") else None,
            "block": block, "relation": raw.get("relation"),
            # "flip in place, then move" (owner 2026-09-19): leg 1 turns the block in hand before the set-down; the 2026-09-16
            # flip-on-the-table record (no in-hand reorient, the turn happens through the re-grasp) leaves the flag unset
            "in_hand_flip": bool(raw.get("in_hand_flip", False))}


#: ``blocks[k].repose.target`` values: the recorded pre-grasp configuration of the block's first grasp
REPOSE_TARGETS = ("pre_grasp",)


def block_repose(b, arm: str, moved_from: Optional[str]) -> Optional[Dict[str, Any]]:
    """The re-pose before a block's first grasp (``blocks[k].repose``, sampler ``generation.repose_after_base_move``, owner 2026-10-12
    item 2): after the between-block base move from ``moved_from`` to the block's stance the working arm makes a planned joint-space
    move to the recorded configuration ``q`` - the pre-grasp of this grasp in the IK branch the sampler tracked the block's legs in -
    before the grasp's transit, and the program check tracks the chain from it. Returns ``{"arm", "target", "q": [7 floats], "pose":
    Pose (the pre-grasp), "from_stance"}``, None without the record. A record without a base move before the block, for another arm,
    from another stance or malformed is a sidecar error."""
    raw = (getattr(b, "extra", None) or {}).get("repose")
    if raw is None:
        return None
    where = f"blocks[{b.order}].repose"
    if not isinstance(raw, dict):
        raise SidecarError(f"{where}: expected an object, got {type(raw).__name__}")
    if moved_from is None:
        raise SidecarError(f"{where}: recorded, but no base move precedes the block (it works from the stance the robot stands at)")
    if raw.get("target") not in REPOSE_TARGETS:
        raise SidecarError(f"{where}.target: one of {REPOSE_TARGETS}, got {raw.get('target')!r}")
    if raw.get("arm", arm) != arm:
        raise SidecarError(f"{where}.arm: the block's working arm is {arm!r}, the re-pose is recorded for {raw.get('arm')!r}")
    if raw.get("from_stance") is not None and raw["from_stance"] != moved_from:
        raise SidecarError(f"{where}.from_stance: recorded {raw['from_stance']!r}, the program arrives from {moved_from!r}")
    q = raw.get("q")
    if not isinstance(q, (list, tuple)) or len(q) != 7 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) for v in q):
        raise SidecarError(f"{where}.q: 7 finite joint values expected, got {q!r}")
    if "pose" not in raw:
        raise SidecarError(f"{where}.pose: required (the pre-grasp pose the configuration reaches)")
    return {"arm": arm, "target": str(raw["target"]), "q": [float(v) for v in q], "pose": Pose.from_dict(raw["pose"], f"{where}.pose"), "from_stance": moved_from}


def program_block_design(sc: Sidecar) -> List[Action]:
    """init → manipulation stance → for each block in the recorded order (anchor first): grasp (recorded cube grasp:
    pinch across ``face_pair``, ``tilt_pick_deg``, wrist yaw ``yaw_deg``) → lift → reorient_in_hand (smallest yaw ψ =
    ``yaw_align_deg`` to the table axes, then one 90° flip about the pinch axis; skipped for a block that needs
    neither — the wrist yaw alone is not a reorientation) → place at its slot → push it left against the placed row
    with the working hand while the closed other gripper braces the row-left blocks (where ``brace_row`` /
    ``push_row`` are recorded) → push it up while the other gripper braces the column-top blocks (where recorded) →
    retreat both grippers.

    The braces are **optional** (owner decision 2026-09-16 11:08, the axial rule): the sampler records ``brace_row`` /
    ``brace_col`` with their paired pushes only when a joint two-arm placement exists; a block without them is
    pick → [reorient] → place → retreat, no brace / push tail (all-or-none per axis is enforced by
    ``Sidecar.problems()`` and here).

    **Flip on the table** (owner decision 2026-09-16 17:10 item 1): a block whose goal face is not up is flipped on
    the table instead of in the hand when the sampler recorded the sub-chain ``blocks[k].table_flip`` (tabletop
    agent's record; the in-hand ``Reorient`` stays for blocks without it): ``{"place": Pose, "pre_place": Pose?,
    "cell": [x, y]?, "regrasp": {"face_pair", "tilt_pick_deg", "yaw_deg", "carry_height", "pick": Pose}}`` — the eef
    pose that sets the block down on a free cell in the intermediate orientation, then the cube grasp of the other
    pitch class that picks it up again (recorded like any grasp / place, all poses in frame). The program emits
    grasp → lift → place(intermediate) → retreat → grasp(re-grasp) → lift → place(slot) → retreat; each grasp →
    release pair is its own gripper-in-FOV span (QA rule (b)).

    **Per-block stances** (owner 2026-09-19: the robot may move between blocks for IK feasibility): a block whose record
    names a ``stance`` other than the one the robot stands at is preceded by the recorded travel (path and / or
    transition, hands empty) from the current stance to it; the first block's stance defaults to ``manipulation``. The
    working arm and the base yaw of the brace / push tail are those of the block's own stance.

    **Re-pose after a between-block base move** (owner 2026-10-12 item 2, ``blocks[k].repose``): the block's first grasp carries the
    recorded joint-space re-pose (:func:`block_repose`, ``Grasp.repose``)."""
    if not sc.blocks:
        raise SidecarError("blocks: required for block design")
    current = "manipulation"
    actions = _travel(sc, "init", current)
    for b in sorted(sc.blocks, key=lambda blk: blk.order):
        wanted = str((b.extra or {}).get("stance") or "manipulation")
        moved_from = None
        if wanted != current:
            actions += _travel(sc, current, wanted)
            moved_from = current
            current = wanted
        arm = _block_working_arm(sc, current)
        base_yaw = sc.stance(current).base_pose2d[2]
        grasp = Grasp(kind="cube", arm=arm, object=b.instance, block=b, repose=block_repose(b, arm, moved_from))
        lift = Lift(arm=arm, object=b.instance, carry_height=b.carry_height, after=grasp)
        actions += [grasp, lift]
        last: Action = lift
        flip = table_flip_record(b)
        if flip is not None:
            # two legs (owner 2026-09-19, "flip in place, then move"): leg 1 turns the block in hand (the record's flip fields:
            # the steep side pick, the recorded flip waypoints) and sets it down again — on its own cell in the goal orientation
            # (``table_flip.place``) — leg 2 re-grasps it with a plain pitched pinch (``table_flip.regrasp``) and places it at
            # the slot; the in-hand flip's place left of the pick cell was the one step no stance passed
            if flip.get("in_hand_flip") and block_reorients(b):
                last = Reorient(mode="flip", arm=arm, object=b.instance, after=lift, block=b, waypoints=b.waypoints)
                actions.append(last)
            mid = Place(kind="support", arm=arm, object=b.instance, place=flip["place"], pre_place=flip.get("pre_place"),
                        relation=str(flip.get("relation") or "back on its own cell, turned to its goal face, for the re-grasp"))
            actions += [mid, Retreat(arm=arm, object=b.instance, after=mid, distance=RETREAT_LITTLE)]
            # the re-grasp may work from another recorded stance (``table_flip.regrasp.stance``): travel with empty hands first
            wanted2 = str((flip["block"].extra or {}).get("stance") or current)
            if wanted2 != current:
                actions += _travel(sc, current, wanted2)
                current = wanted2
                arm = _block_working_arm(sc, current)
                base_yaw = sc.stance(current).base_pose2d[2]
            regrasp = Grasp(kind="cube", arm=arm, object=b.instance, block=flip["block"])
            relift = Lift(arm=arm, object=b.instance, carry_height=flip["block"].carry_height, after=regrasp)
            actions += [regrasp, relift]
            last = relift
        elif block_reorients(b):
            last = Reorient(mode="flip", arm=arm, object=b.instance, after=lift, block=b, waypoints=b.waypoints)
            actions.append(last)
        slot = "at its initial position (the anchor)" if b.order == 0 else "at its slot in the goal grid"
        # the recorded pre-place when the sampler wrote one (``blocks[k].pre_place``: the flip classes pass a 5 cm descent
        # where the 10 cm default does not), else the conventions' PRE_DIST back along the approach
        pre_raw = (b.extra or {}).get("pre_place")
        pre_place = Pose.from_dict(pre_raw, f"blocks[{b.order}].pre_place") if isinstance(pre_raw, dict) else None
        place = Place(kind="support", arm=arm, object=b.instance, place=b.place, pre_place=pre_place, relation=slot)
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
    # packing tail (owner 2026-09-19 21:47, ``sidecar.pack``): with every block at its (loose) place, the columns are closed
    # against the wide-braced left column, then the rows against the wide-braced far (or near) row - the blocks end packed
    pack = (sc.extra or {}).get("pack") if isinstance(getattr(sc, "extra", None), dict) else None
    if isinstance(pack, dict) and pack.get("stages"):
        wanted = str(pack.get("stance") or "manipulation")
        if wanted != current:
            actions += _travel(sc, current, wanted)
            current = wanted
        base_yaw = sc.stance(current).base_pose2d[2]
        for i, stage in enumerate(pack["stages"]):
            brace = BraceContact.from_dict(stage["brace"], f"pack.stages[{i}].brace") if stage.get("brace") else None  # None: pushes only
            pushes = [PushContact.from_dict(p, f"pack.stages[{i}].pushes[{j}]") for j, p in enumerate(stage.get("pushes") or [])]
            if not pushes:
                raise SidecarError(f"pack.stages[{i}]: no pushes")
            actions += _pack_stage_actions(sc, brace, pushes, base_yaw)
    return actions


#: ``articulation.door_class`` of a room door's revolute leaf (sampler ``articulation_large.door_class``, owner decision A4)
ROOM_DOOR_CLASS = "room_door"


def _circle_xy(p0: Sequence[float], p1: Sequence[float], p2: Sequence[float]) -> Tuple[float, float, float]:
    """Centre (x, y) and radius of the circle through three planar points."""
    (x0, y0), (x1, y1), (x2, y2) = p0[:2], p1[:2], p2[:2]
    a11, a12, b1 = 2.0 * (x1 - x0), 2.0 * (y1 - y0), x1 * x1 + y1 * y1 - x0 * x0 - y0 * y0
    a21, a22, b2 = 2.0 * (x2 - x0), 2.0 * (y2 - y0), x2 * x2 + y2 * y2 - x0 * x0 - y0 * y0
    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-12:
        raise SidecarError("articulation.waypoints: the handle contacts are collinear, no hinge circle")
    cx, cy = (b1 * a22 - a12 * b2) / det, (a11 * b2 - b1 * a21) / det
    return cx, cy, math.hypot(x0 - cx, y0 - cy)


def door_leaf_geometry(art) -> Dict[str, object]:
    """The leaf of a vertical-axis door from the recorded handle waypoints h(q_k): the hinge = the centre of the circle
    through the handle contacts at the first, middle and last recorded fraction, the handle radius about it, and the
    closed leaf's handle contact / approach / height = the waypoint at the lower joint limit (open: k = 0; close: k = 1)."""
    keys = [key for _, key in art.fractions()]
    if len(keys) < 3:
        raise SidecarError(f"articulation.waypoints: the door leaf needs >= 3 recorded handle waypoints, got {len(keys)}")
    hx, hy, r = _circle_xy(*(art.waypoints[k].contact for k in (keys[0], keys[len(keys) // 2], keys[-1])))
    closed = art.waypoints[keys[0] if art.q_init <= art.q_goal else keys[-1]]
    return {"hinge_xy": (hx, hy), "handle_radius_m": r, "handle_xy_closed": (float(closed.contact[0]), float(closed.contact[1])),
            "approach_closed": tuple(float(v) for v in closed.approach), "handle_z": float(closed.contact[2])}


#: keys of an ``articulation.push_finish`` record the program and ``push_close`` read (sampler ``articulation_large.plan_push_finish``)
PUSH_FINISH_KEYS = ("arm", "base_pose2d", "k_release", "q_from", "q_to", "hinge_world", "swing_sign", "contact", "normal")


def push_finish_chunks(art) -> List[Dict[str, Any]]:
    """The planned ``push_close`` chunks of a cabinet door's push finish (owner 2026-10-12 ~04:20; ``articulation.push_finish``,
    one record): the hand follows the handle to ``articulation.follow_to`` (= the record's ``k_release``), lets go, and the closed
    hand pushes the leaf from ``q_from`` = q(follow_to) to ``q_to`` with the recorded contact from the recorded stance."""
    rec = (art.extra or {}).get("push_finish")
    w = "articulation.push_finish"
    if not isinstance(rec, dict):
        raise SidecarError(f"{w}: expected a dict, got {type(rec).__name__}")
    missing = [k for k in PUSH_FINISH_KEYS if rec.get(k) is None]
    if missing:
        raise SidecarError(f"{w}: missing {missing}")
    if art.joint_type != "revolute":
        raise SidecarError(f"{w}: a revolute door leaf only, got joint_type {art.joint_type!r}")
    if rec["arm"] not in ARMS:
        raise SidecarError(f"{w}.arm: unknown arm {rec['arm']!r}")
    follow_to = (art.extra or {}).get("follow_to")
    if follow_to is None or not 0.0 < float(follow_to) < 1.0:
        raise SidecarError(f"{w}: articulation.follow_to must be recorded in (0, 1) (the release), got {follow_to!r}")
    if abs(float(follow_to) - float(rec["k_release"])) > 1e-6 or abs(art.q_at(float(follow_to)) - float(rec["q_from"])) > 1e-4:
        raise SidecarError(f"{w}: the push starts where the hand lets go - k_release {rec['k_release']} / q_from {rec['q_from']} vs follow_to {follow_to}")
    if float(rec["swing_sign"]) not in (1.0, -1.0):
        raise SidecarError(f"{w}.swing_sign: +1 or -1, got {rec['swing_sign']!r}")
    for k, n in (("base_pose2d", 3), ("hinge_world", 3), ("normal", 3)):
        if len(rec[k]) != n:
            raise SidecarError(f"{w}.{k}: expected {n} numbers")
    Pose.from_dict(rec["contact"], f"{w}.contact")
    return [dict(rec)]


#: ``articulation.push_close`` keys the program passes to ``PushClose`` for a cabinet door pushed shut room-door style (sampler
#: ``articulation_large.close_by_push_record``); a room door records none and keeps the ``PushClose`` defaults
PUSH_CLOSE_OPTIONS = ("push_inset_m", "push_height_drop_m", "chunk_deg", "drag_m", "contact_clearance_m", "leaf_width_m", "face_offset_m", "handle_standoff_m")


def close_by_push(art) -> bool:
    """Whether the record closes a cabinet door room-door style (``articulation.close_by_push``, sampler ``generation.close_by_push``):
    ``true``; ``false`` / absent = as before; anything else is a sidecar error."""
    v = (art.extra or {}).get("close_by_push")
    if v is None or v is False:
        return False
    if v is not True:
        raise SidecarError(f"articulation.close_by_push: true or false, got {v!r}")
    return True


def push_close_options(art, geo: Dict[str, object]) -> Dict[str, float]:
    """The ``PushClose`` options of a cabinet door pushed shut room-door style (``articulation.push_close``): the push's knobs and the
    leaf the sampler measured (:data:`PUSH_CLOSE_OPTIONS`), plus ``q_closed`` = the lower limit (the closed waypoint's joint value).
    The recorded hinge and swing sign must agree with the handle waypoints' (``geo`` = :func:`door_leaf_geometry`): the hinge within
    3 cm, the sign of the turn that swings the outer face toward the robot as the joint opens."""
    rec = (art.extra or {}).get("push_close")
    w = "articulation.push_close"
    if not isinstance(rec, dict):
        raise SidecarError(f"{w}: expected a dict (articulation.close_by_push), got {type(rec).__name__}")
    missing = [k for k in PUSH_CLOSE_OPTIONS + ("swing_sign", "hinge_world") if rec.get(k) is None]
    if missing:
        raise SidecarError(f"{w}: missing {missing}")

    def number(v) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

    bad = [k for k in PUSH_CLOSE_OPTIONS + ("swing_sign",) if not number(rec[k])]
    if not isinstance(rec["hinge_world"], (list, tuple)) or len(rec["hinge_world"]) != 3 or not all(number(v) for v in rec["hinge_world"]):
        bad.append("hinge_world (3)")
    if bad:
        raise SidecarError(f"{w}: {bad} must be numbers")
    opts = {k: float(rec[k]) for k in PUSH_CLOSE_OPTIONS}
    r = float(geo["handle_radius_m"])
    if not (0.0 < opts["push_inset_m"] < opts["leaf_width_m"] and 0.0 < opts["chunk_deg"] <= 90.0 and opts["drag_m"] >= 0.0 and 0.0 <= opts["contact_clearance_m"] < 0.10
            and 0.0 <= opts["handle_standoff_m"] < r):
        raise SidecarError(f"{w}: out of range - 0 < push_inset_m < leaf_width_m, 0 < chunk_deg <= 90, drag_m >= 0, 0 <= contact_clearance_m < 0.10, "
                           f"0 <= handle_standoff_m < the handle radius {r:.3f}: {opts}")
    hx, hy = geo["hinge_xy"]
    hinge = rec["hinge_world"]
    if math.hypot(float(hinge[0]) - hx, float(hinge[1]) - hy) > 0.03:
        raise SidecarError(f"{w}.hinge_world: {hinge!r} is not the hinge of the handle waypoints ({hx:.3f}, {hy:.3f})")
    (cx, cy), (ax, ay) = geo["handle_xy_closed"], geo["approach_closed"][:2]
    s = 1.0 if (-(cy - hy)) * (-ax) + (cx - hx) * (-ay) > 0.0 else -1.0  # execute.leaf_frame's rule: hinge -> handle turned +90 deg, dotted with -approach
    if float(rec["swing_sign"]) != s:
        raise SidecarError(f"{w}.swing_sign: {rec['swing_sign']!r}, the handle waypoints swing the leaf with {s:+.0f}")
    opts["q_closed"] = min(art.q_init, art.q_goal)
    return opts


def program_articulation_large(sc: Sidecar, direction: str = "move") -> List[Action]:
    """init → manipulation stance (transition) → grasp the handle at h(q_init) → move the joint: base_follow along the
    recorded b_k (``follow_mode`` base) or the arm along h(q_k) (``follow_mode`` arm), up to ``passive_from`` or q_goal
    → release → retreat.

    A **room door is closed by pushing its outer face** (owner decision B, 2026-09-20 20:19; ``articulation.door_class``
    ``room_door``): init → manipulation stance (path + transition) → ``push_close`` with the working arm — no grasp; the
    leaf geometry from the recorded handle waypoints (:func:`door_leaf_geometry`).

    A **cabinet door whose closing the arm follows only part of the way is pushed the rest of the way shut** (owner
    2026-10-12 ~04:20; ``articulation.push_finish``, sampler ``generation.push_finish``): grasp → follow up to
    ``articulation.follow_to`` → release → retreat → ``push_close`` with the planned chunk (:func:`push_finish_chunks`), judged on
    the instance's joint tolerance.

    A **cabinet door closed room-door style** (owner 2026-10-12; ``articulation.close_by_push``, sampler
    ``generation.close_by_push``) takes the room door's branch, with the leaf and the push knobs the sampler recorded
    (``articulation.push_close``, :func:`push_close_options`)."""
    art = sc.articulation
    if art is None:
        raise SidecarError("articulation: required for articulation tasks")
    arm = _working_arm(sc)
    pushed = close_by_push(art)
    if pushed:
        if direction != "close":
            raise SidecarError(f"articulation.close_by_push: a close pushes the leaf shut, the variant is a {direction!r}")
        if art.joint_type != "revolute":
            raise SidecarError(f"articulation.joint_type: a door leaf pushed shut is revolute, got {art.joint_type!r}")
        if (art.extra or {}).get("push_finish") is not None:
            raise SidecarError("articulation: close_by_push (no grasp) and push_finish (grasp, then a push) exclude each other")
    if direction == "close" and ((art.extra or {}).get("door_class") == ROOM_DOOR_CLASS or pushed):
        if art.joint_type != "revolute":
            raise SidecarError(f"articulation.joint_type: a room door leaf is revolute, got {art.joint_type!r}")
        geo = door_leaf_geometry(art)
        opts = push_close_options(art, geo) if pushed else {}
        push = PushClose(arm=arm, object=art.object, joint=art.joint, q_from=art.q_init, q_lower=min(art.q_init, art.q_goal), **geo, **opts)
        return _travel(sc, "init", "manipulation") + [push]
    grasp = Grasp(kind="handle", arm=arm, object=art.object, articulation=art)
    follow = BaseFollow(arm=arm, articulation=art, stance=sc.stance("manipulation"), grasp_action=grasp, direction=direction)
    actions = _travel(sc, "init", "manipulation") + [grasp, follow, Retreat(arm=arm, object=art.object, after=follow, release=True)]
    if direction == "close" and (art.extra or {}).get("push_finish") is not None:
        chunks = push_finish_chunks(art)
        follow_to = float((art.extra or {})["follow_to"])
        actions.append(PushClose(arm=str(chunks[0]["arm"]), object=art.object, joint=art.joint, q_from=art.q_at(follow_to), q_lower=min(art.q_init, art.q_goal),
                                 tol_span_rad=abs(art.q_goal - art.q_init), plan=chunks, drag_m=0.0, **door_leaf_geometry(art)))
    return actions


def _handle_arm(sc: Sidecar, brace) -> str:
    """The arm that grasps the handle of a small-scale articulation: ``articulation.handle_arm`` (the sampler's record,
    tabletop ``articulation_small``), else the manipulation stance's ``handle_arm``, else its single working arm, else —
    two working arms with a brace record — the arm that is not the brace arm."""
    art = sc.articulation
    st = sc.stance("manipulation")
    for src, v in (("articulation.handle_arm", (art.extra or {}).get("handle_arm") if art is not None else None), ("stances.manipulation.handle_arm", st.extra.get("handle_arm"))):
        if v is not None:
            if v not in ARMS:
                raise SidecarError(f"{src}: unknown arm {v!r}")
            return str(v)
    arms = st.working_arms
    if len(arms) == 1:
        return arms[0]
    if brace is not None and len(arms) == 2 and brace.arm in arms:
        return _other_arm(brace.arm)
    raise SidecarError(f"stances['manipulation'].arms: cannot tell the handle arm from {arms!r} (record articulation.handle_arm)")


def program_articulation_small(sc: Sidecar, direction: str = "move") -> List[Action]:
    """init → manipulation stance → [brace the base link (recorded kind) with the brace arm, when the sidecar records
    one] → grasp the handle with the handle arm → move the arm along h(q_k) to the goal (or ``passive_from``) → release
    → retreat (both, when braced).

    Small articulation = **brace optional** (owner decision D-008 = A, 2026-09-16 22:02; the axial / touching rule): the
    sampler (tabletop ``articulation_small``, ``generation.brace_optional``) records ``contacts.brace`` only when a joint
    two-arm cell exists — the handle grasp and the brace pass the exact two-arm IK from the manipulation or a second
    stance; otherwise the instance is handle-only (``contacts.brace`` absent, ``summary.brace: none``,
    ``stances.manipulation.arms == [handle_arm]``, ``articulation.brace_kind: null``) and the program is
    ``transition → grasp(handle) → articulate → release → retreat``. ``articulation.handle_arm`` names the handle arm."""
    art = sc.articulation
    if art is None:
        raise SidecarError("articulation: required for articulation tasks")
    if art.follow_mode != "arm":
        raise SidecarError(f"articulation.follow_mode: small-scale articulation moves the arm, got {art.follow_mode!r}")
    brace = sc.contacts.brace if sc.contacts is not None else None
    arm = _handle_arm(sc, brace)
    lever_braced = bool(((art.extra or {}).get("lever") or {}).get("brace"))  # two-flap v4 (owner 23:20): the pinching hand braces FIRST, then grasps
    if brace is not None:
        if brace.arm != _other_arm(arm) and not lever_braced:
            raise SidecarError(f"contacts.brace.arm: the handle arm is {arm!r}, the brace must use {_other_arm(arm)!r}")
        if brace.object != art.object:
            raise SidecarError("contacts.brace.object: the brace must act on the articulated object's base link")
    grasp = Grasp(kind="handle", arm=arm, object=art.object, articulation=art)
    follow = BaseFollow(arm=arm, articulation=art, stance=sc.stance("manipulation"), grasp_action=grasp, direction=direction)
    head = _travel(sc, "init", "manipulation")
    base_yaw = sc.stance("manipulation").base_pose2d[2]
    # stage 2 (owner 2026-09-18 22:38; sampler ``stage2_push``): the handle follow ends at ``articulation.follow_to`` where the
    # arm's reach ends, the joint brake holds the lid there, the hand lets go, closes, and ``contacts.push`` (a fist on the lid's
    # underside inside the opening) rides the hinge arc to q_goal - the push then retreats
    push = sc.contacts.push if sc.contacts is not None else None
    # two-flap carton (owner 2026-09-19 01:10; sampler ``prep_push: other_flap_down``): ``contacts.push_prep`` = the free hand's
    # closed-gripper press of the OTHER flap down at the seam, held while the handle hand pinches the target flap's edge; the
    # free hand then leaves the flap (it springs back) and, when recorded, braces the box top at that flap's hinge line;
    # then the follow, the release, the optional stage-2 fist push, and every hand retreats
    prep_d = (sc.contacts.extra or {}).get("push_prep") if sc.contacts is not None else None
    # two-flap carton v2 (owner 2026-09-19 14:34; sampler ``prep_push: two_flap_v2``): ``articulation.first`` = a first, small
    # opening by the OTHER hand pinching the target flap's free edge on the centre line of the seam while the main hand's fist
    # presses the other flap down (``contacts.push_prep``, held); the first hand releases and retreats, the fist leaves (the
    # flap springs back), then the first hand's fist presses the other flap down (``contacts.push_prep2``, held) while the main
    # hand pinches the now-ajar flap's edge near the robot (``articulation``, from q_ajar) and opens it to the goal; both retreat
    # push-close (owner 2026-09-19 22:59; sampler ``close_push``): no grasp - the closed fist rides the closing arc at the link's tip
    # in segments (``contacts.push``, ``push2``, ...), each from its own stance (a recorded transition in between, the brake
    # holding the link), each followed by its retreat
    cp = (art.extra or {}).get("close_push")
    if cp is not None:
        from ispatialgym.oracle.sidecar import PushContact

        segs = list(cp.get("segments") or [])
        if not segs:
            raise SidecarError("articulation.close_push: needs at least one segment")
        from ispatialgym.oracle.sidecar import BraceContact

        cx = sc.contacts.extra or {}
        prev = "init"
        actions: List[Action] = []
        for seg in segs:
            st_name = str(seg.get("stance") or "manipulation")
            if st_name != prev:
                actions += _travel(sc, prev, st_name)
                prev = st_name
            slot = str(seg.get("push") or "push")
            pc = push if slot == "push" else (PushContact.from_dict(cx[slot], f"contacts.{slot}") if cx.get(slot) is not None else None)
            if pc is None:
                raise SidecarError(f"articulation.close_push: contacts.{slot} missing")
            if (pc.extra or {}).get("articulation") is None:
                raise SidecarError(f"contacts.{slot}: a push-close segment needs its 'articulation' record (joint, q_from, q_to)")
            yaw = sc.stance(st_name).base_pose2d[2]
            # owner 2026-09-20 16:24: the pushes shove the box along the push direction - the other hand's OPEN gripper braces the
            # face the box slides toward (``contacts.brace`` / ``brace2`` ...; the touching demo's wide stop) for the segment, and
            # lifts off (fingers stay at the opening) after the fist's retreat, before the base moves on
            b = None
            b_slot = seg.get("brace")
            if b_slot is not None:
                bd = sc.contacts.brace if b_slot == "brace" else (BraceContact.from_dict(cx[b_slot], f"contacts.{b_slot}") if cx.get(b_slot) is not None else None)
                if bd is None:
                    raise SidecarError(f"articulation.close_push: contacts.{b_slot} missing")
                if bd.arm == pc.arm:
                    raise SidecarError(f"push-close: brace and push both use the {pc.arm} arm")
                b = Brace(contact=bd, base_yaw=yaw)
                actions.append(b)
            p = Push(contact=pc, base_yaw=yaw)
            # the retreat as far as the segment's pre-contact (the sampler checked that pose; a 10 cm leg along the fist's axis had
            # no solution at the segment ends, 0920-064524)
            d_r = float((pc.extra or {}).get("retreat_m") or 0.06)
            actions += [p, Retreat(arm=pc.arm, object=art.object, after=p, distance=d_r)]
            if b is not None:
                bx = getattr(b.contact, "extra", None) or {}
                wide = bx.get("fingers_opening_m") is not None
                actions.append(Retreat(arm=b.contact.arm, object=art.object, after=b, release=not wide, distance=float(bx.get("retreat_m") or RETREAT_DIST)))
        return actions
    # two-flap v4, the lever (owner 2026-09-20 15:21; sampler ``prep_push: two_flap_v4``): the free hand's flat fist presses the
    # other flap near the seam (``contacts.push_prep``, held), slides into the opening under the target flap's edge and lifts it
    # to q_ajar (``contacts.push_lift``, held; its pre-contact is the press's end pose); the main hand pinches the raised edge
    # (``articulation`` from q_ajar); the fist slides back out to the left and hovers; the main hand opens the flap to the goal,
    # a stage-2 fist push when its reach ends; retreats
    lever = (art.extra or {}).get("lever")
    if lever is not None:
        from ispatialgym.oracle.sidecar import PushContact

        cx = sc.contacts.extra or {}
        if cx.get("push_prep") is None or cx.get("push_lift") is None:
            raise SidecarError("articulation.lever needs contacts.push_prep (the press) and contacts.push_lift (the slide + lift)")
        prep = PushContact.from_dict(cx["push_prep"], "contacts.push_prep")
        lift = PushContact.from_dict(cx["push_lift"], "contacts.push_lift")
        if prep.arm == arm or lift.arm != prep.arm:
            raise SidecarError(f"two-flap v4: the lever must be the hand opposite to the pinching {arm!r} (press {prep.arm!r}, lift {lift.arm!r})")
        for nm, pc in (("push_prep", prep), ("push_lift", lift)):
            if (pc.extra or {}).get("articulation") is None:
                raise SidecarError(f"contacts.{nm}: needs its 'articulation' record (joint, q_from, q_to)")
        p_press = Push(contact=prep, base_yaw=base_yaw)
        p_lift = Push(contact=lift, base_yaw=base_yaw)
        d_lever = float(lever.get("retreat_m") or 0.08)
        # owner 2026-09-20 21:20: the main hand's OPEN gripper braces the box's face on the hinge side (``contacts.brace``, the
        # touching demo's wide stop) through the press / slide / lift - the lever shoved the box 3-4 cm sideways - and lifts off
        # (fingers stay open) before it grasps; the fist's retreat lifts off the pressed flap first, then slides away from the seam
        b = None
        if sc.contacts.brace is not None and lever.get("brace"):
            bd = sc.contacts.brace
            if bd.arm != arm or bd.arm == lift.arm:
                raise SidecarError(f"two-flap v4: the brace must be the pinching {arm!r} hand (recorded {bd.arm!r})")
            b = Brace(contact=bd, base_yaw=base_yaw)
        lever_retreat = Retreat(arm=lift.arm, object=art.object, after=p_lift, distance=d_lever,
                                lift_first_m=(float(lever["retreat_lift_m"]) if lever.get("retreat_lift_m") is not None else None),
                                slide_dir=(list(lever["retreat_dir"]) if lever.get("retreat_dir") is not None else None))
        actions: List[Action] = list(head)
        if b is not None:
            actions.append(b)
        actions += [p_press, p_lift]
        if b is not None:
            # the blade lifts off with its fingers as they are (0921-021042: opening them 2 mm from the wall shoved the box 1.2 cm);
            # the grasp that follows opens the gripper itself
            bx = getattr(bd, "extra", None) or {}
            actions.append(Retreat(arm=bd.arm, object=art.object, after=b, release=False, distance=float(bx.get("retreat_m") or RETREAT_DIST)))
        actions += [grasp, lever_retreat, follow, Retreat(arm=arm, object=art.object, after=follow, release=True)]
        if push is not None and (art.extra or {}).get("follow_to") is not None:
            if (push.extra or {}).get("articulation") is None:
                raise SidecarError("contacts.push: a small-articulation stage-2 push needs its 'articulation' record (joint, q_from, q_to)")
            p2 = Push(contact=push, base_yaw=base_yaw)
            actions += [p2, Retreat(arm=push.arm, object=art.object, after=p2)]
        return actions
    first_d = (art.extra or {}).get("first")
    if first_d is not None:
        from ispatialgym.oracle.sidecar import Articulation, PushContact

        first = Articulation.from_dict(first_d, "articulation.first")
        arm1 = str((first.extra or {}).get("handle_arm") or _other_arm(arm))
        if arm1 not in ARMS:
            raise SidecarError(f"articulation.first.handle_arm: unknown arm {arm1!r}")
        cx = sc.contacts.extra or {}
        single = bool((art.extra or {}).get("single_press"))
        if cx.get("push_prep") is None or (cx.get("push_prep2") is None and not single):
            raise SidecarError("articulation.first needs contacts.push_prep (the press held during the first opening) and contacts.push_prep2 (the press held during the main opening), or one continuous press (single_press)")
        prep1 = PushContact.from_dict(cx["push_prep"], "contacts.push_prep")
        prep2 = prep1 if single else PushContact.from_dict(cx["push_prep2"], "contacts.push_prep2")
        if prep1.arm == arm1 or prep2.arm == arm:
            raise SidecarError(f"two-flap v2: a press must use the hand opposite to the pinching one (push_prep {prep1.arm!r} vs first hand {arm1!r}; push_prep2 {prep2.arm!r} vs main hand {arm!r})")
        for nm, pc in (("push_prep", prep1), ("push_prep2", prep2)):
            if (pc.extra or {}).get("articulation") is None:
                raise SidecarError(f"contacts.{nm}: needs its 'articulation' record (the pressed flap's joint, q_from, q_to)")
        # owner 2026-09-19 16:55: the two phases may use two stances - "manipulation" for the first opening, "manipulation2" for
        # the main one, with a recorded transition between them (the flap ajar, held by the brake, nothing in hand)
        m1 = sc.stance("manipulation")
        try:
            m2 = sc.stance("manipulation2")
        except SidecarError:
            m2 = m1
        p1 = Push(contact=prep1, base_yaw=m1.base_pose2d[2])
        g1 = Grasp(kind="handle", arm=arm1, object=art.object, articulation=first)
        f1 = BaseFollow(arm=arm1, articulation=first, stance=m1, grasp_action=g1, direction=direction)
        follow2 = BaseFollow(arm=arm, articulation=art, stance=m2, grasp_action=grasp, direction=direction)
        # a press's retreat is as long as its pre-contact (the sampler's 6 cm, prep_push_pre_dist_m): the fist's tilted axis makes a
        # 10 cm leg untrackable where the 6 cm approach passed (0920-003018)
        d_press = 0.06
        d_first = float((first.extra or {}).get("retreat_m") or 0.10)  # the first hand's release retreat as the sampler checked it
        phase1 = [p1, g1, f1, Retreat(arm=arm1, object=art.object, after=f1, release=True, distance=d_first)]
        if single and m2 is m1:
            # owner 2026-09-19 19:52: the other hand's fist keeps pressing through both openings; the main hand lets go of the
            # ajar flap and re-grasps it at its near end
            return head + phase1 + [grasp, follow2, Retreat(arm=arm, object=art.object, after=follow2, release=True), Retreat(arm=prep1.arm, object=art.object, after=p1, distance=d_press)]
        p2 = Push(contact=prep2, base_yaw=m2.base_pose2d[2])
        mid = _travel(sc, "manipulation", "manipulation2") if m2 is not m1 else []
        return head + phase1 + [Retreat(arm=prep1.arm, object=art.object, after=p1, distance=d_press)] + mid + \
            [p2, grasp, follow2, Retreat(arm=arm, object=art.object, after=follow2, release=True), Retreat(arm=prep2.arm, object=art.object, after=p2, distance=d_press)]
    if prep_d is not None:
        from ispatialgym.oracle.sidecar import PushContact

        prep = PushContact.from_dict(prep_d, "contacts.push_prep")
        if prep.arm not in ARMS:
            raise SidecarError(f"contacts.push_prep.arm: unknown arm {prep.arm!r}")
        if (prep.extra or {}).get("articulation") is None:
            raise SidecarError("contacts.push_prep: needs its 'articulation' record (the pressed flap's joint, q_from, q_to)")
        p0 = Push(contact=prep, base_yaw=base_yaw)
        actions: List[Action] = head + [p0, grasp, Retreat(arm=prep.arm, object=art.object, after=p0, distance=0.06)]
        if brace is not None:
            if brace.arm == arm:
                raise SidecarError(f"contacts.brace.arm: the handle arm is {arm!r}, the brace must use the other hand")
            b = Brace(contact=brace, base_yaw=base_yaw)
            actions.append(b)
        actions += [follow, Retreat(arm=arm, object=art.object, after=follow, release=True)]
        if push is not None and (art.extra or {}).get("follow_to") is not None:
            if (push.extra or {}).get("articulation") is None:
                raise SidecarError("contacts.push: a small-articulation stage-2 push needs its 'articulation' record (joint, q_from, q_to)")
            p = Push(contact=push, base_yaw=base_yaw)
            actions += [p, Retreat(arm=push.arm, object=art.object, after=p)]
        if brace is not None:
            actions.append(Retreat(arm=brace.arm, object=brace.object, after=b, release=True))
        return actions
    if push is not None and (art.extra or {}).get("follow_to") is not None:
        if push.arm not in ARMS:
            raise SidecarError(f"contacts.push.arm: unknown arm {push.arm!r}")
        if (push.extra or {}).get("articulation") is None:
            raise SidecarError("contacts.push: a small-articulation stage-2 push needs its 'articulation' record (joint, q_from, q_to)")
        if brace is not None:
            raise SidecarError("contacts.brace: a braced two-arm instance cannot carry a stage-2 push (the brace arm would have to let go first)")
        p = Push(contact=push, base_yaw=base_yaw)  # either arm: the other hand's fist is the natural one for a hinge on the handle arm's side
        return head + [grasp, follow, Retreat(arm=arm, object=art.object, after=follow, release=True), p, Retreat(arm=push.arm, object=art.object, after=p)]
    if brace is None:
        return head + [grasp, follow, Retreat(arm=arm, object=art.object, after=follow, release=True)]
    b = Brace(contact=brace, base_yaw=base_yaw)
    return head + [
        b,
        grasp,
        follow,
        Retreat(arm=arm, object=art.object, after=follow, release=True),
        Retreat(arm=brace.arm, object=brace.object, after=b, release=True),
    ]


def _attach_stance(sc: Sidecar, g: GraspRecord) -> str:
    """The stance the attach (and the brace) were solved from: ``grasps[0].attach_stance`` (tabletop ``58afae2``), else
    the recorded ``attach`` stance, else the grasp stance."""
    name = g.extra.get("attach_stance")
    if name is None:
        name = "attach" if "attach" in sc.stances else "manipulation"
    if not isinstance(name, str) or name not in sc.stances:
        raise SidecarError(f"grasps[0].attach_stance: stance {name!r} is not recorded")
    return str(name)


def _carried_lift_pose(sc: Sidecar, g: GraspRecord, frm: str, to: str) -> Pose:
    """Where the lifted grasp sits after the base moved from stance ``frm`` to ``to`` with the arm frozen: the sampler's
    own record (``grasps[0].lift``, the shifted lift the SLERP waypoints start from), else the recorded grasp pose
    lifted by ``carry_height`` and carried along the base motion (the same eef pose in the base frame)."""
    rec = g.extra.get("lift")
    if isinstance(rec, (list, tuple)) and len(rec) == 3:
        return Pose([float(v) for v in rec], list(g.grasp.orientation))
    end = g.grasp.translated([0.0, 0.0, float(g.carry_height)])
    (x0, y0, yaw0), (x1, y1, yaw1) = sc.stance(frm).base_pose2d, sc.stance(to).base_pose2d
    c, s = math.cos(yaw0), math.sin(yaw0)
    dx, dy = end.position[0] - x0, end.position[1] - y0
    bx, by = c * dx + s * dy, -s * dx + c * dy  # the eef in the departing base frame
    c1, s1 = math.cos(yaw1), math.sin(yaw1)
    pos = [x1 + c1 * bx - s1 * by, y1 + s1 * bx + c1 * by, end.position[2]]
    dyaw = yaw1 - yaw0
    if abs(dyaw) > 1e-9:
        half = 0.5 * dyaw
        qz = [0.0, 0.0, math.sin(half), math.cos(half)]
        x, y, z, w = end.orientation
        quat = [
            qz[3] * x + qz[0] * w + qz[1] * z - qz[2] * y,
            qz[3] * y - qz[0] * z + qz[1] * w + qz[2] * x,
            qz[3] * z + qz[0] * y - qz[1] * x + qz[2] * w,
            qz[3] * w - qz[0] * x - qz[1] * y - qz[2] * z,
        ]
    else:
        quat = list(end.orientation)
    return Pose(pos, quat)


def program_attachment(sc: Sidecar) -> List[Action]:
    """init → manipulation stance → grasp the target → lift → [the recorded second base pose: navigate / transition to the
    ``attach`` stance with the male part in hand] → reorient_in_hand toward the attach orientation while moving to the
    pre-attach pose (recorded SLERP waypoints, end = pre-attach 10 cm back along the female insertion axis) → [brace
    the partner on its face opposite to the insertion with the other gripper] → insert along the axis until attached →
    release → retreat (both, when braced).

    Owner decision 2026-09-16 17:10 item 2 (spec §6.8): **attachment keeps the brace**. When the attach pose and the
    brace are not both reachable from the grasp stance the sampler records a **second base pose** (tabletop ``58afae2``:
    ``stances.attach`` = lateral trunk move + torso adjust, ``paths[] manipulation -> attach`` with ``carried`` = the
    male part and / or ``transitions[]`` with ``holding`` = the male part, ``grasps[0].attach_stance``, the SLERP
    waypoints from the carried lift ``grasps[0].lift``); the oracle drives the recorded move with the held object in
    the collision world and the arms frozen at their joints (no re-planning of the stance) and reorients / braces /
    attaches from there. Only when the joint placement has no cell after that is the instance brace-less
    (``summary.brace: none``): one-arm attach, no brace leg."""
    g = _single_grasp(sc, "pre_grasp", "carry_height", "waypoints", "pre_place", "place", "place_on")
    brace = sc.contacts.brace if sc.contacts is not None else None
    if brace is not None:
        if brace.arm != _other_arm(g.arm):
            raise SidecarError(f"contacts.brace.arm: the target is grasped with the {g.arm} arm, the brace must use the other")
        if brace.object != g.place_on:
            raise SidecarError("contacts.brace.object: the brace must hold the attachment partner (grasps[0].place_on)")
    work = _attach_stance(sc, g)
    grasp, lift = _pick(g)
    actions = _travel(sc, "init", "manipulation") + [grasp, lift]
    if work != "manipulation":
        travel = _travel(sc, "manipulation", work, carried=g.object)
        if not travel:
            raise SidecarError(f"attachment: the attach stance {work!r} is reached by neither a recorded path nor a transition from 'manipulation'")
        actions += travel
        start = _carried_lift_pose(sc, g, "manipulation", work)
        reorient = Reorient(mode="slerp", arm=g.arm, object=g.object, start_pose=start, waypoints=g.waypoints)
    else:
        reorient = Reorient(mode="slerp", arm=g.arm, object=g.object, after=lift, waypoints=g.waypoints)
    base_yaw = sc.stance(work).base_pose2d[2]
    place = Place(kind="attach", arm=g.arm, object=g.object, place=g.place, pre_place=g.pre_place, on=g.place_on)
    actions.append(reorient)
    if brace is None:
        return actions + [place, Retreat(arm=g.arm, object=g.object, after=place, distance=RETREAT_LITTLE)]
    b = Brace(contact=brace, base_yaw=base_yaw)
    return actions + [
        b,
        place,
        Retreat(arm=g.arm, object=g.object, after=place, distance=RETREAT_LITTLE),
        Retreat(arm=brace.arm, object=brace.object, after=b, release=True),
    ]


def program_alignment_wall(sc: Sidecar) -> List[Action]:
    """init → manipulation stance → close both grippers → brace at the recorded stop (below_high / above_low /
    side_high) → push at E_low along the arc (δ/2, goal) with the arm only until the upright orientation → retreat
    both. A target without a low corner (round, owner 2026-10-11) records no brace: ``transition → push → retreat``,
    the hinge resistance holds the upright pose."""
    push = _push_contact(sc)
    brace = sc.contacts.brace if sc.contacts is not None else None
    if push.mid is None:
        raise SidecarError("contacts.push.mid: the wall alignment push is an arc (δ/2 waypoint required)")
    if brace is not None and push.object != brace.object:
        raise SidecarError("contacts: push and brace must act on the same target")
    base_yaw = sc.stance("manipulation").base_pose2d[2]
    if brace is None:
        p = Push(contact=push, base_yaw=base_yaw)
        return _travel(sc, "init", "manipulation") + [p, Retreat(arm=push.arm, object=push.object, after=p)]
    return _travel(sc, "init", "manipulation") + _brace_push_retreat(sc, brace, push, base_yaw)


def program_alignment_axial(sc: Sidecar) -> List[Action]:
    """init → manipulation stance → close the gripper(s) → [left gripper brace at the recorded stop edge, when the
    sidecar records one] → right gripper push at the long-axis end along the arc to the aligned yaw → retreat.

    Axial is **push-only** (owner decision 2026-09-16 08:10): the sampler records ``contacts.brace`` only when a joint
    two-arm placement exists; without it the program is ``transition → push → retreat`` and the oracle's push closure
    (hold at the goal, residual re-push) is the stop."""
    push = _push_contact(sc)
    brace = sc.contacts.brace if sc.contacts is not None else None
    if push.arm != "right" or (brace is not None and brace.arm != "left"):
        raise SidecarError("alignment-axial: the right gripper pushes and the left gripper braces (spec §6.10)")
    if push.mid is None:
        raise SidecarError("contacts.push.mid: the axial push is an arc (δ/2 waypoint required)")
    if brace is not None and push.object != brace.object:
        raise SidecarError("contacts: push and brace must act on the same target")
    base_yaw = sc.stance("manipulation").base_pose2d[2]
    if brace is None:
        p = Push(contact=push, base_yaw=base_yaw)
        return _travel(sc, "init", "manipulation") + [p, Retreat(arm=push.arm, object=push.object, after=p)]
    return _travel(sc, "init", "manipulation") + _brace_push_retreat(sc, brace, push, base_yaw)


#: the push-under push's retreat (owner 2026-10-13 ~04:30 item 1a): back along the reversed push direction at the push height first (the
#: corridor the push just swept, at most its length), then up - straight up from the push goal, the wrist camera of the 'pitch 0-' turn
#: (7.6 cm above the eef, up to 7.5 cm toward the bed) met bed_rrcvaq_0's edge one step after the retreat (under_large#01 c3_k0)
PUSH_UNDER_RETREAT_SLIDE_M = 0.10
PUSH_UNDER_RETREAT_LIFT_M = 0.10


def push_under_retreat(push: PushContact) -> Dict[str, Any]:
    """The ``Retreat`` keywords of a push-under push's retreat, from the recorded contact: ``distance`` =
    :data:`PUSH_UNDER_RETREAT_SLIDE_M` (at most the blade's travel ``init -> goal``) along ``slide_dir`` = the push direction reversed in
    the floor plane, then ``lift_after_m`` = :data:`PUSH_UNDER_RETREAT_LIFT_M`."""
    dx, dy = float(push.direction[0]), float(push.direction[1])
    n = math.hypot(dx, dy)
    if n < 1e-6:
        raise SidecarError(f"contacts.push.direction: a push-under push runs along the floor, got {list(push.direction)!r}")
    swept = math.hypot(float(push.goal.position[0]) - float(push.init.position[0]), float(push.goal.position[1]) - float(push.init.position[1]))
    slide = min(PUSH_UNDER_RETREAT_SLIDE_M, swept) if swept > 1e-3 else PUSH_UNDER_RETREAT_SLIDE_M
    return {"distance": slide, "slide_dir": [-dx / n, -dy / n, 0.0], "lift_after_m": PUSH_UNDER_RETREAT_LIFT_M}


def program_iors_large(sc: Sidecar, relation: str = "ontop") -> List[Action]:
    """init A → manipulation B (transition) → grasp → lift (orientation kept) → navigate_to placement C with the object
    attached (recorded path; the retreat transition before lowering the trunk toward the floor band for under, where
    recorded) → place at the goal cell → retreat.

    **Push-under** (owner decision 2026-09-16 item 7, under large-scale): when ``grasps[0].place_kind == "drop"`` the
    object is dropped on the floor outside the partner's footprint edge and pushed under it with the closed gripper
    along the recorded straight floor push ``contacts.push`` (``kind: "push_under"``, from stance C): ``… → place(drop)
    → retreat → push → retreat``. The push may touch the floor and the target, never the partner; ``under`` is read in
    the epilogue after the push. The push's retreat backs out along the push at its height, then lifts (:func:`push_under_retreat`).

    inside / touching / nextto (registered 2026-10-12 ahead of their modes): each relation's own program,
    :func:`program_iors_large_inside` / :func:`program_iors_large_touching` / :func:`program_iors_large_nextto`."""
    mode = {"inside": program_iors_large_inside, "touching": program_iors_large_touching, "nextto": program_iors_large_nextto}.get(relation)
    if mode is not None:
        return mode(sc)
    g = _single_grasp(sc, "pre_grasp", "carry_height", "place", "place_on")
    if sc.path(_grasp_stance(g), "placement") is None:
        raise SidecarError(f"paths: IORS large-scale needs the recorded path from {_grasp_stance(g)!r} to 'placement'")
    pick = _pick_actions(sc, g)
    lift = pick[-1]
    # owner 2026-09-20 05:20: ``grasps[0].waypoints`` = the in-hand line from the lift to the CARRY pose (a compact pose in front
    # of the chest) - the object is carried there through the transitions and the drive (the sampler checked the legs with the
    # arm at that pose); without waypoints the object is carried at the lift configuration as before
    carry = [Reorient(mode="line", arm=g.arm, object=g.object, after=lift, waypoints=g.waypoints)] if g.waypoints else []
    head = _travel(sc, "init", "manipulation") + pick + carry + _travel(sc, _grasp_stance(g), "placement", carried=g.object)
    drop = str(g.extra.get("place_kind") or "support") == "drop"
    if not drop:
        return head + _place_and_retreat(g, "support", _IORS_RELATION_WORDS[relation])
    if relation != "under":
        raise SidecarError(f"grasps[0].place_kind 'drop' is the push-under release of the under relation, not of {relation!r}")
    push = sc.contacts.push if sc.contacts is not None else None
    if push is None or push.kind != "push_under":
        raise SidecarError("contacts.push: a record with kind 'push_under' is required when grasps[0].place_kind == 'drop' (push-under)")
    if push.object != g.object:
        raise SidecarError(f"contacts.push.object {push.object!r} is not the dropped target {g.object!r}")
    if push.arm != g.arm:
        raise SidecarError(f"contacts.push.arm {push.arm!r} differs from the grasp arm {g.arm!r} (the same closed gripper pushes)")
    partner = push.extra.get("partner")
    avoid = (partner,) if isinstance(partner, str) and partner else ()
    place = Place(kind="drop", arm=g.arm, object=g.object, place=g.place, pre_place=g.pre_place, on=g.place_on, relation="under", avoid=avoid)
    # owner 2026-09-20 08:30 (drop first, then crouch): when the sidecar records a separate "push" stance (the deep crouch at the
    # drop stance's base pose), the robot drops from "placement", retreats, takes the crouch with empty hands (the recorded
    # trunk-only transition) and pushes from there; otherwise drop and push share the placement stance as before
    push_stance = "push" if "push" in sc.stances else "placement"
    base_yaw = sc.stance(push_stance).base_pose2d[2]
    p = Push(contact=push, base_yaw=base_yaw, touches=(g.place_on,) if g.place_on else (), avoid=avoid)
    to_push = _travel(sc, "placement", "push") if push_stance == "push" else []
    return (head + [place, Retreat(arm=g.arm, object=g.object, after=place, distance=RETREAT_LITTLE)] + to_push
            + [p, Retreat(arm=push.arm, object=push.object, after=p, **push_under_retreat(push))])


def program_iors_large_inside(sc: Sidecar) -> List[Action]:
    """IORS inside, large-scale (``program_iors_large`` with relation ``inside``; owner 2026-10-12 ~16:05 item 1): init A → manipulation B
    → grasp → lift [→ carry pose] → navigate_to placement C with the object attached → place with kind ``drop`` (the small-scale inside
    ending: the release above the partner's interior surface, the container stays an obstacle) → retreat. The target ends inside the
    partner (``inside`` read in the epilogue). The overhang pick (``contacts.push_pick``, owner 2026-10-13 ~03:15 item 4): push -> retreat
    before the grasp (:func:`_pick_actions`)."""
    g = _single_grasp(sc, "pre_grasp", "carry_height", "place", "place_on")
    if sc.path(_grasp_stance(g), "placement") is None:
        raise SidecarError(f"paths: IORS large-scale needs the recorded path from {_grasp_stance(g)!r} to 'placement'")
    pick = _pick_actions(sc, g)
    lift = pick[-1]
    carry = [Reorient(mode="line", arm=g.arm, object=g.object, after=lift, waypoints=g.waypoints)] if g.waypoints else []
    head = _travel(sc, "init", "manipulation") + pick + carry + _travel(sc, _grasp_stance(g), "placement", carried=g.object)
    return head + _place_and_retreat(g, "drop", _IORS_RELATION_WORDS["inside"])


def program_iors_large_touching(sc: Sidecar) -> List[Action]:
    """IORS touching, large-scale (``program_iors_large`` with relation ``touching``): the program of the touching mode, written
    with that mode; until then a sidecar of the subtask has no program. The target ends touching the partner (``touching`` read
    in the epilogue)."""
    return _iors_large_touching_program(sc)


def program_iors_large_nextto(sc: Sidecar) -> List[Action]:
    """IORS nextto, large-scale (``program_iors_large`` with relation ``nextto``): the program of the nextto mode, written with
    that mode (owner 2026-10-12): init A → push stance ``manipulation`` (recorded path / transition) → ``base_push`` (``contacts.push``
    with ``kind: "base_push"``: the closed hand on the floor-standing target's push face, the base drives it along the recorded line
    ``base_start → base_end`` and stops once the partner is NextTo it) → retreat from where the push leaves the hand; no grasp. The base
    ends at the recorded stance ``contacts.push.end_stance`` (default ``placement``: the push stance moved ``distance_m`` along the line,
    same trunk and arms). The target ends next to the partner (``nextto`` read in the epilogue).

    Two hands (owner 2026-10-12 ~20:17 item C; ``contacts.push.second_hand``): the ``base_push`` carries both arms (``BasePush.second``)
    and each hand retreats from where the push leaves it, the first hand first."""
    push = sc.contacts.push if sc.contacts is not None else None
    if push is None or push.kind != BASE_PUSH_KIND:
        raise SidecarError("contacts.push: IORS nextto large-scale records the base push (kind 'base_push')")
    if sc.grasps:
        raise SidecarError("grasps: the nextto base push grasps nothing")
    end = str(push.extra.get("end_stance") or "placement")
    if end not in sc.stances:
        raise SidecarError(f"stances: the base push ends at the recorded stance {end!r}")
    partner = push.extra.get("partner")
    second = _second_hand(push)
    p = BasePush(contact=push, base_yaw=sc.stance("manipulation").base_pose2d[2], partner=partner if isinstance(partner, str) and partner else None, end_stance=end,
                 second=second)
    retreats = [Retreat(arm=c.arm, object=push.object, after=p.hand(c.arm)) for c in p.contacts]
    return _travel(sc, "init", "manipulation") + [p] + retreats


def _second_hand(push: PushContact) -> Optional[PushContact]:
    """The other hand of a two-hand base push (``contacts.push.second_hand``: its ``arm`` and ``pre`` / ``init`` / ``goal`` poses, the
    push's direction and distance), None for one hand."""
    rec = push.extra.get("second_hand")
    if rec is None:
        return None
    if not isinstance(rec, dict):
        raise SidecarError("contacts.push.second_hand: a two-hand base push records the other hand as a dict")
    if rec.get("arm") != _other_arm(push.arm):
        raise SidecarError(f"contacts.push.second_hand.arm: the first hand is the {push.arm} one, the second must be {_other_arm(push.arm)!r}, got {rec.get('arm')!r}")
    return PushContact.from_dict({**rec, "object": push.object, "direction": list(push.direction), "distance_m": push.distance_m}, "contacts.push.second_hand")


def program_iors_small(sc: Sidecar, relation: str = "ontop") -> List[Action]:
    """init → manipulation stance → grasp → lift (orientation kept) → place at the recorded goal pose (centre of the
    partner; nearest satisfying position for nextto; inside: drop into the fillable volume) → retreat. A nextto place with
    ``grasps[0].raised_release`` (the sampler's ``generation.raised_release``, owner 2026-10-14 ~01:51: the place raised until the
    fully open jaw clears the partner, the target let go at the open command drops onto the table) must not touch the partner
    (``Place.avoid``)."""
    g = _single_grasp(sc, "pre_grasp", "carry_height", "place", "place_on")
    grasp, lift = _pick(g)
    kind = "drop" if relation == "inside" else "support"
    word = _IORS_RELATION_WORDS["under_small" if relation == "under" else relation]
    avoid = (g.place_on,) if relation == "nextto" and g.place_on and g.extra.get("raised_release") else ()
    return _travel(sc, "init", "manipulation") + [grasp, lift] + _place_and_retreat(g, kind, word, avoid=avoid)


def program_iors_touching(sc: Sidecar) -> List[Action]:
    """as nextto to the recorded nextto position, then [brace the partner's far side with the left gripper, when the
    sidecar records one] and push the target toward the partner with the right gripper until touching → retreat.

    Touching = **brace optional** (owner decision 2026-09-16 17:10, the axial rule of 08:10): the sampler records
    ``contacts.brace`` only when the A1 joint two-arm placement admits the partner position and a two-arm candidate
    passes the exact IK (core ``6de8803``: ``summary.brace.present``); without it the instance is push-only —
    ``transition → grasp → lift → carry → place (nextto position) → retreat → push → retreat`` — and the oracle's push
    closure (hold at the goal, residual re-push, ``execute_push``) is the stop, as for ``program_alignment_axial``.

    Shelf layout (``contacts.push.layout`` = :data:`TOUCHING_SHELF_LAYOUT`, touching#03): the same push-only chain with the
    grasping hand as the push arm (the left hand pushing rightward toward a partner on the right board); a brace is refused."""
    g = _single_grasp(sc, "pre_grasp", "carry_height", "place", "place_on")
    push = _push_contact(sc)
    brace = sc.contacts.brace if sc.contacts is not None else None
    if push.extra.get("layout") == TOUCHING_SHELF_LAYOUT:
        if brace is not None or push.arm != g.arm:
            raise SidecarError("touching (shelf layout): the grasping hand pushes the target and nothing braces the partner")
    elif push.arm != "right" or (brace is not None and brace.arm != "left"):
        raise SidecarError("touching: the right gripper pushes the target and the left gripper braces the partner (spec §6.11)")
    if push.object != g.object:
        raise SidecarError("contacts.push.object: the push must act on the placed target")
    if brace is not None and brace.object != g.place_on:
        raise SidecarError("contacts.brace.object: the brace must hold the partner (grasps[0].place_on)")
    # owner 2026-09-18 16:32: a recorded second base pose ``push`` (the base moves along the edge with the box in hand after
    # the lift - ``paths[].carried`` - so the place and the push happen where the grasp did relative to the robot)
    push_stance = "push" if "push" in sc.stances else "manipulation"
    if push_stance == "push":
        grasp, lift = _pick(g)
        head = (_travel(sc, "init", "manipulation") + [grasp, lift] + _travel(sc, "manipulation", "push", carried=g.object)
                + _place_and_retreat(g, "support", _IORS_RELATION_WORDS["touching"]))
    else:
        head = program_iors_small(sc, "nextto")
    base_yaw = sc.stance(push_stance).base_pose2d[2]
    # the release opens the gripper fully beside the partner and shoves it 5-6 mm away from the target (release validation
    # 2026-10-03): the push's goal follows the partner, and so does the brace (the shove exceeds the stop's 5 mm clearance)
    if brace is None:
        p = Push(contact=push, base_yaw=base_yaw, follow=g.place_on)
        return head + [p, Retreat(arm=push.arm, object=push.object, after=p)]
    return head + _brace_push_retreat(sc, brace, push, base_yaw, follow_braced=True)


def _iors_large_touching_program(sc: Sidecar) -> List[Action]:
    """IORS touching, large-scale (:func:`program_iors_large_touching`; owner 2026-10-12 ~16:05): the push-under chain against the
    partner's face - init A → manipulation B (transition) → grasp → lift → [carry pose] → navigate_to the drop stance C with the
    object → place(drop on the floor in front of the face) → retreat → [the deep push stance: trunk-only transition] → push along the
    floor → retreat. ``contacts.push`` (``kind: "push_touching"``) names the ``partner``: the push may touch it (``touches``, no
    ``avoid``) and follows it (``Push.follow``: the pad stops when the target reaches its goal, the touching pose - ``execute_push``'s
    stop rule - moved with the partner if the partner moved); ``touching`` is read in the epilogue."""
    g = _single_grasp(sc, "pre_grasp", "carry_height", "place", "place_on")
    if sc.path(_grasp_stance(g), "placement") is None:
        raise SidecarError(f"paths: IORS large-scale needs the recorded path from {_grasp_stance(g)!r} to 'placement'")
    if str(g.extra.get("place_kind") or "support") != "drop":
        raise SidecarError("grasps[0].place_kind: IORS touching (large-scale) drops the target on the floor before the push ('drop')")
    push = sc.contacts.push if sc.contacts is not None else None
    if push is None or push.kind != "push_touching":
        raise SidecarError("contacts.push: a record with kind 'push_touching' is required (IORS touching, large-scale)")
    if push.object != g.object:
        raise SidecarError(f"contacts.push.object {push.object!r} is not the dropped target {g.object!r}")
    if push.arm != g.arm:
        raise SidecarError(f"contacts.push.arm {push.arm!r} differs from the grasp arm {g.arm!r} (the same closed gripper pushes)")
    partner = push.extra.get("partner")
    if not isinstance(partner, str) or not partner:
        raise SidecarError("contacts.push.partner: the instance the target is pushed against is required (IORS touching, large-scale)")
    pick = _pick_actions(sc, g)
    lift = pick[-1]
    carry = [Reorient(mode="line", arm=g.arm, object=g.object, after=lift, waypoints=g.waypoints)] if g.waypoints else []
    head = _travel(sc, "init", "manipulation") + pick + carry + _travel(sc, _grasp_stance(g), "placement", carried=g.object)
    place = Place(kind="drop", arm=g.arm, object=g.object, place=g.place, pre_place=g.pre_place, on=g.place_on, relation=_IORS_RELATION_WORDS["touching"])
    push_stance = "push" if "push" in sc.stances else "placement"
    base_yaw = sc.stance(push_stance).base_pose2d[2]
    p = Push(contact=push, base_yaw=base_yaw, touches=tuple(n for n in (g.place_on, partner) if n), follow=partner)
    to_push = _travel(sc, "placement", "push") if push_stance == "push" else []
    return head + [place, Retreat(arm=g.arm, object=g.object, after=place, distance=RETREAT_LITTLE)] + to_push + [p, Retreat(arm=push.arm, object=push.object, after=p)]


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
    "alignment-world_vertical": (program_alignment_wall, {}),
    "alignment-axial": (program_alignment_axial, {}),
    "inter_object_relation_satisfaction-ontop_large_scale": (program_iors_large, {"relation": "ontop"}),
    "inter_object_relation_satisfaction-under_large_scale": (program_iors_large, {"relation": "under"}),
    "inter_object_relation_satisfaction-inside_large_scale": (program_iors_large, {"relation": "inside"}),
    "inter_object_relation_satisfaction-touching_large_scale": (program_iors_large, {"relation": "touching"}),
    "inter_object_relation_satisfaction-nextto_large_scale": (program_iors_large, {"relation": "nextto"}),
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


def expand(variant: str, sidecar: Sidecar) -> List[Action]:
    """The action list of ``variant``'s program for ``sidecar`` (§3.7). ``variant`` must be the sidecar's own variant (or
    resolve to the same subtask); the executor's epilogue (§3.8) is not an action."""
    subtask = subtask_of_variant(variant)
    if variant != sidecar.variant and subtask_of_variant(sidecar.variant) != subtask:
        raise SidecarError(f"variant {variant!r} is not the sidecar's variant {sidecar.variant!r}")
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
