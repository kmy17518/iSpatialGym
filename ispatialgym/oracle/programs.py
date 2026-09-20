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

import math
import os
from dataclasses import replace
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
    block_reorients,
)
from ispatialgym.oracle.sidecar import ARMS, GraspRecord, Pose, Sidecar, SidecarError, subtask_of_alias

# ----------------------------------------------------------------------------------------------------------------
# Shared pieces
# ----------------------------------------------------------------------------------------------------------------


def _other_arm(arm: str) -> str:
    return "left" if arm == "right" else "right"


def _travel(sc: Sidecar, frm: str, to: str, carried: Optional[str] = None) -> List[Action]:
    """``Navigate`` (recorded path) and / or ``Transition`` (recorded transition) from stance ``frm`` to ``to``.

    ``carried`` = the bddl instance in the hand during the travel: a recorded path must carry exactly it
    (``paths[].carried``) and a recorded transition that names what it was swept with (``transitions[].holding``,
    core ``bd10c72``) must name it too — the sampler validated the drive-back / trunk / drive-in with that object
    attached to the frozen arm, the oracle drives it with the arms as they are (no re-planning of the stance)."""
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
        actions.append(Navigate(path=path, objects=(carried,) if carried else ()))
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
    block = replace(b, face_pair=str(rg["face_pair"]), tilt_pick_deg=float(rg["tilt_pick_deg"]), yaw_deg=float(rg["yaw_deg"]),
                    carry_height=float(rg.get("carry_height", b.carry_height)), pick=Pose.from_dict(rg["pick"], f"{where}.regrasp.pick"),
                    flip_sign=0, flip_axis=None, waypoints=None, extra={k: v for k, v in (b.extra or {}).items() if k != "table_flip"})
    return {"place": Pose.from_dict(raw["place"], f"{where}.place"),
            "pre_place": Pose.from_dict(raw["pre_place"], f"{where}.pre_place") if raw.get("pre_place") else None,
            "block": block}


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
    release pair is its own gripper-in-FOV span (QA rule (b))."""
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
        flip = table_flip_record(b)
        if flip is not None:
            mid = Place(kind="support", arm=arm, object=b.instance, place=flip["place"], pre_place=flip.get("pre_place"),
                        relation="on a free cell of the table, turned onto its side for the re-grasp")
            regrasp = Grasp(kind="cube", arm=arm, object=b.instance, block=flip["block"])
            relift = Lift(arm=arm, object=b.instance, carry_height=flip["block"].carry_height, after=regrasp)
            actions += [mid, Retreat(arm=arm, object=b.instance, after=mid, distance=RETREAT_LITTLE), regrasp, relift]
            last = relift
        elif block_reorients(b):
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
    if brace is not None:
        if brace.arm != _other_arm(arm):
            raise SidecarError(f"contacts.brace.arm: the handle arm is {arm!r}, the brace must use {_other_arm(arm)!r}")
        if brace.object != art.object:
            raise SidecarError("contacts.brace.object: the brace must act on the articulated object's base link")
    grasp = Grasp(kind="handle", arm=arm, object=art.object, articulation=art)
    follow = BaseFollow(arm=arm, articulation=art, stance=sc.stance("manipulation"), grasp_action=grasp, direction=direction)
    head = _travel(sc, "init", "manipulation")
    if brace is None:
        return head + [grasp, follow, Retreat(arm=arm, object=art.object, after=follow, release=True)]
    base_yaw = sc.stance("manipulation").base_pose2d[2]
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


def program_iors_large(sc: Sidecar, relation: str = "ontop") -> List[Action]:
    """init A → manipulation B (transition) → grasp → lift (orientation kept) → navigate_to placement C with the object
    attached (recorded path; the retreat transition before lowering the trunk toward the floor band for under, where
    recorded) → place at the goal cell → retreat.

    **Push-under** (owner decision 2026-09-16 item 7, under large-scale): when ``grasps[0].place_kind == "drop"`` the
    object is dropped on the floor outside the partner's footprint edge and pushed under it with the closed gripper
    along the recorded straight floor push ``contacts.push`` (``kind: "push_under"``, from stance C): ``… → place(drop)
    → retreat → push → retreat``. The push may touch the floor and the target, never the partner; ``under`` is read in
    the epilogue after the push."""
    g = _single_grasp(sc, "pre_grasp", "carry_height", "place", "place_on")
    if sc.path("manipulation", "placement") is None:
        raise SidecarError("paths: IORS large-scale needs the recorded path from 'manipulation' to 'placement'")
    grasp, lift = _pick(g)
    # owner 2026-09-20 05:20: ``grasps[0].waypoints`` = the in-hand line from the lift to the CARRY pose (a compact pose in front
    # of the chest) - the object is carried there through the transitions and the drive (the sampler checked the legs with the
    # arm at that pose); without waypoints the object is carried at the lift configuration as before
    carry = [Reorient(mode="line", arm=g.arm, object=g.object, after=lift, waypoints=g.waypoints)] if g.waypoints else []
    head = _travel(sc, "init", "manipulation") + [grasp, lift] + carry + _travel(sc, "manipulation", "placement", carried=g.object)
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
    base_yaw = sc.stance("placement").base_pose2d[2]
    p = Push(contact=push, base_yaw=base_yaw, touches=(g.place_on,) if g.place_on else (), avoid=avoid)
    return head + [place, Retreat(arm=g.arm, object=g.object, after=place, distance=RETREAT_LITTLE), p, Retreat(arm=push.arm, object=push.object, after=p)]


def program_iors_small(sc: Sidecar, relation: str = "ontop") -> List[Action]:
    """init → manipulation stance → grasp → lift (orientation kept) → place at the recorded goal pose (centre of the
    partner; nearest satisfying position for nextto; inside: drop into the fillable volume) → retreat."""
    g = _single_grasp(sc, "pre_grasp", "carry_height", "place", "place_on")
    grasp, lift = _pick(g)
    kind = "drop" if relation == "inside" else "support"
    word = _IORS_RELATION_WORDS["under_small" if relation == "under" else relation]
    return _travel(sc, "init", "manipulation") + [grasp, lift] + _place_and_retreat(g, kind, word)


def program_iors_touching(sc: Sidecar) -> List[Action]:
    """as nextto to the recorded nextto position, then [brace the partner's far side with the left gripper, when the
    sidecar records one] and push the target toward the partner with the right gripper until touching → retreat.

    Touching = **brace optional** (owner decision 2026-09-16 17:10, the axial rule of 08:10): the sampler records
    ``contacts.brace`` only when the A1 joint two-arm placement admits the partner position and a two-arm candidate
    passes the exact IK (core ``6de8803``: ``summary.brace.present``); without it the instance is push-only —
    ``transition → grasp → lift → carry → place (nextto position) → retreat → push → retreat`` — and the oracle's push
    closure (hold at the goal, residual re-push, ``execute_push``) is the stop, as for ``program_alignment_axial``."""
    g = _single_grasp(sc, "pre_grasp", "carry_height", "place", "place_on")
    push = _push_contact(sc)
    brace = sc.contacts.brace if sc.contacts is not None else None
    if push.arm != "right" or (brace is not None and brace.arm != "left"):
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
    if brace is None:
        p = Push(contact=push, base_yaw=base_yaw)
        return head + [p, Retreat(arm=push.arm, object=push.object, after=p)]
    return head + _brace_push_retreat(sc, brace, push, base_yaw)


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
