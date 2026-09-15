"""Concrete BDDL problem files for the seven iSpatialGym activities.

The shipped ``ispatialgym/assets/bddl/activity_definitions/*/problemN.bddl`` are structural templates with
representative synsets. The sampler binds *roles* to concrete objects and regenerates the problem with the
concrete synsets, keeping the structure documented in ``docs/task_bddl_formats.md``. The result is parsed and
compiled with :mod:`ispatialgym.activity.problem` before it is written, so a malformed instance cannot be saved.

Roles per activity (``*`` = list role):

* camera_relocalization: floor [, obstacles* (obstructed variant: alias ``obstructed``, group ``ontop`` the floor)]
* object_scaling / mental_rotation: target, floor
* disembedding: target, distractors*, tray_source, tray_dest, table, floor
* configuration_matching-0 (block_design): blocks*, table, floor
* configuration_matching-1/2 (articulation): artic, floor [, table]
* configuration_matching-3 (attachment): male, female, table, floor
* world_alignment-0 (table_surface): target, table, floor
* world_alignment-1 (world_vertical): target, wall, floor
* world_alignment-2 (world_horizontal): target, wall, floor
* inter_object_relation_satisfaction-N: target, partner, floor [, table (init support of the target)]
  [, partner_support (init support of the partner)]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ispatialgym.activity.problem import AGENT_INSTANCE, ISGProblem, parse_problem_string

GROUP_NAMES = {
    "distractors": "distractors_group.n.01_1",
    "blocks": "blocks_group.n.01_1",
    "attachment": "attachment_group.n.01_1",
    "obstacles": "obstacles_group.n.01_1",
}

IORS_RELATIONS = ["ontop", "under", "inside", "nextto", "touching"]


@dataclass
class Bound:
    """One concrete BDDL instance."""

    role: str
    name: str  # BDDL instance name, e.g. bottle__of__beer.n.01_1
    synset: str
    category: Optional[str] = None
    model: Optional[str] = None
    scene_object_name: Optional[str] = None  # set once the object exists in the scene
    dataset: Optional[str] = None


@dataclass
class Bindings:
    roles: Dict[str, List[Bound]] = field(default_factory=dict)
    _counters: Dict[str, int] = field(default_factory=dict)

    def add(self, role: str, synset: str, category=None, model=None, scene_object_name=None, dataset=None) -> Bound:
        k = self._counters.get(synset, 0) + 1
        self._counters[synset] = k
        b = Bound(role, f"{synset}_{k}", synset, category, model, scene_object_name, dataset)
        self.roles.setdefault(role, []).append(b)
        return b

    def one(self, role: str) -> Bound:
        lst = self.roles.get(role, [])
        if len(lst) != 1:
            raise KeyError(f"role {role!r} has {len(lst)} bindings, expected 1")
        return lst[0]

    def many(self, role: str) -> List[Bound]:
        return list(self.roles.get(role, []))

    def has(self, role: str) -> bool:
        return bool(self.roles.get(role))

    def all(self) -> List[Bound]:
        """Every distinct bound instance (a Bound may be aliased under several roles)."""
        out, seen = [], set()
        for lst in self.roles.values():
            for b in lst:
                if b.name not in seen:
                    seen.add(b.name)
                    out.append(b)
        return out

    def by_name(self) -> Dict[str, Bound]:
        return {b.name: b for b in self.all()}


# ----------------------------------------------------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------------------------------------------------


def _objects_section(bounds: Sequence[Bound]) -> List[str]:
    by_syn: Dict[str, List[str]] = {}
    for b in bounds:
        by_syn.setdefault(b.synset, []).append(b.name)
    lines = ["    (:objects"]
    for syn, names in by_syn.items():
        lines.append(f"        {' '.join(names)} - {syn}")
    lines.append(f"        {AGENT_INSTANCE} - agent.n.01")
    lines.append("    )")
    return lines


def _q(name: str) -> str:
    return "?" + name


def render_problem(
    activity: str,
    definition_id: int,
    alias: Optional[str],
    bounds: Sequence[Bound],
    groups: Dict[str, List[str]],
    init: List[str],
    goal: List[str],
    measure: List[str],
    comment: Optional[str] = None,
) -> str:
    lines = [f"(define (problem {activity}-{definition_id})", "    (:domain ispatialgym)"]
    if alias:
        lines.append(f"    (:alias {alias})")
    if comment:
        lines.append("")
        for c in comment.splitlines():
            lines.append(f"    ; {c}")
    lines.append("")
    lines.extend(_objects_section(bounds))
    if groups:
        lines.append("")
        lines.append("    (:groups")
        for g, members in groups.items():
            lines.append(f"        {' '.join(members)} - {g}")
        lines.append("    )")
    lines.append("")
    lines.append("    (:init")
    lines.extend(f"        {lit}" for lit in init)
    lines.append("    )")
    lines.append("")
    lines.append("    (:goal")
    lines.append("        (and")
    lines.extend(f"            {lit}" for lit in goal)
    lines.append("        )")
    lines.append("    )")
    if measure:
        lines.append("")
        lines.append("    (:measure")
        lines.extend(f"        {lit}" for lit in measure)
        lines.append("    )")
    lines.append(")")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------------------------
# Per-activity builders
# ----------------------------------------------------------------------------------------------------------------


def build_problem_text(activity: str, definition_id: int, b: Bindings) -> str:
    agent = AGENT_INSTANCE
    floor = b.one("floor").name
    on_floor = f"(ontop {agent} {floor})"
    if activity == "camera_relocalization":
        init, groups, alias = [on_floor], {}, None
        comment = ("Binary success: the head-camera sensor pose (TRO pose_targets[agent], link = sensor key) is within "
                   "both tolerances of the goal camera pose.")
        if b.has("obstacles"):
            # obstructed variant (docs/task_bddl_formats.md 1b): the obstacles group stands on the floor on the shortest
            # path and is never manipulated; its blocked footprints are in the TRO scene block (scene_variant)
            g = GROUP_NAMES["obstacles"]
            groups = {g: [o.name for o in b.many("obstacles")]}
            init = [f"(ontop {g} {floor})", on_floor]
            alias = "obstructed"
            comment += (" Obstructed: the obstacles group stands on the floor on the shortest path (never manipulated); "
                        "its footprints are recorded as scene_variant.blocked_footprints in the TRO.")
        return render_problem(
            activity, definition_id, alias, b.all(), groups, init,
            [f"(pose_reached {_q(agent)})", f"(pose_settled {_q(agent)})"], [],
            comment=comment,
        )
    if activity in ("object_scaling", "mental_rotation"):
        t = b.one("target").name
        return render_problem(
            activity, definition_id, None, b.all(), {}, [on_floor, f"(grasped {agent} {t})"],
            [f"(grasped {_q(agent)} {_q(t)})", f"(pose_reached {_q(t)})", f"(pose_settled {_q(t)})", f"(pose_settled {_q(agent)})"],
            [],
            comment=(
                "Binary success: the object is still held and its depth along the head-camera optical axis matches the goal "
                "(TRO pose_targets[target]: frame = head camera, position_metric = depth)."
                if activity == "object_scaling"
                else "Binary success: the object is still held and its world-frame orientation is within tolerance of the goal "
                "(TRO pose_targets[target]: orientation_metric = full)."
            ),
        )
    if activity == "disembedding":
        t, src, dst, table = b.one("target").name, b.one("tray_source").name, b.one("tray_dest").name, b.one("table").name
        dis = [d.name for d in b.many("distractors")]
        g = GROUP_NAMES["distractors"]
        init = [f"(ontop {src} {table})", f"(ontop {dst} {table})", f"(ontop {t} {src})", f"(ontop {g} {src})",
                f"(not (ontop {t} {dst}))", f"(not (ontop {g} {dst}))", on_floor]
        goal = [f"(ontop {_q(t)} {_q(dst)})", f"(not (ontop {_q(g)} {_q(dst)}))", f"(not (grasped {_q(agent)} {_q(t)}))",
                f"(pose_settled {_q(t)})", f"(pose_settled {_q(agent)})"]
        return render_problem(activity, definition_id, None, b.all(), {g: dis}, init, goal, [f"(pose_reached {_q(t)})"],
                              comment="Binary success: the target is on the destination tray, no distractor is, and the target is released.")
    if activity == "configuration_matching":
        if definition_id == 0:
            blocks = [x.name for x in b.many("blocks")]
            table = b.one("table").name
            g = GROUP_NAMES["blocks"]
            return render_problem(
                activity, definition_id, "block_design", b.all(), {g: blocks},
                [f"(ontop {g} {table})", on_floor],
                [f"(ontop {_q(g)} {_q(table)})", f"(layout_reached {_q(g)})", f"(not (grasped {_q(agent)} {_q(g)}))",
                 f"(pose_settled {_q(g)})", f"(pose_settled {_q(agent)})"],
                [f"(pose_reached {_q(g)})"],
                comment="Binary success: every block is on the table, released, and the blocks reproduce the pictured relative "
                "arrangement (TRO groups[blocks_group].layout_relations, shared block-to-slot assignment).",
            )
        if definition_id in (1, 2):
            a = b.one("artic").name
            init = [f"(not (open {a}))" if definition_id == 1 else f"(open {a})"]
            goal = [f"(open {_q(a)})" if definition_id == 1 else f"(not (open {_q(a)}))", f"(layout_reached {_q(a)})"]
            measure = []
            if b.has("table"):
                table = b.one("table").name
                init.append(f"(ontop {a} {table})")
                goal.append(f"(ontop {_q(a)} {_q(table)})")
                measure.append(f"(pose_reached {_q(a)})")
            init.append(on_floor)
            goal += [f"(not (grasped {_q(agent)} {_q(a)}))", f"(pose_settled {_q(a)})", f"(pose_settled {_q(agent)})"]
            return render_problem(activity, definition_id, "articulation", b.all(), {}, init, goal, measure,
                                  comment=("Closed -> open" if definition_id == 1 else "Open -> closed") +
                                  ". Binary success: the open guard holds, every non-fixed joint is within tolerance of its "
                                  "target (TRO articulation_targets), and the object is released.")
        if definition_id == 3:
            male, female, table = b.one("male").name, b.one("female").name, b.one("table").name
            g = GROUP_NAMES["attachment"]
            return render_problem(
                activity, definition_id, "attachment", b.all(), {g: [male, female]},
                [f"(ontop {g} {table})", f"(not (attached {male} {female}))", on_floor],
                [f"(attached {_q(male)} {_q(female)})", f"(ontop {_q(female)} {_q(table)})", f"(layout_reached {_q(g)})",
                 f"(not (grasped {_q(agent)} {_q(g)}))", f"(pose_settled {_q(g)})", f"(pose_settled {_q(agent)})"],
                [f"(pose_reached {_q(g)})"],
                comment="Binary success: the male part is attached to the female part, the female part stands on the table, "
                "the male pose in the female frame matches the pictured one (TRO groups[attachment_group], fixed assignment), "
                "and both parts are released.",
            )
        raise ValueError(f"configuration_matching has no definition {definition_id}")
    if activity == "world_alignment":
        t = b.one("target").name
        if definition_id == 0:
            table = b.one("table").name
            return render_problem(
                activity, definition_id, "table_surface", b.all(), {}, [f"(ontop {t} {table})", on_floor],
                [f"(ontop {_q(t)} {_q(table)})", f"(pose_reached {_q(t)})", f"(not (grasped {_q(agent)} {_q(t)}))",
                 f"(pose_settled {_q(t)})", f"(pose_settled {_q(agent)})"], [],
                comment="Binary success: the object is on the table, released, and its orientation is within tolerance of the "
                "goal (TRO pose_targets[target]: orientation_metric = full). A TRO revolute constraint (parent = table) keeps "
                "its position fixed.",
            )
        if definition_id == 1:
            b.one("wall")  # declared for the constraint parent
            return render_problem(
                activity, definition_id, "world_vertical", b.all(), {}, [on_floor],
                [f"(pose_reached {_q(t)})", f"(not (grasped {_q(agent)} {_q(t)}))", f"(pose_settled {_q(t)})",
                 f"(pose_settled {_q(agent)})"], [],
                comment="The object hangs on the wall through a TRO revolute constraint at its centre of mass. Binary success: "
                "the object is released and its orientation is within tolerance of the straight (world vertical) goal.",
            )
        if definition_id == 2:
            b.one("wall")  # declared for the constraint parent
            return render_problem(
                activity, definition_id, "world_horizontal", b.all(), {}, [on_floor],
                [f"(pose_reached {_q(t)})", f"(not (grasped {_q(agent)} {_q(t)}))", f"(pose_settled {_q(t)})",
                 f"(pose_settled {_q(agent)})"], [],
                comment="The object hangs on the wall through a TRO revolute constraint at its centre of mass. Binary success: "
                "the object is released and its orientation is within tolerance of the level (world horizontal) goal; "
                "models that look the same upside down carry the 180-degree flip in orientation_equivalences.",
            )
        raise ValueError(f"world_alignment has no definition {definition_id}")
    if activity == "inter_object_relation_satisfaction":
        rel = IORS_RELATIONS[definition_id]
        t, p = b.one("target").name, b.one("partner").name
        init = []
        if b.has("table"):
            init.append(f"(ontop {t} {b.one('table').name})")
        if b.has("partner_support"):
            init.append(f"(ontop {p} {b.one('partner_support').name})")
        init.append(on_floor)
        spawned_partner = b.one("partner").dataset is not None  # scene objects are bound without a dataset
        # The manipulated target is the first argument, except for ``under`` with a spawned (small) partner: the target
        # is set down centred on the partner (coaster, saucer, tray), so the *partner* ends under the target --
        # ``(under ?partner ?target)`` (docs/task_bddl_formats.md "under", iddl under_small_scale). OmniGibson
        # Under(a, b) = the ray shot up from a's top at its AABB-centre column hits b. The fixed big partner (bed, desk)
        # keeps ``(under ?target ?partner)``: the target ends on the floor under it.
        if rel == "under" and spawned_partner:
            relation = f"({rel} {_q(p)} {_q(t)})"
        else:
            relation = f"({rel} {_q(t)} {_q(p)})"
        goal = [relation, f"(not (grasped {_q(agent)} {_q(t)}))", f"(not (grasped {_q(agent)} {_q(p)}))", f"(pose_settled {_q(t)})"]
        # fixed scene partners (bed, couch) are kinematic: PoseSettled cannot read their velocities, and they never move
        if spawned_partner:
            goal.append(f"(pose_settled {_q(p)})")
        goal.append(f"(pose_settled {_q(agent)})")
        return render_problem(activity, definition_id, rel, b.all(), {}, init, goal,
                              [f"(pose_reached {_q(t)})", f"(pose_reached {_q(p)})"],
                              comment="Binary success: the pictured relation holds and both objects are released. The exact pictured "
                              "poses (TRO pose_targets) are reported only.")
    raise ValueError(f"unknown activity {activity!r}")


def build_problem(activity: str, definition_id: int, bindings: Bindings) -> ISGProblem:
    """Render, parse (validates symbols / predicates / structure) and return the concrete problem."""
    text = build_problem_text(activity, definition_id, bindings)
    problem = parse_problem_string(text, activity=activity, definition_id=definition_id)
    problem.compile()  # raises on compile errors
    problem.text = text  # type: ignore[attr-defined]
    return problem
