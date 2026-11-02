"""Parsing, validation, group expansion and compilation of iSpatialGym BDDL problems.

An iSpatialGym problem is a stock BDDL problem file in the ``ispatialgym`` domain with three extra sections:

* ``(:variant <variant>)`` -- optional, directly after ``:domain``. Recorded as ``task_variant``.
* ``(:groups MEMBER MEMBER ... - GROUP_NAME ...)`` -- typed-list shaped like ``:objects``. Group names have the
  instance-name shape (``blocks_group.n.01_1``) but are not synsets: nothing is loaded for them and they must not
  be declared in ``:objects``.
* ``(:measure (pose_reached ?x) (layout_reached ?g) ...)`` -- after ``:goal``. The measured targets: the listed
  ``pose_reached`` / ``layout_reached`` atoms are configured from the TRO, their errors are reported, and geometric
  success requires every one of them within its tolerance; symbolic success is the goal alone. An atom may also
  appear in the goal, where it gates symbolic success as well (``is_gating``: in ``:goal``).

Groups are expanded on the parsed nested lists before compilation:

=========================================================  ===============================================
Literal                                                    Expansion
=========================================================  ===============================================
``(P g x)`` / ``(P x g)``, pointwise ``P``                 conjunction over members
``(not (P g x))``                                          conjunction over members of ``(not (P m x))``
``(layout_reached g)``, ``(pose_reached g)``               aggregate: kept, evaluated once on the group
group inside ``forall`` / ``exists`` / ``forpairs`` ...    rejected (v1)
=========================================================  ===============================================

The result is compiled with the stock :mod:`bddl.condition_evaluation` machinery, with the group symbols added to
the object scope so the aggregate atoms resolve to a nonphysical group entity at runtime.

Activity names and definition ids
---------------------------------

Problem ``N`` of activity ``A`` lives in ``activity_definitions/A/problemN.bddl`` and is named ``A-N``. A retired
activity name (:data:`LEGACY_ACTIVITIES`, e.g. ``world_alignment`` -> ``alignment``) is accepted wherever an activity
is named -- in the problem name, as the directory of a legacy root, in ``load_problem`` / ``problem_path`` and in
the TRO ``activity`` field -- and compares equal to its canonical name (:func:`canonical_activity`).

``activity_definitions/A/problems.json`` is the optional variant <-> definition id registry written by the v2 grounding
(``ground_isg``): :func:`load_problem_registry`, :func:`variant_of`, :func:`id_of_variant`,
:func:`validate_problem_registry`. Without a registry the variants are read from the problem files themselves.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field
from functools import cached_property
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence

from bddl.condition_evaluation import (
    HEAD,
    compile_state,
    create_scope,
    evaluate_state,
    get_ground_state_options,
)
from bddl.parsing import gen_natural_language_conditions, parse_domain, scan_tokens

from ispatialgym import ASSETS_ROOT

ACTIVITY_DEFINITIONS_ROOT = os.path.join(ASSETS_ROOT, "bddl", "activity_definitions")
DOMAIN_NAME = "ispatialgym"
AGENT_INSTANCE = "agent.n.01_1"

#: Retired activity names -> canonical activity names. A retired name is accepted everywhere an activity is named
#: (problem names, directories of legacy roots, the TRO ``activity`` field) and is equivalent to its canonical name;
#: the v1 instances under ``runlogs/runpod/samp`` were sampled as ``world_alignment``.
LEGACY_ACTIVITIES: Dict[str, str] = {"world_alignment": "alignment"}
#: Section names of problem files written before the rename (the sampler's outputs keep their own BDDL copies).
LEGACY_SECTIONS: Dict[str, str] = {":alias": ":variant"}  # legacy spelling
#: Subtask (iddl) ids renamed on 2026-10-14: the small-scale IORS tasks carry their scale, as ontop / under do. A variant, a
#: variant template or an instance id written before (the eval85 runs, their BDDL copies, sidecars, template metadata and
#: TROs) reads as its current name (:func:`canonical_variant`).
LEGACY_SUBTASK_IDS: Dict[str, str] = {
    "inter_object_relation_satisfaction-inside": "inter_object_relation_satisfaction-inside_small_scale",  # legacy spelling
    "inter_object_relation_satisfaction-nextto": "inter_object_relation_satisfaction-nextto_small_scale",  # legacy spelling
    "inter_object_relation_satisfaction-touching": "inter_object_relation_satisfaction-touching_small_scale",  # legacy spelling
}

#: Optional variant <-> definition id registry of an activity directory (``ground_isg`` allocates ids per variant).
PROBLEM_REGISTRY_FILE = "problems.json"

#: Group symbols by iddl group role (``docs/task_bddl_formats.md`` "Object names") and the v1 spellings that the
#: shipped templates / ``sampling.bddl_writer`` still use. Informational: the parser accepts any group symbol of the
#: instance-name shape and nothing in the evaluator keys on these names.
GROUP_SYMBOLS: Dict[str, str] = {
    "targets": "targets_group.n.01_1",
    "distractors": "distractors_group.n.01_1",
    "assembly": "assembly_group.n.01_1",
    "obstacles": "obstacles_group.n.01_1",
}
LEGACY_GROUP_SYMBOLS: Dict[str, str] = {
    "blocks_group.n.01_1": GROUP_SYMBOLS["targets"],
    "attachment_group.n.01_1": GROUP_SYMBOLS["assembly"],
}

#: Predicates evaluated once on a group (shared member-to-slot assignment) instead of being expanded per member.
AGGREGATE_PREDICATES = frozenset({"layout_reached", "pose_reached"})
#: Predicates allowed in the non-gating ``(:measure ...)`` section.
MEASURE_PREDICATES = AGGREGATE_PREDICATES
SETTLE_PREDICATE = "pose_settled"
#: Evaluator-configured predicates that must not appear in ``:init`` (targets come from the TRO, not from BDDL).
FORBIDDEN_IN_INIT = frozenset({"pose_reached", "pose_settled", "layout_reached"})

LOGICAL_TOKENS = frozenset({"and", "or", "not", "imply", "forall", "exists", "forn", "forpairs", "fornpairs"})
QUANTIFIER_TOKENS = frozenset({"forall", "exists", "forn", "forpairs", "fornpairs"})

#: ``<synset>_<n>``; a synset word may contain single hyphens, as BEHAVIOR's do (``deep-freeze.n.02``, ``mock-up.n.01``)
INSTANCE_RE = re.compile(r"^[a-z0-9_]+(?:-[a-z0-9_]+)*\.n\.\d{2}_\d+$")
#: ``<family>-<subtask>-<category slots>`` (v2, e.g. ``inter_object_relation_satisfaction-ontop_small_scale-apple-
#: plate-breakfast_table``) or a bare v1 variant (``table_surface``): segments of lowercase letters, digits and ``_``
#: joined by single ``-``; the variant starts with a letter and has no empty segment.
VARIANT_RE = re.compile(r"^[a-z][a-z0-9_]*(?:-[a-z0-9_]+)*$")
PROBLEM_FILE_RE = re.compile(r"^problem(\d+)\.bddl$")


class ISGProblemError(ValueError):
    """Raised for a malformed iSpatialGym problem file."""


# ----------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------


def synset_of(instance: str) -> str:
    """``mug.n.04_1`` -> ``mug.n.04``."""
    return instance.rsplit("_", 1)[0]


def is_agent_instance(instance: str) -> bool:
    return instance.startswith("agent.n.")


def canonical_activity(activity: str) -> str:
    """Canonical name of ``activity``: ``world_alignment`` -> ``alignment``; canonical names map to themselves."""
    return LEGACY_ACTIVITIES.get(activity, activity)


def activity_names(activity: str) -> List[str]:
    """Every accepted spelling of ``activity``: the canonical name first, then its retired names."""
    canonical = canonical_activity(activity)
    return [canonical] + [old for old, new in LEGACY_ACTIVITIES.items() if new == canonical]


def canonical_variant(variant):
    """``variant`` (a subtask id, a variant, a variant template or an instance id starting with one) under its current
    subtask id (:data:`LEGACY_SUBTASK_IDS`): ``inter_object_relation_satisfaction-touching-a-b-c`` (legacy spelling) ->
    ``inter_object_relation_satisfaction-touching_small_scale-a-b-c``; anything else (other types included) unchanged."""
    if not isinstance(variant, str):
        return variant
    for old, new in LEGACY_SUBTASK_IDS.items():
        if variant == old or variant.startswith(old + "-"):
            return new + variant[len(old):]
    return variant


def same_activity(a: str, b: str) -> bool:
    """Whether two activity names denote the same activity (retired names equal their canonical name)."""
    return canonical_activity(a) == canonical_activity(b)


def split_problem_name(problem_name: str) -> "tuple[str, int]":
    """``alignment-0`` -> ``("alignment", 0)`` (the activity part is returned as written)."""
    if not isinstance(problem_name, str):
        raise ISGProblemError(f"Problem name must be a symbol, got {problem_name!r}")
    activity, sep, id_str = problem_name.rpartition("-")
    if not sep or not activity or not id_str.isdigit():
        raise ISGProblemError(f"Problem name {problem_name!r} does not have the '<activity>-<definition id>' shape")
    return activity, int(id_str)


def _strip(token: str) -> str:
    return token.lstrip("?")


def _parse_typed_list(tokens: Sequence[str], section: str) -> Dict[str, List[str]]:
    """Parse ``MEMBER MEMBER - TYPE MEMBER - TYPE ...`` preserving declaration order."""
    result: Dict[str, List[str]] = {}
    pending: List[str] = []
    tokens = list(tokens)
    while tokens:
        tok = tokens.pop(0)
        if tok == "-":
            if not pending:
                raise ISGProblemError(f"{section}: unexpected '-' with no members before it")
            if not tokens:
                raise ISGProblemError(f"{section}: missing type name after '-'")
            type_name = tokens.pop(0)
            if not isinstance(type_name, str):
                raise ISGProblemError(f"{section}: type name must be a symbol, got {type_name!r}")
            result.setdefault(type_name, []).extend(pending)
            pending = []
        elif isinstance(tok, str):
            pending.append(tok)
        else:
            raise ISGProblemError(f"{section}: unexpected nested list {tok!r}")
    if pending:
        raise ISGProblemError(f"{section}: members {pending} have no type ('- TYPE' missing)")
    return result


def to_bddl(expr, indent: int = 0) -> str:
    """Serialize a parsed nested-list expression back to BDDL text."""
    pad = "    " * indent
    if isinstance(expr, str):
        return pad + expr
    if all(isinstance(e, str) for e in expr):
        return pad + "(" + " ".join(expr) + ")"
    head = expr[0]
    if head == "not" and len(expr) == 2 and all(isinstance(e, str) for e in expr[1]):
        return pad + "(not (" + " ".join(expr[1]) + "))"
    if head in QUANTIFIER_TOKENS:
        # (forall (?x - cat) body) / (forn (2) (?x - cat) body) / (forpairs (?x - c1) (?y - c2) body)
        parts = [pad + f"({head}"]
        inner = expr[1:]
        body = inner[-1]
        for spec in inner[:-1]:
            parts[0] += " (" + " ".join(spec) + ")"
        parts.append(to_bddl(body, indent + 1))
        parts.append(pad + ")")
        return "\n".join(parts)
    lines = [pad + f"({head}"]
    for child in expr[1:]:
        lines.append(to_bddl(child, indent + 1))
    lines.append(pad + ")")
    return "\n".join(lines)


def _flatten_symbols(expr) -> Iterable[str]:
    if isinstance(expr, str):
        yield expr
    else:
        for e in expr:
            yield from _flatten_symbols(e)


def collect_predicate_args(node, token: str) -> List[str]:
    """Recursively collect the resolved argument names of every ``token`` leaf predicate below a compiled node."""
    args: List[str] = []
    if getattr(node, "STATE_NAME", None) == token:
        args.extend(getattr(node, "inputs", []))
    for child in getattr(node, "children", []):
        if isinstance(child, list):  # ForPairs stores children as a list of lists
            for sub in child:
                args.extend(collect_predicate_args(sub, token))
        else:
            args.extend(collect_predicate_args(child, token))
    return args


# ----------------------------------------------------------------------------------------------------------------
# Problem
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class ISGProblem:
    """A parsed, validated and group-expanded iSpatialGym problem."""

    activity: str
    definition_id: int
    problem_name: str
    domain: str
    variant: Optional[str]
    objects: Dict[str, List[str]]  # synset -> declared instances (declaration order)
    groups: Dict[str, List[str]]  # group symbol -> member instances
    raw_init: list
    raw_goal: list  # top-level goal conjuncts as written
    raw_measure: list = field(default_factory=list)  # (:measure ...) literals as written
    init: list = field(default_factory=list)  # expanded init literals
    goal: list = field(default_factory=list)  # expanded top-level goal conjuncts
    measure: list = field(default_factory=list)  # measured pose_reached / layout_reached literals (may also be in the goal)
    path: Optional[str] = None

    def content_key(self) -> tuple:
        """The objects, groups, expanded init and the union of the goal and measure atoms: what two copies of a problem must share
        whichever section an atom sits in (the 2026-10-14 scoring change copied / moved pose_reached / layout_reached into
        ``:measure``)."""
        flat = lambda exprs: sorted(" ".join(to_bddl(e).split()) for e in exprs)  # noqa: E731
        return ({k: list(v) for k, v in sorted(self.objects.items())}, {k: list(v) for k, v in sorted(self.groups.items())}, flat(self.init),
                sorted(set(flat(self.goal)) | set(flat(self.measure))))

    # ------------------------------------------------------------------ derived

    @property
    def name(self) -> str:
        return f"{self.activity}-{self.definition_id}"

    @property
    def canonical_activity(self) -> str:
        """``activity`` with a retired name (``world_alignment``) replaced by its canonical name (``alignment``)."""
        return canonical_activity(self.activity)

    @property
    def canonical_name(self) -> str:
        return f"{self.canonical_activity}-{self.definition_id}"

    @property
    def task_variant(self) -> Optional[str]:
        return self.variant

    @cached_property
    def instances(self) -> List[str]:
        return [inst for insts in self.objects.values() for inst in insts]

    @cached_property
    def instance_to_synset(self) -> Dict[str, str]:
        return {inst: syn for syn, insts in self.objects.items() for inst in insts}

    @cached_property
    def member_to_group(self) -> Dict[str, str]:
        return {m: g for g, members in self.groups.items() for m in members}

    def is_group(self, symbol: str) -> bool:
        return _strip(symbol) in self.groups

    @cached_property
    def scope_symbols(self) -> List[str]:
        """Every symbol the compiled conditions may reference: declared instances then group symbols."""
        return list(self.instances) + list(self.groups)

    # ------------------------------------------------------------------ goal structure

    @cached_property
    def goal_pose_settled_args(self) -> List[str]:
        return [_strip(g[1]) for g in self.goal if isinstance(g, list) and g and g[0] == SETTLE_PREDICATE]

    @cached_property
    def binary_goal(self) -> list:
        """Goal conjuncts that make up the binary success condition (everything except ``pose_settled``)."""
        return [g for g in self.goal if not (isinstance(g, list) and g and g[0] == SETTLE_PREDICATE)]

    def literal_args(self, predicate: str, where: str = "goal") -> List[str]:
        """Arguments of every (possibly nested) ``predicate`` literal in the expanded goal / init / measure."""
        out: List[str] = []

        def visit(expr):
            if not isinstance(expr, list) or not expr:
                return
            if expr[0] == predicate and all(isinstance(e, str) for e in expr):
                out.extend(_strip(a) for a in expr[1:])
                return
            for child in expr[1:] if expr[0] in LOGICAL_TOKENS else []:
                visit(child)

        source = {"goal": self.goal, "init": self.init, "measure": self.measure}[where]
        for expr in source:
            visit(expr)
        return out

    @cached_property
    def measure_pose_reached_args(self) -> List[str]:
        return _unique(self.literal_args("pose_reached", where="measure"))

    @cached_property
    def measure_layout_reached_args(self) -> List[str]:
        return _unique(self.literal_args("layout_reached", where="measure"))

    # ------------------------------------------------------------------ serialization

    def expanded_bddl(self) -> str:
        """Stock-compatible BDDL text of the expanded problem (groups appear only in aggregate atoms)."""
        lines = [f"(define (problem {self.problem_name})", f"    (:domain {self.domain})"]
        if self.variant:
            lines.append(f"    (:variant {self.variant})")
        lines.append("")
        lines.append("    (:objects")
        for syn, insts in self.objects.items():
            lines.append(f"        {' '.join(insts)} - {syn}")
        lines.append("    )")
        if self.groups:
            lines.append("")
            lines.append("    (:groups")
            for g, members in self.groups.items():
                lines.append(f"        {' '.join(members)} - {g}")
            lines.append("    )")
        lines.append("")
        lines.append("    (:init")
        for lit in self.init:
            lines.append(to_bddl(lit, 2))
        lines.append("    )")
        lines.append("")
        lines.append("    (:goal")
        lines.append(to_bddl(["and"] + list(self.goal), 2))
        lines.append("    )")
        if self.measure:
            lines.append("")
            lines.append("    (:measure")
            for lit in self.measure:
                lines.append(to_bddl(lit, 2))
            lines.append("    )")
        lines.append(")")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ compilation

    def compile(self, generate_ground_options: bool = True) -> "ISGCompiledTask":
        return ISGCompiledTask(self, generate_ground_options=generate_ground_options)


class ISGCompiledTask:
    """Compiled conditions of an :class:`ISGProblem`.

    Duck-type compatible with the attributes of ``bddl.knowledge_base.models.CompiledTask`` that
    ``omnigibson.tasks.behavior_task.BehaviorTask`` consumes (``object_scope``, ``parsed_objects``,
    ``conditions.parsed_goal_conditions``, ``initial_conditions``, ``goal_conditions``,
    ``ground_goal_state_options``, ``natural_language_*``, ``check_goal``).
    """

    def __init__(self, problem: ISGProblem, generate_ground_options: bool = True):
        self.problem = problem
        self.parsed_objects = {syn: list(insts) for syn, insts in problem.objects.items()}
        # Scope: declared instances + group symbols (nonphysical entities bound at runtime).
        self.object_scope = create_scope(self.parsed_objects) | set(problem.groups)
        self.conditions = SimpleNamespace(
            parsed_objects=self.parsed_objects,
            parsed_initial_conditions=copy.deepcopy(problem.init),
            parsed_goal_conditions=copy.deepcopy(problem.goal),
            behavior_activity=problem.activity,
            activity_definition=problem.definition_id,
        )
        self.initial_conditions: List[HEAD] = compile_state(
            [copy.deepcopy(c) for c in problem.init],
            scope=self.object_scope,
            object_map=self.parsed_objects,
            generate_ground_options=generate_ground_options,
        )
        self.goal_conditions: List[HEAD] = compile_state(
            [copy.deepcopy(c) for c in problem.goal],
            scope=self.object_scope,
            object_map=self.parsed_objects,
            generate_ground_options=generate_ground_options,
        )
        self.ground_goal_state_options = (
            get_ground_state_options(self.goal_conditions, scope=self.object_scope, object_map=self.parsed_objects)
            if generate_ground_options
            else []
        )

        # Goal partition: pose_settled heads vs. the binary-success heads.
        self.pose_settled_head_idxs: List[int] = []
        self.binary_goal_head_idxs: List[int] = []
        self.pose_settled_insts: List[str] = []
        for idx, head in enumerate(self.goal_conditions):
            insts = collect_predicate_args(head, SETTLE_PREDICATE)
            if insts:
                self.pose_settled_head_idxs.append(idx)
                for inst in insts:
                    if inst not in self.pose_settled_insts:
                        self.pose_settled_insts.append(inst)
            else:
                self.binary_goal_head_idxs.append(idx)

        # Gating (goal) pose / layout arguments ...
        self.pose_reached_args: List[str] = _unique(
            a for head in self.goal_conditions for a in collect_predicate_args(head, "pose_reached")
        )
        self.layout_reached_args: List[str] = _unique(
            a for head in self.goal_conditions for a in collect_predicate_args(head, "layout_reached")
        )
        # ... and the measured ones of (:measure ...) (an atom may be in both). Both kinds are configured from the TRO and reported.
        self.measure_pose_reached_args: List[str] = list(problem.measure_pose_reached_args)
        self.measure_layout_reached_args: List[str] = list(problem.measure_layout_reached_args)

    @property
    def all_pose_reached_args(self) -> List[str]:
        """Every entity whose pose target is evaluated (gating goal atoms first, then ``:measure`` atoms)."""
        return _unique(self.pose_reached_args + self.measure_pose_reached_args)

    @property
    def all_layout_reached_args(self) -> List[str]:
        return _unique(self.layout_reached_args + self.measure_layout_reached_args)

    def is_gating(self, predicate: str, arg: str) -> bool:
        """The atom ``(predicate arg)`` is in ``:goal``."""
        gated = self.pose_reached_args if predicate == "pose_reached" else self.layout_reached_args
        return arg in gated

    def in_measure(self, predicate: str, arg: str) -> bool:
        """The atom ``(predicate arg)`` is in ``:measure``."""
        measured = self.measure_pose_reached_args if predicate == "pose_reached" else self.measure_layout_reached_args
        return arg in measured

    # BehaviorTask-facing API ---------------------------------------------------------------------------------

    def check_goal(self, evaluate_fn):
        return evaluate_state(self.goal_conditions, evaluate_fn)

    def check_initial_conditions(self, evaluate_fn):
        return evaluate_state(self.initial_conditions, evaluate_fn)

    @staticmethod
    def evaluate_conditions(conditions, evaluate_fn):
        return evaluate_state(conditions, evaluate_fn)

    @cached_property
    def natural_language_initial_conditions(self):
        return gen_natural_language_conditions(self.conditions.parsed_initial_conditions)

    @cached_property
    def natural_language_goal_conditions(self):
        return gen_natural_language_conditions(self.conditions.parsed_goal_conditions)

    @property
    def task(self):
        return self.problem


def _unique(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------------------------------------------


def activity_dir(activity: str, root: Optional[str] = None) -> str:
    """Directory of ``activity`` under ``root`` (default: the shipped ``activity_definitions``).

    The canonical directory (``<root>/alignment``) wins; a directory under a retired name (``<root>/world_alignment``,
    as the v1 samplers wrote next to their templates) is used when the canonical one does not exist. When neither
    exists the canonical path is returned so that callers can name it in their errors.
    """
    root = ACTIVITY_DEFINITIONS_ROOT if root is None else root
    candidates = [os.path.join(root, name) for name in activity_names(activity)]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


def problem_path(activity: str, definition_id: int, root: Optional[str] = None) -> str:
    return os.path.join(activity_dir(activity, root), f"problem{int(definition_id)}.bddl")


def problem_registry_path(activity: str, root: Optional[str] = None) -> str:
    return os.path.join(activity_dir(activity, root), PROBLEM_REGISTRY_FILE)


def list_activities(root: Optional[str] = None) -> List[str]:
    """Activity directories under ``root``, as named on disk (a legacy root may still say ``world_alignment``)."""
    root = ACTIVITY_DEFINITIONS_ROOT if root is None else root
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith(("_", ".")))


def list_problem_ids(activity: str, root: Optional[str] = None) -> List[int]:
    """Definition ids of ``activity``: its ``problemN.bddl`` files, which must be contiguous from ``problem0.bddl``.

    Other files of the directory (``problems.json``, notes) are ignored.
    """
    directory = activity_dir(activity, root)
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"No activity directory for {activity!r}: none of {activity_names(activity)} exists under "
            f"{os.path.dirname(directory)}"
        )
    ids = sorted(int(m.group(1)) for m in (PROBLEM_FILE_RE.match(f) for f in os.listdir(directory)) if m)
    if ids != list(range(len(ids))):
        missing = sorted(set(range(ids[-1] + 1)) - set(ids))
        raise ISGProblemError(
            f"Activity {activity!r} ({directory}) must have contiguous problem files problem0.bddl ... "
            f"problem{ids[-1]}.bddl: found ids {ids}, missing {missing}"
        )
    return ids


def load_problem(activity: str, definition_id: int, root: Optional[str] = None) -> ISGProblem:
    """Load ``problem<definition_id>.bddl`` of ``activity`` (a retired activity name is accepted).

    The returned problem's ``activity`` is the directory it was loaded from: the canonical name for the shipped
    templates (``load_problem("world_alignment", 0).activity == "alignment"``), a retired name only for a legacy
    root that still uses it.
    """
    path = problem_path(activity, definition_id, root)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No iSpatialGym problem file at {path}")
    return parse_problem_file(path, activity=os.path.basename(os.path.dirname(path)), definition_id=int(definition_id))


def parse_problem_file(path: str, activity: Optional[str] = None, definition_id: Optional[int] = None) -> ISGProblem:
    tokens = scan_tokens(filename=path)
    if activity is None or definition_id is None:
        m = PROBLEM_FILE_RE.match(os.path.basename(path))
        activity = os.path.basename(os.path.dirname(path)) if activity is None else activity
        definition_id = int(m.group(1)) if (definition_id is None and m) else definition_id
    problem = parse_problem_tokens(tokens, activity=activity, definition_id=definition_id)
    problem.path = path
    return problem


# ----------------------------------------------------------------------------------------------------------------
# Variant <-> definition id registry (problems.json)
# ----------------------------------------------------------------------------------------------------------------


def _read_variant(path: str) -> Optional[str]:
    """The ``(:variant ...)`` of a problem file without parsing / validating the rest of it."""
    tokens = scan_tokens(filename=path)
    for section in tokens[1:] if isinstance(tokens, list) else []:
        if isinstance(section, list) and section and section[0] in (":variant", *LEGACY_SECTIONS):
            if len(section) != 2 or not isinstance(section[1], str):
                raise ISGProblemError(f"{path}: (:variant ...) must hold exactly one symbol, got {section[1:]!r}")
            return canonical_variant(section[1])
    return None


def _variant_repr(variant: Optional[str]) -> str:
    return f"(:variant {variant})" if variant is not None else "no (:variant ...)"


def problem_variants(activity: str, root: Optional[str] = None) -> Dict[int, Optional[str]]:
    """``{definition id: variant}`` read from the ``problemN.bddl`` files of ``activity`` (``None`` = no variant)."""
    return {pid: _read_variant(problem_path(activity, pid, root)) for pid in list_problem_ids(activity, root)}


def _registry_entries(data, what: str):
    """Normalize the accepted ``problems.json`` shapes to ``(variant, id)`` pairs."""
    if isinstance(data, dict) and "problems" in data:
        data = data["problems"]
    if isinstance(data, list):
        for i, rec in enumerate(data):
            rec = {("variant" if k == "alias" else k): v for k, v in rec.items()} if isinstance(rec, dict) else rec  # legacy spelling
            if not isinstance(rec, dict) or "variant" not in rec:
                raise ISGProblemError(f"{what}: entry {i} must be an object with 'variant' and 'id', got {rec!r}")
            pid = rec.get("id", rec.get("definition_id"))
            if pid is None:
                raise ISGProblemError(f"{what}: entry {i} ({rec.get('variant')!r}) has no 'id' / 'definition_id'")
            yield canonical_variant(rec["variant"]), pid
    elif isinstance(data, dict):
        if data and all(isinstance(k, str) and k.isdigit() for k in data):  # inverse shape {"<id>": "<variant>"}
            for k, variant in data.items():
                yield canonical_variant(variant), int(k)
        else:
            for variant, pid in data.items():
                yield canonical_variant(variant), pid
    else:
        raise ISGProblemError(f"{what}: must be a JSON object {{variant: id}} or a list of {{variant, id}} records")


def load_problem_registry(activity_dir: str, missing_ok: bool = False) -> Dict[str, int]:
    """``{variant: definition id}`` from ``<activity_dir>/problems.json`` (``activity_dir`` may also be the file).

    Canonical shape: a JSON object ``{"<variant>": <id>, ...}``. Also accepted: the inverse ``{"<id>": "<variant>"}``,
    a list of ``{"variant": ..., "id" | "definition_id": ...}`` records, and any of these under a ``"problems"`` key
    (an ``"activity"`` sibling, when present, must name this activity). Variants must match :data:`VARIANT_RE` and be
    unique; ids must be distinct non-negative integers. A missing file raises ``FileNotFoundError`` unless
    ``missing_ok`` (then ``{}``).
    """
    path = activity_dir if os.path.isfile(activity_dir) else os.path.join(activity_dir, PROBLEM_REGISTRY_FILE)
    if not os.path.isfile(path):
        if missing_ok:
            return {}
        raise FileNotFoundError(f"No problem registry at {path}")
    what = f"problem registry {path}"
    with open(path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ISGProblemError(f"{what}: invalid JSON ({e})") from e
    if isinstance(data, dict) and isinstance(data.get("activity"), str):
        directory = os.path.basename(os.path.dirname(os.path.abspath(path)))
        if not same_activity(data["activity"], directory):
            raise ISGProblemError(f"{what}: 'activity' is {data['activity']!r} but the directory is {directory!r}")
    registry: Dict[str, int] = {}
    ids: Dict[int, str] = {}
    for variant, pid in _registry_entries(data, what):
        if not isinstance(variant, str) or not VARIANT_RE.match(variant):
            raise ISGProblemError(f"{what}: invalid variant {variant!r}")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid < 0:
            if isinstance(pid, str) and pid.isdigit():
                pid = int(pid)
            else:
                raise ISGProblemError(f"{what}: id of {variant!r} must be a non-negative integer, got {pid!r}")
        if variant in registry:
            raise ISGProblemError(f"{what}: variant {variant!r} is registered more than once")
        if pid in ids:
            raise ISGProblemError(f"{what}: id {pid} is registered for both {ids[pid]!r} and {variant!r}")
        registry[variant] = pid
        ids[pid] = variant
    return registry


def variant_of(activity: str, definition_id: int, root: Optional[str] = None) -> Optional[str]:
    """Variant of problem ``definition_id`` of ``activity``.

    From ``problems.json`` when it registers the id (checked against the file's ``(:variant ...)`` when the file
    exists), else from the problem file itself; ``None`` when the problem declares no variant.
    """
    definition_id = int(definition_id)
    directory = activity_dir(activity, root)
    registry = load_problem_registry(directory, missing_ok=True)
    registered = next((variant for variant, pid in registry.items() if pid == definition_id), None)
    path = problem_path(activity, definition_id, root)
    if not os.path.isfile(path):
        if registered is not None:
            raise ISGProblemError(f"{PROBLEM_REGISTRY_FILE} of {activity!r} registers {registered!r} -> {definition_id} but {path} does not exist")
        raise FileNotFoundError(f"Activity {activity!r} has no problem {definition_id}: neither a registry entry nor {path}")
    declared = _read_variant(path)
    if registered is not None and declared != registered:
        raise ISGProblemError(
            f"{PROBLEM_REGISTRY_FILE} of {activity!r} registers {registered!r} -> {definition_id} but {path} declares "
            f"{_variant_repr(declared)}"
        )
    return declared


def id_of_variant(activity: str, variant: str, root: Optional[str] = None) -> int:
    """Definition id of the problem of ``activity`` whose variant is ``variant``.

    From ``problems.json`` when it registers the variant (checked against the file when it exists), else by scanning
    the ``(:variant ...)`` of the problem files. Raises :class:`ISGProblemError` naming the known variants otherwise.
    """
    variant = canonical_variant(variant)
    directory = activity_dir(activity, root)
    registry = load_problem_registry(directory, missing_ok=True)
    if variant in registry:
        pid = registry[variant]
        path = problem_path(activity, pid, root)
        if os.path.isfile(path) and _read_variant(path) != variant:
            raise ISGProblemError(
                f"{PROBLEM_REGISTRY_FILE} of {activity!r} registers {variant!r} -> {pid} but {path} declares "
                f"{_variant_repr(_read_variant(path))}"
            )
        return pid
    declared = problem_variants(activity, root)
    matches = [pid for pid, a in declared.items() if a == variant]
    if len(matches) > 1:
        raise ISGProblemError(f"Activity {activity!r}: problems {matches} all declare (:variant {variant})")
    if matches:
        return matches[0]
    known = sorted(set(registry) | {a for a in declared.values() if a})
    raise ISGProblemError(f"Activity {activity!r} has no problem with variant {variant!r}; known variants: {known}")


def validate_problem_registry(activity: str, root: Optional[str] = None) -> Dict[str, int]:
    """Check ``problems.json`` of ``activity`` against its problem files and return the registry.

    Every registered id must have a ``problemN.bddl`` whose ``(:variant ...)`` is the registered variant, every problem
    file that declares a variant must be registered under it, and the files must be contiguous. A problem without an
    variant (the standard camera relocalization) need not be registered.
    """
    registry = load_problem_registry(activity_dir(activity, root))
    declared = problem_variants(activity, root)
    for variant, pid in registry.items():
        if pid not in declared:
            raise ISGProblemError(f"{PROBLEM_REGISTRY_FILE} of {activity!r} registers {variant!r} -> {pid} but problem{pid}.bddl does not exist")
        if declared[pid] != variant:
            raise ISGProblemError(
                f"{PROBLEM_REGISTRY_FILE} of {activity!r} registers {variant!r} -> {pid} but problem{pid}.bddl declares "
                f"{_variant_repr(declared[pid])}"
            )
    for pid, variant in declared.items():
        if variant is not None and registry.get(variant) != pid:
            raise ISGProblemError(
                f"problem{pid}.bddl of {activity!r} declares (:variant {variant}) but {PROBLEM_REGISTRY_FILE} does not register it"
            )
    return registry


def parse_problem_string(text: str, activity: str, definition_id: int) -> ISGProblem:
    return parse_problem_tokens(scan_tokens(string=text), activity=activity, definition_id=definition_id)


def parse_problem_tokens(tokens, activity: str, definition_id: Optional[int]) -> ISGProblem:
    if not isinstance(tokens, list) or not tokens or tokens[0] != "define":
        raise ISGProblemError("Problem file must start with (define ...)")
    sections = tokens[1:]
    problem_name = None
    domain = None
    variant = None
    objects: Dict[str, List[str]] = {}
    groups: Dict[str, List[str]] = {}
    init = None
    goal = None
    measure: list = []
    seen = set()
    order = []
    for group in sections:
        if not isinstance(group, list) or not group:
            raise ISGProblemError(f"Malformed section {group!r}")
        tag = LEGACY_SECTIONS.get(group[0], group[0])
        if tag in seen:
            raise ISGProblemError(f"Duplicate section {tag}")
        seen.add(tag)
        order.append(tag)
        if tag == "problem":
            problem_name = group[-1]
        elif tag == ":domain":
            domain = group[-1]
        elif tag == ":variant":
            if len(group) != 2 or not isinstance(group[1], str):
                raise ISGProblemError(f"(:variant ...) must hold exactly one symbol, got {group[1:]!r}")
            variant = canonical_variant(group[1])
            if not VARIANT_RE.match(variant):
                raise ISGProblemError(f"Invalid variant {variant!r}")
        elif tag == ":objects":
            objects = _parse_typed_list(group[1:], ":objects")
        elif tag == ":groups":
            groups = _parse_typed_list(group[1:], ":groups")
        elif tag == ":init":
            init = group[1:]
        elif tag == ":goal":
            if len(group) != 2:
                raise ISGProblemError("(:goal ...) must hold exactly one expression")
            goal_expr = group[1]
            if isinstance(goal_expr, list) and goal_expr and goal_expr[0] == "and":
                goal = goal_expr[1:]
            else:
                goal = [goal_expr]
        elif tag == ":measure":
            measure = group[1:]
        elif tag == ":requirements":
            pass
        else:
            raise ISGProblemError(f"Unrecognized section {tag!r}")

    if domain != DOMAIN_NAME:
        raise ISGProblemError(f"Problem declares domain {domain!r}; expected {DOMAIN_NAME!r}")
    if problem_name is None:
        raise ISGProblemError("Missing (problem <name>) section")
    if init is None or goal is None:
        raise ISGProblemError("Both (:init ...) and (:goal ...) sections are required")
    if ":variant" in order and order.index(":variant") != order.index(":domain") + 1:
        raise ISGProblemError("(:variant ...) must directly follow (:domain ...)")
    if ":measure" in order and order.index(":measure") < order.index(":goal"):
        raise ISGProblemError("(:measure ...) must come after (:goal ...)")
    # ``<activity>-<definition id>``; a retired activity name (``world_alignment-0`` for ``alignment-0``) is accepted.
    name_activity, name_id = split_problem_name(problem_name)
    if not same_activity(name_activity, activity):
        accepted = " / ".join(f"'{n}-{name_id if definition_id is None else definition_id}'" for n in activity_names(activity))
        raise ISGProblemError(
            f"Problem name {problem_name!r} does not match {accepted} (file naming convention: activity {activity!r})"
        )
    if definition_id is not None and name_id != int(definition_id):
        raise ISGProblemError(
            f"Problem name {problem_name!r} does not match '{activity}-{definition_id}' (file naming convention)"
        )
    if definition_id is None:
        definition_id = name_id

    problem = ISGProblem(
        activity=activity,
        definition_id=int(definition_id),
        problem_name=problem_name,
        domain=domain,
        variant=variant,
        objects=objects,
        groups=groups,
        raw_init=copy.deepcopy(init),
        raw_goal=copy.deepcopy(goal),
        raw_measure=copy.deepcopy(measure),
    )
    _validate_declarations(problem)
    expander = _GroupExpander(problem)
    problem.init = expander.expand_init(init)
    problem.goal = expander.expand_goal(goal)
    problem.measure = _validate_measure(problem, measure)
    _validate_conditions(problem)
    return problem


# ----------------------------------------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------------------------------------


def _validate_declarations(problem: ISGProblem) -> None:
    declared = problem.instances
    if len(set(declared)) != len(declared):
        dupes = sorted({d for d in declared if declared.count(d) > 1})
        raise ISGProblemError(f":objects declares instances more than once: {dupes}")
    for syn, insts in problem.objects.items():
        for inst in insts:
            if not INSTANCE_RE.match(inst):
                raise ISGProblemError(f"Instance {inst!r} does not have the '<synset>_<n>' shape")
            if synset_of(inst) != syn:
                raise ISGProblemError(f"Instance {inst!r} is declared under synset {syn!r}")
    if AGENT_INSTANCE not in declared:
        raise ISGProblemError(f"{AGENT_INSTANCE} must be declared in :objects")

    declared_set = set(declared)
    seen_members: Dict[str, str] = {}
    for g, members in problem.groups.items():
        if not INSTANCE_RE.match(g):
            raise ISGProblemError(f"Group name {g!r} must have the instance-name shape (e.g. blocks_group.n.01_1)")
        if g in declared_set:
            raise ISGProblemError(f"Group {g!r} must not be declared in :objects")
        if not members:
            raise ISGProblemError(f"Group {g!r} has no members")
        for m in members:
            if m in problem.groups:
                raise ISGProblemError(f"Group {g!r} nests group {m!r}; groups do not nest")
            if m not in declared_set:
                raise ISGProblemError(f"Group {g!r} member {m!r} is not declared in :objects")
            if m in seen_members:
                raise ISGProblemError(f"Instance {m!r} belongs to both {seen_members[m]!r} and {g!r}")
            seen_members[m] = g


def _validate_measure(problem: ISGProblem, measure: list) -> list:
    """Validate the ``(:measure ...)`` literals: unary ``pose_reached`` / ``layout_reached`` atoms on declared
    instances or groups, no duplicates (an atom may also be in the goal)."""
    symbols = set(problem.scope_symbols)
    out = []
    seen = set()
    for lit in measure:
        if not isinstance(lit, list) or not lit or not all(isinstance(e, str) for e in lit):
            raise ISGProblemError(f":measure: entries must be flat literals, got {lit!r}")
        pred = lit[0]
        if pred not in MEASURE_PREDICATES:
            raise ISGProblemError(f":measure only accepts {sorted(MEASURE_PREDICATES)} atoms, got {lit}")
        if len(lit) != 2:
            raise ISGProblemError(f":measure: {pred!r} takes exactly one argument, got {lit}")
        arg = _strip(lit[1])
        if arg not in symbols:
            raise ISGProblemError(f":measure: {arg!r} is not a declared instance or group in {lit}")
        if (pred, arg) in seen:
            raise ISGProblemError(f":measure lists ({pred} {arg}) more than once")
        seen.add((pred, arg))
        out.append([pred, "?" + arg])
    return out


def _validate_conditions(problem: ISGProblem) -> None:
    _, _, _, _, predicates = parse_domain(DOMAIN_NAME)
    symbols = set(problem.scope_symbols)

    def check_literal(lit, where):
        pred = lit[0]
        if pred not in predicates:
            raise ISGProblemError(f"{where}: unknown predicate {pred!r} for domain {DOMAIN_NAME}")
        arity = len(predicates[pred])
        args = lit[1:]
        if len(args) != arity:
            raise ISGProblemError(f"{where}: predicate {pred!r} takes {arity} argument(s), got {lit}")
        for a in args:
            name = _strip(a)
            if name in symbols:
                continue
            # Quantifier-bound variables (e.g. ?x) are resolved by the compiler; undeclared constants are errors.
            if where == ":init" or INSTANCE_RE.match(name):
                raise ISGProblemError(f"{where}: {name!r} is not a declared instance or group in {lit}")
        if where == ":init" and pred in FORBIDDEN_IN_INIT:
            raise ISGProblemError(f":init must not contain {pred!r}; targets are configured from the TRO")
        if pred == "grasped" and not is_agent_instance(_strip(args[0])):
            raise ISGProblemError(f"{where}: the first argument of 'grasped' must be the agent, got {lit}")

    def visit(expr, where, top_level=False):
        if not isinstance(expr, list) or not expr:
            raise ISGProblemError(f"{where}: malformed expression {expr!r}")
        head = expr[0]
        if head == SETTLE_PREDICATE and not top_level:
            raise ISGProblemError(f"{where}: (pose_settled ...) must be a direct child of the top-level goal 'and'")
        if head in QUANTIFIER_TOKENS:
            visit(expr[-1], where)
        elif head in LOGICAL_TOKENS:
            for child in expr[1:]:
                visit(child, where)
        else:
            if not all(isinstance(e, str) for e in expr):
                raise ISGProblemError(f"{where}: malformed literal {expr!r}")
            check_literal(expr, where)

    for lit in problem.init:
        if isinstance(lit, list) and lit and lit[0] == "not":
            if len(lit) != 2 or not isinstance(lit[1], list):
                raise ISGProblemError(f":init: malformed negation {lit!r}")
            visit(lit[1], ":init")
            if lit[1][0] == SETTLE_PREDICATE:
                raise ISGProblemError(":init must not contain 'pose_settled'")
        else:
            visit(lit, ":init")
    for conj in problem.goal:
        visit(conj, ":goal", top_level=True)
    if not problem.binary_goal:
        raise ISGProblemError(":goal must contain at least one non-pose_settled condition")
    for arg in problem.goal_pose_settled_args:
        if arg in problem.groups:
            raise ISGProblemError(f"pose_settled on group {arg!r} should have been expanded")  # defensive


# ----------------------------------------------------------------------------------------------------------------
# Group expansion
# ----------------------------------------------------------------------------------------------------------------


class _GroupExpander:
    def __init__(self, problem: ISGProblem):
        self.problem = problem
        self.groups = problem.groups

    def _group_of(self, token: str) -> Optional[str]:
        name = _strip(token)
        return name if name in self.groups else None

    def _members(self, token: str) -> List[str]:
        prefix = "?" if token.startswith("?") else ""
        return [prefix + m for m in self.groups[_strip(token)]]

    def expand_init(self, init: list) -> list:
        out = []
        for lit in init:
            expanded = self.expand(lit, where=":init")
            out.extend(self._flatten_and(expanded))
        return out

    def expand_goal(self, goal: list) -> list:
        out = []
        for conj in goal:
            expanded = self.expand(conj, where=":goal")
            out.extend(self._flatten_and(expanded))
        return out

    @staticmethod
    def _flatten_and(expr) -> list:
        if isinstance(expr, list) and expr and expr[0] == "and" and getattr(expr, "_from_group", False):
            return list(expr[1:])
        return [expr]

    def expand(self, expr, where: str):
        if not isinstance(expr, list) or not expr:
            raise ISGProblemError(f"{where}: malformed expression {expr!r}")
        head = expr[0]
        if head in QUANTIFIER_TOKENS:
            for sym in _flatten_symbols(expr):
                if self._group_of(sym):
                    raise ISGProblemError(f"{where}: group {_strip(sym)!r} inside quantifier {head!r} is not allowed")
            return expr
        if head == "not":
            if len(expr) != 2:
                raise ISGProblemError(f"{where}: malformed negation {expr!r}")
            inner = expr[1]
            if isinstance(inner, list) and inner and inner[0] not in LOGICAL_TOKENS:
                group_positions = [i for i, a in enumerate(inner[1:], start=1) if self._group_of(a)]
                if group_positions and inner[0] not in AGGREGATE_PREDICATES:
                    lits = self._expand_pointwise(inner, group_positions, where)
                    return self._and([["not", lit] for lit in lits])
                return expr
            return ["not", self.expand(inner, where)]
        if head in LOGICAL_TOKENS:
            return [head] + [self.expand(child, where) for child in expr[1:]]
        # predicate literal
        if not all(isinstance(e, str) for e in expr):
            raise ISGProblemError(f"{where}: malformed literal {expr!r}")
        group_positions = [i for i, a in enumerate(expr[1:], start=1) if self._group_of(a)]
        if not group_positions:
            return expr
        if head in AGGREGATE_PREDICATES:
            if len(expr) != 2:
                raise ISGProblemError(f"{where}: aggregate predicate {head!r} must be unary, got {expr}")
            return expr
        if head == SETTLE_PREDICATE and where == ":init":
            raise ISGProblemError(":init must not contain 'pose_settled'")
        return self._and(self._expand_pointwise(expr, group_positions, where))

    def _expand_pointwise(self, lit: list, group_positions: List[int], where: str) -> List[list]:
        if len(group_positions) != 1:
            raise ISGProblemError(f"{where}: a literal may reference at most one group (v1), got {lit}")
        pos = group_positions[0]
        out = []
        for member in self._members(lit[pos]):
            new_lit = list(lit)
            new_lit[pos] = member
            out.append(new_lit)
        return out

    @staticmethod
    def _and(children: List[list]):
        node = _AndList(["and"] + children)
        node._from_group = True
        return node


class _AndList(list):
    """A list subclass so the expander can tag conjunctions it created (flattened at the top level)."""

    _from_group = False
