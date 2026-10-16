"""Parsing, validation, group expansion and compilation of iSpatialGym BDDL problems.

An iSpatialGym problem is a BDDL problem in the ``ispatialgym`` domain with three extra sections:

* ``(:variant <variant>)`` directly after ``:domain``: the grounded setup the problem was written for.
* ``(:groups MEMBER ... - GROUP ...)``: typed lists shaped like ``:objects``. Group names have the instance-name shape
  (``targets_group.n.01_1``) but are not declared in ``:objects`` and load nothing.
* ``(:measure (pose_reached ?x) (layout_reached ?g) ...)`` after ``:goal``: the measured targets. Geometric success
  requires each of them within its tolerance; symbolic success is the goal alone. An atom may also appear in the goal.

Groups are expanded before compilation: a pointwise literal on a group becomes a conjunction over its members, while
``pose_reached`` / ``layout_reached`` on a group stay aggregate atoms evaluated once on the group entity. Groups are not
allowed inside quantifiers. The expanded problem is compiled with the stock ``bddl.condition_evaluation`` machinery.

Problem ``N`` of activity ``A`` lives in ``activity_definitions/A/problemN.bddl`` and is named ``A-N``.
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from types import SimpleNamespace

from bddl.condition_evaluation import HEAD, compile_state, create_scope, evaluate_state, get_ground_state_options
from bddl.parsing import gen_natural_language_conditions, parse_domain, scan_tokens

from ispatialgym import ASSETS_ROOT

ACTIVITY_DEFINITIONS_ROOT = os.path.join(ASSETS_ROOT, "bddl", "activity_definitions")
DOMAIN_NAME = "ispatialgym"
AGENT_INSTANCE = "agent.n.01_1"

AGGREGATE_PREDICATES = frozenset({"layout_reached", "pose_reached"})
MEASURE_PREDICATES = AGGREGATE_PREDICATES
SETTLE_PREDICATE = "pose_settled"
FORBIDDEN_IN_INIT = frozenset({"pose_reached", "pose_settled", "layout_reached"})

LOGICAL_TOKENS = frozenset({"and", "or", "not", "imply", "forall", "exists", "forn", "forpairs", "fornpairs"})
QUANTIFIER_TOKENS = frozenset({"forall", "exists", "forn", "forpairs", "fornpairs"})

INSTANCE_RE = re.compile(r"^[a-z0-9_]+(?:-[a-z0-9_]+)*\.n\.\d{2}_\d+$")
VARIANT_RE = re.compile(r"^[a-z][a-z0-9_]*(?:-[a-z0-9_]+)*$")
PROBLEM_FILE_RE = re.compile(r"^problem(\d+)\.bddl$")


class ISGProblemError(ValueError):
    """A malformed iSpatialGym problem."""


def synset_of(instance: str) -> str:
    """``mug.n.04_1`` -> ``mug.n.04``."""
    return instance.rsplit("_", 1)[0]


def is_agent_instance(instance: str) -> bool:
    return instance.startswith("agent.n.")


def split_problem_name(problem_name: str) -> tuple[str, int]:
    """``alignment-0`` -> ``("alignment", 0)``."""
    if not isinstance(problem_name, str):
        raise ISGProblemError(f"Problem name must be a symbol, got {problem_name!r}")
    activity, sep, id_str = problem_name.rpartition("-")
    if not sep or not activity or not id_str.isdigit():
        raise ISGProblemError(f"Problem name {problem_name!r} does not have the '<activity>-<definition id>' shape")
    return activity, int(id_str)


def _strip(token: str) -> str:
    return token.lstrip("?")


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _parse_typed_list(tokens: Sequence[str], section: str) -> dict[str, list[str]]:
    """Parse ``MEMBER MEMBER - TYPE MEMBER - TYPE ...`` preserving declaration order."""
    result: dict[str, list[str]] = {}
    pending: list[str] = []
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
        parts = [pad + f"({head}"]
        *specs, body = expr[1:]
        for spec in specs:
            parts[0] += " (" + " ".join(spec) + ")"
        parts.append(to_bddl(body, indent + 1))
        parts.append(pad + ")")
        return "\n".join(parts)
    lines = [pad + f"({head}"]
    lines.extend(to_bddl(child, indent + 1) for child in expr[1:])
    lines.append(pad + ")")
    return "\n".join(lines)


def _flatten_symbols(expr) -> Iterable[str]:
    if isinstance(expr, str):
        yield expr
    else:
        for e in expr:
            yield from _flatten_symbols(e)


def collect_predicate_args(node, token: str) -> list[str]:
    """The resolved arguments of every ``token`` leaf predicate below a compiled node."""
    args: list[str] = []
    if getattr(node, "STATE_NAME", None) == token:
        args.extend(getattr(node, "inputs", []))
    for child in getattr(node, "children", []):
        if isinstance(child, list):  # ForPairs stores its children as a list of lists
            for sub in child:
                args.extend(collect_predicate_args(sub, token))
        else:
            args.extend(collect_predicate_args(child, token))
    return args


@dataclass
class ISGProblem:
    """A parsed, validated and group-expanded iSpatialGym problem."""

    activity: str
    definition_id: int
    problem_name: str
    domain: str
    variant: str | None
    objects: dict[str, list[str]]  # synset -> declared instances
    groups: dict[str, list[str]]  # group symbol -> member instances
    init: list = field(default_factory=list)
    goal: list = field(default_factory=list)  # expanded top-level goal conjuncts
    measure: list = field(default_factory=list)
    path: str | None = None

    @property
    def name(self) -> str:
        return f"{self.activity}-{self.definition_id}"

    @cached_property
    def instances(self) -> list[str]:
        return [inst for insts in self.objects.values() for inst in insts]

    @cached_property
    def instance_to_synset(self) -> dict[str, str]:
        return {inst: syn for syn, insts in self.objects.items() for inst in insts}

    @cached_property
    def scope_symbols(self) -> list[str]:
        """Every symbol the compiled conditions may reference: declared instances, then group symbols."""
        return list(self.instances) + list(self.groups)

    @cached_property
    def goal_pose_settled_args(self) -> list[str]:
        return [_strip(g[1]) for g in self.goal if isinstance(g, list) and g and g[0] == SETTLE_PREDICATE]

    @cached_property
    def binary_goal(self) -> list:
        """The goal conjuncts except ``pose_settled``."""
        return [g for g in self.goal if not (isinstance(g, list) and g and g[0] == SETTLE_PREDICATE)]

    def literal_args(self, predicate: str, where: str = "goal") -> list[str]:
        """Arguments of every (possibly nested) ``predicate`` literal of the expanded goal / init / measure."""
        out: list[str] = []

        def visit(expr):
            if not isinstance(expr, list) or not expr:
                return
            if expr[0] == predicate and all(isinstance(e, str) for e in expr):
                out.extend(_strip(a) for a in expr[1:])
                return
            for child in expr[1:] if expr[0] in LOGICAL_TOKENS else []:
                visit(child)

        for expr in {"goal": self.goal, "init": self.init, "measure": self.measure}[where]:
            visit(expr)
        return out

    @cached_property
    def measure_pose_reached_args(self) -> list[str]:
        return _unique(self.literal_args("pose_reached", where="measure"))

    @cached_property
    def measure_layout_reached_args(self) -> list[str]:
        return _unique(self.literal_args("layout_reached", where="measure"))

    def compile(self, generate_ground_options: bool = True) -> ISGCompiledTask:
        return ISGCompiledTask(self, generate_ground_options=generate_ground_options)


class ISGCompiledTask:
    """Compiled conditions of an :class:`ISGProblem`.

    Duck-type compatible with the parts of ``bddl.knowledge_base.models.CompiledTask`` that OmniGibson's
    ``BehaviorTask`` reads.
    """

    def __init__(self, problem: ISGProblem, generate_ground_options: bool = True):
        self.problem = problem
        self.parsed_objects = {syn: list(insts) for syn, insts in problem.objects.items()}
        self.object_scope = create_scope(self.parsed_objects) | set(problem.groups)
        self.conditions = SimpleNamespace(
            parsed_objects=self.parsed_objects,
            parsed_initial_conditions=copy.deepcopy(problem.init),
            parsed_goal_conditions=copy.deepcopy(problem.goal),
            behavior_activity=problem.activity,
            activity_definition=problem.definition_id,
        )
        compile_kwargs = dict(
            scope=self.object_scope, object_map=self.parsed_objects, generate_ground_options=generate_ground_options
        )
        self.initial_conditions: list[HEAD] = compile_state(copy.deepcopy(problem.init), **compile_kwargs)
        self.goal_conditions: list[HEAD] = compile_state(copy.deepcopy(problem.goal), **compile_kwargs)
        self.ground_goal_state_options = (
            get_ground_state_options(self.goal_conditions, scope=self.object_scope, object_map=self.parsed_objects)
            if generate_ground_options
            else []
        )

        self.pose_settled_head_idxs: list[int] = []
        self.binary_goal_head_idxs: list[int] = []
        self.pose_settled_insts: list[str] = []
        for idx, head in enumerate(self.goal_conditions):
            insts = collect_predicate_args(head, SETTLE_PREDICATE)
            if insts:
                self.pose_settled_head_idxs.append(idx)
                self.pose_settled_insts = _unique(self.pose_settled_insts + insts)
            else:
                self.binary_goal_head_idxs.append(idx)

        self.pose_reached_args = _unique(
            a for h in self.goal_conditions for a in collect_predicate_args(h, "pose_reached")
        )
        self.layout_reached_args = _unique(
            a for h in self.goal_conditions for a in collect_predicate_args(h, "layout_reached")
        )
        self.measure_pose_reached_args = list(problem.measure_pose_reached_args)
        self.measure_layout_reached_args = list(problem.measure_layout_reached_args)

    @property
    def all_pose_reached_args(self) -> list[str]:
        """Every entity whose pose target is evaluated: goal atoms first, then ``:measure`` atoms."""
        return _unique(self.pose_reached_args + self.measure_pose_reached_args)

    @property
    def all_layout_reached_args(self) -> list[str]:
        return _unique(self.layout_reached_args + self.measure_layout_reached_args)

    def is_gating(self, predicate: str, arg: str) -> bool:
        """Whether ``(predicate arg)`` is in ``:goal``."""
        return arg in (self.pose_reached_args if predicate == "pose_reached" else self.layout_reached_args)

    def in_measure(self, predicate: str, arg: str) -> bool:
        """Whether ``(predicate arg)`` is in ``:measure``."""
        return arg in (
            self.measure_pose_reached_args if predicate == "pose_reached" else self.measure_layout_reached_args
        )

    def check_goal(self, evaluate_fn):
        return evaluate_state(self.goal_conditions, evaluate_fn)

    @cached_property
    def natural_language_initial_conditions(self):
        return gen_natural_language_conditions(self.conditions.parsed_initial_conditions)

    @cached_property
    def natural_language_goal_conditions(self):
        return gen_natural_language_conditions(self.conditions.parsed_goal_conditions)

    @property
    def task(self):
        return self.problem


def problem_path(activity: str, definition_id: int, root: str | None = None) -> str:
    return os.path.join(root or ACTIVITY_DEFINITIONS_ROOT, activity, f"problem{int(definition_id)}.bddl")


def load_problem(activity: str, definition_id: int, root: str | None = None) -> ISGProblem:
    """Load ``problem<definition_id>.bddl`` of ``activity``."""
    path = problem_path(activity, definition_id, root)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No iSpatialGym problem file at {path}")
    return parse_problem_file(path, activity=activity, definition_id=int(definition_id))


def parse_problem_file(path: str, activity: str | None = None, definition_id: int | None = None) -> ISGProblem:
    if activity is None:
        activity = os.path.basename(os.path.dirname(path))
    if definition_id is None:
        m = PROBLEM_FILE_RE.match(os.path.basename(path))
        definition_id = int(m.group(1)) if m else None
    problem = parse_problem_tokens(scan_tokens(filename=path), activity=activity, definition_id=definition_id)
    problem.path = path
    return problem


def parse_problem_string(text: str, activity: str, definition_id: int) -> ISGProblem:
    return parse_problem_tokens(scan_tokens(string=text), activity=activity, definition_id=definition_id)


def parse_problem_tokens(tokens, activity: str, definition_id: int | None) -> ISGProblem:
    if not isinstance(tokens, list) or not tokens or tokens[0] != "define":
        raise ISGProblemError("Problem file must start with (define ...)")
    problem_name = domain = variant = init = goal = None
    objects: dict[str, list[str]] = {}
    groups: dict[str, list[str]] = {}
    measure: list = []
    order: list[str] = []
    for section in tokens[1:]:
        if not isinstance(section, list) or not section:
            raise ISGProblemError(f"Malformed section {section!r}")
        tag = section[0]
        if tag in order:
            raise ISGProblemError(f"Duplicate section {tag}")
        order.append(tag)
        if tag == "problem":
            problem_name = section[-1]
        elif tag == ":domain":
            domain = section[-1]
        elif tag == ":variant":
            if len(section) != 2 or not isinstance(section[1], str):
                raise ISGProblemError(f"(:variant ...) must hold exactly one symbol, got {section[1:]!r}")
            variant = section[1]
            if not VARIANT_RE.match(variant):
                raise ISGProblemError(f"Invalid variant {variant!r}")
        elif tag == ":objects":
            objects = _parse_typed_list(section[1:], ":objects")
        elif tag == ":groups":
            groups = _parse_typed_list(section[1:], ":groups")
        elif tag == ":init":
            init = section[1:]
        elif tag == ":goal":
            if len(section) != 2:
                raise ISGProblemError("(:goal ...) must hold exactly one expression")
            expr = section[1]
            goal = expr[1:] if isinstance(expr, list) and expr and expr[0] == "and" else [expr]
        elif tag == ":measure":
            measure = section[1:]
        elif tag != ":requirements":
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
    name_activity, name_id = split_problem_name(problem_name)
    if name_activity != activity or (definition_id is not None and name_id != int(definition_id)):
        expected = f"{activity}-{name_id if definition_id is None else definition_id}"
        raise ISGProblemError(f"Problem name {problem_name!r} does not match {expected!r} (file naming convention)")

    problem = ISGProblem(
        activity=activity,
        definition_id=name_id,
        problem_name=problem_name,
        domain=domain,
        variant=variant,
        objects=objects,
        groups=groups,
    )
    _validate_declarations(problem)
    expander = _GroupExpander(problem)
    problem.init = expander.expand_all(init, ":init")
    problem.goal = expander.expand_all(goal, ":goal")
    problem.measure = _validate_measure(problem, measure)
    _validate_conditions(problem)
    return problem


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
    seen_members: dict[str, str] = {}
    for g, members in problem.groups.items():
        if not INSTANCE_RE.match(g):
            raise ISGProblemError(f"Group name {g!r} must have the instance-name shape (e.g. targets_group.n.01_1)")
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
    """Unary ``pose_reached`` / ``layout_reached`` atoms on declared instances or groups, without duplicates."""
    symbols = set(problem.scope_symbols)
    out, seen = [], set()
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
        pred, args = lit[0], lit[1:]
        if pred not in predicates:
            raise ISGProblemError(f"{where}: unknown predicate {pred!r} for domain {DOMAIN_NAME}")
        if len(args) != len(predicates[pred]):
            raise ISGProblemError(f"{where}: predicate {pred!r} takes {len(predicates[pred])} argument(s), got {lit}")
        for a in args:
            name = _strip(a)
            # quantifier-bound variables are resolved by the compiler; undeclared constants are errors
            if name not in symbols and (where == ":init" or INSTANCE_RE.match(name)):
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


class _AndList(list):
    """A conjunction created by group expansion (flattened into the enclosing top-level section)."""


class _GroupExpander:
    def __init__(self, problem: ISGProblem):
        self.groups = problem.groups

    def _group_of(self, token: str) -> str | None:
        name = _strip(token)
        return name if name in self.groups else None

    def _members(self, token: str) -> list[str]:
        prefix = "?" if token.startswith("?") else ""
        return [prefix + m for m in self.groups[_strip(token)]]

    def expand_all(self, exprs: list, where: str) -> list:
        out = []
        for expr in exprs:
            expanded = self.expand(expr, where)
            out.extend(expanded[1:] if isinstance(expanded, _AndList) else [expanded])
        return out

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
                positions = [i for i, a in enumerate(inner[1:], start=1) if self._group_of(a)]
                if positions and inner[0] not in AGGREGATE_PREDICATES:
                    return _AndList(["and"] + [["not", lit] for lit in self._pointwise(inner, positions, where)])
                return expr
            return ["not", self.expand(inner, where)]
        if head in LOGICAL_TOKENS:
            return [head] + [self.expand(child, where) for child in expr[1:]]
        if not all(isinstance(e, str) for e in expr):
            raise ISGProblemError(f"{where}: malformed literal {expr!r}")
        positions = [i for i, a in enumerate(expr[1:], start=1) if self._group_of(a)]
        if not positions:
            return expr
        if head in AGGREGATE_PREDICATES:
            if len(expr) != 2:
                raise ISGProblemError(f"{where}: aggregate predicate {head!r} must be unary, got {expr}")
            return expr
        if head == SETTLE_PREDICATE and where == ":init":
            raise ISGProblemError(":init must not contain 'pose_settled'")
        return _AndList(["and"] + self._pointwise(expr, positions, where))

    def _pointwise(self, lit: list, positions: list[int], where: str) -> list[list]:
        if len(positions) != 1:
            raise ISGProblemError(f"{where}: a literal may reference at most one group, got {lit}")
        pos = positions[0]
        return [lit[:pos] + [member] + lit[pos + 1 :] for member in self._members(lit[pos])]
