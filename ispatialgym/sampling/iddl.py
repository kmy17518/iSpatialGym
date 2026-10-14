"""iddl — parser for the iSpatialGym domain definition language (``ispatialgym/assets/iddl/README.md``).

An iddl file is a BDDL problem skeleton written over bare **roles** instead of synset instances::

    (define (iddl <subtask-id>)
        (:family <family>)
        (:variant <template>)                    ; {role} slots, filled with category names by ground_isg
        (:groups <role> <role> - <group-role>)   ; optional composite groups (attachment assembly)
        (:require (grasps <role> <pipeline>) (contacts <role>) (handle <role>))   ; optional
        (:init <literals>)                       ; no ? prefix
        (:goal (and <literals>))                 ; ? prefix; pose_settled atoms are direct children of the and
        (:measure <literals>)                    ; optional: pose_reached / layout_reached, reported but not gating
    )

The parser knows nothing about categories, models or role kinds — those are the icl's (``roles/candidates.csv``);
it collects the roles from the variant slots, the ``:groups`` members and the literal arguments (``agent`` and ``floor``
are implicit, see :data:`IMPLICIT_ROLES`) and validates the grammar: known predicates with the right arity (the
``ispatialgym`` BDDL domain), ``?`` prefixes per section, ``pose_settled`` only at the top level of the goal,
``:measure`` restricted to ``pose_reached`` / ``layout_reached``, ``:require`` forms, unique sections.

Entry points: :func:`parse` (text), :func:`parse_file`, :func:`load_all` (the 21 files keyed by subtask id).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ispatialgym import ASSETS_ROOT

IDDL_ROOT = os.path.join(ASSETS_ROOT, "iddl")

#: The seven task families (= activity directories, ``docs/instance_sampling_spec.md`` §1).
FAMILIES = (
    "camera_relocalization",
    "object_scaling",
    "mental_rotation",
    "disembedding",
    "configuration_matching",
    "alignment",
    "inter_object_relation_satisfaction",
)

#: Roles every file has without declaring them: the robot and the task room's floor.
IMPLICIT_ROLES = ("agent", "floor")

#: Predicates of the ``ispatialgym`` BDDL domain and their arity (checked against ``bddl`` in the tests).
PREDICATES: Dict[str, int] = {
    "open": 1,
    "ontop": 2,
    "nextto": 2,
    "under": 2,
    "touching": 2,
    "inside": 2,
    "attached": 2,
    "grasped": 2,
    "pose_reached": 1,
    "pose_settled": 1,
    "layout_reached": 1,
}
AGGREGATE_PREDICATES = frozenset({"pose_reached", "layout_reached"})
MEASURE_PREDICATES = AGGREGATE_PREDICATES
SETTLE_PREDICATE = "pose_settled"
#: Evaluator-configured predicates that never appear in ``:init`` (targets come from the TRO).
FORBIDDEN_IN_INIT = frozenset({"pose_reached", "pose_settled", "layout_reached"})
LOGICAL_TOKENS = frozenset({"and", "or", "not", "imply", "forall", "exists", "forn", "forpairs", "fornpairs"})

#: ``:require`` forms: kind -> admissible pipelines (None = the form takes no pipeline argument).
GRASP_PIPELINES = ("floating", "tabletop", "insert", "attachment")
REQUIRE_KINDS: Dict[str, Optional[Tuple[str, ...]]] = {"grasps": GRASP_PIPELINES, "contacts": None, "handle": None}

SLOT_RE = re.compile(r"\{([a-z][a-z0-9_]*)\}")
ROLE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
SUBTASK_RE = re.compile(r"^[a-z][a-z0-9_]*(?:-[a-z][a-z0-9_]*)?$")


class IDDLError(ValueError):
    """Raised for a malformed iddl file."""


# ----------------------------------------------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Require:
    """One ``:require`` line: ``(grasps <role> <pipeline>)``, ``(contacts <role>)`` or ``(handle <role>)``."""

    kind: str
    role: str
    pipeline: Optional[str] = None

    @property
    def form(self) -> str:
        """The icl ``require`` column value this line is checked against (``grasps:tabletop``, ``contacts``, ``handle``)."""
        return f"{self.kind}:{self.pipeline}" if self.kind == "grasps" else self.kind

    @property
    def missing_reason(self) -> str:
        """The customs blacklist reason of a model without an admissible row (``missing_grasps:<pipeline>`` ...)."""
        return f"missing_{self.kind}:{self.pipeline}" if self.kind == "grasps" else f"missing_{self.kind}"

    def to_bddl(self) -> str:
        return f"({self.kind} {self.role} {self.pipeline})" if self.pipeline else f"({self.kind} {self.role})"

    def customs_value(self):
        """The ``roles.<role>.require`` entry of the customs list: ``{"grasps": "<pipeline>"}`` / ``{"contacts": true}``."""
        return {self.kind: self.pipeline if self.kind == "grasps" else True}


@dataclass
class IDDL:
    """A parsed iddl file (roles as bare names; literals as nested lists exactly as written)."""

    subtask: str
    family: str
    variant_template: str
    groups: Dict[str, List[str]] = field(default_factory=dict)  # composite group role -> member roles
    require: List[Require] = field(default_factory=list)
    init: List[list] = field(default_factory=list)
    goal: List[list] = field(default_factory=list)  # top-level conjuncts of (:goal (and ...))
    measure: List[list] = field(default_factory=list)
    path: Optional[str] = None
    comment: str = ""  # the leading ``;`` comment block of the file

    # ------------------------------------------------------------------ roles

    @property
    def variant_slots(self) -> List[str]:
        """Roles named by ``{role}`` slots of the variant template, in template order."""
        return SLOT_RE.findall(self.variant_template)

    @property
    def composite_groups(self) -> List[str]:
        """Group roles declared by ``:groups`` (their members are roles; the group itself is not grounded to a model)."""
        return list(self.groups)

    def literal_roles(self) -> List[str]:
        """Every symbol used as a literal argument (``?`` stripped), in order of first appearance."""
        out: List[str] = []
        for lit in self.init + self.goal + self.measure:
            for sym in _flat_args(lit):
                if sym not in out:
                    out.append(sym)
        return out

    @property
    def roles(self) -> List[str]:
        """Explicit roles to ground: variant slots, ``:groups`` members, literal arguments — without the implicit
        ``agent`` / ``floor`` and without the composite group names. Order of first appearance."""
        out: List[str] = []
        seq: List[str] = list(self.variant_slots)
        for members in self.groups.values():
            seq.extend(members)
        seq.extend(self.literal_roles())
        for r in seq:
            if r in IMPLICIT_ROLES or r in self.groups or r in out:
                continue
            out.append(r)
        return out

    @property
    def all_roles(self) -> List[str]:
        """Explicit roles, then the composite groups, then the implicit roles that appear in the literals."""
        used = set(self.literal_roles()) | set(self.variant_slots)
        return self.roles + self.composite_groups + [r for r in IMPLICIT_ROLES if r in used]

    def require_of(self, role: str) -> List[Require]:
        return [r for r in self.require if r.role == role]

    def is_composite(self, role: str) -> bool:
        return role in self.groups

    def literals_with(self, role: str, section: str = "goal") -> List[list]:
        """Literals of ``section`` (``init`` / ``goal`` / ``measure``) whose arguments name ``role``."""
        src = {"init": self.init, "goal": self.goal, "measure": self.measure}[section]
        return [lit for lit in src if role in _flat_args(lit)]

    def predicates_on(self, role: str) -> Set[str]:
        """Predicate names applied to ``role`` anywhere in the file (``pose_reached``, ``ontop``, ...)."""
        out: Set[str] = set()
        for lit in self.init + self.goal + self.measure:
            for pred, args in _atoms(lit):
                if role in (_strip(a) for a in args):
                    out.add(pred)
        return out

    def pose_evaluated_roles(self) -> List[str]:
        """Roles (incl. composite groups) with a ``pose_reached`` atom in the goal or in ``:measure``."""
        out: List[str] = []
        for lit in self.goal + self.measure:
            for pred, args in _atoms(lit):
                if pred == "pose_reached":
                    r = _strip(args[0])
                    if r not in out:
                        out.append(r)
        return out

    # ------------------------------------------------------------------ variant

    def fill_variant(self, values: Mapping[str, str]) -> str:
        """Fill every ``{role}`` slot from ``values``; a missing slot is an error (no silent placeholder)."""
        missing = [s for s in self.variant_slots if s not in values]
        if missing:
            raise IDDLError(f"{self.subtask}: variant slots without a value: {missing}")
        return SLOT_RE.sub(lambda m: values[m.group(1)], self.variant_template)

    # ------------------------------------------------------------------ serialization

    def to_text(self) -> str:
        """Re-serialize the parsed file (comments dropped) — round-trips through :func:`parse`."""
        lines = [f"(define (iddl {self.subtask})", f"    (:family {self.family})", f"    (:variant {self.variant_template})"]
        if self.groups:
            lines.append("    (:groups")
            for g, members in self.groups.items():
                lines.append(f"        {' '.join(members)} - {g}")
            lines.append("    )")
        if self.require:
            lines.append("    (:require")
            lines.extend(f"        {r.to_bddl()}" for r in self.require)
            lines.append("    )")
        lines.append("    (:init")
        lines.extend(to_sexpr(lit, 2) for lit in self.init)
        lines.append("    )")
        lines.append("    (:goal")
        lines.append(to_sexpr(["and"] + list(self.goal), 2))
        lines.append("    )")
        if self.measure:
            lines.append("    (:measure")
            lines.extend(to_sexpr(lit, 2) for lit in self.measure)
            lines.append("    )")
        lines.append(")")
        return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------------------------
# S-expression helpers
# ----------------------------------------------------------------------------------------------------------------


def tokenize(text: str) -> List[str]:
    """Split iddl text into ``(``, ``)`` and symbol tokens; ``;`` starts a comment to the end of the line."""
    tokens: List[str] = []
    for raw in text.splitlines():
        line = raw.split(";", 1)[0]
        for tok in line.replace("(", " ( ").replace(")", " ) ").split():
            tokens.append(tok)
    return tokens


def read_sexpr(tokens: Sequence[str]):
    """Parse the token list into one nested list (the file holds exactly one top-level form)."""
    pos = 0

    def read():
        nonlocal pos
        if pos >= len(tokens):
            raise IDDLError("unexpected end of file (unbalanced parentheses)")
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            out = []
            while True:
                if pos >= len(tokens):
                    raise IDDLError("unexpected end of file (missing ')')")
                if tokens[pos] == ")":
                    pos += 1
                    return out
                out.append(read())
        if tok == ")":
            raise IDDLError("unexpected ')'")
        return tok

    expr = read()
    if pos != len(tokens):
        raise IDDLError(f"trailing tokens after the top-level form: {tokens[pos:pos + 5]}")
    return expr


def to_sexpr(expr, indent: int = 0) -> str:
    """Serialize a nested list back to text (one literal per line, nested logical forms indented)."""
    pad = "    " * indent
    if isinstance(expr, str):
        return pad + expr
    if all(isinstance(e, str) for e in expr):
        return pad + "(" + " ".join(expr) + ")"
    head = expr[0]
    if head == "not" and len(expr) == 2 and all(isinstance(e, str) for e in expr[1]):
        return pad + "(not (" + " ".join(expr[1]) + "))"
    lines = [pad + f"({head}"]
    for child in expr[1:]:
        lines.append(to_sexpr(child, indent + 1))
    lines.append(pad + ")")
    return "\n".join(lines)


def _strip(token: str) -> str:
    return token[1:] if token.startswith("?") else token


def _atoms(expr) -> Iterable[Tuple[str, List[str]]]:
    """Yield ``(predicate, args)`` of every atom below ``expr`` (logical connectives are descended)."""
    if not isinstance(expr, list) or not expr:
        return
    head = expr[0]
    if head in LOGICAL_TOKENS:
        for child in expr[1:]:
            yield from _atoms(child)
    else:
        yield head, [a for a in expr[1:] if isinstance(a, str)]


def _flat_args(expr) -> List[str]:
    return [_strip(a) for _, args in _atoms(expr) for a in args]


# ----------------------------------------------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------------------------------------------


def _leading_comment(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith(";"):
            lines.append(s.lstrip(";").strip())
        elif s:
            break
    return "\n".join(lines)


def parse(text: str, path: Optional[str] = None) -> IDDL:
    """Parse iddl ``text`` into an :class:`IDDL`; raises :class:`IDDLError` on every grammar violation."""
    where = path or "<string>"
    expr = read_sexpr(tokenize(text))
    if not isinstance(expr, list) or len(expr) < 2 or expr[0] != "define":
        raise IDDLError(f"{where}: the file must be one (define (iddl <subtask>) ...) form")
    head = expr[1]
    if not (isinstance(head, list) and len(head) == 2 and head[0] == "iddl" and isinstance(head[1], str)):
        raise IDDLError(f"{where}: expected (iddl <subtask-id>) after define, got {head!r}")
    subtask = head[1]
    if not SUBTASK_RE.match(subtask):
        raise IDDLError(f"{where}: invalid subtask id {subtask!r} (expected <family>[-<subtask>])")

    sections: Dict[str, list] = {}
    order: List[str] = []
    for sec in expr[2:]:
        if not isinstance(sec, list) or not sec or not isinstance(sec[0], str) or not sec[0].startswith(":"):
            raise IDDLError(f"{where}: malformed section {sec!r}")
        tag = sec[0]
        if tag in sections:
            raise IDDLError(f"{where}: duplicate section {tag}")
        sections[tag] = sec[1:]
        order.append(tag)
    unknown = [t for t in order if t not in (":family", ":variant", ":groups", ":require", ":init", ":goal", ":measure")]
    if unknown:
        raise IDDLError(f"{where}: unknown section(s) {unknown}")
    for required in (":family", ":variant", ":init", ":goal"):
        if required not in sections:
            raise IDDLError(f"{where}: missing section {required}")

    # ---- family / variant
    fam = sections[":family"]
    if len(fam) != 1 or not isinstance(fam[0], str):
        raise IDDLError(f"{where}: (:family ...) must hold one symbol, got {fam!r}")
    family = fam[0]
    if family not in FAMILIES:
        raise IDDLError(f"{where}: unknown family {family!r}; expected one of {FAMILIES}")
    if not (subtask == family or subtask.startswith(family + "-")):
        raise IDDLError(f"{where}: subtask id {subtask!r} must be the family or '<family>-<name>' for family {family!r}")
    al = sections[":variant"]
    if len(al) != 1 or not isinstance(al[0], str):
        raise IDDLError(f"{where}: (:variant ...) must hold one template symbol, got {al!r}")
    variant_template = al[0]
    if not (variant_template == subtask or variant_template.startswith(subtask + "-")):
        raise IDDLError(f"{where}: variant template {variant_template!r} must start with the subtask id {subtask!r}")
    rest = variant_template[len(subtask):]
    if rest and not re.fullmatch(r"(?:-\{[a-z][a-z0-9_]*\})+", rest):
        raise IDDLError(f"{where}: variant template {variant_template!r} must be '<subtask>[-{{role}}...]'")
    slots = SLOT_RE.findall(variant_template)
    if len(set(slots)) != len(slots):
        raise IDDLError(f"{where}: variant template repeats a slot: {slots}")

    # ---- groups
    groups: Dict[str, List[str]] = {}
    if ":groups" in sections:
        groups = _parse_typed_list(sections[":groups"], where)
        for g, members in groups.items():
            if not ROLE_RE.match(g) or any(not ROLE_RE.match(m) for m in members):
                raise IDDLError(f"{where}: :groups names must be bare roles, got {g!r}: {members}")
            if g in members or g in IMPLICIT_ROLES:
                raise IDDLError(f"{where}: composite group {g!r} cannot be its own member or an implicit role")
            if len(set(members)) != len(members):
                raise IDDLError(f"{where}: composite group {g!r} lists a member twice")
        for g, members in groups.items():
            for m in members:
                if m in groups:
                    raise IDDLError(f"{where}: composite group {g!r} nests group {m!r}; groups do not nest")

    # ---- require
    require: List[Require] = []
    for form in sections.get(":require", []):
        if not isinstance(form, list) or not form or not all(isinstance(t, str) for t in form):
            raise IDDLError(f"{where}: malformed :require form {form!r}")
        kind = form[0]
        if kind not in REQUIRE_KINDS:
            raise IDDLError(f"{where}: unknown :require form {kind!r}; expected one of {sorted(REQUIRE_KINDS)}")
        pipelines = REQUIRE_KINDS[kind]
        if pipelines is None:
            if len(form) != 2:
                raise IDDLError(f"{where}: ({kind} <role>) takes exactly one role, got {form}")
            req = Require(kind, form[1])
        else:
            if len(form) != 3:
                raise IDDLError(f"{where}: ({kind} <role> <pipeline>) takes a role and a pipeline, got {form}")
            if form[2] not in pipelines:
                raise IDDLError(f"{where}: unknown {kind} pipeline {form[2]!r}; expected one of {pipelines}")
            req = Require(kind, form[1], form[2])
        if not ROLE_RE.match(req.role) or req.role in IMPLICIT_ROLES:
            raise IDDLError(f"{where}: :require role {req.role!r} must be an explicit role")
        if req in require:
            raise IDDLError(f"{where}: duplicate :require line {req.to_bddl()}")
        require.append(req)

    # ---- init / goal / measure
    init = [_check_literal(lit, ":init", where, prefixed=False, top_level=True) for lit in sections[":init"]]
    goal_sec = sections[":goal"]
    if len(goal_sec) != 1 or not isinstance(goal_sec[0], list) or not goal_sec[0] or goal_sec[0][0] != "and":
        raise IDDLError(f"{where}: (:goal ...) must hold exactly one (and ...) expression")
    goal = [_check_literal(lit, ":goal", where, prefixed=True, top_level=True) for lit in goal_sec[0][1:]]
    if not goal:
        raise IDDLError(f"{where}: the goal (and ...) is empty")
    if not any(not (isinstance(g, list) and g and g[0] == SETTLE_PREDICATE) for g in goal):
        raise IDDLError(f"{where}: the goal must contain at least one non-pose_settled condition")
    measure: List[list] = []
    for lit in sections.get(":measure", []):
        lit = _check_literal(lit, ":measure", where, prefixed=True, top_level=True)
        if not all(isinstance(t, str) for t in lit) or lit[0] not in MEASURE_PREDICATES:
            raise IDDLError(f"{where}: :measure only accepts flat {sorted(MEASURE_PREDICATES)} atoms, got {lit}")
        if lit in measure:
            raise IDDLError(f"{where}: :measure lists {lit} twice")
        measure.append(lit)
    gated = {(p, _strip(a[0])) for g in goal for p, a in _atoms(g) if p in MEASURE_PREDICATES}
    for lit in measure:
        if (lit[0], _strip(lit[1])) in gated:
            raise IDDLError(f"{where}: {lit} is both a gating goal atom and a :measure atom")

    iddl = IDDL(
        subtask=subtask,
        family=family,
        variant_template=variant_template,
        groups=groups,
        require=require,
        init=init,
        goal=goal,
        measure=measure,
        path=path,
        comment=_leading_comment(text),
    )
    _check_roles(iddl, where)
    return iddl


def _parse_typed_list(tokens: Sequence, where: str) -> Dict[str, List[str]]:
    """``MEMBER MEMBER - GROUP ...`` -> ``{group: [members]}`` (declaration order)."""
    result: Dict[str, List[str]] = {}
    pending: List[str] = []
    toks = list(tokens)
    while toks:
        tok = toks.pop(0)
        if not isinstance(tok, str):
            raise IDDLError(f"{where}: :groups: unexpected nested list {tok!r}")
        if tok == "-":
            if not pending or not toks:
                raise IDDLError(f"{where}: :groups: '-' must follow members and precede the group name")
            name = toks.pop(0)
            if not isinstance(name, str):
                raise IDDLError(f"{where}: :groups: group name must be a symbol, got {name!r}")
            if name in result:
                raise IDDLError(f"{where}: :groups: group {name!r} declared twice")
            result[name] = pending
            pending = []
        else:
            pending.append(tok)
    if pending:
        raise IDDLError(f"{where}: :groups: members {pending} have no group ('- GROUP' missing)")
    return result


def _check_literal(lit, section: str, where: str, prefixed: bool, top_level: bool):
    """Validate one literal / logical form of ``section``: predicate + arity, ``?`` prefixes, pose_settled placement."""
    if not isinstance(lit, list) or not lit:
        raise IDDLError(f"{where}: {section}: malformed expression {lit!r}")
    head = lit[0]
    if not isinstance(head, str):
        raise IDDLError(f"{where}: {section}: malformed expression {lit!r}")
    if head in LOGICAL_TOKENS:
        if head in ("forall", "exists", "forn", "forpairs", "fornpairs"):
            raise IDDLError(f"{where}: {section}: quantifiers are not part of iddl ({lit})")
        if head == "not" and len(lit) != 2:
            raise IDDLError(f"{where}: {section}: (not ...) takes one expression, got {lit}")
        for child in lit[1:]:
            _check_literal(child, section, where, prefixed, top_level=False)
        return lit
    if head not in PREDICATES:
        raise IDDLError(f"{where}: {section}: unknown predicate {head!r} in {lit}")
    args = lit[1:]
    if len(args) != PREDICATES[head] or not all(isinstance(a, str) for a in args):
        raise IDDLError(f"{where}: {section}: {head!r} takes {PREDICATES[head]} argument(s), got {lit}")
    for a in args:
        if prefixed and not a.startswith("?"):
            raise IDDLError(f"{where}: {section}: goal / measure arguments carry the ? prefix, got {a!r} in {lit}")
        if not prefixed and a.startswith("?"):
            raise IDDLError(f"{where}: {section}: :init arguments have no ? prefix, got {a!r} in {lit}")
        if not ROLE_RE.match(_strip(a)):
            raise IDDLError(f"{where}: {section}: {a!r} is not a role name in {lit}")
    if section == ":init" and head in FORBIDDEN_IN_INIT:
        raise IDDLError(f"{where}: :init must not contain {head!r} (targets are configured from the TRO)")
    if head == SETTLE_PREDICATE and not (section == ":goal" and top_level):
        raise IDDLError(f"{where}: {section}: (pose_settled ...) must be a direct child of the goal (and ...)")
    if head == "grasped" and _strip(args[0]) != "agent":
        raise IDDLError(f"{where}: {section}: the first argument of 'grasped' must be the agent, got {lit}")
    return lit


def _check_roles(iddl: IDDL, where: str) -> None:
    roles = set(iddl.roles)
    for r in iddl.require:
        if r.role not in roles:
            raise IDDLError(f"{where}: :require names {r.role!r}, which is not a role of the file (roles: {sorted(roles)})")
    used = set(iddl.literal_roles()) | set(iddl.variant_slots)
    for g, members in iddl.groups.items():
        if g not in used:
            raise IDDLError(f"{where}: composite group {g!r} is never used in a literal")
    if "agent" not in set(iddl.literal_roles()):
        raise IDDLError(f"{where}: the agent must appear in the literals ((ontop agent floor))")
    for slot in iddl.variant_slots:
        if slot in IMPLICIT_ROLES or slot in iddl.groups:
            raise IDDLError(f"{where}: variant slot {{{slot}}} must be an explicit non-composite role")


# ----------------------------------------------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------------------------------------------


def parse_file(path: str) -> IDDL:
    with open(path) as f:
        text = f.read()
    return parse(text, path=path)


def expected_subtask_id(path: str, root: Optional[str] = None) -> str:
    """``<root>/<family>/<name>.iddl`` -> ``<family>-<name>`` (``<family>`` when the file is named like the family)."""
    family = os.path.basename(os.path.dirname(os.path.abspath(path)))
    name = os.path.splitext(os.path.basename(path))[0]
    return family if name == family else f"{family}-{name}"


def relative_path(path: str, root: Optional[str] = None) -> str:
    """The ``iddl`` value of a customs entry: ``<family>/<name>.iddl`` relative to the iddl root."""
    root = IDDL_ROOT if root is None else root
    return os.path.relpath(os.path.abspath(path), os.path.abspath(root))


def iter_files(root: Optional[str] = None) -> List[str]:
    root = IDDL_ROOT if root is None else root
    out: List[str] = []
    for family in sorted(os.listdir(root)):
        d = os.path.join(root, family)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith(".iddl"):
                out.append(os.path.join(d, name))
    return out


def load_all(root: Optional[str] = None) -> Dict[str, IDDL]:
    """Parse every ``<root>/<family>/*.iddl`` and return them keyed by subtask id.

    The directory must be the family, the subtask id must be ``<family>-<file stem>`` (or the family for a file
    named like it) and ids must be unique — the file layout and the ids are one contract.
    """
    root = IDDL_ROOT if root is None else root
    out: Dict[str, IDDL] = {}
    for path in iter_files(root):
        iddl = parse_file(path)
        family = os.path.basename(os.path.dirname(path))
        if iddl.family != family:
            raise IDDLError(f"{path}: (:family {iddl.family}) but the file lives under {family!r}")
        expected = expected_subtask_id(path, root)
        if iddl.subtask != expected:
            raise IDDLError(f"{path}: subtask id {iddl.subtask!r} does not match the file layout ({expected!r})")
        if iddl.subtask in out:
            raise IDDLError(f"{path}: subtask id {iddl.subtask!r} already defined by {out[iddl.subtask].path}")
        out[iddl.subtask] = iddl
    return out


__all__ = [
    "AGGREGATE_PREDICATES",
    "FAMILIES",
    "GRASP_PIPELINES",
    "IDDL",
    "IDDLError",
    "IDDL_ROOT",
    "IMPLICIT_ROLES",
    "MEASURE_PREDICATES",
    "PREDICATES",
    "REQUIRE_KINDS",
    "Require",
    "expected_subtask_id",
    "iter_files",
    "load_all",
    "parse",
    "parse_file",
    "relative_path",
    "to_sexpr",
    "tokenize",
]
