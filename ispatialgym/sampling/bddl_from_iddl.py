"""iddl -> concrete BDDL problem text (``docs/task_bddl_formats.md``), validated with ``ispatialgym.activity.problem``.

:func:`render` replaces the roles of an :class:`~ispatialgym.sampling.iddl.IDDL` by instances:

* a ``sampleable`` role -> one ``<synset>_k`` instance (synset = BEHAVIOR-KB synset of the grounded category),
* a ``group`` role -> ``<role>_group.n.01_1`` over N declared members (one ``<synset>_k`` per member; several
  categories allowed), one ``:groups`` line; a ``:groups`` composite of the iddl (``target partner - assembly``) ->
  ``assembly_group.n.01_1`` over the member roles' instances,
* a ``scene`` role -> the bound scene object's synset (two scene roles bound to one object share one instance),
* ``floor`` -> ``floor.n.01_1`` (a scene role bound to the floor maps to the same instance), ``agent`` -> ``agent.n.01_1``.

``:init`` / ``:goal`` / ``:measure`` are copied literal by literal with the roles replaced (``?`` prefixes kept), so
the argument order of every predicate is the iddl's (``(under ?partner ?target)`` for under(small-scale),
``(under ?target ?partner)`` for under(large-scale)). The text is parsed and compiled with
:func:`ispatialgym.activity.problem.parse_problem_string` before anything is written.

Definition ids are allocated per alias in ``activity_definitions/<activity>/problems.json`` (:class:`ProblemRegistry`:
alias <-> id, contiguous with the ``problemN.bddl`` files, never reused; the shape is the record list the
``activity.problem`` registry reader accepts). :func:`bddl_alias` writes the alias with the characters the parser
accepts (``ALIAS_RE``): the full v2 alias when it matches, otherwise ``-`` -> ``__`` and the full alias is recorded
in the registry (``full_alias``).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from ispatialgym.activity import problem as _problem
from ispatialgym.activity.problem import (
    AGENT_INSTANCE,
    ALIAS_RE,
    PROBLEM_FILE_RE,
    ISGProblem,
    parse_problem_string,
)
from ispatialgym.sampling.iddl import IDDL, IMPLICIT_ROLES, LOGICAL_TOKENS, to_sexpr

FLOOR_SYNSET = "floor.n.01"
FLOOR_INSTANCE = f"{FLOOR_SYNSET}_1"
AGENT_SYNSET = "agent.n.01"
GROUP_SUFFIX = "_group.n.01_1"
REGISTRY_FILE = getattr(_problem, "PROBLEM_REGISTRY_FILE", "problems.json")
GROUNDER = "ground_isg 2.0.0"


class GroundingError(ValueError):
    pass


def group_symbol(role: str) -> str:
    """``targets`` -> ``targets_group.n.01_1`` (``docs/task_bddl_formats.md`` "Object names")."""
    return f"{role}{GROUP_SUFFIX}"


def bddl_alias(alias: str) -> str:
    """The alias as the parser accepts it: ``alias`` itself when ``ALIAS_RE`` matches, else ``-`` -> ``__``."""
    if ALIAS_RE.match(alias):
        return alias
    candidate = alias.replace("-", "__")
    if not ALIAS_RE.match(candidate):
        raise GroundingError(f"alias {alias!r} cannot be written in a form the parser accepts ({ALIAS_RE.pattern})")
    return candidate


def full_alias_of(bddl_alias_text: str) -> str:
    """Inverse of :func:`bddl_alias` for the ``__`` spelling (categories never contain a double underscore)."""
    return bddl_alias_text.replace("__", "-") if "__" in bddl_alias_text and "-" not in bddl_alias_text else bddl_alias_text


# ----------------------------------------------------------------------------------------------------------------
# Role groundings
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class RoleGrounding:
    """How one iddl role is declared in the BDDL.

    ``kind``: ``sampleable`` (one instance of ``synset``), ``group`` (``members`` = one ``(category, synset)`` per
    declared member, in declaration order), ``scene`` (one instance of ``synset``; ``same_as`` names another scene
    role bound to the same scene object, or ``floor`` when the role is bound to the floor), ``floor`` / ``agent``.
    """

    role: str
    kind: str
    category: Optional[str] = None
    synset: Optional[str] = None
    members: List[Tuple[str, str]] = field(default_factory=list)
    same_as: Optional[str] = None
    dataset: Optional[str] = None

    def __post_init__(self):
        if self.kind not in ("sampleable", "group", "scene", "floor", "agent"):
            raise GroundingError(f"role {self.role!r}: unknown grounding kind {self.kind!r}")
        if self.kind in ("sampleable", "scene") and not self.synset and not self.same_as:
            raise GroundingError(f"role {self.role!r}: a {self.kind} grounding needs a synset")
        if self.kind == "group" and not self.members:
            raise GroundingError(f"role {self.role!r}: a group grounding needs at least one member")


def floor_grounding(role: str = "floor") -> RoleGrounding:
    return RoleGrounding(role, "floor", category="floors", synset=FLOOR_SYNSET)


def agent_grounding() -> RoleGrounding:
    return RoleGrounding("agent", "agent", synset=AGENT_SYNSET)


@dataclass
class RenderedProblem:
    text: str
    problem: ISGProblem
    alias: str  # as written in the BDDL
    full_alias: str
    activity: str
    definition_id: int
    instances: Dict[str, Union[str, List[str]]]  # role -> instance / group symbol; group role -> also members under "<role>.members"
    objects: Dict[str, List[str]]  # synset -> instances (declaration order)
    groups: Dict[str, List[str]]  # group symbol -> members


# ----------------------------------------------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------------------------------------------


class _Declarations:
    def __init__(self):
        self.objects: Dict[str, List[str]] = {}
        self._counter: Dict[str, int] = {}

    def declare(self, synset: str) -> str:
        k = self._counter.get(synset, 0) + 1
        self._counter[synset] = k
        name = f"{synset}_{k}"
        self.objects.setdefault(synset, []).append(name)
        return name


def _substitute(expr, mapping: Dict[str, str], where: str):
    """Replace role tokens (with or without ``?``) by their instance / group symbols."""
    if isinstance(expr, str):
        prefix = "?" if expr.startswith("?") else ""
        role = expr[len(prefix):]
        if role not in mapping:
            raise GroundingError(f"{where}: role {role!r} has no grounding")
        return prefix + mapping[role]
    if not expr:
        raise GroundingError(f"{where}: empty expression")
    head = expr[0]
    if head in LOGICAL_TOKENS:
        return [head] + [_substitute(child, mapping, where) for child in expr[1:]]
    return [head] + [_substitute(arg, mapping, where) for arg in expr[1:]]


def rewrite_init(lit, init_predicates: Optional[Dict[Tuple[str, str], str]]):
    """``(pred A B)`` with ``(pred, B)`` in ``init_predicates`` -> ``(new_pred A B)`` (B = the role token as the iddl writes it);
    any other literal unchanged. ``{("ontop", "support"): "inside"}`` starts the objects on a support inside it (a shelf in a
    bookcase, a hallstand's shoe shelf)."""
    if not init_predicates or not isinstance(lit, list) or len(lit) != 3 or not isinstance(lit[2], str):
        return lit
    new = init_predicates.get((lit[0], lit[2].lstrip("?")))
    return [new, lit[1], lit[2]] if new else lit


def render(
    iddl: IDDL,
    groundings: Dict[str, RoleGrounding],
    definition_id: int,
    alias: str,
    activity: str,
    comment: Optional[str] = None,
    validate: bool = True,
    init_predicates: Optional[Dict[Tuple[str, str], str]] = None,
) -> RenderedProblem:
    """Render the grounded BDDL of ``iddl`` and validate it (parse + compile).

    ``groundings`` maps every explicit role of the iddl (``iddl.roles``) to a :class:`RoleGrounding`; ``floor`` and
    ``agent`` are added when absent. ``alias`` is the full v2 alias (``bddl_alias`` decides the written spelling).
    ``init_predicates`` rewrites ``:init`` literals (:func:`rewrite_init`).
    """
    groundings = dict(groundings)
    groundings.setdefault("floor", floor_grounding())
    groundings.setdefault("agent", agent_grounding())
    missing = [r for r in iddl.roles if r not in groundings]
    if missing:
        raise GroundingError(f"{iddl.subtask}: roles without a grounding: {missing}")
    extra = [r for r in groundings if r not in iddl.roles and r not in IMPLICIT_ROLES]
    if extra:
        raise GroundingError(f"{iddl.subtask}: groundings for roles the iddl does not have: {extra}")

    decl = _Declarations()
    mapping: Dict[str, str] = {}
    members_of: Dict[str, List[str]] = {}
    groups: Dict[str, List[str]] = {}

    # Declaration order: explicit roles as the iddl lists them (alias slots, :groups members, literals), then the
    # floor, then the agent (matches the shipped templates).
    order = [r for r in iddl.roles] + [r for r in ("floor", "agent") if r in groundings]
    deferred: List[str] = []
    for role in order:
        g = groundings[role]
        if g.kind == "agent":
            mapping[role] = decl.declare(AGENT_SYNSET)
            if mapping[role] != AGENT_INSTANCE:
                raise GroundingError("the agent must be the only agent.n.01 instance")
        elif g.kind == "floor":
            mapping[role] = decl.declare(FLOOR_SYNSET)
        elif g.kind == "group":
            names = [decl.declare(synset) for _, synset in g.members]
            sym = group_symbol(role)
            mapping[role] = sym
            members_of[role] = names
            groups[sym] = names
        elif g.same_as:
            deferred.append(role)
        else:
            mapping[role] = decl.declare(g.synset)  # type: ignore[arg-type]
    for role in deferred:
        target = groundings[role].same_as
        if target == "floor":
            if "floor" not in mapping:
                mapping["floor"] = decl.declare(FLOOR_SYNSET)
            mapping[role] = mapping["floor"]
        elif target in mapping:
            mapping[role] = mapping[target]
        else:
            raise GroundingError(f"{iddl.subtask}: role {role!r} is bound to the same object as {target!r}, which has no grounding")
    # iddl :groups composites (assembly) over the member roles' instances
    for comp, member_roles in iddl.groups.items():
        names: List[str] = []
        for m in member_roles:
            if m in members_of:
                names.extend(members_of[m])
            else:
                names.append(mapping[m])
        sym = group_symbol(comp)
        mapping[comp] = sym
        groups[sym] = names

    written_alias = bddl_alias(alias)
    lines = [f"(define (problem {activity}-{definition_id})", "    (:domain ispatialgym)", f"    (:alias {written_alias})"]
    if comment:
        lines.append("")
        lines.extend(f"    ; {c}" if c else "    ;" for c in comment.splitlines())
    lines.append("")
    lines.append("    (:objects")
    for synset, names in decl.objects.items():
        lines.append(f"        {' '.join(names)} - {synset}")
    lines.append("    )")
    if groups:
        lines.append("")
        lines.append("    (:groups")
        for sym, names in groups.items():
            lines.append(f"        {' '.join(names)} - {sym}")
        lines.append("    )")
    lines.append("")
    lines.append("    (:init")
    for lit in iddl.init:
        lines.append(to_sexpr(_substitute(rewrite_init(lit, init_predicates), mapping, ":init"), 2))
    lines.append("    )")
    lines.append("")
    lines.append("    (:goal")
    lines.append(to_sexpr(["and"] + [_substitute(lit, mapping, ":goal") for lit in iddl.goal], 2))
    lines.append("    )")
    if iddl.measure:
        lines.append("")
        lines.append("    (:measure")
        for lit in iddl.measure:
            lines.append(to_sexpr(_substitute(lit, mapping, ":measure"), 2))
        lines.append("    )")
    lines.append(")")
    text = "\n".join(lines) + "\n"

    problem = None
    if validate:
        problem = validate_text(text, activity, definition_id)
    instances: Dict[str, Union[str, List[str]]] = dict(mapping)
    for role, names in members_of.items():
        instances[f"{role}.members"] = list(names)
    return RenderedProblem(text, problem, written_alias, alias, activity, int(definition_id), instances, dict(decl.objects), groups)


def validate_text(text: str, activity: str, definition_id: int) -> ISGProblem:
    """Parse and compile ``text`` with the iSpatialGym parser (raises :class:`ISGProblemError` when malformed)."""
    problem = parse_problem_string(text, activity=activity, definition_id=int(definition_id))
    problem.compile()
    return problem


# ----------------------------------------------------------------------------------------------------------------
# problems.json registry
# ----------------------------------------------------------------------------------------------------------------


class ProblemRegistry:
    """``activity_definitions/<activity>/problems.json``: alias <-> definition id, contiguous, never reused.

    Shape (accepted by ``ispatialgym.activity.problem.load_problem_registry``)::

        {"activity": "alignment",
         "problems": [{"alias": "table_surface", "id": 0, "legacy": true},
                      {"alias": "alignment-axial-paperback_book-breakfast_table", "id": 3, "subtask": "alignment-axial",
                       "iddl": "alignment/axial.iddl", "grounded": "2026-09-15", "by": "ground_isg 2.0.0", ...}],
         "unregistered": [{"id": 2, "alias": "articulation", "why": "duplicate alias of problem1"}]}

    Legacy template problems (no registry entry yet) are registered from their ``(:alias ...)`` on first use so the
    ids stay contiguous; a file whose alias repeats another file's is listed under ``unregistered`` (one alias = one
    BDDL; the registry reader rejects duplicates).
    """

    def __init__(self, activity_dir: str, activity: str):
        self.activity_dir = activity_dir
        self.activity = activity
        self.records: List[Dict[str, Any]] = []
        self.unregistered: List[Dict[str, Any]] = []
        self.path = os.path.join(activity_dir, REGISTRY_FILE)

    # -- io ---------------------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, activity_dir: str, activity: Optional[str] = None) -> "ProblemRegistry":
        activity = activity or os.path.basename(os.path.normpath(activity_dir))
        reg = cls(activity_dir, activity)
        if os.path.isfile(reg.path):
            with open(reg.path) as f:
                data = json.load(f)
            recs = data.get("problems", data) if isinstance(data, dict) else data
            if isinstance(recs, dict):
                recs = [{"alias": a, "id": i} for a, i in recs.items()]
            for rec in recs:
                if not isinstance(rec, dict) or "alias" not in rec or ("id" not in rec and "definition_id" not in rec):
                    raise GroundingError(f"{reg.path}: malformed record {rec!r}")
                pid = rec.get("id", rec.get("definition_id"))
                rec = {"alias": rec["alias"], "id": int(pid), **{k: v for k, v in rec.items() if k not in ("alias", "id", "definition_id")}}
                reg.records.append(rec)
            if isinstance(data, dict):
                reg.unregistered = list(data.get("unregistered") or [])
        reg._absorb_files()
        reg._check()
        return reg

    def _existing_ids(self) -> List[int]:
        if not os.path.isdir(self.activity_dir):
            return []
        ids = sorted(int(m.group(1)) for m in (PROBLEM_FILE_RE.match(f) for f in os.listdir(self.activity_dir)) if m)
        if ids != list(range(len(ids))):
            raise GroundingError(f"{self.activity_dir}: problem files are not contiguous: {ids}")
        return ids

    def _absorb_files(self) -> None:
        """Register legacy problem files (present on disk, absent from the registry) from their ``(:alias ...)``."""
        known_ids = {r["id"] for r in self.records} | {u["id"] for u in self.unregistered}
        for pid in self._existing_ids():
            if pid in known_ids:
                continue
            alias = _read_alias(os.path.join(self.activity_dir, f"problem{pid}.bddl"))
            if alias is None:
                self.unregistered.append({"id": pid, "alias": None, "why": "the problem declares no alias"})
            elif any(r["alias"] == alias for r in self.records):
                other = next(r["id"] for r in self.records if r["alias"] == alias)
                self.unregistered.append({"id": pid, "alias": alias, "why": f"duplicate alias of problem{other}"})
            else:
                self.records.append({"alias": alias, "id": pid, "legacy": True})
        self.records.sort(key=lambda r: r["id"])

    def _check(self) -> None:
        ids = [r["id"] for r in self.records]
        if len(set(ids)) != len(ids):
            raise GroundingError(f"{self.path}: an id is registered twice: {sorted(i for i in ids if ids.count(i) > 1)}")
        aliases = [r["alias"] for r in self.records]
        if len(set(aliases)) != len(aliases):
            raise GroundingError(f"{self.path}: an alias is registered twice: {sorted(a for a in aliases if aliases.count(a) > 1)}")

    def save(self) -> str:
        os.makedirs(self.activity_dir, exist_ok=True)
        data: Dict[str, Any] = {"activity": self.activity, "problems": sorted(self.records, key=lambda r: r["id"])}
        if self.unregistered:
            data["unregistered"] = sorted(self.unregistered, key=lambda r: r["id"])
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1)
            f.write("\n")
        os.replace(tmp, self.path)
        return self.path

    # -- lookups ------------------------------------------------------------------------------------------------------
    def record(self, alias: str) -> Optional[Dict[str, Any]]:
        for r in self.records:
            if r["alias"] == alias or r.get("full_alias") == alias:
                return r
        return None

    def id_of(self, alias: str) -> Optional[int]:
        r = self.record(alias)
        return None if r is None else int(r["id"])

    def next_id(self) -> int:
        ids = [r["id"] for r in self.records] + [u["id"] for u in self.unregistered] + self._existing_ids()
        return (max(ids) + 1) if ids else 0

    def allocate(self, alias: str, bddl_alias_text: Optional[str] = None, **meta) -> int:
        """The id of ``alias`` (existing record reused and its metadata refreshed; else the next free id)."""
        rec = self.record(alias)
        written = bddl_alias_text or alias
        if rec is None:
            rec = {"alias": written, "id": self.next_id()}
            self.records.append(rec)
        else:
            rec["alias"] = written
        if written != alias:
            rec["full_alias"] = alias
        else:
            rec.pop("full_alias", None)
        rec.pop("legacy", None)
        for k, v in meta.items():
            if v is None:
                rec.pop(k, None)
            else:
                rec[k] = v
        ordered = {"alias": rec["alias"], "id": int(rec["id"])}
        ordered.update({k: v for k, v in rec.items() if k not in ordered})
        self.records[self.records.index(rec)] = ordered
        self.records.sort(key=lambda r: r["id"])
        self._check()
        return int(ordered["id"])

    def aliases(self) -> Dict[str, int]:
        return {r.get("full_alias", r["alias"]): int(r["id"]) for r in self.records}


def _read_alias(path: str) -> Optional[str]:
    """``(:alias X)`` of a problem file, without parsing the rest (comments stripped)."""
    with open(path) as f:
        text = f.read()
    text = "\n".join(line.split(";", 1)[0] for line in text.splitlines())
    m = re.search(r"\(:alias\s+([^\s()]+)\s*\)", text)
    return m.group(1) if m else None


# ----------------------------------------------------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------------------------------------------------


def write_problem(
    bddl_root: str,
    activity: str,
    iddl: IDDL,
    groundings: Dict[str, RoleGrounding],
    alias: str,
    comment: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
    init_predicates: Optional[Dict[Tuple[str, str], str]] = None,
) -> RenderedProblem:
    """Allocate the id of ``alias`` in ``<bddl_root>/<activity>/problems.json``, render, validate and write
    ``problem<N>.bddl`` (an existing problem of the same alias is rewritten under its id). ``meta`` is recorded in
    the registry record (subtask, iddl, pick, slot, ...); ``init_predicates`` see :func:`rewrite_init`."""
    activity_dir = os.path.join(bddl_root, activity)
    registry = ProblemRegistry.load(activity_dir, activity)
    written_alias = bddl_alias(alias)
    existing = registry.record(alias)
    # ``grounded`` = the first grounding date of the alias (a re-grounding keeps it, so the registry only changes
    # where the grounding changed)
    grounded = existing.get("grounded") if existing else None
    definition_id = registry.allocate(alias, written_alias, grounded=grounded or _dt.date.today().isoformat(), by=GROUNDER, **(meta or {}))
    rendered = render(iddl, groundings, definition_id, alias, activity, comment=comment, init_predicates=init_predicates)
    if not dry_run:
        os.makedirs(activity_dir, exist_ok=True)
        path = os.path.join(activity_dir, f"problem{definition_id}.bddl")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(rendered.text)
        os.replace(tmp, path)
        registry.save()
    return rendered


__all__ = [
    "AGENT_SYNSET",
    "FLOOR_INSTANCE",
    "FLOOR_SYNSET",
    "GROUNDER",
    "GroundingError",
    "ProblemRegistry",
    "REGISTRY_FILE",
    "RenderedProblem",
    "RoleGrounding",
    "agent_grounding",
    "bddl_alias",
    "floor_grounding",
    "full_alias_of",
    "group_symbol",
    "render",
    "validate_text",
    "write_problem",
]
