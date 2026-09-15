"""ground_isg — iddl + icl -> grounded BDDL problems + task customs list (``docs/instance_sampling_spec.md`` §9).

For every iddl file (``sampling/iddl.py``) and every category combination admitted by the icl
(``roles/constraints_compatible.csv``; ``--representative-only`` restricts the candidates to the picks of
``roles/representatives.csv``, ``--test-only`` to the ``test = 1`` rows of ``roles/candidates.csv``, ``--all`` takes
every candidate) this module

1. applies the row admissibility of spec §2.2 to the candidate models — ``ready`` (``not_ready``), the iddl
   ``:require`` lines (``missing_grasps:<pipeline>`` / ``missing_handle`` / ``missing_contacts``), the alignment frame
   record (``missing_frame``), symmetric mental-rotation targets (``symmetric_target``), obstacle naturalness
   (``not_natural_in_scene``) — plus the §3.12 look-alike pair rule (``lookalike_pair`` / ``lookalike_of_target``)
   and the global held-out exclusion (every ``heldout`` slot model is blacklisted in every role, reason ``heldout``);
2. renders and validates the BDDL (``sampling/bddl_from_iddl.py``; ids from ``activity_definitions/<activity>/
   problems.json``);
3. writes the customs entry (``sampling/customs.py``, spec §2.3): ``roles``, per-scene ``bindings`` of the scene
   roles, ``whitelist`` (the pick's models; the whole admissible pool with ``--full-whitelists``; always the whole
   admissible pool for group roles), ``blacklist`` with reasons, ``generation`` seeded from the existing entry or
   ``sampling/generation_defaults.py``, ``constraints`` (empty, except the soft ``missing_contacts_pending_B6``
   records while ``require/contacts.csv`` is header-only);
4. calls the representative-test hook :func:`run_representative_test` (template + instance 0; implemented by the v2
   sampler, TODO D2 — skipped with a log line until ``ispatialgym.sampling.v2.core`` provides it).

Run::

    python -m ispatialgym.sampling.ground_isg [--representative-only | --test-only | --all] [--subtask ID ...]
        [--slots representative,variation_geometry,...] [--out-bddl DIR] [--customs FILE] [--scene S ...]
        [--full-whitelists] [--distractors same:2,other:1] [--legacy-activity-names] [--dry-run]

Decisions taken where the docs leave room (all reported by the CLI, see ``--help`` and the run report):

* a group slot whose seeded ``size`` is a range (``distractors`` ``[1, 3]``) gets a concrete member list per problem
  (the BDDL enumerates objects): ``--distractors same:2,other:1`` = 2 same-category + 1 other-category (tier 1)
  distractors, filled from the pools in deterministic order; the grounded entry's ``generation.groups.distractors``
  carries the concrete ``size`` and the per-category ``members`` counts of the alias;
* group alias tokens are ``<category>_<count>``; the custom block (``dataset = ispatialgym-assets``) uses its icl
  ``display_name`` (``block_4``, as ``docs/task_bddl_formats.md`` writes it);
* a scene role bound to the floor (large-scale IORS floor init; small-scale articulation targets taller than 0.6 m,
  §5.4) carries the alias token ``floor`` and the category ``floors``; two scene roles bound to one object (under
  large-scale: the partner's own top) are declared once;
* ``symmetric_180`` empty is read as the icl README says ("not flagged", i.e. no half-turn equivalence); the spec's
  ``missing_frame`` for an unfilled column is behind ``--strict-symmetric-180``.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as _dt
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ispatialgym import ASSETS_ROOT
from ispatialgym.activity.problem import ISGProblemError
from ispatialgym.sampling import customs as C
from ispatialgym.sampling import generation_defaults as G
from ispatialgym.sampling import iddl as I
from ispatialgym.sampling.bddl_from_iddl import (
    GROUNDER,
    GroundingError,
    RenderedProblem,
    RoleGrounding,
    write_problem,
)
from ispatialgym.sampling.customs import Constraint, Customs, CustomsEntry, RoleEntry, SceneEntry

log = logging.getLogger("ispatialgym.ground_isg")

VERSION = "2.0.0"
DEFAULT_ICL = os.path.join(ASSETS_ROOT, "icl")
DEFAULT_IDDL = I.IDDL_ROOT
DEFAULT_BDDL_ROOT = os.path.join(ASSETS_ROOT, "bddl", "activity_definitions")
DEFAULT_CUSTOMS = C.DEFAULT_CUSTOMS_PATH

MODES = ("representative", "test", "all")
SLOTS = ("representative", "variation_texture", "variation_geometry", "heldout")
#: The activity directory of each family; ``--legacy-activity-names`` maps ``alignment`` -> ``world_alignment``.
LEGACY_ACTIVITY_NAMES = {"alignment": "world_alignment"}

FLOOR_CATEGORY = "floors"
FLOOR_ALIAS_TOKEN = "floor"
FLOOR_MODEL_NAMES = ("floor", "floors")
ROOM_ROLE = "room"
TARGET_LIKE_ROLES = ("target", "targets", "obstacles")
CUSTOM_DATASET = "ispatialgym-assets"
#: §5.4: small-scale articulation targets taller than this stand on the floor.
FLOOR_SUPPORT_HEIGHT_M = 0.6
#: §2.2 grasp verdicts: ``ok``, or ``flag`` with flags ⊆ {one_finger}.
TOLERATED_GRASP_FLAGS = {"one_finger"}
HANDLE_ADMISSIBLE_VERDICTS = ("handle", "edge", "manual")
PINCH_RANGE_M = (0.015, 0.080)
PENDING_CONTACTS_REASON = "missing_contacts_pending_B6"

_TRUE = ("1", "true", "True", "yes")


# ----------------------------------------------------------------------------------------------------------------
# icl access
# ----------------------------------------------------------------------------------------------------------------


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _split_rooms(value: str) -> List[str]:
    return [r.strip() for r in (value or "").split(",") if r.strip()]


def _parse_scale(value: str) -> Optional[List[float]]:
    if not value:
        return None
    try:
        v = json.loads(value)
    except Exception:
        try:
            v = [float(x) for x in re.split(r"[ ,;]+", value.strip("[]()")) if x]
        except Exception:
            return None
    if isinstance(v, (int, float)):
        return [float(v)] * 3
    return [float(x) for x in v]


def _scene_object_key(row: Dict[str, str]) -> str:
    """The name a scene candidate row is bound by: the wall region id for wall regions, else the scene object name."""
    return row.get("region_id") or row.get("model") or ""


def _parse_vec(value: str) -> Optional[List[float]]:
    if not value:
        return None
    try:
        return [float(x) for x in json.loads(value)]
    except Exception:
        return None


class ICL:
    """The candidate list: ``roles/`` + ``require/`` sheets with the indexes the grounder needs."""

    def __init__(self, root: Optional[str] = None):
        self.root = root or DEFAULT_ICL
        roles = os.path.join(self.root, "roles")
        req = os.path.join(self.root, "require")
        self.candidates: List[Dict[str, str]] = []
        for r in _read_csv(os.path.join(roles, "candidates.csv")):
            r["_extra"] = ""
            self.candidates.append(r)
        for r in _read_csv(os.path.join(roles, "candidates_extra.csv")):
            r["_extra"] = "1"
            self.candidates.append(r)
        if not self.candidates:
            raise GroundingError(f"no candidates under {roles}")
        self.by_sr: Dict[Tuple[str, str], List[Dict[str, str]]] = collections.defaultdict(list)
        self.by_key: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
        for r in self.candidates:
            self.by_sr[(r["subtask"], r["role"])].append(r)
            self.by_key.setdefault((r["subtask"], r["role"], r["category"], r["model"]), r)
        self.representatives = _read_csv(os.path.join(roles, "representatives.csv"))
        self.picks: Dict[str, Dict[str, Dict[str, Dict[str, str]]]] = collections.defaultdict(lambda: collections.defaultdict(dict))
        for r in self.representatives:
            self.picks[r["pick"]][r["role"]][r["slot"]] = r
        self.heldout_models: Set[Tuple[str, str]] = {
            (r["category"], r["model"]) for r in self.representatives if r["slot"] == "heldout" and r["kind"] != "scene" and r["model"]
        }
        # held-out scene objects are per subtask (a scene instance can be the representative partner of one subtask
        # and the generalization instance of another); held-out models are global (spec §3.12 / A9)
        self.heldout_scene: Set[Tuple[str, str, str, str]] = {
            (r["subtask"], r["role"], r["scene"], r["name"]) for r in self.representatives if r["slot"] == "heldout" and r["kind"] == "scene"
        }
        self.compat = _read_csv(os.path.join(roles, "constraints_compatible.csv"))
        self.conflicting = _read_csv(os.path.join(roles, "constraints_conflicting.csv"))
        self.lookalike_groups: Dict[str, Set[Tuple[str, str]]] = collections.defaultdict(set)
        for r in self.conflicting:
            if r.get("relation", "lookalike") == "lookalike":
                self.lookalike_groups[r["group_id"]].add((r["category"], r["model"]))
        self.lookalike_of: Dict[Tuple[str, str], Set[Tuple[str, str]]] = collections.defaultdict(set)
        for members in self.lookalike_groups.values():
            for m in members:
                self.lookalike_of[m] |= members
        # require sheets
        self.grasps = _read_csv(os.path.join(req, "grasps.csv"))
        self.grasps_ok: Dict[Tuple[str, str, str], List[Dict[str, str]]] = collections.defaultdict(list)
        for g in self.grasps:
            if self.grasp_row_admissible(g):
                self.grasps_ok[(g["category"], g["model"], g["require"])].append(g)
        self.handles = _read_csv(os.path.join(req, "handles.csv"))
        self.handles_canonical: Dict[Tuple[str, str], List[Dict[str, str]]] = collections.defaultdict(list)
        self.handles_instance: Dict[Tuple[str, str], List[Dict[str, str]]] = collections.defaultdict(list)
        for h in self.handles:
            if h.get("scene"):
                self.handles_instance[(h["scene"], h["instance"])].append(h)
            else:
                self.handles_canonical[(h["category"], h["model"])].append(h)
        self.contacts = _read_csv(os.path.join(req, "contacts.csv"))
        self.contacts_by_model: Dict[Tuple[str, str], List[Dict[str, str]]] = collections.defaultdict(list)
        for c in self.contacts:
            self.contacts_by_model[(c["category"], c["model"])].append(c)

    # -- candidates -------------------------------------------------------------------------------------------------
    def rows(self, subtask: str, role: str) -> List[Dict[str, str]]:
        return list(self.by_sr.get((subtask, role), []))

    def row(self, subtask: str, role: str, category: str, model: str) -> Optional[Dict[str, str]]:
        return self.by_key.get((subtask, role, category, model))

    def kind_of(self, subtask: str, role: str) -> str:
        kinds = {r["kind"] for r in self.rows(subtask, role)}
        if not kinds:
            raise GroundingError(f"{subtask}: role {role!r} has no rows in roles/candidates.csv (a role without rows is an error)")
        if len(kinds) > 1:
            raise GroundingError(f"{subtask}: role {role!r} has several kinds {sorted(kinds)}")
        return kinds.pop()

    def has_role(self, subtask: str, role: str) -> bool:
        return bool(self.by_sr.get((subtask, role)))

    def room_rows(self, subtask: str) -> List[Dict[str, str]]:
        return self.rows(subtask, ROOM_ROLE)

    def scene_rows(self, subtask: str, role: str, category: Optional[str] = None, scene: Optional[str] = None) -> List[Dict[str, str]]:
        out = []
        for r in self.rows(subtask, role):
            if r["kind"] != "scene":
                continue
            if category is not None and r["category"] != category:
                continue
            if scene is not None and r["scene"] != scene:
                continue
            out.append(r)
        return out

    # -- compatibility ------------------------------------------------------------------------------------------------
    def partner_rows(self, subtask: str, target_category: str, target_model: Optional[str] = None) -> List[Dict[str, str]]:
        out = []
        for r in self.compat:
            if r["subtask"] != subtask or r["relation"] != "partner_of" or r["category_a"] != target_category:
                continue
            if r.get("model_a") and target_model is not None and r["model_a"] != target_model:
                continue
            out.append(r)
        return out

    def distractor_rows(self, target_category: str, relation: str) -> List[Dict[str, str]]:
        return [
            r for r in self.compat
            if r["subtask"] == "disembedding" and r["relation"] == relation and r["category_a"] == target_category
        ]

    def lookalikes(self, category: str, model: str) -> Set[Tuple[str, str]]:
        return set(self.lookalike_of.get((category, model), set()))

    def are_lookalikes(self, a: Tuple[str, str], b: Tuple[str, str]) -> bool:
        return a != b and b in self.lookalike_of.get(a, set())

    # -- require ------------------------------------------------------------------------------------------------------
    @staticmethod
    def grasp_row_admissible(g: Dict[str, str]) -> bool:
        if g.get("verdict") == "ok":
            return True
        if g.get("verdict") == "flag":
            flags = {f.strip() for f in (g.get("flags") or "").split(",") if f.strip()}
            return flags <= TOLERATED_GRASP_FLAGS
        return False

    def admissible_grasps(self, category: str, model: str, pipeline: str) -> List[Dict[str, str]]:
        """§2.2: rows with ``require = grasps:<pipeline>`` and an admissible verdict (``handle_half`` rows are filed
        under ``grasps:floating`` by the builder, so knives count as floating)."""
        return list(self.grasps_ok.get((category, model, f"grasps:{pipeline}"), []))

    def grasp_key(self, category: str, model: str, pipeline: str) -> Optional[str]:
        rows = self.admissible_grasps(category, model, pipeline)
        keys = sorted({r["key"] for r in rows})
        return keys[0] if keys else None

    @staticmethod
    def handle_row_admissible(h: Dict[str, str], instance: bool) -> bool:
        if h.get("verdict") not in HANDLE_ADMISSIBLE_VERDICTS:
            return False
        if (h.get("review_graspable") or "") == "False":
            return False
        if instance and h.get("width_m"):
            try:
                w = float(h["width_m"])
            except ValueError:
                return False
            if not (PINCH_RANGE_M[0] <= w <= PINCH_RANGE_M[1]):
                return False
        return True

    def admissible_handles(self, category: str, model: str, scene: Optional[str] = None, name: Optional[str] = None) -> List[Dict[str, str]]:
        if scene and name:
            rows = self.handles_instance.get((scene, name), [])
            return [h for h in rows if self.handle_row_admissible(h, instance=True)]
        rows = self.handles_canonical.get((category, model), [])
        return [h for h in rows if self.handle_row_admissible(h, instance=False)]

    def contacts_available(self) -> bool:
        return bool(self.contacts)

    def admissible_contacts(self, category: str, model: str) -> List[Dict[str, str]]:
        return [c for c in self.contacts_by_model.get((category, model), []) if c.get("kind") == "push" or c.get("contact_role") == "push"]


# ----------------------------------------------------------------------------------------------------------------
# Admissibility (spec §2.2, §3.12, §3.13)
# ----------------------------------------------------------------------------------------------------------------


def _is_ready(row: Dict[str, str]) -> bool:
    if row.get("kind") == "scene":
        return True
    v = (row.get("ready") or "").strip()
    if v == "0":
        return False
    if v in _TRUE:
        return True
    try:
        from ispatialgym.sampling import knowledge

        return knowledge.is_ready(row["model"])
    except Exception:  # KB unavailable: the icl column is the only evidence
        return True


def _frame_ok(subtask: str, row: Dict[str, str], strict_symmetric_180: bool) -> Optional[str]:
    """§2.2: alignment targets need a reviewed frame record; wall targets ``symmetric_180`` (strict mode)."""
    front, up, level, long_ = (row.get(k) or "" for k in ("frame_front", "frame_up", "frame_level", "frame_long"))
    if subtask == "alignment-axial":
        ok = bool(front or up) and bool(long_)
    elif subtask in ("alignment-world_vertical", "alignment-world_horizontal"):
        ok = bool(front) and bool(up or level)
        if ok and strict_symmetric_180 and (row.get("symmetric_180") or "") not in ("0", "1"):
            ok = False
    else:
        return None
    return None if ok else "missing_frame"


def _symmetry_tags(row: Dict[str, str]) -> List[str]:
    try:
        return C.parse_symmetry_tags(row.get("symmetry") or "")
    except C.CustomsError:
        return []


@dataclass
class Verdict:
    """Admissibility of one candidate model for one role: ``reason`` (None = admitted), ``pending`` (soft records),
    the admissible handle rows (articulation) and the grasp key of the pipeline."""

    reason: Optional[str] = None
    pending: List[str] = field(default_factory=list)
    handles: List[Dict[str, str]] = field(default_factory=list)
    grasp_key: Optional[str] = None


class Admissibility:
    def __init__(self, icl: ICL, strict_symmetric_180: bool = False):
        self.icl = icl
        self.strict_symmetric_180 = strict_symmetric_180

    def check(self, iddl: I.IDDL, role: str, row: Dict[str, str], scene: Optional[str] = None) -> Verdict:
        """§2.2 row admissibility of ``row`` (a ``roles/candidates.csv`` row of ``(iddl.subtask, role)``)."""
        v = Verdict()
        subtask = iddl.subtask
        category, model = row["category"], row["model"]
        is_scene = row.get("kind") == "scene"
        if not is_scene and (category, model) in self.icl.heldout_models:
            v.reason = "heldout"
            return v
        if not _is_ready(row):
            v.reason = "not_ready"
            return v
        for req in iddl.require_of(role):
            if req.kind == "grasps":
                rows = self.icl.admissible_grasps(category, model, req.pipeline or "")
                if not rows:
                    v.reason = req.missing_reason
                    return v
                v.grasp_key = sorted({r["key"] for r in rows})[0]
            elif req.kind == "handle":
                if is_scene:
                    handles = self.icl.admissible_handles(category, model, scene=row.get("scene"), name=model)
                else:
                    handles = self.icl.admissible_handles(category, model)
                if not handles:
                    v.reason = req.missing_reason
                    return v
                v.handles = handles
            elif req.kind == "contacts":
                if self.icl.contacts_available():
                    if not self.icl.admissible_contacts(category, model):
                        v.reason = req.missing_reason
                        return v
                else:
                    v.pending.append(PENDING_CONTACTS_REASON)
        frame = _frame_ok(subtask, row, self.strict_symmetric_180)
        if frame:
            v.reason = frame
            return v
        if subtask == "mental_rotation" and role == "target" and _symmetry_tags(row):
            v.reason = "symmetric_target"
            return v
        if role == "obstacles" and scene is not None:
            natural = [s.strip() for s in (row.get("natural_scenes") or "").split(",") if s.strip()]
            if natural and scene not in natural:
                v.reason = "not_natural_in_scene"
                return v
        return v


def choose_joint(handles: List[Dict[str, str]], preferred: Optional[str] = None) -> Optional[Tuple[str, Dict[str, str]]]:
    """The joint of an articulation target from its admissible handle rows: the preferred joint when it has an
    admissible row, else the reviewer's ``chosen`` row, else bar / knob handles before edge pinches (name order)."""
    if not handles:
        return None
    if preferred:
        rows = [h for h in handles if h.get("joint") == preferred]
        if not rows:
            return None
        handles = rows
    order = {"handle": 0, "manual": 1, "edge": 2}

    def key(h):
        return (0 if (h.get("chosen") or "") in _TRUE else 1, order.get(h.get("verdict"), 9), h.get("name") or "")

    best = sorted(handles, key=key)[0]
    return best["joint"], best


# ----------------------------------------------------------------------------------------------------------------
# Combinations
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class RolePick:
    """One role of a combination: the grounded category and its model pool / scene objects."""

    role: str
    kind: str  # sampleable | group | scene
    category: Optional[str] = None  # sampleable / scene (``floors`` for the floor)
    models: List[Tuple[str, str]] = field(default_factory=list)  # whitelist seeds [(category, model)]
    members: Dict[str, int] = field(default_factory=dict)  # group: category -> count (alias order)
    pools: Dict[str, List[str]] = field(default_factory=dict)  # group: category -> admissible pool (whitelist)
    scene_objects: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)  # scene: scene -> candidate rows
    same_as: Optional[str] = None  # scene role bound to another role's object ("partner") or to the floor ("floor")
    joint: Optional[str] = None  # articulation: the pick's joint
    pick_row: Optional[Dict[str, str]] = None
    allow_heldout: bool = False  # the heldout slot itself was requested for this role

    @property
    def is_floor(self) -> bool:
        return self.kind == "scene" and self.category == FLOOR_CATEGORY

    def alias_token(self, icl: ICL, subtask: str) -> str:
        if self.kind == "group":
            parts = []
            for cat, n in self.members.items():
                row = next((r for r in icl.rows(subtask, self.role) if r["category"] == cat), None)
                token = cat
                if row is not None and (row.get("dataset") or "") == CUSTOM_DATASET and row.get("display_name"):
                    token = row["display_name"].strip().replace(" ", "_")
                parts.append(f"{token}_{n}")
            return "_".join(parts)
        if self.is_floor:
            return FLOOR_ALIAS_TOKEN
        return self.category or ""


@dataclass
class Combination:
    subtask: str
    pick: str
    slot: str
    roles: Dict[str, RolePick]
    scenes: List[str] = field(default_factory=list)
    rooms: Dict[str, List[str]] = field(default_factory=dict)  # scene -> task rooms
    floor_room: Dict[str, str] = field(default_factory=dict)  # scene -> the floor's room
    notes: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.pick}/{self.slot}"


class Skip(Exception):
    """A combination that is not grounded (reason + detail)."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# ----------------------------------------------------------------------------------------------------------------
# Grounder
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class Options:
    mode: str = "representative"
    slots: Tuple[str, ...] = ("representative",)
    scenes: Optional[Tuple[str, ...]] = None
    full_whitelists: bool = False
    distractor_pattern: Tuple[Tuple[str, int], ...] = (("same", 2), ("other", 1))
    legacy_activity_names: bool = False
    strict_symmetric_180: bool = False
    dry_run: bool = False
    representative_test: bool = True


@dataclass
class GroundedProblem:
    """What one grounded alias produced (the object the D2 hook receives)."""

    alias: str
    bddl_alias: str
    subtask: str
    activity: str
    definition_id: int
    bddl_path: str
    bddl_text: str
    problem: Any  # ispatialgym.activity.problem.ISGProblem
    entry: CustomsEntry
    iddl: I.IDDL
    combination: Combination
    rendered: RenderedProblem


class Report:
    def __init__(self):
        self.problems: Dict[str, List[Tuple[str, int, str]]] = collections.defaultdict(list)  # subtask -> (alias, id, label)
        self.skipped: Dict[str, List[Tuple[str, str]]] = collections.defaultdict(list)  # subtask -> (label, reason)
        self.blacklist: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)  # subtask -> reason -> n
        self.pending: Dict[str, int] = collections.Counter()
        self.notes: Dict[str, List[str]] = collections.defaultdict(list)
        self.tests: List[str] = []

    def text(self) -> str:
        lines = [f"ground_isg {VERSION} report ({_dt.datetime.now().isoformat(timespec='seconds')})", ""]
        lines.append("| subtask | problems | groundings | skipped | blacklist entries (reason: n) |")
        lines.append("|---|---|---|---|---|")
        for st in sorted(set(self.problems) | set(self.skipped) | set(self.blacklist)):
            hist = ", ".join(f"{k}: {v}" for k, v in sorted(self.blacklist[st].items()))
            n_alias = len({a for a, _, _ in self.problems[st]})
            lines.append(f"| {st} | {n_alias} | {len(self.problems[st])} | {len(self.skipped[st])} | {hist} |")
        total = len({a for v in self.problems.values() for a, _, _ in v})
        groundings = sum(len(v) for v in self.problems.values())
        lines.append("")
        lines.append(f"{total} problems written ({groundings} groundings), {sum(len(v) for v in self.skipped.values())} combinations skipped")
        if self.pending:
            lines.append("")
            lines.append("pending requirements (recorded as soft `blacklist` constraint records, models stay whitelisted):")
            for k, v in sorted(self.pending.items()):
                lines.append(f"- {k}: {v}")
        for st in sorted(self.skipped):
            if self.skipped[st]:
                lines.append("")
                lines.append(f"skipped in {st}:")
                for label, reason in self.skipped[st]:
                    lines.append(f"- {label}: {reason}")
        for st in sorted(self.notes):
            if self.notes[st]:
                lines.append("")
                lines.append(f"notes for {st}:")
                for n in sorted(set(self.notes[st])):
                    lines.append(f"- {n}")
        if self.tests:
            lines.append("")
            lines.append("representative tests:")
            lines.extend(f"- {t}" for t in self.tests)
        return "\n".join(lines) + "\n"

    def to_json(self) -> Dict[str, Any]:
        return {
            "version": VERSION,
            "problems": {st: [{"alias": a, "id": i, "from": l} for a, i, l in v] for st, v in self.problems.items()},
            "skipped": {st: [{"combination": l, "reason": r} for l, r in v] for st, v in self.skipped.items()},
            "blacklist": {st: dict(c) for st, c in self.blacklist.items()},
            "pending": dict(self.pending),
            "notes": {st: sorted(set(v)) for st, v in self.notes.items()},
            "tests": list(self.tests),
        }


class Grounder:
    def __init__(
        self,
        iddls: Dict[str, I.IDDL],
        icl: ICL,
        customs: Customs,
        bddl_root: str = DEFAULT_BDDL_ROOT,
        options: Optional[Options] = None,
        iddl_root: Optional[str] = None,
    ):
        self.iddls = iddls
        self.icl = icl
        self.customs = customs
        self.bddl_root = bddl_root
        self.iddl_root = iddl_root or DEFAULT_IDDL
        self.opt = options or Options()
        self.adm = Admissibility(icl, self.opt.strict_symmetric_180)
        self.report = Report()
        self.today = _dt.date.today().isoformat()
        self.grounded: List[GroundedProblem] = []

    # -- helpers ------------------------------------------------------------------------------------------------------
    def activity_of(self, iddl: I.IDDL) -> str:
        if self.opt.legacy_activity_names:
            return LEGACY_ACTIVITY_NAMES.get(iddl.family, iddl.family)
        return iddl.family

    def _reason(self, code: str) -> str:
        return C.blacklist_reason(code, tool="ground_isg", date=self.today)

    @staticmethod
    def _synset(category: str) -> str:
        from ispatialgym.sampling.knowledge import synset_of_category

        return synset_of_category(category)

    def _marked(self, row: Dict[str, str]) -> bool:
        """Whether a candidates row belongs to the selection of the mode (test / all)."""
        if self.opt.mode == "test":
            return (row.get("test") or "") in _TRUE
        if self.opt.mode == "all":
            return not (row.get("status") or "").startswith("geometric_reject")
        return (row.get("representative") or "") in _TRUE

    # -- combinations: representative mode ----------------------------------------------------------------------------
    def combinations(self, subtask: str) -> List[Combination]:
        iddl = self.iddls[subtask]
        if self.opt.mode == "representative":
            return self._combinations_from_picks(iddl)
        return self._combinations_from_candidates(iddl)

    def _combinations_from_picks(self, iddl: I.IDDL) -> List[Combination]:
        subtask = iddl.subtask
        picks = sorted(p for p in self.icl.picks if p.split("#")[0] == subtask)
        out: List[Combination] = []
        if not picks:
            # no object pick (camera_relocalization-standard): one combination from the room rows
            if not iddl.roles or all(self.icl.kind_of(subtask, r) == "scene" for r in iddl.roles):
                try:
                    out.append(self._build(iddl, f"{subtask}#00", "representative", {}))
                except Skip as s:
                    self.report.skipped[subtask].append((f"{subtask}#00/representative", str(s)))
            return out
        for pick in picks:
            roles = self.icl.picks[pick]
            for slot in self.opt.slots:
                if slot != "representative" and not any(slot in slots for slots in roles.values()):
                    continue
                rows: Dict[str, Dict[str, str]] = {}
                for role, slots in roles.items():
                    row = slots.get(slot) or slots.get("representative")
                    if row is not None:
                        rows[role] = row
                try:
                    out.append(self._build(iddl, pick, slot, rows))
                except Skip as s:
                    self.report.skipped[subtask].append((f"{pick}/{slot}", str(s)))
        return out

    def _combinations_from_candidates(self, iddl: I.IDDL) -> List[Combination]:
        """``--test-only`` / ``--all``: category combinations from ``roles/candidates.csv`` marks, admitted by
        ``constraints_compatible.csv`` (never a cartesian product of unrelated lists)."""
        subtask = iddl.subtask
        out: List[Combination] = []
        main_role = next((r for r in TARGET_LIKE_ROLES if r in iddl.roles), None)
        support_cats: List[Optional[str]] = [None]
        if "support" in iddl.roles:
            support_cats = sorted({r["category"] for r in self.icl.scene_rows(subtask, "support") if self._marked(r)}) or []
        if main_role is None:
            try:
                out.append(self._build(iddl, f"{subtask}#00", self.opt.mode, {}))
            except Skip as s:
                self.report.skipped[subtask].append((f"{subtask}#00/{self.opt.mode}", str(s)))
            return out
        main_rows = [r for r in self.icl.rows(subtask, main_role) if self._marked(r)]
        by_cat: Dict[str, List[Dict[str, str]]] = collections.defaultdict(list)
        for r in main_rows:
            by_cat[r["category"]].append(r)
        n = 0
        for cat in sorted(by_cat):
            rows = by_cat[cat]
            main_row = dict(rows[0])
            main_row["_pool"] = json.dumps([r["model"] for r in rows])  # every marked model of the category
            partner_choices: List[Optional[Dict[str, str]]] = [None]
            if "partner" in iddl.roles:
                partner_choices = []
                pkind = self.icl.kind_of(subtask, "partner")
                for pc in sorted({c["category_b"] for c in self.icl.partner_rows(subtask, cat)}):
                    prow = [r for r in self.icl.rows(subtask, "partner") if r["category"] == pc and self._marked(r)]
                    if not prow:
                        continue
                    p = dict(prow[0])
                    if pkind == "scene":
                        p["name"] = p["model"]
                    p["_pool"] = json.dumps([r["model"] for r in prow])
                    partner_choices.append(p)
            for partner in partner_choices:
                for scat in support_cats:
                    rows_sel: Dict[str, Dict[str, str]] = {main_role: main_row}
                    if partner is not None:
                        rows_sel["partner"] = partner
                    if scat is not None:
                        srow = next(r for r in self.icl.scene_rows(subtask, "support", category=scat) if self._marked(r))
                        s = dict(srow)
                        s["name"] = s["model"]
                        rows_sel["support"] = s
                    n += 1
                    label = f"{subtask}#{self.opt.mode}{n:03d}"
                    try:
                        out.append(self._build(iddl, label, self.opt.mode, rows_sel))
                    except Skip as s:
                        self.report.skipped[subtask].append((f"{label}/{self.opt.mode}", str(s)))
        return out

    # -- building one combination ------------------------------------------------------------------------------------
    def _build(self, iddl: I.IDDL, pick: str, slot: str, rows: Dict[str, Dict[str, str]]) -> Combination:
        subtask = iddl.subtask
        comb = Combination(subtask, pick, slot, {})
        for role in iddl.roles:
            kind = self.icl.kind_of(subtask, role)
            row = rows.get(role)
            if row is not None:
                row = self._resolve_row(subtask, role, row)
            if kind == "sampleable":
                comb.roles[role] = self._sampleable_pick(iddl, role, row, slot)
            elif kind == "group":
                comb.roles[role] = self._group_pick(iddl, role, row, comb)
            elif kind == "scene":
                comb.roles[role] = self._scene_pick(iddl, role, row, comb)
            else:
                raise Skip("unknown_kind", f"{role}: {kind}")
        self._pair_rules(iddl, comb)
        self._bind_scenes(iddl, comb)
        return comb

    def _pool_of(self, subtask: str, role: str, category: str) -> List[Dict[str, str]]:
        return [r for r in self.icl.rows(subtask, role) if r["category"] == category and r["kind"] != "scene"]

    def _resolve_row(self, subtask: str, role: str, row: Dict[str, str]) -> Dict[str, str]:
        """The ``roles/candidates.csv`` row behind a selection row (per-model facts: bbox, scale, symmetry, dataset,
        natural_scenes ...), with the selection's own fields (``name``, ``scene``, ``joint``, ``_pool``) on top."""
        cand: Optional[Dict[str, str]] = None
        if row.get("kind") == "scene":
            name = row.get("name") or row.get("model")
            cand = self.icl.row(subtask, role, row["category"], name)
            if cand is None:
                cand = next((r for r in self.icl.scene_rows(subtask, role, category=row["category"]) if _scene_object_key(r) == name), None)
        elif row.get("model"):
            cand = self.icl.row(subtask, role, row["category"], row["model"])
        merged: Dict[str, str] = dict(cand) if cand else {}
        for k, v in row.items():
            if v not in ("", None) or k not in merged:
                merged[k] = v
        if cand is None:
            merged["_synthetic"] = "1"
        return merged

    def _sampleable_pick(self, iddl: I.IDDL, role: str, row: Optional[Dict[str, str]], slot: str) -> RolePick:
        subtask = iddl.subtask
        if row is None:
            # fixed sampleable roles without a pick (disembedding trays): the representative-marked rows
            marked = [r for r in self.icl.rows(subtask, role) if (r.get("representative") or "") in _TRUE] if self.opt.mode == "representative" \
                else [r for r in self.icl.rows(subtask, role) if self._marked(r)]
            if not marked:
                raise Skip("no_candidate", f"role {role!r} has no {self.opt.mode} row")
            cats = sorted({r["category"] for r in marked})
            if len(cats) > 1:
                raise Skip("ambiguous_category", f"role {role!r}: {cats}")
            rp = RolePick(role, "sampleable", cats[0], [(r["category"], r["model"]) for r in marked])
            rp.pick_row = marked[0]
            return rp
        category, model = row["category"], row["model"]
        models = [(category, model)]
        if row.get("_pool"):
            models = [(category, m) for m in json.loads(row["_pool"])]
        rp = RolePick(role, "sampleable", category, models, pick_row=row, joint=row.get("joint") or None)
        rp.allow_heldout = slot == "heldout"
        return rp

    def _group_pick(self, iddl: I.IDDL, role: str, row: Optional[Dict[str, str]], comb: Combination) -> RolePick:
        subtask = iddl.subtask
        gen = G.defaults_for(subtask)
        seed = (gen.get("groups") or {}).get(role)
        if seed is None:
            raise Skip("no_group_seed", f"generation_defaults has no groups.{role} for {subtask}")
        size = seed["size"]
        rp = RolePick(role, "group")
        if role == "distractors":
            target = comb.roles.get("target")
            if target is None or not target.models:
                raise Skip("no_target", "distractors need the target first")
            tcat, tmodel = target.models[0]
            self._distractor_members(rp, tcat, tmodel, size)
            return rp
        if row is None:
            marked = [r for r in self.icl.rows(subtask, role) if self._marked(r)]
            if not marked:
                raise Skip("no_candidate", f"group role {role!r} has no {self.opt.mode} row")
            row = marked[0]
        if isinstance(size, list):
            raise Skip("group_size_range", f"{role}: a range size {size} needs a member pattern (only distractors have one)")
        category, model = row["category"], row["model"]
        rp.members = {category: int(size)}
        rp.pick_row = row
        models = [(category, model)]
        if row.get("_pool"):
            models = [(category, m) for m in json.loads(row["_pool"])]
        rp.models = models
        rp.pools = {category: [m for _, m in models]}
        rp.category = category
        return rp

    def _distractor_members(self, rp: RolePick, tcat: str, tmodel: str, size) -> None:
        """§3.12 rule 1: same-category pool minus the target and its look-alikes, other-category pools (tier order);
        the ``--distractors`` pattern fixes the member counts of the alias."""
        excluded = {(tcat, tmodel)} | self.icl.lookalikes(tcat, tmodel) | self.icl.heldout_models
        same = [(r["category_b"], r["model_b"]) for r in self.icl.distractor_rows(tcat, "distractor_same_category")]
        same = [cm for cm in same if cm not in excluded and _is_ready(self.icl.row("disembedding", "distractors", *cm) or {"kind": "group", "model": cm[1]})]
        other_rows = sorted(self.icl.distractor_rows(tcat, "distractor_other_category"), key=lambda r: (int(r.get("tier") or 9), r["category_b"], r["model_b"]))
        other_by_cat: Dict[str, List[str]] = collections.OrderedDict()
        for r in other_rows:
            cm = (r["category_b"], r["model_b"])
            if cm in excluded:
                continue
            other_by_cat.setdefault(r["category_b"], [])
            if cm[1] not in other_by_cat[r["category_b"]]:
                other_by_cat[r["category_b"]].append(cm[1])
        total = size[1] if isinstance(size, list) else int(size)
        members: Dict[str, int] = collections.OrderedDict()
        pools: Dict[str, List[str]] = {}
        remaining = total
        for kind, n in self.opt.distractor_pattern:
            n = min(n, remaining)
            if n <= 0:
                continue
            if kind == "same":
                if len(same) >= n:
                    members[tcat] = members.get(tcat, 0) + n
                    pools[tcat] = sorted({m for _, m in same})
                    remaining -= n
            elif kind == "other":
                # the first other categories (tier order) with a pool of >= 1 model, one member each
                for cat, ms in other_by_cat.items():
                    if remaining <= 0 or n <= 0:
                        break
                    if cat == tcat or cat in members or not ms:
                        continue
                    members[cat] = 1
                    pools[cat] = sorted(ms)
                    remaining -= 1
                    n -= 1
        if remaining > 0:
            # fill from whatever pool is left (same category first, then other categories)
            if same and remaining > 0 and len(same) >= members.get(tcat, 0) + remaining:
                members[tcat] = members.get(tcat, 0) + remaining
                pools[tcat] = sorted({m for _, m in same})
                remaining = 0
            for cat, ms in other_by_cat.items():
                if remaining <= 0:
                    break
                if cat == tcat or cat in members or not ms:
                    continue
                members[cat] = 1
                pools[cat] = sorted(ms)
                remaining -= 1
        lo = size[0] if isinstance(size, list) else int(size)
        if sum(members.values()) < lo:
            raise Skip("empty_distractor_pool", f"target {tcat}/{tmodel}: pools too small after the look-alike / held-out exclusions")
        rp.members = members
        rp.pools = pools
        rp.models = [(c, m) for c, ms in pools.items() for m in ms]

    def _scene_pick(self, iddl: I.IDDL, role: str, row: Optional[Dict[str, str]], comb: Combination) -> RolePick:
        subtask = iddl.subtask
        rp = RolePick(role, "scene")
        if row is None:
            raise Skip("no_scene_pick", f"scene role {role!r} has no {self.opt.mode} row")
        rp.allow_heldout = comb.slot == "heldout"
        category = row["category"]
        name = row.get("name") or row.get("model")
        scene = row.get("scene")
        if category in FLOOR_MODEL_NAMES or name in FLOOR_MODEL_NAMES:
            # the floor of the scene: the row is the room anchor (the floor rows of the icl carry no room; the
            # partner / target of the combination decides the room)
            rp.category = FLOOR_CATEGORY
            rp.same_as = "floor"
            rp.pick_row = row
            rp.scene_objects = {scene: [row]} if scene else {}
            return rp
        rp.category = category
        rp.pick_row = row
        rp.joint = row.get("joint") or None
        # the same scene object as another role of the combination (under large-scale: partner's own top)
        for other, orp in comb.roles.items():
            if orp.kind == "scene" and other != role and orp.pick_row is not None and orp.pick_row.get("scene") == scene \
                    and (orp.pick_row.get("name") or orp.pick_row.get("model")) == name:
                rp.same_as = other
        # small-scale articulation: targets taller than 0.6 m stand on the floor (§5.4)
        if role == "support" and subtask.startswith("configuration_matching-articulation") and "small_scale" in subtask:
            target = comb.roles.get("target")
            trow = target.pick_row if target else None
            if trow is not None and trow.get("bbox_z"):
                try:
                    if float(trow["bbox_z"]) > FLOOR_SUPPORT_HEIGHT_M:
                        comb.notes.append(f"{target.category}: taller than {FLOOR_SUPPORT_HEIGHT_M} m -> support = floor of the picked "
                                          f"support's room (§5.4)")
                        rp.category = FLOOR_CATEGORY
                        rp.same_as = "floor"
                        rp.scene_objects = {scene: [row]} if scene else {}  # the picked support anchors the room
                        return rp
                except ValueError:
                    pass
        # candidate scene objects per scene: the pick's object + the marked rows of the category. Wall regions are
        # keyed by their region id (``<room>/<wall>/<rNN>``, the pick's ``name``), other scene objects by name.
        objects: Dict[str, List[Dict[str, str]]] = collections.defaultdict(list)
        for r in self.icl.scene_rows(subtask, role, category=category):
            is_pick = r["scene"] == scene and _scene_object_key(r) == name
            if is_pick or self._marked(r) or (self.opt.mode == "representative" and (r.get("representative") or "") in _TRUE):
                if (r.get("status") or "").startswith("geometric_reject") and not is_pick:
                    continue
                if (subtask, role, r["scene"], _scene_object_key(r)) in self.icl.heldout_scene and not (is_pick and rp.allow_heldout):
                    continue  # held-out rule: the role's held-out scene object is never bound in another combination
                if not any(_scene_object_key(o) == _scene_object_key(r) for o in objects[r["scene"]]):
                    objects[r["scene"]].append(r)
        if scene and not any(_scene_object_key(o) == name for o in objects.get(scene, [])):
            # the pick names an object the candidates sheet does not list: keep it (the selection is explicit) and say so
            fake = {k: v for k, v in row.items() if k in ("subtask", "role", "category", "scene", "room", "joint", "handle_key", "note", "status")}
            fake.update({"kind": "scene", "model": name, "representative": "1", "_synthetic": "1"})
            objects[scene].append(fake)
            comb.notes.append(f"{role}: {name}@{scene} is not a roles/candidates.csv row of {subtask} (bound from the selection only)")
        rp.scene_objects = dict(objects)
        return rp

    def _pair_rules(self, iddl: I.IDDL, comb: Combination) -> None:
        """§3.12 rule 2 (look-alike target–partner pair), the compat sheet for target–partner pairs (attachment ready)."""
        subtask = iddl.subtask
        target, partner = comb.roles.get("target"), comb.roles.get("partner")
        if target is None or partner is None:
            return
        tcat = target.category
        if partner.kind == "sampleable":
            pcat = partner.category
            rows = self.icl.partner_rows(subtask, tcat)
            if not any(r["category_b"] == pcat for r in rows):
                raise Skip("not_compatible", f"{tcat} -> {pcat} is not a partner_of pair of {subtask}")
            if subtask == "configuration_matching-attachment":
                ok_models = []
                for tm in [m for _, m in target.models]:
                    for pm in [m for _, m in partner.models]:
                        r = next((c for c in rows if c["model_a"] == tm and c["model_b"] == pm), None)
                        if r is None:
                            continue
                        if (r.get("ready") or "1") not in _TRUE:
                            comb.notes.append(f"attachment pair {tm} -> {pm}: partner KB entry not ready")
                            continue
                        ok_models.append((tm, pm))
                if not ok_models:
                    raise Skip("not_compatible", f"no ready attachment pair between {tcat} {[m for _, m in target.models]} and {pcat}")
            for tm in [m for _, m in target.models]:
                for pm in [m for _, m in partner.models]:
                    if self.icl.are_lookalikes((tcat, tm), (pcat, pm)):
                        if len(target.models) == 1 and len(partner.models) == 1:
                            raise Skip("lookalike_pair", f"{tcat}/{tm} and {pcat}/{pm} share a look-alike group (§3.12 rule 2)")
                        comb.notes.append(f"look-alike pair {tcat}/{tm} – {pcat}/{pm} excluded by the sampler (§3.12 rule 2)")
        else:
            pcat = partner.category
            rows = self.icl.partner_rows(subtask, tcat)
            if rows and not any(r["category_b"] == pcat for r in rows):
                raise Skip("not_compatible", f"{tcat} -> scene partner {pcat} is not a partner_of pair of {subtask}")
            # §6.12 under-large partner admissibility from the B4 note: clearance >= target height + 0.17
            if subtask.endswith("under_large_scale") and target.pick_row is not None:
                extra = G.defaults_for(subtask)["under_clearance_extra_m"]
                try:
                    h = float(target.pick_row.get("bbox_z") or 0.0)
                except ValueError:
                    h = 0.0
                for scene, objs in list(partner.scene_objects.items()):
                    keep = []
                    for o in objs:
                        m = re.search(r"under_clearance_max=([0-9.]+)", o.get("note") or "")
                        if m and h and float(m.group(1)) < h + extra:
                            comb.notes.append(f"{o['model']}@{scene}: under clearance {m.group(1)} < {h:.2f} + {extra} (under_clearance)")
                            continue
                        keep.append(o)
                    partner.scene_objects[scene] = keep
                if not any(partner.scene_objects.values()):
                    raise Skip("under_clearance", f"no partner with enough under clearance for {tcat} (height {h:.2f} m)")

    def _bind_scenes(self, iddl: I.IDDL, comb: Combination) -> None:
        """Which scenes get a block and which rooms: room rows bound the scene; otherwise every scene where each
        scene role has at least one candidate object."""
        subtask = iddl.subtask
        room_rows = self.icl.room_rows(subtask)
        scene_roles = [rp for rp in comb.roles.values() if rp.kind == "scene" and not rp.is_floor and not rp.same_as]
        if room_rows:
            for r in room_rows:
                sc = r["scene"]
                if self.opt.scenes and sc not in self.opt.scenes:
                    continue
                comb.scenes.append(sc)
                comb.rooms[sc] = _split_rooms(r.get("room")) or [r["model"]]
                comb.floor_room[sc] = comb.rooms[sc][0]
        else:
            candidates = set(C.SCENE_CANDIDATES)
            for rp in scene_roles:
                candidates &= {s for s, objs in rp.scene_objects.items() if objs}
            for rp in comb.roles.values():
                if rp.is_floor and rp.scene_objects and not scene_roles:
                    candidates &= set(rp.scene_objects)
            if self.opt.scenes:
                candidates &= set(self.opt.scenes)
            for sc in sorted(candidates, key=lambda s: C.SCENE_CANDIDATES.index(s) if s in C.SCENE_CANDIDATES else 99):
                rooms: List[str] = []
                primary: Optional[str] = None
                # the primary room (the floor's) comes from the partner (large-scale IORS), the scene target
                # (large-scale articulation) or the support (tabletop); a floor-bound support contributes its anchor row
                for rname in ("partner", "target", "support"):
                    rp = comb.roles.get(rname)
                    if rp is None or rp.kind != "scene" or (rp.same_as and not rp.is_floor):
                        continue
                    for o in rp.scene_objects.get(sc, []):
                        for room in _split_rooms(o.get("room")):
                            if room not in rooms:
                                rooms.append(room)
                            if primary is None:
                                primary = room
                comb.scenes.append(sc)
                comb.rooms[sc] = rooms
                if primary is not None:
                    comb.floor_room[sc] = primary
        if not comb.scenes:
            missing = [rp.role for rp in scene_roles if not any(rp.scene_objects.values())]
            raise Skip("no_scene", f"no candidate scene has bindings for every scene role (empty: {missing})")

    # -- grounding ----------------------------------------------------------------------------------------------------
    def ground(self, comb: Combination) -> GroundedProblem:
        iddl = self.iddls[comb.subtask]
        subtask = iddl.subtask
        activity = self.activity_of(iddl)
        # alias
        slot_values = {r: comb.roles[r].alias_token(self.icl, subtask) for r in iddl.alias_slots}
        alias = iddl.fill_alias(slot_values)
        # BDDL groundings
        groundings: Dict[str, RoleGrounding] = {}
        for role in iddl.roles:
            rp = comb.roles[role]
            if rp.kind == "group":
                members = []
                for cat, n in rp.members.items():
                    members.extend([(cat, self._synset(cat))] * n)
                dataset = None
                if rp.pick_row is not None and rp.pick_row.get("dataset"):
                    dataset = rp.pick_row["dataset"]
                groundings[role] = RoleGrounding(role, "group", rp.category, members=members, dataset=dataset)
            elif rp.kind == "sampleable":
                groundings[role] = RoleGrounding(role, "sampleable", rp.category, self._synset(rp.category))
            elif rp.same_as:
                groundings[role] = RoleGrounding(role, "scene", rp.category, same_as=rp.same_as)
            else:
                groundings[role] = RoleGrounding(role, "scene", rp.category, self._synset(rp.category))
        comment = f"{GROUNDER} {self.today}: {alias} ({I.relative_path(iddl.path, self.iddl_root) if iddl.path else iddl.subtask}; {comb.label})"
        if iddl.comment:
            comment += "\n" + iddl.comment
        rendered = write_problem(
            self.bddl_root,
            activity,
            iddl,
            groundings,
            alias,
            comment=comment,
            meta={"subtask": subtask, "iddl": I.relative_path(iddl.path, self.iddl_root) if iddl.path else None,
                  "pick": comb.pick, "slot": comb.slot, "mode": self.opt.mode},
            dry_run=self.opt.dry_run,
        )
        entry = self._customs_entry(iddl, comb, alias, rendered.definition_id)
        gp = GroundedProblem(
            alias=alias,
            bddl_alias=rendered.alias,
            subtask=subtask,
            activity=activity,
            definition_id=rendered.definition_id,
            bddl_path=os.path.join(self.bddl_root, activity, f"problem{rendered.definition_id}.bddl"),
            bddl_text=rendered.text,
            problem=rendered.problem,
            entry=entry,
            iddl=iddl,
            combination=comb,
            rendered=rendered,
        )
        self.grounded.append(gp)
        self.report.problems[subtask].append((alias, rendered.definition_id, comb.label))
        for n in comb.notes:
            self.report.notes[subtask].append(n)
        return gp

    def _customs_entry(self, iddl: I.IDDL, comb: Combination, alias: str, definition_id: int) -> CustomsEntry:
        subtask = iddl.subtask
        existing = self.customs.entries.get(alias)
        existing_dict = existing.to_dict() if existing is not None else None
        entry = self.customs.get_or_create_entry(alias, iddl=I.relative_path(iddl.path, self.iddl_root) if iddl.path else None, definition_id=definition_id)
        entry.generation = G.seed_generation(subtask, existing_dict)
        old_roles = dict(entry.roles)
        old_scenes = dict(entry.scenes)
        entry.roles = {}
        entry.scenes = {}
        pose_roles = set(iddl.pose_evaluated_roles())
        for comp, members in iddl.groups.items():
            if comp in pose_roles:
                pose_roles |= set(members)
        verdicts: Dict[Tuple[str, str, str], Verdict] = {}

        # ---- roles ---------------------------------------------------------------------------------------------------
        for role in iddl.roles:
            rp = comb.roles[role]
            re_ = RoleEntry()
            if rp.kind == "group":
                cats = list(rp.members)
                re_.category = cats[0] if len(cats) == 1 else cats
                re_.sampleable = True
                if rp.pick_row is not None and rp.pick_row.get("dataset"):
                    re_.dataset = rp.pick_row["dataset"]
                if rp.pick_row is not None and rp.kind == "group" and role != "distractors":
                    re_.representative = rp.pick_row["model"]
                seed = entry.generation.setdefault("groups", {}).setdefault(role, G.group_seed(subtask, role))
                seed["size"] = sum(rp.members.values())
                seed["members"] = dict(rp.members)
            elif rp.kind == "sampleable":
                re_.category = rp.category
                re_.sampleable = True
                if rp.pick_row is not None:
                    re_.representative = rp.pick_row["model"]
                elif rp.models:
                    re_.representative = rp.models[0][1]
            else:
                re_.category = rp.category
                re_.sampleable = False
            for req in iddl.require_of(role):
                re_.require.update(req.customs_value())
            # an alias grounded before (or by another slot of this run: same categories, other models) keeps its
            # per-model facts and its representative model; the whitelists are unioned below
            old = old_roles.get(role)
            if old is not None:
                re_.scale.update(old.scale)
                re_.joint.update(old.joint)
                re_.handle_key.update(old.handle_key)
                re_.grasp_keys.update(old.grasp_keys)
                re_.orientation_equivalences.update(old.orientation_equivalences)
                if old.representative and comb.slot != "representative":
                    re_.representative = old.representative
            entry.roles[role] = re_
        entry.roles["floor"] = RoleEntry(sampleable=False)

        # ---- per-model facts of the whitelisted / pool models -------------------------------------------------------
        role_models: Dict[str, List[Tuple[str, str, Dict[str, str]]]] = {}
        for role in iddl.roles:
            rp = comb.roles[role]
            if rp.kind == "scene":
                continue
            wanted: List[Tuple[str, str]] = list(rp.models)
            if rp.kind == "group" and role == "distractors":
                wanted = [(c, m) for c, ms in rp.pools.items() for m in ms]
            elif self.opt.full_whitelists:
                cats = list(rp.members) if rp.kind == "group" else [rp.category]
                for cat in cats:
                    for r in self._pool_of(subtask, role, cat):
                        if (cat, r["model"]) not in wanted:
                            wanted.append((cat, r["model"]))
            rows = []
            for cat, m in wanted:
                r = self.icl.row(subtask, role, cat, m)
                if r is None and role == "distractors":
                    r = self.icl.row("disembedding", "distractors", cat, m)
                if r is None:
                    r = {"subtask": subtask, "role": role, "category": cat, "model": m, "kind": rp.kind, "ready": "", "_synthetic": "1"}
                rows.append((cat, m, r))
            role_models[role] = rows

        # ---- scenes ---------------------------------------------------------------------------------------------------
        for sc in comb.scenes:
            se = SceneEntry(rooms=list(comb.rooms.get(sc, [])))
            for role in iddl.roles:
                rp = comb.roles[role]
                if rp.kind != "scene":
                    continue
                if rp.is_floor:
                    se.bindings[role] = {"room": comb.floor_room.get(sc)} if comb.floor_room.get(sc) else {"room": None}
                    continue
                if rp.same_as and rp.same_as != "floor":
                    other = comb.roles[rp.same_as]
                    se.bindings[role] = [o["model"] for o in other.scene_objects.get(sc, [])]
                    continue
                objs = rp.scene_objects.get(sc, [])
                if objs and any(o.get("region_id") or "/" in (o.get("model") or "") for o in objs):
                    # wall regions (§5.5): bindings.support = wall object names, bindings.support_regions = the segments
                    def _wall(o):
                        return o["model"] if o.get("region_id") else (o["model"].split("/")[1] if o["model"].count("/") == 2 else o["model"])

                    se.bindings[role] = sorted({_wall(o) for o in objs})
                    se.bindings[f"{role}_regions"] = [
                        {"id": o.get("region_id") or o["model"], "wall": _wall(o), "room": o.get("room"), "p0": _parse_vec(o.get("p0")),
                         "p1": _parse_vec(o.get("p1")), "normal": _parse_vec(o.get("normal")),
                         "hang_z": float(o["hang_z"]) if o.get("hang_z") else None,
                         "length_m": float(o["length_m"]) if o.get("length_m") else None,
                         "max_object_m": float(o["max_object_m"]) if o.get("max_object_m") else None}
                        for o in objs
                    ]
                else:
                    se.bindings[role] = [o["model"] for o in objs]
                # scene articulated targets: joint / handle key per scene object (§5.3)
                if any(r.kind == "handle" for r in iddl.require_of(role)):
                    for o in objs:
                        v = self.adm.check(iddl, role, o, scene=sc)
                        if v.reason:
                            self.report.blacklist[subtask][v.reason] += 1
                            entry.roles[role].joint.pop(f"{sc}/{o['model']}", None)
                            comb.notes.append(f"{o['model']}@{sc}: {v.reason} (scene object kept in bindings, no admissible joint)")
                            continue
                        preferred = rp.joint if (rp.pick_row is not None and o["model"] == (rp.pick_row.get("name") or rp.pick_row.get("model"))) else (o.get("joint") or None)
                        chosen = choose_joint(v.handles, preferred) or choose_joint(v.handles)
                        if chosen is None:
                            continue
                        joint, hrow = chosen
                        entry.roles[role].joint[f"{sc}/{o['model']}"] = joint
                        entry.roles[role].handle_key[f"{sc}/{o['model']}"] = hrow.get("key") or o.get("handle_key") or ""
            se.bindings["floor"] = {"room": comb.floor_room.get(sc)} if comb.floor_room.get(sc) else {"room": None}
            # whitelists / blacklists of the spawned roles
            for role, rows in role_models.items():
                rp = comb.roles[role]
                for cat, m, r in rows:
                    v = verdicts.get((role, cat, m))
                    if v is None or role == "obstacles":
                        v = self.adm.check(iddl, role, r, scene=sc)
                        if rp.allow_heldout and v.reason == "heldout" and (cat, m) in rp.models:
                            v = self._recheck_without_heldout(iddl, role, r, sc)
                        verdicts[(role, cat, m)] = v
                    if role == "distractors" and self.icl.are_lookalikes((cat, m), tuple(comb.roles["target"].models[0])):
                        v = Verdict(reason="lookalike_of_target")
                    if v.reason:
                        se.blacklist_add(role, cat, m, self._reason(v.reason))
                        self.report.blacklist[subtask][v.reason] += 1
                        continue
                    se.whitelist_add(role, cat, m)
                    self._model_facts(entry.roles[role], iddl, role, cat, m, r, v, rp)
                    for p in v.pending:
                        rec = Constraint("blacklist", {"scene": sc, role: C.model_key(cat, m)}, False,
                                         C.constraint_source("ground_isg", VERSION, date=self.today), {"reason": p, "pending": True})
                        if entry.add_constraint(rec):
                            self.report.pending[p] += 1
            # held-out models of every spawned role's categories (global rule)
            for role in iddl.roles:
                rp = comb.roles[role]
                if rp.kind == "scene":
                    continue
                cats = list(rp.members) if rp.kind == "group" else [rp.category]
                for cat, m in sorted(self.icl.heldout_models):
                    if cat in cats and not (rp.allow_heldout and (cat, m) in rp.models):
                        if not se.is_blacklisted(role, cat, m):
                            se.blacklist_add(role, cat, m, self._reason("heldout"))
                            self.report.blacklist[subtask]["heldout"] += 1
            # an existing scene block: its whitelisted models of the same categories stay (union), the rejections a
            # sampler recorded stay (ground_isg's own reasons were re-derived above)
            old = old_scenes.get(sc)
            if old is not None:
                for role, cats in old.whitelist.items():
                    rp = comb.roles.get(role)
                    if rp is None or rp.kind == "scene":
                        continue
                    own = set(rp.members) if rp.kind == "group" else {rp.category}
                    for cat, models in cats.items():
                        if cat in own:
                            for m in models:
                                se.whitelist_add(role, cat, m)
                for role, cat, m, reason in old.blacklist_reasons():
                    if not se.is_blacklisted(role, cat, m) and not reason.startswith("ground_isg"):
                        se.blacklist_add(role, cat, m, reason)
            entry.scenes[sc] = se
        return entry

    def _recheck_without_heldout(self, iddl: I.IDDL, role: str, row: Dict[str, str], scene: str) -> Verdict:
        """The ``heldout`` slot was requested for this role: run the other checks on the held-out model."""
        saved = set(self.icl.heldout_models)
        self.icl.heldout_models.discard((row["category"], row["model"]))
        try:
            return self.adm.check(iddl, role, row, scene=scene)
        finally:
            self.icl.heldout_models |= saved

    def _model_facts(self, re_: RoleEntry, iddl: I.IDDL, role: str, cat: str, m: str, row: Dict[str, str], v: Verdict, rp: RolePick) -> None:
        key = C.model_key(cat, m)
        subtask = iddl.subtask
        if v.grasp_key:
            re_.grasp_keys[key] = v.grasp_key
        scale = _parse_scale(row.get("scale") or "")
        if scale is not None and key not in re_.scale:
            re_.scale[key] = scale
        if v.handles:
            preferred = rp.joint if (rp.pick_row is not None and rp.pick_row.get("model") == m) else (row.get("joint") or None)
            chosen = choose_joint(v.handles, preferred) or choose_joint(v.handles)
            if chosen is not None:
                joint, hrow = chosen
                re_.joint.setdefault(key, joint)
                re_.handle_key.setdefault(key, hrow.get("key") or row.get("handle_key") or f"{cat}__{m}")
        if role in set(iddl.pose_evaluated_roles()) | {mm for comp, ms in iddl.groups.items() if comp in iddl.pose_evaluated_roles() for mm in ms}:
            tags = self._orientation_equivalences(subtask, role, row)
            if tags is not None:
                re_.orientation_equivalences[key] = tags

    @staticmethod
    def _orientation_equivalences(subtask: str, role: str, row: Dict[str, str]) -> Optional[List[str]]:
        """§3.13 rule 2: the symmetry tags of the model (wall targets: the reviewed ``symmetric_180`` half turn about
        the wall normal = front axis); None where the orientation is not evaluated (object scaling depth metric,
        inside target, blocks)."""
        if subtask == "object_scaling":
            return None
        if subtask == "configuration_matching-block_design":
            return None
        if subtask == "inter_object_relation_satisfaction-inside" and role == "target":
            return None
        if subtask in ("alignment-world_vertical", "alignment-world_horizontal"):
            if (row.get("symmetric_180") or "") == "1" and row.get("frame_front"):
                return [C.half_turn_tag(row["frame_front"])]
            return []
        if not (row.get("symmetry") or ""):
            return None  # not surveyed (scene rows, trays, articulated targets): no claim
        return _symmetry_tags(row)

    # -- run ------------------------------------------------------------------------------------------------------------
    def run(self, subtasks: Optional[Sequence[str]] = None) -> Report:
        wanted = list(subtasks) if subtasks else sorted(self.iddls)
        unknown = [s for s in wanted if s not in self.iddls]
        if unknown:
            raise GroundingError(f"unknown subtask id(s) {unknown}; known: {sorted(self.iddls)}")
        for subtask in wanted:
            combos = self.combinations(subtask)
            if not combos and not self.report.skipped.get(subtask):
                self.report.skipped[subtask].append((subtask, f"no_combination: no candidate rows carry the {self.opt.mode} mark"))
            for comb in combos:
                try:
                    gp = self.ground(comb)
                except Skip as s:
                    self.report.skipped[subtask].append((comb.label, str(s)))
                    continue
                except ISGProblemError as e:
                    # the rendered BDDL is rejected by ispatialgym.activity.problem (e.g. a synset the INSTANCE_RE
                    # of the parser does not accept): nothing is written for this combination
                    self.report.skipped[subtask].append((comb.label, f"bddl_rejected: {e}"))
                    log.warning("%s: BDDL rejected by the parser: %s", comb.label, e)
                    continue
                except (GroundingError, I.IDDLError, KeyError) as e:
                    self.report.skipped[subtask].append((comb.label, f"error: {e}"))
                    log.warning("%s: %s", comb.label, e)
                    continue
                if self.opt.representative_test and not self.opt.dry_run:
                    for sc in comb.scenes:
                        outcome = run_representative_test(gp, self.customs, sc)
                        if outcome:
                            self.report.tests.append(f"{gp.alias} @ {sc}: {outcome}")
            if not self.opt.dry_run:
                self.customs.save()
        return self.report


# ----------------------------------------------------------------------------------------------------------------
# D2 hook
# ----------------------------------------------------------------------------------------------------------------


def run_representative_test(problem: GroundedProblem, customs: Customs, scene: str) -> None:
    """**D2 HOOK — representative-model test of a grounded alias (spec §9): template + instance 0.**

    ``ground_isg`` calls this once per grounded problem and candidate scene after the BDDL and the customs entry are
    written. The v2 sampler (TODO D2, ``ispatialgym.sampling.v2``) implements it as::

        ispatialgym.sampling.v2.core.run_representative_test(problem: ispatialgym.sampling.ground_isg.GroundedProblem,
                                                             customs: ispatialgym.sampling.customs.Customs,
                                                             scene: str) -> Optional[str]

    where ``problem`` carries ``alias`` / ``bddl_alias`` / ``subtask`` / ``activity`` / ``definition_id`` /
    ``bddl_path`` / ``bddl_text`` / ``problem`` (the parsed ``ISGProblem``) / ``entry`` (the live ``CustomsEntry``,
    same object as ``customs[problem.alias]``) / ``iddl`` / ``combination`` (the picks per role), and the
    implementation (a) runs the full initialization once with the representative models on the scene's bindings, (b)
    on success keeps template + instance 0 and appends the learned constraints to ``problem.entry.constraints``, (c) on
    failure moves the representative model to ``problem.entry.scenes[scene].blacklist`` with the rejection reason
    (``customs.blacklist_reason`` format) and retries the next representative once, and (d) returns a one-line
    outcome (or None). ``ground_isg`` saves the customs file after every subtask, so in-place edits of ``problem.entry``
    are persisted. The import is lazy (the sampler needs the simulator stack); until the function exists the test is
    skipped with a log line and the whitelists stay as the icl derived them.
    """
    try:
        from ispatialgym.sampling.v2 import core as v2_core  # noqa: WPS433 (lazy: needs OmniGibson)
    except Exception as e:  # ImportError, or the simulator stack failing to import
        _log_skip(f"ispatialgym.sampling.v2.core not available ({e})", problem.alias, scene)
        return None
    impl = getattr(v2_core, "run_representative_test", None)
    if impl is None:
        _log_skip("ispatialgym.sampling.v2.core has no run_representative_test yet", problem.alias, scene)
        return None
    return impl(problem, customs, scene)


_SKIP_LOGGED: Set[str] = set()


def _log_skip(why: str, alias: str, scene: str) -> None:
    """One INFO line per reason (the per-problem lines go to DEBUG)."""
    if why not in _SKIP_LOGGED:
        _SKIP_LOGGED.add(why)
        log.info("representative test skipped (TODO D2): %s — first for %s @ %s; further skips logged at DEBUG", why, alias, scene)
    else:
        log.debug("representative test skipped for %s @ %s: %s", alias, scene, why)


# ----------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------


def _parse_pattern(text: str) -> Tuple[Tuple[str, int], ...]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        kind, _, n = part.partition(":")
        if kind not in ("same", "other") or not n.isdigit():
            raise argparse.ArgumentTypeError(f"--distractors expects 'same:N,other:M', got {text!r}")
        out.append((kind, int(n)))
    if not out:
        raise argparse.ArgumentTypeError("--distractors must name at least one pool")
    return tuple(out)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ispatialgym.sampling.ground_isg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--representative-only", dest="mode", action="store_const", const="representative", help="ground the picks of roles/representatives.csv (default)")
    mode.add_argument("--test-only", dest="mode", action="store_const", const="test", help="ground the test = 1 candidates of roles/candidates.csv")
    mode.add_argument("--all", dest="mode", action="store_const", const="all", help="ground every admitted candidate combination")
    ap.set_defaults(mode="representative")
    ap.add_argument("--subtask", action="append", default=None, help="iddl subtask id (repeatable); default: all 21")
    ap.add_argument("--slots", default="representative", help="comma list of representatives.csv slots to ground (representative, variation_texture, variation_geometry, heldout)")
    ap.add_argument("--scene", action="append", default=None, help="restrict the scene blocks to these scenes (repeatable)")
    ap.add_argument("--iddl", default=DEFAULT_IDDL, help="iddl root")
    ap.add_argument("--icl", default=DEFAULT_ICL, help="icl root")
    ap.add_argument("--out-bddl", default=DEFAULT_BDDL_ROOT, help="activity_definitions root the problem<N>.bddl / problems.json are written to")
    ap.add_argument("--customs", default=DEFAULT_CUSTOMS, help="task customs list (created when absent, entries updated in place)")
    ap.add_argument("--full-whitelists", action="store_true", help="whitelist every admissible candidate of the category, not only the pick's model")
    ap.add_argument("--distractors", type=_parse_pattern, default=(("same", 2), ("other", 1)), help="disembedding distractor member pattern (default same:2,other:1)")
    ap.add_argument("--legacy-activity-names", action="store_true", help="write the alignment family under the retired activity name world_alignment")
    ap.add_argument("--strict-symmetric-180", action="store_true", help="spec 2.2 literally: wall targets without a filled symmetric_180 are missing_frame")
    ap.add_argument("--no-representative-test", action="store_true", help="do not call the D2 representative-test hook")
    ap.add_argument("--dry-run", action="store_true", help="ground and report, write nothing")
    ap.add_argument("--report", default=None, help="also write the report (markdown) to this path; .json alongside")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    slots = tuple(s.strip() for s in args.slots.split(",") if s.strip())
    bad = [s for s in slots if s not in SLOTS]
    if bad:
        print(f"unknown slot(s) {bad}; expected {SLOTS}", file=sys.stderr)
        return 2
    opt = Options(
        mode=args.mode,
        slots=slots,
        scenes=tuple(args.scene) if args.scene else None,
        full_whitelists=args.full_whitelists,
        distractor_pattern=args.distractors,
        legacy_activity_names=args.legacy_activity_names,
        strict_symmetric_180=args.strict_symmetric_180,
        dry_run=args.dry_run,
        representative_test=not args.no_representative_test,
    )
    iddls = I.load_all(args.iddl)
    icl = ICL(args.icl)
    customs = Customs.load(args.customs)
    if customs.comment is None:
        customs.comment = ("iSpatialGym task customs list (docs/instance_sampling_spec.md 2.3), written by ground_isg; one entry per grounded "
                           "alias: roles, per-scene bindings / whitelist / blacklist, generation (seeded from sampling/generation_defaults.py), constraints.")
    grounder = Grounder(iddls, icl, customs, bddl_root=args.out_bddl, options=opt, iddl_root=args.iddl)
    report = grounder.run(args.subtask)
    text = report.text()
    print(text)
    if args.report:
        with open(args.report, "w") as f:
            f.write(text)
        with open(os.path.splitext(args.report)[0] + ".json", "w") as f:
            json.dump(report.to_json(), f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
