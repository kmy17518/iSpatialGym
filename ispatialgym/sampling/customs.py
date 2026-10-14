"""Task customs list (iSpatialGym's task-custom list) — the schema of ``docs/instance_sampling_spec.md`` §2.3.

One entry per subtask (``ispatialgym.eval.names.subtask_name``: the task and its objects, the support left out)::

    {
      "<subtask>": {
        "iddl": "inter_object_relation_satisfaction/nextto.iddl",
        "definition_id": 3,
        "roles": {
          "target":  {"category": "spatula", "sampleable": true, "require": {"grasps": "tabletop"}, "representative": "hkvsep"},
          "partner": {"category": "soup_ladle", "sampleable": true, "representative": "cfzneb"},
          "support": {"category": "breakfast_table", "sampleable": false},
          "floor":   {"sampleable": false}
        },
        "scenes": {
          "house_single_floor": {
            "rooms": ["dining_room_0"],
            "bindings": {"support": ["breakfast_table_rhjoby_0"], "floor": {"room": "dining_room_0"}},
            "whitelist": {"target": {"spatula": ["hkvsep", ...]}, "partner": {"soup_ladle": [...]}},
            "blacklist": {"target": {"spatula": {"pdqgzq": "ground_isg 2026-09-20: missing_grasps:tabletop"}}}
          }
        },
        "generation": {...},            # sampling/generation_defaults.py seeds it; generation only (never read at evaluation)
        "evaluation": {"tolerances": {"pose": {"position": m, "orientation": rad}, "layout": {...},
                                      "joints": {"revolute": rad, "prismatic": m, "fraction": f}}},  # optional, hand-tuned
                                      # evaluation tolerances overriding the isg-v2 profile (ispatialgym.eval.tolerance_profile)
        "constraints": [ {"scope": {...}, "type": "stance", ..., "hard": false, "source": {...}},
                         {"scope": {}, "type": "hold", "robot": "r1pro", "depth_m": 0.5, "trunk": [0.5, 0.0], "roll_deg": 15.0,
                          "eef_in_base": {"position": [], "orientation": []}, ..., "hard": false, "source": {...}} ]
      }
    }

``roles.<role>`` optional keys (§2.3): ``scale`` (``{"<category>/<model>": s | [sx, sy, sz], "default": s}`` — the
spawn scale per model; ``default`` is the spec §5.5 dict form for a role-wide scale, written by the scale fallbacks),
``joint`` / ``handle_key`` (``{"<category>/<model>" | "<scene>/<object name>": ...}``), ``dataset`` (non-BEHAVIOR
assets), ``orientation_equivalences`` (§3.13: ``{"<category>/<model>": [symmetry tags]}`` — the rotation group is
enumerated with :func:`equivalence_group`), ``grasp_keys`` (``{"<category>/<model>": "<require/grasps.csv key>"}``,
the explicit library key of the pipeline so no sampler guesses a suffix). ``category`` is a string, or a list for a
group role whose members come from several categories (the subtask name lists them).

``scenes.<scene>.bindings.<role>`` of a scene role is a list of scene object names, or — when the role is bound to the
**floor** of the task room (large-scale IORS floor init, §5.6; small-scale articulation targets taller than 0.6 m,
§5.4; the implicit ``floor`` role) — ``{"room": "<room instance>"}``: the subtask's sampling key carries the token
``floor`` and ``roles.<role>.category`` is ``floors``. Wall regions add ``bindings.<role>_regions`` (§5.5).

The grounding's fallback fields (``fallbacks``, ``fallback_of``, ``fallback_rank``, ``fallback_kind``, ``fallback_params``,
``superseded_by``) are read and written when present; the eval release's list has none (the fallback entries they name
are not part of it).

Dataclasses here are thin, typed views over that JSON; :class:`Customs` loads / saves the file and hands out entries.
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ispatialgym import ASSETS_ROOT

log = logging.getLogger("ispatialgym.customs")

DEFAULT_CUSTOMS_PATH = os.path.join(ASSETS_ROOT, "task_specs", "isg_customs.json")

#: The seven candidate scenes (spec §2.3 / §3.1).
SCENE_CANDIDATES = (
    "house_single_floor",
    "house_double_floor_upper",
    "house_double_floor_lower",
    "Rs_int",
    "hotel_suite_large",
    "restaurant_diner",
    "office_cubicles_right",
)

#: §2.3 constraint record types and scope keys. ``hold`` (beyond the §2.3 list) is the robot-level constant of the
#: held-object tasks (spec §6.3, ``sampling/v2/tasks/held_object.py``): the fixed hold the right arm carries the
#: object in, written once per entry with an empty scope (it depends on the robot, not on a scene / support / model):
#: ``robot``, ``depth_m`` (pad centre on the optical axis), ``trunk`` ``[translate, tilt]``, ``roll_deg`` (closing
#: axis rolled from world down about the approach), the hold eef pose ``eef_in_base`` (``{"position", "orientation"}``
#: in the base frame), ``approach`` / ``closing`` descriptions, ``right_arm_q`` (the IK solution) and ``idle_arms``.
CONSTRAINT_TYPES = ("stance", "grasp_set", "joint_set", "placement", "scale", "band_ref", "contacts", "blacklist", "hold")
SCOPE_KEYS = ("scene", "support", "partner", "target", "model", "support_height", "trunk")
#: Fields a ``hold`` record carries (``Constraint.fields``); the write-back adds ``source``.
HOLD_FIELDS = ("robot", "depth_m", "trunk", "roll_deg", "eef_in_base", "approach", "closing", "right_arm_q", "idle_arms")
#: Keys of a customs entry, in the order they are written (the fallback keys only when set). Any other top-level key of
#: an entry is a hand-written owner / agent record (D-004 ``f8e9aa7`` ``blacklisted``: the office_cubicles_right cabinets
#: removed from the subtask; ``pick_note`` of the Rs_int drawer pick) — kept verbatim in ``CustomsEntry.notes`` and written
#: back after the schema keys, never interpreted here.
ENTRY_KEYS = ("iddl", "definition_id", "fallback_of", "fallback_rank", "fallback_kind", "fallback_params", "fallbacks", "superseded_by",
              "roles", "scenes", "generation", "constraints", "evaluation")
#: ``fallback_kind`` values (see the module docstring).
FALLBACK_KINDS = ("init_surface", "scale", "next_model")
#: Keys of a role entry, in the order they are written (§2.3 + the documented extensions).
ROLE_KEYS = (
    "category",
    "sampleable",
    "require",
    "representative",
    "scale",
    "joint",
    "handle_key",
    "dataset",
    "orientation_equivalences",
    "grasp_keys",
)
SCENE_KEYS = ("rooms", "bindings", "whitelist", "blacklist")

BEHAVIOR_DATASET = "behavior-1k-assets"


class CustomsError(ValueError):
    pass


def blacklist_reason(reason: str, tool: str = "ground_isg", date: Optional[str] = None) -> str:
    """The blacklist value format of the §2.3 example: ``"ground_isg 2026-09-20: missing_grasps:tabletop"``."""
    return f"{tool} {date or _dt.date.today().isoformat()}: {reason}"


def reason_code(value: str) -> str:
    """``"ground_isg 2026-09-20: missing_grasps:tabletop"`` -> ``"missing_grasps:tabletop"`` (the first token of the
    reason; a free-text reason keeps its first word)."""
    text = value.split(": ", 1)[1] if ": " in value else value
    return text.split()[0] if text.split() else text


def model_key(category: str, model: str) -> str:
    return f"{category}/{model}"


# ----------------------------------------------------------------------------------------------------------------
# Roles
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class RoleEntry:
    """``roles.<role>`` of an entry."""

    category: Union[None, str, List[str]] = None
    sampleable: bool = False
    require: Dict[str, Any] = field(default_factory=dict)  # {"grasps": "<pipeline>"} | {"contacts": true} | {"handle": true}
    representative: Optional[str] = None
    scale: Dict[str, Any] = field(default_factory=dict)  # "<category>/<model>" -> s | [sx, sy, sz]
    joint: Dict[str, str] = field(default_factory=dict)  # "<category>/<model>" | "<scene>/<name>" -> joint
    handle_key: Dict[str, str] = field(default_factory=dict)
    dataset: Optional[str] = None  # None = BEHAVIOR assets
    orientation_equivalences: Dict[str, List[str]] = field(default_factory=dict)  # "<category>/<model>" -> tags
    grasp_keys: Dict[str, str] = field(default_factory=dict)  # "<category>/<model>" -> grasp library key

    @property
    def categories(self) -> List[str]:
        if self.category is None:
            return []
        return [self.category] if isinstance(self.category, str) else list(self.category)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.category is not None:
            d["category"] = self.category
        d["sampleable"] = bool(self.sampleable)
        if self.require:
            d["require"] = dict(self.require)
        if self.representative is not None:
            d["representative"] = self.representative
        for key in ("scale", "joint", "handle_key"):
            val = getattr(self, key)
            if val:
                d[key] = dict(val)
        if self.dataset:
            d["dataset"] = self.dataset
        if self.orientation_equivalences:
            d["orientation_equivalences"] = {k: list(v) for k, v in self.orientation_equivalences.items()}
        if self.grasp_keys:
            d["grasp_keys"] = dict(self.grasp_keys)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RoleEntry":
        unknown = set(d) - set(ROLE_KEYS)
        if unknown:
            raise CustomsError(f"role entry has unknown keys {sorted(unknown)}")
        return cls(
            category=d.get("category"),
            sampleable=bool(d.get("sampleable", False)),
            require=dict(d.get("require") or {}),
            representative=d.get("representative"),
            scale=dict(d.get("scale") or {}),
            joint=dict(d.get("joint") or {}),
            handle_key=dict(d.get("handle_key") or {}),
            dataset=d.get("dataset"),
            orientation_equivalences={k: list(v) for k, v in (d.get("orientation_equivalences") or {}).items()},
            grasp_keys=dict(d.get("grasp_keys") or {}),
        )


# ----------------------------------------------------------------------------------------------------------------
# Scenes
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class SceneEntry:
    """``scenes.<scene>``: task rooms, bindings of the scene roles, per-scene whitelist / blacklist."""

    rooms: List[str] = field(default_factory=list)
    bindings: Dict[str, Any] = field(default_factory=dict)  # role -> [scene object names] | {"room": ...} | {"category": [...]}
    whitelist: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)  # role -> category -> [models]
    blacklist: Dict[str, Dict[str, Dict[str, str]]] = field(default_factory=dict)  # role -> category -> model -> reason
    # hand-written comments of the scene block (keys starting with ``_``, e.g. the sandal subtasks' ``_note``): kept verbatim
    notes: Dict[str, Any] = field(default_factory=dict)

    def whitelist_add(self, role: str, category: str, model: str) -> None:
        if self.is_blacklisted(role, category, model):
            return
        models = self.whitelist.setdefault(role, {}).setdefault(category, [])
        if model not in models:
            models.append(model)

    def blacklist_add(self, role: str, category: str, model: str, reason: str) -> None:
        """Blacklist ``model`` for ``role`` with ``reason`` (removed from the whitelist; the first reason is kept)."""
        entry = self.blacklist.setdefault(role, {}).setdefault(category, {})
        entry.setdefault(model, reason)
        wl = self.whitelist.get(role, {}).get(category)
        if wl and model in wl:
            wl.remove(model)
        if wl is not None and not wl:  # no empty lists linger (canonical form, `whitelist.get(role)` falsy when empty)
            del self.whitelist[role][category]
            if not self.whitelist[role]:
                del self.whitelist[role]

    def is_blacklisted(self, role: str, category: str, model: str) -> bool:
        return model in self.blacklist.get(role, {}).get(category, {})

    def whitelisted(self, role: str) -> Dict[str, List[str]]:
        return {c: list(ms) for c, ms in self.whitelist.get(role, {}).items()}

    def blacklist_reasons(self) -> Iterable[Tuple[str, str, str, str]]:
        for role, cats in self.blacklist.items():
            for cat, models in cats.items():
                for model, reason in models.items():
                    yield role, cat, model, reason

    def to_dict(self) -> Dict[str, Any]:
        """Canonical form (a re-grounding writes identical text): models sorted, empty categories / roles dropped."""
        whitelist = {r: {c: sorted(ms) for c, ms in cats.items() if ms} for r, cats in self.whitelist.items()}
        blacklist = {r: {c: {m: ms[m] for m in sorted(ms)} for c, ms in cats.items() if ms} for r, cats in self.blacklist.items()}
        return {
            "rooms": list(self.rooms),
            "bindings": copy.deepcopy(self.bindings),
            "whitelist": {r: cats for r, cats in whitelist.items() if cats},
            "blacklist": {r: cats for r, cats in blacklist.items() if cats},
            **copy.deepcopy(self.notes),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SceneEntry":
        notes = {k: copy.deepcopy(v) for k, v in d.items() if k.startswith("_")}
        unknown = set(d) - set(SCENE_KEYS) - set(notes)
        if unknown:
            raise CustomsError(f"scene entry has unknown keys {sorted(unknown)}")
        return cls(
            rooms=list(d.get("rooms") or []),
            bindings=copy.deepcopy(d.get("bindings") or {}),
            whitelist={r: {c: list(ms) for c, ms in (cats or {}).items()} for r, cats in (d.get("whitelist") or {}).items()},
            blacklist={r: {c: dict(ms) for c, ms in (cats or {}).items()} for r, cats in (d.get("blacklist") or {}).items()},
            notes=notes,
        )


# ----------------------------------------------------------------------------------------------------------------
# Constraints
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class Constraint:
    """A typed, scoped record (§2.3): ``type`` ∈ :data:`CONSTRAINT_TYPES`, ``scope`` over :data:`SCOPE_KEYS`,
    ``hard`` (rejection rule) or soft (prior), ``source`` (instance, sampler, date) and the type's own fields."""

    type: str
    scope: Dict[str, Any] = field(default_factory=dict)
    hard: bool = False
    source: Dict[str, Any] = field(default_factory=dict)
    fields: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.type not in CONSTRAINT_TYPES:
            raise CustomsError(f"unknown constraint type {self.type!r}; expected one of {CONSTRAINT_TYPES}")
        unknown = set(self.scope) - set(SCOPE_KEYS)
        if unknown:
            raise CustomsError(f"constraint scope has unknown keys {sorted(unknown)}; allowed: {SCOPE_KEYS}")
        clash = set(self.fields) & {"type", "scope", "hard", "source"}
        if clash:
            raise CustomsError(f"constraint fields must not use the reserved keys {sorted(clash)}")

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"scope": copy.deepcopy(self.scope), "type": self.type}
        d.update(copy.deepcopy(self.fields))
        d["hard"] = bool(self.hard)
        d["source"] = copy.deepcopy(self.source)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Constraint":
        if "type" not in d:
            raise CustomsError(f"constraint record without type: {d}")
        fields = {k: v for k, v in d.items() if k not in ("type", "scope", "hard", "source")}
        return cls(type=d["type"], scope=dict(d.get("scope") or {}), hard=bool(d.get("hard", False)), source=dict(d.get("source") or {}), fields=fields)

    def same_as(self, other: "Constraint") -> bool:
        """Equal type / scope / fields (the source may differ): used to avoid duplicate records on re-grounding."""
        return self.type == other.type and self.scope == other.scope and self.fields == other.fields and self.hard == other.hard


def constraint_source(tool: str = "ground_isg", version: str = "2.0.0", instance: Optional[str] = None, date: Optional[str] = None) -> Dict[str, Any]:
    src: Dict[str, Any] = {"sampler": f"{tool} {version}", "date": date or _dt.date.today().isoformat()}
    if instance is not None:
        src["instance"] = instance
    return src


# ----------------------------------------------------------------------------------------------------------------
# Entry / file
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class CustomsEntry:
    """One subtask of the customs list."""

    subtask: str
    iddl: Optional[str] = None  # "<family>/<task>.iddl" relative to the iddl root
    definition_id: Optional[int] = None
    roles: Dict[str, RoleEntry] = field(default_factory=dict)
    scenes: Dict[str, SceneEntry] = field(default_factory=dict)
    generation: Dict[str, Any] = field(default_factory=dict)
    constraints: List[Constraint] = field(default_factory=list)
    # read at evaluation only (``evaluation.tolerances``: ispatialgym.eval.tolerance_profile); empty unless tuned by hand
    evaluation: Dict[str, Any] = field(default_factory=dict)
    # fallback fields (module docstring): set on a fallback entry ...
    fallback_of: Optional[str] = None
    fallback_rank: Optional[int] = None
    fallback_kind: Optional[str] = None
    fallback_params: Dict[str, Any] = field(default_factory=dict)
    # ... and, on a primary, its fallbacks in rank order
    fallbacks: List[str] = field(default_factory=list)
    # set when a later grounding replaced this entry (the replacing entry's key, or True): dormant, never queued
    superseded_by: Union[None, str, bool] = None
    # hand-written owner / agent records outside the schema (``blacklisted``, ``pick_note``, ...): kept verbatim
    notes: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_fallback(self) -> bool:
        return self.fallback_of is not None

    @property
    def is_live(self) -> bool:
        return not self.superseded_by

    def set_fallback(self, primary: str, rank: int, kind: str, params: Optional[Dict[str, Any]] = None) -> None:
        if kind not in FALLBACK_KINDS:
            raise CustomsError(f"unknown fallback kind {kind!r}; expected one of {FALLBACK_KINDS}")
        if rank < 1:
            raise CustomsError("fallback_rank starts at 1")
        self.fallback_of, self.fallback_rank, self.fallback_kind = primary, int(rank), kind
        self.fallback_params = dict(params or {})

    def role(self, name: str) -> RoleEntry:
        return self.roles.setdefault(name, RoleEntry())

    def scene(self, name: str) -> SceneEntry:
        if name not in SCENE_CANDIDATES:
            raise CustomsError(f"{name!r} is not a candidate scene {SCENE_CANDIDATES}")
        return self.scenes.setdefault(name, SceneEntry())

    def whitelist(self, scene: str, role: str, category: str, model: str) -> None:
        self.scene(scene).whitelist_add(role, category, model)

    def blacklist(self, scene: str, role: str, category: str, model: str, reason: str) -> None:
        self.scene(scene).blacklist_add(role, category, model, reason)

    def blacklist_everywhere(self, role: str, category: str, model: str, reason: str) -> None:
        for sc in self.scenes.values():
            sc.blacklist_add(role, category, model, reason)

    def add_constraint(self, constraint: Constraint, dedupe: bool = True) -> bool:
        """Append ``constraint``; with ``dedupe`` an equal record (type / scope / fields) is not appended twice."""
        if dedupe and any(c.same_as(constraint) for c in self.constraints):
            return False
        self.constraints.append(constraint)
        return True

    def constraints_of(self, ctype: Optional[str] = None, **scope) -> List[Constraint]:
        out = []
        for c in self.constraints:
            if ctype is not None and c.type != ctype:
                continue
            if all(k not in c.scope or c.scope[k] == v for k, v in scope.items()):
                out.append(c)
        return out

    def sampleable_roles(self) -> List[str]:
        return [r for r, e in self.roles.items() if e.sampleable]

    def scene_roles(self) -> List[str]:
        return [r for r, e in self.roles.items() if not e.sampleable]

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"iddl": self.iddl, "definition_id": self.definition_id}
        if self.fallback_of is not None:
            d["fallback_of"] = self.fallback_of
            d["fallback_rank"] = self.fallback_rank
            d["fallback_kind"] = self.fallback_kind
            if self.fallback_params:
                d["fallback_params"] = copy.deepcopy(self.fallback_params)
        if self.fallbacks:
            d["fallbacks"] = list(self.fallbacks)
        if self.superseded_by:
            d["superseded_by"] = self.superseded_by
        d.update(
            {
                "roles": {r: e.to_dict() for r, e in self.roles.items()},
                "scenes": {s: e.to_dict() for s, e in self.scenes.items()},
                "generation": copy.deepcopy(self.generation),
                "constraints": [c.to_dict() for c in self.constraints],
            }
        )
        if self.evaluation:
            d["evaluation"] = copy.deepcopy(self.evaluation)
        for k, v in self.notes.items():
            if k not in d:
                d[k] = copy.deepcopy(v)
        return d

    @classmethod
    def from_dict(cls, subtask: str, d: Dict[str, Any]) -> "CustomsEntry":
        unknown = sorted(set(d) - set(ENTRY_KEYS))
        if unknown:
            # hand-written records (owner blacklists, pick notes) ride along; a schema key misspelt by hand would too, so say so
            log.info("customs entry %r: keys outside the schema kept verbatim: %s", subtask, unknown)
        entry = cls(
            subtask=subtask,
            iddl=d.get("iddl"),
            definition_id=None if d.get("definition_id") is None else int(d["definition_id"]),
            roles={r: RoleEntry.from_dict(e) for r, e in (d.get("roles") or {}).items()},
            scenes={s: SceneEntry.from_dict(e) for s, e in (d.get("scenes") or {}).items()},
            generation=copy.deepcopy(d.get("generation") or {}),
            constraints=[Constraint.from_dict(c) for c in (d.get("constraints") or [])],
            evaluation=copy.deepcopy(d.get("evaluation") or {}),
            fallbacks=list(d.get("fallbacks") or []),
            superseded_by=d.get("superseded_by") or None,
            notes={k: copy.deepcopy(d[k]) for k in unknown},
        )
        if d.get("fallback_of") is not None:
            entry.set_fallback(d["fallback_of"], int(d.get("fallback_rank") or 1), d.get("fallback_kind") or "init_surface", d.get("fallback_params"))
            legacy = entry.fallback_params.pop("superseded_by", None)  # written at the top level since 2026-09-16
            if legacy and not entry.superseded_by:
                entry.superseded_by = legacy
        return entry


class Customs:
    """The customs list file: ``{subtask: entry}`` (a leading ``_comment`` string is kept)."""

    def __init__(self, path: Optional[str] = None, entries: Optional[Dict[str, CustomsEntry]] = None, comment: Optional[str] = None):
        self.path = path
        self.entries: Dict[str, CustomsEntry] = dict(entries or {})
        self.comment = comment

    @classmethod
    def load(cls, path: Optional[str] = None, missing_ok: bool = True) -> "Customs":
        path = path or DEFAULT_CUSTOMS_PATH
        if not os.path.exists(path):
            if not missing_ok:
                raise FileNotFoundError(path)
            return cls(path=path)
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise CustomsError(f"{path}: the customs list must be a JSON object keyed by subtask")
        comment = data.pop("_comment", None)
        entries = {subtask: CustomsEntry.from_dict(subtask, d) for subtask, d in data.items()}
        return cls(path=path, entries=entries, comment=comment)

    def save(self, path: Optional[str] = None) -> str:
        path = path or self.path or DEFAULT_CUSTOMS_PATH
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data: Dict[str, Any] = {}
        if self.comment:
            data["_comment"] = self.comment
        for subtask in sorted(self.entries):
            data[subtask] = self.entries[subtask].to_dict()
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1)
            f.write("\n")
        os.replace(tmp, path)
        self.path = path
        return path

    def __contains__(self, subtask: str) -> bool:
        return subtask in self.entries

    def __getitem__(self, subtask: str) -> CustomsEntry:
        return self.entries[subtask]

    def __len__(self) -> int:
        return len(self.entries)

    def get_or_create_entry(self, subtask: str, iddl: Optional[str] = None, definition_id: Optional[int] = None) -> CustomsEntry:
        """The entry of ``subtask`` (created empty when absent; ``iddl`` / ``definition_id`` are set when given)."""
        entry = self.entries.get(subtask)
        if entry is None:
            entry = CustomsEntry(subtask=subtask)
            self.entries[subtask] = entry
        if iddl is not None:
            entry.iddl = iddl
        if definition_id is not None:
            entry.definition_id = int(definition_id)
        return entry

    def to_dict(self) -> Dict[str, Any]:
        return {subtask: e.to_dict() for subtask, e in sorted(self.entries.items())}


# ----------------------------------------------------------------------------------------------------------------
# Orientation equivalences (spec §3.13)
# ----------------------------------------------------------------------------------------------------------------

_AXES = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}
#: Enumeration step of continuous symmetry axes (``xinf`` ...): 5° = 72 quaternions per axis, finer than every
#: orientation tolerance of §8.
CONTINUOUS_STEP_DEG = 5.0
#: Safety cap on the enumerated group (``zinf x2`` gives 144 elements, ``z4 x2`` 8).
MAX_GROUP_ELEMENTS = 4096


def parse_symmetry_tags(value: Optional[str]) -> List[str]:
    """``"zinf x2"`` / ``"z2,x2"`` -> ``["zinf", "x2"]``; ``none`` / empty -> ``[]``."""
    if not value:
        return []
    tags = [t for t in value.replace(",", " ").split() if t]
    if tags == ["none"]:
        return []
    for t in tags:
        if t != "full" and not (len(t) >= 2 and t[0] in _AXES and (t[1:] == "inf" or t[1:].isdigit())):
            raise CustomsError(f"unknown symmetry tag {t!r} in {value!r}")
    return tags


def _quat_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Quaternion (x, y, z, w) of a rotation by ``angle`` about the unit ``axis``."""
    s = np.sin(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0)])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _canonical(q: np.ndarray) -> Tuple[int, ...]:
    """Dedupe key of a rotation (q and −q are the same rotation): rounded, sign-normalised components."""
    q = q / np.linalg.norm(q)
    if q[3] < -1e-9 or (abs(q[3]) <= 1e-9 and (q[2] < -1e-9 or (abs(q[2]) <= 1e-9 and (q[1] < -1e-9 or (abs(q[1]) <= 1e-9 and q[0] < 0))))):
        q = -q
    return tuple(int(round(c * 1e5)) for c in q)


def equivalence_group(tags: Sequence[str], step_deg: float = CONTINUOUS_STEP_DEG) -> Optional[List[List[float]]]:
    """The rotation group generated by the symmetry ``tags`` (§3.13 rule 2), as unit quaternions ``[x, y, z, w]`` in
    the object's canonical frame — the identity first. ``None`` for ``full`` (sphere: ``orientation_metric: null``).

    Finite generators (``x2``, ``z4``, ``z6`` ...) are closed over composition; a continuous axis (``zinf``) is
    enumerated at ``step_deg`` (72 elements at 5°). An empty tag list gives the identity only.
    """
    tags = list(tags)
    if "full" in tags:
        return None
    generators: List[np.ndarray] = []
    for t in tags:
        axis = _AXES[t[0]]
        if t[1:] == "inf":
            generators.append(_quat_about(axis, np.deg2rad(step_deg)))
        else:
            n = int(t[1:])
            if n < 2:
                raise CustomsError(f"symmetry tag {t!r}: the fold must be >= 2")
            generators.append(_quat_about(axis, 2.0 * np.pi / n))
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    elements: Dict[Tuple[int, ...], np.ndarray] = {_canonical(identity): identity}
    frontier = [identity]
    while frontier:
        nxt = []
        for q in frontier:
            for g in generators:
                r = _quat_mul(g, q)
                r = r / np.linalg.norm(r)
                key = _canonical(r)
                if key not in elements:
                    if len(elements) >= MAX_GROUP_ELEMENTS:
                        raise CustomsError(f"symmetry tags {tags} generate more than {MAX_GROUP_ELEMENTS} rotations")
                    elements[key] = r
                    nxt.append(r)
        frontier = nxt
    out = [identity.tolist()] + [q.tolist() for k, q in elements.items() if k != _canonical(identity)]
    return [[float(round(c, 8)) for c in q] for q in out]


def half_turn_tag(axis: str) -> str:
    """The tag of the half turn about a frame axis given as ``+x`` / ``-y`` / ``z`` (sign irrelevant): ``x2``."""
    letter = axis.strip().lstrip("+-").lower()
    if letter not in _AXES:
        raise CustomsError(f"unknown frame axis {axis!r}")
    return f"{letter}2"


__all__ = [
    "BEHAVIOR_DATASET",
    "CONSTRAINT_TYPES",
    "CONTINUOUS_STEP_DEG",
    "Constraint",
    "Customs",
    "CustomsEntry",
    "CustomsError",
    "DEFAULT_CUSTOMS_PATH",
    "ENTRY_KEYS",
    "FALLBACK_KINDS",
    "HOLD_FIELDS",
    "ROLE_KEYS",
    "RoleEntry",
    "SCENE_CANDIDATES",
    "SCENE_KEYS",
    "SCOPE_KEYS",
    "SceneEntry",
    "blacklist_reason",
    "constraint_source",
    "equivalence_group",
    "half_turn_tag",
    "model_key",
    "parse_symmetry_tags",
    "reason_code",
]
