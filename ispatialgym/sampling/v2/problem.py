"""Grounded problem = one customs entry (spec §2.3) + its grounded BDDL + the scene bindings; the §9 model draw.

Pure module (no simulator): ``ground_isg`` (D1) and the CLIs import it, the tests exercise the draw rules without a
scene. The inputs are the D1 artefacts only (README contract 1):

* the customs entry ``task_custom_lists.json[alias]`` — ``iddl``, ``definition_id``, ``roles.<role>.{category,
  sampleable, require, representative, scale, joint, dataset}``, ``scenes.<scene>.{rooms, bindings, whitelist,
  blacklist}``, ``generation`` (``templates_max``, ``instances_per_template``, ``tolerances``, ``groups.<role>.{size,
  draw}``, ``path_buckets``, task numbers), ``constraints`` (typed, scoped records);
* the grounded BDDL ``<bddl_root>/<activity>/problem<definition_id>.bddl`` (``activity/problem.py`` parses it);
* the iddl file (``sampling/iddl.py`` when D1 lands; until then the file is only located).

:func:`iter_template_choices` implements the §9 ``sample_isg`` model draw: iterate over all whitelisted models before
reusing one (exact duplicates count as one model, texture variants as distinct models, §3.12 rule 4), unique models
across roles within a template, look-alike target–partner pairs rejected (§3.12 rule 2), group members per
``generation.groups.<role>`` (``size`` drawn per template; ``draw = unique`` pairwise distinct with at most one model
per look-alike group, or ``same_model``), distractor pools minus the target's look-alike group (§3.12 rule 1).

The v1-compatible facade (``.generation``, ``.activity``, ``.definition_id``, ``.name``, ``gen()``, ``spawned_roles()``,
``fixed_roles()``, ``scenes[scene]`` with ``.rooms`` / ``.fixed``) keeps the ``TaskSampler`` machinery of
``sampling/tasks/base.py`` working on a :class:`GroundedProblem`; :meth:`GroundedProblem.from_v1_spec` builds one from a
v1 spec entry (the smoke path until D1 lands).
"""

from __future__ import annotations

import copy
import csv
import datetime as _dt
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from ispatialgym import ASSETS_ROOT
from ispatialgym.sampling.spec import ModelChoice, RoleSpec, SceneSpec

#: the D1 customs file (``customs.py`` writes ``isg_customs.json``); the v1 list is the fallback for the facade path
_CUSTOMS_CANDIDATES = (os.path.join(ASSETS_ROOT, "task_specs", "isg_customs.json"), os.path.join(ASSETS_ROOT, "task_specs", "task_custom_lists.json"))
DEFAULT_CUSTOMS = next((p for p in _CUSTOMS_CANDIDATES if os.path.exists(p)), _CUSTOMS_CANDIDATES[0])
DEFAULT_BDDL_ROOT = os.path.join(ASSETS_ROOT, "bddl", "activity_definitions")
DEFAULT_IDDL_ROOT = os.path.join(ASSETS_ROOT, "iddl")
DEFAULT_ICL_ROOT = os.path.join(ASSETS_ROOT, "icl")

SCENE_CANDIDATES = ("house_single_floor", "house_double_floor_upper", "house_double_floor_lower", "Rs_int", "hotel_suite_large",
                    "restaurant_diner", "office_cubicles_right")

#: subtask id -> (activity directory, task family) for the aliases ``<subtask>-<categories...>``
SUBTASK_FAMILY = {
    "camera_relocalization-standard": ("camera_relocalization", "camera_relocalization"),
    "camera_relocalization-obstructed": ("camera_relocalization", "camera_relocalization"),
    "object_scaling": ("object_scaling", "object_scaling"),
    "mental_rotation": ("mental_rotation", "mental_rotation"),
    "disembedding": ("disembedding", "disembedding"),
    "configuration_matching-block_design": ("configuration_matching", "configuration_matching"),
    # the iddl / D1 / F6 ids of the four articulation subtasks (``<direction>_<scale>_scale``) ...
    "configuration_matching-articulation_open_large_scale": ("configuration_matching", "configuration_matching"),
    "configuration_matching-articulation_close_large_scale": ("configuration_matching", "configuration_matching"),
    "configuration_matching-articulation_open_small_scale": ("configuration_matching", "configuration_matching"),
    "configuration_matching-articulation_close_small_scale": ("configuration_matching", "configuration_matching"),
    # ... and the spelling the first task modules registered under (kept as aliases)
    "configuration_matching-articulation_large_scale-open": ("configuration_matching", "configuration_matching"),
    "configuration_matching-articulation_large_scale-close": ("configuration_matching", "configuration_matching"),
    "configuration_matching-articulation_small_scale-open": ("configuration_matching", "configuration_matching"),
    "configuration_matching-articulation_small_scale-close": ("configuration_matching", "configuration_matching"),
    "configuration_matching-attachment": ("configuration_matching", "configuration_matching"),
    "alignment-world_vertical": ("alignment", "alignment"),
    "alignment-world_horizontal": ("alignment", "alignment"),
    "alignment-axial": ("alignment", "alignment"),
    "inter_object_relation_satisfaction-ontop_small_scale": ("inter_object_relation_satisfaction", "inter_object_relation_satisfaction"),
    "inter_object_relation_satisfaction-under_small_scale": ("inter_object_relation_satisfaction", "inter_object_relation_satisfaction"),
    "inter_object_relation_satisfaction-inside": ("inter_object_relation_satisfaction", "inter_object_relation_satisfaction"),
    "inter_object_relation_satisfaction-nextto": ("inter_object_relation_satisfaction", "inter_object_relation_satisfaction"),
    "inter_object_relation_satisfaction-touching": ("inter_object_relation_satisfaction", "inter_object_relation_satisfaction"),
    "inter_object_relation_satisfaction-ontop_large_scale": ("inter_object_relation_satisfaction", "inter_object_relation_satisfaction"),
    "inter_object_relation_satisfaction-under_large_scale": ("inter_object_relation_satisfaction", "inter_object_relation_satisfaction"),
}


class GroundingError(ValueError):
    pass


def subtask_of_alias(alias: str, iddl: Optional[str] = None) -> str:
    """``inter_object_relation_satisfaction-ontop_small_scale-apple-plate-breakfast_table`` -> the subtask id (longest
    known prefix); the iddl path (``<family>/<subtask>.iddl``) decides when it is given."""
    if iddl:
        fam, name = os.path.splitext(iddl)[0].split("/")[-2:]
        cand = f"{fam}-{name}" if fam != name else name
        if cand in SUBTASK_FAMILY:
            return cand
    best = ""
    for st in SUBTASK_FAMILY:
        if (alias == st or alias.startswith(st + "-")) and len(st) > len(best):
            best = st
    if not best:
        raise GroundingError(f"alias {alias!r} matches no known subtask id {sorted(SUBTASK_FAMILY)}")
    return best


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------------------------------------------
# Look-alikes (spec §3.12)
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class LookalikeIndex:
    """Exact duplicates, texture variants and look-alike groups of the icl (§3.12): ``duplicate_of`` maps a model to
    the canonical member of its exact-duplicate group, ``geometry_group`` to its same-geometry group id, ``groups`` are
    the reviewed look-alike groups (``constraints_conflicting.csv``). Until A6 lands the pair rule treats exact
    duplicates and texture variants as look-alikes (§3.12 status note)."""

    duplicate_of: Dict[Tuple[str, str], Tuple[str, str]] = field(default_factory=dict)
    geometry_group: Dict[Tuple[str, str], str] = field(default_factory=dict)
    groups: List[Set[Tuple[str, str]]] = field(default_factory=list)

    @staticmethod
    def load(icl_root: Optional[str] = None) -> "LookalikeIndex":
        """Read ``roles/constraints_conflicting.csv`` (look-alike groups), ``roles/candidates.csv`` (``duplicate_of`` /
        ``geometry_group`` columns when present) and ``roles/representatives.csv`` (``exact_group`` / ``geometry_group``).
        Missing files are tolerated (empty index)."""
        root = icl_root or DEFAULT_ICL_ROOT
        idx = LookalikeIndex()
        p = os.path.join(root, "roles", "constraints_conflicting.csv")
        if os.path.exists(p):
            by_group: Dict[str, Set[Tuple[str, str]]] = {}
            with open(p, newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("relation", "lookalike") != "lookalike":
                        continue
                    by_group.setdefault(row["group_id"], set()).add((row["category"], row["model"]))
            idx.groups = [g for g in by_group.values() if len(g) >= 2]
        for name in ("candidates.csv", "representatives.csv"):
            p = os.path.join(root, "roles", name)
            if not os.path.exists(p):
                continue
            with open(p, newline="") as f:
                reader = csv.DictReader(f)
                cols = set(reader.fieldnames or [])
                for row in reader:
                    cat, model = row.get("category", ""), row.get("model", "")
                    if not model:
                        continue
                    if "duplicate_of" in cols and row.get("duplicate_of"):
                        idx.duplicate_of[(cat, model)] = (cat, row["duplicate_of"])
                    if "exact_group" in cols and row.get("exact_group"):
                        # ``apple__c2/exact:obixxh+zlxfxt``: the first member is the canonical one
                        members = row["exact_group"].split(":")[-1].split("+")
                        for m in members:
                            idx.duplicate_of.setdefault((cat, m), (cat, members[0]))
                    if "geometry_group" in cols and row.get("geometry_group"):
                        idx.geometry_group[(cat, model)] = row["geometry_group"]
        return idx

    def canonical(self, category: str, model: str) -> Tuple[str, str]:
        """The exact-duplicate group representative (the model itself when it has none)."""
        return self.duplicate_of.get((category, model), (category, model))

    def same_geometry(self, a: Tuple[str, str], b: Tuple[str, str]) -> bool:
        if a == b or self.canonical(*a) == self.canonical(*b):
            return True
        ga, gb = self.geometry_group.get(a), self.geometry_group.get(b)
        return ga is not None and ga == gb

    def lookalike_group_of(self, category: str, model: str) -> Set[Tuple[str, str]]:
        out: Set[Tuple[str, str]] = set()
        for g in self.groups:
            if (category, model) in g:
                out |= g
        return out

    def lookalikes(self, a: Tuple[str, str], b: Tuple[str, str]) -> bool:
        """§3.12 rule 2 / 1: reviewed look-alike group, or (stand-in until A6) exact duplicate / texture variant."""
        if a == b:
            return True
        if b in self.lookalike_group_of(*a):
            return True
        return self.same_geometry(a, b)


# ----------------------------------------------------------------------------------------------------------------
# Grounded problem
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class RoleEntry:
    """``roles.<role>`` of the customs entry (§2.3)."""

    name: str
    category: Optional[str]  # one category, or the first of a list (``categories`` keeps them all)
    sampleable: bool
    require: Dict[str, Any] = field(default_factory=dict)
    representative: Optional[str] = None
    scale: Dict[str, Any] = field(default_factory=dict)  # "<category>/<model>" -> scalar or [sx, sy, sz]
    joint: Any = None
    handle_key: Any = None
    dataset: str = "behavior-1k-assets"
    kind: str = "sampleable"  # sampleable | group | scene
    categories: List[str] = field(default_factory=list)
    grasp_keys: Dict[str, str] = field(default_factory=dict)  # "<category>/<model>" -> grasp library key (D1 extension)
    orientation_equivalences: Dict[str, Any] = field(default_factory=dict)  # "<category>/<model>" -> symmetry tags (D1 extension)

    def grasp_pipeline(self) -> Optional[str]:
        g = self.require.get("grasps")
        return g if isinstance(g, str) else None

    def scale_of(self, category: str, model: str):
        return self.scale.get(f"{category}/{model}", self.scale.get(model))

    def grasp_key_of(self, category: str, model: str) -> Optional[str]:
        return self.grasp_keys.get(f"{category}/{model}", self.grasp_keys.get(model))

    def joint_of(self, key: str) -> Optional[str]:
        """The chosen joint of an articulation target (``"<category>/<model>"`` or ``"<scene>/<name>"``)."""
        if isinstance(self.joint, dict):
            return self.joint.get(key)
        return self.joint

    def equivalence_tags(self, category: str, model: str) -> List[str]:
        tags = self.orientation_equivalences.get(f"{category}/{model}", self.orientation_equivalences.get(model, []))
        return list(tags) if isinstance(tags, (list, tuple)) else ([tags] if tags else [])


class GroundedProblem:
    """One customs entry, its grounded BDDL and the scene bindings (README module map: ``v2/problem.py``)."""

    def __init__(self, alias: str, entry: Dict[str, Any], bddl=None, iddl_path: Optional[str] = None, customs_path: Optional[str] = None,
                 bddl_root: Optional[str] = None, lookalikes: Optional[LookalikeIndex] = None, subtask: Optional[str] = None,
                 bddl_text: Optional[str] = None):
        self.alias = alias
        self.entry = entry
        self.customs_path = customs_path
        self.bddl_root = bddl_root or DEFAULT_BDDL_ROOT
        self.iddl_path = iddl_path
        self.iddl = None  # ``sampling.iddl`` object once D1 lands (see :meth:`load`)
        self.bddl = bddl  # ISGProblem of the grounded BDDL (None for the v1 facade until it is rendered)
        self.bddl_text = bddl_text
        self.subtask = subtask or subtask_of_alias(alias, entry.get("iddl"))
        self.activity, self.family = SUBTASK_FAMILY[self.subtask]
        self.definition_id = int(entry.get("definition_id", 0))
        self.lookalikes = lookalikes or LookalikeIndex()
        self.roles: Dict[str, RoleEntry] = {}
        for name, r in (entry.get("roles") or {}).items():
            groups = (entry.get("generation") or {}).get("groups") or {}
            kind = "group" if name in groups else ("sampleable" if r.get("sampleable", False) else "scene")
            cat = r.get("category")
            cats = [cat] if isinstance(cat, str) else list(cat or [])
            self.roles[name] = RoleEntry(name, cats[0] if cats else None, bool(r.get("sampleable", False)), dict(r.get("require") or {}),
                                         r.get("representative"), dict(r.get("scale") or {}), r.get("joint"), r.get("handle_key"),
                                         r.get("dataset") or "behavior-1k-assets", kind, cats, dict(r.get("grasp_keys") or {}),
                                         dict(r.get("orientation_equivalences") or {}))
        self.entry.setdefault("constraints", [])
        self.entry.setdefault("scenes", {})
        self.entry.setdefault("generation", {})

    # -- loading ------------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, alias: str, customs_path: Optional[str] = None, bddl_root: Optional[str] = None, icl_root: Optional[str] = None,
             iddl_root: Optional[str] = None) -> "GroundedProblem":
        """Load ``customs_path[alias]`` (default ``assets/task_specs/task_custom_lists.json``), the grounded BDDL
        ``<bddl_root>/<activity>/problem<definition_id>.bddl`` and, when ``sampling.iddl`` exists (D1), the parsed iddl."""
        customs_path = customs_path or DEFAULT_CUSTOMS
        with open(customs_path) as f:
            data = json.load(f)
        if alias not in data:
            raise GroundingError(f"alias {alias!r} not in {customs_path}; have {sorted(k for k in data if not k.startswith('_'))[:20]}...")
        entry = data[alias]
        gp = cls(alias, entry, customs_path=customs_path, bddl_root=bddl_root, lookalikes=LookalikeIndex.load(icl_root))
        gp.iddl_path = os.path.join(iddl_root or DEFAULT_IDDL_ROOT, entry["iddl"]) if entry.get("iddl") else None
        gp.iddl = _load_iddl(gp.iddl_path)
        from ispatialgym.activity.problem import load_problem, problem_path

        path = problem_path(gp.activity, gp.definition_id, gp.bddl_root)
        if not os.path.isfile(path):
            raise GroundingError(f"grounded BDDL missing: {path} (alias {alias}, definition_id {gp.definition_id})")
        gp.bddl = load_problem(gp.activity, gp.definition_id, gp.bddl_root)
        with open(path) as f:
            gp.bddl_text = f.read()
        if gp.bddl.alias and gp.bddl.alias != alias and not alias.startswith(gp.bddl.alias):
            # the BDDL alias may be the short variant name (v1 style); the customs alias is authoritative
            pass
        return gp

    @classmethod
    def from_v1_spec(cls, problem_spec, subtask: str, alias: Optional[str] = None, scene: Optional[str] = None) -> "GroundedProblem":
        """v1 facade (the smoke path until D1 lands): build a customs-shaped entry from a v1 ``ProblemSpec``
        (``sampling.spec``): spawned roles -> sampleable roles with a one-model whitelist, fixed roles -> scene roles
        bound per scene (``table`` is renamed ``support``), ``generation`` copied. The BDDL is rendered per template by
        ``core`` through the v1 ``bddl_writer`` (``bddl`` stays None here)."""
        isg = (problem_spec.raw or {}).get("_isg") or {}
        alias = alias or isg.get("alias") or f"{subtask}-v1-{problem_spec.name}"
        roles: Dict[str, Any] = {}
        for r in problem_spec.roles.values():
            role_name = "support" if r.name == "table" else r.name
            if r.fixed:
                roles[role_name] = {"category": None, "sampleable": False}
            else:
                cat = next(iter(r.models))
                roles[role_name] = {"category": cat, "sampleable": True, "representative": r.models[cat][0]}
                if r.grasps.get("key"):
                    suffix = r.grasps["key"].rsplit("__", 1)[-1] if r.grasps["key"].count("__") >= 2 else "f"
                    roles[role_name]["require"] = {"grasps": {"t": "tabletop", "i": "insert", "fl": "insert", "a": "attachment", "h": "floating"}.get(suffix, "floating")}
                    roles[role_name]["grasp_keys"] = {f"{cat}/{m}": (r.grasps["key"] if len(r.models[cat]) == 1 else r.grasps.get("keys", {}).get(m))
                                                      for m in r.models[cat]}
                if r.options.get("scale") is not None:
                    roles[role_name]["scale"] = {f"{cat}/{m}": r.options["scale"] for m in r.models[cat]}
                if r.dataset != "behavior-1k-assets":
                    roles[role_name]["dataset"] = r.dataset
        scenes: Dict[str, Any] = {}
        for sc, ss in problem_spec.scenes.items():
            if scene is not None and sc != scene:
                continue
            bindings = {}
            for role, req in ss.fixed.items():
                bindings["support" if role == "table" else role] = req
            wl = {r.name: {cat: list(ms) for cat, ms in r.models.items()} for r in problem_spec.spawned_roles()}
            scenes[sc] = {"rooms": list(ss.rooms), "bindings": bindings, "whitelist": wl, "blacklist": {}}
        pipelines = {}
        for req in isg.get("require", []):
            if req.get("kind") == "grasps":
                pipelines[req["role"]] = req["pipeline"]
        for role, pipe in pipelines.items():
            roles.setdefault(role, {}).setdefault("require", {})["grasps"] = pipe
        entry = {"iddl": isg.get("iddl"), "definition_id": problem_spec.definition_id, "roles": roles, "scenes": scenes,
                 "generation": dict(problem_spec.generation), "constraints": [], "_v1": {"name": problem_spec.name, "alias": problem_spec.alias}}
        return cls(alias, entry, subtask=subtask)

    # -- v1-compatible facade -----------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.alias

    @property
    def generation(self) -> Dict[str, Any]:
        return self.entry["generation"]

    @property
    def constraints(self) -> List[Dict[str, Any]]:
        return self.entry["constraints"]

    @property
    def raw(self) -> Dict[str, Any]:
        return self.entry

    def gen(self, key: str, default=None):
        return self.generation.get(key, default)

    @property
    def scenes(self) -> Dict[str, SceneSpec]:
        """v1 ``SceneSpec`` view of ``scenes.<scene>`` (``fixed`` = the bindings, plus ``whitelist`` / ``blacklist`` in options)."""
        out = {}
        for sc, sd in self.entry["scenes"].items():
            out[sc] = SceneSpec(sc, list(sd.get("rooms") or []), dict(sd.get("bindings") or {}),
                                {"whitelist": sd.get("whitelist") or {}, "blacklist": sd.get("blacklist") or {}})
        return out

    def spawned_roles(self) -> List[RoleSpec]:
        out = []
        for r in self.roles.values():
            if r.kind == "scene":
                continue
            out.append(RoleSpec(r.name, fixed=False, count=1, models={r.category: []} if r.category else {}, dataset=r.dataset,
                                grasps={"pipeline": r.grasp_pipeline()} if r.grasp_pipeline() else {}, options={"kind": r.kind}))
        return out

    def fixed_roles(self) -> List[RoleSpec]:
        return [RoleSpec(r.name, fixed=True) for r in self.roles.values() if r.kind == "scene"]

    # -- role queries -------------------------------------------------------------------------------------------
    def sampleable_roles(self) -> List[str]:
        return [r.name for r in self.roles.values() if r.kind == "sampleable"]

    def group_roles(self) -> List[str]:
        return [r.name for r in self.roles.values() if r.kind == "group"]

    def scene_roles(self) -> List[str]:
        return [r.name for r in self.roles.values() if r.kind == "scene"]

    def group_spec(self, role: str) -> Dict[str, Any]:
        """``generation.groups.<role>`` (``size`` int or [min, max]; ``draw`` unique | same_model)."""
        g = dict((self.generation.get("groups") or {}).get(role) or {})
        g.setdefault("size", 1)
        g.setdefault("draw", "unique")
        return g

    def scene_names(self) -> List[str]:
        return list(self.entry["scenes"])

    def rooms(self, scene: str) -> List[str]:
        return list(self.entry["scenes"][scene].get("rooms") or [])

    def bindings(self, scene: str) -> Dict[str, Any]:
        return dict(self.entry["scenes"][scene].get("bindings") or {})

    def whitelist(self, scene: str, role: str) -> Dict[str, List[str]]:
        """``{category: [models]}`` admitted for ``role`` in ``scene`` (the blacklist already removed)."""
        sd = self.entry["scenes"].get(scene) or {}
        wl = copy.deepcopy((sd.get("whitelist") or {}).get(role) or {})
        bl = (sd.get("blacklist") or {}).get(role) or {}
        for cat, models in list(wl.items()):
            bad = set((bl.get(cat) or {}).keys()) if isinstance(bl.get(cat), dict) else set(bl.get(cat) or [])
            wl[cat] = [m for m in models if m not in bad]
        return wl

    def blacklist(self, scene: str, role: str) -> Dict[str, Dict[str, str]]:
        sd = self.entry["scenes"].get(scene) or {}
        return copy.deepcopy((sd.get("blacklist") or {}).get(role) or {})

    def add_blacklist(self, scene: str, role: str, category: str, model: str, reason: str) -> None:
        """§9: move ``category/model`` of ``role`` to the scene's blacklist with ``reason`` (whitelist pruned)."""
        sd = self.entry["scenes"].setdefault(scene, {"rooms": [], "bindings": {}, "whitelist": {}, "blacklist": {}})
        sd.setdefault("blacklist", {}).setdefault(role, {}).setdefault(category, {})[model] = reason
        wl = sd.setdefault("whitelist", {}).get(role) or {}
        if category in wl and model in wl[category]:
            wl[category] = [m for m in wl[category] if m != model]

    def add_constraint(self, record: Dict[str, Any], source_instance: Optional[str] = None, sampler_version: str = "2.0.0-dev") -> Dict[str, Any]:
        """§2.3 / §9 constraints write-back: append a typed, scoped record with ``source`` (instance, sampler, date)."""
        rec = dict(record)
        rec.setdefault("scope", {})
        rec.setdefault("hard", False)
        if "type" not in rec:
            raise GroundingError(f"constraint record without type: {rec}")
        rec["source"] = {"instance": source_instance, "sampler": sampler_version, "date": _dt.date.today().isoformat(), **(rec.get("source") or {})}
        self.entry["constraints"].append(rec)
        return rec

    def constraints_for(self, ctype: Optional[str] = None, **scope) -> List[Dict[str, Any]]:
        """Records of ``type`` whose scope agrees with every given scope key (priors before drawing)."""
        out = []
        for rec in self.entry["constraints"]:
            if ctype is not None and rec.get("type") != ctype:
                continue
            sc = rec.get("scope") or {}
            if all(k not in sc or sc[k] == v for k, v in scope.items()):
                out.append(rec)
        return out

    def hard_blacklisted(self, scene: str, role: str, category: str, model: str) -> bool:
        for rec in self.constraints_for("blacklist", scene=scene):
            sc = rec.get("scope") or {}
            if sc.get(role) == f"{category}/{model}" or sc.get("model") == f"{category}/{model}":
                return True
        return False

    def tolerances(self) -> Dict[str, float]:
        return dict(self.generation.get("tolerances") or {"position": 0.03, "orientation": 0.20})

    def orientation_equivalences(self, role: str, category: str, model: str) -> Optional[List[List[float]]]:
        """§3.13: the rotation group of the model's symmetry tags as quaternions (xyzw) for ``pose_targets[...]
        .orientation_equivalences``; ``None`` = no equivalence (asymmetric / not surveyed), ``[]`` with the tag ``full``
        means ``orientation_metric: null``. Uses ``customs.equivalence_group`` (D1) when it exists."""
        tags = self.roles[role].equivalence_tags(category, model) if role in self.roles else []
        if not tags or tags == ["none"]:
            return None
        try:
            from ispatialgym.sampling import customs as _customs  # type: ignore

            return _customs.equivalence_group(tags)
        except Exception:
            return None

    def path_buckets(self) -> Optional[List[Dict[str, Any]]]:
        return self.generation.get("path_buckets")

    # -- BDDL instance mapping ----------------------------------------------------------------------------------
    def instances_of_synset(self, synset: str) -> List[str]:
        if self.bddl is None:
            return []
        return list(self.bddl.objects.get(synset, []))

    def save_customs(self, path: Optional[str] = None) -> str:
        """Write the (possibly updated) entry back into the customs file it came from (§9: blacklists, constraints)."""
        path = path or self.customs_path
        if path is None:
            raise GroundingError("no customs path to save to")
        data = {}
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
        data[self.alias] = self.entry
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, path)
        return path

    def spec_hash(self) -> str:
        """Hash of the entry (whitelists, generation, bindings; constraints excluded) recorded in the TRO / template_meta."""
        e = {k: v for k, v in self.entry.items() if k not in ("constraints",)}
        return sha256_of_text(json.dumps(e, sort_keys=True, default=str))


def _load_iddl(path: Optional[str]):
    """Parse the iddl through ``sampling.iddl`` (D1) when it exists; None otherwise (the file is only located)."""
    if not path or not os.path.exists(path):
        return None
    try:
        from ispatialgym.sampling import iddl as _iddl  # type: ignore
    except Exception:
        return None
    for fn in ("parse_file", "load", "parse"):
        f = getattr(_iddl, fn, None)
        if f is not None:
            try:
                return f(path)
            except Exception:
                return None
    return None


# ----------------------------------------------------------------------------------------------------------------
# Template choices (§9 model draw)
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class TemplateChoice:
    """Models of one template: ``models`` per sampleable role, ``groups`` member lists per group role (``size`` drawn),
    the scene and its ``bindings`` / ``rooms``. ``choices`` is the v1 ``{role: ModelChoice}`` view (group members as
    ``<role>[i]``) for the ``TaskSampler`` machinery; ``template_id`` follows spec §1."""

    scene: str
    models: Dict[str, ModelChoice]
    groups: Dict[str, List[ModelChoice]] = field(default_factory=dict)
    bindings: Dict[str, Any] = field(default_factory=dict)
    rooms: List[str] = field(default_factory=list)

    @property
    def choices(self) -> Dict[str, ModelChoice]:
        out = dict(self.models)
        for role, members in self.groups.items():
            for i, m in enumerate(members):
                out[f"{role}[{i}]"] = m
        return out

    def all_models(self) -> List[Tuple[str, ModelChoice]]:
        out = [(r, c) for r, c in self.models.items()]
        for role, members in self.groups.items():
            out += [(role, m) for m in members]
        return out

    @property
    def template_id(self) -> str:
        parts = [f"{role}-{c.category}-{c.model}" for role, c in sorted(self.models.items())]
        for role in sorted(self.groups):
            members = self.groups[role]
            cats: Dict[str, List[str]] = {}
            for m in members:
                cats.setdefault(m.category, []).append(m.model)
            parts.append(f"{role}-" + "-".join(f"{c}_{len(ms)}" for c, ms in sorted(cats.items())))
        return "__".join([self.scene] + parts) if parts else self.scene

    def to_json(self) -> Dict[str, Any]:
        return {"scene": self.scene, "rooms": list(self.rooms), "bindings": self.bindings,
                "models": {r: c.__dict__ for r, c in self.models.items()},
                "groups": {r: [m.__dict__ for m in ms] for r, ms in self.groups.items()}, "template_id": self.template_id}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "TemplateChoice":
        return TemplateChoice(d["scene"], {r: ModelChoice(**c) for r, c in d["models"].items()},
                              {r: [ModelChoice(**m) for m in ms] for r, ms in (d.get("groups") or {}).items()}, dict(d.get("bindings") or {}),
                              list(d.get("rooms") or []))


class ModelCycler:
    """Iterate over all whitelisted models of one role before reusing one (§9; §3.12 rule 4: exact duplicates count
    as one model — one member of each exact-duplicate group is in the cycle —, texture variants as distinct models).
    Every full pass is a fresh random permutation."""

    def __init__(self, role: str, whitelist: Dict[str, List[str]], rng, lookalikes: LookalikeIndex, dataset: str = "behavior-1k-assets",
                 grasp_keys: Optional[Dict[str, str]] = None, scales: Optional[Dict[str, Any]] = None):
        self.role = role
        self.rng = rng
        self.dataset = dataset
        self.grasp_keys = grasp_keys or {}
        self.scales = scales or {}
        seen_canon: Set[Tuple[str, str]] = set()
        self.pool: List[Tuple[str, str]] = []
        for cat, models in whitelist.items():
            for m in models:
                canon = lookalikes.canonical(cat, m)
                if canon in seen_canon:
                    continue
                seen_canon.add(canon)
                self.pool.append((cat, m))
        self._order: List[int] = []
        self.n_passes = 0

    def __len__(self) -> int:
        return len(self.pool)

    def choice(self, cat: str, model: str) -> ModelChoice:
        key = self.grasp_keys.get(f"{cat}/{model}")
        scale = self.scales.get(f"{cat}/{model}", self.scales.get(model))
        if scale is not None and not isinstance(scale, list):
            scale = [float(scale)] * 3
        return ModelChoice(cat, model, self.dataset, key, scale)

    def next(self, exclude: Set[Tuple[str, str]] = frozenset()) -> Optional[ModelChoice]:
        """Next model of the cycle not in ``exclude`` (looks ahead within the pass; starts a new pass when exhausted)."""
        if not self.pool:
            return None
        for _ in range(2):
            if not self._order:
                self._order = [int(i) for i in self.rng.permutation(len(self.pool))]
                self.n_passes += 1
            for k, i in enumerate(self._order):
                if self.pool[i] not in exclude:
                    self._order.pop(k)
                    return self.choice(*self.pool[i])
            self._order = []
        return None

    def remove(self, cat: str, model: str) -> None:
        """Whitelist pruning after a failure (§9): the model leaves the cycle."""
        if (cat, model) in self.pool:
            i = self.pool.index((cat, model))
            self.pool.pop(i)
            self._order = [j if j < i else j - 1 for j in self._order if j != i]


def _draw_size(spec_size, rng) -> int:
    if isinstance(spec_size, (list, tuple)):
        lo, hi = int(spec_size[0]), int(spec_size[1])
        return int(rng.integers(lo, hi + 1))
    return int(spec_size)


def draw_group(problem: GroundedProblem, role: str, scene: str, rng, taken: Set[Tuple[str, str]], target: Optional[ModelChoice] = None,
               pools: Optional[Dict[str, List[str]]] = None) -> Optional[List[ModelChoice]]:
    """Members of group ``role`` per ``generation.groups.<role>``: ``size`` drawn, ``draw = same_model`` (one model, N
    copies) or ``unique`` (pairwise distinct, at most one model per look-alike group, none a look-alike of the target,
    none in ``taken``). ``members`` (D1: ``{category: count}`` in alias order, mixed-category distractor / obstacle
    aliases) fixes how many members each category contributes — the grounded BDDL declares exactly those instances.
    ``pools`` overrides the whitelist (disembedding distractor pools). None when a pool is short."""
    spec = problem.group_spec(role)
    wl = pools if pools is not None else problem.whitelist(scene, role)
    entry = problem.roles[role]
    grasp_keys = entry.grasp_keys
    la = problem.lookalikes

    def mk(c, m):
        scale = entry.scale_of(c, m)
        if scale is not None and not isinstance(scale, list):
            scale = [float(scale)] * 3
        return ModelChoice(c, m, entry.dataset, grasp_keys.get(f"{c}/{m}"), scale)

    def admissible(cats: Optional[Sequence[str]]) -> List[Tuple[str, str]]:
        out = [(c, m) for c, ms in wl.items() if cats is None or c in cats for m in ms]
        if target is not None:
            out = [cm for cm in out if not la.lookalikes(cm, (target.category, target.model))]
        return [cm for cm in out if cm not in taken]

    def draw(cands: List[Tuple[str, str]], n: int, chosen: List[Tuple[str, str]]) -> Optional[List[Tuple[str, str]]]:
        if not cands:
            return None
        if spec["draw"] == "same_model":
            c, m = cands[int(rng.integers(len(cands)))]
            return [(c, m)] * n
        out: List[Tuple[str, str]] = []
        for cm in (cands[int(i)] for i in rng.permutation(len(cands))):
            if len(out) >= n:
                break
            if cm in chosen or cm in out or any(la.lookalikes(cm, other) for other in chosen + out):
                continue
            out.append(cm)
        return out if len(out) == n else None

    members_spec = spec.get("members")
    chosen: List[Tuple[str, str]] = []
    if isinstance(members_spec, dict) and members_spec:
        for cat, n in members_spec.items():  # alias order = BDDL instance order
            part = draw(admissible([cat]), int(n), chosen)
            if part is None:
                return None
            chosen += part
    else:
        size = _draw_size(spec["size"], rng)
        part = draw(admissible(None), size, chosen)
        if part is None:
            return None
        chosen = part
    return [mk(c, m) for c, m in chosen]


def iter_template_choices(problem: GroundedProblem, scene: str, rng, constraints: Optional[Sequence[Dict[str, Any]]] = None,
                          exclude: Sequence[Dict[str, str]] = (), max_templates: Optional[int] = None,
                          distractor_pools=None) -> Iterator[TemplateChoice]:
    """§9 ``sample_isg`` model draw for ``scene``: yields up to ``N = min(#valid combinations, generation.templates_max)``
    templates (``max_templates`` caps further). Sampleable roles cycle through their whitelists (:class:`ModelCycler`),
    models are unique across roles within a template, look-alike target–partner pairs are skipped (§3.12 rule 2),
    groups are drawn with :func:`draw_group`. ``exclude`` lists ``{role: "category/model"}`` combinations already
    produced (the representative template of ``ground_isg``). ``constraints`` (hard ``blacklist`` records) remove
    models before drawing; soft records are priors read by the task modules, not here. ``distractor_pools(target) ->
    {category: [models]}`` supplies the disembedding pools (§3.12 rule 1)."""
    la = problem.lookalikes
    roles = problem.sampleable_roles()
    cyclers: Dict[str, ModelCycler] = {}
    n_combos = 1
    for role in roles:
        wl = problem.whitelist(scene, role)
        for rec in list(constraints or []) + problem.constraints_for("blacklist", scene=scene):
            if rec.get("type") != "blacklist" or not rec.get("hard", True):
                continue
            sc = rec.get("scope") or {}
            cm = sc.get(role) or sc.get("model")
            if cm and "/" in cm:
                c, m = cm.split("/", 1)
                if c in wl and m in wl[c]:
                    wl[c] = [x for x in wl[c] if x != m]
        cy = ModelCycler(role, wl, rng, la, problem.roles[role].dataset, problem.roles[role].grasp_keys, problem.roles[role].scale)
        cyclers[role] = cy
        n_combos *= max(len(cy), 0)
    if n_combos == 0:
        return
    excluded = {tuple(sorted((r, cm) for r, cm in e.items())) for e in exclude}
    n_max = int(problem.gen("templates_max", n_combos) or n_combos)
    n_target = min(n_combos, n_max) - len(excluded)
    if max_templates is not None:
        n_target = min(n_target, int(max_templates))
    produced: Set[tuple] = set()
    tries = 0
    while len(produced) < n_target and tries < 20 * max(1, n_target):
        tries += 1
        models: Dict[str, ModelChoice] = {}
        taken: Set[Tuple[str, str]] = set()
        ok = True
        for role in roles:
            cy = cyclers[role]
            pick = None
            for _ in range(len(cy) + 1):
                cand = cy.next(exclude=taken)
                if cand is None:
                    break
                # §3.12 rule 2: the pair must not be look-alikes (target–partner, or any two sampleable roles)
                if any(la.lookalikes((cand.category, cand.model), (o.category, o.model)) for o in models.values()):
                    continue
                pick = cand
                break
            if pick is None:
                ok = False
                break
            models[role] = pick
            taken.add(la.canonical(pick.category, pick.model))
        if not ok:
            break
        key = tuple(sorted((r, f"{c.category}/{c.model}") for r, c in models.items()))
        if key in produced or key in excluded:
            continue
        groups: Dict[str, List[ModelChoice]] = {}
        for role in problem.group_roles():
            target = models.get("target")
            pools = distractor_pools(target) if (distractor_pools is not None and role == "distractors" and target is not None) else None
            members = draw_group(problem, role, scene, rng, taken, target=target, pools=pools)
            if members is None:
                ok = False
                break
            groups[role] = members
            taken |= {la.canonical(m.category, m.model) for m in members}
        if not ok:
            continue
        produced.add(key)
        yield TemplateChoice(scene, models, groups, problem.bindings(scene), problem.rooms(scene))


# ----------------------------------------------------------------------------------------------------------------
# Run report helpers (spec §3.5 "Report"; layout ``problems.<problem>.<template>.{n_written, time_s}`` kept from v1)
# ----------------------------------------------------------------------------------------------------------------


class RejectionLog:
    """Rejections per reason with the first ``keep`` examples (spec §3.5)."""

    def __init__(self, keep: int = 3):
        self.counts: Dict[str, int] = {}
        self.examples: Dict[str, List[Dict[str, Any]]] = {}
        self.keep = keep

    def add(self, reason: str, message: str = "", **where) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + 1
        ex = self.examples.setdefault(reason, [])
        if len(ex) < self.keep:
            ex.append({"message": str(message)[:400], **where})

    def to_json(self) -> Dict[str, Any]:
        return {"counts": dict(self.counts), "examples": {k: list(v) for k, v in self.examples.items()}}

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def new_run_summary(sampler_version: str, customs_path: Optional[str], out_root: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Top-level ``summary.json`` of a ``sample_isg`` / ``multiply_isg`` run."""
    return {"sampler_version": sampler_version, "schema": "isg-sampling-summary-v2", "customs": customs_path, "out_root": out_root,
            "started": _dt.datetime.now().isoformat(timespec="seconds"), "args": args or {}, "problems": {}, "scenes": {}, "errors": []}


def template_summary(n_written: int, time_s: float, rejections: RejectionLog, seeds: Sequence[int] = (), extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """``problems.<alias>.<template_id>`` entry: the v1 keys ``n_written`` / ``time_s`` (read by ``bench_report.py``) plus
    the per-reason ``rejections`` (attempts rejected, with examples), ``counters`` (bookkeeping such as ``scene_reset``
    — not rejections; ``extra["counters"]``), and the seeds."""
    counts = {k: v for k, v in rejections.counts.items() if k != "scene_reset"}
    counters = dict((extra or {}).get("counters") or {})
    if "scene_reset" in rejections.counts:
        counters["scene_reset"] = counters.get("scene_reset", 0) + int(rejections.counts["scene_reset"])
    d = {"n_written": int(n_written), "time_s": round(float(time_s), 2), "rejections": counts,
         "rejection_examples": {k: list(v) for k, v in rejections.examples.items() if k != "scene_reset"}, "counters": counters,
         "seeds": [int(s) for s in seeds]}
    if extra:
        d.update({k: v for k, v in extra.items() if k != "counters"})
    return d
