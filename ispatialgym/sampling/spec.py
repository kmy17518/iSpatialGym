"""Task spec (iSpatialGym's task-custom list) loading and validation.

Schema (one entry per ``<activity>-<def_id>``)::

    {
      "alias": null | "<variant>",                 # must equal the problem's (:alias)
      "roles": {
        "<role>": {"models": {"<category>": ["<model>", ...]}, "dataset": "behavior-1k-assets",
                   "grasps": {"key": "<grasp library key>"}},          # spawned role
        "<role>": {"fixed": true}                                      # bound to a scene object
      },
      "scenes": {
        "<scene_model>": {"rooms": ["<room_instance>", ...],
                          "fixed": {"<role>": ["<scene object name>", ...] | {"category": [...]} | {"room": "..."}}}
      },
      "generation": {"instances_per_template": 5, "templates_per_scene": null, ...task parameters...},
      "distractors": {                                  # disembedding only: distractor pools per target category
        "families": {"<family>": ["<category>", ...]},  # informational: how other_categories were derived
        "lookalike_groups": [["<category>/<model>", ...] | {"members": [...], "reason": "..."}, ...],
        "pools": {
          "<category>": {"same_category": ["<model>", ...], "other_categories": ["<category>", ...],
                         "other_models": {"<category>": ["<model>", ...]}}   # other_models optional
        }
      }
    }

Spawned roles produce one *template* per combination of model choices (cartesian product, capped by
``generation.templates_per_scene``); ``count`` (block design) spawns several copies of the same model.

Distractor pools (disembedding): for a target ``(C, m)`` the *same-category pool* is
``pools[C].same_category`` minus ``m`` and minus every model that shares a look-alike group with ``m``; the
*other-category pool* is the union of ``pools[C'].same_category`` over ``pools[C].other_categories`` plus the
explicit ``pools[C].other_models`` (again minus look-alikes of ``m``, groups may cross categories). See
:class:`DistractorSpec`.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ispatialgym.sampling import knowledge

SCENE_CANDIDATES = (
    "house_single_floor",
    "house_double_floor_upper",
    "house_double_floor_lower",
    "Rs_int",
    "hotel_suite_large",
    "restaurant_diner",
    "office_cubicles_right",
)


class SpecError(ValueError):
    pass


@dataclass
class ModelChoice:
    category: str
    model: str
    dataset: str = "behavior-1k-assets"
    grasp_key: Optional[str] = None
    scale: Optional[List[float]] = None

    @property
    def synset(self) -> str:
        return knowledge.synset_of_category(self.category)

    @property
    def tag(self) -> str:
        return f"{self.category}-{self.model}"


@dataclass
class RoleSpec:
    name: str
    fixed: bool = False
    count: int = 1
    models: Dict[str, List[str]] = field(default_factory=dict)  # category -> models
    dataset: str = "behavior-1k-assets"
    grasps: Dict[str, Any] = field(default_factory=dict)  # {"key": ...} or {"keys": {model: key}}
    pick: str = "product"  # product | random (one random model per template)
    options: Dict[str, Any] = field(default_factory=dict)

    def choices(self) -> List[ModelChoice]:
        out = []
        for cat, models in self.models.items():
            for m in models:
                key = None
                if "key" in self.grasps and len(self.models) == 1 and len(models) == 1:
                    key = self.grasps["key"]
                elif "keys" in self.grasps:
                    key = self.grasps["keys"].get(m)
                elif "suffix" in self.grasps:
                    key = f"{cat}__{m}{self.grasps['suffix']}"
                out.append(ModelChoice(cat, m, self.dataset, key, self.options.get("scale")))
        return out


@dataclass
class SceneSpec:
    scene_model: str
    rooms: List[str] = field(default_factory=list)
    fixed: Dict[str, Any] = field(default_factory=dict)  # role -> binding request
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CategoryPool:
    """Distractor pool of one target category (``distractors.pools.<category>``)."""

    category: str
    same_category: List[str]  # models of ``category`` that may be drawn as same-category distractors
    other_categories: List[str]  # categories whose ``same_category`` lists form the other-category pool
    other_models: Dict[str, List[str]] = field(default_factory=dict)  # explicit other-category models (added to the pool)
    note: Optional[str] = None


@dataclass
class DistractorSpec:
    """``distractors`` section of a problem: per-category pools plus look-alike groups.

    A look-alike group lists ``(category, model)`` pairs that cannot be told apart from the robot's initial view
    (exact duplicates, label-only variants, ...). When the target belongs to a group, the other members are never
    drawn as distractors, from either pool.
    """

    pools: Dict[str, CategoryPool]
    lookalike_groups: List[List[Tuple[str, str]]] = field(default_factory=list)
    families: Dict[str, List[str]] = field(default_factory=dict)
    dataset: str = "behavior-1k-assets"

    def lookalikes_of(self, category: str, model: str) -> Set[Tuple[str, str]]:
        out: Set[Tuple[str, str]] = set()
        for grp in self.lookalike_groups:
            if (category, model) in grp:
                out.update(grp)
        out.discard((category, model))
        return out

    def _choice(self, category: str, model: str) -> ModelChoice:
        return ModelChoice(category, model, self.dataset)

    def same_category_choices(self, target: ModelChoice) -> List[ModelChoice]:
        pool = self.pools.get(target.category)
        if pool is None:
            return []
        skip = self.lookalikes_of(target.category, target.model)
        return [self._choice(target.category, m) for m in pool.same_category
                if m != target.model and (target.category, m) not in skip]

    def other_category_choices(self, target: ModelChoice) -> List[ModelChoice]:
        """Union of the ``same_category`` pools of ``other_categories`` and the explicit ``other_models``, without the
        target's category and look-alikes, each model once (in spec order)."""
        pool = self.pools.get(target.category)
        if pool is None:
            return []
        skip = self.lookalikes_of(target.category, target.model)
        out, seen = [], set()
        pairs = [(cat, m) for cat in pool.other_categories for m in self.pools[cat].same_category]
        pairs += [(cat, m) for cat, models in pool.other_models.items() for m in models]
        for cat, m in pairs:
            if cat == target.category or (cat, m) in skip or (cat, m) in seen:
                continue
            seen.add((cat, m))
            out.append(self._choice(cat, m))
        return out


@dataclass
class ProblemSpec:
    name: str  # <activity>-<def_id>
    alias: Optional[str]
    roles: Dict[str, RoleSpec]
    scenes: Dict[str, SceneSpec]
    generation: Dict[str, Any]
    raw: Dict[str, Any]
    distractors: Optional[DistractorSpec] = None
    _distractors_parsed: bool = field(default=False, repr=False, compare=False)

    def distractor_spec(self) -> Optional[DistractorSpec]:
        """The problem's ``distractors`` section (None when absent). Parsed lazily from ``raw`` for ProblemSpecs that
        other scripts rebuild field by field (e.g. with rewritten ``scenes``) without carrying ``distractors`` over."""
        if self.distractors is None and not self._distractors_parsed:
            self._distractors_parsed = True
            if self.raw.get("distractors") is not None:
                self.distractors = _parse_distractors(self.name, self.raw["distractors"], self.roles, check_ready=False)
        return self.distractors

    @property
    def activity(self) -> str:
        return self.name.rsplit("-", 1)[0]

    @property
    def definition_id(self) -> int:
        return int(self.name.rsplit("-", 1)[1])

    def spawned_roles(self) -> List[RoleSpec]:
        return [r for r in self.roles.values() if not r.fixed]

    def fixed_roles(self) -> List[RoleSpec]:
        return [r for r in self.roles.values() if r.fixed]

    def template_choices(self, rng=None) -> List[Dict[str, ModelChoice]]:
        """Model assignment per template: cartesian product over spawned roles (``pick="product"``)."""
        product_roles = [r for r in self.spawned_roles() if r.pick == "product" and r.models]
        random_roles = [r for r in self.spawned_roles() if r.pick == "random" and r.models]
        lists = [r.choices() for r in product_roles]
        combos = list(itertools.product(*lists)) if lists else [()]
        cap = self.generation.get("templates_per_scene")
        if cap is not None:
            combos = combos[: int(cap)]
        out = []
        for combo in combos:
            d = {r.name: c for r, c in zip(product_roles, combo)}
            for r in random_roles:
                ch = r.choices()
                d[r.name] = ch[rng.integers(len(ch))] if rng is not None else ch[0]
            out.append(d)
        return out

    def gen(self, key: str, default=None):
        return self.generation.get(key, default)


@dataclass
class TaskSpecFile:
    path: str
    problems: Dict[str, ProblemSpec]
    sha256: str

    def __getitem__(self, name: str) -> ProblemSpec:
        if name not in self.problems:
            raise SpecError(f"problem {name!r} not in {self.path}; have {sorted(self.problems)}")
        return self.problems[name]


def _parse_role(name: str, d: Dict[str, Any]) -> RoleSpec:
    d = dict(d)
    fixed = bool(d.pop("fixed", False))
    models = d.pop("models", {}) or {}
    role = RoleSpec(
        name=name,
        fixed=fixed,
        count=int(d.pop("count", 1)),
        models={k: list(v) for k, v in models.items()},
        dataset=d.pop("dataset", "behavior-1k-assets"),
        grasps=d.pop("grasps", {}) or {},
        pick=d.pop("pick", "product"),
        options=d,
    )
    if not fixed and not role.models:
        raise SpecError(f"spawned role {name!r} needs 'models'")
    return role


def _parse_distractors(problem: str, d: Dict[str, Any], roles: Dict[str, RoleSpec], check_ready: bool) -> DistractorSpec:
    """Validate the ``distractors`` section: known categories, ready models of the right category, other_categories
    that reference pools, look-alike members that are pooled (or targets), and a pool for every target category."""
    where = f"{problem}: distractors"
    pools_raw = d.get("pools")
    if not isinstance(pools_raw, dict) or not pools_raw:
        raise SpecError(f"{where}.pools must map categories to {{same_category, other_categories}}")
    dataset = d.get("dataset", "behavior-1k-assets")
    pools: Dict[str, CategoryPool] = {}
    for cat, body in pools_raw.items():
        if cat.startswith("_"):
            continue
        if not isinstance(body, dict):
            raise SpecError(f"{where}.pools.{cat} must be an object")
        try:
            knowledge.synset_of_category(cat)
        except KeyError as e:
            raise SpecError(f"{where}.pools: {e}")
        def check_models(cat_: str, models_: List[str], what: str):
            if len(set(models_)) != len(models_):
                raise SpecError(f"{where}.pools.{cat}.{what} lists a model twice")
            if not (check_ready and dataset == "behavior-1k-assets"):
                return
            known = set(knowledge.models_of_category(cat_, ready_only=False))
            for m in models_:
                if m not in known:
                    raise SpecError(f"{where}.pools.{cat}.{what}: {m} is not a {cat_} model in the BEHAVIOR inventory")
                if not knowledge.is_ready(m):
                    raise SpecError(f"{where}.pools.{cat}.{what}: {m} is not 'ready' (complaints: {knowledge.complaints(m)})")

        models = list(body.get("same_category") or [])
        check_models(cat, models, "same_category")
        other_models: Dict[str, List[str]] = {}
        for ocat, oms in (body.get("other_models") or {}).items():
            if ocat == cat:
                raise SpecError(f"{where}.pools.{cat}.other_models lists the category itself")
            try:
                knowledge.synset_of_category(ocat)
            except KeyError as e:
                raise SpecError(f"{where}.pools.{cat}.other_models: {e}")
            other_models[ocat] = list(oms or [])
            check_models(ocat, other_models[ocat], f"other_models.{ocat}")
        pools[cat] = CategoryPool(cat, models, list(body.get("other_categories") or []), other_models, body.get("note"))
    for cat, pool in pools.items():
        for other in pool.other_categories:
            if other == cat:
                raise SpecError(f"{where}.pools.{cat}.other_categories lists the category itself")
            if other not in pools:
                raise SpecError(f"{where}.pools.{cat}.other_categories: {other!r} has no pool")
    pooled = {(cat, m) for cat, pool in pools.items() for m in pool.same_category}
    pooled |= {(ocat, m) for pool in pools.values() for ocat, ms in pool.other_models.items() for m in ms}
    targets = {(cat, m) for r in roles.values() for cat, ms in r.models.items() for m in ms if r.name == "target"}
    groups: List[List[Tuple[str, str]]] = []
    for grp in d.get("lookalike_groups") or []:
        if isinstance(grp, dict):  # {"members": [...], "reason": "..."} form
            grp = grp.get("members") or []
        members: List[Tuple[str, str]] = []
        for entry in grp:
            if not isinstance(entry, str) or entry.count("/") != 1:
                raise SpecError(f"{where}.lookalike_groups: entries are '<category>/<model>' strings, got {entry!r}")
            cat, m = entry.split("/")
            if (cat, m) not in pooled and (cat, m) not in targets:
                raise SpecError(f"{where}.lookalike_groups: {entry} is neither pooled nor a target model")
            members.append((cat, m))
        if len(members) < 2:
            raise SpecError(f"{where}.lookalike_groups: a group needs >= 2 members, got {grp}")
        groups.append(members)
    for cat, _ in targets:
        if cat not in pools:
            raise SpecError(f"{where}.pools has no entry for target category {cat!r}")
    families = {k: list(v) for k, v in (d.get("families") or {}).items()}
    for fam, cats in families.items():
        for cat in cats:
            if cat not in pools:
                raise SpecError(f"{where}.families.{fam}: {cat!r} has no pool")
    return DistractorSpec(pools=pools, lookalike_groups=groups, families=families, dataset=dataset)


def load_task_spec(path: str, check_ready: bool = True) -> TaskSpecFile:
    with open(path, "rb") as f:
        raw_bytes = f.read()
    data = json.loads(raw_bytes)
    problems = {}
    for name, entry in data.items():
        if name.startswith("_"):
            continue  # comments / metadata
        if "-" not in name or not name.rsplit("-", 1)[1].isdigit():
            raise SpecError(f"spec key {name!r} must look like '<activity>-<def_id>'")
        roles = {r: _parse_role(r, d) for r, d in (entry.get("roles") or {}).items()}
        scenes = {}
        for sc, sd in (entry.get("scenes") or {}).items():
            if sc not in SCENE_CANDIDATES:
                raise SpecError(f"{name}: scene {sc!r} is not a candidate scene {SCENE_CANDIDATES}")
            sd = dict(sd or {})
            scenes[sc] = SceneSpec(sc, list(sd.pop("rooms", []) or []), dict(sd.pop("fixed", {}) or {}), sd)
        if check_ready:
            for r in roles.values():
                if r.dataset != "behavior-1k-assets":
                    continue
                for cat, models in r.models.items():
                    for m in models:
                        if not knowledge.is_ready(m):
                            raise SpecError(f"{name}: {cat}/{m} is not 'ready' (complaints: {knowledge.complaints(m)})")
        distractors = None
        if entry.get("distractors") is not None:
            distractors = _parse_distractors(name, entry["distractors"], roles, check_ready)
        problems[name] = ProblemSpec(
            name=name,
            alias=entry.get("alias"),
            roles=roles,
            scenes=scenes,
            generation=dict(entry.get("generation") or {}),
            raw=copy.deepcopy(entry),
            distractors=distractors,
        )
    return TaskSpecFile(path=os.path.abspath(path), problems=problems, sha256=hashlib.sha256(raw_bytes).hexdigest())


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
