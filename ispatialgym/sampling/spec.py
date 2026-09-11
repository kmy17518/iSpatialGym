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
      "generation": {"instances_per_template": 5, "templates_per_scene": null, ...task parameters...}
    }

Spawned roles produce one *template* per combination of model choices (cartesian product, capped by
``generation.templates_per_scene``); ``count`` (block design) spawns several copies of the same model.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
class ProblemSpec:
    name: str  # <activity>-<def_id>
    alias: Optional[str]
    roles: Dict[str, RoleSpec]
    scenes: Dict[str, SceneSpec]
    generation: Dict[str, Any]
    raw: Dict[str, Any]

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
        problems[name] = ProblemSpec(
            name=name,
            alias=entry.get("alias"),
            roles=roles,
            scenes=scenes,
            generation=dict(entry.get("generation") or {}),
            raw=copy.deepcopy(entry),
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
