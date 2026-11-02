"""Scene crop (train-fast 2026-11-01, ``ISPATIALGYM_FAST_SCENE_CROP``): load only the part of the scene the sampler can use.

Today a sampler world loads the task rooms + their neighbours (``SimWorld.adjacent_rooms``, ``ISG_LOAD_ADJACENT``) and, through
OmniGibson's partial-room rule (``InteractiveTraversableScene._should_load_object``), **every** building structure and door of the
whole scene (house_single_floor: 443 objects loaded for a dining-room table task, 213 of them walls / ceilings / roofs / fences /
doors, 16 in the task room; 500-530 s of scene load). The sampler's collision world only ever holds the objects that pass
``SimWorld.obstacle_filter`` (the task rooms' floor extent +/- ``ISG_OBSTACLE_MARGIN_M``), and a tabletop program never leaves the
support's surroundings. Modes (:data:`MODES`):

* ``full``   - today's loading, unchanged (the scene file is the variant / stable json, ``load_room_instances`` = rooms + adjacent).
* ``rooms``  - ``load_room_instances`` = the task rooms only, and every object (structure included) whose json AABB (grown by
  ``reach_bands.SLIM_MARGIN_M``) misses the task rooms' floor extent grown by ``ISG_OBSTACLE_MARGIN_M`` is left out: exactly what the
  collision world could ever contain. Kept always: the task-room floors, the bound scene objects, objects without a known box, and
  the resting closure (what a kept object stands on).
* ``region`` - ``rooms`` further restricted to the objects whose json AABB (grown by :data:`REGION_BOX_MARGIN_M`) meets, in xy, the
  support's AABB (united with its declared neighbour supports) grown by ``ISPATIALGYM_FAST_SCENE_CROP_RADIUS_M`` (default 2.5 m:
  stand-off <= 0.55 m + base half-size ~0.4 m + arm reach ~1.0 m + 0.5 m margin). Kept always as for ``rooms`` plus the support and
  its neighbours. A problem without exactly one bound support scene object falls back to ``rooms``.

The crop is a pure function of the scene json (:func:`compute_crop`, no simulator), written as a cropped copy of the scene file
(:func:`write_cropped_scene`, ``<out>/scene_variants/<scene>__crop_<hash>.json``: the kept objects only, metadata unchanged) that the
sampler loads with ``load_room_instances`` = the task rooms. ``SimWorld.stable_scene`` (the ``template.json`` merge source, the
reset / scene-unchanged reference) stays the uncropped variant, so ``template.json`` is still the full scene; only what is loaded
(and so ``template-partial_rooms.json``) shrinks. ``template_meta.scene_crop`` (:meth:`CropPlan.meta`) records it.

The json boxes and the always-loaded rule are those of the reach bands' slim load (``reach_bands._json_aabb`` /
``reach_bands._slim_drop_list``), generalised here: the candidates follow OmniGibson's loader rule for the rooms actually loaded, the
xy-only region test, and an articulated object is grown by its largest extent only where needed (``rooms``: any object with joints,
as the calibrated slim load; ``region``: only when a joint of it is open in the scene file - a closed scene door / drawer stays closed:
``reset_scene_objects`` restores every articulated scene object before every attempt).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

log = logging.getLogger("ispatialgym.sampling")

MODES = ("full", "rooms", "region")

#: the default crop per subtask (``ISPATIALGYM_FAST_SCENE_CROP`` overrides it for every family; owner decision 2026-11-01 after
#: reviewing what each visibility check reads):
#: * ``full``: camera relocalization (>= 3 non-structural scene objects with >= 200 seg pixels at the goal view, >= 30 % of them also
#:   visible from the init view) and alignment-world_vertical / alignment_wall (>= 5 visible scene objects) count every scene object in
#:   view, objects seen through doorways included;
#: * ``region``: the tabletop families, and disembedding (its checks render only the target with / without the distractors and the
#:   destination: task objects + nearby occluders);
#: * ``rooms``: the families whose robot navigates through the task room (large-scale IORS, articulation large) and the held-object
#:   families (object scaling, mental rotation: self-occlusion by the robot and the held target out of view; no single support, so
#:   ``region`` would fall back to ``rooms`` anyway).
SCENE_CROP_DEFAULTS: Dict[str, str] = {
    # pixel families counting every object in view
    "camera_relocalization-standard": "full",
    "camera_relocalization-obstructed": "full",
    "alignment-world_vertical": "full",
    # held-object families
    "object_scaling": "rooms",
    "mental_rotation": "rooms",
    # navigation through the room
    "inter_object_relation_satisfaction-ontop_large_scale": "rooms",
    "inter_object_relation_satisfaction-under_large_scale": "rooms",
    "inter_object_relation_satisfaction-inside_large_scale": "rooms",
    "inter_object_relation_satisfaction-touching_large_scale": "rooms",
    "inter_object_relation_satisfaction-nextto_large_scale": "rooms",
    "configuration_matching-articulation_open_large_scale": "rooms",
    "configuration_matching-articulation_close_large_scale": "rooms",
    "configuration_matching-articulation_large_scale-open": "rooms",
    "configuration_matching-articulation_large_scale-close": "rooms",
    # tabletop
    "inter_object_relation_satisfaction-ontop_small_scale": "region",
    "inter_object_relation_satisfaction-under_small_scale": "region",
    "inter_object_relation_satisfaction-inside": "region",
    "inter_object_relation_satisfaction-nextto": "region",
    "inter_object_relation_satisfaction-touching": "region",
    "configuration_matching-articulation_open_small_scale": "region",
    "configuration_matching-articulation_close_small_scale": "region",
    "configuration_matching-articulation_small_scale-open": "region",
    "configuration_matching-articulation_small_scale-close": "region",
    "configuration_matching-attachment": "region",
    "configuration_matching-block_design": "region",
    "alignment-axial": "region",
    "disembedding": "region",
}
#: a subtask not in the table (and a missing subtask) loads as today
UNKNOWN_SUBTASK_MODE = "full"
#: ``region``: the support's AABB grown by this in xy (``ISPATIALGYM_FAST_SCENE_CROP_RADIUS_M``)
DEFAULT_RADIUS_M = 2.5
#: ``region``: every candidate's json box is grown by this before the region test (the json box vs loaded collision AABB mismatch
#: measured for the slim load, 2026-09-16: median 0, max 0.33 m)
REGION_BOX_MARGIN_M = 0.35
#: a joint farther than this from 0 (rad / m) in the scene file counts as open (``region``: the object is grown by its largest extent)
OPEN_JOINT_TOL = 0.01
#: resting closure: a candidate's top within this (+ twice the boxes' margin) of a kept object's bottom, footprints overlapping
REST_TOL_M = 0.3

BoxFn = Callable[[str, Dict[str, Any], Dict[str, Any], float, bool], Optional[Tuple[Any, Any]]]


# ----------------------------------------------------------------------------------------------------------------
# the switch
# ----------------------------------------------------------------------------------------------------------------


def default_mode(subtask: Optional[str]) -> str:
    """The family default of :data:`SCENE_CROP_DEFAULTS` (:data:`UNKNOWN_SUBTASK_MODE` for an unknown / missing subtask)."""
    return SCENE_CROP_DEFAULTS.get(str(subtask or ""), UNKNOWN_SUBTASK_MODE)


def requested_mode(subtask: Optional[str]) -> str:
    """``ISPATIALGYM_FAST_SCENE_CROP`` (``full`` / ``rooms`` / ``region``), default :func:`default_mode`; ``full`` with
    ``ISPATIALGYM_FAST=0`` (every train-fast path off: today's loading)."""
    from ispatialgym.fastflags import fast_master, fast_value

    if not fast_master():
        return "full"
    dflt = default_mode(subtask)
    value = str(fast_value("SCENE_CROP", dflt, cast=str)).strip().lower()
    if value in ("0", "off", "false", "no", "none"):
        return "full"
    if value not in MODES:
        log.warning("ISPATIALGYM_FAST_SCENE_CROP=%r is not one of %s: using %s", value, MODES, dflt)
        return dflt
    return value


def crop_radius() -> float:
    from ispatialgym.fastflags import fast_value

    return float(fast_value("SCENE_CROP_RADIUS_M", DEFAULT_RADIUS_M, cast=float))


def support_of(bindings: Dict[str, Any]) -> Tuple[Optional[str], List[str], str]:
    """``(support, centre names, why)`` from a scene block's ``bindings``: the single scene object bound as ``support`` and the
    neighbour supports its ``support_surfaces`` entry declares (``v2.neighbors``); ``(None, [], why)`` without exactly one."""
    req = (bindings or {}).get("support")
    names = [n for n in req if isinstance(n, str)] if isinstance(req, list) else []
    if not names:
        return None, [], "no support scene object bound (floor-bound / navigation task)"
    if len(names) > 1:
        return None, [], f"{len(names)} support candidates bound ({', '.join(names)})"
    sup = names[0]
    centre = [sup]
    rows = (bindings or {}).get("support_surfaces") or []
    for row in (rows.values() if isinstance(rows, dict) else rows):
        if isinstance(row, dict) and row.get("name") == sup:
            centre += [n for n in (row.get("neighbors") or []) if isinstance(n, str) and n not in centre]
    return sup, centre, ""


def bound_names(bindings: Dict[str, Any]) -> List[str]:
    """Every scene object name a scene block's ``bindings`` mentions (any role: support, partner, target, ...)."""
    out: List[str] = []
    for v in (bindings or {}).values():
        if isinstance(v, list):
            out += [n for n in v if isinstance(n, str) and n not in out]
    return out


# ----------------------------------------------------------------------------------------------------------------
# the loader rule and the json boxes
# ----------------------------------------------------------------------------------------------------------------


def always_loaded_categories() -> Set[str]:
    """OmniGibson's ``is_building_structure`` rule: loaded whatever the room list (``reach_bands._slim_drop_list``)."""
    from omnigibson.utils.constants import GROUND_CATEGORIES as OG_GROUND
    from omnigibson.utils.constants import STRUCTURE_CATEGORIES as OG_STRUCTURE

    return (set(OG_STRUCTURE) - set(OG_GROUND)) | {"door", "sliding_door"}


#: OmniGibson's GROUND_CATEGORIES: loaded only with their room (not by the structure rule); a neighbour room's floor the robot may
#: stand on is a crop candidate of its own (2026-11-01: the region crop of nextto_02 / touching_01 dropped the living-room / kitchen
#: floors next to the dining table; bases there stood 1.5 cm low on no floor and the arm met the wall at the first transit)
GROUND_CATEGORIES = ("floors", "lawn", "driveway")


def _in_rooms(args: Dict[str, Any]) -> List[str]:
    r = args.get("in_rooms") or []
    return [r] if isinstance(r, str) else list(r)


def is_robot(info: Dict[str, Any]) -> bool:
    cls = str(info.get("class_name", ""))
    return cls == "Robot" or cls.lower() in ("r1", "r1pro")


def loaded_by_rule(args: Dict[str, Any], load_rooms: Optional[Iterable[str]], always: Set[str]) -> bool:
    """Whether OmniGibson's partial-room loading brings the object in for ``load_room_instances = load_rooms`` (None: everything)."""
    if load_rooms is None:
        return True
    return bool(set(_in_rooms(args)) & set(load_rooms)) or args.get("category") in always


def _joint_open(state: Dict[str, Any]) -> bool:
    jp = state.get("joint_pos")
    if not jp:
        return False
    vals = jp.values() if isinstance(jp, dict) else jp
    try:
        return any(abs(float(v)) > OPEN_JOINT_TOL for v in vals)
    except (TypeError, ValueError):
        return True


def default_box_fn(name: str, args: Dict[str, Any], state: Dict[str, Any], margin: float, articulated: bool):
    """``reach_bands._json_aabb`` (the scaled canonical bounding box, fully rotated, grown)."""
    from ispatialgym.sampling.reach_bands import _json_aabb

    return _json_aabb(args, state, margin, articulated)


def _xyz(box) -> Tuple[List[float], List[float]]:
    lo, hi = box
    return [float(v) for v in lo[:3]], [float(v) for v in hi[:3]]


def _meets(lo, hi, blo, bhi, dims: int = 3) -> bool:
    return all(hi[k] >= blo[k] and lo[k] <= bhi[k] for k in range(dims))


def _resting_closure(keep: Set[str], cand: Set[str], boxes: Dict[str, Tuple[List[float], List[float]]], cats: Dict[str, str],
                     margin: float, structure: Set[str]) -> Set[str]:
    """``reach_bands._slim_drop_list``'s closure: keep every candidate a kept non-structure object may rest on (footprints overlapping,
    the candidate's top within ``REST_TOL_M`` + 2 ``margin`` of the kept object's bottom), iterated. Returns the added names."""
    added: Set[str] = set()
    changed = True
    while changed and cand:
        changed = False
        supports = [boxes[k] for k in keep | added if k in boxes and cats.get(k) not in structure]
        for name in sorted(cand):
            if name not in boxes:
                continue
            lo, hi = boxes[name]
            for klo, khi in supports:
                if _meets(lo, hi, klo, khi, dims=2) and abs(hi[2] - klo[2]) <= REST_TOL_M + 2 * margin:
                    added.add(name)
                    cand.discard(name)
                    changed = True
                    break
    return added


# ----------------------------------------------------------------------------------------------------------------
# the plan
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class CropPlan:
    """What a world loads. ``kept``: the scene-json objects of the cropped file (robots excluded); ``dropped``: the objects today's
    loading (``today_rooms`` + the structure rule) brings in that the crop leaves out."""

    mode: str
    requested_mode: str
    rooms: List[str]
    loaded_rooms: Optional[List[str]]
    today_rooms: Optional[List[str]]
    kept: List[str]
    dropped: List[str]
    n_json: int
    n_today: int
    radius_m: Optional[float] = None
    center_object: Optional[str] = None
    center_names: List[str] = field(default_factory=list)
    region_xy: Optional[Tuple[List[float], List[float]]] = None
    note: str = ""
    dropped_by_category: Dict[str, int] = field(default_factory=dict)
    path: Optional[str] = None  # the cropped scene file (write_cropped_scene)

    @property
    def cropped(self) -> bool:
        return self.mode != "full"

    def key(self) -> Tuple:
        """What decides whether two templates can share the world (``sample_isg.WorldPool``)."""
        if self.mode == "region":
            return ("region", round(float(self.radius_m or 0.0), 3), self.center_object, tuple(self.center_names))
        return (self.mode,)

    def meta(self) -> Dict[str, Any]:
        """``template_meta.scene_crop`` (the interface the collector / replay read; ``full``: ``{"mode": "full"}`` only)."""
        if not self.cropped:
            return {"mode": "full"}
        out = {"mode": self.mode, "radius_m": self.radius_m, "center_object": self.center_object, "n_loaded_json": len(self.kept),
               "n_dropped": len(self.dropped), "dropped": sorted(self.dropped), "loaded_rooms": list(self.loaded_rooms or []),
               "loaded_scene_file": "template-partial_rooms.json"}
        if self.requested_mode != self.mode:
            out["requested_mode"] = self.requested_mode
            out["note"] = self.note
        return out

    def describe(self) -> str:
        """The one INFO line of a world build."""
        if not self.cropped:
            fb = f" (requested {self.requested_mode}: {self.note})" if self.requested_mode != "full" else ""
            return (f"SCENE_CROP full: loading {len(self.kept)} of {self.n_json} scene-json objects (rooms {self.loaded_rooms} + every building "
                    f"structure / door){fb}")
        where = f"task rooms {self.rooms}"
        if self.mode == "region":
            where += f", radius {self.radius_m:.2f} m around {self.center_object}" + (
                f" + neighbours {self.center_names[1:]}" if len(self.center_names) > 1 else "")
        ranked = sorted(self.dropped_by_category.items(), key=lambda kv: (-kv[1], kv[0]))
        cats = ", ".join(f"{c} {n}" for c, n in ranked[:8]) + (f", {sum(n for _, n in ranked[8:])} more in {len(ranked) - 8} categories" if len(ranked) > 8 else "")
        fb = f"; requested {self.requested_mode}: {self.note}" if self.requested_mode != self.mode else ""
        return (f"SCENE_CROP {self.mode}: loading {len(self.kept)} of {self.n_json} scene-json objects ({where}): dropped {len(self.dropped)} of "
                f"the {self.n_today} today's load would bring ({cats or 'none'}){fb}")


def compute_crop(scene: Dict[str, Any], mode: str, rooms: Sequence[str], today_rooms: Optional[Sequence[str]], bindings: Optional[Dict[str, Any]] = None,
                 radius: float = DEFAULT_RADIUS_M, obstacle_margin: Optional[float] = None, slim_margin: Optional[float] = None,
                 region_margin: float = REGION_BOX_MARGIN_M, box_fn: Optional[BoxFn] = None, always: Optional[Set[str]] = None,
                 structure: Optional[Set[str]] = None, extra_keep: Iterable[str] = ()) -> CropPlan:
    """The :class:`CropPlan` of ``scene`` (a scene json: the variant / stable file the world would load) for the task ``rooms``.
    ``today_rooms``: the ``load_room_instances`` of today's world (rooms + adjacent with ``ISG_LOAD_ADJACENT``); ``bindings``: the
    scene block's bindings (support, partner, ...: kept, and the ``region`` centre). ``box_fn(name, args, state, margin, articulated)``
    -> ``(lo, hi)`` or None (default ``reach_bands._json_aabb``); ``always``: the categories loaded whatever the rooms (default
    OmniGibson's structure rule); ``structure``: the categories that never count as a support for the resting closure (default
    ``knowledge.STRUCTURE_CATEGORIES``)."""
    if mode not in MODES:
        raise ValueError(f"scene crop mode must be one of {MODES}, got {mode!r}")
    if box_fn is None:
        box_fn = default_box_fn
    if always is None:
        always = always_loaded_categories()
    if structure is None:
        from ispatialgym.sampling.knowledge import STRUCTURE_CATEGORIES

        structure = set(STRUCTURE_CATEGORIES)
    if slim_margin is None:
        from ispatialgym.sampling.reach_bands import SLIM_MARGIN_M

        slim_margin = SLIM_MARGIN_M
    m_obst = float(os.environ.get("ISG_OBSTACLE_MARGIN_M", "1.0")) if obstacle_margin is None else float(obstacle_margin)
    init = scene["objects_info"]["init_info"]
    reg = scene["state"]["registry"]["object_registry"]
    rooms = list(rooms or [])
    rooms_set = set(rooms)
    names = [n for n, info in init.items() if not is_robot(info)]
    cats = {n: init[n]["args"].get("category") for n in names}
    today = [n for n in names if loaded_by_rule(init[n]["args"], today_rooms, always)]
    bindings = bindings or {}
    keep_always = set(bound_names(bindings)) | set(extra_keep)

    def full(note: str = "") -> CropPlan:
        return CropPlan("full", mode, rooms, list(today_rooms) if today_rooms is not None else None, list(today_rooms) if today_rooms is not None else None,
                        sorted(today), [], len(names), len(today), note=note)

    if mode == "full" or not rooms:
        return full("" if rooms else "no task rooms")
    # the task rooms' floor extent (json boxes, as reach_bands._slim_drop_list) grown by the obstacle filter's margin
    flo, fhi = [], []
    task_floors = set()
    for n in names:
        a = init[n]["args"]
        if cats[n] == "floors" and set(_in_rooms(a)) & rooms_set:
            task_floors.add(n)
            box = box_fn(n, a, reg[n], slim_margin, False) if n in reg else None
            if box is not None:
                lo, hi = _xyz(box)
                flo.append(lo)
                fhi.append(hi)
    if not flo:
        return full("no floor of the task rooms with a known box")
    blo = [min(v[k] for v in flo) - m_obst for k in range(3)]
    bhi = [max(v[k] for v in fhi) + m_obst for k in range(3)]
    bhi[2] = 10.0  # everything above the floor (SimWorld.task_bounds)
    cand_all = [n for n in names if loaded_by_rule(init[n]["args"], rooms, always)]
    # the floors today's world loads with the neighbour rooms: candidates like any object (kept by the same box tests); the rooms of
    # the kept ones join load_room_instances (the cropped file holds only kept objects, so nothing else of those rooms loads)
    from ispatialgym.fastflags import fast_on

    cand_set = set(cand_all)
    neighbour_ground = {n for n in today if n not in cand_set and cats[n] in GROUND_CATEGORIES} if fast_on("CROP_NEIGHBOUR_FLOORS") else set()
    cand_all += sorted(neighbour_ground)
    # rooms: what can meet the collision world's bounds
    slim_boxes: Dict[str, Tuple[List[float], List[float]]] = {}
    keep: Set[str] = set()
    drop: Set[str] = set()
    for n in cand_all:
        state = reg.get(n)
        box = box_fn(n, init[n]["args"], state, slim_margin, bool(state.get("joint_pos"))) if state is not None else None
        if box is not None:
            slim_boxes[n] = _xyz(box)
        if n in task_floors or n in keep_always or box is None or _meets(*slim_boxes[n], blo, bhi):
            keep.add(n)
        else:
            drop.add(n)
    keep |= _resting_closure(keep, drop, slim_boxes, cats, slim_margin, structure)
    eff, radius_m, centre_obj, centre_names, region, note = "rooms", None, None, [], None, ""
    if mode == "region":
        sup, centre_names, why = support_of(bindings)
        cboxes = []
        for c in centre_names:
            if c in init and c in reg:
                box = box_fn(c, init[c]["args"], reg[c], 0.0, _joint_open(reg[c]))
                if box is not None:
                    cboxes.append(_xyz(box))
                elif c == sup:
                    cboxes = []
                    why = f"the support {sup} has no known box"
                    break
            elif c == sup:
                why = f"the support {sup} is not in the scene file"
                cboxes = []
                break
        if cboxes:
            r = float(radius)
            region = ([min(b[0][k] for b in cboxes) - r for k in range(2)], [max(b[1][k] for b in cboxes) + r for k in range(2)])
            reg_boxes: Dict[str, Tuple[List[float], List[float]]] = {}
            rkeep: Set[str] = set()
            rdrop: Set[str] = set()
            for n in cand_all:
                state = reg.get(n)
                box = box_fn(n, init[n]["args"], state, region_margin, _joint_open(state)) if state is not None else None
                if box is not None:
                    reg_boxes[n] = _xyz(box)
                if n in task_floors or n in keep_always or n in centre_names or box is None or (n in keep and _meets(*reg_boxes[n], region[0], region[1], dims=2)):
                    rkeep.add(n)
                else:
                    rdrop.add(n)
            rkeep |= _resting_closure(rkeep, rdrop, reg_boxes, cats, region_margin, structure)
            keep = rkeep
            eff, radius_m, centre_obj = "region", r, sup
        else:
            centre_names = []
            note = why or "no support box"
    kept = sorted(keep)
    dropped = sorted(set(today) - keep)
    by_cat = Counter(cats[n] or "?" for n in dropped)
    extra_rooms = sorted({r for n in keep & neighbour_ground for r in _in_rooms(init[n]["args"])} - rooms_set)
    return CropPlan(eff, mode, rooms, rooms + extra_rooms, list(today_rooms) if today_rooms is not None else None, kept, dropped, len(names), len(today),
                    radius_m=radius_m, center_object=centre_obj, center_names=list(centre_names), region_xy=region, note=note,
                    dropped_by_category=dict(by_cat))


def cropped_scene(scene: Dict[str, Any], plan: CropPlan) -> Dict[str, Any]:
    """A copy of ``scene`` with only ``plan.kept`` (and any robot entry) in ``objects_info.init_info`` and the object registry;
    metadata, systems and everything else unchanged (the scene's metadata becomes the task metadata of the loaded world and so of
    ``template-partial_rooms.json`` / ``template.json``: it must not differ from the uncropped file's)."""
    keep = set(plan.kept)
    out = {k: v for k, v in scene.items() if k not in ("objects_info", "state")}
    out = copy.deepcopy(out)
    init = scene["objects_info"]["init_info"]
    out["objects_info"] = {**{k: copy.deepcopy(v) for k, v in scene["objects_info"].items() if k != "init_info"},
                           "init_info": {n: copy.deepcopy(info) for n, info in init.items() if n in keep or is_robot(info)}}
    state = {k: copy.deepcopy(v) for k, v in scene["state"].items() if k != "registry"}
    registry = {k: copy.deepcopy(v) for k, v in scene["state"]["registry"].items() if k != "object_registry"}
    registry["object_registry"] = {n: copy.deepcopy(s) for n, s in scene["state"]["registry"]["object_registry"].items()
                                   if n in keep or n not in init or is_robot(init[n])}
    state["registry"] = registry
    out["state"] = state
    return out


def crop_tag(source_name: str, plan: CropPlan) -> str:
    key = "\n".join([os.path.basename(source_name), plan.mode] + plan.kept)
    return hashlib.sha256(key.encode()).hexdigest()[:10]


def write_cropped_scene(scene: Dict[str, Any], plan: CropPlan, out_dir: str, scene_model: str, source_name: str) -> str:
    """Write :func:`cropped_scene` to ``<out_dir>/<scene_model>__crop_<hash>.json`` (hash over the source file name, the mode and
    the kept names; written once, atomically) and return the path (also set on ``plan.path``)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{scene_model}__crop_{crop_tag(source_name, plan)}.json")
    if not os.path.exists(path):
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(cropped_scene(scene, plan), f)
        os.replace(tmp, path)
    plan.path = path
    return path


def request_key(mode: str, bindings: Dict[str, Any], radius: float = DEFAULT_RADIUS_M) -> Tuple:
    """The :meth:`CropPlan.key` a request will have (before the scene json is read): ``region`` without a single support is
    ``rooms``."""
    if mode == "region":
        sup, centre, _ = support_of(bindings)
        if sup is None:
            return ("rooms",)
        return ("region", round(float(radius), 3), sup, tuple(centre))
    return (mode,)


def covers(live: Optional[Tuple], wanted: Tuple) -> bool:
    """Whether a world cropped as ``live`` (:meth:`CropPlan.key`) holds everything a world cropped as ``wanted`` would (same scene,
    rooms and removed chairs assumed): ``full`` holds every crop, ``rooms`` every ``region``, a ``region`` only the same one."""
    if live is None:
        return False
    if live[0] == "full":
        return True
    if live[0] == "rooms":
        return wanted[0] in ("rooms", "region")
    return live == wanted
