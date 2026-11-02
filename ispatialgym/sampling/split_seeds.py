"""Per-variant, per-split seeds and the scene / room draw of the held-object tasks (owner 2026-10-11: distinct eval init / goal
and scenes, train policy). Pure module: no simulator, the room measurement reads the layout PNGs only.

Seeds
-----
``split_seed(split, variant, k)`` is the sampler seed (``sample_isg --seed-split``, ``multiply_isg --split``) of ``variant`` in
``split`` (``eval`` / ``train``): ``2 * h + parity`` with ``h`` the first 8 bytes of SHA-256 of ``"<split>:<variant>"``
(``k = 0``) or ``"<split>:<variant>:<k>"`` (``k > 0``: the k-th next seed, for an eval offender re-sampled) modulo 2**29, and
parity 0 for eval, 1 for train. Guarantees:

* a train seed never equals an eval seed (any variant, any k): eval seeds are even, train seeds odd;
* no two variants share a stream: within a split the seeds of ``(variant, k)`` are pairwise distinct for every variant of the
  shipped customs list and k < ``CHECKED_INDICES`` (:func:`check_unique`, run by the unit tests; a new variant collides with
  probability ~1e-5), and distinct seeds give disjoint attempt-seed streams - the samplers derive
  ``(seed * 7919 + t) * 1000003 + index * 1009 + attempt`` (t = template ordinal / multiply offset < 7919, index <= 990,
  attempt < 1009), an injective map;
* seeds stay below 2**30, so every derived attempt seed fits in int64.

Scene / room (held-object tasks: object scaling, mental rotation - their only scene role is the floor)
-----------------------------------------------------------------------------------------------------
Eligible rooms (:func:`eligible_rooms`, owner 2026-10-11 ~13:50): every room of the ``SCENE_CANDIDATES`` that is not outdoor
(``OUTDOOR_ROOM_TYPES``: gardens and the like, owner 13:31) and has at least ``MIN_NAV_FREE_M2`` (1 m^2) of robot-reachable
floor - free floor of the robot-eroded navigation map, the cells the held-object start base is drawn from
(``HeldObjectV2.sample_start_base``). No room-type whitelist. 45 rooms with 612 m^2 of reachable floor: house_single_floor 15,
house_double_floor_lower 5, house_double_floor_upper 5, Rs_int 1 (living_room_0), restaurant_diner 5, hotel_suite_large 2,
office_cubicles_right 12; out: the closets and Rs_int's bedroom / kitchen / entryway / bathroom (< 1 m^2), both gardens.

``draw_scene_room(split, variant, k)``; the eval and the train rooms both come from ``eligible_rooms``:

* eval: the variant's entry of ``EVAL_ROOMS``, whatever k (a re-sampled offender keeps its room; only its seed changes). The
  table is :func:`eval_assignment` of the 10 + 10 variants, frozen so that a new variant never moves an existing eval instance
  (the unit tests recompute it). Per task, in hash-seeded orders (SHA-256, as the seeds): 10 distinct rooms covering all 7
  scenes - one room per scene, one more in each of the 3 scenes with the most eligible rooms (house_single_floor,
  office_cubicles_right and a 5-room scene, ties in the task's order); mental rotation avoids object scaling's rooms, so the
  tasks share only Rs_int's single room. A variant that cannot be sampled in its room moves to its task's next unused room in
  the task's order (``EVAL_ROOM_MOVES``, reported). The 1 m separation check stays as a guard; it cannot fire within a task.
* train (``multiply_isg --split train``, one draw per train instance g): one room among all eligible rooms, no scene-first
  step, weighted by its reachable floor capped at ``TRAIN_CAP_X_MEDIAN`` (3) x the median eligible room (7.23 -> 21.69 m^2:
  six rooms capped, ~5.1 % of train each; the smallest ~3 per 1000), from SHA-256 of ``"scene:train:<variant>[:<g>]"``.

The numbers (``ROOM_FLOOR_M2``, every room of the candidate scenes) are measured without the simulator by :func:`measure_rooms`
from the scene's layout maps the way OmniGibson reads them: room instances from ``floor_insseg_0`` / ``floor_semseg_0``
(nearest-resized to 0.1 m, named as ``SegmentationMap`` names them), free floor from ``floor_trav_0`` (the map with objects,
resized to 0.1 m, < 255 -> 0), reachable floor = that map eroded with a square ``NAV_ERODE_PX`` kernel. 6 cells is the
samplers' own erosion: ceil((``robot_utils.BASE_ROOM_RADIUS`` 0.33 m, the chassis half-diagonal, + 0.2 m) / 0.1 m). Check:
the 10 s1_3tasks camera-relocalization paths (``paths[].waypoints`` of the oracle sidecars, planned and line-of-sight-checked
on the sampler's eroded map) stay on free floor of house_single_floor's map for kernels up to 6 cells; at 7 cells 2 of 10
cross blocked floor, at 10 cells 8 of 10; their minimum clearance to blocked floor is 0.32 m. (The earlier 10-cell kernel
took the R1Pro extent with the arms, 0.8165 x 1.2688 m.) The kernel only feeds this measurement, not the samplers. The table
is frozen here so that the draws never change with the data. ``python -m ispatialgym.sampling.split_seeds rooms`` prints the
measured table, ``... assign`` the eval assignment, ``... shares`` the train shares, ``... seed <split> <variant> [k]`` and
``... draw <split> <variant> [k]`` one seed / draw.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ispatialgym import REPO_ROOT

SPLITS = ("eval", "train")
_PARITY = {"eval": 0, "train": 1}
#: seeds are ``2 * h + parity`` with h < 2**29: below 2**30, every derived attempt seed fits in int64
HASH_BITS = 29
#: re-sample indices covered by the uniqueness check of the tests
CHECKED_INDICES = 16

#: subtasks whose eval / train (scene, room) comes from :func:`draw_scene_room` (their only scene role is the floor)
SCENE_DRAW_SUBTASKS = ("object_scaling", "mental_rotation")
#: the scenes of the draw (owner 2026-10-11)
SCENE_CANDIDATES = ("house_single_floor", "house_double_floor_lower", "house_double_floor_upper", "Rs_int", "restaurant_diner",
                    "hotel_suite_large", "office_cubicles_right")
#: outdoor room types the draws never pick (owner 2026-10-11 13:31: object scaling / mental rotation never in a garden or any
#: other outdoor room, eval and train); a type containing "garden", "yard" or "outdoor" counts too
OUTDOOR_ROOM_TYPES = ("garden", "balcony", "porch", "patio", "terrace", "deck", "yard", "backyard", "front_yard", "courtyard", "veranda",
                      "lawn", "driveway", "outdoor", "exterior")
#: minimum robot-reachable floor (free floor of the eroded navigation map) of an eligible room (owner 2026-10-11 ~13:50)
MIN_NAV_FREE_M2 = 1.0
MAP_RESOLUTION_M = 0.1
#: the samplers' erosion: ceil((robot_utils.BASE_ROOM_RADIUS 0.33 + 0.2) / 0.1) = ceil(5.3); the check is in the module docstring
NAV_ERODE_PX = 6
#: train draw: a room's weight is its reachable floor, capped at this multiple of the median eligible room
TRAIN_CAP_X_MEDIAN = 3

#: ``{scene: {room: (room area, free floor, reachable floor = free floor of the eroded navigation map)}}`` in m^2, every room of the
#: candidate scenes, measured by :func:`measure_rooms` (2026-10-11, behavior-1k-assets of Behavior-1K 6a6194600, 6-cell kernel)
ROOM_FLOOR_M2: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    "house_single_floor": {
        "bathroom_0": (11.44, 7.68, 2.65),
        "bathroom_1": (7.26, 3.81, 1.45),
        "bedroom_0": (20.24, 13.01, 7.34),
        "childs_room_0": (12.33, 6.89, 3.16),
        "childs_room_1": (13.86, 6.35, 2.98),
        "childs_room_2": (12.90, 7.76, 4.08),
        "closet_0": (3.11, 1.68, 0.55),
        "closet_1": (3.99, 1.94, 0.85),
        "closet_2": (2.73, 1.09, 0.08),
        "closet_3": (2.47, 1.15, 0.15),
        "closet_4": (2.73, 1.00, 0.08),
        "corridor_0": (78.13, 65.28, 45.26),
        "dining_room_0": (14.63, 9.21, 4.23),
        "entryway_0": (24.01, 18.74, 9.19),
        "garage_0": (34.81, 18.84, 12.23),
        "garden_0": (2513.32, 2255.17, 2003.58),
        "kitchen_0": (23.10, 13.44, 8.23),
        "living_room_0": (54.78, 39.00, 27.05),
        "living_room_1": (20.35, 13.86, 5.53),
        "sauna_0": (29.84, 21.08, 12.42),
        "utility_room_0": (16.92, 8.84, 5.27),
    },
    "house_double_floor_lower": {
        "bathroom_0": (6.08, 3.31, 1.10),
        "corridor_0": (27.75, 20.53, 13.25),
        "garage_0": (50.70, 36.91, 25.05),
        "garden_0": (948.36, 739.07, 531.69),
        "kitchen_0": (36.14, 22.27, 15.16),
        "living_room_0": (23.65, 12.70, 8.04),
    },
    "house_double_floor_upper": {
        "bathroom_0": (14.70, 6.86, 3.52),
        "bedroom_0": (17.48, 9.35, 5.36),
        "bedroom_1": (19.32, 10.83, 5.74),
        "bedroom_2": (18.90, 9.00, 4.13),
        "living_room_0": (40.69, 24.50, 14.18),
    },
    "Rs_int": {
        "bathroom_0": (3.61, 1.56, 0.16),
        "bedroom_0": (6.84, 1.91, 0.45),
        "entryway_0": (2.85, 1.79, 0.67),
        "kitchen_0": (4.75, 1.77, 0.55),
        "living_room_0": (22.40, 12.74, 5.35),
    },
    "restaurant_diner": {
        "bar_0": (14.43, 7.34, 4.29),
        "bathroom_0": (33.18, 21.18, 12.42),
        "corridor_0": (2.21, 2.06, 1.22),
        "dining_room_0": (68.35, 46.73, 30.24),
        "kitchen_0": (55.44, 34.76, 21.55),
    },
    "hotel_suite_large": {
        "bathroom_0": (22.80, 15.86, 9.66),
        "bedroom_0": (39.78, 20.41, 10.35),
    },
    "office_cubicles_right": {
        "bathroom_0": (35.19, 28.18, 18.29),
        "conference_hall_0": (60.59, 55.68, 48.13),
        "copy_room_0": (8.47, 5.74, 3.69),
        "corridor_0": (193.03, 177.04, 141.11),
        "empty_room_0": (9.66, 8.80, 5.85),
        "lobby_0": (29.16, 20.34, 12.21),
        "meeting_room_0": (33.58, 20.76, 11.05),
        "private_office_0": (15.17, 9.50, 2.68),
        "private_office_1": (21.84, 14.59, 6.30),
        "private_office_2": (21.45, 15.28, 7.17),
        "private_office_3": (21.86, 15.65, 7.23),
        "shared_office_0": (39.59, 17.52, 6.34),
    },
}

#: the eval (scene, room) of every held-object variant: :func:`eval_assignment` of these variants with ``EVAL_ROOM_MOVES``, frozen
#: (owner 2026-10-11 ~13:50) so that a new variant never moves an existing eval instance; the unit tests recompute it
EVAL_ROOMS: Dict[str, Tuple[str, str]] = {
    "object_scaling-apple": ("house_double_floor_lower", "living_room_0"),
    "object_scaling-beefsteak_tomato": ("restaurant_diner", "bathroom_0"),
    "object_scaling-beeswax_candle": ("office_cubicles_right", "empty_room_0"),
    "object_scaling-bell_pepper": ("hotel_suite_large", "bedroom_0"),
    "object_scaling-bottle_of_catsup": ("Rs_int", "living_room_0"),
    "object_scaling-bottle_of_medicine": ("house_single_floor", "kitchen_0"),
    "object_scaling-bottle_of_pop": ("office_cubicles_right", "private_office_0"),
    "object_scaling-can_of_soda": ("house_single_floor", "living_room_0"),
    "object_scaling-canned_food": ("house_double_floor_upper", "bedroom_1"),
    "object_scaling-pack_of_cigarettes": ("house_double_floor_upper", "bedroom_0"),
    "mental_rotation-carving_knife": ("house_single_floor", "bathroom_0"),
    "mental_rotation-classroom_mock_up": ("office_cubicles_right", "corridor_0"),
    "mental_rotation-crab": ("house_double_floor_lower", "garage_0"),
    "mental_rotation-hook": ("restaurant_diner", "kitchen_0"),
    "mental_rotation-mug": ("Rs_int", "living_room_0"),
    "mental_rotation-soup_ladle": ("house_single_floor", "dining_room_0"),
    "mental_rotation-statue": ("hotel_suite_large", "bathroom_0"),
    "mental_rotation-teaspoon": ("office_cubicles_right", "bathroom_0"),
    "mental_rotation-toy_figure": ("house_double_floor_lower", "kitchen_0"),
    "mental_rotation-walker": ("house_double_floor_upper", "living_room_0"),
}
#: variants that could not be sampled in their eval room, in move order: each went to its task's next unused room (reported)
EVAL_ROOM_MOVES: Tuple[str, ...] = ()


# ----------------------------------------------------------------------------------------------------------------
# seeds
# ----------------------------------------------------------------------------------------------------------------


def _key(prefix: str, split: str, variant: str, k: int) -> str:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if int(k) < 0:
        raise ValueError(f"seed index must be >= 0, got {k}")
    base = f"{prefix}{split}:{variant}"
    return base if int(k) == 0 else f"{base}:{int(k)}"


def _hash_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def split_seed(split: str, variant: str, k: int = 0) -> int:
    """The sampler seed of ``variant`` in ``split`` (``k``: the k-th next seed); see the module docstring for the guarantees."""
    h = _hash_int(_key("", split, variant, k)) % (1 << HASH_BITS)
    return 2 * h + _PARITY[split]


def check_unique(variants: Iterable[str], indices: Sequence[int] = range(CHECKED_INDICES)) -> Dict[int, Tuple[str, str, int]]:
    """Every ``split_seed(split, variant, k)`` of ``variants`` x ``SPLITS`` x ``indices``, keyed by seed; ``ValueError`` on a
    collision (two keys, one seed)."""
    seen: Dict[int, Tuple[str, str, int]] = {}
    for variant in sorted(set(variants)):
        for split in SPLITS:
            for k in indices:
                s = split_seed(split, variant, k)
                if s in seen:
                    raise ValueError(f"split seed collision: {(split, variant, k)} and {seen[s]} -> {s}")
                seen[s] = (split, variant, int(k))
    return seen


# ----------------------------------------------------------------------------------------------------------------
# scene / room draw
# ----------------------------------------------------------------------------------------------------------------


def room_type(room: str) -> str:
    """``living_room_0`` -> ``living_room`` (OmniGibson names a room instance ``<type>_<i>``)."""
    head, _, tail = room.rpartition("_")
    return head if head and tail.isdigit() else room


def is_outdoor(room: str) -> bool:
    """Whether a room instance is of an outdoor type (:data:`OUTDOOR_ROOM_TYPES`): never drawn."""
    t = room_type(room)
    return t in OUTDOOR_ROOM_TYPES or any(w in t for w in ("garden", "yard", "outdoor"))


def scene_draw_applies(subtask: str) -> bool:
    """Whether the subtask's (scene, room) comes from :func:`draw_scene_room` (the held-object tasks)."""
    return any(subtask == s or subtask.startswith(s + "-") for s in SCENE_DRAW_SUBTASKS)


def scene_draw_task(variant: str) -> Optional[str]:
    """The held-object task (``SCENE_DRAW_SUBTASKS``) of a variant, else None."""
    return next((s for s in SCENE_DRAW_SUBTASKS if variant == s or variant.startswith(s + "-")), None)


def eligible_rooms(scene: Optional[str] = None, table: Optional[Dict[str, Dict[str, Tuple[float, float, float]]]] = None,
                   min_nav_free_m2: float = MIN_NAV_FREE_M2) -> List[Tuple[str, str]]:
    """``(scene, room)`` pairs of the draws, in ``SCENE_CANDIDATES`` order then room name: never an outdoor room (:func:`is_outdoor`),
    at least ``min_nav_free_m2`` of reachable floor (``table`` defaults to ``ROOM_FLOOR_M2``). The eval assignment and every train
    draw (``multiply_isg --split train``) go through this function."""
    table = ROOM_FLOOR_M2 if table is None else table
    out = []
    for sc in SCENE_CANDIDATES:
        if scene is not None and sc != scene:
            continue
        for room, (_, _, nav) in sorted((table.get(sc) or {}).items()):
            if not is_outdoor(room) and float(nav) >= float(min_nav_free_m2):
                out.append((sc, room))
    return out


def _hash_order(items: Iterable[str], salt: str) -> List[str]:
    """``items`` sorted by SHA-256 of ``"<salt>:<item>"`` (the seeds' hash): a fixed pseudo-random order."""
    return sorted(items, key=lambda x: (_hash_int(f"{salt}:{x}"), x))


def eval_assignment(variants: Iterable[str], table: Optional[Dict[str, Dict[str, Tuple[float, float, float]]]] = None,
                    moves: Sequence[str] = ()) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, List[Tuple[str, str]]]]:
    """``{variant: (scene, room)}`` of the held-object ``variants`` (the others are ignored) and ``{task: the task's room order}``.

    Per task, in ``SCENE_DRAW_SUBTASKS`` order, with its variants, its scenes and the eligible rooms each in a hash-seeded order
    keyed by the task: scene quotas of one room per scene, then one more per scene in order of decreasing eligible rooms (ties:
    the task's scene order) until every variant has a slot; the rooms, walked in the task's room order, fill the quotas, skipping
    an earlier task's rooms unless nothing else fills a quota (Rs_int); the i-th variant gets the i-th room taken. ``moves``: each
    variant in turn goes to the first room of its task's order that no variant holds or has left."""
    rooms = eligible_rooms(table=table)
    by_scene: Dict[str, List[str]] = {}
    for sc, r in rooms:
        by_scene.setdefault(sc, []).append(r)
    out: Dict[str, Tuple[str, str]] = {}
    orders: Dict[str, List[Tuple[str, str]]] = {}
    for task in SCENE_DRAW_SUBTASKS:
        als = _hash_order(sorted({a for a in variants if scene_draw_task(a) == task}), f"eval-room:{task}:alias")  # legacy spelling: the salt EVAL_ROOMS was frozen with
        if not als:
            continue
        if len(als) > len(rooms):
            raise ValueError(f"{task}: {len(als)} variants but {len(rooms)} eligible rooms")
        order = [tuple(x.split("/", 1)) for x in _hash_order([f"{sc}/{r}" for sc, r in rooms], f"eval-room:{task}:room")]
        orders[task] = order
        ranked = sorted(_hash_order(by_scene, f"eval-room:{task}:scene"), key=lambda sc: -len(by_scene[sc]))
        quota = dict.fromkeys(by_scene, 0)
        while sum(quota.values()) < len(als):
            for sc in ranked:
                if sum(quota.values()) < len(als) and quota[sc] < len(by_scene[sc]):
                    quota[sc] += 1
        earlier = set(out.values())
        chosen: List[Tuple[str, str]] = []
        for skip_earlier in (True, False):
            for sc, r in order:
                if (sc, r) in chosen or sum(c[0] == sc for c in chosen) >= quota[sc] or (skip_earlier and (sc, r) in earlier):
                    continue
                chosen.append((sc, r))
        out.update(zip(als, chosen))
    held, left = set(out.values()), set()
    for a in moves:
        if a not in out:
            raise KeyError(f"moved variant {a} has no eval room")
        left.add(out[a])
        free = [x for x in orders[scene_draw_task(a)] if x not in held and x not in left]
        if not free:
            raise ValueError(f"{a}: no unused room left in its task's order")
        out[a] = free[0]
        held.add(free[0])
    return out, orders


def eval_scene_room(variant: str) -> Tuple[str, str]:
    """The variant's frozen eval (scene, room) (``EVAL_ROOMS``); ``KeyError`` for a variant the table does not list."""
    if variant not in EVAL_ROOMS:
        raise KeyError(f"{variant} has no eval room in split_seeds.EVAL_ROOMS: give a new held-object variant its task's next unused room "
                       f"(split_seeds assign) and add it to the table")
    return EVAL_ROOMS[variant]


def train_weights(table: Optional[Dict[str, Dict[str, Tuple[float, float, float]]]] = None) -> Tuple[List[Tuple[str, str]], List[int], int]:
    """The train draw's rooms (``eligible_rooms`` order), their weights in 0.01 m^2 - reachable floor capped at
    ``TRAIN_CAP_X_MEDIAN`` x the median eligible room - and that cap."""
    table = ROOM_FLOOR_M2 if table is None else table
    rooms = eligible_rooms(table=table)
    if not rooms:
        raise ValueError("no eligible room")
    navs = [int(round(float(table[sc][r][2]) * 100)) for sc, r in rooms]
    cap = int(round(TRAIN_CAP_X_MEDIAN * statistics.median(navs)))
    return rooms, [min(n, cap) for n in navs], cap


def draw_scene_room(split: str, variant: str, k: int = 0) -> Tuple[str, str]:
    """The (scene, room) of ``variant`` in ``split``. eval: its frozen room (:func:`eval_scene_room`) whatever ``k``; train: the k-th
    draw, one eligible room weighted by its capped reachable floor (:func:`train_weights`)."""
    h = _hash_int(_key("scene:", split, variant, k))
    if split == "eval":
        return eval_scene_room(variant)
    rooms, weights, _ = train_weights()
    x = h % sum(weights)
    for room, w in zip(rooms, weights):
        if x < w:
            return room
        x -= w
    raise AssertionError("unreachable: x < sum(weights)")


# ----------------------------------------------------------------------------------------------------------------
# room measurement (layout maps; reproduces OmniGibson's SegmentationMap / TraversableMap at 0.1 m)
# ----------------------------------------------------------------------------------------------------------------


def _data_path(data_path: Optional[str] = None) -> str:
    return data_path or os.environ.get("OMNIGIBSON_DATA_PATH") or os.path.join(REPO_ROOT, "Behavior-1K", "datasets")


def measure_rooms(scene: str, data_path: Optional[str] = None, erode_px: int = NAV_ERODE_PX) -> Dict[str, Tuple[float, float, float]]:
    """``{room: (area, free floor, eroded free floor)}`` (m^2, rounded to 0.01) of every room instance of ``scene``."""
    import cv2
    import numpy as np

    root = os.path.join(_data_path(data_path), "behavior-1k-assets")
    layout = os.path.join(root, "scenes", scene, "layout")
    with open(os.path.join(root, "metadata", "room_categories.txt")) as f:
        cats = [line.rstrip() for line in f]
    ins = cv2.imread(os.path.join(layout, "floor_insseg_0.png"), cv2.IMREAD_GRAYSCALE)
    sem = cv2.imread(os.path.join(layout, "floor_semseg_0.png"), cv2.IMREAD_GRAYSCALE)
    trav = cv2.imread(os.path.join(layout, "floor_trav_0.png"), cv2.IMREAD_GRAYSCALE)
    if ins is None or sem is None or trav is None:
        raise FileNotFoundError(f"layout maps of {scene} missing under {layout}")
    n = int(ins.shape[0] * 0.01 / MAP_RESOLUTION_M)
    ins = cv2.resize(ins, (n, n), interpolation=cv2.INTER_NEAREST)
    sem = cv2.resize(sem, (n, n), interpolation=cv2.INTER_NEAREST)
    nt = int(trav.shape[0] * 0.01 / MAP_RESOLUTION_M)
    trav = cv2.resize(trav, (nt, nt))
    trav[trav < 255] = 0
    nav = cv2.erode(trav, np.ones((int(erode_px), int(erode_px))))
    # SegmentationMap: instance ids ascending, each instance's type from its first pixel (row-major), named <type>_<i> with
    # i counting the instances of that type in id order
    by_type: Dict[int, List[int]] = {}
    for i in sorted(set(np.unique(ins).tolist()) - {0}):
        xs, ys = np.where(ins == i)
        by_type.setdefault(int(sem[xs[0], ys[0]]), []).append(int(i))
    cell = MAP_RESOLUTION_M ** 2
    out: Dict[str, Tuple[float, float, float]] = {}
    for sem_id, ids in by_type.items():
        for j, i in enumerate(ids):
            mask = ins == i
            out[f"{cats[sem_id - 1]}_{j}"] = (round(float(mask.sum()) * cell, 2), round(float((mask & (trav == 255)).sum()) * cell, 2),
                                             round(float((mask & (nav == 255)).sum()) * cell, 2))
    return out


def measured_table(data_path: Optional[str] = None, erode_px: int = NAV_ERODE_PX) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """``ROOM_FLOOR_M2`` as measured now: every room of every candidate scene (``eligible_rooms`` filters)."""
    return {sc: dict(sorted(measure_rooms(sc, data_path, erode_px).items())) for sc in SCENE_CANDIDATES}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("seed", "draw"):
        p = sub.add_parser(name)
        p.add_argument("split", choices=SPLITS)
        p.add_argument("variant")
        p.add_argument("k", type=int, nargs="?", default=0)
    p = sub.add_parser("rooms", help="measure every candidate scene's rooms and list the eligible ones")
    p.add_argument("--erode-px", type=int, default=NAV_ERODE_PX)
    p.add_argument("--min-nav-free-m2", type=float, default=MIN_NAV_FREE_M2)
    sub.add_parser("assign", help="the eval rooms of the held-object variants (frozen vs recomputed) and each task's unused rooms")
    sub.add_parser("shares", help="the train draw's room and scene shares")
    args = parser.parse_args(argv)
    if args.cmd == "seed":
        print(split_seed(args.split, args.variant, args.k))
    elif args.cmd == "draw":
        print("\t".join(draw_scene_room(args.split, args.variant, args.k)))
    elif args.cmd == "assign":
        got, orders = eval_assignment(EVAL_ROOMS, moves=EVAL_ROOM_MOVES)
        held = set(EVAL_ROOMS.values())
        for task in SCENE_DRAW_SUBTASKS:
            for a in sorted(x for x in EVAL_ROOMS if scene_draw_task(x) == task):
                sc, r = EVAL_ROOMS[a]
                print(f"{a:40s} {sc:26s} {r:18s} reachable {ROOM_FLOOR_M2[sc][r][2]:7.2f} m^2")
            print(f"{task} unused rooms in order: " + ", ".join(f"{s}/{r}" for s, r in orders.get(task, []) if (s, r) not in held))
        print(json.dumps({"moves": list(EVAL_ROOM_MOVES), "frozen_matches_recomputed": got == EVAL_ROOMS}))
    elif args.cmd == "shares":
        rooms, weights, cap = train_weights()
        total = sum(weights)
        per_scene: Dict[str, int] = {}
        for (sc, _), w in zip(rooms, weights):
            per_scene[sc] = per_scene.get(sc, 0) + w
        median = statistics.median(ROOM_FLOOR_M2[s][r][2] for s, r in rooms)
        print(f"{len(rooms)} rooms, median {median:.2f} m^2, cap {cap / 100:.2f} m^2 ({TRAIN_CAP_X_MEDIAN} x the median)")
        for sc in SCENE_CANDIDATES:
            print(f"{sc:26s} {100 * per_scene.get(sc, 0) / total:5.1f} %")
        for (sc, r), w in sorted(zip(rooms, weights), key=lambda t: (-t[1], t[0])):
            print(f"  {sc + '/' + r:46s} {100 * w / total:5.2f} %{'  (capped)' if w == cap else ''}")
    else:
        table = measured_table(erode_px=args.erode_px)
        for sc, rooms in table.items():
            for r, (area, free, nav) in rooms.items():
                print(f"{sc:26s} {r:18s} area {area:7.2f} free {free:7.2f} nav_free {nav:7.2f}")
        ok = eligible_rooms(table=table, min_nav_free_m2=args.min_nav_free_m2)
        print(json.dumps({"eligible": [f"{s}/{r}" for s, r in ok], "n_eligible": len(ok),
                          "reachable_m2": round(sum(table[s][r][2] for s, r in ok), 2),
                          "frozen_table_matches": table == {k: dict(v) for k, v in ROOM_FLOOR_M2.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
