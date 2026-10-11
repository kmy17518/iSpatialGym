"""Per-alias, per-split seeds and the scene / room draw of the held-object tasks (owner 2026-10-11: distinct eval init / goal
and scenes, train policy). Pure module: no simulator, the room measurement reads the layout PNGs only.

Seeds
-----
``split_seed(split, alias, k)`` is the sampler seed (``sample_isg --seed-split``, ``multiply_isg --split``) of ``alias`` in
``split`` (``eval`` / ``train``): ``2 * h + parity`` with ``h`` the first 8 bytes of SHA-256 of ``"<split>:<alias>"``
(``k = 0``) or ``"<split>:<alias>:<k>"`` (``k > 0``: the k-th next seed, for an eval offender re-sampled) modulo 2**29, and
parity 0 for eval, 1 for train. Guarantees:

* a train seed never equals an eval seed (any alias, any k): eval seeds are even, train seeds odd;
* no two aliases share a stream: within a split the seeds of ``(alias, k)`` are pairwise distinct for every alias of the
  shipped customs list and k < ``CHECKED_INDICES`` (:func:`check_unique`, run by the unit tests; a new alias collides with
  probability ~1e-5), and distinct seeds give disjoint attempt-seed streams - the samplers derive
  ``(seed * 7919 + t) * 1000003 + index * 1009 + attempt`` (t = template ordinal / multiply offset < 7919, index <= 990,
  attempt < 1009), an injective map;
* seeds stay below 2**30, so every derived attempt seed fits in int64.

Scene / room draw (held-object tasks: object scaling, mental rotation - their only scene role is the floor)
------------------------------------------------------------------------------------------------------
``draw_scene_room(split, alias, k)`` = the k-th (scene, room) draw of the alias in the split (eval: k = the seed index, so a
re-sampled offender moves too; train: k = the train instance index, one draw per instance), from SHA-256 of
``"scene:<split>:<alias>[:<k>]"``: the scene uniformly among the ``SCENE_CANDIDATES`` that have an eligible room, then a room
uniformly among that scene's eligible rooms. Eligible (:func:`eligible_rooms`): a room of a ``ROOM_TYPES`` type (living /
dining / bed / kitchen / office; never bathrooms, corridors, closets, storage, utility rooms, garages, gardens, entryways)
with at least ``MIN_NAV_FREE_M2`` (2.0 m^2) of free floor in the robot-eroded navigation map - the cells the held-object
start base is drawn from (``HeldObjectV2.sample_start_base``: a free cell of the eroded map inside the task room).

The numbers (``ROOM_FLOOR_M2``) are measured without the simulator by :func:`measure_rooms` from the scene's layout maps the
way OmniGibson reads them: room instances from ``floor_insseg_0`` / ``floor_semseg_0`` (nearest-resized to 0.1 m, named as
``SegmentationMap`` names them), free floor from ``floor_trav_0`` (the map with objects, resized to 0.1 m, < 255 -> 0), the
navigation map = that map eroded with a square ``NAV_ERODE_PX`` kernel (``TraversableMap._erode_trav_map``: radius = half
the R1Pro chassis diagonal (0.816 x 1.269 m) + 0.2 = 0.954 m -> 10 px). The eligible set depends on the kernel near the
threshold (an 11 px kernel leaves Rs_int / living_room_0 at 1.54 and house_double_floor_upper / bedroom_1 at 1.80 m^2);
the table is frozen here so that the draws never change with the data. ``python -m ispatialgym.sampling.split_seeds rooms``
prints the measured table, ``... seed <split> <alias> [k]`` and ``... draw <split> <alias> [k]`` one seed / draw.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
#: room types with floor to stand and hold an object in: living / dining / bed / kitchen / office
ROOM_TYPES = ("living_room", "television_room", "dining_room", "bedroom", "childs_room", "kitchen", "private_office", "shared_office")
#: minimum free floor of the robot-eroded navigation map inside the room
MIN_NAV_FREE_M2 = 2.0
MAP_RESOLUTION_M = 0.1
#: R1Pro: ceil((||(0.8165, 1.2688)|| / 2 + 0.2) / 0.1) = ceil(9.544)
NAV_ERODE_PX = 10

#: ``{scene: {room: (room area, free floor, free floor of the eroded navigation map)}}`` in m^2, every room of a ``ROOM_TYPES``
#: type, measured by :func:`measure_rooms` (2026-10-11, behavior-1k-assets of Behavior-1K 6a6194600)
ROOM_FLOOR_M2: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    "house_single_floor": {
        "bedroom_0": (20.24, 13.01, 3.20),
        "childs_room_0": (12.33, 6.89, 0.70),
        "childs_room_1": (13.86, 6.35, 1.15),
        "childs_room_2": (12.90, 7.76, 1.82),
        "dining_room_0": (14.63, 9.21, 1.76),
        "kitchen_0": (23.10, 13.44, 4.02),
        "living_room_0": (54.78, 39.00, 20.32),
        "living_room_1": (20.35, 13.86, 1.61),
    },
    "house_double_floor_lower": {
        "kitchen_0": (36.14, 22.27, 10.34),
        "living_room_0": (23.65, 12.70, 4.97),
    },
    "house_double_floor_upper": {
        "bedroom_0": (17.48, 9.35, 2.98),
        "bedroom_1": (19.32, 10.83, 2.44),
        "bedroom_2": (18.90, 9.00, 1.44),
        "living_room_0": (40.69, 24.50, 7.24),
    },
    "Rs_int": {
        "bedroom_0": (6.84, 1.91, 0.02),
        "kitchen_0": (4.75, 1.77, 0.00),
        "living_room_0": (22.40, 12.74, 2.18),
    },
    "restaurant_diner": {
        "dining_room_0": (68.35, 46.73, 21.17),
        "kitchen_0": (55.44, 34.76, 12.05),
    },
    "hotel_suite_large": {
        "bedroom_0": (39.78, 20.41, 4.24),
    },
    "office_cubicles_right": {
        "private_office_0": (15.17, 9.50, 0.42),
        "private_office_1": (21.84, 14.59, 2.60),
        "private_office_2": (21.45, 15.28, 2.75),
        "private_office_3": (21.86, 15.65, 2.92),
        "shared_office_0": (39.59, 17.52, 1.62),
    },
}


# ----------------------------------------------------------------------------------------------------------------
# seeds
# ----------------------------------------------------------------------------------------------------------------


def _key(prefix: str, split: str, alias: str, k: int) -> str:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if int(k) < 0:
        raise ValueError(f"seed index must be >= 0, got {k}")
    base = f"{prefix}{split}:{alias}"
    return base if int(k) == 0 else f"{base}:{int(k)}"


def _hash_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def split_seed(split: str, alias: str, k: int = 0) -> int:
    """The sampler seed of ``alias`` in ``split`` (``k``: the k-th next seed); see the module docstring for the guarantees."""
    h = _hash_int(_key("", split, alias, k)) % (1 << HASH_BITS)
    return 2 * h + _PARITY[split]


def check_unique(aliases: Iterable[str], indices: Sequence[int] = range(CHECKED_INDICES)) -> Dict[int, Tuple[str, str, int]]:
    """Every ``split_seed(split, alias, k)`` of ``aliases`` x ``SPLITS`` x ``indices``, keyed by seed; ``ValueError`` on a
    collision (two keys, one seed)."""
    seen: Dict[int, Tuple[str, str, int]] = {}
    for alias in sorted(set(aliases)):
        for split in SPLITS:
            for k in indices:
                s = split_seed(split, alias, k)
                if s in seen:
                    raise ValueError(f"split seed collision: {(split, alias, k)} and {seen[s]} -> {s}")
                seen[s] = (split, alias, int(k))
    return seen


# ----------------------------------------------------------------------------------------------------------------
# scene / room draw
# ----------------------------------------------------------------------------------------------------------------


def room_type(room: str) -> str:
    """``living_room_0`` -> ``living_room`` (OmniGibson names a room instance ``<type>_<i>``)."""
    head, _, tail = room.rpartition("_")
    return head if head and tail.isdigit() else room


def scene_draw_applies(subtask: str) -> bool:
    """Whether the subtask's (scene, room) comes from :func:`draw_scene_room` (the held-object tasks)."""
    return any(subtask == s or subtask.startswith(s + "-") for s in SCENE_DRAW_SUBTASKS)


def eligible_rooms(scene: Optional[str] = None, table: Optional[Dict[str, Dict[str, Tuple[float, float, float]]]] = None,
                   min_nav_free_m2: float = MIN_NAV_FREE_M2) -> List[Tuple[str, str]]:
    """``(scene, room)`` pairs of the draw, in ``SCENE_CANDIDATES`` order then room name: a ``ROOM_TYPES`` room with at least
    ``min_nav_free_m2`` of eroded free floor (``table`` defaults to ``ROOM_FLOOR_M2``)."""
    table = ROOM_FLOOR_M2 if table is None else table
    out = []
    for sc in SCENE_CANDIDATES:
        if scene is not None and sc != scene:
            continue
        for room, (_, _, nav) in sorted((table.get(sc) or {}).items()):
            if room_type(room) in ROOM_TYPES and float(nav) >= float(min_nav_free_m2):
                out.append((sc, room))
    return out


def draw_scene_room(split: str, alias: str, k: int = 0) -> Tuple[str, str]:
    """The k-th (scene, room) draw of ``alias`` in ``split``: the scene uniformly among the candidates with an eligible room,
    then one of its eligible rooms uniformly."""
    h = _hash_int(_key("scene:", split, alias, k))
    scenes = [sc for sc in SCENE_CANDIDATES if eligible_rooms(sc)]
    scene = scenes[h % len(scenes)]
    rooms = [r for _, r in eligible_rooms(scene)]
    return scene, rooms[(h // len(scenes)) % len(rooms)]


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
    """``ROOM_FLOOR_M2`` as measured now: every ``ROOM_TYPES`` room of every candidate scene."""
    out = {}
    for sc in SCENE_CANDIDATES:
        out[sc] = {r: v for r, v in sorted(measure_rooms(sc, data_path, erode_px).items()) if room_type(r) in ROOM_TYPES}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("seed", "draw"):
        p = sub.add_parser(name)
        p.add_argument("split", choices=SPLITS)
        p.add_argument("alias")
        p.add_argument("k", type=int, nargs="?", default=0)
    p = sub.add_parser("rooms", help="measure every candidate scene's rooms and list the eligible ones")
    p.add_argument("--erode-px", type=int, default=NAV_ERODE_PX)
    p.add_argument("--min-nav-free-m2", type=float, default=MIN_NAV_FREE_M2)
    args = parser.parse_args(argv)
    if args.cmd == "seed":
        print(split_seed(args.split, args.alias, args.k))
    elif args.cmd == "draw":
        print("\t".join(draw_scene_room(args.split, args.alias, args.k)))
    else:
        table = measured_table(erode_px=args.erode_px)
        for sc, rooms in table.items():
            for r, (area, free, nav) in rooms.items():
                print(f"{sc:26s} {r:18s} area {area:7.2f} free {free:7.2f} nav_free {nav:7.2f}")
        ok = eligible_rooms(table=table, min_nav_free_m2=args.min_nav_free_m2)
        print(json.dumps({"eligible": [f"{s}/{r}" for s, r in ok], "frozen_table_matches": table == {k: dict(v) for k, v in ROOM_FLOOR_M2.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
