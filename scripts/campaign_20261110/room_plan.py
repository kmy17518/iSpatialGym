#!/usr/bin/env python3
"""Select ten shared train rooms per free-room task, preserving eval siblings.

Floor-map measurements are CPU-only candidate screens. They do not replace
visibility, collision, placement, or open-loop validation in the loaded scene.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.campaign_20261110.contracts import SCENES, require, write_json_atomic
from ispatialgym.sampling.split_seeds import MAP_RESOLUTION_M, NAV_ERODE_PX


TASKS = ("camera_relocalization-standard", "camera_relocalization-obstructed", "object_scaling", "mental_rotation")


def path_extent(mask):
    """Longest two-sweep 8-neighbor path within a room, in metres (lower bound)."""
    import numpy as np

    points = np.argwhere(mask)
    if not len(points):
        return 0.0
    height, width = mask.shape
    steps = [(dy, dx, math.hypot(dy, dx)) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]

    def sweep(start):
        dist, queue = {start: 0.0}, [(0.0, start)]
        while queue:
            d, (y, x) = heapq.heappop(queue)
            if d != dist[(y, x)]:
                continue
            for dy, dx, cost in steps:
                yy, xx = y + dy, x + dx
                if not (0 <= yy < height and 0 <= xx < width and mask[yy, xx]):
                    continue
                # No diagonal shortcut through blocked corner cells.
                if dy and dx and not (mask[y, xx] and mask[yy, x]):
                    continue
                nxt = (yy, xx)
                if d + cost < dist.get(nxt, math.inf):
                    dist[nxt] = d + cost
                    heapq.heappush(queue, (d + cost, nxt))
        farthest = max(dist, key=dist.get)
        return farthest, dist

    remaining = {tuple(map(int, p)) for p in points}
    longest = 0.0
    while remaining:
        endpoint, reached = sweep(min(remaining))
        remaining.difference_update(reached)
        _, distances = sweep(endpoint)
        longest = max(longest, max(distances.values()))
    return round(longest * MAP_RESOLUTION_M, 3)


def measure_scene(dataset, scene):
    import cv2
    import numpy as np

    root = Path(dataset) / "behavior-1k-assets"
    cats = (root / "metadata/room_categories.txt").read_text().splitlines()
    layout = root / "scenes" / scene / "layout"
    images = [cv2.imread(str(layout / name), cv2.IMREAD_GRAYSCALE)
              for name in ("floor_insseg_0.png", "floor_semseg_0.png", "floor_trav_0.png")]
    require(all(im is not None for im in images), f"missing room maps for {scene}")
    ins, sem, trav = images
    side = int(ins.shape[0] * 0.01 / MAP_RESOLUTION_M)
    ins = cv2.resize(ins, (side, side), interpolation=cv2.INTER_NEAREST)
    sem = cv2.resize(sem, (side, side), interpolation=cv2.INTER_NEAREST)
    trav = cv2.resize(trav, (side, side))
    nav = cv2.erode((trav == 255).astype("uint8"), np.ones((NAV_ERODE_PX, NAV_ERODE_PX))) > 0
    by_type = {}
    for instance in sorted(set(np.unique(ins).tolist()) - {0}):
        rows, cols = np.where(ins == instance)
        by_type.setdefault(int(sem[rows[0], cols[0]]), []).append(int(instance))
    result = []
    for semantic, instances in by_type.items():
        for index, instance in enumerate(instances):
            room = f"{cats[semantic - 1]}_{index}"
            mask = ins == instance
            free = mask & nav
            result.append({"scene": scene, "room": room, "area_m2": round(float(mask.sum()) * MAP_RESOLUTION_M ** 2, 2),
                           "traversable_m2": round(float(free.sum()) * MAP_RESOLUTION_M ** 2, 2),
                           "path_extent_lower_bound_m": path_extent(free), "source": str(layout),
                           "validation": "floor_map_screen_only"})
    return result


def choose_rooms(measurements, standards, reserved_rooms, *, count=10):
    reserved = {(r["scene"], r["room"]) for r in reserved_rooms}
    by_key = {(r["scene"], r["room"]): r for r in measurements}
    eligible = {key: row for key, row in by_key.items() if key not in reserved and row["traversable_m2"] >= 1
                and not key[1].startswith(("garden_", "balcony_", "terrace_", "outdoor_"))}
    result = {}
    for task in TASKS:
        required = {(row["scene"], room) for row in standards if row["task"] == task for room in row["rooms"]}
        require(not required & reserved, f"{task}: a standard sibling room is held out")
        require(required <= set(eligible), f"{task}: required room fails traversability screen")
        require(len(required) <= count, f"{task}: more pinned standard rooms than allowed anchors")
        selected = set(required)
        # Fill missing scenes first, then add the largest remaining feasible room.
        while len(selected) < count:
            represented = {scene for scene, _ in selected}
            choices = [key for key in eligible if key not in selected]
            require(bool(choices), f"{task}: insufficient eligible rooms")
            key = max(choices, key=lambda k: (k[0] not in represented, eligible[k]["traversable_m2"], k))
            selected.add(key)
        rows = [dict(eligible[key], required_for_eval_sibling=key in required) for key in sorted(selected)]
        total = sum(r["traversable_m2"] for r in rows)
        for row in rows:
            row["sampling_weight"] = row["traversable_m2"] / total
        result[task] = rows
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--standards", required=True)
    parser.add_argument("--reserved-room", action="append", default=[], help="scene/room, repeatable")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    measurements = [row for scene in sorted(SCENES) for row in measure_scene(args.datasets, scene)]
    reserved = [dict(zip(("scene", "room"), value.split("/", 1))) for value in args.reserved_room]
    standards = json.loads(Path(args.standards).read_text())["templates"]
    plan = choose_rooms(measurements, standards, reserved)
    write_json_atomic(args.out, {"schema": "isg-room-plan-v1", "rooms_per_task": 10, "reserved_rooms": reserved,
                                 "sampling_room_weights": "traversable_area", "training_buckets": "soft",
                                 "tasks": plan, "measurements": measurements,
                                 "simulation_validation": "pending; inserted furniture may change traversability"})
    print(json.dumps({task: {"rooms": len(rows), "scenes": len({r['scene'] for r in rows}),
                             "long_path_rooms": sum(r["path_extent_lower_bound_m"] > 5 for r in rows)}
                      for task, rows in plan.items()}))


if __name__ == "__main__":
    main()
