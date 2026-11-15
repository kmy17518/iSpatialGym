"""Offline (CPU-only) in-view scoring of replay-time removal candidates (owner 2026-11-15 04:15).

The batched replay should prefer removing an object the head camera actually sees, without any extra render pass. The render
planner scores every candidate per demo from recorded data only, and the replay just reads the ranking (``--scene-pert-plan``):

* the merged scene = the full template (``isg_full_scene_file``) with the recorded crop's entries; an object's world box comes from
  its pose and scale in that scene file and its model's ``misc/metadata.json`` (``bbox_size``, ``base_link_offset``);
* the head camera = the recorded ``isg/head_camera_pose`` (every ``step``-th frame), USD convention (looks along -z, +y up), with
  the replay head camera's pinhole intrinsics (720 x 720, fx = fy = 306 by default: the replayer's ``goal_observation.intrinsics``);
* a frame counts when the projected box covers at least ``min_area`` of the image and the object is not behind a wall by the
  cheap room test: it stands in the camera's room (rooms = the floors objects' xy boxes), or the 2D segment camera -> object passes
  a door object's box (an opening into a neighbour room);
* score = counted frames / sampled frames. The planner ranks a goal group's candidates by their summed score (above
  ``threshold``); the replay removes the first removable single-object unit of that ranking and falls back to its group-keyed
  draw when none is in view.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCHEMA = "isg-replay-inview-v1"
DEFAULT_INTRINSICS = {"fx": 306.0, "fy": 306.0, "cx": 360.0, "cy": 360.0, "width": 720, "height": 720}
STRUCTURAL_EXTRA = ("floors", "walls", "ceilings")


def _quat_to_mat(q) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _registry(scene_file: Mapping[str, Any]) -> Mapping[str, Any]:
    state = scene_file.get("state") or {}
    return (state.get("registry") or {}).get("object_registry", {}) if "registry" in state else state.get("object_registry", {})


def merged_objects(full: Mapping[str, Any], recorded: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """{name: category, model, scale, pos, quat, in_rooms, in_crop} of the merged scene (recorded entries win)."""
    out = {}
    rec_init = (recorded or {}).get("objects_info", {}).get("init_info", {})
    rec_state = _registry(recorded or {})
    full_state = _registry(full)
    for name, info in {**full.get("objects_info", {}).get("init_info", {}), **rec_init}.items():
        args = info.get("args") or {}
        st = (rec_state.get(name) if name in rec_init else None) or full_state.get(name) or {}
        root = st.get("root_link") or {}
        pos, ori = root.get("pos"), root.get("ori")
        if pos is None or ori is None:
            continue
        scale = args.get("scale")
        scale = [1.0, 1.0, 1.0] if scale is None else ([float(scale)] * 3 if isinstance(scale, (int, float)) else [float(v) for v in scale])
        out[name] = {"category": args.get("category"), "model": args.get("model"), "scale": scale, "pos": [float(v) for v in pos],
                     "quat": [float(v) for v in ori], "in_rooms": list(args.get("in_rooms") or []), "in_crop": name in rec_init,
                     "type": args.get("type", "")}
    return out


_META: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}


def model_box(category: str, model: str, roots: Sequence[str]) -> Optional[Dict[str, Any]]:
    key = (category, model)
    if key not in _META:
        _META[key] = None
        for r in roots:
            p = os.path.join(r, "objects", str(category), str(model), "misc", "metadata.json")
            if os.path.exists(p):
                with open(p) as f:
                    m = json.load(f)
                if m.get("bbox_size") is not None:
                    _META[key] = {"size": [float(v) for v in m["bbox_size"]],
                                  "offset": [float(v) for v in (m.get("base_link_offset") or [0, 0, 0])]}
                break
    return _META[key]


def world_corners(obj: Mapping[str, Any], box: Mapping[str, Any]) -> np.ndarray:
    R = _quat_to_mat(obj["quat"])
    s = np.asarray(obj["scale"])
    center = np.asarray(obj["pos"]) + R @ (np.asarray(box["offset"]) * s)
    half = np.asarray(box["size"]) * s / 2
    signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=float)
    return center + (signs * half) @ R.T


def projected_area(corners: np.ndarray, cam_pos, cam_quat, K: Mapping[str, float], near: float = 0.05) -> float:
    """Fraction of the image covered by the clipped 2D box of the corners in front of the camera (0 when none is)."""
    R = _quat_to_mat(cam_quat)
    pc = (corners - np.asarray(cam_pos)) @ R  # camera frame: x right, y up, looking along -z
    front = pc[:, 2] < -near
    if not front.any():
        return 0.0
    if not front.all():  # box straddles the image plane: clip conservatively to its front corners
        pc = pc[front]
    zf = -pc[:, 2]
    u = K["cx"] + K["fx"] * pc[:, 0] / zf
    v = K["cy"] - K["fy"] * pc[:, 1] / zf
    w, h = float(K["width"]), float(K["height"])
    u0, u1 = max(0.0, float(u.min())), min(w, float(u.max()))
    v0, v1 = max(0.0, float(v.min())), min(h, float(v.max()))
    if u1 <= u0 or v1 <= v0:
        return 0.0
    return (u1 - u0) * (v1 - v0) / (w * h)


def _xy_box(corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return corners[:, :2].min(axis=0), corners[:, :2].max(axis=0)


def _segment_hits_box(p, q, lo, hi) -> bool:
    """2D segment p->q intersects the axis-aligned box [lo, hi] (slab test)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    t0, t1 = 0.0, 1.0
    for i in range(2):
        if abs(d[i]) < 1e-12:
            if p[i] < lo[i] or p[i] > hi[i]:
                return False
        else:
            a, b = (lo[i] - p[i]) / d[i], (hi[i] - p[i]) / d[i]
            a, b = min(a, b), max(a, b)
            t0, t1 = max(t0, a), min(t1, b)
            if t0 > t1:
                return False
    return True


def room_of(xy, rooms: Mapping[str, Tuple[np.ndarray, np.ndarray]]) -> Optional[str]:
    hits = [(float(np.prod(hi - lo)), r) for r, (lo, hi) in rooms.items() if np.all(xy >= lo) and np.all(xy <= hi)]
    return min(hits)[1] if hits else None  # the smallest room box containing the point


def score_demo(demo_dir: str, roots: Sequence[str], K: Optional[Mapping[str, float]] = None, step: int = 10,
               min_area: float = 0.002, door_margin: float = 0.3, structural: Sequence[str] = (),
               extra: Sequence[str] = ()) -> Dict[str, Any]:
    """In-view scores of the outside-crop, non-structural objects of one demo's merged scene (CPU, recorded data only)."""
    import h5py

    t0 = time.process_time()
    K = dict(K or DEFAULT_INTRINSICS)
    with h5py.File(os.path.join(demo_dir, "demo.hdf5"), "r") as f:
        attrs = f["data"].attrs
        recorded = json.loads(attrs["scene_file"])
        full_ref = json.loads(attrs["isg_full_scene_file"]) if "isg_full_scene_file" in attrs else None
        cams = f["data/demo_0/isg/head_camera_pose"][()]
        load_rooms = json.loads(attrs["isg_full_load_room_instances"]) if "isg_full_load_room_instances" in attrs else None
    full = recorded
    if isinstance(full_ref, str) and os.path.exists(full_ref):
        with open(full_ref) as fh:
            full = json.load(fh)
    objs = merged_objects(full, recorded if full is not recorded else None)
    cropped = full is not recorded
    corners, missing = {}, []
    for n, o in objs.items():
        b = model_box(o["category"], o["model"], roots)
        if b is None:
            missing.append(n)
            continue
        corners[n] = world_corners(o, b)
    rooms: Dict[str, List[np.ndarray]] = {}
    for n, o in objs.items():
        if o["category"] == "floors" and n in corners:
            lo, hi = _xy_box(corners[n])
            for r in o["in_rooms"]:
                if r in rooms:
                    rooms[r] = [np.minimum(rooms[r][0], lo), np.maximum(rooms[r][1], hi)]
                else:
                    rooms[r] = [lo, hi]
    rooms = {r: (v[0], v[1]) for r, v in rooms.items()}
    doors = [(_xy_box(corners[n])[0] - door_margin, _xy_box(corners[n])[1] + door_margin)
             for n, o in objs.items() if n in corners and "door" in str(o["category"])]
    skip = set(structural) | set(STRUCTURAL_EXTRA)
    # the merged replay scene loads the task rooms + their neighbours only (isg_full_load_room_instances)
    loaded = set(load_rooms) if load_rooms else None
    cands = [n for n, o in objs.items() if n in corners and (not cropped or not o["in_crop"])
             and (loaded is None or o["in_crop"] or bool(loaded & set(o["in_rooms"])))
             and str(o["category"]) not in skip and not str(o.get("type", "")).endswith("Robot")]
    probe = list(extra or ())
    idx = list(range(0, len(cams), max(1, int(step))))
    if idx and idx[-1] != len(cams) - 1:
        idx.append(len(cams) - 1)
    scored = cands + [n for n in probe if n in corners and n not in cands]
    scores = {n: 0 for n in scored}
    for i in idx:
        cp, cq = cams[i, :3].astype(float), cams[i, 3:7].astype(float)
        croom = room_of(cp[:2], rooms)
        for n in scored:
            if projected_area(corners[n], cp, cq, K) < min_area:
                continue
            c = corners[n].mean(axis=0)
            oroom = room_of(c[:2], rooms)
            same = croom is not None and (oroom == croom or croom in objs[n]["in_rooms"])
            ok = same or (croom is not None and oroom is not None
                          and any(_segment_hits_box(cp[:2], c[:2], lo, hi) for lo, hi in doors))
            if ok:
                scores[n] += 1
    n_frames = len(idx)
    return {"schema": SCHEMA, "demo": os.path.basename(os.path.normpath(demo_dir)), "cropped": cropped, "frames": n_frames,
            "step": int(step), "min_area": min_area, "intrinsics": K, "candidates": len(cands), "missing_metadata": len(missing),
            "rooms": len(rooms), "doors": len(doors), "load_rooms": sorted(loaded) if loaded else None,
            "probe": {n: round(scores.get(n, 0) / max(1, len(idx)), 4) for n in probe},
            "scores": {n: round(c / max(1, n_frames), 4) for n, c in sorted(scores.items()) if c and n in set(cands)},
            "cpu_s": round(time.process_time() - t0, 3)}


def group_rankings(per_demo: Mapping[str, Mapping[str, Any]], groups: Mapping[str, str], threshold: float = 0.05) -> Dict[str, List[str]]:
    """{goal group: candidate names by summed in-view score over its demos, those with mean score >= threshold}."""
    acc: Dict[str, Dict[str, float]] = {}
    count: Dict[str, int] = {}
    for demo, rec in per_demo.items():
        g = groups.get(demo, demo)
        count[g] = count.get(g, 0) + 1
        for n, s in rec.get("scores", {}).items():
            acc.setdefault(g, {})[n] = acc.setdefault(g, {}).get(n, 0.0) + float(s)
    out = {}
    for g, sc in acc.items():
        out[g] = [n for n, s in sorted(sc.items(), key=lambda kv: (-kv[1], kv[0])) if s / count[g] >= threshold]
    for g in count:
        out.setdefault(g, [])
    return out


def main(argv=None) -> int:
    """Render planner step: ``python -m ispatialgym.multienv.replay_inview --demos @list --out plan.json``."""
    import argparse

    from ispatialgym.multienv.replay_scene_perturbation import goal_group, STRUCTURAL, EXTRA_STRUCTURAL

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--roots", default=None, help="asset roots with objects/<category>/<model>/misc/metadata.json (comma separated); "
                                                  "default: $OMNIGIBSON_DATA_PATH/{behavior-1k-assets,ispatialgym-assets}")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--min-area", type=float, default=0.002)
    ap.add_argument("--threshold", type=float, default=0.05)
    a = ap.parse_args(argv)
    demos = [l.strip() for l in open(a.demos[1:])] if a.demos.startswith("@") else a.demos.split(",")
    demos = [d for d in demos if d]
    data = os.environ.get("OMNIGIBSON_DATA_PATH", "")
    roots = a.roots.split(",") if a.roots else [os.path.join(data, "behavior-1k-assets"), os.path.join(data, "ispatialgym-assets")]
    groups = {os.path.basename(os.path.normpath(d)): goal_group(d) for d in demos}
    plan = build_plan(demos, groups, roots, threshold=a.threshold, step=a.step, min_area=a.min_area,
                      structural=sorted(STRUCTURAL | EXTRA_STRUCTURAL))
    with open(a.out, "w") as f:
        json.dump(plan, f, indent=1)
    print(json.dumps({"demos": len(plan["demos"]), "cpu_s_total": plan["cpu_s_total"],
                      "ranked": {g: r[:3] for g, r in plan["rankings"].items()}}))
    return 0



def build_plan(demo_dirs: Sequence[str], groups: Mapping[str, str], roots: Sequence[str], threshold: float = 0.05,
               **kwargs) -> Dict[str, Any]:
    per = {}
    for d in demo_dirs:
        try:
            rec = score_demo(d, roots, **kwargs)
        except Exception as e:  # noqa: BLE001  (no ranking for this demo: the replay falls back to its group-keyed draw)
            rec = {"schema": SCHEMA, "demo": os.path.basename(os.path.normpath(d)), "error": repr(e)[:300], "scores": {}, "cpu_s": 0.0}
        per[rec["demo"]] = rec
    return {"schema": SCHEMA, "threshold": threshold, "demos": per, "groups": {d: groups.get(d, d) for d in per},
            "rankings": group_rankings(per, groups, threshold),
            "cpu_s_total": round(sum(r["cpu_s"] for r in per.values()), 3)}


if __name__ == "__main__":
    raise SystemExit(main())
