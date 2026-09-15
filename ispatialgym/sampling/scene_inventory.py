"""Scene inventories for ``ground_isg`` (TODO B4): one OmniGibson process per scene, full scene loaded, no robot motion.

Three inventories per scene, written under ``<out>/<scene>/``:

* ``artic/<name>.json`` — every articulated scene object (fixed base, openable joints): instance scale, root pose,
  rooms, per joint the empirically measured axis / hinge point / opening sign **in the instance frame** (so the
  numbers already include the per-axis scene scale), the handle candidates of the same geometric heuristic that produced
  ``runlogs/grasp_library_20260905/artic/handles`` (bars / knobs / edge fallbacks, contact + pinch axis + approach +
  pull direction), world-frame extras (handle heights, outward normal, face centre, free floor in front of the face)
  and the prior reviewer verdict of the canonical model when one exists (``artic_results.csv``). ``artic/strips/`` holds
  the review renders (overview / per-joint zoom with lettered markers / opened 60 %), same layout as the canonical
  strips, shot in the scene with the viewer camera.
* ``supports/<name>.json`` — every work surface of the scene (tables, desks, counters, cabinets, booths, ...): top
  raster summary (largest horizontal face, 2 cm raycast raster), hull polygon, the four local-frame edges with
  midpoint / inward normal / tangent / length / AABB overhang, free-floor runs in front of each edge at stand-offs
  0.35 / 0.45 / 0.55 m (with and without the chairs within 1 m, which the sampler removes), walls near the edge, the
  chairs, occupants and rooms.
* ``partners/<name>.json`` — large fixed partners for the IORS large-scale subtasks (beds, sofas, desks, breakfast
  tables, countertops): face raster + edges + free-floor runs as for supports, an under-clearance map (5 cm floor
  raster inside the footprint, clearance = height of the first partner hit of an upward ray) with per-edge summaries,
  and the supports within 3.5 m (AABB distance + geodesic on the robot-eroded map).

Usage::

    python -m ispatialgym.sampling.scene_inventory --scene house_single_floor --out runlogs/scene_inventory_20260915
    python -m ispatialgym.sampling.scene_inventory --scene Rs_int --parts artic --no-render

Everything is measured in place; joint perturbations are undone with a full simulator state restore per object.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym import REPO_ROOT

log = logging.getLogger("ispatialgym.sampling.scene_inventory")

SCENES = ("house_single_floor", "house_double_floor_upper", "house_double_floor_lower", "Rs_int", "hotel_suite_large",
          "restaurant_diner", "office_cubicles_right")
#: categories treated as large fixed partners (IORS ontop / under large-scale)
PARTNER_CATEGORIES = ("bed", "sofa", "couch", "desk", "breakfast_table", "countertop")
#: work-surface categories (the table-selection inventory's vocabulary)
SUPPORT_CATEGORIES = ("breakfast_table", "desk", "coffee_table", "conference_table", "pedestal_table", "countertop", "bar", "nightstand",
                      "console_table", "side_table", "commercial_kitchen_table", "cubicle", "booth", "bottom_cabinet", "bottom_cabinet_no_top",
                      "metal_bottom_cabinet", "serving_cart", "stand", "shelf", "bench", "ottoman")
#: categories whose openable joints are inventoried when the object has any movable joint
ARTIC_CATEGORIES = ("bottom_cabinet", "bottom_cabinet_no_top", "top_cabinet", "fridge", "oven", "dishwasher", "door", "clothes_dryer",
                    "washer", "microwave", "sliding_door", "openable_window", "trash_can", "toilet", "cedar_chest", "cooler")
STAND_OFFS = (0.35, 0.45, 0.55)
CHAIR_DISTANCE = 1.0
GLIB = os.path.join(REPO_ROOT, "runlogs", "grasp_library_20260905", "artic")

# ------------------------------------------------------------------------------------------------ handle heuristic
# Verbatim port of runlogs/grasp_library_20260905/scripts/artic_handles.py (in-memory pieces instead of OBJ files), so
# that scene instances get the same candidates / kinds / fallbacks the reviewers judged on the canonical models.
PROTRUDE_MIN = 0.008
GRASP_DEPTH_MIN = 0.012
BIG_FOOTPRINT = 0.20
MAX_PINCH = 0.085
EDGE_CLEAR = 0.03


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _basis_perp(n):
    a = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    u = _unit(np.cross(n, a))
    v = _unit(np.cross(n, u))
    return u, v


def infer_handles(info: dict, j: dict, pieces: List[Tuple[str, np.ndarray]]) -> dict:
    allv = np.concatenate([v for _, v in pieces]) if pieces else np.zeros((1, 3))
    lo, hi = np.array(j["link_bbox_min"]), np.array(j["link_bbox_max"])
    ext = hi - lo
    ctr = (lo + hi) / 2
    blo, bhi = np.array(info["base_link"]["bbox_min"]), np.array(info["base_link"]["bbox_max"])
    bctr = (blo + bhi) / 2
    axis = np.array(j["axis_dir_obj"])
    kind = "drawer" if j["type"] == "PrismaticJoint" else ("door" if abs(axis[2]) > 0.7 else "flap")
    if kind == "drawer":
        n = _unit(axis)
    else:
        cand = [i for i in range(3) if abs(axis[i]) < 0.7]
        thin = min(cand, key=lambda i: ext[i])
        n = np.zeros(3)
        n[thin] = 1.0
        if np.dot(ctr - bctr, n) < 0:
            n = -n
        if kind == "flap" and thin == 2 and (ctr[2] > bhi[2] - 0.02):
            n = np.array([0, 0, 1.0])
    u, v = _basis_perp(n)
    proj = allv @ n
    face_area = ext[np.argsort(np.abs(n))[:2]].prod()
    piece_stats = []
    for name, pv in pieces:
        p = pv @ n
        fu, fv = pv @ u, pv @ v
        foot = max(np.ptp(fu), 1e-4) * max(np.ptp(fv), 1e-4)
        piece_stats.append({"name": name, "pmax": float(p.max()), "pmin": float(p.min()), "foot": float(foot / max(face_area, 1e-6)), "verts": pv})
    big = [p for p in piece_stats if p["foot"] >= BIG_FOOTPRINT]
    s = max(p["pmax"] for p in big) if big else float(np.percentile(proj, 60))
    handles = [p for p in piece_stats if p["pmax"] > s + PROTRUDE_MIN and p["foot"] < BIG_FOOTPRINT]
    clusters = []
    for p in handles:
        pv = p["verts"][p["verts"] @ n > s + 0.003]
        if len(pv) < 3:
            continue
        c = pv.mean(0)
        placed = False
        for cl in clusters:
            if np.linalg.norm((c - cl["c"]) - np.dot(c - cl["c"], n) * n) < 0.06 and abs(np.dot(c - cl["c"], n)) < 0.05:
                cl["verts"].append(pv)
                cl["c"] = np.concatenate(cl["verts"]).mean(0)
                placed = True
                break
        if not placed:
            clusters.append({"verts": [pv], "c": c})
    cands = []
    for cl in clusters:
        pv = np.concatenate(cl["verts"])
        depth = float((pv @ n).max() - s)
        eu, ev = (pv @ u), (pv @ v)
        len_u, len_v = float(np.ptp(eu)), float(np.ptp(ev))
        long_dir, short_dir, L, W = (u, v, len_u, len_v) if len_u >= len_v else (v, u, len_v, len_u)
        if W > MAX_PINCH and L > MAX_PINCH:
            continue
        if depth < GRASP_DEPTH_MIN:
            cands.append({"kind": "flush_handle", "center": pv.mean(0).round(4).tolist(), "depth": round(depth, 4), "note": "protrudes < 12 mm; fingertips cannot get behind it"})
            continue
        centre_face = np.array([np.dot(pv.mean(0), u), np.dot(pv.mean(0), v)])
        contact_depth = s + max(depth / 2, depth - 0.012)
        contact = u * centre_face[0] + v * centre_face[1] + n * contact_depth
        cands.append({"kind": "bar" if L > 2.5 * W else "knob", "contact": contact.round(4).tolist(), "pinch_axis": short_dir.round(4).tolist(),
                      "approach_dir": (-n).round(4).tolist(), "width": round(W, 4), "length": round(L, 4), "depth": round(depth, 4), "face_s": round(s, 4),
                      "n_pieces": len(cl["verts"])})
    for sl in j.get("sub_links", []):
        slo, shi = np.array(sl["bbox_min"]), np.array(sl["bbox_max"])
        sc = (slo + shi) / 2
        se = shi - slo
        depth = float(max((np.array([[slo[i], shi[i]][int(n[i] > 0)] for i in range(3)]) @ n) - s, 0.0))
        inplane = [i for i in range(3) if abs(n[i]) < 0.7]
        short_i = min(inplane, key=lambda i: se[i])
        long_i = max(inplane, key=lambda i: se[i])
        pinch = np.zeros(3)
        pinch[short_i] = 1.0
        cands.append({"kind": "bar" if se[long_i] > 2.5 * se[short_i] else "knob", "contact": sc.round(4).tolist(), "pinch_axis": pinch.round(4).tolist(),
                      "approach_dir": (-n).round(4).tolist(), "width": round(float(se[short_i]), 4), "length": round(float(se[long_i]), 4), "depth": round(depth, 4),
                      "face_s": round(s, 4), "n_pieces": 0, "source": f"handle link {sl['link']}"})
    usable = [c for c in cands if c["kind"] in ("bar", "knob")]
    fallback = None
    if kind == "flap":
        hinge = np.array(j["hinge_point_obj"])
        e = ctr - hinge
        e = _unit(e - np.dot(e, axis) * axis)
        far = allv[np.argsort(allv @ e)[-max(3, len(allv) // 20):]]
        edge_c = far.mean(0)
        thickness = float(np.ptp(allv[(allv @ e) > (allv @ e).max() - 0.03] @ n))
        base_beyond = float(max(np.dot(np.array([blo[i] if e[i] < 0 else bhi[i] for i in range(3)]), e) - edge_c @ e, 0.0))
        exposed = base_beyond < 0.01 or (n[2] > 0.7 and lo[2] > bhi[2] - 0.02)
        fallback = {"kind": "edge_lid", "contact": (edge_c - e * 0.035).round(4).tolist(), "pinch_axis": n.round(4).tolist(), "approach_dir": (-e).round(4).tolist(),
                    "width": round(min(thickness, ext[np.argmax(np.abs(n))]), 4), "exposed": bool(exposed), "note": "pinch the lid/flap across its thickness at the free edge"}
    elif kind == "drawer":
        top = allv[(allv @ n) > s - 0.03]
        top_z = top[:, 2].max()
        others_above = any(np.array(o["link_bbox_min"])[2] < top_z + EDGE_CLEAR and np.array(o["link_bbox_max"])[2] > top_z and o["joint"] != j["joint"]
                           for o in info["joints"]) or (bhi[2] > top_z + 0.01)
        thickness = float(np.ptp(top[top[:, 2] > top_z - 0.03] @ n))
        fc = u * np.dot(ctr, u) + v * np.dot(ctr, v) + n * (s - thickness / 2)
        fc = np.array(fc, dtype=float)
        fc[2] = top_z - 0.035
        fallback = {"kind": "edge_drawer_top", "contact": fc.round(4).tolist(), "pinch_axis": n.round(4).tolist(), "approach_dir": [0, 0, -1.0],
                    "width": round(thickness, 4), "exposed": not others_above, "note": "pinch the drawer front across its thickness from above"}
    else:
        hinge = np.array(j["hinge_point_obj"])
        e = ctr - hinge
        e = _unit(e - np.dot(e, axis) * axis)
        far = allv[np.argsort(allv @ e)[-max(3, len(allv) // 20):]]
        edge_c = far.mean(0)
        others = [o for o in info["joints"] if o["joint"] != j["joint"]]
        blocked = any(np.all(np.array(o["link_bbox_min"]) - 0.005 < edge_c + e * 0.02) and np.all(np.array(o["link_bbox_max"]) + 0.005 > edge_c + e * 0.02) for o in others) or \
            (np.all(blo - 0.005 < edge_c + e * 0.02) and np.all(bhi + 0.005 > edge_c + e * 0.02))
        fallback = {"kind": "edge_door", "contact": (edge_c - e * 0.035).round(4).tolist(), "pinch_axis": n.round(4).tolist(), "approach_dir": (-e).round(4).tolist(),
                    "width": round(float(ext[np.argmax(np.abs(n))]), 4), "exposed": not blocked, "note": "pinch the door leaf across its thickness at the free edge"}
    for c in usable + ([fallback] if fallback else []):
        p = np.array(c["contact"])
        if kind == "drawer":
            pull = axis
        else:
            hinge = np.array(j["hinge_point_obj"])
            r = p - hinge
            r = r - np.dot(r, axis) * axis
            pull = _unit(np.cross(axis, r))
        c["pull_dir"] = np.array(pull, dtype=float).round(4).tolist()
    return {"joint": j["joint"], "type": j["type"], "kind": kind, "child_link": j["child_link"], "outward_normal": n.round(4).tolist(), "face_s": round(float(s), 4),
            "axis_dir_obj": j["axis_dir_obj"], "hinge_point_obj": j.get("hinge_point_obj"), "open_sign": j["open_sign"], "lower": j["lower"], "upper": j["upper"],
            "closed_q": j["closed_q"], "link_bbox_min": j["link_bbox_min"], "link_bbox_max": j["link_bbox_max"], "handles": usable,
            "flush_handles": [c for c in cands if c["kind"] == "flush_handle"], "fallback": fallback, "sub_links": j.get("sub_links", []),
            "status": "handle" if usable else ("edge" if fallback and fallback.get("exposed") else "ungraspable")}


# ------------------------------------------------------------------------------------------------ geometry helpers

def convex_hull_xy(pts: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull of (N, 2) points."""
    pts = np.unique(np.round(pts, 4), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def prior_reviews() -> Dict[Tuple[str, str, str], dict]:
    """(category, model, joint) -> reviewer verdict + opening-test result of the canonical model (artic_results.csv)."""
    path = os.path.join(GLIB, "artic_results.csv")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            out[(r["category"], r["model"], r["joint"])] = {
                "chosen": r.get("review_chosen"), "pinch": r.get("review_pinch"), "motion": r.get("review_motion"),
                "graspable": r.get("review_graspable"), "handle_desc": r.get("handle_desc"), "result": r.get("result"), "setting": r.get("setting")}
    return out


class Inventory:
    def __init__(self, scene_model: str, out_root: str, render: bool = True):
        from ispatialgym.sampling import robot_utils  # noqa: F401  (warp pinning before omnigibson)
        from ispatialgym.sampling.sim_utils import SimWorld

        self.scene_model = scene_model
        self.out = os.path.join(out_root, scene_model)
        for d in ("artic", "artic/renders", "artic/strips", "supports", "partners"):
            os.makedirs(os.path.join(self.out, d), exist_ok=True)
        t0 = time.time()
        self.world = SimWorld(scene_model, rooms=None)
        self.scene = self.world.scene
        self.trav = self.scene.trav_map
        self.floor_map = self.trav.floor_map[0]
        self.eroded = self.trav._erode_trav_map(th.clone(self.floor_map), robot=self.world.robot)
        self.load_s = time.time() - t0
        self.render = render
        self.markers = None
        if render:
            # Isaac Sim 5.1's synthetic-data graph (the robot camera's seg_instance annotator) segfaults with light
            # fixtures and glass in view (see SimWorld). This module only needs rgb from the viewer camera, so the robot
            # is parked far outside the scene (its camera sees only sky) and the light fixtures are shown again for the
            # review renders. Detaching the annotator instead (remove_modality) left the physics view invalid.
            from ispatialgym.sampling import geometry as G

            robot = self.world.robot
            robot.set_position_orientation(th.tensor([60.0, 60.0, 5.0]), G.IDENTITY_QUAT)
            robot.disable_gravity()
            robot.keep_still()
            for o in self.world.render_hidden:
                o.visible = True
        self.summary: Dict[str, Any] = {"scene": scene_model, "load_s": round(self.load_s, 1), "artic": {}, "supports": {}, "partners": {}}
        self._prim_to_obj = {l.prim_path: o for o in self.scene.objects for l in o.links.values()}

    # -- floor / rooms -------------------------------------------------------------------------------------------
    def _map_value(self, m: th.Tensor, xy) -> bool:
        ij = self.trav.world_to_map(th.as_tensor([float(xy[0]), float(xy[1])]))
        i, j = int(ij[0]), int(ij[1])
        if i < 0 or j < 0 or i >= m.shape[0] or j >= m.shape[1]:
            return False
        return bool(m[i, j] == 255)

    def free(self, xy) -> bool:
        return self._map_value(self.floor_map, xy)

    def free_eroded(self, xy) -> bool:
        return self._map_value(self.eroded, xy)

    def room_of(self, xy) -> Optional[str]:
        try:
            return self.scene.seg_map.get_room_instance_by_point(th.as_tensor([float(xy[0]), float(xy[1])]))
        except Exception:
            return None

    def floor_z_under(self, obj) -> float:
        lo, hi = obj.aabb
        c = (lo + hi) / 2
        start = th.tensor([float(c[0]), float(c[1]), float(lo[2]) + 0.02])
        end = start.clone()
        end[2] = float(lo[2]) - 1.5
        own = {l.prim_path for l in obj.links.values()}
        for h in self.world.raycast(start.unsqueeze(0), end.unsqueeze(0)):
            if h["hit"] and h["rigidBody"] not in own:
                return float(h["position"][2])
        return float(self.world.floor_z)

    # -- shared: edges with free-floor runs -----------------------------------------------------------------------
    def chairs_near(self, obj, distance: float = CHAIR_DISTANCE) -> List[Any]:
        from ispatialgym.sampling.scene_variants import CHAIR_CATEGORIES

        lo, hi = obj.aabb
        out = []
        for o in self.scene.objects:
            if o.category not in CHAIR_CATEGORIES or o is obj:
                continue
            clo, chi = o.aabb
            c = (clo + chi) / 2
            d = th.maximum(th.maximum(lo[:2] - c[:2], c[:2] - hi[:2]), th.zeros(2))
            if float(th.linalg.norm(d)) <= distance:
                out.append(o)
        return out

    def _inside_any_aabb(self, xy, objs) -> bool:
        for o in objs:
            lo, hi = o.aabb
            if float(lo[0]) <= xy[0] <= float(hi[0]) and float(lo[1]) <= xy[1] <= float(hi[1]):
                return True
        return False

    def edges_with_runs(self, obj, top, chairs: List[Any], lateral_step: float = 0.05) -> List[dict]:
        """The four local-frame edges of ``top`` with the free-floor run of base positions in front of each."""
        from ispatialgym.sampling import geometry as G

        lo, hi = obj.aabb
        aabb_xy = G.aabb_corners(lo, hi)[:, :2]
        out = []
        for name, mid, normal, length in top.edges():
            inward = -normal
            tangent = th.tensor([-float(inward[1]), float(inward[0])])
            overhang = max(0.0, float((aabb_xy @ normal).max()) - float(th.dot(mid, normal)))
            if overhang < 0.05:
                overhang = 0.0
            half = length / 2
            lats = th.arange(-half, half + 1e-6, lateral_step).tolist() if length > lateral_step else [0.0]
            runs = {}
            for s in STAND_OFFS:
                free_with, free_wo = [], []
                for lat in lats:
                    p = mid + normal * (s + overhang) + tangent * lat
                    f = self.free(p)
                    free_with.append(f)
                    free_wo.append(f or self._inside_any_aabb((float(p[0]), float(p[1])), chairs))
                runs[f"{s:.2f}"] = {
                    "free_fraction": round(sum(free_with) / max(len(lats), 1), 3),
                    "free_fraction_chairs_removed": round(sum(free_wo) / max(len(lats), 1), 3),
                    "longest_free_run_m": round(_longest_run(free_wo) * lateral_step, 2),
                    "free_lateral_range_m": _free_range(free_wo, lats),
                }
            # wall in front of the edge (horizontal ray from the edge midpoint at top height + 0.3 m, 1.2 m out)
            start = th.tensor([float(mid[0]), float(mid[1]), top.z + 0.3])
            end = start + th.tensor([float(normal[0]), float(normal[1]), 0.0]) * 1.2
            wall_d = None
            for h in self.world.raycast(start.unsqueeze(0), end.unsqueeze(0)):
                if h["hit"]:
                    o = self._prim_to_obj.get(h["rigidBody"])
                    if o is not None and o is not obj:
                        wall_d = {"object": o.name, "category": o.category, "distance_m": round(float(h["distance"]), 3)}
            room = self.room_of(mid + normal * 0.6)
            out.append({"name": name, "mid": [round(float(v), 4) for v in mid], "inward": [round(float(v), 4) for v in inward],
                        "tangent": [round(float(v), 4) for v in tangent], "length_m": round(float(length), 3), "overhang_m": round(overhang, 3),
                        "room_in_front": room, "blocked_in_front": wall_d, "runs": runs})
        return out

    # -- (a) articulated scene objects ----------------------------------------------------------------------------
    def inventory_artic(self, only: Optional[Sequence[str]] = None):
        import omnigibson as og

        reviews = prior_reviews()
        objs = [o for o in self.scene.objects if o.fixed_base and getattr(o, "n_joints", 0) > 0 and o.n_joints > 0
                and (o.category in ARTIC_CATEGORIES or self._openable_meta(o) is not None)]
        if only:
            objs = [o for o in objs if o.name in only]
        log.info("%s: %d articulated scene objects", self.scene_model, len(objs))
        for obj in objs:
            t0 = time.time()
            state = og.sim.dump_state()
            try:
                rec = self._artic_record(obj, reviews)
                og.sim.load_state(state)
                self.world.keep_still(self.scene.objects)
                if self.render and rec.get("joints"):
                    try:
                        rec["strip"] = self._render_strip(obj, rec)
                    except Exception as e:  # renders are a convenience for the review, never a reason to lose the record
                        log.warning("%s: render failed: %r", obj.name, e)
                        rec["strip"] = None
                    og.sim.load_state(state)
                    self.world.keep_still(self.scene.objects)
                with open(os.path.join(self.out, "artic", f"{obj.name}.json"), "w") as f:
                    json.dump(rec, f, indent=1)
                self.summary["artic"][obj.name] = {"category": obj.category, "model": obj.model, "n_joints": len(rec.get("joints", [])),
                                                    "status": [j["status"] for j in rec.get("joints", [])], "s": round(time.time() - t0, 1)}
                log.info("artic %s (%s/%s): %d joints %s in %.1fs", obj.name, obj.category, obj.model, len(rec.get("joints", [])),
                         [j["status"] for j in rec.get("joints", [])], time.time() - t0)
            except Exception as e:
                og.sim.load_state(state)
                log.error("artic %s failed: %r\n%s", obj.name, e, traceback.format_exc())
                self.summary["artic"][obj.name] = {"category": obj.category, "model": obj.model, "error": repr(e)}

    def rerender_artic(self, only: Optional[Sequence[str]] = None):
        """Re-render the review strips from the existing ``artic/<name>.json`` records (no re-measurement)."""
        import omnigibson as og

        adir = os.path.join(self.out, "artic")
        names = sorted(fn[:-5] for fn in os.listdir(adir) if fn.endswith(".json"))
        if only:
            names = [n for n in names if n in only]
        for name in names:
            with open(os.path.join(adir, f"{name}.json")) as f:
                rec = json.load(f)
            if not rec.get("joints"):
                continue
            try:
                obj = self.world.obj(name)
            except KeyError:
                continue
            state = og.sim.dump_state()
            try:
                rec["strip"] = self._render_strip(obj, rec)
                with open(os.path.join(adir, f"{name}.json"), "w") as f:
                    json.dump(rec, f, indent=1)
                log.info("re-rendered %s", name)
            except Exception as e:
                log.warning("%s: render failed: %r", name, e)
            finally:
                og.sim.load_state(state)
                self.world.keep_still(self.scene.objects)

    def _openable_meta(self, obj) -> Optional[list]:
        from omnigibson.utils.asset_utils import get_dataset_path

        path = os.path.join(get_dataset_path("behavior-1k-assets"), "objects", obj.category, obj.model, "misc", "metadata.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                md = json.load(f)
        except Exception:
            return None
        oj = md.get("openable_joint_ids")
        return oj if oj else None

    def _link_pieces_local(self, link) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        import omnigibson.utils.transform_utils as T
        from omnigibson.utils.usd_utils import mesh_prim_shape_to_trimesh_mesh

        lp, lq = link.get_position_orientation()
        out = []
        for cm_name, cm in link.collision_meshes.items():
            pos, quat = cm.get_position_orientation()
            pos, quat = T.relative_pose_transform(pos, quat, lp, lq)
            if cm.geom_type == "Mesh":
                verts = cm.points.numpy().astype(np.float64)
                faces = cm.faces.numpy().astype(np.int64)
            else:
                tm = mesh_prim_shape_to_trimesh_mesh(cm.prim)
                verts = np.asarray(tm.vertices, dtype=np.float64)
                faces = np.asarray(tm.faces, dtype=np.int64)
            scale = cm.get_world_scale().numpy().astype(np.float64)
            R = T.quat2mat(quat).numpy().astype(np.float64)
            out.append((cm_name, (verts * scale) @ R.T + pos.numpy().astype(np.float64), faces))
        return out

    def _artic_record(self, obj, reviews) -> dict:
        import omnigibson as og
        import omnigibson.utils.transform_utils as T

        def to_obj(p, q, rp, rq):
            pos, quat = T.relative_pose_transform(th.as_tensor(p, dtype=th.float32), th.as_tensor(q, dtype=th.float32), rp, rq)
            return pos.numpy().astype(np.float64), quat.numpy().astype(np.float64)

        oj = self._openable_meta(obj) or []
        joint_dir = {}
        for e in oj:
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                joint_dir[e[1]] = (e[2] if len(e) > 2 else 1)
        if not joint_dir:
            joint_dir = {jn: 1 for jn in obj.joints}
        jl, ju = obj.joint_lower_limits.clone(), obj.joint_upper_limits.clone()
        names = list(obj.joints.keys())
        q_now = obj.get_joint_positions().clone()
        q_closed = jl.clone()
        for i, jn in enumerate(names):
            if joint_dir.get(jn, 1) < 0:
                q_closed[i] = ju[i]
        obj.set_joint_positions(q_closed)
        obj.set_joint_velocities(th.zeros_like(q_closed))
        obj.keep_still()
        og.sim.step_physics()
        rp, rq = obj.root_link.get_position_orientation()
        rp, rq = rp.float(), rq.float()
        alo, ahi = obj.aabb
        scale = [round(float(s), 4) for s in obj.scale] if hasattr(obj, "scale") else None
        rooms = list(obj.in_rooms or [])
        info: Dict[str, Any] = {
            "scene": self.scene_model, "name": obj.name, "category": obj.category, "model": obj.model, "key": f"{obj.category}__{obj.model}",
            "scale": scale, "rooms": rooms, "root_pose_world": {"position": rp.tolist(), "orientation": rq.tolist()},
            "aabb_world": {"min": alo.tolist(), "max": ahi.tolist(), "extent": (ahi - alo).tolist()},
            "joint_positions_in_scene": {n: float(q_now[i]) for i, n in enumerate(names)},
            "openable_joint_ids": oj, "canonical_annotation": os.path.exists(os.path.join(GLIB, "handles", f"{obj.category}__{obj.model}.json")),
            "joints": [], "base_link": {},
        }
        base_pieces = self._link_pieces_local(obj.root_link)
        bl_p, bl_q = to_obj(*obj.root_link.get_position_orientation(), rp, rq)
        Rb = T.quat2mat(th.as_tensor(bl_q, dtype=th.float32)).numpy()
        bv = np.concatenate([v @ Rb.T + bl_p for _, v, _ in base_pieces]) if base_pieces else np.zeros((1, 3))
        info["base_link"] = {"bbox_min": bv.min(0).round(4).tolist(), "bbox_max": bv.max(0).round(4).tolist(), "n_pieces": len(base_pieces)}
        raw_joints = []
        for i, jn in enumerate(names):
            if jn not in joint_dir:
                continue
            j = obj.joints[jn]
            span = float(ju[i] - jl[i])
            if span <= 1e-6:
                continue
            child = j.body1.split("/")[-1]
            if child not in obj.links:
                continue
            link = obj.links[child]
            d = joint_dir[jn]
            delta = min(0.05 if j.joint_type == "PrismaticJoint" else 0.25, 0.5 * span) * d
            p0, o0 = link.get_position_orientation()
            q1 = q_closed.clone()
            q1[i] = q_closed[i] + delta
            obj.set_joint_positions(q1)
            obj.keep_still()
            og.sim.step_physics()
            p1, o1 = link.get_position_orientation()
            obj.set_joint_positions(q_closed)
            obj.keep_still()
            og.sim.step_physics()
            p0o, o0o = to_obj(p0, o0, rp, rq)
            p1o, o1o = to_obj(p1, o1, rp, rq)
            entry = {"joint": jn, "type": j.joint_type, "child_link": child, "lower": float(jl[i]), "upper": float(ju[i]), "closed_q": float(q_closed[i]),
                     "open_sign": int(d), "link_pos_obj": p0o.round(5).tolist(), "link_quat_obj": o0o.round(5).tolist()}
            if j.joint_type == "PrismaticJoint":
                v = p1o - p0o
                entry["axis_dir_obj"] = (v / max(np.linalg.norm(v), 1e-9)).round(5).tolist()
                entry["travel_per_unit"] = float(np.linalg.norm(v) / abs(delta))
                entry["hinge_point_obj"] = p0o.round(5).tolist()
            else:
                R0 = T.quat2mat(th.as_tensor(o0o, dtype=th.float32)).numpy()
                R1 = T.quat2mat(th.as_tensor(o1o, dtype=th.float32)).numpy()
                Rrel = R1 @ R0.T
                ang = float(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1)))
                ax = np.array([Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0], Rrel[1, 0] - Rrel[0, 1]])
                ax = ax / max(np.linalg.norm(ax), 1e-9)
                A = np.eye(3) - Rrel
                b = p1o - Rrel @ p0o
                c, *_ = np.linalg.lstsq(A, b, rcond=None)
                entry["axis_dir_obj"] = ax.round(5).tolist()
                entry["hinge_point_obj"] = c.round(5).tolist()
                entry["measured_angle"] = round(ang, 4)
                entry["expected_angle"] = round(abs(delta), 4)
            pieces = self._link_pieces_local(link)
            lp, lq = to_obj(p0, o0, rp, rq)
            Rl = T.quat2mat(th.as_tensor(lq, dtype=th.float32)).numpy()
            pieces_obj = [(pn, v @ Rl.T + lp) for pn, v, _ in pieces]
            allv = np.concatenate([v for _, v in pieces_obj]) if pieces_obj else np.zeros((1, 3))
            entry.update({"n_pieces": len(pieces_obj), "link_bbox_min": allv.min(0).round(4).tolist(), "link_bbox_max": allv.max(0).round(4).tolist()})
            subs = []
            for jn2, j2 in obj.joints.items():
                if j2.body0.split("/")[-1] == child:
                    sname = j2.body1.split("/")[-1]
                    if sname not in obj.links:
                        continue
                    sl = obj.links[sname]
                    sp = self._link_pieces_local(sl)
                    slp, slq = to_obj(*sl.get_position_orientation(), rp, rq)
                    Rs = T.quat2mat(th.as_tensor(slq, dtype=th.float32)).numpy()
                    if sp:
                        sv = np.concatenate([v @ Rs.T + slp for _, v, _ in sp])
                        subs.append({"link": sname, "joint": jn2, "bbox_min": sv.min(0).round(4).tolist(), "bbox_max": sv.max(0).round(4).tolist()})
            entry["sub_links"] = subs
            raw_joints.append((entry, pieces_obj))
        info["joints"] = [e for e, _ in raw_joints]
        # handle inference on the instance geometry (already scaled), then world-frame extras + prior verdicts
        results = []
        R = T.quat2mat(rq).numpy().astype(np.float64)
        rpn = rp.numpy().astype(np.float64)
        for entry, pieces_obj in raw_joints:
            try:
                r = infer_handles(info, entry, pieces_obj)
            except Exception as e:
                r = {"joint": entry["joint"], "type": entry["type"], "child_link": entry["child_link"], "status": "error", "error": repr(e)}
            if "outward_normal" in r:
                n_w = R @ np.array(r["outward_normal"])
                r["outward_normal_world"] = n_w.round(4).tolist()
                lo, hi = np.array(r["link_bbox_min"]), np.array(r["link_bbox_max"])
                face_c_obj = (lo + hi) / 2
                face_c_obj = face_c_obj + np.array(r["outward_normal"]) * (r["face_s"] - np.dot(face_c_obj, np.array(r["outward_normal"])))
                P = R @ face_c_obj + rpn
                r["face_centre_world"] = P.round(4).tolist()
                for h in r.get("handles", []) + ([r["fallback"]] if r.get("fallback") else []):
                    cw = R @ np.array(h["contact"]) + rpn
                    h["contact_world"] = cw.round(4).tolist()
                    h["height_world"] = round(float(cw[2]), 3)
                    h["pinch_axis_world"] = (R @ np.array(h["pinch_axis"])).round(4).tolist()
                    h["approach_dir_world"] = (R @ np.array(h["approach_dir"])).round(4).tolist()
                    h["pull_dir_world"] = (R @ np.array(h["pull_dir"])).round(4).tolist()
                # free floor in front of the face: walk along the horizontal outward normal from the face centre
                if abs(float(n_w[2])) < 0.9:
                    nh = np.array([n_w[0], n_w[1], 0.0])
                    nh = nh / max(np.linalg.norm(nh), 1e-9)
                    r["front_free_m"] = self._free_run_along(P[:2], nh[:2], self.free)
                    r["front_free_eroded_m"] = self._free_run_along(P[:2], nh[:2], self.free_eroded)
                    r["room_in_front"] = self.room_of(P[:2] + nh[:2] * 0.8)
            r["prior_review"] = reviews.get((obj.category, obj.model, entry["joint"]))
            results.append(r)
        info["joints"] = results
        return info

    def _free_run_along(self, xy, direction, test, step: float = 0.05, max_d: float = 3.0) -> float:
        d = 0.0
        run = 0.0
        started = False
        while d <= max_d:
            p = np.asarray(xy) + np.asarray(direction) * d
            f = test(p)
            if f:
                started = True
                run = d
            elif started:
                break
            d += step
        return round(run, 2)

    # -- renders ----------------------------------------------------------------------------------------------------
    def _ensure_markers(self):
        if self.markers is not None:
            return
        import omnigibson as og
        from omnigibson.objects.primitive_object import PrimitiveObject

        colors = {"A": (1.0, 0.1, 0.1), "B": (0.1, 0.8, 0.1), "C": (0.1, 0.3, 1.0), "D": (1.0, 0.5, 0.0), "F": (1.0, 0.9, 0.1), "X": (0.5, 0.5, 0.5)}
        state = og.sim.dump_state()
        og.sim.stop()
        pool = {}
        for letter, col in colors.items():
            pool[letter] = []
            for k in range(6):
                m = PrimitiveObject(name=f"isg_mk_{letter}_{k}", primitive_type="Sphere", radius=0.016, visual_only=True, rgba=[*col, 1.0])
                self.scene.add_object(m)
                pool[letter].append(m)
        og.sim.play()
        og.sim.load_state(state)
        self.markers = pool
        self._far = th.tensor([60.0, 60.0, -5.0])
        self._park_markers()

    def _park_markers(self):
        for ms in self.markers.values():
            for m in ms:
                m.set_position_orientation(self._far, th.tensor([0.0, 0.0, 0.0, 1.0]))

    def _place_markers(self, items):
        import omnigibson as og

        self._park_markers()
        used = {k: 0 for k in self.markers}
        for letter, p in items:
            k = used.get(letter, 0)
            if k < len(self.markers[letter]):
                self.markers[letter][k].set_position_orientation(th.as_tensor(p, dtype=th.float32), th.tensor([0.0, 0.0, 0.0, 1.0]))
                used[letter] = k + 1
        og.sim.step()

    def _grab(self, cam_pos, target, path, size: int = 420):
        import omnigibson as og
        import omnigibson.utils.transform_utils as T
        from PIL import Image

        cam = og.sim.viewer_camera
        cam_pos = th.as_tensor(cam_pos, dtype=th.float32)
        target = th.as_tensor(target, dtype=th.float32)
        f = target - cam_pos
        f = f / th.linalg.norm(f)
        up = th.tensor([0.0, 0.0, 1.0])
        r = th.linalg.cross(f, up)
        if th.linalg.norm(r) < 1e-3:
            r = th.linalg.cross(f, th.tensor([1.0, 0.0, 0.0]))
        r = r / th.linalg.norm(r)
        u = th.linalg.cross(r, f)
        cam.set_position_orientation(cam_pos, T.mat2quat(th.stack([r, u, -f], dim=1)))
        for _ in range(4):
            og.sim.render()
        rgb = cam.get_obs()[0]["rgb"][..., :3].cpu().numpy()
        h, w = rgb.shape[:2]
        s = min(h, w)
        Image.fromarray(rgb[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]).resize((size, size), Image.LANCZOS).save(path)

    def _render_strip(self, obj, rec: dict) -> str:
        import omnigibson as og
        from PIL import Image, ImageDraw, ImageFont

        self._ensure_markers()
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
        rp = np.array(rec["root_pose_world"]["position"])
        import omnigibson.utils.transform_utils as T

        R = T.quat2mat(th.tensor(rec["root_pose_world"]["orientation"])).numpy()
        alo, ahi = np.array(rec["aabb_world"]["min"]), np.array(rec["aabb_world"]["max"])
        ext = ahi - alo
        c = (alo + ahi) / 2
        joints = [j for j in rec["joints"] if "outward_normal_world" in j]
        n0 = np.array(joints[0]["outward_normal_world"]) if joints else np.array([1.0, 0, 0])
        if abs(n0[2]) > 0.7:
            n0 = np.array([1.0, 0, 0])
        n0[2] = 0.0
        n0 = _unit(n0)
        side = np.cross(n0, [0, 0, 1.0])
        d = 1.2 * float(np.linalg.norm(ext)) + 0.4
        # stay inside the room: the camera may not go further out than the free floor in front of the face allows
        front = [j.get("front_free_m") for j in joints if j.get("front_free_m") is not None]
        if front:
            d = max(0.9, min(d, max(front) + 0.25))
        names = list(obj.joints.keys())
        q = obj.get_joint_positions().clone()
        for j in joints:
            if j["joint"] in names:
                q[names.index(j["joint"])] = j["closed_q"]
        obj.set_joint_positions(q)
        obj.keep_still()
        og.sim.step()  # full step: the render transforms only follow physics in a full simulator step

        def world(p):
            return R @ np.array(p) + rp

        items = []
        for j in joints:
            for k, h in enumerate(j.get("handles", [])[:4]):
                items.append(("ABCD"[k], world(h["contact"])))
            if j.get("fallback") and j.get("status") != "handle":
                items.append(("F", world(j["fallback"]["contact"])))
            for x in j.get("flush_handles", [])[:2]:
                items.append(("X", world(x["center"])))
        rdir = os.path.join(self.out, "artic", "renders")
        views = []
        self._place_markers(items)
        p_over = os.path.join(rdir, f"{obj.name}__overview.png")
        self._grab(c + n0 * d * 0.9 + side * d * 0.2 + np.array([0, 0, d * 0.25]), c, p_over)
        views.append(("overview (closed)", p_over))
        for ji, j in enumerate(joints):
            jitems = [("ABCD"[k], world(h["contact"])) for k, h in enumerate(j.get("handles", [])[:4])]
            if j.get("fallback") and j.get("status") != "handle":
                jitems.append(("F", world(j["fallback"]["contact"])))
            jitems += [("X", world(x["center"])) for x in j.get("flush_handles", [])[:2]]
            self._place_markers(jitems)
            lo, hi = world(j["link_bbox_min"]), world(j["link_bbox_max"])
            lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
            lc = (lo + hi) / 2
            lext = float(np.linalg.norm(hi - lo))
            n = np.array(j["outward_normal_world"])
            sd = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.7 else np.array([1.0, 0, 0])
            dz = 1.1 * lext + 0.25
            p = os.path.join(rdir, f"{obj.name}__{ji}_zoom.png")
            self._grab(lc + n * dz * 0.85 + sd * dz * 0.3 + np.array([0, 0, dz * 0.25]), lc, p)
            views.append((f"{j['joint']} ({j.get('kind')}, {j['status']})", p))
        self._place_markers(items)
        q2 = q.clone()
        for j in joints:
            if j["joint"] in names:
                i = names.index(j["joint"])
                q2[i] = j["closed_q"] + j["open_sign"] * 0.6 * (j["upper"] - j["lower"])
        obj.set_joint_positions(q2)
        obj.keep_still()
        og.sim.step()
        p_open = os.path.join(rdir, f"{obj.name}__open.png")
        self._grab(c + n0 * d * 0.9 + side * d * 0.2 + np.array([0, 0, d * 0.25]), c, p_open)
        views.append(("opened 60 % (markers at closed positions)", p_open))
        obj.set_joint_positions(q)
        obj.keep_still()
        og.sim.step()
        self._park_markers()
        S = 420
        cols = min(len(views), 4)
        rows = (len(views) + cols - 1) // cols
        strip = Image.new("RGB", (cols * S, rows * (S + 26) + 30 + 28), (255, 255, 255))
        dr = ImageDraw.Draw(strip)
        hdr = f"{self.scene_model} / {obj.name} ({rec['category']}/{rec['model']}, scale {rec['scale']})  aabb {np.round(ext, 2).tolist()} m  joints: " + \
            ", ".join(f"{j['joint']}={j.get('kind')}/{j['status']}" for j in joints)
        dr.text((6, 5), hdr[:230], fill=(0, 0, 0), font=font)
        for k, (title, path) in enumerate(views):
            im = Image.open(path).resize((S, S))
            x, y = (k % cols) * S, 30 + (k // cols) * (S + 26)
            strip.paste(im, (x, y + 26))
            dr.text((x + 4, y + 4), title[:60], fill=(0, 0, 0), font=font)
        dr.text((6, strip.height - 24), "markers: A/B/C/D protruding handle candidates (red/green/blue/orange), F edge fallback (yellow), X flush handle (grey)",
                fill=(0, 0, 0), font=font)
        out = os.path.join(self.out, "artic", "strips", f"{obj.name}.png")
        strip.save(out)
        return os.path.relpath(out, self.out)

    # -- (b) supports ----------------------------------------------------------------------------------------------
    def inventory_supports(self, names: Optional[Sequence[str]] = None, union: Sequence[str] = ()):
        objs = [o for o in self.scene.objects if o.category in SUPPORT_CATEGORIES] if not names else [self.world.obj(n) for n in names]
        objs = sorted({o.name: o for o in objs}.values(), key=lambda o: o.name)
        log.info("%s: %d support candidates", self.scene_model, len(objs))
        for obj in objs:
            t0 = time.time()
            try:
                rec = self._surface_record(obj)
                rec["in_union"] = obj.name in set(union)
                with open(os.path.join(self.out, "supports", f"{obj.name}.json"), "w") as f:
                    json.dump(rec, f, indent=1)
                self.summary["supports"][obj.name] = {"category": obj.category, "top_z": rec["top"]["z"], "n_free_cells": rec["top"]["n_free_cells"],
                                                       "best_edge_free_run_m": max((e["runs"]["0.45"]["longest_free_run_m"] for e in rec["edges"]), default=0.0),
                                                       "in_union": rec["in_union"], "s": round(time.time() - t0, 1)}
                log.info("support %s (%s): top z %.2f, %d free cells, best run %.2f m in %.1fs", obj.name, obj.category, rec["top"]["z"], rec["top"]["n_free_cells"],
                         self.summary["supports"][obj.name]["best_edge_free_run_m"], time.time() - t0)
            except Exception as e:
                log.error("support %s failed: %r\n%s", obj.name, e, traceback.format_exc())
                self.summary["supports"][obj.name] = {"category": obj.category, "error": repr(e)}

    def _surface_record(self, obj) -> dict:
        top = self.world.support_top(obj, resolution=0.02)
        chairs = self.chairs_near(obj)
        lo, hi = obj.aabb
        cells = top.cells(free_only=False).numpy()
        hull = convex_hull_xy(cells) if len(cells) >= 3 else cells
        umin, umax, vmin, vmax = top.local_bounds() if len(cells) else (0, 0, 0, 0)
        rec = {
            "scene": self.scene_model, "name": obj.name, "category": obj.category, "model": obj.model,
            "scale": [round(float(s), 4) for s in obj.scale] if hasattr(obj, "scale") else None, "rooms": list(obj.in_rooms or []),
            "aabb_world": {"min": lo.tolist(), "max": hi.tolist(), "extent": (hi - lo).tolist()},
            "top": {"z": round(top.z, 4), "yaw": round(top.yaw, 4), "resolution": top.resolution, "n_cells": int(top.mask.sum()), "n_free_cells": int(top.free.sum()),
                    "area_m2": round(float(top.mask.sum()) * top.resolution ** 2, 3), "local_bounds_uv": [round(float(v), 3) for v in (umin, umax, vmin, vmax)],
                    "hull_xy": np.round(hull, 3).tolist(), "occupants": top.occupants},
            "chairs_within_1m": [c.name for c in chairs],
            "edges": self.edges_with_runs(obj, top, chairs) if len(cells) else [],
        }
        return rec

    # -- (c) large partners -----------------------------------------------------------------------------------------
    def inventory_partners(self, support_records: Optional[Dict[str, dict]] = None):
        objs = sorted([o for o in self.scene.objects if o.category in PARTNER_CATEGORIES and o.fixed_base], key=lambda o: o.name)
        log.info("%s: %d large partner candidates", self.scene_model, len(objs))
        for obj in objs:
            t0 = time.time()
            try:
                rec = self._surface_record(obj)
                rec.update(self._under_clearance(obj))
                rec["nearby_supports"] = self._nearby_supports(obj, support_records or {})
                with open(os.path.join(self.out, "partners", f"{obj.name}.json"), "w") as f:
                    json.dump(rec, f, indent=1)
                uc = rec["under_clearance"]
                self.summary["partners"][obj.name] = {"category": obj.category, "top_z": rec["top"]["z"], "face_cells": rec["top"]["n_free_cells"],
                                                       "under_frac_ge_0.15": uc["fraction_ge"]["0.15"], "under_frac_ge_0.25": uc["fraction_ge"]["0.25"],
                                                       "n_supports_within_3m": sum(1 for s in rec["nearby_supports"] if (s["geodesic_m"] or 99) <= 3.0), "s": round(time.time() - t0, 1)}
                log.info("partner %s (%s): face z %.2f, under >=15cm %.0f%%, supports within 3 m: %d (%.1fs)", obj.name, obj.category, rec["top"]["z"],
                         100 * uc["fraction_ge"]["0.15"], self.summary["partners"][obj.name]["n_supports_within_3m"], time.time() - t0)
            except Exception as e:
                log.error("partner %s failed: %r\n%s", obj.name, e, traceback.format_exc())
                self.summary["partners"][obj.name] = {"category": obj.category, "error": repr(e)}

    def _under_clearance(self, obj, res: float = 0.05, shrink: float = 0.05) -> dict:
        lo, hi = obj.aabb
        fz = self.floor_z_under(obj)
        xs = th.arange(float(lo[0]) + shrink, float(hi[0]) - shrink + 1e-6, res)
        ys = th.arange(float(lo[1]) + shrink, float(hi[1]) - shrink + 1e-6, res)
        if len(xs) == 0 or len(ys) == 0:
            return {"under_clearance": {"floor_z": fz, "resolution": res, "origin": [float(lo[0]) + shrink, float(lo[1]) + shrink], "grid": [], "fraction_ge": {}}}
        gx, gy = th.meshgrid(xs, ys, indexing="ij")
        starts = th.stack([gx.flatten(), gy.flatten(), th.full_like(gx.flatten(), fz + 0.005)], dim=1)
        ends = starts.clone()
        ends[:, 2] = float(hi[2]) + 0.5
        hits = self.world.raycast(starts, ends)
        own = {l.prim_path for l in obj.links.values()}
        grid = np.full((len(xs), len(ys)), -1.0)
        for k, h in enumerate(hits):
            i, j = divmod(k, len(ys))
            if h["hit"] and h["rigidBody"] in own:
                grid[i, j] = round(float(h["position"][2]) - fz, 3)
        vals = grid[grid >= 0]
        n = grid.size
        frac = {f"{t:.2f}": round(float((vals >= t).sum()) / max(n, 1), 3) for t in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)}
        return {"under_clearance": {"floor_z": round(fz, 4), "resolution": res, "origin": [round(float(xs[0]), 4), round(float(ys[0]), 4)], "shape": list(grid.shape),
                                    "grid": np.round(grid, 3).tolist(), "fraction_hit": round(float(len(vals)) / max(n, 1), 3), "fraction_ge": frac,
                                    "max_clearance_m": round(float(vals.max()), 3) if len(vals) else 0.0}}

    def _nearby_supports(self, obj, support_records: Dict[str, dict], max_euclid: float = 3.5) -> List[dict]:
        lo, hi = obj.aabb
        out = []
        for name, rec in support_records.items():
            if name == obj.name:
                continue
            slo, shi = th.tensor(rec["aabb_world"]["min"]), th.tensor(rec["aabb_world"]["max"])
            d = th.maximum(th.maximum(lo[:2] - shi[:2], slo[:2] - hi[:2]), th.zeros(2))
            euclid = float(th.linalg.norm(d))
            if euclid > max_euclid:
                continue
            geo = None
            try:
                a = self._nearest_eroded_free(((lo + hi) / 2)[:2], ((slo + shi) / 2)[:2])
                b = self._nearest_eroded_free(((slo + shi) / 2)[:2], ((lo + hi) / 2)[:2])
                if a is not None and b is not None:
                    _, dist = self.trav.get_shortest_path(0, a, b, entire_path=False, robot=self.world.robot)
                    geo = None if dist is None else round(float(dist), 2)
            except Exception:
                geo = None
            out.append({"support": name, "category": rec["category"], "aabb_distance_m": round(euclid, 2), "geodesic_m": geo})
        return sorted(out, key=lambda r: (r["geodesic_m"] is None, r["geodesic_m"] or 0, r["aabb_distance_m"]))

    def _nearest_eroded_free(self, from_xy, toward_xy, max_d: float = 2.0):
        """First eroded-free cell walking from ``from_xy`` toward ``toward_xy`` (the partner's footprint is blocked)."""
        f = th.as_tensor([float(from_xy[0]), float(from_xy[1])])
        t = th.as_tensor([float(toward_xy[0]), float(toward_xy[1])])
        d = t - f
        n = float(th.linalg.norm(d))
        if n < 1e-6:
            return f if self.free_eroded(f) else None
        d = d / n
        for s in th.arange(0.0, max_d + 1e-6, 0.05).tolist():
            p = f + d * s
            if self.free_eroded(p):
                return p
        return None

    def write_summary(self):
        with open(os.path.join(self.out, "scene.json"), "w") as f:
            json.dump(self.summary, f, indent=1)


def _longest_run(flags: List[bool]) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def _free_range(flags: List[bool], lats: List[float]) -> Optional[List[float]]:
    xs = [l for f, l in zip(flags, lats) if f]
    return [round(min(xs), 2), round(max(xs), 2)] if xs else None


def load_support_union(path: Optional[str]) -> Dict[str, List[str]]:
    """scene -> support names that appear as spec / viable in any task of task_object_selection.csv (the 31 of the union)."""
    path = path or os.path.join(REPO_ROOT, "runlogs", "grasp_library_20260905", "select", "task_object_selection.csv")
    out: Dict[str, set] = {}
    if not os.path.exists(path):
        return {}
    cur = None
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["task"]:
                cur = (r["task"], r["role"])
                continue
            if cur and cur[1] == "table" and r["model"] and r["selection"] in ("viable", "spec"):
                scene = r["pair"].split(":")[0].strip()
                out.setdefault(scene, set()).add(r["model"])
    return {k: sorted(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, choices=SCENES)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "runlogs", "scene_inventory_20260915"))
    ap.add_argument("--parts", default="artic,supports,partners", help="comma-separated subset of artic,supports,partners,renders (renders = re-render strips from existing artic JSONs)")
    ap.add_argument("--only", nargs="*", default=None, help="restrict the artic / supports parts to these scene object names")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--log-level", default=os.environ.get("ISPATIALGYM_LOG_LEVEL", "INFO"))
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="[%(levelname)s %(name)s] %(message)s")
    parts = set(args.parts.split(","))
    inv = Inventory(args.scene, args.out, render=not args.no_render)
    t0 = time.time()
    support_records: Dict[str, dict] = {}
    if "supports" in parts:
        union = load_support_union(None).get(args.scene, [])
        inv.inventory_supports(names=args.only, union=union)
        sdir = os.path.join(inv.out, "supports")
        for fn in os.listdir(sdir):
            if fn.endswith(".json"):
                with open(os.path.join(sdir, fn)) as f:
                    support_records[fn[:-5]] = json.load(f)
    if "partners" in parts:
        inv.inventory_partners(support_records)
    if "artic" in parts:
        inv.inventory_artic(only=args.only)
    if "renders" in parts:
        inv.rerender_artic(only=args.only)
    inv.summary["total_s"] = round(time.time() - t0, 1)
    inv.write_summary()
    print(json.dumps({"scene": args.scene, "artic": len(inv.summary["artic"]), "supports": len(inv.summary["supports"]), "partners": len(inv.summary["partners"]),
                      "load_s": inv.summary["load_s"], "total_s": inv.summary["total_s"], "out": inv.out}), flush=True)
    import omnigibson as og

    og.shutdown()


if __name__ == "__main__":
    main()
