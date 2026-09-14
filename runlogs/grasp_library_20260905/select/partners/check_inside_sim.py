#!/usr/bin/env python3
"""Check in OmniGibson that ``Inside`` evaluates true for the IORS-2 partner containers listed in the sheet.

For every listed container model x every test pick (pick 1 / pick 2 / alternate) of its paired target categories the test
reproduces the sampler's goal (``IORSSampler._goal_pose`` for ``inside``): container upright on the ground, target in its
grasp-library rest orientation (scaled variants at their scale) centred on the container's ``fillable`` meta-link, its
lowest point 5 mm above the container top, then dropped and settled (>= 60 physics steps, until it stops moving, max 300).
Recorded per pair: ``target.states[Inside].get_value(container)`` after settling (cache cleared first), ``Touching``,
the target centre height relative to the container top, container displacement and whether the target came to rest.
Flowers are tested twice: lying (rest orientation, what the sampler does today) and standing on the stem end.

Usage (ispatialgym conda env, headless, ~4 GB GPU):  python check_inside_sim.py [--out inside_sim.json] [--only bowl,vase]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import numpy as np
import torch as th
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.abspath(os.path.join(HERE, "..", ".."))
FIT = os.path.join(HERE, "partner_fit.json")
TASK = "inter_object_relation_satisfaction-2"
FLOWERS = {"sunflower", "lily", "dahlia_flower"}
PICK_SEL = ("pick 1", "pick 2", "alternate")
TEST_XY = (0.0, 0.0)
PARK_Z = 0.5


def rest_quat_xyzw(key):
    y = yaml.safe_load(open(os.path.join(LIB, "cands", f"{key}.yaml")))
    w, x, y_, z = y["ispatialgym"]["rest_quat_wxyz"]
    return th.tensor([x, y_, z, w], dtype=th.float32)


def quat_from_vectors(a, b):
    """Quaternion (xyzw) rotating unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = float(np.dot(a, b))
    if c > 1 - 1e-9:
        return th.tensor([0.0, 0.0, 0.0, 1.0])
    if c < -1 + 1e-9:
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        return th.tensor([axis[0], axis[1], axis[2], 0.0], dtype=th.float32)
    v = np.cross(a, b)
    s = math.sqrt((1 + c) * 2)
    q = np.array([v[0] / s, v[1] / s, v[2] / s, s / 2])
    return th.tensor(q / np.linalg.norm(q), dtype=th.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "inside_sim.json"))
    ap.add_argument("--only", default="", help="comma-separated container categories")
    ap.add_argument("--max-partial", type=int, default=10, help="also test this many PARTIAL vases (standing only)")
    ap.add_argument("--missing-only", action="store_true", help="keep the results already in --out and test only the missing pairs")
    args = ap.parse_args()
    fit = json.load(open(FIT))
    only = {c for c in args.only.split(",") if c}
    previous = json.load(open(args.out)) if (args.missing_only and os.path.exists(args.out)) else {"results": [], "containers": {}}
    done = {(r["container"], r["target"], r["mode"]) for r in previous["results"] if "error" not in r}

    # ---------------------------------------------------------------- test plan
    plan = []  # (container cat, container model, target cat, target model, key, scale, mode)
    containers, targets = {}, {}
    for pcat, d in fit["tasks"][TASK].items():
        if only and pcat not in only:
            continue
        listed = [m for m in d["models"] if m["verdict"] == "ok"][:12]
        extra = [m for m in d["models"] if m["verdict"] == "partial"][: args.max_partial] if pcat == "vase" else []
        picks = [t for t in fit["targets"][TASK] if t["selection"] in PICK_SEL and t["category"] in d["targets"]]
        for m in listed + extra:
            containers[(pcat, m["model"])] = None
            for t in picks:
                targets[(t["category"], t["model"])] = t
                modes = ["lying", "standing"] if t["category"] in FLOWERS else ["rest"]
                if m in extra:
                    modes = ["standing"]
                for mode in modes:
                    if (f"{pcat}/{m['model']}", f"{t['category']}/{t['model']}", mode) in done:
                        continue
                    plan.append((pcat, m["model"], t["category"], t["model"], t["key"], t["scale"], mode, m["verdict"]))
    # only load what the plan needs
    containers = {k: None for k in containers if any(p[0] == k[0] and p[1] == k[1] for p in plan)}
    targets = {k: v for k, v in targets.items() if any(p[2] == k[0] and p[3] == k[1] for p in plan)}
    print(f"{len(plan)} tests: {len(containers)} containers x {len(targets)} targets" + (f" ({len(done)} pairs kept from {args.out})" if done else ""), flush=True)
    if not plan:
        print("nothing to do")
        return

    # ---------------------------------------------------------------- simulator
    import omnigibson as og
    import omnigibson.utils.transform_utils as T
    from omnigibson.macros import macros
    from omnigibson.object_states import Inside, Touching
    from omnigibson.objects.dataset_object import DatasetObject

    env = og.Environment(configs={"scene": {"type": "Scene"}})
    scene = env.scene
    for _ in range(3):
        og.sim.step()
    objs = {}
    park = {}
    i = 0

    def load(cat, model, scale=None):
        nonlocal i
        kw = {"scale": th.tensor(scale, dtype=th.float32)} if scale and any(abs(s - 1) > 1e-6 for s in scale) else {}
        o = DatasetObject(name=f"o_{cat}_{model}", category=cat, model=model, fixed_base=False, **kw)
        scene.add_object(o)
        p = th.tensor([4.0 + 0.8 * (i % 12), 4.0 + 0.8 * (i // 12), PARK_Z])
        i += 1
        o.set_position_orientation(p, th.tensor([0.0, 0.0, 0.0, 1.0]))
        o.disable_gravity()
        o.keep_still()
        og.sim.step()
        park[o.name] = p
        return o

    t0 = time.time()
    for (cat, model) in containers:
        objs[("c", cat, model)] = load(cat, model)
    for (cat, model), t in targets.items():
        objs[("t", cat, model)] = load(cat, model, t["scale"])
    print(f"loaded {len(objs)} objects in {time.time() - t0:.0f} s", flush=True)

    def to_park(o):
        o.disable_gravity()
        o.set_position_orientation(park[o.name], th.tensor([0.0, 0.0, 0.0, 1.0]))
        o.keep_still()

    def full_step():
        """A rendering step: physics -> USD/Fabric sync refreshes the per-step caches (AABBs, contacts) that
        og.sim.step_physics() leaves stale (see ispatialgym.sampling.sim_utils.SimWorld.full_step)."""
        og.sim.step()

    def aabb_after_pose(o, pos, quat):
        o.set_position_orientation(pos, quat)
        o.keep_still()
        full_step()
        o.keep_still()
        return o.aabb

    def container_volume_centre(o):
        lo, hi = o.aabb
        centre = (lo + hi) / 2
        for link in o.links.values():
            if link.is_meta_link and link.meta_link_type in macros.object_states.contains.CONTAINER_META_LINK_TYPES:
                p, _ = link.get_position_orientation()
                return th.tensor([float(p[0]), float(p[1])]), float(hi[2]), True
        return centre[:2], float(hi[2]), False

    def cavity_diagnostic(o, centre):
        """Along the vertical line through the fillable centre: how many sample heights are inside the fillable volume and
        how many are inside a COLLISION mesh (a convex hull that fills the cavity makes Inside physically unreachable)."""
        lo, hi = o.aabb
        zs = th.linspace(float(lo[2]) + 0.005, float(hi[2]) - 0.005, 12)
        pts = th.stack([th.full_like(zs, float(centre[0])), th.full_like(zs, float(centre[1])), zs], dim=1)
        in_fill = th.zeros(len(zs), dtype=th.bool)
        in_coll = th.zeros(len(zs), dtype=th.bool)
        n_coll = 0
        for link in o.links.values():
            if link.is_meta_link:
                if link.meta_link_type in macros.object_states.contains.CONTAINER_META_LINK_TYPES:
                    in_fill |= link.check_points_in_volume(pts)
                continue
            for m in link.collision_meshes.values():
                n_coll += 1
                try:
                    in_coll |= m.check_points_in_volume(pts)
                except Exception:  # noqa: BLE001
                    pass
        return {"n_collision_meshes": n_coll, "centre_line_in_fillable": int(in_fill.sum()), "centre_line_in_collision": int(in_coll.sum()),
                "centre_line_samples": len(zs), "fillable_and_collision": int((in_fill & in_coll).sum())}

    def standing_quat(o):
        """Flower: long canonical axis vertical with the head (the end with the larger perpendicular spread) up."""
        pts = []
        for link in o.links.values():
            p = link.collision_boundary_points_world
            if p is not None and len(p):
                pts.append(p)
        pts = th.cat(pts).numpy()
        rp, rq = o.get_position_orientation()
        R = T.quat2mat(rq).numpy()
        v = (pts - rp.numpy()) @ R  # canonical frame
        lo, hi = v.min(0), v.max(0)
        ext = hi - lo
        ax = int(np.argmax(ext))
        t = (v[:, ax] - lo[ax]) / ext[ax]
        perp = np.delete(v, ax, axis=1)
        sp = lambda m: float(np.ptp(perp[m], axis=0).max()) if m.sum() > 3 else 0.0  # noqa: E731
        head_hi = sp(t > 0.7) >= sp(t < 0.3)
        up = np.zeros(3)
        up[ax] = 1.0 if head_hi else -1.0  # canonical direction stem -> head
        return quat_from_vectors(up, np.array([0.0, 0.0, 1.0]))

    results = list(previous["results"])
    diagnostics = dict(previous.get("containers", {}))
    t_all = time.time()
    for n, (pcat, pm, tcat, tm, key, scale, mode, pverdict) in enumerate(plan):
        c = objs[("c", pcat, pm)]
        t = objs[("t", tcat, tm)]
        rec = {"container": f"{pcat}/{pm}", "container_verdict": pverdict, "target": f"{tcat}/{tm}", "key": key, "scale": scale, "mode": mode}
        try:
            # container upright on the ground at the test spot
            lo, hi = aabb_after_pose(c, th.tensor([TEST_XY[0], TEST_XY[1], 0.5]), th.tensor([0.0, 0.0, 0.0, 1.0]))
            cz = 0.5 - float(lo[2]) + 0.002
            c.set_position_orientation(th.tensor([TEST_XY[0], TEST_XY[1], cz]), th.tensor([0.0, 0.0, 0.0, 1.0]))
            c.keep_still()
            c.enable_gravity()
            for _ in range(40):
                og.sim.step_physics()
            c.keep_still()
            full_step()
            c_pos0, _ = c.get_position_orientation()
            centre, top, has_fill = container_volume_centre(c)
            rec["container_has_fillable"] = has_fill
            rec["container_top_z"] = round(top, 4)
            if rec["container"] not in diagnostics:
                diagnostics[rec["container"]] = cavity_diagnostic(c, centre)
                print(f"    {rec['container']}: {diagnostics[rec['container']]}", flush=True)
            # target pose: rest orientation (or standing for flowers), lowest point 5 mm above the container top
            quat = rest_quat_xyzw(key)
            if mode == "standing":
                t.set_position_orientation(th.tensor([2.0, 2.0, 1.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
                t.keep_still()
                full_step()
                quat = standing_quat(t)
            tlo, thi = aabb_after_pose(t, th.tensor([2.0, 2.0, 1.0]), quat)
            tpos, _ = t.get_position_orientation()
            off = float(tpos[2] - tlo[2])
            rec["target_extent"] = [round(float(x), 4) for x in (thi - tlo)]
            start = th.tensor([float(centre[0]), float(centre[1]), top + 0.005 + off])
            t.set_position_orientation(start, quat)
            t.keep_still()
            t.enable_gravity()
            settled = False
            for step in range(300):
                og.sim.step_physics()
                if step >= 60 and step % 10 == 0:
                    if float(th.linalg.norm(t.get_linear_velocity())) < 5e-3 and float(th.linalg.norm(t.get_angular_velocity())) < 5e-2:
                        settled = True
                        break
            rec["settled"] = settled
            rec["settle_steps"] = step + 1
            rec["final_lin_vel"] = round(float(th.linalg.norm(t.get_linear_velocity())), 4)
            t.wake()
            c.wake()
            full_step()  # refresh AABBs / contacts before evaluating the predicates
            t.states[Inside].clear_cache()
            t.states[Touching].clear_cache()
            inside = bool(t.states[Inside].get_value(c))
            touching = bool(t.states[Touching].get_value(c))
            tlo2, thi2 = t.aabb
            clo2, chi2 = c.aabb
            c_pos1, _ = c.get_position_orientation()
            rec.update({
                "inside": inside, "touching": touching,
                "target_centre_below_container_top": round(float(chi2[2] - (tlo2[2] + thi2[2]) / 2), 4),
                "target_bottom_above_ground": round(float(tlo2[2]), 4),
                "target_xy_offset": round(float(th.linalg.norm(((tlo2 + thi2) / 2)[:2] - c_pos1[:2])), 4),
                "container_moved": round(float(th.linalg.norm(c_pos1 - c_pos0)), 4),
            })
        except Exception as e:  # noqa: BLE001
            rec["error"] = repr(e)
        finally:
            to_park(t)
            to_park(c)
            full_step()
        results.append(rec)
        flag = "OK " if rec.get("inside") else ("ERR" if "error" in rec else "no ")
        print(f"[{n + 1}/{len(plan)}] {flag} {rec['container']:26s} <- {rec['target']:28s} {mode:8s} inside={rec.get('inside')} touching={rec.get('touching')} "
              f"centre_below_top={rec.get('target_centre_below_container_top')} settled={rec.get('settled')} moved={rec.get('container_moved')}", flush=True)
        if (n + 1) % 20 == 0:
            json.dump({"n": len(results), "containers": diagnostics, "results": results}, open(args.out, "w"), indent=1)
    json.dump({"n": len(results), "elapsed_s": round(time.time() - t_all, 1), "containers": diagnostics, "results": results}, open(args.out, "w"), indent=1)
    ok = sum(1 for r in results if r.get("inside"))
    print(f"DONE {ok}/{len(results)} inside=True in {time.time() - t_all:.0f} s -> {args.out}", flush=True)
    og.shutdown()


if __name__ == "__main__":
    sys.exit(main())
