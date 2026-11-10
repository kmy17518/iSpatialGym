"""i7-sink 2026-11-09 (owner A3, plate-in-sink): does a plate lie flat in a sink basin? Geometry + the goal check's drop, measured live.

A Kit job (one scene per run); writes ``--out`` (JSON: measured numbers only, no asset data).

Sink (``--res`` raster, default 5 mm, over the sink's AABB; every body but the robot counts - the counter around a drop-in sink too):
  * the basin FLOOR: downward rays from above the AABB; among the hits on the sink, the LOWEST 2 mm height bin holding at least
    ``max(30, 2 %)`` of them (a drain hole is smaller; the counter flange / faucet are higher);
  * the OCCUPANCY at height ``h`` above the floor (``HEIGHTS``, 2 mm steps to 4 cm, then 6 / 8 / 10 / 15 cm): a downward ray starting at
    ``floor + h`` meets something above ``floor + 3 mm`` (a ray that starts inside a wall hits at ~0 m) - rays below the faucet / the
    rim, so an arching spout is no wall; ``surf_h`` per cell = the HIGHEST occupied ``h`` (0: the floor itself; a wall's material
    fills its column from below, so the plate's underside at that cell must be above it);
  * the flat floor's inscribed radius (the free cells at ``h`` = 4 mm) and the basin's half width ``half_width(h)`` about that centre.
Plate (spawned and parked: gravity off, nothing under it): an upward 2 mm raster of its underside -> the outline radius ``R`` about the
hits' centroid and the underside height above its lowest point ``u(r)`` (minimum per 2 mm radius bin).
Fit (level plate, foot on the floor): for each centre on the floor cells, every cell within ``R`` must have ``surf_h <= u(r)``;
``margin_mm`` = the best centre's least ``u(r) - surf_h`` (2 mm resolution; negative: a wall / the flare under the rim).
Drops = the goal check (``SamplerV2.check_goal_feasible``: teleport to the release, ``world.settle`` 120 + 10 physics steps, the up axis
vs the level orientation <= 15 deg): at the fit centre (+-2 cm), level and tilted 30 / 45 deg about a horizontal axis (that rim down) in
four directions, the lowest point ``--drop`` above the floor, raised in 5 mm steps until no hull point starts inside a body; the tilt after
130 steps and after 480, the rest height of the root above the floor and the xy shift.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

HEIGHTS = [0.002 * k for k in range(1, 21)] + [0.06, 0.08, 0.10]
CEILING_M = 0.15
BASIN_LINK_M = 0.03


def _components(mask):
    """4-connected labels of a boolean raster (0 = background)."""
    lab = np.zeros(mask.shape, dtype=int)
    n = 0
    for i0, j0 in zip(*np.nonzero(mask)):
        if lab[i0, j0]:
            continue
        n += 1
        stack = [(i0, j0)]
        lab[i0, j0] = n
        while stack:
            i, j = stack.pop()
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < mask.shape[0] and 0 <= b < mask.shape[1] and mask[a, b] and not lab[a, b]:
                    lab[a, b] = n
                    stack.append((a, b))
    return lab


def _bins_min(r, u, dr=0.002):
    nb = int(math.ceil(float(r.max()) / dr)) + 1
    out = np.full(nb, np.nan)
    idx = np.minimum((r / dr).astype(int), nb - 1)
    for i, v in zip(idx, u):
        if np.isnan(out[i]) or v < out[i]:
            out[i] = v
    last = np.nan
    for i in range(nb):  # a bin without a hit: the previous bin's value (inner rings first)
        if np.isnan(out[i]):
            out[i] = last if not np.isnan(last) else 0.0
        last = out[i]
    return out


def fit_plate(sink, plate):
    """(margin m, centre xy) of the best centre for the level plate (foot on the floor)."""
    xs, ys, wh = np.array(sink["xs"]), np.array(sink["ys"]), np.array(sink["surf_h"], dtype=float)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    R, u, dr = plate["R"], np.array(plate["u"]), plate["dr"]
    floor = wh < 0.004
    best = (-9.0, None)
    for cx, cy in zip(gx[floor], gy[floor]):
        d = np.hypot(gx - cx, gy - cy)
        m = d <= R
        ui = u[np.minimum((d[m] / dr).astype(int), len(u) - 1)]
        marg = float((ui - wh[m]).min())
        if marg > best[0]:
            best = (marg, (float(cx), float(cy)))
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--rooms", nargs="*", default=None)
    ap.add_argument("--sinks", nargs="+", required=True)
    ap.add_argument("--plates", nargs="*", default=[], help="category/model[@scale] (scale: one float)")
    ap.add_argument("--res", type=float, default=0.005)
    ap.add_argument("--drop", type=float, default=0.02)
    ap.add_argument("--tilts", nargs="*", type=float, default=[0.0, 30.0, 45.0])
    ap.add_argument("--drops-for", nargs="*", default=None, help="plate specs dropped (default every plate); the others only fitted")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import torch as th
    from omnigibson.utils.sampling_utils import raytest_batch

    from ispatialgym.sampling import geometry as G
    from ispatialgym.sampling.sim_utils import SimWorld

    t0 = time.time()
    world = SimWorld(a.scene, a.rooms, cameras=False)
    robot_paths = [str(lk.prim_path) for lk in world.robot.links.values()]
    out = {"scene": a.scene, "rooms": a.rooms, "res": a.res, "drop": a.drop, "heights": HEIGHTS, "sinks": {}, "plates": {}, "fits": [], "drops": []}

    def rays(st, en, ignore):
        hits = []
        for i in range(0, len(st), 4096):
            hits += raytest_batch(st[i:i + 4096], en[i:i + 4096], only_closest=True, ignore_bodies=ignore)
        return hits

    # ---- sinks
    for sn in a.sinks:
        s = world.obj(sn)
        lo, hi = [t.tolist() for t in s.aabb]
        xs = np.arange(lo[0] + a.res / 2, hi[0], a.res)
        ys = np.arange(lo[1] + a.res / 2, hi[1], a.res)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        X, Y = gx.ravel(), gy.ravel()
        paths = tuple(str(lk.prim_path) for lk in s.links.values())
        hits = rays([[float(x), float(y), hi[2] + 0.05] for x, y in zip(X, Y)], [[float(x), float(y), lo[2] - 0.05] for x, y in zip(X, Y)], robot_paths)
        ztop = np.array([float(h["position"][2]) if h.get("hit") else np.nan for h in hits])
        on = np.array([bool(h.get("hit")) and str(h.get("rigidBody", "")).startswith(paths) for h in hits])
        zs = np.sort(ztop[on])
        if len(zs) == 0:
            out["sinks"][sn] = {"error": "no hit on the sink"}
            continue
        edges = np.arange(zs[0], zs[-1] + 0.004, 0.002)
        hist, _ = np.histogram(zs, bins=edges)
        need = max(30, int(0.02 * len(zs)))
        k = int(np.nonzero(hist >= need)[0][0])
        sel = zs[(zs >= edges[k]) & (zs < edges[k + 1])]
        zf = float(np.median(sel))
        wall_h = np.zeros(len(X))
        for h in HEIGHTS:  # the highest occupied height wins
            P = [[float(x), float(y), zf + h] for x, y in zip(X, Y)]
            hh = rays(P, [[float(x), float(y), zf - 0.1] for x, y in zip(X, Y)], robot_paths)
            # ... or a ceiling within CEILING_M over that point (under the counter beside a drop-in basin, a lip over the wall); a faucet
            # spout over the basin is higher
            hu = rays(P, [[float(x), float(y), zf + CEILING_M] for x, y in zip(X, Y)], robot_paths)
            occ = np.array([(bool(r.get("hit")) and float(r["position"][2]) > zf + 0.003) or bool(q.get("hit")) for r, q in zip(hh, hu)])
            wall_h[occ] = h
        # the basin: the connected region under BASIN_LINK_M (4-neighbours) holding the most cells open from above onto the floor
        open_floor = (np.abs(ztop - zf) <= 0.003).reshape(gx.shape)
        wall_h = wall_h.reshape(gx.shape)
        low = wall_h < BASIN_LINK_M  # the basin region: floor + its walls' foot (a drain ring does not split it)
        lab = _components(low)
        if lab.max() > 0:
            counts = np.bincount(lab[open_floor & (lab > 0)].ravel(), minlength=lab.max() + 1)
            keep = int(counts.argmax()) if counts.max() > 0 else 0
            wall_h[(lab != keep) & low] = HEIGHTS[-1]  # low cells outside the basin (in the cabinet below the counter) count as walls
        free = wall_h < 0.004
        bx, by = gx[~free], gy[~free]
        best = (0.0, None)
        for cx, cy in zip(gx[free], gy[free]):
            r = float(np.hypot(bx - cx, by - cy).min()) if len(bx) else 9.0
            if r > best[0]:
                best = (r, (float(cx), float(cy)))
        hw = {}
        if best[1] is not None:
            d = np.hypot(gx - best[1][0], gy - best[1][1])
            for h in HEIGHTS:
                blocked = wall_h > h + 1e-9
                hw[f"{h:.3f}"] = round(float(d[blocked].min()), 4) if blocked.any() else None
        rec = {"aabb": [lo, hi], "xs": xs.tolist(), "ys": ys.tolist(), "surf_h": [[round(float(v), 4) for v in row] for row in wall_h],
               "floor_z": round(zf, 4), "top_z_max_on_sink": round(float(zs[-1]), 4), "free_floor_cells": int(free.sum()), "flat_inscribed_r": round(best[0], 4),
               "flat_centre": best[1], "half_width_at_h": hw,
               "free_extent_xy": [round(float(free.any(axis=1).sum() * a.res), 4), round(float(free.any(axis=0).sum() * a.res), 4)]}
        out["sinks"][sn] = rec
        print(f"SINK {sn} floor {zf:.4f} free cells {rec['free_floor_cells']} inscribed r {rec['flat_inscribed_r']:.4f} at {rec['flat_centre']} "
              f"extent {rec['free_extent_xy']} half_width(h) " + " ".join(f"{k}:{v}" for k, v in hw.items() if k in ("0.004", "0.010", "0.020", "0.030", "0.060", "0.100")),
              flush=True)

    # ---- plates
    plates = {}
    for i, spec in enumerate(a.plates):
        cm, _, sc = spec.partition("@")
        cat, model = cm.split("/")
        scale = [float(sc)] * 3 if sc else None
        name = f"probe_{cat}_{model}_{i}"
        try:
            o = world.spawn(name, cat, model, scale=scale)
        except Exception as e:  # noqa: BLE001
            print(f"PLATE {spec} spawn failed: {e!r}", flush=True)
            continue
        world.step_physics(2, still=[o])
        world.sync_render_transforms()
        lo, hi = [t.tolist() for t in o.aabb]
        xs = np.arange(lo[0] - 0.01, hi[0] + 0.01, 0.002)
        ys = np.arange(lo[1] - 0.01, hi[1] + 0.01, 0.002)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        X, Y = gx.ravel(), gy.ravel()
        paths = tuple(str(lk.prim_path) for lk in o.links.values())
        hh = rays([[float(x), float(y), lo[2] - 0.2] for x, y in zip(X, Y)], [[float(x), float(y), hi[2] + 0.05] for x, y in zip(X, Y)], robot_paths)
        on = np.array([bool(h.get("hit")) and str(h.get("rigidBody", "")).startswith(paths) for h in hh])
        z = np.array([float(h["position"][2]) if h.get("hit") else np.nan for h in hh])
        cx, cy = float(X[on].mean()), float(Y[on].mean())
        r = np.hypot(X[on] - cx, Y[on] - cy)
        u = z[on] - float(z[on].min())
        prof = _bins_min(r, u)
        rec = {"spec": spec, "R": round(float(r.max()), 4), "height": round(float(hi[2] - lo[2]), 4), "dr": 0.002, "u": [round(float(v), 4) for v in prof],
               "foot_r": round(float(r[u <= 0.001].max()), 4), "rim_u": round(float(prof.max()), 4)}
        plates[spec] = (o, i)
        out["plates"][spec] = rec
        print(f"PLATE {spec} R {rec['R']:.4f} height {rec['height']:.4f} foot_r {rec['foot_r']:.4f} rim_u {rec['rim_u']:.4f} "
              f"u(0.9R) {prof[int(0.9 * rec['R'] / 0.002)]:.4f}", flush=True)

    # ---- fits and drops
    up = th.tensor([0.0, 0.0, 1.0])
    fits = {}
    for sn, srec in out["sinks"].items():  # every fit first (cheap), then the drops
        if "error" in srec:
            continue
        for spec in plates:
            marg, c = fit_plate(srec, out["plates"][spec])
            fits[(sn, spec)] = (marg, c)
            out["fits"].append({"sink": sn, "plate": spec, "margin_mm": round(marg * 1000, 1), "centre": c, "fits_flat": bool(marg > 0.0)})
            print(f"FIT {sn} {spec} R {out['plates'][spec]['R']:.4f} margin {marg * 1000:+.1f} mm centre {c} -> {'FLAT' if marg > 0 else 'BLOCKED'}", flush=True)
    for sn, srec in out["sinks"].items():
        if "error" in srec:
            continue
        s = world.obj(sn)
        zf = srec["floor_z"]
        for spec, (o, idx) in plates.items():
            marg, c = fits[(sn, spec)]
            if c is None or (a.drops_for is not None and spec not in a.drops_for):
                continue
            hull = world.local_hull(o)
            sub = hull[:: max(1, len(hull) // 300)]
            plate_paths = [str(lk.prim_path) for lk in o.links.values()]
            for dx, dy in ((0.0, 0.0), (0.02, 0.0), (-0.02, 0.0), (0.0, 0.02), (0.0, -0.02)):
                for tilt in a.tilts:
                    if (dx, dy) != (0.0, 0.0) and tilt != 0.0:
                        continue  # the offsets only for the level release
                    for ux, uy in ([(1.0, 0.0)] if tilt == 0.0 else [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]):
                        axis = th.tensor([-uy, ux, 0.0])  # the rim toward (ux, uy) goes down
                        quat = G.quat_about_axis(axis, math.radians(tilt)) if tilt else th.tensor([0.0, 0.0, 0.0, 1.0])
                        pts = G.local_points_world(sub, (th.zeros(3), quat))
                        base = zf - float(pts[:, 2].min()) + a.drop
                        cx, cy = c[0] + dx, c[1] + dy
                        lift = 0.0
                        while lift < 0.15:  # no hull point may start inside a body (the plate is parked meanwhile)
                            P = [[cx + float(p[0]), cy + float(p[1]), base + lift + float(p[2])] for p in pts]
                            hh = rays(P, [[p[0], p[1], p[2] - 0.003] for p in P], robot_paths + plate_paths)
                            if not any(h.get("hit") and float(h.get("distance", 1.0)) < 5e-4 for h in hh):
                                break
                            lift += 0.005
                        o.set_position_orientation(th.tensor([cx, cy, base + lift]), quat)
                        world.activate(o)
                        world.settle([o], steps=120, still=[s])
                        _, q1 = o.get_position_orientation()
                        t130 = math.degrees(math.acos(float(th.clamp(G.rotate(q1.float(), up)[2], -1, 1))))
                        world.step_physics(350, still=[s, world.robot])
                        p2, q2 = o.get_position_orientation()
                        t480 = math.degrees(math.acos(float(th.clamp(G.rotate(q2.float(), up)[2], -1, 1))))
                        d = {"sink": sn, "plate": spec, "offset": [dx, dy], "tilt": tilt, "dir": [ux, uy], "lift_mm": round(lift * 1000, 1),
                             "tilt_130": round(t130, 2), "tilt_480": round(t480, 2), "root_above_floor": round(float(p2[2]) - zf, 4),
                             "xy_shift": [round(float(p2[0]) - cx, 4), round(float(p2[1]) - cy, 4)], "ok_15": bool(t130 <= 15.0)}
                        out["drops"].append(d)
                        print(f"DROP {sn} {spec} off {dx:+.2f},{dy:+.2f} tilt {tilt:.0f} dir {ux:+.0f},{uy:+.0f} lift {d['lift_mm']} mm -> "
                              f"{t130:.1f} deg (130) {t480:.1f} deg (480) root {d['root_above_floor']:.4f} shift {d['xy_shift']}", flush=True)
                        world.park(o, idx)
            n = [d for d in out["drops"] if d["sink"] == sn and d["plate"] == spec]
            print(f"DROPS {sn} {spec}: {sum(d['ok_15'] for d in n)}/{len(n)} within 15 deg; max {max(d['tilt_130'] for d in n):.1f} "
                  f"tilted(30/45) {sum(d['ok_15'] for d in n if d['tilt'] > 0)}/{sum(1 for d in n if d['tilt'] > 0)}", flush=True)
    out["elapsed_s"] = round(time.time() - t0, 1)
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"BASIN_FIT_DONE {a.out} {out['elapsed_s']} s", flush=True)
    import omnigibson as og

    og.shutdown()


if __name__ == "__main__":
    main()
