"""i7-sink 2026-11-09 (owner A3, plate-in-sink): does a plate lie flat in a sink basin? Geometry + the goal check's drop, measured live.

A Kit job (one scene per run). Per sink: a vertical ray raster (``--res``, default 5 mm) over the sink's AABB - every body counts (the
faucet, the counter around the sink); the basin floor = the most common hit height in the lower half of the AABB (2 mm bins). Per plate
(spawned and parked: gravity off, no floor under it): an upward ray raster (2 mm) of its underside -> its outline radius ``R`` and the
underside height above its lowest point ``u(r)`` (the minimum per 2 mm radius bin). Fit: for a centre ``c`` (5 mm lattice over the
basin's floor cells) the plate lies flat when every sink cell within ``R`` of ``c`` is at most ``u(r)`` above the floor; ``margin_mm`` =
the best centre's least clearance (negative: a wall / the rim is in the way of the level plate). Profile ``half_width(h)``: the
basin's inscribed radius about the best centre of the region at most ``h`` above the floor.

The goal check's drop (``SamplerV2.check_goal_feasible``: teleport to the release, ``world.settle`` 120 + 10 physics steps, up axis vs the
level orientation <= 15 deg), repeated here per (plate, sink): at the fit centre (and +-2 cm), level and tilted 30 / 45 deg about a
horizontal axis (far rim down) in four directions, the lowest point ``--drop`` above the floor (raised clear of the raster where the tilted
hull would sit in a wall); the tilt after 130 steps and after 480. Writes ``--out`` (JSON): no asset data, only the measured numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np


def _bins_min(r, u, dr=0.002):
    nb = int(math.ceil(float(r.max()) / dr)) + 1
    out = np.full(nb, np.nan)
    idx = np.minimum((r / dr).astype(int), nb - 1)
    for i, v in zip(idx, u):
        if np.isnan(out[i]) or v < out[i]:
            out[i] = v
    # a bin without a hit (none inside the first ring at the centre): the next bin's value
    for i in range(nb - 2, -1, -1):
        if np.isnan(out[i]):
            out[i] = out[i + 1]
    return out


def fit_plate(sink, plate, tol=0.001):
    """Best centre for the level plate in the sink raster: (margin m, centre xy)."""
    xs, ys, z, zf = np.array(sink["xs"]), np.array(sink["ys"]), np.array(sink["z"], dtype=float), sink["floor_z"]
    zrel = z - zf
    zrel[np.isnan(zrel)] = -1.0  # nothing under the cell: a hole
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    R, u, dr = plate["R"], np.array(plate["u"]), plate["dr"]
    floor = np.abs(zrel) <= 0.003
    cand = np.stack([gx[floor], gy[floor]], 1)
    best = (-9.0, None)
    for c in cand:
        d = np.hypot(gx - c[0], gy - c[1])
        m = d <= R
        if not m.any():
            continue
        ui = u[np.minimum((d[m] / dr).astype(int), len(u) - 1)]
        marg = float((ui - zrel[m]).min())
        if marg > best[0]:
            best = (marg, (float(c[0]), float(c[1])))
    return best


def half_width_profile(sink, c, hs=(0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08)):
    xs, ys, z, zf = np.array(sink["xs"]), np.array(sink["ys"]), np.array(sink["z"], dtype=float), sink["floor_z"]
    zrel = z - zf
    zrel[np.isnan(zrel)] = -1.0
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    d = np.hypot(gx - c[0], gy - c[1])
    out = {}
    for h in hs:
        bad = zrel > h + 0.003
        out[f"{h:.3f}"] = round(float(d[bad].min()), 4) if bad.any() else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--rooms", nargs="*", default=None)
    ap.add_argument("--sinks", nargs="+", required=True)
    ap.add_argument("--plates", nargs="*", default=[], help="category/model[@scale] (scale: one float)")
    ap.add_argument("--res", type=float, default=0.005)
    ap.add_argument("--drop", type=float, default=0.02)
    ap.add_argument("--tilts", nargs="*", type=float, default=[0.0, 30.0, 45.0])
    ap.add_argument("--no-drops", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import torch as th

    from ispatialgym.sampling import geometry as G
    from ispatialgym.sampling.sim_utils import SimWorld

    t0 = time.time()
    world = SimWorld(a.scene, a.rooms, cameras=False)
    robot_paths = [str(lk.prim_path) for lk in world.robot.links.values()]
    out = {"scene": a.scene, "rooms": a.rooms, "res": a.res, "drop": a.drop, "sinks": {}, "plates": {}, "fits": [], "drops": []}

    def ray_grid(x0, x1, y0, y1, z_from, z_to, res, ignore=None):
        xs = np.arange(x0 + res / 2, x1, res)
        ys = np.arange(y0 + res / 2, y1, res)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        st = [[float(x), float(y), z_from] for x, y in zip(gx.ravel(), gy.ravel())]
        en = [[float(x), float(y), z_to] for x, y in zip(gx.ravel(), gy.ravel())]
        from omnigibson.utils.sampling_utils import raytest_batch

        hits = []
        for i in range(0, len(st), 4096):
            hits += raytest_batch(st[i:i + 4096], en[i:i + 4096], only_closest=True, ignore_bodies=ignore)
        z = np.full(gx.shape, np.nan).ravel()
        body = [""] * len(z)
        for i, h in enumerate(hits):
            if h.get("hit"):
                z[i] = float(h["position"][2])
                body[i] = str(h.get("rigidBody", ""))
        return xs, ys, z.reshape(gx.shape), body

    # ---- sinks
    for sn in a.sinks:
        s = world.obj(sn)
        lo, hi = [t.tolist() for t in s.aabb]
        xs, ys, z, body = ray_grid(lo[0] - 0.05, hi[0] + 0.05, lo[1] - 0.05, hi[1] + 0.05, hi[2] + 0.3, lo[2] - 0.3, a.res, robot_paths)
        paths = tuple(str(lk.prim_path) for lk in s.links.values())
        on_sink = np.array([b.startswith(paths) for b in body]).reshape(z.shape)
        low = z[(z < 0.5 * (lo[2] + hi[2])) & on_sink & ~np.isnan(z)]
        if len(low) == 0:
            out["sinks"][sn] = {"error": "no hit in the lower half of the AABB"}
            continue
        hist, edges = np.histogram(low, bins=np.arange(low.min(), low.max() + 0.004, 0.002))
        zf = float(edges[int(hist.argmax())] + 0.001)
        rec = {"aabb": [lo, hi], "xs": xs.tolist(), "ys": ys.tolist(), "z": [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in z],
               "floor_z": round(zf, 4), "floor_cells": int((np.abs(z - zf) <= 0.003).sum()), "res": a.res,
               "rim_z": round(float(np.nanmax(np.where(on_sink, z, np.nan))), 4)}
        # the flat floor's inscribed circle (no plate): the centre with the farthest cell above the floor + 3 mm
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        fl = np.abs(z - zf) <= 0.003
        hole = np.isnan(z) | (z < zf - 0.003)
        bad = ~(fl | hole)
        bx, by = gx[bad], gy[bad]
        best = (0.0, None)
        for cx, cy in zip(gx[fl], gy[fl]):
            r = float(np.hypot(bx - cx, by - cy).min())
            if r > best[0]:
                best = (r, (float(cx), float(cy)))
        rec["flat_inscribed_r"] = round(best[0], 4)
        rec["flat_centre"] = best[1]
        if best[1] is not None:
            rec["half_width_at_h"] = half_width_profile(rec, best[1])
        ext = fl.any(axis=1), fl.any(axis=0)
        rec["flat_extent_xy"] = [round(float(ext[0].sum() * a.res), 4), round(float(ext[1].sum() * a.res), 4)]
        out["sinks"][sn] = rec
        print(f"SINK {sn} floor {zf:.4f} rim {rec['rim_z']:.4f} flat cells {rec['floor_cells']} inscribed r {rec['flat_inscribed_r']:.4f} at {rec['flat_centre']} "
              f"extent {rec['flat_extent_xy']} half_width {rec.get('half_width_at_h')}", flush=True)

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
        p, q = o.get_position_orientation()
        lo, hi = [t.tolist() for t in o.aabb]
        xs, ys, z, body = ray_grid(lo[0] - 0.01, hi[0] + 0.01, lo[1] - 0.01, hi[1] + 0.01, lo[2] - 0.2, hi[2] + 0.05, 0.002)
        paths = tuple(str(lk.prim_path) for lk in o.links.values())
        on = np.array([b.startswith(paths) for b in body]).reshape(z.shape) & ~np.isnan(z)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        cx, cy = float(p[0]), float(p[1])
        r = np.hypot(gx[on] - cx, gy[on] - cy)
        zl = float(z[on].min())
        u = z[on] - zl
        prof = _bins_min(r, u)
        rec = {"spec": spec, "R": round(float(r.max()), 4), "height": round(float(hi[2] - lo[2]), 4), "dr": 0.002, "u": [round(float(v), 4) for v in prof],
               "foot_r": round(float(r[u <= 0.001].max()), 4), "rim_u": round(float(prof[-1]), 4)}
        plates[spec] = (o, i)
        out["plates"][spec] = rec
        print(f"PLATE {spec} R {rec['R']:.4f} height {rec['height']:.4f} foot_r {rec['foot_r']:.4f} rim_u {rec['rim_u']:.4f} "
              f"u(0.8R) {prof[int(0.8 * rec['R'] / 0.002)]:.4f}", flush=True)

    # ---- fits and drops
    up = th.tensor([0.0, 0.0, 1.0])
    for sn, srec in out["sinks"].items():
        if "error" in srec:
            continue
        s = world.obj(sn)
        for spec, (o, idx) in plates.items():
            prec = out["plates"][spec]
            marg, c = fit_plate(srec, prec)
            fit = {"sink": sn, "plate": spec, "margin_mm": round(marg * 1000, 1), "centre": c, "fits_flat": bool(marg >= 0.0)}
            out["fits"].append(fit)
            print(f"FIT {sn} {spec} margin {fit['margin_mm']:+.1f} mm centre {c} -> {'FLAT' if marg >= 0 else 'TILTED'}", flush=True)
            if a.no_drops or c is None:
                continue
            hull = world.local_hull(o)
            zf = srec["floor_z"]
            xs, ys, zr = np.array(srec["xs"]), np.array(srec["ys"]), np.array(srec["z"], dtype=float)

            def raster_z(x, y):
                i = int(np.clip(np.searchsorted(xs, x), 0, len(xs) - 1))
                j = int(np.clip(np.searchsorted(ys, y), 0, len(ys) - 1))
                v = zr[i, j]
                return -9.0 if np.isnan(v) else float(v)

            for dx, dy in ((0.0, 0.0), (0.02, 0.0), (-0.02, 0.0), (0.0, 0.02), (0.0, -0.02)):
                for tilt in a.tilts:
                    dirs = [(1.0, 0.0)] if tilt == 0.0 else [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
                    if (dx, dy) != (0.0, 0.0) and tilt != 0.0:
                        continue  # the offsets only for the level release
                    for ux, uy in dirs:
                        # tilt about the horizontal axis perpendicular to (ux, uy): the rim toward (ux, uy) goes down
                        axis = th.tensor([-uy, ux, 0.0])
                        quat = G.quat_about_axis(axis, math.radians(tilt)) if tilt else th.tensor([0.0, 0.0, 0.0, 1.0])
                        pts = G.local_points_world(hull, (th.zeros(3), quat))
                        base = zf - float(pts[:, 2].min()) + a.drop
                        cx, cy = c[0] + dx, c[1] + dy
                        lift = 0.0
                        for pt in pts:  # raise until no hull point sits under the raster (a wall)
                            gz = raster_z(cx + float(pt[0]), cy + float(pt[1]))
                            lift = max(lift, gz + 0.002 - (base + float(pt[2])))
                        pos = th.tensor([cx, cy, base + lift])
                        o.set_position_orientation(pos, quat)
                        world.activate(o)
                        world.settle([o], steps=120, still=[s])
                        _, q1 = o.get_position_orientation()
                        t130 = math.degrees(math.acos(float(th.clamp(G.rotate(q1.float(), up)[2], -1, 1))))
                        world.step_physics(350, still=[s, world.robot])
                        p2, q2 = o.get_position_orientation()
                        t480 = math.degrees(math.acos(float(th.clamp(G.rotate(q2.float(), up)[2], -1, 1))))
                        d = {"sink": sn, "plate": spec, "offset": [dx, dy], "tilt": tilt, "dir": [ux, uy], "lift_mm": round(lift * 1000, 1),
                             "tilt_130": round(t130, 2), "tilt_480": round(t480, 2), "z_480_above_floor": round(float(p2[2]) - zf, 4),
                             "xy_480": [round(float(p2[0]), 4), round(float(p2[1]), 4)], "ok_15": bool(t130 <= 15.0)}
                        out["drops"].append(d)
                        print(f"DROP {sn} {spec} off {dx:+.2f},{dy:+.2f} tilt {tilt:.0f} dir {ux:+.0f},{uy:+.0f} lift {d['lift_mm']} mm -> "
                              f"{t130:.1f} deg (130) {t480:.1f} deg (480) z {d['z_480_above_floor']:.4f}", flush=True)
                        world.park(o, idx)
            n = [d for d in out["drops"] if d["sink"] == sn and d["plate"] == spec]
            print(f"DROPS {sn} {spec}: {sum(d['ok_15'] for d in n)}/{len(n)} within 15 deg; max {max(d['tilt_130'] for d in n):.1f}", flush=True)
    out["elapsed_s"] = round(time.time() - t0, 1)
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"BASIN_FIT_DONE {a.out} {out['elapsed_s']} s", flush=True)
    import omnigibson as og

    og.shutdown()


if __name__ == "__main__":
    main()
