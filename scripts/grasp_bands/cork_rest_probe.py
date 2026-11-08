"""i7-cork rest survey (2026-11-08, owner decision 4.2): how stable is the cork ncxgpe on each of its candidate rests?

The attachment sampler's nudge test (``tasks/attachment.Attachment.check_rest_stability``: the settled part tilted 2 deg about the world
x axis and lifted 1 mm, 300 physics steps with the robot held, the up axis within 3 deg of the settled one) rejected 24 of 41 attempts of
attachment_03 with the cork standing on its tapered male end (r0). This probe runs the same nudge in the grasp library's validation world
(``validate_tabletop.TabletopWorld``: the demo physics, the slab, the sleep assist) for three rest families:

* ``male_down``  - r0, the library rest (identity): the tapered male end (ring r 6.1-6.5 mm) on the slab;
* ``wide_down``  - turned 180 deg about x: the wide end (r ~7.8 mm at the top face) on the slab (route i, the hand-over);
* ``lying``      - turned 90 deg about x, rolled about the cork axis (route ii, the 90 deg reorient).

Per trial: the drop + settle of the validator (``TabletopWorld.settle``, varied xy / yaw), then the sampler's nudge about the world x and
the world y axis (each from the settled snapshot): the up-axis tilt (the sampler's metric: the object's local +z), the full rotation and
the translation after 300 steps. One ``CORK_REST {json}`` line per trial (also ``<out>/rest_probe.jsonl``).

    python -u scripts/grasp_bands/cork_rest_probe.py --out OUT [--trials 12] [--model-dir DIR] [--support desk/mdhelw]

``--support <category>/<model>``: the cork settles on that B1K object (fixed, its top found by a drop) instead of the validator's slab, at a
random xy over the inner 80 % of the top (attachment_03's desk mdhelw: is the instability the desk's surface?). Same nudge.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

NUDGE_DEG = 2.0  # tasks/attachment.REST_NUDGE_DEG
STEPS = 300  # REST_STABILITY_STEPS
TIP_MAX_DEG = 3.0  # REST_TIP_MAX_DEG


def qmul(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def qaxis(axis, ang):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    s = math.sin(ang / 2)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(ang / 2)])


def rot(q, v):
    x, y, z, w = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return R @ np.asarray(v, dtype=float)


def qang(a, b):
    return 2.0 * math.acos(min(1.0, abs(float(np.dot(a, b)))))


def rests(rng, trials):
    """(family, quat xyzw) per trial: male_down / wide_down with a random yaw, lying with a random roll about the cork axis and yaw."""
    out = []
    for k in range(trials):
        yaw = qaxis([0, 0, 1], float(rng.uniform(-math.pi, math.pi)))
        out.append(("male_down", yaw))
        out.append(("wide_down", qmul(yaw, qaxis([1, 0, 0], math.pi))))
        roll = qaxis([0, 0, 1], float(rng.uniform(-math.pi, math.pi)))
        out.append(("lying", qmul(yaw, qmul(qaxis([1, 0, 0], math.pi / 2), roll))))
    return out


def settle_on(world, obj, geom, q0, xy, top_z, drop=0.003):
    """The validator's drop + settle (``TabletopWorld.settle``'s rule: still over 30 steps, 1 mm / 3 mrad, slow) on a B1K support whose top is
    at ``top_z`` (the AABB's: a drop from 3 mm above it; a lower surface under the xy is reached by the fall). ``(record, state)``."""
    import torch as th

    cfg = world.cfg
    V = geom.union.vertices
    R0 = np.array([rot(q0, e) for e in np.eye(3)]).T
    bottom = -float((V @ R0.T)[:, 2].min())
    world.park()
    obj.set_position_orientation(th.tensor([xy[0], xy[1], top_z + bottom + drop], dtype=th.float32), th.tensor(q0, dtype=th.float32))
    obj.keep_still()
    a = world.hold_action()
    hist, stable, steps = [], False, 0
    for steps in range(1, cfg.settle_max_steps + 1):
        world.env.step(a)
        p, q = obj.get_position_orientation()
        hist.append((p.numpy().astype(float).copy(), q.numpy().astype(float).copy()))
        if steps >= cfg.settle_window:
            ps = np.array([h[0] for h in hist[-cfg.settle_window:]])
            qs = [h[1] for h in hist[-cfg.settle_window:]]
            dp = float(np.linalg.norm(ps - ps[-1], axis=1).max())
            dr = max(qang(qq, qs[-1]) for qq in qs)
            if dp <= cfg.settle_pos_m and dr <= cfg.settle_rot_rad and float(th.linalg.norm(obj.get_linear_velocity())) <= cfg.settle_speed_m_s \
                    and float(th.linalg.norm(obj.get_angular_velocity())) <= cfg.settle_ang_rad_s:
                stable = True
                break
    p, q = obj.get_position_orientation()
    up_dev = math.acos(float(np.clip(np.dot(rot(q.numpy().astype(float), [0, 0, 1]), rot(q0, [0, 0, 1])), -1, 1)))
    rec = {"stable": stable, "settle_steps": steps, "tilt_from_intended_rad": up_dev, "h_root": round(float(p[2]) - top_z, 6)}
    return rec, world.scene.dump_state(serialized=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--model-dir", default=None, help="a geometry export dir (default <out>/model)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--support", default=None, help="category/model of a B1K support (default: the validator's slab)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(a.out, "rest_probe.jsonl")
    from ispatialgym.grasping import object_geometry as OG
    from ispatialgym.grasping import validate_tabletop as VT

    world = VT.TabletopWorld()
    import omnigibson as og
    import torch as th

    job = {"category": "cork", "model": "ncxgpe", "scale": [1.0, 1.0, 1.0], "dataset": "behavior-1k-assets", "id": "cork_rest_probe"}
    mdir = a.model_dir or os.path.join(a.out, "model")
    os.makedirs(mdir, exist_ok=True)
    obj = world.load(job, mdir)
    geom = OG.load(mdir)
    rng = np.random.default_rng(a.seed)
    desk = None
    if a.support:
        from omnigibson.objects.dataset_object import DatasetObject

        cat, mod = a.support.split("/")
        desk = DatasetObject(name="probe_support", category=cat, model=mod, dataset_name="behavior-1k-assets", fixed_base=True, include_default_states=False)
        world.scene.add_object(desk)
        desk.set_position_orientation(th.tensor([2.0, 2.0, 0.6]), th.tensor([0.0, 0.0, 0.0, 1.0]))
        for _ in range(3):
            og.sim.step()
        lo, hi = (v.numpy().astype(float) for v in desk.aabb)
        desk.set_position_orientation(th.tensor([2.0, 2.0, 0.6 - float(lo[2])]), th.tensor([0.0, 0.0, 0.0, 1.0]))
        for _ in range(3):
            og.sim.step()
        lo, hi = (v.numpy().astype(float) for v in desk.aabb)
        world.slab.set_position_orientation(th.tensor([3.0, 3.0, -5.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))  # the slab out of the way
        print("CORK_REST_SUPPORT " + json.dumps({"support": a.support, "aabb_lo": lo.round(4).tolist(), "aabb_hi": hi.round(4).tolist()}), flush=True)
    for k, (fam, q0) in enumerate(rests(rng, a.trials)):
        xy = (float(rng.uniform(-0.05, 0.05)), float(rng.uniform(-0.05, 0.05)))
        t0 = time.time()
        if desk is not None:
            xy = (float(rng.uniform(lo[0] + 0.1 * (hi[0] - lo[0]), hi[0] - 0.1 * (hi[0] - lo[0]))),
                  float(rng.uniform(lo[1] + 0.1 * (hi[1] - lo[1]), hi[1] - 0.1 * (hi[1] - lo[1]))))
            rec, state = settle_on(world, obj, geom, q0, xy, float(hi[2]))
        else:
            rec, state = world.settle(obj, geom, q0, obj_xy=xy)
        p_s, q_s = obj.get_position_orientation()
        p_s, q_s = p_s.numpy().astype(float), q_s.numpy().astype(float)
        row = {"trial": k, "family": fam, "xy": [round(v, 4) for v in xy], "support": a.support or "slab", "start_quat_xyzw": [round(float(v), 5) for v in q0], "settled_quat_xyzw": [round(float(v), 5) for v in q_s],
               "stable": rec["stable"], "settle_steps": rec["settle_steps"], "tilt_from_start_deg": round(math.degrees(rec["tilt_from_intended_rad"]), 2),
               "axis_world": [round(float(v), 3) for v in rot(q_s, [0, 0, 1])], "h_root_m": rec["h_root"], "nudge": {}}
        for nm, ax in (("x", [1, 0, 0]), ("y", [0, 1, 0])):
            world.scene.load_state(state, serialized=False)
            if desk is None:
                world.place_slab(rec["slab"]["centre"], rec["slab"]["dims"])
            qn = qmul(qaxis(ax, math.radians(NUDGE_DEG)), q_s)
            obj.set_position_orientation(th.tensor(p_s + np.array([0, 0, 0.001]), dtype=th.float32), th.tensor(qn, dtype=th.float32))
            obj.keep_still()
            obj.wake()
            for _ in range(STEPS):
                og.sim.step_physics()
                world.robot.keep_still()
            p1, q1 = obj.get_position_orientation()
            p1, q1 = p1.numpy().astype(float), q1.numpy().astype(float)
            c = float(np.clip(np.dot(rot(q_s, [0, 0, 1]), rot(q1, [0, 0, 1])), -1.0, 1.0))
            tilt = math.degrees(math.acos(c))
            row["nudge"][nm] = {"up_axis_tilt_deg": round(tilt, 2), "rotation_deg": round(math.degrees(qang(q_s, q1)), 2),
                                "moved_mm": round(float(np.linalg.norm(p1 - p_s)) * 1000, 2), "pass": tilt <= TIP_MAX_DEG}
        row["pass_both"] = all(v["pass"] for v in row["nudge"].values())
        row["wall_s"] = round(time.time() - t0, 2)
        line = json.dumps(row)
        print("CORK_REST " + line, flush=True)
        with open(out_path, "a") as f:
            f.write(line + "\n")
    summ = {}
    for line in open(out_path):
        r = json.loads(line)
        s = summ.setdefault(r["family"], {"trials": 0, "settled_stable": 0, "pass_x": 0, "pass_y": 0, "pass_both": 0, "tilts": []})
        s["trials"] += 1
        s["settled_stable"] += int(bool(r["stable"]))
        s["pass_x"] += int(r["nudge"]["x"]["pass"])
        s["pass_y"] += int(r["nudge"]["y"]["pass"])
        s["pass_both"] += int(r["pass_both"])
        s["tilts"] += [r["nudge"]["x"]["up_axis_tilt_deg"], r["nudge"]["y"]["up_axis_tilt_deg"]]
    for s in summ.values():
        t = sorted(s.pop("tilts"))
        s["tilt_deg_median"] = t[len(t) // 2] if t else None
        s["tilt_deg_max"] = t[-1] if t else None
    print("CORK_REST_SUMMARY " + json.dumps(summ), flush=True)
    with open(os.path.join(a.out, "rest_probe_summary.json"), "w") as f:
        json.dump(summ, f, indent=1)


if __name__ == "__main__":
    main()
