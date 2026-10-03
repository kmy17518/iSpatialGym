"""Fabric-direct replay prototype (a measurement tool, not used by production): does the renderer draw poses written
straight into Fabric (no state load, no PhysX writes), what must be written for the cameras to follow, and how fast
is a tick?

    python -m ispatialgym.multienv.replay_fabric --demo <physics-only demo dir> --out <dir> [--start 1] [--frames N]
        [--variants plain2,links,links_camfabric,links_camusd]

One slot (the replayer's environment, :func:`ispatialgym.multienv.replay.env_config`).
  pass 1 (plain): every recorded state loaded as the exact replayer does (``og.sim.batched_usd_edits()``), one render,
           the robot cameras' rgb read; per frame every link prim's Fabric world transform (kept where it changed; it
           stands in for the per-step link poses a collector would record), and per camera its view matrix
           (``camera_parameters["cameraViewTransform"]``), its prim's Fabric world transform and its USD world transform.
  pass 2, one per variant, each from the first state loaded the plain way, then per state only writes, one render:
    plain2           plain loading again: the renderer's run-to-run floor for this demo (run it first)
    links            the changed links' pass-1 world transforms (``fabric_hierarchy.set_world_xform``) +
                     ``update_world_xforms``
    links_camfabric  links, and every camera prim's pass-1 Fabric world transform
    links_camusd     links, and every camera prim's pass-1 USD world transform written to USD (``editing_usd``)
    nosync_links     links with ``/physics/fabricUpdateTransformations`` off (PhysX would rewrite its own poses into
                     Fabric on every update); ``nosync_links_all`` writes every link, not only the changed ones
    rec              the demo's recorded ``isg/link_poses`` rows (ISPATIALGYM_RECORD_LINK_POSES=1) instead of pass 1's
                     transforms, the links that moved since the previous row, synchronization off
  suffixes: _ahead (the next frame's transforms), _u2 (two transform updates), _r2 (two renders), _sync
    (``SynchronizeToFabric`` after the writes), _kick (plus a real USD edit per tick: a hidden cube's colour), _cbE /
    _cbL (the writes from an app-update callback, early / late order).
Images are read from the sensors (``env.get_obs()``'s proprioception reads synchronize PhysX into Fabric).
Per variant: ms per tick (write / update / render / read), rgb PSNR per camera against pass 1 (mean, min, first and
last tenth; per frame in the json; also against pass 1 shifted by one frame), and per camera the frames whose view
matrix differs from pass 1's; ``--dump-frames N`` saves the N frames per camera furthest below the floor as PNGs.
Saves ``<out>/fabric.json`` (and the view matrices in ``<out>/views.npz``); prints ``FABRIC {json}``.
Run under ``tools/nothp.py`` on a node whose job cgroup is full (it holds every frame in memory).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback

import numpy as np

from ispatialgym.multienv import replay as RP


def psnr_stats(a_list, b_list):
    a, b = np.stack(a_list).astype(np.float64), np.stack(b_list).astype(np.float64)
    mse = ((a - b) ** 2).reshape(len(a), -1).mean(axis=1)
    p = 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-10))
    k = max(1, len(p) // 10)
    return {"mean": round(float(p.mean()), 2), "min": round(float(p.min()), 2), "first_tenth": round(float(p[:k].mean()), 2),
            "last_tenth": round(float(p[-k:].mean()), 2)}


def quat_xyzw_to_rot(q):
    """Column-vector rotation matrix of quaternions (..., 4) xyzw."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def pose_matrices(poses, scales):
    """USD (row-vector) world matrices (..., 4, 4) from poses (..., 7: pos xyz, quat xyzw) and per-link scales (L, 3)."""
    p = poses.astype(np.float64)
    q = p[..., 3:7] / np.linalg.norm(p[..., 3:7], axis=-1, keepdims=True)
    m = np.zeros(p.shape[:-1] + (4, 4))
    m[..., :3, :3] = scales[..., :, None] * np.swapaxes(quat_xyzw_to_rot(q), -1, -2)
    m[..., 3, :3] = p[..., :3]
    m[..., 3, 3] = 1.0
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--frames", type=int, default=100000)
    ap.add_argument("--warmup-s", type=float, default=60.0)
    ap.add_argument("--variants", default="plain2,nosync_links,nosync_links_all,links")
    ap.add_argument("--dump-frames", type=int, default=0,
                    help="per variant and camera, PNGs of the N frames furthest below the floor (plain2 run first): plain, "
                         "variant, |difference| x 8")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    demo = RP.Demo(args.demo)

    import omnigibson as og
    import omnigibson.lazy as lazy
    from omnigibson.macros import gm
    from omnigibson.sensors.vision_sensor import VisionSensor

    for name, value in (("ENABLE_TRANSITION_RULES", False), ("ENABLE_TILED_RENDERING", False), ("VISUAL_ONLY", True)):
        if getattr(gm, name) != value:
            with gm.unlocked():
                setattr(gm, name, value)
    env = og.Environment(configs=RP.env_config(demo, [demo], demo.dir, 1))
    scene = env.scenes[0]
    robot = scene.robots[0]
    slot = RP.Slot(0, scene, None, robot.name)
    slot.start(demo)
    cams = [n for n, s in robot.sensors.items() if isinstance(s, VisionSensor)]
    sensors = {n: robot.sensors[n] for n in cams}
    cam_paths = {n: s.prim_path for n, s in sensors.items()}
    paths = sorted({link.prim_path for obj in scene.objects for link in obj.links.values()})
    fh = og.sim.fabric_hierarchy
    sdf = lazy.usdrt.Sdf.Path
    stage = og.sim.stage
    rep = {"demo": demo.dir, "links": len(paths), "cameras": cam_paths, "start": args.start}

    def read(_=None):
        # the cameras' own observations: env.get_obs() also reads proprioception, whose prim reads synchronize PhysX
        # into Fabric (og.sim.sync_physx_to_fabric), which would undo Fabric writes
        out = {}
        for n in cams:
            o = sensors[n].get_obs()
            o = o[0] if isinstance(o, tuple) else o
            out[n] = o["rgb"][..., :3].detach().cpu().numpy().astype(np.uint8)
        return out

    settings = lazy.carb.settings.get_settings()

    def views():
        return {n: np.array(s.camera_parameters["cameraViewTransform"], dtype=np.float64).reshape(-1) for n, s in sensors.items()}

    def usd_world(path):
        return lazy.omni.usd.get_world_transform_matrix(stage.GetPrimAtPath(path))

    def load_plain(i):
        with og.sim.batched_usd_edits():
            slot.scene.load_state(demo.state_at(i), serialized=True)

    kick_color = None
    if any("_kick" in v for v in args.variants.split(",")):
        # a 1 cm cube 100 m below the floor (there before the warm-up and pass 1, so every pass renders the same scene)
        # whose display colour changes every tick: a real USD edit for editing_usd()'s SynchronizeToFabric to carry
        with og.sim.editing_usd():
            cube = lazy.pxr.UsdGeom.Cube.Define(stage, "/World/isg_replay_kick")
            cube.GetSizeAttr().Set(0.01)
            lazy.pxr.UsdGeom.XformCommonAPI(cube).SetTranslate(lazy.pxr.Gf.Vec3d(0.0, 0.0, -100.0))
            kick_color = cube.CreateDisplayColorAttr()
            kick_color.Set(lazy.pxr.Vt.Vec3fArray([lazy.pxr.Gf.Vec3f(0.5, 0.5, 0.5)]))
    load_plain(0)
    t_w = time.time()
    while time.time() - t_w < args.warmup_s:
        og.sim.render()
    end = min(demo.T, args.start + args.frames)
    n1 = end - args.start
    # pass 1: plain loads; link transforms (changed ones), camera views / Fabric / USD transforms recorded
    load_plain(args.start)
    og.sim.render()
    prev = {p: np.array(fh.get_world_xform(sdf(p))) for p in paths}
    changes, cam_fab, cam_usd, frames1, views1 = [], [], [], {n: [] for n in cams}, {n: [] for n in cams}
    rec_paths, rec_poses, rec_ref = None, None, []
    if any(v.startswith("rec_") for v in args.variants.split(",")):
        import h5py

        with h5py.File(os.path.join(args.demo, "demo.hdf5"), "r") as h:
            rec_paths = json.loads(h["data"].attrs["isg_link_pose_paths"])
            rec_poses = np.asarray(h["data/demo_0/isg/link_poses"])
        rec_scales = np.stack([np.linalg.norm(prev[p][:3, :3], axis=1) for p in rec_paths])
        rep["recorded"] = {"links": len(rec_paths), "rows": int(rec_poses.shape[0]), "dataset_bytes_raw": int(rec_poses.nbytes)}
    t_load = t_render = t_read = 0.0
    for i in range(args.start, end):
        t0 = time.perf_counter()
        load_plain(i)
        t1 = time.perf_counter()
        og.sim.render()
        t2 = time.perf_counter()
        f = read(env.get_obs()[0][0])
        t3 = time.perf_counter()
        v = views()
        for n in cams:
            frames1[n].append(f[n])
            views1[n].append(v[n])
        ch, cur = {}, {}
        for p in paths:
            m = fh.get_world_xform(sdf(p))
            a = np.array(m)
            cur[p] = a
            if not np.array_equal(a, prev[p]):
                ch[p] = m
                prev[p] = a
        changes.append(ch)
        if rec_paths is not None:
            rec_ref.append(np.stack([cur[p] for p in rec_paths]))
        cam_fab.append({n: fh.get_world_xform(sdf(p)) for n, p in cam_paths.items()})
        cam_usd.append({n: usd_world(p) for n, p in cam_paths.items()})
        t_load += t1 - t0
        t_render += t2 - t1
        t_read += t3 - t2
    rep["frames"] = n1
    rep["plain_ms_per_tick"] = {"load": round(1000 * t_load / n1, 1), "render": round(1000 * t_render / n1, 1), "read": round(1000 * t_read / n1, 1)}
    rep["changed_links_per_state"] = round(float(np.mean([len(c) for c in changes])), 1)
    rec_off = 0
    if rec_paths is not None:
        errs = {}
        for off in (-1, 0, 1):
            e = []
            for k, i in enumerate(range(args.start, end)):
                r = i + off
                if 0 <= r < len(rec_poses):
                    e.append(float(np.abs(pose_matrices(rec_poses[r], rec_scales) - rec_ref[k]).max()))
            errs[off] = (round(float(np.mean(e)), 7), round(float(np.max(e)), 7)) if e else None
        rec_off = min((o for o in errs if errs[o]), key=lambda o: errs[o][0])
        rep["recorded"].update({"row_offset": rec_off, "matrix_error_mean_max_by_offset": {str(k): v for k, v in errs.items()}})
        gf_m = lazy.usdrt.Gf.Matrix4d

        def to_gf(a):
            return gf_m(*[float(x) for x in a.reshape(-1)])

        probe = prev[rec_paths[0]]
        rep["recorded"]["gf_roundtrip_exact"] = bool(np.array_equal(np.array(to_gf(probe)), probe))
    rep["variants"] = {}
    view_dump = {f"plain::{n}": np.stack(v) for n, v in views1.items()}
    for variant in [v for v in args.variants.split(",") if v]:
        res = {}
        frames2, views2 = {n: [] for n in cams}, {n: [] for n in cams}
        sub = None
        try:
            load_plain(args.start)
            og.sim.render()
            og.sim.render()  # the camera takes its pose one update late: start from a settled frame
            nosync = variant.startswith(("nosync_", "rec_"))
            if nosync:
                # PhysX writes its (unchanged) poses into Fabric on every app update otherwise
                settings.set_bool("/physics/fabricUpdateTransformations", False)
            write_all = "_all" in variant
            last = {p: fh.get_world_xform(sdf(p)) for p in paths} if write_all else None
            pending, sub = [], None
            if "_cbE" in variant or "_cbL" in variant:
                # the writes happen inside render()'s app update, from an update-event callback (early or late order)
                def _apply(_e, pending=pending):
                    if pending:
                        for p, m in pending:
                            fh.set_world_xform(sdf(p), m)
                        fh.update_world_xforms()
                        pending.clear()

                sub = lazy.omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
                    _apply, order=-1000 if "_cbE" in variant else 1000, name="isg_replay_fabric")
            via_cb = sub is not None
            t_write = t_upd = t_render2 = t_read2 = 0.0
            for k, i in enumerate(range(args.start, end)):
                t0 = time.perf_counter()
                if variant == "plain2":
                    load_plain(i)
                elif variant.startswith("rec_"):
                    r = min(max(i + rec_off + (1 if "_ahead" in variant else 0), 0), len(rec_poses) - 1)
                    rp = min(max(r - 1, 0), len(rec_poses) - 1)
                    mats = pose_matrices(rec_poses[r], rec_scales)
                    moved = np.flatnonzero(np.any(rec_poses[r] != rec_poses[rp], axis=-1)) if k else range(len(rec_paths))
                    if via_cb:
                        pending.extend((rec_paths[l], to_gf(mats[l])) for l in moved)
                    else:
                        for l in moved:
                            fh.set_world_xform(sdf(rec_paths[l]), to_gf(mats[l]))
                else:
                    kk = min(k + 1, len(changes) - 1) if "_ahead" in variant else k
                    if write_all:
                        last.update(changes[kk])
                        for p, m in last.items():
                            fh.set_world_xform(sdf(p), m)
                    else:
                        for p, m in changes[kk].items():
                            fh.set_world_xform(sdf(p), m)
                    if "links_camfabric" in variant:
                        for n, p in cam_paths.items():
                            fh.set_world_xform(sdf(p), cam_fab[k][n])
                t1 = time.perf_counter()
                if "_kick" in variant:
                    with og.sim.editing_usd():
                        kick_color.Set(lazy.pxr.Vt.Vec3fArray([lazy.pxr.Gf.Vec3f(0.5 + 1e-3 * (k % 2), 0.5, 0.5)]))
                elif "_sync" in variant:
                    og.sim.usdrt_stage.SynchronizeToFabric()
                if variant != "plain2" and not via_cb:
                    fh.update_world_xforms()
                    if "_u2" in variant:
                        # the written links' descendants (cameras, visual meshes) take the parents' new transforms in a
                        # second pass only
                        fh.update_world_xforms()
                if "links_camusd" in variant:
                    with og.sim.editing_usd():
                        for n, p in cam_paths.items():
                            lazy.omni.usd.set_prim_world_transform(stage.GetPrimAtPath(p), cam_usd[k][n]) \
                                if hasattr(lazy.omni.usd, "set_prim_world_transform") else _set_usd_world(lazy, stage, p, cam_usd[k][n])
                t2 = time.perf_counter()
                og.sim.render()
                if variant.endswith("_r2"):
                    og.sim.render()
                t3 = time.perf_counter()
                f = read()
                t4 = time.perf_counter()
                v = views()
                for n in cams:
                    frames2[n].append(f[n])
                    views2[n].append(v[n])
                t_write += t1 - t0
                t_upd += t2 - t1
                t_render2 += t3 - t2
                t_read2 += t4 - t3
            res["ms_per_tick"] = {"write_or_load": round(1000 * t_write / n1, 2), "update": round(1000 * t_upd / n1, 2),
                                  "render": round(1000 * t_render2 / n1, 1), "read": round(1000 * t_read2 / n1, 1)}
            res["psnr_vs_plain"] = {n.split(":")[1]: psnr_stats(frames1[n], frames2[n]) for n in cams}
            res["psnr_per_frame"] = {n.split(":")[1]: [round(float(10 * np.log10(255.0 ** 2 / max(1e-10, ((a.astype(np.float64) - b) ** 2).mean()))), 2)
                                                       for a, b in zip(frames1[n], frames2[n])] for n in cams}
            # against the plain pass shifted by one frame: a lagging (prev) or leading (next) replay
            res["psnr_vs_plain_prev"] = {n.split(":")[1]: psnr_stats(frames1[n][:-1], frames2[n][1:]) for n in cams}
            res["psnr_vs_plain_next"] = {n.split(":")[1]: psnr_stats(frames1[n][1:], frames2[n][:-1]) for n in cams}
            res["view_mismatch_frames"] = {n.split(":")[1]: int(sum(not np.allclose(a, b, atol=1e-6) for a, b in zip(views1[n], views2[n])))
                                           for n in cams}
            view_dump.update({f"{variant}::{n}": np.stack(v) for n, v in views2.items()})
            floor = rep["variants"].get("plain2", {}).get("psnr_per_frame")
            if args.dump_frames and variant != "plain2" and floor:
                from PIL import Image

                res["dumped"] = {}
                for n in cams:
                    c = n.split(":")[1]
                    gap = np.array(res["psnr_per_frame"][c]) - np.array(floor[c])
                    worst = [int(k) for k in np.argsort(gap)[: args.dump_frames]]
                    res["dumped"][c] = {str(k): round(float(gap[k]), 2) for k in worst}
                    for k in worst:
                        a, b = frames1[n][k], frames2[n][k]
                        diff = np.clip(np.abs(a.astype(np.int16) - b.astype(np.int16)) * 8, 0, 255).astype(np.uint8)
                        Image.fromarray(np.concatenate([a, b, diff], axis=1)).save(
                            os.path.join(args.out, f"{variant}__{c}__f{k + args.start:04d}__gap{gap[k]:.1f}.png"))
        except Exception as e:  # noqa: BLE001
            res["error"] = repr(e)
            res["trace"] = traceback.format_exc()[-1500:]
        finally:
            settings.set_bool("/physics/fabricUpdateTransformations", True)
            if sub is not None:
                sub.unsubscribe()
        rep["variants"][variant] = res
        print("FABRIC_VARIANT " + json.dumps({"variant": variant, **{k: v for k, v in res.items() if k not in ("trace", "psnr_per_frame")}}), flush=True)
    np.savez_compressed(os.path.join(args.out, "views.npz"), **view_dump)
    with open(os.path.join(args.out, "fabric.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("FABRIC " + json.dumps({k: v for k, v in rep.items() if k != "variants"}), flush=True)
    return 0


def _set_usd_world(lazy, stage, path, world):
    """Author ``path``'s local transform so that its USD world transform is ``world`` (Gf.Matrix4d)."""
    prim = stage.GetPrimAtPath(path)
    parent_world = lazy.omni.usd.get_world_transform_matrix(prim.GetParent())
    local = world * parent_world.GetInverse()
    xf = lazy.pxr.UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    if len(ops) == 1 and ops[0].GetOpType() == lazy.pxr.UsdGeom.XformOp.TypeTransform:
        ops[0].Set(local)
    else:
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(local)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    with contextlib.suppress(Exception):
        import omnigibson as og

        og.shutdown()
    os._exit(rc)
