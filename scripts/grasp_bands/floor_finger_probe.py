"""i7-floorfix probe (2026-11-07): is the floor kind's finger rule defeated by the IK's own ground slab?

The floor reach maps (``reach_maps`` kind ``floor``) keep the cuRobo ground slab ``__ground__`` in the IK collision world; the floor IS
the support there, so a padded finger sphere that touches the slab fails the pre-grasp / grasp IK before the contact policy's 2 mm
finger rule (``HandChecker.check(floor=True)``) is consulted. This probe runs the map builder's own code path (``reach_maps.compute_map``
and its ``_solve``) in three IK variants:

* ``as_built``     - the slab in the IK for every sphere (the maps as built);
* ``fingers_off``  - the working arm's finger spheres switched off in the IK solver's sphere model (world AND self-collision: cuRobo has
                     no per-pair mask; the probe only), the slab in for every other sphere;
* ``slab_off``     - the whole slab out of the IK; every non-finger link then judged only by the map's explicit sphere check
                     (``contact_policy.sphere_support_ok``: pre-grasp, grasp and 3 interpolated configurations).

Part A: the fix1 configuration (umbrella top_2 and its twin, trunk t2.00/k0, the object at the fix1 base-frame pose; both arms): per
variant the per-seed pre-grasp / grasp IK successes and, at the solutions, the finger / non-finger sphere and hull clearances to the
floor, plus the map status of the restricted map (the fix1 cell and its 4 nearest lattice cells). Part B: whole floor maps per variant
(status histograms, feasible counts). Part D: the off-curve deeper trunk (torso_joint1 at its upper limit, joints 2 / 3 of t2.00/k0) as
an extra pseudo-trunk ``t2.25`` (report only). One ``FLOOR_PROBE {json}`` line per result (also appended to ``<out>/probe.jsonl``).

    python -u scripts/grasp_bands/floor_finger_probe.py --glib GLIB --out OUT [--entries E ...] [--deep]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from contextlib import contextmanager

import numpy as np
import torch as th

#: fix1 (inter_object_relation_satisfaction-touching_large_scale_02, instance 0000): base_pose2d, umbrella root xy and yaw (world)
FIX1_BASE = (-0.30720019340515137, 4.54558801651001, -1.0782334285508506)
FIX1_OBJ = (0.1447465717792511, 4.23214864730835, 3.634270946426124)
VARIANTS = ("as_built", "fingers_off", "slab_off")
DEEP_KEY_TRANSLATE = 2.25  # the pseudo-trunk of Part D (torso_joint1 at its upper limit; never a real translate: the curve ends at 2.0)

STATE = {"variant": "as_built"}
OUT = {"path": None}


def emit(rec):
    line = json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("FLOOR_PROBE " + line, flush=True)
    if OUT["path"]:
        with open(OUT["path"], "a") as f:
            f.write(line + "\n")


def load_entry(S, glib, eid):
    """The v2 library entry ``eid`` (a directory of ``<glib>/models``) through ``schema.load_model`` with its own fields and its
    candidates' designated rest hint (the briefcase's ``__r2cc3ac``)."""
    d = os.path.join(S.library_root(glib), "models", eid)
    lib_d = S.load_json(os.path.join(d, "library.json"))
    cand = S.load_json(os.path.join(d, "candidates.json")) or {}
    hint = cand.get("rest_hint_wxyz") if re.search(r"__r[0-9a-f]{6}$", eid) else None
    lib = S.load_model(lib_d["category"], lib_d["model"], lib_d.get("scale"), glib, constraint=lib_d.get("constraint") or "tabletop", rest_hint_wxyz=hint)
    if lib is None or os.path.basename(os.path.dirname(lib.path)) != eid:
        raise RuntimeError(f"{eid}: load_model resolved {None if lib is None else lib.path}")
    return lib


@contextmanager
def variant_ctx(world, entry, name, arm):
    """The IK variant for the block (only the batched IK solver of ``entry`` is touched: the whole-body FK / collision entries keep
    their spheres and the slab)."""
    from ispatialgym.grasping import contact_policy as CP

    done = []
    cub = getattr(world.backend.checker.data, "cuboids", None)
    slab = world.backend.GROUND_SLAB_NAME
    if name == "fingers_off":
        kcs = {}
        for kin in (entry.fk, getattr(entry.rsc, "kinematics", None)):
            kc = getattr(kin, "kinematics_config", None)
            if kc is not None:
                kcs[id(kc)] = kc
        for kc in kcs.values():
            for link in CP.DEFAULT.finger_links(arm):
                kc.disable_link_spheres(link)
                done.append((kc, link))
    elif name == "slab_off":
        cub.set_enabled(slab, False)
    try:
        yield
    finally:
        for kc, link in done:
            kc.enable_link_spheres(link)
        if name == "slab_off":
            cub.set_enabled(slab, True)


def install(RM):
    orig = RM._solve

    def _solve_variant(world, entry, T, q_lock, seed_active=None):
        with variant_ctx(world, entry, STATE["variant"], world.cfg.arm):
            return orig(world, entry, T, q_lock, seed_active)

    RM._solve = _solve_variant
    return orig


def install_deep_trunk(world):
    """Part D: ``q_config(DEEP_KEY_TRANSLATE, 0)`` = t2.00/k0 with torso_joint1 at its upper limit (joints 2 / 3 unchanged)."""
    orig = world.q_config

    def q_config(translate, tilt, posture="untucked", fingers=0.05):
        if abs(float(translate) - DEEP_KEY_TRANSLATE) < 1e-6:
            q = orig(2.0, tilt, posture, fingers)
            j1 = int(world.robot.trunk_control_idx[0])
            q[j1] = float(world.trunk_limits[1][0])
            return q
        return orig(translate, tilt, posture, fingers)

    world.q_config = q_config


def set_arm(world, RM, arm):
    world.cfg = RM.floor_config(arm=arm, mode="full", lattice_m=0.08, seeds=8, batch=2048, twins=True)
    world._shoulder = {}
    return world.cfg


def summarize(RM, meta, arrays):
    S_ = RM.STATUS
    names = RM.STATUS_NAME
    hist = {}
    feas_cells = None
    per_trunk = {}
    attempted = {}
    for t in meta["trunks"]:
        a = arrays[f"s0.00_{t['key']}"]
        u, c = np.unique(a, return_counts=True)
        for k, v in zip(u.tolist(), c.tolist()):
            hist[names[k]] = hist.get(names[k], 0) + v
        f = (a == S_["feasible"])
        per_trunk[t["key"]] = int(f.sum())
        fc = f.any(axis=(1, 2))
        feas_cells = fc if feas_cells is None else (feas_cells | fc)
        # hand_policy also marks the IK-independent rejections (left out here): the problems that went to the IK and ended so
        for k in ("ik_pre", "ik_grasp", "slab_spheres", "object_arm", "branch_jump", "feasible"):
            attempted[k] = attempted.get(k, 0) + int((a == S_[k]).sum())
    n_ik = sum(attempted.values())
    return {"hist": hist, "feasible_by_trunk": {k: v for k, v in per_trunk.items() if v}, "n_feasible": int(sum(per_trunk.values())),
            "trunks_reaching": sorted(k for k, v in per_trunk.items() if v), "cells_any_feasible": int(feas_cells.sum()) if feas_cells is not None else 0,
            "n_cells": int(len(arrays["cells"])), "ik_attempt_breakdown": {k: round(v / max(n_ik, 1), 4) for k, v in attempted.items()}, "n_ik_attempted": n_ik,
            "timing": meta.get("timing")}


def fix1_frame():
    bx, by, byaw = FIX1_BASE
    ox, oy, oyaw_w = FIX1_OBJ
    dx, dy = ox - bx, oy - by
    c, s = math.cos(byaw), math.sin(byaw)
    return (c * dx + s * dy, -s * dx + c * dy), (oyaw_w - byaw) % (2 * math.pi)


def sphere_report(world, entry, q_act, q_lock, arm, links, ground):
    """min (z - r) to the floor (z = 0) per link group at the working-arm solution ``q_act``: the two fingers, the hand's rigid links,
    the rest (ground-contact links excluded)."""
    from ispatialgym.grasping import contact_policy as CP

    sp = world.spheres(world.full_q(entry, q_act.unsqueeze(0), q_lock))[0].float().cpu()
    d = sp[:, 2] - sp[:, 3]
    on = sp[:, 3] > 0
    fing = set(CP.DEFAULT.finger_links(arm))
    hand = set(CP.DEFAULT.hand_rigid_links(arm))
    out = {}
    for name, sel in (("finger", lambda l: l in fing), ("hand_rigid", lambda l: l in hand), ("other", lambda l: l and l not in fing and l not in hand and l not in ground)):
        m = th.tensor([bool(sel(l)) for l in links]) & on
        if bool(m.any()):
            k = int(th.argmin(th.where(m, d, th.full_like(d, 1e9))))
            out[name] = {"min_mm": round(1000 * float(d[k]), 2), "link": links[k], "r_mm": round(1000 * float(sp[k, 3]), 2)}
    for lk in sorted(fing):
        m = th.tensor([l == lk for l in links]) & on
        if bool(m.any()):
            out[lk] = round(1000 * float(d[m].min()), 2)
    return out


def part_a(world, RM, CP, S, OG, lib, geom, rid, arms):
    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling import reach_bands as RB

    (px, py), oyaw = fix1_frame()
    emit({"part": "A", "what": "fix1 frame", "object_base_xy": [round(px, 4), round(py, 4)], "object_yaw_deg": round(math.degrees(oyaw), 2)})
    rest = lib.rests[rid]
    for arm in arms:
        cfg = set_arm(world, RM, arm)
        grasps = RM.grasp_set(lib, rid, cfg)
        names = [g["name"] for g in grasps]
        lat = RM.lattice(cfg)
        near = np.argsort(np.hypot(lat[:, 0] - px, lat[:, 1] - py))[:4]
        cells = np.concatenate([np.array([[px, py]]), lat[near]])
        # the map's own code path at the fix1 cell (+ the 4 nearest lattice cells), trunk t2.00/k0 only
        orig_lat, orig_tg = RM.lattice, RB.trunk_grid
        RM.lattice = lambda _cfg, _c=cells: _c
        RB.trunk_grid = lambda h: [(2.0, 0.0)]
        posture = None
        try:
            for v in VARIANTS:
                STATE["variant"] = v
                meta, arrays = RM.compute_map(world, lib, rid, 0.0, RM.FLOOR_ANCHOR, geom=geom, cfg=cfg, kind="floor")
                tr = meta["trunks"][0]
                posture = tr.get("posture")
                st = arrays["s0.00_t2.00_k+0.0"]
                yi = int(np.argmin([abs(((oyaw - y + math.pi) % (2 * math.pi)) - math.pi) for y in arrays["yaws"]]))
                rows = {}
                for ci in range(len(cells)):
                    rows[f"({cells[ci, 0]:.3f},{cells[ci, 1]:.3f})"] = {names[g]: RM.STATUS_NAME[int(st[ci, yi, g])] for g in range(len(names))
                                                                       if grasps[g]["base"] == "top_2" or len(names) <= 4}
                emit({"part": "A", "what": "map status", "arm": arm, "variant": v, "trunk": tr["key"], "posture": posture, "yaw_index": yi,
                      "yaw_deg": round(math.degrees(float(arrays["yaws"][yi])), 1), "cells": rows})
        finally:
            RM.lattice, RB.trunk_grid = orig_lat, orig_tg
            STATE["variant"] = "as_built"
        # per-seed detail at the fix1 pose (the exact yaw), the map's q_lock (posture of the trunk's eligibility)
        entry = world.entry()
        links = world.link_of_sphere()
        ground = set(getattr(world.backend, "ground_contact_links", ()) or ())
        q_lock = world.q_config(2.0, 0.0, posture or "untucked")
        hc = CP.HandChecker(arm, CP.DEFAULT, device=str(world.dev), spacing=cfg.hand_spacing_m)
        plane = RM.floor_plane()
        Ry = RM._yaw_R([oyaw])[0]
        To = np.eye(4)
        To[:3, :3] = Ry @ S.quat_to_mat(rest.quat_xyzw)
        To[:3, 3] = [px, py, rest.h_root]
        for g in grasps:
            if g["base"] != "top_2" or abs(g["param"]) > 1e-9:
                continue
            TE = th.as_tensor(To @ g["T"], dtype=th.float32, device=world.dev).unsqueeze(0)
            TP = TE.clone()
            TP[:, :3, 3] -= C.PRE_DIST * TP[:, :3, 2]
            pq, cq = float(g["preshape_q"]), float(g["contact_q"])
            hull_target = {}
            for nm, T, fq in (("pre@preshape", TP, pq), ("grasp@mid", TE, 0.5 * (pq + cq)), ("grasp@contact", TE, cq), ("grasp@open0.05", TE, 0.05)):
                cl = hc.clearance(T, plane, (fq, fq))
                hull_target[nm] = {r: round(1000 * float(v[0]), 2) for r, v in cl.items()}
            for v in VARIANTS:
                STATE["variant"] = v
                try:
                    sp_ok, q_pre = RM._solve(world, entry, TP, q_lock)
                    rec = {"part": "A", "what": "ik detail", "arm": arm, "grasp": g["name"], "variant": v, "posture": posture,
                           "eef_target_z_mm": round(1000 * float(TE[0, 2, 3]), 2), "approach_axis": [round(float(x), 3) for x in TE[0, :3, 2].tolist()],
                           "closing_axis": [round(float(x), 3) for x in TE[0, :3, 1].tolist()], "preshape_q": pq, "contact_q": round(cq, 5),
                           "hull_at_targets_mm": hull_target, "pre_seeds_ok": [bool(x) for x in sp_ok[0].tolist()]}
                    if bool(sp_ok[0].any()):
                        s0 = int(th.argmax(sp_ok[0].int()))
                        q1 = q_pre[0, s0]
                        rec["pre_spheres"] = sphere_report(world, entry, q1, q_lock, arm, links, ground)
                        sg_ok, q_g = RM._solve(world, entry, TE, q_lock, seed_active=q1.unsqueeze(0))
                        rec["grasp_seeds_ok"] = [bool(x) for x in sg_ok[0].tolist()]
                        if bool(sg_ok[0].any()):
                            s1 = int(th.argmax(sg_ok[0].int()))
                            qg = q_g[0, s1]
                            rec["grasp_spheres"] = sphere_report(world, entry, qg, q_lock, arm, links, ground)
                            qf = world.full_q(entry, qg.unsqueeze(0), q_lock)
                            Tg = world.eef(qf)
                            rec["grasp_fk_hull_mid_mm"] = {r: round(1000 * float(x[0]), 2) for r, x in hc.clearance(Tg, plane, (0.5 * (pq + cq),) * 2).items()}
                            ok_s, bad = CP.sphere_support_ok(world.spheres(qf), links, plane, arm, CP.DEFAULT, ground_contact_links=sorted(ground), floor_support=True)
                            rec["grasp_sphere_policy_ok"] = [bool(ok_s[0]), bad[0]]
                            rec["grasp_self_collision_pairs"] = [p["links"] for p in world.backend.self_collision_pairs(qf[0].cpu(), "whole_body")][:6]
                            rec["branch_jump_rad"] = round(float((qg - q1).abs().max()), 3)
                except Exception as e:  # noqa: BLE001
                    import traceback

                    rec = {"part": "A", "what": "ik detail", "arm": arm, "grasp": g["name"], "variant": v, "error": repr(e), "tb": traceback.format_exc()[-1200:]}
                finally:
                    STATE["variant"] = "as_built"
                emit(rec)


def part_b(world, RM, S, OG, glib, entries, arms, variants, deep, full_variants=()):
    from ispatialgym.sampling import reach_bands as RB

    for eid in entries:
        try:
            lib = load_entry(S, glib, eid)
        except Exception as e:  # noqa: BLE001
            emit({"part": "B", "entry": eid, "error": repr(e)})
            continue
        geom = OG.load(os.path.dirname(lib.path))
        for arm in arms:
            cfg = set_arm(world, RM, arm)
            for rid, r in lib.rests.items():
                if not r.stable or not lib.grasps(rid):
                    continue
                for v in (VARIANTS if eid in full_variants else variants):
                    STATE["variant"] = v
                    t0 = time.time()
                    try:
                        meta, arrays = RM.compute_map(world, lib, rid, 0.0, RM.FLOOR_ANCHOR, geom=geom, cfg=cfg, kind="floor")
                        emit({"part": "B", "entry": eid, "arm": arm, "rest": rid, "variant": v, "n_grasps": len(meta["grasps"]), **summarize(RM, meta, arrays),
                              "wall_s": round(time.time() - t0, 1)})
                    except Exception as e:  # noqa: BLE001
                        import traceback

                        emit({"part": "B", "entry": eid, "arm": arm, "rest": rid, "variant": v, "error": repr(e), "tb": traceback.format_exc()[-1200:]})
                    finally:
                        STATE["variant"] = "as_built"
                if deep:
                    # Part D: the deeper pseudo-trunk alone and t2.00/k0 alone, under the slab_off rule (what the fix applies)
                    orig_tg = RB.trunk_grid
                    for tr_list, tag in (([(DEEP_KEY_TRANSLATE, 0.0)], "deep_j1_limit"), ([(2.0, 0.0)], "t2.00_k0")):
                        RB.trunk_grid = lambda h, _t=tr_list: list(_t)
                        STATE["variant"] = "slab_off"
                        try:
                            meta, arrays = RM.compute_map(world, lib, rid, 0.0, RM.FLOOR_ANCHOR, geom=geom, cfg=cfg, kind="floor")
                            t = meta["trunks"][0]
                            a = arrays[f"s0.00_{t['key']}"]
                            fc = (a == RM.STATUS["feasible"]).any(axis=(1, 2))
                            emit({"part": "D", "entry": eid, "arm": arm, "rest": rid, "trunk": tag, "eligible": t.get("eligible"), "posture": t.get("posture"),
                                  "reason": t.get("reason"), "shoulder": t.get("shoulder"), "arm_len": t.get("arm_len"), "n_feasible": int((a == RM.STATUS["feasible"]).sum()),
                                  "cells_any_feasible": int(fc.sum()), "feasible_cells": np.round(arrays["cells"][fc], 2).tolist()})
                        except Exception as e:  # noqa: BLE001
                            emit({"part": "D", "entry": eid, "arm": arm, "trunk": tag, "error": repr(e)})
                        finally:
                            RB.trunk_grid = orig_tg
                            STATE["variant"] = "as_built"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--glib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--entries", nargs="*", default=["umbrella__ziqzqo__s1", "mouse__pljjtu__s1", "sandal__chgcrm__s1", "soap_dispenser__zwudcz__s1",
                                                       "briefcase__lwbctq__s1__r2cc3ac"])
    ap.add_argument("--slow-entries", nargs="*", default=["trash_can__jlawet__s1"], help="whole maps of the right arm only, slab_off (as_built: the built map)")
    ap.add_argument("--deep", action="store_true", help="Part D: the off-curve deeper trunk (torso_joint1 at its limit)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    OUT["path"] = os.path.join(a.out, "probe.jsonl")
    t0 = time.time()
    from ispatialgym.sampling import reach_maps as RM  # SynthWorld imports the planner (warp pinning) before omnigibson

    from ispatialgym.grasping import contact_policy as CP
    from ispatialgym.grasping import object_geometry as OG
    from ispatialgym.grasping import schema as S

    world = RM.SynthWorld(RM.floor_config(arm="left", mode="full", lattice_m=0.08, seeds=8, batch=2048, twins=True))
    install(RM)
    if a.deep:
        install_deep_trunk(world)
    ent = world.entry()
    kc_ptrs = {k: (int(getattr(getattr(o, "kinematics_config", None), "link_spheres").data_ptr()) if getattr(o, "kinematics_config", None) is not None else None)
               for k, o in (("entry.fk", ent.fk), ("ik.kinematics", getattr(ent.rsc, "kinematics", None)))}
    cub = world.backend.checker.data.cuboids
    emit({"part": "setup", "setup_s": round(world.setup_s, 1), "root_z": world.root_z, "slab": world.backend.GROUND_SLAB_NAME, "cuboid_names": cub.names[0] if cub is not None else None,
          "use_cuda_graph": getattr(world.backend, "use_cuda_graph", None), "link_spheres_ptrs": kc_ptrs,
          "ground_contact_links": sorted(world.backend.ground_contact_links), "finger_links": {arm: list(CP.DEFAULT.finger_links(arm)) for arm in ("left", "right")},
          "finger_spheres_local": {lk: ent.fk.kinematics_config.get_link_spheres(lk).cpu().numpy().round(4).tolist() for lk in CP.DEFAULT.finger_links("left")}})
    umb = load_entry(S, a.glib, "umbrella__ziqzqo__s1")
    part_a(world, RM, CP, S, OG, umb, OG.load(os.path.dirname(umb.path)), "r0", ("left", "right"))
    emit({"part": "A", "done_s": round(time.time() - t0, 1)})
    # the fingers_off variant (the IK's steering of the non-finger links kept) only where it tells something: umbrella, sandal
    part_b(world, RM, S, OG, a.glib, a.entries, ("left", "right"), ("as_built", "slab_off"), a.deep, full_variants=("umbrella__ziqzqo__s1", "sandal__chgcrm__s1"))
    emit({"part": "B", "done_s": round(time.time() - t0, 1)})
    part_b(world, RM, S, OG, a.glib, a.slow_entries, ("right",), ("slab_off",), False)
    emit({"part": "done", "total_s": round(time.time() - t0, 1)})
    import omnigibson as og

    og.shutdown()


if __name__ == "__main__":
    main()
