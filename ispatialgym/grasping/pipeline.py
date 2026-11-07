"""Grasp library v2 pipeline - stages over a job manifest (:mod:`manifest`), each a persistent worker (one simulator for all its models):

    python -m ispatialgym.grasping.pipeline export     --jobs M.json [--shard i/n] [--only ID ...]   # GPU: collision geometry
    python -m ispatialgym.grasping.pipeline candidates --jobs M.json [--workers 8]                  # CPU: candidates + families
    python -m ispatialgym.grasping.pipeline validate   --jobs M.json [--shard i/n]                  # GPU: rests + physics trials
    python -m ispatialgym.grasping.pipeline all        --jobs M.json [--only ID ...]                # the three, one process (pilots)
    python -m ispatialgym.grasping.pipeline lean       --jobs M.json [--shard i/n] [--arm left]     # GPU: the lean jobs end to end
    python -m ispatialgym.grasping.pipeline overhang   --jobs M.json [--shard i/n]                  # GPU: the overhang jobs end to end
    python -m ispatialgym.grasping.pipeline summary    --jobs M.json

Floating jobs (``constraint`` "floating", i7-floating 2026-11-07; ``floating.floating_job``): export -> GraspGen in its own env
(``scripts/grasp_bands/graspgen_floating.py``) -> candidates (``floating.generate``) -> validate (``validate_floating``: in hand, gravity,
the held motions); ``scripts/grasp_bands/job_floating.sh`` runs the four. No settle survey, no rest retry.
Lean rests (i7-lean 2026-11-07, :mod:`ispatialgym.grasping.lean`; jobs from ``python -m ispatialgym.grasping.lean manifest``): the
``lean`` stage settles each lean job's object against the vertical board (``TabletopWorld.settle(lean=)``) and writes
``<entry>/lean_settle.json``; the candidates are then generated at that MEASURED rest with the board as an obstacle and validated with the
board present. ``all`` runs it between export and candidates for the manifest's lean jobs; ``lean`` runs export + lean settle +
candidates + validate for the lean jobs only (``job_glib.sh`` ``GB_STAGES=lean``). Lean jobs skip the settle-first survey and the
measured-rest retry (a lean rest that leans elsewhere keeps its ``<tag>_other`` record and serves nothing).

Overhang rests (i7-overhang 2026-11-07, :mod:`ispatialgym.grasping.overhang`; jobs from ``python -m ispatialgym.grasping.overhang manifest``):
the ``overhang`` stage settles each overhang job's object on the slab whose edge lies under it (``TabletopWorld.settle(overhang=)``) and writes
``<entry>/overhang_settle.json``; the candidates are then generated at that MEASURED rest with the slab box as the support and validated on
that slab. ``all`` runs it between export and candidates for the manifest's overhang jobs; ``overhang`` runs export + overhang settle +
candidates + validate for the overhang jobs only (``job_glib.sh`` ``GB_STAGES=overhang``). Overhang jobs skip the settle-first survey and the
measured-rest retry.

Left-arm lean validation (i7-overhang 2026-11-07, owner decision; lean rests only): ``lean --arm left`` runs the validation world with the
LEFT arm and the object mirrored to the robot's left (``validate_tabletop.mirrored_offsets``, as ``validate_artic motions --arm left``),
generates the lean jobs' candidates for the left hand and validates only the manifest's lean jobs whose ``validate_arm`` is that arm
(``lean manifest --validate-arm left``: their own entries ``...__valleft``, the arm in the library and its identity). Every other stage
and job runs the right arm as before.

``--root`` (default ``ISPATIALGYM_GRASP_LIBRARY_V2`` or ``runlogs/grasp_library_v2``) holds ``models/<entry id>/``; finished stages are
skipped per model (``--force`` redoes them). Every stage appends one json line per model to ``<root>/logs/<stage>_<shard>.jsonl``
(timings, counts, errors) and prints ``GB_<STAGE>_DONE``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

from ispatialgym.grasping import schema as S


def _jobs(args) -> List[Dict[str, Any]]:
    with open(args.jobs) as f:
        jobs = json.load(f)
    if args.only:
        want = set(args.only)
        jobs = [j for j in jobs if j["id"] in want or j.get("legacy_key") in want]
    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        jobs = [j for k, j in enumerate(jobs) if k % n == i]
    return jobs


def _log(args, stage: str, rec: Dict[str, Any]) -> None:
    root = S.library_root(args.root)
    os.makedirs(os.path.join(root, "logs"), exist_ok=True)
    tag = (args.shard or "0/1").replace("/", "of")
    with open(os.path.join(root, "logs", f"{stage}_{tag}.jsonl"), "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print(f"GB_{stage.upper()} {json.dumps(rec, default=str)[:600]}", flush=True)


def stage_export(args, jobs, world=None):
    from ispatialgym.grasping import object_geometry as OG

    todo = [j for j in jobs if args.force or OG.load(S.job_dir(j, args.root)) is None]
    if not todo:
        print("GB_EXPORT_DONE (nothing to do)", flush=True)
        return world
    if world is None:
        world = _world(args)
    for j in todo:
        t0 = time.time()
        d = S.job_dir(j, args.root)
        os.makedirs(d, exist_ok=True)
        try:
            obj = world.load(j, d)  # exports when missing
            if args.force:
                OG.export_loaded(obj, d, j)
            world.unload(obj)
            _log(args, "export", {"id": j["id"], "ok": True, "s": round(time.time() - t0, 2)})
        except Exception as e:  # noqa: BLE001
            _log(args, "export", {"id": j["id"], "ok": False, "error": repr(e), "tb": traceback.format_exc()[-800:]})
    print("GB_EXPORT_DONE", flush=True)
    return world


def candidate_overrides(job: Optional[Dict[str, Any]] = None):
    """of-v2maps 2026-11-07 (owner: retry the objects without feasible grasps with more candidates + time, recorded): opt-in
    ``ISG_GLIB_CAND_OVERRIDES`` (JSON of ``candidates.CandidateConfig`` fields, e.g. n_candidates / n_orientations / max_representatives /
    topk) and ``ISG_GLIB_DENSIFY`` ("4x2,8x4,16x8": the densify passes, candidate x orientation multipliers; default (4, 2), (8, 4)).
    i7-plate 2026-11-07: a manifest entry's own ``cand_overrides`` (same fields, e.g. ``{"families": ["lip_shallow", "rim_straddle"]}``)
    applies to that entry only, over the run's. Returns ``(CandidateConfig, densify)``."""
    from dataclasses import asdict

    from ispatialgym.grasping import candidates as CA

    base = CA.CandidateConfig()
    ov = json.loads(os.environ.get("ISG_GLIB_CAND_OVERRIDES") or "{}")
    ov = {**ov, **((job or {}).get("cand_overrides") or {})}
    unknown = sorted(set(ov) - set(asdict(base)))
    if unknown:
        raise ValueError(f"unknown CandidateConfig fields in the candidate overrides: {unknown}")
    cfg = CA.CandidateConfig(**{**asdict(base), **ov}) if ov else base
    dz = os.environ.get("ISG_GLIB_DENSIFY", "").strip()
    densify = tuple(tuple(int(v) for v in p.lower().split("x")) for p in dz.split(",") if p) if dz else ((4, 2), (8, 4))
    return cfg, densify


def validate_overrides():
    """Opt-in ``ISG_GLIB_VAL_OVERRIDES`` (JSON of ``validate_tabletop.ValidateConfig`` fields, e.g. max_trials / max_valid)."""
    from ispatialgym.grasping import validate_tabletop as VT

    ov = json.loads(os.environ.get("ISG_GLIB_VAL_OVERRIDES") or "{}")
    return VT.ValidateConfig(**ov)


def _cand_one(payload):
    j, root, force = payload[:3]
    from ispatialgym.grasping import candidates as CA
    from ispatialgym.grasping import lean as LN

    d = S.job_dir(j, root)
    j = LN.resolve_job(j, d)  # i7-lean: a lean job's measured rest from its lean_settle.json
    from ispatialgym.grasping import overhang as OV

    j = OV.resolve_job(j, d)  # i7-overhang: an overhang job's measured rest from its overhang_settle.json
    t0 = time.time()
    if not force and os.path.exists(os.path.join(d, "candidates.json")):
        return {"id": j["id"], "ok": True, "skipped": True}
    if (j.get("constraint") or "tabletop") == S.FLOATING:
        return _cand_floating(j, d, t0)
    try:
        cfg, densify = candidate_overrides(j)
        # i7-overhang: a lean job validated with the left arm gets the left hand's candidates (every other job: the right hand as before)
        arm = LN.job_arm(j) if LN.is_lean_job(j) else "right"
        out = CA.generate(j, d, cfg=cfg, densify=densify, arm=arm)
        n_ok = sum(1 for c in out["candidates"] if c["geometry"]["ok"])
        fams = out["families"]
        pools = tuple(cfg.modes) + tuple(f for f in cfg.families if f not in CA.POSE_FAMILIES)
        by_pool: Dict[str, Dict[str, Any]] = {p: {"n": 0, "geometry_ok": 0, "reasons": {}} for p in pools}
        for c in out["candidates"]:  # i7-plate: per pool (mode / family) the selected candidates and the gate each one failed
            b = by_pool.setdefault(c.get("family") or c["mode"], {"n": 0, "geometry_ok": 0, "reasons": {}})
            b["n"] += 1
            b["geometry_ok"] += int(bool(c["geometry"]["ok"]))
            b["reasons"][c["geometry"]["reason"]] = b["reasons"].get(c["geometry"]["reason"], 0) + 1
        return {"id": j["id"], "ok": True, "s": round(time.time() - t0, 2), "n_candidates": len(out["candidates"]), "n_geometry_ok": n_ok,
                "cand_config": {k: getattr(cfg, k) for k in ("n_candidates", "n_orientations", "cone_deg", "max_representatives", "topk", "families",
                                                              "blade_contact_at_retreat", "retreat_shift_exact", "sweep_from_preshape")},
                "densify": [list(x) for x in densify], "densified_from": (out["log"] or {}).get("densified_from") is not None,
                "n_representatives": len(out["representatives"]), "n_families": len(fams),
                "family_types": sorted({f.get("type") for f in fams.values()}),
                # i7-ladder: a given-pose family logs one pool per proposal / legacy grasp (agent_p<i>[_side], legacy_<name>[_side])
                "log": {k: out["log"][k] for k in out["log"] if k in pools + ("n_sampled",) or (isinstance(out["log"][k], dict) and "source" in out["log"][k])},
                "by_pool": by_pool, "gate_dump": CA.GATE_DUMP_FILE if cfg.gate_dump else None,
                "reasons": sorted({c["geometry"]["reason"] for c in out["candidates"] if not c["geometry"]["ok"]})}
    except Exception as e:  # noqa: BLE001
        return {"id": j["id"], "ok": False, "error": repr(e), "tb": traceback.format_exc()[-800:]}


def floating_overrides(job: Optional[Dict[str, Any]] = None):
    """i7-floating 2026-11-07: the floating candidates' config (``floating.FloatingConfig``), opt-in ``ISG_GLIB_FLOAT_OVERRIDES`` (JSON of its
    fields) and a manifest entry's own ``cand_overrides`` over it."""
    from dataclasses import asdict

    from ispatialgym.grasping import floating as FL

    base = FL.FloatingConfig()
    ov = json.loads(os.environ.get("ISG_GLIB_FLOAT_OVERRIDES") or "{}")
    ov = {**ov, **((job or {}).get("cand_overrides") or {})}
    unknown = sorted(set(ov) - set(asdict(base)))
    if unknown:
        raise ValueError(f"unknown FloatingConfig fields in the floating overrides: {unknown}")
    return FL.FloatingConfig(**{**asdict(base), **ov}) if ov else base


def _cand_floating(j, d, t0):
    """i7-floating 2026-11-07: a ``floating`` job's candidates from its GraspGen output (``floating.generate``)."""
    from ispatialgym.grasping import floating as FL

    try:
        cfg = floating_overrides(j)
        out = FL.generate(j, d, cfg=cfg)
        lg = out["log"]["graspgen"]
        return {"id": j["id"], "ok": True, "s": round(time.time() - t0, 2), "constraint": S.FLOATING, "floating_mode": out["floating_mode"],
                "n_candidates": len(out["candidates"]), "n_geometry_ok": len(out["candidates"]), "n_representatives": len(out["representatives"]),
                "gates": {k: v for k, v in lg.items() if k.startswith("n_")}, "gate_dump": CA_GATE_DUMP if cfg.gate_dump else None}
    except Exception as e:  # noqa: BLE001
        return {"id": j["id"], "ok": False, "constraint": S.FLOATING, "error": repr(e), "tb": traceback.format_exc()[-800:]}


CA_GATE_DUMP = "gate_dump.json"


def stage_candidates(args, jobs):
    payloads = [(j, args.root, args.force) for j in jobs]
    if args.workers > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(args.workers) as pool:
            for rec in pool.imap_unordered(_cand_one, payloads):
                _log(args, "candidates", rec)
    else:
        for p in payloads:
            _log(args, "candidates", _cand_one(p))
    print("GB_CANDIDATES_DONE", flush=True)


def _world(args):
    from dataclasses import replace

    from ispatialgym.grasping import validate_tabletop as VT

    cfg = validate_overrides()  # of-v2maps 2026-11-07: ISG_GLIB_VAL_OVERRIDES (default: ValidateConfig())
    arm = getattr(args, "arm", None) or "right"
    if arm != "right":
        # i7-overhang 2026-11-07 (owner: a left-arm validator for lean rests): the left arm, the object mirrored to the robot's left
        cfg = replace(cfg, arm=arm, base_offsets=VT.mirrored_offsets(cfg.base_offsets))
    return VT.TabletopWorld(cfg)


def stage_validate(args, jobs, world=None):
    from ispatialgym.grasping import validate_tabletop as VT

    todo = [j for j in jobs if args.force or not os.path.exists(os.path.join(S.job_dir(j, args.root), "library.json"))]
    if not todo:
        print("GB_VALIDATE_DONE (nothing to do)", flush=True)
        return world
    if world is None:
        world = _world(args)
    from ispatialgym.grasping import lean as LN

    for j in todo:
        d = S.job_dir(j, args.root)
        j = LN.resolve_job(j, d)  # i7-lean: a lean job's measured rest from its lean_settle.json
        from ispatialgym.grasping import overhang as OV

        j = OV.resolve_job(j, d)  # i7-overhang: an overhang job's measured rest from its overhang_settle.json
        try:
            if (j.get("constraint") or "tabletop") == S.FLOATING:  # i7-floating: in hand, gravity, the held motions (validate_floating)
                from ispatialgym.grasping import validate_floating as VF

                rec = VF.run_model(world, j, d, cfg=floating_validate_overrides())
            else:
                rec = VT.run_model(world, j, d)
            rec["ok"] = True
        except Exception as e:  # noqa: BLE001
            rec = {"id": j["id"], "ok": False, "error": repr(e), "tb": traceback.format_exc()[-1500:]}
        _log(args, "validate", rec)
    print("GB_VALIDATE_DONE", flush=True)
    return world


def floating_validate_overrides():
    """i7-floating 2026-11-07: opt-in ``ISG_GLIB_FLOAT_VAL_OVERRIDES`` (JSON of ``validate_floating.FloatingValidateConfig`` fields)."""
    from ispatialgym.grasping import validate_floating as VF

    ov = json.loads(os.environ.get("ISG_GLIB_FLOAT_VAL_OVERRIDES") or "{}")
    return VF.FloatingValidateConfig(**ov)


def settle_first_on() -> bool:
    """Opt-in ``GB_SETTLE_FIRST=1`` (of-v2maps 2026-11-07): the settle survey runs before the candidates (:mod:`settle_first`)."""
    return os.environ.get("GB_SETTLE_FIRST", "") == "1"


def stage_settle(args, jobs, world=None):
    """``GB_SETTLE_FIRST``: per job, :data:`settle_first.SURVEY_YAWS_DEG` x :data:`settle_first.SURVEY_DROPS_M` drops from the designated
    rest (``TabletopWorld.settle_survey``), clustered into distinct rests (``settle_first.cluster_rests``) and written to
    ``<job dir>/settle_first.json``; returns the jobs to generate and validate (``settle_first.jobs_for_rests``: the job at its most
    frequent stable rest under its own id, a derived job per other stable rest seen twice or more) and the world. A survey that finds
    no stable rest keeps the job as it was (designated)."""
    import numpy as np

    from ispatialgym.grasping import object_geometry as OG
    from ispatialgym.grasping import settle_first as SF

    from ispatialgym.grasping import lean as LN

    if world is None:
        world = _world(args)
    cfg = world.cfg
    out: List[Dict[str, Any]] = []
    from ispatialgym.grasping import overhang as OV

    for j in jobs:
        # i7-floating: a floating object is never set down; i7-lean / i7-overhang: settled by their own stage, never surveyed on the bare slab
        if (j.get("constraint") or "tabletop") == S.FLOATING or LN.is_lean_job(j) or OV.is_overhang_job(j):
            out.append(j)
            continue
        d = S.job_dir(j, args.root)
        t0 = time.time()
        path = os.path.join(d, "settle_first.json")
        try:
            doc = None if args.force else S.load_json(path)
            if doc is None:
                obj = world.load(j, d)
                geom = OG.load(d)
                hint = j.get("rest_hint_wxyz")
                q_rest = np.array([0, 0, 0, 1.0]) if hint is None else np.array([hint[1], hint[2], hint[3], hint[0]])
                quat0 = S.quat_mul(S.quat_yaw(cfg.reference_yaw), q_rest)
                try:
                    outcomes = world.settle_survey(obj, geom, quat0, SF.survey_plan())
                finally:
                    world.unload(obj)
                clusters = SF.cluster_rests(outcomes, geom.union.vertices, designated_xyzw=quat0)
                doc = {"schema": "isg-settle-first/1", "id": j["id"], "method": SF.METHOD, "designated_quat_xyzw": [round(float(v), 7) for v in quat0],
                       "plan": [{"yaw0": round(y, 5), "drop_m": dm} for y, dm in SF.survey_plan()], "clusters": clusters,
                       "outcomes": [{k: o.get(k) for k in ("yaw0", "drop_m", "quat_xyzw", "h_root", "stable", "tilt_from_intended_rad", "settle_steps",
                                                         "lowest_point_gap_m", "support_load_N", "weight_N")} for o in outcomes],
                       "wall_s": round(time.time() - t0, 2), "created": S.now()}
                os.makedirs(d, exist_ok=True)
                S.save_json(path, doc)
            new = SF.jobs_for_rests(j, doc["clusters"]) or [j]
            for nj in new:
                if nj.get("derived_from"):
                    nd = S.job_dir(nj, args.root)
                    os.makedirs(nd, exist_ok=True)
                    S.save_json(os.path.join(nd, "job.json"), nj)
                    geo = os.path.join(d, "geometry.json")
                    if os.path.exists(geo) and not os.path.exists(os.path.join(nd, "geometry.json")):
                        import shutil

                        shutil.copy2(geo, os.path.join(nd, "geometry.json"))
            out.extend(new)
            _log(args, "settle", {"id": j["id"], "ok": True, "s": doc.get("wall_s"), "rests": [(c["rest_id"], c["n"], c["n_stable"], c["h_root"],
                                                                                              c["tilt_from_designated_rad"], len(c["roll_merged"])) for c in doc["clusters"]],
                                  "jobs": [nj["id"] for nj in new]})
        except Exception as e:  # noqa: BLE001
            out.append(j)
            _log(args, "settle", {"id": j["id"], "ok": False, "error": repr(e), "tb": traceback.format_exc()[-800:]})
    print("GB_SETTLE_DONE", flush=True)
    return out, world


def stage_lean(args, jobs, world=None):
    """i7-lean 2026-11-07: per lean job, the object settled against the vertical board (``TabletopWorld.settle(lean=)`` from the job's
    upright base leaned to its spec: ``lean.lean_quat``), the measured lean rest written to ``<entry>/lean_settle.json`` (schema
    ``lean.SETTLE_SCHEMA``: the spec, the start, the rest record); returns the jobs (each lean job with its measured rest:
    ``lean.job_at_rest``) and the world. Other jobs pass unchanged."""
    import numpy as np

    from ispatialgym.grasping import lean as LN
    from ispatialgym.grasping import object_geometry as OG

    out: List[Dict[str, Any]] = []
    for j in jobs:
        if not LN.is_lean_job(j):
            out.append(j)
            continue
        d = S.job_dir(j, args.root)
        path = os.path.join(d, LN.SETTLE_FILE)
        t0 = time.time()
        try:
            doc = None if args.force else S.load_json(path)
            if doc is None:
                if world is None:
                    world = _world(args)
                spec = LN.spec_of(j)
                quat0 = LN.lean_quat(LN.base_quat(j), spec)
                obj = world.load(j, d)
                try:
                    rest, _ = world.settle(obj, OG.load(d), quat0, lean=spec)
                finally:
                    world.unload(obj)
                ok = bool(rest["lean"]["lean_ok"]) and bool(rest["stable"])
                rest["rest_id"] = spec.rest_id() if ok else spec.rest_id() + "_other"
                doc = {"schema": LN.SETTLE_SCHEMA, "id": j["id"], "lean": spec.to_json(), "start_quat_xyzw": [round(float(v), 7) for v in np.asarray(quat0)],
                       "lean_base_wxyz": j.get("lean_base_wxyz"), "lean_base": j.get("lean_base"), "rest": rest, "ok": ok,
                       "wall_s": round(time.time() - t0, 2), "created": S.now()}
                os.makedirs(d, exist_ok=True)
                S.save_json(path, doc)
            out.append(LN.job_at_rest(j, doc["rest"]))
            r = doc["rest"]
            _log(args, "lean", {"id": j["id"], "ok": True, "lean_ok": doc.get("ok"), "rest_id": r.get("rest_id"), "measured_deg": r["lean"].get("measured_deg"),
                                "side": r["lean"].get("measured_side"), "h_root": r.get("h_root"), "board_face_y_m": r["lean"].get("board_face_y_m"),
                                "board_load_ratio": r["lean"].get("board_load_ratio"), "why": r["lean"].get("lean_why"), "s": doc.get("wall_s")})
        except Exception as e:  # noqa: BLE001
            out.append(j)
            _log(args, "lean", {"id": j["id"], "ok": False, "error": repr(e), "tb": traceback.format_exc()[-800:]})
    print("GB_LEAN_DONE", flush=True)
    return out, world


def stage_overhang(args, jobs, world=None):
    """i7-overhang 2026-11-07: per overhang job, the object settled on the slab whose edge lies under it (``TabletopWorld.settle(overhang=)``
    from the job's designated rest at the reference yaw; ``overhang.slab_for``: the named face ``distance_m`` beyond the edge), the measured
    overhang rest written to ``<entry>/overhang_settle.json`` (schema ``overhang.SETTLE_SCHEMA``: the spec, the start, the rest record);
    returns the jobs (each overhang job with its measured rest: ``overhang.job_at_rest``) and the world. Other jobs pass unchanged."""
    import numpy as np

    from ispatialgym.grasping import object_geometry as OG
    from ispatialgym.grasping import overhang as OV

    out: List[Dict[str, Any]] = []
    for j in jobs:
        if not OV.is_overhang_job(j):
            out.append(j)
            continue
        d = S.job_dir(j, args.root)
        path = os.path.join(d, OV.SETTLE_FILE)
        t0 = time.time()
        try:
            doc = None if args.force else S.load_json(path)
            if doc is None:
                if world is None:
                    world = _world(args)
                spec = OV.spec_of(j)
                hint = j.get("rest_hint_wxyz")
                q_rest = np.array([0, 0, 0, 1.0]) if hint is None else np.array([hint[1], hint[2], hint[3], hint[0]])
                quat0 = S.quat_mul(S.quat_yaw(world.cfg.reference_yaw), q_rest)
                obj = world.load(j, d)
                try:
                    rest, _ = world.settle(obj, OG.load(d), quat0, overhang=spec)
                finally:
                    world.unload(obj)
                ok = bool(rest["overhang"]["overhang_ok"]) and bool(rest["stable"])
                rest["rest_id"] = spec.rest_id() if ok else spec.rest_id() + "_other"
                doc = {"schema": OV.SETTLE_SCHEMA, "id": j["id"], "overhang": spec.to_json(), "start_quat_xyzw": [round(float(v), 7) for v in np.asarray(quat0)],
                       "overhang_base": j.get("overhang_base"), "rest": rest, "ok": ok, "wall_s": round(time.time() - t0, 2), "created": S.now()}
                os.makedirs(d, exist_ok=True)
                S.save_json(path, doc)
            out.append(OV.job_at_rest(j, doc["rest"]))
            r = doc["rest"]
            _log(args, "overhang", {"id": j["id"], "ok": True, "overhang_ok": doc.get("ok"), "rest_id": r.get("rest_id"),
                                    "measured_distance_m": r["overhang"].get("measured_distance_m"), "com_inside_m": r["overhang"].get("com_inside_m"),
                                    "edge_along_m": r["overhang"].get("edge_along_m"), "flush_turn_deg": r["overhang"].get("flush_turn_deg"),
                                    "h_root": r.get("h_root"), "stable": r.get("stable"), "why": r["overhang"].get("overhang_why"), "s": doc.get("wall_s")})
        except Exception as e:  # noqa: BLE001
            out.append(j)
            _log(args, "overhang", {"id": j["id"], "ok": False, "error": repr(e), "tb": traceback.format_exc()[-800:]})
    print("GB_OVERHANG_DONE", flush=True)
    return out, world


def rest_retry_jobs(args, jobs) -> List[Dict[str, Any]]:
    """Jobs whose object settled in a rest other than the designated one (``rests.json``: ``intended`` False) and got no valid grasp:
    derived jobs designated at that MEASURED rest (``rest_hint_wxyz`` = its quaternion, wxyz), candidates generated for it - a new model
    with an unstable canonical pose gets its grasps on the rest it really takes."""
    out = []
    for j in jobs:
        # i7-floating: no support, no rest to retry at; i7-lean / i7-overhang: never retried as a stable rest
        if (j.get("constraint") or "tabletop") == S.FLOATING or j.get("rest_kind") in ("lean", "overhang"):
            continue
        d = S.job_dir(j, args.root)
        lib = S.load_json(os.path.join(d, "library.json")) or {}
        if lib.get("grasps"):
            continue
        rests = (S.load_json(os.path.join(d, "rests.json")) or {}).get("rests", {})
        other = [r for r in rests.values() if not r.get("intended", True) and r.get("stable")]
        if not other or j.get("derived_from"):
            continue
        x, y, z, w = other[0]["quat_xyzw"]
        nj = dict(j, rest_hint_wxyz=[round(w, 6), round(x, 6), round(y, 6), round(z, 6)], rest_pose="measured_other_rest", derived_from=j["id"])
        nj["id"] = S.entry_id(nj["category"], nj["model"], nj["scale"], nj.get("constraint") or "tabletop", nj["rest_hint_wxyz"])
        if os.path.exists(os.path.join(S.job_dir(nj, args.root), "library.json")) and not args.force:
            continue
        os.makedirs(S.job_dir(nj, args.root), exist_ok=True)
        S.save_json(os.path.join(S.job_dir(nj, args.root), "job.json"), nj)  # the parent: reach maps shard a derived job with it
        geo = os.path.join(d, "geometry.json")
        if os.path.exists(geo):  # the same object: its exported geometry is reused
            import shutil

            shutil.copy2(geo, os.path.join(S.job_dir(nj, args.root), "geometry.json"))
        out.append(nj)
    return out


def summary(args, jobs):
    rows = []
    for j in jobs:
        d = S.job_dir(j, args.root)
        c = S.load_json(os.path.join(d, "candidates.json")) or {}
        v = S.load_json(os.path.join(d, "validation.json")) or {}
        lib = S.load_json(os.path.join(d, "library.json")) or {}
        tr = v.get("trials", [])
        rows.append({"id": j["id"], "legacy_key": j.get("legacy_key"), "cands": len(c.get("candidates", [])),
                     "geo_ok": sum(1 for x in c.get("candidates", []) if x["geometry"]["ok"]), "reps": len(c.get("representatives", [])),
                     "families": {t: sum(1 for f in (c.get("families") or {}).values() if f.get("type") == t) for t in ("rotational", "translational", "none")},
                     "tested": sum(1 for t in tr if "ag" in t), "valid": len(lib.get("grasps", [])),
                     "modes_valid": sorted({g.get("mode") for g in lib.get("grasps", [])}),
                     "rest": {k: (r.get("stable"), r.get("h_root")) for k, r in (lib.get("rests") or {}).items()},
                     "fail": sorted({r for t in tr if t.get("verdict") == "invalid" for r in (t.get("reasons") or [])}),
                     "model_s": (v.get("timing") or {}).get("model_s")})
    for r in rows:
        print(json.dumps(r))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["export", "settle", "lean", "overhang", "candidates", "validate", "retry", "all", "summary"])
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--arm", default="right", choices=["right", "left"],
                    help="the validation world's arm (i7-overhang: left for the lean stage only - the lean jobs whose validate_arm is left)")
    args = ap.parse_args(argv)
    if args.arm != "right" and args.stage not in ("lean", "summary"):
        raise SystemExit(f"--arm {args.arm}: the left-arm validator is for the lean stage only (owner 2026-11-07: lean rests only)")
    jobs = _jobs(args)
    if args.stage == "summary":
        summary(args, jobs)
        return
    world = None
    if args.stage == "lean":
        # i7-lean 2026-11-07: the manifest's lean jobs end to end (export, the settle against the board, candidates, validation)
        from ispatialgym.grasping import lean as LN

        jobs = [j for j in jobs if LN.is_lean_job(j)]
        # i7-overhang: the lean jobs validated with this run's arm (``--arm``; a job's ``validate_arm``, right when absent)
        skipped = [j["id"] for j in jobs if LN.job_arm(j) != args.arm]
        jobs = [j for j in jobs if LN.job_arm(j) == args.arm]
        print(f"GB_LEAN_JOBS {len(jobs)} (arm {args.arm}{f'; other arms skipped: {skipped}' if skipped else ''})", flush=True)
    if args.stage == "overhang":
        # i7-overhang 2026-11-07: the manifest's overhang jobs end to end (export, the settle at the slab edge, candidates, validation)
        from ispatialgym.grasping import overhang as OV

        jobs = [j for j in jobs if OV.is_overhang_job(j)]
        print(f"GB_OVERHANG_JOBS {len(jobs)}", flush=True)
    if args.stage in ("export", "all", "lean", "overhang"):
        world = stage_export(args, jobs, world)
    if args.stage == "settle" or (args.stage == "all" and settle_first_on()):
        jobs, world = stage_settle(args, jobs, world)  # of-v2maps 2026-11-07: GB_SETTLE_FIRST - the measured rests first
    if args.stage in ("all", "lean"):
        jobs, world = stage_lean(args, jobs, world)  # i7-lean: the lean jobs' measured rests (other jobs pass unchanged)
    if args.stage in ("all", "overhang"):
        jobs, world = stage_overhang(args, jobs, world)  # i7-overhang: the overhang jobs' measured rests (other jobs pass unchanged)
    if args.stage in ("candidates", "all", "lean", "overhang"):
        stage_candidates(args, jobs)
    if args.stage in ("validate", "all", "lean", "overhang"):
        world = stage_validate(args, jobs, world)
    if args.stage in ("retry", "all"):
        derived = rest_retry_jobs(args, jobs)
        print(f"GB_RETRY {len(derived)} jobs at a measured other rest", flush=True)
        if derived:
            stage_candidates(args, derived)
            world = stage_validate(args, derived, world)
    if world is not None:
        import omnigibson as og

        og.shutdown()


if __name__ == "__main__":
    main()
