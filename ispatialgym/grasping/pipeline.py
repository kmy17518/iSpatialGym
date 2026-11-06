"""Grasp library v2 pipeline - stages over a job manifest (:mod:`manifest`), each a persistent worker (one simulator for all its models):

    python -m ispatialgym.grasping.pipeline export     --jobs M.json [--shard i/n] [--only ID ...]   # GPU: collision geometry
    python -m ispatialgym.grasping.pipeline candidates --jobs M.json [--workers 8]                  # CPU: candidates + families
    python -m ispatialgym.grasping.pipeline validate   --jobs M.json [--shard i/n]                  # GPU: rests + physics trials
    python -m ispatialgym.grasping.pipeline all        --jobs M.json [--only ID ...]                # the three, one process (pilots)
    python -m ispatialgym.grasping.pipeline summary    --jobs M.json

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


def _cand_one(payload):
    j, root, force = payload
    from ispatialgym.grasping import candidates as CA

    d = S.job_dir(j, root)
    t0 = time.time()
    if not force and os.path.exists(os.path.join(d, "candidates.json")):
        return {"id": j["id"], "ok": True, "skipped": True}
    try:
        out = CA.generate(j, d)
        n_ok = sum(1 for c in out["candidates"] if c["geometry"]["ok"])
        fams = out["families"]
        return {"id": j["id"], "ok": True, "s": round(time.time() - t0, 2), "n_candidates": len(out["candidates"]), "n_geometry_ok": n_ok,
                "n_representatives": len(out["representatives"]), "n_families": len(fams),
                "family_types": sorted({f.get("type") for f in fams.values()}), "log": {k: out["log"][k] for k in out["log"] if k in ("top", "side", "n_sampled")},
                "reasons": sorted({c["geometry"]["reason"] for c in out["candidates"] if not c["geometry"]["ok"]})}
    except Exception as e:  # noqa: BLE001
        return {"id": j["id"], "ok": False, "error": repr(e), "tb": traceback.format_exc()[-800:]}


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
    from ispatialgym.grasping import validate_tabletop as VT

    return VT.TabletopWorld(VT.ValidateConfig())


def stage_validate(args, jobs, world=None):
    from ispatialgym.grasping import validate_tabletop as VT

    todo = [j for j in jobs if args.force or not os.path.exists(os.path.join(S.job_dir(j, args.root), "library.json"))]
    if not todo:
        print("GB_VALIDATE_DONE (nothing to do)", flush=True)
        return world
    if world is None:
        world = _world(args)
    for j in todo:
        d = S.job_dir(j, args.root)
        try:
            rec = VT.run_model(world, j, d)
            rec["ok"] = True
        except Exception as e:  # noqa: BLE001
            rec = {"id": j["id"], "ok": False, "error": repr(e), "tb": traceback.format_exc()[-1500:]}
        _log(args, "validate", rec)
    print("GB_VALIDATE_DONE", flush=True)
    return world


def rest_retry_jobs(args, jobs) -> List[Dict[str, Any]]:
    """Jobs whose object settled in a rest other than the designated one (``rests.json``: ``intended`` False) and got no valid grasp:
    derived jobs designated at that MEASURED rest (``rest_hint_wxyz`` = its quaternion, wxyz), candidates generated for it - a new model
    with an unstable canonical pose gets its grasps on the rest it really takes."""
    out = []
    for j in jobs:
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
    ap.add_argument("stage", choices=["export", "candidates", "validate", "retry", "all", "summary"])
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    jobs = _jobs(args)
    if args.stage == "summary":
        summary(args, jobs)
        return
    world = None
    if args.stage in ("export", "all"):
        world = stage_export(args, jobs, world)
    if args.stage in ("candidates", "all"):
        stage_candidates(args, jobs)
    if args.stage in ("validate", "all"):
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
