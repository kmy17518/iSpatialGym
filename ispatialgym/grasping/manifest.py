"""Job manifests of the grasp-library v2 pipeline: one job per (category, model, scale, constraint, designated rest).

Sources:

* ``--legacy-keys K1 K2 ...`` - models of existing library keys (``runlogs/grasp_library_20260905``): their constraint, scale and
  designated rest (``rest_quat_wxyz`` / ``rest_pose`` of the key's candidates) are carried over, the key is recorded as
  ``legacy_key`` (what the v2 entry replaces);
* ``--candidates-csv`` - every sampleable target row of ``assets/icl/roles/candidates.csv`` (the ~1000-model universe): rows with a
  ``grasp_key`` as above, rows without one as NEW models (``tabletop``, canonical rest - no recorded physics settings);
* ``--isg1020`` - the 16 tabletop / attachment targets of the isg-1020 aliases (the validation set);
* ``--floating-keys K1 K2 ...`` - legacy floating (GraspGen) keys of the held-object tasks -> ``floating`` jobs (i7-floating 2026-11-07,
  :func:`floating_job_of`): entry ``<category>__<model>__<scale tag>__floating``, ``legacy_key`` = the key.

    python -m ispatialgym.grasping.manifest --isg1020 --out runlogs/grasp_library_v2/manifests/isg1020.json
    python -m ispatialgym.grasping.manifest --candidates-csv ispatialgym/assets/icl/roles/candidates.csv --out .../all.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence

from ispatialgym import REPO_ROOT
from ispatialgym.grasping import schema as S

LEGACY_ROOT = os.path.join(REPO_ROOT, "runlogs", "grasp_library_20260905")

#: isg-1020's tabletop / attachment grasp keys (13 IORS + 3 attachment aliases; RECOVERY.md, recovery/train85/manifests/counted.tsv)
ISG1020_KEYS = (
    "brussels_sprouts__mbkrxe__i", "sunflower__dkajbh__r50", "gym_shoe__kmcbym__t", "salt_shaker__fonexa__t", "tablespoon__yxxfdj__st",
    "bar_soap__utgixp", "pack_of_kielbasa__dgodnn__t", "box_of_cereal__iucshm__s080", "box_of_sanitary_napkins__humkqw", "beer_glass__mxsliu__t2f",
    "plate__qtfzeq__st", "spatula__hkvsep__t076", "teacup__wopjex__t", "cap__xxqyrt", "lid__pbmsji__s065", "lid__tnwisp",
)


def legacy_job(key: str, root: str = LEGACY_ROOT) -> Optional[Dict[str, Any]]:
    import yaml

    res = S.load_json(os.path.join(root, "results", f"{key}.json")) or {}
    cpath = os.path.join(root, "cands", f"{key}.yaml")
    meta: Dict[str, Any] = {}
    if os.path.exists(cpath):
        with open(cpath) as f:
            meta = (yaml.safe_load(f) or {}).get("ispatialgym") or {}
    cat = res.get("category") or (meta.get("object") or {}).get("category")
    model = res.get("model") or (meta.get("object") or {}).get("model")
    if not cat or not model:
        parts = key.split("__")
        cat, model = parts[0], parts[1]
    constraint = res.get("constraint") or meta.get("constraint") or "tabletop"
    if res.get("pipeline") == "graspgen" or meta.get("pipeline") == "graspgen":
        return None  # floating (GraspGen) keys: out of the tabletop / attachment scope
    rest = meta.get("rest_quat_wxyz") if meta.get("rest_pose", "canonical") != "canonical" else None
    return {"category": cat, "model": model, "scale": list(S.canonical_scale(res.get("scale"))), "dataset": res.get("dataset") or "behavior-1k-assets",
            "constraint": constraint, "rest_hint_wxyz": rest, "rest_pose": meta.get("rest_pose", "canonical"), "legacy_key": key,
            "kind": "attachment" if constraint == "attachment_free" else "tabletop", "source": "legacy_key"}


def floating_job_of(key: str, root: str = LEGACY_ROOT, held_tasks: Optional[Sequence[str]] = None) -> Optional[Dict[str, Any]]:
    """i7-floating 2026-11-07: the v2 ``floating`` job of a legacy floating (GraspGen) key - its model, spawn scale, dataset, legacy mode and
    held task(s) read from ``results/<key>.json`` (metadata only: the legacy grasps are not used; ``floating.floating_job``). None for a
    key of another pipeline."""
    from ispatialgym.grasping import floating as FL

    res = S.load_json(os.path.join(root, "results", f"{key}.json")) or {}
    if res.get("pipeline") != "graspgen":
        return None
    tasks = list(held_tasks) if held_tasks else [t for t in [res.get("task")] if t in ("mental_rotation", "object_scaling")]
    return FL.floating_job(key, res, tasks)


def from_candidates_csv(path: str) -> List[Dict[str, Any]]:
    jobs = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("role") not in ("target",) or row.get("kind") != "sampleable":
                continue
            key = (row.get("grasp_key") or "").strip()
            j = legacy_job(key) if key else None
            if key and j is None:
                # a floating (GraspGen) key: the object still starts on a support and is picked from it -> a tabletop job of its own
                # (not a replacement of the floating key: no legacy_key; the key's scale is kept)
                res = S.load_json(os.path.join(LEGACY_ROOT, "results", f"{key}.json")) or {}
                j = {"category": row["category"], "model": row["model"], "scale": list(S.canonical_scale(res.get("scale"))),
                     "dataset": row.get("dataset") or res.get("dataset") or "behavior-1k-assets", "constraint": "tabletop", "rest_hint_wxyz": None,
                     "rest_pose": "canonical", "legacy_key": None, "floating_key": key, "kind": "tabletop", "source": "floating_key"}
            if j is None:
                sc = row.get("scale") or ""
                j = {"category": row["category"], "model": row["model"], "scale": list(S.canonical_scale(json.loads(sc) if sc.startswith("[") else (float(sc) if sc else None))),
                     "dataset": row.get("dataset") or "behavior-1k-assets", "constraint": "tabletop", "rest_hint_wxyz": None, "rest_pose": "canonical",
                     "legacy_key": None, "kind": "tabletop", "source": "new_model"}
            j["subtasks"] = [row.get("subtask")]
            jobs.append(j)
    return merge(jobs)


def merge(jobs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for j in jobs:
        k = json.dumps([j["category"], j["model"], j["scale"], j["constraint"], j.get("rest_hint_wxyz")])
        if k in out:
            out[k].setdefault("subtasks", [])
            out[k]["subtasks"] = sorted(set(out[k]["subtasks"]) | set(j.get("subtasks") or []))
            continue
        out[k] = dict(j)
    res = list(out.values())
    for j in res:
        j["id"] = S.entry_id(j["category"], j["model"], j["scale"], j["constraint"], j.get("rest_hint_wxyz"))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--legacy-keys", nargs="*", default=[])
    ap.add_argument("--candidates-csv", nargs="*", default=None)
    ap.add_argument("--isg1020", action="store_true")
    ap.add_argument("--floating-keys", nargs="*", default=[], help="legacy floating (GraspGen) keys -> v2 floating jobs (i7-floating)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    jobs = []
    keys = list(a.legacy_keys) + (list(ISG1020_KEYS) if a.isg1020 else [])
    for k in keys:
        j = legacy_job(k)
        if j is not None:
            jobs.append(j)
    for c in a.candidates_csv or []:
        jobs += from_candidates_csv(c)
    jobs = merge(jobs)
    for k in a.floating_keys:  # i7-floating: ids are their own (floating.floating_job), never merged with a tabletop job of the model
        j = floating_job_of(k)
        if j is None:
            raise SystemExit(f"{k}: not a legacy floating (graspgen) key")
        jobs.append(j)
    S.save_json(a.out, jobs)
    n_new = sum(1 for j in jobs if j["source"] == "new_model")
    print(f"{len(jobs)} jobs ({n_new} new models without a library key) -> {a.out}")


if __name__ == "__main__":
    main()
