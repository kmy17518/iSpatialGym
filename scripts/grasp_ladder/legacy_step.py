"""Grasp ladder step 4 (last resort): an entry's LEGACY grasps (runlogs/grasp_library_20260905, the key's cands/<key>.yaml) as given
poses through the same v2 gates and physics validation (owner decision 2026-11-07). The owner decides before any alias uses them:

    legacy_step.py list   --batch B [--report B/report.json]     # the entries / aliases that would end here (after step 3), never builds
    legacy_step.py submit --batch B --code <snapshot> --out <new root> --owner-approved [--run]

``list`` reads the step-3 batch (entries.json, report.json): an entry that is still at 0 after step 3 (no proposals, infeasible, or 0
physics-valid) and has a legacy key with grasps would end at step 4; it prints the entry, its legacy key, the number of legacy grasps
(and how many the legacy validation held) and every customs alias that reads the key. ``submit`` refuses without ``--owner-approved``;
even then the build root is separate and nothing is linked or staged for any alias by this script.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from ispatialgym import REPO_ROOT  # noqa: E402
from ispatialgym.grasping import ladder as LD  # noqa: E402
from ispatialgym.grasping import schema as S  # noqa: E402


def step4_rows(batch, report=None, legacy_root=LD.LEGACY_ROOT):
    """The step-4 candidates of a step-3 batch: ``[{"id", "alias_key", "legacy_key", "after_step3", "n_legacy", "n_legacy_held",
    "aliases"}]`` for every entry still at 0 after step 3 (``after_step3``: why)."""
    entries = S.load_json(os.path.join(batch, "entries.json")) or []
    rep = {r["id"]: r for r in ((S.load_json(report or os.path.join(batch, "report.json")) or {}).get("rows") or [])}
    customs = S.load_json(os.path.join(REPO_ROOT, "ispatialgym", "assets", "task_specs", "isg_customs.json")) or {}
    out = []
    for e in entries:
        j = e["job"]
        r = rep.get(j["id"])
        if r is None:
            why = "step 3 not reported"
        elif r.get("proposals") is None:
            why = "no proposals"
        elif not r.get("feasible"):
            why = "proposer: infeasible"
        elif "n_valid" not in r:
            why = "proposals not built"
        elif r["n_valid"] > 0:
            continue  # step 3 produced grasps: the entry does not reach step 4
        else:
            why = "0 physics-valid after step 3"
        key = j.get("legacy_key")
        gs = LD.legacy_grasps(key, legacy_root) if key else []
        role = LD.task_role(j, customs)
        out.append({"id": j["id"], "alias_key": j.get("alias_key"), "legacy_key": key, "after_step3": why, "n_legacy": len(gs),
                    "n_legacy_held": sum(1 for g in gs if g.get("legacy_physics")), "aliases": sorted({a["customs"] for a in role["aliases_of_the_key"]})})
    return out


def cmd_list(a):
    rows = step4_rows(a.batch, a.report)
    for r in rows:
        tail = (f"legacy {r['legacy_key']}: {r['n_legacy']} grasps ({r['n_legacy_held']} held in the legacy validation)" if r["n_legacy"]
                else f"NO legacy grasps (key {r['legacy_key']}) - nothing for step 4")
        print(f"STEP4 {r['id']} alias {r['alias_key']} [{r['after_step3']}] {tail} | customs aliases: {', '.join(r['aliases']) or '-'}")
    S.save_json(os.path.join(a.batch, "step4_list.json"), {"rows": rows, "created": S.now(), "note": "owner decision before any use"})
    print(f"STEP4_LIST {len(rows)} entries would end at step 4 or later ({sum(1 for r in rows if r['n_legacy'])} with legacy grasps) -> "
          f"{os.path.join(a.batch, 'step4_list.json')}")


def cmd_submit(a):
    if not a.owner_approved:
        raise SystemExit("step 4 uses legacy grasps: the owner decides first (pass --owner-approved only with that decision recorded)")
    rows = {r["id"]: r for r in step4_rows(a.batch, a.report) if r["n_legacy"]}
    jobs = []
    for e in S.load_json(os.path.join(a.batch, "entries.json")) or []:
        j = e["job"]
        if j["id"] not in rows or (a.only and j["id"] not in a.only):
            continue
        key = j["legacy_key"]
        jobs.append(LD.legacy_job(j, LD.legacy_grasps(key), key))
        d = os.path.join(a.out, "glib", "models", j["id"])
        os.makedirs(d, exist_ok=True)
        if not os.path.exists(os.path.join(d, "geometry.json")):
            shutil.copy2(os.path.join(e["src_dir"], "geometry.json"), os.path.join(d, "geometry.json"))
    if not jobs:
        print("STEP4_SUBMIT nothing to build")
        return
    m = os.path.join(a.out, "manifest.json")
    S.save_json(m, jobs)
    tag = os.path.basename(os.path.normpath(a.out))
    line = (f"GBH={a.gbh} ISG_GB_ROOT={a.code} GB_OUT={a.out} GB_GLIB={a.out}/glib GB_TAG={tag} GB_MANIFEST={m} GB_NSHARD={len(jobs)} GB_STAGES=glib "
            f"ISG_GLIB_VAL_OVERRIDES='{{\"max_trials\":30,\"max_valid\":30}}' sbatch --parsable -A vision -p svl --exclude=svl9 "
            f"--gres=gpu:titanrtx:1 --array=0-{len(jobs) - 1} -J isgof_legacy -o {a.gbh}/slurm/isgof_legacy_%A_%a.out {a.code}/scripts/grasp_bands/job_glib.sh")
    print("STEP4_SBATCH", line)
    if a.run:
        print("STEP4_JOB", subprocess.check_output(line, shell=True, text=True).strip())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("list")
    p.add_argument("--batch", required=True)
    p.add_argument("--report", default=None)
    p.set_defaults(fn=cmd_list)
    p = sp.add_parser("submit")
    p.add_argument("--batch", required=True)
    p.add_argument("--report", default=None)
    p.add_argument("--code", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gbh", default="/vision/u/minyeong/projects/isg_home/oracle_fix")
    p.add_argument("--owner-approved", action="store_true")
    p.add_argument("--only", nargs="*")
    p.add_argument("--run", action="store_true")
    p.set_defaults(fn=cmd_submit)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
