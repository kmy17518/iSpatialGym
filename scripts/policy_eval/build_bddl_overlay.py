#!/usr/bin/env python3
"""Per-instance BDDL roots for an eval dataset whose problems are not the repository's release tree (the v2 batched-replay export,
scripts/batched_replay/export_eval.py, names each template's own problem id: README "Problems: {'template': N}"): for every instance,
the exact problem its demo was collected with - ``<collection template dir>/activity_definitions/<activity>/problem<id>.bddl`` (the
demo's meta.json ``template_dir``), checked against the instance file's ``bddl_sha256`` - copied to
``<out>/<instance id>/<activity>/problem<id>.bddl``. The evaluator reads it through ``ISPATIALGYM_EVAL_BDDL_OVERLAY=<out>``
(ispatialgym.eval.instances.instance_bddl_root).

    build_bddl_overlay.py --instances <ispatialgym-eval-instances root> --out <overlay dir>
"""
import argparse
import collections
import hashlib
import json
import os
import shutil
import sys


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    stats, bad = collections.Counter(), []
    for line in open(os.path.join(a.instances, "metadata/instances.jsonl")):
        r = json.loads(line)
        iid = os.path.basename(r["files"]["tro_state"])[: -len("-tro_state.json")]
        tm = json.load(open(os.path.join(a.instances, r["files"]["template_meta"])))
        act, did = tm["problem"].rsplit("-", 1)
        tro = json.load(open(os.path.join(a.instances, r["files"]["tro_state"])))["ispatialgym"]
        if int(tro["activity_definition_id"]) != int(did) or tro["activity"] != act:
            bad.append((iid, "tro problem != template meta problem"))
            continue
        demo_meta = json.load(open(os.path.join(r["source"]["demo_dir"], "meta.json")))
        src = os.path.join(demo_meta["template_dir"], "activity_definitions", act, f"problem{did}.bddl")
        if not os.path.exists(src) or sha(src) != tro["bddl_sha256"]:
            bad.append((iid, f"collection BDDL {src} missing or not the TRO's bddl_sha256"))
            continue
        dst = os.path.join(a.out, iid, act, f"problem{did}.bddl")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        stats[r["task_family"]] += 1
    json.dump({"instances": sum(stats.values()), "by_family": stats, "failed": bad}, open(os.path.join(a.out, "overlay.json"), "w"), indent=1)
    print(f"overlay: {sum(stats.values())} instances {dict(stats)}; failed {len(bad)}", bad[:5])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
