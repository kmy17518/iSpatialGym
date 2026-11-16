#!/usr/bin/env python3
"""Re-export the v2 eval dataset so the evaluator reads exactly what the instances were collected and rendered with (2026-11-16):

    fix_eval_dataset.py --src <eval-layout dir (ispatialgym-eval-instances + ispatialgym-eval-demos)> --out <new dir> --overlay <bddl overlay>

The new dir starts as a hard-linked copy of --src; every changed file is written to a new inode (the source stays untouched):
  * BDDL: each instance's exact collection problem (``build_bddl_overlay.py``: sha256 = the TRO's ``bddl_sha256``) ships in the dataset as
    ``ispatialgym-eval-instances/bddl/<instance id>/<activity>/problem<id>.bddl``; the index record names it (``files.bddl_root``).
  * TROs: the ``goal_observation_collection`` copy leaves the ``ispatialgym`` block (the index record keeps it); the oracle sidecars'
    ``tro_sha256`` follow.
  * template meta: the collection's ``scene_crop`` moves to ``eval.collection_scene_crop`` (the dataset ships the full template; the goal
    images and training renders come from the full world).
  * index records: ``eval_lighting: "stock"`` (the batched replay ignored collection lighting draws; the evaluator does the same for
    dataset instances).
  * README: the evaluator notes.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

TRO_SUFFIX = "-tro_state.json"


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def write_new(path, text):
    """Replace ``path`` with a new file (a new inode: a hard link to the source stays as it was)."""
    tmp = path + ".tmp_fix"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


README_NOTE = """
## Evaluator notes (re-export 2026-11-16)

This dataset needs the evaluator of the `campaign-20261115-eval` branch (`ispatialgym.eval.instances`, `ispatialgym.oracle.run_isg`), which:

- loads each instance's own template (`files.template` / `files.template_meta` of its index record: `<scene>_task_<subtask>_<t>_<i>_template`,
  t = the subtask's t-th distinct template), not the first template of its subtask;
- loads each instance's BDDL problem from the dataset (`files.bddl_root` = `bddl/<instance id>`: the exact problem the demo was
  collected with, sha256 = the instance file's `bddl_sha256`), not from the repository's release tree (its ids differ);
- builds the full world from the template (the collection's cropped world, `eval.collection_scene_crop` in the template meta, is not
  shipped; the goal images and the training-format renders come from the full world);
- lights every instance with the stock lights (`eval_lighting: "stock"`): 74 instances promoted from the train run carry a
  collection-time lighting draw in their instance file (`campaign.perturbations.lighting`), which the batched replay ignored when it
  rendered their goal images, so the evaluator ignores it too;
- gives a policy-mode robot BEHAVIOR's 61-D proprioception (the training data's layout).

The collection's goal record (`goal_observation_collection`) is kept in `metadata/instances.jsonl` only.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlay", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    subprocess.run(["cp", "-al", a.src, a.out], check=True)
    root = os.path.join(a.out, "ispatialgym-eval-instances")
    idx_path = os.path.join(root, "metadata", "instances.jsonl")
    recs = [json.loads(l) for l in open(idx_path) if l.strip()]
    metas_done, stats = set(), {"bddl": 0, "tro": 0, "oracle": 0, "template_meta": 0, "crop_moved": 0}
    for r in recs:
        tro_rel = r["files"]["tro_state"]
        iid = os.path.basename(tro_rel)[: -len(TRO_SUFFIX)]
        tro_path = os.path.join(root, tro_rel)
        tro = json.load(open(tro_path))
        blk = tro["ispatialgym"]
        # BDDL
        tm_path = os.path.join(root, r["files"]["template_meta"])
        tm = json.load(open(tm_path))
        act, did = tm["problem"].rsplit("-", 1)
        src_bddl = os.path.join(a.overlay, iid, act, f"problem{did}.bddl")
        if sha256(src_bddl) != blk["bddl_sha256"]:
            raise SystemExit(f"{iid}: overlay BDDL is not the TRO's bddl_sha256")
        dst = os.path.join(root, "bddl", iid, act, f"problem{did}.bddl")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src_bddl, dst)
        r["files"]["bddl_root"] = f"bddl/{iid}"
        stats["bddl"] += 1
        # TRO
        if "goal_observation_collection" in blk:
            if r.get("goal_observation_collection") not in (None, blk["goal_observation_collection"]):
                raise SystemExit(f"{iid}: index and TRO goal_observation_collection differ")
            r["goal_observation_collection"] = blk.pop("goal_observation_collection")
            write_new(tro_path, json.dumps(tro, indent=2))
            stats["tro"] += 1
        # oracle sidecar
        if r["files"].get("oracle"):
            op = os.path.join(root, r["files"]["oracle"])
            sc = json.load(open(op))
            new = sha256(tro_path)
            if sc.get("tro_sha256") != new:
                sc["tro_sha256"] = new
                write_new(op, json.dumps(sc, indent=2) + "\n")
                stats["oracle"] += 1
        # template meta (shared by the instances of one template)
        if tm_path not in metas_done:
            metas_done.add(tm_path)
            if "scene_crop" in tm:
                tm.setdefault("eval", {})["collection_scene_crop"] = tm.pop("scene_crop")
                stats["crop_moved"] += 1
                write_new(tm_path, json.dumps(tm, indent=1))
                stats["template_meta"] += 1
        r["eval_lighting"] = "stock"
    write_new(idx_path, "".join(json.dumps(r) + "\n" for r in recs))
    rd = os.path.join(root, "README.md")
    text = open(rd).read()
    if "## Evaluator notes" not in text:
        write_new(rd, text.rstrip("\n") + "\n" + README_NOTE)
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
