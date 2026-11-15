#!/usr/bin/env python3
"""Check the merged datasets of REPLAY_DIFF_20261115 against the curated eval_production tree before pushing.

    verify_merge.py --train <lerobot dir> --eval <lerobot dir> --eval-layout <instances dir> --heldout <lerobot dir>
                    --heldout-layout <instances dir>

Expected (the tree's index.json, SUMMARY.json current_by_category, the flag lists, heldout_replay_plan.jsonl):
  train      = every standard-eval-sibling + train instance, minus flagged duplicate starts, each once
  eval       = the 1,000 standard-eval instances, each once (LeRobot and eval layout)
  heldout    = the plan's 312 replays: (instance, clean|perturbed), none rendered of heldout_not_rendered
  no instance of one set in another; no flagged instance anywhere; one chunk per task family
Exit 1 on any mismatch.
"""
import argparse
import collections
import json
import os
import sys

TREE = "/vision/u/minyeong/ispatialgym/eval_production"
DIFF = f"{TREE}/REPLAY_DIFF_20261115"
FAMILIES = ("camera_relocalization", "object_scaling", "mental_rotation", "disembedding", "alignment", "configuration_matching",
            "inter_object_relation_satisfaction")


def jl(p):
    return [json.loads(x) for x in open(p) if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    for k in ("train", "eval", "eval-layout", "heldout", "heldout-layout"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    idx = {r["instance_id"]: r for r in json.load(open(f"{TREE}/index.json")) if r.get("kind") == "demo"}
    flagged = set()
    for f in ("flagged_not_listed.jsonl", "flagged_already_rendered.jsonl", "drop_replayed_flagged.jsonl"):
        flagged |= {r["instance_id"] for r in jl(f"{DIFF}/{f}")}
    summary = json.load(open(f"{DIFF}/SUMMARY.json"))
    by_cat = collections.Counter(r["category"] for r in idx.values())
    errors = []
    print("tree categories", dict(by_cat), "summary", summary["current_by_category"])
    if dict(by_cat) != summary["current_by_category"]:
        errors.append(f"tree index {dict(by_cat)} != SUMMARY {summary['current_by_category']}")
    want = {
        "train": {i for i, r in idx.items() if r["category"] in ("standard-eval-sibling", "train")} - flagged,
        "eval": {i for i, r in idx.items() if r["category"] == "standard-eval"} - flagged,
    }
    plan = jl(f"{DIFF}/heldout_replay_plan.jsonl")
    want_h = {(r["instance_id"], r["variant"]) for r in plan}
    not_rendered = {r["instance_id"] for r in jl(f"{DIFF}/heldout_not_rendered.jsonl")}

    def lerobot(root):
        eps = jl(f"{root}/isg_meta/episodes.jsonl")
        info = json.load(open(f"{root}/meta/info.json"))
        chunks = json.load(open(f"{root}/isg_meta/task_groups.json"))["chunks"]
        bad = [e["episode_index"] for e in eps if chunks.get(e["task_family"]) != e.get("chunk_index")
               or FAMILIES.index(e["task_family"]) != e.get("chunk_index")]
        if bad:
            errors.append(f"{root}: {len(bad)} episodes outside their family's chunk")
        if info["total_episodes"] != len(eps):
            errors.append(f"{root}: info {info['total_episodes']} episodes vs {len(eps)} records")
        dirs = sorted(os.listdir(f"{root}/data"))
        if dirs != [f"chunk-{FAMILIES.index(f):03d}" for f in FAMILIES if f in chunks]:
            errors.append(f"{root}: data chunks {dirs}")
        return eps

    sets = {}
    for name in ("train", "eval"):
        eps = lerobot(getattr(a, name))
        ids = [e["instance"].get("instance_id") or e["instance"].get("demo_name") for e in eps]
        dup = [i for i, n in collections.Counter(ids).items() if n > 1]
        got = set(ids)
        sets[name] = got
        cats = collections.Counter(idx[i]["category"] if i in idx else "missing" for i in got)
        print(f"{name}: {len(eps)} episodes, {len(got)} instances, categories {dict(cats)}; expected {len(want[name])}")
        for what, s in (("missing", want[name] - got), ("extra", got - want[name]), ("duplicated", set(dup)),
                        ("flagged", got & flagged)):
            if s:
                errors.append(f"{name}: {len(s)} {what}, e.g. {sorted(s)[:3]}")
    lay = jl(f"{a.eval_layout}/metadata/instances.jsonl")
    lay_ids = [r["source"]["instance_id"] for r in lay]
    print(f"eval layout: {len(lay)} instances")
    if set(lay_ids) != want["eval"] or len(lay_ids) != len(set(lay_ids)):
        errors.append(f"eval layout: {len(lay_ids)} records, {len(set(lay_ids) ^ want['eval'])} differ from the standard-eval set")
    heps = lerobot(a.heldout)
    hgot = [((e["instance"].get("instance_id") or e["instance"].get("demo_name")), e.get("heldout_variant")) for e in heps]
    hlay = [(r["source"]["instance_id"], r.get("heldout_variant")) for r in jl(f"{a.heldout_layout}/metadata/instances.jsonl")]
    print(f"heldout: {len(heps)} episodes ({collections.Counter(v for _, v in hgot)}), layout {len(hlay)}; expected {len(want_h)}")
    for name, got in (("heldout lerobot", hgot), ("heldout layout", hlay)):
        if set(got) != want_h or len(got) != len(set(got)):
            errors.append(f"{name}: {len(got)} replays; missing {len(want_h - set(got))}, extra {len(set(got) - want_h)}")
        if {i for i, _ in got} & (not_rendered | flagged):
            errors.append(f"{name}: holds not-rendered / flagged members")
    hid = {i for i, _ in hgot}
    for x, y in (("train", "eval"),):
        if sets[x] & sets[y]:
            errors.append(f"{x} and {y} share {len(sets[x] & sets[y])} instances")
    if hid & sets["train"]:
        errors.append(f"held-out instances in train: {len(hid & sets['train'])}")
    goals_h = {idx[i]["group_id"] for i in hid if i in idx}
    leak = {i for i in sets["train"] if idx.get(i, {}).get("group_id") in goals_h}
    if leak:
        errors.append(f"train holds {len(leak)} instances of held-out goal groups")
    print("\n".join(f"ERROR {e}" for e in errors) if errors else "VERIFY OK")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
