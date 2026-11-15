#!/usr/bin/env python3
"""Batched replay -> the eval dataset layout (git@github.com:kmy17518/ispatialgym-eval-data: ispatialgym-eval-instances +
ispatialgym-eval-demos, read by ispatialgym.eval.instances / run_isg --task).

    PYTHONPATH=<code> OMNIGIBSON_DATA_PATH=<code>/Behavior-1K/datasets \\
      export_eval.py --runs <run>[,<run>...] --out <root>/ispatialgym-eval-instances [--demos-out <root>/ispatialgym-eval-demos]

Reads the eval units pulled by sync.py ($RV/runs/<run>/out/<unit>/: episodes_isg.jsonl with each episode's instance linkage, the
goal_images/ of the replay). Per replayed instance (one kept episode), named as scripts/eval_instances/export_counted.py names them
(task family -> task -> subtask from the TRO's variant, ispatialgym.eval.names):
  scenes/<scene>/json/<scene>_task_<subtask>_<t>_0_template.json / -meta.json      the template (t: the subtask's t-th distinct
                                                                                   template.json; template_meta + an "eval" block)
  scenes/<scene>/json/<scene>_task_<subtask>_instances/<scene>_task_<subtask>_<t>_<i>_template-tro_state.json   the instance's TRO,
      activity_definition_id of the repository's problem when isg_customs maps it, goal_observation = the replay's goal image
  ..._<t>_<i>_template.oracle.json     the oracle sidecar (tro_sha256 refreshed)
  ..._<t>_<i>_template-goal_rgb.png    the goal image: the replayed episode's last head-camera frame at the replay's size (224)
  <demos-out>/scenes/.../..._<t>_<i>_template-demo.hdf5   the instance's demo (the recorded one, unchanged)
and metadata/{instances.jsonl, episodes.jsonl, tasks.yaml, available_tasks.yaml, task.jsonl}, README.md. instances.jsonl `source` keeps
the linkage (eval manifest unit, EVAL_INSTANCE_ID, demo dir + sha256, TRO / template paths + sha256) and `replay` the batched-replay
episode the goal image came from (run, unit, episode, settle, image size).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

RV = Path(os.environ.get("RV", "/vision/u/minyeong/projects/isg_home/campaign_20261110/training_generation/batched_replay_r1"))
FAMILY_ORDER = ("alignment", "camera_relocalization", "configuration_matching", "disembedding", "inter_object_relation_satisfaction",
                "mental_rotation", "object_scaling")


def sha256_file(p, bs=8 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, data, indent, newline=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)
        if newline:
            f.write("\n")


def episodes_of(runs):
    out = []
    for run in runs:
        spec = json.load(open(RV / "runs" / run / "units.json"))
        for u in spec["units"]:
            d = RV / "runs" / run / "out" / u["unit"]
            if u["purpose"] != "eval":
                continue
            if not (d / ".synced").exists():
                print(f"skip {run}/{u['unit']}: not pulled")
                continue
            for line in (d / "episodes_isg.jsonl").read_text().splitlines():
                if line.strip():
                    e = json.loads(line)
                    if e.get("kept"):
                        out.append((run, u, d, e))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--demos-out", default=None)
    a = ap.parse_args()
    from ispatialgym.activity.problem import canonical_variant
    from ispatialgym.eval import names as N
    from ispatialgym.eval.instances import DEMO_SUFFIX, EVALUATION_PROFILE
    from ispatialgym.oracle.sidecar import from_legacy_keys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval_instances"))
    import export_counted as EC  # the 2026-10-14 exporter: repository problem ids (isg_customs), README text

    out = Path(a.out).resolve()
    demos = Path(a.demos_out).resolve() if a.demos_out else out.parent / "ispatialgym-eval-demos"
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty")
    customs = json.load(open(EC.CUSTOMS))
    items = []
    for run, u, d, e in episodes_of(a.runs.split(",")):
        inst = e["instance"]
        tro = json.load(open(inst["tro_path"]))
        block = tro["ispatialgym"]
        variant = canonical_variant(block.get("task_variant") or inst.get("variant"))
        tmeta = from_legacy_keys(json.load(open(inst["template_meta"]))) if inst.get("template_meta") and os.path.exists(inst["template_meta"]) else {}
        items.append({"run": run, "unit": u, "dir": d, "ep": e, "inst": inst, "tro": tro, "variant": variant, "tmeta": tmeta,
                      "task_family": N.task_family(variant), "task": N.task_name(variant), "subtask": N.subtask_name(variant),
                      "scene": tmeta.get("scene") or block.get("scene", {}).get("model"),
                      "order": (inst.get("class_id") or "", inst.get("eval_unit_id") or inst["demo_name"])})
    items.sort(key=lambda x: (FAMILY_ORDER.index(x["task_family"]), x["task"], x["subtask"], x["order"]))
    index, episodes, tasks, available, task_lines = [], [], {}, {}, []
    per_sub = collections.defaultdict(list)
    for it in items:
        per_sub[it["subtask"]].append(it)
    task_counts = collections.Counter()
    problems = collections.Counter()
    for sub, its in per_sub.items():
        tsha = {}
        for i, it in enumerate(its):
            scene = it["scene"]
            sha = sha256_file(it["inst"]["template_json"])
            t = tsha.setdefault(sha, len(tsha))
            json_dir = Path("scenes") / scene / "json"
            prefix = f"{scene}_task_{sub}"
            inst_dir = json_dir / f"{prefix}_instances"
            name = f"{prefix}_{t}_{i}_template"
            files = {"template": str(json_dir / f"{prefix}_{t}_0_template.json"), "template_meta": str(json_dir / f"{prefix}_{t}_0_template-meta.json"),
                     "tro_state": str(inst_dir / f"{name}-tro_state.json"), "oracle": str(inst_dir / f"{name}.oracle.json"),
                     "goal_rgb": str(inst_dir / f"{name}-goal_rgb.png")}
            demo_rel = str(inst_dir / (name + DEMO_SUFFIX))
            # the repository's problem (isg_customs), else the template's own
            activity, did = (it["tmeta"].get("problem") or "-0").rsplit("-", 1)
            did, added, psrc = int(did), {}, "template"
            try:
                bddl = os.path.join(it["inst"]["template_dir"], "activity_definitions", activity, f"problem{did}.bddl")
                activity, did, _, added = EC.release_bddl({"subtask": sub, "bddl": bddl, "run_problem": it["tmeta"]["problem"]}, customs)
                psrc = "repository (isg_customs)"
            except (KeyError, SystemExit, FileNotFoundError, ValueError) as err:
                print(f"{sub} #{i}: problem from the template ({str(err)[:120]})")
            problems[psrc] += 1
            if not (out / files["template"]).exists():
                (out / files["template"]).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(it["inst"]["template_json"], out / files["template"])
                tm = dict(it["tmeta"])
                tm["problem"] = f"{activity}-{did}"
                tm["eval"] = {"task_family": it["task_family"], "task": it["task"], "subtask": sub, "variant": it["variant"]}
                write_json(out / files["template_meta"], tm, 1)
            png = it["dir"] / it["ep"]["goal_image"]["rgb_path"]
            (out / files["goal_rgb"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(png, out / files["goal_rgb"])
            tro = json.loads(json.dumps(it["tro"]))
            blk = tro["ispatialgym"]
            blk["activity_definition_id"] = did
            for key in ("task_variant", "instance_id"):
                if key in blk:
                    blk[key] = canonical_variant(blk[key])
            if added:
                EC.pin_added_members(blk, added, os.path.join(it["inst"]["demo_dir"], "demo.hdf5"))
            g = it["ep"]["goal_image"]
            if blk.get("goal_observation") is not None:
                blk["goal_observation_collection"] = blk["goal_observation"]
            blk["goal_observation"] = {"sensor_key": g["sensor_key"], "rgb_path": os.path.basename(files["goal_rgb"]),
                                       "rgb_sha256": sha256_file(out / files["goal_rgb"]), "resolution": g["resolution"],
                                       "intrinsics": g.get("intrinsics"), "camera_pose_world": g["camera_pose_world"],
                                       "evaluation_profile": EVALUATION_PROFILE, "n_render_iterations": g.get("n_render_iterations", 1)}
            write_json(out / files["tro_state"], tro, 2)
            if it["inst"].get("oracle_sidecar"):
                sc = from_legacy_keys(json.load(open(it["inst"]["oracle_sidecar"])))
                sc["tro_sha256"] = sha256_file(out / files["tro_state"])
                write_json(out / files["oracle"], sc, 2, newline=True)
            else:
                files.pop("oracle")
            demo_src = os.path.join(it["inst"]["demo_dir"], "demo.hdf5")
            (demos / demo_rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(demo_src, demos / demo_rel)
            meta = json.load(open(os.path.join(it["inst"]["demo_dir"], "meta.json")))
            length = int(it["ep"]["length_recorded"])
            ti = task_counts[it["task"]]
            task_counts[it["task"]] += 1
            names_ = {"task_family": it["task_family"], "task": it["task"], "subtask": sub, "instance": i, "task_instance": ti}
            src = {k: it["inst"].get(k) for k in ("class_id", "eval_manifest", "eval_unit_id", "eval_unit_dir", "eval_instance_id",
                                                   "eval_source", "instance_id", "variant", "demo_dir", "demo_sha256", "template_dir",
                                                   "template_json", "tro_path", "tro_sha256", "oracle_sidecar")}
            src.update({"template_sha256": sha, "template_index": t, "problem_source": psrc})
            replay = {"run": it["run"], "unit": it["unit"]["unit"], "dataset": it["ep"]["dataset"], "episode_index": it["ep"]["episode_index"],
                      "length": it["ep"]["length"], "settle": it["ep"].get("settle"), "image_size": it["ep"].get("image_size"),
                      "scene_mode": it["ep"].get("scene_mode"), "perturbation": it["ep"].get("perturbation")}
            index.append({**names_, "scene": scene, "problem": f"{activity}-{did}", "length": length, "files": files, "source": src,
                          "replay": replay})
            episodes.append({"episode_index": len(episodes), **names_, "length": length, "action_steps": meta.get("n_action_steps"),
                             "settle_steps": meta.get("n_settle_steps"), "demo": demo_rel, "demo_sha256": sha256_file(demos / demo_rel)})
            tasks.setdefault(it["task_family"], {}).setdefault(it["task"], [])
            if sub not in tasks[it["task_family"]][it["task"]]:
                tasks[it["task_family"]][it["task"]].append(sub)
            poses = {k.lower(): v for k, v in (it["tro"].get("robot_poses") or {}).items()}
            pose = (poses.get("robot") or poses.get("r1pro") or [{}])[0]
            available.setdefault(sub, {})[i] = {"scene_model": scene, "task": it["task"], "task_family": it["task_family"],
                                                "robot_start_position": [float(v) for v in pose.get("position", [])],
                                                "robot_start_orientation": [float(v) for v in pose.get("orientation", [])]}
        task_lines.append({"task_index": len(task_lines), "task_name": sub, "task": its[0]["task"], "task_family": its[0]["task_family"],
                           "num_instances": len(its), "length": max(r["length"] for r in index if r["subtask"] == sub)})
    (out / "metadata").mkdir(parents=True, exist_ok=True)
    for name, lines in (("instances.jsonl", index), ("episodes.jsonl", episodes), ("task.jsonl", task_lines)):
        with open(out / "metadata" / name, "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
    with open(out / "metadata" / "tasks.yaml", "w") as f:
        yaml.safe_dump(tasks, f, sort_keys=False, default_flow_style=False)
    with open(out / "metadata" / "available_tasks.yaml", "w") as f:
        yaml.safe_dump(available, f, sort_keys=False, default_flow_style=None)
    rows = "\n".join(f"| {fam} | {task} | {len(subs)} | {sum(1 for r in index if r['task'] == task)} |"
                     for fam, ts in tasks.items() for task, subs in ts.items())
    (out / "README.md").write_text(f"""# iSpatialGym eval instances (batched replay {', '.join(a.runs.split(','))})

{len(index)} eval instances of the campaign_20261110 standard-eval set, laid out like BEHAVIOR-1K's `2026-challenge-task-instances`
(the layout of scripts/eval_instances/export_counted.py: task family -> task -> subtask, `<scene>_task_<subtask>_<template>_<instance>`),
with their demos in the sibling dataset `ispatialgym-eval-demos`. Written by `scripts/batched_replay/export_eval.py`.

- Goal image: each instance's `-goal_rgb.png` is the last rendered head-camera frame of its demo's batched replay (224 x 224, `--scene
  full`: the recorded crop merged into the template's task + adjacent rooms; no replay-time perturbation for eval instances), registered
  as the TRO's `goal_observation`; the collection's, if any, is kept as `goal_observation_collection`.
- `metadata/instances.jsonl` `source`: the eval-1000 manifest unit, EVAL_INSTANCE_ID, the demo / TRO / template paths and sha256;
  `replay`: the batched-replay run / unit / episode, its settle window and size. Problems: {dict(problems)}.

| task family | task | subtasks | instances |
|---|---|---|---|
{rows}
""")
    print(f"{len(index)} instances, {len(task_lines)} subtasks -> {out}; demos -> {demos}; problems {dict(problems)}")
    from ispatialgym.eval.instances import load_index

    recs = load_index(str(out))
    bad = [r for r in recs if not all(os.path.exists(r.path(k)) for k in r.files) or not os.path.exists(r.demo_path())]
    print(f"load_index: {len(recs)} records, {len(bad)} with missing files")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
