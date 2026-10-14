#!/usr/bin/env python3
"""Export eval instances of the eval85 runs into the dataset layout of ispatialgym.eval.instances: the release's runs
(scripts/eval_instances/release_runs.tsv, frozen), or with ``--runs counted`` the runs counted now (runs/<tag>/SELECTED.tsv
rows with counted = yes, for s2_3tasks, s3_55tasks and iors_large); one subtask per instance, named by the task family ->
task -> subtask hierarchy (ispatialgym.eval.names), the files renamed after BEHAVIOR-1K's 2026-challenge-task-instances.

    python scripts/eval_instances/export_counted.py [--eval-root /vision/u/minyeong/projects/isg_home/eval85] [--out <root>]
        [--demos-out <demos root>] [--runs scripts/eval_instances/release_runs.tsv | counted] [--clean]

Per instance: template.json; template_meta.json with the problem name in the repository's BDDL tree (``isg_customs.json``
``definition_id``; the run's own copy of the problem must be that problem) and an "eval" block with the names; the
-tro_state.json with that ``activity_definition_id``; the .oracle.json with the instance file's ``tro_sha256``; the
demo.hdf5 into the demos dataset. The BDDL problems stay in the repository (as BEHAVIOR's stay in its bddl package). Writes
metadata/{instances.jsonl, episodes.jsonl, tasks.yaml, available_tasks.yaml, task.jsonl} and README.md; sets every
instance's goal image from a successful replay of its demo (ispatialgym.eval.instances.set_goal, which keeps where it came
from in an internal record outside the dataset; exits non-zero naming the instances without one); deletes the files of
instances no longer exported.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import shutil
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from ispatialgym.activity.problem import parse_problem_file, parse_problem_string  # noqa: E402
from ispatialgym.eval import names as N  # noqa: E402
from ispatialgym.eval.instances import DEMO_SUFFIX, GoalImageError, default_root, demos_root, goal_record_path, load_index, set_goal, sha256_of  # noqa: E402
from ispatialgym.oracle.sidecar import from_legacy_keys  # noqa: E402

TAGS = ("s2_3tasks", "s3_55tasks", "iors_large")
TRO_SUFFIX = "-tro_state.json"
FAMILY_ORDER = ("alignment", "camera_relocalization", "configuration_matching", "disembedding", "inter_object_relation_satisfaction",
                "mental_rotation", "object_scaling")
BDDL_ROOT = os.path.join(REPO, "ispatialgym", "assets", "bddl", "activity_definitions")
CUSTOMS = os.path.join(REPO, "ispatialgym", "assets", "task_specs", "isg_customs.json")


def counted_runs(eval_root: str):
    for tag in TAGS:
        with open(os.path.join(eval_root, "runs", tag, "SELECTED.tsv")) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["counted"] == "yes":
                    yield tag, r["pick"], r["run"]


def release_runs(path: str):
    """``[(subtask, tag, pick, run)]`` of a release file (tab-separated, ``#`` comments)."""
    with open(path) as f:
        rows = list(csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t"))
    return [(r["subtask"], r["tag"], r["pick"], r["run"]) for r in rows]


def read_instance(eval_root: str, tag: str, pick: str, run: str) -> dict:
    run_dir = os.path.join(eval_root, "runs", tag, run)
    tros = glob.glob(os.path.join(run_dir, "inst", "*", "*", "instances", "*" + TRO_SUFFIX))
    if len(tros) != 1:
        raise RuntimeError(f"{run_dir}: expected one instance, found {len(tros)}")
    tro = tros[0]
    iid = os.path.basename(tro)[: -len(TRO_SUFFIX)]
    tdir = os.path.dirname(os.path.dirname(tro))
    with open(os.path.join(tdir, "template_meta.json")) as f:
        tmeta = from_legacy_keys(json.load(f))
    activity, did = tmeta["problem"].rsplit("-", 1)
    demo_dir = os.path.join(run_dir, "demo", iid)
    with open(os.path.join(demo_dir, "meta.json")) as f:
        demo_meta = json.load(f)
    with open(os.path.join(run_dir, "replay_log", iid, "meta.json")) as f:
        replay_meta = json.load(f)
    with open(tro) as f:
        tro_json = json.load(f)
    poses = {k.lower(): v for k, v in (tro_json.get("robot_poses") or {}).items()}
    pose = (poses.get("robot") or poses.get("r1pro") or [{}])[0]
    variant = tmeta["variant"]
    review = os.path.join(eval_root, "review_videos", "replays", tag, run, "replay_log", iid)
    import h5py

    with h5py.File(os.path.join(demo_dir, "demo.hdf5"), "r") as f:
        n_rows = int(f["data"]["demo_0"]["action"].shape[0])  # the recorded steps: the program's actions and its settle window
    return {
        "tag": tag, "pick": pick, "run": run, "run_dir": run_dir, "iid": iid, "tdir": tdir, "tro": tro, "tro_json": tro_json,
        "bddl": os.path.join(tdir, "activity_definitions", activity, f"problem{did}.bddl"), "activity": activity, "run_problem": tmeta["problem"],
        "tmeta": tmeta, "variant": variant, "task_family": N.task_family(variant), "task": N.task_name(variant), "subtask": N.subtask_name(variant),
        "scene": tmeta["scene"], "robot_pose": pose, "length": n_rows, "action_steps": demo_meta.get("n_action_steps"),
        "settle_steps": demo_meta.get("n_settle_steps"),
        "source": {
            "tag": tag, "pick": pick, "run": run, "run_dir": run_dir, "variant": variant, "problem": tmeta["problem"],
            "template_id": tmeta.get("template_id"), "demo_hdf5": os.path.join(demo_dir, "demo.hdf5"), "demo_meta": os.path.join(demo_dir, "meta.json"),
            "demo_videos": [os.path.join(demo_dir, v) for v in ("head_rgb.mp4", "external_rgb.mp4")],
            "replay_meta": os.path.join(run_dir, "replay_log", iid, "meta.json"),
            "replay_videos": [os.path.join(review, v) for v in ("head_rgb.mp4", "external_rgb.mp4") if os.path.exists(os.path.join(review, v))],
            "code": (replay_meta.get("provenance") or {}).get("isg_root"),
        },
    }


def pick_number(pick: str) -> int:
    m = re.search(r"#(\d+)$", pick)
    return int(m.group(1)) if m else 0


def release_bddl(d: dict, customs: dict) -> tuple:
    """``(activity, definition id, path)`` of the instance's problem in the release's BDDL tree; the run's own copy must be the
    same problem (sections, not comments) under its old name."""
    entry = customs[d["subtask"]]
    activity, did = entry["iddl"].split("/")[0], int(entry["definition_id"])
    path = os.path.join(BDDL_ROOT, activity, f"problem{did}.bddl")
    with open(d["bddl"]) as f:
        run_text = f.read().replace(f"(define (problem {d['run_problem']})", f"(define (problem {activity}-{did})")
    if parse_problem_string(run_text, activity, did).expanded_bddl() != parse_problem_file(path, activity, did).expanded_bddl():
        raise SystemExit(f"{d['subtask']}: {path} is not the run's problem {d['run_problem']} ({d['bddl']})")
    return activity, did, path


def write_json(path: str, data, indent: int, newline: bool = False) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)
        if newline:
            f.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval-root", default="/vision/u/minyeong/projects/isg_home/eval85")
    ap.add_argument("--out", default=None, help="dataset root (default: ispatialgym.eval.instances.default_root())")
    ap.add_argument("--demos-out", default=None, help="demos dataset root (default: the sibling ispatialgym-eval-demos of --out)")
    ap.add_argument("--clean", action="store_true", help="delete both datasets first (goal images too)")
    ap.add_argument("--runs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "release_runs.tsv"),
                    help="release file of the runs to export (subtask, tag, pick, run; default: the release's); 'counted' = the runs counted now")
    a = ap.parse_args()
    out = os.path.abspath(a.out or default_root())
    demos = os.path.abspath(a.demos_out or demos_root(out))
    if a.runs == "counted":
        items = [read_instance(a.eval_root, *r) for r in counted_runs(a.eval_root)]
    else:
        items = []
        for subtask, tag, pick, run in release_runs(a.runs):
            d = read_instance(a.eval_root, tag, pick, run)
            if d["subtask"] != subtask:
                raise SystemExit(f"{a.runs}: {tag}/{run} is subtask {d['subtask']}, not {subtask}")
            items.append(d)
    items.sort(key=lambda d: (FAMILY_ORDER.index(d["task_family"]), d["task"], pick_number(d["pick"]), d["subtask"]))
    names = [d["subtask"] for d in items]
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        raise SystemExit(f"subtask names not unique: {dups}")
    if a.clean:
        for root in (out, demos):
            if os.path.isdir(root):
                shutil.rmtree(root)
    os.makedirs(os.path.join(out, "metadata"), exist_ok=True)
    with open(CUSTOMS) as f:
        customs = json.load(f)

    index, episodes, tasks, available, task_lines = [], [], {}, {}, []
    task_counts = {}
    for d in items:
        scene, sub = d["scene"], d["subtask"]
        activity, did, _ = release_bddl(d, customs)
        json_dir = os.path.join("scenes", scene, "json")
        prefix = f"{scene}_task_{sub}"
        inst_dir = os.path.join(json_dir, f"{prefix}_instances")
        inst = f"{prefix}_0_0_template"
        files = {
            "template": os.path.join(json_dir, f"{prefix}_0_0_template.json"),
            "template_meta": os.path.join(json_dir, f"{prefix}_0_0_template-meta.json"),
            "tro_state": os.path.join(inst_dir, f"{inst}-tro_state.json"),
            "oracle": os.path.join(inst_dir, f"{inst}.oracle.json"),
            "goal_rgb": os.path.join(inst_dir, f"{inst}-goal_rgb.png"),
        }
        demo_rel = os.path.join(inst_dir, inst + DEMO_SUFFIX)
        for folder in (os.path.join(out, inst_dir), os.path.join(demos, inst_dir)):
            os.makedirs(folder, exist_ok=True)
        shutil.copyfile(os.path.join(d["tdir"], "template.json"), os.path.join(out, files["template"]))
        tmeta = dict(d["tmeta"])
        tmeta["problem"] = f"{activity}-{did}"
        tmeta["eval"] = {"task_family": d["task_family"], "task": d["task"], "subtask": sub, "variant": d["variant"]}
        write_json(os.path.join(out, files["template_meta"]), tmeta, 1)
        tro = dict(d["tro_json"])
        tro["ispatialgym"] = {**tro["ispatialgym"], "activity_definition_id": did}
        write_json(os.path.join(out, files["tro_state"]), tro, 2)
        with open(os.path.join(d["tdir"], "instances", d["iid"] + ".oracle.json")) as f:
            sidecar = from_legacy_keys(json.load(f))
        sidecar["tro_sha256"] = sha256_of(os.path.join(out, files["tro_state"]))
        write_json(os.path.join(out, files["oracle"]), sidecar, 2, newline=True)
        demo_dst = os.path.join(demos, demo_rel)
        if not (os.path.exists(demo_dst) and os.path.getsize(demo_dst) == os.path.getsize(d["source"]["demo_hdf5"])):
            shutil.copyfile(d["source"]["demo_hdf5"], demo_dst)
        ti = task_counts.get(d["task"], 0)
        task_counts[d["task"]] = ti + 1
        names_ = {"task_family": d["task_family"], "task": d["task"], "subtask": sub, "instance": 0, "task_instance": ti}
        index.append({**names_, "scene": scene, "problem": f"{activity}-{did}", "length": d["length"], "files": files, "source": d["source"]})
        episodes.append({"episode_index": len(episodes), **names_, "length": d["length"], "action_steps": d["action_steps"], "settle_steps": d["settle_steps"],
                         "demo": demo_rel, "demo_sha256": sha256_of(demo_dst)})
        tasks.setdefault(d["task_family"], {}).setdefault(d["task"], []).append(sub)
        pose = d["robot_pose"] or {}
        available[sub] = {0: {"scene_model": scene, "task": d["task"], "task_family": d["task_family"],
                              "robot_start_position": [float(v) for v in pose.get("position", [])],
                              "robot_start_orientation": [float(v) for v in pose.get("orientation", [])]}}
        task_lines.append({"task_index": len(task_lines), "task_name": sub, "task": d["task"], "task_family": d["task_family"], "num_instances": 1,
                           "length": d["length"]})

    for name, lines in (("instances.jsonl", index), ("episodes.jsonl", episodes), ("task.jsonl", task_lines)):
        with open(os.path.join(out, "metadata", name), "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
    with open(os.path.join(out, "metadata", "tasks.yaml"), "w") as f:
        yaml.safe_dump(tasks, f, sort_keys=False, default_flow_style=False)
    with open(os.path.join(out, "metadata", "available_tasks.yaml"), "w") as f:
        yaml.safe_dump(available, f, sort_keys=False, default_flow_style=None)
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(readme(index, tasks))
    if os.path.isdir(os.path.join(out, "activity_definitions")):  # the problems are the repository's (exports before 2026-10-14 copied them)
        shutil.rmtree(os.path.join(out, "activity_definitions"))
    stale = delete_unindexed(out, ("scenes",), {p for rec in index for p in rec["files"].values()})
    stale += delete_unindexed(demos, ("scenes",), {e["demo"] for e in episodes})
    if stale:
        print(f"deleted {len(stale)} file(s) of instances no longer exported")
    sources, failed = {}, []
    for rec in load_index(out):
        # a goal image set from an explicit replay (set-goal --run, e.g. another image size) stays the instance's
        prev = {}
        if os.path.exists(goal_record_path(rec)):
            with open(goal_record_path(rec)) as f:
                prev = json.load(f).get("source") or {}
        try:
            info = set_goal(rec, prev["run_dir"] if prev.get("replay") == "run" else None)
        except GoalImageError as e:
            failed.append(str(e))
            continue
        sources[info["source"]["replay"]] = sources.get(info["source"]["replay"], 0) + 1
    n_fam, n_task = len(tasks), sum(len(t) for t in tasks.values())
    print(f"{len(index)} instances (goal images from replays: {sources}), {n_task} tasks, {n_fam} task families -> {out}; demos -> {demos}")
    if failed:
        raise SystemExit(f"{len(failed)} instance(s) without a goal image:\n" + "\n".join(failed))
    return 0


def delete_unindexed(root: str, subdirs, keep_rel) -> list:
    """Delete the files under ``root/<subdirs>`` not in ``keep_rel`` (paths relative to ``root``) and the emptied folders."""
    keep = {os.path.normpath(os.path.join(root, p)) for p in keep_rel}
    stale = []
    for sub in subdirs:
        for dirpath, _dirs, files in os.walk(os.path.join(root, sub), topdown=False):
            for name in files:
                path = os.path.normpath(os.path.join(dirpath, name))
                if path not in keep:
                    os.remove(path)
                    stale.append(path)
            if not os.listdir(dirpath):
                os.rmdir(dirpath)
    return stale


def readme(index, tasks) -> str:
    rows = "\n".join(f"| {fam} | {task} | {len(subs)} |" for fam, ts in tasks.items() for task, subs in ts.items())
    return f"""# iSpatialGym eval instances

{len(index)} eval instances, one per subtask, laid out like BEHAVIOR-1K's `2026-challenge-task-instances` and kept next to it
in OmniGibson's data folder, with their demos in the sibling dataset `ispatialgym-eval-demos`. Written by
`scripts/eval_instances/export_counted.py` from the eval runs that `scripts/eval_instances/release_runs.tsv` lists (the
counted instances of 2026-10-13 ~20:40, frozen), each with its goal image.

## Hierarchy

- **task family**: alignment, camera_relocalization, configuration_matching, disembedding, inter_object_relation_satisfaction,
  mental_rotation, object_scaling.
- **task**: a family's task, e.g. `alignment_axial`, `alignment_vertical`, `camera_relocalization_obstructed`,
  `configuration_matching_attachment`.
- **subtask**: a task with its objects, the support left out, e.g. `alignment_axial_board_game`; each subtask has one scene
  and one template, and its instances differ only in their initial state. `isg_customs.json` is keyed by subtask.
- **variant**: the sampler's grounded setup of a subtask, its objects and support, e.g.
  `alignment-axial-board_game-breakfast_table` (the BDDL problem's `(:variant ...)`, the instance file's `task_variant`).
  A subtask may have several; here each has one, named by `variant` in the template metadata and its `eval` block.

| task family | task | subtasks |
|---|---|---|
{rows}

## Layout

```text
metadata/instances.jsonl       one line per instance: task_family, task, subtask, instance (index within the subtask),
                               task_instance (index within the task), scene, problem, length, files (relative to this
                               folder), source (the eval run it came from)
metadata/episodes.jsonl        one line per instance's demo: episode_index, the names, length (the demo's recorded steps,
                               the actions a replay steps), action_steps + settle_steps (the oracle's program and its
                               settle window, which make up length), demo (its file in ../ispatialgym-eval-demos), demo_sha256
metadata/tasks.yaml            task family -> task -> [subtask, ...]
metadata/available_tasks.yaml  subtask -> {{0: scene_model, robot_start_position, robot_start_orientation, task, task_family}}
metadata/task.jsonl            one line per subtask: task_index, task_name (= subtask), task, task_family, num_instances, length
scenes/<scene>/json/<scene>_task_<subtask>_0_0_template.json          the scene with the task objects
scenes/<scene>/json/<scene>_task_<subtask>_0_0_template-meta.json     the sampler's template facts (+ "eval": the names)
scenes/<scene>/json/<scene>_task_<subtask>_instances/
    <scene>_task_<subtask>_0_<i>_template-tro_state.json   instance i
    <scene>_task_<subtask>_0_<i>_template.oracle.json      its oracle sidecar (the demo program)
    <scene>_task_<subtask>_0_<i>_template-goal_rgb.png     its goal image
../ispatialgym-eval-demos/scenes/<scene>/json/<scene>_task_<subtask>_instances/
    <scene>_task_<subtask>_0_<i>_template-demo.hdf5        instance i's demo (the actions a replay steps)
```

As in BEHAVIOR's layout the first number after the subtask is the template (always 0: one template per subtask) and the
second the instance. The BDDL problems are not here, as BEHAVIOR's are not in its task instances: an instance names its
problem (`problem` = `<activity>-<id>` in the index and the template metadata, `activity_definition_id` in the instance
file), loaded from the iSpatialGym repository's `ispatialgym/assets/bddl/activity_definitions` (numbered from 0 per
activity). The files inside keep the sampler's own fields.

## Loading

```bash
# a task's instances are its subtasks' instances in the index order; a subtask's are its own
python -m ispatialgym.oracle.run_isg --task alignment_axial --instance-indices 0 2 --mode replay          # the instances' demos
python -m ispatialgym.oracle.run_isg --task alignment_axial_board_game --instance-indices 0 --mode policy --policy ...
python -m ispatialgym.eval.instances list --level task          # names and instance counts
python -m ispatialgym.eval.instances show alignment_axial 0 1   # what an index resolves to (files, goal image, demo)
```

`ISPATIALGYM_INSTANCES_PATH` (or `--instances-root`) points at this folder, `ISPATIALGYM_DEMOS_PATH` at the demos; the
defaults are `$OMNIGIBSON_DATA_PATH/ispatialgym-eval-instances` and its sibling `ispatialgym-eval-demos`. With `--task`,
run_isg gives an instance `max(300, 2 x its demo's length)` steps (`--max-steps`), a settle window of 0.2 x that
(`--settle-steps`) and 1 s per step of wall clock (`--wallclock`), and replays its demo unless `--replay` names another.
run_isg never writes into this folder: runs go to `--out` (default `./runs`) and the goal / sidecar write-backs of oracle
runs are skipped for dataset instances.

## Goal images

The goal image is the head camera's last frame (`final_rgb.png`, 1080 x 1080) of a successful replay of the instance's demo:
the eval run's replay re-run with the current code, else its original replay (`python -m ispatialgym.eval.instances
set-goal`). A replay renders the whole episode; a frame rendered straight at the demo's final state comes out about 1 %
darker (3-4 % in shadows), because the renderer's bounced light depends on what it rendered before. Each instance's
`files.goal_rgb` (here) names it, and its instance file registers it (`ispatialgym.goal_observation`, as `run_isg
--goal-image` writes it: `rgb_path`, sha256, size, the head camera key and intrinsics, and the camera's world pose: the
demo's last recorded head camera pose, which the replay holds through its settle window), which is where the policy
harness reads it. The sidecar's `tro_sha256` follows the instance file.

Another image size: replay the instance at that size (`--record video`: no demo.hdf5 / LeRobot copy of the replay), then
make that run's final frame the goal image (one goal image per instance; a re-export keeps it). The replay keeps going
through the demo's own contacts with the objects: run_isg only logs the QA metrics of a replay (`--qa`), it never stops
at them.

```bash
sbatch scripts/eval_instances/job_run_isg.sh --task alignment_axial --instance-indices 0 --mode replay \\
    --image-size 720 --record video --out <dir>
python -m ispatialgym.eval.instances set-goal alignment_axial 0 --run <dir>
python -m ispatialgym.eval.instances set-goal alignment_axial 0          # back to the eval run's replay (1080)
```
"""


if __name__ == "__main__":
    raise SystemExit(main())
