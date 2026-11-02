"""The instance lists of the ISG B1K runs' evaluation (``run_isg_vec --instances``): per training task its validation
instances (TRO files of the train85 production, the dataset's last head frame as the goal image) and its eval-dataset
instance(s) (the 83 counted instances, matched by alias), plus the run table.

    python scripts/vec_eval/build_instance_lists.py --out /vision/u/minyeong/projects/isg_home/vec_eval85

Writes ``<out>/instances/val/<task>.json``, ``<out>/instances/eval/<task>.json``, ``<out>/instances/task_ids.json``,
``<out>/instances/runs.json`` and the val goal images ``<out>/goals/val/<episode_index>.png`` (decoded exactly as ACT's
``B1KDataset._goal_frame`` with ``goal_source=episode_last``: the row timestamp of the episode's last recorded frame plus the
camera stream's ``from_timestamp``, nearest frame within 8 ms).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

DATASET = "/vision/u/minyeong/projects/isg_home/train85/datasets/isg-1020-b1k"
SPLIT = "/vision/u/minyeong/projects/isg-buildbox-coord-side/coordinator/inbox/buildbox1/pi_split_act_multitask5.json"
EVAL_ROOT = "/vision/u/minyeong/projects/ispatialgym-eval/Behavior-1K/datasets/ispatialgym-eval-instances"
HEAD_KEY = "observation.rgb.zed_link_camera_0"
TOL = 0.008

RUNS = {
    "act-isg1020-objscale-bs1024-s0": {"kind": "act", "groups": ["object_scaling"], "code": "act isg-wave2-dev-todo1 @ 68860d3", "steps": list(range(10000, 200001, 10000))},
    "act-isg1020-reloc-bs1024-s0": {"kind": "act", "groups": ["camera_relocalization"], "code": "act isg-wave2-dev-todo1 @ 68860d3", "steps": list(range(10000, 200001, 10000))},
    "act-isg1020-iors-small-bs1024-s0": {"kind": "act", "groups": ["iors_small"], "code": "act isg-wave2-dev-todo1 @ 68860d3", "steps": list(range(10000, 200001, 10000))},
    "act-isg1020-multitask5-bs1024-s0": {"kind": "act", "groups": ["camera_relocalization", "object_scaling", "mental_rotation", "cm", "iors_small"],
                                         "code": "act isg-wave2-dev-todo1 @ 68860d3", "steps": list(range(10000, 200001, 10000))},
    "pi05-isg1020-iors-small-gbs256-s0": {"kind": "openpi", "groups": ["iors_small"], "code": "openpi my-clean-todo1 @ 93f8a4e", "steps": list(range(5000, 60001, 5000))},
    "act-isg1020-multitask5-relq-ddp4-bs1024-s0": {"kind": "act", "groups": ["camera_relocalization", "object_scaling", "mental_rotation", "cm", "iors_small"],
                                                   "code": "act isg-wave2-dev-todo1-side @ 8b44f32", "steps": list(range(10000, 200001, 10000))},
    "act-isg1020-multitask5-late-ddp4-bs1024-s0": {"kind": "act", "groups": ["camera_relocalization", "object_scaling", "mental_rotation", "cm", "iors_small"],
                                                   "code": "act isg-wave2-dev-todo1-side @ 8b44f32", "steps": list(range(10000, 200001, 10000))},
}


def expand_groups(groups_doc, names):
    groups = groups_doc["groups"]
    out = []

    def walk(n):
        if n in groups:
            for m in groups[n]:
                walk(m)
        else:
            out.append(n)

    for n in names:
        walk(n)
    return list(dict.fromkeys(out))


def legacy_variants(alias: str):
    """The spellings an eval instance's ``source.variant`` may use for a training alias (the 2026-10-14 rename of the small-scale
    IORS tasks: inside / nextto / touching -> *_small_scale, and the reverse)."""
    out = {alias}
    for a, b in (("-inside-", "-inside_small_scale-"), ("-nextto-", "-nextto_small_scale-"), ("-touching-", "-touching_small_scale-")):
        if a in alias:
            out.add(alias.replace(a, b, 1))
        if b in alias:
            out.add(alias.replace(b, a, 1))
    return out


def decode_frame(path, timestamp):
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        container.seek(max(0, int(timestamp / float(stream.time_base))), stream=stream, backward=True)
        best, distance = None, float("inf")
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            d = abs(t - timestamp)
            if d < distance:
                best, distance = frame, d
            if t >= timestamp:
                break
    if best is None or distance > TOL:
        raise ValueError(f"{path}: no frame within {TOL}s of {timestamp:.6f} (nearest {distance:.6f})")
    return best.to_ndarray(format="rgb24")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--eval-root", default=EVAL_ROOT)
    ap.add_argument("--goals", action=argparse.BooleanOptionalAction, default=True, help="decode the val goal frames")
    args = ap.parse_args(argv)
    inst_dir = os.path.join(args.out, "instances")
    os.makedirs(os.path.join(inst_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(inst_dir, "eval"), exist_ok=True)
    goal_dir = os.path.join(args.out, "goals", "val")
    os.makedirs(goal_dir, exist_ok=True)

    with open(os.path.join(args.dataset, "isg_meta", "task_groups.json")) as f:
        groups_doc = json.load(f)
    with open(args.split) as f:
        split = json.load(f)
    rows = {}
    with open(os.path.join(args.dataset, "meta", "episode_aliases.csv")) as f:
        for r in csv.DictReader(f):
            rows[int(r["episode_index"])] = r
    task_ids = {}
    for r in rows.values():
        task_ids[r["alias"]] = int(r["task_index"])
    with open(os.path.join(inst_dir, "task_ids.json"), "w") as f:
        json.dump(task_ids, f, indent=1, sort_keys=True)

    # the row timestamps of the last frames (one parquet read per data file)
    import pyarrow.parquet as pq

    last_ts = {}
    by_file = defaultdict(list)
    for e, r in rows.items():
        by_file[(int(r["chunk_index"]), int(r["data_file_index"]))].append(e)
    for (chunk, fidx), eps in sorted(by_file.items()):
        path = os.path.join(args.dataset, f"data/chunk-{chunk:03d}/file-{fidx:03d}.parquet")
        table = pq.read_table(path, columns=["timestamp"])
        ts = table.column("timestamp").to_numpy()
        base = int(rows[eps[0]]["dataset_from_index"]) if False else None  # offsets are global dataset indices
        # the data file holds the rows [file_from, file_to); a row's position in the file = dataset index - first dataset index of the file
        first = min(int(rows[e]["dataset_from_index"]) for e in eps)
        for e in eps:
            last_ts[e] = float(ts[int(rows[e]["dataset_to_index"]) - 1 - first])

    # eval dataset index by alias
    eval_by_variant = defaultdict(list)
    with open(os.path.join(args.eval_root, "metadata", "instances.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            eval_by_variant[rec["source"]["variant"]].append(rec)
    episodes = {}
    with open(os.path.join(args.eval_root, "metadata", "episodes.jsonl")) as f:
        for line in f:
            ep = json.loads(line)
            episodes[(ep["subtask"], ep["instance"])] = ep

    tasks_all = sorted(split["val_tasks"])
    summary = {"tasks": {}, "runs": {}}
    missing_goals = []
    for task in tasks_all:
        val_eps = sorted(int(e) for e in split["val_tasks"][task]["val"])
        specs = []
        for e in val_eps:
            r = rows[e]
            alias = r["alias"]
            assert alias == task, (e, alias, task)
            base = r["source_export"].split("/demo/")[0]
            tro = glob.glob(os.path.join(base, "inst", alias, "*", "instances", r["instance_id"] + "-tro_state.json"))
            if len(tro) != 1:
                print(f"WARN episode {e}: {len(tro)} TRO candidates under {base}", file=sys.stderr)
                continue
            demo = os.path.join(base, "demo", r["instance_id"], "demo.hdf5")
            goal_png = os.path.join(goal_dir, f"{e}.png")
            if args.goals and not os.path.exists(goal_png):
                video = os.path.join(args.dataset, f"videos/{HEAD_KEY}/chunk-{int(r['chunk_index']):03d}/file-{int(r['video_file_index']):03d}.mp4")
                try:
                    img = decode_frame(video, float(r["video_from_timestamp"]) + last_ts[e])
                    import av

                    with av.open(goal_png, mode="w") as c:
                        s = c.add_stream("png")
                        s.width, s.height = img.shape[1], img.shape[0]
                        s.pix_fmt = "rgb24"
                        for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                            c.mux(p)
                        for p in s.encode():
                            c.mux(p)
                except Exception as ex:  # noqa: BLE001
                    print(f"WARN episode {e}: goal frame not decoded: {ex!r}", file=sys.stderr)
                    missing_goals.append(e)
            specs.append({"name": r["instance_id"], "tro": tro[0], "demo": demo if os.path.exists(demo) else None, "length": int(r["length"]), "episode_index": e,
                          "alias": alias, "task_id": task_ids[alias], "goal_png": goal_png, "subtask_legacy": r["subtask"], "source": "val",
                          "video_file": f"videos/{HEAD_KEY}/chunk-{int(r['chunk_index']):03d}/file-{int(r['video_file_index']):03d}.mp4",
                          "goal_timestamp": float(r["video_from_timestamp"]) + last_ts[e]})
        with open(os.path.join(inst_dir, "val", task + ".json"), "w") as f:
            json.dump({"task": task, "task_id": task_ids[task], "instances": specs}, f, indent=1)
        # eval instances of the alias
        ev = []
        for v in legacy_variants(task):
            for rec in eval_by_variant.get(v, []):
                ep = episodes.get((rec["subtask"], rec["instance"])) or {}
                iid = os.path.basename(rec["files"]["tro_state"])[: -len("-tro_state.json")]
                ev.append({"name": iid, "tro": os.path.join(args.eval_root, rec["files"]["tro_state"]), "demo": os.path.join(os.path.dirname(args.eval_root), "ispatialgym-eval-demos", ep["demo"]) if ep.get("demo") else None,
                           "length": ep.get("length"), "action_steps": ep.get("action_steps"), "settle_steps": ep.get("settle_steps"), "alias": task, "variant": rec["source"]["variant"],
                           "task_id": task_ids[task], "task_family": rec["task_family"], "task": rec["task"], "subtask": rec["subtask"], "instance": rec["instance"],
                           "goal_png_released": os.path.join(args.eval_root, rec["files"]["goal_rgb"]), "goal_png": os.path.join(args.out, "goals", "eval", iid + "-goal_rgb.png"),
                           "source": "eval-dataset", "dataset_task": rec["task"], "dataset_index": rec["task_instance"]})
        with open(os.path.join(inst_dir, "eval", task + ".json"), "w") as f:
            json.dump({"task": task, "task_id": task_ids[task], "instances": ev}, f, indent=1)
        summary["tasks"][task] = {"task_id": task_ids[task], "val": len(specs), "eval": len(ev), "eval_names": [x["name"] for x in ev]}
    for run, spec in RUNS.items():
        tasks = expand_groups(groups_doc, spec["groups"])
        summary["runs"][run] = {**spec, "tasks": tasks, "n_val": sum(summary["tasks"][t]["val"] for t in tasks), "n_eval": sum(summary["tasks"][t]["eval"] for t in tasks)}
    with open(os.path.join(inst_dir, "runs.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: {"val": v["val"], "eval": v["eval"]} for k, v in summary["tasks"].items()}, indent=0))
    print({run: (v["n_val"], v["n_eval"], len(v["tasks"])) for run, v in summary["runs"].items()})
    if missing_goals:
        print("missing goals:", missing_goals)


if __name__ == "__main__":
    main()
