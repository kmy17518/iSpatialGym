#!/usr/bin/env python3
"""Author distinct object-scaling goals before starts and queue them in one worker."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.campaign_20261110 import goal_groups as g
from scripts.campaign_20261110.contracts import digest, file_digest, require, write_json_atomic
from ispatialgym.sampling.campaign_goal_queue import ENV, SCHEMA, canonical_goal, read_queue
from ispatialgym.sampling.campaign_held_goals import GoalPlanError, draw_start_depth


def rotate(q, v):
    x, y, z, w = q
    cross = lambda a, b: (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    uv = cross((x,y,z), v)
    uuv = cross((x,y,z), uv)
    return [v[i] + 2*(w*uv[i]+uuv[i]) for i in range(3)]


def depth_goal_proposals(anchor, count, *, magnitude_range, seed, minimum_separation_m=.005):
    """Use fresh template depth windows; every resulting goal still needs native IK."""
    require(isinstance(count, int) and 1 <= count <= 100, "invalid batch goal count")
    require(minimum_separation_m >= .005, "depth goals need at least 5 mm separation")
    summary, meta = anchor["summary"], anchor["template_meta"]
    goal, base = summary["goal_realised"], summary["base_pose2d"]
    old_depth = float(summary["goal"]["goal_depth_m"])
    camera = meta["hold"]["camera_in_base"]
    q = [float(v) for v in camera["orientation"]]
    require(abs(sum(v*v for v in q)-1) < 1e-4, "anchor camera orientation invalid")
    norm = math.sqrt(sum(v*v for v in q))
    local_axis = rotate([v/norm for v in q], (0,0,-1))
    yaw = base[2]
    axis = [math.cos(yaw)*local_axis[0]-math.sin(yaw)*local_axis[1],
            math.sin(yaw)*local_axis[0]+math.cos(yaw)*local_axis[1], local_axis[2]]
    require(all(abs(float(a)-float(b)) < .001 for a,b in zip(summary["trunk"], meta["hold"]["trunk"])),
            "fresh anchor trunk differs from validated camera frame")
    windows = [v["window"] for v in meta["template_facts"]["depth_windows"].values()]
    require(windows and all(len(w)==2 and w[0]<w[1] for w in windows), "anchor lacks native depth windows")
    rng = random.Random(seed)
    proposals, selected = [], [old_depth]
    for _ in range(count*1000):
        window = windows[rng.randrange(len(windows))]
        lo, hi = float(window[0])+.005, float(window[1])-.005
        if lo >= hi:
            continue
        depth = rng.uniform(lo,hi)
        if any(abs(depth-v) < minimum_separation_m for v in selected):
            continue
        try:
            draw_start_depth(rng, depth, window, magnitude_range)
        except GoalPlanError:
            continue
        pose = {"position": [float(goal["position"][i])+axis[i]*(depth-old_depth) for i in range(3)],
                "orientation": list(goal["orientation"])}
        proposals.append({"goal_pose": pose, "depth_m": depth, "source_goal_depth_m": old_depth,
                          "native_depth_window_m": list(window), "optical_axis_world": axis,
                          "minimum_separation_m": minimum_separation_m, "proposal_validation": "native_interval_only; full sampler IK/path/physics still required"})
        selected.append(depth)
        if len(proposals) == count:
            return proposals
    raise ValueError("native goal region cannot supply the requested separated goals")


def prepare_batch(*, anchor, template, campaign_root, batch_id, count, magnitude_range, seed,
                  campaign_id, code_sha256, physics_sha256, data_sha256, budget_seconds=900):
    require(template.task == "object_scaling", "automatic batch goal proposals currently support object scaling only")
    root = Path(campaign_root).resolve()
    require(batch_id and all(c.isalnum() or c in "_.-" for c in batch_id), "invalid batch ID")
    queue_dir = root / "queues" / batch_id
    require(not queue_dir.exists(), "queue already exists")
    proposals = depth_goal_proposals(anchor,count,magnitude_range=magnitude_range,seed=seed)
    # Existing group plans are also excluded, including diagnostic/superseded ones.
    prior_goals = set()
    for path in (root/"groups").glob("*/goal_plan.json"):
        old = g.read_json(path)
        if old.get("task") == template.task and old.get("scene") == template.scene:
            prior_goals.add(digest(canonical_goal(old)))
    entries, groups = [], []
    for i, proposal in enumerate(proposals):
        require(digest(canonical_goal(proposal)) not in prior_goals, "proposed goal duplicates a previous group")
        group_id = f"{batch_id}_{i:04d}"
        derived = copy.deepcopy(anchor)
        derived["summary"]["goal_realised"] = proposal["goal_pose"]
        derived["goal_proposal_source"] = "fresh_anchor_native_depth_region_before_any_group_start"
        directory = root/"groups"/group_id
        g.prepare_group(anchor=derived,template=template,campaign_id=campaign_id,group_id=group_id,output=directory,
                        code_sha256=code_sha256,physics_sha256=physics_sha256,data_sha256=data_sha256,
                        magnitude_range=magnitude_range,budget_seconds=budget_seconds,max_members=3)
        write_json_atomic(directory/"goal_proposal.json", {**proposal,"anchor_sources":anchor["sources"],"seed":seed,"ordinal":i})
        plan_path = directory/"goal_plan.json"
        entries.append({"path":str(plan_path),"sha256":file_digest(plan_path)})
        groups.append(group_id)
    config_path = root/"groups"/groups[0]/"campaign_config.json"
    require(all(file_digest(root/"groups"/name/"campaign_config.json") == file_digest(config_path) for name in groups),
            "queued groups differ in scene/scale/light/config identity")
    queue_dir.mkdir(parents=True,exist_ok=False)
    queue_path=queue_dir/"queue.json"
    write_json_atomic(queue_path,{"schema":SCHEMA,"campaign_id":campaign_id,"config_sha256":file_digest(config_path),
                                "config_path":str(config_path),"entries":entries,"group_ids":groups,"seed":seed})
    read_queue(queue_path,config_path)
    return {"queue":str(queue_path),"config":str(config_path),"group_ids":groups,"candidate_instances":3*len(groups)}


def submission_command(queue_path, **kwargs):
    raw=g.read_json(queue_path)
    state=read_queue(queue_path,raw["config_path"])
    first=Path(state["entries"][0]["path"]).parent
    command=g.submission_command(first,**kwargs)
    command[command.index("submit")+2]=str(sum(row["max_members"] for row in state["entries"]))
    budgets=sum(float(g.read_json(row["path"])["budget_seconds"]) for row in state["entries"])
    command[command.index("--sample-limit-s")+1]=str(int(budgets+1800))
    command[command.index("--time-s")+1]=str(int(budgets+3600))
    index=next(i for i,value in enumerate(command) if value.startswith("ISPATIALGYM_CAMPAIGN_GOAL_PLAN="))
    del command[index-1:index+1]
    command.extend(["--extra-env",ENV+"="+str(Path(queue_path).resolve())])
    return command


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("anchor-job","campaign-root","anchor-ledger","contracts","class-id","campaign-id","batch-id","code-identity-file","physics-spec","data-manifest"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--campaign-started-at",type=float,required=True)
    parser.add_argument("--count",type=int,required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--magnitude-range",type=float,nargs=2,required=True)
    parser.add_argument("--budget-seconds",type=float,default=900)
    a=parser.parse_args()
    anchor=g.successful_anchor(a.anchor_job,campaign_root=a.campaign_root,ledger=a.anchor_ledger,campaign_started_at=a.campaign_started_at)
    print(json.dumps(prepare_batch(anchor=anchor,template=g.load_template_contract(a.contracts,a.class_id),campaign_root=a.campaign_root,
        batch_id=a.batch_id,count=a.count,magnitude_range=a.magnitude_range,seed=a.seed,campaign_id=a.campaign_id,
        code_sha256=file_digest(a.code_identity_file),physics_sha256=file_digest(a.physics_spec),data_sha256=file_digest(a.data_manifest),
        budget_seconds=a.budget_seconds)))


if __name__ == "__main__":
    main()
