#!/usr/bin/env python3
"""Author distinct held-object goals before starts and queue them in one worker."""
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
from ispatialgym.sampling.campaign_goal_queue import ENV, SCHEMA, read_queue
from ispatialgym.sampling.campaign_held_goals import GoalPlanError, draw_start_depth
from scripts.campaign_20261110.goal_reuse_policy import POLICY as GOAL_REUSE_POLICY


def rotate(q, v):
    x, y, z, w = q
    cross = lambda a, b: (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    uv = cross((x,y,z), v)
    uuv = cross((x,y,z), uv)
    return [v[i] + 2*(w*uv[i]+uuv[i]) for i in range(3)]


def quaternion_product(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    out = [aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx,
           aw*bz+ax*by-ay*bx+az*bw, aw*bw-ax*bx-ay*by-az*bz]
    norm = math.sqrt(sum(v*v for v in out))
    require(norm > 0 and math.isfinite(norm), "invalid composed goal orientation")
    return [v/norm for v in out]


def axis_angle(axis, degrees):
    norm = math.sqrt(sum(v*v for v in axis))
    require(norm > 0 and math.isfinite(norm), "invalid rotation axis")
    half = math.radians(degrees)/2
    return [v/norm*math.sin(half) for v in axis]+[math.cos(half)]


def rotation_goal_proposals(anchor, count, *, seed, minimum_separation_deg=5, maximum_anchor_offset_deg=20):
    """Native goal-angle region about a fresh reference pose, before new starts."""
    require(isinstance(count, int) and 1 <= count <= 100, "invalid batch goal count")
    require(minimum_separation_deg >= 5, "rotation goals need at least five degrees separation")
    require(minimum_separation_deg < maximum_anchor_offset_deg <= 45, "invalid bounded goal neighborhood")
    summary, meta = anchor["summary"], anchor["template_meta"]
    goal = summary["goal_realised"]
    lo, hi = (float(v) for v in meta["init_orientation"]["goal_angle_range_deg"])
    require(0 <= lo < hi <= 180, "invalid native rotation range")
    source_rotation = summary["goal"]
    # Reconstruct the fresh reference from its recorded world-axis turn. This
    # reference never becomes a final group member or constrains future starts.
    inverse = axis_angle(source_rotation["axis_world"], -float(source_rotation["angle_deg"]))
    reference = quaternion_product(inverse, goal["orientation"])
    rng, proposals, selected = random.Random(seed), [], [goal["orientation"]]
    for _ in range(count*1000):
        z, phi = rng.uniform(-1, 1), rng.uniform(-math.pi, math.pi)
        axis = [math.sqrt(1-z*z)*math.cos(phi), math.sqrt(1-z*z)*math.sin(phi), z]
        offset = rng.uniform(minimum_separation_deg, maximum_anchor_offset_deg)
        orientation = quaternion_product(axis_angle(axis, offset),goal["orientation"])
        relative = quaternion_product(orientation,[-reference[0],-reference[1],-reference[2],reference[3]])
        if relative[3] < 0:
            relative = [-v for v in relative]
        angle = math.degrees(2*math.acos(min(1,relative[3])))
        if not lo <= angle <= hi:
            continue
        scale = math.sqrt(sum(v*v for v in relative[:3]))
        if scale < 1e-10:
            continue
        native_axis = [v/scale for v in relative[:3]]
        distance = lambda q: math.degrees(2*math.acos(min(1, abs(sum(a*b for a,b in zip(q,orientation))))))
        if any(distance(q) < minimum_separation_deg for q in selected):
            continue
        proposals.append({"goal_pose": {"position": list(goal["position"]), "orientation": orientation},
            "reference_orientation": reference, "axis_world": native_axis, "goal_angle_deg": angle,
            "anchor_goal_offset_deg":offset,"maximum_anchor_offset_deg":maximum_anchor_offset_deg,
            "native_goal_angle_range_deg": [lo, hi], "minimum_separation_deg": minimum_separation_deg,
            "proposal_validation": "native_angle_region_only; full sampler frame/IK/path/physics still required"})
        selected.append(orientation)
        if len(proposals) == count:
            return proposals
    raise ValueError("native rotation region cannot supply separated goals")


def move_goal_with_base(proposal, original_base, new_base):
    """Rigidly move the preselected world goal with its base-relative camera."""
    old_x, old_y, old_yaw = (float(v) for v in original_base)
    x, y, yaw = (float(v) for v in new_base)
    require(all(math.isfinite(v) for v in (x,y,yaw)), "nonfinite base proposal")
    delta = yaw-old_yaw
    rotation = [0,0,math.sin(delta/2),math.cos(delta/2)]
    goal = proposal["goal_pose"]
    offset = rotate(rotation, [goal["position"][0]-old_x,goal["position"][1]-old_y,goal["position"][2]])
    moved = {**proposal, "goal_pose": {"position": [x+offset[0],y+offset[1],offset[2]],
        "orientation": quaternion_product(rotation,goal["orientation"])}, "base_pose2d": [x,y,yaw],
        "source_base_pose2d": list(original_base)}
    for name in ("axis_world", "optical_axis_world"):
        if name in proposal:
            moved[name] = rotate(rotation, proposal[name])
    if "reference_orientation" in proposal:
        moved["reference_orientation"] = quaternion_product(rotation, proposal["reference_orientation"])
    return moved


def projected_goal_clear(base, goal, free_cells, resolution, origin=(0, 0)):
    """Conservative proposal filter; loaded-scene arm/object checks remain native."""
    dx, dy = goal[0]-base[0], goal[1]-base[1]
    steps = max(1, math.ceil(math.hypot(dx, dy)/(resolution/2)))
    cells = [(round((base[0]+dx*i/steps-origin[0])/resolution), round((base[1]+dy*i/steps-origin[1])/resolution))
             for i in range(steps+1)]
    # Adjacent diagonal steps must not cut the corner of a blocked cell.
    for before, after in zip(cells, cells[1:]):
        if after not in free_cells or (before[0] != after[0] and before[1] != after[1]
                and ((before[0],after[1]) not in free_cells or (after[0],before[1]) not in free_cells)):
            return False
    return cells[0] in free_cells


def apply_base_region(proposals, anchor, region, *, seed):
    """Select distinct declared free room cells; physical feasibility stays native."""
    require(region.get("schema") == "isg-room-base-region-v1", "missing room base region")
    meta = anchor["template_meta"]
    require(region["scene"] == meta["scene"], "base region scene differs from anchor")
    require(region["room"] in anchor["tro"]["ispatialgym"]["scene"]["task_rooms"], "base region outside task rooms")
    sources = region.get("sources", [])
    require(sources and all(file_digest(row["path"]) == row["sha256"] for row in sources), "base region map source changed")
    from scripts.campaign_20261110.base_regions import room_base_region
    require(digest(room_base_region(region["dataset"],region["scene"],region["room"])) == digest(region),
            "base-region cells differ from their actual frozen maps")
    points = region["free_xy"]
    require(len(points) >= len(proposals) and all(len(p)==2 and all(math.isfinite(float(v)) for v in p) for p in points),
            "base region lacks enough finite free cells")
    require(len({tuple(p) for p in points}) == len(points), "duplicate free cells in base region")
    rng = random.Random(seed)
    resolution = float(region["map_resolution_m"])
    origin = points[0]
    cells = {(round((x-origin[0])/resolution),round((y-origin[1])/resolution)) for x,y in points}
    used, result = set(), []
    for proposal in proposals:
        for _ in range(max(1000, len(points)*10)):
            xy = rng.choice(points)
            if tuple(xy) in used:
                continue
            moved = move_goal_with_base(proposal,anchor["summary"]["base_pose2d"],[*xy,rng.uniform(-math.pi,math.pi)])
            if not projected_goal_clear(xy, moved["goal_pose"]["position"], cells, resolution, origin):
                continue
            used.add(tuple(xy))
            result.append({**moved,"base_region_sha256":digest(region),
                "base_region_validation":"base and projected base-to-goal segment in eroded room free cells without corner cutting; native collision/IK/path checks still required"})
            break
        else:
            raise ValueError("native room map cannot supply a new clear base-to-goal proposal")
    return result


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
                  campaign_id, code_sha256, physics_sha256, data_sha256, budget_seconds=900, base_region=None):
    require(template.task in ("object_scaling", "mental_rotation"), "automatic batch requires a held-object task")
    root = Path(campaign_root).resolve()
    require(batch_id and all(c.isalnum() or c in "_.-" for c in batch_id), "invalid batch ID")
    queue_dir = root / "queues" / batch_id
    require(not queue_dir.exists(), "queue already exists")
    proposals = (depth_goal_proposals(anchor,count,magnitude_range=magnitude_range,seed=seed) if template.task == "object_scaling"
                 else rotation_goal_proposals(anchor,count,seed=seed))
    if base_region is not None:
        proposals = apply_base_region(proposals,anchor,base_region,seed=seed)
    entries, groups = [], []
    for i, proposal in enumerate(proposals):
        group_id = f"{batch_id}_{i:04d}"
        derived = copy.deepcopy(anchor)
        derived["summary"]["goal_realised"] = proposal["goal_pose"]
        if "base_pose2d" in proposal:
            derived["summary"]["base_pose2d"] = proposal["base_pose2d"]
        derived["goal_proposal_source"] = "fresh_anchor_native_goal_region_before_any_group_start"
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
                                "config_path":str(config_path),"entries":entries,"group_ids":groups,"seed":seed,
                                "goal_reuse_policy":GOAL_REUSE_POLICY,
                                "base_region_sha256": digest(base_region) if base_region is not None else None})
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
    parser.add_argument("--base-region",help="Hashed scene-map free-cell manifest for varied room bases")
    a=parser.parse_args()
    anchor=g.successful_anchor(a.anchor_job,campaign_root=a.campaign_root,ledger=a.anchor_ledger,campaign_started_at=a.campaign_started_at)
    print(json.dumps(prepare_batch(anchor=anchor,template=g.load_template_contract(a.contracts,a.class_id),campaign_root=a.campaign_root,
        batch_id=a.batch_id,count=a.count,magnitude_range=a.magnitude_range,seed=a.seed,campaign_id=a.campaign_id,
        code_sha256=file_digest(a.code_identity_file),physics_sha256=file_digest(a.physics_spec),data_sha256=file_digest(a.data_manifest),
        budget_seconds=a.budget_seconds,base_region=g.read_json(a.base_region) if a.base_region else None)))


if __name__ == "__main__":
    main()
