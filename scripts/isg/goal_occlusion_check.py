#!/usr/bin/env python3
"""The pre-generation goal-image occlusion rule on instances already written: what ``SamplerV2.check_goal_occlusion``
predicts for an instance sampled before the rule existed (a queue of instances waiting for demos, or the instances behind
recorded demos to compare with their measured last frames).

Per instance, from its summary (the sampler's own records): the goal entities at the goal check's settled poses, the
articulated objects at their TRO joint targets, the robot at the program check's last witness (the last action's retreat /
end pose) at that action's stance; then one rendering step and the head camera's depth through
``ispatialgym.eval.goal_occlusion``'s box / panel screens. An instance whose last action records no witness (a closed-loop
push-close) is reported as skipped. The world is built once per template (run_isg's --queue reuse) and only read: nothing
is written next to the instances.

    python scripts/isg/goal_occlusion_check.py --out <results.jsonl> <tro> [<tro> ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import torch as th

from ispatialgym.eval import goal_occlusion as GO
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle import run_isg as RI
from ispatialgym.oracle.head_depth import settled_depth
from ispatialgym.oracle.qa import CameraModel


def final_configuration(summary):
    acts = (summary.get("program_check") or {}).get("actions") or []
    if not acts:
        return None
    last = acts[-1]
    wit = last.get("witnesses") or {}
    key = "retreat" if "retreat" in wit else (list(wit)[-1] if wit else None)
    stance = (summary.get("stances") or {}).get(last.get("stance") or "")
    if key is None or wit[key].get("q") is None or stance is None:
        return None
    return wit[key]["q"], stance["base_pose2d"], {"action": last.get("name"), "index": last.get("index"), "witness": key, "stance": last.get("stance")}


def predict(loader, summary, raw_tro):
    import omnigibson as og
    from omnigibson.utils.constants import JointType

    env, robot = loader.env, loader.robot
    final = final_configuration(summary)
    if final is None:
        return {"skipped": "no witness at the program's end"}
    q, base_pose2d, source = final
    scene = env.scene
    moved = []
    for name, pose in ((summary.get("goal_check") or {}).get("settled_poses") or {}).items():
        obj = scene.object_registry("name", name)
        if obj is not None:
            obj.set_position_orientation(th.tensor(pose[0], dtype=th.float32), th.tensor(pose[1], dtype=th.float32))
            moved.append(obj)
    for inst, target in (raw_tro["ispatialgym"].get("articulation_targets") or {}).items():
        obj = loader.physical.get(inst)
        if obj is None:
            continue
        qs = obj.get_joint_positions().clone()
        names = list(obj.joints)
        for joint, rec in (target.get("joints") or {}).items():
            if joint in names:
                qs[names.index(joint)] = float(rec["position"])
        obj.set_joint_positions(qs)
        obj.set_joint_velocities(th.zeros_like(qs))
        moved.append(obj)
    qk = th.tensor(q, dtype=th.float32)
    qk[[M.ALL_JOINTS.index(n) for n in M.BASE_VIRTUAL_JOINTS]] = 0.0
    for drive in (False, True):  # the joint state and the drive targets (SimWorld.teleport_robot)
        robot.set_joint_positions(qk, drive=drive)
        robot.set_joint_velocities(th.zeros_like(qk), drive=drive)
    z = float(raw_tro["robot_poses"]["robot"][0]["position"][2])
    x, y, yaw = (float(v) for v in base_pose2d)
    robot.set_position_orientation(th.tensor([x, y, z]), th.tensor([0.0, 0.0, float(th.sin(th.tensor(yaw / 2))), float(th.cos(th.tensor(yaw / 2)))]))
    loader.hold.hold_current()
    robot.apply_action(loader.hold.action())
    for obj in moved:
        obj.wake()
    robot.keep_still()
    og.sim.step()
    for obj in list(loader.physical.values()):
        obj.keep_still()
    robot.keep_still()
    sensor = robot.sensors[loader.cam_key]
    judged = [o for o in dict.fromkeys(loader.tracked.values()) if GO.role(getattr(o, "name", ""))]
    without = None
    depth = settled_depth(sensor, og.sim.render)
    cp, cq = sensor.get_position_orientation()
    if any(GO.role(o.name) == "fixture" for o in judged):
        robot.visible = False
        try:
            without = settled_depth(sensor, og.sim.render)
        finally:
            robot.visible = True
    objects = GO.screen_scene(depth, (cp.float().tolist(), cq.float().tolist()), GO.intrinsics_of(CameraModel.from_sensor(sensor)), judged,
                              depth_without_robot=without, revolute=JointType.JOINT_REVOLUTE)
    return GO.report(objects, method="predicted from the instance summary (goal_occlusion_check.py), head-camera depth", robot=source)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tros", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    tros = sorted(os.path.abspath(t) for t in args.tros)
    shared = {}
    for tro in tros:
        t0 = time.time()
        rec = {"tro": tro, "instance": os.path.basename(tro)[: -len("-tro_state.json")]}
        try:
            summary = json.load(open(tro.replace("-tro_state.json", ".summary.json")))
            raw = json.load(open(tro))
            loader = RI.InstanceLoader(tro)
            if shared.get("env") is not None and os.path.abspath(shared["template_dir"]) == os.path.abspath(loader.template_dir):
                loader.attach(shared)
            else:
                if shared.get("env") is not None:
                    raise RuntimeError("one template per process: run the other template's instances in another process")
                loader.build_env()
                loader.share(shared)
            shared["loader"] = loader
            loader.restore()
            rec["alias"] = loader.tmeta.get("alias")
            if GO.exempt(rec["alias"]):
                rec["skipped"] = "no target / partner"
            else:
                rec.update(predict(loader, summary, raw))
        except Exception as e:  # noqa: BLE001
            rec["error"] = repr(e)
            rec["traceback"] = traceback.format_exc()[-2000:]
        rec["seconds"] = round(time.time() - t0, 1)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print("GOAL_OCCLUSION_CHECK " + json.dumps({k: rec.get(k) for k in ("instance", "max_occluded", "worst", "occluded", "skipped", "error")}), flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
