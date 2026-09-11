"""Reload every generated instance in a fresh process and re-check it the way the evaluator will see it.

For each ``<out>/<problem>/<template>/`` the scene template is loaded with ``InteractiveTraversableScene`` (the robot
comes from the template), an ``ISpatialGymTask`` is built from the template's ``activity_definitions`` and each
``instances/*-tro_state.json`` is applied with ``ISpatialGymTask.load_tro_instance``. Checks per instance:

* the BDDL initial conditions hold after loading,
* the binary goal is *not* satisfied at init,
* stability: after ``--extra-steps`` further physics steps no task object (or the robot base) drifted,
* the robot base pose / joint positions match the TRO,
* goal feasibility: objects teleported to their world-frame pose targets / group slots / joint targets satisfy the
  binary goal after settling (skipped for targets that are not full world poses).

Instances whose TRO carries ``grasps`` or ``constraints`` need loader support that does not exist yet (assisted
grasp / hinge restoration); their grasp- or hinge-dependent checks are reported as ``skipped`` instead of failed.

Example::

    OMNIGIBSON_HEADLESS=1 python -m ispatialgym.sampling.validate_instances --out /tmp/isg_out
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
import traceback

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import torch as th  # noqa: E402

log = logging.getLogger("ispatialgym.validate")


def _quat_angle(q0, q1) -> float:
    q0 = th.as_tensor(q0, dtype=th.float32)
    q1 = th.as_tensor(q1, dtype=th.float32)
    d = float(abs(th.dot(q0 / th.linalg.norm(q0), q1 / th.linalg.norm(q1))))
    return 2.0 * float(th.arccos(th.clamp(th.tensor(d), max=1.0)))


def validate_template(template_dir: str, extra_steps: int, check_goal: bool, pos_tol: float, ang_tol: float):
    import omnigibson as og
    from omnigibson.macros import gm
    from omnigibson.utils.python_utils import recursively_convert_to_torch

    for name, value in (("ENABLE_OBJECT_STATES", True), ("USE_GPU_DYNAMICS", False), ("ENABLE_TRANSITION_RULES", False)):
        if getattr(gm, name) != value:
            with gm.unlocked():
                setattr(gm, name, value)

    from ispatialgym.eval.group_entity import GroupEntity
    from ispatialgym.eval.tro import load_tro
    from ispatialgym.oracle import r1pro_model as M
    from ispatialgym.tasks import ISpatialGymTask  # noqa: F401  (registers the task type)

    with open(os.path.join(template_dir, "template_meta.json")) as f:
        meta = json.load(f)
    problem = meta["problem"]
    activity, did = problem.rsplit("-", 1)
    tros = sorted(glob.glob(os.path.join(template_dir, "instances", "*-tro_state.json")))
    results = {"template": template_dir, "problem": problem, "instances": {}}
    if not tros:
        results["error"] = "no instances"
        return results
    from ispatialgym.sampling.sim_utils import IMAGE_SIZE, PARK_POSITION

    cfg = {
        "env": {"action_frequency": 30, "physics_frequency": 120},
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": meta["scene"],
            "scene_file": os.path.join(template_dir, "template.json"),
            "load_room_instances": meta.get("load_room_instances") or None,
        },
        # the robot saved in the template carries no room assignment, so partial-room loading skips it: the evaluation
        # robot comes from the config (same JoyLo configuration as the sampler) and the TRO restores its pose / joints
        "robots": [
            M.joylo_robot_config(
                "robot_r1",
                obs_modalities=["rgb", "seg_instance"],
                sensor_config={"VisionSensor": {"sensor_kwargs": {"image_height": IMAGE_SIZE, "image_width": IMAGE_SIZE}}},
                position=PARK_POSITION.tolist(),
            )
        ],
        "task": {
            "type": "ISpatialGymTask",
            "activity_name": activity,
            "activity_definition_id": int(did),
            "activity_definitions_root": os.path.join(template_dir, "activity_definitions"),
            "tro_path": tros[0],
            "termination_config": {"max_steps": 10 ** 6},
        },
    }
    # the template must equal the stable scene json except for the task objects and the robot
    from ispatialgym.sampling.sim_utils import stable_scene_path

    with open(stable_scene_path(meta["scene"])) as f:
        stable_reg = json.load(f)["state"]["registry"]["object_registry"]
    with open(os.path.join(template_dir, "template.json")) as f:
        tmpl_reg = json.load(f)["state"]["registry"]["object_registry"]
    task_names = set(meta.get("inst_to_name", {}).values())
    devs = {}
    for name, st in tmpl_reg.items():
        if name in task_names or name not in stable_reg or "root_link" not in st:
            continue
        dp = float(th.linalg.norm(th.tensor(st["root_link"]["pos"]) - th.tensor(stable_reg[name]["root_link"]["pos"])))
        da = _quat_angle(st["root_link"]["ori"], stable_reg[name]["root_link"]["ori"])
        if dp > 1e-3 or da > 5e-3:
            devs[name] = {"pos": round(dp, 4), "ang": round(da, 4)}
    results["template_scene_deviations"] = devs
    if devs:
        log.warning("%s: %d non-task scene objects differ from the stable scene json: %s", problem, len(devs), dict(list(devs.items())[:5]))

    t0 = time.time()
    env = og.Environment(configs=cfg)
    results["scene_load_s"] = time.time() - t0
    task = env.task
    robot = env.robots[0]
    for path in tros:
        iid = os.path.basename(path).replace("-tro_state.json", "")
        rec = {"ok": True, "checks": {}, "notes": []}
        t1 = time.time()
        try:
            tro = load_tro(path)
            extras = tro.meta.extras
            constrained = {c["child"] for c in (extras.get("constraints") or [])} | {g["object"] for g in (extras.get("grasps") or [])}
            if constrained:
                rec["notes"].append(f"loader cannot restore grasps / constraints yet for {sorted(constrained)}; dependent checks skipped")
            task.load_tro_instance(env, tro=tro, settle_steps=25)
            scope = task.object_scope
            physical = {k: v for k, v in scope.items() if v is not None and not isinstance(v, GroupEntity)}
            # kinematic predicates cache per step and treat sleeping objects as unchanged: wake + one full step
            for ent in physical.values():
                if hasattr(ent, "wake"):
                    ent.wake()
            og.sim.step()
            for ent in task.group_entities.values():
                ent.invalidate()
            # robot pose / joints
            rp = tro.robot_poses["robot"][0]
            p, q = robot.get_position_orientation()
            rec["checks"]["robot_pose"] = {"pos_err": float(th.linalg.norm(p - th.tensor(rp["position"]))), "ang_err": _quat_angle(q, rp["orientation"])}
            jp = (tro.meta.robot_state or {}).get("joint_positions") or {}
            names = list(robot.joints.keys())
            cur = robot.get_joint_positions()
            jerr = max([abs(float(cur[names.index(n)]) - float(v)) for n, v in jp.items() if n in names] or [0.0])
            rec["checks"]["robot_joint_err"] = jerr
            if devs:
                rec["ok"] = False
                rec["notes"].append(f"template differs from the stable scene json for {len(devs)} non-task objects (max {max(d['pos'] for d in devs.values()):.3f} m)")
            # init conditions
            ok_init, status = task.compiled_task.check_initial_conditions(task._evaluate_predicate)
            if not ok_init and constrained:
                rec["checks"]["init"] = "skipped (grasp/constraint)"
            else:
                rec["checks"]["init"] = bool(ok_init)
                if not ok_init:
                    rec["ok"] = False
                    rec["notes"].append(f"init false: {status}")
            # goal not satisfied at init
            _, gstatus = task.compiled_task.check_goal(task._evaluate_predicate)
            binary = set(task.compiled_task.binary_goal_head_idxs)
            goal_now = binary <= set(gstatus["satisfied"])
            rec["checks"]["goal_false_at_init"] = not goal_now
            if goal_now:
                rec["ok"] = False
                rec["notes"].append("binary goal already satisfied at init")
            # stability
            before = {k: tuple(x.clone() for x in v.get_position_orientation()) for k, v in physical.items()}
            for _ in range(extra_steps):
                og.sim.step()
            drift = {}
            for k, v in physical.items():
                if k in constrained or v is robot:
                    continue
                p1, q1 = v.get_position_orientation()
                dp = float(th.linalg.norm(p1 - before[k][0]))
                da = _quat_angle(q1, before[k][1])
                if dp > pos_tol or da > ang_tol:
                    drift[k] = {"pos": dp, "ang": da}
            rp1, _ = robot.get_position_orientation()
            rb = float(th.linalg.norm(rp1[:2] - p[:2]))
            rec["checks"]["drift"] = drift
            rec["checks"]["robot_base_drift"] = rb
            if drift or rb > pos_tol:
                rec["ok"] = False
                rec["notes"].append(f"unstable after {extra_steps} steps: {drift} robot {rb:.3f}")
            # goal feasibility (world-frame full poses / slots / joints). Informational: the sampler already verified the
            # goal with the task-specific mechanics (attachment joints, hinges, contact nudging) that a plain teleport
            # cannot reproduce here.
            if check_goal:
                if extras.get("attachment_targets"):
                    rec["checks"]["goal_feasible"] = "skipped (attachment needs AttachedTo.set_value)"
                else:
                    rec["checks"]["goal_feasible"] = _goal_feasible(env, task, tro, physical, constrained, og, recursively_convert_to_torch)
                    if rec["checks"]["goal_feasible"] is False:
                        rec["notes"].append("warning: binary goal not satisfied after a plain teleport to the targets")
            env.reset()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            rec["ok"] = False
            rec["notes"].append(f"exception: {e!r}")
        rec["time_s"] = time.time() - t1
        results["instances"][iid] = rec
        log.info("%s: %s %s", iid, "OK" if rec["ok"] else "FAIL", "; ".join(rec["notes"]))
    og.clear()
    return results


def _goal_feasible(env, task, tro, physical, constrained, og, to_torch):
    """Teleport objects to their targets, settle, evaluate the binary goal, restore via env.reset()."""
    meta = tro.meta
    moved = False
    for inst, tgt in meta.pose_targets.items():
        ent = physical.get(inst)
        if ent is None or ent is env.robots[0] or inst in constrained:
            continue
        if tgt.frame != "world" or tgt.position is None or tgt.orientation is None or tgt.position_metric != "full":
            continue
        ent.set_position_orientation(th.as_tensor(tgt.position, dtype=th.float32), th.as_tensor(tgt.orientation, dtype=th.float32))
        ent.keep_still()
        moved = True
    for gname, g in meta.groups.items():
        members = task.groups.get(gname, [])
        slots = list(g.target_slots.items())
        if g.assignment_mode == "fixed" and g.member_to_slot:
            pairs = [(m, g.target_slots[g.member_to_slot[m]]) for m in members if m in g.member_to_slot]
        else:
            pairs = [(m, s) for m, (_, s) in zip(members, slots)]
        for m, slot in pairs:
            ent = physical.get(m)
            if ent is None or m in constrained:
                continue
            ent.set_position_orientation(th.as_tensor(slot.position, dtype=th.float32), th.as_tensor(slot.orientation, dtype=th.float32))
            ent.keep_still()
            moved = True
    for inst, art in meta.articulation_targets.items():
        ent = physical.get(inst)
        if ent is None:
            continue
        names = list(ent.joints.keys())
        q = ent.get_joint_positions().clone()
        for jn, (jpos, _tol) in art.joints.items():
            if jn in names:
                q[names.index(jn)] = float(jpos)
        ent.set_joint_positions(q)
        ent.set_joint_velocities(th.zeros_like(q))
        ent.keep_still()
        moved = True
    if not moved:
        return "skipped (no world-frame targets)"
    for _ in range(120):
        og.sim.step_physics()
        env.robots[0].keep_still()
    # kinematic predicates treat sleeping objects as unchanged: wake everything and take one full step first
    for ent in physical.values():
        if hasattr(ent, "wake"):
            ent.wake()
    og.sim.step()
    for ent in task.group_entities.values():
        ent.invalidate()
    _, status = task.compiled_task.check_goal(task._evaluate_predicate)
    binary = set(task.compiled_task.binary_goal_head_idxs)
    ok = binary <= set(status["satisfied"])
    if not ok:
        log.info("goal status after teleport: unsatisfied %s", sorted(binary - set(status["satisfied"])))
    return bool(ok)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="sampler output directory")
    parser.add_argument("--problem", action="append", default=None)
    parser.add_argument("--extra-steps", type=int, default=60)
    parser.add_argument("--pos-tol", type=float, default=0.01)
    parser.add_argument("--ang-tol", type=float, default=0.05)
    parser.add_argument("--no-goal", action="store_true", help="skip the goal-feasibility teleport check")
    parser.add_argument("--report", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="[%(levelname)s %(name)s] %(message)s", stream=sys.stdout, force=True)

    templates = []
    for problem_dir in sorted(glob.glob(os.path.join(args.out, "*"))):
        pname = os.path.basename(problem_dir)
        if not os.path.isdir(problem_dir) or (args.problem and pname not in args.problem):
            continue
        for tdir in sorted(glob.glob(os.path.join(problem_dir, "*"))):
            if os.path.exists(os.path.join(tdir, "template_meta.json")):
                templates.append(tdir)
    report = {"templates": [], "n_ok": 0, "n_fail": 0}
    import omnigibson as og

    try:
        for tdir in templates:
            try:
                res = validate_template(tdir, args.extra_steps, not args.no_goal, args.pos_tol, args.ang_tol)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                res = {"template": tdir, "error": repr(e), "instances": {}}
                try:
                    og.clear()
                except Exception:
                    pass
            report["templates"].append(res)
            for rec in res["instances"].values():
                report["n_ok" if rec["ok"] else "n_fail"] += 1
    finally:
        path = args.report or os.path.join(args.out, f"validation_report_{int(time.time())}.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print("VALIDATION_REPORT " + json.dumps({"n_ok": report["n_ok"], "n_fail": report["n_fail"], "report": path,
                                                 "fails": [f"{os.path.basename(t['template'])}/{i}: {r['notes']}" for t in report["templates"] for i, r in t["instances"].items() if not r["ok"]]
                                                 + [f"{t['template']}: {t['error']}" for t in report["templates"] if t.get("error")]}), flush=True)
        og.shutdown()


if __name__ == "__main__":
    main()
