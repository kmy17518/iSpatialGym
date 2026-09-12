"""Render the init state of generated instances in the BEHAVIOR replay layout.

Layout (``joylo/scripts/replay_data.py``): top row = left wrist | head | right wrist camera, bottom = an external
camera attached to the robot's ``base_link`` at (-0.4, 0, 2.0) m looking down over the robot's shoulder, three camera
widths wide and two high. One PNG per instance (``<iid>-init.png`` next to the TRO) plus one contact sheet per problem
(``<out>/renders/<problem>.png``, first instance of every template).

Loading mirrors ``validate_instances.py``. Objects the loader cannot restore yet (assisted grasps, hinge constraints)
are posed from the TRO with gravity disabled so the picture shows the sampled init state.

Example::

    OMNIGIBSON_HEADLESS=1 python -m ispatialgym.sampling.render_instances --out /tmp/isg_full11
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import traceback

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import numpy as np  # noqa: E402
import torch as th  # noqa: E402
from PIL import Image  # noqa: E402

log = logging.getLogger("ispatialgym.render")

CAM_RES = 640  # side of each robot camera tile in the sheet (``--res``); 384 with 4 iterations looked blurred
RENDER_ITERS = 10  # frames rendered after a teleport before the sensors are read (``--render-iters``)


def _to_uint8(img: th.Tensor) -> np.ndarray:
    arr = img.cpu().numpy()
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return arr[..., :3]


def _tile(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(arr).resize((w, h), Image.BILINEAR))


def compose(left, head, right, external, res: int = CAM_RES) -> np.ndarray:
    top = np.concatenate([_tile(left, res, res), _tile(head, res, res), _tile(right, res, res)], axis=1)
    bottom = _tile(external, 3 * res, 2 * res)
    return np.concatenate([top, bottom], axis=0)


def render_template(template_dir: str, robot_name: str = "robot_r1", res: int = None, render_iters: int = None):
    res = CAM_RES if res is None else int(res)
    render_iters = RENDER_ITERS if render_iters is None else int(render_iters)
    import omnigibson as og
    from omnigibson.macros import gm

    for name, value in (("ENABLE_OBJECT_STATES", True), ("USE_GPU_DYNAMICS", False), ("ENABLE_TRANSITION_RULES", False)):
        if getattr(gm, name) != value:
            with gm.unlocked():
                setattr(gm, name, value)
    from ispatialgym.eval.group_entity import GroupEntity
    from ispatialgym.eval.tro import load_tro
    from ispatialgym.oracle import r1pro_model as M
    from ispatialgym.sampling.sim_utils import PARK_POSITION
    from ispatialgym.tasks import ISpatialGymTask  # noqa: F401

    with open(os.path.join(template_dir, "template_meta.json")) as f:
        meta = json.load(f)
    problem = meta["problem"]
    activity, did = problem.rsplit("-", 1)
    tros = sorted(glob.glob(os.path.join(template_dir, "instances", "*-tro_state.json")))
    if not tros:
        return []
    # BEHAVIOR replay external camera (joylo/scripts/replay_data.py): over the robot's shoulder, looking down-forward
    external_sensors = [
        {
            "sensor_type": "VisionSensor",
            "name": "external_sensor0",
            "relative_prim_path": f"/controllable__r1pro__{robot_name}/base_link/external_sensor0",
            "modalities": ["rgb"],
            "sensor_kwargs": {"image_height": res * 2, "image_width": res * 3, "horizontal_aperture": 40.0},
            "position": th.tensor([-0.4, 0.0, 2.0], dtype=th.float32),
            "orientation": th.tensor([0.2706, -0.2706, -0.6533, 0.6533], dtype=th.float32),
            "pose_frame": "parent",
        }
    ]
    cfg = {
        "env": {"action_frequency": 30, "physics_frequency": 120, "external_sensors": external_sensors},
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": meta["scene"],
            "scene_file": os.path.join(template_dir, "template.json"),
            "load_room_instances": meta.get("load_room_instances") or None,
        },
        "robots": [
            M.joylo_robot_config(
                robot_name,
                obs_modalities=["rgb"],
                sensor_config={"VisionSensor": {"sensor_kwargs": {"image_height": res, "image_width": res}}},
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
    env = og.Environment(configs=cfg)
    task = env.task
    robot = env.robots[0]
    head_key = [k for k in robot.sensors if k.endswith(M.HEAD_CAMERA_SENSOR_SUFFIX)][0]
    left_key = [k for k in robot.sensors if "left" in k][0]
    right_key = [k for k in robot.sensors if "right" in k][0]
    for k in (head_key, left_key, right_key):
        robot.sensors[k].horizontal_aperture = 40.0
    external = env.external_sensors["external_sensor0"]
    written = []
    for path in tros:
        iid = os.path.basename(path).replace("-tro_state.json", "")
        try:
            tro = load_tro(path)
            task.load_tro_instance(env, tro=tro, settle_steps=10)
            scope = task.object_scope
            # objects the loader cannot restore yet: pose them from the TRO with gravity off so the picture is the init
            extras = tro.meta.extras
            frozen = {c["child"] for c in (extras.get("constraints") or [])} | {g["object"] for g in (extras.get("grasps") or [])}
            for inst in frozen:
                ent = scope.get(inst)
                st = tro.object_states.get(inst)
                if ent is None or isinstance(ent, GroupEntity) or st is None:
                    continue
                root = st["root_link"]
                ent.set_position_orientation(th.tensor(root["pos"], dtype=th.float32), th.tensor(root["ori"], dtype=th.float32))
                ent.disable_gravity()
                ent.keep_still()
            robot.keep_still()
            og.sim.step_physics()
            for _ in range(render_iters):
                og.sim.render()
            obs, _ = robot.get_obs()
            ext_obs, _ = external.get_obs()
            sheet = compose(_to_uint8(obs[left_key]["rgb"]), _to_uint8(obs[head_key]["rgb"]), _to_uint8(obs[right_key]["rgb"]), _to_uint8(ext_obs["rgb"]), res)
            out = path.replace("-tro_state.json", "-init.png")
            Image.fromarray(sheet).save(out)
            written.append(out)
            log.info("%s -> %s", iid, out)
            env.reset()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            log.error("%s failed: %r", iid, e)
    og.clear()
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--problem", action="append", default=None)
    parser.add_argument("--res", type=int, default=CAM_RES, help="side of each robot camera tile")
    parser.add_argument("--render-iters", type=int, default=RENDER_ITERS)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="[%(levelname)s %(name)s] %(message)s", stream=sys.stdout, force=True)
    import omnigibson as og

    sheets = {}
    try:
        for problem_dir in sorted(glob.glob(os.path.join(args.out, "*"))):
            pname = os.path.basename(problem_dir)
            if not os.path.isdir(problem_dir) or pname == "renders" or (args.problem and pname not in args.problem):
                continue
            for tdir in sorted(glob.glob(os.path.join(problem_dir, "*"))):
                if os.path.exists(os.path.join(tdir, "template_meta.json")):
                    try:
                        written = render_template(tdir, res=args.res, render_iters=args.render_iters)
                    except Exception as e:  # noqa: BLE001
                        traceback.print_exc()
                        log.error("template %s failed: %r", tdir, e)
                        try:
                            og.clear()
                        except Exception:
                            pass
                        continue
                    if written:
                        sheets.setdefault(pname, []).append(written[0])
        # contact sheet per problem: first instance of every template, stacked vertically
        os.makedirs(os.path.join(args.out, "renders"), exist_ok=True)
        for pname, paths in sheets.items():
            imgs = [np.asarray(Image.open(p)) for p in paths]
            Image.fromarray(np.concatenate(imgs, axis=0)).save(os.path.join(args.out, "renders", f"{pname}.png"))
        print("RENDER_REPORT " + json.dumps({k: len(v) for k, v in sheets.items()}), flush=True)
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
