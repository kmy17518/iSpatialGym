"""Rewrite a single-env ``demo.hdf5`` as if it had been recorded in scene slot ``--slot`` of an ``--num-envs`` simulator
(a validation tool for ispatialgym.multienv, not used by production).

    python -m ispatialgym.multienv.rehost_demo --demo <demo dir> --tro <iid>-tro_state.json --out <new demo dir>
        [--num-envs 2] [--slot 1]

Every recorded state (``data/demo_0/state``) is loaded into the slot's scene and dumped again from it with the same objects
(the per-step recording filter's selection is kept by name); the scene file is the recorded one loaded into the slot's
scene and saved from it (with the recorded task metadata). Both therefore carry the slot's scene pose and slot-frame
object poses, as a recording made in that slot does; everything else in the file is copied unchanged.
``debug_full_states.pt`` (record_debug_states) is copied as it is: a playback in a one-scene simulator must reproduce the
original slot-0 states. Prints ``REHOST {json}`` with the slot's scene pose and the round-trip check (the rehosted
state loaded back into the slot vs the original loaded there).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch as th

from ispatialgym.oracle import run_isg as R  # noqa: F401  (curobo before omnigibson)

import omnigibson as og  # noqa: E402
from omnigibson.macros import gm  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--tro", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--slot", type=int, default=1)
    args = ap.parse_args(argv)
    import h5py

    for name, value in (("ENABLE_OBJECT_STATES", True), ("USE_GPU_DYNAMICS", False), ("ENABLE_TRANSITION_RULES", False)):
        if getattr(gm, name) != value:
            with gm.unlocked():
                setattr(gm, name, value)
    tro_path = os.path.abspath(args.tro)
    iid, template_dir, tmeta, activity, did = R.resolve_instance(tro_path)
    tro = R.load_tro(tro_path)
    cfg = R.build_env_config(template_dir, tmeta, tro, tro_path, activity, did, R.robot_name_from_tro(tro), R.DEFAULT_IMAGE_SIZE)
    cfg["task"] = {"type": "DummyTask"}
    cfg["env"]["num_envs"] = int(args.num_envs)
    env = og.Environment(configs=cfg)
    scene = env.scenes[args.slot]
    reg = scene.object_registry
    src = os.path.join(args.demo, "demo.hdf5")
    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, "demo.hdf5")
    shutil.copyfile(src, dst)
    for extra in ("debug_full_states.pt", "meta.json", "annotations.json"):
        if os.path.exists(os.path.join(args.demo, extra)):
            shutil.copyfile(os.path.join(args.demo, extra), os.path.join(args.out, extra))

    with h5py.File(src, "r") as f:
        states = th.as_tensor(f["data/demo_0/state"][()], dtype=th.float32)
        sizes = f["data/demo_0/state_size"][()]
        scene_file = json.loads(f["data"].attrs["scene_file"])

    def names_in(flat):
        d, _ = scene.deserialize(flat)
        return set(d["registry"]["object_registry"].keys())

    # -- scene file: the recorded one restored into the slot's scene, saved from it
    prev = reg._dump_filter
    scene.restore(scene_file)
    reg.set_dump_filter(lambda obj: True)
    new_scene_file = scene.save(as_dict=True)
    reg.set_dump_filter(prev)
    new_scene_file["metadata"] = scene_file.get("metadata", new_scene_file.get("metadata"))

    # -- per-step states
    out_states, max_rt = [], 0.0
    for i in range(states.shape[0]):
        flat = states[i, : int(sizes[i])]
        keep = names_in(flat)
        scene.load_state(flat, serialized=True)
        reg.set_dump_filter(lambda obj, keep=keep: obj.name in keep)
        try:
            new = scene.dump_state(serialized=True).detach().cpu().float().clone()
        finally:
            reg.set_dump_filter(prev)
        assert new.numel() == flat.numel(), (i, new.numel(), flat.numel())
        out_states.append(new)
        if i % 50 == 0 or i == states.shape[0] - 1:
            # round trip: the rehosted state loaded back into the slot must give the same slot-frame state again
            scene.load_state(new, serialized=True)
            reg.set_dump_filter(lambda obj, keep=keep: obj.name in keep)
            try:
                again = scene.dump_state(serialized=True).detach().cpu().float()
            finally:
                reg.set_dump_filter(prev)
            max_rt = max(max_rt, float(th.max(th.abs(again - new))))
    padded = th.zeros((len(out_states), max(int(s.numel()) for s in out_states)), dtype=th.float32)
    for i, s in enumerate(out_states):
        padded[i, : s.numel()] = s
    with h5py.File(dst, "r+") as f:
        demo = f["data/demo_0"]
        del demo["state"]
        demo.create_dataset("state", data=padded.numpy(), compression="gzip")
        f["data"].attrs["scene_file"] = json.dumps(new_scene_file, default=R.json_default)
        f["data"].attrs["isg_rehosted"] = json.dumps({"from": src, "num_envs": args.num_envs, "slot": args.slot})
    pos, ori = scene.get_position_orientation()
    rec = {"out": dst, "slot": args.slot, "num_envs": args.num_envs, "slot_scene_pos": pos.tolist(), "slot_scene_ori": ori.tolist(),
           "recorded_scene_pos": scene_file["state"].get("pos"), "n_states": len(out_states), "roundtrip_max_abs": max_rt,
           "scene_file_state_pos": [float(v) for v in new_scene_file["state"]["pos"]],
           "state0_pos_field": out_states[0][:3].tolist()}
    print("REHOST " + json.dumps(rec, default=R.json_default), flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
