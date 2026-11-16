#!/usr/bin/env python3
"""Input-consistency audit: the observations a policy receives at eval time (``GTPolicy(dump_steps=...)`` ->
``<run out>/<instance>/obs_dump.npz``) against the rendered training-format episode of the same instance (the eval LeRobot
merge: the same demo, the same batched-replay world and cameras), and the eval goal PNG against the episode's decoded last
head frame (what the goal-conditioned runs trained on, ``GOAL_SOURCE=episode_last``).

    compare_obs.py --lerobot <eval-lerobot root> --instances <ispatialgym-eval-instances root> <run out dir>... [--json out.json]

Per dumped step and camera: mean |diff| and PSNR of the uint8 images; per proprio group (BEHAVIOR's PROPRIOCEPTION_INDICES)
the max |diff|; the 23-D training state (indices of the three training code bases) max |diff|.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

CAMS = {"zed_link": "observation.rgb.zed_link_camera_0", "left_realsense_link": "observation.rgb.left_realsense_link_camera_0",
        "right_realsense_link": "observation.rgb.right_realsense_link_camera_0"}
PROPRIO = {"base_qvel": (0, 3), "arm_left_qpos": (3, 10), "arm_left_qvel": (10, 17), "eef_left_pos": (17, 20), "eef_left_quat": (20, 24),
           "gripper_left_qpos": (24, 26), "gripper_left_qvel": (26, 28), "arm_right_qpos": (28, 35), "arm_right_qvel": (35, 42),
           "eef_right_pos": (42, 45), "eef_right_quat": (45, 49), "gripper_right_qpos": (49, 51), "gripper_right_qvel": (51, 53),
           "trunk_qpos": (53, 57), "trunk_qvel": (57, 61)}


def train_state(s: np.ndarray) -> np.ndarray:
    """The 23-D state of pi / ACT / DP: base vel, torso, left arm, left gripper (q1+q2), right arm, right gripper."""
    s = np.asarray(s, dtype=np.float64)
    return np.concatenate([s[..., 0:3], s[..., 53:57], s[..., 3:10], s[..., 24:25] + s[..., 25:26], s[..., 28:35], s[..., 49:50] + s[..., 50:51]], -1)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot", required=True)
    ap.add_argument("--instances", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    by_tro = {}
    for line in open(os.path.join(a.instances, "metadata/instances.jsonl")):
        r = json.loads(line)
        by_tro[os.path.abspath(os.path.join(a.instances, r["files"]["tro_state"]))] = r
    ep_of = {}
    for line in open(os.path.join(a.lerobot, "isg_meta/episodes.jsonl")):
        e = json.loads(line)
        inst = e.get("instance") or {}
        ep_of[inst.get("instance_id") or inst.get("demo_name")] = e
    ds = LeRobotDataset(repo_id="isg-eval", root=a.lerobot)
    starts = {}
    eidx = np.asarray(ds.hf_dataset["episode_index"])
    for ep in np.unique(eidx):
        w = np.nonzero(eidx == ep)[0]
        starts[int(ep)] = (int(w[0]), int(w[-1]))
    report = []
    dumps = []
    for run in a.runs:
        dumps += sorted(glob.glob(os.path.join(run, "*", "obs_dump.npz")))
    for path in dumps:
        d = os.path.dirname(path)
        meta = json.load(open(os.path.join(d, "meta.json")))
        rec = by_tro.get(os.path.abspath(meta["tro_path"]))
        if rec is None:
            print(f"{d}: tro not in the eval dataset", file=sys.stderr)
            continue
        e = ep_of[rec["source"]["instance_id"]]
        first, last = starts[int(e["episode_index"])]
        dump = np.load(path)
        row = {"instance": rec["source"]["instance_id"], "subtask": rec["subtask"], "episode_index": e["episode_index"], "steps": {}}
        for k, step in enumerate(dump["step"].tolist()):
            item = ds[first + step]
            st = {}
            for cam, key in CAMS.items():
                ref = (item[key].numpy().transpose(1, 2, 0) * 255.0).round().clip(0, 255).astype(np.uint8)
                got = dump[cam][k]
                st[cam] = {"shape": list(got.shape), "mean_abs": float(np.abs(got.astype(np.int16) - ref.astype(np.int16)).mean()), "psnr": psnr(got, ref)}
            sref = item["observation.state"].numpy()
            sgot = dump["proprio"][k]
            if sgot.shape != sref.shape:  # a run before the policy-mode 61-D proprio fix: images only
                st["proprio_max_abs"], st["train_state_max_abs"], st["train_state_absdiff"] = {}, float("nan"), None
                st["proprio_dims"] = [int(sgot.shape[0]), int(sref.shape[0])]
            else:
                st["proprio_max_abs"] = {g: float(np.abs(sgot[i:j] - sref[i:j]).max()) for g, (i, j) in PROPRIO.items()}
                st["train_state_max_abs"] = float(np.abs(train_state(sgot) - train_state(sref)).max())
                st["train_state_absdiff"] = np.round(np.abs(train_state(sgot) - train_state(sref)), 5).tolist()
            row["steps"][int(step)] = st
        # goal: the released PNG vs the episode's decoded last head frame
        goal_png = np.asarray(Image.open(os.path.join(a.instances, rec["files"]["goal_rgb"])).convert("RGB"))
        lastf = (ds[last][CAMS["zed_link"]].numpy().transpose(1, 2, 0) * 255.0).round().clip(0, 255).astype(np.uint8)
        row["goal_png_vs_decoded_last"] = {"mean_abs": float(np.abs(goal_png.astype(np.int16) - lastf.astype(np.int16)).mean()), "psnr": psnr(goal_png, lastf)}
        report.append(row)
        s0 = row["steps"].get(0) or next(iter(row["steps"].values()))
        print(f"{row['subtask'][:60]:60s} step0 psnr head {s0['zed_link']['psnr']:.1f} L {s0['left_realsense_link']['psnr']:.1f} "
              f"R {s0['right_realsense_link']['psnr']:.1f}  state23 max {s0['train_state_max_abs']:.2e}  goal png vs decoded {row['goal_png_vs_decoded_last']['psnr']:.1f} dB")
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
