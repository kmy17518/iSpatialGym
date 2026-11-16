#!/usr/bin/env python3
"""Side-by-side PNGs of an instance's eval-time observation (obs_dump.npz) and its rendered training-format episode frame:
per dumped step, rows = cameras, columns = eval | episode | |diff| x4.

    compare_frames_png.py --lerobot <eval-lerobot> --instances <eval instances root> --out <dir> <run dir>/<instance dir>...
"""
import argparse
import json
import os

import numpy as np

CAMS = {"zed_link": "observation.rgb.zed_link_camera_0", "left_realsense_link": "observation.rgb.left_realsense_link_camera_0",
        "right_realsense_link": "observation.rgb.right_realsense_link_camera_0"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot", required=True)
    ap.add_argument("--instances", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("dirs", nargs="+")
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
        ep_of[(e.get("instance") or {}).get("instance_id")] = e
    ds = LeRobotDataset(repo_id="isg-eval", root=a.lerobot)
    eidx = np.asarray(ds.hf_dataset["episode_index"])
    os.makedirs(a.out, exist_ok=True)
    for d in a.dirs:
        meta = json.load(open(os.path.join(d, "meta.json")))
        rec = by_tro[os.path.abspath(meta["tro_path"])]
        e = ep_of[rec["source"]["instance_id"]]
        first = int(np.nonzero(eidx == int(e["episode_index"]))[0][0])
        dump = np.load(os.path.join(d, "obs_dump.npz"))
        for k, step in enumerate(dump["step"].tolist()[:2]):
            item = ds[first + step]
            rows = []
            for cam, key in CAMS.items():
                ref = (item[key].numpy().transpose(1, 2, 0) * 255).round().clip(0, 255).astype(np.uint8)
                got = dump[cam][k]
                diff = np.clip(np.abs(got.astype(np.int16) - ref.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
                rows.append(np.concatenate([got, ref, diff], 1))
            Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(a.out, f"{rec['subtask']}_step{step}.png"))
        print("wrote", rec["subtask"])


if __name__ == "__main__":
    main()
