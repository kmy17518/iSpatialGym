"""isg-1020 demos -> the robot-relative grasp situation of each episode (stdlib only; runs on the cluster login node):

    python3 scripts/grasp_bands/extract_demo_stances.py --dataset $TRAIN --out demo_stances.json [--match kielbasa teacup ...]

Per episode of the matched aliases: the manipulation stance of the oracle record (base pose2d, trunk translate / tilt, stand-off, edge),
the grasp (key, name), the target object's initial pose (its TRO state) and that pose in the stance's base frame (cell xy, object yaw
relative to the base). Used to compare reach maps with what the demos actually did (``reach_maps`` cells are robot-relative)."""

import argparse
import csv
import glob
import json
import math
import os


def yaw_of(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--match", nargs="*", default=[])
    a = ap.parse_args()
    rows = list(csv.DictReader(open(os.path.join(a.dataset, "meta", "episode_aliases.csv"))))
    out = []
    for r in rows:
        if a.match and not any(m in r["alias"] for m in a.match):
            continue
        src = r["source_export"]
        job = src.split("/demo/")[0]
        inst = r["instance_id"]
        cand = glob.glob(os.path.join(job, "inst", "*", "*", "instances", inst + ".oracle.json"))
        tro = glob.glob(os.path.join(job, "inst", "*", "*", "instances", inst + "-tro_state.json"))
        if not cand or not tro:
            out.append({"episode": r["episode_index"], "alias": r["alias"], "instance": inst, "error": "no oracle/tro"})
            continue
        o = json.load(open(cand[0]))
        t = json.load(open(tro[0]))
        st = (o.get("stances") or {}).get("manipulation") or {}
        g = (o.get("grasps") or [{}])[0]
        target = [k for k in t if k not in ("robot_poses", "ispatialgym") and isinstance(t[k], dict) and "root_link" in t[k]]
        # the grasped object: the binding whose synset matches the grasp key's category
        cat = (g.get("grasp_key") or "").split("__")[0]
        pick = [k for k in target if k.split(".n.")[0].replace("__", "_") == cat] or target[:1]
        rl = t[pick[0]]["root_link"]
        bx, by, byaw = st.get("base_pose2d", [0, 0, 0])
        dx, dy = rl["pos"][0] - bx, rl["pos"][1] - by
        c, s = math.cos(-byaw), math.sin(-byaw)
        out.append({"episode": int(r["episode_index"]), "alias": r["alias"], "instance": inst, "grasp_key": g.get("grasp_key"), "grasp_name": g.get("grasp_name"),
                    "stance": {k: st.get(k) for k in ("base_pose2d", "translate", "tilt", "stand_off", "edge")},
                    "object": {"binding": pick[0], "pos": rl["pos"], "ori": rl["ori"]},
                    "cell_base": [c * dx - s * dy, s * dx + c * dy], "z": rl["pos"][2], "yaw_rel": math.atan2(math.sin(yaw_of(rl["ori"]) - byaw), math.cos(yaw_of(rl["ori"]) - byaw))})
    json.dump(out, open(a.out, "w"), indent=1)
    print(len(out), "episodes ->", a.out)


if __name__ == "__main__":
    main()
