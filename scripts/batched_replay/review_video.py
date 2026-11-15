#!/usr/bin/env python3
"""Annotated tiled review video of batched-replay episodes, concatenated into one mp4.

    review_video.py --out review.mp4 [--stride 1] <episode dir> [<episode dir> ...]

An episode dir holds episodes_isg.jsonl (the replayer's records) and the LeRobot dataset it names (<dir>/<dataset>, or <dir>/<unit>
for a Vast unit whose dataset was renamed). Every kept episode becomes one segment: a header with its family / class / instance,
crop and scene mode, the perturbation (TRAIN in orange: lighting draw and removed object; EVAL in green: none), then the frames tiled
[head | left wrist | right wrist | goal image png], each frame annotated with its index, the action / settle split and the padded
rows (frames past the recorded end that repeat the last recorded row). A 1.5 s title card opens each segment.
"""
from __future__ import annotations

import argparse
import json
import os
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CAMS = (("observation.rgb.zed_link_camera_0", "head"), ("observation.rgb.left_realsense_link_camera_0", "left wrist"),
        ("observation.rgb.right_realsense_link_camera_0", "right wrist"))
S = 224
HEADER = 78


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_HEAD, F_SMALL, F_TITLE = font(13), font(11), font(20)


def decode(path, t0, t1, fps):
    out = []
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        for fr in c.decode(st):
            t = float(fr.pts * st.time_base)
            if t < t0 - 0.5 / fps:
                continue
            if t >= t1 - 0.5 / fps:
                break
            out.append(fr.to_ndarray(format="rgb24"))
    return out


def episode_frames(ds: Path, idx: int):
    import pyarrow.parquet as pq

    info = json.load(open(ds / "meta/info.json"))
    rows = [r for p in sorted((ds / "meta/episodes").glob("*/*.parquet")) for r in pq.read_table(p).to_pylist()]
    r = next(x for x in rows if x["episode_index"] == idx)
    frames = {}
    for key, _ in CAMS:
        p = ds / info["video_path"].format(video_key=key, chunk_index=r[f"videos/{key}/chunk_index"], file_index=r[f"videos/{key}/file_index"])
        frames[key] = decode(p, r[f"videos/{key}/from_timestamp"], r[f"videos/{key}/to_timestamp"], info["fps"])
    return frames, int(r["length"]), info["fps"]


def describe(e):
    inst = e.get("instance") or {}
    p = e.get("perturbation") or {}
    cls = inst.get("class_id") or e.get("task")
    fam = inst.get("task_family") or cls.split("-")[0]
    train = p.get("scene") == "remove" or p.get("lighting") == "replay"
    lines = [f"{fam} | {cls} | {inst.get('eval_unit_id') or inst.get('demo_name', '')[:40]}",
             f"recorded world: {'CROPPED (' + inst['scene_crop'] + ')' if inst.get('scene_crop') else 'full (uncropped)'} -> replay --scene "
             f"{e.get('scene_mode')} (task rooms + adjacent) | {e.get('image_size')} px | length {e['length']} (recorded {e.get('length_recorded')})"]
    if train:
        fd = p.get("fixtures_draw") or {}
        sky = p.get("skybox") or {}
        light = (f"lighting x{fd.get('intensity_factor', 1):.2f} tint {[round(c, 2) for c in fd.get('color_factors', [1, 1, 1])]}"
                 if fd.get("perturbed") else "lighting: default (p draw: no)")
        light += f", skybox x{sky.get('intensity_factor', 1):.2f}" if sky.get("perturbed") else ""
        if p.get("camera_collection_removals"):
            scene = "scene: no replay-time removal (camera relocalization: collection-time removal kept)"
        else:
            scene = f"scene: removed {', '.join(p.get('removed') or []) or 'nothing (p draw: no)'}"
        lines.append(f"TRAIN perturbation: {light} | {scene}")
    else:
        lines.append("EVAL: no replay-time perturbation (default lighting, full merged scene)")
    return lines, train


def card(lines, train, w, h):
    im = Image.new("RGB", (w, h), (20, 20, 20))
    d = ImageDraw.Draw(im)
    y = 30
    for i, ln in enumerate(lines):
        d.text((20, y), ln, fill=((255, 170, 60) if train else (120, 230, 120)) if i == len(lines) - 1 else (235, 235, 235),
               font=F_TITLE if i == 0 else F_HEAD)
        y += 34 if i == 0 else 24
    return np.asarray(im)


def compose(frames, goal, i, e, lines, train):
    w = S * 4
    im = Image.new("RGB", (w, HEADER + S + 16), (0, 0, 0))
    d = ImageDraw.Draw(im)
    color = (255, 170, 60) if train else (120, 230, 120)
    d.rectangle([0, 0, w, HEADER], fill=(30, 30, 30))
    d.text((6, 4), lines[0], fill=(240, 240, 240), font=F_HEAD)
    d.text((6, 22), lines[1], fill=(200, 200, 200), font=F_SMALL)
    d.text((6, 38), lines[2], fill=color, font=F_SMALL)
    st = e.get("settle") or {}
    a, rec = st.get("action_steps"), e.get("length_recorded")
    phase = "ACTION" if a is None or i < a else ("SETTLE (padded: repeats last recorded row)" if rec and i >= rec else "SETTLE (recorded)")
    d.text((6, 56), f"frame {i + 1}/{e['length']} | action steps {a} + settle {st.get('settle_steps')} (recorded {st.get('settle_steps_recorded')}, "
                    f"padded {st.get('padded_rows')}) | {phase}", fill=(255, 255, 120) if a is not None and i >= a else (230, 230, 230), font=F_SMALL)
    for k, (key, label) in enumerate(CAMS):
        fr = frames[key][min(i, len(frames[key]) - 1)]
        im.paste(Image.fromarray(fr), (k * S, HEADER))
        d.text((k * S + 4, HEADER + S + 1), label, fill=(220, 220, 220), font=F_SMALL)
    im.paste(goal, (3 * S, HEADER))
    d.text((3 * S + 4, HEADER + S + 1), "goal image (saved png = last frame)", fill=(220, 220, 220), font=F_SMALL)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()
    eps = []
    for d in map(Path, a.dirs):
        for line in (d / "episodes_isg.jsonl").read_text().splitlines():
            e = json.loads(line)
            if e.get("kept"):
                ds = d / e["dataset"]
                if not ds.exists():
                    ds = d / e["dataset"].rsplit("_slot", 1)[0]
                eps.append((d, ds, e))
    out = av.open(a.out, "w")
    vs = out.add_stream("libx264", rate=a.fps)
    vs.width, vs.height, vs.pix_fmt = S * 4, HEADER + S + 16, "yuv420p"
    vs.options = {"crf": "23", "preset": "veryfast"}
    index = []
    t = 0.0
    for n, (d, ds, e) in enumerate(eps):
        lines, train = describe(e)
        lines[0] = f"[{n + 1}/{len(eps)}] " + lines[0]
        frames, L, fps = episode_frames(ds, e["episode_index"])
        goal = Image.open(d / e["goal_image"]["rgb_path"]).convert("RGB").resize((S, S))
        title = card(lines + [f"goal image: {e['goal_image']['rgb_path']} (sha {e['goal_image']['rgb_sha256'][:12]})"], train, vs.width, vs.height)
        index.append({"segment": n + 1, "start_s": round(t, 1), "class": (e.get("instance") or {}).get("class_id") or e.get("task"),
                      "train_perturbed": train, "lines": lines})
        for _ in range(int(1.5 * a.fps)):
            for p in vs.encode(av.VideoFrame.from_ndarray(title, format="rgb24")):
                out.mux(p)
        n_fr = 0
        for i in range(0, L, a.stride):
            img = compose(frames, goal, i, e, lines, train)
            for p in vs.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(p)
            n_fr += 1
        t += 1.5 + n_fr / a.fps
        print(f"segment {n + 1}: {lines[0]} ({L} frames, decoded {[len(v) for v in frames.values()]})", flush=True)
    for p in vs.encode():
        out.mux(p)
    out.close()
    Path(a.out).with_suffix(".index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {a.out} ({t / 60:.1f} min, {len(eps)} episodes); index {Path(a.out).with_suffix('.index.json')}")


if __name__ == "__main__":
    main()
