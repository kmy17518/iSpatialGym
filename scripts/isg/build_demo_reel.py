#!/usr/bin/env python3
"""Stitch every demo of a first-demo tag into one labeled mp4 with chapters.

Each demo keeps its native 544x544 head-camera frames (with --overhead, the
overhead camera side by side); a label bar is padded on top (index, subtask,
objects, run, where it was made, arm, live step counter), so nothing in the
image is covered. One chapter per demo, plus a plain-text timestamp index.

Usage: build_demo_reel.py [--src DIR] [--out-dir DIR] [--overhead] [--preview]
  e.g. build_demo_reel.py --src runlogs/first-demo-v3.0 --overhead   -> runlogs/tmp/first-demo-v3.0_all_demos_with_overhead.mp4 + .txt
"""
import argparse
import json
import math
import re
import subprocess
from pathlib import Path

RUNLOGS = Path(__file__).resolve().parents[2] / "runlogs"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
BAR = 64
BG = "0x181818"
PREVIEW_FRAME = 45
VIEWS = {"head": ("videos", "video"), "overhead": ("videos_overhead", "video_overhead")}


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = map(int, s["avg_frame_rate"].split("/"))
    return int(s["width"]), int(s["height"]), num / den, int(s["nb_frames"])


def made(made_on):
    """('at', 'v1.0') for a demo made at a first-demo tag, ('on', '<commit>') otherwise."""
    lead = re.match(r"(\S+) \(([0-9a-f]{7,})\)", made_on or "")
    if lead:
        tag = re.fullmatch(r"first-demo-(v[\d.]+)", lead.group(1))
        return ("at", tag.group(1)) if tag else ("on", lead.group(2))
    tag = re.search(r"first-demo-(v[\d.]+)", made_on or "")
    return ("at", tag.group(1)) if tag else ("at", "?")


def arm_of(entry):
    arm = entry.get("arm")
    if isinstance(arm, dict):
        arm = arm.get("arm")
    return arm if isinstance(arm, str) else None


def load_entries(src, views):
    manifest = json.loads((src / "manifest.json").read_text())
    entries = []
    for i, e in enumerate(manifest["entries"]):
        paths = []
        for view in views:
            folder, key = VIEWS[view]
            if key not in e:
                raise SystemExit(f"manifest has no {key!r} for {e['alias']}")
            video = src / folder / f"{e['dir']}.mp4"
            if video.resolve() != (src / e[key]).resolve():
                raise SystemExit(f"{folder}/ link and manifest disagree for {e['alias']}:\n"
                                 f"  {video.resolve()}\n  {(src / e[key]).resolve()}")
            paths.append(video)
        probes = [probe(p) for p in paths]
        if len(set(probes)) != 1:
            raise SystemExit(f"{e['alias']}: views differ in size, fps or length: {probes}")
        w, h, fps, frames = probes[0]
        if e.get("n_steps") not in (None, frames):
            print(f"note: {e['alias']}: {frames} video frames vs n_steps={e['n_steps']}")
        alias, subtask = e["alias"], e["subtask"]
        if alias.startswith(subtask + "-"):
            rest = alias[len(subtask) + 1:]
        else:
            rest = "" if alias == subtask else alias
        entries.append(dict(idx=i, alias=alias, subtask=subtask, rest=rest, run=e["run_id"],
                            made=made(e.get("made_on")), arm=arm_of(e), videos=paths,
                            size=(w, h), fps=fps, frames=frames))
    if len({(e["size"], e["fps"]) for e in entries}) != 1:
        raise SystemExit("demos differ in resolution or fps; add scaling before concatenating")
    return manifest, entries


def chain(e, n, work, w, h, preview):
    tag = f"{e['idx']:02d}"

    def textfile(name, text):
        p = work / f"{name}_{tag}.txt"
        p.write_text(text)
        return p

    info = [f"run {e['run']}", "made {} {}".format(*e["made"])] + ([f"{e['arm']} arm"] if e["arm"] else [])
    title = textfile("title", f"[{e['idx'] + 1:02d}/{n}] {e['subtask']}")
    run = textfile("run", " | ".join(info))
    step = textfile("step", f"step %{{eif:n:d}}/{e['frames']}")

    k = len(e["videos"])
    inputs = [f"[{e['idx'] * k + j}:v]setpts=PTS-STARTPTS" for j in range(k)]
    if k == 1:
        f = [inputs[0]]
    else:
        f = [";".join(f"{s}[s{tag}_{j}]" for j, s in enumerate(inputs)) + ";"
             + "".join(f"[s{tag}_{j}]" for j in range(k)) + f"hstack=inputs={k}"]
    f += [f"pad={w * k}:{h + BAR}:0:{BAR}:color={BG}",
          f"drawtext=fontfile={FONT_BOLD}:textfile={title}:fontsize=14:fontcolor=white:x=8:y=6"]
    if e["rest"]:
        objects = textfile("objects", e["rest"])
        f.append(f"drawtext=fontfile={FONT}:textfile={objects}:fontsize=14:fontcolor=0xd8d8d8:x=8:y=25")
    f.append(f"drawtext=fontfile={FONT}:textfile={run}:fontsize=13:fontcolor=0x9a9a9a:x=8:y=45")
    f.append(f"drawtext=fontfile={FONT}:textfile={step}:fontsize=13:fontcolor=0x9a9a9a:x=w-tw-8:y=45")
    if preview:
        p = min(PREVIEW_FRAME, e["frames"] - 1)
        f += [f"trim=start_frame={p}:end_frame={p + 1}", "setpts=PTS-STARTPTS", f"crop={w * k}:{BAR}:0:0"]
    return ",".join(f) + f"[v{e['idx']}]"


def ffmeta_escape(s):
    return re.sub(r"([=;#\\\n])", r"\\\1", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=RUNLOGS / "first-demo-v1.2")
    ap.add_argument("--out-dir", type=Path, default=RUNLOGS / "tmp")
    ap.add_argument("--work", type=Path, default=Path("/tmp/demo_reel/work"))
    ap.add_argument("--overhead", action="store_true", help="put the overhead camera next to the head camera")
    ap.add_argument("--preview", action="store_true", help="only render the label bars to a PNG")
    args = ap.parse_args()

    views = ["head", "overhead"] if args.overhead else ["head"]
    manifest, entries = load_entries(args.src, views)
    n, k = len(entries), len(views)
    (w, h), fps = entries[0]["size"], entries[0]["fps"]
    tag = manifest.get("tag", args.src.name)
    args.work.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    chains = [chain(e, n, args.work, w, h, args.preview) for e in entries]
    tail = f"concat=n={n}:v=1:a=0" + (f",tile=1x{n}" if args.preview else "") + "[out]"
    graph = args.work / ("preview_graph.txt" if args.preview else "graph.txt")
    graph.write_text(";\n".join(chains) + ";\n" + "".join(f"[v{i}]" for i in range(n)) + tail)

    cmd = ["ffmpeg", "-hide_banner", "-y", "-loglevel", "error", "-stats"]
    for e in entries:
        for video in e["videos"]:
            cmd += ["-i", str(video)]

    if args.preview:
        png = args.work / "preview_labels.png"
        cmd += ["-filter_complex_script", str(graph), "-map", "[out]", "-frames:v", "1", str(png)]
        subprocess.run(cmd, check=True)
        print(png)
        return

    starts, t = [], 0
    for e in entries:
        starts.append(t)
        t += e["frames"]
    total_frames = t

    def ms(frames):
        return round(frames * 1000 / fps)

    commit = manifest.get("commit", "?")
    meta = [";FFMETADATA1", "title=" + ffmeta_escape(f"{tag} ({commit}): all {n} demos, {' + '.join(views)}"), ""]
    for e, s in zip(entries, starts):
        meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={ms(s)}", f"END={ms(s + e['frames'])}",
                 "title=" + ffmeta_escape(f"{e['idx'] + 1:02d} {e['alias']}"), ""]
    chapters = args.work / "chapters.txt"
    chapters.write_text("\n".join(meta))

    out = args.out_dir / f"{tag}_all_demos{'_with_overhead' if args.overhead else ''}.mp4"
    partial = out.with_name(out.stem + ".partial.mp4")
    cmd += ["-f", "ffmetadata", "-i", str(chapters),
            "-filter_complex_script", str(graph), "-map", "[out]",
            "-map_metadata", str(n * k), "-map_chapters", str(n * k),
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-g", "60",
            "-pix_fmt", "yuv420p", "-r", f"{fps:g}", "-movflags", "+faststart", str(partial)]
    subprocess.run(cmd, check=True)
    partial.replace(out)

    def clock(frames, up=False):
        t = frames * 10 / fps
        tenths = math.ceil(t - 1e-9) if up else round(t)
        return f"{tenths // 600:02d}:{tenths % 600 // 10:02d}.{tenths % 10}"

    picture = (f"{w}x{h} head camera" if k == 1
               else f"head camera (left) and overhead camera (right), {w}x{h} each,")
    width = max(len(e["alias"]) for e in entries)
    lines = [f"{tag} ({commit}): {n} demos in {out.name}, {clock(total_frames)} total",
             f"{fps:g} fps, {picture} with a {BAR} px label bar on top; "
             "'step i/N' in the bar is the 0-based frame index within that demo.",
             "One chapter per demo (VLC: Playback > Chapter, mpv: PgUp/PgDn).",
             "",
             f"{'start':<8}  {'length':>7}  {'#':>2}  {'demo':<{width}}  {'run':<11}  {'made':<8}  arm"]
    for e, s in zip(entries, starts):
        lines.append(f"{clock(s, up=True):<8}  {e['frames'] / fps:>6.1f}s  {e['idx'] + 1:>2}  "
                     f"{e['alias']:<{width}}  {e['run']:<11}  {e['made'][1]:<8}  {e['arm'] or '-'}")
    index = out.with_suffix(".txt")
    index.write_text("\n".join(lines) + "\n")
    print(out)
    print(index)


if __name__ == "__main__":
    main()
