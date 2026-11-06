#!/usr/bin/env python3
"""trainfast refill (2026-11-01): keep the GPU caps busy with the fixed-code production configuration (crop default, BAND_EVIDENCE, measured
repeat rule, shared exact-IK cache, 12 CPUs, k=2, n=24 per job, tags pilot2 / pilot3 / pilot4). Each pass: per alias of those tags, the
instances written so far (plus 24 per job still queued or running); while below TARGET and a slot is free under the caps (svl 48 TITAN
GPUs, viscam 30 L40S), submit the next job as tag pilot4. Stop file: $F/refill.stop.
    refill.py [--target N] [--dry-run]"""
import argparse, glob, os, subprocess, sys
F = "/vision/u/minyeong/projects/isg_home/trainfast"
W = "/vision/u/minyeong/projects/ispatialgym-train-fast"
TAGS = ("pilot2", "pilot3", "pilot4")
SKIP = {"configuration_matching-block_design_01", "inter_object_relation_satisfaction-inside_large_scale_03"}
CAPS = {"svl": 48, "viscam": 30}

def my_jobs():
    out = subprocess.run(["squeue", "-u", "minyeong", "-h", "-o", "%i|%j|%P|%T"], capture_output=True, text=True).stdout
    return [l.split("|") for l in out.splitlines() if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=int(os.environ.get("REFILL_TARGET", "96")))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(os.path.join(F, "refill.stop")):
        print("REFILL stopped (refill.stop present)"); return 0
    jobs = my_jobs()
    gpus = {p: sum(1 for j in jobs if j[2] == p and j[1].startswith("isg")) for p in CAPS}
    live = {}
    for tag in TAGS:
        for d in glob.glob(f"{F}/out/{tag}/*/g*"):
            if d.endswith((".cancelled", ".failed_submit")):
                continue
            jt = open(os.path.join(d, "job.txt")).read() if os.path.exists(os.path.join(d, "job.txt")) else ""
            key = os.path.basename(os.path.dirname(d))
            w = 0
            if os.path.exists(os.path.join(d, "sample.log")):
                w = sum(1 for l in open(os.path.join(d, "sample.log"), errors="replace") if " written (attempt " in l)
            ended = "ISGF_END" in jt
            live.setdefault(key, [0, 0])
            live[key][0] += w if ended else max(w, 24)  # a queued / running job counts its 24
            live[key][1] += 0 if ended else 1
    want = sorted((v[0], k) for k, v in live.items() if k not in SKIP and v[0] < a.target)
    sub = 0
    for written, key in want:
        part = "svl" if gpus["svl"] < CAPS["svl"] else ("viscam" if gpus["viscam"] < CAPS["viscam"] else None)
        if part is None:
            break
        cmd = ["python3", f"{F}/scripts/submit_fast.py", "submit", key, "24", "--flow", "fast", "--code", W, "--kcollect", "2", "--cpus", "12",
               "--mem", "32G", "--sample-limit-s", "16000", "--time-s", "21600", "--tag", "pilot4"]
        if part == "viscam":
            cmd += ["--part", "viscam", "--acct", "viscam", "--gres", "l40s"]
        if a.dry_run:
            print("DRY", part, key, written); gpus[part] += 1; sub += 1; continue
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=F)
        print(f"REFILL {part} {key} (had {written}): {(r.stdout or r.stderr).strip().splitlines()[-1][:120]}")
        gpus[part] += 1
        sub += 1
    print(f"REFILL pass: target {a.target}, {sub} submitted, GPUs svl {gpus['svl']}/{CAPS['svl']} viscam {gpus['viscam']}/{CAPS['viscam']}, "
          f"{len(want)} aliases below target")
    return 0

if __name__ == "__main__":
    sys.exit(main())
