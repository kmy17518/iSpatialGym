"""Packing test (owner item 2, 2026-11-01): successes per GPU-hour per arm and key, one GPU job = one unit (a single job, or a
job_pair.sh pair: both members' successes over the pair's wall). Also: sampler s per attempt, collector busy share, GPU utilisation
and peak memory (gpu.log), and failures (CUDA out-of-memory lines, slurm states).
    pack_ab.py [tag ...]   (default: the pk_* tags)"""
import collections, glob, json, os, re, subprocess, sys, time
from datetime import datetime

F = "/vision/u/minyeong/projects/isg_home/trainfast"
TAGS = ("pk_k2_titan", "pk_k3_titan", "pk_pair1_titan", "pk_k2_l40s", "pk_k3_l40s", "pk_pair1_l40s", "pk_pair2_l40s")
RX_END = re.compile(r"^ISGF_END \S+ wall_s=(\d+)")
RX_START = re.compile(r"^ISGF_SAMPLE .* start=(\S+)$")
RX_PEND = re.compile(r"^PAIR_END \S+ wall_s=(\d+)")
RX_PSTART = re.compile(r"^PAIR_START .* start=(\S+)$")
RX_PSUB = re.compile(r"^PAIR_SUBMITTED (\d+)")
RX_COL = re.compile(r"COLLECT_EXIT c\d+ q=\d+ rc=\d+ wall_s=(\d+)")
RX_GPU = re.compile(r"^\d+ (\d+)MiB,(\d+)%")


def sacct(jid):
    out = subprocess.run(["sacct", "-j", str(jid), "-X", "-n", "-P", "-o", "State,ElapsedRaw"], capture_output=True, text=True).stdout.split("\n")
    return out[0].split("|") if out and out[0] else None


def lines(p):
    try:
        return open(p).read().splitlines()
    except OSError:
        return []


def wall_of(job, pair_id):
    """(GPU seconds of the unit, state)."""
    if pair_id:
        L = lines(f"{job}/pair.txt")
        for l in L:
            m = RX_PEND.match(l)
            if m:
                return int(m.group(1)), "done"
        rec = sacct(pair_id)
        if rec and rec[0].split()[0] not in ("RUNNING", "PENDING"):
            return int(rec[1]), rec[0].split()[0].lower()
        st = [RX_PSTART.match(l) for l in L]
        st = [m.group(1) for m in st if m]
        return (time.time() - datetime.fromisoformat(st[0]).timestamp(), "running") if st else (0, "pending")
    L = lines(f"{job}/job.txt")
    for l in L:
        m = RX_END.match(l)
        if m:
            return int(m.group(1)), "done"
    try:
        rec = sacct(open(f"{job}/jobid").read().split()[0])
    except (OSError, IndexError):
        rec = None
    if rec and rec[0].split()[0] not in ("RUNNING", "PENDING"):
        return int(rec[1]), rec[0].split()[0].lower()
    st = [RX_START.match(l) for l in L]
    st = [m.group(1) for m in st if m]
    return (time.time() - datetime.fromisoformat(st[0]).timestamp(), "running") if st else (0, "pending")


def member(job):
    res = [json.loads(l) for l in lines(f"{job}/results.jsonl") if l.strip().startswith("{")]
    sl = lines(f"{job}/sample.log")
    written = sum(" written (attempt " in l for l in sl)
    rejected = sum(" rejected in " in l and "attempt " in l for l in sl)
    att_s = sum(float(m.group(1)) for l in sl for m in [re.search(r"(?:rejected in \w+ after|written \(attempt \d+,) ([\d.]+)s", l)] if m)
    busy = 0.0
    for p in glob.glob(f"{job}/demo/*/meta.json"):
        try:
            busy += float((json.load(open(p)).get("timings") or {}).get("total_s") or 0)
        except Exception:  # noqa: BLE001 - a demo still being written
            pass
    cwall = sum(int(m.group(1)) for l in lines(f"{job}/job.txt") for m in [RX_COL.search(l)] if m)
    oom = sum(1 for p in glob.glob(f"{job}/*.log") for l in lines(p) if "out of memory" in l.lower())
    return {"succ": sum(bool(r.get("success")) for r in res), "demos": len(res), "written": written, "attempts": written + rejected,
            "att_s": att_s, "busy": busy, "cwall": cwall, "oom": oom}


def gpu(job):
    s = [tuple(map(int, m.groups())) for m in (RX_GPU.match(l) for l in lines(f"{job}/gpu.log")) if m]
    s = [x for x in s if x[0] > 1000]
    return (sum(u for _, u in s) / len(s), max(m for m, _ in s) / 1024) if s else (None, None)


def main(tags):
    print(f"{'arm':<15} {'key':<22} units  succ  GPU-h  succ/GPU-h  per-unit rates      s/att  coll busy  GPU util  peak GB  OOM lines  states")
    for tag in tags:
        units = collections.defaultdict(list)
        for job in sorted(glob.glob(f"{F}/out/{tag}/*/g*")):
            if job.endswith(".failed_submit"):
                continue
            key = os.path.basename(os.path.dirname(job))
            pid = next((m.group(1) for l in lines(f"{job}/pair.txt") for m in [RX_PSUB.match(l)] if m), None)
            units[(key, pid or job)].append(job)
        by_key = collections.defaultdict(list)
        for (key, uid), jobs in units.items():
            pid = uid if not uid.startswith("/") else None
            w, state = wall_of(jobs[0], pid)
            ms = [member(j) for j in jobs]
            g = gpu(jobs[0])
            by_key[key].append({"w": w, "state": state, "ms": ms, "gpu": g})
        for key in sorted(by_key):
            us = by_key[key]
            succ = sum(m["succ"] for u in us for m in u["ms"])
            gh = sum(u["w"] for u in us) / 3600
            rates = [sum(m["succ"] for m in u["ms"]) / (u["w"] / 3600) if u["w"] > 0 else 0 for u in us]
            att = sum(m["attempts"] for u in us for m in u["ms"])
            att_s = sum(m["att_s"] for u in us for m in u["ms"])
            busy = sum(m["busy"] for u in us for m in u["ms"])
            cwall = sum(m["cwall"] for u in us for m in u["ms"])
            gu = [u["gpu"][0] for u in us if u["gpu"][0] is not None]
            pk = [u["gpu"][1] for u in us if u["gpu"][1] is not None]
            short = key.replace("inter_object_relation_satisfaction-", "iors-")[:22]
            print(f"{tag:<15} {short:<22} {len(us):5d} {succ:5d} {gh:6.2f} {succ / gh if gh else 0:10.1f}  "
                  f"{', '.join(f'{r:.1f}' for r in rates):<18} {att_s / att if att else 0:6.0f} {100 * busy / cwall if cwall else 0:8.0f} % "
                  f"{(sum(gu) / len(gu)) if gu else 0:7.0f} % {max(pk) if pk else 0:7.1f} {sum(m['oom'] for u in us for m in u['ms']):9d}  "
                  f"{','.join(sorted(collections.Counter(u['state'] for u in us)))}")


if __name__ == "__main__":
    main(sys.argv[1:] or TAGS)
