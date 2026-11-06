"""Like-for-like speed-up of the fast tags over train85 (production, 1 sampler + 1 rendered collector per GPU): for each tag, the
pooled successes per GPU-hour (whole job wall: ISGF_END / ISGT_END, or slurm's Elapsed for a job cancelled before it) over the
alias keys the tag ran, against train85's pooled rate on the SAME keys. Successes = results.jsonl rows with success true.
    speed_compare.py <tag> [<tag> ...]"""
import collections, glob, json, os, re, subprocess, sys

H = "/vision/u/minyeong/projects/isg_home"
F = f"{H}/trainfast"
RX_END = re.compile(r"^ISG[FT]_END \S+ wall_s=(\d+)")
RX_GPU = re.compile(r'gpu="([^"]+)"')


def elapsed(job):
    try:
        jid = open(os.path.join(job, "jobid")).read().split()[0]
    except (OSError, IndexError):
        return None
    out = subprocess.run(["sacct", "-j", jid, "-X", "-n", "-P", "-o", "State,ElapsedRaw"], capture_output=True, text=True).stdout.split("\n")
    if not out or not out[0]:
        return None
    state, el = out[0].split("|")
    return None if state.split()[0] in ("RUNNING", "PENDING") else int(el)


def job_rows(root):
    """(key, wall_s, successes, demos, gpu) per job under <root>/<key>/g*."""
    for job in glob.glob(f"{root}/*/g*"):
        key = os.path.basename(os.path.dirname(job))
        wall, gpu = None, "?"
        try:
            for line in open(f"{job}/job.txt"):
                m = RX_END.match(line)
                if m:
                    wall = int(m.group(1))
                g = RX_GPU.search(line)
                if g and gpu == "?":
                    gpu = g.group(1)
        except OSError:
            continue
        if wall is None:
            wall = elapsed(job)
        if not wall:
            continue
        succ = demos = 0
        try:
            for line in open(f"{job}/results.jsonl"):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                demos += 1
                succ += bool(r.get("success"))
        except OSError:
            pass
        yield key, wall, succ, demos, gpu


def pooled(rows):
    w = sum(r[1] for r in rows) / 3600
    s = sum(r[2] for r in rows)
    return s, w, (s / w if w else 0.0)


def main(tags):
    base = collections.defaultdict(list)
    for r in job_rows(f"{H}/train85/out"):
        base[r[0]].append(r)
    s, w, rate = pooled([r for v in base.values() for r in v])
    print(f"train85: {sum(len(v) for v in base.values())} jobs, {len(base)} keys, {s} successes / {w:.0f} GPU-h = {rate:.2f} per GPU-h")
    for tag in tags:
        rows = list(job_rows(f"{F}/out/{tag}"))
        keys = sorted({r[0] for r in rows} & set(base))
        mine = [r for r in rows if r[0] in keys]
        theirs = [r for k in keys for r in base[k]]
        s1, w1, r1 = pooled(mine)
        s0, w0, r0 = pooled(theirs)
        gpus = collections.Counter(r[4].replace("NVIDIA ", "") for r in rows)
        per = []
        for k in keys:
            a = pooled([r for r in mine if r[0] == k])
            b = pooled(base[k])
            if a[1] > 0 and b[2] > 0:
                per.append((a[2] / b[2], k))
        per.sort()
        med = per[len(per) // 2][0] if per else float("nan")
        print(f"{tag:<14} {len(rows):3d} jobs on {dict(gpus)}; {len(keys)} keys shared with train85: {s1} succ / {w1:.1f} GPU-h = {r1:5.1f} per GPU-h "
              f"vs train85 {s0} / {w0:.0f} = {r0:4.1f} on the same keys -> x{(r1 / r0) if r0 else float('nan'):.1f} pooled; per key median x{med:.1f}"
              + (f" (range x{per[0][0]:.1f} {per[0][1][-30:]} .. x{per[-1][0]:.1f} {per[-1][1][-30:]})" if per else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
