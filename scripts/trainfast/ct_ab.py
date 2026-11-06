"""Contention test E16 (out/ct/<instance>/<arm>/p<i>/): per collector process the outcome (success, steps, first success step), the
per-step costs (meta fast.step_costs), episode and total time, CPU seconds and the share of a core it got (/usr/bin/time -v), the
PhysX threads (meta loader.physx_threads) and the peak thread count (ps.log); per arm the job wall and GPU peak memory.
    ct_ab.py [instance]   (default de03)"""
import glob, json, os, re, sys

F = "/vision/u/minyeong/projects/isg_home/trainfast"


def tv(p):
    out = {}
    try:
        for l in open(p):
            k, _, v = l.strip().partition(": ")
            out[k] = v
    except OSError:
        pass
    cpu = float(out.get("User time (seconds)", 0) or 0) + float(out.get("System time (seconds)", 0) or 0)
    return cpu, out.get("Percent of CPU this job got", "?")


def main(inst):
    print(f"{'arm':<11} {'p':<2} {'ok':<2} {'steps':>5} {'1st':>4} {'env ms/st':>9} {'qa ms/st':>8} {'episode s':>9} {'total s':>7} {'CPU s':>6} {'CPU %':>5} "
          f"{'physx':>9} {'thr':>4}  arm wall s / GPU GB")
    for arm in sorted(glob.glob(f"{F}/out/ct/{inst}/*")):
        jt = open(f"{arm}/job.txt").read() if os.path.exists(f"{arm}/job.txt") else ""
        m = re.search(r"ISGC_END \S+ wall_s=(\d+)", jt)
        gm = [int(x.group(1)) for x in (re.match(r"^\d+ (\d+)MiB", l) for l in open(f"{arm}/gpu.log")) if x] if os.path.exists(f"{arm}/gpu.log") else []
        thr = {}
        for l in open(f"{arm}/ps.log") if os.path.exists(f"{arm}/ps.log") else []:
            parts = l.split()
            if len(parts) == 3:
                thr[parts[1]] = max(thr.get(parts[1], 0), int(parts[2]))
        tmax = max(thr.values()) if thr else 0
        for p in sorted(glob.glob(f"{arm}/p*/")):
            metas = glob.glob(f"{p}demo/**/meta.json", recursive=True)
            cpu, pct = tv(f"{p}time.txt")
            if not metas:
                print(f"{os.path.basename(arm):<11} {os.path.basename(p[:-1]):<2} (no meta)")
                continue
            mt = json.load(open(metas[0]))
            sc = (mt.get("fast") or {}).get("step_costs") or {}
            n = sc.get("steps") or 1
            t = mt.get("timings") or {}
            px = (mt.get("loader") or {}).get("physx_threads") or {}
            print(f"{os.path.basename(arm):<11} {os.path.basename(p[:-1]):<2} {'y' if mt.get('success') else 'n':<2} {mt.get('n_steps', 0):5d} "
                  f"{str(mt.get('first_success_step')):>4} {1000 * sc.get('env_step_s', 0) / n:9.1f} {1000 * sc.get('qa_s', 0) / n:8.1f} "
                  f"{t.get('episode_s', 0):9.1f} {t.get('total_s', 0):7.1f} {cpu:6.0f} {pct:>5} {str(px.get('set', px.get('default'))):>9} {tmax:4d}  "
                  f"{m.group(1) if m else '-'} / {max(gm) / 1024 if gm else 0:.1f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "de03")
