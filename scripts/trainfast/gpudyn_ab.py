"""F1 CPU vs GPU PhysX collector A/B (out/gpudyn/<instance>/<cpu|gpu>_r<k>/): per run the outcome (success, status / reason, steps,
first success step, QA reason), the per-step costs (meta fast.step_costs), episode and total time, the process's CPU seconds and
peak RSS (/usr/bin/time -v) and the GPU memory / utilisation (gpu.log, 10 s samples).
    gpudyn_ab.py"""
import glob, json, os, re, statistics as st

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
    return cpu, int(out.get("Maximum resident set size (kbytes)", 0) or 0) / 1e6


def gpu(p):
    s = []
    try:
        for l in open(p):
            m = re.match(r"^\d+ (\d+)MiB,(\d+)%", l)
            if m:
                s.append((int(m.group(1)), int(m.group(2))))
    except OSError:
        pass
    s = [x for x in s if x[0] > 500]
    return (st.mean(u for _, u in s), max(m for m, _ in s) / 1024) if s else (0, 0)


def main():
    print(f"{'instance':<9} {'run':<7} {'ok':<3} {'status':<10} {'steps':>5} {'1st succ':>8} {'env ms/st':>9} {'qa ms/st':>8} {'episode s':>9} "
          f"{'total s':>7} {'CPU s':>6} {'RSS GB':>6} {'GPU %':>5} {'GPU GB':>6}  reason")
    for inst in sorted(glob.glob(f"{F}/out/gpudyn/*")):
        for run in sorted(glob.glob(f"{inst}/*_r*")):
            metas = glob.glob(f"{run}/demo/**/meta.json", recursive=True)
            cpu, rss = tv(f"{run}/time.txt")
            gu, gm = gpu(f"{run}/gpu.log")
            end = [l for l in open(f"{run}/job.txt")] if os.path.exists(f"{run}/job.txt") else []
            if not metas:
                print(f"{os.path.basename(inst):<9} {os.path.basename(run):<7} -   (no meta) {''.join(l.strip() for l in end if 'ISGD_END' in l)}")
                continue
            m = json.load(open(metas[0]))
            sc = (m.get("fast") or {}).get("step_costs") or {}
            n = sc.get("steps") or 1
            t = m.get("timings") or {}
            print(f"{os.path.basename(inst):<9} {os.path.basename(run):<7} {'y' if m.get('success') else 'n':<3} {str(m.get('status'))[:10]:<10} "
                  f"{m.get('n_steps', 0):5d} {str(m.get('first_success_step')):>8} {1000 * sc.get('env_step_s', 0) / n:9.1f} {1000 * sc.get('qa_s', 0) / n:8.1f} "
                  f"{t.get('episode_s', 0):9.1f} {t.get('total_s', 0):7.1f} {cpu:6.0f} {rss:6.1f} {gu:5.0f} {gm:6.1f}  "
                  f"{str(m.get('reason') or m.get('qa_reason') or '')[:60]} gpu_dyn={(m.get('fast') or {}).get('flags', {}).get('ISPATIALGYM_FAST_COLLECTOR_GPU_DYNAMICS')}")


if __name__ == "__main__":
    main()
