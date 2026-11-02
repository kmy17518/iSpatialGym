"""Before / after table of the vec_next build change from the lambda runs: world build seconds, batch and env steps per second, loop stage
shares (reads <run>.log and <run>/_batches/batch_0000.json).  python table.py <label>=<run dir> ..."""
import glob, json, os, re, sys

rows = []
for arg in sys.argv[1:]:
    label, d = arg.split("=", 1)
    log = d + ".log"
    build = steps = env = None
    if os.path.exists(log):
        for line in open(log, errors="replace"):
            m = re.search(r"world built in ([0-9.]+)s", line)
            if m:
                build = float(m.group(1))
            m = re.search(r"batch_steps/s=([0-9.]+) env_steps/s=([0-9.]+)", line)
            if m:
                steps, env = float(m.group(1)), float(m.group(2))
    b = sorted(glob.glob(os.path.join(d, "_batches", "batch_*.json")))
    loop = {}
    if b:
        j = json.load(open(b[0]))
        loop = j.get("loop") or j.get("timings", {}).get("loop") or {}
    rows.append((label, build, steps, env, loop))
print("| run | world build s | batch steps/s | env steps/s | env_step_s | record_s | monitors_s | post_step_s |")
print("|---|---|---|---|---|---|---|---|")
for label, build, steps, env, loop in rows:
    f = lambda k: ("%.2f" % loop[k]) if k in loop else "-"
    print(f"| {label} | {build if build is not None else '-'} | {steps if steps is not None else '-'} | {env if env is not None else '-'} | {f('env_step_s')} | {f('record_s')} | {f('monitors_s')} | {f('post_step_s')} |")
