"""Offline: every written train instance of the campaign judged against its alias's eval instance by the old (2026-10-11) and the
measured (2026-11-01) repeat rule. (Written instances passed the old rule at write time, unless written without it; the point is
the measured rule's verdicts and gaps.)"""
import collections, glob, json, os, sys
from ispatialgym.sampling import instance_separation as SEP
F = "/vision/u/minyeong/projects/isg_home/trainfast"
idx = f"{F}/eval_index_v2/per_alias"
cnt = collections.Counter()
for key_dir in sorted(glob.glob(f"{F}/out/*/*")):
    key = os.path.basename(key_dir)
    ip = os.path.join(idx, key + ".json")
    if not os.path.exists(ip):
        continue
    ev = SEP.load_instances([ip])
    for t in glob.glob(f"{key_dir}/g*/inst/*/*/instances/*-tro_state.json"):
        try:
            r = SEP.read_instance(t)
        except Exception:
            continue
        os.environ["ISPATIALGYM_FAST_REPEAT_MEASURED"] = "0"
        old = SEP.find_repeat(r, ev) is not None
        os.environ["ISPATIALGYM_FAST_REPEAT_MEASURED"] = "1"
        hit = SEP.find_repeat(r, ev)
        cnt[(key, old, hit is not None, bool(r.measured_goal))] += 1
by = collections.defaultdict(lambda: [0, 0, 0, 0])
for (key, old, new, has), n in cnt.items():
    b = by[key]; b[0] += n; b[1] += n * old; b[2] += n * new; b[3] += n * has
print(f"{'alias':58s} {'n':>5s} {'old-rule repeats':>16s} {'measured repeats':>16s} {'with measured':>13s}")
for key, (n, o, m, h) in sorted(by.items()):
    print(f"{key[:58]:58s} {n:5d} {o:16d} {m:16d} {h:13d}")
