"""Which objects a demo's per-step states record, and how much each moves: object starts are found by the md5 name hashes
(omnigibson get_uuid) of the template's objects; the 3 floats after an object's hash are its root position.
    decode_states.py <demo.hdf5> <template.json> [<other demo.hdf5>]"""
import collections, json, sys
from hashlib import md5
import h5py
import numpy as np

def uuid(name):
    return float(np.float32(int(md5(name.encode()).hexdigest(), 16) % 10 ** 8))

def objects_per_step(path, names, stride=10):
    by = {uuid(n): n for n in names}
    seen = collections.defaultdict(list)
    with h5py.File(path) as f:
        g = f["data/demo_0"]
        st, ss = g["state"][:], g["state_size"][:]
    counts = []
    for i in range(0, len(ss), stride):
        s = st[i, : int(ss[i])]
        hits = [(j, by[float(v)]) for j, v in enumerate(s) if float(v) in by]
        counts.append(len(hits))
        for j, n in hits:
            seen[n].append(s[j + 1 : j + 4].astype(np.float64))
    return counts, seen

demo, tmpl = sys.argv[1], sys.argv[2]
names = list(json.load(open(tmpl))["objects_info"]["init_info"]) + ["robot_r1"]
c, seen = objects_per_step(demo, names)
print(f"{demo.split('/pv/')[-1][:90]}: objects recorded per step median {int(np.median(c))} (min {min(c)} max {max(c)})")
other = None
if len(sys.argv) > 3:
    c2, seen2 = objects_per_step(sys.argv[3], names)
    print(f"  other: objects per step median {int(np.median(c2))}")
    other = set(seen2)
rows = []
for n, ps in seen.items():
    p = np.stack(ps)
    rows.append((n, len(ps), float(np.linalg.norm(p.max(0) - p.min(0))) * 1000))
rows.sort(key=lambda r: -r[1])
print("  object                                   steps-seen  extent of root motion (mm)" + ("  [not in other]" if other is not None else ""))
for n, k, ext in rows[:40]:
    flag = "  *" if other is not None and n not in other else ""
    print(f"  {n[:40]:40s} {k:10d}  {ext:10.3f}{flag}")
