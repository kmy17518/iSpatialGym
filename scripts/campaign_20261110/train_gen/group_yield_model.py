"""Per-class train group model from eval member records (ungated demos).

For each finalized eval group, replay its member publications under the train
sampling limit L: a member counts if it was published within L of group start
and its demo eventually succeeded (finish time unbounded except the safety cap).
Sampling occupancy = time until the 3rd counted publication or L (move-on).
Outputs per class: E[members/group], P(>=1), P(>=2), mean sampling occupancy s,
mean demo tail after the last counted publication.
"""
import csv, json, statistics, sys
from collections import defaultdict


def model(members_json, proposal_csv, limits=None):
    rows = {r['class_id']: r for r in csv.DictReader(open(proposal_csv))}
    by = defaultdict(list)
    for g in json.load(open(members_json)):
        by[g['class']].append(g)
    out = {}
    for cid, groups in by.items():
        if cid not in rows or rows[cid]['status'] != 'RUN':
            continue
        L = float((limits or {}).get(cid) or rows[cid]['train_group_limit_s'])
        cap = float(rows[cid]['demo_safety_cap_s'])
        ms, occ, tail, walls = [], [], [], []
        for g in groups:
            good = [m for m in sorted(g['members'], key=lambda m: m['pub'] or 9e18)
                    if m.get('pub') and m['pub'] - g['start'] <= L and m.get('success')
                    and (m.get('fin') or 9e18) - m['pub'] <= cap]
            good = good[:3]
            ms.append(len(good))
            if len(good) == 3:
                o = good[2]['pub'] - g['start']
            else:
                o = L
            occ.append(o)
            end = max([m['fin'] - g['start'] for m in good] + [o])
            walls.append(end)
            tail.append(end - o)
        n = len(ms)
        out[cid] = {'n_groups': n, 'L': L, 'members_mean': statistics.mean(ms), 'p_ge1': sum(m >= 1 for m in ms) / n,
                    'p_ge2': sum(m >= 2 for m in ms) / n, 'occupancy_mean_s': statistics.mean(occ),
                    'tail_mean_s': statistics.mean(tail), 'group_wall_mean_s': statistics.mean(walls)}
    return out


if __name__ == '__main__':
    res = model(sys.argv[1], sys.argv[2])
    json.dump(res, open(sys.argv[3], 'w'), indent=1, sort_keys=True)
    for c, r in sorted(res.items()):
        print(f"{c:62s} n={r['n_groups']:3d} L={r['L']:6.0f} m/g={r['members_mean']:.2f} p1={r['p_ge1']:.2f} p2={r['p_ge2']:.2f} occ={r['occupancy_mean_s']:6.0f} wall={r['group_wall_mean_s']:6.0f}")
