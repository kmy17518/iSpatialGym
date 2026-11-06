"""Reach maps vs the isg-1020 demos: is the demonstrated grasp situation inside the map's feasible region?

    python scripts/grasp_bands/compare_demos.py --stances demo_stances.json --maps MAPS --glib GLIB --jobs isg1020.json --heights isg1020_heights.json

For each demo (``extract_demo_stances.py``): the map of its target model at its alias's support height, the stand-off of its
manipulation stance (edge distance, overhang 0), the grid trunk with the same translate and the nearest tilt (as ``ReachBands.band``),
the map cell nearest to the demonstrated object root in the stance's base frame and the yaw step nearest to its yaw about the rest.
``exact``: some grasp of the v2 library is feasible there; ``near``: within one cell / one yaw step; the status counts say why not.
The demos ran the LEGACY grasps (other rows, the 8 mm finger allowance, the whole-support ignore) - agreement measures how much of the
demonstrated workspace the stricter v2 policy keeps, not the recall of a grasp row."""

import argparse
import collections
import json
import math
import os

import numpy as np

from ispatialgym.grasping import schema as S
from ispatialgym.sampling import reach_maps as RM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stances", required=True)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--glib", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--heights", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    demos = json.load(open(a.stances))
    jobs = {j["legacy_key"]: j for j in json.load(open(a.jobs))}
    heights = json.load(open(a.heights))
    maps = {}
    rows = collections.defaultdict(lambda: collections.Counter())
    detail = []
    for d in demos:
        j = jobs.get(d.get("grasp_key"))
        if j is None:
            rows[d["alias"]]["no_job"] += 1
            continue
        key = j["id"]
        if key not in maps:
            lib = S.load_model(j["category"], j["model"], j.get("scale"), a.glib, constraint=j.get("constraint") or "tabletop", rest_hint_wxyz=j.get("rest_hint_wxyz"))
            rm = None
            if lib is not None and lib.rests and lib.grasps():
                rid = sorted(lib.rests, key=lambda r: (not lib.rests[r].stable, r))[0]
                h0 = float(heights[key][0])
                rm = RM.ReachMap.find(os.path.join(a.maps, key), lib, h0, rest_id=rid)
            maps[key] = (rm, lib)
        rm, lib = maps[key]
        if rm is None:
            rows[d["alias"]]["no_map"] += 1
            continue
        st = d["stance"]
        so = float(st["stand_off"])
        d_e = min(rm.meta["edge_distances"], key=lambda v: abs(v - so))
        tr = [t for t in rm.trunks() if abs(t[0] - float(st["translate"])) < 0.13]
        if not tr:
            rows[d["alias"]]["no_trunk"] += 1
            continue
        trunk = min(tr, key=lambda t: (abs(t[0] - float(st["translate"])), abs(t[1] - float(st["tilt"]))))
        stat = rm.status(d_e, *trunk)
        # the demonstrated yaw about the rest: q_obj = R_z(psi) q_rest -> psi = yaw(q_obj q_rest^-1), relative to the base yaw
        rest = lib.rests[rm.meta["rest_id"]]
        qr = rest.quat_xyzw
        qinv = np.array([-qr[0], -qr[1], -qr[2], qr[3]])
        psi = S.yaw_of_quat(S.quat_mul(np.asarray(d["object"]["ori"]), qinv)) - float(st["base_pose2d"][2])
        psi = math.atan2(math.sin(psi), math.cos(psi))
        cell = np.asarray(d["cell_base"])
        ci = int(np.argmin(np.linalg.norm(rm.cells - cell[None], axis=1)))
        nY = len(rm.yaws)
        yi = int(round((psi % (2 * math.pi)) / (2 * math.pi / nY))) % nY
        lat = float(rm.meta["config"]["lattice_m"])
        exact = bool((stat[ci, yi] == RM.STATUS["feasible"]).any())
        near_c = np.nonzero(np.linalg.norm(rm.cells - rm.cells[ci][None], axis=1) <= lat * 1.5)[0]
        near = bool((stat[np.ix_(near_c, [(yi - 1) % nY, yi, (yi + 1) % nY])] == RM.STATUS["feasible"]).any())
        why = collections.Counter(RM.STATUS_NAME[int(v)] for v in stat[ci, yi])
        r = rows[d["alias"]]
        r["n"] += 1
        r["exact"] += exact
        r["near"] += near
        r["off_lattice_m_gt_lat"] += float(np.linalg.norm(rm.cells[ci] - cell)) > lat
        if not exact:
            r["why:" + why.most_common(1)[0][0]] += 1
        detail.append({"episode": d["episode"], "alias": d["alias"], "map": os.path.basename(rm.path), "d_e": d_e, "trunk": trunk, "cell": rm.cells[ci].tolist(),
                       "demo_cell": cell.tolist(), "yaw_index": yi, "psi": psi, "exact": exact, "near": near, "status": dict(why)})
    tot = collections.Counter()
    for al, r in sorted(rows.items()):
        n = max(r["n"], 1)
        tot.update(r)
        print(f"{al[:70]:70s} n {r['n']:3d} exact {r['exact'] / n:5.2f} near {r['near'] / n:5.2f} "
              + " ".join(f"{k}={v}" for k, v in sorted(r.items()) if k not in ("n", "exact", "near")))
    n = max(tot["n"], 1)
    print(f"ALL n {tot['n']} exact {tot['exact'] / n:.3f} near {tot['near'] / n:.3f}")
    if a.out:
        json.dump({"rows": {k: dict(v) for k, v in rows.items()}, "detail": detail}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
