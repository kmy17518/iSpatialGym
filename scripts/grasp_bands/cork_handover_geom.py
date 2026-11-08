"""i7-cork (owner decision 2026-11-08 4.2 (i)): can both R1Pro hands hold the cork ncxgpe at once (the hand-over), CPU only?

The cork is 34.5 mm long (object z -18.3 male end .. +16.1 wide end), r 6.5 mm at the male end, 8.9 mm at its widest. At the hand-over the
right hand holds the narrow (male) end, the left hand the wide end; both pinch ACROSS the axis (a pinch along the axis would cover an end).
Two pinch families per hand:

* ``axial``: the approach along the cork axis from its own end (right: +z_obj from below the male end; left: -z_obj from beyond the wide end),
  the eef ``d`` mm outside the end face, the fingertips reaching 18.8 - d mm along the cork;
* ``side``: the approach perpendicular to the axis, the pad centre at a height ``z`` of the cork.

For each pair (right x left x the left's turn about the axis) the exact hand hulls (``hand_geometry``, every role: fingers closed to the cork's
width at their pinch, palm, wrist camera, wrist) of both hands: the minimum distance between them (convex hull vs convex hull, by the
separating-axis-free approximation: surface samples at 1.5 mm + the inside test), and each hand's finger extent along the cork axis.
Prints one ``HANDOVER_GEOM {json}`` line per pair and a summary (the pairs with >= 3 mm between the hands).

    ssh svl11 cat <glib>/models/cork__ncxgpe__s1__attachment_free/geometry.json | python scripts/grasp_bands/cork_handover_geom.py --geometry -
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np


def rotm(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def frame(approach, closing):
    """eef rotation (columns x, y = closing, z = approach)."""
    z = np.asarray(approach, float) / np.linalg.norm(approach)
    y = np.asarray(closing, float) - np.dot(closing, z) * z
    y /= np.linalg.norm(y)
    return np.stack([np.cross(y, z), y, z], axis=1)


def radius_at(V, z0, band=0.002):
    m = np.abs(V[:, 2] - z0) <= band
    return float(np.hypot(V[m, 0], V[m, 1]).max()) if m.any() else None


def hand_points(hg, R, p, width, roles=("finger", "palm", "wrist_camera", "wrist")):
    """World (object-frame) surface points of one hand at eef pose (R, p), fingers closed to ``width`` (each finger at width / 2)."""
    q = max(0.0, min(0.05, 0.5 * width))
    P, idx = hg.surface(spacing=0.0015, roles=roles, finger_q=(q, q))
    return P @ R.T + p, idx


def hull_dist(A, B):
    from scipy.spatial import cKDTree

    d, _ = cKDTree(B).query(A, k=1)
    return float(d.min())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", required=True, help="the entry's geometry.json ('-' = stdin)")
    a = ap.parse_args(argv)
    geo = json.load(sys.stdin if a.geometry == "-" else open(a.geometry))
    V = np.concatenate([np.asarray(pc["vertices"], float) for pc in geo["pieces"]])
    z_lo, z_hi = float(V[:, 2].min()), float(V[:, 2].max())
    from ispatialgym.grasping import hand_geometry as HG

    hr, hl = HG.load("right"), HG.load("left")
    tip = 0.0188
    rows = []
    # right hand on the male (narrow) end
    rights = []
    for d in (0.0, 0.003, 0.006, 0.009):  # axial: eef d outside the male end face, approach +z
        zc = z_lo - d + tip - 0.006  # the pinch about 6 mm short of the fingertips
        w = 2 * (radius_at(V, max(zc, z_lo + 0.002)) or 0.0065)
        rights.append(("axial", d, frame([0, 0, 1], [1, 0, 0]), np.array([0, 0, z_lo - d]), w, (z_lo, z_lo - d + tip)))
    for zp in (-0.014, -0.010, -0.006):  # side: the pad centre at zp, approach -x
        w = 2 * (radius_at(V, zp) or 0.007)
        R = frame([-1, 0, 0], [0, 1, 0])
        p = np.array([0, 0, zp]) - R[:, 2] * (-0.01186)  # the pad centre on the axis
        rights.append(("side", zp, R, p, w, None))
    lefts = []
    for d in (0.0, 0.003, 0.006, 0.009):  # axial: eef d beyond the wide end face, approach -z
        zc = z_hi + d - tip + 0.006
        w = 2 * (radius_at(V, min(zc, z_hi - 0.002)) or 0.008)
        lefts.append(("axial", d, frame([0, 0, -1], [1, 0, 0]), np.array([0, 0, z_hi + d]), w, (z_hi + d - tip, z_hi)))
    for zp in (0.004, 0.008, 0.012):
        w = 2 * (radius_at(V, zp) or 0.009)
        R = frame([1, 0, 0], [0, 1, 0])
        p = np.array([0, 0, zp]) - R[:, 2] * (-0.01186)
        lefts.append(("side", zp, R, p, w, None))
    for (rk, rv, Rr, pr, wr, rext) in rights:
        Pr, ir = hand_points(hr, Rr, pr, wr)
        fr = Pr[ir == 0]
        for (lk, lv, Rl0, pl0, wl, lext) in lefts:
            for phi in (0.0, 45.0, 90.0, 135.0, 180.0):
                Rz = rotm([0, 0, 1], math.radians(phi))
                Rl, pl = Rz @ Rl0, Rz @ pl0
                Pl, il = hand_points(hl, Rl, pl, wl)
                fl = Pl[il == 0]
                dist = hull_dist(Pr, Pl)
                row = {"right": [rk, round(rv * 1000, 1)], "left": [lk, round(lv * 1000, 1)], "left_turn_deg": phi, "hands_min_mm": round(dist * 1000, 2),
                       "right_finger_z_mm": [round(float(fr[:, 2].min()) * 1000, 1), round(float(fr[:, 2].max()) * 1000, 1)],
                       "left_finger_z_mm": [round(float(fl[:, 2].min()) * 1000, 1), round(float(fl[:, 2].max()) * 1000, 1)],
                       "widths_mm": [round(wr * 1000, 1), round(wl * 1000, 1)], "mating_face_free_mm": round((float(fl[:, 2].min()) - z_lo) * 1000, 1)}
                rows.append(row)
                print("HANDOVER_GEOM " + json.dumps(row), flush=True)
    ok = [r for r in rows if r["hands_min_mm"] >= 3.0 and r["mating_face_free_mm"] >= 2.0]
    best = sorted(rows, key=lambda r: -r["hands_min_mm"])[:5]
    print("HANDOVER_GEOM_SUMMARY " + json.dumps({"cork_z_mm": [round(z_lo * 1000, 1), round(z_hi * 1000, 1)], "pairs": len(rows), "pairs_3mm_apart": len(ok),
                                                  "best": best}), flush=True)


if __name__ == "__main__":
    main()
