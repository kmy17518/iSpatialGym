"""Drop test of the puzzle board (``tray/isgboard``): does a Kohs cube released above the rim seat itself in its pocket?

Owner 2026-09-20 23:00 (option 2 follow-up): the sampler releases every block with its bottom 4 mm above the rim, aimed
``bias`` toward the pocket's outer corner; the landing error on the table was 2.5-4.7 mm plus the drop, so the chamfered
mouth must bring an error of a few mm back onto the seat. This script spawns the board (fixed) and one cube in an empty
OmniGibson scene and drops the cube over each pocket with the bias plus extra offsets (inward / outward / yaw), then
reports where it came to rest: the centre's xy error from the pocket centre, the yaw error and the bottom height.

Pass (owner): seated in its own pocket within 1 mm / 2 deg for every case except the divider-apex probe (which only
reports where the cube goes). Run (local GPU ~3 GB or a pod)::

    OMNIGIBSON_HEADLESS=1 python drop_test.py [--bias 0.004] [--settle-s 2.0] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import torch as th  # noqa: E402

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "model" / "isgboard" / "build_manifest.json"


def quat_from_yaw(yaw: float) -> th.Tensor:
    return th.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def yaw_of(q: th.Tensor) -> float:
    x, y, z, w = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_mod90_deg(q: th.Tensor) -> float:
    """Yaw error of a cube modulo its 90-degree symmetry, in degrees, in (-45, 45]."""
    d = math.degrees(yaw_of(q)) % 90.0
    return d - 90.0 if d > 45.0 else d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bias", type=float, default=0.004, help="release aim toward the pocket's outer corner (m, per axis)")
    ap.add_argument("--above-rim", type=float, default=0.004, help="cube bottom above the rim top at release (m)")
    ap.add_argument("--settle-s", type=float, default=2.0)
    ap.add_argument("--json", default=str(HERE / "model" / "isgboard" / "drop_test.json"))
    ap.add_argument("--trace", default="", help="case name(s), comma separated, whose fall is printed step by step")
    ap.add_argument("--board-friction", type=float, default=None, help="override the board's static / dynamic friction (PhysicsMaterial on base_link)")
    ap.add_argument("--cube-friction", type=float, default=None, help="override the cube's static / dynamic friction")
    ap.add_argument("--only", default="", help="run only these case names (comma separated)")
    ap.add_argument("--sweep", action="store_true", help="dense error grid (+-6 mm per axis / diagonal, yaw +-2) on pockets 3 and 0")
    args = ap.parse_args()

    import omnigibson as og
    from omnigibson.objects import DatasetObject

    geo = json.loads(MANIFEST.read_text())
    edge = float(geo["edge_m"])
    floor_z = float(geo["pocket_floor_z_m"])
    rim_top = float(geo["rim_top_z_m"])
    centres = [tuple(float(v) for v in c) for c in geo["pocket_centres"]]
    signs = [tuple(int(v) for v in s) for s in geo["outer_corner_sign"]]

    cfg = {"env": {"action_frequency": 30, "physics_frequency": 120}, "scene": {"type": "Scene"}, "robots": []}
    env = og.Environment(configs=cfg)
    scene = env.scene
    b_kw = {}
    if args.board_friction is not None:
        b_kw["link_physics_materials"] = {"base_link": {"static_friction": args.board_friction, "dynamic_friction": args.board_friction, "restitution": 0.0}}
    c_kw = {}
    if args.cube_friction is not None:
        c_kw["link_physics_materials"] = {"base_link": {"static_friction": args.cube_friction, "dynamic_friction": args.cube_friction, "restitution": 0.0}}
    board = DatasetObject(name="board", category="tray", model="isgboard", dataset_name="ispatialgym-assets", fixed_base=True, **b_kw)
    cube = DatasetObject(name="cube", category="toy_dice", model="isgblk", dataset_name="ispatialgym-assets", **c_kw)
    og.sim.stop()
    scene.add_object(board)
    scene.add_object(cube)
    og.sim.play()
    board.set_position_orientation(th.tensor([0.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    board.keep_still()
    for _ in range(5):
        og.sim.step_physics()
    lo, hi = board.aabb
    print(f"board aabb x {float(lo[0]):.4f}..{float(hi[0]):.4f} y {float(lo[1]):.4f}..{float(hi[1]):.4f} z {float(lo[2]):.4f}..{float(hi[2]):.4f}", flush=True)
    board_bottom = float(lo[2])
    n_steps = int(round(args.settle_s * 30))  # action_frequency 30 Hz full steps

    def drop(xy, yaw_deg: float, trace: bool = False):
        """Release the cube with its bottom ``above_rim`` above the rim top at ``xy`` with ``yaw``; return its rest pose facts."""
        q = quat_from_yaw(math.radians(yaw_deg))
        # the cube's root sits at its bottom centre (base_link_offset z = +edge / 2 -> the bbox centre is edge / 2 above the root)
        cube.set_position_orientation(th.tensor([xy[0], xy[1], 0.5]), q)
        cube.keep_still()
        with og.sim.render_on_step(True):
            og.sim.step()
        cube.set_position_orientation(th.tensor([xy[0], xy[1], 0.5]), q)
        cube.keep_still()
        with og.sim.render_on_step(True):
            og.sim.step()
        c_lo, c_hi = cube.aabb
        root_p, _ = cube.get_position_orientation()
        bottom_off = float(c_lo[2]) - float(root_p[2])  # root -> bottom
        z_root = board_bottom + rim_top + args.above_rim - bottom_off
        cube.set_position_orientation(th.tensor([xy[0], xy[1], z_root]), q)
        cube.set_linear_velocity(th.zeros(3))
        cube.set_angular_velocity(th.zeros(3))
        cube.wake()  # a body posed while asleep stays put (the floating-block pitfall of 0919-183037)
        # full (rendering) simulator steps: the physics -> USD/Fabric sync that refreshes poses / AABBs happens there
        # (sim_utils.full_step); ``step_physics`` alone leaves every reading stale
        for k in range(n_steps):
            with og.sim.render_on_step(True):
                og.sim.step()
            if k % 15 == 0:
                cube.wake()
            if trace and (k < 24 or k % 15 == 0):
                c_lo_, c_hi_ = cube.aabb
                p_, q_ = cube.get_position_orientation()
                cen_ = 0.5 * (c_lo_ + c_hi_)
                print(f"    t={k / 30:.3f}s centre ({(float(cen_[0]) - xy[0]) * 1000:+.2f}, {(float(cen_[1]) - xy[1]) * 1000:+.2f}) mm from release, bottom {(float(c_lo_[2]) - board_bottom - floor_z) * 1000:+.2f} mm, "
                      f"tilt {math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (float(q_[0]) ** 2 + float(q_[1]) ** 2))))):.1f} deg", flush=True)
        c_lo, c_hi = cube.aabb
        p, q_end = cube.get_position_orientation()
        centre = 0.5 * (c_lo + c_hi)
        if trace:
            # the cube's corners (root = bottom centre; edge / 2 above it is the geometric centre) - where does it rest?
            x_, y_, z_, w_ = (float(v) for v in q_end)
            R = th.tensor([[1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - z_ * w_), 2 * (x_ * z_ + y_ * w_)],
                           [2 * (x_ * y_ + z_ * w_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - x_ * w_)],
                           [2 * (x_ * z_ - y_ * w_), 2 * (y_ * z_ + x_ * w_), 1 - 2 * (x_ * x_ + y_ * y_)]])
            geo_c = th.tensor([float(p[0]), float(p[1]), float(p[2])]) + R @ th.tensor([0.0, 0.0, edge / 2])
            corners = []
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        cpt = geo_c + R @ th.tensor([sx * edge / 2, sy * edge / 2, sz * edge / 2])
                        corners.append((float(cpt[2]) - board_bottom - floor_z, float(cpt[0]), float(cpt[1])))
            corners.sort()
            print("    lowest corners (z above floor, x, y) mm: " + "; ".join(f"({z * 1000:+.1f}, {x * 1000:+.1f}, {y * 1000:+.1f})" for z, x, y in corners[:4]), flush=True)
        return {"centre_xy": [float(centre[0]), float(centre[1])], "bottom_z_above_floor": float(c_lo[2]) - board_bottom - floor_z,
                "yaw_err_deg": yaw_mod90_deg(q_end), "tilt_deg": math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (float(q_end[0]) ** 2 + float(q_end[1]) ** 2)))))}

    cases = []
    # the base case over every pocket: the bias toward the outer corner, no extra error
    for i, (c, sgn) in enumerate(zip(centres, signs)):
        cases.append((f"pocket{i}_bias", i, (args.bias * sgn[0], args.bias * sgn[1]), 0.0))
    # pocket 3 (near-right): extra errors along the diagonal (both axes) and along one axis toward the divider
    i = 3
    c, sgn = centres[i], signs[i]
    for inward in (0.003, 0.005):
        cases.append((f"pocket{i}_inward{inward * 1000:.0f}mm_diag", i, ((args.bias - inward) * sgn[0], (args.bias - inward) * sgn[1]), 0.0))
        cases.append((f"pocket{i}_inward{inward * 1000:.0f}mm_x", i, ((args.bias - inward) * sgn[0], args.bias * sgn[1]), 0.0))
    for outward in (0.005, 0.010):
        cases.append((f"pocket{i}_outward{outward * 1000:.0f}mm_diag", i, ((args.bias + outward) * sgn[0], (args.bias + outward) * sgn[1]), 0.0))
    for yaw in (3.0, -3.0):
        cases.append((f"pocket{i}_yaw{yaw:+.0f}deg", i, (args.bias * sgn[0], args.bias * sgn[1]), yaw))
    cases.append((f"pocket{i}_inward5mm_yaw3", i, ((args.bias - 0.005) * sgn[0], (args.bias - 0.005) * sgn[1]), 3.0))
    # pocket 0 (far-left) with the worst inward error too (the flipped blocks land there)
    c0, s0 = centres[0], signs[0]
    cases.append(("pocket0_inward5mm_diag", 0, ((args.bias - 0.005) * s0[0], (args.bias - 0.005) * s0[1]), 0.0))
    if args.sweep:
        # a denser error grid on two pockets (3: near-right, 0: far-left): +-6 mm along x, along y and on the diagonal, yaw +-2
        cases = []
        for pi in (3, 0):
            sgn = signs[pi]
            for e in (-0.006, -0.004, -0.002, 0.0, 0.002, 0.004, 0.006):  # inward negative, outward positive (per axis, on top of the bias)
                cases.append((f"p{pi}_x{e * 1000:+.0f}", pi, ((args.bias + e) * sgn[0], args.bias * sgn[1]), 0.0))
                cases.append((f"p{pi}_y{e * 1000:+.0f}", pi, (args.bias * sgn[0], (args.bias + e) * sgn[1]), 0.0))
                cases.append((f"p{pi}_d{e * 1000:+.0f}", pi, ((args.bias + e) * sgn[0], (args.bias + e) * sgn[1]), 0.0))
            for yaw in (-2.0, 2.0):
                cases.append((f"p{pi}_yaw{yaw:+.0f}_in4", pi, ((args.bias - 0.004) * sgn[0], (args.bias - 0.004) * sgn[1]), yaw))
                cases.append((f"p{pi}_yaw{yaw:+.0f}_out4", pi, ((args.bias + 0.004) * sgn[0], (args.bias + 0.004) * sgn[1]), yaw))
    # the divider-apex probe: centred over the ridge between pockets 2 and 3 (x = 0)
    cases.append(("apex_between_2_3", 3, (-centres[3][0], 0.0), 0.0))

    results = []
    all_ok = True
    only = {n for n in args.only.split(",") if n}
    traced = {n for n in args.trace.split(",") if n}
    for name, pi, off, yaw in cases:
        if only and name not in only:
            continue
        c = centres[pi]
        xy = (c[0] + off[0], c[1] + off[1])
        r = drop(xy, yaw, trace=name in traced)
        err = [r["centre_xy"][0] - c[0], r["centre_xy"][1] - c[1]]
        # 1 mm = the seat clearance: a cube flush against a seat wall reads 1.00 mm (+ rounding) and is seated
        seated = abs(err[0]) <= 0.00105 and abs(err[1]) <= 0.00105 and abs(r["yaw_err_deg"]) <= 2.0 and abs(r["bottom_z_above_floor"]) <= 0.0015 and r["tilt_deg"] <= 2.0
        probe = name.startswith("apex")
        if not probe and not seated:
            all_ok = False
        results.append({"case": name, "pocket": pi, "release_offset_mm": [off[0] * 1000, off[1] * 1000], "release_yaw_deg": yaw,
                        "rest_err_mm": [err[0] * 1000, err[1] * 1000], "yaw_err_deg": r["yaw_err_deg"], "tilt_deg": r["tilt_deg"],
                        "bottom_above_floor_mm": r["bottom_z_above_floor"] * 1000, "seated": seated, "probe": probe})
        print(f"{name:32s} release ({off[0] * 1000:+.1f}, {off[1] * 1000:+.1f}) mm yaw {yaw:+.0f} -> rest err ({err[0] * 1000:+.2f}, {err[1] * 1000:+.2f}) mm, "
              f"yaw {r['yaw_err_deg']:+.2f} deg, tilt {r['tilt_deg']:.2f} deg, bottom {r['bottom_z_above_floor'] * 1000:+.2f} mm {'SEATED' if seated else ('probe' if probe else 'NOT SEATED')}",
              flush=True)
    out = {"bias_m": args.bias, "above_rim_m": args.above_rim, "settle_s": args.settle_s, "geometry": {k: geo[k] for k in ("pocket_pitch_m", "seat_inner_m", "wall_h_m", "chamfer_h_m", "divider_w_m", "ridge_apex_z_m")},
           "all_seated": all_ok, "results": results}
    Path(args.json).write_text(json.dumps(out, indent=2) + "\n")
    print(f"DROP_TEST_{'PASSED' if all_ok else 'FAILED'} -> {args.json}", flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
