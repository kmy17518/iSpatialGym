"""Robot-only end-effector reach volume for the large-scale stances (articulation of scene objects, wall alignment,
IORS with a big partner).

For every trunk of the grid (translate {0, 0.25, 0.5, 0.75, 1.0} x tilt {0, -0.3, -0.6, -0.9}) and each arm:

1. whole-body self-collision check of the trunk with untucked arms (floor plane only, no scene); colliding trunks are
   skipped;
2. a 5 cm lattice of eef positions in the base frame, x in [0.25, 0.90], lateral in [-0.70, 0.70], z per class group,
   pruned geometrically to positions within (arm length + 0.05) of that arm's shoulder;
3. six orientation classes (a = approach = eef +z, c = closing axis = eef +y): ``fwd_pinch_v`` (a +x, c +z) and
   ``fwd_pinch_h`` (a +x, c +y) for handles / knobs (z in [0.20, 1.70]); ``push_up`` (a +z), ``push_down`` (a -z),
   ``push_left`` (a +y), ``push_right`` (a -y), all with c = +x (pads parallel to the wall; z in [0.90, 1.90]). A cell is
   feasible when the pose and its pre-pose (0.10 m back along -a) pass the 12-seed batched IK with the locked joints of
   the untucked init configuration at that trunk (idle arm untucked), the floor plane being the only obstacle;
4. output per (arm, trunk, class): a boolean mask over the lattice; :meth:`ReachVolume.reachable` is the nearest-cell
   lookup used by the samplers for screening (scene collisions are checked at the concrete stance and by the exact IK).

The result depends on the robot model, the arm posture, the trunk grid, the lattice, the classes and the IK settings
only - not on the scene or the target - so it is computed once per robot and cached (``ISPATIALGYM_REACH_VOLUME``,
default ``~/.cache/ispatialgym/reach_volume``). The computation is sharded per (arm, trunk) so that it can run on
several machines::

    python -m ispatialgym.sampling.eef_reach_volume compute --arm right --translates 0 0.25 --tilts 0 -0.3 --out DIR
    python -m ispatialgym.sampling.eef_reach_volume compute --arm left --all --out DIR
    python -m ispatialgym.sampling.eef_reach_volume merge --out DIR       # shards -> reach_volume.npz + meta.json
    python -m ispatialgym.sampling.eef_reach_volume info --out DIR

``--smoke`` uses a 15 cm lattice and one trunk for a quick end-to-end test.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

log = logging.getLogger("ispatialgym.sampling.reach_volume")

#: JoyLo translate 1.0 (the "downward" keyframe itself) self-collides in cuRobo's sphere model with *every* arm posture
#: (measured 2026-09-15: untucked / tucked / ready all collide at 1.0, untucked is free up to 0.95), so the deepest grid
#: trunk is 0.95. Tilts are additive on torso_joint3 (negative = forward / down).
TRANSLATES = [0.0, 0.25, 0.5, 0.75, 0.95]
TILTS = [0.0, -0.3, -0.6, -0.9]
#: W1 (owner 2026-09-20 22:15): the GROUND half of the JoyLo curve (translate 1 -> 2: downward -> ground keyframe; tilt 0 - the
#: curve's own torso_joint3 already runs -0.94 -> its limit -1.83 there). Reachable since the base_link <-> torso_link2 sphere
#: pair is exempt (``r1pro_model.EXTRA_SELF_COLLISION_IGNORE_PAIRS``); probe 2026-09-21 on the floor blade (pitch 30, z 5.2 cm):
#: (0.95, -1.2) reaches 0.40-0.70 m ahead, (1.75, 0) 0.40-0.80, (2.0, 0) 0.40-0.85; 1.25 / 1.5 at tilt 0 reach nothing (the
#: shoulder still high and back). Added to the trunk grid of LOW tops (floor, low tables) only: ``reach_bands.trunk_grid``.
GROUND_TRUNKS = [(1.75, 0.0), (2.0, 0.0)]
LATTICE = 0.05
X_RANGE = (0.25, 0.90)
LAT_RANGE = (-0.70, 0.70)
PRE_POSE = 0.10
REACH_MARGIN = 0.05
#: orientation classes: approach axis a (eef +z), closing axis c (eef +y), z range of the lattice, in the base frame
CLASSES: Dict[str, Dict] = {
    "fwd_pinch_v": {"a": (1.0, 0.0, 0.0), "c": (0.0, 0.0, 1.0), "z": (0.20, 1.70), "group": "pinch"},
    "fwd_pinch_h": {"a": (1.0, 0.0, 0.0), "c": (0.0, 1.0, 0.0), "z": (0.20, 1.70), "group": "pinch"},
    "push_up": {"a": (0.0, 0.0, 1.0), "c": (1.0, 0.0, 0.0), "z": (0.90, 1.90), "group": "push"},
    "push_down": {"a": (0.0, 0.0, -1.0), "c": (1.0, 0.0, 0.0), "z": (0.90, 1.90), "group": "push"},
    "push_left": {"a": (0.0, 1.0, 0.0), "c": (1.0, 0.0, 0.0), "z": (0.90, 1.90), "group": "push"},
    "push_right": {"a": (0.0, -1.0, 0.0), "c": (1.0, 0.0, 0.0), "z": (0.90, 1.90), "group": "push"},
}
ARMS = ("right", "left")


#: committed copy of the merged volume (what the samplers read on every machine; env var overrides)
REPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runlogs", "reach_volume_20260915")


def default_dir() -> str:
    env = os.environ.get("ISPATIALGYM_REACH_VOLUME")
    if env:
        return env
    if os.path.exists(os.path.join(REPO_DIR, "reach_volume.npz")):
        return REPO_DIR
    return os.path.expanduser("~/.cache/ispatialgym/reach_volume")


def _arange(lo: float, hi: float, step: float) -> np.ndarray:
    n = int(round((hi - lo) / step)) + 1
    return lo + step * np.arange(n)


def lattice(step: float = LATTICE) -> Dict[str, np.ndarray]:
    return {
        "x": _arange(X_RANGE[0], X_RANGE[1], step),
        "lat": _arange(LAT_RANGE[0], LAT_RANGE[1], step),
        "z_pinch": _arange(CLASSES["fwd_pinch_v"]["z"][0], CLASSES["fwd_pinch_v"]["z"][1], step),
        "z_push": _arange(CLASSES["push_up"]["z"][0], CLASSES["push_up"]["z"][1], step),
    }


def class_quat(cls: str) -> th.Tensor:
    """eef orientation (xyzw) in the base frame for a class: columns [x, y, z] = [c x a, c, a]."""
    import omnigibson.utils.transform_utils as T

    a = th.tensor(CLASSES[cls]["a"], dtype=th.float32)
    c = th.tensor(CLASSES[cls]["c"], dtype=th.float32)
    x = th.cross(c, a, dim=0)
    return T.mat2quat(th.stack([x, c, a], dim=1))


def shard_name(arm: str, translate: float, tilt: float) -> str:
    return f"{arm}__t{translate:.2f}_k{tilt:+.1f}.npz"


def grid_meta(step: float) -> Dict:
    return {"translates": TRANSLATES, "tilts": TILTS, "lattice": step, "x_range": X_RANGE, "lat_range": LAT_RANGE, "pre_pose": PRE_POSE,
            "reach_margin": REACH_MARGIN, "classes": CLASSES, "posture": "untucked", "seeds": 12}


# ----------------------------------------------------------------------------------------------- lookup


@dataclass
class ReachVolume:
    """Merged reach volume: ``masks[arm][(translate, tilt)][cls]`` -> bool array (nx, nlat, nz)."""

    lat: Dict[str, np.ndarray]
    masks: Dict[str, Dict[Tuple[float, float], Dict[str, np.ndarray]]]
    meta: Dict

    @classmethod
    def load(cls, out_dir: Optional[str] = None) -> "ReachVolume":
        out_dir = out_dir or default_dir()
        npz = np.load(os.path.join(out_dir, "reach_volume.npz"))
        with open(os.path.join(out_dir, "meta.json")) as f:
            meta = json.load(f)
        lat = {k: npz[k] for k in ("x", "lat", "z_pinch", "z_push")}
        masks: Dict[str, Dict[Tuple[float, float], Dict[str, np.ndarray]]] = {}
        for key in npz.files:
            if "__" not in key or key.count("__") < 2:
                continue
            arm, trunk, c = key.split("__")
            t, k = trunk.split("_k")
            masks.setdefault(arm, {}).setdefault((float(t[1:]), float(k)), {})[c] = npz[key]
        return cls(lat, masks, meta)

    def trunks(self, arm: str) -> List[Tuple[float, float]]:
        return sorted(self.masks.get(arm, {}))

    def nearest_trunk(self, arm: str, translate: float, tilt: float) -> Optional[Tuple[float, float]]:
        ts = self.trunks(arm)
        if not ts:
            return None
        same_t = [t for t in ts if abs(t[0] - translate) < 1e-6]
        pool = same_t or ts
        return min(pool, key=lambda t: (abs(t[0] - translate), abs(t[1] - tilt)))

    def reachable(self, arm: str, translate: float, tilt: float, cls: str, p_base) -> bool:
        """Nearest-cell lookup of a base-frame eef position for the grid trunk with the same translate and the nearest
        tilt (``False`` outside the lattice or for a trunk that collided)."""
        trunk = self.nearest_trunk(arm, translate, tilt)
        if trunk is None or cls not in self.masks[arm].get(trunk, {}):
            return False
        m = self.masks[arm][trunk][cls]
        zs = self.lat["z_pinch"] if CLASSES[cls]["group"] == "pinch" else self.lat["z_push"]
        p = np.asarray([float(v) for v in p_base])
        step = float(self.meta["lattice"])
        ix = int(round((p[0] - self.lat["x"][0]) / step))
        il = int(round((p[1] - self.lat["lat"][0]) / step))
        iz = int(round((p[2] - zs[0]) / step))
        if not (0 <= ix < m.shape[0] and 0 <= il < m.shape[1] and 0 <= iz < m.shape[2]):
            return False
        return bool(m[ix, il, iz])

    @staticmethod
    def classify(a_base, c_base, pinch_tol_deg: float = 20.0, push_tol_deg: float = 45.0) -> Optional[str]:
        """Class whose (a, c) match the given base-frame approach / closing axes: pinch classes need both axes within
        ``pinch_tol_deg`` (the closing axis is undirected), push classes the closing axis within ``pinch_tol_deg`` of
        +x and the nearest of the four in-plane approach directions within ``push_tol_deg``. ``None`` = no class."""
        a = np.asarray(a_base, dtype=float)
        a = a / np.linalg.norm(a)
        c = np.asarray(c_base, dtype=float)
        c = c / np.linalg.norm(c)
        best, best_ang = None, None
        for name, spec in CLASSES.items():
            ca, cc = np.asarray(spec["a"]), np.asarray(spec["c"])
            ang_a = math.degrees(math.acos(max(-1.0, min(1.0, float(a @ ca)))))
            ang_c = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(c @ cc))))))
            tol_a = pinch_tol_deg if spec["group"] == "pinch" else push_tol_deg
            if ang_a <= tol_a and ang_c <= pinch_tol_deg and (best_ang is None or ang_a < best_ang):
                best, best_ang = name, ang_a
        return best


# ----------------------------------------------------------------------------------------------- computation


class _RobotOnly:
    """Minimal OmniGibson world: empty scene with a floor plane and the JoyLo R1Pro, plus the cuRobo backend."""

    def __init__(self, arms: Sequence[str]):
        from ispatialgym.oracle.curobo_planner import R1ProCuRobo  # warp pinning before omnigibson
        import omnigibson as og
        from ispatialgym.oracle import r1pro_model as M

        self.M = M
        cfg = {"env": {"action_frequency": 30, "physics_frequency": 120}, "scene": {"type": "Scene"},
               "robots": [M.joylo_robot_config("robot_r1", position=[0.0, 0.0, 0.0])]}
        self.env = og.Environment(configs=cfg)
        self.og = og
        self.robot = self.env.scene.robots[0]
        for _ in range(5):
            og.sim.step()
        self.backend = R1ProCuRobo(self.robot, prebuild=tuple(f"{a}_arm" for a in arms))
        self.backend.update_world()  # ground slab is the only obstacle
        self.jidx = self.backend._jidx
        idx = self.robot.trunk_control_idx
        self.trunk_limits = (self.robot.joint_lower_limits[idx].clone(), self.robot.joint_upper_limits[idx].clone())
        self.root_pos = self.backend.root_pose()[0].clone()

    def q_config(self, translate: float, tilt: float) -> th.Tensor:
        """28-D configuration at the trunk with untucked arms and open grippers, base_link upright at the root."""
        from ispatialgym.oracle.trunk import torso_q
        from ispatialgym.sampling.robot_utils import FINGER_OPEN, NOMINAL_BASE_Z

        q = th.zeros(28)
        q[self.jidx["base_footprint_z_joint"]] = NOMINAL_BASE_Z - float(self.root_pos[2])
        q[self.robot.trunk_control_idx] = torso_q(translate, tilt, self.trunk_limits)
        for arm in ARMS:
            q[self.robot.arm_control_idx[arm]] = self.robot.untucked_default_joint_pos[self.robot.arm_control_idx[arm]].clone()
            q[self.robot.gripper_control_idx[arm]] = FINGER_OPEN
        return q

    def base_pose(self, q: th.Tensor):
        return self.backend.fk([self.M.BASE_LINK], q)[self.M.BASE_LINK]

    def link_pos_base(self, q: th.Tensor, link: str) -> th.Tensor:
        """World position of a robot link at ``q`` expressed in the base frame (through the simulated robot; cuRobo FK
        only knows the tool frames)."""
        from ispatialgym.sampling import geometry as G

        self.robot.set_joint_positions(q)
        self.robot.keep_still()
        self.og.sim.step_physics()
        self.robot.set_joint_positions(q)
        self.robot.keep_still()
        lk = self.robot.eef_links[link] if link in ARMS else self.robot.links[link]
        p = lk.get_position_orientation()[0].float()
        bp, bq = self.robot.get_position_orientation()
        return G.rotate(G.qinv(bq.float()), p - bp.float())

    def arm_length(self, arm: str) -> float:
        """Shoulder -> eef distance with the arm hanging straight (all arm joints zero)."""
        q = self.q_config(0.5, 0.0)
        q[self.robot.arm_control_idx[arm]] = 0.0
        sh = self.link_pos_base(q, f"{arm}_arm_link1")
        ee = self.link_pos_base(q, arm)  # eef link of that arm
        return float(th.linalg.norm(ee - sh))


def compute_shard(world: _RobotOnly, arm: str, translate: float, tilt: float, step: float, out_dir: str, batch_size: int = 64) -> Optional[str]:
    """One (arm, trunk) shard; returns the written path or None when the trunk collides."""
    from ispatialgym.sampling import geometry as G

    t0 = time.time()
    lat = lattice(step)
    q = world.q_config(translate, tilt)
    coll = world.backend.check_collision(q.unsqueeze(0), emb_name="whole_body", sync_world=False, margin=0.0)
    path = os.path.join(out_dir, "shards", shard_name(arm, translate, tilt))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if bool(coll[0]):
        log.warning("%s trunk (%.2f, %.1f) collides with untucked arms; skipped", arm, translate, tilt)
        np.savez_compressed(path, collides=np.array(True), **{k: v for k, v in lat.items()})
        return None
    base = world.base_pose(q)
    shoulder = world.link_pos_base(q, f"{arm}_arm_link1")
    reach = world.arm_length(arm) + REACH_MARGIN
    results = {}
    n_total = n_pruned = n_ok = 0
    for cls, spec in CLASSES.items():
        zs = lat["z_pinch"] if spec["group"] == "pinch" else lat["z_push"]
        mask = np.zeros((len(lat["x"]), len(lat["lat"]), len(zs)), dtype=bool)
        pts, idx = [], []
        for i, x in enumerate(lat["x"]):
            for j, y in enumerate(lat["lat"]):
                for k, z in enumerate(zs):
                    p = th.tensor([float(x), float(y), float(z)])
                    n_total += 1
                    if float(th.linalg.norm(p - shoulder)) > reach:
                        n_pruned += 1
                        continue
                    pts.append(p)
                    idx.append((i, j, k))
        if not pts:
            results[cls] = mask
            continue
        quat_b = class_quat(cls)
        a_b = th.tensor(spec["a"], dtype=th.float32)
        poses = [G.compose(base, (p, quat_b)) for p in pts]
        pres = [G.compose(base, (p - a_b * PRE_POSE, quat_b)) for p in pts]
        succ, _ = world.backend.solve_ik_batch(f"{arm}_arm", world.M.EEF_LINK[arm], poses, q, batch_size=batch_size)
        sel = [i for i, s in enumerate(succ.tolist()) if s]
        if sel:
            succ_pre, _ = world.backend.solve_ik_batch(f"{arm}_arm", world.M.EEF_LINK[arm], [pres[i] for i in sel], q, batch_size=batch_size)
            for i, s in zip(sel, succ_pre.tolist()):
                if s:
                    mask[idx[i]] = True
                    n_ok += 1
        results[cls] = mask
        log.info("%s (%.2f, %.1f) %-12s %d candidates, %d pose-feasible, %d with pre-pose  [%.0fs]", arm, translate, tilt, cls, len(pts), len(sel),
                 int(mask.sum()), time.time() - t0)
    np.savez_compressed(path, collides=np.array(False), shoulder=shoulder.numpy(), reach=np.array(reach), **lat,
                        **{f"{arm}__t{translate:.2f}_k{tilt:+.1f}__{c}": m for c, m in results.items()})
    log.info("shard %s: %d lattice points, %d pruned, %d feasible cells, %.0fs", os.path.basename(path), n_total, n_pruned, n_ok, time.time() - t0)
    return path


def merge(out_dir: str, step: float):
    shards = sorted(glob.glob(os.path.join(out_dir, "shards", "*.npz")))
    if not shards:
        raise SystemExit(f"no shards in {out_dir}/shards")
    arrays: Dict[str, np.ndarray] = dict(lattice(step))
    collided = []
    for s in shards:
        d = np.load(s)
        if bool(d["collides"]):
            collided.append(os.path.basename(s)[:-4])
            continue
        for k in d.files:
            if k.count("__") >= 2:
                arrays[k] = d[k]
    meta = {**grid_meta(step), "shards": [os.path.basename(s) for s in shards], "collided_trunks": collided,
            "sha256": hashlib.sha256(json.dumps(grid_meta(step), sort_keys=True).encode()).hexdigest()[:16]}
    np.savez_compressed(os.path.join(out_dir, "reach_volume.npz"), **arrays)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"merged {len(shards)} shards ({len(collided)} colliding trunks) -> {out_dir}/reach_volume.npz")


def info(out_dir: str):
    rv = ReachVolume.load(out_dir)
    for arm in rv.masks:
        for trunk in rv.trunks(arm):
            cells = {c: int(m.sum()) for c, m in rv.masks[arm][trunk].items()}
            print(f"{arm} translate {trunk[0]:.2f} tilt {trunk[1]:+.1f}: " + ", ".join(f"{c} {n}" for c, n in cells.items()))
    print("collided trunks:", rv.meta.get("collided_trunks"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--arm", choices=list(ARMS) + ["both"], default="both")
    c.add_argument("--translates", type=float, nargs="*", default=None)
    c.add_argument("--tilts", type=float, nargs="*", default=None)
    c.add_argument("--all", action="store_true", help="every trunk of the grid")
    c.add_argument("--out", default=None)
    c.add_argument("--smoke", action="store_true", help="15 cm lattice, one trunk (0.5, 0.0): end-to-end test in ~1 min")
    c.add_argument("--skip-existing", action="store_true")
    m = sub.add_parser("merge"); m.add_argument("--out", default=None); m.add_argument("--lattice", type=float, default=LATTICE)
    i = sub.add_parser("info"); i.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = args.out or default_dir()
    os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(level=os.environ.get("ISPATIALGYM_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
    if args.cmd == "merge":
        return merge(out_dir, args.lattice)
    if args.cmd == "info":
        return info(out_dir)
    arms = list(ARMS) if args.arm == "both" else [args.arm]
    step = 0.15 if args.smoke else LATTICE
    translates = [0.5] if args.smoke else (TRANSLATES if args.all or not args.translates else args.translates)
    tilts = [0.0] if args.smoke else (TILTS if args.all or not args.tilts else args.tilts)
    world = _RobotOnly(arms)
    for arm in arms:
        log.info("%s arm length %.3f m", arm, world.arm_length(arm))
    n = 0
    for arm in arms:
        for tr in translates:
            for ti in tilts:
                path = os.path.join(out_dir, "shards", shard_name(arm, tr, ti))
                if args.skip_existing and os.path.exists(path):
                    continue
                compute_shard(world, arm, tr, ti, step, out_dir)
                n += 1
    print(f"REACH_VOLUME_DONE shards={n} out={out_dir}")
    world.og.shutdown()


if __name__ == "__main__":
    main()
