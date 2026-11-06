"""Scene-independent reach maps: where, relative to the robot, the right hand can grasp an object model resting on a table edge.

Replaces the per-scene / per-support bands of :mod:`reach_bands` (kept, explicitly separate: a map is never returned for a band
query or vice versa). A map is computed in a minimal synthetic world - the R1Pro alone on the floor (cuRobo: the floor cuboid and
self-collision; no household scene is loaded) - for one (object library entry, rest, exact support-height anchor, arm), and holds:

* the **frame**: base_link projected on the floor, +x ahead, +y the robot's left, z up from the floor;
* the **support**: a synthetic slab whose top is at the anchor height ``h`` above the floor, its near edge ``d_e`` ahead of the base
  (``d_e`` = the actual base-to-edge distance: the sampler's stand-off plus the support's overhang), ``slab_depth_m`` deep, ``slab_long_m``
  wide (>= the longest support edge of the inventory + margin), ``slab_thickness_m`` thick (a slab, never a block to the floor);
* per **edge distance** ``d_e`` x **trunk** (``reach_bands.trunk_grid(h)``; eligibility: the idle-arm posture free of the slab and of
  itself) x **cell** (an ``lattice_m`` lattice of object-root xy) x **object yaw** (``n_yaws``) x **grasp** (validated grasps, their
  finger-swapped twins unless stored, members of non-yaw-equivalent symmetry families): a status code (:data:`STATUS`) - feasible, or why
  not, or ``skipped`` (unknown: the accelerated mode's early cut; never labelled exhaustive).

Per (trunk, cell, yaw, grasp): ``T_world_eef = place_object(rest, yaw, cell, h) @ T_object_eef`` (``schema.place_object``: measured
``h_root``, no offset), the demo pre-grasp (``conventions.pre_pose``, 0.10 m back along +z); the hand's exact hulls against the slab
under the contact policy at the pre-grasp, along the straight approach and through the closing sweep (IK-independent, per ``d_e``);
batched IK of the pre-grasp (``seeds`` seeds) and of the grasp seeded from it (same branch: max joint step <= ``branch_max_rad``);
then, per ``d_e``, every non-finger sphere of the whole robot at the pre-grasp, the grasp and ``approach_configs`` joint-interpolated
configurations clear of the slab (``contact_policy.sphere_support_ok``), the hand hulls at the FK eef poses within the finger
allowance, and the arm's spheres (not the hand's) clear of the object at the cell. The IK itself never sees the slab - that is
what lets one IK pass serve every ``d_e`` (all seeds' solutions are kept and the first passing one counts).

Maps are proposal / filter data: the sampler transforms cells into a real stance on a real edge and re-checks everything exactly
(scene geometry, clutter, legs, approach) - a map's feasible cell is not an executable trajectory.

    python -m ispatialgym.sampling.reach_maps bench
    python -m ispatialgym.sampling.reach_maps compute --jobs M.json --glib ROOT --anchors anchors.json [--heights 0.7351 ...] --out MAPS
    python -m ispatialgym.sampling.reach_maps info MAP.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym.grasping import contact_policy as CP
from ispatialgym.grasping import hand_geometry as H
from ispatialgym.grasping import schema as S

SCHEMA = "isg-reach-map/1"
STATUS = {"unevaluated": 0, "feasible": 1, "off_support": 2, "reach_bound": 3, "hand_policy": 4, "ik_pre": 5, "ik_grasp": 6,
          "slab_spheres": 7, "object_arm": 8, "skipped": 9, "trunk_ineligible": 10, "branch_jump": 11}
STATUS_NAME = {v: k for k, v in STATUS.items()}


@dataclass
class MapConfig:
    arm: str = "right"
    lattice_m: float = 0.08
    x_max_m: float = 1.20
    y_half_m: float = 0.88
    edge_distances: Tuple[float, ...] = (0.35, 0.45, 0.55)
    n_yaws: int = 8
    seeds: int = 8
    batch: int = 2048
    branch_max_rad: float = 1.0
    approach_configs: int = 3
    family_samples: int = 4
    twins: bool = True
    mode: str = "full"  # "accelerated": the early cut (measured 2026-11-02: 1-4 % fewer IK problems, up to 6.5 % fewer feasible entries)
    slab_depth_m: float = 1.2
    slab_long_m: float = 3.2
    slab_thickness_m: float = 0.04
    hand_spacing_m: float = 0.006
    reach_margin_m: float = 0.03
    postures: Tuple[str, ...] = ("untucked", "ready", "tucked")

    def key(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("batch")  # does not change results
        return d


def lattice(cfg: MapConfig) -> np.ndarray:
    xs = np.arange(min(cfg.edge_distances), cfg.x_max_m + 1e-9, cfg.lattice_m)
    ys = np.arange(-cfg.y_half_m, cfg.y_half_m + 1e-9, cfg.lattice_m)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], 1)


# ------------------------------------------------------------------------------------------------ grasp set


def grasp_set(lib: S.ModelLibrary, rest_id: str, cfg: MapConfig) -> List[Dict[str, Any]]:
    """The map's grasp axis: every validated grasp of the rest, its twin (eef turned pi about +z) unless an equivalent row is stored,
    members of its family when the family is NOT vertical-yaw-equivalent (``family_samples`` over the interval incl. endpoints; the
    yaw-equivalent ones are covered by the yaw axis and keep the representative with ``yaw_equivalent``)."""
    from ispatialgym.grasping import symmetry as SY

    out: List[Dict[str, Any]] = []
    fams = lib.families()
    rows = lib.grasps(rest_id)
    Ts = [S.pose_of(g["T_object_eef"]) for g in rows]
    for g, T in zip(rows, Ts):
        fam = fams.get(g.get("family_id") or "", {"type": "none"})
        params = [0.0]
        if fam.get("type") in ("rotational", "translational") and not fam.get("vertical_yaw_equivalent"):
            # the representative (the physics-tested pose) always, plus samples over the interval incl. its endpoints
            params = sorted({0.0, *[round(float(v), 6) for v in S.family_samples(fam, cfg.family_samples)]})
        for p in params:
            Tm = S.family_member_T(fam, T, p)
            base = {"base": g["name"], "family_id": g.get("family_id"), "param": round(float(p), 5), "preshape_q": g.get("preshape_q", 0.05),
                    "contact_q": 0.5 * float(g.get("width_m", 0.04)) + 0.0009, "physics": "tested" if abs(p) < 1e-9 else "family_member",
                    "yaw_equivalent": bool(fam.get("vertical_yaw_equivalent"))}
            out.append({**base, "name": g["name"] if abs(p) < 1e-9 else f"{g['name']}@{p:+.3f}", "twin": False, "T": Tm})
            if cfg.twins:
                Tt = SY.twin_T(Tm)
                if not any(SY._pose_close(Tt, T2, 0.015, 20.0) for T2 in Ts):
                    out.append({**base, "name": f"{out[-1]['name']}~r180", "twin": True, "T": Tt})
    return out


# ------------------------------------------------------------------------------------------------ brace / contact poses (prompt 2 §10)

BRACE_FRACTIONS = (0.2, 0.4, 0.6, 0.8)
BRACE_SIDES = {"near": (1.0, 0.0), "far": (-1.0, 0.0), "left": (0.0, -1.0), "right": (0.0, 1.0)}  # horizontal approach in the map frame
BRACE_PITCHES_DEG = (0.0, 45.0)
#: the oracle's brace stop: the fingertip plane this far before the face along the approach (``conventions.fingertip_eef``, spec 5.5)
BRACE_CLEARANCE_M = 0.005


def brace_classes(fractions: Sequence[float] = BRACE_FRACTIONS) -> List[Dict[str, Any]]:
    """The contact classes of a brace map (``reach_bands.CONTACT_CLASSES`` generalised): side ``near`` (pads on the face toward the robot,
    pushing away: approach +x of the map frame), ``far`` (-x, pulling), ``left`` / ``right`` (-y / +y, across); approach horizontal or
    pitched 45 deg DOWN toward the face; closing axis ``v`` (vertical-most axis perpendicular to the approach: fingers stacked) or ``h``
    (horizontal, perpendicular to the approach: fingers side by side); at each height fraction of the object's settled height; plus the
    top press (approach straight down, two wrist yaws) at the top. Closed fingers, the fingertip plane ``BRACE_CLEARANCE_M`` before the
    contact point (``conventions.fingertip_eef``: the oracle's brace stop)."""
    out = []
    for side in BRACE_SIDES:
        for pitch in BRACE_PITCHES_DEG:
            for ax in ("v", "h"):
                for f in fractions:
                    out.append({"name": f"{side}_{ax}_p{int(pitch)}_f{f:.1f}", "base": f"{side}_{ax}_p{int(pitch)}", "side": side, "pitch_deg": pitch,
                                "closing": ax, "fraction": f, "twin": False, "family_id": None, "param": None, "preshape_q": 0.0, "contact_q": 0.0,
                                "physics": "geometric_contact", "yaw_equivalent": False})
    for wy in (0, 90):
        out.append({"name": f"top_press_y{wy}", "base": f"top_press_y{wy}", "side": "top", "pitch_deg": 90.0, "closing": f"y{wy}", "fraction": 1.0,
                    "twin": False, "family_id": None, "param": None, "preshape_q": 0.0, "contact_q": 0.0, "physics": "geometric_contact",
                    "yaw_equivalent": False})
    return out


def brace_targets(V_rest: np.ndarray, yaws: Sequence[float], height: float, h_root: float, classes: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """(Y, G, 4, 4) contact eef poses with the object root at the map origin xy, and (Y, G, 3) contact points: per yaw the object's
    hull (``V_rest``: collision vertices in the rest frame) is placed, each class's contact point is the hull's extreme point against the
    approach at the class's height (cross-section +-1 cm; the top press: the top centre), ``fingertip_eef`` stops the fingertips 5 mm before it."""
    from ispatialgym.robot import conventions as C

    nY, nG = len(yaws), len(classes)
    TE = np.tile(np.eye(4), (nY, nG, 1, 1))
    P = np.zeros((nY, nG, 3))
    for yi, yaw in enumerate(yaws):
        R = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.0]])
        W = V_rest @ R.T
        W[:, 2] += height + h_root
        zlo, zhi = float(W[:, 2].min()), float(W[:, 2].max())
        H = zhi - height
        for gi, c in enumerate(classes):
            if c["side"] == "top":
                top = W[W[:, 2] >= zhi - 0.01]
                p = np.array([top[:, 0].mean(), top[:, 1].mean(), zhi])
                a = np.array([0.0, 0.0, -1.0])
                cl = np.array([1.0, 0.0, 0.0]) if c["closing"] == "y0" else np.array([0.0, 1.0, 0.0])
            else:
                ah = np.array([*BRACE_SIDES[c["side"]], 0.0])
                pr = math.radians(c["pitch_deg"])
                a = ah * math.cos(pr) + np.array([0, 0, -1.0]) * math.sin(pr)
                z = height + c["fraction"] * H
                sl = W[np.abs(W[:, 2] - z) <= 0.01]
                if len(sl) == 0:
                    sl = W
                ctr = sl[:, :2].mean(0)
                e = float(((sl[:, :2] - ctr) @ (-ah[:2])).max())
                p = np.array([ctr[0] - ah[0] * e, ctr[1] - ah[1] * e, z])
                hz = np.array([0.0, 0.0, 1.0])
                cl = hz - (hz @ a) * a if c["closing"] == "v" else np.cross(hz, ah)
                cl = cl / np.linalg.norm(cl)
            pos, quat = C.fingertip_eef(th.as_tensor(p, dtype=th.float32), th.as_tensor(a, dtype=th.float32), th.as_tensor(cl, dtype=th.float32), clearance=BRACE_CLEARANCE_M)
            TE[yi, gi] = S.pose_mat(pos.numpy(), quat.numpy())
            P[yi, gi] = p
    return TE, P


# ------------------------------------------------------------------------------------------------ the synthetic world


class SynthWorld:
    """The R1Pro on an empty floor with its cuRobo backend (floor cuboid only): kinematics, batched IK, sphere FK. One per process,
    reused across heights / models (``reset`` between maps: the IK solvers' seed streams)."""

    def __init__(self, cfg: MapConfig):
        from ispatialgym.oracle.curobo_planner import R1ProCuRobo  # warp pinning before omnigibson
        import omnigibson as og

        from ispatialgym.oracle import r1pro_model as M

        t0 = time.time()
        self.cfg, self.og, self.M = cfg, og, M
        self.env = og.Environment(configs={"env": {"action_frequency": 30, "physics_frequency": 120}, "scene": {"type": "Scene"},
                                           "robots": [M.joylo_robot_config("robot", obs_modalities=["proprio"], position=[0.0, 0.0, 0.0])]})
        self.robot = self.env.scene.robots[0]
        for _ in range(3):
            og.sim.step()
        self.backend = R1ProCuRobo(self.robot, prebuild=(), mesh_cache_size=8, cuboid_cache_size=8)
        self.backend.update_world()
        rp, rq = self.backend.root_pose()
        assert float(th.linalg.norm(rp[:2])) < 1e-4 and abs(float(rq[3])) > 0.99999, f"the synthetic world needs the root at the origin ({rp}, {rq})"
        self.root_z = float(rp[2])
        self.jidx = self.backend._jidx
        self.trunk_limits = (self.robot.joint_lower_limits[self.robot.trunk_control_idx].clone(), self.robot.joint_upper_limits[self.robot.trunk_control_idx].clone())
        self.dev = self.backend.device_cfg.device
        self.setup_s = time.time() - t0
        self._shoulder: Dict[Tuple[float, float], Tuple[np.ndarray, float]] = {}

    def q_config(self, translate: float, tilt: float, posture: str = "untucked", fingers: float = 0.05) -> th.Tensor:
        from ispatialgym.robot import conventions as C

        q = th.zeros(28)
        q[self.jidx["base_footprint_z_joint"]] = C.NOMINAL_BASE_Z - self.root_z
        q[self.robot.trunk_control_idx] = C.torso_q(translate, tilt, self.trunk_limits)
        for arm in ("left", "right"):
            if posture == "untucked":
                q[self.robot.arm_control_idx[arm]] = self.robot.untucked_default_joint_pos[self.robot.arm_control_idx[arm]].clone()
            elif posture == "ready":
                q[self.robot.arm_control_idx[arm]] = th.tensor(self.M.ARM_READY_Q[arm], dtype=th.float32)
            else:
                q[self.robot.arm_control_idx[arm]] = 0.0
            q[self.robot.gripper_control_idx[arm]] = fingers
        return q

    def entry(self):
        return self.backend._batch_ik_entry(f"{self.cfg.arm}_arm", self.cfg.batch, seeds=self.cfg.seeds)

    def full_q(self, entry, q_active: th.Tensor, q_lock: th.Tensor) -> th.Tensor:
        """(N, 28) configurations (GPU): ``q_lock`` with the IK entry's active joints from ``q_active``."""
        names = list(entry.rsc.joint_names)
        idx = th.tensor([self.jidx[n] for n in names], device=self.dev)
        q = q_lock.to(self.dev).unsqueeze(0).repeat(q_active.shape[0], 1)
        q[:, idx] = q_active.to(self.dev)
        return q

    def _wb(self):
        return self.backend.kin_entry("whole_body")

    def _fk(self, q_full: th.Tensor, chunk: int = 8192):
        wb = self._wb()
        self.backend.refresh_locked_joints("whole_body", q_full[0].cpu())
        for s0 in range(0, q_full.shape[0], chunk):
            js = self.backend._active_js(wb, q_full[s0:s0 + chunk].contiguous())
            with th.no_grad():
                yield wb.fk.compute_kinematics(js)

    def spheres(self, q_full: th.Tensor) -> th.Tensor:
        """(N, S, 4) whole-body spheres (root = world frame; radii with the model's buffer) of full configurations (any batch)."""
        n = q_full.shape[0]
        return th.cat([st.robot_spheres.reshape(-1, st.robot_spheres.shape[-2], 4) for st in self._fk(q_full)]).reshape(n, -1, 4)

    def eef(self, q_full: th.Tensor) -> th.Tensor:
        """(N, 4, 4) eef poses (world) of full configurations."""
        out = []
        for st in self._fk(q_full):
            p = st.tool_poses.get_link_pose(self.M.EEF_LINK[self.cfg.arm])
            out.append(_mat(p.position.reshape(-1, 3), p.quaternion.reshape(-1, 4)))
        return th.cat(out)

    def link_of_sphere(self) -> List[str]:
        return list(self.backend._collision_costs(self._wb())["link_of_sphere"])

    def shoulder(self, translate: float, tilt: float) -> Tuple[np.ndarray, float]:
        """(shoulder xyz, arm reach) of the trunk (``eef_reach_volume``'s geometric bound: link1 position, straight-arm length)."""
        k = (round(translate, 4), round(tilt, 4))
        if k not in self._shoulder:
            arm = self.cfg.arm
            q = self.q_config(translate, tilt)
            sh = self._link_pos(q, f"{arm}_arm_link1")
            q0 = q.clone()
            q0[self.robot.arm_control_idx[arm]] = 0.0
            L = float(np.linalg.norm(self._link_pos(q0, self.M.EEF_LINK[arm]) - self._link_pos(q0, f"{arm}_arm_link1")))
            self._shoulder[k] = (sh, L)
        return self._shoulder[k]

    def _link_pos(self, q: th.Tensor, link: str) -> np.ndarray:
        self.robot.set_joint_positions(q)
        self.robot.keep_still()
        self.og.sim.step_physics()
        self.robot.set_joint_positions(q)
        lk = self.robot.eef_links[self.cfg.arm] if link == self.M.EEF_LINK[self.cfg.arm] else self.robot.links[link]
        return lk.get_position_orientation()[0].numpy().astype(float)


def _mat(pos: th.Tensor, quat_wxyz: th.Tensor) -> th.Tensor:
    w, x, y, z = quat_wxyz.unbind(-1)
    R = th.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                  2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                  2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)
    T = th.eye(4, device=pos.device).repeat(pos.shape[0], 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = pos
    return T


def _quat_wxyz(R: th.Tensor) -> th.Tensor:
    """Rotation matrices (N, 3, 3) -> unit quaternions (N, 4) wxyz (robust branch per row)."""
    m = R
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = th.zeros(R.shape[0], 4, device=R.device)
    c0 = t > 0
    s = th.sqrt(th.clamp(t + 1.0, min=1e-12)) * 2
    q[c0] = th.stack([0.25 * s, (m[:, 2, 1] - m[:, 1, 2]) / s, (m[:, 0, 2] - m[:, 2, 0]) / s, (m[:, 1, 0] - m[:, 0, 1]) / s], 1)[c0]
    c1 = (~c0) & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    s = th.sqrt(th.clamp(1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], min=1e-12)) * 2
    q[c1] = th.stack([(m[:, 2, 1] - m[:, 1, 2]) / s, 0.25 * s, (m[:, 0, 1] + m[:, 1, 0]) / s, (m[:, 0, 2] + m[:, 2, 0]) / s], 1)[c1]
    c2 = (~c0) & (~c1) & (m[:, 1, 1] > m[:, 2, 2])
    s = th.sqrt(th.clamp(1.0 + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2], min=1e-12)) * 2
    q[c2] = th.stack([(m[:, 0, 2] - m[:, 2, 0]) / s, (m[:, 0, 1] + m[:, 1, 0]) / s, 0.25 * s, (m[:, 1, 2] + m[:, 2, 1]) / s], 1)[c2]
    c3 = (~c0) & (~c1) & (~c2)
    s = th.sqrt(th.clamp(1.0 + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1], min=1e-12)) * 2
    q[c3] = th.stack([(m[:, 1, 0] - m[:, 0, 1]) / s, (m[:, 0, 2] + m[:, 2, 0]) / s, (m[:, 1, 2] + m[:, 2, 1]) / s, 0.25 * s], 1)[c3]
    return q / th.linalg.norm(q, dim=1, keepdim=True)


def _solve(world: SynthWorld, entry, T: th.Tensor, q_lock: th.Tensor, seed_active: Optional[th.Tensor] = None):
    """Batched IK of eef poses ``T`` (N, 4, 4; world = root frame) with the locked joints of ``q_lock``: ``(success (N, S), q (N, S, dof))``
    - every seed's solution (the caller keeps the first that passes its own checks). ``seed_active`` (N, dof): one seed per problem (the
    pre-grasp solution of a grasp), the rest of the ``seeds`` random. Poses go to cuRobo as tensors (no per-pose Python), in chunks of the
    entry's batch size, the last one padded."""
    be = world.backend
    ik = entry.rsc
    B = world.cfg.batch
    be._refresh_locked_entry(entry, q_lock)
    frames = list(ik.kinematics.tool_frames)
    dof = len(ik.joint_names)
    n = T.shape[0]
    S_ = world.cfg.seeds
    succ = th.zeros(n, S_, dtype=th.bool, device=world.dev)
    qs = th.zeros(n, S_, dof, device=world.dev)
    cur_full = be._js_full(q_lock.unsqueeze(0).repeat(B, 1))
    cur = entry.fk.get_active_js(cur_full)
    from ispatialgym.oracle.curobo_planner import CuPose, GoalToolPose

    for s0 in range(0, n, B):
        Tc = T[s0:s0 + B].to(world.dev)
        b = Tc.shape[0]
        if b < B:
            Tc = th.cat([Tc, Tc[-1:].repeat(B - b, 1, 1)])
        pose = CuPose(position=Tc[:, :3, 3].contiguous(), quaternion=_quat_wxyz(Tc[:, :3, :3]).contiguous())
        goal = GoalToolPose.from_poses({frames[0]: pose}, ordered_tool_frames=frames, num_goalset=1)
        if seed_active is not None:
            sd = seed_active[s0:s0 + B].to(world.dev)
            if sd.shape[0] < B:
                sd = th.cat([sd, sd[-1:].repeat(B - sd.shape[0], 1)])
            seed = sd.view(B, 1, dof).clone()
        else:
            seed = cur.position.view(B, 1, -1).clone()
        res = ik.solve_pose(goal, current_state=cur, seed_config=seed, return_seeds=S_)
        succ[s0:s0 + b] = res.success.reshape(B, -1)[:b, :S_]
        qs[s0:s0 + b] = res.solution.reshape(B, -1, dof)[:b, :S_]
    return succ, qs


# ------------------------------------------------------------------------------------------------ the computation


def slab(h: float, d_e: float, cfg: MapConfig) -> CP.Support:
    return CP.Support.box([d_e + cfg.slab_depth_m / 2, 0.0, h - cfg.slab_thickness_m / 2],
                          [cfg.slab_depth_m / 2, cfg.slab_long_m / 2, cfg.slab_thickness_m / 2], name="synthetic_slab")


def _yaw_R(yaws: Sequence[float]) -> np.ndarray:
    return np.stack([np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]]) for a in yaws])


def compute_map(world: SynthWorld, lib: S.ModelLibrary, rest_id: str, height: float, anchor: Dict[str, Any], geom=None,
                cfg: Optional[MapConfig] = None, policy: CP.ContactPolicy = CP.DEFAULT, kind: str = "tabletop") -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """One map: (meta, arrays). ``geom`` (the object's collision pieces, :mod:`object_geometry`) for the arm-vs-object check.
    ``kind="brace"``: the contact classes of :func:`brace_classes` instead of the library grasps (closed fingers, the pads on the
    object's faces; single-arm proposals only - simultaneous brace / manipulation and the arms' mutual collisions are checked later)."""
    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling.reach_bands import trunk_grid

    cfg = cfg or world.cfg
    t_all = time.time()
    tim: Dict[str, float] = {}
    dev = world.dev
    rest = lib.rests[rest_id]
    grasps = grasp_set(lib, rest_id, cfg) if kind == "tabletop" else brace_classes()
    if kind == "brace" and geom is None:
        raise ValueError("a brace map needs the object's collision geometry")
    cells = lattice(cfg)
    yaws = [2 * math.pi * k / cfg.n_yaws for k in range(cfg.n_yaws)]
    nC, nY, nG = len(cells), len(yaws), len(grasps)
    trunks = trunk_grid(height)
    arrays: Dict[str, np.ndarray] = {"cells": cells.astype(np.float32), "yaws": np.array(yaws, dtype=np.float32)}
    meta_trunks: List[Dict[str, Any]] = []
    if nG == 0:
        raise ValueError(f"{lib.path}: no validated grasps on rest {rest_id}")
    # -- object poses and eef targets (cells x yaws x grasps), world frame
    t0 = time.time()
    Ry = _yaw_R(yaws)
    Rr = S.quat_to_mat(rest.quat_xyzw)
    To = np.zeros((nY, 4, 4))
    To[:, :3, :3] = Ry @ Rr
    To[:, 3, 3] = 1.0
    To[:, 2, 3] = height + rest.h_root
    if kind == "brace":
        TE, contact_pts = brace_targets(np.asarray(geom.union.vertices) @ Rr.T, yaws, height, rest.h_root, grasps)
    else:
        T_og = np.stack([g["T"] for g in grasps])  # (G, 4, 4)
        TE = np.einsum("yij,gjk->ygik", To, T_og)  # object at the origin xy: (Y, G, 4, 4)
    TE_t = th.as_tensor(TE, dtype=th.float32, device=dev).reshape(1, nY, nG, 4, 4).repeat(nC, 1, 1, 1, 1)
    TE_t[..., 0, 3] += th.as_tensor(cells[:, 0], dtype=th.float32, device=dev).view(nC, 1, 1)
    TE_t[..., 1, 3] += th.as_tensor(cells[:, 1], dtype=th.float32, device=dev).view(nC, 1, 1)
    TE_flat = TE_t.reshape(-1, 4, 4)
    TP_flat = TE_flat.clone()
    TP_flat[:, :3, 3] -= C.PRE_DIST * TP_flat[:, :3, 2]
    N = TE_flat.shape[0]
    # footprint of the rest at each yaw: the object's xy reach toward the robot (support fit at d_e)
    if geom is not None:
        V = np.asarray(geom.union.vertices) @ Rr.T
        reach_back = np.array([float((V @ R.T)[:, 0].min()) for R in Ry])  # most negative x of the footprint at each yaw
    else:
        reach_back = np.zeros(nY)
    tim["targets_s"] = round(time.time() - t0, 3)
    # -- the hand vs the slab per d_e (IK-independent): pre-grasp, approach, grasp, closing sweep
    t0 = time.time()
    hc = CP.HandChecker(cfg.arm, policy, device=str(dev), spacing=cfg.hand_spacing_m)
    hand_ok: Dict[float, th.Tensor] = {}
    pre_q = float(max(g["preshape_q"] for g in grasps))
    path = CP.approach_path(TP_flat, TE_flat, policy.approach_step_m * 2)
    for d_e in cfg.edge_distances:
        sb = slab(height, d_e, cfg)
        ok = th.ones(N, dtype=th.bool, device=dev)
        for k in range(path.shape[0]):
            o, _, _ = hc.check(path[k], sb, (pre_q, pre_q))
            ok &= o
        for j in range(1, 3):
            q = pre_q + (min(g["contact_q"] for g in grasps) - pre_q) * j / 2
            o, _, _ = hc.check(TE_flat, sb, (q, q))
            ok &= o
        hand_ok[d_e] = ok
    del path
    if th.cuda.is_available():
        th.cuda.empty_cache()
    tim["hand_policy_s"] = round(time.time() - t0, 3)
    # off-support per (d_e, cell, yaw)
    onsup = {}
    for d_e in cfg.edge_distances:
        m = (cells[:, None, 0] + reach_back[None, :]) >= d_e  # (C, Y)
        onsup[d_e] = th.as_tensor(np.repeat(m[:, :, None], nG, axis=2).reshape(-1), device=dev)
    # -- arm-vs-object planes (object at each cell / yaw): world-frame planes per (cell, yaw) are built on the fly
    pieces = []
    if geom is not None:
        for pc in geom.pieces:
            pieces.append((th.as_tensor(pc.face_normals, dtype=th.float32, device=dev), th.as_tensor(np.einsum("ij,ij->i", pc.face_normals, pc.triangles[:, 0]), dtype=th.float32, device=dev)))
    hand_links = set(policy.finger_links(cfg.arm)) | set(policy.hand_rigid_links(cfg.arm))
    entry = world.entry()
    links = world.link_of_sphere()
    arm_mask = th.tensor([(l not in hand_links) and bool(l) and (l.startswith(f"{cfg.arm}_arm")) for l in links], device=dev)
    # object frame of each problem (for the arm-vs-object test)
    To_t = th.as_tensor(To, dtype=th.float32, device=dev)
    obj_idx_y = th.arange(nY, device=dev).view(1, nY, 1).repeat(nC, 1, nG).reshape(-1)
    obj_xy = th.as_tensor(cells, dtype=th.float32, device=dev).view(nC, 1, 1, 2).repeat(1, nY, nG, 1).reshape(-1, 2)
    t_ik = t_chk = 0.0
    n_ik = 0
    for tr, ti in trunks:
        key = f"t{tr:.2f}_k{ti:+.1f}"
        rec: Dict[str, Any] = {"translate": tr, "tilt": ti, "key": key}
        status = {d_e: th.zeros(N, dtype=th.uint8, device=dev) for d_e in cfg.edge_distances}
        # eligibility: the first idle-arm posture free of self-collision and of the slab at every d_e (support factor, shared by objects)
        posture, q_lock = None, None
        for p in cfg.postures:
            q = world.q_config(tr, ti, p)
            free = not bool(world.backend.check_collision(q.unsqueeze(0), emb_name="whole_body", margin=0.0)[0])
            if not free:
                continue
            sph, ln = world.backend.robot_spheres_world(q.unsqueeze(0))
            slab_free = [bool(CP.sphere_support_ok(sph, ln, slab(height, d_e, cfg), cfg.arm, policy)[0][0]) for d_e in cfg.edge_distances]
            if all(slab_free):
                posture, q_lock = p, q
                break
            rec.setdefault("slab_collide_by_posture", {})[p] = slab_free
        rec["posture"] = posture
        if q_lock is None:
            rec["eligible"] = False
            rec["reason"] = "every idle-arm posture collides (self or the slab at some d_e)"
            for d_e in cfg.edge_distances:
                status[d_e].fill_(STATUS["trunk_ineligible"])
                arrays[f"s{d_e:.2f}_{key}"] = status[d_e].view(nC, nY, nG).cpu().numpy()
            meta_trunks.append(rec)
            continue
        rec["eligible"] = True
        sh, L = world.shoulder(tr, ti)
        sh_t = th.as_tensor(sh, dtype=th.float32, device=dev)
        reach = L + cfg.reach_margin_m
        in_reach = (th.linalg.norm(TE_flat[:, :3, 3] - sh_t, dim=1) <= reach) & (th.linalg.norm(TP_flat[:, :3, 3] - sh_t, dim=1) <= reach)
        rec.update({"shoulder": [round(float(v), 4) for v in sh], "arm_len": round(L, 4)})
        any_hand = th.zeros(N, dtype=th.bool, device=dev)
        for d_e in cfg.edge_distances:
            any_hand |= hand_ok[d_e] & onsup[d_e]
        cand = in_reach & any_hand
        idx = th.nonzero(cand).flatten()
        rec["n_problems"] = int(idx.numel())
        # accelerated mode: one round of the robot-facing yaws first; cells with no pre-grasp in it skip the rest
        if cfg.mode == "accelerated" and idx.numel() > 0:
            yaw_of = obj_idx_y[idx]
            first = (yaw_of == (nY // 2)) | (yaw_of == (nY // 2 - 1)) | (yaw_of == (nY // 2 + 1) % nY)
            order = th.cat([idx[first], idx[~first]])
        else:
            order = idx
        ta = time.time()
        succ_pre, q_pre = _solve(world, entry, TP_flat[order], q_lock)
        n_ik += int(order.numel())
        t_ik += time.time() - ta
        pre_any = succ_pre.any(dim=1)
        if cfg.mode == "accelerated" and order.numel() > 0:
            cell_of = order // (nY * nG)
            first_n = int(first.sum())
            hit = th.zeros(nC, dtype=th.bool, device=dev)
            hit.index_put_((cell_of[:first_n][pre_any[:first_n]],), th.ones(int(pre_any[:first_n].sum()), dtype=th.bool, device=dev))
            skip = (~hit[cell_of]) & (th.arange(order.numel(), device=dev) >= first_n)
        else:
            skip = th.zeros(order.numel(), dtype=th.bool, device=dev)
        # the grasp IK seeded from the first successful pre-grasp seed of each problem
        first_seed = th.argmax(succ_pre.int(), dim=1)
        q_pre1 = q_pre[th.arange(order.numel(), device=dev), first_seed]
        sel = th.nonzero(pre_any & ~skip).flatten()
        ta = time.time()
        succ_g, q_g = _solve(world, entry, TE_flat[order[sel]], q_lock, seed_active=q_pre1[sel]) if sel.numel() else (th.zeros(0, cfg.seeds, dtype=th.bool, device=dev), None)
        n_ik += int(sel.numel())
        t_ik += time.time() - ta
        tb = time.time()
        # per d_e post checks over the seeds of each grasp solution (first passing seed counts)
        feas = {d_e: th.zeros(sel.numel(), dtype=th.bool, device=dev) for d_e in cfg.edge_distances}
        why = {d_e: th.full((sel.numel(),), STATUS["ik_grasp"], dtype=th.uint8, device=dev) for d_e in cfg.edge_distances}
        if sel.numel():
            qp = q_pre1[sel]
            for s in range(cfg.seeds):
                ok_s = succ_g[:, s]
                if not bool(ok_s.any()):
                    continue
                qg = q_g[:, s]
                jump = (qg - qp).abs().max(dim=1).values > cfg.branch_max_rad
                cand_s = ok_s & ~jump
                for d_e in cfg.edge_distances:
                    cur = cand_s & ~feas[d_e]
                    why[d_e] = th.where(ok_s & jump & ~feas[d_e], th.full_like(why[d_e], STATUS["branch_jump"]), why[d_e])
                    if not bool(cur.any()):
                        continue
                    ii = th.nonzero(cur).flatten()
                    good = th.ones(ii.numel(), dtype=th.bool, device=dev)
                    sb = slab(height, d_e, cfg)
                    for t in [0.0] + [k / (cfg.approach_configs + 1) for k in range(1, cfg.approach_configs + 1)] + [1.0]:
                        for c0 in range(0, ii.numel(), 8192):
                            iic = ii[c0:c0 + 8192]
                            qq = qp[iic] * (1 - t) + qg[iic] * t
                            sp = world.spheres(world.full_q(entry, qq, q_lock))
                            o, _ = CP.sphere_support_ok(sp, links, sb, cfg.arm, policy)
                            good[c0:c0 + 8192] &= o.to(dev)
                    w_slab = ~good
                    # the hand hulls at the FK eef of the grasp and the pre-grasp (IK tolerance: 5 mm / 0.05 rad)
                    for qq, fq in ((qg[ii], 0.5 * (pre_q + min(g["contact_q"] for g in grasps))), (qp[ii], pre_q)):
                        o, _, _ = hc.check(world.eef(world.full_q(entry, qq, q_lock)), sb, (fq, fq))
                        good &= o
                    w_hand = ~good & ~w_slab
                    # the arm (not the hand) clear of the object at the cell
                    if pieces:
                        # chunked: (problems x arm spheres x faces of a piece) does not fit a 24 GB card for large grasp sets in one go
                        o = th.ones(ii.numel(), dtype=th.bool, device=dev)
                        for c0 in range(0, ii.numel(), 2048):
                            iic = ii[c0:c0 + 2048]
                            sp = world.spheres(world.full_q(entry, qg[iic], q_lock))[:, arm_mask]
                            oi = order[sel[iic]]
                            Ro = To_t[obj_idx_y[oi], :3, :3]
                            to = th.cat([obj_xy[oi], To_t[obj_idx_y[oi], 2, 3:4]], 1)
                            Po = th.einsum("nji,nsj->nsi", Ro, sp[..., :3] - to[:, None])
                            deep = th.full(Po.shape[:2], float("inf"), device=dev)
                            for nrm, dd in pieces:
                                deep = th.minimum(deep, (Po @ nrm.T - dd).max(dim=-1).values)
                            o[c0:c0 + 2048] = (((deep - sp[..., 3]) >= 0.0) | (sp[..., 3] <= 0)).all(dim=1)
                        w_obj = good & ~o
                        good &= o
                    else:
                        w_obj = th.zeros_like(good)
                    feas[d_e][ii] |= good
                    w = why[d_e][ii]
                    w = th.where(w_slab, th.full_like(w, STATUS["slab_spheres"]), w)
                    w = th.where(w_hand, th.full_like(w, STATUS["hand_policy"]), w)
                    w = th.where(w_obj, th.full_like(w, STATUS["object_arm"]), w)
                    why[d_e][ii] = w
        t_chk += time.time() - tb
        for d_e in cfg.edge_distances:
            st = status[d_e]
            st.fill_(STATUS["unevaluated"])
            st[~in_reach] = STATUS["reach_bound"]
            st[in_reach & ~hand_ok[d_e]] = STATUS["hand_policy"]
            st[~onsup[d_e]] = STATUS["off_support"]
            ev = order
            vals = th.full((ev.numel(),), STATUS["ik_pre"], dtype=th.uint8, device=dev)
            vals[skip] = STATUS["skipped"]
            if sel.numel():
                v2 = th.where(feas[d_e], th.full_like(why[d_e], STATUS["feasible"]), why[d_e])
                vals[sel] = v2
            ok_here = hand_ok[d_e][ev] & onsup[d_e][ev]
            st[ev] = th.where(ok_here, vals, st[ev])
            arrays[f"s{d_e:.2f}_{key}"] = st.view(nC, nY, nG).cpu().numpy()
            rec.setdefault("n_feasible", {})[f"{d_e:.2f}"] = int((st == STATUS["feasible"]).sum())
            rec.setdefault("n_skipped", {})[f"{d_e:.2f}"] = int((st == STATUS["skipped"]).sum())
        meta_trunks.append(rec)
    tim.update({"ik_s": round(t_ik, 2), "checks_s": round(t_chk, 2), "ik_problems": n_ik, "total_s": round(time.time() - t_all, 2)})
    meta = {"schema": SCHEMA, "kind": kind, "arm": cfg.arm, "frame": "base_link on the floor: +x ahead, +y left, z up from the floor",
            "object": lib.identity(), "object_key": lib.identity_key(), "rest_id": rest_id,
            "height": {"value": height, "anchor_key": anchor.get("key"), "anchor_surfaces": anchor.get("surfaces", [])[:50]},
            "edge_distances": list(cfg.edge_distances), "config": cfg.key(), "policy_key": policy.key(cfg.arm), "policy": policy.meta(cfg.arm),
            "slab": {"depth_m": cfg.slab_depth_m, "long_m": cfg.slab_long_m, "thickness_m": cfg.slab_thickness_m},
            "ik": {"seeds": cfg.seeds, "position_tolerance": world.backend.position_tolerance, "orientation_tolerance": world.backend.orientation_tolerance,
                   "collision_world": "floor cuboid + self-collision (slab and object checked explicitly)", "pre_dist": C.PRE_DIST},
            "approach_check": f"hand hulls at the pre-grasp, every {policy.approach_step_m * 2:.3f} m of the straight approach and 2 sweep openings; "
                              f"whole-robot spheres at {cfg.approach_configs} joint-interpolated configurations",
            "screening": cfg.mode, "exhaustive": cfg.mode == "full",
            "grasps": ([{k: v for k, v in g.items() if k != "T"} | {"T_object_eef": S.mat_pose(g["T"])} for g in grasps] if kind == "tabletop" else
                       [dict(g, contact_point_world_yaw0=[round(float(v), 5) for v in contact_pts[0, i]],
                             contact_to_eef_yaw0=S.mat_pose(np.linalg.inv(S.pose_mat(contact_pts[0, i], [0, 0, 0, 1.0])) @ TE[0, i])) for i, g in enumerate(grasps)]),
            "brace": ({"fractions": list(BRACE_FRACTIONS), "pitches_deg": list(BRACE_PITCHES_DEG), "sides": list(BRACE_SIDES), "fingers": "closed",
                       "frames": "approach / closing / contact point in the map frame (base on the floor); fingertip_eef: fingertip plane 5 mm before the point",
                       "note": "single-arm feasibility proposal; simultaneous brace + manipulation and inter-arm collisions are validated later"}
                      if kind == "brace" else None),
            "trunks": meta_trunks, "status_codes": STATUS, "robot": robot_digest(), "timing": tim, "created": S.now()}
    meta["key"] = map_key(meta)
    return meta, arrays


def robot_digest() -> str:
    from ispatialgym.oracle import r1pro_model as M

    h = hashlib.sha1()
    for p in (M.source_urdf_path(), M.source_curobo_yaml_path()):
        with open(p, "rb") as f:
            h.update(f.read())
    h.update(json.dumps(M.EXTRA_SELF_COLLISION_IGNORE_PAIRS).encode())
    return h.hexdigest()[:12]


def map_key(meta: Dict[str, Any]) -> str:
    ident = {k: meta[k] for k in ("schema", "kind", "arm", "object_key", "rest_id", "edge_distances", "config", "policy_key", "robot")}
    ident["brace"] = meta.get("brace")
    ident["height"] = meta["height"]["value"]
    return hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]


def save(out_dir: str, meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"map__{meta['object']['category']}__{meta['object']['model']}__h{meta['height']['value']:.4f}__{meta['key']}")
    np.savez_compressed(base + ".npz", **arrays)
    S.save_json(base + ".json", meta)
    return base + ".json"


# ------------------------------------------------------------------------------------------------ reading maps


class ReachMap:
    """One map (``.json`` + ``.npz``). Status queries per (edge distance, trunk, cell); cell weights for sampling."""

    def __init__(self, path_json: str):
        self.path = path_json
        with open(path_json) as f:
            self.meta = json.load(f)
        if self.meta.get("schema") != SCHEMA:
            raise S.IncompatibleRecord(f"{path_json}: schema {self.meta.get('schema')!r} is not {SCHEMA}")
        z = np.load(path_json[:-5] + ".npz")
        self.arrays = {k: z[k] for k in z.files}
        self.cells = self.arrays["cells"]
        self.yaws = self.arrays["yaws"]

    @staticmethod
    def find(maps_root: str, lib: S.ModelLibrary, height: float, *, rest_id: str, arm: str = "right", policy_key: Optional[str] = None,
             anchors=None, kind: str = "tabletop") -> Optional["ReachMap"]:
        """The map of this library entry / rest / arm at the anchor that covers ``height`` (``height_anchors.AnchorSet.lookup``: exact,
        or within the anchor's clipped range); None when uncovered or when no compatible map exists - never a band, never a map of other
        metadata (object identity, rest, policy, schema are all checked)."""
        target = height
        if anchors is not None:
            a, kind = anchors.lookup(height)
            if a is None:
                return None
            target = a["height"]
        want = lib.identity_key()
        for fn in sorted(os.listdir(maps_root)) if os.path.isdir(maps_root) else []:
            if not fn.endswith(".json") or not fn.startswith("map__"):
                continue
            p = os.path.join(maps_root, fn)
            try:
                with open(p) as f:
                    m = json.load(f)
            except Exception:  # noqa: BLE001
                continue
            if m.get("schema") != SCHEMA or m.get("kind", "tabletop") != kind or m.get("object_key") != want or m.get("rest_id") != rest_id or m.get("arm") != arm:
                continue
            if policy_key is not None and m.get("policy_key") != policy_key:
                continue
            if abs(float(m["height"]["value"]) - target) > 1e-6:
                continue
            return ReachMap(p)
        return None

    def trunks(self) -> List[Tuple[float, float]]:
        return [(t["translate"], t["tilt"]) for t in self.meta["trunks"] if t.get("eligible")]

    def status(self, edge_distance: float, translate: float, tilt: float) -> np.ndarray:
        d = min(self.meta["edge_distances"], key=lambda v: abs(v - edge_distance))
        if abs(d - edge_distance) > 1e-6:
            raise KeyError(f"edge distance {edge_distance} is not in the map ({self.meta['edge_distances']})")
        return self.arrays[f"s{d:.2f}_t{translate:.2f}_k{tilt:+.1f}"]

    def feasible_cells(self, edge_distance: float, translate: float, tilt: float) -> np.ndarray:
        st = self.status(edge_distance, translate, tilt)
        return self.cells[(st == STATUS["feasible"]).any(axis=(1, 2))]

    def pairs(self, edge_distance: float, translate: float, tilt: float, cell_index: int) -> List[Tuple[int, int]]:
        st = self.status(edge_distance, translate, tilt)[cell_index]
        return [tuple(map(int, v)) for v in np.argwhere(st == STATUS["feasible"])]

    def cell_weights(self, edge_distance: float, translate: float, tilt: float) -> np.ndarray:
        """Plausibility per cell, normalised for discretisation: a feasible (yaw, grasp) adds 1 / n_yaws of the yaw axis times the grasp's
        share of its family (a family sampled at k points contributes its measure once, whatever k), twins as their own wrist choice."""
        st = self.status(edge_distance, translate, tilt) == STATUS["feasible"]
        g = self.meta["grasps"]
        fam_n: Dict[Tuple[str, bool], int] = {}
        for x in g:
            k = (x["base"], x["twin"])
            fam_n[k] = fam_n.get(k, 0) + 1
        w_g = np.array([1.0 / fam_n[(x["base"], x["twin"])] for x in g])
        return (st * w_g[None, None, :]).sum(axis=(1, 2)) / len(self.yaws)


# ------------------------------------------------------------------------------------------------ articulation opening prefixes (prompt 2 §7)


def compute_prefix_map(world: SynthWorld, alib: Dict[str, Any], height: float, anchor: Dict[str, Any], cfg: Optional[MapConfig] = None,
                       policy: CP.ContactPolicy = CP.DEFAULT, min_open_rad: float = 0.2, subsample: int = 2, step_max_rad: float = 0.5
                       ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Reachable opening prefixes of an articulated tabletop object (``validate_artic``'s ``artic_library.json``: the measured rest and,
    per tested handle, the FIRST fixed-stance pull segment as object-relative eef waypoints with the joint value). Per (edge distance,
    trunk, cell, yaw, segment): the pre-grasp and the waypoints in order by batched IK, each seeded from the previous waypoint's
    solution; a configuration counts only while every joint stays within ``step_max_rad`` of the previous one (no branch jumps between
    disconnected successes) and the whole robot keeps the contact policy against the slab (spheres; the hand hulls at the FK eef).
    Stored: the contiguous prefix (rad of joint travel from the initial state) - ``accepted`` where it reaches ``min_open_rad`` (the full
    validated range is not required). Hybrid demonstrations contribute their first fixed-stance segment only; the achieved range is that
    of the tested grasp / stance / controller, not the object's limit. The moving link's geometry is not posed per waypoint: the arm-vs-
    object check is not run here (the exact sampler checks the articulated target)."""
    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling.reach_bands import trunk_grid

    cfg = cfg or world.cfg
    t_all = time.time()
    dev = world.dev
    rest = S.Rest.from_record(alib["rests"]["r0"])
    cells = lattice(cfg)
    yaws = [2 * math.pi * k / cfg.n_yaws for k in range(cfg.n_yaws)]
    nC, nY = len(cells), len(yaws)
    To = np.stack([S.place_object(rest, y, (0.0, 0.0), height) for y in yaws])  # (Y, 4, 4) at the origin cell
    arrays: Dict[str, np.ndarray] = {"cells": cells.astype(np.float32), "yaws": np.array(yaws, dtype=np.float32)}
    entry = world.entry()
    links = world.link_of_sphere()
    hc = CP.HandChecker(cfg.arm, policy, device=str(dev), spacing=cfg.hand_spacing_m)
    segs_meta = []
    t_ik = 0.0
    n_ik = 0
    for si, seg in enumerate(alib.get("segments") or []):
        wps = seg["waypoints"][::max(1, subsample)]
        if len(wps) < 2:
            continue
        q0 = float(wps[0]["q"])
        travel = np.array([abs(float(w["q"]) - q0) for w in wps])
        W = np.stack([S.pose_of(w["eef_in_object"]) for w in wps])  # (K, 4, 4)
        TW = np.einsum("yij,kjl->ykil", To, W)  # (Y, K, 4, 4)
        T = th.as_tensor(TW, dtype=th.float32, device=dev).unsqueeze(0).repeat(nC, 1, 1, 1, 1)
        T[..., 0, 3] += th.as_tensor(cells[:, 0], dtype=th.float32, device=dev).view(nC, 1, 1)
        T[..., 1, 3] += th.as_tensor(cells[:, 1], dtype=th.float32, device=dev).view(nC, 1, 1)
        T = T.reshape(nC * nY, len(wps), 4, 4)
        segs_meta.append({"index": si, "candidate": seg.get("candidate"), "joint": seg.get("joint"), "n_waypoints": len(wps),
                          "validated_travel_rad": round(float(travel.max()), 4), "trial_stop": seg.get("stop"), "profile": seg.get("profile")})
        for tr, ti in trunk_grid(height):
            key = f"t{tr:.2f}_k{ti:+.1f}"
            q_lock = world.q_config(tr, ti, "untucked")
            if bool(world.backend.check_collision(q_lock.unsqueeze(0), emb_name="whole_body", margin=0.0)[0]):
                for d_e in cfg.edge_distances:
                    arrays[f"s{d_e:.2f}_{key}__seg{si}"] = np.full((nC, nY), -1.0, dtype=np.float32)
                continue
            sh, L = world.shoulder(tr, ti)
            sh_t = th.as_tensor(sh, dtype=th.float32, device=dev)
            in_reach = (th.linalg.norm(T[:, 0, :3, 3] - sh_t, dim=1) <= L + cfg.reach_margin_m)
            prefix = {d_e: th.full((nC * nY,), -1.0, device=dev) for d_e in cfg.edge_distances}
            alive = {d_e: in_reach.clone() for d_e in cfg.edge_distances}
            # the pre-grasp of the first waypoint, then the chain
            Tp = T[:, 0].clone()
            Tp[:, :3, 3] -= C.PRE_DIST * Tp[:, :3, 2]
            idx = th.nonzero(in_reach).flatten()
            if idx.numel() == 0:
                for d_e in cfg.edge_distances:
                    arrays[f"s{d_e:.2f}_{key}__seg{si}"] = prefix[d_e].view(nC, nY).cpu().numpy()
                continue
            ta = time.time()
            sp_ok, qp = _solve(world, entry, Tp[idx], q_lock)
            t_ik += time.time() - ta
            n_ik += int(idx.numel())
            ok_any = sp_ok.any(dim=1)
            prev = th.zeros(nC * nY, qp.shape[-1], device=dev)
            prev[idx] = qp[th.arange(idx.numel(), device=dev), th.argmax(sp_ok.int(), dim=1)]
            for d_e in cfg.edge_distances:
                a = alive[d_e]
                a[idx[~ok_any]] = False
            for k in range(len(wps)):
                live = alive[cfg.edge_distances[0]].clone()
                for d_e in cfg.edge_distances[1:]:
                    live |= alive[d_e]
                ii = th.nonzero(live).flatten()
                if ii.numel() == 0:
                    break
                ta = time.time()
                sk, qk = _solve(world, entry, T[ii, k], q_lock, seed_active=prev[ii])
                t_ik += time.time() - ta
                n_ik += int(ii.numel())
                best = th.argmax(sk.int(), dim=1)
                qsel = qk[th.arange(ii.numel(), device=dev), best]
                okk = sk.any(dim=1) & ((qsel - prev[ii]).abs().max(dim=1).values <= step_max_rad)
                qf = world.full_q(entry, qsel, q_lock)
                sp = world.spheres(qf)
                eef = world.eef(qf)
                for d_e in cfg.edge_distances:
                    sb = slab(height, d_e, cfg)
                    o_s, _ = CP.sphere_support_ok(sp, links, sb, cfg.arm, policy)
                    o_h, _, _ = hc.check(eef, sb, (0.0, 0.0))
                    good = okk & o_s.to(dev) & o_h & alive[d_e][ii]
                    alive[d_e][ii] = good
                    prefix[d_e][ii[good]] = float(travel[k])
                prev[ii] = qsel
            for d_e in cfg.edge_distances:
                arrays[f"s{d_e:.2f}_{key}__seg{si}"] = prefix[d_e].view(nC, nY).cpu().numpy()
    meta = {"schema": SCHEMA, "kind": "articulation_prefix", "arm": cfg.arm, "frame": "base_link on the floor: +x ahead, +y left, z up from the floor",
            "object": {"category": alib.get("category"), "model": alib.get("model"), "scale": alib.get("scale"), "rests": {"r0": {k: alib["rests"]["r0"][k] for k in ("quat_xyzw", "h_root")}}},
            "object_key": S.digest({"artic": alib.get("category"), "m": alib.get("model"), "rest": alib["rests"]["r0"]["quat_xyzw"], "segs": [s_["candidate"] for s_ in segs_meta]}, 16),
            "rest_id": "r0", "height": {"value": height, "anchor_key": anchor.get("key")}, "edge_distances": list(cfg.edge_distances), "config": cfg.key(),
            "policy_key": policy.key(cfg.arm), "segments": segs_meta, "min_open_rad": min_open_rad, "step_max_rad": step_max_rad,
            "value": "contiguous joint travel (rad) from the initial state reachable from the cell / yaw / trunk; -1 = not even the grasp",
            "scope": "the tested grasp / stance / controller's first fixed-stance segment - not the object's global opening limit",
            "robot": robot_digest(), "timing": {"ik_s": round(t_ik, 2), "ik_problems": n_ik, "total_s": round(time.time() - t_all, 2)}, "created": S.now()}
    meta["key"] = S.digest({k: meta[k] for k in ("schema", "kind", "arm", "object_key", "edge_distances", "config", "policy_key", "robot")} | {"h": height}, 16)
    return meta, arrays


# ------------------------------------------------------------------------------------------------ CLI


def _bench(args):
    cfg = MapConfig(batch=args.batch, seeds=args.seeds)
    w = SynthWorld(cfg)
    entry = w.entry()
    q = w.q_config(0.5, 0.0)
    ai = w.robot.arm_control_idx[cfg.arm]
    g = th.Generator().manual_seed(0)
    n = args.n
    Q = q.unsqueeze(0).repeat(n, 1)
    Q[:, ai] += 0.6 * (th.rand(n, len(ai), generator=g) - 0.5)
    T = w.eef(Q.to(w.dev))
    _solve(w, entry, T[: cfg.batch], q)  # warm (graph capture)
    th.cuda.synchronize()
    t0 = time.time()
    succ, qs = _solve(w, entry, T, q)
    th.cuda.synchronize()
    dt = time.time() - t0
    t1 = time.time()
    sp = w.spheres(w.full_q(entry, qs[:, 0], q))
    th.cuda.synchronize()
    print(f"BENCH seeds {cfg.seeds} batch {cfg.batch}: {n} IK problems in {dt:.2f}s = {n / dt:.0f}/s, any-seed success {float(succ.any(1).float().mean()):.3f}; "
          f"sphere FK {n} in {time.time() - t1:.3f}s; setup {w.setup_s:.1f}s; peak {th.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--n", type=int, default=16384)
    b.add_argument("--batch", type=int, default=2048)
    b.add_argument("--seeds", type=int, default=8)
    c = sub.add_parser("compute")
    c.add_argument("--jobs", required=True)
    c.add_argument("--glib", default=None)
    c.add_argument("--anchors", required=True)
    c.add_argument("--heights", nargs="*", default=None, help="anchor keys (default: every anchor)")
    c.add_argument("--heights-by-id", default=None, help="json {job id: [anchor keys]} (overrides --heights per job)")
    c.add_argument("--only", nargs="*", default=None)
    c.add_argument("--shard", default=None, help="i/n")
    c.add_argument("--out", required=True)
    c.add_argument("--mode", default="full", choices=["accelerated", "full"])
    c.add_argument("--lattice", type=float, default=0.08)
    c.add_argument("--seeds", type=int, default=8)
    c.add_argument("--batch", type=int, default=2048)
    c.add_argument("--kind", default="tabletop", choices=["tabletop", "brace"])
    c.add_argument("--skip-existing", action="store_true", help="skip (job, height) pairs whose map file exists already")
    c.add_argument("--include-derived", action="store_true", help="also every library entry of --glib that is not a manifest job (the measured-rest retries)")
    c.add_argument("--arm", default="right", choices=["right", "left"])
    pf = sub.add_parser("prefix")
    pf.add_argument("--artic-lib", nargs="+", required=True, help="artic_library.json files (validate_artic)")
    pf.add_argument("--anchors", required=True)
    pf.add_argument("--heights", nargs="+", required=True)
    pf.add_argument("--out", required=True)
    pf.add_argument("--min-open-rad", type=float, default=0.2)
    i = sub.add_parser("info")
    i.add_argument("map")
    a = ap.parse_args(argv)
    if a.cmd == "bench":
        _bench(a)
    elif a.cmd == "prefix":
        an = json.load(open(a.anchors))
        anchors = {x["key"]: x for x in an["anchors"]}
        w = SynthWorld(MapConfig())
        for p in a.artic_lib:
            alib = json.load(open(p))
            for hk in a.heights:
                try:
                    meta, arrays = compute_prefix_map(w, alib, float(anchors[hk]["height"]), anchors[hk], min_open_rad=a.min_open_rad)
                    base = os.path.join(a.out, f"prefix__{alib['category']}__{alib['model']}__h{float(hk):.4f}__{meta['key']}")
                    os.makedirs(a.out, exist_ok=True)
                    np.savez_compressed(base + ".npz", **arrays)
                    S.save_json(base + ".json", meta)
                    acc = {k: int((v >= a.min_open_rad).sum()) for k, v in arrays.items() if k.startswith("s0.45") and k.endswith("__seg0")}
                    print("GB_PREFIX " + json.dumps({"lib": p, "h": hk, "timing": meta["timing"], "segments": meta["segments"], "accepted_cells_x_yaws@0.45": acc,
                                                     "max_prefix_rad": max(float(v.max()) for k, v in arrays.items() if "__seg" in k)}), flush=True)
                except Exception as e:  # noqa: BLE001
                    import traceback

                    print("GB_PREFIX " + json.dumps({"lib": p, "h": hk, "error": repr(e), "tb": traceback.format_exc()[-1500:]}), flush=True)
    elif a.cmd == "info":
        m = ReachMap(a.map)
        print(json.dumps({k: m.meta[k] for k in ("key", "rest_id", "height", "edge_distances", "screening", "timing")}, indent=1))
        for t in m.meta["trunks"]:
            print(t["key"], t.get("eligible"), t.get("posture"), t.get("n_problems"), t.get("n_feasible"), t.get("n_skipped"))
    else:
        _compute_cli(a)
    import omnigibson as og

    if a.cmd != "info":
        og.shutdown()


def _compute_cli(a):
    from ispatialgym.grasping import object_geometry as OG
    from ispatialgym.sampling.height_anchors import AnchorSet

    with open(a.jobs) as f:
        jobs = json.load(f)
    if a.only:
        jobs = [j for j in jobs if j["id"] in set(a.only) or j.get("legacy_key") in set(a.only)]
    derived = []
    if a.include_derived and a.glib:
        known = {j["id"] for j in jobs}
        for p in sorted(glob.glob(os.path.join(S.library_root(a.glib), "models", "*", "library.json"))):
            eid = os.path.basename(os.path.dirname(p))
            if eid in known:
                continue
            lib = json.load(open(p))
            cand = S.load_json(os.path.join(os.path.dirname(p), "candidates.json")) or {}
            parent = (S.load_json(os.path.join(os.path.dirname(p), "job.json")) or {}).get("derived_from")
            derived.append({"id": eid, "category": lib["category"], "model": lib["model"], "scale": lib.get("scale"), "constraint": lib.get("constraint") or "tabletop",
                            "rest_hint_wxyz": cand.get("rest_hint_wxyz"), "derived": True, "derived_from": parent})
    if a.shard:
        # a derived entry goes to its parent job's shard - the shard whose glib stage made it, so it exists when this stage lists the
        # library; sharding the appended list by index instead raced the other shards' retries (each saw a different list). The parent
        # is job.json's derived_from; entries made before it was written fall back to the first job with the same id minus the rest digest
        i, n = (int(v) for v in a.shard.split("/"))
        base = lambda eid: re.sub(r"__r[0-9a-f]{6}$", "", eid)  # noqa: E731 - the entry without its designated-rest digest
        index, first = {}, {}
        for k, j in enumerate(jobs):
            index[j["id"]] = k
            first.setdefault(base(j["id"]), k)
        home = lambda d: index[d["derived_from"]] if d.get("derived_from") in index else first.get(base(d["id"]), 0)  # noqa: E731
        jobs = [j for k, j in enumerate(jobs) if k % n == i] + [d for d in derived if home(d) % n == i]
    else:
        jobs += derived
    with open(a.anchors) as f:
        an = json.load(f)
    aset = AnchorSet(an["anchors"], an.get("max_approx_m", 0.01))
    want = set(a.heights) if a.heights else None
    anchors = [x for x in aset.anchors if want is None or x["key"] in want]
    cfg = MapConfig(mode=a.mode, lattice_m=a.lattice, seeds=a.seeds, batch=a.batch, arm=a.arm, twins=a.kind == "tabletop")
    by_id = json.load(open(a.heights_by_id)) if a.heights_by_id else {}
    w = SynthWorld(cfg)
    for j in jobs:
        anchors_j = [x for x in aset.anchors if x["key"] in set(by_id[j["id"]])] if j["id"] in by_id else anchors
        lib = S.load_model(j["category"], j["model"], j.get("scale"), a.glib, constraint=j.get("constraint") or "tabletop", rest_hint_wxyz=j.get("rest_hint_wxyz"))
        if lib is None or not lib.grasps():
            print(f"GB_MAP {json.dumps({'id': j['id'], 'ok': False, 'reason': 'no v2 library entry with validated grasps'})}", flush=True)
            continue
        geom = OG.load(os.path.dirname(lib.path))
        for rid, r in lib.rests.items():
            if not r.stable or not lib.grasps(rid):
                continue
            for anc in anchors_j:
                if a.skip_existing and glob.glob(os.path.join(a.out, j["id"], f"map__*__h{float(anc['height']):.4f}__*.json")):
                    continue
                try:
                    meta, arrays = compute_map(w, lib, rid, float(anc["height"]), anc, geom=geom, cfg=cfg, kind=a.kind)
                    p = save(os.path.join(a.out, j["id"]), meta, arrays)
                    nf = {t["key"]: t.get("n_feasible") for t in meta["trunks"] if t.get("eligible")}
                    print(f"GB_MAP {json.dumps({'id': j['id'], 'ok': True, 'rest': rid, 'h': anc['key'], 'path': p, 'timing': meta['timing'], 'n_grasps': len(meta['grasps']), 'feasible': nf})}", flush=True)
                except Exception as e:  # noqa: BLE001
                    import traceback

                    print(f"GB_MAP {json.dumps({'id': j['id'], 'ok': False, 'rest': rid, 'h': anc['key'], 'error': repr(e), 'tb': traceback.format_exc()[-1200:]})}", flush=True)
    print("GB_MAP_DONE", flush=True)


if __name__ == "__main__":
    main()
