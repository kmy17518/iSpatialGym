"""Supported physics validation of tabletop / attachment grasp candidates in one persistent world (no scene, no reload per model).

The world is the demo collection's (``run_isg``): OmniGibson at 30 Hz control / 120 Hz physics on CPU PhysX, the JoyLo R1Pro
(``r1pro_model.joylo_robot_config``: assisted grasping, smooth multi-finger gripper control) with the evaluator's 250 kg base
(``apply_eval_base_mass``) and the task's assisted-grasp rules (``apply_environment_rules``: FixedJoint on moving links, pad-clear
release 0.2 s), the rigid sleep threshold 0.005 on the object (``sleep_assist.apply_to_objects``) and the sleep assist 30,1,3 in the
loop (``SleepAssist``). On the floor stands one fixed slab - a local tabletop under the object, ``slab_half_m`` around it (the
support under the object and the table everywhere the hand, wrist camera, palm and the approach / closing envelope can reach), its
near edge at least ``slab_edge_clear_m`` from the robot base like a table edge at the sampler's stand-offs - with full contact
response against every robot link (nothing is filtered, nothing is disabled).

Per model (``run_model``):

1. the object is imported (``scene.add_object``; the persistent world keeps the robot, the slab and the cuRobo backend) and its
   collision geometry exported when the library has none;
2. **rest**: the robot parked 2.5 m away, the object dropped ``drop_m`` above the slab in its designated rest (canonical or the key's
   rest hint) at ``reference_yaw``, settled until its pose moved less than ``settle_pos_m`` / ``settle_rot_rad`` over
   ``settle_window`` control steps, it is slow and it presses on the slab (measured normal force) - not a fixed step count. The measured
   rest (support-frame quaternion, ``h_root``, lowest collision point gap, velocities, drift, contact force) is saved BEFORE any gripper
   contact; a rest materially different from the intended one (tilt > ``rest_tilt_tol_rad``) is saved as its own rest_id;
3. **placement without approach**: for every family representative (one per symmetry family) the eef target
   ``T_world_object(rest) @ T_object_eef`` is solved by batched cuRobo IK over base yaws x base offsets x trunks (the base, the trunk,
   the right arm - the whole robot, not a floating gripper); solutions are kept only when the contact policy holds for the WHOLE
   robot at that configuration (hand hulls vs the slab with the 2 mm finger allowance, every non-finger sphere clear of the slab, arm
   spheres clear of the object; ``contact_policy``) and the most comfortable one (closest to the untucked arm) is used;
4. **trial** (state restored from the rest snapshot, verified): the robot is set at that configuration with the fingers at the oracle's
   pre-shape (pinch width / 2 + 10 mm), two hold steps (drive targets are not part of the state), then the oracle's own closure
   (``Motions.gripper`` with ``execute.closure_for_width``: ramp 2.25 mm / step, contact hold 1 mm inside the touch, thin pinches
   <= 6 mm at 0.5 mm / step without squeeze, the assisted grasp's 0.3 s window, the full-close fallback), ``hold_steps`` of hold, then
   the support is unloaded (the slab lowered ``unload_drop_m``: the object hangs on the grasp) for ``unload_steps``;
5. **acceptance**: AG formed on the object and still holds after the unload with the object within ``retain_*`` of its in-hand pose;
   no non-finger robot link touched the slab (``non_finger_contact_N``) - an AG success does not override it; the fingers' measured
   force on the slab never exceeded ``finger_force_max_N`` for ``force_window`` consecutive control steps; the object's load on the
   slab never exceeded ``object_load_ratio_max`` x its weight for ``force_window`` steps (the hand pushing the object into the table,
   which the finger contacts alone miss); the achieved grasp stayed within ``on_target_*`` of the candidate (the historical knob vs rim
   miss). Every force series is summarised in the trial (units: N; ratio: dimensionless, per control step = max over its 4 physics
   steps). The thresholds are PROVISIONAL (not calibrated; the recorded distributions are for calibrating them).

Fidelity notes: the approach is not executed (the hand starts at the grasp pose with the pre-shaped fingers, as the oracle's approach
ends); a policy-passing start can still carry <= 2 mm finger/table overlap that PhysX resolves in the first step. The slab is a box
with the primitive's default physics material, not the scene table's; its edge sits at the configuration's own distance.

    python -m ispatialgym.grasping.pipeline validate --jobs <manifest.json> [--shard i/n] [--only ID ...]
"""

from __future__ import annotations

import math
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ispatialgym.grasping import contact_policy as CP
from ispatialgym.grasping import object_geometry as OG
from ispatialgym.grasping import schema as S

VALIDATOR_VERSION = "isg-tabletop-validator/1"


@dataclass
class ValidateConfig:
    arm: str = "right"
    slab_top_z: float = 0.75
    slab_thickness_m: float = 0.05
    slab_half_m: float = 0.20          # beyond the object's footprint radius
    slab_edge_clear_m: float = 0.30    # base origin to the slab's near edge, at least
    drop_m: float = 0.003              # sampler Z_OFFSET / PLACE_DROP: the physical initialization drop
    reference_yaw: float = 0.0
    settle_window: int = 30
    settle_max_steps: int = 300
    settle_pos_m: float = 0.001
    settle_rot_rad: float = 0.003
    settle_speed_m_s: float = 0.005
    settle_ang_rad_s: float = 0.05
    rest_tilt_tol_rad: float = 0.05
    restore_pos_m: float = 0.0005
    restore_rot_rad: float = 0.002
    trunks: Tuple[Tuple[float, float], ...] = ((0.5, 0.0), (0.25, 0.0), (0.75, -0.3), (0.5, -0.3))
    base_yaws: int = 8
    #: the object in the base frame: ahead = the slab's half size + slab_edge_clear_m + dx (a table edge at a sampler stand-off), left = dy
    base_offsets: Tuple[Tuple[float, float], ...] = ((0.0, -0.15), (0.0, -0.28), (0.08, -0.05), (0.08, -0.22))
    ik_seeds: int = 12
    init_hold_steps: int = 2
    hold_steps: int = 10
    unload_drop_m: float = 0.10
    unload_steps: int = 30
    retain_pos_m: float = 0.01
    retain_rot_rad: float = 0.10
    on_target_pos_m: float = 0.015
    on_target_rot_rad: float = math.radians(15.0)
    finger_force_max_N: float = 10.0
    object_load_ratio_max: float = 2.0
    force_window: int = 10
    non_finger_contact_N: float = 0.5
    sleep_assist: bool = True
    max_trials: int = 10
    max_valid: int = 6
    max_rest_lift_m: float = 0.003


# ------------------------------------------------------------------------------------------------ force probe


class ForceProbe:
    """Contact forces between the robot links / the object (sensors) and the slab / the object (filters), read on every physics step
    (hooked into ``RigidContactAPI.read_from_physx``, which the simulator calls after each one) and folded per control step into the
    maxima the acceptance rules read. Units: N (``get_contact_force_matrix(dt)``)."""

    def __init__(self, robot, slab, obj, arm: str):
        import omnigibson as og
        from omnigibson.utils.usd_utils import RigidContactAPI

        self.og = og
        self.arm = arm
        fingers = set(f"{arm}_gripper_finger_link{i}" for i in (1, 2))
        self.robot_links = [(n, l.prim_path) for n, l in robot.links.items() if len(l.collision_meshes) > 0]
        self.obj_links = [(n, l.prim_path) for n, l in obj.links.items()]
        self.slab_path = slab.root_link.prim_path
        sensors = [p for _, p in self.robot_links] + [p for _, p in self.obj_links]
        filters = [self.slab_path] + [p for _, p in self.obj_links]
        # the list form takes one filter list per sensor (omni.physics.tensors SimulationView.create_rigid_contact_view overload 2)
        self.view = og.sim.physics_sim_view.create_rigid_contact_view(sensors, filter_patterns=[list(filters) for _ in sensors], max_contact_data_count=4096)
        self.n_robot = len(self.robot_links)
        self.finger_rows = [i for i, (n, _) in enumerate(self.robot_links) if n in fingers]
        self.other_rows = [i for i, (n, _) in enumerate(self.robot_links) if n not in fingers]
        self.obj_rows = list(range(self.n_robot, self.n_robot + len(self.obj_links)))
        self.reset()
        self._api = RigidContactAPI
        self._orig = None

    def reset(self):
        self.acc = {"finger_slab": 0.0, "finger_slab_z": 0.0, "other_slab": 0.0, "other_link": None, "obj_slab_z": 0.0, "finger_obj": 0.0}

    def install(self):
        api = self._api
        if self._orig is not None:
            return
        orig = api.read_from_physx
        probe = self

        def hooked():
            orig()
            probe.read()

        self._orig = orig
        api.read_from_physx = hooked

    def uninstall(self):
        if self._orig is not None:
            self._api.read_from_physx = self._orig
            self._orig = None

    def read(self):
        try:
            F = self.view.get_contact_force_matrix(dt=self.og.sim.get_physics_dt())
        except Exception:  # noqa: BLE001 - invalid view (objects changed): no sample
            return
        F = F.detach().cpu().float()
        a = self.acc
        fs = F[self.finger_rows, 0]  # fingers vs slab
        a["finger_slab"] = max(a["finger_slab"], float(fs.norm(dim=-1).sum()))
        a["finger_slab_z"] = max(a["finger_slab_z"], float(fs[:, 2].abs().sum()))
        if self.other_rows:
            on = F[self.other_rows, 0].norm(dim=-1)
            k = int(on.argmax())
            if float(on[k]) > a["other_slab"]:
                a["other_slab"] = float(on[k])
                a["other_link"] = self.robot_links[self.other_rows[k]][0]
        a["obj_slab_z"] = max(a["obj_slab_z"], float(F[self.obj_rows, 0, 2].sum()))
        if F.shape[1] > 1:
            a["finger_obj"] = max(a["finger_obj"], float(F[self.finger_rows, 1:].norm(dim=-1).sum()))

    def take(self) -> Dict[str, Any]:
        out = dict(self.acc)
        self.reset()
        return out


def _sustained(series: Sequence[float], thr: float, window: int) -> Tuple[bool, int]:
    run = best = 0
    for v in series:
        run = run + 1 if v > thr else 0
        best = max(best, run)
    return best >= window, best


def _summ(series: Sequence[float]) -> Dict[str, float]:
    if not series:
        return {"max": 0.0, "mean": 0.0, "p90": 0.0, "n": 0}
    a = np.asarray(series, dtype=float)
    return {"max": round(float(a.max()), 3), "mean": round(float(a.mean()), 3), "p90": round(float(np.percentile(a, 90)), 3), "n": int(len(a))}


# ------------------------------------------------------------------------------------------------ the world


class _PlannerStub:
    control_dt = 1.0 / 30.0


class TabletopWorld:
    """The persistent validation world: robot, slab, cuRobo backend; objects come and go."""

    def __init__(self, cfg: ValidateConfig = ValidateConfig(), policy: CP.ContactPolicy = CP.DEFAULT):
        from ispatialgym.oracle.curobo_planner import R1ProCuRobo  # warp pinning before omnigibson
        import omnigibson as og
        import torch as th
        from omnigibson.objects.primitive_object import PrimitiveObject

        from ispatialgym.oracle import r1pro_model as M
        from ispatialgym.oracle import sleep_assist as SA
        from ispatialgym.tasks.ispatialgym_task import apply_environment_rules

        t0 = time.time()
        self.cfg, self.policy, self.og, self.th, self.M, self.SA = cfg, policy, og, th, M, SA
        os.environ.setdefault("ISG_SLEEP_ASSIST", "30,1,3" if cfg.sleep_assist else "off")
        self.env = og.Environment(configs={"env": {"action_frequency": 30, "physics_frequency": 120},
                                           "scene": {"type": "Scene"},
                                           "robots": [M.joylo_robot_config("robot", obs_modalities=["proprio"], position=[0.0, 0.0, 0.0])]})
        self.robot = self.env.scene.robots[0]
        self.scene = self.env.scene
        self.slab = PrimitiveObject(name="gb_slab", primitive_type="Cube", fixed_base=True, size=1.0, rgba=(0.6, 0.5, 0.4, 1.0),
                                    scale=th.tensor([0.6, 0.6, cfg.slab_thickness_m]))
        self.scene.add_object(self.slab)
        self.slab.set_position_orientation(th.tensor([3.0, 3.0, cfg.slab_top_z - cfg.slab_thickness_m / 2]), th.tensor([0.0, 0.0, 0.0, 1.0]))
        for _ in range(3):
            og.sim.step()
        self.base_mass = M.apply_eval_base_mass(self.robot)
        apply_environment_rules([self.robot])
        self.backend = R1ProCuRobo(self.robot, prebuild=(), mesh_cache_size=64, cuboid_cache_size=8)
        self.backend.update_world(ignore_objects=[self.slab])
        self.root = self.backend.root_pose()
        self.jidx = self.backend._jidx
        self.trunk_limits = (self.robot.joint_lower_limits[self.robot.trunk_control_idx].clone(),
                             self.robot.joint_upper_limits[self.robot.trunk_control_idx].clone())
        self.hand = CP.HandChecker(cfg.arm, policy, spacing=0.004)
        self.t_setup = time.time() - t0
        self.timing: Dict[str, float] = {"setup_s": round(self.t_setup, 2)}
        self.physics_profile = {"action_hz": 30, "physics_hz": 120, "gpu_dynamics": False, "base_mass": self.base_mass,
                                "rigid_sleep_threshold": SA.RIGID_SLEEP_THRESHOLD, "sleep_assist": os.environ.get("ISG_SLEEP_ASSIST"),
                                "ag_rules": {"moving_link_joint": "FixedJoint", "release_on_pad_clear": True, "release_max_delay_s": 0.2},
                                "slab": {"thickness_m": cfg.slab_thickness_m, "material": "primitive default", "fixed_base": True}}

    # ------------------------------------------------------------------------------------------ configurations

    def q_config(self, translate: float, tilt: float, base_xy_yaw: Tuple[float, float, float], fingers: float = 0.05):
        th = self.th
        from ispatialgym.robot import conventions as C

        q = th.zeros(28)
        q[self.robot.trunk_control_idx] = C.torso_q(translate, tilt, self.trunk_limits)
        for arm in ("left", "right"):
            q[self.robot.arm_control_idx[arm]] = self.robot.untucked_default_joint_pos[self.robot.arm_control_idx[arm]].clone()
            q[self.robot.gripper_control_idx[arm]] = fingers
        rp, rq = self.root
        root2d = (float(rp[0]), float(rp[1]), C.yaw_of(rq.float()))
        return C.q_with_base_pose(q, *base_xy_yaw, root_pose2d=root2d, base_z=C.NOMINAL_BASE_Z, root_z=float(rp[2]), joint_index=self.jidx)

    def park(self):
        q = self.q_config(0.5, 0.0, (-2.5, 0.0, 0.0))
        self.robot.set_joint_positions(q)
        self.robot.keep_still()

    # ------------------------------------------------------------------------------------------ objects

    def load(self, job: Dict[str, Any], model_dir: str):
        th = self.th
        from omnigibson.objects.dataset_object import DatasetObject

        t0 = time.time()
        kw = {}
        sc = S.canonical_scale(job.get("scale"))
        if sc != (1.0, 1.0, 1.0):
            kw["scale"] = th.tensor(sc, dtype=th.float32)
        name = f"gb_{job['category']}_{job['model']}_{int(time.time() * 1000) % 100000}"
        obj = DatasetObject(name=name, category=job["category"], model=job["model"], dataset_name=job.get("dataset") or "behavior-1k-assets",
                            fixed_base=False, include_default_states=False, **kw)
        self.scene.add_object(obj)
        try:
            obj.set_position_orientation(th.tensor([3.0, -3.0, 1.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
            obj.keep_still()
            self.og.sim.step()
            self.SA.apply_to_objects([obj], {"rigid_sleep_threshold": self.SA.RIGID_SLEEP_THRESHOLD})
            if OG.load(model_dir) is None:
                OG.export_loaded(obj, model_dir, job)
            # the placement IK runs against the floor only (the object and the slab are checked explicitly, contact_policy), so the
            # cuRobo world is never synced with the objects of the persistent world - its mesh store stays empty for any number of models
            self.probe = ForceProbe(self.robot, self.slab, obj, self.cfg.arm)
            self.probe.install()
        except Exception:
            self.scene.remove_object(obj)
            self.og.sim.step()
            raise
        self.timing["load_s"] = round(time.time() - t0, 2)
        return obj

    def unload(self, obj):
        try:
            self.probe.uninstall()
        except Exception:  # noqa: BLE001
            pass
        for arm in ("left", "right"):
            try:
                self.robot.release_grasp_immediately(arm)
            except Exception:  # noqa: BLE001
                pass
        self.scene.remove_object(obj)
        self.og.sim.step()

    # ------------------------------------------------------------------------------------------ stepping

    def step(self, action, sa=None, k: int = 0) -> Dict[str, Any]:
        self.env.step(action)
        if sa is not None:
            sa.step(k)
        return self.probe.take()

    def hold_action(self):
        from ispatialgym.oracle.action import ActionState

        st = ActionState(self.robot)
        st.hold_current()
        return st.action()

    # ------------------------------------------------------------------------------------------ rest

    def slab_pose_for(self, obj_xy, radius: float) -> Tuple[List[float], List[float]]:
        half = radius + self.cfg.slab_half_m
        return [float(obj_xy[0]), float(obj_xy[1]), self.cfg.slab_top_z - self.cfg.slab_thickness_m / 2], [2 * half, 2 * half, self.cfg.slab_thickness_m]

    def place_slab(self, centre, dims):
        th = self.th
        self.slab.set_position_orientation(th.tensor(centre, dtype=th.float32), th.tensor([0.0, 0.0, 0.0, 1.0]))
        sc = th.tensor(dims, dtype=th.float32)
        if not th.allclose(self.slab.scale, sc, atol=1e-5):
            self.slab.scale = sc
        self.support = CP.Support.box(centre, [d / 2 for d in dims], name="slab")

    def settle_survey(self, obj, geom: OG.ObjectGeometry, quat0: np.ndarray, plan: Sequence[Tuple[float, float]]) -> List[Dict[str, Any]]:
        """of-v2maps 2026-11-07 (``GB_SETTLE_FIRST``, :mod:`ispatialgym.grasping.settle_first`): one :meth:`settle` per ``(yaw0, drop_m)`` of
        ``plan`` from the designated rest ``quat0`` turned by ``yaw0`` about +z; the rest records with ``yaw0`` / ``drop_m`` (no snapshot)."""
        out = []
        for yaw0, drop in plan:
            q = S.quat_mul(S.quat_yaw(float(yaw0)), np.asarray(quat0, dtype=float))
            rec, _ = self.settle(obj, geom, q, drop_m=float(drop))
            rec.update(yaw0=round(float(yaw0), 5), drop_m=float(drop))
            out.append(rec)
        return out

    def settle(self, obj, geom: OG.ObjectGeometry, quat0: np.ndarray, obj_xy=(0.0, 0.0), drop_m: Optional[float] = None) -> Tuple[Dict[str, Any], Any]:
        """Drop and settle the object on the slab with the robot parked; the measured rest record and the scene snapshot. ``drop_m``:
        the drop height (default ``cfg.drop_m``; the settle-first survey varies it)."""
        th, cfg = self.th, self.cfg
        drop = cfg.drop_m if drop_m is None else float(drop_m)
        t0 = time.time()
        V = geom.union.vertices
        R0 = S.quat_to_mat(quat0)
        bottom = -float((V @ R0.T)[:, 2].min())
        radius = float(np.linalg.norm((V @ R0.T)[:, :2], axis=1).max())
        centre, dims = self.slab_pose_for(obj_xy, radius)
        self.park()
        self.place_slab(centre, dims)
        z0 = cfg.slab_top_z + bottom + drop
        obj.set_position_orientation(th.tensor([obj_xy[0], obj_xy[1], z0], dtype=th.float32), th.tensor(quat0, dtype=th.float32))
        obj.keep_still()
        a = self.hold_action()
        hist: List[Tuple[np.ndarray, np.ndarray]] = []
        stable = False
        steps = 0
        load = []
        for steps in range(1, cfg.settle_max_steps + 1):
            f = self.step(a)
            p, q = obj.get_position_orientation()
            hist.append((p.numpy().astype(float).copy(), q.numpy().astype(float).copy()))
            load.append(f["obj_slab_z"])
            if steps >= cfg.settle_window:
                ps = np.array([h[0] for h in hist[-cfg.settle_window:]])
                qs = [h[1] for h in hist[-cfg.settle_window:]]
                dp = float(np.linalg.norm(ps - ps[-1], axis=1).max())
                dr = max(_qang(qq, qs[-1]) for qq in qs)
                v = float(th.linalg.norm(obj.get_linear_velocity()))
                w = float(th.linalg.norm(obj.get_angular_velocity()))
                if dp <= cfg.settle_pos_m and dr <= cfg.settle_rot_rad and v <= cfg.settle_speed_m_s and w <= cfg.settle_ang_rad_s:
                    stable = True
                    break
        # the support load, measured awake (PhysX reports no contacts of a sleeping body): two steps after a wake
        weight = float(obj.mass) * 9.81
        obj.wake()
        wl = []
        for _ in range(2):
            wl.append(self.step(a)["obj_slab_z"])
        support_load = float(np.mean(wl))
        contact = support_load > 0.5 * weight
        p, q = obj.get_position_orientation()
        p, q = p.numpy().astype(float), q.numpy().astype(float)
        ps = np.array([h[0] for h in hist[-cfg.settle_window:]])
        qs = [h[1] for h in hist[-cfg.settle_window:]]
        Vw = V @ S.quat_to_mat(q).T + p
        rec = {"schema": S.REST, "rest_id": None, "quat_convention": "xyzw; q_object = R_z(yaw) * quat_xyzw",
               "support_frame": "z = 0 at the slab top, +z up, axes of the validation world; x/y diagnostic only",
               "reference_yaw": cfg.reference_yaw, "intended_quat_xyzw": [round(float(v), 7) for v in quat0],
               "quat_xyzw": [round(float(v), 7) for v in q], "h_root": round(float(p[2] - cfg.slab_top_z), 6),
               "pose_support": {"position": [round(float(p[0] - obj_xy[0]), 6), round(float(p[1] - obj_xy[1]), 6), round(float(p[2] - cfg.slab_top_z), 6)],
                                "orientation_xyzw": [round(float(v), 7) for v in q]},
               "tilt_from_intended_rad": round(S.tilt_distance(q, quat0), 5), "yaw_from_reference_rad": round(S.yaw_of_quat(q) - cfg.reference_yaw, 5),
               "lowest_point_gap_m": round(float(Vw[:, 2].min() - cfg.slab_top_z), 6), "drop_m": drop,
               "stable": bool(stable and contact), "pose_still": bool(stable), "contact": bool(contact),
               "support_load_N": round(support_load, 3), "weight_N": round(weight, 3), "contact_rule": "awake support load > 0.5 x weight",
               "settle_steps": steps, "settle_time_s": round(steps / 30.0, 3),
               "final_lin_vel_m_s": round(float(th.linalg.norm(obj.get_linear_velocity())), 6),
               "final_ang_vel_rad_s": round(float(th.linalg.norm(obj.get_angular_velocity())), 6),
               "drift_window": {"steps": cfg.settle_window, "pos_m": round(float(np.linalg.norm(ps - ps[-1], axis=1).max()), 6),
                                "rot_rad": round(max(_qang(qq, qs[-1]) for qq in qs), 6)},
               "slab": {"centre": centre, "dims": dims}, "physics_profile": self.physics_profile, "policy_key": self.policy.key(self.cfg.arm),
               "validator": VALIDATOR_VERSION, "created": S.now(), "wall_s": round(time.time() - t0, 2)}
        state = self.scene.dump_state(serialized=False)
        return rec, state

    def restore(self, obj, state, rest: Dict[str, Any]) -> Dict[str, Any]:
        """The rest snapshot back, then verified against the saved rest (pose within ``restore_*``)."""
        th = self.th
        for arm in ("left", "right"):
            try:
                self.robot.release_grasp_immediately(arm)
            except Exception:  # noqa: BLE001
                pass
        self.scene.load_state(state, serialized=False)
        sl = rest["slab"]
        self.place_slab(sl["centre"], sl["dims"])
        p, q = obj.get_position_orientation()
        dz = abs(float(p[2]) - self.cfg.slab_top_z - rest["h_root"])
        dr = _qang(q.numpy(), np.asarray(rest["quat_xyzw"]))
        return {"restore_dz_m": round(dz, 6), "restore_rot_rad": round(dr, 6), "restore_ok": dz <= self.cfg.restore_pos_m and dr <= self.cfg.restore_rot_rad}

    # ------------------------------------------------------------------------------------------ placement by IK

    def plan(self, obj, geom: OG.ObjectGeometry, cands: List[Dict[str, Any]], obj_pose) -> Dict[str, Dict[str, Any]]:
        """Whole-robot configurations putting the eef at each candidate (batched IK over base yaw x offset x trunk), checked under the
        contact policy for the whole robot; the most comfortable passing one per candidate."""
        th, cfg = self.th, self.cfg
        t0 = time.time()
        T_wo = S.pose_mat(*obj_pose)
        targets = {c["name"]: T_wo @ S.pose_of(c["T_object_eef"]) for c in cands}
        # a start within the policy's finger allowance but below the slab top starts where the oracle's approach would stop: the
        # fingers just on the surface (a teleport into the overlap is resolved by PhysX as an impulse); the transform used is recorded
        lifts = {}
        if targets:
            names0 = list(targets)
            cl = self.hand.clearance(th.as_tensor(np.stack([targets[n] for n in names0]), dtype=th.float32), self.support,
                                     (max(c["preshape_q"] for c in cands),) * 2)
            for i, n in enumerate(names0):
                lift = max(0.0, -float(cl["finger"][i]) + 0.0005) if float(cl["finger"][i]) < 0.0 else 0.0
                if lift > 0.0:
                    targets[n] = targets[n].copy()
                    targets[n][2, 3] += lift
                lifts[n] = lift
        names = list(targets)
        best: Dict[str, Dict[str, Any]] = {}
        n_ik = 0
        untucked = self.robot.untucked_default_joint_pos[self.robot.arm_control_idx[cfg.arm]].float()
        pieces = [(th.as_tensor(pc.face_normals, dtype=th.float32), th.as_tensor(np.einsum("ij,ij->i", pc.face_normals, pc.triangles[:, 0]), dtype=th.float32))
                  for pc in geom.pieces]
        Rw, tw = th.as_tensor(T_wo[:3, :3], dtype=th.float32), th.as_tensor(T_wo[:3, 3], dtype=th.float32)
        for k in range(cfg.base_yaws):
            for off in cfg.base_offsets:
                for tr, ti in cfg.trunks:
                    todo = [n for n in names if n not in best or best[n]["cost"] > 0.5]
                    if not todo:
                        break
                    rel = th.tensor([float(T_wo[0, 3]), float(T_wo[1, 3])])
                    poses = []
                    for n in todo:
                        T = targets[n]
                        poses.append((th.as_tensor(T[:3, 3], dtype=th.float32), th.as_tensor(S.mat_to_quat(T[:3, :3]), dtype=th.float32)))
                    # the base yaw (k of base_yaws), the object at ``off`` in the base frame
                    yaw = 2 * math.pi * k / cfg.base_yaws
                    c, s = math.cos(yaw), math.sin(yaw)
                    ahead = self.support.half[0] + self.cfg.slab_edge_clear_m + off[0]
                    bx = float(rel[0]) - (c * ahead - s * off[1])
                    by = float(rel[1]) - (s * ahead + c * off[1])
                    edge = self.cfg.slab_edge_clear_m
                    # the slab's near edge must stay >= slab_edge_clear_m from the base origin (a table edge at a sampler stand-off)
                    sc = self.support
                    d_edge = _box_distance_2d(sc, bx, by)
                    if d_edge < edge:
                        continue
                    q_seed = self.q_config(tr, ti, (bx, by, yaw))
                    succ, qs = self.backend.solve_ik_batch(f"{cfg.arm}_arm", self.M.EEF_LINK[cfg.arm], poses, q_seed, batch_size=max(8, len(poses)), seeds=cfg.ik_seeds)
                    n_ik += len(poses)
                    ok_idx = [i for i, s_ in enumerate(succ.tolist()) if s_]
                    if not ok_idx:
                        continue
                    Q = qs[ok_idx]
                    sph, links = self.backend.robot_spheres_world(Q)
                    ok_s, worst = CP.sphere_support_ok(sph, links, self.support, cfg.arm, self.policy)
                    # arm spheres (not the hand) clear of the object's pieces
                    hand = set(self.policy.finger_links(cfg.arm)) | set(self.policy.hand_rigid_links(cfg.arm))
                    m = th.tensor([(l not in hand) and bool(l) for l in links])
                    P = sph[:, m, :3]
                    r = sph[:, m, 3]
                    Po = (P - tw) @ Rw  # object frame
                    deep = th.full(r.shape, float("inf"))
                    for nrm, d in pieces:
                        deep = th.minimum(deep, (Po @ nrm.T - d).max(dim=-1).values)
                    ok_o = ((deep - r) >= 0.0) | (r <= 0)
                    ok_o = ok_o.all(dim=1)
                    for j, i in enumerate(ok_idx):
                        n = todo[i]
                        if not (bool(ok_s[j]) and bool(ok_o[j])):
                            best.setdefault(n, {"cost": float("inf"), "fail": worst[j] or "arm_object"})
                            continue
                        qa = Q[j][self.robot.arm_control_idx[cfg.arm]].float()
                        cost = float(th.linalg.norm(qa - untucked))
                        if n not in best or cost < best[n]["cost"]:
                            best[n] = {"cost": cost, "q": Q[j].clone(), "base": [round(bx, 4), round(by, 4), round(yaw, 4)], "trunk": [tr, ti], "fail": None,
                                       "start_lift_m": round(lifts.get(n, 0.0), 5), "T_object_eef_used": S.mat_pose(np.linalg.inv(T_wo) @ targets[n])}
        self.timing["plan_s"] = round(time.time() - t0, 2)
        self.timing["plan_ik_poses"] = n_ik
        return best

    # ------------------------------------------------------------------------------------------ trial

    def trial(self, obj, cand: Dict[str, Any], plan: Dict[str, Any], state, rest: Dict[str, Any]) -> Dict[str, Any]:
        from ispatialgym.oracle import execute as EX
        from ispatialgym.oracle.motions import Motions, MotionsConfig

        th, cfg, arm = self.th, self.cfg, self.cfg.arm
        t0 = time.time()
        rec: Dict[str, Any] = {"schema": S.TRIAL, "candidate": cand["name"], "family_id": cand.get("family_id"), "rest_id": rest["rest_id"],
                               "mode": cand.get("mode"), "base": plan["base"], "trunk": plan["trunk"]}
        rec.update(self.restore(obj, state, rest))
        if not rec["restore_ok"]:
            rec.update({"verdict": "invalid", "reasons": ["rest_not_restored"]})
            return rec
        width = float(cand["width_m"])
        mcfg = MotionsConfig()
        pre_q = min(0.05, 0.5 * width + mcfg.gripper_preshape_margin)
        q = plan["q"].clone()
        q[self.robot.gripper_control_idx[arm]] = pre_q
        self.robot.set_joint_positions(q)
        self.robot.keep_still()
        motions = Motions(self.env, _PlannerStub(), cfg=mcfg, robot=self.robot)
        motions.state.hold_current()
        motions.state.gripper_cmd[arm] = motions.gripper_command(arm, pre_q)
        obj.wake()
        sa = self.SA.SleepAssist(self.robot, {obj.name: obj}, "gb") if cfg.sleep_assist else None
        p_rest, q_rest = (x.clone() for x in obj.get_position_orientation())
        series: Dict[str, List[float]] = {"finger_slab": [], "other_slab": [], "obj_slab_ratio": [], "finger_obj": []}
        other_links: List[str] = []
        weight = float(obj.mass) * 9.81
        k = [0]

        asleep = [0]

        def run(gen, phase, wake=False):
            for a in gen:
                if wake and self.robot._ag_obj_in_hand.get(arm) is not obj:
                    # the demo's object is awake when the pads close (SleepAssist wakes it for a moving gripper within 5 cm; a slept body
                    # the pads meet unwoken takes the closure as a kick) - here the hand starts at the grasp, so the wake is explicit
                    obj.wake()
                f = self.step(a, sa, k[0])
                try:
                    asleep[0] += int(bool(obj.root_link.is_asleep))
                except Exception:  # noqa: BLE001
                    pass
                k[0] += 1
                series["finger_slab"].append(f["finger_slab"])
                series["other_slab"].append(f["other_slab"])
                series["obj_slab_ratio"].append(f["obj_slab_z"] / max(weight, 1e-6))
                series["finger_obj"].append(f["finger_obj"])
                if f["other_slab"] > cfg.non_finger_contact_N and f["other_link"]:
                    other_links.append(f"{phase}:{f['other_link']}")
        run(motions.hold(cfg.init_hold_steps), "init", wake=True)
        push = float(th.linalg.norm(obj.get_position_orientation()[0] - p_rest))
        rate, squeeze, thin = EX.closure_for_width(width, mcfg, library=True)
        n_close0 = k[0]
        run(motions.gripper(arm, True, rate=rate, squeeze=squeeze), "close", wake=True)
        ag_step = None
        held = self.robot._ag_obj_in_hand.get(arm)
        attached = held is obj
        evs = [e for e in motions.release_events if e.get("event") == "contact_hold"]
        if attached:
            ag_step = (evs[-1].get("attached_after_steps") if evs else None)
        T_oe_attach = self._T_object_eef(obj)
        run(motions.hold(cfg.hold_steps), "hold")
        held_hold = self.robot._ag_obj_in_hand.get(arm) is obj
        # support unloading: the slab drops away, the object hangs on the grasp
        sl = rest["slab"]
        c = list(sl["centre"])
        c[2] -= cfg.unload_drop_m
        self.slab.set_position_orientation(th.tensor(c, dtype=th.float32), th.tensor([0.0, 0.0, 0.0, 1.0]))
        T_oe0 = self._T_object_eef(obj)
        run(motions.hold(cfg.unload_steps), "unload")
        T_oe1 = self._T_object_eef(obj)
        retained = self.robot._ag_obj_in_hand.get(arm) is obj
        dpos = float(np.linalg.norm(T_oe1[:3, 3] - T_oe0[:3, 3]))
        drot = _Rang(T_oe0[:3, :3], T_oe1[:3, :3])
        T_c = S.pose_of(cand["T_object_eef"])
        off_pos = float(np.linalg.norm(T_oe_attach[:3, 3] - T_c[:3, 3]))
        off_rot = _Rang(T_oe_attach[:3, :3], T_c[:3, :3])
        finger_bad, finger_run = _sustained(series["finger_slab"], cfg.finger_force_max_N, cfg.force_window)
        load_bad, load_run = _sustained(series["obj_slab_ratio"], cfg.object_load_ratio_max, cfg.force_window)
        reasons = []
        if not attached:
            reasons.append("ag_not_formed")
        elif not held_hold:
            reasons.append("ag_lost_in_hold")
        if attached and not retained:
            reasons.append("dropped_on_unload")
        if retained and (dpos > cfg.retain_pos_m or drot > cfg.retain_rot_rad):
            reasons.append("slipped_on_unload")
        if other_links:
            reasons.append("non_finger_table_contact")
        if finger_bad:
            reasons.append("finger_pushes_table")
        if load_bad:
            reasons.append("object_pressed_into_table")
        if attached and (off_pos > cfg.on_target_pos_m or off_rot > cfg.on_target_rot_rad):
            reasons.append("grasp_off_target")
        fingers = self.robot.get_joint_positions()[self.robot.gripper_control_idx[arm]].tolist()
        ag_link = None
        try:
            ag_link = (self.robot._ag_obj_constraint_params.get(arm) or {}).get("ag_link_prim_path")
        except Exception:  # noqa: BLE001
            pass
        rec.update({
            "verdict": "valid" if not reasons else "invalid", "reasons": reasons,
            "ag": {"formed": bool(attached), "attached_after_steps": ag_step, "held_after_hold": bool(held_hold), "retained_after_unload": bool(retained),
                   "unload_drift_m": round(dpos, 5), "unload_drift_rad": round(drot, 5), "ag_link": ag_link},
            "closure": {"rate_m_per_step": rate, "squeeze_m": squeeze, "thin_pinch": thin, "preshape_q": round(pre_q, 5), "events": motions.release_events[-3:],
                        "fingers_final_m": [round(float(v), 5) for v in fingers]},
            "achieved": {"T_object_eef_at_attach": S.mat_pose(T_oe_attach), "off_target_m": round(off_pos, 5), "off_target_rad": round(off_rot, 5),
                         "placement_push_m": round(push, 5)},
            "forces": {"units": "N per control step (max over its physics steps); obj_slab_ratio = object normal load / weight",
                       "finger_slab": _summ(series["finger_slab"]), "finger_slab_sustained_steps": finger_run,
                       "other_slab": _summ(series["other_slab"]), "other_links": sorted(set(other_links))[:8],
                       "obj_slab_ratio": _summ(series["obj_slab_ratio"]), "obj_slab_ratio_sustained_steps": load_run,
                       "finger_obj": _summ(series["finger_obj"]),
                       "thresholds": {"finger_force_max_N": cfg.finger_force_max_N, "object_load_ratio_max": cfg.object_load_ratio_max,
                                      "force_window_steps": cfg.force_window, "non_finger_contact_N": cfg.non_finger_contact_N, "calibrated": False}},
            "eef_in_object_used": plan.get("T_object_eef_used", cand["T_object_eef"]), "start_lift_m": plan.get("start_lift_m", 0.0),
            "object_asleep_steps": asleep[0], "steps": k[0], "wall_s": round(time.time() - t0, 3),
        })
        if sa is not None:
            try:
                rec["sleep_assist"] = {"sleeps": sa.sleeps, "wakes": sa.wakes}
            except Exception:  # noqa: BLE001
                pass
        return rec

    def _T_object_eef(self, obj) -> np.ndarray:
        p, q = obj.get_position_orientation()
        ep, eq = self.robot.eef_links[self.cfg.arm].get_position_orientation()
        return np.linalg.inv(S.pose_mat(p.numpy(), q.numpy())) @ S.pose_mat(ep.numpy(), eq.numpy())


def _box_distance_2d(sup: CP.Support, x: float, y: float) -> float:
    cx, cy = sup.centre[0], sup.centre[1]
    hx, hy = sup.half[0], sup.half[1]
    dx = max(abs(x - cx) - hx, 0.0)
    dy = max(abs(y - cy) - hy, 0.0)
    return math.hypot(dx, dy)


def _qang(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = abs(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))
    return 2 * math.acos(min(1.0, d))


def _Rang(A, B) -> float:
    R = np.asarray(A).T @ np.asarray(B)
    return math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))


# ------------------------------------------------------------------------------------------------ one model


def run_model(world: TabletopWorld, job: Dict[str, Any], model_dir: str) -> Dict[str, Any]:
    """Rest, placement and trials of one model; writes ``rests.json``, ``validation.json`` and ``library.json``."""
    cfg = world.cfg
    t0 = time.time()
    cand_doc = S.load_json(os.path.join(model_dir, "candidates.json"))
    if cand_doc is None:
        raise FileNotFoundError(f"{model_dir}: no candidates.json")
    by_name = {c["name"]: c for c in cand_doc["candidates"]}
    reps = [by_name[n] for n in cand_doc.get("representatives", []) if n in by_name][: cfg.max_trials]
    obj = world.load(job, model_dir)
    geom = OG.load(model_dir)
    rests: Dict[str, Dict[str, Any]] = {}
    trials: List[Dict[str, Any]] = []
    timing = dict(world.timing)
    try:
        hint = job.get("rest_hint_wxyz")
        q_rest = np.array([0, 0, 0, 1.0]) if hint is None else np.array([hint[1], hint[2], hint[3], hint[0]])
        quat0 = S.quat_mul(S.quat_yaw(cfg.reference_yaw), q_rest)
        rest, state = world.settle(obj, geom, quat0)
        intended = rest["tilt_from_intended_rad"] <= cfg.rest_tilt_tol_rad
        rest["rest_id"] = "r0" if intended else "r0_other"
        rest["intended"] = bool(intended)
        rest.update({"category": job["category"], "model": job["model"], "scale": list(S.canonical_scale(job.get("scale"))), "dataset": job.get("dataset"),
                     "asset": {"category": job["category"], "model": job["model"], "dataset": job.get("dataset") or "behavior-1k-assets"},
                     "provenance": {"job": job.get("id"), "legacy_key": job.get("legacy_key"), "validator": VALIDATOR_VERSION}})
        if job.get("settle_first"):
            # of-v2maps 2026-11-07 (GB_SETTLE_FIRST): the surveyed rest the candidates were generated at; this settle is the guard
            rest["settle_first"] = dict(job["settle_first"], designated_rest_hint_wxyz=job.get("designated_rest_hint_wxyz"))
        rests[rest["rest_id"]] = rest
        timing["settle_s"] = rest["wall_s"]
        p, q = obj.get_position_orientation()
        obj_pose = (p.numpy().astype(float), q.numpy().astype(float))
        # the representatives re-checked against the MEASURED rest's slab (the candidates were screened on the designated rest)
        import torch as th

        T_wo = S.pose_mat(*obj_pose)
        Tw = th.as_tensor(np.stack([T_wo @ S.pose_of(c["T_object_eef"]) for c in reps]), dtype=th.float32) if reps else None
        geo = CP.check_grasp_geometry(world.hand, Tw, world.support, [c["contact_q"] for c in reps], open_q=max([c["preshape_q"] for c in reps] + [0.0])) if reps else None
        # a candidate whose fingers end up just past the allowance at the MEASURED rest (it settles a fraction of a mm off the canonical
        # rest the generator used) is lifted along the support normal by the deficit (<= max_rest_lift_m) and re-checked; the lifted
        # transform is the one tested and recorded
        reps = [dict(c) for c in reps]
        for i, c in enumerate(reps):
            if geo is None or bool(geo["ok"][i]) or geo["reason"][i] != "finger_penetration":
                continue
            need = -world.policy.finger_support_max_penetration_m + 0.001 - float(geo["finger_min"][i])
            if not 0.0 < need <= cfg.max_rest_lift_m:
                continue
            T_l = T_wo @ S.pose_of(c["T_object_eef"])
            T_l[2, 3] += need
            g1 = CP.check_grasp_geometry(world.hand, th.as_tensor(T_l[None], dtype=th.float32), world.support, [c["contact_q"]], open_q=c["preshape_q"])
            if bool(g1["ok"][0]):
                c["T_object_eef"] = S.mat_pose(np.linalg.inv(T_wo) @ T_l)
                c["rest_lift_m"] = round(need, 5)
                geo["ok"][i] = True
                geo["reason"][i] = "ok"
        plans = world.plan(obj, geom, [c for i, c in enumerate(reps) if bool(geo["ok"][i])], obj_pose) if reps else {}
        timing["plan_s"] = world.timing.get("plan_s")
        n_valid = 0
        for i, c in enumerate(reps):
            if n_valid >= cfg.max_valid:
                break
            if not bool(geo["ok"][i]):
                trials.append({"schema": S.TRIAL, "candidate": c["name"], "family_id": c.get("family_id"), "rest_id": rest["rest_id"], "verdict": "invalid",
                               "reasons": [f"geometry_at_measured_rest:{geo['reason'][i]}@{geo['stage'][i]}"]})
                continue
            pl = plans.get(c["name"])
            if pl is None or pl.get("q") is None:
                trials.append({"schema": S.TRIAL, "candidate": c["name"], "family_id": c.get("family_id"), "rest_id": rest["rest_id"], "verdict": "invalid",
                               "reasons": ["no_placement" if pl is None else f"no_placement:{pl.get('fail')}"]})
                continue
            tr = world.trial(obj, c, pl, state, rest)
            trials.append(tr)
            n_valid += tr["verdict"] == "valid"
    except Exception as e:  # noqa: BLE001
        trials.append({"error": repr(e), "traceback": traceback.format_exc()[-2000:]})
    finally:
        world.unload(obj)
    timing["model_s"] = round(time.time() - t0, 2)
    timing["trial_s"] = round(sum(t.get("wall_s", 0.0) for t in trials), 2)
    S.save_json(os.path.join(model_dir, "rests.json"), {"schema": S.REST, "rests": rests})
    S.save_json(os.path.join(model_dir, "validation.json"), {"schema": S.TRIAL, "validator": VALIDATOR_VERSION, "config": asdict(cfg),
                                                             "policy_key": world.policy.key(cfg.arm), "trials": trials, "timing": timing, "created": S.now()})
    lib = build_library(job, cand_doc, rests, trials, world.policy, world.physics_profile)
    S.save_json(os.path.join(model_dir, "library.json"), lib)
    return {"id": job.get("id"), "n_reps": len(reps), "n_valid": sum(1 for t in trials if t.get("verdict") == "valid"), "timing": timing,
            "rests": {k: {"stable": v["stable"], "h_root": v["h_root"], "tilt": v["tilt_from_intended_rad"]} for k, v in rests.items()},
            "reasons": [t.get("reasons") or t.get("error") for t in trials if t.get("verdict") != "valid"]}


def build_library(job, cand_doc, rests, trials, policy: CP.ContactPolicy, physics_profile) -> Dict[str, Any]:
    by_name = {c["name"]: c for c in cand_doc["candidates"]}
    grasps = []
    for t in trials:
        if t.get("verdict") != "valid":
            continue
        c = by_name[t["candidate"]]
        grasps.append({"name": c["name"], "rest_id": t["rest_id"], "family_id": c.get("family_id"), "mode": c.get("mode"),
                       "T_object_eef": t.get("eef_in_object_used", c["T_object_eef"]), "T_object_eef_at_attach": t["achieved"]["T_object_eef_at_attach"],
                       "width_m": c["width_m"], "preshape_q": c["preshape_q"], "finger_q": t["closure"]["fingers_final_m"],
                       "physics": "tested", "trial_forces": {k: t["forces"][k]["max"] for k in ("finger_slab", "obj_slab_ratio")}})
    fams = {fid: f for fid, f in (cand_doc.get("families") or {}).items() if any(g["family_id"] == fid for g in grasps)}
    for f in fams.values():
        f["physics"] = "representative_only"
        f["members_physics"] = "not tested: identified geometrically (members folded by symmetry.dedup); reach maps / sampling check the member they use"
    return {"schema": S.LIBRARY, "category": job["category"], "model": job["model"], "scale": list(S.canonical_scale(job.get("scale"))),
            "dataset": job.get("dataset"), "constraint": job.get("constraint"), "legacy_key": job.get("legacy_key"), "arm": policy_arm(cand_doc),
            "policy_key": policy.key(policy_arm(cand_doc)), "policy": policy.meta(policy_arm(cand_doc)), "physics_profile": physics_profile,
            "rests": {k: v for k, v in rests.items()}, "families": fams, "grasps": grasps,
            "twins": "finger-swapped twins (eef turned pi about +z) are NOT stored: reach maps evaluate them with their own IK / policy checks",
            "provenance": {"candidates": cand_doc.get("pipeline"), "validator": VALIDATOR_VERSION, "created": S.now()}}


def policy_arm(cand_doc) -> str:
    return cand_doc.get("arm", "right")
