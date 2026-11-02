"""cuRobo v2 (``nvidia-curobo>=0.8``) backend for R1Pro in OmniGibson.

One :class:`curobo.motion_planner.MotionPlanner` is built per *embodiment* (active joint group,
see :data:`ispatialgym.oracle.r1pro_model.EMBODIMENTS`). All planners share one scene collision
checker so the world is loaded once. Before every query the locked joints of the selected
planner are refreshed from the simulator state, so the planner sees the true current pose of
every joint it is not allowed to move.

Frames: cuRobo kinematics are rooted at ``base_footprint_x`` (the OmniGibson articulation root,
fixed in the world during an episode). Public methods take and return **world-frame** poses with
xyzw quaternions; conversion to the root frame and to wxyz happens here.
"""

from __future__ import annotations

import importlib.metadata
import logging
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th


#: pip-warp submodules cuRobo touches lazily; import them eagerly before Isaac Sim starts.
_WARP_EAGER_SUBMODULES = ("warp.config", "warp.context", "warp.types", "warp.utils", "warp.torch", "warp.constants")


class _PipWarpFinder:
    """``sys.meta_path`` finder pinning ``warp`` and ``warp.*`` to the pip package.

    Isaac Sim's Kit *fast importer* (``omni.ext._impl.fast_importer.FastFinder``) maps module
    names to files under its extension folders regardless of the parent package's ``__path__``.
    Once Kit is running, a lazily imported ``warp.torch`` therefore resolves to the bundled
    ``omni.warp.core`` (warp 1.8) instead of the pip ``warp-lang`` cuRobo v0.8 was built against.
    Placing this finder in front of ``FastFinder`` restores standard resolution for ``warp``.
    """

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if fullname == "warp" or fullname.startswith("warp."):
            import importlib.machinery

            return importlib.machinery.PathFinder.find_spec(fullname, path, target)
        return None


def _ensure_pip_warp():
    """Make sure ``import warp`` resolves to the pip ``warp-lang`` that cuRobo needs (import time)."""
    if "warp" in sys.modules:
        wp = sys.modules["warp"]
        if "extscache" in (getattr(wp, "__file__", "") or ""):
            raise ImportError(
                f"Isaac Sim's bundled warp ({wp.__file__}) was imported before cuRobo. Import "
                "`ispatialgym.oracle.curobo_planner` (or `warp`) before `omnigibson`."
            )
    else:
        try:
            site_dir = str(importlib.metadata.distribution("warp-lang").locate_file(""))
            if site_dir in sys.path:
                sys.path.remove(site_dir)
            sys.path.insert(0, site_dir)
        except importlib.metadata.PackageNotFoundError:
            pass
        import warp  # noqa: F401
    for name in _WARP_EAGER_SUBMODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            pass


def pin_pip_warp_after_launch():
    """Install :class:`_PipWarpFinder` ahead of Kit's fast importer (call once Isaac Sim runs)."""
    if not any(f is _PipWarpFinder for f in sys.meta_path):
        sys.meta_path.insert(0, _PipWarpFinder)
    # Drop bundled-warp copies of modules that also exist in pip warp so re-imports get pip's.
    import importlib.machinery

    for name, mod in list(sys.modules.items()):
        if not (name == "warp" or name.startswith("warp.")):
            continue
        if "omni.warp.core" not in (getattr(mod, "__file__", "") or ""):
            continue
        parent = sys.modules.get(name.rpartition(".")[0]) if "." in name else None
        ppath = list(getattr(parent, "__path__", [])) if parent is not None else None
        if ppath and importlib.machinery.PathFinder.find_spec(name, ppath) is not None:
            del sys.modules[name]


_ensure_pip_warp()

import omnigibson as og  # noqa: E402
import omnigibson.utils.transform_utils as T  # noqa: E402
from omnigibson.utils.usd_utils import mesh_prim_shape_to_trimesh_mesh  # noqa: E402

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria  # noqa: E402
from curobo._src.util.warp import init_warp  # noqa: E402
from curobo._src.geom.collision import SceneCollisionCfg, create_collision_checker
from curobo._src.graph_planner.graph_planner_prm import PRMGraphPlanner
from curobo._src.robot.types.joint_types import JointType
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_trajopt import TrajOptSolver
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Mesh, Scene
from curobo.types import DeviceCfg, GoalToolPose, JointState
from curobo.types import Pose as CuPose

from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.poses import Pose, as_tensor, quat_wxyz_to_xyzw, quat_xyzw_to_wxyz
from ispatialgym.fastflags import fast_on, fast_value  # 2026-10-31 train-fast: switches / tunables of the batched IK (off = the former behaviour)

log = logging.getLogger(__name__)

#: ISPATIALGYM_BATCHED_WORLD_SYNC=1 (opt-in): ``update_world`` writes the poses of the meshes already in the store in one
#: batched step at the end (one Pose inverse over all of them, one indexed assignment) instead of a one-mesh Pose,
#: inverse, name lookup and write per mesh; the per-mesh pose reads and relative transforms are unchanged. =check also
#: inverts every pose on its own and prints WORLD_SYNC_CHECK counts when a batched row differs.
BATCHED_WORLD_SYNC = os.environ.get("ISPATIALGYM_BATCHED_WORLD_SYNC", "")
_WS_COUNTS = {"calls": 0, "meshes": 0, "mismatches": 0}

#: ISPATIALGYM_SEEDED_SPHERE_FIT (changes the fitted spheres, hence sampled instances and demos, once): =1 seeds each
#: attach sphere fit (morphit: torch and numpy random draws) from a hash of its exact inputs - the collision meshes'
#: prim paths, vertices, faces, scales and float32 poses in the eef frame, and the sphere count - with the global torch /
#: numpy / random generators saved and restored around it, so a fit no longer shifts the process's other random draws
#: and the same inputs always give the same spheres; =cache also keeps every fit by that key (an exact cache, since the
#: fit is then a function of its key). Default off: the global generators, as production (v2.0-multienv's default was cache).
SEEDED_SPHERE_FIT = os.environ.get("ISPATIALGYM_SEEDED_SPHERE_FIT", "off")
if SEEDED_SPHERE_FIT in ("off", "0"):
    SEEDED_SPHERE_FIT = ""
_FIT_CACHE: Dict[str, th.Tensor] = {}
_FIT_COUNTS = {"fits": 0, "fit_s": 0.0, "hits": 0}


def shrink_link_spheres(spheres: th.Tensor, mask: th.Tensor, margin: float) -> Tuple[th.Tensor, float]:
    """``(spheres, activation distance)`` of a per-link proximity query at ``margin``: a negative margin shrinks the radii of the
    ``mask``-ed enabled spheres (``(..., S, 4)``, r <= 0 = disabled; ``mask`` broadcast over the sphere axis) by ``-margin`` (to >= 0.1 mm)
    and the query counts penetration only (activation 0) - those spheres may sink up to ``-margin`` into an obstacle."""
    if margin >= 0.0:
        return spheres, float(margin)
    r = spheres[..., 3]
    out = spheres.clone()
    out[..., 3] = th.where(mask.reshape(*([1] * (r.dim() - 1)), -1) & (r > 0), (r + float(margin)).clamp(min=1e-4), r)
    return out, 0.0


def _sphere_fit_key(obstacles, num_spheres: int) -> str:
    import hashlib

    h = hashlib.sha256()
    for ob in obstacles:
        h.update(str(ob.name).encode())
        h.update(np.ascontiguousarray(ob.vertices, dtype=np.float32).tobytes())
        h.update(np.ascontiguousarray(ob.faces, dtype=np.int64).tobytes())
        h.update(np.asarray(1.0 if ob.scale is None else ob.scale, dtype=np.float32).tobytes())
        pose = np.asarray(ob.pose, dtype=np.float32)
        if fast_on("ATTACH_FAST"):
            # train-fast 2026-10-31: the mesh pose in the eef frame differs by float noise between two attaches of the same grasp (the object
            # teleported into the hand each time), which made every SEEDED_SPHERE_FIT=cache lookup miss; rounded to 1 mm / 1e-3 quaternion units
            pose = np.round(pose, 3)
        h.update(pose.tobytes())
    h.update(str(int(num_spheres)).encode())
    return h.hexdigest()


def fit_attach_spheres(am, obstacles, num_spheres: int) -> th.Tensor:
    """``am.fit_spheres(obstacles, num_spheres=num_spheres)``, seeded / cached per ISPATIALGYM_SEEDED_SPHERE_FIT."""
    if not SEEDED_SPHERE_FIT:
        return am.fit_spheres(obstacles, num_spheres=num_spheres)
    import random

    key = _sphere_fit_key(obstacles, num_spheres)
    if SEEDED_SPHERE_FIT == "cache" and key in _FIT_CACHE:
        _FIT_COUNTS["hits"] += 1
        return _FIT_CACHE[key].clone()
    seed = int(key[:16], 16)
    t0 = time.perf_counter()
    np_state, py_state = np.random.get_state(), random.getstate()
    with th.random.fork_rng(devices=list(range(th.cuda.device_count()))):
        th.manual_seed(seed)
        np.random.seed(seed % 2**32)
        random.seed(seed)
        try:
            out = am.fit_spheres(obstacles, num_spheres=num_spheres)
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)
    _FIT_COUNTS["fits"] += 1
    _FIT_COUNTS["fit_s"] += time.perf_counter() - t0
    if SEEDED_SPHERE_FIT == "cache":
        _FIT_CACHE[key] = out.detach().clone()
    return out


class _PoseBatch:
    def __init__(self, planner):
        self.planner = planner
        self.names: List[str] = []
        self.pos: List[th.Tensor] = []
        self.quat: List[th.Tensor] = []

    def add(self, name: str, pos: th.Tensor, quat_xyzw: th.Tensor) -> None:
        self.names.append(name)
        self.pos.append(pos)
        self.quat.append(quat_xyzw_to_wxyz(quat_xyzw))

    def write(self) -> None:
        if not self.names:
            return
        dev = self.planner.device_cfg.device
        meshes = self.planner.checker.data.meshes
        first = {}
        for i, n in enumerate(meshes.names[0]):
            first.setdefault(n, i)
        idx = th.tensor([first[n] for n in self.names], dtype=th.long, device=dev)
        vec = CuPose(position=th.stack(self.pos).to(dev), quaternion=th.stack(self.quat).to(dev)).inverse().get_pose_vector()
        _WS_COUNTS["calls"] += 1
        _WS_COUNTS["meshes"] += len(self.names)
        if BATCHED_WORLD_SYNC == "check":
            for k in range(len(self.names)):
                one = CuPose(position=self.pos[k].unsqueeze(0).to(dev), quaternion=self.quat[k].unsqueeze(0).to(dev)).inverse().get_pose_vector()
                if not th.equal(one.reshape(-1), vec[k].reshape(-1)):
                    _WS_COUNTS["mismatches"] += 1
            if _WS_COUNTS["calls"] % 20 == 0 or _WS_COUNTS["mismatches"]:
                print(f"WORLD_SYNC_CHECK {_WS_COUNTS}", flush=True)
        meshes.inv_pose[0, idx, :7] = vec


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------


@dataclass
class PlanResult:
    """Outcome of a planning query.

    ``q_traj`` is a ``(T, 28)`` float tensor of full OmniGibson joint positions (locked joints held
    constant at their current values) sampled every ``dt`` seconds.
    """

    success: bool
    q_traj: Optional[th.Tensor] = None
    dt: float = 1.0 / 30.0
    #: human-readable outcome; on failure it names the stage and the most likely cause
    status: str = ""
    plan_time: float = 0.0
    #: raw numbers behind ``status`` (best IK errors, feasible-seed counts, start-state collision, ...)
    diagnostics: Dict[str, object] = field(default_factory=dict)

    def __bool__(self):
        return bool(self.success)

    @property
    def duration(self) -> float:
        return 0.0 if self.q_traj is None else (len(self.q_traj) - 1) * self.dt


# --------------------------------------------------------------------------------------
# Shared-world MotionPlanner
# --------------------------------------------------------------------------------------


class _SharedWorldMotionPlanner(MotionPlanner):
    """MotionPlanner that reuses an externally created scene collision checker."""

    def __init__(self, config: MotionPlannerCfg, checker):
        self._shared_checker = checker
        super().__init__(config)

    def _initialize_components(self):
        self.scene_collision_checker = self._shared_checker
        self.ik_solver = IKSolver(self.config.ik_solver_config, self.scene_collision_checker)
        self.trajopt_solver = TrajOptSolver(self.config.trajopt_solver_config, self.scene_collision_checker)
        self.graph_planner = None
        if self.config.graph_planner_config is not None:
            self.graph_planner = PRMGraphPlanner(self.config.graph_planner_config, self.scene_collision_checker)


@dataclass
class _PlannerEntry:
    embodiment: M.Embodiment
    planner: MotionPlanner
    #: per locked joint: (child link index in fixed_transforms, origin 4x4, JointType, scale, offset)
    lock_info: Dict[str, tuple] = field(default_factory=dict)
    default_criteria: Dict[str, ToolPoseCriteria] = field(default_factory=dict)
    #: lazily created scene/self collision cost terms for plain collision queries
    rsc: object = None
    #: dedicated Kinematics instance (shares the planner's tensors) for cheap batched FK queries
    fk: object = None


# --------------------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------------------


class R1ProCuRobo:
    """cuRobo v2 planning/kinematics backend bound to one OmniGibson R1Pro instance."""

    def __init__(
        self,
        robot,
        device: str = "cuda:0",
        control_dt: Optional[float] = None,
        # 1 cm for the IK solvers AND the motion planner's trajopt (one MotionPlannerCfg configures both). 2026-09-25 (teacup
        # 0925-222212, wave 3: the pre-grasp transit skimmed the saucer 30 cm beside the teacup and pushed it 20 cm) 2 cm was tried:
        # it changed the sampler's IK solutions as well (wave 5, bar soap 0925-235551: 8/8 attempts ik_joint_limit at the place) and
        # the transit still brushed the saucer (0925-235559, 12 cm) - reverted; the transit needs a via-above leg, not a margin
        collision_activation_distance: float = 0.01,
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.05,
        num_ik_seeds: int = 32,
        num_trajopt_seeds: int = 4,
        mesh_cache_size: int = 8192,  # owner decision 2026-09-16: 8192 (the room-AABB obstacle filter stays; no radius filter); the pilot overflowed 2048 in the kitchen scope
        cuboid_cache_size: int = 16,
        prebuild: Sequence[str] = ("right_arm", "base"),
        use_cuda_graph: bool = True,
        warmup_iterations: int = 3,
        sdf_grid: Optional[Dict[str, object]] = None,
    ):
        """``sdf_grid`` (opt-in, ``reach_bands --collision sdf-grid``): ``{"res": voxel size (m), "dims": [x, y, z] grid extents (m),
        "mesh_cache": capacity of the mesh store}`` - the checker gets a ``len(sdf_grid.LAYERS)``-layer cuRobo voxel store of
        that size (see :mod:`ispatialgym.oracle.sdf_grid`) and callers move static obstacles into it with
        :meth:`load_sdf_layers`; only the objects passing :attr:`mesh_object_filter` are loaded as meshes. ``None`` (default):
        the exact mesh-only world, unchanged."""
        assert robot.model == "r1pro", f"R1ProCuRobo only supports r1pro, got {robot.model}"
        self.robot = robot
        self.device_cfg = DeviceCfg(device=th.device(device), dtype=th.float32)
        pin_pip_warp_after_launch()
        init_warp(device_cfg=self.device_cfg)
        # A planner alternates between pose goals (plan_pose) and joint goals (plan_cspace); with CUDA
        # graphs enabled cuRobo must re-capture its graphs on such a switch, which it only allows
        # when this runtime flag is set (requires CUDA >= 12).
        import curobo._src.runtime as _curobo_runtime_src
        import curobo.runtime as _curobo_runtime  # the public module is what cuRobo's helpers read

        _curobo_runtime_src.cuda_graph_reset = True
        _curobo_runtime.cuda_graph_reset = True
        self.control_dt = float(og.sim.get_sim_step_dt() if control_dt is None else control_dt)
        self.collision_activation_distance = collision_activation_distance
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.num_ik_seeds = num_ik_seeds
        self.num_trajopt_seeds = num_trajopt_seeds
        # 2026-10-31 train-fast (BATCH_SEEDS): seeds of the batched screening solvers (``_batch_ik_entry``), 12 as before unless
        # ISPATIALGYM_FAST_BATCH_IK_SEEDS=<n>; a caller may still ask for other seeds per call (``solve_ik_batch(seeds=...)``)
        self.batch_ik_seeds = int(fast_value("BATCH_IK_SEEDS", 12))
        self.use_cuda_graph = use_cuda_graph
        self.warmup_iterations = int(os.environ.get("ISPATIALGYM_CUROBO_WARMUP_ITERS", warmup_iterations))

        self.joint_names: List[str] = list(robot.joints.keys())
        assert self.joint_names == M.ALL_JOINTS, "Unexpected R1Pro joint order"
        #: links that legitimately touch / sit within ~1 cm of the floor (wheels, base body). Their
        #: proximity to *ground-like* obstacles is not counted as a violation of margin queries; against
        #: everything else (walls, furniture) they are treated like any other link.
        self.ground_contact_links = set(robot.floor_touching_base_link_names) | {M.BASE_LINK}
        self._ground_mesh_names: set = set()
        self._jidx = {n: i for i, n in enumerate(self.joint_names)}

        # Derived URDF + v1 sphere data
        self.urdf_path = M.generate_holonomic_urdf()
        self._v1_kin = M.load_v1_kinematics_yaml()
        self._joint_child = self._parse_joint_children(self.urdf_path)
        self._default_q = {n: float(v) for n, v in zip(self.joint_names, robot.reset_joint_pos)}

        # Shared world. NOTE: cuRobo 0.8 launches its sphere-vs-mesh kernel over spheres x ``mesh_cache_size`` threads (the
        # cache capacity, not the number of loaded meshes; ``wp_autograd.py``: ``dim = b * h * n * max_n``), so an oversized
        # cache is paid on every collision evaluation: measured 2026-09-16, a 128-pose x 12-seed batched IK call with 28
        # enabled meshes took 8.0 s with the 8192 cache and 5.2 s with *no* enabled mesh at all. Callers that know their
        # world (reach_bands.load_world) pass a cache sized to the loaded meshes.
        self.mesh_cache_size = int(mesh_cache_size)
        cache = {"mesh": self.mesh_cache_size, "cuboid": cuboid_cache_size}
        self.sdf_grid = dict(sdf_grid) if sdf_grid else None
        if self.sdf_grid:
            from ispatialgym.oracle import sdf_grid as SG

            self.mesh_cache_size = int(self.sdf_grid.get("mesh_cache", self.mesh_cache_size))
            cache = {"mesh": self.mesh_cache_size, "cuboid": cuboid_cache_size,
                     "voxel": {"layers": len(SG.LAYERS), "dims": [float(v) for v in self.sdf_grid["dims"]], "voxel_size": float(self.sdf_grid["res"])}}
        self.checker = create_collision_checker(
            SceneCollisionCfg(
                device_cfg=self.device_cfg,
                scene_model=None,
                cache=cache,
            )
        )
        self._world_loaded = False
        self._world_mesh_names: set = set()
        self._disabled_obstacles: set = set()
        self._attached: Dict[str, dict] = {}  # arm -> info
        #: optional ``callable(obj) -> bool``: scene objects for which it returns False are never loaded as obstacles
        #: (e.g. furniture of neighbouring rooms that only exists for rendering; the mesh cache holds ``mesh_cache_size``)
        self.obstacle_filter = None
        #: optional ``callable(obj) -> bool`` (sdf-grid mode): objects for which it returns False are not loaded into the *mesh*
        #: store - they are expected to be represented by the voxel layers (:meth:`load_sdf_layers`) or to be irrelevant
        self.mesh_object_filter = None
        #: voxel layer name -> {"objects": [scene object names], "ground_like": bool} (sdf-grid mode, :meth:`load_sdf_layers`)
        self._grid_layers: Dict[str, Dict[str, object]] = {}
        self._grid_centre_world: Optional[np.ndarray] = None

        self._entries: Dict[str, _PlannerEntry] = {}
        self._kin_entries: Dict[str, _PlannerEntry] = {}
        for name in prebuild:
            self.get_entry(name)

    # ------------------------------------------------------------------ construction

    @staticmethod
    def _parse_joint_children(urdf_path: str) -> Dict[str, str]:
        root = ET.parse(urdf_path).getroot()
        return {j.get("name"): j.find("child").get("link") for j in root.findall("joint")}

    def get_entry(self, emb_name: str) -> _PlannerEntry:
        """Entry with a full MotionPlanner (built lazily, ~10 s)."""
        if emb_name not in self._entries:
            self._entries[emb_name] = self._build_entry(M.EMBODIMENTS[emb_name])
            self._kin_entries.pop(emb_name, None)
        return self._entries[emb_name]

    def kin_entry(self, emb_name: str) -> _PlannerEntry:
        """Entry for FK / collision queries only: reuses the planner if built, else a cheap
        kinematics-only model (no optimizers)."""
        if emb_name in self._entries:
            return self._entries[emb_name]
        if emb_name not in self._kin_entries:
            emb = M.EMBODIMENTS[emb_name]
            q = self.q_full()
            lock_values = {n: float(q[self._jidx[n]]) for n in emb.locked_joints}
            robot_dict = M.build_curobo_robot_dict(emb, self.urdf_path, lock_values, self._default_q, self._v1_kin)
            kin = Kinematics(KinematicsCfg.from_data_dict(robot_dict, device_cfg=self.device_cfg))
            entry = _PlannerEntry(embodiment=emb, planner=None, fk=kin)
            self._fill_lock_info(entry, kin)
            self._kin_entries[emb_name] = entry
        return self._kin_entries[emb_name]

    def planner(self, emb_name: str) -> MotionPlanner:
        return self.get_entry(emb_name).planner

    def _fill_lock_info(self, entry: _PlannerEntry, kin: Kinematics):
        kc = kin.kinematics_config
        parser = kin.config.kinematics_parser
        for jn in entry.embodiment.locked_joints:
            child = self._joint_child[jn]
            if child not in kc.link_name_to_idx_map:
                continue  # link not part of the kinematic tree
            lp = parser.get_link_parameters(child)
            origin = np.eye(4, dtype=np.float32)
            origin[:3, :4] = np.asarray(lp.fixed_transform, dtype=np.float32)
            entry.lock_info[jn] = (
                kc.link_name_to_idx_map[child],
                th.as_tensor(origin, device=self.device_cfg.device),
                lp.joint_type,
                float(lp.joint_offset[0]),
                float(lp.joint_offset[1]),
            )

    def _build_entry(self, emb: M.Embodiment) -> _PlannerEntry:
        t0 = time.time()
        q = self.q_full()
        lock_values = {n: float(q[self._jidx[n]]) for n in emb.locked_joints}
        robot_dict = M.build_curobo_robot_dict(emb, self.urdf_path, lock_values, self._default_q, self._v1_kin)
        t_dict = time.time()
        cfg = MotionPlannerCfg.create(
            robot=robot_dict,
            scene_model=None,
            collision_cache=None,
            self_collision_check=True,
            device_cfg=self.device_cfg,
            num_ik_seeds=self.num_ik_seeds,
            num_trajopt_seeds=self.num_trajopt_seeds,
            position_tolerance=self.position_tolerance,
            orientation_tolerance=self.orientation_tolerance,
            use_cuda_graph=self.use_cuda_graph,
            optimizer_collision_activation_distance=self.collision_activation_distance,
        )
        cfg.trajopt_solver_config.interpolation_dt = self.control_dt
        t_cfg = time.time()
        planner = _SharedWorldMotionPlanner(cfg, self.checker)
        t_solvers = time.time()
        if self.warmup_iterations > 0:
            planner.warmup(enable_graph=True, num_warmup_iterations=self.warmup_iterations)
        t_warm = time.time()

        entry = _PlannerEntry(embodiment=emb, planner=planner)
        entry.fk = Kinematics(planner.kinematics.config)  # same KinematicsCfg -> shares tensors with the solvers
        self._fill_lock_info(entry, entry.fk)
        entry.default_criteria = {f: ToolPoseCriteria(device_cfg=self.device_cfg) for f in planner.tool_frames}
        t1 = time.time()
        log.info("Built cuRobo embodiment '%s' (%d active joints) in %.1fs (robot dict %.1fs, config %.1fs, solvers %.1fs, warm-up %.1fs x%d, fk %.1fs)",
                 emb.name, len(emb.active_joints), t1 - t0, t_dict - t0, t_cfg - t_dict, t_solvers - t_cfg, t_warm - t_solvers, self.warmup_iterations,
                 t1 - t_warm)
        return entry

    def _kin_state(self, entry: _PlannerEntry, q_batch: th.Tensor):
        """Batched FK (root frame) through the entry's dedicated Kinematics instance."""
        return entry.fk.compute_kinematics(self._active_js(entry, q_batch))

    # ------------------------------------------------------------------ state helpers

    def q_full(self) -> th.Tensor:
        """Current 28-D joint positions with the base yaw wrapped to [-pi, pi]."""
        q = self.robot.get_joint_positions().clone().float()
        rz = self._jidx["base_footprint_rz_joint"]
        q[rz] = (q[rz] + math.pi) % (2 * math.pi) - math.pi
        return q

    def root_pose(self) -> Pose:
        pos, quat = self.robot.root_link.get_position_orientation()
        return pos.float(), quat.float()

    def world_to_root(self, pose: Pose) -> Pose:
        rp, rq = self.root_pose()
        return T.relative_pose_transform(as_tensor(pose[0]), as_tensor(pose[1]), rp, rq)

    def root_to_world(self, pose: Pose) -> Pose:
        rp, rq = self.root_pose()
        return T.pose_transform(rp, rq, as_tensor(pose[0]), as_tensor(pose[1]))

    def _cu_pose_from_world(self, poses: Sequence[Pose]) -> CuPose:
        # one simulator read of the root pose for the whole batch (it was read per pose: 128 PhysX view reads = 40 ms of a
        # 128-pose batched IK call, measured 2026-09-16); the per-pose arithmetic is unchanged, so results are identical
        root_p, root_q = self.root_pose()
        pos, quat = [], []
        for p in poses:
            rp, rq = T.relative_pose_transform(as_tensor(p[0]), as_tensor(p[1]), root_p, root_q)
            pos.append(rp)
            quat.append(quat_xyzw_to_wxyz(rq))
        return CuPose(
            position=th.stack(pos).to(self.device_cfg.device),
            quaternion=th.stack(quat).to(self.device_cfg.device),
        )

    def _js_full(self, q_full: th.Tensor) -> JointState:
        q = as_tensor(q_full)
        if q.ndim == 1:
            q = q.unsqueeze(0)
        return JointState.from_position(q.to(self.device_cfg.device), joint_names=list(self.joint_names))

    def _active_js(self, entry: _PlannerEntry, q_full: th.Tensor) -> JointState:
        return entry.fk.get_active_js(self._js_full(q_full))

    def _full_from_active(self, entry: _PlannerEntry, q_active: th.Tensor, active_names: List[str], q_ref: th.Tensor) -> th.Tensor:
        """Expand ``(T, n_active)`` active-joint positions to ``(T, 28)`` using ``q_ref`` for the rest."""
        q_active = q_active.detach().cpu().float().reshape(-1, len(active_names))
        out = as_tensor(q_ref).clone().unsqueeze(0).repeat(q_active.shape[0], 1)
        for i, n in enumerate(active_names):
            out[:, self._jidx[n]] = q_active[:, i]
        return out

    # ------------------------------------------------------------------ locked joints

    def refresh_locked_joints(self, emb_name: str, q_full: Optional[th.Tensor] = None):
        """Write the current values of the locked joints into the planner's fixed transforms."""
        entry = self.kin_entry(emb_name)
        q_full = self.q_full() if q_full is None else as_tensor(q_full)
        kc = entry.fk.kinematics_config
        lock_js = kc.lock_jointstate
        for jn, (link_idx, origin, jtype, scale, offset) in entry.lock_info.items():
            val = float(q_full[self._jidx[jn]]) * scale + offset
            kc.fixed_transforms[link_idx] = _lock_transform(origin, jtype, val, self.device_cfg.device)
        if lock_js is not None:
            for i, jn in enumerate(lock_js.joint_names):
                lock_js.position[..., i] = float(q_full[self._jidx[jn]])

    # ------------------------------------------------------------------ world

    #: BEHAVIOR object categories the robot stands / drives on. Wheels and base body may be within the
    #: proximity margin of these without it counting as a collision risk.
    GROUND_CATEGORIES = {"floors", "carpet", "mat", "gym_mat", "yoga_mat", "lawn", "driveway", "ceramic_tile"}
    GROUND_SLAB_NAME = "__ground__"

    def _is_ground_like(self, obj) -> bool:
        return getattr(obj, "category", None) in self.GROUND_CATEGORIES

    def _iter_collision_meshes(self, ignore_objects=(), ignore_links=(), objects=None):
        """``objects``: restrict the iteration to these scene objects (a partial world update)."""
        ignore = set()
        for o in ignore_objects or ():
            ignore.add(o if isinstance(o, str) else o.name)
        # (object name, link name) pairs left out of the world (2026-09-20: the flap a fist presses while the other hand
        # plans past the box - the box body and the other flap stay obstacles)
        ign_links, ign_meshes = set(), set()
        for ent in (ignore_links or ()):
            o = ent[0] if isinstance(ent[0], str) else ent[0].name
            if len(ent) >= 3 and ent[2]:
                ign_meshes.add((o, ent[1], str(ent[2])))  # one collision piece of a link (the wall a braced blade stands against, 2026-09-21)
            else:
                ign_links.add((o, ent[1]))
        rp, rq = self.root_pose()
        for obj in (self.robot.scene.objects if objects is None else objects):
            if obj is self.robot or getattr(obj, "visual_only", False) or obj.name in ignore:
                continue
            if self.obstacle_filter is not None and not self.obstacle_filter(obj):
                continue
            if self.mesh_object_filter is not None and not self.mesh_object_filter(obj):
                continue
            for link_name, link in obj.links.items():
                if (obj.name, link_name) in ign_links:
                    continue
                for cm in link.collision_meshes.values():
                    if ign_meshes and (obj.name, link_name, str(cm.prim_path).rsplit("/", 1)[-1]) in ign_meshes:
                        continue
                    pos, quat = cm.get_position_orientation()
                    pos, quat = T.relative_pose_transform(pos, quat, rp, rq)
                    if self._is_ground_like(obj):
                        self._ground_mesh_names.add(cm.prim_path)
                    yield obj, link, cm, (pos, quat)

    def _mesh_geometry(self, cm):
        # train-fast 2026-10-31 (ATTACH_FAST): a collision mesh's vertices / faces never change in a process, yet every attach_object
        # (6-8 per sampler attempt; every carried action of the oracle) re-read them through USD - cached per prim path; the world
        # scale is read live (an object scaling task sets it at spawn).
        cache = self.__dict__.setdefault("_mesh_geom_cache", {}) if fast_on("ATTACH_FAST") else None
        key = getattr(cm, "prim_path", None)
        if cache is not None and key in cache:
            verts, faces = cache[key]
        else:
            if cm.geom_type == "Mesh":
                verts = cm.points.numpy().astype(np.float32)
                faces = cm.faces.numpy().astype(np.int64)
            else:
                tm = mesh_prim_shape_to_trimesh_mesh(cm.prim)
                verts = np.asarray(tm.vertices, dtype=np.float32)
                faces = np.asarray(tm.faces, dtype=np.int64)
            if cache is not None and key is not None:
                cache[key] = (verts, faces)
        scale = cm.get_world_scale().numpy().astype(np.float32)
        return verts, faces, scale

    def update_world(self, ignore_objects=(), ground: bool = True, ignore_links=(), only_objects=None):
        """Synchronise the collision world with the simulator.

        First call loads every collision mesh (Warp BVH build, slow); later calls only update
        poses of already-known meshes and add new ones. Meshes of ignored / removed objects are
        disabled rather than deleted so the Warp cache stays valid.

        ``only_objects`` (a loaded world): refresh the poses of these objects' meshes alone and leave the rest as
        they are - the articulation search poses one joint hundreds of times while nothing else moves, and a full
        pass over the house's 4155 meshes cost ~5 s each time (door 0921-041232: 30 s per look-ahead candidate).
        """
        t0 = time.time()
        if self._world_loaded and only_objects is not None:
            batch = self._pose_batch() if BATCHED_WORLD_SYNC else None
            for obj, link, cm, (pos, quat) in self._iter_collision_meshes(ignore_objects, ignore_links, objects=list(only_objects)):
                name = cm.prim_path
                if name in self._world_mesh_names:
                    if batch is not None:
                        batch.add(name, pos, quat)
                        continue
                    self._mesh_update_pose(name, CuPose(position=pos.unsqueeze(0).to(self.device_cfg.device),
                                                        quaternion=quat_xyzw_to_wxyz(quat).unsqueeze(0).to(self.device_cfg.device)))
            if batch is not None:
                batch.write()
            return
        if not self._world_loaded:
            meshes, names = [], set()
            for obj, link, cm, (pos, quat) in self._iter_collision_meshes(ignore_objects, ignore_links):
                verts, faces, scale = self._mesh_geometry(cm)
                meshes.append(
                    Mesh(
                        name=cm.prim_path,
                        pose=th.cat([pos, quat_xyzw_to_wxyz(quat)]).tolist(),
                        vertices=verts,
                        faces=faces,
                        scale=scale,
                    )
                )
                names.add(cm.prim_path)
            cuboids = []
            if ground and og.sim.floor_plane is not None:
                rp, rq = self.root_pose()
                # Thin slab whose top face is the world ground plane (z=0, optionally lowered by
                # GROUND_CLEARANCE), expressed in the root frame.
                gpos, gquat = T.relative_pose_transform(th.tensor([0.0, 0.0, -0.05 - self.GROUND_CLEARANCE]), th.tensor([0.0, 0.0, 0.0, 1.0]), rp, rq)
                cuboids.append(Cuboid(name=self.GROUND_SLAB_NAME, dims=[60.0, 60.0, 0.1], pose=th.cat([gpos, quat_xyzw_to_wxyz(gquat)]).tolist()))
            scene = Scene(mesh=meshes, cuboid=cuboids if cuboids else None)
            self.checker.load_collision_model(scene)
            self._patch_mesh_query_dims()
            for e in self._entries.values():
                if e.planner.graph_planner is not None:
                    e.planner.graph_planner.reset_buffer()
            self._world_mesh_names = names
            self._world_loaded = True
            self._grid_update_poses()
            log.info("cuRobo world loaded: %d meshes in %.2fs", len(meshes), time.time() - t0)
            return

        seen = set()
        batch = self._pose_batch() if BATCHED_WORLD_SYNC else None
        for obj, link, cm, (pos, quat) in self._iter_collision_meshes(ignore_objects, ignore_links):
            name = cm.prim_path
            seen.add(name)
            if batch is not None and name in self._world_mesh_names:
                batch.add(name, pos, quat)
                if name in self._disabled_obstacles and name not in self._attached_mesh_names():
                    self._mesh_enable(name, True)
                    self._disabled_obstacles.discard(name)
                continue
            cu_pose = CuPose(
                position=pos.unsqueeze(0).to(self.device_cfg.device),
                quaternion=quat_xyzw_to_wxyz(quat).unsqueeze(0).to(self.device_cfg.device),
            )
            if name in self._world_mesh_names:
                self._mesh_update_pose(name, cu_pose)
                if name in self._disabled_obstacles and name not in self._attached_mesh_names():
                    self._mesh_enable(name, True)
                    self._disabled_obstacles.discard(name)
            else:
                verts, faces, scale = self._mesh_geometry(cm)
                self.checker.data.meshes.add(
                    Mesh(name=name, pose=th.cat([pos, quat_xyzw_to_wxyz(quat)]).tolist(), vertices=verts, faces=faces, scale=scale)
                )
                self._world_mesh_names.add(name)
        if batch is not None:
            batch.write()
        for name in self._world_mesh_names - seen:
            if name not in self._disabled_obstacles:
                self._mesh_enable(name, False)
                self._disabled_obstacles.add(name)
        self._patch_mesh_query_dims()
        self._grid_update_poses()
        for e in self._entries.values():
            if e.planner.graph_planner is not None:
                e.planner.graph_planner.reset_buffer()

    # ------------------------------------------------------------------ sdf-grid mode (opt-in; see ispatialgym.oracle.sdf_grid)

    def reload_mesh_store(self, ignore_objects=()):
        """Empty the mesh store (in place: the solvers' CUDA graphs keep their buffers) and load it again from the simulator
        through :attr:`mesh_object_filter`; the voxel layers are dropped too (``load_collision_model`` clears every store) and
        must be loaded again with :meth:`load_sdf_layers`. sdf-grid mode only: the store is small and holds one band's
        movable objects; the worker calls this between bands."""
        assert self.sdf_grid, "reload_mesh_store is for the sdf-grid mode"
        self.checker.data.meshes.clear(clear_warp_cache=True)
        self._world_loaded = False
        self._world_mesh_names = set()
        self._disabled_obstacles = set()
        self._ground_mesh_names = set()
        self._grid_layers = {}
        self.update_world(ignore_objects=ignore_objects)

    def _grid_pose_root(self, centre_world) -> List[float]:
        """``[x, y, z, qw, qx, qy, qz]`` of the world-axis-aligned grid centred at ``centre_world`` in the cuRobo (root) frame."""
        rp, rq = self.root_pose()
        c = th.as_tensor(np.asarray(centre_world, dtype=np.float64), dtype=th.float32)
        p, q = T.relative_pose_transform(c, th.tensor([0.0, 0.0, 0.0, 1.0]), rp, rq)
        return th.cat([p, quat_xyzw_to_wxyz(q)]).tolist()

    def _grid_update_poses(self):
        """Re-pose the voxel layers relative to the robot root (they are defined in the world frame)."""
        if not self._grid_layers or self._grid_centre_world is None:
            return
        pose = self._grid_pose_root(self._grid_centre_world)
        dev = self.device_cfg.device
        cu = CuPose(position=th.tensor(pose[:3], device=dev).reshape(1, 3), quaternion=th.tensor(pose[3:], device=dev).reshape(1, 4))
        for name in self._grid_layers:
            self.checker.data.voxels.update_pose(name, w_obj_pose=cu)

    def load_sdf_layers(self, layers: Dict[str, Sequence[object]], centre_world, cache_dir: Optional[str], scene_name: str) -> Dict[str, object]:
        """Represent the collision meshes of ``layers[layer name] = [scene objects]`` by one signed-distance voxel layer each
        (:data:`sdf_grid.LAYERS`; a layer with no objects is left disabled), sampled with cuRobo's own mesh SDF at the voxel
        centres of the grid of :attr:`sdf_grid` centred (x, y) at ``centre_world`` (world frame; z from the config), cached
        on disk under ``cache_dir`` by scene, grid, and the meshes' prim paths / poses / geometry. The objects must not be in
        the mesh store (see :attr:`mesh_object_filter`). Returns the meta record (per layer: objects, meshes, cache hit, build
        time, file) for the band's ``schedule.collision``."""
        from ispatialgym.oracle import sdf_grid as SG
        from curobo._src.geom.types import VoxelGrid

        assert self.sdf_grid, "load_sdf_layers needs a backend built with sdf_grid=..."
        t0 = time.time()
        res = float(self.sdf_grid["res"])
        n = np.asarray(self.sdf_grid["n"], dtype=int)
        centre = np.asarray(centre_world, dtype=np.float64)
        pose_root = self._grid_pose_root(centre)
        grids, meta_layers = [], {}
        mesh_store = set(self._world_mesh_names)
        for lname in SG.LAYERS:
            objs = list(layers.get(lname, []) or [])
            meshes, records, obj_names = [], [], []
            for obj in objs:
                obj_names.append(obj.name)
                for link in obj.links.values():
                    for cm in link.collision_meshes.values():
                        if cm.prim_path in mesh_store:
                            raise ValueError(f"{cm.prim_path} is in the mesh store and cannot be gridded too (mesh_object_filter)")
                        verts, faces, scale = self._mesh_geometry(cm)
                        pos, quat = cm.get_position_orientation()
                        pose7 = th.cat([pos.float(), quat_xyzw_to_wxyz(quat.float())]).tolist()
                        meshes.append({"name": cm.prim_path, "pose": pose7, "scale": [float(v) for v in scale], "vertices": verts, "faces": faces})
                        records.append(SG.mesh_record(cm.prim_path, pose7, scale, verts, faces))
            rec = {"objects": obj_names, "n_meshes": len(meshes), "ground_like": bool(objs) and all(self._is_ground_like(o) for o in objs)}
            if not meshes:
                rec["enabled"] = False
                meta_layers[lname] = rec
                continue
            key = SG.cache_key(scene_name, res, n, centre, records, self.MESH_QUERY_MIN_DIMS)
            feat = SG.load_cached(cache_dir, key, n, self.device_cfg.device) if cache_dir else None
            rec.update({"key": key, "cache_hit": feat is not None})
            if feat is None:
                t1 = time.time()
                feat, st = SG.sample_layer(meshes, n, res, centre, self.device_cfg, self.MESH_QUERY_MIN_DIMS)
                rec.update({"sample": st, "build_s": round(time.time() - t1, 3)})
                if cache_dir:
                    path, size = SG.save_cached(cache_dir, key, feat, {"scene": scene_name, "layer": lname, "res": res, "n": n.tolist(), "centre": centre.tolist(),
                                                                      "min_dims": self.MESH_QUERY_MIN_DIMS, "far": SG.FAR, "objects": obj_names, "meshes": records, "sample": st})
                    rec.update({"file": path, "file_bytes": size})
            else:
                rec["file"] = SG.cache_paths(cache_dir, key)[0]
            rec.update(SG.layer_stats(feat, res))
            rec["enabled"] = True
            grids.append(VoxelGrid(name=lname, pose=list(pose_root), dims=[float(v) for v in self.sdf_grid["dims"]], voxel_size=res, feature_tensor=feat))
            meta_layers[lname] = rec
        vox = self.checker.data.voxels
        vox.load_batch(grids, 0)
        params = vox.params[0, :len(grids)].detach().cpu().numpy()
        assert all(int(params[l, k]) == int(n[k]) for l in range(len(grids)) for k in range(3)), (params, n)
        self._grid_layers = {g.name: {"objects": list(meta_layers[g.name]["objects"]), "ground_like": bool(meta_layers[g.name]["ground_like"])} for g in grids}
        self._grid_centre_world = centre
        lo, hi = SG.grid_bounds(n, res, centre)
        meta = {"type": "sdf_grid", "res": res, "n": n.tolist(), "n_voxels": int(np.prod(n)), "bounds_world": [lo.tolist(), hi.tolist()], "centre_world": centre.tolist(),
                "features": "float16", "far": SG.FAR, "layers": meta_layers, "mesh_cache": self.mesh_cache_size, "load_s": round(time.time() - t0, 3)}
        log.info("sdf-grid world: res %.3f m, %s voxels/layer, %d layers (%s), mesh store %d/%d in %.1fs", res, "x".join(str(int(v)) for v in n), len(grids),
                 ", ".join(f"{g.name}: {meta_layers[g.name]['n_meshes']} meshes of {len(meta_layers[g.name]['objects'])} objects" + (" [cached]" if meta_layers[g.name]["cache_hit"] else "") for g in grids),
                 len(self._world_mesh_names), self.mesh_cache_size, time.time() - t0)
        return meta

    def grid_layer_enable(self, name: str, enabled: bool):
        if name in self._grid_layers:
            self.checker.data.voxels.set_enabled(name, enabled)

    def grid_layer_of_object(self, obj_name: str) -> Optional[str]:
        for lname, info in self._grid_layers.items():
            if obj_name in info["objects"]:
                return lname
        return None

    def grid_object_enable(self, obj_name: str, enabled: bool):
        """Enable / disable the voxel layer holding ``obj_name`` (no-op when the object is not gridded). Note a layer holds
        every object listed for it, so callers keep objects toggled separately in their own layer (the band's support)."""
        lname = self.grid_layer_of_object(obj_name)
        if lname is not None:
            self.checker.data.voxels.set_enabled(lname, enabled)

    # cuRobo 0.8.0's SceneData routers (update_obstacle_pose / enable_obstacle) only catch
    # RuntimeError while the cuboid store raises ValueError for unknown names, so mesh obstacles
    # are addressed through the mesh store directly.
    def _mesh_update_pose(self, name: str, pose: CuPose):
        self.checker.data.meshes.update_pose(name, w_obj_pose=pose)

    def _pose_batch(self) -> "_PoseBatch":
        return _PoseBatch(self)

    def _mesh_enable(self, name: str, enabled: bool):
        self.checker.data.meshes.set_enabled(name, enabled)

    #: Minimum bounding-box extent (m) assumed for every mesh obstacle in the SDF query (see
    #: ``_patch_mesh_query_dims``). The query radius is half the box diagonal, 0.35 m here, which must
    #: exceed the largest robot sphere (0.18 m) plus buffer and the largest activation/margin used
    #: (0.03 m); it also leaves room for sphere-fitted grasped objects.
    MESH_QUERY_MIN_DIMS = 0.4
    #: Distance (m) the ground slab is lowered below z=0. Zero: the floor is modelled where it is,
    #: because floor-contact links are exempt from proximity margins (see ``check_collision``).
    GROUND_CLEARANCE = 0.0

    def _patch_mesh_query_dims(self):
        """Work around a cuRobo 0.8.0 mesh-SDF quirk for small meshes.

        ``compute_local_sdf_with_grad`` queries ``wp.mesh_query_point`` with ``max_distance`` equal
        to half the mesh bounding-box diagonal and, when nothing is found within that radius,
        returns ``max_distance`` as the signed distance. For a robot sphere whose radius exceeds
        that half-diagonal (e.g. a 15 cm base sphere vs a 5 cm object) this yields a positive
        penetration regardless of where the sphere is, i.e. phantom collisions everywhere.
        The stored ``dims`` are only used to size that query, so inflating them keeps the SDF exact
        and simply makes the query look further.
        """
        meshes = getattr(self.checker.data, "meshes", None)
        if meshes is None or meshes.dims is None:
            return
        with th.no_grad():
            d = meshes.dims[..., :3]
            d.clamp_(min=self.MESH_QUERY_MIN_DIMS)

    def _attached_mesh_names(self) -> set:
        out = set()
        for info in self._attached.values():
            out |= set(info["mesh_names"])
        return out

    # ------------------------------------------------------------------ attachment

    def attach_object(self, arm: str, obj, emb_names: Optional[Sequence[str]] = None, num_spheres: Optional[int] = None):
        """Attach ``obj``'s collision geometry to the ``arm`` end-effector in the given planners.

        Spheres are fitted once in the eef frame and written to ``attached_object_<arm>_eef_link``
        of every planner in ``emb_names`` (default: all built planners that know that link). The
        object's world obstacles are disabled.
        """
        eef_pose = self.robot.eef_links[arm].get_position_orientation()
        obstacles, mesh_names = [], []
        for link in obj.links.values():
            for cm in link.collision_meshes.values():
                verts, faces, scale = self._mesh_geometry(cm)
                pos, quat = cm.get_position_orientation()
                pos, quat = T.relative_pose_transform(pos, quat, eef_pose[0], eef_pose[1])  # mesh in eef frame
                obstacles.append(Mesh(name=cm.prim_path, pose=th.cat([pos, quat_xyzw_to_wxyz(quat)]).tolist(), vertices=verts, faces=faces, scale=scale))
                mesh_names.append(cm.prim_path)
        if not obstacles:
            log.warning("attach_object: %s has no collision meshes", obj.name)
            return

        from curobo._src.collision.attachment_manager import AttachmentManager

        link_name = M.ATTACHED_OBJECT_LINK[arm]
        targets = list(emb_names) if emb_names else list(self._entries) + [n for n in self._kin_entries if n not in self._entries]
        sphere_tensor = None
        for name in targets:
            entry = self.kin_entry(name)
            kc = entry.fk.kinematics_config
            if link_name not in kc.link_name_to_idx_map:
                continue
            # operates on the KinematicsParams shared by all solvers of this planner
            am = AttachmentManager(kinematics=entry.fk, scene_collision=self.checker, device_cfg=self.device_cfg)
            if sphere_tensor is None:
                sphere_tensor = fit_attach_spheres(am, obstacles, num_spheres or M.ATTACHED_OBJECT_NUM_SPHERES)
            am.update(sphere_tensor, self._active_js(entry, self.q_full()), link_name=link_name, world_objects_pose_offset=None)
        for n in mesh_names:
            if n in self._world_mesh_names:
                self._mesh_enable(n, False)
                self._disabled_obstacles.add(n)
        self._attached[arm] = {"obj": obj, "mesh_names": mesh_names, "planners": targets, "link": link_name,
                               "spheres": None if sphere_tensor is None else sphere_tensor.detach().cpu().clone()}  # (n, 4): x y z r in the link frame (diagnostics)
        log.info("Attached %s to %s (%d spheres)", obj.name, link_name, 0 if sphere_tensor is None else sphere_tensor.shape[0])

    def detach_object(self, arm: str):
        info = self._attached.pop(arm, None)
        if info is None:
            return
        for name in info["planners"]:
            entry = self._entries.get(name) or self._kin_entries.get(name)
            if entry is None:
                continue
            kc = entry.fk.kinematics_config
            if info["link"] in kc.link_name_to_idx_map:
                kc.reset_link_spheres(info["link"])
        for n in info["mesh_names"]:
            if n in self._world_mesh_names:
                self._mesh_enable(n, True)
                self._disabled_obstacles.discard(n)

    # ------------------------------------------------------------------ kinematics

    def fk(self, frames: Sequence[str], q_full: Optional[th.Tensor] = None, emb_name: str = "whole_body") -> Dict[str, Pose]:
        """World-frame poses of ``frames`` at ``q_full`` (default: current state).

        ``frames`` must be tool frames of the embodiment (``whole_body`` knows both eefs, the head
        camera link, ``base_link`` and ``torso_link4``).
        """
        entry = self.kin_entry(emb_name)
        q_full = self.q_full() if q_full is None else as_tensor(q_full)
        self.refresh_locked_joints(emb_name, q_full)
        state = self._kin_state(entry, q_full)
        out = {}
        for f in frames:
            p = state.tool_poses.get_link_pose(f)
            pos = p.position.reshape(-1, 3)[0].cpu()
            quat = quat_wxyz_to_xyzw(p.quaternion.reshape(-1, 4)[0].cpu())
            out[f] = self.root_to_world((pos, quat))
        return out

    def fk_batch(self, frame: str, q_batch: th.Tensor, emb_name: str = "whole_body") -> Tuple[th.Tensor, th.Tensor]:
        """World poses ``(N,3), (N,4 xyzw)`` of ``frame`` for a batch ``(N,28)`` of full configurations."""
        entry = self.kin_entry(emb_name)
        q_batch = as_tensor(q_batch)
        self.refresh_locked_joints(emb_name, q_batch[0])
        state = self._kin_state(entry, q_batch)
        p = state.tool_poses.get_link_pose(frame)
        pos = p.position.reshape(-1, 3).cpu()
        quat = quat_wxyz_to_xyzw(p.quaternion.reshape(-1, 4).cpu())
        rp, rq = self.root_pose()
        out_pos, out_quat = [], []
        for i in range(pos.shape[0]):
            wp, wq = T.pose_transform(rp, rq, pos[i], quat[i])
            out_pos.append(wp)
            out_quat.append(wq)
        return th.stack(out_pos), th.stack(out_quat)

    def _collision_costs(self, entry: _PlannerEntry):
        """Lazily create the self-collision cost term and a per-sphere world-collision buffer for the entry."""
        if entry.rsc is None:
            from curobo._src.cost.cost_self_collision import SelfCollisionCost
            from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg

            kin = entry.fk
            self_cfg = SelfCollisionCostCfg(weight=1.0, device_cfg=self.device_cfg, self_collision_kin_config=kin.get_self_collision_config())
            kc = kin.kinematics_config
            inv = {v: k for k, v in kc.link_name_to_idx_map.items()}
            link_of_sphere = [inv.get(int(i), "") for i in kc.link_sphere_idx_map.cpu().tolist()]
            ground_contact = th.tensor([l in self.ground_contact_links for l in link_of_sphere], device=self.device_cfg.device)
            entry.rsc = {"self": SelfCollisionCost(self_cfg), "batch": 0, "buffer": None, "ground_contact": ground_contact, "link_of_sphere": link_of_sphere}
        return entry.rsc

    def link_proximity(self, q_batch: th.Tensor, links: Sequence[str], margin: float, emb_name: str = "whole_body", sync_world: bool = False) -> th.Tensor:
        """``(N,)`` bool tensor: True where a sphere of one of ``links`` lies within ``margin`` (m) of a world obstacle (the ignored
        / attached obstacles excluded as in :meth:`check_collision`). A per-link proximity check: the block-design sampler keeps
        the ARM links (not the pinching fingers, which work 1 mm above the table by design) clear of the table (demo
        0919-164732: ``right_arm_link5`` brushed the table during a 45 deg pitched pinch the margin-0 checks had passed).
        A negative ``margin`` lets those spheres penetrate an obstacle by up to ``-margin``: their radii shrink by it (to >= 0.1 mm)
        and only penetration counts (the sampler's approach finger-support touch, owner 2026-10-13)."""
        from curobo._src.geom.collision.buffer_collision import CollisionBuffer
        from curobo._src.robot.kinematics.kinematics_state import KinematicsState

        entry = self.kin_entry(emb_name)
        if sync_world:
            self.update_world()
        q_batch = as_tensor(q_batch)
        if q_batch.ndim == 1:
            q_batch = q_batch.unsqueeze(0)
        n = q_batch.shape[0]
        self.refresh_locked_joints(emb_name, q_batch[0])
        js = self._active_js(entry, q_batch)
        costs = self._collision_costs(entry)
        state = entry.fk.compute_kinematics(js)
        spheres = state.robot_spheres.reshape(n, 1, -1, 4).contiguous()
        if costs["batch"] != n:
            costs["self"].setup_batch_tensors(n, 1)
            costs["buffer"] = CollisionBuffer.from_shape(spheres.shape, self.device_cfg)
            costs["batch"] = n
        dev = self.device_cfg.device
        wanted = set(links)
        mask = th.tensor([l in wanted for l in costs["link_of_sphere"]], device=dev).unsqueeze(0)
        weight = th.ones(1, device=dev)
        spheres, act = shrink_link_spheres(spheres, mask, float(margin))
        with th.no_grad():
            qstate = KinematicsState(robot_spheres=spheres)
            dm = self.checker.get_sphere_collision(qstate, costs["buffer"], weight, th.full((1,), act, device=dev)).reshape(n, -1)
            near = ((dm > 0.0) & mask).any(dim=1)
        return near.cpu()

    def robot_spheres_world(self, q_batch: th.Tensor, emb_name: str = "whole_body") -> Tuple[th.Tensor, List[str]]:
        """``(N, S, 4)`` world-frame collision spheres (x y z r; r <= 0 = a disabled sphere) of the full configurations
        ``q_batch`` (N, 28) in the embodiment ``emb_name`` - an attached object's spheres included - and the link of every
        sphere. Forward kinematics only: no world query, no planner or world state is touched (the oracle's protected-object
        transit check, ``protection.check_trajectory``)."""
        entry = self.kin_entry(emb_name)
        q_batch = as_tensor(q_batch)
        if q_batch.ndim == 1:
            q_batch = q_batch.unsqueeze(0)
        n = q_batch.shape[0]
        self.refresh_locked_joints(emb_name, q_batch[0])
        js = self._active_js(entry, q_batch)
        with th.no_grad():
            sp = entry.fk.compute_kinematics(js).robot_spheres.reshape(n, -1, 4).float().cpu()
        rp, rq = self.root_pose()
        R = T.quat2mat(rq.float().cpu()).float()
        world = sp.clone()
        world[..., :3] = sp[..., :3] @ R.T + rp.float().cpu().reshape(1, 1, 3)
        return world, list(self._collision_costs(entry)["link_of_sphere"])

    def _ground_obstacles_enabled(self, enabled: bool):
        """Enable / disable every ground-like obstacle (slab + floor/carpet/... meshes).

        Meshes that are disabled for other reasons (ignored objects, attached objects) stay disabled.
        """
        cub = getattr(self.checker.data, "cuboids", None)
        if cub is not None and self.GROUND_SLAB_NAME in cub.names[0]:
            cub.set_enabled(self.GROUND_SLAB_NAME, enabled)
        for lname, info in self._grid_layers.items():  # sdf-grid mode: layers of ground-like objects, like the meshes below
            if info["ground_like"]:
                self.checker.data.voxels.set_enabled(lname, enabled)
        held_off = getattr(self, "_held_off", set())  # meshes an active sampler ignore context (ik._Ignoring) switched off
        for name in self._ground_mesh_names:
            if name not in self._world_mesh_names:
                continue
            if enabled and (name in self._disabled_obstacles or name in held_off):
                # owner 2026-09-25 (shampoo 0925-171050 .. 205946, 'lift: 1/11 tracked chain configurations collide', body = the utility
                # room floor): every margin > 0 check (step 3 of check_collision) switched the ground meshes off and back ON, re-enabling
                # the floor mesh a sampler ignore context had removed for a floor-class lift - the carried dispenser's spheres then met
                # the floor 10 mm into the leg
                continue
            self._mesh_enable(name, enabled)

    def check_collision(self, q_batch: th.Tensor, emb_name: str = "whole_body", sync_world: bool = False, self_collision: bool = True, margin: float = 0.0) -> th.Tensor:
        """``(N,)`` bool tensor: True where the full configuration collides with the world or with itself.

        ``margin`` (m) makes world *proximity* count as a collision for every (link, obstacle) pair
        except (ground-contact link, ground-like obstacle): wheels and base body sit within ~1 cm of
        the floor by design. Actual penetration is always checked for every pair.
        """
        from curobo._src.geom.collision.buffer_collision import CollisionBuffer
        from curobo._src.robot.kinematics.kinematics_state import KinematicsState

        entry = self.kin_entry(emb_name)
        if sync_world:
            self.update_world()
        q_batch = as_tensor(q_batch)
        if q_batch.ndim == 1:
            q_batch = q_batch.unsqueeze(0)
        n = q_batch.shape[0]
        self.refresh_locked_joints(emb_name, q_batch[0])
        js = self._active_js(entry, q_batch)
        costs = self._collision_costs(entry)
        state = entry.fk.compute_kinematics(js)
        spheres = state.robot_spheres.reshape(n, 1, -1, 4).contiguous()
        if costs["batch"] != n:
            costs["self"].setup_batch_tensors(n, 1)
            costs["buffer"] = CollisionBuffer.from_shape(spheres.shape, self.device_cfg)
            costs["batch"] = n
        weight = th.ones(1, device=self.device_cfg.device)
        dev = self.device_cfg.device
        with th.no_grad():
            qstate = KinematicsState(robot_spheres=spheres)
            # 1) actual penetration, every sphere against every obstacle
            d0 = self.checker.get_sphere_collision(qstate, costs["buffer"], weight, th.zeros(1, device=dev)).reshape(n, -1)
            in_coll = (d0 > 0.0).any(dim=1)
            if margin > 0.0:
                act = th.full((1,), float(margin), device=dev)
                # 2) proximity for the non-ground-contact links against everything
                dm = self.checker.get_sphere_collision(qstate, costs["buffer"], weight, act).reshape(n, -1)
                gc = costs["ground_contact"].unsqueeze(0)
                in_coll = in_coll | ((dm > 0.0) & ~gc).any(dim=1)
                # 3) proximity for the ground-contact links against everything *except* ground-like obstacles
                self._ground_obstacles_enabled(False)
                try:
                    dg = self.checker.get_sphere_collision(qstate, costs["buffer"], weight, act).reshape(n, -1)
                finally:
                    self._ground_obstacles_enabled(True)
                in_coll = in_coll | ((dg > 0.0) & gc).any(dim=1)
            if self_collision:
                ds = costs["self"].forward(spheres).reshape(n, -1).amax(dim=1)
                in_coll = in_coll | (ds > 0.0)
        return in_coll.cpu()

    def self_collision_pairs(self, q_full: th.Tensor, emb_name: str = "whole_body", margin: float = 0.0) -> List[Dict[str, object]]:
        """The link pairs of cuRobo's self-collision list whose padded spheres overlap at ``q_full``:
        ``[{"links": (a, b), "penetration": m}]`` (the worst sphere pair per link pair). ``margin`` (m): pairs closer
        than this also count (a sampler witness the demo's trunk, a centimetre off the recorded one, must not push
        into the padding: dishwasher_wide 0921-044216, link5 - torso_link4 by 9 mm at the recorded retreat)."""
        entry = self.kin_entry(emb_name)
        q_full = as_tensor(q_full).reshape(-1)
        self.refresh_locked_joints(emb_name, q_full)
        js = self._active_js(entry, q_full.unsqueeze(0))
        with th.no_grad():
            sp = entry.fk.compute_kinematics(js).robot_spheres.reshape(-1, 4)
            scfg = entry.fk.get_self_collision_config()
            pairs = scfg.collision_pairs.long()
            pad = scfg.sphere_padding.to(sp.device)
            i, j = pairs[:, 0], pairs[:, 1]
            d = th.linalg.norm(sp[i, :3] - sp[j, :3], dim=1)
            pen = (sp[i, 3] + pad[i]) + (sp[j, 3] + pad[j]) + float(margin) - d
            active = (sp[i, 3] > 0) & (sp[j, 3] > 0)
            hot = th.nonzero((pen > 0) & active).reshape(-1)
        if hot.numel() == 0:
            return []
        kc = entry.fk.kinematics_config
        link_of = {}
        for name in kc.link_name_to_idx_map:
            for k in kc.get_sphere_index_from_link_name(name).tolist():
                link_of[int(k)] = name
        worst: Dict[Tuple[str, str], float] = {}
        for k in hot.tolist():
            a, b = link_of.get(int(i[k]), str(int(i[k]))), link_of.get(int(j[k]), str(int(j[k])))
            key = (a, b) if a <= b else (b, a)
            worst[key] = max(worst.get(key, 0.0), float(pen[k]))
        return [{"links": key, "penetration": v} for key, v in sorted(worst.items(), key=lambda kv: -kv[1])]

    @contextmanager
    def static_links_relaxed(self, emb_name: str, q_start: th.Tensor, moving_arms: Sequence[str]):
        """Plan an arm while the **other** arm holds a contact it was driven into by an unchecked straight leg (a brace
        approach, a grasp): when the start configuration violates cuRobo's self-collision margin between two links
        the plan cannot move (the held arm against the trunk), those held-arm links leave the sphere model for the
        duration of the block (``kinematics_config.disable_link_spheres``, shared by every solver of the planner) and
        are restored afterwards. Yields the disabled link names. Pairs that involve a planned arm, or the trunk /
        base alone, are genuine constraints and stay."""
        moving = set(moving_arms)
        disable: List[str] = []
        try:
            pairs = self.self_collision_pairs(q_start, emb_name)
        except Exception as e:  # noqa: BLE001
            log.warning("static_links_relaxed: self-collision pairs unavailable (%r)", e)
            pairs = []
        for pr in pairs:
            arms = [next((arm for arm in M.ARMS if link.startswith(arm + "_")), None) for link in pr["links"]]
            if any(arm in moving for arm in arms if arm is not None):
                continue
            for link, arm in zip(pr["links"], arms):
                if arm is not None and link not in disable:
                    disable.append(link)
        if not disable:
            yield []
            return
        entry = self.kin_entry(emb_name)
        kc = entry.fk.kinematics_config
        done = []
        try:
            for link in disable:
                if link in kc.link_name_to_idx_map:
                    kc.disable_link_spheres(link)
                    done.append(link)
            log.warning("planning %s with the held arm's %s out of the sphere model: the start violates the self-collision margin by %.1f mm (%s)",
                        "/".join(sorted(moving)), done, 1000 * pairs[0]["penetration"], " ".join(f"{a}-{b}" for a, b in (pr["links"] for pr in pairs)))
            yield done
        finally:
            for link in done:
                kc.enable_link_spheres(link)

    @staticmethod
    def _arm_of_link(link: str) -> Optional[str]:
        """``left`` / ``right`` for an arm link or an attached object (``attached_object_<arm>_eef_link``), None for the trunk / base."""
        for arm in M.ARMS:
            if link.startswith(arm + "_") or link == f"attached_object_{arm}_eef_link":
                return arm
        return None

    @contextmanager
    def start_collisions_relaxed(self, emb_name: str, q_start: th.Tensor, moving_arms: Sequence[str]):
        """A collision-checked leg (``solve_ik_path(check_collisions=True)``) must fail on NEW collisions only (owner 2026-09-24
        item 2): every self-collision pair the start configuration already violates between the moving arm and **another
        arm's** link or attached object (the bracing hand's fingers inside the sphere model of the carton the other hand
        holds, 23-50 mm; 0925-103154 retreat[6] 'start configuration is in collision') is exempted for the block by taking
        the static member's spheres out of the model; pairs within the moving arm itself or against the trunk / base stay
        (the 0918-000750 elbow fold, 3 mrad short of the joint limit, is such a pair and is still caught), and pairs between
        two static arms are relaxed as ``static_links_relaxed`` does. Yields the disabled link names."""
        moving = set(moving_arms)
        disable: List[str] = []
        try:
            pairs = self.self_collision_pairs(q_start, emb_name)
        except Exception as e:  # noqa: BLE001
            log.warning("start_collisions_relaxed: self-collision pairs unavailable (%r)", e)
            pairs = []
        for pr in pairs:
            arms = [self._arm_of_link(link) for link in pr["links"]]
            if all(a is not None and a in moving for a in arms):
                continue  # within the moving arm: a genuine constraint
            if any(a is None for a in arms) and any(a is not None and a in moving for a in arms):
                continue  # the moving arm against the trunk / base: genuine
            for link, arm in zip(pr["links"], arms):
                if arm is not None and arm not in moving and link not in disable:
                    disable.append(link)  # the static arm's member (its link or its attached object) leaves the model for this leg
        if not disable:
            yield []
            return
        entry = self.kin_entry(emb_name)
        kc = entry.fk.kinematics_config
        done: List[str] = []
        try:
            for link in disable:
                if link in kc.link_name_to_idx_map:
                    kc.disable_link_spheres(link)
                    done.append(link)
            log.warning("checking %s's leg with %s out of the sphere model: the start already violates the self-collision margin by %.1f mm (%s) - only new collisions fail the leg",
                        "/".join(sorted(moving)), done, 1000 * pairs[0]["penetration"], " ".join(f"{a}-{b}" for a, b in (pr["links"] for pr in pairs)))
            yield done
        finally:
            for link in done:
                kc.enable_link_spheres(link)

    # ------------------------------------------------------------------ goal construction

    def _goal_and_criteria(self, entry: _PlannerEntry, targets: Dict[str, Pose], q_full: th.Tensor, criteria: Optional[Dict[str, ToolPoseCriteria]] = None):
        """Build a GoalToolPose covering all planner tool frames; unconstrained frames get zero weight."""
        planner = entry.planner
        frames = list(planner.tool_frames)
        unknown = set(targets) - set(frames)
        if unknown:
            raise ValueError(f"Frames {unknown} are not tool frames of embodiment '{entry.embodiment.name}' ({frames})")
        current = self._kin_state(entry, q_full).tool_poses
        pose_dict, crit = {}, {}
        for f in frames:
            if f in targets:
                pose_dict[f] = self._cu_pose_from_world([targets[f]])
                crit[f] = (criteria or {}).get(f) or entry.default_criteria[f]
            else:
                cp = current.get_link_pose(f)
                pose_dict[f] = CuPose(position=cp.position.reshape(-1, 3)[:1].clone(), quaternion=cp.quaternion.reshape(-1, 4)[:1].clone())
                crit[f] = ToolPoseCriteria(terminal_pose_axes_weight_factor=[0.0] * 6, non_terminal_pose_axes_weight_factor=[0.0] * 6, device_cfg=self.device_cfg)
        goal = GoalToolPose.from_poses(pose_dict, ordered_tool_frames=frames, num_goalset=1)
        return goal, crit

    def _apply_criteria(self, entry: _PlannerEntry, crit: Dict[str, ToolPoseCriteria]):
        entry.planner.update_tool_pose_criteria(crit)

    def _restore_criteria(self, entry: _PlannerEntry):
        entry.planner.update_tool_pose_criteria(dict(entry.default_criteria))

    # ------------------------------------------------------------------ planning

    def plan_pose(
        self,
        emb_name: str,
        targets: Dict[str, Pose],
        q_start: Optional[th.Tensor] = None,
        max_attempts: int = 4,
        criteria: Optional[Dict[str, ToolPoseCriteria]] = None,
        sync_world: bool = True,
    ) -> PlanResult:
        """Plan a collision-free joint trajectory bringing tool frames to world-frame ``targets``."""
        entry = self.get_entry(emb_name)
        if sync_world:
            self.update_world()
        q_start = self.q_full() if q_start is None else as_tensor(q_start)
        self.refresh_locked_joints(emb_name, q_start)
        goal, crit = self._goal_and_criteria(entry, targets, q_start, criteria)
        self._apply_criteria(entry, crit)
        t0 = time.time()
        try:
            cur = self._active_js(entry, q_start)
            result = entry.planner.plan_pose(goal, cur, max_attempts=max_attempts)
            out = self._result_from_trajopt(entry, result, q_start, time.time() - t0)
            if not out.success:
                # MotionPlanner.plan_pose returns None when IK never succeeded, otherwise the trajopt result
                out.diagnostics.update(self._diagnose_start(emb_name, q_start))
                if result is None:
                    out.diagnostics.update(self._diagnose_ik(entry, goal, cur))
                    out.status = "IK failed: " + out.diagnostics["ik_verdict"]
                else:
                    out.diagnostics.update(self._diagnose_trajopt(result))
                    out.status = "IK found a goal configuration but trajectory optimization failed: " + out.diagnostics["trajopt_verdict"]
                    out.diagnostics.update(self._diagnose_path(entry, result, q_start, emb_name))
                    psc = out.diagnostics.get("path_self_collision")
                    if isinstance(psc, dict):
                        out.status += f" [self-collision along the path: {psc['links'][0]} - {psc['links'][1]} by {1000 * psc['penetration']:.0f} mm at waypoint {psc['waypoint']}/{psc['of']}]"
                if out.diagnostics.get("start_in_collision"):
                    out.status += " [start configuration is in collision" + self._start_detail(out.diagnostics) + "]"
        finally:
            self._restore_criteria(entry)
        return out

    @staticmethod
    def _start_detail(diag: Dict[str, object]) -> str:
        parts = []
        if diag.get("start_world_collision"):
            parts.append("world")
        sc = diag.get("start_self_collision")
        if sc:
            parts.append("self: " + ", ".join(f"{p['links'][0]}-{p['links'][1]} {p['penetration_mm']:.0f} mm" for p in sc))
        off = diag.get("start_world_offenders")
        if off:
            parts.append("offenders: " + ", ".join(str(o).split("/")[-3:][0] + "/" + str(o).split("/")[-1] if "/" in str(o) else str(o) for o in off))
        return (": " + "; ".join(parts)) if parts else ""

    def plan_joint(self, emb_name: str, q_goal_full: th.Tensor, q_start: Optional[th.Tensor] = None, max_attempts: int = 4, sync_world: bool = True) -> PlanResult:
        """Plan a collision-free trajectory to a joint-space goal (active joints of the embodiment)."""
        entry = self.get_entry(emb_name)
        if sync_world:
            self.update_world()
        q_start = self.q_full() if q_start is None else as_tensor(q_start)
        self.refresh_locked_joints(emb_name, q_start)
        goal_js = self._active_js(entry, as_tensor(q_goal_full))
        t0 = time.time()
        result = entry.planner.plan_cspace(goal_js, self._active_js(entry, q_start), max_attempts=max_attempts)
        out = self._result_from_trajopt(entry, result, q_start, time.time() - t0)
        if not out.success:
            out.diagnostics.update(self._diagnose_start(emb_name, q_start))
            goal_coll = bool(self.check_collision(as_tensor(q_goal_full), emb_name=emb_name)[0])
            out.diagnostics["goal_in_collision"] = goal_coll
            if result is not None:
                out.diagnostics.update(self._diagnose_trajopt(result))
                out.status = "trajectory optimization to the joint goal failed: " + out.diagnostics["trajopt_verdict"]
                out.diagnostics.update(self._diagnose_path(entry, result, q_start, emb_name))
                psc = out.diagnostics.get("path_self_collision")
                if isinstance(psc, dict):
                    out.status += f" [self-collision along the path: {psc['links'][0]} - {psc['links'][1]} by {1000 * psc['penetration']:.0f} mm at waypoint {psc['waypoint']}/{psc['of']}]"
            else:
                out.status = "no trajectory to the joint goal (graph search found no seed path)"
            if goal_coll:
                out.status += " [goal configuration is in collision]"
            if out.diagnostics.get("start_in_collision"):
                out.status += " [start configuration is in collision" + self._start_detail(out.diagnostics) + "]"
        return out

    def _result_from_trajopt(self, entry: _PlannerEntry, result, q_ref: th.Tensor, plan_time: float) -> PlanResult:
        if result is None or result.success is None or not bool(result.success.any()):
            return PlanResult(success=False, status="planning failed", dt=self.control_dt, plan_time=plan_time)
        interp = result.get_interpolated_plan()
        names = list(interp.joint_names)
        q_active = interp.position.reshape(-1, len(names))
        q_traj = self._full_from_active(entry, q_active, names, q_ref)
        return PlanResult(success=True, q_traj=q_traj, dt=self.control_dt, status="ok", plan_time=plan_time)

    # ------------------------------------------------------------------ failure diagnostics

    def _diagnose_start(self, emb_name: str, q_start: th.Tensor) -> Dict[str, object]:
        try:
            d: Dict[str, object] = {"start_in_collision": bool(self.check_collision(q_start, emb_name=emb_name)[0])}
        except Exception as e:  # diagnostics must never mask the original failure
            return {"start_in_collision": f"unknown ({e})"}
        if d["start_in_collision"]:
            # which kind (2026-09-19, a locked arm pressing an object while the other plans): world penetration vs self-collision
            try:
                d["start_world_collision"] = bool(self.check_collision(q_start, emb_name=emb_name, self_collision=False)[0])
                pairs = self.self_collision_pairs(q_start, emb_name=emb_name)
                if pairs:
                    d["start_self_collision"] = [{"links": list(p["links"]), "penetration_mm": round(1000 * float(p["penetration"]), 1)} for p in pairs[:4]]
                if d["start_world_collision"]:
                    d["start_world_offenders"] = self.world_collision_offenders(q_start, emb_name=emb_name)
            except Exception as e:  # noqa: BLE001
                d["start_collision_detail"] = f"unknown ({e})"
        return d

    def start_world_offenders(self, q_full: th.Tensor, emb_name: str = "whole_body", limit: int = 6) -> List[str]:
        """The world meshes the configuration ``q_full`` already penetrates in the planner's sphere model (padding /
        activation distance included), ``[]`` when the world check passes - the per-mesh search of
        :meth:`world_collision_offenders` runs only then (owner 2026-09-25: a leg that leaves a contact is judged on
        NEW collisions; the pad that ended a push against the partner reads as a start collision here while the
        physics contact report is empty, touching 0925-134446 retreat[8])."""
        if not bool(self.check_collision(q_full, emb_name=emb_name, self_collision=False)[0]):
            return []
        return self.world_collision_offenders(q_full, emb_name=emb_name, limit=limit)

    def objects_of_world_meshes(self, names: Sequence[str]) -> List[Any]:
        """The scene objects owning the collision meshes ``names`` (prim paths as :meth:`world_collision_offenders` lists
        them; a ``together: a, b`` entry is split). The ground slab and unknown names map to nothing."""
        wanted = set()
        for n in names or ():
            s = str(n)
            if s.startswith("together: "):
                wanted.update(x.strip() for x in s[len("together: "):].split(","))
            else:
                wanted.add(s)
        out: List[Any] = []
        for obj in self.robot.scene.objects:
            if obj is self.robot:
                continue
            for link in getattr(obj, "links", {}).values():
                if any(str(cm.prim_path) in wanted for cm in getattr(link, "collision_meshes", {}).values()):
                    out.append(obj)
                    break
        return out

    def world_collision_offenders(self, q_full: th.Tensor, emb_name: str = "whole_body", limit: int = 6) -> List[str]:
        """The enabled world meshes (and the ground slab) whose removal clears the penetration at ``q_full``: each enabled
        obstacle is disabled in turn and the world check re-run (2026-09-19 diagnostic; the world is restored)."""
        out: List[str] = []
        names = [n for n in sorted(self._world_mesh_names) if n not in self._disabled_obstacles]
        for name in names:
            self._mesh_enable(name, False)
            try:
                hit = bool(self.check_collision(q_full, emb_name=emb_name, self_collision=False)[0])
            finally:
                self._mesh_enable(name, True)
            if not hit:
                out.append(name)
                if len(out) >= limit:
                    return out
        cub = getattr(self.checker.data, "cuboids", None)
        if cub is not None and self.GROUND_SLAB_NAME in cub.names[0]:
            cub.set_enabled(self.GROUND_SLAB_NAME, False)
            try:
                hit = bool(self.check_collision(q_full, emb_name=emb_name, self_collision=False)[0])
            finally:
                cub.set_enabled(self.GROUND_SLAB_NAME, True)
            if not hit:
                out.append(self.GROUND_SLAB_NAME)
        if out:
            return out
        # several penetrate: greedy - keep disabling the mesh whose removal reduces the penetration most until the check clears
        disabled: List[str] = []
        try:
            remaining = list(names)
            for _ in range(min(limit, len(remaining))):
                best, best_pen = None, None
                for name in remaining:
                    self._mesh_enable(name, False)
                    try:
                        pen = self._world_penetration(q_full, emb_name)
                    finally:
                        self._mesh_enable(name, True)
                    if best is None or pen < best_pen:
                        best, best_pen = name, pen
                if best is None:
                    break
                disabled.append(best)
                remaining.remove(best)
                self._mesh_enable(best, False)
                if not bool(self.check_collision(q_full, emb_name=emb_name, self_collision=False)[0]):
                    break
        finally:
            for name in disabled:
                self._mesh_enable(name, True)
        return [f"together: {', '.join(disabled)}"] if disabled else ["(no single obstacle clears it: several penetrate)"]

    def _world_penetration(self, q_full: th.Tensor, emb_name: str) -> float:
        """Summed world penetration of the robot's spheres at ``q_full`` (diagnostic)."""
        from curobo._src.geom.collision.buffer_collision import CollisionBuffer
        from curobo._src.robot.kinematics.kinematics_state import KinematicsState

        entry = self.kin_entry(emb_name)
        q = as_tensor(q_full).reshape(1, -1)
        self.refresh_locked_joints(emb_name, q[0])
        js = self._active_js(entry, q)
        costs = self._collision_costs(entry)
        state = entry.fk.compute_kinematics(js)
        spheres = state.robot_spheres.reshape(1, 1, -1, 4).contiguous()
        if costs["batch"] != 1:
            costs["self"].setup_batch_tensors(1, 1)
            costs["buffer"] = CollisionBuffer.from_shape(spheres.shape, self.device_cfg)
            costs["batch"] = 1
        weight = th.ones(1, device=self.device_cfg.device)
        with th.no_grad():
            d0 = self.checker.get_sphere_collision(KinematicsState(robot_spheres=spheres), costs["buffer"], weight, th.zeros(1, device=self.device_cfg.device)).reshape(-1)
        return float(d0.clamp(min=0).sum())

    def _diagnose_ik(self, entry: _PlannerEntry, goal, cur) -> Dict[str, object]:
        """Classify an IK failure: unreachable pose vs. reachable but infeasible (collision / limits).

        cuRobo marks a seed ``success`` only if it converged (pose error below tolerance) *and* is
        feasible (no collision, inside joint limits). Looking at the two parts separately tells the
        two failure modes apart.
        """
        ik = entry.planner.ik_solver
        try:
            res = ik.solve_pose(goal, current_state=cur, seed_config=cur.position.view(1, 1, -1).clone(), return_seeds=ik.config.num_seeds)
        except Exception as e:
            return {"ik_verdict": f"could not re-run IK for diagnostics ({e})"}
        pos_err = res.position_error.reshape(-1)
        rot_err = res.rotation_error.reshape(-1)
        n = int(pos_err.numel())
        pos_tol, orient_tol = float(ik.config.position_tolerance), float(ik.config.orientation_tolerance)
        converged = (pos_err <= pos_tol) & (rot_err <= orient_tol)
        feasible = res.feasible.reshape(-1).bool() if res.feasible is not None else th.ones(n, dtype=th.bool, device=pos_err.device)
        d = {
            "ik_best_pos_err_m": float(pos_err.min()),
            "ik_best_rot_err_rad": float(rot_err.min()),
            "ik_seeds": n,
            "ik_converged_seeds": int(converged.sum()),
            "ik_feasible_seeds": int(feasible.sum()),
            "ik_converged_and_feasible": int((converged & feasible).sum()),
        }
        if d["ik_converged_seeds"] == 0:
            # The IK optimizer includes the collision cost, so a pose inside an obstacle also shows up as
            # "not converged". Re-run once with every world obstacle disabled to tell the cases apart.
            free = self._ik_without_world(ik, goal, cur)
            d["ik_converged_without_world"] = free
            if free:
                d["ik_verdict"] = (
                    f"goal pose is kinematically reachable but blocked by world obstacles (best pose error with obstacles "
                    f"{d['ik_best_pos_err_m']*1000:.1f} mm / {d['ik_best_rot_err_rad']:.3f} rad; converges once obstacles are ignored)"
                )
            else:
                d["ik_verdict"] = (
                    f"goal pose unreachable for the active joints (best pose error {d['ik_best_pos_err_m']*1000:.1f} mm / "
                    f"{d['ik_best_rot_err_rad']:.3f} rad over {n} seeds even without obstacles; tolerances {pos_tol*1000:.0f} mm / {orient_tol:.2f} rad)"
                )
        else:
            d["ik_verdict"] = (
                f"goal pose reachable ({d['ik_converged_seeds']}/{n} seeds converge) but every converged solution violates "
                f"collision or joint-limit constraints ({d['ik_feasible_seeds']}/{n} seeds feasible overall)"
            )
        return d

    def _ik_without_world(self, ik, goal, cur) -> bool:
        """Re-run IK with all world obstacles disabled (self-collision stays on); restore afterwards."""
        stores = [s for s in (getattr(self.checker.data, "meshes", None), getattr(self.checker.data, "cuboids", None)) if s is not None and getattr(s, "enable", None) is not None]
        saved = [s.enable.clone() for s in stores]
        try:
            for s in stores:
                s.enable.zero_()
            res = ik.solve_pose(goal, current_state=cur, seed_config=cur.position.view(1, 1, -1).clone(), return_seeds=ik.config.num_seeds)
            pos_err = res.position_error.reshape(-1)
            rot_err = res.rotation_error.reshape(-1)
            return bool(((pos_err <= ik.config.position_tolerance) & (rot_err <= ik.config.orientation_tolerance)).any())
        except Exception:
            return False
        finally:
            for s, e in zip(stores, saved):
                s.enable.copy_(e)

    def _diagnose_path(self, entry: _PlannerEntry, result, q_ref: th.Tensor, emb_name: str, samples: int = 12) -> Dict[str, object]:
        """The worst self-collision pair along the trajectory the failed optimisation returned (the pair the plan
        could not get around, e.g. the other hand's wrist when both hands work 13 cm apart)."""
        try:
            plan = result.get_interpolated_plan()
            names = list(plan.joint_names)
            q_active = plan.position.reshape(-1, len(names))
            q_traj = self._full_from_active(entry, q_active, names, q_ref)
            worst: Optional[Dict[str, object]] = None
            idx = th.linspace(0, len(q_traj) - 1, min(samples, len(q_traj))).long().tolist()
            for k in idx:
                pairs = self.self_collision_pairs(q_traj[k], emb_name)
                if pairs and (worst is None or pairs[0]["penetration"] > worst["penetration"]):
                    worst = {"links": list(pairs[0]["links"]), "penetration": pairs[0]["penetration"], "waypoint": int(k), "of": int(len(q_traj))}
            return {} if worst is None else {"path_self_collision": worst}
        except Exception as e:  # noqa: BLE001
            return {"path_self_collision": f"unknown ({e})"}

    @staticmethod
    def _diagnose_trajopt(result) -> Dict[str, object]:
        d: Dict[str, object] = {}
        try:
            # cuRobo's own verdict (MotionGenStatus: TrajOpt Fail / IK Fail / Graph Fail / Invalid start ... ) - a plan whose
            # final pose error is 0.0 mm and that still fails (blender eyedvd pre-grasp 0918-222128) is only readable from it
            st = getattr(result, "status", None)
            if st is not None:
                d["status"] = str(getattr(st, "value", st))
            if getattr(result, "valid_query", None) is not None:
                d["valid_query"] = bool(result.valid_query)
            if getattr(result, "optimized_dt", None) is not None:
                try:
                    d["optimized_dt"] = float(th.as_tensor(result.optimized_dt).reshape(-1)[0])
                except Exception:  # noqa: BLE001
                    pass
            n = int(result.success.numel())
            d["trajopt_seeds"] = n
            if result.feasible is not None:
                d["trajopt_feasible_seeds"] = int(result.feasible.reshape(-1).bool().sum())
            if result.position_error is not None:
                d["trajopt_best_final_pos_err_m"] = float(result.position_error.reshape(-1).min())
            if result.rotation_error is not None:
                d["trajopt_best_final_rot_err_rad"] = float(result.rotation_error.reshape(-1).min())
            parts = []
            if "trajopt_feasible_seeds" in d:
                parts.append(f"{d['trajopt_feasible_seeds']}/{n} trajectories collision/limit feasible")
            if "trajopt_best_final_pos_err_m" in d:
                parts.append(f"best final pose error {d['trajopt_best_final_pos_err_m']*1000:.1f} mm / {d.get('trajopt_best_final_rot_err_rad', float('nan')):.3f} rad")
            if d.get("status"):
                parts.append(f"cuRobo status {d['status']}")
            d["trajopt_verdict"] = "; ".join(parts) if parts else "no per-seed metrics available"
            if d.get("trajopt_feasible_seeds", 1) == 0:
                d["trajopt_verdict"] += " -> the path between start and goal is blocked (collision), not the goal itself"
        except Exception as e:
            d["trajopt_verdict"] = f"could not read trajopt metrics ({e})"
        return d

    # ------------------------------------------------------------------ IK

    def solve_ik(
        self,
        emb_name: str,
        targets: Dict[str, Pose],
        q_seed: Optional[th.Tensor] = None,
        criteria: Optional[Dict[str, ToolPoseCriteria]] = None,
        return_all: bool = False,
        sync_world: bool = True,
    ):
        """Collision-aware IK. Returns the successful full configuration closest to ``q_seed``
        (or ``None``); with ``return_all`` returns ``(N, 28)`` of all successful solutions."""
        entry = self.get_entry(emb_name)
        if sync_world:
            self.update_world()
        q_seed = self.q_full() if q_seed is None else as_tensor(q_seed)
        self.refresh_locked_joints(emb_name, q_seed)
        goal, crit = self._goal_and_criteria(entry, targets, q_seed, criteria)
        self._apply_criteria(entry, crit)
        try:
            ik = entry.planner.ik_solver
            n_seeds = ik.config.num_seeds
            cur = self._active_js(entry, q_seed)
            res = ik.solve_pose(goal, current_state=cur, seed_config=cur.position.view(1, 1, -1).clone(), return_seeds=n_seeds)
        finally:
            self._restore_criteria(entry)
        success = res.success.view(-1)
        if not bool(success.any()):
            return None
        names = list(ik.joint_names)
        sols = res.solution.reshape(-1, len(names))[success]
        q_full = self._full_from_active(entry, sols, names, q_seed)
        if return_all:
            return q_full
        seed_active = cur.position.view(1, -1).cpu()
        d = th.linalg.norm(sols.cpu() - seed_active, dim=1)
        return q_full[int(th.argmin(d))]

    # ------------------------------------------------------------------ batched IK (reachability queries)

    def _batch_ik_entry(self, emb_name: str, batch_size: int, seeds: Optional[int] = None) -> _PlannerEntry:
        """Dedicated IK solver (``max_batch_size = batch_size``) for ``emb_name`` sharing the scene checker; the
        collision spheres / self-collision data are the planner's, so results agree with :meth:`solve_ik`.
        ``seeds`` (2026-10-31 train-fast): seeds of that solver, default :attr:`batch_ik_seeds`; one solver is cached per
        (embodiment, batch size, seeds)."""
        seeds = int(getattr(self, "batch_ik_seeds", 12)) if seeds is None else int(seeds)
        key = (emb_name, int(batch_size), seeds)
        cache = getattr(self, "_batch_ik", None)
        if cache is None:
            cache = self._batch_ik = {}
        if key not in cache:
            from curobo._src.solver.solver_ik_cfg import IKSolverCfg

            emb = M.EMBODIMENTS[emb_name]
            q = self.q_full()
            lock_values = {n: float(q[self._jidx[n]]) for n in emb.locked_joints}
            robot_dict = M.build_curobo_robot_dict(emb, self.urdf_path, lock_values, self._default_q, self._v1_kin)
            t0 = time.time()
            # cuRobo IK cost is ~1 ms per (problem, seed) on an RTX 4090 regardless of batching, so screening
            # queries use fewer seeds than the planner's IK (``batch_ik_seeds``, default 12: ~3 % recall loss).
            cfg = IKSolverCfg.create(
                robot=robot_dict,
                scene_model=None,
                collision_cache=None,
                self_collision_check=True,
                device_cfg=self.device_cfg,
                num_seeds=seeds,
                position_tolerance=self.position_tolerance,
                orientation_tolerance=self.orientation_tolerance,
                use_cuda_graph=self.use_cuda_graph,
                optimizer_collision_activation_distance=self.collision_activation_distance,
                max_batch_size=int(batch_size),
            )
            ik = IKSolver(cfg, self.checker)
            entry = _PlannerEntry(embodiment=emb, planner=None, fk=Kinematics(ik.kinematics.config))
            entry.rsc = ik  # store the solver in the otherwise unused slot
            self._fill_lock_info(entry, entry.fk)
            cache[key] = entry
            log.info("Built batched IK solver '%s' x%d in %.1fs", emb_name, batch_size, time.time() - t0)
        return cache[key]

    def _refresh_locked_entry(self, entry: _PlannerEntry, q_full: th.Tensor):
        kc = entry.fk.kinematics_config
        lock_js = kc.lock_jointstate
        for jn, (link_idx, origin, jtype, scale, offset) in entry.lock_info.items():
            val = float(q_full[self._jidx[jn]]) * scale + offset
            kc.fixed_transforms[link_idx] = _lock_transform(origin, jtype, val, self.device_cfg.device)
        if lock_js is not None:
            for i, jn in enumerate(lock_js.joint_names):
                lock_js.position[..., i] = float(q_full[self._jidx[jn]])

    @staticmethod
    def _batch_size_ladder() -> List[int]:
        """Solver sizes BATCH_SIZE_FIT picks from (``ISPATIALGYM_FAST_BATCH_SIZES``, comma separated; default ``8,16,32,64``)."""
        raw = str(fast_value("BATCH_SIZES", "8,16,32,64"))
        out = sorted({int(t) for t in raw.replace(";", ",").split(",") if t.strip().isdigit() and int(t) > 0})
        return out or [8, 16, 32, 64]

    def solve_ik_batch(self, emb_name: str, frame: str, poses: Sequence[Pose], q_seed: th.Tensor, batch_size: int = 64, sync_world: bool = False,
                       seeds: Optional[int] = None):
        """Collision-aware IK for many world poses of one tool frame at once.

        Returns ``(success (N,) bool, q (N, 28))`` where ``q[i]`` is the best solution (full configuration, locked
        joints from ``q_seed``) when ``success[i]``. Poses are solved in chunks of ``batch_size``.

        ``seeds`` (2026-10-31 train-fast): seeds of the batched solver (default :attr:`batch_ik_seeds`, 12). BATCH_SIZE_FIT
        (``ISPATIALGYM_FAST_BATCH_SIZE_FIT=0`` off): the solver's batch is the smallest rung of :meth:`_batch_size_ladder` that
        holds every pose, never above ``batch_size`` - every chunk was padded to a full ``batch_size`` (64) before, so one pose cost
        a 64-problem x seeds solve (5-10 s against a large mesh store: the sphere-vs-mesh kernel scales with the store's capacity);
        a solver per (embodiment, size, seeds) is built once (a few seconds) and cached. More poses than the largest rung are
        chunked by ``batch_size`` as before.
        """
        if sync_world:
            self.update_world()
        if fast_on("BATCH_SIZE_FIT") and len(poses) > 0:
            fit = [b for b in self._batch_size_ladder() if len(poses) <= b <= int(batch_size)]
            if fit:
                batch_size = min(fit)
        entry = self._batch_ik_entry(emb_name, batch_size, seeds=seeds)
        ik: IKSolver = entry.rsc
        q_seed = as_tensor(q_seed)
        self._refresh_locked_entry(entry, q_seed)
        frames = list(ik.kinematics.tool_frames)
        if frame not in frames:
            raise ValueError(f"{frame} is not a tool frame of {emb_name} ({frames})")
        n = len(poses)
        success = th.zeros(n, dtype=th.bool)
        q_out = q_seed.clone().unsqueeze(0).repeat(n, 1)
        names = None
        if len(frames) > 1:
            # a bare IKSolver has no per-frame zero-weight criteria, so batched IK is restricted to the single-tool-frame
            # (arm) embodiments
            raise ValueError("solve_ik_batch supports single-tool-frame embodiments only")
        for start in range(0, n, batch_size):
            chunk = list(poses[start : start + batch_size])
            b = len(chunk)
            # cuRobo's input padding does not cover every JointState field; always solve a full batch instead
            padded = chunk + [chunk[-1]] * (batch_size - b)
            goal = GoalToolPose.from_poses({frame: self._cu_pose_from_world(padded)}, ordered_tool_frames=frames, num_goalset=1)
            cur = entry.fk.get_active_js(self._js_full(q_seed.unsqueeze(0).repeat(batch_size, 1)))
            res = ik.solve_pose(goal, current_state=cur, seed_config=cur.position.view(batch_size, 1, -1).clone(), return_seeds=1)
            names = list(ik.joint_names) if names is None else names
            success[start : start + b] = res.success.reshape(-1)[:b].cpu()
            sols = res.solution.reshape(batch_size, -1, len(names))[:b, 0]
            q_out[start : start + b] = self._full_from_active(entry, sols, names, q_seed)
        return success, q_out

    # ------------------------------------------------------------------ differential IK (Cartesian tracking)

    def solve_ik_path(
        self,
        emb_name: str,
        waypoints: Sequence[Dict[str, Pose]],
        q_start: Optional[th.Tensor] = None,
        pos_tol: float = 2e-3,
        rot_tol: float = 1e-2,
        max_iters: int = 30,
        damping: float = 1e-3,
        max_joint_step: float = 0.2,
        check_collisions: bool = True,
        sync_world: bool = True,
        soft_pos_tol: Optional[float] = None,
        soft_rot_tol: Optional[float] = None,
        sigma_thr: float = 0.08,
        damping_boost: float = 0.02,
        nullspace_gain: float = 0.0,
        limit_margin: float = 0.0,
    ) -> PlanResult:
        """Track dense Cartesian ``waypoints`` (each ``{frame: (pos, quat_xyzw)}`` in world) with
        damped-least-squares differential IK on the embodiment's active joints.

        ``soft_pos_tol`` / ``soft_rot_tol``: a waypoint the iterations leave within these (but outside ``pos_tol`` /
        ``rot_tol``) is accepted with its residual instead of failing the path (``diagnostics["soft_waypoints"]`` lists
        them) — a straight approach that stalls 2 mm short of a 2 mm tolerance is not a failed approach.

        ``limit_margin`` (rad): the joint box the solve may use is shrunk by this at both ends (never past the start
        configuration: a joint already inside the margin may move away from its limit, not toward it). The R1Pro
        elbow's forearm folds onto its upper arm within the last degree of its range (0918-000750 retreat: joint4 at
        -2.0912 of -2.0944, link3 - link5 0.5 mm into each other, the drive fighting the contact for the rest of the
        episode), so free-space legs keep clear of the limits.

        ``check_collisions``: the path is validated for collisions (world + self); on a collision the collision-free
        prefix is returned as ``q_traj`` with ``diagnostics`` ``{"stage": "collision", "fail_index", "converged", ...}``
        so a caller may execute it and finish the segment another way (``Motions.arm_along(finish=...)``).

        Near a kinematic singularity (smallest singular value of the Jacobian below ``sigma_thr``) the damping is
        raised (``damping + damping_boost·(1 − σ/σ_thr)²``); an optional **nullspace step** toward the joint
        mid-ranges (``nullspace_gain``, projected with the damped pseudo-inverse, its own 0.05 rad clamp) is available
        but off by default — on the pilot's stalled lift configuration (R1Pro ``arm_joint2 = 0.003``) the plain DLS
        solved the 31 x 1 cm lift and its 5 mm-offset variant while the escape stalled the task (probe 2026-09-16); the
        stall of that run came from the lift waypoints asking the hand to correct the approach residual first.

        Successive waypoints are solved from the previous solution, which yields a continuous
        joint-space path for straight-line / arc end-effector motions (something trajopt does not
        guarantee). Jacobians are obtained by batched finite differences of cuRobo FK. The result
        is validated for collisions (world + self) unless ``check_collisions`` is False. Fails on
        unreachable waypoints, joint jumps above ``max_joint_step`` (rad) or collisions.
        """
        entry = self.kin_entry(emb_name)
        if sync_world:
            self.update_world()
        t0 = time.time()
        q_full = self.q_full() if q_start is None else as_tensor(q_start).clone()
        self.refresh_locked_joints(emb_name, q_full)
        active = list(entry.embodiment.active_joints)
        aidx = th.tensor([self._jidx[n] for n in active])
        lo = self.robot.joint_lower_limits[aidx].clone()
        hi = self.robot.joint_upper_limits[aidx].clone()
        # unbounded virtual base joints -> generous box around the current value
        unbounded = (hi - lo) > 100.0
        lo[unbounded] = q_full[aidx][unbounded] - 10.0
        hi[unbounded] = q_full[aidx][unbounded] + 10.0
        if limit_margin > 0.0:
            q0a = q_full[aidx]
            lo = th.where(unbounded, lo, th.minimum(lo + float(limit_margin), q0a))
            hi = th.where(unbounded, hi, th.maximum(hi - float(limit_margin), q0a))
        n = len(active)
        eps = 1e-4
        dev = self.device_cfg.device
        frames = list(waypoints[0].keys())
        for wp in waypoints:
            if list(wp.keys()) != frames:
                raise ValueError("all waypoints must constrain the same frames")
        unknown = set(frames) - set(entry.fk.tool_frames)
        if unknown:
            raise ValueError(f"Frames {unknown} are not tool frames of embodiment '{emb_name}'")

        def targets_root(wp):
            pos, quat = [], []
            for f in frames:
                rp, rq = self.world_to_root(wp[f])
                pos.append(rp)
                quat.append(rq)
            return th.stack(pos), th.stack(quat)  # (F,3), (F,4 xyzw) on CPU

        def pose_error(qb):
            """Return the stacked 6-D pose error of every frame for each of the N configs: (N, 6F)."""
            errs = []
            state = self._kin_state(entry, qb.to(dev))
            for f, tp, tq in zip(frames, *tgt):
                lp = state.tool_poses.get_link_pose(f)
                p = lp.position.reshape(-1, 3).cpu()
                q_xyzw = quat_wxyz_to_xyzw(lp.quaternion.reshape(-1, 4).cpu())
                e_p = tp.unsqueeze(0) - p
                # orientation error as rotation vector of R_t R_c^T (root frame)
                R_c = T.quat2mat(q_xyzw)
                R_t = T.quat2mat(tq).unsqueeze(0)
                R_err = R_t @ R_c.transpose(-1, -2)
                e_o = T.quat2axisangle(T.mat2quat_batch(R_err))
                errs.append(th.cat([e_p, e_o], dim=-1))
            return th.cat(errs, dim=-1)

        path = [q_full.clone()]
        q_cur = q_full.clone()
        soft: List[Dict[str, float]] = []
        n_singular = 0
        for wi, wp in enumerate(waypoints):
            tgt = targets_root(wp)
            converged = False
            for it in range(max_iters):
                qb = q_cur.unsqueeze(0).repeat(n + 1, 1)
                for k in range(n):
                    qb[k + 1, aidx[k]] += eps
                e = pose_error(qb)  # (n+1, 6F) on CPU
                err = e[0]
                err_f = err.reshape(len(frames), 6)
                if float(err_f[:, :3].norm(dim=-1).max()) < pos_tol and float(err_f[:, 3:].norm(dim=-1).max()) < rot_tol:
                    converged = True
                    break
                # err = target - fk(q)  =>  d err / d q = -J_fk ; we want dq with J_fk dq = err
                J = -((e[1:] - e[0].unsqueeze(0)) / eps).T  # (6F, n)
                q_act = q_cur[aidx]
                free = th.ones(n, dtype=th.bool)
                q_mid = 0.5 * (lo + hi)
                q_mid[unbounded] = q_act[unbounded]  # virtual base joints: no centring
                for _ in range(n):
                    Jf = J * free.to(J.dtype).unsqueeze(0)
                    try:
                        s_min = float(th.linalg.svdvals(Jf).min())
                    except Exception:  # noqa: BLE001
                        s_min = float("inf")
                    lam = damping + (damping_boost * (1.0 - s_min / sigma_thr) ** 2 if s_min < sigma_thr else 0.0)
                    JJt = Jf @ Jf.T
                    inv = th.linalg.inv(JJt + lam * th.eye(JJt.shape[0]))
                    dq = Jf.T @ (inv @ err)
                    dq_ns = th.zeros_like(dq)
                    if s_min < sigma_thr and nullspace_gain > 0.0:
                        # singularity escape: joint centring in the nullspace of the (damped) Jacobian, its own
                        # (small) step so it never scales the task step down
                        N = th.eye(n) - Jf.T @ inv @ Jf
                        dq_ns = N @ (nullspace_gain * (q_mid - q_act) * free.to(J.dtype))
                        ns_step = float(dq_ns.abs().max())
                        if ns_step > 0.05:
                            dq_ns = dq_ns * (0.05 / ns_step)
                        n_singular += 1
                    # joints pushed beyond a limit are saturated and removed from the solve
                    saturated = ((q_act + dq + dq_ns > hi) & (dq + dq_ns > 0)) | ((q_act + dq + dq_ns < lo) & (dq + dq_ns < 0))
                    saturated &= free
                    if not bool(saturated.any()):
                        break
                    free &= ~saturated
                step = float(dq.abs().max())
                if step > 0.15:
                    dq = dq * (0.15 / step)
                q_cur[aidx] = th.clamp(q_act + dq + dq_ns, lo, hi)
            if not converged:
                err_f = err.reshape(len(frames), 6)
                pos_err = float(err_f[:, :3].norm(dim=-1).max())
                rot_err = float(err_f[:, 3:].norm(dim=-1).max())
                if soft_pos_tol is not None and pos_err <= soft_pos_tol and rot_err <= (soft_rot_tol if soft_rot_tol is not None else rot_tol):
                    soft.append({"index": wi, "pos_err": pos_err, "rot_err": rot_err})
                    converged = True
            if not converged:
                at_lo = [active[k] for k in range(n) if abs(float(q_cur[aidx[k]] - lo[k])) < 1e-4]
                at_hi = [active[k] for k in range(n) if abs(float(q_cur[aidx[k]] - hi[k])) < 1e-4]
                status = (
                    f"differential IK did not converge at waypoint {wi}/{len(waypoints)}: "
                    f"pos_err={pos_err*1000:.1f}mm rot_err={rot_err:.3f}rad "
                    f"at_lower={at_lo} at_upper={at_hi} singular_iterations={n_singular}"
                )
                # the converged prefix (start + waypoints 0..wi-1) is returned so a caller may execute it and finish
                # the segment another way (``Motions.arm_along(finish=...)``)
                partial = th.stack(path) if len(path) > 1 else None
                diag = {"stage": "diff_ik", "fail_index": wi, "n_waypoints": len(waypoints), "pos_err": pos_err, "rot_err": rot_err,
                        "at_lower": at_lo, "at_upper": at_hi, "converged": len(path) - 1, "singular_iterations": n_singular}
                return PlanResult(success=False, q_traj=partial, status=status, dt=self.control_dt, plan_time=time.time() - t0, diagnostics=diag)
            jump = float(th.max(th.abs(q_cur[aidx] - path[-1][aidx])))
            if jump > max_joint_step:
                return PlanResult(success=False, status=f"joint jump {jump:.2f} rad at waypoint {wi}", dt=self.control_dt, plan_time=time.time() - t0)
            path.append(q_cur.clone())
        q_traj = th.stack(path)
        if check_collisions:
            coll = self.check_collision(q_traj, emb_name=emb_name)
            if bool(coll.any()):
                first = int(th.nonzero(coll).view(-1)[0])  # index into q_traj: 0 = the start, k = waypoint k - 1
                kind, pairs = "world", []
                try:
                    if not bool(self.check_collision(q_traj[first], emb_name=emb_name, self_collision=False)[0]):
                        kind = "self"
                        pairs = [{"links": list(p["links"]), "penetration_mm": round(1000 * float(p["penetration"]), 1)} for p in self.self_collision_pairs(q_traj[first], emb_name)[:3]]
                except Exception:  # noqa: BLE001 - diagnostics only
                    pass
                prefix = q_traj[:first] if first > 1 else None  # the start + the collision-free waypoints
                diag = {"stage": "collision", "kind": kind, "fail_index": max(first - 1, 0), "n_waypoints": len(waypoints), "converged": max(first - 1, 0),
                        "self_collision": pairs, "at_lower": [], "at_upper": []}
                what = f"{kind} collision" + (f" ({', '.join('-'.join(p['links']) for p in pairs)})" if pairs else "")
                return PlanResult(success=False, q_traj=prefix, status=f"{what} at waypoint {max(first - 1, 0)}/{len(waypoints)}", dt=self.control_dt,
                                  plan_time=time.time() - t0, diagnostics=diag)
        status = "ok" if not soft else f"ok ({len(soft)} waypoints within the soft tolerance, max {max(w['pos_err'] for w in soft) * 1000:.1f} mm)"
        diag = {"singular_iterations": n_singular}
        if soft:
            diag["soft_waypoints"] = soft
        return PlanResult(success=True, q_traj=q_traj, dt=self.control_dt, status=status, plan_time=time.time() - t0, diagnostics=diag)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _lock_transform(origin: th.Tensor, jtype: JointType, val: float, device) -> th.Tensor:
    """``(origin @ motion(val))[:3, :4]`` of a locked joint, composed in **float64**. cuRobo (``seed_ik_solver``) and
    OmniGibson's curobo module switch ``torch.backends.cuda.matmul.allow_tf32`` on process-wide, so a float32 CUDA
    matmul here runs with a 10-bit mantissa: the R1Pro's articulation root stays at the spawn point (40, 40, 5) while
    its holonomic base moves through the six virtual joints, so a locked base joint carries 40-60 m and the composed
    transform was off by ~2^-11 of that - 14 mm at the door_crtrpa_0 stance (probe 0919 fk_probe: arm embodiments vs
    whole body < 0.4 mm within 4 m of the root, 14.4 mm at 58 m; the whole body keeps those joints active in cuRobo's
    own kernels and matched the simulator). Every arm IK / polish of the samplers and the oracle went through this."""
    m = _joint_motion(jtype, val, device, dtype=th.float64)
    return (origin.double() @ m)[:3, :4].to(origin.dtype)


def _joint_motion(jtype: JointType, val: float, device, dtype=th.float32) -> th.Tensor:
    """Homogeneous transform of a 1-DoF joint motion (cuRobo positive-axis convention)."""
    m = th.eye(4, device=device, dtype=dtype)
    t = jtype.value if isinstance(jtype, JointType) else int(jtype)
    if t == JointType.FIXED.value:
        return m
    neg = t >= JointType.X_PRISM_NEG.value
    if neg:
        t -= 6
        val = -val
    axis = t % 3
    if t < 3:  # prismatic
        m[axis, 3] = val
        return m
    c, s = math.cos(val), math.sin(val)
    if axis == 0:
        m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    elif axis == 1:
        m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    else:
        m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m
