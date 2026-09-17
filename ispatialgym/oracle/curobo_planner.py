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
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

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

log = logging.getLogger(__name__)


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
    ):
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
        self.use_cuda_graph = use_cuda_graph
        self.warmup_iterations = warmup_iterations

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

        # Shared world
        self.checker = create_collision_checker(
            SceneCollisionCfg(
                device_cfg=self.device_cfg,
                scene_model=None,
                cache={"mesh": mesh_cache_size, "cuboid": cuboid_cache_size},
            )
        )
        self._world_loaded = False
        self._world_mesh_names: set = set()
        self._disabled_obstacles: set = set()
        self._attached: Dict[str, dict] = {}  # arm -> info
        #: optional ``callable(obj) -> bool``: scene objects for which it returns False are never loaded as obstacles
        #: (e.g. furniture of neighbouring rooms that only exists for rendering; the mesh cache holds ``mesh_cache_size``)
        self.obstacle_filter = None

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
        planner = _SharedWorldMotionPlanner(cfg, self.checker)
        if self.warmup_iterations > 0:
            planner.warmup(enable_graph=True, num_warmup_iterations=self.warmup_iterations)

        entry = _PlannerEntry(embodiment=emb, planner=planner)
        entry.fk = Kinematics(planner.kinematics.config)  # same KinematicsCfg -> shares tensors with the solvers
        self._fill_lock_info(entry, entry.fk)
        entry.default_criteria = {f: ToolPoseCriteria(device_cfg=self.device_cfg) for f in planner.tool_frames}
        log.info("Built cuRobo embodiment '%s' (%d active joints) in %.1fs", emb.name, len(emb.active_joints), time.time() - t0)
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
        pos, quat = [], []
        for p in poses:
            rp, rq = self.world_to_root(p)
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
            kc.fixed_transforms[link_idx] = (origin @ _joint_motion(jtype, val, self.device_cfg.device))[:3, :4]
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

    def _iter_collision_meshes(self, ignore_objects=()):
        ignore = set()
        for o in ignore_objects or ():
            ignore.add(o if isinstance(o, str) else o.name)
        rp, rq = self.root_pose()
        for obj in self.robot.scene.objects:
            if obj is self.robot or getattr(obj, "visual_only", False) or obj.name in ignore:
                continue
            if self.obstacle_filter is not None and not self.obstacle_filter(obj):
                continue
            for link in obj.links.values():
                for cm in link.collision_meshes.values():
                    pos, quat = cm.get_position_orientation()
                    pos, quat = T.relative_pose_transform(pos, quat, rp, rq)
                    if self._is_ground_like(obj):
                        self._ground_mesh_names.add(cm.prim_path)
                    yield obj, link, cm, (pos, quat)

    def _mesh_geometry(self, cm):
        if cm.geom_type == "Mesh":
            verts = cm.points.numpy().astype(np.float32)
            faces = cm.faces.numpy().astype(np.int64)
        else:
            tm = mesh_prim_shape_to_trimesh_mesh(cm.prim)
            verts = np.asarray(tm.vertices, dtype=np.float32)
            faces = np.asarray(tm.faces, dtype=np.int64)
        scale = cm.get_world_scale().numpy().astype(np.float32)
        return verts, faces, scale

    def update_world(self, ignore_objects=(), ground: bool = True):
        """Synchronise the collision world with the simulator.

        First call loads every collision mesh (Warp BVH build, slow); later calls only update
        poses of already-known meshes and add new ones. Meshes of ignored / removed objects are
        disabled rather than deleted so the Warp cache stays valid.
        """
        t0 = time.time()
        if not self._world_loaded:
            meshes, names = [], set()
            for obj, link, cm, (pos, quat) in self._iter_collision_meshes(ignore_objects):
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
            log.info("cuRobo world loaded: %d meshes in %.2fs", len(meshes), time.time() - t0)
            return

        seen = set()
        for obj, link, cm, (pos, quat) in self._iter_collision_meshes(ignore_objects):
            name = cm.prim_path
            seen.add(name)
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
        for name in self._world_mesh_names - seen:
            if name not in self._disabled_obstacles:
                self._mesh_enable(name, False)
                self._disabled_obstacles.add(name)
        self._patch_mesh_query_dims()
        for e in self._entries.values():
            if e.planner.graph_planner is not None:
                e.planner.graph_planner.reset_buffer()

    # cuRobo 0.8.0's SceneData routers (update_obstacle_pose / enable_obstacle) only catch
    # RuntimeError while the cuboid store raises ValueError for unknown names, so mesh obstacles
    # are addressed through the mesh store directly.
    def _mesh_update_pose(self, name: str, pose: CuPose):
        self.checker.data.meshes.update_pose(name, w_obj_pose=pose)

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
                sphere_tensor = am.fit_spheres(obstacles, num_spheres=num_spheres or M.ATTACHED_OBJECT_NUM_SPHERES)
            am.update(sphere_tensor, self._active_js(entry, self.q_full()), link_name=link_name, world_objects_pose_offset=None)
        for n in mesh_names:
            if n in self._world_mesh_names:
                self._mesh_enable(n, False)
                self._disabled_obstacles.add(n)
        self._attached[arm] = {"obj": obj, "mesh_names": mesh_names, "planners": targets, "link": link_name}
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
            entry.rsc = {"self": SelfCollisionCost(self_cfg), "batch": 0, "buffer": None, "ground_contact": ground_contact}
        return entry.rsc

    def _ground_obstacles_enabled(self, enabled: bool):
        """Enable / disable every ground-like obstacle (slab + floor/carpet/... meshes).

        Meshes that are disabled for other reasons (ignored objects, attached objects) stay disabled.
        """
        cub = getattr(self.checker.data, "cuboids", None)
        if cub is not None and self.GROUND_SLAB_NAME in cub.names[0]:
            cub.set_enabled(self.GROUND_SLAB_NAME, enabled)
        for name in self._ground_mesh_names:
            if name not in self._world_mesh_names:
                continue
            if enabled and name in self._disabled_obstacles:
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

    def self_collision_pairs(self, q_full: th.Tensor, emb_name: str = "whole_body") -> List[Dict[str, object]]:
        """The link pairs of cuRobo's self-collision list whose padded spheres overlap at ``q_full``:
        ``[{"links": (a, b), "penetration": m}]`` (the worst sphere pair per link pair)."""
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
            pen = (sp[i, 3] + pad[i]) + (sp[j, 3] + pad[j]) - d
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
                if out.diagnostics.get("start_in_collision"):
                    out.status += " [start configuration is in collision]"
        finally:
            self._restore_criteria(entry)
        return out

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
            else:
                out.status = "no trajectory to the joint goal (graph search found no seed path)"
            if goal_coll:
                out.status += " [goal configuration is in collision]"
            if out.diagnostics.get("start_in_collision"):
                out.status += " [start configuration is in collision]"
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
            return {"start_in_collision": bool(self.check_collision(q_start, emb_name=emb_name)[0])}
        except Exception as e:  # diagnostics must never mask the original failure
            return {"start_in_collision": f"unknown ({e})"}

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

    @staticmethod
    def _diagnose_trajopt(result) -> Dict[str, object]:
        d: Dict[str, object] = {}
        try:
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

    def _batch_ik_entry(self, emb_name: str, batch_size: int) -> _PlannerEntry:
        """Dedicated IK solver (``max_batch_size = batch_size``) for ``emb_name`` sharing the scene checker; the
        collision spheres / self-collision data are the planner's, so results agree with :meth:`solve_ik`."""
        seeds = int(getattr(self, "batch_ik_seeds", 12))
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
            kc.fixed_transforms[link_idx] = (origin @ _joint_motion(jtype, val, self.device_cfg.device))[:3, :4]
        if lock_js is not None:
            for i, jn in enumerate(lock_js.joint_names):
                lock_js.position[..., i] = float(q_full[self._jidx[jn]])

    def solve_ik_batch(self, emb_name: str, frame: str, poses: Sequence[Pose], q_seed: th.Tensor, batch_size: int = 64, sync_world: bool = False):
        """Collision-aware IK for many world poses of one tool frame at once.

        Returns ``(success (N,) bool, q (N, 28))`` where ``q[i]`` is the best solution (full configuration, locked
        joints from ``q_seed``) when ``success[i]``. Poses are solved in chunks of ``batch_size``.
        """
        if sync_world:
            self.update_world()
        entry = self._batch_ik_entry(emb_name, batch_size)
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
    ) -> PlanResult:
        """Track dense Cartesian ``waypoints`` (each ``{frame: (pos, quat_xyzw)}`` in world) with
        damped-least-squares differential IK on the embodiment's active joints.

        ``soft_pos_tol`` / ``soft_rot_tol``: a waypoint the iterations leave within these (but outside ``pos_tol`` /
        ``rot_tol``) is accepted with its residual instead of failing the path (``diagnostics["soft_waypoints"]`` lists
        them) — a straight approach that stalls 2 mm short of a 2 mm tolerance is not a failed approach.

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
                first = int(th.nonzero(coll).view(-1)[0])
                return PlanResult(success=False, status=f"collision at waypoint {first}/{len(waypoints)}", dt=self.control_dt, plan_time=time.time() - t0)
        status = "ok" if not soft else f"ok ({len(soft)} waypoints within the soft tolerance, max {max(w['pos_err'] for w in soft) * 1000:.1f} mm)"
        diag = {"singular_iterations": n_singular}
        if soft:
            diag["soft_waypoints"] = soft
        return PlanResult(success=True, q_traj=q_traj, dt=self.control_dt, status=status, plan_time=time.time() - t0, diagnostics=diag)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _joint_motion(jtype: JointType, val: float, device) -> th.Tensor:
    """Homogeneous transform of a 1-DoF joint motion (cuRobo positive-axis convention)."""
    m = th.eye(4, device=device, dtype=th.float32)
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
