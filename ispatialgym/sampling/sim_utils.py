"""OmniGibson-side helpers for the sampler: environment construction, spawning, placement geometry
(support rasters), state snapshots, renders and the BEHAVIOR settle / validation checks.

Import order matters: :mod:`ispatialgym.oracle.curobo_planner` pins pip-warp and must be imported before
``omnigibson`` (done by :mod:`ispatialgym.sampling.robot_utils`, which this module imports first).
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch as th

from ispatialgym.sampling import robot_utils  # noqa: F401  (import before omnigibson: warp pinning)

import omnigibson as og  # noqa: E402
import omnigibson.utils.transform_utils as T  # noqa: E402
from omnigibson.macros import gm  # noqa: E402
from omnigibson.objects.dataset_object import DatasetObject  # noqa: E402
from omnigibson.utils.asset_utils import get_dataset_path  # noqa: E402
from omnigibson.utils.config_utils import TorchEncoder  # noqa: E402
from omnigibson.utils.data_utils import merge_scene_files  # noqa: E402
from omnigibson.utils.sampling_utils import raytest_batch  # noqa: E402

from ispatialgym.oracle import r1pro_model as M  # noqa: E402
from ispatialgym.sampling import geometry as G  # noqa: E402
from ispatialgym.sampling.knowledge import STRUCTURE_CATEGORIES  # noqa: E402

log = logging.getLogger("ispatialgym.sampling")

PARK_POSITION = th.tensor([40.0, 40.0, 5.0])
HEAD_APERTURE_MM = 40.0
IMAGE_SIZE = 1080


class Reject(Exception):
    """An instance attempt failed a check (counted per reason, attempt retried)."""

    def __init__(self, reason: str, message: str = "", **details):
        super().__init__(f"[{reason}] {message}" + (f" {details}" if details else ""))
        self.reason = reason
        self.details = details


def stable_scene_path(scene_model: str) -> str:
    return os.path.join(get_dataset_path("2026-challenge-task-instances"), "scenes", scene_model, "json", f"{scene_model}_stable.json")


def adjacent_rooms(scene_model: str, rooms: Sequence[str], reach_m: float = 0.6) -> List[str]:
    """Room instances that share a wall / doorway with any of ``rooms``: the scene's room instance segmentation map is
    dilated by ``reach_m`` (walls are 10-30 cm thick) and every other room instance the dilated mask touches counts.
    Only the map PNGs are read, so this can run before the scene is loaded."""
    from omnigibson.maps.segmentation_map import SegmentationMap
    from omnigibson.utils.asset_utils import get_scene_path

    seg = SegmentationMap(scene_dir=get_scene_path(scene_model), map_resolution=0.1)
    ids = [seg.room_ins_name_to_ins_id[r] for r in rooms if r in seg.room_ins_name_to_ins_id]
    if not ids:
        return []
    mask = th.zeros_like(seg.room_ins_map, dtype=th.bool)
    for i in ids:
        mask |= seg.room_ins_map == i
    k = 2 * int(round(reach_m / seg.map_resolution)) + 1
    dilated = th.nn.functional.max_pool2d(mask.float()[None, None], k, stride=1, padding=k // 2)[0, 0] > 0.5
    touched = th.unique(seg.room_ins_map[dilated]).tolist()
    # ``ISG_ADJACENT_EXCLUDE`` (comma separated name substrings) drops rooms from the neighbour list. Default none: the
    # garden and the kitchen of house_single_floor used to crash the sampler's seg_instance renders, which was the
    # light-shape bug guarded in ``SimWorld._set_measurement_render_settings`` and not a property of those rooms.
    exclude = tuple(s for s in os.environ.get("ISG_ADJACENT_EXCLUDE", "").split(",") if s)
    names = sorted(seg.room_ins_id_to_ins_name[int(i)] for i in touched if int(i) in seg.room_ins_id_to_ins_name and int(i) not in ids and int(i) != 0)
    return [n for n in names if not any(s in n for s in exclude)]


# ----------------------------------------------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------------------------------------------


class SimWorld:
    """One OmniGibson environment (scene + R1Pro) plus the helpers every task sampler needs."""

    def __init__(self, scene_model: str, rooms: Optional[Sequence[str]], robot_name: str = "robot_r1", image_size: int = IMAGE_SIZE,
                 load_adjacent: Optional[bool] = None, scene_variant: Optional[dict] = None, cameras: bool = True):
        """``cameras=False``: the robot carries no vision sensor (``obs_modalities=["proprio"]`` - OmniGibson creates a
        ``VisionSensor`` only for requested vision modalities), so no 1080^2 rgb + seg_instance buffers, annotators or
        per-frame synthetic-data graph exist; ``cam_key`` / ``cam`` / ``camera_model`` are ``None`` and the render /
        segmentation helpers raise. For the stages that never read pixels (reach bands: kinematics, raycasts, cuRobo)."""
        # macros lock once they have been read (second SimWorld in one process after og.clear())
        for name, value in (("ENABLE_OBJECT_STATES", True), ("USE_GPU_DYNAMICS", False), ("ENABLE_TRANSITION_RULES", False)):
            if getattr(gm, name) != value:
                with gm.unlocked():
                    setattr(gm, name, value)
        self.scene_model = scene_model
        #: ``rooms``: where the task takes place (stances, spawned objects, parking). ``adjacent``: their neighbours
        #: (shared wall / doorway), so that walls, doors and furniture seen through doorways and windows exist at
        #: evaluation time (no sky through missing walls). ``eval_rooms`` (rooms + adjacent) is what templates record as
        #: ``load_room_instances`` and, with ``load_adjacent`` (``ISG_LOAD_ADJACENT``, default on), also what the sampler
        #: itself loads, so that sampling, evaluation and the oracle share one collision and visual world (a stance
        #: validated here cannot collide with a neighbour's furniture in the demo). v1 loaded the task rooms only because
        #: the measurement renders segfaulted with the neighbours in view; see ``_set_measurement_render_settings``.
        self.rooms = list(rooms) if rooms else None
        if load_adjacent is None:
            load_adjacent = os.environ.get("ISG_LOAD_ADJACENT", "1") == "1"
        self.adjacent = adjacent_rooms(scene_model, self.rooms) if self.rooms else []
        self.eval_rooms = (self.rooms + [r for r in self.adjacent if r not in self.rooms]) if self.rooms else None
        self.loaded_rooms = self.eval_rooms if load_adjacent else self.rooms
        self.robot_name = robot_name
        #: scene variant (``scene_variants.py``): stable scene minus e.g. the chairs around the work table. Everything
        #: downstream (template merge, scene-unchanged checks, resets) compares against the variant.
        self.scene_variant = scene_variant or {}
        self.stable_path = self.scene_variant.get("path") or stable_scene_path(scene_model)
        with open(self.stable_path) as f:
            self.stable_scene = json.load(f)
        cfg = {
            "env": {"action_frequency": 30, "physics_frequency": 120},
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": scene_model,
                "scene_file": self.stable_path,
                "seg_map_resolution": 0.1,
                "trav_map_resolution": 0.1,
                "load_room_instances": self.loaded_rooms,
            },
            "robots": [
                M.joylo_robot_config(
                    robot_name,
                    obs_modalities=["rgb", "seg_instance"] if cameras else ["proprio"],
                    sensor_config={"VisionSensor": {"sensor_kwargs": {"image_height": image_size, "image_width": image_size}}} if cameras else {},
                    position=PARK_POSITION.tolist(),
                    **M.camera_overrides(),
                )
            ],
        }
        t0 = time.time()
        self.env = og.Environment(configs=cfg)
        self._set_measurement_render_settings()
        self.scene = self.env.scene
        self.robot = self.env.scene.robots[0]
        # the BEHAVIOR-1K evaluator's 250 kg chassis, so the sampler's physics probes (passive continuation, stable set,
        # the drop tests) see the grader's robot (owner 2026-09-21 23:49; ``r1pro_model.apply_eval_base_mass``)
        self.base_link_mass_kg = M.apply_eval_base_mass(self.robot)
        log.info("robot base_link mass %.2f -> %.2f kg (BEHAVIOR-1K eval: %.0f)", self.base_link_mass_kg["before"], self.base_link_mass_kg["after"], M.EVAL_BASE_LINK_MASS_KG)
        self.cameras = cameras
        self.cam_key = self.cam = self.camera_model = None
        if cameras:
            self.cam_key = [k for k in self.robot.sensors if k.endswith(M.HEAD_CAMERA_SENSOR_SUFFIX)][0]
            self.cam = self.robot.sensors[self.cam_key]
            self.cam.horizontal_aperture = HEAD_APERTURE_MM
            self.camera_model = G.CameraModel.from_sensor(float(self.cam.focal_length), HEAD_APERTURE_MM, image_size, image_size)
        for obj in self.scene.objects:
            obj.keep_still()
        self.n_scene_objects = len(self.scene.objects)
        self.scene_object_names = {o.name for o in self.scene.objects}
        self.spawned: Dict[str, DatasetObject] = {}
        #: the traversability map's nominal floor height (OmniGibson's default 0.0 in every scene) — only a fallback: the
        #: task room's actual floor surface is measured below (``measure_floor_z``) and drives every robot height
        self.floor_z_default = float(self.scene.trav_map.floor_heights[0]) if getattr(self.scene, "trav_map", None) is not None else 0.0
        self.floor_z = self.floor_z_default
        self.floor_z_by_room: Dict[str, float] = {}
        self.floor_probe: Dict[str, Any] = {}
        if self.scene_variant.get("freed_footprints"):
            # the traversability PNG is baked with the furniture: free the removed objects' footprints
            from ispatialgym.sampling.scene_variants import patch_trav_map

            n_freed = patch_trav_map(self.scene, self.scene_variant["freed_footprints"])
            log.info("traversability map patched for the scene variant: %d cells freed", n_freed)
        #: the simulation state right after the environment's last rendering step: what the render side (Fabric transforms:
        #: ``aabb``, ``collision_boundary_points_world``, the collision-mesh poses cuRobo reads) keeps showing while only
        #: physics steps follow (``park_robot``'s five). A process that reuses this world for several independent
        #: computations restores it, then the parked state, before each one (``reach_bands.reset_world_between_bands``), so
        #: that every computation sees exactly the render-side and physics-side poses of a fresh process (measured
        #: 2026-09-16: without it the support table had crept by ~4e-6 m after two more rendering steps and 7 of 111
        #: (yaw, grasp) pairs of a band changed). ``park_cell`` / ``measure_floor_z`` below only raycast (no step).
        self.initial_state = og.sim.dump_state()
        xy = self.park_cell()
        self.floor_z = self.measure_floor_z(xy)
        self.robot_park_xy = self.park_robot(xy, floor_z=self.floor_z)
        self.probe_floor_height(self.robot_park_xy)
        self.scene.update_initial_file()
        # ``ISG_HIDE_CATEGORY_SUBSTRINGS`` (comma separated category substrings, default none): emergency override that
        # hides scene objects while sampling and shows them again around template saving (see save_template). v1 hid
        # ``light,lamp,chandelier`` against the segmentation crash; ``_set_measurement_render_settings`` removes its
        # cause, so nothing is hidden any more and every measurement sees the scene as evaluation renders it.
        subs = tuple(s for s in os.environ.get("ISG_HIDE_CATEGORY_SUBSTRINGS", "").split(",") if s)
        self.render_hidden = [o for o in self.scene.objects if any(s in o.category for s in subs)]
        for o in self.render_hidden:
            o.visible = False
        if self.render_hidden:
            log.info("hidden for rendering: %s", [o.name for o in self.render_hidden])
        self.load_time = time.time() - t0
        self.robot_visible = True
        # collision-world relevance: the task rooms' floor extent plus a margin. Neighbouring rooms are loaded for what
        # the cameras see through doorways and windows; their furniture is not an obstacle for a robot working here.
        self.task_bounds = None
        if self.rooms:
            los, his = [], []
            for o in self.scene.objects:
                if o.category == "floors" and set(o.in_rooms or []) & set(self.rooms):
                    lo, hi = o.aabb
                    los.append(lo)
                    his.append(hi)
            if los:
                m = float(os.environ.get("ISG_OBSTACLE_MARGIN_M", "1.0"))
                self.task_bounds = (th.stack(los).min(dim=0).values - m, th.stack(his).max(dim=0).values + m)
                self.task_bounds[1][2] = 10.0  # everything above the floor (ceilings, wall-mounted objects)
        n_obst = sum(1 for o in self.scene.objects if self.obstacle_filter(o))
        log.info("scene %s loaded (%d objects, %d collision-relevant, rooms=%s, adjacent=%s) in %.1fs", scene_model, self.n_scene_objects, n_obst,
                 self.rooms, self.adjacent, self.load_time)

    def obstacle_filter(self, obj) -> bool:
        """Whether ``obj`` is loaded into the IK collision world: spawned task objects always, scene objects when their
        AABB intersects the task rooms' extent (walls / doors / furniture of neighbours near a doorway included)."""
        if self.task_bounds is None or obj.name in self.spawned or obj is self.robot:
            return True
        lo, hi = obj.aabb
        blo, bhi = self.task_bounds
        return bool(((hi >= blo) & (lo <= bhi)).all())

    # ------------------------------------------------------------------ objects

    def obj(self, name: str):
        o = self.scene.object_registry("name", name)
        if o is None:
            raise KeyError(f"no object {name!r} in scene {self.scene_model}")
        return o

    def spawn(self, name: str, category: str, model: str, dataset: str = "behavior-1k-assets", scale=None, fixed_base: bool = False,
              extra_abilities: Optional[Dict[str, dict]] = None) -> DatasetObject:
        kwargs = {}
        if scale is not None:
            kwargs["scale"] = th.tensor(scale, dtype=th.float32)
        if self.rooms:
            # partial-room loading (``load_room_instances``) skips scene-file objects without a room assignment, so a
            # template must record the loaded rooms on every spawned object (BEHAVIOR templates do the same)
            kwargs["in_rooms"] = list(self.rooms)
        if extra_abilities:
            # object states follow the abilities; e.g. AttachedTo needs "attachable", which the taxonomy does not give
            # every model that carries attachment meta links
            from ispatialgym.sampling import knowledge as K

            abilities = K.abilities_of_category(category)
            abilities.update(extra_abilities)
            kwargs["abilities"] = abilities
        obj = DatasetObject(name=name, category=category, model=model, dataset_name=dataset, fixed_base=fixed_base, **kwargs)
        # Adding (articulated) objects while playing can invalidate PhysX tensor views after earlier removals; do it
        # stopped and restore the simulation state afterwards (as BEHAVIOR's sample_robot_pose.py does).
        state = og.sim.dump_state()
        og.sim.stop()
        self.scene.add_object(obj)
        og.sim.play()
        og.sim.load_state(state)
        # stop / play rebuilt PhysX from USD: the objects parked earlier must stay parked (gravity off is written to USD by
        # ``park`` since 2026-09-16, and their poses / velocities are re-applied here so nothing drifts before the next read)
        for i, other in enumerate(self.spawned.values()):
            self.park(other, i)
        self.spawned[name] = obj
        self.park(obj, len(self.spawned) - 1)
        og.sim.step()
        return obj

    def remove(self, obj):
        self.spawned.pop(obj.name, None)
        # never remove an object the robot still holds with assisted grasping: the robot's state would keep referring to it
        try:
            for arm, held in getattr(self.robot, "_ag_obj_in_hand", {}).items():
                if held is obj:
                    self.robot.release_grasp_immediately(arm)
        except Exception:
            pass
        state = og.sim.dump_state()
        og.sim.stop()
        self.scene.remove_object(obj)
        og.sim.play()
        try:
            og.sim.load_state(state)
        except Exception:
            pass
        og.sim.step()

    def park(self, obj, index: int = 0):
        """Move ``obj`` far away with gravity disabled (there is no floor there: partial-room loading). Gravity is
        switched off **persistently** (:meth:`set_gravity`): the PhysX view alone forgets it at the next
        ``og.sim.stop()`` / ``play()`` of a later :meth:`spawn`."""
        obj.set_position_orientation(PARK_POSITION + th.tensor([index * 1.0, 0.0, 0.0]), G.IDENTITY_QUAT)
        self.set_gravity(obj, False)
        obj.keep_still()

    def activate(self, obj):
        """Re-enable gravity for an object that has been placed. The body is woken explicitly: a sleeping body that was
        teleported through the physics API keeps its old render transform (it is then absent from segmentation renders
        although PhysX has it in view) until it is active during a rendering step."""
        self.set_gravity(obj, True)
        obj.wake()
        obj.keep_still()

    def set_gravity(self, obj, enabled: bool):
        """Gravity on / off for every rigid link of ``obj``, in PhysX **and** in USD.

        ``RigidPrimView.disable_gravities`` writes the PhysX tensor view while the simulation plays and the USD attribute
        only when it is stopped; ``og.sim.stop()`` / ``play()`` (every later :meth:`spawn`) rebuilds PhysX from USD, so a
        PhysX-only switch is undone and parked objects fall (measured in the E1 pilot: the source tray's root at z 4.84
        instead of 5.00 and still falling at ``sample_template`` time). The USD ``PhysxRigidBodyAPI.disableGravity``
        attribute is therefore written as well (own ``editing_usd`` block — the prim methods open their own and the
        context must not nest)."""
        if enabled:
            obj.enable_gravity()
        else:
            obj.disable_gravity()
        import omnigibson.lazy as lazy

        with og.sim.editing_usd():
            for link in obj.links.values():
                prim = getattr(link, "prim", None)
                if prim is None or not prim.HasAPI(lazy.pxr.UsdPhysics.RigidBodyAPI):
                    continue
                api = lazy.pxr.PhysxSchema.PhysxRigidBodyAPI(prim) if prim.HasAPI(lazy.pxr.PhysxSchema.PhysxRigidBodyAPI) \
                    else lazy.pxr.PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                api.GetDisableGravityAttr().Set(not enabled)

    def gravity_disabled_in_usd(self, obj) -> bool:
        """Whether every rigid link of ``obj`` has ``disableGravity = True`` in USD (probe helper)."""
        import omnigibson.lazy as lazy

        vals = []
        for link in obj.links.values():
            prim = getattr(link, "prim", None)
            if prim is None or not prim.HasAPI(lazy.pxr.UsdPhysics.RigidBodyAPI):
                continue
            if not prim.HasAPI(lazy.pxr.PhysxSchema.PhysxRigidBodyAPI):
                vals.append(False)
                continue
            vals.append(bool(lazy.pxr.PhysxSchema.PhysxRigidBodyAPI(prim).GetDisableGravityAttr().Get()))
        return bool(vals) and all(vals)

    def park_cell(self) -> th.Tensor:
        """World xy of the robot's parking cell: the free (eroded) traversability cell nearest the centre of the free
        cells inside the task rooms (elsewhere there is no floor under partial-room loading and the robot falls, which
        silently shifts the cuRobo world: meshes are stored relative to the robot root at sync time)."""
        trav = self.scene.trav_map
        m = trav._erode_trav_map(th.clone(trav.floor_map[0]), robot=self.robot)
        free = th.nonzero(m == 255)
        if len(free) == 0:
            raise RuntimeError("no free traversable cell to park the robot")
        if self.rooms:
            seg = self.scene.seg_map
            in_rooms = th.tensor([seg.get_room_instance_by_point(trav.map_to_world(c)) in self.rooms for c in free])
            if bool(in_rooms.any()):
                free = free[in_rooms]
            else:
                log.warning("no free traversable cell inside %s; parking the robot elsewhere", self.rooms)
        centre = free.float().mean(dim=0)
        k = int(th.argmin(th.linalg.norm(free.float() - centre, dim=1)))
        return trav.map_to_world(free[k])

    #: a floor hit counts when it lies within this of the nominal floor height (a step / mezzanine is another floor)
    FLOOR_MEASURE_RANGE = 0.30

    def room_of(self, xy) -> Optional[str]:
        try:
            return self.scene.seg_map.get_room_instance_by_point(G.t(xy)[:2])
        except Exception:  # noqa: BLE001 - no segmentation map for this scene
            return None

    def measure_floor_z(self, xy, room: Optional[str] = None) -> float:
        """Height of the walkable floor surface under world ``xy`` — what the robot actually stands on.

        ``trav_map.floor_heights[0]`` is OmniGibson's default 0.0 in every scene, but a room's floor slab may sit
        elsewhere (``office_cubicles_right`` / ``shared_office_0``: ``floors_ovpbsn_0`` collision top at z +0.0071); a stance
        check standing ``base_link`` at ``0.0 + NOMINAL_BASE_Z`` then sinks the wheel spheres 5 mm into the slab and cuRobo's
        penetration query — which has no ground-like exemption — fails **every** configuration in the room (0 / 205 in
        the articulation diagnosis). Order: (1) downward raycast at ``xy`` (the robot's bodies ignored) hitting within
        ``FLOOR_MEASURE_RANGE`` of the nominal height; (2) the room's ``floors`` object's collision-hull top; (3) its
        AABB top; (4) the nominal height. The result is cached per room (``floor_z_by_room``) and logged."""
        room = room or self.room_of(xy)
        if room in self.floor_z_by_room:
            return self.floor_z_by_room[room]
        z0 = self.floor_z_default
        x, y = float(G.t(xy)[0]), float(G.t(xy)[1])
        z, how = None, None
        try:
            ignore = [link.prim_path for link in self.robot.links.values()]
            hits = raytest_batch([[x, y, z0 + 0.5]], [[x, y, z0 - 0.5]], only_closest=True, ignore_bodies=ignore)
            h = hits[0] if hits else {}
            if h.get("hit") and abs(float(h["position"][2]) - z0) <= self.FLOOR_MEASURE_RANGE:
                z, how = float(h["position"][2]), f"raycast on {h.get('rigidBody')}"
        except Exception as e:  # noqa: BLE001
            log.debug("floor raycast failed at (%.2f, %.2f): %r", x, y, e)
        if z is None and room is not None:
            try:
                floors = self.floor_of_room(room)
                self.sync_render_transforms()
                tops = [float(p[:, 2].max()) for p in (l.collision_boundary_points_world for l in floors.links.values()) if p is not None and len(p)]
                if tops and abs(max(tops) - z0) <= self.FLOOR_MEASURE_RANGE:
                    z, how = max(tops), f"collision hull top of {floors.name}"
                else:
                    top = float(floors.aabb[1][2])
                    if abs(top - z0) <= self.FLOOR_MEASURE_RANGE:
                        z, how = top, f"AABB top of {floors.name}"
            except Exception as e:  # noqa: BLE001 - no floors object in the room
                log.debug("no floors object for room %s: %r", room, e)
        if z is None:
            z, how = z0, "trav_map.floor_heights[0] (nothing measured)"
        z = float(z)
        if room is not None:
            self.floor_z_by_room[room] = z
        log.info("floor height in %s at (%.2f, %.2f): z = %.4f (%s; nominal %.4f, delta %+.1f mm)", room or "?", x, y, z, how, z0, (z - z0) * 1000)
        return z

    def probe_floor_height(self, xy, tol_m: float = 0.002) -> Dict[str, Any]:
        """After parking at the measured floor height: the physically settled ``base_link`` z minus ``NOMINAL_BASE_Z`` must
        agree with the raycast floor within ``tol_m`` (2 mm); the result is kept in ``floor_probe`` (template meta) and a
        disagreement is logged as a warning (the checks would stand the robot at a wrong height)."""
        settled = float(self.robot.get_position_orientation()[0][2])
        floor_from_settled = settled - robot_utils.NOMINAL_BASE_Z
        diff = floor_from_settled - self.floor_z
        self.floor_probe = {"room": self.room_of(xy), "floor_z_measured": round(self.floor_z, 4), "floor_z_default": round(self.floor_z_default, 4),
                            "settled_base_z": round(settled, 4), "settled_minus_nominal": round(floor_from_settled, 4), "diff_mm": round(diff * 1000, 2),
                            "ok": abs(diff) <= tol_m}
        (log.info if abs(diff) <= tol_m else log.warning)("floor probe: raycast floor %.4f vs settled base %.4f - %.4f = %.4f (%+.1f mm)%s", self.floor_z,
                                                           settled, robot_utils.NOMINAL_BASE_Z, floor_from_settled, diff * 1000,
                                                           "" if abs(diff) <= tol_m else " - DISAGREE beyond 2 mm")
        return self.floor_probe

    def park_robot(self, xy: Optional[th.Tensor] = None, floor_z: Optional[float] = None):
        """Stand the robot on a free traversability-map cell of the loaded floor (it must not float / fall) at
        ``floor_z + NOMINAL_BASE_Z`` (``floor_z`` = the measured floor of the task room by default)."""
        if xy is None:
            xy = self.park_cell()
        if floor_z is None:
            floor_z = self.floor_z
        self.robot.set_joint_positions(self.robot.tucked_default_joint_pos)
        self.robot.set_joint_velocities(th.zeros(self.robot.n_joints))
        self.robot.set_position_orientation(th.tensor([float(xy[0]), float(xy[1]), floor_z + robot_utils.NOMINAL_BASE_Z]), G.IDENTITY_QUAT)
        self.robot.keep_still()
        for _ in range(5):
            og.sim.step_physics()
            self.robot.keep_still()
        return xy

    def local_hull(self, obj) -> th.Tensor:
        """Collision boundary points of every link in the object's root frame (for footprints at any orientation).

        ``link.collision_boundary_points_world`` is transformed with the **render-side** (Fabric / USD) link transform,
        which physics-only steps do not refresh (``step_physics`` runs ``PhysicsContext._step(update_fabric=False)``),
        while the root pose comes from PhysX. A body that moved since the last render or teleport therefore gave a hull
        shifted by that motion (E1 pilot: +6.5 cm in z for a falling tray, ``bottom_offset`` off by the same amount, the
        tray placed inside the table slab). PhysX is synchronised to Fabric first, so both sides describe the same pose.
        """
        self.sync_render_transforms()
        root = obj.get_position_orientation()
        pts = []
        for link in obj.links.values():
            p = link.collision_boundary_points_world
            if p is None or len(p) == 0:
                continue
            pts.append(p)
        if not pts:
            return th.zeros(0, 3)
        world = th.cat(pts, dim=0)
        Rt = T.quat2mat(root[1]).T
        return (world - root[0]) @ Rt.T

    def sync_render_transforms(self):
        """Push the PhysX poses to the render-side (Fabric) transforms that ``collision_boundary_points_world`` / ``aabb``
        read (``og.sim.sync_physx_to_fabric``; a single render frame when the fabric interface is unavailable)."""
        try:
            og.sim.sync_physx_to_fabric()
        except Exception as e:  # noqa: BLE001 - e.g. no fabric interface in this build: a render does the same, slower
            log.debug("sync_physx_to_fabric unavailable (%r); rendering one frame instead", e)
            og.sim.render()

    def hull_at(self, local_hull: th.Tensor, pose: G.Pose) -> th.Tensor:
        return G.local_points_world(local_hull, pose)

    def footprint(self, local_hull: th.Tensor, pose: G.Pose) -> th.Tensor:
        return G.convex_hull_2d(self.hull_at(local_hull, pose)[:, :2])

    def bottom_offset(self, local_hull: th.Tensor, quat) -> float:
        """Distance from the root origin down to the lowest collision point when the object has orientation ``quat``."""
        pts = G.local_points_world(local_hull, (th.zeros(3), G.t(quat)))
        return -float(pts[:, 2].min())

    def top_offset(self, local_hull: th.Tensor, quat) -> float:
        pts = G.local_points_world(local_hull, (th.zeros(3), G.t(quat)))
        return float(pts[:, 2].max())

    def place(self, obj, xy, quat, support_z: float, local_hull: Optional[th.Tensor] = None, drop: float = 0.003):
        """Put ``obj`` with orientation ``quat`` so that its lowest collision point is ``drop`` above ``support_z``."""
        hull = self.local_hull(obj) if local_hull is None else local_hull
        z = support_z + self.bottom_offset(hull, quat) + drop
        pos = th.tensor([float(xy[0]), float(xy[1]), z])
        obj.set_position_orientation(pos, G.t(quat))
        self.activate(obj)
        return pos

    # ------------------------------------------------------------------ sim stepping / state

    def step_physics(self, n: int, still: Iterable = ()):
        still = list(still)
        for _ in range(n):
            og.sim.step_physics()
            for e in still:
                e.keep_still()

    def full_step(self, render: bool = True):
        """One full simulator step. With ``render`` the physics -> USD/Fabric transform sync happens, which is what
        refreshes the per-step object-state caches (AABBs, contacts, kinematic 'changed' flags): a non-rendering full
        step leaves AABB-based predicates (Inside, OnTop) stale, so rendering stays on wherever predicates follow."""
        with og.sim.render_on_step(render):
            og.sim.step()

    def step(self, n: int = 1, render: bool = True):
        for _ in range(n):
            self.full_step(render)

    def control_step(self, action: th.Tensor, n: int = 1):
        """Apply a JoyLo action and step the simulator without rendering and without the environment's observation
        pipeline (``env.step`` also reads and remaps six 1080^2 sensor buffers per step). Assisted grasping runs inside
        the simulator step (``Robot.post_step``) on PhysX state, so grasps are established exactly as through
        ``env.step``; callers that evaluate AABB-based predicates afterwards take one rendering :meth:`full_step`."""
        for _ in range(n):
            self.robot.apply_action(action)
            self.full_step(render=False)

    def snapshot(self):
        return og.sim.dump_state(serialized=False)

    def restore(self, state):
        og.sim.load_state(state, serialized=False)
        og.sim.step_physics()

    def keep_still(self, entities: Iterable):
        for e in entities:
            e.keep_still()

    def wake(self, entities: Iterable):
        for e in entities:
            if isinstance(e, DatasetObject):
                e.wake()

    # ------------------------------------------------------------------ renders

    #: RTX flag that draws the *shapes* of light sources (the UsdLux prims inside light-fixture models) as visible
    #: emitters. OmniGibson switches it on at launch (``Simulator._set_renderer_settings``). Isaac Sim 5.1's
    #: instance-segmentation reduction (``omni.replicator.core.InstanceSegmentationReduction``, the C++ node behind
    #: ``seg_instance``, ``seg_instance_fast`` and ``seg_semantic``) cannot map the instance ids of those shapes: with a
    #: light shape in view (directly or through glass) other pixels come out ``unlabelled`` (up to a third of the
    #: image on house_single_floor) and with enough of them the node indexes out of bounds and segfaults the process.
    #: Bisected 2026-09-15 on house_single_floor with the neighbours loaded, 8 living-room views (sliding door + garden,
    #: kitchen, dining room, corridor, high window) x 4 renders: flag off + nothing hidden = no crash and zero
    #: unlabelled pixels; the v1 guard (light fixtures hidden, flag on) still crashed on the kitchen view, as did the
    #: viewer-camera path of BEHAVIOR-1K #2312 and translucency off. Only the emitter shape disappears from rgb; the
    #: fixture meshes stay visible and the illumination is unchanged.
    LIGHT_SHAPES_SETTING = "/rtx/raytracing/showLights"

    def _set_measurement_render_settings(self):
        """Make every rendered frame safe for the head camera's ``seg_instance`` annotator, which runs the synthetic-data
        graph on *every* frame (rgb reads, rendering simulator steps), not only inside :meth:`seg_instance`; a guard
        that only wraps the measurement renders would leave those frames exposed. Nothing is hidden: see
        ``LIGHT_SHAPES_SETTING``."""
        import carb

        settings = carb.settings.get_settings()
        if settings.get(self.LIGHT_SHAPES_SETTING) != 0:
            settings.set_int(self.LIGHT_SHAPES_SETTING, 0)
            log.info("renderer: %s=0 (light shapes not drawn; seg_instance crash guard)", self.LIGHT_SHAPES_SETTING)

    def render(self, n: int = 3):
        for _ in range(n):
            og.sim.render()

    def _require_camera(self):
        if self.cam is None:
            raise RuntimeError("this SimWorld was created with cameras=False (no head camera): renders / segmentation / camera poses are unavailable")

    #: frames rendered per measurement: at least this many (see ``seg_instance``), at most ``SEG_MAX_RENDERS``
    SEG_MIN_RENDERS = 4
    SEG_MAX_RENDERS = 8

    def _raw_seg_instance(self) -> th.Tensor:
        """The head camera's instance-segmentation buffer as the annotator currently holds it (unremapped ids, a
        copy); a cheap fetch (no rgb / semantic read, no remapping) used to detect when the buffer has settled."""
        self._require_camera()
        raw = self.cam._annotators["seg_instance"].get_data(device=og.sim.device)
        data = raw["data"] if isinstance(raw, dict) else raw
        seg = self.cam._preprocess_cpu_obs(data, "seg_instance") if og.sim.device == "cpu" else self.cam._preprocess_gpu_obs(data, "seg_instance")
        return seg.clone()

    def seg_instance(self, n_renders: int = 3) -> Tuple[th.Tensor, Dict[int, str]]:
        """Instance segmentation of the head camera: ``(label map (H, W) int64, {id: object name | 'background' |
        'unlabelled' | 'groundPlane' | particle system})``. Callers compare the mapping values (object names), never
        the ids.

        The annotator data lags the scene: a pose or visibility change reaches the renderer one frame late, and the
        buffer a read returns trails the last rendered frame by a further, load-dependent one to two frames (measured
        2026-09-15: change, 3 renders, read = the old view; a second read without rendering = the same; under GPU load
        even render, read, render, read returned the old view). A fixed number of frames is therefore not enough. This
        renders at least ``max(n_renders, SEG_MIN_RENDERS)`` frames and keeps rendering one frame at a time until two
        consecutive fetches of the raw buffer are identical (the scene is static during a measurement, so a settled
        buffer repeats exactly), up to ``SEG_MAX_RENDERS``; then the remapped observation is read."""
        self._require_camera()
        n_min = max(int(n_renders), self.SEG_MIN_RENDERS)
        self.render(n_min - 2)
        prev = self._raw_seg_instance()
        n = n_min - 2
        while True:
            self.render(1)
            n += 1
            cur = self._raw_seg_instance()
            if n >= n_min and th.equal(cur, prev):
                break
            if n >= self.SEG_MAX_RENDERS:
                log.warning("seg_instance: label map still changing after %d frames; using the latest", n)
                break
            prev = cur
        obs, info = self.robot.get_obs()
        seg = obs[self.cam_key]["seg_instance"]
        mapping = {int(k): v for k, v in info[self.cam_key]["seg_instance"].items()}
        return seg.cpu(), mapping

    def depth_linear(self) -> th.Tensor:
        """The head camera's linear depth ``(H, W)`` in m (distance along the optical axis), rendered until the buffer has
        settled (``ispatialgym.oracle.head_depth``: a kept extra annotator, the camera's observations unchanged)."""
        from ispatialgym.oracle.head_depth import settled_depth

        self._require_camera()
        return th.as_tensor(settled_depth(self.cam, lambda: self.render(1)))

    def visible_pixels(self, n_renders: int = 3, exclude_structure: bool = True) -> Dict[str, int]:
        seg, mapping = self.seg_instance(n_renders)
        ids, counts = th.unique(seg, return_counts=True)
        out = {}
        for i, c in zip(ids.tolist(), counts.tolist()):
            name = mapping.get(int(i))
            if name is None or name in ("background", "unlabelled", "groundPlane", "BACKGROUND", "UNLABELLED"):
                continue
            if exclude_structure:
                o = self.scene.object_registry("name", name)
                if o is not None and o.category in STRUCTURE_CATEGORIES:
                    continue
            out[name] = int(c)
        return out

    def set_visible(self, objs: Iterable, visible: bool):
        for o in objs:
            o.visible = visible

    def object_pixels(self, obj, hidden: Iterable = (), n_renders: int = 3) -> int:
        hidden = list(hidden)
        self.set_visible(hidden, False)
        try:
            px = self.visible_pixels(n_renders, exclude_structure=False).get(obj.name, 0)
        finally:
            self.set_visible(hidden, True)
        return px

    def ensure_segmented(self, obj) -> bool:
        """True if ``obj`` appears in the instance-segmentation label map.

        A teleported sleeping body occasionally keeps a stale render transform (it is physically in view but the
        replicator never lists it), non-deterministically. Remedies in order: wake + full simulator step (physics ->
        render transform sync), visibility toggle, re-writing the pose. Each is followed by fresh renders.
        """

        def present():
            _, mapping = self.seg_instance()
            return obj.name in mapping.values()

        if present():
            return True
        self.wake([obj])
        self.robot.keep_still()
        self.full_step(render=True)
        if present():
            return True
        obj.visible = False
        self.render(2)
        obj.visible = True
        self.render(3)
        if present():
            return True
        p, q = obj.get_position_orientation()
        obj.set_position_orientation(p, q)
        obj.keep_still()
        self.full_step(render=True)
        self.robot.keep_still()
        if present():
            return True
        # nothing at all labelled (only background / unlabelled): the instance-segmentation annotator went stale
        # (seen in the second world of a process after og.clear()); re-create it on the head camera
        _, mapping = self.seg_instance()
        if len(mapping) <= 2:
            log.warning("instance segmentation returned no labels; re-creating the annotator on %s", self.cam_key)
            try:
                self.cam.remove_modality("seg_instance")
                self.render(2)
                self.cam.add_modality("seg_instance")
                self.render(3)
            except Exception as e:  # noqa: BLE001
                log.warning("re-creating the seg_instance annotator failed: %r", e)
        return present()

    def rgb(self, n_renders: int = 3):
        """Head camera rgb (debug / contact sheets, no measurement). Read after the segmentation buffer has settled
        (:meth:`seg_instance`): a lone read returns a frame one to two renders old, see there."""
        self.seg_instance(n_renders)
        obs, _ = self.robot.get_obs()
        return obs[self.cam_key]["rgb"][..., :3].cpu()

    # ------------------------------------------------------------------ raycasts / supports

    def support_top(self, support, resolution: float = 0.02, z_margin: float = 0.03, ignore: Iterable = ()) -> "SupportTop":
        """Raster of the support's top surface: cells whose downward ray first hits the support near its top."""
        lo, hi = support.aabb
        top = float(hi[2])
        xs = th.arange(float(lo[0]) + resolution / 2, float(hi[0]), resolution)
        ys = th.arange(float(lo[1]) + resolution / 2, float(hi[1]), resolution)
        gx, gy = th.meshgrid(xs, ys, indexing="ij")
        starts = th.stack([gx.flatten(), gy.flatten(), th.full_like(gx.flatten(), top + 0.30)], dim=1)
        ends = starts.clone()
        ends[:, 2] = float(lo[2]) - 0.05
        ignore_paths = [self.robot.prim_path] + [o.prim_path for o in ignore]
        prim_to_obj = {}
        for o in self.scene.objects:
            for l in o.links.values():
                prim_to_obj[l.prim_path] = o
        hits = raytest_batch(starts.tolist(), ends.tolist(), only_closest=True)
        mask = th.zeros(len(xs), len(ys), dtype=th.bool)
        occupied = th.zeros_like(mask)
        zs = th.full((len(xs), len(ys)), float("nan"))
        occupants = set()
        own_hits = []  # (i, j, z) of rays that first hit the support itself
        other_hits = []  # (i, j, z, obj) of rays that first hit another object
        for k, h in enumerate(hits):
            i, j = divmod(k, len(ys))
            if not h["hit"]:
                continue
            body = h["rigidBody"]
            if any(body.startswith(p) for p in ignore_paths):
                continue
            o = prim_to_obj.get(body)
            hz = float(h["position"][2])
            if o is support:
                own_hits.append((i, j, hz))
            elif o is not None:
                other_hits.append((i, j, hz, o))
        # the usable surface is the support's largest horizontal face, not necessarily its AABB top (a bed's headboard
        # or a chair back is higher than the mattress / seat): take the most populated 2 cm height bin
        if own_hits:
            hz_all = th.tensor([z for _, _, z in own_hits])
            bins = th.round((hz_all - float(lo[2])) / 0.02).long()
            best = int(th.mode(bins).values)
            z_top = float(hz_all[bins == best].median())
        else:
            z_top = top
        for i, j, hz in own_hits:
            if abs(hz - z_top) <= z_margin:
                mask[i, j] = True
                zs[i, j] = hz
        for i, j, hz, o in other_hits:
            if hz > z_top - z_margin:
                occupied[i, j] = True
                occupants.add(o.name)
        yaw = G.yaw_of(support.get_position_orientation()[1])
        return SupportTop(support.name, xs, ys, mask, occupied, z_top, yaw, sorted(occupants), resolution)

    def raycast(self, starts: th.Tensor, ends: th.Tensor):
        return raytest_batch(G.t(starts).tolist(), G.t(ends).tolist(), only_closest=True)

    def surface_z_under(self, xy, from_z: float, ignore: Sequence[Any] = (), depth: float = 1.0) -> Optional[Dict[str, Any]]:
        """The actual surface under world ``xy`` (owner item 3, 2026-09-16 10:15): a downward raycast from ``from_z`` (the
        release height) to ``from_z - depth`` with the rigid bodies of ``ignore`` (the placed object itself, the robot)
        excluded — the saucer well, the plate centre, the tray floor, the table under a hole — instead of the partner's
        hull top. Returns ``{"z", "body", "normal"}`` or None when nothing is hit. Task modules place / release 1 cm
        above ``z`` (IORS ontop / under, the disembedding destination, attachment place, block slots)."""
        bodies: List[str] = []
        for o in list(ignore) + [self.robot]:
            for link in getattr(o, "links", {}).values():
                bodies.append(link.prim_path)
        x, y = float(G.t(xy)[0]), float(G.t(xy)[1])
        hits = raytest_batch([[x, y, float(from_z)]], [[x, y, float(from_z) - float(depth)]], only_closest=True, ignore_bodies=bodies or None)
        h = hits[0] if hits else {}
        if not h.get("hit"):
            return None
        return {"z": float(h["position"][2]), "body": str(h.get("rigidBody")), "normal": [float(v) for v in h.get("normal", (0.0, 0.0, 1.0))]}

    # ------------------------------------------------------------------ validation (BEHAVIOR semantics)

    #: residual-motion thresholds after settling. BEHAVIOR's ``_validate_object_state_stability(strict=False)`` accepts
    #: 1 m/s / pi rad/s and relies on the pose comparison over the following 10 steps (which we keep, see
    #: ``drift_after``); ours tolerate the slow rocking of round objects (ladles, lemons) but reject anything sliding.
    STABLE_LIN_VEL = 0.05
    STABLE_ANG_VEL = 0.50

    def check_stability(self, entities: Iterable, lin_tol: float = STABLE_LIN_VEL, ang_tol: float = STABLE_ANG_VEL) -> Optional[str]:
        for e in entities:
            if not isinstance(e, DatasetObject):
                continue
            v = float(th.linalg.norm(e.get_linear_velocity()))
            w = float(th.linalg.norm(e.get_angular_velocity()))
            if v > lin_tol or w > ang_tol:
                return f"{e.name} moving (v={v:.3f} m/s, w={w:.3f} rad/s)"
        return None

    def check_scene_unchanged(self, task_object_names: Iterable[str], pos_tol: float = 0.05, ori_tol: float = 0.05) -> Optional[str]:
        """Non-task scene objects must still be at their stable-scene poses (BEHAVIOR validate_task step 2)."""
        task_names = set(task_object_names) | set(self.spawned) | {self.robot.name}
        ref = self.stable_scene["state"]["registry"]["object_registry"]
        for o in self.scene.objects:
            if o.name in task_names or o.name not in ref:
                continue
            r = ref[o.name]["root_link"]
            p, q = o.get_position_orientation()
            if float(th.linalg.norm(p - th.tensor(r["pos"]))) > pos_tol:
                return f"scene object {o.name} moved by {float(th.linalg.norm(p - th.tensor(r['pos']))):.3f} m"
            if G.quat_angle(q, th.tensor(r["ori"])) > ori_tol:
                return f"scene object {o.name} rotated"
        return None

    def reset_scene_objects(self, task_object_names: Iterable[str] = ()) -> int:
        """Put every non-task scene object back into exactly the state of the stable scene json (pose, velocities,
        joints). Called before every instance attempt and before a template is saved, so that instances differ from
        the scene json only in the task-relevant objects and the robot, whatever physics did meanwhile (objects
        nudged by a settling task object, slow creep of stacked furniture over thousands of steps, ...).
        Returns the number of objects whose pose actually changed by more than 1 mm / 0.005 rad."""
        from omnigibson.utils.python_utils import recursively_convert_to_torch

        task_names = set(task_object_names) | set(self.spawned) | {self.robot.name}
        ref = self.stable_scene["state"]["registry"]["object_registry"]
        n_changed = 0
        for o in self.scene.objects:
            if o.name in task_names or o.name not in ref:
                continue
            r = ref[o.name]["root_link"]
            p, q = o.get_position_orientation()
            moved = float(th.linalg.norm(p - th.tensor(r["pos"]))) > 0.001 or G.quat_angle(q, th.tensor(r["ori"])) > 0.005
            if moved:
                n_changed += 1
            if moved or o.n_joints > 0:  # articulated scene objects (doors, drawers) can drift in their joints too
                o.load_state(recursively_convert_to_torch(copy.deepcopy(ref[o.name])), serialized=False)
                o.keep_still()
        return n_changed

    def settle(self, task_entities: Iterable, steps: int = 60, robot_still: bool = True, still: Iterable = ()) -> Optional[str]:
        """Let the task objects settle under physics (robot base held) and report residual motion. ``still`` = entities
        held at rest every step (2026-09-25: the scene support - a free-standing table is a rigid body on four convex-hull
        legs; the trays' 3 mm drop sets it chattering on the floor mesh and the top then shakes every object on it for
        seconds: headless test, identical placements - fixed table <= 0.2 mm per frame, dynamic table 1-150 mm. The demo
        loads the table at rest from the TRO and never kicks it, so holding it here matches the demo)."""
        ents = list(task_entities)
        held = [e for e in still if e is not None]
        self.wake(ents)
        q0 = self.robot.get_joint_positions().clone()
        for _ in range(steps):
            og.sim.step_physics()
            if robot_still:
                self.robot.keep_still()
            for e in held:
                e.keep_still()
        for _ in range(10):
            og.sim.step_physics()
            if robot_still:
                self.robot.keep_still()
            for e in held:
                e.keep_still()
        # one full step refreshes the per-step caches (AABBs, contacts) that step_physics leaves stale
        self.full_step()
        if robot_still:
            self.robot.keep_still()
        # diagnostic (2026-09-25): a robot whose drives target another configuration creeps during a physics-only settle
        dq = (self.robot.get_joint_positions() - q0).abs()
        self.last_robot_drift_rad = float(dq.max())
        if self.last_robot_drift_rad > 0.01:
            names = list(getattr(self.robot, "joints", {}).keys())
            worst = names[int(dq.argmax())] if len(names) == len(dq) else int(dq.argmax())
            log.warning("settle: the robot's joints drifted up to %.3f rad (%s) over %d + 10 physics steps - stale drive targets?", self.last_robot_drift_rad, worst, steps)
        return self.check_stability(ents)

    def drift_after(self, entities: Iterable, steps: int = 10, pos_tol: float = 0.01, ang_tol: float = 0.05, still: Iterable = ()) -> Optional[str]:
        ents = list(entities)
        held = [e for e in still if e is not None]
        before = {e.name: e.get_position_orientation() for e in ents}
        joints = {e.name: e.get_joint_positions().clone() for e in ents if e.n_joints > 0}
        for _ in range(steps):
            og.sim.step_physics()
            self.robot.keep_still()
            for e in held:
                e.keep_still()
        for e in ents:
            p, q = e.get_position_orientation()
            p0, q0 = before[e.name]
            if float(th.linalg.norm(p - p0)) > pos_tol or G.quat_angle(q, q0) > ang_tol:
                return f"{e.name} drifted {float(th.linalg.norm(p - p0)):.3f} m / {G.quat_angle(q, q0):.3f} rad in {steps} steps"
            if e.name in joints and float((e.get_joint_positions() - joints[e.name]).abs().max()) > 0.02:
                return f"{e.name} joints drifted"
        return None

    #: trembling gate (2026-09-24, disembedding milk bottles 0916-141051 / 0916-181259): after the settle every free task
    #: object must be *visually* still for ``TREMBLE_FRAMES`` action frames. PhysX contact chatter - a body lying across the
    #: seams of a convex-decomposition floor (the gsxbym tray: four slabs, 1.6 mm step), a cylinder on its side with no
    #: restoring torque - keeps a body awake with 10-50 mm/s of alternating velocity that ``check_stability`` (50 mm/s,
    #: 0.5 rad/s) and ``drift_after`` (10 mm / 0.05 rad over 10 physics steps) let through; the demos then showed 0.5-2 mm and
    #: 1-8 deg jumps per frame from step 0, 17 mm of drift over the transition, and ``pose_settled`` never held. Calibration
    #: (head_rgb demos, per 1/30 s frame): still boxes on a tray <= 0.04 mm / 0.22 deg (float32 pose noise); trembling bottles
    #: 0.12-2.2 mm / 0.27-7.9 deg at their worst frame, 0.02-0.6 mm / 0.02-0.4 deg on average. Velocity is deliberately not
    #: gated: PhysX reports 5-30 mm/s of sign-alternating velocity for bodies whose pose does not move at all.
    TREMBLE_FRAMES = 45  # 1.5 s at 30 Hz
    #: per-frame limits (0.5 mm / 0.5 deg): about a head-camera pixel (~0.5 mm at the tray); the settling jiggle of a bottle
    #: seconds after its release reads 0.1-0.5 mm single-frame hops with zero net motion (roll / crowding tests 0924), the
    #: net limits decide
    TREMBLE_POS_TOL = 0.0005  # m per frame
    TREMBLE_ANG_TOL = 0.0087  # rad per frame (0.5 deg)
    TREMBLE_DRIFT_POS_TOL = 0.001  # m over the window (a slowly rolling cylinder: small steps, steady drift)
    TREMBLE_DRIFT_ANG_TOL = 0.01  # rad over the window

    def tremble_after(self, entities: Iterable, frames: int = TREMBLE_FRAMES, pos_tol: float = TREMBLE_POS_TOL, ang_tol: float = TREMBLE_ANG_TOL,
                      drift_pos_tol: float = TREMBLE_DRIFT_POS_TOL, drift_ang_tol: float = TREMBLE_DRIFT_ANG_TOL, robot_still: bool = True,
                      still: Iterable = ()) -> Optional[str]:
        """Per-frame pose change of ``entities`` over ``frames`` action frames (``n_physics_timesteps_per_render`` physics
        steps each, the robot held): the worst frame of every entity must stay within ``pos_tol`` / ``ang_tol`` and its net
        motion over the window within ``drift_pos_tol`` / ``drift_ang_tol``. Objects held by the robot (assisted grasp) are
        skipped. Returns a description of the worst offender or None; ``self.last_tremble`` keeps the per-entity maxima
        (mm, deg) for the instance summary."""
        held = set()
        for obj in (getattr(self.robot, "_ag_obj_in_hand", {}) or {}).values():
            if obj is not None:
                held.add(obj.name)
        ents = [e for e in entities if isinstance(e, DatasetObject) and e.name not in held]
        held_still = [e for e in still if e is not None]
        try:
            n_sub = max(1, int(og.sim.n_physics_timesteps_per_render))
        except Exception:
            n_sub = 4
        prev = {e.name: e.get_position_orientation() for e in ents}
        first = dict(prev)
        worst = {e.name: {"step_mm": 0.0, "step_deg": 0.0, "first_move_frame": None, "worst_frame": None} for e in ents}
        for fi in range(int(frames)):
            for _ in range(n_sub):
                og.sim.step_physics()
                if robot_still:
                    self.robot.keep_still()
                for e in held_still:
                    e.keep_still()
            for e in ents:
                p, q = e.get_position_orientation()
                p0, q0 = prev[e.name]
                w = worst[e.name]
                step_mm = float(th.linalg.norm(p - p0)) * 1000.0
                if step_mm > w["step_mm"]:
                    w["step_mm"], w["worst_frame"] = step_mm, fi
                if w["first_move_frame"] is None and step_mm > 0.2:
                    w["first_move_frame"] = fi
                w["step_deg"] = max(w["step_deg"], math.degrees(G.quat_angle(q, q0)))
                prev[e.name] = (p, q)
        # as in ``settle``: one full step refreshes the per-step caches (AABBs, contacts) the physics-only steps leave stale
        self.full_step()
        if robot_still:
            self.robot.keep_still()
        self.last_tremble = {}
        offender = None
        for e in ents:
            p, q = prev[e.name]
            p0, q0 = first[e.name]
            w = worst[e.name]
            d = (p - p0) * 1000.0
            w.update({"drift_mm": float(th.linalg.norm(p - p0)) * 1000.0, "drift_deg": math.degrees(G.quat_angle(q, q0)), "asleep": bool(getattr(e, "is_asleep", False)),
                      "frames": int(frames), "drift_xyz_mm": [round(float(v), 2) for v in d.tolist()], "z0": round(float(p0[2]), 4)})
            self.last_tremble[e.name] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in w.items()}
            bad = w["step_mm"] > pos_tol * 1000.0 or w["step_deg"] > math.degrees(ang_tol) or w["drift_mm"] > drift_pos_tol * 1000.0 or w["drift_deg"] > math.degrees(drift_ang_tol)
            if bad and (offender is None or w["step_mm"] + w["drift_mm"] > offender[1]):
                offender = (e.name, w["step_mm"] + w["drift_mm"])
        if offender is None:
            return None
        w = worst[offender[0]]
        # who moved when: the initiator of a chain is the entity with the earliest first_move_frame
        timeline = "; ".join(f"{n}: first>0.2mm@f{v['first_move_frame']} worst@f{v['worst_frame']} {v['step_mm']:.2f}mm net {v['drift_xyz_mm']}"
                             for n, v in sorted(self.last_tremble.items(), key=lambda kv: (kv[1]["first_move_frame"] is None, kv[1]["first_move_frame"] or 0))
                             if v["step_mm"] > 0.1 or v["drift_mm"] > 0.3)
        return (f"{offender[0]} trembles: worst frame {w['step_mm']:.2f} mm / {w['step_deg']:.2f} deg (limits {pos_tol * 1000:.2f} mm / {math.degrees(ang_tol):.2f} deg), "
                f"net {w['drift_mm']:.2f} mm / {w['drift_deg']:.2f} deg over {frames} frames (limits {drift_pos_tol * 1000:.1f} mm / {math.degrees(drift_ang_tol):.2f} deg)"
                f" [timeline: {timeline}]")

    # ------------------------------------------------------------------ templates

    def save_template(self, out_dir: str, inst_to_name: Dict[str, str], robot_poses: Dict) -> Tuple[str, str]:
        """Write ``template-partial_rooms.json`` (what is loaded) and ``template.json`` (merged with the stable scene)."""
        os.makedirs(out_dir, exist_ok=True)
        self.scene.write_task_metadata(key="inst_to_name", data=dict(inst_to_name))
        self.scene.write_task_metadata(key="robot_poses", data=robot_poses)
        partial = os.path.join(out_dir, "template-partial_rooms.json")
        for o in self.render_hidden:  # the template must not record the render workaround
            o.visible = True
        try:
            self.scene.save(json_path=partial)
        finally:
            for o in self.render_hidden:
                o.visible = False
        with open(partial) as f:
            scene_b = json.load(f)
        merged = merge_scene_files(copy.deepcopy(self.stable_scene), scene_b, keep_robot_from="b")
        full = os.path.join(out_dir, "template.json")
        with open(full, "w") as f:
            json.dump(merged, f, cls=TorchEncoder, indent=2)
        return partial, full

    # ------------------------------------------------------------------ robot placement

    def teleport_robot(self, base_pose2d: Tuple[float, float, float], q: Optional[th.Tensor] = None, z: Optional[float] = None):
        """Set the robot base (x, y, yaw) and optionally the full joint vector; keep it still."""
        x, y, yaw = base_pose2d
        zb = (self.floor_z + robot_utils.NOMINAL_BASE_Z) if z is None else float(z)
        if q is not None:
            self.robot.set_joint_positions(G.t(q))
            self.robot.set_joint_velocities(th.zeros_like(G.t(q)))
            # 2026-09-25: the joint *drive targets* too. ``set_joint_positions`` (drive=False) writes the joint state only;
            # the position drives kept the targets of the last ``apply_action`` (a program-check grasp over the tray, the
            # previous attempt's stance), and during the physics-only settle (``keep_still`` zeroes the velocities every
            # step, the drive torque still integrates ½·a·dt² per step) the arm crept toward them - hundreds of steps
            # later the hand had swept into the tray and shoved the bottles (the disembedding milk alias never settled
            # once its distractors filled the tray; the objects nearest the hand moved first)
            self.robot.set_joint_positions(G.t(q), drive=True)
            self.robot.set_joint_velocities(th.zeros_like(G.t(q)), drive=True)
        self.robot.set_position_orientation(th.tensor([x, y, zb]), G.quat_from_yaw(yaw))
        self.robot.keep_still()
        og.sim.step_physics()
        self.robot.keep_still()

    def robot_floor_z(self) -> float:
        """The measured floor height the robot stands on (see ``measure_floor_z``)."""
        return float(self.floor_z)

    def camera_pose(self) -> G.Pose:
        self._require_camera()
        p, q = self.cam.get_position_orientation()
        return p.float(), q.float()

    def floor_of_room(self, room: str):
        for o in self.scene.objects:
            if o.category == "floors" and room in (o.in_rooms or []):
                return o
        raise KeyError(f"no floors object in room {room!r}")

    def floor_object_under(self, xy):
        """The ``floors`` scene object whose collision surface a downward raycast at world ``xy`` hits (the robot's bodies
        ignored), or None. A room may hold several overlapping floor objects (house_single_floor garden_0: the lawn
        ``floors_khjmrr_0`` and ``floors_pyoemr_0`` both span the sauna door's approach); ``floor_of_room`` returns the first
        one, and the BDDL ``(ontop agent floor)`` is judged against the object the robot actually stands on
        (door_crtrpa_0 0919-153516: init_false at settle with the room's first floor bound)."""
        z0 = self.floor_z_default
        x, y = float(G.t(xy)[0]), float(G.t(xy)[1])
        try:
            ignore = [link.prim_path for link in self.robot.links.values()]
            hits = raytest_batch([[x, y, z0 + 0.5]], [[x, y, z0 - 0.5]], only_closest=True, ignore_bodies=ignore)
            h = hits[0] if hits else {}
            body = str(h.get("rigidBody") or "") if h.get("hit") else ""
        except Exception as e:  # noqa: BLE001
            log.debug("floor raycast failed at (%.2f, %.2f): %r", x, y, e)
            return None
        if not body:
            return None
        for o in self.scene.objects:
            if o.category == "floors" and (body == o.prim_path or body.startswith(o.prim_path + "/")):
                return o
        return None


@dataclass
class SupportTop:
    """Raster of a support's usable top surface (world frame, axis-aligned grid)."""

    name: str
    xs: th.Tensor
    ys: th.Tensor
    mask: th.Tensor  # (nx, ny) True where the support top is exposed
    occupied: th.Tensor  # (nx, ny) True where another object sits on the support
    z: float
    yaw: float  # support root yaw (its local x axis direction)
    occupants: List[str] = field(default_factory=list)
    resolution: float = 0.02

    @property
    def free(self) -> th.Tensor:
        return self.mask & ~self.occupied

    def cells(self, free_only: bool = True) -> th.Tensor:
        m = self.free if free_only else self.mask
        idx = th.nonzero(m)
        return th.stack([self.xs[idx[:, 0]], self.ys[idx[:, 1]]], dim=1)

    def local_axes(self) -> Tuple[th.Tensor, th.Tensor]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return th.tensor([c, s]), th.tensor([-s, c])

    def local_bounds(self):
        """Bounds of the exposed top in the support's local (yaw) frame: (umin, umax, vmin, vmax), and the centre."""
        pts = self.cells(free_only=False)
        ax, ay = self.local_axes()
        u = pts @ ax
        v = pts @ ay
        return float(u.min()), float(u.max()), float(v.min()), float(v.max())

    def contains(self, xy, margin: float = 0.0) -> bool:
        """True if world ``xy`` lies on an exposed (free) cell; ``margin`` requires neighbours within it to be free too."""
        pts = [G.t(xy)]
        if margin > 0:
            for dx, dy in ((margin, 0), (-margin, 0), (0, margin), (0, -margin), (margin, margin), (-margin, -margin), (margin, -margin), (-margin, margin)):
                pts.append(G.t(xy) + th.tensor([dx, dy]))
        for p in pts:
            i = int(th.round((p[0] - self.xs[0]) / self.resolution))
            j = int(th.round((p[1] - self.ys[0]) / self.resolution))
            if i < 0 or j < 0 or i >= len(self.xs) or j >= len(self.ys) or not bool(self.free[i, j]):
                return False
        return True

    def polygon_supported(self, poly_xy: th.Tensor, margin: float = 0.02) -> bool:
        """Every vertex and the centroid of a footprint polygon lie on free cells (with margin)."""
        for p in poly_xy:
            if not self.contains(p, margin):
                return False
        return self.contains(poly_xy.mean(dim=0), 0.0)

    def edges(self):
        """The four sides of the exposed top in the local frame: list of (name, midpoint_world, outward_normal_world, length)."""
        umin, umax, vmin, vmax = self.local_bounds()
        ax, ay = self.local_axes()
        uc, vc = (umin + umax) / 2, (vmin + vmax) / 2
        out = []
        for name, mid_uv, normal, length in (
            ("+u", (umax, vc), ax, vmax - vmin),
            ("-u", (umin, vc), -ax, vmax - vmin),
            ("+v", (uc, vmax), ay, umax - umin),
            ("-v", (uc, vmin), -ay, umax - umin),
        ):
            mid = mid_uv[0] * ax + mid_uv[1] * ay
            out.append((name, mid, normal, length))
        return out

    def to_json(self):
        umin, umax, vmin, vmax = self.local_bounds()
        return {"name": self.name, "z": self.z, "yaw": self.yaw, "local_bounds": [umin, umax, vmin, vmax],
                "n_free_cells": int(self.free.sum()), "occupants": self.occupants, "resolution": self.resolution}
