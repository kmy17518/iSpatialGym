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
#: the parked robot's clearance (owner 2026-10-12 ~02:40): its tucked footprint grown by this on every side must not overlap
#: any scene object's AABB (``SimWorld.park_clear_mask``)
PARK_CLEARANCE_M = 0.05
#: objects lying within this height above the floor (the two carpets of the scene set, fixed) do not count: the robot stands on them
PARK_FLOOR_BAND_M = 0.02
HEAD_APERTURE_MM = 40.0
IMAGE_SIZE = 1080


class Reject(Exception):
    """An instance attempt failed a check (counted per reason, attempt retried)."""

    def __init__(self, reason: str, message: str = "", **details):
        super().__init__(f"[{reason}] {message}" + (f" {details}" if details else ""))
        self.reason = reason
        self.details = details


def stable_scene_path(scene_model: str) -> str:
    from ispatialgym.sampling.scene_variants import stable_scene_override

    return stable_scene_override(scene_model) or os.path.join(get_dataset_path("2026-challenge-task-instances"), "scenes", scene_model, "json", f"{scene_model}_stable.json")


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


def _collision_piece_aabbs(obj) -> List[Tuple[th.Tensor, th.Tensor]]:
    """World AABBs of ``obj``'s collision meshes, from the render-side link transforms as ``obj.aabb``; the object's AABB
    when the meshes cannot be read; [] without collision geometry."""
    out = []
    try:
        for link in obj.links.values():
            meshes = link.collision_meshes
            if not meshes:
                continue
            tf = link.scaled_transform
            for mesh in meshes.values():
                pts = mesh.points_in_parent_frame
                if pts is not None and len(pts):
                    w = T.transform_points(pts, tf)
                    out.append((w.min(dim=0).values.float(), w.max(dim=0).values.float()))
    except Exception:  # noqa: BLE001 - e.g. a mesh type without readable points
        out = []
    if out:
        return out
    try:
        lo, hi = obj.aabb
    except Exception:  # noqa: BLE001
        return []
    return [(th.as_tensor(lo, dtype=th.float32), th.as_tensor(hi, dtype=th.float32))]


def _fmt_blockers(blockers: Sequence[Tuple[str, float]]) -> str:
    if not blockers:
        return "no scene object (its footprint leaves the loaded rooms)"
    return ", ".join(f"{n} by {d * 100:.1f} cm" for n, d in blockers)


# ----------------------------------------------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------------------------------------------


class SimWorld:
    """One OmniGibson environment (scene + R1Pro) plus the helpers every task sampler needs."""

    def __init__(self, scene_model: str, rooms: Optional[Sequence[str]], robot_name: str = "robot_r1", image_size: int = IMAGE_SIZE,
                 load_adjacent: Optional[bool] = None, scene_variant: Optional[dict] = None, cameras: bool = True, usd_camera_model: bool = False,
                 scene_crop: Optional[dict] = None):
        """``cameras=False``: the robot carries no vision sensor (``obs_modalities=["proprio"]`` - OmniGibson creates a
        ``VisionSensor`` only for requested vision modalities), so no 1080^2 rgb + seg_instance buffers, annotators or
        per-frame synthetic-data graph exist; ``cam_key`` / ``cam`` / ``camera_model`` are ``None`` and the render /
        segmentation helpers raise. For the stages that never read pixels (reach bands: kinematics, raycasts, cuRobo).

        ``usd_camera_model=True`` with ``cameras=False`` (train-fast SAMPLER_CAMERAS, 2026-10-31: the sampler's camera-free world
        for the families that use frustum math / raycasts but no pixels): ``camera_model`` is built from the head camera prim's
        USD ``focalLength`` (the sensor the camera world reads it from wraps that prim; ``HEAD_CAMERA_FOCAL_MM_FALLBACK`` when the
        prim is not found) with the same aperture / image size, and :meth:`camera_pose` falls back to the ``zed_link`` pose
        composed with the USD camera offset (the kit's FK of the current configuration). ``cam`` stays ``None``: every pixel
        helper still raises ``_require_camera``'s message.

        ``scene_crop`` (train-fast SCENE_CROP, 2026-11-01; ``ispatialgym.sampling.scene_crop``; None = today's loading): ``{"path": the
        cropped copy of the variant / stable json, "loaded_rooms": the load_room_instances it is loaded with, "meta":
        template_meta.scene_crop, "zone_xy": ((x0, y0), (x1, y1)) or None, "log": the SCENE_CROP line}``. Only what is LOADED
        changes: ``stable_path`` / ``stable_scene`` (the template.json merge source, the reset and scene-unchanged references, the
        traversability patch) stay the uncropped variant, ``eval_rooms`` stays rooms + adjacent, ``loaded_rooms`` becomes the crop's."""
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
        # train-fast SCENE_CROP (2026-11-01): a cropped world loads the cropped scene file with its own room list (the task rooms);
        # ``loaded_rooms`` (templates' ``sampled_with_rooms``, the park-cell room mask) follows what is actually loaded
        self.scene_crop = dict(scene_crop) if scene_crop and scene_crop.get("path") else None
        if self.scene_crop is not None:
            self.loaded_rooms = list(self.scene_crop["loaded_rooms"])
        #: train-fast SCENE_CROP ``region``: the xy box inside which every scene object is loaded (park cells stay inside it)
        self.crop_zone_xy = (self.scene_crop or {}).get("zone_xy")
        self.robot_name = robot_name
        #: scene variant (``scene_variants.py``): stable scene minus e.g. the chairs around the work table. Everything
        #: downstream (template merge, scene-unchanged checks, resets) compares against the variant.
        self.scene_variant = scene_variant or {}
        self.stable_path = self.scene_variant.get("path") or stable_scene_path(scene_model)
        with open(self.stable_path) as f:
            self.stable_scene = json.load(f)
        campaign_support = ((self.stable_scene.get("metadata") or {}).get("ispatialgym") or {}).get("campaign_added_support")
        if campaign_support:
            self.scene_variant = {**self.scene_variant, "path": self.stable_path,
                                  "campaign_added_support": copy.deepcopy(campaign_support)}
        cfg = {
            "env": {"action_frequency": 30, "physics_frequency": 120},
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": scene_model,
                "scene_file": self.scene_crop["path"] if self.scene_crop is not None else self.stable_path,  # SCENE_CROP: the cropped copy
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
        if self.scene_crop is not None and self.scene_crop.get("log"):
            log.info("%s", self.scene_crop["log"])
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
        #: train-fast SAMPLER_CAMERAS: ``camera_pose`` from the zed link's pose when there is no sensor (see ``usd_camera_model``)
        self._fk_camera = False
        if cameras:
            self.cam_key = [k for k in self.robot.sensors if k.endswith(M.HEAD_CAMERA_SENSOR_SUFFIX)][0]
            self.cam = self.robot.sensors[self.cam_key]
            self.cam.horizontal_aperture = HEAD_APERTURE_MM
            self.camera_model = G.CameraModel.from_sensor(float(self.cam.focal_length), HEAD_APERTURE_MM, image_size, image_size)
        elif usd_camera_model:
            focal, how = self.usd_head_camera_focal_length()
            self.camera_model = G.CameraModel.from_sensor(focal, HEAD_APERTURE_MM, image_size, image_size)
            self._fk_camera = True
            log.info("camera-free world (SAMPLER_CAMERAS): camera model from %s (focal %.2f mm, aperture %.0f mm, %d px: hfov %.1f deg); "
                     "camera_pose = zed_link pose o USD camera offset", how, focal, HEAD_APERTURE_MM, image_size, math.degrees(self.camera_model.hfov))
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
        if self.scene_variant.get("campaign_added_support"):
            from ispatialgym.sampling.scene_variants import block_campaign_support

            n_blocked = block_campaign_support(self.scene, self.scene_variant["campaign_added_support"])
            log.info("traversability map patched for campaign support: %d cells blocked", n_blocked)
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
        log.info("scene %s loaded (%d objects, %d collision-relevant, rooms=%s, adjacent=%s%s) in %.1fs", scene_model, self.n_scene_objects, n_obst,
                 self.rooms, self.adjacent, f", SCENE_CROP {self.scene_crop['meta']['mode']}: loaded rooms {self.loaded_rooms}" if self.scene_crop else "",
                 self.load_time)

    def scene_crop_meta(self) -> Dict[str, Any]:
        """``template_meta.scene_crop`` of this world (train-fast SCENE_CROP): ``{"mode": "full"}`` for today's loading."""
        sc = getattr(self, "scene_crop", None)
        return copy.deepcopy(sc["meta"]) if sc else {"mode": "full"}

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
        """World xy of the robot's parking cell: of the free (eroded) traversability cells inside the task rooms (elsewhere
        there is no floor under partial-room loading and the robot falls, which silently shifts the cuRobo world: meshes
        are stored relative to the robot root at sync time), the one nearest their centre that keeps the tucked robot
        clear of every scene object (:meth:`choose_park_cell`)."""
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
        k = self.choose_park_cell(trav.map_to_world(free), th.linalg.norm(free.float() - centre, dim=1), "world load")
        return trav.map_to_world(free[k])

    def park_box(self) -> Tuple[th.Tensor, th.Tensor]:
        """``(lo, hi)``: the box the robot parked by ``park_robot`` (tucked, yaw 0) occupies relative to its base position on
        the floor (``r1pro_model.TUCKED_BOX_IN_BASE``)."""
        lo, hi = M.TUCKED_BOX_IN_BASE
        return th.tensor(lo, dtype=th.float32), th.tensor(hi, dtype=th.float32)

    def park_obstacles(self) -> Tuple[List[str], th.Tensor, th.Tensor]:
        """``(names, lo, hi)``: one AABB ``(M, 3)`` per collision mesh (a convex piece; its object's name in ``names``) of
        every scene object a parked robot could touch - all but the robot, the floors and visual-only objects -, read once,
        at the world-load park: the scene file's poses, which ``reset_scene_objects`` restores before every later park. Per
        piece, not per object: the box of a multi-part object (a paving, a wall run, an L-shaped bar, stairs) covers floor
        it leaves free. Objects spawned later are kept away by ``park_pose``."""
        if getattr(self, "_park_obstacles", None) is None:
            names, los, his = [], [], []
            for o in self.scene.objects:
                if o is self.robot or o.category == "floors" or getattr(o, "visual_only", False):
                    continue
                for lo, hi in _collision_piece_aabbs(o):
                    names.append(o.name)
                    los.append(lo)
                    his.append(hi)
            self._park_obstacles = (names, th.stack(los) if los else th.zeros(0, 3), th.stack(his) if his else th.zeros(0, 3))
        return self._park_obstacles

    def _park_overlaps(self, pts: th.Tensor, margin: float) -> Tuple[th.Tensor, th.Tensor]:
        """``(idx, over)``: the pieces of :meth:`park_obstacles` within the robot's height and reach of ``pts`` (indices), and
        ``(N, len(idx))`` how far (m) the parked robot's box (:meth:`park_box`) at each xy of ``pts``, grown by ``margin``,
        reaches into each of their AABBs (the smaller of the x / y overlaps; <= 0: apart)."""
        _, lo, hi = self.park_obstacles()
        box_lo, box_hi = self.park_box()
        rlo, rhi = box_lo[:2] - margin, box_hi[:2] + margin
        z0 = float(self.floor_z)
        pts = th.as_tensor(pts, dtype=th.float32).reshape(-1, 2)
        reach_lo, reach_hi = pts.min(dim=0).values + rlo, pts.max(dim=0).values + rhi
        near = (hi[:, 2] > z0 + PARK_FLOOR_BAND_M) & (lo[:, 2] < z0 + float(box_hi[2]) + margin) & (hi[:, 0] > reach_lo[0]) & (lo[:, 0] < reach_hi[0]) & \
               (hi[:, 1] > reach_lo[1]) & (lo[:, 1] < reach_hi[1])
        idx = th.nonzero(near).flatten()
        lo, hi = lo[idx], hi[idx]
        ox = th.minimum(pts[:, None, 0] + rhi[0], hi[None, :, 0]) - th.maximum(pts[:, None, 0] + rlo[0], lo[None, :, 0])
        oy = th.minimum(pts[:, None, 1] + rhi[1], hi[None, :, 1]) - th.maximum(pts[:, None, 1] + rlo[1], lo[None, :, 1])
        return idx, th.minimum(ox, oy)

    def park_clear_mask(self, pts: th.Tensor, margin: float = PARK_CLEARANCE_M) -> th.Tensor:
        """``(N,)`` bool per world xy of ``pts``: the tucked robot parked there (yaw 0) keeps ``margin`` from the AABB of
        every collision piece of the scene objects (:meth:`park_obstacles`; from ``PARK_FLOOR_BAND_M`` above the floor to its
        height + ``margin``) and its footprint lies over the loaded rooms (partial-room loading leaves no floor elsewhere).
        An AABB test, not the kit's whole-body check: the world-load park runs before the cuRobo world exists."""
        pts = th.as_tensor(pts, dtype=th.float32).reshape(-1, 2)
        # in chunks: without a free cell in the task rooms the candidates are every free cell of the scene (~10^5)
        ok = th.cat([~(self._park_overlaps(c, margin)[1] > 0).any(dim=1) for c in th.split(pts, 1024)]) if len(pts) else th.zeros(0, dtype=th.bool)
        rooms = self.loaded_rooms
        seg = getattr(self.scene, "seg_map", None)
        if rooms and seg is not None:
            ids = th.tensor([seg.room_ins_name_to_ins_id[r] for r in rooms if r in seg.room_ins_name_to_ins_id], dtype=th.long)
            box_lo, box_hi = self.park_box()
            shape = seg.room_ins_map.shape
            for cx, cy in ((box_lo[0], box_lo[1]), (box_lo[0], box_hi[1]), (box_hi[0], box_lo[1]), (box_hi[0], box_hi[1])):
                corner = pts + th.stack([cx, cy])
                # SegmentationMap.world_to_map for a batch (its flip over every dim reverses the point order of an (N, 2) input)
                ij = th.flip(corner / seg.map_resolution + seg.map_size / 2.0, dims=(1,)).int()
                inside = (ij[:, 0] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[1])
                room = th.zeros(len(pts), dtype=th.long)
                room[inside] = seg.room_ins_map[ij[inside, 0], ij[inside, 1]].long()
                ok &= th.isin(room, ids)
        zone = getattr(self, "crop_zone_xy", None)
        if zone is not None and len(pts):
            # train-fast SCENE_CROP region (2026-11-01): only inside the crop zone is every scene object loaded, so only there does a
            # clear cell stay clear in the full scene (template.json); prefer cells whose tucked box + margin lies inside it
            box_lo, box_hi = self.park_box()
            zlo, zhi = th.tensor(zone[0][:2], dtype=th.float32), th.tensor(zone[1][:2], dtype=th.float32)
            inside = ((pts + box_lo[:2] - margin) >= zlo).all(dim=1) & ((pts + box_hi[:2] + margin) <= zhi).all(dim=1)
            if bool((ok & inside).any()):
                ok = ok & inside
            elif bool(ok.any()):
                log.warning("SCENE_CROP: no clear park cell inside the crop zone %s; a cell outside it may overlap an object that is not loaded",
                            [[round(float(v), 2) for v in zone[0][:2]], [round(float(v), 2) for v in zone[1][:2]]])
        return ok

    def park_blockers(self, xy, margin: float = PARK_CLEARANCE_M) -> List[Tuple[str, float]]:
        """``[(name, overlap m)]``, deepest first: the scene objects whose collision pieces' AABBs the parked robot's box at
        ``xy``, grown by ``margin``, reaches into (the deepest piece per object)."""
        names = self.park_obstacles()[0]
        idx, over = self._park_overlaps(th.as_tensor(xy, dtype=th.float32)[:2], margin)
        deepest: Dict[str, float] = {}
        for i, d in zip(idx.tolist(), over[0].tolist()):
            if d > 0:
                deepest[names[i]] = max(d, deepest.get(names[i], 0.0))
        return sorted(deepest.items(), key=lambda b: -b[1])

    def choose_park_cell(self, pts: th.Tensor, dist: th.Tensor, who: str) -> int:
        """Index into ``pts`` (world xy ``(N, 2)``, the admissible park cells) of the cell to park on: the smallest ``dist``
        (the caller's distance to its centre) among the cells :meth:`park_clear_mask` keeps clear - OmniGibson's eroded
        traversability map keeps only ~0.35 m from mapped obstacles, less than the robot reaches from its base (0.40 m
        behind), so a free cell can still put the robot into furniture (2026-10-12: ottoman_hnmndv_1 pushed 13-17 cm). The
        first minimum wins ties, so a clear cell that was the nearest stays the choice. No cell clear: the nearest cell,
        with a warning."""
        k0 = int(th.argmin(dist))
        clear = self.park_clear_mask(pts)
        if not bool(clear.any()):
            log.warning("%s: no park cell keeps the tucked robot %.2f m clear of the scene objects and inside %s; parking at the nearest "
                        "cell to the centre, (%.2f, %.2f), which overlaps %s", who, PARK_CLEARANCE_M, self.loaded_rooms, float(pts[k0][0]),
                        float(pts[k0][1]), _fmt_blockers(self.park_blockers(pts[k0])))
            return k0
        k = int(th.argmin(th.where(clear, dist, th.full_like(dist, math.inf))))
        if k != k0:
            log.info("%s: park cell (%.2f, %.2f); the nearest cell to the centre, (%.2f, %.2f), overlaps %s with the tucked footprint + %.2f m",
                     who, float(pts[k][0]), float(pts[k][1]), float(pts[k0][0]), float(pts[k0][1]), _fmt_blockers(self.park_blockers(pts[k0])),
                     PARK_CLEARANCE_M)
        return k

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

    def local_collision_pieces(self, obj) -> List[th.Tensor]:
        """The points of every collision mesh of every link, one tensor per mesh, in the object's root frame.

        :meth:`local_hull` merges a link's meshes into one convex hull; here each mesh stays a piece of its own (PhysX collides
        the convex hull of a ``convexHull``-approximated mesh, the approximation of BEHAVIOR-1K's object collision meshes), placed
        by the same transform (``points_in_parent_frame`` through the link's render-side scaled transform, see
        :meth:`local_hull`), so the pieces' points together span the same hull."""
        self.sync_render_transforms()
        root = obj.get_position_orientation()
        Rt = T.quat2mat(root[1]).T
        out = []
        for link in obj.links.values():
            for mesh in link.collision_meshes.values():
                p = mesh.points_in_parent_frame
                if p is None or len(p) == 0:
                    continue
                out.append((link.transform_local_points_to_world(p) - root[0]) @ Rt.T)
        return out

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
            raise RuntimeError("this SimWorld was created with cameras=False (no head camera): renders / segmentation / pixel counts are unavailable"
                               + (" - the sampler built it camera-free for a family listed as pixel-free (sample_isg.PIXEL_SUBTASK_PREFIXES); "
                                  "add the family there or run with ISPATIALGYM_FAST_SAMPLER_CAMERAS=0" if getattr(self, "_fk_camera", False) else ""))

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
        # i7-split 2026-11-07: the per-cell hit heights of the top cells are kept (``SupportTop.zs``) - the support split derives its
        # tolerance from their spread about the plane (``v2.support_split.tolerance_from_top``)
        return SupportTop(support.name, xs, ys, mask, occupied, z_top, yaw, sorted(occupants), resolution, zs=zs)

    def support_interior(self, support, z_surface: float, resolution: float = 0.02, z_margin: float = 0.03, ignore: Iterable = (),
                         probe: float = 0.04, headroom_cap: float = 1.5, board_max_thickness: float = 0.12) -> "SupportInterior":
        """Raster of an interior surface of ``support`` (a shelf inside a bookcase / wardrobe / hallstand) at world height
        ``z_surface``; :meth:`support_top` only sees the topmost face. Per cell of the support's AABB: a downward ray from
        ``z_surface + probe`` whose first hit is the support itself within ``z_margin`` of ``z_surface`` on an upward face
        marks a board cell (a ray that starts inside a wall / panel / the board above hits at distance ~0 and is dropped);
        another object hit there marks the cell occupied. Headroom of a board cell = distance from the board to the first
        body an upward ray meets (any object, the board above included; ``headroom_cap`` when nothing is hit). Recorded:
        the board pieces (the support's collision meshes whose top face is at the surface, at most
        ``board_max_thickness`` tall, under the board cells) and the open sides of the compartment (horizontal rays from
        just inside a side, at half the median headroom up to 0.15 m, leave the support's AABB without hitting it)."""
        lo, hi = support.aabb
        z0 = float(z_surface)
        xs = th.arange(float(lo[0]) + resolution / 2, float(hi[0]), resolution)
        ys = th.arange(float(lo[1]) + resolution / 2, float(hi[1]), resolution)
        gx, gy = th.meshgrid(xs, ys, indexing="ij")
        starts = th.stack([gx.flatten(), gy.flatten(), th.full_like(gx.flatten(), z0 + probe)], dim=1)
        ends = starts.clone()
        ends[:, 2] = z0 - 2 * z_margin
        ignore_paths = [self.robot.prim_path] + [o.prim_path for o in ignore]
        prim_to_obj = {}
        for o in self.scene.objects:
            for l in o.links.values():
                prim_to_obj[l.prim_path] = o
        mask = th.zeros(len(xs), len(ys), dtype=th.bool)
        occupied = th.zeros_like(mask)
        zs = th.full((len(xs), len(ys)), float("nan"))
        occupants = set()
        for k, h in enumerate(raytest_batch(starts.tolist(), ends.tolist(), only_closest=True)):
            i, j = divmod(k, len(ys))
            if not h["hit"] or float(h.get("distance", 1.0)) < 1e-3:
                continue
            body = h["rigidBody"]
            if any(body.startswith(p) for p in ignore_paths):
                continue
            o = prim_to_obj.get(body)
            hz = float(h["position"][2])
            if o is support:
                if abs(hz - z0) <= z_margin and float(h["normal"][2]) >= 0.9:
                    mask[i, j] = True
                    zs[i, j] = hz
            elif o is not None and hz > z0 - z_margin:
                occupied[i, j] = True
                occupants.add(o.name)
        if not bool(mask.any()):
            raise RuntimeError(f"{support.name}: no board cells within {z_margin} m of z = {z0:.3f} (interior surface)")
        z_meas = float(zs[mask].median())
        idx = th.nonzero(mask)
        headroom = th.zeros(len(xs), len(ys))
        up_s = th.stack([xs[idx[:, 0]], ys[idx[:, 1]], zs[idx[:, 0], idx[:, 1]] + 0.005], dim=1)
        up_e = up_s.clone()
        up_e[:, 2] += headroom_cap
        for (i, j), h in zip(idx.tolist(), raytest_batch(up_s.tolist(), up_e.tolist(), only_closest=True)):
            if not h["hit"] or any(h["rigidBody"].startswith(p) for p in ignore_paths):
                headroom[i, j] = headroom_cap
            else:
                headroom[i, j] = max(0.0, float(h.get("distance", 0.0)))
        cell_xy = th.stack([xs[idx[:, 0]], ys[idx[:, 1]]], dim=1)
        cmin, cmax = cell_xy.min(dim=0).values, cell_xy.max(dim=0).values
        board = []
        for link in support.links.values():
            for cm in link.collision_meshes.values():
                plo, phi = cm.aabb
                if abs(float(phi[2]) - z_meas) > z_margin or float(phi[2] - plo[2]) > board_max_thickness:
                    continue
                if float(plo[0]) > float(cmax[0]) or float(phi[0]) < float(cmin[0]) or float(plo[1]) > float(cmax[1]) or float(phi[1]) < float(cmin[1]):
                    continue
                board.append(str(cm.prim_path))
        yaw = G.yaw_of(support.get_position_orientation()[1])
        out = SupportInterior(support.name, xs, ys, mask, occupied, z_meas, yaw, sorted(occupants), resolution, headroom=headroom, z_requested=z0,
                              board_pieces=board, headroom_cap=headroom_cap, zs=zs)
        z_probe = z_meas + min(0.5 * float(headroom[mask].median()), 0.15)
        corners = G.aabb_corners(lo, hi)[:, :2]
        for name, mid, normal, length in out.edges():
            tangent = th.tensor([-float(normal[1]), float(normal[0])])
            pts = [mid - normal * 0.04 + tangent * float(t) for t in th.linspace(-0.4 * float(length), 0.4 * float(length), 7)]
            reach = float((corners @ normal).max()) - float(th.dot(mid, normal)) + 0.10
            s = [[float(p[0]), float(p[1]), z_probe] for p in pts]
            e = [[float(p[0] + normal[0] * reach), float(p[1] + normal[1] * reach), z_probe] for p in pts]
            blocked = sum(1 for h in raytest_batch(s, e, only_closest=True)
                          if h["hit"] and (prim_to_obj.get(h["rigidBody"]) is support or float(h.get("distance", 1.0)) < 1e-3))
            if blocked <= 1:
                out.open_edges.append(name)
        log.info("interior surface of %s at z %.3f (requested %.3f): %d board cells (%d occupied), headroom median %.3f m, open sides %s, board pieces %d",
                 support.name, z_meas, z0, int(mask.sum()), int((mask & occupied).sum()), float(headroom[mask].median()), out.open_edges or "none", len(board))
        return out

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

    def reset_scene_objects(self, task_object_names: Iterable[str] = (), context: str = "") -> int:
        """Put every non-task scene object back into exactly the state of the stable scene json (pose, velocities,
        joints). Called before every instance attempt and before a template is saved, so that instances differ from
        the scene json only in the task-relevant objects and the robot, whatever physics did meanwhile (objects
        nudged by a settling task object, slow creep of stacked furniture over thousands of steps, ...).
        Returns the number of objects whose pose actually changed by more than 1 mm / 0.005 rad; those are logged by
        name with their displacement (``context``: where in the run) and kept in ``last_reset`` as ``(name, m, rad)``."""
        from omnigibson.utils.python_utils import recursively_convert_to_torch

        task_names = set(task_object_names) | set(self.spawned) | {self.robot.name}
        ref = self.stable_scene["state"]["registry"]["object_registry"]
        moved_objects = []
        reloaded: List[str] = []
        for o in self.scene.objects:
            if o.name in task_names or o.name not in ref:
                continue
            r = ref[o.name]["root_link"]
            p, q = o.get_position_orientation()
            d, a = float(th.linalg.norm(p - th.tensor(r["pos"]))), float(G.quat_angle(q, th.tensor(r["ori"])))
            moved = d > 0.001 or a > 0.005
            if moved:
                moved_objects.append((o.name, d, a))
            if moved or o.n_joints > 0:  # articulated scene objects (doors, drawers) can drift in their joints too
                o.load_state(recursively_convert_to_torch(copy.deepcopy(ref[o.name])), serialized=False)
                o.keep_still()
                reloaded.append(o.name)
        #: every object whose state this call re-wrote (the moved ones and all articulated ones) - what a partial cuRobo re-sync
        #: after the park must refresh besides the task objects (train-fast PARK_ONCE, ``SamplerV2._sync_after_park``)
        self.last_reloaded = reloaded
        self.last_reset = sorted(moved_objects, key=lambda m: -m[1])
        if self.last_reset:
            shown = ", ".join(f"{n} {d * 1000:.1f} mm / {math.degrees(a):.1f} deg" for n, d, a in self.last_reset[:12])
            more = f" (+{len(self.last_reset) - 12} more)" if len(self.last_reset) > 12 else ""
            log.info("%s%d scene objects put back to the scene file: %s%s", f"{context}: " if context else "", len(self.last_reset), shown, more)
        return len(moved_objects)

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
        variant_meta = (self.stable_scene.get("metadata") or {}).get("ispatialgym") or {}
        if variant_meta.get("campaign_added_support"):
            merged.setdefault("metadata", {}).setdefault("ispatialgym", {})["campaign_added_support"] = copy.deepcopy(variant_meta["campaign_added_support"])
        if variant_meta.get("moved_objects"):
            # the scene variant's objects parked outside the scene, each with its pose in the source scene file (owner 2026-10-11)
            isg = merged.setdefault("metadata", {}).setdefault("ispatialgym", {})
            isg["moved_objects"] = copy.deepcopy(variant_meta["moved_objects"])
            isg["source_scene_file"] = variant_meta.get("source_scene_file")
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
            # later the hand had swept into the tray and shoved the bottles (the disembedding milk variant never settled
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

    #: the R1Pro head camera prim's ``focalLength`` (mm) when the prim cannot be read in a camera-free world (the value the
    #: camera world measures from the sensor; ``tests/test_v2_oracle.FakeSensor`` carries the same)
    HEAD_CAMERA_FOCAL_MM_FALLBACK = 17.0

    def usd_head_camera_focal_length(self) -> Tuple[float, str]:
        """``(focal length mm, where it came from)`` of the head camera: the ``Camera`` prim under the robot's ``zed_link`` read
        through USD (the prim exists whether or not OmniGibson wrapped it in a ``VisionSensor``), else
        ``HEAD_CAMERA_FOCAL_MM_FALLBACK``."""
        try:
            link = self.robot.links[M.HEAD_CAMERA_LINK]
            for prim in link.prim.GetChildren():
                if prim.GetPrimTypeInfo().GetTypeName() != "Camera":
                    continue
                attr = prim.GetAttribute("focalLength")
                value = attr.Get() if attr and attr.IsValid() else None
                if value is not None and float(value) > 0:
                    return float(value), f"USD {prim.GetPrimPath()}"
        except Exception as e:  # noqa: BLE001 - no such link / prim API: the constant
            log.warning("head camera prim not readable (%r); focal length %.1f mm assumed", e, self.HEAD_CAMERA_FOCAL_MM_FALLBACK)
        else:
            log.warning("no Camera prim under %s; focal length %.1f mm assumed", M.HEAD_CAMERA_LINK, self.HEAD_CAMERA_FOCAL_MM_FALLBACK)
        return float(self.HEAD_CAMERA_FOCAL_MM_FALLBACK), "fallback constant"

    def camera_pose(self) -> G.Pose:
        """World pose of the head camera sensor frame. Camera world: the sensor prim. Camera-free sampler world
        (``usd_camera_model``): the ``zed_link`` pose composed with the USD camera offset (``conventions.CAMERA_IN_ZED``, what the kit's
        ``cam_in_zed`` is in such a world) - the FK of the robot's current configuration as PhysX holds it."""
        if self.cam is None and self._fk_camera:
            from ispatialgym.robot import conventions as C

            zp, zq = self.robot.links[M.HEAD_CAMERA_LINK].get_position_orientation()
            p, q = C.compose((zp.float(), zq.float()), C.camera_offset_in_zed(None))
            return p.float(), q.float()
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
    #: i7-split 2026-11-07: (nx, ny) hit height of each top cell (``nan`` off the top; :meth:`SimWorld.support_top` / ``support_interior``),
    #: None for a raster built without them
    zs: Optional[th.Tensor] = None

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

    @staticmethod
    def _margin_offsets(margin: float) -> th.Tensor:
        """The (dx, dy) of :meth:`contains`'s neighbours (the point itself first), as ``th.tensor([dx, dy])`` made them one by one."""
        if margin > 0:
            return th.tensor([[0.0, 0.0], [margin, 0], [-margin, 0], [0, margin], [0, -margin], [margin, margin], [-margin, -margin], [margin, -margin],
                              [-margin, margin]])
        return th.zeros(1, 2)

    def _all_free(self, pts: th.Tensor) -> bool:
        """Every point of ``pts`` (N, 2) on a free raster cell: the index arithmetic of the former per-point loop, elementwise (the same
        float ops, so the same cells), with ``free`` built once instead of per point (i7-stancememo r3: disembedding's start draw spent
        220 of 285 s in 343k scalar ``contains`` calls)."""
        i = th.round((pts[:, 0] - self.xs[0]) / self.resolution)
        j = th.round((pts[:, 1] - self.ys[0]) / self.resolution)
        if bool(((i < 0) | (j < 0) | (i >= len(self.xs)) | (j >= len(self.ys))).any()):
            return False
        return bool(self.free[i.long(), j.long()].all())

    def contains(self, xy, margin: float = 0.0) -> bool:
        """True if world ``xy`` lies on an exposed (free) cell; ``margin`` requires neighbours within it to be free too."""
        p = G.t(xy)
        offs = self._margin_offsets(margin)
        return self._all_free(p.reshape(1, 2) + offs if margin > 0 else p.reshape(1, 2))

    def polygon_supported(self, poly_xy: th.Tensor, margin: float = 0.02) -> bool:
        """Every vertex and the centroid of a footprint polygon lie on free cells (with margin)."""
        verts = G.t(poly_xy)
        pts = (verts[:, None, :] + self._margin_offsets(margin)[None]).reshape(-1, 2) if margin > 0 else verts
        return self._all_free(pts) and self.contains(poly_xy.mean(dim=0), 0.0)

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


@dataclass
class SupportInterior(SupportTop):
    """Raster of an interior surface of a support (a shelf inside a bookcase / wardrobe / hallstand) at a chosen height
    (:meth:`SimWorld.support_interior`): :class:`SupportTop` plus the headroom above each board cell, the open sides of the
    compartment and the collision pieces of the board the objects stand on."""

    headroom: Optional[th.Tensor] = None  # (nx, ny) m from the board to the first body above; 0 off the board
    z_requested: float = 0.0
    open_edges: List[str] = field(default_factory=list)
    board_pieces: List[str] = field(default_factory=list)  # collision mesh prim paths of the board under the cells
    headroom_cap: float = 1.5

    def usable(self, height: float, margin: float) -> th.Tensor:
        """Free board cells whose headroom fits an object of ``height`` plus ``margin``."""
        return self.free & (self.headroom >= height + margin)

    def cells_usable(self, height: float, margin: float) -> th.Tensor:
        idx = th.nonzero(self.usable(height, margin))
        return th.stack([self.xs[idx[:, 0]], self.ys[idx[:, 1]]], dim=1)

    def to_json(self):
        d = super().to_json()
        hr = self.headroom[self.mask] if self.headroom is not None and bool(self.mask.any()) else th.zeros(0)
        d.update({"kind": "interior", "z_requested": self.z_requested, "open_edges": list(self.open_edges), "board_pieces": list(self.board_pieces),
                  "n_board_cells": int(self.mask.sum()), "headroom_cap": self.headroom_cap,
                  "headroom_m": {"min": round(float(hr.min()), 4), "median": round(float(hr.median()), 4), "max": round(float(hr.max()), 4)} if len(hr) else None})
        return d
