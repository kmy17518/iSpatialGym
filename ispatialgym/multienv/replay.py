"""Kinematic replayer: BEHAVIOR LeRobot v3 shards from physics-only ``demo.hdf5`` files (``ISPATIALGYM_PHYSICS_ONLY=1``
collection, :mod:`ispatialgym.oracle.physics_only`).

    python -m ispatialgym.multienv.replay --demos <demo dir>[,<demo dir>...] --root <lerobot root> --task <variant>
        [--batched-usd-edits] [--slots 1] [--tiled] [--repo-prefix replay] [--async-flush] [--first-renders 10]
        [--renders 1] [--warmup-s 60] [--proprio recorded|loaded] [--results <jsonl>]
    python -m ispatialgym.multienv.replay --watch <demo root> --done-file <path> ...   (a pipeline stage: see below)

One process per variant, ``--slots`` scenes of the variant's scene in one simulator, each slot playing a different demo (a
slot takes the next demo of the list when its demo ends). The demos must share a template (the same scene objects), as
production's one-world replay (``replay_obs_isg.py --one-world``). Per tick every slot loads its demo's next recorded
state into its scene (no physics step), the cameras render once, and each slot's cameras feed that slot's BEHAVIOR
:class:`~ispatialgym.oracle.behavior_lerobot.ShardWriter`: one LeRobot v3 dataset per slot, ``<root>/<repo
prefix>_slot<k>``, BEHAVIOR's encoders, background flush with ``--async-flush``. Measured (dishwasher, 12 CPUs): one slot
with ``--batched-usd-edits`` replays ~2x faster than plain loads with the same images; several slots in one process
were not faster than one (a state load costs more per slot in a bigger stage), and ``--tiled`` changes the rgb images
(see its help), so a GPU is best served by several one-slot processes.

The environment is production's aligned-replay one (``DataPlaybackWrapper.create_from_hdf5`` with replay_obs's settings:
the recorded env config and scene file, visual-only objects, replay_obs's sensors, modalities and proprioception keys)
with ``num_envs`` = slots and ``DummyTask`` (``ISpatialGymTask`` binds one environment; it adds nothing visible).

Frames pair as production's ``AlignedLeRobotPlaybackWrapper`` and live collection: frame i = the observation of state i
with action i (reward / terminated / truncated i). ``observation.state`` is the collection-time proprioception
(``isg/behavior_proprio``, recorded by physics-only collection); ``--proprio loaded`` takes the robot's proprioception
of the loaded state instead. Renders per tick: ``--first-renders`` when a slot has just loaded the first state of a demo
(production's replay renders 10 times there), else ``--renders`` (production's ``n_render_iterations`` = 1).

``--watch``: the demos are the successful runs appearing under ``<demo root>`` (``meta.json`` with status success, written
by run_isg after demo.hdf5), taken in the order they finish; the environment is built from the first one, idle slots
take new demos as they appear, and the process ends when ``--done-file`` exists and every finished demo is replayed.

Prints ``REPLAY_DEMO {json}`` per finished demo and ``REPLAY_SUMMARY {json}`` (env-steps/s, per-tick time split, CPU,
GPU memory) at the end; ``--results`` appends the same lines as json.

``--scene`` (2026-11-01 train-fast PARTIAL_SCENE): a demo collected in the sampler's cropped world (``ISPATIALGYM_FAST_PARTIAL_SCENE``,
hdf5 attr ``isg_scene_crop``) records the cropped world's scene file. ``full`` builds the environment from the full template
(``isg_full_scene_file``, else ``template.json`` of the template the config's ``tro_path`` names) with the full evaluation rooms
(``isg_full_load_room_instances``), and each slot restores the template restricted to the objects the scene loaded with every
recorded object's entry taken from the demo (:func:`merge_partial_scene_file`); the per-step states then load by object hash into
the full scene and the objects outside the crop keep the template's state. ``recorded``: the recorded scene file, as before.
``auto`` (default): ``full`` when the first demo is cropped, else ``recorded``.

``--goal-occlusion`` (owner 2026-11-05: goal occlusion is screened in the sampler for mental_rotation and disembedding, in this replay
stage for every other family): at a demo's last frame - its settled end - the oracle's screen of :mod:`ispatialgym.eval.goal_occlusion`
(the head camera's depth, the box screen of every target / partner, the panel screen of a fixture from a second depth render with the
slot's robot hidden; an inside goal judges its partner; camera relocalization is exempt). A demo with a judged object at least
``GO.THRESHOLD`` (50 %) hidden is left out of the dataset (its episode is dropped, ``REPLAY_DEMO.kept`` false) and every verdict is
printed as ``REPLAY_OCCLUSION {json}`` (alias, demo, max_occluded, worst, per-object shares). ``auto`` (default) screens a demo whose
instance the sampler did not measure (its ``summary.json`` carries ``"visibility": "pending"`` - sampled camera-free - or no measured
``goal_occlusion``); ``on`` every demo; ``off`` none. Not with ``--tiled`` (no per-slot depth annotator).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch as th

BEHAVIOR_PROPRIO_KEY = "isg/behavior_proprio"

# ----------------------------------------------------------------------------------------------------------------------------
# 2026-11-01 train-fast (PARTIAL_SCENE): partial demo -> full variant scene
# ----------------------------------------------------------------------------------------------------------------------------

SCENE_MODES = ("auto", "recorded", "full")


def is_cropped(scene_crop: Optional[Dict[str, Any]]) -> bool:
    """Was the demo collected in a cropped world (hdf5 attr ``isg_scene_crop`` with a mode other than ``full``)?"""
    return isinstance(scene_crop, dict) and (scene_crop.get("mode") or "full") != "full"


def scene_mode(requested: str, scene_crop: Optional[Dict[str, Any]]) -> str:
    """``--scene``: ``auto`` is ``full`` for a cropped demo, else ``recorded`` (today's replay)."""
    if requested not in SCENE_MODES:
        raise ValueError(f"--scene {requested!r}: one of {SCENE_MODES}")
    if requested == "auto":
        return "full" if is_cropped(scene_crop) else "recorded"
    return requested


def _object_registry(scene_file: Dict[str, Any]) -> Dict[str, Any]:
    state = scene_file.get("state") or {}
    return (state.get("registry") or {}).get("object_registry", {}) if "registry" in state else state.get("object_registry", {})


def _system_registry(scene_file: Dict[str, Any]) -> Dict[str, Any]:
    state = scene_file.get("state") or {}
    return (state.get("registry") or {}).get("system_registry", {}) if "registry" in state else state.get("system_registry", {})


def _is_robot_info(name: str, info: Dict[str, Any]) -> bool:
    """A robot entry of a scene file: OmniGibson's robot classes (a recorded ``R1Pro``) or merge_scene_files' generic ``Robot``
    (the template's)."""
    return str(info.get("class_module") or "").startswith("omnigibson.robots") or info.get("class_name") == "Robot"


def merge_partial_scene_file(full: Dict[str, Any], recorded: Dict[str, Any], loaded_names=None, is_robot=None,
                             to_scene_frame=None) -> Dict[str, Any]:
    """The scene file a full-scene replay restores for a demo recorded in a cropped world: the full template's objects (its robot
    entries dropped) and every object of the recorded scene file, a recorded object taking the recorded entry (init_info and
    state), the robot, ``metadata``, ``versions`` and the scene pose from the recorded file, the systems of both (recorded wins),
    the scene ``init_info`` from the full template (the replay scene is the full one). ``loaded_names`` (the object names of the
    replay scene's registry): the result holds exactly those of them that either file knows - so ``Scene.restore`` sees no
    topology change - and a recorded object the scene did not load is a ValueError. ``to_scene_frame``
    (``Scene._scene_file_in_scene_frame``): both files are first moved to their scene frame (it returns None for a file saved at the
    origin). The inputs are left untouched; the result is a deep copy (``restore(update_initial_file=True)`` keeps and edits it)."""
    is_robot = is_robot or _is_robot_info
    if to_scene_frame is not None:
        full = to_scene_frame(full) or full
        recorded = to_scene_frame(recorded) or recorded
    f_info, r_info = full["objects_info"]["init_info"], recorded["objects_info"]["init_info"]
    f_reg, r_reg = _object_registry(full), _object_registry(recorded)
    f_state, r_state = full.get("state") or {}, recorded.get("state") or {}
    for key, ident in (("pos", [0.0, 0.0, 0.0]), ("ori", [0.0, 0.0, 0.0, 1.0])):
        a = [float(v) for v in (f_state.get(key) if f_state.get(key) is not None else ident)]
        b = [float(v) for v in (r_state.get(key) if r_state.get(key) is not None else ident)]
        if len(a) != len(b) or any(abs(x - y) > 1e-6 for x, y in zip(a, b)):
            raise ValueError(f"the full template and the recorded scene file were saved at different scene poses ({key} {a} vs {b}); "
                             "pass to_scene_frame=Scene._scene_file_in_scene_frame")
    names = [n for n, v in f_info.items() if n not in r_info and not is_robot(n, v)] + list(r_info)
    if loaded_names is not None:
        loaded = set(loaded_names)
        missing = sorted(n for n in r_info if n not in loaded)
        if missing:
            raise ValueError(f"{len(missing)} object(s) of the recorded (cropped) scene file are not in the full replay scene "
                             f"(not in the full template / outside its load_room_instances?): {missing[:20]}")
        names = [n for n in names if n in loaded]
    init_info: Dict[str, Any] = {}
    objects: Dict[str, Any] = {}
    for n in names:
        info, reg = (r_info, r_reg) if n in r_info else (f_info, f_reg)
        init_info[n] = info[n]
        if n in reg:
            objects[n] = reg[n]
    registry = {**(r_state.get("registry") or {}), "object_registry": objects, "system_registry": {**_system_registry(full), **_system_registry(recorded)}}
    out = {k: v for k, v in recorded.items() if k not in ("objects_info", "state", "init_info")}
    for k in ("metadata", "versions"):
        if k not in out and k in full:
            out[k] = full[k]
    out["init_info"] = full.get("init_info", recorded.get("init_info"))
    out["objects_info"] = {**full["objects_info"], **recorded["objects_info"], "init_info": init_info}
    out["state"] = {**r_state, "registry": registry}
    return copy.deepcopy(out)


def full_scene_path(demo: "Demo") -> str:
    """The full template scene file of a demo: ``isg_full_scene_file``, else ``<template dir>/template.json`` of the config's task
    ``tro_path`` (``<template dir>/instances/<iid>-tro_state.json``), else of meta.json's ``template_dir`` - the first that exists."""
    cands = []
    if demo.full_scene_file:
        cands.append(demo.full_scene_file)
    if demo.tro_path:
        cands.append(os.path.join(os.path.dirname(os.path.dirname(demo.tro_path)), "template.json"))
    try:
        with open(os.path.join(demo.dir, "meta.json")) as f:
            tdir = json.load(f).get("template_dir")
        if tdir:
            cands.append(os.path.join(tdir, "template.json"))
    except (OSError, ValueError):
        pass
    for c in cands:
        if os.path.exists(c):
            return c
    raise SystemExit(f"--scene full: no full template scene file for {demo.dir} (tried {cands})")


_FULL_SCENES: Dict[str, Dict[str, Any]] = {}


def load_full_scene(demo: "Demo") -> Dict[str, Any]:
    """The demo's full template scene file (read once per path; callers merge, which copies)."""
    path = os.path.abspath(full_scene_path(demo))
    if path not in _FULL_SCENES:
        with open(path, "rb") as f:
            raw = f.read()
        if demo.full_scene_sha256 and hashlib.sha256(raw).hexdigest() != demo.full_scene_sha256:
            print(f"WARNING --scene full: {path} is not the template.json {demo.dir} was collected against (sha256 differs)", flush=True)
        _FULL_SCENES[path] = json.loads(raw)
    return _FULL_SCENES[path]


def _scene_frame_fn():
    try:
        from omnigibson.scenes.scene_base import Scene

        return getattr(Scene, "_scene_file_in_scene_frame", None)
    except Exception:  # noqa: BLE001
        return None


class Demo:
    """The recorded arrays of one demo's committed episode (``data/demo_0``)."""

    def __init__(self, demo_dir: str, campaign_record=None):
        self.dir = demo_dir
        self.name = os.path.basename(os.path.normpath(demo_dir))
        self.campaign_meta = None
        if campaign_record is not None:
            from ispatialgym.multienv.campaign_rendering import checked_metadata

            self.campaign_meta = checked_metadata(campaign_record, demo_dir)
        with h5py.File(os.path.join(demo_dir, "demo.hdf5"), "r") as f:
            g = f["data/demo_0"]
            self.scene_file = json.loads(f["data"].attrs["scene_file"])
            # 2026-11-01 train-fast (PARTIAL_SCENE): the recorder's crop attrs (absent: collected in the full world)
            attrs = f["data"].attrs
            self.scene_crop = json.loads(attrs["isg_scene_crop"]) if "isg_scene_crop" in attrs else None
            self.full_scene_file = json.loads(attrs["isg_full_scene_file"]) if "isg_full_scene_file" in attrs else None
            self.full_scene_sha256 = json.loads(attrs["isg_full_scene_sha256"]) if "isg_full_scene_sha256" in attrs else None
            self.full_rooms_known = "isg_full_load_room_instances" in attrs
            self.full_rooms = json.loads(attrs["isg_full_load_room_instances"]) if self.full_rooms_known else None
            try:
                self.tro_path = (json.loads(attrs["config"]).get("task") or {}).get("tro_path")
            except (KeyError, ValueError, AttributeError):
                self.tro_path = None
            if campaign_record is not None:
                # The collection HDF5 can retain a node-local scratch path;
                # the accepted manifest pins its unchanged durable TRO bytes.
                self.tro_path = campaign_record["artifacts"]["tro"]["path"]
            self.state = g["state"][()]
            self.state_size = g["state_size"][()]
            self.action = th.as_tensor(g["action"][()], dtype=th.float32)
            self.reward = g["reward"][()]
            self.terminated = g["terminated"][()]
            self.truncated = g["truncated"][()]
            self.proprio = g[BEHAVIOR_PROPRIO_KEY][()] if BEHAVIOR_PROPRIO_KEY in g else None
        self.T = int(self.action.shape[0])
        if self.state.shape[0] != self.T:
            raise ValueError(f"{demo_dir}: {self.state.shape[0]} states for {self.T} actions")

    def state_at(self, i: int) -> th.Tensor:
        return th.as_tensor(self.state[i, : int(self.state_size[i])], dtype=th.float32)


class DemoSource:
    """The demos to replay: a fixed list, or (``watch``) the successful runs appearing under a demo root until
    ``done_file`` exists."""

    def __init__(self, demo_dirs: List[str], watch: Optional[str] = None, done_file: Optional[str] = None, poll_s: float = 5.0,
                 claims: Optional[str] = None, campaign_records=None):
        self.pending: List[str] = list(demo_dirs)
        self.watch, self.done_file, self.poll_s = watch, done_file, float(poll_s)
        self.claims = claims
        self.campaign_records = campaign_records
        if claims:
            os.makedirs(claims, exist_ok=True)
        self.seen = set(demo_dirs)
        self._last_scan = 0.0

    def _claim(self, demo_dir: str) -> bool:
        """``claims``: several replayers watching one demo root take each demo once (an exclusive create)."""
        if not self.claims:
            return True
        try:
            os.close(os.open(os.path.join(self.claims, os.path.basename(os.path.normpath(demo_dir))), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return True
        except FileExistsError:
            return False

    def _scan(self, force: bool = False) -> None:
        if self.watch is None or (not force and time.time() - self._last_scan < self.poll_s):
            return
        self._last_scan = time.time()
        found = []
        for name in os.listdir(self.watch) if os.path.isdir(self.watch) else []:
            d = os.path.join(self.watch, name)
            meta = os.path.join(d, "meta.json")
            if d in self.seen or not os.path.exists(meta) or not os.path.exists(os.path.join(d, "demo.hdf5")):
                continue
            self.seen.add(d)
            try:
                with open(meta) as f:
                    ok = json.load(f).get("status") == "success"
            except (OSError, ValueError):
                self.seen.discard(d)  # meta.json being written: next scan
                continue
            if ok:
                found.append((os.path.getmtime(meta), d))
        self.pending += [d for _, d in sorted(found)]

    def next(self, force: bool = False) -> Optional[Demo]:
        self._scan(force)
        while self.pending:
            d = self.pending.pop(0)
            if self._claim(d):
                record = None if self.campaign_records is None else self.campaign_records[os.path.realpath(d)]
                return Demo(d, campaign_record=record)
        return None

    def finished(self) -> bool:
        """No demo left and none to come."""
        if self.watch is None:
            return not self.pending
        if not (self.done_file and os.path.exists(self.done_file)):
            return False
        self._scan(force=True)
        return not self.pending

    def first(self) -> Demo:
        while True:
            d = self.next(force=True)
            if d is not None:
                return d
            if self.finished():
                raise SystemExit("no demo to replay")
            time.sleep(self.poll_s)


def env_config(first: Demo, demos: List[Demo], first_dir: str, n_slots: int, scene: str = "recorded") -> Dict[str, Any]:
    """``create_from_hdf5``'s config edits for replay_obs's settings (include_contacts=False, include_task_obs=False,
    replay_obs's robot sensors / modalities / proprioception), with ``n_slots`` envs and DummyTask. ``scene="full"`` (2026-11-01
    train-fast PARTIAL_SCENE): the scene file is the full template merged with the first demo's recorded objects
    (:func:`merge_partial_scene_file`) and the rooms the full evaluation rooms."""
    from omnigibson.envs.data_wrapper import _add_recorded_non_kin_states_to_scene_file

    from ispatialgym.oracle import behavior_lerobot as BL

    with h5py.File(os.path.join(first_dir, "demo.hdf5"), "r") as f:
        config = json.loads(f["data"].attrs["config"])
    config["env"]["flatten_obs_space"] = True
    config["env"]["num_envs"] = int(n_slots)
    if scene == "full":
        # 2026-11-01 train-fast (PARTIAL_SCENE): every object of the full template (the cropped-away ones at the template state), the
        # recorded objects as recorded; the recorded non-kin schema goes onto the recorded objects as before (a recorded object the
        # base does not know raises in _add_recorded_non_kin_states_to_scene_file)
        scene_file = merge_partial_scene_file(load_full_scene(first), first.scene_file, None, to_scene_frame=_scene_frame_fn())
        if first.full_rooms_known:
            config["scene"]["load_room_instances"] = first.full_rooms
    else:
        scene_file = json.loads(json.dumps(first.scene_file))
    _add_recorded_non_kin_states_to_scene_file(scene_file=scene_file, recorded_scene_files=[d.scene_file for d in demos])
    config["scene"]["scene_file"] = scene_file
    config["task"] = {"type": "DummyTask", "include_obs": False}
    config["objects"] = []
    for robot_cfg in config["robots"]:
        robot_cfg.update(BL.robot_overrides())
    return config


def make_slot_view(env, k: int):
    """Slot ``k`` of the multi-env environment as the one-scene, one-robot environment BEHAVIOR's writer expects."""
    from omnigibson.envs.env_wrapper import EnvironmentWrapper

    class SlotView(EnvironmentWrapper):
        def __init__(self, env, k):
            super().__init__(env=env)
            object.__setattr__(self, "_slot", int(k))

        @property
        def num_envs(self):
            return 1

        @property
        def scene(self):
            return self.env.scenes[self._slot]

        @property
        def scenes(self):
            return [self.env.scenes[self._slot]]

        @property
        def robots(self):
            return self.env.scenes[self._slot].robots

        @property
        def external_sensors(self):
            return None

    return SlotView(env, k)


@contextlib.contextmanager
def individual_render_products(env):
    """With tiled rendering the environment pauses every camera's own render product; BEHAVIOR's writer reads each
    camera's intrinsics from it (the camera-params annotator) when it creates its dataset. Resume them meanwhile."""
    from omnigibson.sensors.vision_sensor import VisionSensor

    import omnigibson as og

    paused = []
    if getattr(env, "_tiled_sensor", None) is not None:
        with og.sim.editing_usd():
            for scene in env.scenes:
                for sensor in scene.robots[0].sensors.values():
                    if isinstance(sensor, VisionSensor) and not sensor.render_product.hydra_texture.updates_enabled:
                        sensor.render_product.hydra_texture.updates_enabled = True
                        paused.append(sensor)
    try:
        yield
    finally:
        if paused:
            with og.sim.editing_usd():
                for sensor in paused:
                    sensor.render_product.hydra_texture.updates_enabled = False


def process_gpu_mib() -> Optional[int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            pid, mib = [x.strip() for x in line.split(",")]
            if int(pid) == os.getpid():
                return int(mib)
    except Exception:  # noqa: BLE001
        return None
    return None


def _same(a, b) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, th.Tensor) or isinstance(b, th.Tensor):
        return isinstance(a, th.Tensor) and isinstance(b, th.Tensor) and a.shape == b.shape and th.equal(a, b)
    return a == b


# ----------------------------------------------------------------------------------------------------------------------------
# 2026-11-05 one trunk (owner): goal occlusion screened at the replay stage for the families the sampler does not screen
# ----------------------------------------------------------------------------------------------------------------------------


def instance_record(demo: "Demo") -> Dict[str, Any]:
    """The demo's instance summary (``<iid>.summary.json`` next to its TRO) and its variant / subtask (the demo's meta.json first)."""
    rec: Dict[str, Any] = {"summary": None, "variant": None, "subtask": None, "tracked": None}
    meta_p = os.path.join(demo.dir, "meta.json")
    if os.path.exists(meta_p):
        try:
            m = json.load(open(meta_p))
            rec["variant"], rec["subtask"] = m.get("variant"), m.get("subtask")
            if m.get("tracked_objects"):
                rec["tracked"] = sorted(set((m.get("tracked_objects") or {}).values()))
        except ValueError:
            pass
    if rec["tracked"] is None and demo.tro_path and os.path.exists(demo.tro_path):
        try:  # the task's objects as the TRO binds them (the oracle judges its tracked objects only)
            binds = (json.load(open(demo.tro_path)).get("ispatialgym") or {}).get("object_bindings") or {}
            rec["tracked"] = sorted({b["scene_object_name"] for b in binds.values() if isinstance(b, dict) and b.get("scene_object_name")}) or None
        except (ValueError, OSError):
            pass
    if demo.tro_path and demo.tro_path.endswith("-tro_state.json"):
        sp = demo.tro_path[: -len("-tro_state.json")] + ".summary.json"
        if os.path.exists(sp):
            try:
                rec["summary"] = json.load(open(sp))
            except ValueError:
                pass
    return rec


def needs_occlusion_screen(mode: str, rec: Dict[str, Any]) -> Tuple[bool, str]:
    """(screen, why) under ``--goal-occlusion`` ``mode`` for a demo's :func:`instance_record`."""
    from ispatialgym.eval import goal_occlusion as GO

    sub = rec.get("subtask") or rec.get("variant")
    if mode == "off":
        return False, "off"
    if GO.exempt(sub):
        return False, "exempt subtask"
    if mode == "on":
        return True, "on"
    summ = rec.get("summary") or {}
    if summ.get("visibility") == "pending":
        return True, "visibility pending (sampled camera-free)"
    if (summ.get("goal_occlusion") or {}).get("max_occluded") is not None or "occluded" in (summ.get("goal_occlusion") or {}):
        return False, "measured by the sampler"
    return True, "not measured by the sampler"


def occlusion_screen(scene, render, subtask: Optional[str], tracked: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """:mod:`ispatialgym.eval.goal_occlusion` on the slot's current (loaded) state, as ``Executor.goal_occlusion_report`` screens a
    demo's last frame: the head camera's settled depth, the task's (``tracked``: the demo's tracked objects) targets / partners and
    fixtures - every other door / dishwasher of the scene is not the task's; without ``tracked`` the target / partner objects only -; the
    robot hidden for a second render when a fixture is judged."""
    from omnigibson.utils.constants import JointType

    from ispatialgym.eval import goal_occlusion as GO
    from ispatialgym.oracle.head_depth import settled_depth
    from ispatialgym.oracle.qa import CameraModel
    from ispatialgym.oracle.r1pro_model import HEAD_CAMERA_SENSOR_SUFFIX

    robot = scene.robots[0]
    key = next((k for k in robot.sensors if k.endswith(HEAD_CAMERA_SENSOR_SUFFIX)), None)
    if key is None:
        return {"status": "no head camera", "occluded": False, "max_occluded": None}
    sensor = robot.sensors[key]
    intr = GO.intrinsics_of(CameraModel.from_sensor(sensor))
    names = set(tracked) if tracked else None
    judged = [o for o in scene.objects if GO.role(getattr(o, "name", "")) and (o.name in names if names is not None else GO.role(o.name) == "object")]
    depth = settled_depth(sensor, render)
    depth_without = None
    if any(GO.role(o.name) == "fixture" for o in judged):
        robot.visible = False
        try:
            depth_without = settled_depth(sensor, render)
        finally:
            robot.visible = True
            for _ in range(3):
                render()
    cp, cq = sensor.get_position_orientation()
    objects = GO.screen_scene(depth, (cp.float().tolist(), cq.float().tolist()), intr, judged, depth_without_robot=depth_without,
                              revolute=JointType.JOINT_REVOLUTE, subtask=subtask)
    return GO.report(objects, method="replay head-camera depth", camera_key=key, resolution=[intr[0], intr[1]])


class Slot:
    def __init__(self, k: int, scene, writer, robot_name: str, scene_mode: str = "recorded"):
        self.k = k
        self.scene = scene
        #: 2026-11-01 train-fast (PARTIAL_SCENE): "full" restores each demo merged into the full template
        self.scene_mode = scene_mode
        self.n_objects_scene: Optional[int] = None
        self.n_objects_recorded: Optional[int] = None
        self.writer = writer
        self.proprio_key = f"{robot_name}::proprio"
        self.demo: Optional[Demo] = None
        self.i = 0
        self.t0 = 0.0
        self.last_obs = None
        #: --delta-load: every object's recorded state as last loaded into this slot (None: load everything next)
        self.loaded: Optional[Dict[str, Any]] = None
        self.campaign_restoration = None

    def start(self, demo: Demo) -> None:
        from omnigibson.envs.data_wrapper import _align_scene_object_states_with_recorded_schema

        scene_file = demo.scene_file
        self.n_objects_recorded = len(demo.scene_file["objects_info"]["init_info"])
        if self.scene_mode == "full":
            # 2026-11-01 train-fast (PARTIAL_SCENE): the full template restricted to the objects this scene loaded, the recorded objects
            # as recorded - restore sees no topology change; a recorded object the full scene lacks fails here
            loaded = list(self.scene.object_registry.get_dict("name").keys())
            try:
                scene_file = merge_partial_scene_file(load_full_scene(demo), demo.scene_file, loaded, to_scene_frame=_scene_frame_fn())
            except ValueError as e:
                raise SystemExit(f"--scene full: {demo.dir}: {e}") from e
            unknown = sorted(set(loaded) - set(scene_file["objects_info"]["init_info"]))
            if unknown:
                print(f"WARNING --scene full: {len(unknown)} object(s) of the replay scene are in neither the template nor {demo.name}'s "
                      f"recorded scene file; restore removes them: {unknown[:10]}", flush=True)
        elif is_cropped(demo.scene_crop):
            print(f"WARNING --scene recorded: {demo.name} was collected in a cropped world ({demo.scene_crop.get('mode')}); restoring its "
                  "recorded scene file changes this scene's objects to the crop", flush=True)
        self.scene.restore(scene_file, update_initial_file=True)
        _align_scene_object_states_with_recorded_schema(scene=self.scene, recorded_scene_file=scene_file)
        self.campaign_restoration = None
        if getattr(demo, "campaign_meta", None) is not None:
            import omnigibson as og
            from ispatialgym.multienv.campaign_rendering import restore_campaign_scene

            if self.scene_mode != "recorded":
                raise ValueError("campaign rendering requires the exact recorded scene")
            self.scene.load_state(demo.state_at(0), serialized=True)
            self.campaign_restoration = restore_campaign_scene(self.scene, og.sim, demo.campaign_meta)
        self.n_objects_scene = len(self.scene.object_registry.get_dict("name"))
        self.demo, self.i, self.t0 = demo, 0, time.time()
        self.loaded = None

    def load(self, flat: th.Tensor, delta: bool) -> Tuple[int, int]:
        """Load a recorded state into the slot's scene; returns (objects in the state, objects loaded). ``delta``: objects
        whose recorded state equals the one last loaded into this slot are left as they are (nothing else changes a
        slot's objects between loads: no physics step), which leaves the scene as a full load would."""
        if not delta:
            self.scene.load_state(flat, serialized=True)
            return 0, 0
        state, _ = self.scene.deserialize(flat)
        objs = state["registry"]["object_registry"]
        last = self.loaded
        changed = {n: s for n, s in objs.items() if last is None or n not in last or not _same(s, last[n])}
        registry = dict(state["registry"])
        registry["object_registry"] = changed
        self.scene.load_state({**state, "registry": registry}, serialized=False)
        self.loaded = dict(objs) if last is None else {**last, **objs}
        return len(objs), len(changed)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos", default="", help="demo dirs (demo.hdf5 inside), comma-separated, or @file with one per line")
    ap.add_argument("--watch", default=None, help="replay the successful runs appearing under this demo root (see the module doc)")
    ap.add_argument("--done-file", default=None, help="--watch: no more demos come once this file exists")
    ap.add_argument("--claims", default=None, help="--watch: claim directory shared by several replayers of one demo root")
    ap.add_argument("--campaign-manifest", default=None, help="accepted campaign instances.jsonl; verify inputs and restore exact augmentations")
    ap.add_argument("--sequential-augmentation", action="store_true", help="explicit single-slot, fixed-inventory lighting/parking sequence")
    ap.add_argument("--diagnostic-episode-fault", choices=("failure", "interruption"), default=None,
                    help="explicit single-slot campaign test: abort first episode after one recorded action")
    ap.add_argument("--poll-s", type=float, default=5.0)
    ap.add_argument("--root", required=True, help="LeRobot root dir: one dataset per slot, <root>/<repo prefix>_slot<k>")
    ap.add_argument("--task", required=True, help="task name written into the datasets (the variant)")
    ap.add_argument("--slots", type=int, default=1)
    ap.add_argument("--tiled", action="store_true",
                    help="with several slots, one tiled render product per camera name. Depth matches collection, rgb does not: the "
                         "renderer's post-processing (exposure, denoising) runs over the whole tiled image, so the tiles come out "
                         "darker than a camera's own render product (head camera 12.7 dB PSNR vs a collection frame)")
    ap.add_argument("--repo-prefix", default="replay")
    ap.add_argument("--async-flush", action="store_true", help="the writers' 32-step segments go to a background thread")
    ap.add_argument("--first-renders", type=int, default=10)
    ap.add_argument("--renders", type=int, default=1)
    ap.add_argument("--delta-load", action="store_true",
                    help="per slot, load only the objects whose recorded state differs from the one last loaded into it. The scene "
                         "state matches a full load (poses bitwise, only PhysX sleep flags differ) but the images do not quite: "
                         "right-wrist depth 0.33 mm mean error vs collection (0.12 with full loads), and with --batched-usd-edits the "
                         "moving parts render one state late")
    ap.add_argument("--batched-usd-edits", action="store_true",
                    help="load each tick's states inside og.sim.batched_usd_edits(): one Fabric synchronization per tick instead "
                         "of one per USD edit (~150 per state); state, poses and images as with plain loads, load 2.2x faster")
    ap.add_argument("--warmup-s", type=float, default=60.0,
                    help="render the slots' first states this long before the first frame: a fresh renderer's images keep changing for "
                         "~300 frames (texture streaming, exposure), which a collection process has behind it after its first demo")
    ap.add_argument("--proprio", choices=("recorded", "loaded"), default="recorded")
    ap.add_argument("--scene", choices=SCENE_MODES, default="auto",
                    help="2026-11-01 train-fast PARTIAL_SCENE: 'full' replays a demo collected in a cropped world (hdf5 isg_scene_crop) in the "
                         "full template (task rooms + neighbours; objects outside the crop at the template state), 'recorded' in its "
                         "recorded scene file (as before); 'auto': full for a cropped first demo, else recorded")
    ap.add_argument("--results", default=None, help="append REPLAY_DEMO / REPLAY_SUMMARY json lines here")
    ap.add_argument("--goal-occlusion", choices=("auto", "on", "off"), default="auto",
                    help="screen the goal occlusion at each demo's last frame and leave an occluded demo out of the dataset (module doc); "
                         "auto: the demos the sampler did not measure")
    ap.add_argument("--profile-ticks", default=None, help="START:COUNT: cProfile those ticks, stats to <results>.prof.txt (or stdout)")
    args = ap.parse_args(argv)

    if args.demos.startswith("@"):
        with open(args.demos[1:]) as f:
            demo_dirs = [line.strip() for line in f if line.strip()]
    else:
        demo_dirs = [d for d in args.demos.split(",") if d]
    if not demo_dirs and not args.watch:
        ap.error("--demos or --watch")
    if args.watch and not args.done_file:
        ap.error("--watch needs --done-file")
    campaign_records = None
    if args.sequential_augmentation and (not args.campaign_manifest or int(args.slots) != 1):
        ap.error('sequential augmentation requires an exact campaign manifest and one slot')
    if args.campaign_manifest:
        if args.watch or args.scene != "recorded" or args.tiled or args.delta_load:
            ap.error("campaign rendering requires a fixed accepted list, --scene recorded, individual cameras and full state loads")
        from ispatialgym.multienv.campaign_rendering import selected_records

        campaign_records = selected_records(args.campaign_manifest, demo_dirs,
            sequential_augmentation=args.sequential_augmentation, slots=int(args.slots))
    if args.diagnostic_episode_fault and (not args.sequential_augmentation or
            os.environ.get("ISPATIALGYM_CAMPAIGN_RESTORATION_PROBE") != "1"):
        ap.error("episode-fault control requires an explicit sequential campaign probe")
    n_slots = max(1, int(args.slots) if args.watch else min(int(args.slots), len(demo_dirs)))
    tiled = args.tiled and n_slots > 1
    t_start = time.time()
    source = DemoSource(demo_dirs, args.watch, args.done_file, args.poll_s, claims=args.claims, campaign_records=campaign_records)
    first = source.first()

    def usable(demo: Demo) -> Demo:
        if args.proprio == "recorded" and demo.proprio is None:
            raise SystemExit(f"no {BEHAVIOR_PROPRIO_KEY} in {demo.dir} (record with ISPATIALGYM_PHYSICS_ONLY=1 or "
                             "ISPATIALGYM_RECORD_BEHAVIOR_PROPRIO=1, or use --proprio loaded)")
        return demo

    usable(first)

    import omnigibson as og
    from omnigibson.macros import gm

    if args.sequential_augmentation:
        from ispatialgym.oracle import run_isg
        from ispatialgym.oracle.campaign_batch import saved_world_scope

        saved_world_scope(run_isg, list(campaign_records.values()))

    for name, value in (("ENABLE_TRANSITION_RULES", False), ("ENABLE_TILED_RENDERING", tiled), ("VISUAL_ONLY", True)):
        if getattr(gm, name) != value:
            with gm.unlocked():
                setattr(gm, name, value)

    from ispatialgym.oracle import behavior_lerobot as BL

    # the environment of the first demo, as production's one-world replay: the others align their object-state schema
    # when their slot starts them
    # 2026-11-01 train-fast (PARTIAL_SCENE): the world - recorded scene file or the full template - from the first demo
    scene = scene_mode(args.scene, first.scene_crop)
    config = env_config(first, [first], first.dir, n_slots, scene=scene)
    env = og.Environment(configs=config)
    load_s = time.time() - t_start
    robot_name = env.scenes[0].robots[0].name
    with individual_render_products(env):
        writers = [BL.ShardWriter(make_slot_view(env, k), args.root, f"{args.repo_prefix}_slot{k}", args.task, async_flush=args.async_flush)
                   for k in range(n_slots)]
    slots = [Slot(k, env.scenes[k], writers[k], robot_name, scene_mode=scene) for k in range(n_slots)]
    slots[0].start(first)
    n_objects_recorded_first = slots[0].n_objects_recorded  # REPLAY_SUMMARY (PARTIAL_SCENE): the first demo's recorded objects
    n_demos = 1

    def fill_idle(force: bool = False) -> None:
        nonlocal n_demos
        for s in slots:
            if s.demo is None:
                d = source.next(force)
                if d is None:
                    return
                s.start(usable(d))
                n_demos += 1

    fill_idle(force=True)
    t_warm = time.time()
    n_warm = 0
    if args.warmup_s > 0:
        for s in slots:
            if s.demo is not None:
                s.scene.load_state(s.demo.state_at(0), serialized=True)
        while time.time() - t_warm < args.warmup_s:
            og.sim.render()
            n_warm += 1
    warmup = {"s": round(time.time() - t_warm, 1), "renders": n_warm}
    out = open(args.results, "a") if args.results else None

    def emit(tag: str, rec: Dict[str, Any]) -> None:
        line = f"{tag} {json.dumps(rec)}"
        print(line, flush=True)
        if out is not None:
            out.write(json.dumps({"tag": tag, **rec}) + "\n")
            out.flush()

    t_ticks = defaultdict(float)
    n_occl = defaultdict(int)  # goal-occlusion verdicts: kept / dropped / not_screened
    n_ticks = n_steps = 0
    diagnostic_fault_pending = args.diagnostic_episode_fault
    objs_in = objs_loaded = 0
    idle_s = 0.0
    r0, w0 = resource.getrusage(resource.RUSAGE_SELF), time.time()
    gpu_peak = process_gpu_mib() or 0
    prof_window = tuple(int(v) for v in args.profile_ticks.split(":")) if args.profile_ticks else None
    prof = None
    while True:
        if prof_window is not None:
            if n_ticks == prof_window[0]:
                import cProfile

                prof = cProfile.Profile()
                prof.enable()
            elif prof is not None and n_ticks == prof_window[0] + prof_window[1]:
                prof.disable()
                import io
                import pstats

                buf = io.StringIO()
                pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(50)
                pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(30)
                target = (args.results + ".prof.txt") if args.results else None
                if target:
                    with open(target, "w") as f:
                        f.write(buf.getvalue())
                else:
                    print(buf.getvalue(), flush=True)
                prof, prof_window = None, None
        active = [s for s in slots if s.demo is not None]
        if not active:
            if source.finished():
                break
            t_idle = time.time()
            time.sleep(args.poll_s)
            fill_idle(force=True)
            idle_s += time.time() - t_idle
            continue
        t1 = time.perf_counter()
        with og.sim.batched_usd_edits() if args.batched_usd_edits else contextlib.nullcontext():
            for s in active:
                n_in, n_loaded = s.load(s.demo.state_at(s.i), args.delta_load)
                objs_in += n_in
                objs_loaded += n_loaded
        t2 = time.perf_counter()
        for _ in range(args.first_renders if any(s.i == 0 for s in active) else args.renders):
            og.sim.render()
        t3 = time.perf_counter()
        obs_list, _ = env.get_obs()
        t4 = time.perf_counter()
        for s in active:
            d, i = s.demo, s.i
            # tiled camera buffers are persistent CUDA tensors (per-slot views), and the writer converts frames with .numpy()
            obs = {key: ((v.detach().cpu() if v.is_cuda else v.clone()) if isinstance(v, th.Tensor) else v) for key, v in obs_list[s.k].items()}
            if args.proprio == "recorded":
                obs[s.proprio_key] = th.as_tensor(d.proprio[i], dtype=th.float32)
            from ispatialgym.multienv.campaign_render_cleanup import write_frame
            error, cleanup = write_frame(s.writer, d, i, obs, fault=diagnostic_fault_pending)
            if error is not None:
                emit("REPLAY_EPISODE_ABORTED", {"demo": d.dir, "slot": s.k, "kept": False,
                    "campaign_restoration": s.campaign_restoration, "cleanup": cleanup,
                    "wall_s": round(time.time()-s.t0, 2)})
                if not cleanup["expected_controlled_failure"]:
                    raise error
                diagnostic_fault_pending = None
                s.demo, s.loaded, s.last_obs = None, None, None
                continue
            s.i += 1
            n_steps += 1
            if s.i == d.T:
                # the last action; the observation it led to is not a frame (the writer pairs it with no action)
                s.writer.step(d.action[d.T - 1], obs, float(d.reward[d.T - 1]), bool(d.terminated[d.T - 1]), bool(d.truncated[d.T - 1]))
                keep, occl = True, None
                rec = instance_record(d)
                screen, why = needs_occlusion_screen(args.goal_occlusion, rec)
                if screen and tiled:
                    screen, why = False, "not measured: tiled rendering"
                if screen:
                    t_o = time.perf_counter()
                    try:
                        occl = occlusion_screen(s.scene, og.sim.render, rec.get("subtask") or rec.get("variant"), rec.get("tracked"))
                    except Exception as e:  # noqa: BLE001  (a failed screen keeps the demo and says why)
                        occl = {"status": f"error: {e!r}", "occluded": False, "max_occluded": None}
                    t_ticks["goal_occlusion"] += time.perf_counter() - t_o
                    keep = not occl.get("occluded")
                    n_occl[("dropped" if not keep else "kept")] += 1
                else:
                    n_occl["not_screened"] += 1
                emit("REPLAY_OCCLUSION", {"demo": d.dir, "variant": rec.get("variant"), "screened": screen, "why": why, "kept": keep,
                                          "max_occluded": (occl or {}).get("max_occluded"), "worst": (occl or {}).get("worst"),
                                          "occluded": (occl or {}).get("occluded"), "status": (occl or {}).get("status"),
                                          "objects": {k: {kk: v.get(kk) for kk in ("status", "occluded", "judged", "visible_px", "total_px") if kk in v}
                                                      for k, v in ((occl or {}).get("objects") or {}).items()},
                                          "sampler_predicted": ((rec.get("summary") or {}).get("goal_occlusion") or {}).get("max_occluded")})
                index = s.writer.end(save=keep)
                emit("REPLAY_DEMO", {"demo": d.dir, "slot": s.k, "frames": d.T, "episode_index": index, "wall_s": round(time.time() - s.t0, 2),
                                     "dataset": os.path.join(args.root, f"{args.repo_prefix}_slot{s.k}"), "kept": keep,
                                     "campaign_restoration": s.campaign_restoration,
                                     "goal_occlusion": None if occl is None else {"max_occluded": occl.get("max_occluded"), "worst": occl.get("worst"),
                                                                                  "occluded": occl.get("occluded")}})
                s.demo = None
        fill_idle()
        t5 = time.perf_counter()
        t_ticks["load_states"] += t2 - t1
        t_ticks["render"] += t3 - t2
        t_ticks["get_obs"] += t4 - t3
        t_ticks["writers"] += t5 - t4
        n_ticks += 1
        if n_ticks % 200 == 0:
            gpu_peak = max(gpu_peak, process_gpu_mib() or 0)
    if args.diagnostic_episode_fault and diagnostic_fault_pending is not None:
        raise RuntimeError("requested renderer episode failure was not exercised")
    t_flush = time.perf_counter()
    for w in writers:
        w.close()
    close_s = time.perf_counter() - t_flush
    w1, r1 = time.time(), resource.getrusage(resource.RUSAGE_SELF)
    wall = w1 - w0
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    busy = wall - idle_s
    emit("REPLAY_SUMMARY", {
        "slots": n_slots, "tiled": tiled, "demos": n_demos, "env_steps": n_steps, "ticks": n_ticks, "load_s": round(load_s, 1), "warmup": warmup,
        "replay_wall_s": round(wall, 2), "idle_s": round(idle_s, 2), "close_s": round(close_s, 2),
        "env_steps_per_s": round(n_steps / max(wall, 1e-9), 2), "env_steps_per_busy_s": round(n_steps / max(busy, 1e-9), 2),
        "ms_per_tick": {k: round(1000 * v / max(1, n_ticks), 2) for k, v in t_ticks.items()},
        "cpu_cores_busy": round(cpu / max(wall, 1e-9), 2), "cpu_ms_per_env_step": round(1000 * cpu / max(1, n_steps), 1),
        "gpu_mib_peak": max(gpu_peak, process_gpu_mib() or 0), "async_flush": bool(args.async_flush), "proprio": args.proprio,
        "delta_load": bool(args.delta_load), "batched_usd_edits": bool(args.batched_usd_edits),
        "objects_per_state": round(objs_in / max(1, n_steps), 1) if args.delta_load else None,
        "objects_loaded_per_state": round(objs_loaded / max(1, n_steps), 1) if args.delta_load else None,
        "renders": args.renders, "first_renders": args.first_renders, "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "scene": scene, "goal_occlusion": {"mode": args.goal_occlusion, **n_occl}, "n_objects_scene": slots[0].n_objects_scene, "n_objects_recorded": n_objects_recorded_first,
        "cpus_allowed": len(os.sched_getaffinity(0)), "host": os.uname().nodename, "pid": os.getpid()})
    if out is not None:
        out.close()
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    try:
        import omnigibson as og

        og.shutdown()
    except OSError:  # Kit's temp-dir cleanup at exit; the datasets are written by then
        pass
    sys.exit(rc)
