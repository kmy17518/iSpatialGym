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

``--scene-pert remove`` (owner 2026-11-14: training-set scene perturbation at batched-replay time, every class but camera
relocalization): per slot and replayed episode, a deterministic (``--scene-pert-seed``, demo, replay index) draw hides removable
scene objects - never the task's objects, anything the demo moves or that touches them or the robot, nothing structural; see
:mod:`ispatialgym.multienv.replay_scene_perturbation`. The recorded states still load, so the scene state is the recorded one; only
the images change. Each episode's record is printed as ``REPLAY_SCENE_PERT {json}``, carried by ``REPLAY_DEMO.scene_perturbation``
and appended to ``<root>/scene_perturbation.jsonl``; ``--scene-pert-from`` replays the removals of earlier records exactly.

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
        #: --settle-pad: frame i of the replayed episode plays recorded row rows[i] (None: frame i = row i)
        self.rows: Optional[np.ndarray] = None
        self.T_recorded = self.T
        self.settle: Optional[Dict[str, Any]] = None

    def state_at(self, i: int) -> th.Tensor:
        if self.rows is not None:
            i = int(self.rows[i])
        return th.as_tensor(self.state[i, : int(self.state_size[i])], dtype=th.float32)

    def pad_settle(self, frac: float, min_steps: int) -> Dict[str, Any]:
        """Batched-replay settle window (owner 2026-11-15): the replayed episode is the demo's action steps A followed by
        ``max(ceil(frac * A), min_steps)`` settle steps - it REPLACES the recorded settle window (collection: 45 steps), it is not
        added to it. Rows past the recorded end repeat the last recorded row (state, action, reward; the scene is settled and the
        replay kinematic, so the frames are the settled scene); a shorter window drops recorded settle rows. ``terminated`` /
        ``truncated`` are the recorded ones on recorded non-final rows and the recorded final row's on the last frame only."""
        import math

        a, s_rec, src = settle_split(self)
        s_out = max(int(math.ceil(float(frac) * a)), int(min_steps))
        n = a + s_out
        rows = np.minimum(np.arange(n), self.T_recorded - 1)
        last = self.T_recorded - 1
        self.rows = rows
        self.action = self.action[th.as_tensor(rows)]
        self.reward = np.asarray(self.reward)[rows]
        term, trunc = np.asarray(self.terminated), np.asarray(self.truncated)
        self.terminated = np.where(rows == last, False, term[rows]).astype(term.dtype)
        self.truncated = np.where(rows == last, False, trunc[rows]).astype(trunc.dtype)
        self.terminated[-1], self.truncated[-1] = term[last], trunc[last]
        if self.proprio is not None:
            self.proprio = self.proprio[rows]
        self.T = int(n)
        self.settle = {"action_steps": int(a), "settle_steps_recorded": int(s_rec), "settle_steps": int(s_out),
                       "length_recorded": int(self.T_recorded), "length": int(n), "frac": float(frac), "min_steps": int(min_steps),
                       "padded_rows": int(max(0, n - self.T_recorded)), "dropped_rows": int(max(0, self.T_recorded - n)),
                       "source": src}
        return self.settle


def settle_split(demo: "Demo") -> Tuple[int, int, str]:
    """(action steps, recorded settle steps, where from) of a demo: meta.json ``n_action_steps`` (run_isg: the program's actions;
    ``n_settle_steps`` its settle window, together the recorded rows), else the hdf5 ``isg/phase`` rows marked settle."""
    try:
        with open(os.path.join(demo.dir, "meta.json")) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        meta = {}
    a = meta.get("n_action_steps")
    if isinstance(a, int) and 0 < a <= demo.T_recorded:
        s = meta.get("n_settle_steps")
        if isinstance(s, int) and a + s != demo.T_recorded:
            print(f"WARNING settle: {demo.name}: n_action_steps {a} + n_settle_steps {s} != {demo.T_recorded} recorded rows; "
                  "the action steps decide", flush=True)
        return int(a), int(demo.T_recorded - a), "meta.n_action_steps"
    with h5py.File(os.path.join(demo.dir, "demo.hdf5"), "r") as f:
        g = f["data/demo_0"]
        if "isg/phase" in g:
            ph = g["isg/phase"][()]
            ph = [p.decode() if isinstance(p, bytes) else str(p) for p in np.asarray(ph).reshape(-1)]
            settle = [i for i, p in enumerate(ph) if "settle" in p.lower()]
            if settle and settle == list(range(settle[0], len(ph))):
                return settle[0], len(ph) - settle[0], "isg/phase"
    raise SystemExit(f"--settle-pad: {demo.dir}: no action/settle split (meta.json n_action_steps or isg/phase settle rows)")


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


def merged_extent(demo: "Demo", adjacent=None) -> Dict[str, Any]:
    """Render switch 2026-11-15 (owner): the merged scene = the task rooms + their adjacent rooms, not the whole house. The demo's
    ``isg_full_load_room_instances`` (the rooms ``--scene full`` loads) must equal its TRO ``scene.task_rooms`` U
    ``sim_utils.adjacent_rooms(scene, task_rooms)``; ``adjacent`` overrides that function (tests)."""
    tro = {}
    if demo.tro_path and os.path.exists(demo.tro_path):
        with open(demo.tro_path) as f:
            tro = json.load(f).get("ispatialgym") or {}
    scene_info = tro.get("scene") or {}
    tasks = list(scene_info.get("task_rooms") or [])
    model = scene_info.get("model") or (demo.scene_file.get("init_info") or {}).get("args", {}).get("scene_model")
    if adjacent is None:
        from ispatialgym.sampling.sim_utils import adjacent_rooms as adjacent
    adj = list(adjacent(model, tasks)) if tasks else []
    loaded = sorted(demo.full_rooms or [])
    expect = sorted(set(tasks) | set(adj))
    return {"scene_extent": "task_rooms+adjacent", "task_rooms": tasks, "adjacent": sorted(adj), "loaded": loaded,
            "equal": bool(tasks) and expect == loaded}


def sensor_config(image_size: Optional[int]) -> Dict[str, Any]:
    """The robot's sensor config: BEHAVIOR's replay_obs one (wrists 480, head 720 with a 40 mm aperture), or with ``image_size``
    (batched replay 2026-11-15: 224) every camera at image_size x image_size - the same apertures, so the same fields of view."""
    from ispatialgym.oracle import behavior_lerobot as BL

    cfg = copy.deepcopy(BL.ROBOT_SENSOR_CONFIG)
    if image_size:
        for v in cfg.values():
            v.setdefault("sensor_kwargs", {}).update({"image_height": int(image_size), "image_width": int(image_size)})
    return cfg


def env_config(first: Demo, demos: List[Demo], first_dir: str, n_slots: int, scene: str = "recorded",
               image_size: Optional[int] = None) -> Dict[str, Any]:
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
        robot_cfg["sensor_config"] = sensor_config(image_size)
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
        #: --scene-pert: the objects hidden in this slot's scene for the current episode, and its record
        self.hidden: List[str] = []
        self.scene_pert: Optional[Dict[str, Any]] = None
        self.pert_check: Dict[str, Any] = {}
        #: replay-time lighting: campaign restoration leaves the recorded collection lighting out
        self.ignore_recorded_lighting = False

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

            if self.scene_mode not in ("recorded", "full"):
                raise ValueError("campaign rendering requires the recorded or the merged scene")
            self.scene.load_state(demo.state_at(0), serialized=True)
            # merged (full) scene of a cropped demo: restoration against the recorded crop's objects, outside fixtures lit alike
            crop = (set(demo.scene_file["objects_info"]["init_info"]) if self.scene_mode == "full" and is_cropped(demo.scene_crop)
                    else None)
            self.campaign_restoration = restore_campaign_scene(self.scene, og.sim, demo.campaign_meta, crop_names=crop,
                                                               ignore_lighting=self.ignore_recorded_lighting)
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


def RSP_camera(d: Demo) -> bool:
    from ispatialgym.multienv.replay_scene_perturbation import _read_json, is_camera_relocalization

    return is_camera_relocalization(_read_json(os.path.join(d.dir, "meta.json")))


def replay_goal_image(s: "Slot", d: Demo, obs: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    """Camera relocalization (owner 2026-11-15, eval semantics: ``eval.instances.set_goal`` = the head camera's last frame of a replay of
    the demo, posed at the demo's last ``isg/head_camera_pose``): the last replayed frame of this episode - same env, same replay
    lighting and removals - saved next to the replayed TRO as ``<instance>[-r<k>]-goal_rgb.png`` and registered as its
    ``goal_observation``."""
    ri = (s.scene_pert or {}).get("replay_index", 0)
    from ispatialgym.multienv.replay_lighting import fit_name

    name = fit_name(d.name + (f"-r{ri}" if ri else "") + "-goal_rgb.png", "-goal_rgb.png")
    return save_goal_image(s, d, obs, os.path.join(out_dir, name), rel=name)


def save_goal_image(s: "Slot", d: Demo, obs: Dict[str, Any], path: str, rel: Optional[str] = None) -> Dict[str, Any]:
    """The head camera's rgb of ``obs`` (the episode's last rendered frame) as a png at ``path``, and its goal-observation
    record (``rgb_path`` = ``rel``, sha256, resolution, intrinsics, the demo's last recorded head-camera pose)."""
    import hashlib as _h

    from PIL import Image

    from ispatialgym.oracle.r1pro_model import HEAD_CAMERA_SENSOR_SUFFIX

    robot = s.scene.robots[0]
    key = next(k for k in robot.sensors if k.endswith(HEAD_CAMERA_SENSOR_SUFFIX))
    rgb_key = next(k for k in obs if k.endswith("rgb") and key.replace(robot.name + ":", "") in k)
    a = obs[rgb_key].numpy()[..., :3].astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(a).save(path)
    name = rel if rel is not None else os.path.basename(path)
    with h5py.File(os.path.join(d.dir, "demo.hdf5"), "r") as f:
        pose = [float(v) for v in f["data/demo_0/isg/head_camera_pose"][-1]]
    try:
        from ispatialgym.oracle.qa import CameraModel

        intr = CameraModel.from_sensor(robot.sensors[key]).intrinsics()
    except Exception:  # noqa: BLE001
        intr = None
    return {"sensor_key": key, "rgb_path": name, "rgb_sha256": _h.sha256(open(path, "rb").read()).hexdigest(),
            "resolution": [int(a.shape[1]), int(a.shape[0])], "intrinsics": intr,
            "camera_pose_world": {"position": pose[:3], "orientation": pose[3:7]}, "n_render_iterations": 1,
            "source": "batched replay: the episode's last frame (eval set_goal semantics), replay lighting and removals of this episode",
            "frame": int(d.T - 1), "recorded_row": int(d.rows[-1]) if d.rows is not None else int(d.T - 1)}


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def instance_link(d: Demo, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """What ties a replayed episode to its instance / template (batched replay 2026-11-15; train and eval alike): the demo dir and
    its hdf5 sha256, meta.json names (instance_id, variant, subtask, template_dir), the source TRO path + sha256 and its template
    (template.json next to ``instances/``), the oracle sidecar and summary next to the TRO, EVAL_INSTANCE_ID when the demo dir has
    one, plus the caller's ``--instance-meta`` row (class / unit / purpose / source receipt)."""
    meta = {}
    try:
        with open(os.path.join(d.dir, "meta.json")) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        pass
    tro = d.tro_path
    tdir = os.path.dirname(os.path.dirname(tro)) if tro else meta.get("template_dir")
    stem = tro[: -len("-tro_state.json")] if tro and tro.endswith("-tro_state.json") else None
    rec = {"demo_dir": os.path.realpath(d.dir), "demo_name": d.name,
           "demo_sha256": file_sha256(os.path.join(d.dir, "demo.hdf5")),
           "instance_id": meta.get("instance_id"), "variant": meta.get("variant"), "subtask": meta.get("subtask"),
           "template_dir": tdir, "template_json": os.path.join(tdir, "template.json") if tdir else None,
           "template_meta": os.path.join(tdir, "template_meta.json") if tdir else None,
           "tro_path": tro, "tro_sha256": file_sha256(tro) if tro and os.path.exists(tro) else None,
           "oracle_sidecar": (stem + ".oracle.json") if stem and os.path.exists(stem + ".oracle.json") else None,
           "summary": (stem + ".summary.json") if stem and os.path.exists(stem + ".summary.json") else None,
           "scene_crop": (d.scene_crop or {}).get("mode") if d.scene_crop else None}
    eid = os.path.join(d.dir, "EVAL_INSTANCE_ID")
    if os.path.exists(eid):
        with open(eid) as f:
            rec["eval_instance_id"] = f.read().strip()
    if extra:
        rec.update({("listed_demo_dir" if k == "demo_dir" else k): v for k, v in extra.items()})
    return rec


def read_instance_meta(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """``--instance-meta``: jsonl rows with ``demo_dir`` (or ``demo_name``) and any fields to carry into the episode records."""
    out: Dict[str, Dict[str, Any]] = {}
    if not path:
        return out
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                key = row.get("demo_dir") or row.get("demo_name")
                out[os.path.basename(os.path.normpath(key))] = row
    return out


def seg_visibility(s: "Slot", d: Demo, render, n_poses: int = 3, renders: int = 4) -> Dict[str, Any]:
    """Owner 2026-11-15 06:05: visible pixels per object of the slot's (unperturbed) scene from its head camera placed at
    ``n_poses`` recorded head-camera poses of the demo (first, middle, last; ``isg/head_camera_pose``, scene frame), instance
    segmentation only (an annotator added for these renders and removed after). The camera's parent-frame pose is restored."""
    from ispatialgym.oracle.r1pro_model import HEAD_CAMERA_SENSOR_SUFFIX

    t0 = time.perf_counter()
    robot = s.scene.robots[0]
    key = next(k for k in robot.sensors if k.endswith(HEAD_CAMERA_SENSOR_SUFFIX))
    sensor = robot.sensors[key]
    with h5py.File(os.path.join(d.dir, "demo.hdf5"), "r") as f:
        cams = f["data/demo_0/isg/head_camera_pose"][()]
    idx = sorted({0, len(cams) // 2, len(cams) - 1})[:max(1, int(n_poses))]
    saved = sensor.get_position_orientation(frame="parent")
    added = "seg_instance" not in sensor.modalities
    px: Dict[str, int] = defaultdict(int)
    n = 0
    # the seg_instance annotator segfaults the renderer with light shapes drawn (house_single_floor kitchen/garden views;
    # sim_utils.SimWorld.LIGHT_SHAPES_SETTING): off for these renders only, restored before the next replay frame
    import carb

    from ispatialgym.sampling.sim_utils import SimWorld

    settings = carb.settings.get_settings()
    shapes = settings.get(SimWorld.LIGHT_SHAPES_SETTING)
    settings.set_int(SimWorld.LIGHT_SHAPES_SETTING, 0)
    try:
        if added:
            sensor.add_modality("seg_instance")
        for i in idx:
            sensor.set_position_orientation(th.as_tensor(cams[i, :3], dtype=th.float32), th.as_tensor(cams[i, 3:7], dtype=th.float32),
                                            frame="scene")
            for _ in range(int(renders)):
                render()
                n += 1
            obs, info = sensor.get_obs()
            seg = obs["seg_instance"]
            mapping = {int(k): str(v) for k, v in (info.get("seg_instance") or {}).items()}
            ids, counts = th.unique(th.as_tensor(seg).cpu(), return_counts=True)
            for i_, c in zip(ids.tolist(), counts.tolist()):
                name = mapping.get(int(i_))
                if name:
                    px[name.rsplit("/", 1)[-1]] += int(c)
    finally:
        sensor.set_position_orientation(*saved, frame="parent")
        if added:
            sensor.remove_modality("seg_instance")
        if shapes is not None:
            settings.set(SimWorld.LIGHT_SHAPES_SETTING, shapes)
        for _ in range(2):
            render()
            n += 1
    return {"px": dict(px), "poses": [int(i) for i in idx], "renders": n, "s": round(time.perf_counter() - t0, 3)}


def save_frames(out_dir: str, s: "Slot", d: Demo, i: int, obs: Dict[str, Any]) -> None:
    """--scene-pert-frames: the slot's rgb observations of frame ``i`` as png."""
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    ri = (s.scene_pert or {}).get("replay_index", 0)
    for key, v in obs.items():
        if not key.endswith("rgb") or not isinstance(v, th.Tensor):
            continue
        a = v.numpy()
        if a.ndim != 3:
            continue
        a = a[..., :3].astype(np.uint8)
        cam = key.split("::")[-2].replace(":", "_").replace("/", "_") if "::" in key else key.replace(":", "_")
        Image.fromarray(a).save(os.path.join(out_dir, f"slot{s.k}_{d.name[-14:]}_r{ri}_f{i:05d}_{cam}.png"))


def n_slots_req(args) -> int:
    return int(args.slots)


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
    ap.add_argument("--diagnostic-fault-boundary", choices=("writer", "post_render"), default="writer")
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
    ap.add_argument("--scene-pert", choices=("off", "remove"), default="off",
                    help="replay-time scene perturbation: hide removable non-task scene objects per slot and episode (module doc)")
    ap.add_argument("--scene-pert-seed", type=int, default=0)
    ap.add_argument("--scene-pert-scope", choices=("outside_crop", "all"), default="outside_crop",
                    help="outside_crop (owner 2026-11-15): for a cropped demo only objects of the replay scene outside the recorded crop "
                         "are candidates (with --scene full: the merged scene's objects the collection never simulated); an uncropped "
                         "demo: every object. all: every object of the replay scene")
    ap.add_argument("--scene-pert-plan", default=None,
                    help="the render planner's in-view ranking (ispatialgym.multienv.replay_inview, offline CPU): a perturbed group "
                         "removes the first removable single object of its ranking, else the group-keyed draw")
    ap.add_argument("--scene-pert-visibility", choices=("off", "seg"), default="off",
                    help="seg (owner 2026-11-15): per goal group, before its first episode, instance segmentation of the unperturbed "
                         "scene at 3 recorded head-camera poses ranks the removal units by visible pixels (the group's choice); "
                         "nothing visible: the offline ranking / group-keyed draw")
    ap.add_argument("--scene-pert-vis-threshold", type=int, default=500, help="min visible pixels of a unit (summed over the poses)")
    ap.add_argument("--scene-pert-inview-pick", choices=("first", "random"), default="first",
                    help="which in-view candidate a perturbed episode removes: the top-ranked one, or (owner 2026-11-15, variance) a "
                         "uniform pick among all in-view candidates seeded by the draw key")
    ap.add_argument("--scene-pert-p-episode", type=float, default=0.5, help="probability that a replayed episode is perturbed")
    ap.add_argument("--scene-pert-p-unit", type=float, default=0.5, help="--scene-pert-mode subset: probability that a removable unit is removed")
    ap.add_argument("--scene-pert-mode", choices=("one", "subset"), default="one",
                    help="one (owner 2026-11-15): a perturbed instance loses exactly one single-object unit, drawn per goal group; "
                         "subset: every removable unit with --scene-pert-p-unit (per episode)")
    ap.add_argument("--scene-pert-index-base", type=int, default=0,
                    help="replay index of a demo's first replay in this run (the n-th replay of a demo in the run is base + n)")
    ap.add_argument("--scene-pert-from", default=None, help="REPLAY_SCENE_PERT records (results jsonl or log): apply exactly these removals")
    ap.add_argument("--scene-pert-verify", action="store_true",
                    help="per tick: the slots' tracked-object poses against the recorded isg/object_pose; at the last frame the "
                         "loaded scene state against the recorded one")
    ap.add_argument("--scene-pert-frames", default=None, help="save each slot's rgb frames at --scene-pert-frame-at to this dir (png)")
    ap.add_argument("--scene-pert-frame-at", default="0,0.5,1", help="frame positions as fractions of the demo length")
    ap.add_argument("--scene-pert-probe", default=None,
                    help="after the slots' first episodes start: point every slot's head camera at up to 3 objects removed in some "
                         "slots and kept in others, save the views to this dir and print REPLAY_SCENE_PERT_PROBE (cameras restored)")
    ap.add_argument("--scene-pert-probe-only", action="store_true", help="stop after the probe (no dataset written)")
    ap.add_argument("--lighting-pert", choices=("auto", "off", "replay"), default="auto",
                    help="replay-time lighting (owner 2026-11-15; ispatialgym.multienv.replay_lighting): replay = ignore the recorded "
                         "collection lighting and apply a fresh draw per env (fixtures) and per batch (skybox); auto = replay with "
                         "--campaign-manifest unless every demo is camera relocalization (exempt: recorded lighting), else off")
    ap.add_argument("--lighting-pert-seed", type=int, default=None, help="default: --scene-pert-seed")
    ap.add_argument("--lighting-pert-p", type=float, default=0.5)
    ap.add_argument("--lighting-pert-dist", choices=("uniform", "normal"), default="uniform",
                    help="intensity factor: uniform on --lighting-pert-range (owner 2026-11-15 03:55), or N(1, sigma) clipped to it")
    ap.add_argument("--lighting-pert-sigma", type=float, default=0.17, help="--lighting-pert-dist normal: N(1, sigma)")
    ap.add_argument("--lighting-pert-range", default="0.5,1.5", help="intensity factor clip lo,hi")
    ap.add_argument("--lighting-pert-color-sigma", type=float, default=0.04, help="fixture colour factors ~ N(1, sigma) per channel")
    ap.add_argument("--lighting-skybox", choices=("batch", "episode", "group"), default="batch",
                    help="batch: one skybox draw per process (its goal groups); episode (--slots 1 only; 'group' is an alias): drawn with "
                         "each episode's draw key (--pert-unit) and applied before it")
    ap.add_argument("--pert-unit", choices=("demo", "group"), default="demo",
                    help="what one replay-time draw covers: demo (owner 2026-11-15: an independent coin flip and draw per demo) or group "
                         "(one shared draw per goal group)")
    ap.add_argument("--lighting-pert-from", default=None, help="REPLAY_SCENE_PERT records: re-apply their lighting exactly")
    ap.add_argument("--replay-tro-dir", default=None,
                    help="replayed-instance TROs (campaign.replay_perturbations, written to <root>/instances by every perturbed replay): "
                         "re-apply their removals and lighting exactly (the loader path)")
    ap.add_argument("--lighting-sweep", default=None, help="review sheet: factors (comma separated) applied to every fixture and the skybox")
    ap.add_argument("--lighting-sweep-dir", default=None, help="--lighting-sweep: save the slots' head rgb at each factor here, then stop")
    ap.add_argument("--lighting-sweep-at", type=float, default=0.5, help="--lighting-sweep: the demo frame (fraction of its length)")
    ap.add_argument("--profile-ticks", default=None, help="START:COUNT: cProfile those ticks, stats to <results>.prof.txt (or stdout)")
    # batched replay 2026-11-15 (owner)
    ap.add_argument("--settle-pad", type=float, default=None,
                    help="replace the recorded settle window: the episode = the demo's action steps A + max(ceil(SETTLE_PAD * A), "
                         "--settle-pad-min) settle steps (rows past the recorded end repeat the last recorded row); default: as recorded")
    ap.add_argument("--settle-pad-min", type=int, default=50)
    ap.add_argument("--image-size", type=int, default=None,
                    help="every camera (head and wrists) renders and is written at IMAGE_SIZE x IMAGE_SIZE (same apertures); "
                         "default: BEHAVIOR's 720 head / 480 wrists")
    ap.add_argument("--goal-images", choices=("on", "off"), default="off",
                    help="on: save every kept episode's last rendered head-camera frame (its goal image) as "
                         "<root>/goal_images/<dataset>/episode_<index>.png, recorded in <root>/episodes_isg.jsonl")
    ap.add_argument("--instance-meta", default=None,
                    help="jsonl rows (demo_dir or demo_name + any fields: class, unit, purpose, receipt) carried into "
                         "<root>/episodes_isg.jsonl with the demo's instance / template / TRO linkage")
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
        if args.watch or args.scene not in ("recorded", "full") or args.tiled or args.delta_load:
            ap.error("campaign rendering requires a fixed accepted list, --scene recorded (or full: the merged scene), individual "
                     "cameras and full state loads")
        from ispatialgym.multienv.campaign_rendering import selected_records

        replay_light = (args.lighting_pert in ("auto", "replay") or bool(args.lighting_pert_from) or bool(args.replay_tro_dir))
        campaign_records = selected_records(args.campaign_manifest, demo_dirs,
            sequential_augmentation=args.sequential_augmentation, slots=int(args.slots), replay_lighting=replay_light)
    if args.diagnostic_episode_fault and (not args.campaign_manifest or int(args.slots) != 1 or
            os.environ.get("ISPATIALGYM_CAMPAIGN_RESTORATION_PROBE") != "1"):
        ap.error("episode-fault control requires an explicit single-slot campaign probe")
    n_slots = max(1, int(args.slots) if args.watch else min(int(args.slots), len(demo_dirs)))
    tiled = args.tiled and n_slots > 1
    t_start = time.time()
    source = DemoSource(demo_dirs, args.watch, args.done_file, args.poll_s, claims=args.claims, campaign_records=campaign_records)
    first = source.first()

    imeta_early = read_instance_meta(args.instance_meta)

    def usable(demo: Demo) -> Demo:
        if args.proprio == "recorded" and demo.proprio is None:
            raise SystemExit(f"no {BEHAVIOR_PROPRIO_KEY} in {demo.dir} (record with ISPATIALGYM_PHYSICS_ONLY=1 or "
                             "ISPATIALGYM_RECORD_BEHAVIOR_PROPRIO=1, or use --proprio loaded)")
        row = (imeta_early or {}).get(demo.name) or {}
        # batched replay 2026-11-15: the caller's own TRO / full template for the demo (the hdf5's recorded tro_path can name
        # another instance of the same collector job; eval_production's tro.json is the instance's own)
        if row.get("tro_override"):
            demo.tro_path = row["tro_override"]
        if row.get("full_scene_override"):
            demo.full_scene_file = row["full_scene_override"]
        if args.settle_pad is not None and demo.settle is None:
            demo.pad_settle(args.settle_pad, args.settle_pad_min)
            print("REPLAY_SETTLE " + json.dumps({"demo": demo.name, **demo.settle}), flush=True)
        return demo

    usable(first)

    import omnigibson as og
    from omnigibson.macros import gm

    # replay-time lighting: camera relocalization keeps its collection lighting; a run must not mix it with the other classes
    from ispatialgym.multienv import replay_lighting as RL
    from ispatialgym.multienv.replay_scene_perturbation import _read_json as _rj, exempt as _exempt

    exempt_dirs = {d for d in demo_dirs if _exempt(_rj(os.path.join(d, "meta.json")))}
    lighting_mode = args.lighting_pert
    if lighting_mode == "auto":
        lighting_mode = "replay" if (args.campaign_manifest and len(exempt_dirs) < len(demo_dirs)) else "off"
    if args.lighting_pert_from:
        lighting_mode = "replay"
    if lighting_mode == "replay" and exempt_dirs:
        ap.error("replay-time lighting: an exempt family (collection lighting) renders in its own batch")

    from ispatialgym.multienv.replay_scene_perturbation import goal_group

    groups_of = {os.path.basename(os.path.normpath(d)): goal_group(d) for d in demo_dirs}
    # batched replay 2026-11-15: a goal group named by the caller (--instance-meta goal_group_id, e.g. eval_production's
    # <class>__<goal key>) keys the replay-time draws
    for name, row in imeta_early.items():
        if row.get("goal_group_id") and name in groups_of:
            groups_of[name] = str(row["goal_group_id"])
    # owner 2026-11-15 11:xx: --pert-unit demo (default) = an independent draw per demo (lighting coin + factor, skybox, scene coin +
    # object), keyed by (seed, demo); group = one shared draw per goal group (the 2026-11-15 morning render switch)
    draw_key_of = {n: (n if args.pert_unit == "demo" else g) for n, g in groups_of.items()}
    if args.lighting_skybox in ("group", "episode") and n_slots_req(args) != 1:
        ap.error("--lighting-skybox group needs --slots 1 (the skybox is one per simulator)")
    inview_plan = None
    if args.scene_pert_plan == "auto":
        # batched replay 2026-11-15: the render planner's offline in-view ranking (CPU, recorded data only), built here
        from ispatialgym.multienv.replay_inview import build_plan
        from ispatialgym.multienv.replay_scene_perturbation import STRUCTURAL, EXTRA_STRUCTURAL

        data = os.environ.get("OMNIGIBSON_DATA_PATH") or ""
        t_plan = time.time()
        inview_plan = build_plan(demo_dirs, {os.path.basename(os.path.normpath(d)): draw_key_of[os.path.basename(os.path.normpath(d))]
                                             for d in demo_dirs},
                                 [os.path.join(data, "behavior-1k-assets"), os.path.join(data, "ispatialgym-assets")],
                                 structural=sorted(STRUCTURAL | EXTRA_STRUCTURAL))
        print(f"REPLAY_INVIEW_PLAN demos={len(inview_plan['demos'])} cpu_s={inview_plan['cpu_s_total']} wall_s={time.time() - t_plan:.1f} "
              f"errors={sum(1 for r in inview_plan['demos'].values() if r.get('error'))}", flush=True)
        if args.results:
            os.makedirs(os.path.dirname(os.path.abspath(args.results)), exist_ok=True)
            with open(os.path.join(os.path.dirname(os.path.abspath(args.results)), "scene_pert_plan.json"), "w") as fh:
                json.dump(inview_plan, fh)
    elif args.scene_pert_plan:
        with open(args.scene_pert_plan) as fh:
            inview_plan = json.load(fh)
        if inview_plan.get("schema") != "isg-replay-inview-v1":
            ap.error("--scene-pert-plan: unknown schema")
    lo, hi = (float(x) for x in args.lighting_pert_range.split(","))
    light_cfg = {"dist": args.lighting_pert_dist, "p": args.lighting_pert_p, "sigma": args.lighting_pert_sigma, "lo": lo, "hi": hi,
                 "color_sigma": args.lighting_pert_color_sigma, "sky_sigma": args.lighting_pert_sigma, "sky_lo": lo, "sky_hi": hi}
    light_seed = args.scene_pert_seed if args.lighting_pert_seed is None else args.lighting_pert_seed
    light_from = RL.load_from_records(args.lighting_pert_from) if args.lighting_pert_from else None
    tro_removed = None
    if args.replay_tro_dir:
        tro_removed, tro_light = RL.load_replayed_tros(args.replay_tro_dir)
        if tro_light:
            light_from, lighting_mode = tro_light, "replay"
    sky_draw = None
    if lighting_mode == "replay":
        names = [os.path.basename(os.path.normpath(d)) for d in demo_dirs]
        if light_from is not None:
            mine = [r for (n, _), r in light_from.items() if n in set(names)]
            RL.check_batch_skybox(mine)
            if not mine:
                ap.error("--lighting-pert-from: no record for these demos")
            sky_draw = mine[0]["skybox"]
        else:
            sky_draw = RL.draw_skybox(light_seed, RL.batch_key([groups_of[n] for n in names]) + f"|r{args.scene_pert_index_base}", light_cfg)

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
    extent = None
    if scene == "full" and is_cropped(first.scene_crop):
        extent = merged_extent(first)
        print("REPLAY_SCENE_EXTENT " + json.dumps(extent), flush=True)
        if not extent["equal"]:
            raise SystemExit(f"--scene full: the merged scene's rooms {extent['loaded']} are not the task rooms + adjacent rooms "
                             f"{sorted(set(extent['task_rooms']) | set(extent['adjacent']))}")
    config = env_config(first, [first], first.dir, n_slots, scene=scene, image_size=args.image_size)
    env = og.Environment(configs=config)
    load_s = time.time() - t_start
    robot_name = env.scenes[0].robots[0].name
    sky_applied = None
    if lighting_mode == "replay":
        for sc in env.scenes:  # untouched fixture defaults of every slot, before any restoration
            RL.fixture_defaults(sc)
        RL.sky_defaults(og.sim)
        sky_applied = RL.apply_skybox(og.sim, sky_draw)
        print("REPLAY_LIGHTING_SKYBOX " + json.dumps({"draw": sky_draw, "applied": sky_applied}), flush=True)
    from ispatialgym.multienv.campaign_render_cleanup import processing_cleanup
    writers = []
    with processing_cleanup(writers):
        with individual_render_products(env):
            for k in range(n_slots):
                writers.append(BL.ShardWriter(make_slot_view(env, k), args.root, f"{args.repo_prefix}_slot{k}", args.task, async_flush=args.async_flush))
        slots = [Slot(k, env.scenes[k], writers[k], robot_name, scene_mode=scene) for k in range(n_slots)]
        replay_counts: Dict[str, int] = defaultdict(int)
        seg_decisions: Dict[str, Dict[str, Any]] = {}  # goal group -> its segmentation counts (computed once, before its first episode)
        seg_time = [0.0, 0]
        pert_from = None
        if args.scene_pert_from:
            from ispatialgym.multienv.replay_scene_perturbation import load_from_records

            pert_from = load_from_records(args.scene_pert_from)
        if tro_removed is not None:
            pert_from = tro_removed
        pert_path = os.path.join(args.root, "scene_perturbation.jsonl")
        episodes_path = os.path.join(args.root, "episodes_isg.jsonl")
        instance_meta = read_instance_meta(args.instance_meta)

        def start_slot(s: "Slot", demo: Demo) -> None:
            s.ignore_recorded_lighting = lighting_mode == "replay"
            s.start(demo)
            if args.scene_pert_verify:
                from ispatialgym.multienv.replay_scene_perturbation import _read_json

                meta = _read_json(os.path.join(demo.dir, "meta.json"))
                s.pert_check = {"tracked": meta.get("tracked_objects") or {}, "pos_m": 0.0, "ori_rad": 0.0, "frames": 0,
                                "recorded_success": meta.get("success")}
            if args.scene_pert == "off" and pert_from is None and lighting_mode != "replay":
                return
            from ispatialgym.multienv import replay_scene_perturbation as RSP

            RSP.clear(s.scene, s.hidden)
            s.hidden = []
            ri = args.scene_pert_index_base + replay_counts[demo.dir]
            replay_counts[demo.dir] += 1
            t_p = time.perf_counter()
            if args.scene_pert != "off" or pert_from is not None:
                forced = None
                if pert_from is not None:
                    forced = pert_from.get((demo.name, ri))
                    if forced is None:
                        raise SystemExit(f"--scene-pert-from: no record for {demo.name} replay {ri}")
                inside = None
                if args.scene_pert_scope == "outside_crop" and is_cropped(demo.scene_crop):
                    inside = set(demo.scene_file["objects_info"]["init_info"])
                if RSP.is_camera_relocalization(RSP._read_json(os.path.join(demo.dir, "meta.json"))) and forced is None:
                    # owner 2026-11-15: camera relocalization keeps its collection-time removals; no replay-time removal
                    rec = {"schema": RSP.SCHEMA, "demo": demo.dir, "demo_name": demo.name, "replay_index": ri,
                           "seed": args.scene_pert_seed, "removed": [], "perturbed": False, "camera": "collection-time removals kept"}
                else:
                    g = draw_key_of.get(demo.name, demo.name)
                    vis_px = None
                    if (args.scene_pert_visibility == "seg" and forced is None and args.scene_pert_mode == "one"
                            and RSP.group_gate(RSP.group_key(args.scene_pert_seed, g, args.scene_pert_index_base), args.scene_pert_p_episode)):
                        if g not in seg_decisions:
                            seg_decisions[g] = seg_visibility(s, demo, og.sim.render)
                            seg_time[0] += seg_decisions[g]["s"]
                            seg_time[1] += 1
                        vis_px = seg_decisions[g]["px"]
                    rec = RSP.plan(s.scene, demo, ri, args.scene_pert_seed,
                                   {"mode": args.scene_pert_mode, "p_episode": args.scene_pert_p_episode, "p_unit": args.scene_pert_p_unit,
                                    "prefer_pick": args.scene_pert_inview_pick},
                                   forced=forced, inside_crop=inside, group=draw_key_of.get(demo.name), round_=args.scene_pert_index_base,
                                   prefer=(inview_plan or {}).get("rankings", {}).get(draw_key_of.get(demo.name)),
                                   visible_px=vis_px, vis_threshold=args.scene_pert_vis_threshold)
                    if vis_px is not None and rec.get("visibility") is not None:
                        rec["visibility"].update({k: seg_decisions[g][k] for k in ("poses", "renders", "s")})
                rec["scope"], rec["scene_mode"] = args.scene_pert_scope, s.scene_mode
                s.hidden = RSP.apply(s.scene, rec["removed"])
            else:
                rec = {"schema": RSP.SCHEMA, "demo": demo.dir, "demo_name": demo.name, "replay_index": ri, "seed": args.scene_pert_seed,
                       "removed": [], "perturbed": False, "scene_pert": "off"}
            if lighting_mode == "replay":
                if light_from is not None:
                    lrec = light_from.get((demo.name, ri))
                    if lrec is None:
                        raise SystemExit(f"--lighting-pert-from: no record for {demo.name} replay {ri}")
                    fdraw = lrec["fixtures_draw"]
                    RL.check_batch_skybox([lrec, {"skybox": sky_draw}])
                else:
                    fdraw = RL.draw_fixtures(light_seed, draw_key_of.get(demo.name, demo.name), args.scene_pert_index_base, light_cfg)
                fixtures = RL.apply_fixtures(s.scene, og.sim, fdraw)
                sky_ep = sky_draw
                if args.lighting_skybox in ("group", "episode") and light_from is None:
                    # one slot: the simulator's one skybox belongs to this episode, drawn with the episode's draw key (--pert-unit)
                    sky_ep = RL.draw_skybox(light_seed, f"episode:{draw_key_of.get(demo.name, demo.name)}|r{args.scene_pert_index_base}", light_cfg)
                elif abs(float(og.sim.skybox.intensity) - sky_applied["intensity"]) > 1e-5 * max(1.0, sky_applied["intensity"]):
                    RL.apply_skybox(og.sim, sky_draw)  # a restoration reset it: the batch keeps one skybox
                rec["lighting"] = {"schema": RL.SCHEMA, "mode": "replay", "seed": light_seed, "cfg": light_cfg,
                                   "fixtures_draw": fdraw, "fixtures_applied": fixtures, "skybox": sky_ep,
                                   "skybox_scope": args.lighting_skybox,
                                   "skybox_applied": RL.apply_skybox(og.sim, sky_ep),
                                   "recorded_ignored": RL.ignored_recorded(getattr(demo, "campaign_meta", None)),
                                   "forced": light_from is not None}
            s.scene.load_state(demo.state_at(0), serialized=True)
            rec["group"] = groups_of.get(demo.name, demo.name)
            rec["draw_key"], rec["pert_unit"] = draw_key_of.get(demo.name, demo.name), args.pert_unit
            rec.setdefault("round", args.scene_pert_index_base)
            rec["slot"], rec["plan_s"] = s.k, round(time.perf_counter() - t_p, 2)
            s.scene_pert = rec
            print("REPLAY_SCENE_PERT " + json.dumps(rec), flush=True)
            if args.results:
                with open(args.results, "a") as fo:
                    fo.write(json.dumps({"tag": "REPLAY_SCENE_PERT", **rec}) + "\n")

        def pert_visibility(tag: str) -> None:
            if args.scene_pert == "off" and pert_from is None:
                return
            from ispatialgym.multienv.replay_scene_perturbation import visibility_matrix

            vm = visibility_matrix([s.scene for s in slots], [s.hidden for s in slots])
            vm["when"] = tag
            print("REPLAY_SCENE_PERT_VISIBILITY " + json.dumps(vm), flush=True)
            if args.results:
                with open(args.results, "a") as fo:
                    fo.write(json.dumps({"tag": "REPLAY_SCENE_PERT_VISIBILITY", **vm}) + "\n")

        start_slot(slots[0], first)
        n_objects_recorded_first = slots[0].n_objects_recorded  # REPLAY_SUMMARY (PARTIAL_SCENE): the first demo's recorded objects
        n_demos = 1

        def fill_idle(force: bool = False) -> None:
            nonlocal n_demos
            started: List[int] = []
            try:
                _fill(force, started)
            finally:
                if started:
                    pert_visibility(f"start slots {started}")

        def _fill(force: bool, started: List[int]) -> None:
            nonlocal n_demos
            for s in slots:
                if s.demo is None:
                    d = source.next(force)
                    if d is None:
                        return
                    start_slot(s, usable(d))
                    n_demos += 1
                    started.append(s.k)

        fill_idle(force=True)
        if args.lighting_sweep:
            from PIL import Image
            from ispatialgym.oracle.r1pro_model import HEAD_CAMERA_SENSOR_SUFFIX

            os.makedirs(args.lighting_sweep_dir, exist_ok=True)
            rows = []
            for f in [float(x) for x in args.lighting_sweep.split(",")]:
                d0 = {"intensity_factor": f, "color_factors": [1.0, 1.0, 1.0]}
                RL.apply_skybox(og.sim, d0)
                for s in slots:
                    if s.demo is not None:
                        RL.apply_fixtures(s.scene, og.sim, d0)
                        s.scene.load_state(s.demo.state_at(min(s.demo.T - 1, int(args.lighting_sweep_at * (s.demo.T - 1)))), serialized=True)
                for _ in range(40):
                    og.sim.render()
                for s in slots:
                    if s.demo is None:
                        continue
                    robot = s.scene.robots[0]
                    key = next(k for k in robot.sensors if k.endswith(HEAD_CAMERA_SENSOR_SUFFIX))
                    obs = robot.sensors[key].get_obs()
                    obs = obs[0] if isinstance(obs, tuple) else obs
                    a = obs["rgb"].detach().cpu().numpy()[..., :3].astype(np.uint8)
                    Image.fromarray(a).save(os.path.join(args.lighting_sweep_dir, f"sweep_slot{s.k}_{s.demo.name[-14:]}_x{f:.2f}.png"))
                    rows.append({"slot": s.k, "demo": s.demo.name, "factor": f, **RL.sweep_stats(a)})
            print("REPLAY_LIGHTING_SWEEP " + json.dumps(rows), flush=True)
            if args.results:
                with open(args.results, "a") as fo:
                    fo.write(json.dumps({"tag": "REPLAY_LIGHTING_SWEEP", "rows": rows}) + "\n")
            return 0
        if args.scene_pert_probe:
            from ispatialgym.multienv.replay_scene_perturbation import probe_views

            for s in slots:
                if s.demo is not None:
                    s.scene.load_state(s.demo.state_at(0), serialized=True)
            probe = probe_views([s.scene for s in slots], [s.hidden for s in slots], og.sim.render, args.scene_pert_probe)
            print("REPLAY_SCENE_PERT_PROBE " + json.dumps(probe), flush=True)
            if args.results:
                with open(args.results, "a") as fo:
                    fo.write(json.dumps({"tag": "REPLAY_SCENE_PERT_PROBE", **probe}) + "\n")
            if args.scene_pert_probe_only:
                return 0
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
            if args.scene_pert_verify:
                from ispatialgym.multienv.replay_scene_perturbation import pose_errors, trajectory_update

                for s in active:
                    pc = s.pert_check
                    e = pose_errors(s.scene, s.demo, s.i, pc.get("tracked") or {})
                    pc["pos_m"], pc["ori_rad"] = max(pc.get("pos_m", 0.0), e["pos_m"]), max(pc.get("ori_rad", 0.0), e["ori_rad"])
                    pc["frames"] = pc.get("frames", 0) + 1
                    pc["n_tracked"] = e["n"]
                    trajectory_update(pc, s.scene, pc.get("tracked") or {})
                    if s.i == s.demo.T - 1:
                        pc["final_pos_m"], pc["final_ori_rad"] = e["pos_m"], e["ori_rad"]
                        pc["traj_sha256"] = pc.pop("_traj").hexdigest()
                        cur = s.scene.dump_state(serialized=True)
                        cur = cur if isinstance(cur, th.Tensor) else th.as_tensor(cur)
                        pc["final_scene_state_sha256"] = hashlib.sha256(cur.detach().cpu().double().numpy().tobytes()).hexdigest()
                        pc["final_scene_state_len"] = int(cur.numel())
            t2 = time.perf_counter()
            for _ in range(args.first_renders if any(s.i == 0 for s in active) else args.renders):
                og.sim.render()
            t3 = time.perf_counter()
            if diagnostic_fault_pending and args.diagnostic_fault_boundary == "post_render" and any(s.i == 1 for s in active):
                from ispatialgym.multienv.campaign_render_cleanup import abort_after_render
                assert len(active) == 1
                s = active[0]
                error, cleanup = abort_after_render(s.writer, s.i, diagnostic_fault_pending)
                emit("REPLAY_EPISODE_ABORTED", {"demo": s.demo.dir, "slot": s.k, "kept": False,
                    "campaign_restoration": s.campaign_restoration, "cleanup": cleanup,
                    "wall_s": round(time.time()-s.t0, 2)})
                diagnostic_fault_pending = None
                s.demo, s.loaded, s.last_obs = None, None, None
                fill_idle()
                continue
            obs_list, _ = env.get_obs()
            t4 = time.perf_counter()
            for s in active:
                d, i = s.demo, s.i
                # tiled camera buffers are persistent CUDA tensors (per-slot views), and the writer converts frames with .numpy()
                obs = {key: ((v.detach().cpu() if v.is_cuda else v.clone()) if isinstance(v, th.Tensor) else v) for key, v in obs_list[s.k].items()}
                if args.proprio == "recorded":
                    obs[s.proprio_key] = th.as_tensor(d.proprio[i], dtype=th.float32)
                if args.scene_pert_frames:
                    at = {min(d.T - 1, int(round(float(x) * (d.T - 1)))) for x in args.scene_pert_frame_at.split(",") if x}
                    if i in at:
                        save_frames(args.scene_pert_frames, s, d, i, obs)
                from ispatialgym.multienv.campaign_render_cleanup import write_frame
                error, cleanup = write_frame(s.writer, d, i, obs, fault=diagnostic_fault_pending if args.diagnostic_fault_boundary == "writer" else None)
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
                                         "scene_pert_check": {k: v for k, v in s.pert_check.items() if not k.startswith("_")} or None,
                                         "scene_perturbation": None if s.scene_pert is None else {
                                             "removed": s.scene_pert["removed"], "replay_index": s.scene_pert["replay_index"],
                                             "seed": s.scene_pert["seed"], "perturbed": s.scene_pert["perturbed"],
                                             "exempt": s.scene_pert.get("exempt"),
                                             "lighting": (s.scene_pert.get("lighting") or {}).get("fixtures_draw")},
                                         "goal_occlusion": None if occl is None else {"max_occluded": occl.get("max_occluded"), "worst": occl.get("worst"),
                                                                                      "occluded": occl.get("occluded")}})
                    goal = None
                    if args.goal_images == "on" and keep and index is not None:
                        dataset = f"{args.repo_prefix}_slot{s.k}"
                        rel = os.path.join("goal_images", dataset, f"episode_{int(index):06d}.png")
                        goal = save_goal_image(s, d, obs, os.path.join(args.root, rel), rel=rel)
                    if s.scene_pert is not None and d.tro_path and RSP_camera(d):
                        s.scene_pert["goal_observation"] = replay_goal_image(s, d, obs, os.path.join(args.root, "instances"))
                    if s.scene_pert is not None and (s.scene_pert.get("removed") or s.scene_pert.get("lighting")) and d.tro_path:
                        s.scene_pert["replayed_tro"] = RL.write_replayed_tro(d.tro_path, os.path.join(args.root, "instances"), s.scene_pert)
                    sp = s.scene_pert or {}
                    lt = sp.get("lighting") or {}
                    ep = {"schema": "isg-batched-replay-episode-v1", "dataset": f"{args.repo_prefix}_slot{s.k}", "episode_index": index,
                          "kept": keep, "task": args.task, "length": d.T, "length_recorded": d.T_recorded, "settle": d.settle,
                          "image_size": args.image_size, "scene_mode": s.scene_mode, "goal_image": goal,
                          "camera_goal_observation": sp.get("goal_observation"), "replayed_tro": sp.get("replayed_tro"),
                          "perturbation": {"scene": args.scene_pert, "lighting": lighting_mode,
                                           "scene_perturbed": bool(sp.get("perturbed")), "removed": sp.get("removed") or [],
                                           "camera_collection_removals": sp.get("camera"),
                                           "lighting_perturbed": bool(((lt.get("fixtures_draw") or {}).get("perturbed"))),
                                           "fixtures_draw": lt.get("fixtures_draw"), "skybox": lt.get("skybox"),
                                           "group": sp.get("group"), "replay_index": sp.get("replay_index")},
                          "instance": instance_link(d, instance_meta.get(d.name))}
                    os.makedirs(args.root, exist_ok=True)
                    with open(episodes_path, "a") as fo:
                        fo.write(json.dumps(ep) + "\n")
                    if s.scene_pert is not None:
                        os.makedirs(args.root, exist_ok=True)
                        with open(pert_path, "a") as fo:
                            fo.write(json.dumps({"dataset": f"{args.repo_prefix}_slot{s.k}", "episode_index": index, "kept": keep,
                                                 "check": {k: v for k, v in s.pert_check.items() if not k.startswith("_")},
                                                 **s.scene_pert}) + "\n")
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
            "scene": scene, "goal_occlusion": {"mode": args.goal_occlusion, **n_occl},
            "image_size": args.image_size, "settle_pad": args.settle_pad, "settle_pad_min": args.settle_pad_min,
            "lighting_pert": lighting_mode, "scene_pert": args.scene_pert,
            "scene_pert_seg": {"groups": seg_time[1], "s": round(seg_time[0], 3)}, "n_objects_scene": slots[0].n_objects_scene, "n_objects_recorded": n_objects_recorded_first,
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
