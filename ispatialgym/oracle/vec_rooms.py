"""Room-loading probes of the vectorized evaluator (campaign rooms, 2026-10-28).

``FrameDumper``: lossless PNGs of the robot cameras (head / left wrist / right wrist, and the external camera) at chosen
steps of an episode, so two runs of the same demo under different room loadings can be compared pixel by pixel
(``run_isg_vec --dump-frames "0-60:5,25"``: steps 0..60 every 5, then every 25th).

``RoomVisibility``: per step, the instance segmentation of every robot camera (``seg_instance``: per-pixel object ids,
``info[cam]["seg_instance"]`` = {id: object name | 'background' | 'unlabelled' | 'groundPlane'}) is reduced to the set
of visible objects and their pixel counts; objects are mapped to the room instances they belong to (``obj.in_rooms``) and
the per-room / per-object exposure is written to ``rooms_visibility.json`` of the episode. Walls, ceilings, roofs, fences,
backgrounds and doors are loaded whatever ``load_room_instances`` says (OmniGibson ``_should_load_object``), so they are
reported but flagged ``always_loaded``; floors, windows, lights and furniture decide which rooms an episode needs.

The segmentation buffer trails the rendered frame by one to three frames (``sampling.sim_utils.SimWorld.seg_instance``);
over a whole episode that lag is irrelevant. Run with the sampler's crash guard (``/rtx/raytracing/showLights`` = 0:
light emitter shapes are not drawn, the illumination is unchanged) - the instance-segmentation node segfaults on light
shapes in view.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import torch as th

#: categories OmniGibson loads whatever the room filter says (InteractiveTraversableScene._should_load_object, fork 6a6194600)
ALWAYS_LOADED_CATEGORIES = frozenset({"walls", "ceilings", "roof", "fence", "background", "door", "sliding_door"})
NON_OBJECT_LABELS = frozenset({"background", "unlabelled", "groundPlane", "BACKGROUND", "UNLABELLED", "ground_plane"})


def parse_dump_spec(spec: Optional[str]) -> Optional["DumpSpec"]:
    """``"0-60:5,25"`` -> steps 0, 5, ..., 60 and every 25th step; ``"10"`` -> every 10th step; None / "" -> no dumps."""
    if not spec:
        return None
    ranges, every = [], []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo_hi, _, k = item.partition(":")
            lo, hi = lo_hi.split("-")
            ranges.append((int(lo), int(hi), int(k or 1)))
        else:
            every.append(int(item))
    return DumpSpec(ranges, every)


class DumpSpec:
    def __init__(self, ranges: Sequence[tuple], every: Sequence[int]):
        self.ranges, self.every = list(ranges), [int(k) for k in every if int(k) > 0]

    def wants(self, step: int) -> bool:
        for lo, hi, k in self.ranges:
            if lo <= step <= hi and (step - lo) % max(1, k) == 0:
                return True
        return any(step % k == 0 for k in self.every)

    def describe(self) -> str:
        return ",".join([f"{lo}-{hi}:{k}" for lo, hi, k in self.ranges] + [str(k) for k in self.every])


def _to_uint8(img) -> Optional[np.ndarray]:
    if img is None:
        return None
    t = th.as_tensor(img)
    if t.ndim == 3 and t.shape[-1] >= 3:
        t = t[..., :3]
    if t.dtype != th.uint8:
        t = (t.clamp(0, 1) * 255).round().to(th.uint8) if t.is_floating_point() else t.to(th.uint8)
    return t.detach().cpu().numpy()


class FrameDumper:
    """PNG dumps of ``frames`` (a dict camera -> uint8 HxWx3) at the steps of ``spec`` into ``<out_dir>/frames/``."""

    def __init__(self, out_dir: str, spec: DumpSpec):
        self.dir = os.path.join(out_dir, "frames")
        self.spec = spec
        self.n = 0
        os.makedirs(self.dir, exist_ok=True)

    def maybe(self, step: int, frames: Dict[str, Optional[np.ndarray]]) -> None:
        if not self.spec.wants(step):
            return
        from PIL import Image

        for cam, img in frames.items():
            if img is None:
                continue
            Image.fromarray(np.ascontiguousarray(img)).save(os.path.join(self.dir, f"step_{step:05d}_{cam}.png"), compress_level=1)
        self.n += 1


class RoomVisibility:
    """Accumulates which objects (and rooms) the robot cameras of one slot see over an episode."""

    def __init__(self, scene, robot_name: str, cam_keys: Dict[str, Optional[str]], every: int = 1, min_px: int = 1):
        self.scene, self.robot_name = scene, robot_name
        self.cam_keys = {k: v for k, v in cam_keys.items() if v}
        self.every, self.min_px = max(1, int(every)), int(min_px)
        self.objects: Dict[str, Dict[str, Dict[str, Any]]] = {cam: {} for cam in self.cam_keys}   # cam -> name -> stats
        self.rooms: Dict[str, Dict[str, Dict[str, Any]]] = {cam: {} for cam in self.cam_keys}     # cam -> room -> stats
        self.steps_sampled = 0
        self.pixels: Dict[str, int] = {}
        self._rooms_of: Dict[str, Optional[List[str]]] = {}
        self._category: Dict[str, Optional[str]] = {}
        self.unmapped: Dict[str, int] = {"ids": 0, "px": 0, "frames": 0}  # pixels whose id the info mapping did not name (diagnostic)
        self.mapping_keys_sample: List[Any] = []

    def _resolve(self, name: str) -> None:
        if name in self._rooms_of:
            return
        obj = None
        try:
            obj = self.scene.object_registry("name", name)
        except Exception:  # noqa: BLE001
            obj = None
        if obj is None:
            # the mapping may carry the prim path or a scene-prefixed name
            try:
                obj = self.scene.object_registry("prim_path", name)
            except Exception:  # noqa: BLE001
                obj = None
        if obj is None and ":" in name:
            try:
                obj = self.scene.object_registry("name", name.split(":")[-1])
            except Exception:  # noqa: BLE001
                obj = None
        if obj is None:
            self._rooms_of[name], self._category[name] = None, None
            return
        rooms = getattr(obj, "in_rooms", None)
        self._rooms_of[name] = list(rooms) if rooms else []
        self._category[name] = getattr(obj, "category", None)

    def observe(self, step: int, obs, info) -> None:
        if step % self.every:
            return
        robs = (obs or {}).get(self.robot_name) or {}
        # og.Environment.step / reset put the sensors' info under info["obs_info"][robot][sensor]; robot.get_obs() returns it flat
        info = info or {}
        rinfo = ((info.get("obs_info") or {}).get(self.robot_name)) or info.get(self.robot_name) or {}
        any_cam = False
        for cam, key in self.cam_keys.items():
            seg = (robs.get(key) or {}).get("seg_instance")
            mapping = (rinfo.get(key) or {}).get("seg_instance") or {}
            if seg is None:
                continue
            any_cam = True
            seg = th.as_tensor(seg)
            if cam not in self.pixels:
                self.pixels[cam] = int(seg.numel())
            ids, counts = th.unique(seg.reshape(-1), return_counts=True)
            seen_rooms: Dict[str, int] = {}
            if not self.mapping_keys_sample and mapping:
                self.mapping_keys_sample = [(k, mapping[k]) for k in list(mapping)[:5]]
            unm = 0
            for i, c in zip(ids.tolist(), counts.tolist()):
                name = mapping.get(int(i), mapping.get(str(i)))
                if name is None:
                    unm += int(c)
                    self.unmapped["ids"] += 1
                    continue
                if name in NON_OBJECT_LABELS or c < self.min_px:
                    continue
                name = str(name)
                if name == self.robot_name or name.startswith(self.robot_name):
                    continue
                st = self.objects[cam].get(name)
                if st is None:
                    self._resolve(name)
                    st = self.objects[cam][name] = {"frames": 0, "max_px": 0, "sum_px": 0, "first_step": int(step), "rooms": self._rooms_of.get(name),
                                                    "category": self._category.get(name),
                                                    "always_loaded": (self._category.get(name) in ALWAYS_LOADED_CATEGORIES)}
                st["frames"] += 1
                st["max_px"] = max(st["max_px"], int(c))
                st["sum_px"] += int(c)
                if not st["always_loaded"]:
                    for r in (self._rooms_of.get(name) or []):
                        seen_rooms[r] = seen_rooms.get(r, 0) + int(c)
            if unm:
                self.unmapped["px"] += unm
                self.unmapped["frames"] += 1
            for r, px in seen_rooms.items():
                rs = self.rooms[cam].setdefault(r, {"frames": 0, "max_px": 0, "sum_px": 0, "first_step": int(step)})
                rs["frames"] += 1
                rs["max_px"] = max(rs["max_px"], px)
                rs["sum_px"] += px
        if any_cam:
            self.steps_sampled += 1

    def summary(self) -> Dict[str, Any]:
        rooms_all: Dict[str, Dict[str, Any]] = {}
        for cam, rooms in self.rooms.items():
            for r, st in rooms.items():
                a = rooms_all.setdefault(r, {"frames": 0, "max_px": 0, "sum_px": 0, "max_frac": 0.0, "cams": []})
                a["frames"] = max(a["frames"], st["frames"])
                a["max_px"] = max(a["max_px"], st["max_px"])
                a["sum_px"] += st["sum_px"]
                a["max_frac"] = max(a["max_frac"], st["max_px"] / max(1, self.pixels.get(cam, 1)))
                a["cams"].append(cam)
        return {"steps_sampled": self.steps_sampled, "every": self.every, "pixels": self.pixels, "rooms": rooms_all, "unmapped": self.unmapped,
                "mapping_keys_sample": self.mapping_keys_sample,
                "rooms_by_camera": self.rooms, "objects_by_camera": self.objects}

    def write(self, out_dir: str) -> str:
        p = os.path.join(out_dir, "rooms_visibility.json")
        with open(p, "w") as f:
            json.dump(self.summary(), f, indent=1, default=str)
        return p


def resolve_rooms_override(path: Optional[str], alias: Optional[str]) -> Optional[List[str]]:
    """``--rooms-override <json>``: {alias: [room instances]} (or {alias: {"rooms": [...]}}); None when the alias has no entry."""
    if not path:
        return None
    with open(path) as f:
        d = json.load(f)
    d = d.get("overrides", d) if isinstance(d, dict) else {}
    e = d.get(alias) if alias else None
    if e is None:
        return None
    rooms = e.get("rooms") if isinstance(e, dict) else e
    return [str(r) for r in rooms] if rooms else None


__all__ = ["ALWAYS_LOADED_CATEGORIES", "DumpSpec", "FrameDumper", "RoomVisibility", "parse_dump_spec", "resolve_rooms_override"]
