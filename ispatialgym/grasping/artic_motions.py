"""The oracle's small-articulation motions per alias (i7-artic 2026-11-07, owner decision: handles stay with the ARTICULATION library) -
what ``validate_artic motions`` validates in physics and the articulation prefix maps (``reach_maps`` kind ``articulation_prefix``) are
computed from.

A motion is built with the SAMPLER'S OWN code (``sampling.v2.tasks.articulation_small``: ``_load_handle``'s handle choice, ``_facing_local``,
the front-end shift of ``sample_template``, ``_build_v2_handles``, ``_build_close_push_handle``; the methods run on a small stand-in
object, :class:`_Shim`, that carries the attributes they read) from the alias's customs entry, the handle annotation and the object's
local hull, so the handle poses, the joint values and the face yaw are the ones the sampler uses for that alias:

* ``pull_open`` (open, no prep push; ``articulation_open_small#01`` blender eyedvd): the annotated bar / knob (else the fallback) pinched at
  q_init = the lower limit, pulled open along the hinge arc (``BaseFollow`` arm mode); stage 2's fist push is not part of it;
* ``v2_ajar_pull`` (open, ``prep_push`` ``two_flap_v2`` / ``two_flap_v4``; ``#02`` carton cdmmwy): the MAIN pinch (handle B: every
  ``handle_variants`` site / yaw / pitch the exact IK tries, else ``handle``) at q_ajar (``v2_ajar_rad``), pulled open from there; the
  first opening (v2's other hand, v4's lever fist) is not part of it;
* ``push_close`` (close, ``close_push``; ``articulation_close_small#01`` / ``#02``): the closed fist on the moving link's tip
  (``_build_close_push_handle``) at q_init (``close_init_rad`` / ``close_init_fraction``), pushed along the closing arc to the lower limit
  plus ``close_push_overshoot_rad`` (the first segment from the manipulation stance; the re-stanced segments and the wide-stop brace are not
  part of it).

Every pinch is a candidate with both finger assignments (``_pinch_variants``: the hand rolled pi about the approach; the fist's two turns
for the push). ``handle_signature`` digests the base handle poses and the joint geometry the motion was built from: the map's identity
carries it and the sampler's lookup recomputes it from its own handles, so a changed generation parameter, annotation or facing never
reads a map of other handle poses."""

from __future__ import annotations

import copy
import json
import math
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ispatialgym import REPO_ROOT
from ispatialgym.grasping import schema as S

MOTIONS = ("pull_open", "v2_ajar_pull", "push_close")
CUSTOMS = os.path.join(REPO_ROOT, "ispatialgym", "assets", "task_specs", "isg_customs.json")
#: the joint value / handle vector rounding of :func:`handle_signature`
SIG_DECIMALS = 4
#: the oracle's pre-shape margin of a handle pinch (``MotionsConfig.gripper_preshape_margin``; the sampler's ``grasp_fingers``)
PRESHAPE_MARGIN_M = 0.010
#: the pads' closed opening on a pinch of width w: w / 2 + this (``validate_artic`` / ``reach_maps.grasp_set`` contact_q)
CONTACT_EXTRA_M = 0.0009


def load_alias(alias: str, customs_path: Optional[str] = None) -> Dict[str, Any]:
    with open(customs_path or CUSTOMS) as f:
        d = json.load(f)
    if alias not in d:
        raise KeyError(f"{alias!r} is not a customs entry of {customs_path or CUSTOMS}")
    return d[alias]


def direction_of(entry: Dict[str, Any]) -> str:
    g = entry.get("generation") or {}
    if g.get("direction") in ("open", "close"):
        return str(g["direction"])
    return "close" if "close" in str(entry.get("iddl", "")) else "open"


def motion_kind(gen: Dict[str, Any], direction: str) -> Optional[str]:
    """The motion of an alias's generation block (the sampler's branches): None for a close without ``close_push`` (the free-hinge close
    of a stable open lid - no prefix motion of it is validated)."""
    if direction == "close":
        return "push_close" if gen.get("close_push") else None
    if gen.get("prep_push") in ("two_flap_v2", "two_flap_v4"):
        return "v2_ajar_pull"
    return "pull_open"


def defaults_key(direction: str) -> str:
    return "configuration_matching-articulation_open_small_scale" if direction == "open" else "configuration_matching-articulation_close_small_scale"


def target_model(entry: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """(category, model, joint, handle key) of the alias's target (the representative; ``roles.target.joint`` / ``handle_key``)."""
    r = (entry.get("roles") or {}).get("target") or {}
    cat, model = r.get("category"), r.get("representative")
    if not cat or not model:
        raise ValueError("roles.target needs category + representative")
    key = f"{cat}/{model}"
    joint = r.get("joint")
    joint = joint.get(key) if isinstance(joint, dict) else joint
    hkey = r.get("handle_key")
    hkey = hkey.get(key) if isinstance(hkey, dict) else hkey
    return cat, model, joint, hkey or f"{cat}__{model}"


def target_scale(entry: Dict[str, Any]) -> Optional[List[float]]:
    """Resolve the representative's explicit scale with the sampler's own rules.

    Missing scales retain the nominal validation path. An explicit scale must
    also key the loaded geometry and motion library; scaling only the handle
    annotation would validate a different physical object.
    """
    from ispatialgym.sampling.v2.problem import resolve_scale

    cat, model, _, _ = target_model(entry)
    scale = resolve_scale(entry["roles"]["target"].get("scale"), cat, model)
    if scale is not None and (len(scale) != 3 or any(not math.isfinite(v) or v <= 0 for v in scale)):
        raise ValueError(f"invalid articulation target scale: {scale!r}")
    return scale


class _Shim:
    """The attributes ``ArticulationSmall._facing_local`` / ``_build_v2_handles`` / ``_build_close_push_handle`` / ``_pinch_variants``
    read (``gen``, ``g``, ``jinfo``, ``handle``, ``hull``, ``direction``, ``facing_local``, ``q_init`` / ``q_goal``, ``gp.variant``)."""

    def __init__(self, alias: str, gen: Dict[str, Any], defaults: Dict[str, Any], direction: str):
        self.gp = SimpleNamespace(variant=alias)
        self._gen, self.defaults, self.direction = dict(gen), dict(defaults), direction
        self.jinfo: Dict[str, Any] = {}
        self.handle: Dict[str, Any] = {}
        self.hull = None
        self.q_init: Optional[float] = None
        self.q_goal: Optional[float] = None

    def gen(self, key: str, default=None):
        return self._gen.get(key, default)

    def g(self, key: str):
        return self.gen(key, self.defaults.get(key))


def _jsonable(o):
    import torch as th

    if isinstance(o, th.Tensor):
        return [float(v) for v in o.reshape(-1).tolist()] if o.numel() > 1 else float(o)
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def _r(v) -> List[float]:
    return [round(float(x), SIG_DECIMALS) for x in v]


def handle_signature(handles: Sequence[Dict[str, Any]], jinfo: Dict[str, Any]) -> str:
    """Digest of the base handle poses (object frame, closed configuration: contact / approach / pinch, rounded) and the joint geometry
    ``conventions.handle_waypoints`` turns them with (axis, hinge, closed q, type) - the sampler recomputes it from its own handles."""
    rows = [{"c": _r(h["contact"]), "a": _r(h.get("approach_dir", h.get("approach_dir_obj"))), "p": _r(h["pinch_axis"])} for h in handles]
    j = {"axis": _r(jinfo["axis_dir_obj"]), "hinge": _r(jinfo.get("hinge_point_obj", [0.0, 0.0, 0.0])),
         "closed_q": round(float(jinfo.get("closed_q", jinfo.get("lower", 0.0))), SIG_DECIMALS), "type": str(jinfo.get("type", ""))}
    return S.digest({"handles": rows, "joint": j}, 16)


def face_yaw_offset(facing_local) -> float:
    """The object's yaw relative to a robot facing the edge (``articulation_small._brace_band``: face_yaw - base yaw = pi - atan2(f))."""
    f = [float(v) for v in facing_local]
    a = math.pi - math.atan2(f[1], f[0])
    return math.atan2(math.sin(a), math.cos(a))


def needed_travel(q_values: Sequence[float], q_start: float) -> float:
    """The joint travel from ``q_start`` a band of ``q_values`` requires (``articulation_small._band_q_values``)."""
    return max([0.0] + [abs(float(q) - float(q_start)) for q in q_values])


def base_handles(s) -> List[Dict[str, Any]]:
    """The handles a sampler (or the shim) grasps / pushes for its motion: v2's ``handle_variants`` (else ``handle``), else ``handle``."""
    if getattr(s, "v2", False):
        return list(getattr(s, "handle_variants", None) or [s.handle])
    if push_base_sig_on() and getattr(s, "close_push", False) and getattr(s, "_close_push_handle0", None) is not None:
        # i7-artsmall: the push-close fist's BASE turn (the map was validated from it, both turns as variants) - an attempt that ran the
        # flow with the twin turn leaves self.handle on the twin, and under START_DRAW the next attempt's prefix lookup read the twin's
        # signature: missing_map on every later attempt (as_off / as_on close_01 361000: 45 of 46 attempts)
        return [s._close_push_handle0]
    return [s.handle]


def push_base_sig_on() -> bool:
    """i7-artsmall (2026-11-10, a sampler-only bug fix, default ON since lead #80c; "0" off): ``ISPATIALGYM_ARTIC_PUSH_BASE_SIG`` - the
    push-close prefix map lookup signs the fist's base turn (:func:`base_handles`), not the turn the previous attempt left in ``handle``."""
    return os.environ.get("ISPATIALGYM_ARTIC_PUSH_BASE_SIG", "1") != "0"


def build(alias: str, entry: Dict[str, Any], handle_doc: Dict[str, Any], hull, joint_limits: Tuple[float, float], scale=(1.0, 1.0, 1.0)) -> Dict[str, Any]:
    """The alias's motion (module docstring) from its customs ``entry``, the handle annotation ``handle_doc`` (``artic/handles/<key>.json``),
    the object's local hull ``hull`` ((N, 3) tensor, root frame, joints closed: ``sim_utils.World.local_hull``) and the target joint's
    simulator limits ``joint_limits`` - in ``ArticulationSmall.sample_template``'s order."""
    import torch as th

    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling import generation_defaults as GD
    from ispatialgym.sampling.tasks.articulation import scale_handle, scale_joint_info
    from ispatialgym.sampling.v2.tasks.articulation_small import ArticulationSmall as AS

    gen = dict(entry.get("generation") or {})
    direction = direction_of(entry)
    motion = motion_kind(gen, direction)
    if motion is None:
        raise ValueError(f"{alias}: a {direction} without close_push has no prefix motion (the free-hinge close of a stable open lid)")
    cat, model, joint_name, hkey = target_model(entry)
    sh = _Shim(alias, gen, GD.defaults_for(defaults_key(direction)), direction)
    # _load_handle
    joints = [j for j in handle_doc["joints"] if joint_name is None or j["joint"] == joint_name]
    if not joints:
        raise ValueError(f"{alias}: joint {joint_name!r} not in the annotation of {hkey}")
    j = joints[0]
    hs = [h for h in j.get("handles", []) if h.get("kind") in ("bar", "knob")] or ([j["fallback"]] if j.get("fallback") else [])
    if not hs:
        raise ValueError(f"{alias}: no graspable handle for {hkey}/{j['joint']}")
    sc = th.as_tensor([float(v) for v in S.canonical_scale(scale)], dtype=th.float32)
    sh.handle, sh.jinfo = scale_handle(hs[0], sc), scale_joint_info(j, sc)
    all_jinfo = {jj["joint"]: scale_joint_info(jj, sc) for jj in handle_doc["joints"]}
    sh.jinfo.setdefault("closed_q", sh.jinfo.get("lower", 0.0))
    w = sh.handle.get("width")
    sh.hull = th.as_tensor(hull, dtype=th.float32)
    facing = AS._facing_local(sh)
    sh.facing_local = facing
    if sh.handle.get("front_end_along_hinge"):
        # sample_template's front-end shift (the same formula)
        axis = C.unit(th.tensor(sh.jinfo["axis_dir_obj"], dtype=th.float32))
        side = 1.0 if float(th.dot(axis, facing)) >= 0 else -1.0
        hinge = th.tensor(sh.jinfo["hinge_point_obj"], dtype=th.float32)
        c = th.tensor(sh.handle["contact"], dtype=th.float32)
        ax = float(th.dot(c - hinge, axis))
        c = c + axis * (side * abs(ax) - ax)
        a = C.unit(th.tensor(sh.handle["approach_dir"], dtype=th.float32))
        aa = float(th.dot(a, axis))
        a = C.unit(a + axis * (-side * abs(aa) - aa))
        sh.handle = dict(sh.handle, contact=[float(v) for v in c.tolist()], approach_dir=[float(v) for v in a.tolist()])
    sh.v4 = gen.get("prep_push") == "two_flap_v4"
    sh.v2 = gen.get("prep_push") == "two_flap_v2" or sh.v4
    sh.q_ajar = None
    if sh.v2:
        AS._build_v2_handles(sh)
    # the arm (sample_template): the handle's side of the hull centre, v2's main arm
    hc = th.tensor(sh.handle["contact"], dtype=th.float32)
    side_axis = th.tensor([float(facing[1]), -float(facing[0]), 0.0])
    lat = float(th.dot(hc - sh.hull.mean(dim=0), side_axis))
    arm_cfg = sh.g("arm")
    handle_arm = arm_cfg if arm_cfg in ("left", "right") else ("left" if lat > float(sh.g("handle_arm_side_threshold_m")) else "right")
    if sh.v2:
        handle_arm = str(sh.gen("v2_main_arm", "right"))
    pinned = arm_cfg in ("left", "right") or sh.v2
    arms = [handle_arm] if pinned else [handle_arm, "left" if handle_arm == "right" else "right"]
    lower, upper = float(joint_limits[0]), float(joint_limits[1])
    sh.jinfo["lower"], sh.jinfo["upper"] = lower, upper
    sh.jinfo.setdefault("closed_q", lower)
    out: Dict[str, Any] = {"alias": alias, "category": cat, "model": model, "scale": list(S.canonical_scale(scale)), "handle_key": hkey,
                           "joint": j["joint"], "joint_type": "prismatic" if "prismatic" in str(sh.jinfo.get("type", "")).lower() else "revolute",
                           "direction": direction, "motion": motion, "lower": lower, "upper": upper,
                           "facing_local": [round(float(v), 6) for v in facing.tolist()], "face_yaw_offset": round(face_yaw_offset(facing), 6),
                           "handle_arm": handle_arm, "arms_offered": arms, "handle_lateral_m": round(lat, 4)}
    if motion == "push_close":
        q0 = sh.gen("close_init_rad", None)
        q_init = float(q0) if q0 is not None else float(lower + float(sh.gen("close_init_fraction", 0.6)) * (upper - lower))
        q_init = float(min(max(q_init, lower + 0.05), upper - 1e-3))
        sh.q_init, sh.q_goal = q_init, lower
        AS._build_close_push_handle(sh)
        out.update({"q_start": q_init, "q_goal": lower, "overshoot_rad": float(sh.gen("close_push_overshoot_rad", 0.15)),
                    "goal_tolerance_rad": float(sh.gen("close_tolerance_rad", (sh.g("tolerances") or {}).get("revolute", 0.05)))})
        out["q_arc_end"] = lower - out["overshoot_rad"]
        pre_q = con_q = 0.0  # the closed fist (C.FINGER_CLOSED)
    elif motion == "v2_ajar_pull":
        out.update({"q_start": float(sh.q_ajar), "q_goal": None, "q_arc_end": upper})
        wv = float(w or 0.0)
        pre_q = float(min(C.FINGER_OPEN, 0.5 * wv + float(sh.gen("v2_second_preshape_margin_m", 0.02))))
        con_q = 0.5 * wv + CONTACT_EXTRA_M
    else:
        out.update({"q_start": lower, "q_goal": None, "q_arc_end": upper})
        wv = float(w or 0.0)
        pre_q = float(min(C.FINGER_OPEN, 0.5 * wv + PRESHAPE_MARGIN_M))
        con_q = 0.5 * wv + CONTACT_EXTRA_M
    bases = base_handles(sh)
    out["handle_sig"] = handle_signature(bases, sh.jinfo)
    out["base_handles"] = _jsonable(bases)
    variants = []
    for i, hb in enumerate(bases):
        tag = str(hb.get("site") or ("tip" if motion == "push_close" else "h"))
        stem = f"{tag}#{i}" + (f"_y{float(hb['yaw_deg']):g}" if hb.get("yaw_deg") is not None else "") + (f"_p{float(hb['pitch_deg']):g}" if hb.get("pitch_deg") is not None else "")
        for k, hv in enumerate(AS._pinch_variants(sh, hb)):
            variants.append({"name": f"{stem}{'+' if k == 0 else '-'}", "base_index": i, "assignment": "+pinch" if k == 0 else "-pinch",
                             "handle": _jsonable(hv), "preshape_q": round(pre_q, 5), "contact_q": round(con_q, 5), "width_m": w})
    out["variants"] = variants
    out["jinfo"] = _jsonable({k: v for k, v in sh.jinfo.items() if k not in ("handles", "flush_handles", "fallback")})
    out["all_jinfo"] = {k: _jsonable({kk: vv for kk, vv in v.items() if kk not in ("handles", "flush_handles", "fallback")}) for k, v in all_jinfo.items()}
    out["profile_spec"] = {"joint_friction": copy.deepcopy(gen.get("joint_friction")), "joint_drives": copy.deepcopy(gen.get("joint_drives") or {}),
                           "open_fraction_range": list(sh.g("open_fraction_range")), "stable_set_fractions": int(sh.g("stable_set_fractions")),
                           "stable_hold_steps": int(sh.g("stable_hold_steps")), "stable_drift_tolerance_fraction": float(sh.g("stable_drift_tolerance_fraction")),
                           "stable_probe_tolerance": float((sh.g("stable_probe_tolerances") or {}).get(out["joint_type"], 0.1))}
    return out


def brake_candidates(joint_friction) -> List[Any]:
    """The sampler's ``joint_friction`` candidates in order (``sample_template``): a list, one value, or [None] (the stock joint)."""
    jf = joint_friction
    return list(jf) if isinstance(jf, (list, tuple)) else ([jf] if jf is not None else [None])
