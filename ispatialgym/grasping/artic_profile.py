"""Articulation physics profiles: the joint physics demo collection applies, resolved per (model, scale, joint) and cached.

Demo collection (the sampler writes ``constraints`` records, ``run_isg`` re-applies them; ``sampling/constraints.py``):

* small lids / flaps of free-standing tabletop bodies: a torque-limited hinge BRAKE - ``joint_brake_spec(max_effort)`` = zero stiffness,
  damping ``max_effort / JOINT_BRAKE_OMEGA_RAD_S`` (0.0025 rad/s), the force capped at ``max_effort`` (``apply_joint_brake``); the effort
  comes from the alias's ``joint_friction`` ladder, the first value whose held angles are stable (``articulation_small``);
* horizontal-axis doors of large articulations (a dishwasher door): COUNTERBALANCE - gravity off on the moving leaf assembly, joint
  damping 5 s^-1 (``apply_counterbalance``, ``articulation_large`` ``counterbalance: auto``);
* configured two-flap cartons: the OTHER flap's drive (``generation.joint_drives``: stiffness 1, damping 0.15, max effort 5, lower limit
  -0.55 rad on ``j_left_lid``) - only where the alias configures it for that model and the joint exists;
* the assisted grasp on a moving link: a FixedJoint (the task's ``apply_environment_rules``).

:func:`resolve` returns a profile from, in order: the cache (``<library>/profiles/<entry>__<joint>.json``, valid while its key - model,
scale, joint, joint limits / axis, physics version, ladder - matches), the RECORDED demo settings of the customs entries that use this
model (``recorded``), else a bounded automatic calibration (:func:`calibrate_brake`, once per model / scale / joint - never per grasp
candidate) for small flaps, the counterbalance rule for large horizontal-axis leaves. Nothing carton-specific is ever applied to a model
whose own customs entry does not configure it.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ispatialgym import REPO_ROOT
from ispatialgym.grasping import schema as S

PHYSICS_VERSION = "isg-artic-physics/1"  # bump when constraints.py's brake / counterbalance semantics change
DEFAULT_LADDER = ("brake:0.15", "brake:0.3", "brake:0.5", "brake:1.0", "brake:2.0", "brake:4.0")
CUSTOMS = os.path.join(REPO_ROOT, "ispatialgym", "assets", "task_specs", "isg_customs.json")


def classify(joint: Dict[str, Any], fixed_base: bool, aabb_extent: Sequence[float]) -> str:
    """``small_flap`` (a free-standing tabletop body's revolute flap / lid), ``large_horizontal`` (a revolute leaf whose axis is
    horizontal on a fixed-base body - the dishwasher door), ``large_vertical`` (a door / cabinet leaf), ``prismatic``."""
    if "Prismatic" in str(joint.get("type", "")):
        return "prismatic"
    ax = np.asarray(joint.get("axis_dir_obj") or [0, 0, 1], dtype=float)
    horizontal = abs(float(ax[2])) < 0.3
    if not fixed_base and max(aabb_extent) < 0.8:
        return "small_flap"
    return "large_horizontal" if horizontal else "large_vertical"


def recorded(category: str, model: str, joint: str, customs_path: str = CUSTOMS) -> Dict[str, Any]:
    """The demo settings of every customs entry whose target is ``category/model`` with ``joint``: the ``joint_friction`` ladders,
    ``joint_drives`` (per OTHER joint), ``counterbalance`` / ``counterbalance_damping``. Entry keys are recorded as provenance."""
    if not os.path.exists(customs_path):
        return {}
    d = json.load(open(customs_path))
    out: Dict[str, Any] = {"entries": [], "joint_friction": [], "joint_drives": {}, "counterbalance": None}
    target = f"{category}/{model}"

    def mentions(o) -> bool:
        s = json.dumps(o)
        return target in s or f"{category}__{model}" in s

    for k, v in d.items():
        if not isinstance(v, dict) or not mentions(v):
            continue
        gen = v.get("generation") or {}
        hit = False
        jf = gen.get("joint_friction")
        if jf:
            out["joint_friction"].append(jf)
            hit = True
        jd = gen.get("joint_drives")
        if isinstance(jd, dict):
            for jn, spec in jd.items():
                if jn != joint:
                    out["joint_drives"].setdefault(jn, spec)
                    hit = True
        if "counterbalance" in gen:
            out["counterbalance"] = {"mode": gen.get("counterbalance"), "damping": gen.get("counterbalance_damping", 5.0)}
            hit = True
        if hit:
            out["entries"].append(k)
    return out if out["entries"] else {}


def profile_key(category: str, model: str, scale, joint: Dict[str, Any], ladder: Sequence[str]) -> str:
    return S.digest({"physics": PHYSICS_VERSION, "category": category, "model": model, "scale": list(S.canonical_scale(scale)),
                     "joint": {k: joint.get(k) for k in ("joint", "type", "axis_dir_obj", "hinge_point_obj", "lower", "upper")},
                     "ladder": list(ladder)}, 16)


def profile_path(root: Optional[str], entry_id: str, joint_name: str) -> str:
    return os.path.join(S.library_root(root), "profiles", f"{entry_id}__{joint_name}.json")


def resolve(category: str, model: str, scale, joint: Dict[str, Any], *, fixed_base: bool, aabb_extent: Sequence[float], root: Optional[str] = None,
            calibrate=None, ladder: Sequence[str] = DEFAULT_LADDER) -> Dict[str, Any]:
    """The profile (cached / recorded / calibrated). ``calibrate(ladder) -> (chosen spec, probe log)`` runs in the caller's world (the
    validator's :meth:`ArticWorld.calibrate_brake`) when a small flap has neither a cache entry nor recorded settings."""
    entry = S.entry_id(category, model, scale)
    key = profile_key(category, model, scale, joint, ladder)
    path = profile_path(root, entry, joint["joint"])
    cached = S.load_json(path)
    if cached and cached.get("key") == key and cached.get("schema") == S.PROFILE:
        cached["source_used"] = "cache"
        return cached
    kind = classify(joint, fixed_base, aabb_extent)
    rec = recorded(category, model, joint["joint"])
    prof: Dict[str, Any] = {"schema": S.PROFILE, "key": key, "physics_version": PHYSICS_VERSION, "category": category, "model": model,
                            "scale": list(S.canonical_scale(scale)), "joint": joint["joint"], "kind": kind, "ag_moving_link_joint": "FixedJoint",
                            "brake": None, "counterbalance": None, "other_joint_drives": {}, "provenance": {}, "created": S.now()}
    if kind == "small_flap":
        ladders = rec.get("joint_friction") or []
        if ladders:
            first = ladders[0]
            spec = first[0] if isinstance(first, (list, tuple)) else first
            prof["brake"] = {"spec": spec, "ladder": first if isinstance(first, (list, tuple)) else [first]}
            prof["provenance"]["brake"] = {"source": "recorded", "entries": rec["entries"],
                                           "note": "the alias's joint_friction ladder; the sampler takes its first stable value, calibrated per instance"}
        elif calibrate is not None:
            spec, probe = calibrate(ladder)
            prof["brake"] = {"spec": spec, "ladder": list(ladder)}
            prof["provenance"]["brake"] = {"source": "calibrated", "probe": probe}
        else:
            prof["provenance"]["brake"] = {"source": "none", "note": "no recorded setting and no calibration run"}
    elif kind == "large_horizontal":
        cb = rec.get("counterbalance") or {"mode": "auto", "damping": 5.0}
        prof["counterbalance"] = {"damping": float(cb.get("damping", 5.0)), "mode": cb.get("mode", "auto")}
        prof["provenance"]["counterbalance"] = {"source": "recorded" if rec.get("counterbalance") else "articulation_large default (auto, 5 s^-1)",
                                                "entries": rec.get("entries", [])}
    for jn, spec in (rec.get("joint_drives") or {}).items():
        prof["other_joint_drives"][jn] = spec  # only joints the model's own customs entries configure
    if prof["other_joint_drives"]:
        prof["provenance"]["other_joint_drives"] = {"source": "recorded", "entries": rec["entries"]}
    S.save_json(path, prof)
    prof["source_used"] = "new"
    return prof


def apply(obj, profile: Dict[str, Any], leaf_links: Sequence[str] = ()) -> Dict[str, Any]:
    """The profile on a loaded articulated object (``constraints``' functions, as the loader applies the records): brake on the
    target joint, counterbalance (gravity off on ``leaf_links``, damping), the other joints' drives. USD edits: apply before the rest
    snapshot (they survive ``load_state``)."""
    from ispatialgym.sampling import constraints as CN

    out: Dict[str, Any] = {}
    j = obj.joints[profile["joint"]]
    if profile.get("brake") and str(profile["brake"]["spec"]).startswith("brake:"):
        eff = float(str(profile["brake"]["spec"]).split(":", 1)[1])
        CN.apply_joint_brake(j, CN.joint_brake_spec(eff))
        out["brake"] = eff
    if profile.get("counterbalance"):
        CN.apply_counterbalance(obj, j, list(leaf_links), float(profile["counterbalance"]["damping"]))
        out["counterbalance"] = list(leaf_links)
    drives = profile.get("other_joint_drives") or {}
    for jn, spec in drives.items():
        if jn not in obj.joints:
            continue  # never a joint the model does not have
        out.setdefault("other_joint_drives", {})[jn] = CN.apply_joint_brake(obj.joints[jn], spec)  # as articulation_small applies joint_drives
    if any((spec or {}).get("lower_limit_rad") is not None for spec in drives.values()):
        CN.restrict_openable_joints(obj, [profile["joint"]])  # OmniGibson's Open state must not read the extended-limit flap as open
    return out
