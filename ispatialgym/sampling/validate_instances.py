"""Reload validation (spec §3.5 "Reload validation is a pipeline stage", §3.14 loader contract): every generated instance
is re-loaded in a fresh process from its TRO and re-checked the way the evaluator will see it.

Two artefact layouts are handled; the mode is detected per template directory (:func:`detect_layout`):

**v2** (``sample_isg`` / ``multiply_isg``, spec §1: ``<root>/<alias>/<template_id>/{template.json, template_meta.json,
instances/<iid>-tro_state.json, <iid>.oracle.json, <iid>.summary.json}``; detected by the ``<iid>.oracle.json``
sidecars / the v2 ``template_meta.json``). Per instance, in a worker process (one per template directory, or one per
instance with ``--fresh-process-per-instance``):

1. the instance is loaded through the evaluator's loader (:class:`ispatialgym.oracle.run_isg.InstanceLoader`: the
   template scene with the ``isg-v1`` camera profile, ``ISpatialGymTask``, ``load_tro_instance`` and the §3.14
   restores -- constraints / hinges, assisted grasps, attachment collision filters, articulation joint values),
2. the scene settles ``--settle-steps`` (60) physics steps without anything held and must reproduce the stored state:
   robot base within 2 mm / 0.02 rad, every object within 1 mm / 0.005 rad, every joint (robot and task objects) within
   half its tolerance (:data:`RELOAD_TOLERANCES`, :func:`joint_tolerance`),
3. the BDDL ``:init`` conditions hold and the binary goal is **false**,
4. the loader contract items the TRO carries were restored (every hinge exists, every grasp is attached, every
   attachment pair filtered, articulation joints at their initial values),
5. the sidecar loads (``Sidecar.load``), validates (``problems(check_program=True)``) and ``verify_tro`` matches the TRO,
6. the program check is re-run through ``ispatialgym.sampling.v2.program_check.run`` -- ``expand(alias, sidecar)``,
   then the IK / collision check of every action's targets from the recorded stances (reported per action name),
7. a contact sheet (init view; goal view when the goal can be materialised by teleport as the sampler did) is written
   to ``<template>/validation/<iid>.png``.

The per-template report goes to ``<template>/validation/report.json`` (:data:`REPORT_SCHEMA`); with ``--delete-failed``
the instances that failed the reload are deleted (TRO, sidecar, summary) and counted as ``reload_failed`` -- never
patched. One summary line per template (``VALIDATION_TEMPLATE {...}``) and one for the run (``VALIDATION_REPORT``).

**v1** (``sample_instances``: ``<out>/<problem>/<template>/``, no sidecar) keeps the previous behaviour
(:func:`validate_template`).

Examples::

    OMNIGIBSON_HEADLESS=1 python -m ispatialgym.sampling.validate_instances --root /tmp/isg_v2 \\
        [--alias <alias> ...] [--template <template_id | dir> ...] [--delete-failed] [--no-render] \\
        [--fresh-process-per-instance] [--settle-steps 60] [--no-program-check] [--no-goal]
    OMNIGIBSON_HEADLESS=1 python -m ispatialgym.sampling.validate_instances --out /tmp/isg_out      # v1 layout
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import numpy as np  # noqa: E402
import torch as th  # noqa: E402

# pure (torch only): the draft records program_check reads
from ispatialgym.sampling.v2.stances import (  # noqa: E402
    TRANSITION_ORDER,
    ContactRecord,
    GraspRecord,
    PathRecord,
    SidecarDraft,
    Stance,
    Transition,
    pose_from_sidecar,
)

log = logging.getLogger("ispatialgym.validate")

TRO_SUFFIX = "-tro_state.json"
SIDECAR_SUFFIX = ".oracle.json"
SUMMARY_SUFFIX = ".summary.json"
VALIDATION_DIR = "validation"
REPORT_SCHEMA = "isg-validation-v1"
#: directories under ``<root>`` / ``<root>/<alias>`` that are not aliases / templates
NON_TEMPLATE_DIRS = {"renders", "scene_variants", "validation", "activity_definitions", "runs", "instances"}

#: §3.5 reload tolerances: the fresh process must reproduce the stored state within these
RELOAD_TOLERANCES: Dict[str, float] = {
    "robot_base_position_m": 0.002,
    "robot_base_orientation_rad": 0.02,
    "object_position_m": 0.001,
    "object_orientation_rad": 0.005,
    "joint_tolerance_fraction": 0.5,  # of the joint's evaluation tolerance (articulation target) or of the §8 floor
}
#: spec §8 articulation tolerance floors per joint type (the tolerance of a joint without a TRO target)
JOINT_TOLERANCE_FLOOR: Dict[str, float] = {"revolute": 0.05, "prismatic": 0.01}
#: roles whose objects the sampler ignores for the final grasp / contact pose (``SamplerV2.support_objects``)
SUPPORT_ROLES: Tuple[str, ...] = ("support", "tray_source", "tray_dest", "wall")
#: robot joints the reload check skips: the holonomic base's virtual joints only hold the offset to the articulation root
BASE_VIRTUAL_JOINTS: Tuple[str, ...] = (
    "base_footprint_x_joint", "base_footprint_y_joint", "base_footprint_z_joint",
    "base_footprint_rx_joint", "base_footprint_ry_joint", "base_footprint_rz_joint",
)
DEFAULT_SETTLE_STEPS = 60
GOAL_SETTLE_STEPS = 120
RENDER_ITERS = 3
DEFAULT_IMAGE_SIZE = 1080  # evaluation profile isg-v1
DEFAULT_HEAD_APERTURE_MM = 40.0
SHEET_TILE = 540  # contact sheet tile side (the 1080² views downscaled by 2)


def _quat_angle(q0, q1) -> float:
    q0 = th.as_tensor(q0, dtype=th.float32)
    q1 = th.as_tensor(q1, dtype=th.float32)
    d = float(abs(th.dot(q0 / th.linalg.norm(q0), q1 / th.linalg.norm(q1))))
    return 2.0 * float(th.arccos(th.clamp(th.tensor(d), max=1.0)))


def pose_deviation(p0, q0, p1, q1) -> Tuple[float, float]:
    """``(position error m, orientation error rad)`` between two poses."""
    dp = float(th.linalg.norm(th.as_tensor(p1, dtype=th.float32) - th.as_tensor(p0, dtype=th.float32)))
    return dp, _quat_angle(q0, q1)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
        f.write("\n")


# ==========================================================================================
# layout (spec §1)
# ==========================================================================================


@dataclass
class InstanceFiles:
    """The files of one instance under ``<template>/instances/``."""

    iid: str
    tro_path: str
    sidecar_path: Optional[str] = None
    summary_path: Optional[str] = None

    @property
    def paths(self) -> List[str]:
        return [p for p in (self.tro_path, self.sidecar_path, self.summary_path) if p]


def list_instances(template_dir: str) -> List[InstanceFiles]:
    """Every ``instances/<iid>-tro_state.json`` of a template with its sidecar / summary when present, sorted by id."""
    out = []
    for tro in sorted(glob.glob(os.path.join(template_dir, "instances", f"*{TRO_SUFFIX}"))):
        iid = os.path.basename(tro)[: -len(TRO_SUFFIX)]
        base = tro[: -len(TRO_SUFFIX)]
        sidecar = base + SIDECAR_SUFFIX
        summary = base + SUMMARY_SUFFIX
        out.append(InstanceFiles(iid, tro, sidecar if os.path.exists(sidecar) else None, summary if os.path.exists(summary) else None))
    return out


def detect_layout(template_dir: str) -> str:
    """``"v2"`` for a ``sample_isg`` template (an ``<iid>.oracle.json`` sidecar next to a TRO, or a v2 ``template_meta.json``
    -- ``sampler_version`` 2.x / the ``subtask`` + ``spec_hash`` keys the v2 core writes), else ``"v1"``."""
    if glob.glob(os.path.join(template_dir, "instances", f"*{SIDECAR_SUFFIX}")):
        return "v2"
    meta_path = os.path.join(template_dir, "template_meta.json")
    if os.path.exists(meta_path):
        try:
            meta = _read_json(meta_path)
        except (OSError, json.JSONDecodeError):
            return "v1"
        if str(meta.get("sampler_version", "")).startswith("2"):
            return "v2"
        if "subtask" in meta and "spec_hash" in meta:
            return "v2"
    return "v1"


def discover_templates(root: str, aliases: Optional[Sequence[str]] = None, templates: Optional[Sequence[str]] = None) -> List[str]:
    """Template directories under ``root`` (``<root>/<alias | problem>/<template_id>/template_meta.json``), filtered by
    alias / problem directory name and by template id; a ``--template`` entry that is itself a template directory
    (contains ``template_meta.json``) is taken as is."""
    out: List[str] = []
    for t in templates or []:
        if os.path.isdir(t) and os.path.exists(os.path.join(t, "template_meta.json")):
            out.append(os.path.abspath(t))
    wanted_ids = {os.path.basename(os.path.normpath(t)) for t in (templates or []) if os.path.abspath(t) not in out}
    if not os.path.isdir(root):
        return out
    for alias_dir in sorted(glob.glob(os.path.join(root, "*"))):
        alias = os.path.basename(alias_dir)
        if not os.path.isdir(alias_dir) or alias in NON_TEMPLATE_DIRS or (aliases and alias not in aliases):
            continue
        for tdir in sorted(glob.glob(os.path.join(alias_dir, "*"))):
            tid = os.path.basename(tdir)
            if not os.path.isdir(tdir) or tid in NON_TEMPLATE_DIRS or not os.path.exists(os.path.join(tdir, "template_meta.json")):
                continue
            if templates and tid not in wanted_ids:
                continue
            if os.path.abspath(tdir) not in out:
                out.append(os.path.abspath(tdir))
    return out


def template_identity(template_dir: str, meta: Mapping[str, Any]) -> Dict[str, Any]:
    """Alias / template id / problem / subtask / scene of a template (v2 ``template_meta.json`` keys, v1 fallbacks)."""
    tdir = os.path.normpath(os.path.abspath(template_dir))
    alias = meta.get("alias") or os.path.basename(os.path.dirname(tdir))
    return {
        "template": tdir,
        "template_id": meta.get("template_id") or os.path.basename(tdir),
        "alias": alias,
        "problem": meta.get("problem"),
        "subtask": meta.get("subtask"),
        "scene": meta.get("scene"),
        "sampler_version": meta.get("sampler_version"),
    }


def support_instances(meta: Mapping[str, Any]) -> List[str]:
    """BDDL instances the sampler treats as supports (``template_meta.bindings`` roles of :data:`SUPPORT_ROLES`), the
    objects ``program_check`` ignores for the final grasp / contact pose (``SamplerV2.support_objects``)."""
    out = []
    for b in meta.get("bindings") or []:
        if isinstance(b, dict) and b.get("role") in SUPPORT_ROLES and b.get("name"):
            out.append(b["name"])
    return out


# ==========================================================================================
# tolerances and the state comparison (spec §3.5)
# ==========================================================================================


def joint_tolerance(articulation_targets: Optional[Mapping[str, Any]], inst: str, joint: str, joint_type: str = "revolute") -> float:
    """Reload tolerance of one joint: ``joint_tolerance_fraction`` (½) of the joint's evaluation tolerance -- the TRO
    ``articulation_targets[inst].joints[joint]`` tolerance when the joint is a target, else the spec §8 floor of its
    type (0.05 rad revolute / 0.01 m prismatic). ``articulation_targets`` may hold ``ArticulationTarget`` objects or
    plain ``{inst: {joint: (position, tolerance)}}`` dicts."""
    tol = None
    art = (articulation_targets or {}).get(inst)
    joints = getattr(art, "joints", art) if art is not None else None
    if isinstance(joints, Mapping) and joint in joints:
        entry = joints[joint]
        if isinstance(entry, Mapping):
            tol = entry.get("tolerance")
        elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
            tol = entry[1]
    if tol is None:
        tol = JOINT_TOLERANCE_FLOOR.get(joint_type, JOINT_TOLERANCE_FLOOR["revolute"])
    return RELOAD_TOLERANCES["joint_tolerance_fraction"] * float(tol)


def stored_state(object_states: Mapping[str, Any], robot_poses: Optional[Mapping[str, Any]], robot_state: Optional[Mapping[str, Any]],
                 joint_names: Optional[Mapping[str, Sequence[str]]] = None) -> Dict[str, Any]:
    """The state a TRO stores, in the shape :func:`evaluate_state` compares: ``robot_base`` (pos, ori), ``robot_joints``
    ``{name: q}`` (base virtual joints excluded), ``objects`` ``{inst: (pos, ori)}`` from each object's ``root_link``,
    ``object_joints`` ``{inst: {joint: q}}`` from the dumped ``joint_pos`` list of an articulated object (BEHAVIOR's
    ``EntityPrim._dump_state``: values in the object's joint order, named through ``joint_names[inst]`` -- the loaded
    object's ``joints.keys()`` -- or ``joint_<i>`` without it; a ``joints`` dict of per-joint states is read as well)."""
    out: Dict[str, Any] = {"robot_base": None, "robot_joints": {}, "objects": {}, "object_joints": {}}
    if robot_poses:
        poses = {k.lower(): v for k, v in robot_poses.items()}
        lst = poses.get("robot") or poses.get("r1pro") or next(iter(poses.values()), None)
        if lst:
            out["robot_base"] = (list(lst[0]["position"]), list(lst[0]["orientation"]))
    jp = (robot_state or {}).get("joint_positions") or {}
    out["robot_joints"] = {n: float(v) for n, v in jp.items() if n not in BASE_VIRTUAL_JOINTS}
    for inst, st in object_states.items():
        if not isinstance(st, dict):
            continue
        root = st.get("root_link") or {}
        if "pos" in root and "ori" in root:
            out["objects"][inst] = (list(root["pos"]), list(root["ori"]))
        q: Dict[str, float] = {}
        jp = st.get("joint_pos")
        if isinstance(jp, (list, tuple)) and jp:
            names = list((joint_names or {}).get(inst) or [])
            for i, v in enumerate(jp):
                q[names[i] if i < len(names) else f"joint_{i}"] = float(v)
        joints = st.get("joints") or {}
        if isinstance(joints, dict):
            for jn, js in joints.items():
                pos = js.get("pos") if isinstance(js, dict) else js
                if isinstance(pos, (int, float)):
                    q[jn] = float(pos)
                elif isinstance(pos, (list, tuple)) and len(pos) == 1:
                    q[jn] = float(pos[0])
        if q:
            out["object_joints"][inst] = q
    return out


def evaluate_state(stored: Mapping[str, Any], current: Mapping[str, Any], joint_tolerances: Optional[Mapping[Tuple[str, str], float]] = None,
                   robot_joint_tolerance: Optional[float] = None, tolerances: Mapping[str, float] = RELOAD_TOLERANCES) -> Tuple[Dict[str, Any], List[str]]:
    """Compare ``current`` (same shape as :func:`stored_state`) against ``stored`` with the §3.5 tolerances. Returns the
    per-item report (errors and ``ok`` flags) and the list of failure strings (empty = the state is reproduced).
    ``joint_tolerances`` maps ``(inst, joint)`` to the reload tolerance (:func:`joint_tolerance`); joints without an
    entry use the revolute floor. ``robot_joint_tolerance`` defaults to the revolute floor as well."""
    joint_tolerances = joint_tolerances or {}
    floor = tolerances["joint_tolerance_fraction"] * JOINT_TOLERANCE_FLOOR["revolute"]
    robot_joint_tolerance = floor if robot_joint_tolerance is None else float(robot_joint_tolerance)
    failures: List[str] = []
    checks: Dict[str, Any] = {"robot_base": None, "robot_joints": None, "objects": {}, "object_joints": {}}

    if stored.get("robot_base") is not None and current.get("robot_base") is not None:
        (p0, q0), (p1, q1) = stored["robot_base"], current["robot_base"]
        dp, da = pose_deviation(p0, q0, p1, q1)
        dxy = float(th.linalg.norm(th.as_tensor(p1, dtype=th.float32)[:2] - th.as_tensor(p0, dtype=th.float32)[:2]))
        ok = dp <= tolerances["robot_base_position_m"] and da <= tolerances["robot_base_orientation_rad"]
        checks["robot_base"] = {"pos_err_m": dp, "xy_err_m": dxy, "z_err_m": float(p1[2] - p0[2]), "ang_err_rad": da, "ok": ok}
        if not ok:
            failures.append(f"robot base moved {1000 * dp:.1f} mm / {da:.4f} rad (tolerance {1000 * tolerances['robot_base_position_m']:.0f} mm / {tolerances['robot_base_orientation_rad']} rad)")

    rj0, rj1 = stored.get("robot_joints") or {}, current.get("robot_joints") or {}
    if rj0:
        errs = {n: abs(float(rj1[n]) - float(v)) for n, v in rj0.items() if n in rj1}
        worst = max(errs, key=errs.get) if errs else None
        bad = sorted(n for n, e in errs.items() if e > robot_joint_tolerance)
        checks["robot_joints"] = {"max_err": errs[worst] if worst else 0.0, "worst_joint": worst, "n_compared": len(errs),
                                  "tolerance": robot_joint_tolerance, "failed": bad, "ok": not bad}
        if bad:
            failures.append(f"robot joints off by more than {robot_joint_tolerance:.4f}: " + ", ".join(f"{n} ({errs[n]:.4f})" for n in bad[:6]))

    for inst, (p0, q0) in (stored.get("objects") or {}).items():
        cur = (current.get("objects") or {}).get(inst)
        if cur is None:
            checks["objects"][inst] = {"ok": False, "missing": True}
            failures.append(f"{inst}: not in the loaded scope")
            continue
        dp, da = pose_deviation(p0, q0, cur[0], cur[1])
        ok = dp <= tolerances["object_position_m"] and da <= tolerances["object_orientation_rad"]
        checks["objects"][inst] = {"pos_err_m": dp, "ang_err_rad": da, "ok": ok}
        if not ok:
            failures.append(f"{inst} moved {1000 * dp:.2f} mm / {da:.4f} rad (tolerance {1000 * tolerances['object_position_m']:.0f} mm / {tolerances['object_orientation_rad']} rad)")

    for inst, q0 in (stored.get("object_joints") or {}).items():
        q1 = (current.get("object_joints") or {}).get(inst) or {}
        per: Dict[str, Any] = {}
        for jn, v0 in q0.items():
            if jn not in q1:
                continue
            tol = joint_tolerances.get((inst, jn), floor)
            err = abs(float(q1[jn]) - float(v0))
            per[jn] = {"err": err, "tolerance": tol, "ok": err <= tol}
            if err > tol:
                failures.append(f"{inst}.{jn} off by {err:.4f} (tolerance {tol:.4f})")
        if per:
            checks["object_joints"][inst] = per
    return checks, failures


def settle_drift(loaded: Mapping[str, Any], settled: Mapping[str, Any]) -> Dict[str, Any]:
    """How far the robot base and every object moved during the free settle (both in the :func:`stored_state` shape):
    tells a pose the loader did not reproduce from one that drifted afterwards."""
    out: Dict[str, Any] = {"robot_base": None, "objects": {}}
    if loaded.get("robot_base") is not None and settled.get("robot_base") is not None:
        dp, da = pose_deviation(loaded["robot_base"][0], loaded["robot_base"][1], settled["robot_base"][0], settled["robot_base"][1])
        out["robot_base"] = {"pos_m": dp, "ang_rad": da}
    for inst, (p0, q0) in (loaded.get("objects") or {}).items():
        cur = (settled.get("objects") or {}).get(inst)
        if cur is not None:
            dp, da = pose_deviation(p0, q0, cur[0], cur[1])
            out["objects"][inst] = {"pos_m": dp, "ang_rad": da}
    return out


# ==========================================================================================
# report schema
# ==========================================================================================


def new_template_report(template_dir: str, meta: Mapping[str, Any], layout: str, settle_steps: int = DEFAULT_SETTLE_STEPS) -> Dict[str, Any]:
    rep: Dict[str, Any] = {"schema": REPORT_SCHEMA, "layout": layout, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "finished": None, "time_s": None,
                           "settle_steps": int(settle_steps), "goal_settle_steps": GOAL_SETTLE_STEPS, "tolerances": dict(RELOAD_TOLERANCES),
                           "joint_tolerance_floor": dict(JOINT_TOLERANCE_FLOOR), "loader": {}, "instances": {},
                           "n_instances": 0, "n_ok": 0, "reload_failed": 0, "n_error": 0, "deleted": [], "error": None}
    rep.update(template_identity(template_dir, meta))
    return rep


def new_instance_record(iid: str) -> Dict[str, Any]:
    return {"iid": iid, "ok": False, "status": "pending", "failures": [], "warnings": [], "checks": {}, "contact_sheet": None, "time_s": None}


def finish_instance_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """``status``: ``ok`` (every §3.5 / §3.14 check passed), ``reload_failed`` (a check failed) or ``error`` (an
    exception that is not a loader-contract violation; kept, not deleted)."""
    if rec.get("status") == "error":
        rec["ok"] = False
        return rec
    rec["ok"] = not rec["failures"]
    rec["status"] = "ok" if rec["ok"] else "reload_failed"
    return rec


def finalize_template_report(report: Dict[str, Any], t0: Optional[float] = None) -> Dict[str, Any]:
    insts = report["instances"]
    report["n_instances"] = len(insts)
    report["n_ok"] = sum(1 for r in insts.values() if r.get("status") == "ok")
    report["reload_failed"] = sum(1 for r in insts.values() if r.get("status") == "reload_failed")
    report["n_error"] = sum(1 for r in insts.values() if r.get("status") == "error")
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if t0 is not None:
        report["time_s"] = round(time.time() - t0, 1)
    return report


def merge_template_reports(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a per-instance worker report into the template report (``--fresh-process-per-instance``)."""
    for iid, rec in other.get("instances", {}).items():
        base["instances"][iid] = rec
    for k in ("loader",):
        if other.get(k) and not base.get(k):
            base[k] = other[k]
    if other.get("error") and not base.get("error"):
        base["error"] = other["error"]
    base["deleted"] = sorted(set(base.get("deleted") or []) | set(other.get("deleted") or []))
    return finalize_template_report(base)


def failed_instance_ids(report: Mapping[str, Any]) -> List[str]:
    return sorted(iid for iid, r in report.get("instances", {}).items() if r.get("status") == "reload_failed")


def delete_failed_instances(template_dir: str, iids: Iterable[str]) -> List[str]:
    """§3.5: instances that failed the reload are deleted (TRO, sidecar, summary and any v1 ``-init.png``), never
    patched. Returns the paths removed."""
    removed = []
    for iid in iids:
        base = os.path.join(template_dir, "instances", iid)
        for path in (base + TRO_SUFFIX, base + SIDECAR_SUFFIX, base + SUMMARY_SUFFIX, base + "-init.png"):
            if os.path.exists(path):
                os.remove(path)
                removed.append(path)
    return removed


def summary_line(report: Mapping[str, Any]) -> str:
    fails = {iid: (r.get("failures") or [r.get("error", "error")])[:3] for iid, r in report.get("instances", {}).items() if r.get("status") != "ok"}
    body = {"alias": report.get("alias"), "template_id": report.get("template_id"), "n_instances": report.get("n_instances", 0), "n_ok": report.get("n_ok", 0),
            "reload_failed": report.get("reload_failed", 0), "n_error": report.get("n_error", 0), "deleted": len(report.get("deleted") or []),
            "time_s": report.get("time_s"), "fails": fails}
    if report.get("error"):
        body["error"] = report["error"]
    return "VALIDATION_TEMPLATE " + json.dumps(body, default=str)


# ==========================================================================================
# sidecar -> program check wiring (spec §3.5 program check, oracle doc §3.7)
# ==========================================================================================


def check_sidecar(sidecar_path: str, tro_path: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """``Sidecar.load`` + ``problems(check_program=True)`` (schema, required blocks, the program expands and annotates)
    + ``verify_tro`` (the sidecar's ``tro_sha256`` is this TRO's). Returns ``(record, raw sidecar dict or None)``."""
    from ispatialgym.oracle.sidecar import Sidecar, SidecarError

    rec: Dict[str, Any] = {"path": os.path.basename(sidecar_path), "loaded": False, "valid": False, "tro_sha256_ok": False, "problems": [],
                           "subtask": None, "alias": None, "ok": False}
    try:
        sc = Sidecar.load(sidecar_path)
    except SidecarError as e:
        rec["problems"].append(str(e))
        return rec, None
    rec["loaded"] = True
    rec["alias"] = sc.alias
    try:
        rec["subtask"] = sc.subtask
    except SidecarError as e:
        rec["problems"].append(str(e))
    try:
        problems = sc.problems(check_program=True)
    except Exception as e:  # noqa: BLE001  (a program that raises something else is still a problem to report)
        problems = [f"validation raised {e!r}"]
    rec["problems"].extend(problems)
    rec["valid"] = not problems
    try:
        sc.verify_tro(tro_path)
        rec["tro_sha256_ok"] = True
    except SidecarError as e:
        rec["problems"].append(str(e))
    rec["ok"] = rec["valid"] and rec["tro_sha256_ok"]
    return rec, _read_json(sidecar_path)


class LoadedSidecar(SidecarDraft):
    """A written ``<iid>.oracle.json`` as the ``SidecarDraft`` ``program_check.run`` reads: the typed records
    (``stance()``, ``grasp_of()``, ``stances``, ``transitions``) rebuilt from the file, ``to_dict()`` = the written dict
    itself (no lossy re-serialisation: the program is expanded from exactly what the sampler wrote)."""

    def __init__(self, alias: str, task_family: str, instance_id: str, raw: Mapping[str, Any]):
        super().__init__(alias, task_family, instance_id)
        self.raw: Dict[str, Any] = dict(raw)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.raw)


def draft_from_sidecar(d: Mapping[str, Any]) -> LoadedSidecar:
    """:class:`LoadedSidecar` from a written sidecar dict, so that the §3.5 program check runs on a written instance
    exactly as it ran in the sampler."""
    sd = LoadedSidecar(d["alias"], d.get("task_family", ""), d.get("instance_id", "draft"), d)
    sd.path_bucket_m = d.get("path_bucket_m")
    sd.tro_sha256 = d.get("tro_sha256")
    for name, st in (d.get("stances") or {}).items():
        sd.add_stance(Stance.from_sidecar(name, st))
    for t in d.get("transitions") or []:
        sd.add_transition(Transition(t["from"], t["to"], float(t["stand_off"]), tuple(t.get("order") or TRANSITION_ORDER)))
    for p in d.get("paths") or []:
        sd.add_path(PathRecord(p["from"], p["to"], p.get("mode", "local"), [tuple(float(v) for v in w) for w in p["waypoints"]],
                               float(p.get("length_m", 0.0)), p.get("posture", "untucked"), p.get("carried"), style=p.get("style")))
    for g in d.get("grasps") or []:
        ch = g.get("carry_height", 0.12)
        sd.add_grasp(GraspRecord(g["object"], g["arm"], g.get("grasp_key"), g.get("grasp_name"), float(g.get("yaw", 0.0) or 0.0),
                                 pose_from_sidecar(g.get("pre_grasp")), pose_from_sidecar(g.get("grasp")), pose_from_sidecar(g.get("pre_place")),
                                 pose_from_sidecar(g.get("place")), 0.0 if ch is None else float(ch), finger_q=g.get("finger_q"),
                                 eef_in_object=pose_from_sidecar(g.get("eef_in_object")), stance=g.get("stance", "manipulation"),
                                 place_on=g.get("place_on"), waypoints=[pose_from_sidecar(w) for w in g["waypoints"]] if g.get("waypoints") else None))
    for kind, c in (d.get("contacts") or {}).items():
        direction = c.get("direction")
        sd.add_contact(ContactRecord(kind, c["arm"], pre=pose_from_sidecar(c.get("pre")),
                                     contact=pose_from_sidecar(c.get("init") if kind == "push" else c.get("contact")),
                                     mid=pose_from_sidecar(c.get("mid")), goal=pose_from_sidecar(c.get("goal")),
                                     direction=None if direction is None else th.tensor(direction, dtype=th.float32),
                                     distance_m=c.get("distance_m"), brace_kind=c.get("kind"), object=c.get("object"), stance=c.get("stance", "manipulation")))
    sd.articulation = copy.deepcopy(d.get("articulation")) or None
    sd.blocks = list(copy.deepcopy(d.get("blocks") or []))
    sd.goal = copy.deepcopy(d.get("goal")) or None
    sd.language = copy.deepcopy(d.get("language") or {"objects": {}, "goal_numbers": {}})
    sd.waypoint_policy = d.get("waypoint_policy", "checked")
    return sd


class ValidationSampler:
    """What ``program_check.run`` needs from "the sampler" (its docstring): ``ikc`` (``IKChecker``), ``kit``
    (``RobotKit``), ``objects`` (bddl instance -> sim object), ``support_objects()`` and ``subtask``, over the
    evaluator's loaded instance."""

    def __init__(self, kit, ikc, objects: Mapping[str, Any], supports: Sequence[Any], subtask: str):
        self.kit = kit
        self.ikc = ikc
        self.objects = dict(objects)
        self._supports = list(supports)
        self.subtask = subtask

    def support_objects(self) -> List[Any]:
        return list(self._supports)


def run_program_check(alias: str, draft, sampler, subtask: str) -> Dict[str, Any]:
    """The §3.5 program check on a written instance through ``ispatialgym.sampling.v2.program_check.run`` (the same
    call the sampler makes): ``{"ok", "mode", "actions": [{"name", "stance", "carried", "n_poses"}], "failed_action",
    "reason", "time_s"}``. A ``Reject("program_check_failed", action=<name>)`` becomes ``ok = False`` with the action
    name; any other exception is reported as ``failed_action = "run"``."""
    from ispatialgym.sampling.v2 import program_check as PC

    t0 = time.time()
    out: Dict[str, Any] = {"ok": False, "mode": None, "actions": [], "failed_action": None, "reason": None, "time_s": None}
    try:
        rep = PC.run(alias, draft, sampler, subtask=subtask)
        out.update({"ok": True, "mode": rep.get("mode"), "actions": list(rep.get("actions") or []), "blocks": rep.get("blocks")})
    except Exception as e:  # noqa: BLE001  (``sim_utils.Reject`` needs the simulator stack: duck-typed on ``reason``)
        details = getattr(e, "details", None) or {}
        if getattr(e, "reason", None) == "program_check_failed":
            out["failed_action"] = details.get("action") or "unknown"
        else:
            out["failed_action"] = "run"
            log.debug("program check raised", exc_info=True)
        out["reason"] = str(e)
    out["time_s"] = round(time.time() - t0, 2)
    return out


def loader_contract_report(extras: Mapping[str, Any], loader_report: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Spec §3.14: what the TRO asked the loader to restore versus what ``InstanceLoader.report`` says was restored --
    every ``constraints`` hinge exists (and was re-created after the reset when needed), every ``grasps`` record is
    attached (at load and after the reset), every ``attachment_targets`` pair has its collision filter. Returns the
    record and the failure strings."""
    failures: List[str] = []
    want_c = list(extras.get("constraints") or [])
    have_c = list(loader_report.get("constraints") or [])
    after = loader_report.get("after_reset") or {}
    rec: Dict[str, Any] = {
        "constraints": {"expected": len(want_c), "restored": len(have_c), "recreated_after_reset": int(after.get("hinges_recreated", 0)),
                        "ok": len(have_c) == len(want_c)},
        "grasps": [],
        "attachments": {"expected": len(extras.get("attachment_targets") or []), "restored": len(loader_report.get("attachments") or []), "ok": True},
        "articulation_reset": list(after.get("articulation_reset") or []),
    }
    if not rec["constraints"]["ok"]:
        failures.append(f"loader restored {len(have_c)} of {len(want_c)} constraints")
    want_g = {(g.get("arm"), g.get("object")) for g in (extras.get("grasps") or [])}
    seen = set()
    for g in loader_report.get("grasps") or []:
        att = bool(g.get("attached")) and g.get("attached_after_reset", True) is not False
        rec["grasps"].append({"arm": g.get("arm"), "object": g.get("object"), "attached": bool(g.get("attached")), "attached_after_reset": g.get("attached_after_reset"),
                              "object_in_eef_dev_mm": g.get("object_in_eef_dev_mm"), "object_in_eef_dev_deg": g.get("object_in_eef_dev_deg"), "ok": att})
        seen.add((g.get("arm"), g.get("object")))
        if not att:
            failures.append(f"grasp of {g.get('object')} ({g.get('arm')}) not attached")
    for arm, obj in sorted(want_g - seen, key=str):
        rec["grasps"].append({"arm": arm, "object": obj, "attached": False, "ok": False})
        failures.append(f"grasp of {obj} ({arm}) was not restored")
    n_att, n_have = rec["attachments"]["expected"], rec["attachments"]["restored"]
    if n_have != n_att or any(not a.get("collision_filtered", True) for a in (loader_report.get("attachments") or [])):
        rec["attachments"]["ok"] = False
        failures.append(f"loader filtered {n_have} of {n_att} attachment pairs")
    rec["ok"] = not failures
    return rec, failures


# ==========================================================================================
# contact sheet
# ==========================================================================================


def _to_uint8(img) -> np.ndarray:
    arr = img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return arr[..., :3]


def compose_contact_sheet(init_rgb, goal_rgb=None, caption: str = "", note: str = "", tile: int = SHEET_TILE) -> np.ndarray:
    """``[init view | goal view]`` side by side (``tile`` px each) under a caption bar; without a goal view the right
    tile is a grey panel with ``note`` (why the goal was not materialised)."""
    from PIL import Image, ImageDraw

    def _tile(arr):
        return Image.fromarray(_to_uint8(arr)).convert("RGB").resize((tile, tile), Image.BILINEAR)

    bar = 28
    sheet = Image.new("RGB", (2 * tile, tile + bar), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 7), caption[:160], fill=(240, 240, 240))
    sheet.paste(_tile(init_rgb), (0, bar))
    if goal_rgb is not None:
        sheet.paste(_tile(goal_rgb), (tile, bar))
    else:
        panel = Image.new("RGB", (tile, tile), (60, 60, 60))
        ImageDraw.Draw(panel).multiline_text((10, tile // 2 - 10), ("goal view not materialised\n" + note)[:400], fill=(220, 220, 220))
        sheet.paste(panel, (tile, bar))
    draw.text((6 + 0, bar + 6), "init", fill=(255, 255, 0))
    draw.text((tile + 6, bar + 6), "goal", fill=(255, 255, 0))
    return np.asarray(sheet)


# ==========================================================================================
# v2: the simulator side
# ==========================================================================================


class KitWorld:
    """The ``world`` a ``RobotKit`` reads (duck-typed ``SimWorld``) over the evaluator's environment: robot, head camera
    key, floor height, task rooms, the collision-relevance filter (task rooms' floor extent + 1 m, task objects always)
    and ``teleport_robot``."""

    def __init__(self, env, robot, cam_key: str, rooms: Sequence[str], always: Sequence[str]):
        from ispatialgym.oracle.run_isg import make_obstacle_filter, task_bounds

        self.env = env
        self.scene = env.scene
        self.robot = robot
        self.cam_key = cam_key
        self.rooms = list(rooms)
        tm = getattr(env.scene, "trav_map", None)
        heights = getattr(tm, "floor_heights", None) if tm is not None else None
        self.floor_z = float(heights[0]) if heights is not None and len(heights) else 0.0
        bounds = task_bounds(env.scene, self.rooms) if self.rooms else None
        self.obstacle_filter = make_obstacle_filter(bounds, robot, list(always))

    def teleport_robot(self, base_pose2d, q=None, z=None):
        import omnigibson as og

        from ispatialgym.robot.conventions import NOMINAL_BASE_Z, quat_from_yaw

        x, y, yaw = base_pose2d
        zb = (self.floor_z + NOMINAL_BASE_Z) if z is None else float(z)
        if q is not None:
            self.robot.set_joint_positions(th.as_tensor(q, dtype=th.float32))
            self.robot.set_joint_velocities(th.zeros(len(q)))
        self.robot.set_position_orientation(th.tensor([x, y, zb]), quat_from_yaw(yaw))
        self.robot.keep_still()
        og.sim.step_physics()
        self.robot.keep_still()


def current_state(robot, physical: Mapping[str, Any], agent_instance: str) -> Dict[str, Any]:
    """The loaded scene in the shape of :func:`stored_state`."""
    p, q = robot.get_position_orientation()
    names = list(robot.joints.keys())
    rq = robot.get_joint_positions()
    out: Dict[str, Any] = {"robot_base": (p.tolist(), q.tolist()), "robot_joints": {n: float(rq[i]) for i, n in enumerate(names) if n not in BASE_VIRTUAL_JOINTS},
                           "objects": {}, "object_joints": {}}
    for inst, ent in physical.items():
        if inst == agent_instance or ent is robot:
            continue
        ep, eq = ent.get_position_orientation()
        out["objects"][inst] = (ep.tolist(), eq.tolist())
        joints = getattr(ent, "joints", None) or {}
        if joints:
            jq = ent.get_joint_positions()
            out["object_joints"][inst] = {n: float(jq[i]) for i, n in enumerate(joints.keys())}
    return out


def joint_tolerances_for(meta, physical: Mapping[str, Any], agent_instance: str) -> Dict[Tuple[str, str], float]:
    """``(inst, joint) -> reload tolerance`` for every joint of every articulated task object (:func:`joint_tolerance`
    with the joint type read from the simulator)."""
    out: Dict[Tuple[str, str], float] = {}
    for inst, ent in physical.items():
        if inst == agent_instance:
            continue
        for jn, joint in (getattr(ent, "joints", None) or {}).items():
            jtype = str(getattr(joint, "joint_type", "revolute") or "revolute").lower()
            jtype = "prismatic" if "prismatic" in jtype else "revolute"
            out[(inst, jn)] = joint_tolerance(meta.articulation_targets, inst, jn, jtype)
    return out


def _head_rgb(robot, cam_key: str, og, render_iters: int = RENDER_ITERS) -> np.ndarray:
    for _ in range(max(1, int(render_iters))):
        og.sim.render()
    obs, _ = robot.get_obs()
    return _to_uint8(obs[cam_key]["rgb"])


def _wake_and_step(og, task, physical: Mapping[str, Any], n_full_steps: int = 3) -> None:
    """Kinematic predicates cache per step and treat sleeping objects as unchanged (§3.5): wake + full steps."""
    for ent in physical.values():
        if hasattr(ent, "wake"):
            ent.wake()
    for _ in range(n_full_steps):
        og.sim.step()
    for ge in task.group_entities.values():
        ge.invalidate()


def _condition_texts(conditions: Optional[Sequence[Any]], idxs: Iterable[int]) -> List[str]:
    """BDDL text of the conditions at ``idxs`` (the index itself when the expanded list is not available)."""
    out = []
    for i in idxs:
        if conditions is not None and 0 <= int(i) < len(conditions):
            out.append(" ".join(str(conditions[int(i)]).split()))
        else:
            out.append(str(i))
    return out


def _binary_goal(task) -> Tuple[bool, List[str]]:
    _, status = task.compiled_task.check_goal(task._evaluate_predicate)
    binary = set(task.compiled_task.binary_goal_head_idxs)
    satisfied = set(status["satisfied"])
    return binary <= satisfied, _condition_texts(getattr(task.problem, "goal", None), sorted(binary - satisfied))


def _teleport_robot_to_goal(robot, goal_block: Mapping[str, Any]) -> None:
    """Camera relocalization: the robot at the sidecar's ``goal`` stance (base pose, trunk, arms untucked when asked),
    the joint hold re-targeted so the controllers keep it there."""
    from ispatialgym.oracle.action import ActionState
    from ispatialgym.robot.conventions import quat_from_yaw, torso_q

    x, y, yaw = [float(v) for v in goal_block["base_pose2d"]]
    q = robot.get_joint_positions().clone()
    idx = robot.trunk_control_idx
    q[idx] = torso_q(float(goal_block["translate"]), float(goal_block["tilt"]), (robot.joint_lower_limits[idx], robot.joint_upper_limits[idx]))
    if goal_block.get("arms") == "untucked":
        for arm, aidx in robot.arm_control_idx.items():
            q[aidx] = robot.untucked_default_joint_pos[aidx]
    robot.set_joint_positions(q)
    robot.set_joint_velocities(th.zeros_like(q))
    p, _ = robot.get_position_orientation()
    robot.set_position_orientation(th.tensor([x, y, float(p[2])]), quat_from_yaw(yaw))
    robot.keep_still()
    hold = ActionState(robot)
    hold.hold_current()
    robot.apply_action(hold.action())


def materialise_goal(env, task, tro, physical: Mapping[str, Any], robot, sidecar_dict: Optional[Mapping[str, Any]], og,
                     settle_steps: int = GOAL_SETTLE_STEPS) -> Dict[str, Any]:
    """Teleport the task to its goal configuration the way the sampler's goal check did (world-frame position targets --
    with the target orientation when one is set, else the current one --, group slots, articulation joint targets; an
    orientation-only target of a hinged object is rotated about its hinge anchor), settle, evaluate the binary goal.
    The robot is put at the sidecar's ``goal`` stance when the block exists (the camera-relocalization goal; the stance
    the executor captures the goal image from otherwise), else it stays at init. Targets a plain teleport cannot
    materialise (held objects, camera-frame depth targets, attachments) are skipped and named. The caller restores with
    ``env.reset()``. Informational: the sampler verified the goal with the task mechanics."""
    from ispatialgym.activity.problem import AGENT_INSTANCE
    from ispatialgym.robot.conventions import qinv, qmul, quat_rotate

    meta = tro.meta
    extras = meta.extras or {}
    out: Dict[str, Any] = {"materialised": False, "moved": [], "skipped": [], "binary_goal_after_teleport": None, "unsatisfied": [], "robot_view": "init stance"}
    if extras.get("attachment_targets"):
        out["skipped"].append("attachment needs AttachedTo.set_value")
        return out
    held = {g["object"] for g in (extras.get("grasps") or []) if isinstance(g, dict)}
    hinges = {c["child"]: c for c in (extras.get("constraints") or []) if isinstance(c, dict) and c.get("type", "revolute") == "revolute"}  # joint_friction records name no child
    for inst, tgt in meta.pose_targets.items():
        if inst == AGENT_INSTANCE:
            continue
        ent = physical.get(inst)
        if ent is None or ent is robot:
            continue
        if inst in held:
            out["skipped"].append(f"{inst}: held object")
            continue
        world = tgt.frame == "world"
        if world and tgt.position is not None and tgt.position_metric == "full":
            q_goal = th.as_tensor(tgt.orientation, dtype=th.float32) if tgt.orientation is not None else ent.get_position_orientation()[1].float()
            ent.set_position_orientation(th.as_tensor(tgt.position, dtype=th.float32), q_goal)
            ent.keep_still()
            out["moved"].append(inst)
        elif world and tgt.orientation is not None and tgt.position is None and inst in hinges:
            p, q = (x.float() for x in ent.get_position_orientation())
            anchor = th.tensor(hinges[inst]["anchor_position"], dtype=th.float32)
            q_goal = th.as_tensor(tgt.orientation, dtype=th.float32)
            dq = qmul(q_goal, qinv(q))
            ent.set_position_orientation(anchor + quat_rotate(dq, p - anchor), q_goal)
            ent.keep_still()
            out["moved"].append(f"{inst} (about the hinge)")
        else:
            out["skipped"].append(f"{inst}: frame {tgt.frame}, position {tgt.position_metric}, orientation {tgt.orientation_metric}")
    for gname, g in meta.groups.items():
        members = task.groups.get(gname, [])
        slots = list(g.target_slots.items())
        if not slots:
            continue
        if g.assignment_mode == "fixed" and g.member_to_slot:
            pairs = [(m, g.target_slots[g.member_to_slot[m]]) for m in members if m in g.member_to_slot]
        else:
            pairs = [(m, s) for m, (_, s) in zip(members, slots)]
        for m, slot in pairs:
            ent = physical.get(m)
            if ent is None or m in held:
                continue
            ent.set_position_orientation(th.as_tensor(slot.position, dtype=th.float32), th.as_tensor(slot.orientation, dtype=th.float32))
            ent.keep_still()
            out["moved"].append(m)
    for inst, art in meta.articulation_targets.items():
        ent = physical.get(inst)
        if ent is None:
            continue
        names = list(ent.joints.keys())
        q = ent.get_joint_positions().clone()
        for jn, (jpos, _tol) in art.joints.items():
            if jn in names:
                q[names.index(jn)] = float(jpos)
        ent.set_joint_positions(q)
        ent.set_joint_velocities(th.zeros_like(q))
        ent.keep_still()
        out["moved"].append(f"{inst} joints")
    goal_block = (sidecar_dict or {}).get("goal")
    if goal_block and (out["moved"] or AGENT_INSTANCE in meta.pose_targets):
        _teleport_robot_to_goal(robot, goal_block)
        out["robot_view"] = "sidecar goal stance"
        if AGENT_INSTANCE in meta.pose_targets:
            out["moved"].append(f"{AGENT_INSTANCE} (sidecar goal stance)")
    elif AGENT_INSTANCE in meta.pose_targets:
        out["skipped"].append(f"{AGENT_INSTANCE}: camera target without a sidecar goal block")
    if not out["moved"]:
        return out
    for _ in range(int(settle_steps)):
        og.sim.step_physics()
        robot.keep_still()
    _wake_and_step(og, task, physical, 1)
    ok, unsatisfied = _binary_goal(task)
    out.update({"materialised": True, "binary_goal_after_teleport": bool(ok), "unsatisfied": unsatisfied})
    return out


def _release_previous(loader, robot, og) -> None:
    """Undo the previous instance's §3.14 restores before the next TRO is loaded into the same environment: the hinge
    prims and the assisted grasps."""
    if loader is None:
        return
    try:
        from ispatialgym.sampling.constraints import remove_hinge

        for h in list(getattr(loader, "hinges", None) or []):
            remove_hinge(h)
        if getattr(loader, "hinges", None):
            og.sim.update_handles()
    except Exception:  # noqa: BLE001
        log.warning("hinge cleanup failed", exc_info=True)
    for arm, obj in list(getattr(robot, "_ag_obj_in_hand", {}).items()):
        if obj is not None:
            try:
                robot.release_grasp_immediately(arm)
            except Exception:  # noqa: BLE001
                log.warning("grasp release of %s failed", arm, exc_info=True)


def _adopt_env(loader, env, cfg) -> None:
    """Point an ``InstanceLoader`` at the already built environment of this template (one process per template)."""
    from ispatialgym.oracle import r1pro_model as M

    loader.env = env
    loader.cfg = cfg
    loader.task = env.task
    loader.robot = env.robots[0]
    loader.cam_key = next(k for k in loader.robot.sensors if k.endswith(M.HEAD_CAMERA_SENSOR_SUFFIX))
    loader.report["head_camera_key"] = loader.cam_key
    loader.report["evaluation_profile"] = loader.tro.meta.extras.get("evaluation_profile") or "isg-v1"


def _build_kit(env, robot, cam_key: str, rooms: Sequence[str], always: Sequence[str], arms: Sequence[str]):
    """``RobotKit`` (cuRobo world + IK) and ``IKChecker`` over the evaluator's environment; the IK cache namespace is the
    validation's own so that no cached sampler verdict is read back (every recorded pose is re-solved here)."""
    from ispatialgym.sampling.robot_utils import RobotKit
    from ispatialgym.sampling.v2.ik import IKChecker

    world = KitWorld(env, robot, cam_key, rooms, always)
    kit = RobotKit(world, prebuild=tuple(f"{a}_arm" for a in arms))
    kit.sync_world()
    ikc = IKChecker(kit, {"scene": getattr(env.scene, "scene_model", None), "rooms": sorted(rooms), "floor_z": world.floor_z, "validation": "reload"})
    return kit, ikc


def _program_arms(sidecar_dicts: Iterable[Mapping[str, Any]]) -> List[str]:
    """Arms the programs of these sidecars use (planner embodiments to prebuild), from ``expand``; ``["right"]`` when
    nothing can be expanded."""
    from ispatialgym.oracle.programs import expand
    from ispatialgym.oracle.sidecar import Sidecar

    arms = set()
    for d in sidecar_dicts:
        try:
            for a in expand(d["alias"], Sidecar.from_dict(d)):
                arms.update(a.arms)
        except Exception:  # noqa: BLE001
            continue
    return sorted(arms) or ["right"]


def validate_template_v2(template_dir: str, instances: Optional[Sequence[str]] = None, settle_steps: int = DEFAULT_SETTLE_STEPS, render: bool = True,
                         check_goal: bool = True, program_check: bool = True, render_iters: int = RENDER_ITERS, image_size: int = DEFAULT_IMAGE_SIZE,
                         head_aperture: float = DEFAULT_HEAD_APERTURE_MM, delete_failed: bool = False, report_path: Optional[str] = None) -> Dict[str, Any]:
    """Reload validation of one v2 template directory in this process (see the module docstring). Writes
    ``<template>/validation/report.json`` (or ``report_path``) and returns the report."""
    from ispatialgym.oracle import run_isg as RI  # imports the cuRobo planner before omnigibson (warp pinning)

    import omnigibson as og

    from ispatialgym.activity.problem import AGENT_INSTANCE

    t_start = time.time()
    template_dir = os.path.abspath(template_dir)
    meta = _read_json(os.path.join(template_dir, "template_meta.json"))
    report = new_template_report(template_dir, meta, "v2", settle_steps)
    val_dir = os.path.join(template_dir, VALIDATION_DIR)
    report_path = report_path or os.path.join(val_dir, "report.json")
    todo = [i for i in list_instances(template_dir) if instances is None or i.iid in set(instances)]
    if not todo:
        report["error"] = "no instances" if instances is None else f"instances {list(instances)} not found"
        finalize_template_report(report, t_start)
        _write_json(report_path, report)
        return report
    os.makedirs(val_dir, exist_ok=True)
    supports = support_instances(meta)
    sidecar_dicts = []
    for inst in todo:
        if inst.sidecar_path:
            try:
                sidecar_dicts.append(_read_json(inst.sidecar_path))
            except (OSError, json.JSONDecodeError):
                pass
    if not report.get("subtask") and sidecar_dicts:  # a v1 template_meta with v2 sidecars: the subtask comes from the sidecar
        try:
            from ispatialgym.oracle.sidecar import subtask_of_alias

            report["subtask"] = subtask_of_alias(sidecar_dicts[0]["alias"])
        except Exception:  # noqa: BLE001
            pass
    env = cfg = None
    kit = ikc = None
    prev_loader = None
    for inst in todo:
        rec = new_instance_record(inst.iid)
        t1 = time.time()
        loader = None
        try:
            loader = RI.InstanceLoader(inst.tro_path, image_size=image_size, head_aperture=head_aperture)
            if env is None:
                t0 = time.time()
                env = loader.build_env()
                cfg = loader.cfg
                report["loader"] = {"scene_load_s": round(time.time() - t0, 1), **{k: loader.report.get(k) for k in ("trav_cells_freed", "head_camera_key", "head_camera_aperture_mm", "evaluation_profile")}}
            else:
                _release_previous(prev_loader, env.robots[0], og)
                _adopt_env(loader, env, cfg)
            try:
                obs, _info = loader.restore()
            except RI.LoadError as e:
                rec["failures"].append(f"loader contract: [{e.reason}] {e.message}")
                rec["checks"]["loader_contract"] = {"ok": False, "load_error": e.reason, "message": e.message, "report": loader.report}
                raise _Skip()
            task, robot, cam_key = loader.task, loader.robot, loader.cam_key
            physical = loader.physical
            tro = loader.tro
            rec["checks"]["trav_cells_blocked"] = loader.report.get("trav_cells_blocked")
            # -- §3.5: settle without anything held, then reproduce the stored state ------------------------------
            loaded = current_state(robot, physical, AGENT_INSTANCE)  # right after the loader (diagnostic: load error vs settle drift)
            for _ in range(int(settle_steps)):
                og.sim.step_physics()
            _wake_and_step(og, task, physical, 3)
            current = current_state(robot, physical, AGENT_INSTANCE)
            stored = stored_state(tro.object_states, tro.robot_poses, tro.meta.robot_state,
                                  {inst: list(ent.joints.keys()) for inst, ent in physical.items() if getattr(ent, "joints", None)})
            checks, failures = evaluate_state(stored, current, joint_tolerances_for(tro.meta, physical, AGENT_INSTANCE))
            checks["settle_drift"] = settle_drift(loaded, current)
            rec["checks"]["state"] = checks
            rec["failures"] += failures
            # -- :init true, binary goal false ---------------------------------------------------------------------
            ok_init, status = task.compiled_task.check_initial_conditions(task._evaluate_predicate)
            rec["checks"]["init_true"] = bool(ok_init)
            if not ok_init:
                rec["failures"].append(f"init false: {_condition_texts(getattr(task.problem, 'init', None), status.get('unsatisfied', []))}")
            goal_now, _ = _binary_goal(task)
            rec["checks"]["goal_false_at_init"] = not goal_now
            if goal_now:
                rec["failures"].append("binary goal already satisfied at init")
            # -- §3.14 loader contract ----------------------------------------------------------------------------
            lc, lc_fail = loader_contract_report(tro.meta.extras or {}, loader.report)
            rec["checks"]["loader_contract"] = lc
            rec["failures"] += lc_fail
            # -- init view -----------------------------------------------------------------------------------------
            init_rgb = _head_rgb(robot, cam_key, og, render_iters) if render else None
            # -- sidecar -------------------------------------------------------------------------------------------
            sc_dict = None
            if inst.sidecar_path:
                sc_rec, sc_dict = check_sidecar(inst.sidecar_path, inst.tro_path)
                rec["checks"]["sidecar"] = sc_rec
                if not sc_rec["ok"]:
                    rec["failures"].append("sidecar: " + "; ".join(sc_rec["problems"][:4]))
            else:
                rec["checks"]["sidecar"] = {"ok": False, "problems": ["no sidecar next to the TRO"]}
                rec["failures"].append(f"no sidecar {inst.iid}{SIDECAR_SUFFIX}")
            # -- program check (§3.5) ------------------------------------------------------------------------------
            if program_check and sc_dict is not None and rec["checks"]["sidecar"].get("loaded"):
                alias = sc_dict.get("alias") or meta.get("alias")
                subtask = rec["checks"]["sidecar"].get("subtask") or meta.get("subtask") or alias
                rooms = list((tro.meta.scene or {}).get("task_rooms") or meta.get("task_rooms") or meta.get("rooms") or [])
                objects = {k: v for k, v in physical.items() if k != AGENT_INSTANCE and v is not robot}
                if kit is None:
                    t0 = time.time()
                    kit, ikc = _build_kit(env, robot, cam_key, rooms, [o.name for o in objects.values()], _program_arms(sidecar_dicts))
                    report["loader"]["planner_build_s"] = round(time.time() - t0, 1)
                else:
                    kit.sync_world()
                sampler = ValidationSampler(kit, ikc, objects, [objects[s] for s in supports if s in objects], subtask)
                pc = run_program_check(alias, draft_from_sidecar(sc_dict), sampler, subtask)
                rec["checks"]["program_check"] = pc
                if not pc["ok"]:
                    rec["failures"].append(f"program check failed at {pc['failed_action']}: {pc['reason']}")
            elif program_check:
                rec["checks"]["program_check"] = {"ok": False, "skipped": "sidecar not loaded"}
            # -- goal by teleport + goal view ----------------------------------------------------------------------
            goal_rgb = None
            if check_goal:
                g = materialise_goal(env, task, tro, physical, robot, sc_dict, og, GOAL_SETTLE_STEPS)
                rec["checks"]["goal"] = g
                if g["materialised"]:
                    if render:
                        goal_rgb = _head_rgb(robot, cam_key, og, render_iters)
                    if g["binary_goal_after_teleport"] is False:
                        rec["warnings"].append(f"binary goal not satisfied after a plain teleport to the targets: {g['unsatisfied']}")
            # -- contact sheet -------------------------------------------------------------------------------------
            if render and init_rgb is not None:
                finish_instance_record(rec)
                note = "; ".join((rec["checks"].get("goal") or {}).get("skipped") or []) or ("goal check off" if not check_goal else "")
                caption = f"{inst.iid}  [{rec['status']}]" + (f"  {rec['failures'][0]}" if rec["failures"] else "")
                sheet = compose_contact_sheet(init_rgb, goal_rgb, caption, note)
                from PIL import Image

                out_png = os.path.join(val_dir, f"{inst.iid}.png")
                Image.fromarray(sheet).save(out_png)
                rec["contact_sheet"] = os.path.relpath(out_png, template_dir)
            env.reset()
        except _Skip:
            pass
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            rec["status"] = "error"
            rec["error"] = repr(e)
            rec["failures"].append(f"exception: {e!r}")
        finish_instance_record(rec)
        rec["time_s"] = round(time.time() - t1, 1)
        report["instances"][inst.iid] = rec
        prev_loader = loader
        log.info("%s: %s %s", inst.iid, rec["status"].upper(), "; ".join(rec["failures"][:3]))
    finalize_template_report(report, t_start)
    if delete_failed:
        failed = failed_instance_ids(report)
        report["deleted"] = failed
        report["deleted_paths"] = delete_failed_instances(template_dir, failed)
    _write_json(report_path, report)
    try:
        og.clear()
    except Exception:  # noqa: BLE001
        pass
    return report


class _Skip(Exception):
    """Internal: stop the checks of one instance after a load failure that is already recorded."""


# ==========================================================================================
# v1 (sample_instances layout), unchanged
# ==========================================================================================


def validate_template(template_dir: str, extra_steps: int, check_goal: bool, pos_tol: float, ang_tol: float):
    import omnigibson as og
    from omnigibson.macros import gm
    from omnigibson.utils.python_utils import recursively_convert_to_torch

    for name, value in (("ENABLE_OBJECT_STATES", True), ("USE_GPU_DYNAMICS", False), ("ENABLE_TRANSITION_RULES", False)):
        if getattr(gm, name) != value:
            with gm.unlocked():
                setattr(gm, name, value)

    from ispatialgym.eval.group_entity import GroupEntity
    from ispatialgym.eval.tro import load_tro
    from ispatialgym.oracle import r1pro_model as M
    from ispatialgym.tasks import ISpatialGymTask  # noqa: F401  (registers the task type)

    with open(os.path.join(template_dir, "template_meta.json")) as f:
        meta = json.load(f)
    problem = meta["problem"]
    activity, did = problem.rsplit("-", 1)
    tros = sorted(glob.glob(os.path.join(template_dir, "instances", "*-tro_state.json")))
    results = {"template": template_dir, "problem": problem, "instances": {}}
    if not tros:
        results["error"] = "no instances"
        return results
    from ispatialgym.sampling.sim_utils import IMAGE_SIZE, PARK_POSITION

    cfg = {
        "env": {"action_frequency": 30, "physics_frequency": 120},
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": meta["scene"],
            "scene_file": os.path.join(template_dir, "template.json"),
            "load_room_instances": meta.get("load_room_instances") or None,
        },
        # the robot saved in the template carries no room assignment, so partial-room loading skips it: the evaluation
        # robot comes from the config (same JoyLo configuration as the sampler) and the TRO restores its pose / joints
        "robots": [
            M.joylo_robot_config(
                "robot_r1",
                obs_modalities=["rgb"],  # validation renders contact sheets only; seg reads live in SimWorld (B10: showLights=0 guard)
                sensor_config={"VisionSensor": {"sensor_kwargs": {"image_height": IMAGE_SIZE, "image_width": IMAGE_SIZE}}},
                position=PARK_POSITION.tolist(),
            )
        ],
        "task": {
            "type": "ISpatialGymTask",
            "activity_name": activity,
            "activity_definition_id": int(did),
            "activity_definitions_root": os.path.join(template_dir, "activity_definitions"),
            "tro_path": tros[0],
            "termination_config": {"max_steps": 10 ** 6},
        },
    }
    # the template must equal the stable scene json except for the task objects and the robot
    from ispatialgym.sampling.sim_utils import stable_scene_path

    # templates made from a scene variant (chairs removed) are compared against that variant
    sv = meta.get("scene_variant") or {}
    ref_path = os.path.normpath(os.path.join(template_dir, sv["scene_file"])) if sv.get("scene_file") else stable_scene_path(meta["scene"])
    if not os.path.exists(ref_path):
        log.warning("%s: scene variant %s not found, comparing against the stable scene", problem, ref_path)
        ref_path = stable_scene_path(meta["scene"])
    with open(ref_path) as f:
        stable_reg = json.load(f)["state"]["registry"]["object_registry"]
    with open(os.path.join(template_dir, "template.json")) as f:
        tmpl_reg = json.load(f)["state"]["registry"]["object_registry"]
    task_names = set(meta.get("inst_to_name", {}).values())
    devs = {}
    for name, st in tmpl_reg.items():
        if name in task_names or name not in stable_reg or "root_link" not in st:
            continue
        dp = float(th.linalg.norm(th.tensor(st["root_link"]["pos"]) - th.tensor(stable_reg[name]["root_link"]["pos"])))
        da = _quat_angle(st["root_link"]["ori"], stable_reg[name]["root_link"]["ori"])
        if dp > 1e-3 or da > 5e-3:
            devs[name] = {"pos": round(dp, 4), "ang": round(da, 4)}
    results["template_scene_deviations"] = devs
    if devs:
        log.warning("%s: %d non-task scene objects differ from the stable scene json: %s", problem, len(devs), dict(list(devs.items())[:5]))

    t0 = time.time()
    env = og.Environment(configs=cfg)
    results["scene_load_s"] = time.time() - t0
    from ispatialgym.sampling.scene_variants import patch_trav_map_from_meta

    results["trav_cells_freed"] = patch_trav_map_from_meta(env.scene, meta)  # what evaluation must do as well
    task = env.task
    robot = env.robots[0]
    for path in tros:
        iid = os.path.basename(path).replace("-tro_state.json", "")
        rec = {"ok": True, "checks": {}, "notes": []}
        t1 = time.time()
        try:
            tro = load_tro(path)
            extras = tro.meta.extras
            constrained = {c.get("child") or c.get("object") for c in (extras.get("constraints") or [])} | {g["object"] for g in (extras.get("grasps") or [])}
            if constrained:
                rec["notes"].append(f"loader cannot restore grasps / constraints yet for {sorted(constrained)}; dependent checks skipped")
            task.load_tro_instance(env, tro=tro, settle_steps=25)
            scope = task.object_scope
            physical = {k: v for k, v in scope.items() if v is not None and not isinstance(v, GroupEntity)}
            # kinematic predicates cache per step and treat sleeping objects as unchanged: wake + a few full steps (the
            # robot needs to rest on the floor for `ontop agent floor`: its base starts a few mm above it)
            for ent in physical.values():
                if hasattr(ent, "wake"):
                    ent.wake()
            for _ in range(3):
                og.sim.step()
            for ent in task.group_entities.values():
                ent.invalidate()
            # robot pose / joints
            rp = tro.robot_poses["robot"][0]
            p, q = robot.get_position_orientation()
            rec["checks"]["robot_pose"] = {"pos_err": float(th.linalg.norm(p - th.tensor(rp["position"]))), "ang_err": _quat_angle(q, rp["orientation"])}
            jp = (tro.meta.robot_state or {}).get("joint_positions") or {}
            names = list(robot.joints.keys())
            cur = robot.get_joint_positions()
            jerr = max([abs(float(cur[names.index(n)]) - float(v)) for n, v in jp.items() if n in names] or [0.0])
            rec["checks"]["robot_joint_err"] = jerr
            if devs:
                rec["ok"] = False
                rec["notes"].append(f"template differs from the stable scene json for {len(devs)} non-task objects (max {max(d['pos'] for d in devs.values()):.3f} m)")
            # init conditions
            ok_init, status = task.compiled_task.check_initial_conditions(task._evaluate_predicate)
            if not ok_init and constrained:
                rec["checks"]["init"] = "skipped (grasp/constraint)"
            else:
                rec["checks"]["init"] = bool(ok_init)
                if not ok_init:
                    rec["ok"] = False
                    rec["notes"].append(f"init false: {status}")
            # goal not satisfied at init
            _, gstatus = task.compiled_task.check_goal(task._evaluate_predicate)
            binary = set(task.compiled_task.binary_goal_head_idxs)
            goal_now = binary <= set(gstatus["satisfied"])
            rec["checks"]["goal_false_at_init"] = not goal_now
            if goal_now:
                rec["ok"] = False
                rec["notes"].append("binary goal already satisfied at init")
            # stability
            before = {k: tuple(x.clone() for x in v.get_position_orientation()) for k, v in physical.items()}
            for _ in range(extra_steps):
                og.sim.step()
            drift = {}
            for k, v in physical.items():
                if k in constrained or v is robot:
                    continue
                p1, q1 = v.get_position_orientation()
                dp = float(th.linalg.norm(p1 - before[k][0]))
                da = _quat_angle(q1, before[k][1])
                if dp > pos_tol or da > ang_tol:
                    drift[k] = {"pos": dp, "ang": da}
            rp1, _ = robot.get_position_orientation()
            rb = float(th.linalg.norm(rp1[:2] - p[:2]))
            rec["checks"]["drift"] = drift
            rec["checks"]["robot_base_drift"] = rb
            if drift or rb > pos_tol:
                rec["ok"] = False
                rec["notes"].append(f"unstable after {extra_steps} steps: {drift} robot {rb:.3f}")
            # goal feasibility (world-frame full poses / slots / joints). Informational: the sampler already verified the
            # goal with the task-specific mechanics (attachment joints, hinges, contact nudging) that a plain teleport
            # cannot reproduce here.
            if check_goal:
                if extras.get("attachment_targets"):
                    rec["checks"]["goal_feasible"] = "skipped (attachment needs AttachedTo.set_value)"
                else:
                    rec["checks"]["goal_feasible"] = _goal_feasible(env, task, tro, physical, constrained, og, recursively_convert_to_torch)
                    if rec["checks"]["goal_feasible"] is False:
                        rec["notes"].append("warning: binary goal not satisfied after a plain teleport to the targets")
            env.reset()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            rec["ok"] = False
            rec["notes"].append(f"exception: {e!r}")
        rec["time_s"] = time.time() - t1
        results["instances"][iid] = rec
        log.info("%s: %s %s", iid, "OK" if rec["ok"] else "FAIL", "; ".join(rec["notes"]))
    og.clear()
    return results


def _goal_feasible(env, task, tro, physical, constrained, og, to_torch):
    """Teleport objects to their targets, settle, evaluate the binary goal, restore via env.reset()."""
    meta = tro.meta
    moved = False
    for inst, tgt in meta.pose_targets.items():
        ent = physical.get(inst)
        if ent is None or ent is env.robots[0] or inst in constrained:
            continue
        if tgt.frame != "world" or tgt.position is None or tgt.orientation is None or tgt.position_metric != "full":
            continue
        ent.set_position_orientation(th.as_tensor(tgt.position, dtype=th.float32), th.as_tensor(tgt.orientation, dtype=th.float32))
        ent.keep_still()
        moved = True
    for gname, g in meta.groups.items():
        members = task.groups.get(gname, [])
        slots = list(g.target_slots.items())
        if g.assignment_mode == "fixed" and g.member_to_slot:
            pairs = [(m, g.target_slots[g.member_to_slot[m]]) for m in members if m in g.member_to_slot]
        else:
            pairs = [(m, s) for m, (_, s) in zip(members, slots)]
        for m, slot in pairs:
            ent = physical.get(m)
            if ent is None or m in constrained:
                continue
            ent.set_position_orientation(th.as_tensor(slot.position, dtype=th.float32), th.as_tensor(slot.orientation, dtype=th.float32))
            ent.keep_still()
            moved = True
    for inst, art in meta.articulation_targets.items():
        ent = physical.get(inst)
        if ent is None:
            continue
        names = list(ent.joints.keys())
        q = ent.get_joint_positions().clone()
        for jn, (jpos, _tol) in art.joints.items():
            if jn in names:
                q[names.index(jn)] = float(jpos)
        ent.set_joint_positions(q)
        ent.set_joint_velocities(th.zeros_like(q))
        ent.keep_still()
        moved = True
    if not moved:
        return "skipped (no world-frame targets)"
    for _ in range(120):
        og.sim.step_physics()
        env.robots[0].keep_still()
    # kinematic predicates treat sleeping objects as unchanged: wake everything and take one full step first
    for ent in physical.values():
        if hasattr(ent, "wake"):
            ent.wake()
    og.sim.step()
    for ent in task.group_entities.values():
        ent.invalidate()
    _, status = task.compiled_task.check_goal(task._evaluate_predicate)
    binary = set(task.compiled_task.binary_goal_head_idxs)
    ok = binary <= set(status["satisfied"])
    if not ok:
        log.info("goal status after teleport: unsatisfied %s", sorted(binary - set(status["satisfied"])))
    return bool(ok)


# ==========================================================================================
# CLI
# ==========================================================================================


def _worker_argv(args, template_dir: str, iid: Optional[str] = None) -> List[str]:
    argv = [sys.executable, "-m", "ispatialgym.sampling.validate_instances", "--worker", "--root", args.root or args.out or os.path.dirname(os.path.dirname(template_dir)),
            "--template", template_dir, "--settle-steps", str(args.settle_steps), "--render-iters", str(args.render_iters), "--image-size", str(args.image_size),
            "--log-level", str(args.log_level)]
    if iid:
        argv += ["--instance", iid]
    for flag, on in (("--no-render", args.no_render), ("--no-goal", args.no_goal), ("--no-program-check", args.no_program_check)):
        if on:
            argv.append(flag)
    return argv


def _run_worker(argv: List[str], timeout: Optional[float]) -> int:
    log.info("worker: %s", " ".join(argv[3:]))
    try:
        return subprocess.run(argv, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        log.error("worker timed out after %s s", timeout)
        return 124


def _worker_report_path(template_dir: str, iid: Optional[str]) -> str:
    return os.path.join(template_dir, VALIDATION_DIR, "report.json" if iid is None else f"report-{iid}.json")


def validate_template_in_processes(args, template_dir: str) -> Dict[str, Any]:
    """Run the worker(s) for one template (a fresh simulator process per template, or per instance) and return the
    merged ``<template>/validation/report.json``."""
    meta = _read_json(os.path.join(template_dir, "template_meta.json"))
    insts = list_instances(template_dir)
    final_path = _worker_report_path(template_dir, None)
    if args.fresh_process_per_instance:
        report = new_template_report(template_dir, meta, "v2", args.settle_steps)
        if not insts:
            report["error"] = "no instances"
        for inst in insts:
            path = _worker_report_path(template_dir, inst.iid)
            if os.path.exists(path):
                os.remove(path)
            rc = _run_worker(_worker_argv(args, template_dir, inst.iid), args.worker_timeout)
            if os.path.exists(path):
                merge_template_reports(report, _read_json(path))
                os.remove(path)
            else:
                rec = new_instance_record(inst.iid)
                rec["status"] = "error"
                rec["error"] = f"worker exited {rc} without a report"
                rec["failures"].append(rec["error"])
                report["instances"][inst.iid] = rec
        finalize_template_report(report)
    else:
        if os.path.exists(final_path):
            os.remove(final_path)
        rc = _run_worker(_worker_argv(args, template_dir), args.worker_timeout)
        if os.path.exists(final_path):
            report = _read_json(final_path)
        else:
            report = new_template_report(template_dir, meta, "v2", args.settle_steps)
            report["error"] = f"worker exited {rc} without a report"
            for inst in insts:
                rec = new_instance_record(inst.iid)
                rec["status"] = "error"
                rec["error"] = report["error"]
                rec["failures"].append(report["error"])
                report["instances"][inst.iid] = rec
            finalize_template_report(report)
    if args.delete_failed:
        failed = failed_instance_ids(report)
        report["deleted"] = sorted(set(report.get("deleted") or []) | set(failed))
        report["deleted_paths"] = sorted(set(report.get("deleted_paths") or []) | set(delete_failed_instances(template_dir, failed)))
    _write_json(final_path, report)
    return report


def _worker_main(args) -> int:
    """``--worker``: validate one template (or one instance of it) in this process and exit (Isaac teardown must not
    change the exit code)."""
    tdir = os.path.abspath(args.template[0])
    iid = args.instance[0] if args.instance else None
    code = 0
    try:
        report = validate_template_v2(tdir, instances=[iid] if iid else None, settle_steps=args.settle_steps, render=not args.no_render, check_goal=not args.no_goal,
                                      program_check=not args.no_program_check, render_iters=args.render_iters, image_size=args.image_size,
                                      report_path=_worker_report_path(tdir, iid))
        print("VALIDATION_WORKER " + json.dumps({"template": tdir, "instance": iid, "n_ok": report["n_ok"], "reload_failed": report["reload_failed"],
                                                 "n_error": report["n_error"], "error": report.get("error")}, default=str), flush=True)
    except BaseException as e:  # noqa: BLE001
        traceback.print_exc()
        code = 1
        print("VALIDATION_WORKER " + json.dumps({"template": tdir, "instance": iid, "error": repr(e)}), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    og = sys.modules.get("omnigibson")
    if og is not None and getattr(og, "app", None) is not None:
        try:
            og.shutdown()
        except BaseException:  # noqa: BLE001
            pass
    sys.stdout.flush()
    os._exit(code)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None, help="v2 output root (<root>/<alias>/<template_id>/...)")
    parser.add_argument("--out", default=None, help="v1 sampler output directory (<out>/<problem>/<template>/...); same as --root for the v1 layout")
    parser.add_argument("--alias", action="append", default=None, help="restrict to these aliases (v2) / problem directories (v1); repeatable")
    parser.add_argument("--template", action="append", default=None, help="restrict to these template ids or template directories; repeatable")
    parser.add_argument("--problem", action="append", default=None, help="(v1) problem directories; alias of --alias")
    parser.add_argument("--delete-failed", action="store_true", help="delete the instances that failed the reload (counted as reload_failed)")
    parser.add_argument("--no-render", action="store_true", help="no contact sheets")
    parser.add_argument("--no-goal", action="store_true", help="skip the goal materialisation (teleport) and the goal view")
    parser.add_argument("--no-program-check", action="store_true", help="skip the program check re-run (no cuRobo)")
    parser.add_argument("--fresh-process-per-instance", action="store_true", help="one simulator process per instance instead of per template")
    parser.add_argument("--settle-steps", type=int, default=DEFAULT_SETTLE_STEPS, help="physics steps of the free settle before the state comparison (§3.5: 60)")
    parser.add_argument("--render-iters", type=int, default=RENDER_ITERS)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--worker-timeout", type=float, default=None, help="seconds per worker process")
    parser.add_argument("--extra-steps", type=int, default=60, help="(v1) further physics steps for the stability check")
    parser.add_argument("--pos-tol", type=float, default=0.01, help="(v1) drift tolerance")
    parser.add_argument("--ang-tol", type=float, default=0.05, help="(v1) drift tolerance")
    parser.add_argument("--report", default=None, help="aggregate report path (default <root>/validation_report_<ts>.json)")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--instance", action="append", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="[%(levelname)s %(name)s] %(message)s", stream=sys.stdout, force=True)
    if args.worker:
        return _worker_main(args)
    root = args.root or args.out
    if not root:
        raise SystemExit("--root (v2) or --out (v1) is required")
    aliases = list(args.alias or []) + list(args.problem or []) or None
    templates = discover_templates(root, aliases, args.template)
    v2 = [t for t in templates if detect_layout(t) == "v2"]
    v1 = [t for t in templates if t not in v2]
    log.info("%d templates under %s (%d v2, %d v1)", len(templates), root, len(v2), len(v1))
    report: Dict[str, Any] = {"schema": REPORT_SCHEMA, "root": os.path.abspath(root), "templates": [], "n_ok": 0, "n_fail": 0, "reload_failed": 0, "n_error": 0,
                              "deleted": [], "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    path = args.report or os.path.join(root, f"validation_report_{int(time.time())}.json")
    try:
        for tdir in v2:
            try:
                res = validate_template_in_processes(args, tdir)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                res = {"template": tdir, "error": repr(e), "instances": {}, "n_ok": 0, "reload_failed": 0, "n_error": 0}
            report["templates"].append(res)
            report["n_ok"] += res.get("n_ok", 0)
            report["reload_failed"] += res.get("reload_failed", 0)
            report["n_error"] += res.get("n_error", 0)
            report["n_fail"] += res.get("reload_failed", 0) + res.get("n_error", 0)
            report["deleted"] += list(res.get("deleted") or [])
            print(summary_line(res), flush=True)
        if v1:
            from ispatialgym.sampling import robot_utils  # noqa: F401  (cuRobo / warp pinning before omnigibson)

            import omnigibson as og

            try:
                for tdir in v1:
                    try:
                        res = validate_template(tdir, args.extra_steps, not args.no_goal, args.pos_tol, args.ang_tol)
                    except Exception as e:  # noqa: BLE001
                        traceback.print_exc()
                        res = {"template": tdir, "error": repr(e), "instances": {}}
                        try:
                            og.clear()
                        except Exception:  # noqa: BLE001
                            pass
                    res["layout"] = "v1"
                    report["templates"].append(res)
                    for rec in res["instances"].values():
                        report["n_ok" if rec["ok"] else "n_fail"] += 1
            finally:
                og.shutdown()
    finally:
        report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _write_json(path, report)
        fails = []
        for t in report["templates"]:
            for iid, r in t.get("instances", {}).items():
                if not r.get("ok"):
                    fails.append(f"{os.path.basename(t.get('template', ''))}/{iid}: {r.get('failures') or r.get('notes')}")
            if t.get("error"):
                fails.append(f"{t.get('template')}: {t['error']}")
        print("VALIDATION_REPORT " + json.dumps({"n_ok": report["n_ok"], "n_fail": report["n_fail"], "reload_failed": report["reload_failed"], "n_error": report["n_error"],
                                                 "deleted": len(report["deleted"]), "report": path, "fails": fails}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    main()
