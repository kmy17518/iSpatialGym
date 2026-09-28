"""The start state of a loaded instance (engineering track, owner 2026-09-27): what the executor starts from at the end of
``InstanceLoader.restore`` - the robot's base pose, joint positions and velocities and controller goals, the assisted grasp per
arm, every task object's root pose and joint positions, the loader's hinges / constraint and attachment records, and the render
settings the demo depends on. A warm demo (the sampler and the demo of a task in one Isaac process) must start from what a fresh
reload of the same TRO starts from; :func:`compare` decides that on tolerances (``validate_instances.EXACT_TOLERANCES``: 0.1 mm /
0.1 mrad right after the load) plus exact equality of the discrete items. The sha256 of the quantized values only says whether two
states are bit-identical at the quantization.

    fp = fingerprint(loader)            # after loader.restore(); run_isg writes it to meta["start_state"]
    rep = compare(fp_warm, fp_fresh)    # {"ok": bool, "max": {...}, "fails": [...]}
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Sequence

#: quantization of the fingerprint's values (m, rad, quaternion components)
QUANTUM = 1e-6
#: pass / fail tolerances (``validate_instances.EXACT_TOLERANCES``)
TOLERANCES: Dict[str, float] = {"pos_m": 1e-4, "rot_rad": 1e-4, "joint": 1e-4, "vel": 1e-3}


def _q(v: Any, quantum: float = QUANTUM) -> Any:
    """Floats rounded to ``quantum``, recursively (tensors / arrays through ``tolist``)."""
    if hasattr(v, "detach"):
        v = v.detach().cpu().tolist()
    elif hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
        v = v.tolist()
    if isinstance(v, float):
        return round(round(v / quantum) * quantum, 9) if math.isfinite(v) else str(v)
    if isinstance(v, dict):
        return {str(k): _q(x, quantum) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_q(x, quantum) for x in v]
    if isinstance(v, (int, bool, str)) or v is None:
        return v
    name = getattr(v, "name", None)
    if isinstance(name, str):  # a sim object (e.g. the assisted grasp's target): its repr carries a per-process address
        return f"<{type(v).__name__} {name}>"
    return str(v)


def _try(fn, default=None):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - an unreadable item is recorded as such (compare flags it)
        return {"unreadable": repr(e)[:200]} if default is None else default


def _robot(robot) -> Dict[str, Any]:
    names = list(getattr(robot, "joints", {}) or {})
    out: Dict[str, Any] = {"base": _try(lambda: [list(x.tolist()) for x in robot.get_position_orientation()]),
                           "joint_names": names,
                           "joints": _try(lambda: robot.get_joint_positions().tolist()),
                           "joint_velocities": _try(lambda: robot.get_joint_velocities().tolist())}
    goals = {}
    for name, c in (getattr(robot, "controllers", None) or {}).items():
        g = getattr(c, "_goal", None)
        goals[name] = _try(lambda g=g: {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in (g or {}).items()})
    out["controller_goals"] = goals
    in_hand = getattr(robot, "_ag_obj_in_hand", None) or {}
    out["assisted_grasp"] = {arm: (getattr(o, "name", None) if o is not None else None) for arm, o in in_hand.items()}
    params = getattr(robot, "_ag_obj_constraint_params", None) or {}
    out["assisted_grasp_params"] = {arm: _try(lambda p=p: {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in (p or {}).items()}) for arm, p in params.items()}
    return out


def _objects(physical: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for inst in sorted(physical):
        ent = physical[inst]
        d = {"pose": _try(lambda ent=ent: [list(x.tolist()) for x in ent.get_position_orientation()])}
        if int(getattr(ent, "n_joints", 0) or 0) > 0:
            d["joints"] = _try(lambda ent=ent: ent.get_joint_positions().tolist())
        out[inst] = d
    return out


def _render() -> Dict[str, Any]:
    def read():
        import carb

        s = carb.settings.get_settings()
        return {"showLights": s.get("/rtx/raytracing/showLights")}
    return _try(read, default={"showLights": None})


def fingerprint(loader, render: bool = True) -> Dict[str, Any]:
    """The quantized start state of ``loader`` (an ``InstanceLoader`` after ``restore``) and its sha256. Reads only: no physics step,
    no render."""
    robot = loader.robot
    values: Dict[str, Any] = {"robot": _robot(robot), "objects": _objects(getattr(loader, "physical", {}) or {}),
                              "hinges": len(getattr(loader, "hinges", []) or []),
                              "constraints": _try(lambda: [{k: v for k, v in dict(c).items() if k in ("type", "object", "child", "parent", "ok")}
                                                           for c in (loader.report.get("constraints") or [])]),
                              "attachments": _try(lambda: loader.report.get("attachments")),
                              "grasps": _try(lambda: [{"arm": g.get("arm"), "object": g.get("object")} for g in (loader.grasps or [])]),
                              "controller_targets_restored": _try(lambda: loader.report.get("controller_targets_restored")),
                              "modalities": _try(lambda: sorted(getattr(robot, "obs_modalities", None) or []))}
    if render:
        values["render"] = _render()
    qv = _q(values)
    return {"sha256": hashlib.sha256(json.dumps(qv, sort_keys=True, default=str).encode()).hexdigest(), "quantum": QUANTUM, "values": qv}


def _quat_angle(a: Sequence[float], b: Sequence[float]) -> float:
    """The rotation angle between two quaternions, normalized first: quantized components leave |q| a few 1e-7 off 1, which
    ``acos`` of the raw dot product turns into mrad for identical rotations."""
    na = math.sqrt(sum(float(x) ** 2 for x in a)) or 1.0
    nb = math.sqrt(sum(float(y) ** 2 for y in b)) or 1.0
    ua, ub = [float(x) / na for x in a], [float(y) / nb for y in b]
    s = 1.0 if sum(x * y for x, y in zip(ua, ub)) >= 0.0 else -1.0
    diff = math.sqrt(sum((x - s * y) ** 2 for x, y in zip(ua, ub)))
    summ = math.sqrt(sum((x + s * y) ** 2 for x, y in zip(ua, ub)))
    return 4.0 * math.atan2(diff, summ)


def _pose_diff(a, b):
    if not (isinstance(a, list) and isinstance(b, list) and len(a) == 2 and len(b) == 2):
        return None, None
    dp = math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a[0], b[0])))
    return dp, _quat_angle(a[1], b[1])


def _max_abs(a, b) -> Optional[float]:
    if not (isinstance(a, list) and isinstance(b, list)) or len(a) != len(b):
        return None
    return max((abs(float(x) - float(y)) for x, y in zip(a, b)), default=0.0)


def compare(a: Dict[str, Any], b: Dict[str, Any], tol: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Two fingerprints (``fingerprint`` outputs, or their ``values``): ``ok`` when every pose / joint / velocity is within ``tol``
    and every discrete item is equal; ``max`` = the largest differences, ``fails`` = what broke the tolerance or the equality."""
    tol = dict(TOLERANCES, **(tol or {}))
    va, vb = a.get("values", a), b.get("values", b)
    fails: List[str] = []
    mx = {"pos_m": 0.0, "rot_rad": 0.0, "joint": 0.0, "vel": 0.0}

    def upd(key, v, what):
        if v is None:
            fails.append(f"{what}: not comparable")
            return
        mx[key] = max(mx[key], v)
        if v > tol[key]:
            fails.append(f"{what}: {key} {v:.3g} > {tol[key]:.3g}")

    ra, rb = va.get("robot", {}), vb.get("robot", {})
    dp, dr = _pose_diff(ra.get("base"), rb.get("base"))
    upd("pos_m", dp, "robot base")
    upd("rot_rad", dr, "robot base")
    if ra.get("joint_names") != rb.get("joint_names"):
        fails.append("robot joint names differ")
    upd("joint", _max_abs(ra.get("joints"), rb.get("joints")), "robot joints")
    upd("vel", _max_abs(ra.get("joint_velocities"), rb.get("joint_velocities")), "robot joint velocities")
    for name in sorted(set(ra.get("controller_goals", {})) | set(rb.get("controller_goals", {}))):
        ga, gb = ra.get("controller_goals", {}).get(name), rb.get("controller_goals", {}).get(name)
        for k in sorted(set(ga or {}) | set(gb or {})):
            upd("joint", _max_abs((ga or {}).get(k), (gb or {}).get(k)) if isinstance((ga or {}).get(k), list) else (0.0 if (ga or {}).get(k) == (gb or {}).get(k) else None),
                f"controller {name}.{k}")
    for key in ("assisted_grasp",):
        if ra.get(key) != rb.get(key):
            fails.append(f"robot {key} differs: {ra.get(key)} vs {rb.get(key)}")
    oa, ob = va.get("objects", {}), vb.get("objects", {})
    if sorted(oa) != sorted(ob):
        fails.append(f"objects differ: {sorted(set(oa) ^ set(ob))}")
    for inst in sorted(set(oa) & set(ob)):
        dp, dr = _pose_diff(oa[inst].get("pose"), ob[inst].get("pose"))
        upd("pos_m", dp, f"object {inst}")
        upd("rot_rad", dr, f"object {inst}")
        if "joints" in oa[inst] or "joints" in ob[inst]:
            upd("joint", _max_abs(oa[inst].get("joints"), ob[inst].get("joints")), f"object {inst} joints")
    for key in ("hinges", "constraints", "attachments", "grasps", "controller_targets_restored", "modalities", "render"):
        if va.get(key) != vb.get(key):
            fails.append(f"{key} differ: {str(va.get(key))[:120]} vs {str(vb.get(key))[:120]}")
    return {"ok": not fails, "max": {k: round(v, 9) for k, v in mx.items()}, "fails": fails,
            "identical": a.get("sha256") is not None and a.get("sha256") == b.get("sha256")}
