"""Floating grasps of the grasp library v2 (i7-floating 2026-11-07, owner: "floating grasps from GraspGen, as the legacy library used"):
the held-object tasks (``object_scaling``, ``mental_rotation``; ``sampling/v2/tasks/held_object.py`` §2.2) hold the object in the air -
it never rests on a support - so their grasps are a constraint of their own, ``floating``, with no support.

Port of the legacy floating pipeline (``runlogs/grasp_library_20260905``: ``scripts/graspgen_batch.py`` + ``gen_candidates.run_graspgen`` +
``ag_validate_render.py``; design doc §3.2):

1. **Proposals** - GraspGen (NVlabs, the Franka Panda checkpoint, diffusion generator + discriminator) on 2000 surface points of the
   object's COLLISION geometry (the v2 export, ``geometry.json``; no visual asset is decoded): 2000 grasps, the top 500 by the
   discriminator (``scripts/grasp_bands/graspgen_floating.py`` in the ``graspgen`` env writes ``graspgen.npz``: Panda base poses in the
   object frame + confidences).
2. **Re-targeting** to the R1Pro eef (:data:`T_PANDA_TO_EEF`): the Panda base -> its TCP (``depth`` 0.10527 m along +z), our pad centre
   on it (``conventions.CONTACT_CENTER_Z``: the eef origin 1.186 cm past the pad centre), and a quarter turn about +z because GraspGen
   closes along x and the R1Pro along y.
3. **The v2 gates that apply to a floating object** (shared with ``candidates`` - the same functions and thresholds, no gate relaxed):
   width / antipodal cone at the pad centre, the hand's cuRobo spheres clear of the object at the HELD sampler's opening (the fingers
   fully open: ``HeldObjectV2.establish_grasp`` teleports the object into the fully open hand), the closing sweep (the pads meet the
   intended contact first), the lever-arm cap, the approach from the pre-grasp clear of the object, and the finger blades' exact hulls
   outside the object from the opening down to the pads' first touch. There is no support, so the support policy (finger allowance,
   palm / camera / wrist vs the support, the support retreat / lift) does not apply.
4. **The legacy ranking modes** (per legacy key: ``C`` confidence gate then CoM, ``V`` vessels the same, ``H`` confidence-first,
   ``H+handle`` the knives' handle half) and NMS to ``topk``; no symmetry folding (the held sampler draws library rows, never family
   members - every representative is a row it can use).

The physics validation (in hand, under gravity, then the task's held motions) is :mod:`validate_floating`. ``floating`` is the entry's
constraint, so its id is ``<category>__<model>__<scale tag>__floating`` (:func:`schema.entry_id`) - it never aliases a tabletop /
attachment entry of the same model - and a floating entry has no rest (``rests`` empty, every grasp ``rest_id`` None).
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ispatialgym.grasping import candidates as CA
from ispatialgym.grasping import contact_policy as CP
from ispatialgym.grasping import object_geometry as OG
from ispatialgym.grasping import schema as S

CONSTRAINT = "floating"
PIPELINE = "graspgen_v2"
#: ``<model dir>/graspgen.npz`` (``T`` (N, 4, 4) Panda base poses in the object frame, ``conf`` (N,)) + ``graspgen.json`` (its record)
GRASPGEN_FILE = "graspgen.npz"
GRASPGEN_META = "graspgen.json"
#: the Panda's base -> TCP depth (GraspGen ``config/grippers/franka_panda.yaml`` ``depth``; legacy ``gen_candidates.PANDA_DEPTH``)
PANDA_DEPTH = 0.10527314
#: legacy floating modes (``build_rounds.method_for`` / the curated plan): the ranking of ``gen_candidates.run_graspgen``
MODES = ("C", "V", "H", "H+handle")


def _rot_z(a: float) -> np.ndarray:
    T = np.eye(4)
    c, s = math.cos(a), math.sin(a)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def _trans(v) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = v
    return T


#: Panda base pose -> R1Pro eef pose (right-multiplied): ``R_z(pi/2) @ T_z(PANDA_DEPTH - CONTACT_Z)`` - the legacy ``T_CONV``
T_PANDA_TO_EEF = _rot_z(math.pi / 2) @ _trans([0.0, 0.0, PANDA_DEPTH - CA.CONTACT_Z])


def panda_to_eef(T_panda: np.ndarray) -> np.ndarray:
    """GraspGen's Panda base poses (N, 4, 4) -> R1Pro eef poses in the same frame: the pad centre on the Panda TCP, the approach kept
    (+z), the closing axis Panda x -> eef y."""
    T = np.asarray(T_panda, dtype=float).reshape(-1, 4, 4)
    return np.einsum("nij,jk->nik", T, T_PANDA_TO_EEF)


@dataclass
class FloatingConfig:
    aperture_m: float = CA.GRIPPER_MAX_OPENING - 0.008
    min_width_m: float = 0.003
    # the shared cone (candidates.CandidateConfig.cone_deg, ISG_GLIB_CONE_DEG)
    cone_deg: float = field(default_factory=lambda: float(os.environ.get("ISG_GLIB_CONE_DEG") or 30.0))
    obj_margin_m: float = 0.001
    #: the opening the hand holds while the object is put into it: the held sampler opens the fingers fully (``establish_grasp``: q = upper)
    open_q: float = CA.FINGER_OPEN
    sweep_margin_m: float = 0.008
    lever_cap: float = 0.40
    #: legacy modes C / V: the CoM-ranked pool is restricted to pinches whose CoM lies within this fraction of the size from the pinch line
    #: (``gen_candidates.LEVER_CAP`` on the line distance) when any does - a ranking rule of the mode, not a gate
    mode_lever_cap: float = 0.40
    #: legacy H+handle (knives): the contact in the handle half of the long axis
    handle_t_max: float = 0.5
    approach_obj_check: bool = True
    topk: int = 12
    nms_pos_m: float = 0.015
    nms_rot_deg: float = 30.0
    max_representatives: int = 10
    gate_dump: bool = field(default_factory=lambda: os.environ.get("ISG_GLIB_GATE_DUMP", "") == "1")
    # Default-off recovery for multipart objects whose internal collision-piece
    # faces make centre-origin rays report a pinch narrower than the outer body.
    # The closing pads encounter the outside faces first. All subsequent sweep,
    # hand/object, approach, blade and native physics checks remain unchanged.
    outer_pinch: bool = False


def contact_measurement(union, poses, cfg: FloatingConfig):
    if type(cfg.outer_pinch) is not bool:
        raise ValueError("floating outer_pinch must be boolean")
    if cfg.outer_pinch and cfg.open_q != CA.FINGER_OPEN:
        raise ValueError("floating outer_pinch requires the held sampler's full opening")
    return CA.contact_features(union, poses, CA.CONTACT_Z,
                               outer_half=CA.PAD_OPEN_HALF if cfg.outer_pinch else None)


def load_graspgen(model_dir: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """``(T_panda (N, 4, 4), conf (N,), record)`` of the model's GraspGen output (FileNotFoundError when the stage has not run)."""
    p = os.path.join(model_dir, GRASPGEN_FILE)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{model_dir}: no {GRASPGEN_FILE} (run scripts/grasp_bands/graspgen_floating.py in the graspgen env)")
    d = np.load(p)
    T = np.asarray(d["T"], dtype=float).reshape(-1, 4, 4)
    conf = np.asarray(d["conf"], dtype=float).reshape(-1)
    return T, conf, S.load_json(os.path.join(model_dir, GRASPGEN_META)) or {}


def line_perp(T: np.ndarray, com: np.ndarray) -> np.ndarray:
    """Distance of the CoM from each pinch line (through the pad centre along the closing axis): the legacy floating ranking's ``perp``."""
    c = T[:, :3, 3] + T[:, :3, 2] * CA.CONTACT_Z
    y = T[:, :3, 1]
    v = com[None] - c
    along = np.einsum("ij,ij->i", v, y)
    return np.linalg.norm(v - along[:, None] * y, axis=1)


def handle_t(T: np.ndarray, union, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Legacy ``handle_half``: the contact's position along the long axis, 0 at the handle end (the thicker end), 1 at the blade tip."""
    ext = hi - lo
    ax = int(np.argmax(ext))
    L = float(ext[ax])
    v = np.asarray(union.vertices)
    tt = (v[:, ax] - lo[ax]) / L
    perp_v = np.delete(v, ax, axis=1)

    def thick(mask):
        return float(np.ptp(perp_v[mask], axis=0).min()) if mask.sum() > 3 else 0.0

    handle_hi = thick(tt > 0.65) >= thick(tt < 0.35)
    d = np.zeros(3)
    d[ax] = -1.0 if handle_hi else 1.0
    hpt = (lo + hi) / 2
    hpt[ax] = hi[ax] if handle_hi else lo[ax]
    c = T[:, :3, 3] + T[:, :3, 2] * CA.CONTACT_Z
    return np.einsum("ij,j->i", c - hpt[None], d) / L


def mode_scores(mode: str, conf: np.ndarray, perp: np.ndarray, size: float, antip: np.ndarray, cov: np.ndarray, clear: np.ndarray,
                cfg: FloatingConfig, th_: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """The legacy floating ranking (``gen_candidates.run_graspgen``) per mode: ``(score, keep)`` - ``keep`` is the C / V mode's own lever
    restriction (all True when no candidate passes it, as the legacy rule did)."""
    n = len(conf)
    keep = np.ones(n, dtype=bool)
    if n == 0:
        return np.zeros(0), keep
    if mode == "H+handle":
        return conf * 100.0 + 20.0 * np.exp(-((th_ - 0.3) / 0.15) ** 2), keep
    if mode == "H":
        score = conf * 100.0 - np.clip(perp, 0, 0.3)
        band = conf >= conf.max() - 0.01
        return np.where(band, conf.max() * 100.0 - perp, score), keep
    k = perp <= cfg.mode_lever_cap * size
    if k.any():
        keep = k
    cmax = conf[keep].max()
    gate = conf >= cmax - 0.05
    f_com = np.exp(-perp / max(0.01, 0.25 * size))
    score = np.where(gate, 10.0 + 5.0 * f_com + antip + cov + np.clip(clear / 0.02, 0, 1) + 2.0 * conf, conf)
    return score, keep


def generate(job: Dict[str, Any], model_dir: str, cfg: Optional[FloatingConfig] = None, policy: CP.ContactPolicy = CP.DEFAULT,
             arm: str = "right") -> Dict[str, Any]:
    """Floating candidates of one model (``job``: category, model, scale, ``constraint`` "floating", ``floating_mode``) from its GraspGen
    output -> ``candidates.json`` (schema CANDIDATES, pipeline :data:`PIPELINE`) and, with ``cfg.gate_dump``, ``gate_dump.json``."""
    cfg = FloatingConfig() if cfg is None else cfg  # built per call: its env-read defaults (ISG_GLIB_GATE_DUMP, ISG_GLIB_CONE_DEG) are current
    t0 = time.time()
    geom = OG.load(model_dir)
    if geom is None:
        raise FileNotFoundError(f"no geometry.json in {model_dir} (run the export stage)")
    mode = job.get("floating_mode") or "C"
    if mode not in MODES:
        raise ValueError(f"unknown floating mode {mode!r} (known: {MODES})")
    T_panda, conf, gg_meta = load_graspgen(model_dir)
    union = geom.union
    lo, hi = union.vertices.min(0), union.vertices.max(0)
    size = float(np.max(hi - lo))
    com = union.center_mass if union.volume > 1e-9 else union.centroid
    dump = CA._GateDump(cfg.gate_dump)
    pool = "graspgen"
    T = panda_to_eef(T_panda)
    log: Dict[str, Any] = {"n_graspgen": int(len(T)), "size": round(size, 4), "mode": mode, "com": [round(float(v), 5) for v in com]}
    dump.gate(pool, "pool", len(T), len(T))
    idx = np.arange(len(T))

    def _where(Tm):
        cm = Tm[:, :3, 3] + Tm[:, :3, 2] * CA.CONTACT_Z
        return {"contact_mm": cm * 1000, "approach_down_deg": CA.tilt_deg(Tm)}

    def step(name, keep, measures=None, why=None):
        nonlocal T, conf, idx
        if dump.on:
            dump.gate(pool, name, len(T), int(keep.sum()), np.where(~keep)[0], {**_where(T), **(measures or {})},
                      None if why is None else np.asarray(why)[~keep])
        T, conf, idx = T[keep], conf[keep], idx[keep]
        log[f"n_{name}"] = int(len(T))
        return keep

    # width / antipodal cone at the pad centre (candidates.contact_features, the tabletop modes' first gate)
    _, width, antip, cov = contact_measurement(union, T, cfg)
    ok_w = np.isfinite(width) & (width >= cfg.min_width_m) & (width <= cfg.aperture_m)
    keep = ok_w & (antip >= math.cos(math.radians(cfg.cone_deg)))
    why = np.where(~np.isfinite(width), "no_width", np.where(~ok_w, "width_range", "antipodal"))
    k = step("width_antipodal", keep, {"width_mm": width * 1000, "antipodal": antip, "graspgen_conf": conf, "why": why}, why)
    width, antip, cov = width[k], antip[k], cov[k]
    # the hand's cuRobo spheres clear of the object at the held sampler's opening (fully open)
    clear_fn = CA.SphereClearance(geom, arm)
    clear = clear_fn(T, cfg.open_q) if len(T) else np.zeros(0)
    k = step("object_clear", clear >= cfg.obj_margin_m, {"sphere_clear_mm": clear * 1000})
    width, antip, cov, clear = width[k], antip[k], cov[k], clear[k]
    # the closing sweep: the pads meet the intended contact first
    touch = CA.pad_sweep(union, T) if len(T) else np.zeros(0)
    k = step("sweep", touch <= width / 2 + cfg.sweep_margin_m, {"width_mm": width * 1000, "touch_mm": touch * 1000,
                                                                "limit_mm": (width / 2 + cfg.sweep_margin_m) * 1000})
    width, antip, cov, clear = width[k], antip[k], cov[k], clear[k]
    # the lever-arm cap (candidates._score's perp: the CoM offset along the pad-width axis)
    c = T[:, :3, 3] + T[:, :3, 2] * CA.CONTACT_Z
    perp_x = np.abs(np.einsum("ij,ij->i", com[None] - c, T[:, :3, 0])) if len(T) else np.zeros(0)
    k = step("lever", perp_x <= cfg.lever_cap * size, {"com_perp_mm": perp_x * 1000, "cap_mm": np.full(len(T), cfg.lever_cap * size * 1000)})
    width, antip, cov, clear, perp_x = width[k], antip[k], cov[k], clear[k], perp_x[k]
    th_ = None
    if mode == "H+handle":
        th_ = handle_t(T, union, lo, hi) if len(T) else np.zeros(0)
        k = step("handle_half", th_ <= cfg.handle_t_max, {"handle_t": th_})
        width, antip, cov, clear, perp_x, th_ = width[k], antip[k], cov[k], clear[k], perp_x[k], th_[k]
    # the full geometry of every survivor: the approach from the pre-grasp and the finger blades through the closure
    hc = CP.HandChecker(arm, policy, spacing=0.004)
    pre_q = np.full(len(T), float(cfg.open_q))
    contact_q = np.clip(0.5 * width + CA.pad_face_offset(arm), 0.0, CA.FINGER_OPEN)
    appr = CA._approach_object_clear(clear_fn, T, pre_q, cfg) if (cfg.approach_obj_check and len(T)) else np.ones(len(T), dtype=bool)
    blades, deep = CA.finger_blades_clear(geom, hc, T, pre_q, contact_q) if len(T) else (np.ones(0, dtype=bool), np.zeros(0))
    reason = np.where(~appr, "approach_hits_object", np.where(~blades, "finger_blade_in_object", "ok"))
    k = step("geometry", appr & blades, {"blade_depth_mm": deep * 1000, "approach_object_clear": appr, "why": reason}, reason)
    width, antip, cov, clear, perp_x, contact_q, deep = width[k], antip[k], cov[k], clear[k], perp_x[k], contact_q[k], deep[k]
    th_ = th_[k] if th_ is not None else None
    # the legacy mode's ranking, then NMS
    perp = line_perp(T, com) if len(T) else np.zeros(0)
    score, mkeep = mode_scores(mode, conf, perp, size, antip, cov, clear, cfg, th_)
    if len(T) and not mkeep.all():
        log["n_mode_lever"] = int(mkeep.sum())
        dump.gate(pool, "mode_lever", len(T), int(mkeep.sum()))
    sel_pool = np.where(mkeep)[0]
    sel = sel_pool[CA.nms(T[sel_pool], score[sel_pool], cfg.topk, cfg.nms_pos_m, cfg.nms_rot_deg)] if len(sel_pool) else np.zeros(0, dtype=int)
    log["n_selected"] = int(len(sel))
    dump.gate(pool, "nms", len(sel_pool), len(sel))
    cands = []
    for r, i in enumerate(sel):
        cands.append({
            "name": f"gg_{r}", "mode": "floating", "floating_mode": mode, "score": round(float(score[i]), 4), "graspgen_index": int(idx[i]),
            "graspgen_conf": round(float(conf[i]), 5), "T_object_eef": S.mat_pose(T[i]), "width_m": round(float(width[i]), 5),
            "preshape_q": round(float(cfg.open_q), 5), "contact_q": round(float(contact_q[i]), 5),
            "features": {"width": round(float(width[i]), 5), "antipodal": round(float(antip[i]), 5), "coverage": round(float(cov[i]), 5),
                         "clearance": round(float(clear[i]), 5), "com_perp": round(float(perp_x[i]), 5), "com_line_dist": round(float(perp[i]), 5),
                         "approach_down_deg": round(float(CA.tilt_deg(T[i:i + 1])[0]), 3),
                         **({"handle_t": round(float(th_[i]), 5)} if th_ is not None else {})},
            "geometry": {"ok": True, "reason": "ok", "stage": "", "blade_object_depth_m": round(float(deep[i]), 5), "support": None},
        })
    reps = [c["name"] for c in cands][: cfg.max_representatives]
    out = {"schema": S.CANDIDATES, "pipeline": PIPELINE, "category": job["category"], "model": job["model"],
           "scale": list(S.canonical_scale(job.get("scale"))), "dataset": job.get("dataset"), "constraint": CONSTRAINT, "rest_hint_wxyz": None,
           "generation_rest_wxyz": None, "legacy_key": job.get("legacy_key"), "floating_mode": mode, "arm": arm, "policy": policy.meta(arm),
           "policy_key": policy.key(arm), "config": asdict(cfg), "graspgen": gg_meta, "candidates": cands, "families": {}, "representatives": reps,
           "log": {pool: log}, "t_s": round(time.time() - t0, 2), "created": S.now(),
           "support": "none (floating: the support policy's gates do not apply)",
           "families_note": "no symmetry folding: the held sampler draws library rows, never family members"}
    S.save_json(os.path.join(model_dir, "candidates.json"), out)
    if cfg.gate_dump:
        S.save_json(os.path.join(model_dir, CA.GATE_DUMP_FILE), {
            "schema": CA.GATE_DUMP_SCHEMA, "id": job.get("id"), "category": job.get("category"), "model": job.get("model"), "constraint": CONSTRAINT,
            "floating_mode": mode, "passes": [{"n_graspgen": int(log["n_graspgen"]), "log": {pool: log}, "pools": dump.pools}], "final_pass": 0,
            "notes": "floating (i7-floating): one pool (graspgen), gates width_antipodal, object_clear (spheres at the held sampler's full opening), "
                     "sweep, lever, [handle_half], geometry (approach_hits_object / finger_blade_in_object), then the legacy mode ranking + NMS. "
                     "No support gates. Lengths in mm.", "created": S.now()})
    return out


def floating_job(key: str, legacy: Dict[str, Any], held_tasks: Sequence[str] = ()) -> Dict[str, Any]:
    """The v2 floating job of a legacy floating key (``legacy`` = its ``results/<key>.json``): the model, the spawn scale the held
    sampler uses (the key's ``scale``, e.g. ``__sf`` thickness scaling), its dataset and legacy mode (``handle_half`` -> ``H+handle``);
    ``legacy_key`` = the key (what the entry serves). The legacy grasps themselves are not used."""
    parts = key.split("__")
    cat = legacy.get("category") or parts[0]
    model = legacy.get("model") or parts[1]
    mode = (legacy.get("mode") or "C")[0]
    if (legacy.get("constraint") == "handle_half") and mode == "H":
        mode = "H+handle"
    j = {"category": cat, "model": model, "scale": list(S.canonical_scale(legacy.get("scale"))), "dataset": legacy.get("dataset") or "behavior-1k-assets",
         "constraint": CONSTRAINT, "rest_hint_wxyz": None, "rest_pose": None, "legacy_key": key, "kind": CONSTRAINT, "source": "floating_key",
         "floating_mode": mode, "held_tasks": sorted(set(held_tasks))}
    j["id"] = S.entry_id(cat, model, j["scale"], CONSTRAINT)
    return j
