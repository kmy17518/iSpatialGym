"""The grasp ladder, steps 3 and 4 (owner decision 2026-11-07, docs/GRASP_BANDS.md "a grasp ladder for every v2 entry").

Every library entry climbs: (1) round 1, the standard build; (2) round 2, the retry ladder + per-object fixes; (3) AGENT PROPOSALS for
an entry still at 0 physics-valid grasps after round 2; (4) last resort, its LEGACY grasps (``runlogs/grasp_library_20260905``)
re-validated - reported to the owner before any alias uses them.

Steps 3 and 4 reuse the articulation library's handle-review interface (``runlogs/grasp_library_20260905/artic/handle_review*_contract.md``):
a geometric heuristic proposes candidates (here: the round's own near misses from the gate dump, lettered markers), rendered strips show
them, and the proposer answers a JSON contract (``scripts/grasp_ladder/CONTRACT.md``, :data:`CONTRACT`). The proposals become GIVEN
poses (``candidates.agent_proposal_poses``) that go through the same v2 gates and the same physics validation as every other candidate:
no gate is relaxed, the entry records ``source: "agent"`` (step 4: ``"legacy"``) on its candidates and grasps.

This module is the CPU side (selection, inputs, strips, contract checks, manifests, reports); ``scripts/grasp_ladder/agent_step.py``
and ``legacy_step.py`` are the command lines. Nothing here writes object meshes: the inputs carry piece summaries, the strips are PNGs
of the pipeline's collision export (``geometry.json``), which stays where the pipeline wrote it.
"""

from __future__ import annotations

import copy
import json
import math
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ispatialgym import REPO_ROOT
from ispatialgym.grasping import schema as S

CONTRACT = "isg-grasp-proposal/1"
LEGACY_ROOT = os.path.join(REPO_ROOT, "runlogs", "grasp_library_20260905")
MARKERS = "ABCDEFGH"
MAX_PROPOSALS = 8
#: the order of the candidate gates (candidates.py) - a candidate that died later got further
GATE_ORDER = ("pool", "width_antipodal", "support_retreat", "support_lift", "object_clear", "sweep", "lever", "attachment_free",
              "flower_stem_top", "nms", "geometry", "physics")
CONFIDENCE = ("high", "medium", "low")


# ------------------------------------------------------------------------------------------------ status of an entry over the rounds


def n_valid(model_dir: str) -> Optional[int]:
    """Physics-valid grasps of a built entry (``library.json``); None when the entry was never validated there."""
    lib = S.load_json(os.path.join(model_dir, "library.json"))
    return None if lib is None else len(lib.get("grasps") or [])


def entry_status(entry_id: str, roots: Sequence[str]) -> Dict[str, Optional[int]]:
    """``{glib root: n_valid or None}`` for one entry id over the rounds' library roots (``<root>/models/<id>``)."""
    return {r: n_valid(os.path.join(r, "models", entry_id)) for r in roots}


def select_zero(jobs: Sequence[Dict[str, Any]], roots: Sequence[str], exclude: Iterable[str] = ()) -> List[Dict[str, Any]]:
    """The entries still at 0 physics-valid grasps after the given rounds: built in at least one root and valid in none (an entry
    that was never built is not a step-3 entry - it has not had its round 1). Returns ``[{"job", "src_dir", "status"}]`` with
    ``src_dir`` the latest root (in ``roots`` order) that holds its exported geometry."""
    out = []
    ex = set(exclude)
    for j in jobs:
        if j["id"] in ex:
            continue
        st = entry_status(j["id"], roots)
        built = {r: v for r, v in st.items() if v is not None}
        if not built or any(v > 0 for v in built.values()):
            continue
        src = [os.path.join(r, "models", j["id"]) for r in roots if os.path.exists(os.path.join(r, "models", j["id"], "geometry.json"))]
        if not src:
            continue
        out.append({"job": j, "src_dir": src[-1], "status": st})
    return out


# ------------------------------------------------------------------------------------------------ inputs: geometry, role, history, markers


def geometry_summary(geom, R: np.ndarray) -> Dict[str, Any]:
    """The rest-frame geometry the proposer reads (mm): extents, support and top heights, the CoM, per piece its bbox / centroid / volume,
    the plate axis and the attachment meta links (rest frame). ``geom`` is the canonical object geometry, ``R`` the rest rotation."""
    from ispatialgym.grasping import candidates as CA

    g = geom.rotated(R)
    u = g.union
    lo, hi = u.vertices.min(0), u.vertices.max(0)
    com = u.center_mass if u.volume > 1e-9 else u.centroid
    pieces = []
    for i, p in enumerate(g.pieces):
        plo, phi = p.vertices.min(0), p.vertices.max(0)
        pieces.append({"i": i, "bbox_min_mm": _mm(plo), "bbox_max_mm": _mm(phi), "centroid_mm": _mm(p.centroid),
                       "volume_cm3": round(float(abs(p.volume)) * 1e6, 3)})
    cx, cy, r_max = CA.plate_axis(u)
    ev, evec = np.linalg.eigh(np.cov((u.vertices - u.vertices.mean(0)).T))
    mates = []
    if geom.info.get("meta_links"):
        info = dict(geom.info)
        info["coll_bbox_min"], info["coll_bbox_max"] = lo.tolist(), hi.tolist()
        for p, m in CA.attachment_frames(info, R):
            mates.append({"point_mm": _mm(p), "mating_direction": [round(float(v), 4) for v in m]})
    return {"frame": "rest frame: the object in its designated rest, z up, the support plane at z = support_z_mm; millimetres",
            "bbox_min_mm": _mm(lo), "bbox_max_mm": _mm(hi), "extents_mm": _mm(hi - lo), "support_z_mm": round(float(lo[2]) * 1000, 2),
            "top_z_mm": round(float(hi[2]) * 1000, 2), "com_mm": _mm(com), "mass_kg": geom.info.get("mass"), "n_pieces": len(pieces),
            "pieces": pieces, "plate_axis_xy_mm": [round(cx * 1000, 2), round(cy * 1000, 2)], "max_radius_about_axis_mm": round(r_max * 1000, 2),
            "principal_axes": [{"axis": [round(float(v), 4) for v in evec[:, k]], "std_mm": round(math.sqrt(max(ev[k], 0.0)) * 1000, 2)}
                               for k in (2, 1, 0)],
            "attachment_mates": mates, "meta_links": sorted((geom.info.get("meta_links") or {}).keys())}


def _mm(v) -> List[float]:
    return [round(float(x) * 1000, 2) for x in np.asarray(v, dtype=float).reshape(-1)]


def gripper_facts() -> Dict[str, Any]:
    """The R1Pro parallel gripper in the eef frame (+z approach, +y closing) the contract is written against."""
    from ispatialgym.grasping import candidates as CA
    from ispatialgym.robot import conventions as C

    return {"frame": "eef: +z = approach (the direction the hand moves onto the object), +y = closing axis (the fingers close along +-y), "
                     "x = y cross z (the pads' width direction)",
            "max_opening_mm": round(CA.GRIPPER_MAX_OPENING * 1000, 1), "usable_pinch_width_mm": [round(CA.CandidateConfig().min_width_m * 1000, 1),
                                                                                                   round(CA.CandidateConfig().aperture_m * 1000, 1)],
            "pad_span_along_approach_mm": [round(v * 1000, 2) for v in C.PAD_Z_RANGE], "pad_centre_mm": round(CA.CONTACT_Z * 1000, 2),
            "fingertip_mm": round(C.FINGERTIP_LENGTH * 1000, 2), "max_contact_toward_tips_mm": round(CA.CandidateConfig().tip_max_m * 1000, 1),
            "pad_width_mm": 20.0,
            "preshape": "the oracle opens each finger to pinch width / 2 + 10 mm before it descends, then closes until the pads touch",
            "pre_grasp_mm": round(C.PRE_DIST * 1000, 1), "notes": "the hand (palm, wrist camera) sits behind the eef origin along -z; a top "
            "pinch of a low object needs its contact high enough that the fingers stay within 2 mm of the support"}


def task_role(job: Dict[str, Any], customs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """What the grasp is for: the aliases / subtasks of the entry, its constraint and what the oracle does with the object."""
    aliases = []
    key = job.get("legacy_key")
    if customs and key:
        for k, v in customs.items():
            if not isinstance(v, dict):
                continue
            for role, r in (v.get("roles") or {}).items():
                if key in ((r or {}).get("grasp_keys") or {}).values():
                    aliases.append({"customs": k, "role": role, "support": ((v.get("roles") or {}).get("support") or {}).get("category")})
    rules = {"tabletop": "picked from its support, carried, placed (released on the goal)",
             "tabletop_insert": "picked from its support and inserted into a container: the fingers must leave the bottom free",
             "tabletop_level_pinch": "picked from its support with a level pinch",
             "attachment_free": "picked and attached to a partner: the contact on the far side of the mating face, no finger past it",
             "flower_stem_top": "picked by the stem below the head"}
    c = job.get("constraint") or "tabletop"
    return {"alias_key": job.get("alias_key"), "subtasks": job.get("subtasks"), "legacy_key": key, "constraint": c, "use": rules.get(c, c),
            "support_height_m": job.get("h_live"), "aliases_of_the_key": aliases}


def history(src_dir: str, status: Dict[str, Optional[int]]) -> Dict[str, Any]:
    """The rounds so far: per root the physics-valid count, the last build's per-pool gate counts and geometry reasons."""
    cand = S.load_json(os.path.join(src_dir, "candidates.json")) or {}
    val = S.load_json(os.path.join(src_dir, "validation.json")) or {}
    reasons = Counter()
    for t in val.get("trials") or []:
        for r in (t.get("reasons") or (["valid"] if t.get("verdict") == "valid" else [])):
            reasons[str(r).split("@")[0]] += 1
    return {"rounds": status, "last_build": src_dir, "log": {k: v for k, v in (cand.get("log") or {}).items() if isinstance(v, dict)},
            "geometry_reasons": dict(Counter(f"{c.get('family') or c.get('mode')}/{c['geometry']['reason']}" for c in cand.get("candidates") or []).most_common()),
            "physics_reasons": dict(reasons)}


def _rank(gate: str) -> int:
    return GATE_ORDER.index(gate) if gate in GATE_ORDER else -1


def markers_from_dump(dump: Optional[Dict[str, Any]], cand_doc: Optional[Dict[str, Any]], k: int = len(MARKERS), sep_mm: float = 15.0) -> List[Dict[str, Any]]:
    """The heuristic's proposals for the strips: the round's NEAR MISSES - the dropped candidates that got furthest through the gates
    (selected candidates with their geometry reason first, then the gate dump's last pass, deepest gate first), spread at least
    ``sep_mm`` apart, at most ``k``. Each marker: letter, pool, the gate that dropped it, contact (mm), approach / closing axis, width and
    the measures that gate decided on."""
    from ispatialgym.grasping import candidates as CA

    rows: List[Tuple[int, float, Dict[str, Any]]] = []
    for c in (cand_doc or {}).get("candidates") or []:
        T = S.pose_of(c["T_rest_eef"])
        sh = float((c.get("features") or {}).get("shift_m", 0.0)) if c.get("mode") == "top" else 0.0
        p = T[:3, 3] + T[:3, 2] * (CA.CONTACT_Z + sh)
        rows.append((_rank("geometry") + (1 if c["geometry"]["ok"] else 0), float(c.get("score", 0.0)),
                     {"pool": c.get("family") or c.get("mode"), "gate": "geometry" if not c["geometry"]["ok"] else "physics",
                      "why": c["geometry"]["reason"], "contact_mm": _mm(p), "approach": [round(float(v), 3) for v in T[:3, 2]],
                      "closing_axis": [round(float(v), 3) for v in T[:3, 1]], "width_mm": round(float(c["width_m"]) * 1000, 2),
                      "contact_offset_mm": round(sh * 1000, 2), "measures": {kk: c["geometry"].get(kk) for kk in ("stage", "finger_min_m", "hand_min_m", "blade_object_depth_m")}}))
    passes = (dump or {}).get("passes") or []
    if passes:
        for pool, stages in (passes[-1].get("pools") or {}).items():
            for st in stages:
                if st["gate"] in ("pool", "nms", "geometry"):
                    continue
                for d in st.get("dropped") or []:
                    if not d.get("contact_mm") or None in d["contact_mm"]:
                        continue
                    meas = {kk: v for kk, v in d.items() if kk not in ("contact_mm", "approach_k", "closing_k", "why", "contact_azimuth_deg")}
                    m = {"pool": pool, "gate": st["gate"], "why": d.get("why") or st["gate"], "contact_mm": d["contact_mm"],
                         "width_mm": d.get("width_mm"), "contact_offset_mm": d.get("contact_offset_mm"), "measures": meas}
                    if d.get("approach_k"):
                        m["approach"] = [round(v / 1000.0, 3) for v in d["approach_k"]]
                        m["closing_axis"] = [round(v / 1000.0, 3) for v in d["closing_k"]]
                    rows.append((_rank(st["gate"]), -abs(float(d.get("touch_mm") or 0.0) - float(d.get("limit_mm") or 0.0)), m))
    rows.sort(key=lambda r: (-r[0], -r[1]))
    out: List[Dict[str, Any]] = []
    for _, _, m in rows:
        p = np.asarray(m["contact_mm"], dtype=float)
        if any(np.linalg.norm(p - np.asarray(o["contact_mm"])) < sep_mm and o["pool"] == m["pool"] for o in out):
            continue
        out.append(m)
        if len(out) >= k:
            break
    for letter, m in zip(MARKERS, out):
        m["marker"] = letter
    return out


# ------------------------------------------------------------------------------------------------ strips (PNG of the collision export)


def render_strip(geom_rest, markers: Sequence[Dict[str, Any]], header: str, path: str) -> str:
    """One PNG per entry: orthographic top / front / side views and two vertical sections (through the plate axis) of the collision
    pieces in the REST frame (mm axes, the support line), the markers lettered with their closing axis (segment) and approach (arrow), and
    an oblique view. Matplotlib (Agg); only the collision export is drawn."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    u = geom_rest.union
    V = np.asarray(u.vertices) * 1000.0
    F = np.asarray(u.faces)
    N = np.asarray(u.face_normals)
    lo, hi = V.min(0), V.max(0)
    ctr = 0.5 * (lo + hi)
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    light = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    views = [("top (looking down -z)", (0, 1), 2, +1), ("front (looking along +y)", (0, 2), 1, -1), ("side (looking along -x)", (1, 2), 0, +1)]
    fig = plt.figure(figsize=(22, 8.5))
    fig.suptitle(header, fontsize=10, family="monospace", x=0.01, ha="left")
    for vi, (title, (a, b), dax, sgn) in enumerate(views):
        ax = fig.add_subplot(2, 4, vi + 1)
        depth = V[F].mean(1)[:, dax] * sgn
        order = np.argsort(depth)
        shade = 0.35 + 0.6 * np.clip(np.abs(N @ light), 0, 1)
        polys = V[F][order][:, :, [a, b]]
        ax.add_collection(PolyCollection(polys, facecolors=np.c_[np.repeat(shade[order, None], 3, 1) * [0.55, 0.65, 0.8], np.ones(len(order))],
                                         edgecolors=(0, 0, 0, 0.08), linewidths=0.2))
        if b == 2:
            ax.axhline(lo[2], color="saddlebrown", lw=1.5)
        _draw_markers(ax, markers, a, b, cols)
        _frame(ax, lo, hi, a, b, title)
    # sections through the plate axis: the walls / holes the views hide
    for si, (nrm, a, title) in enumerate((([0, 1.0, 0], 0, "section y = centre (x right, z up)"), ([1.0, 0, 0], 1, "section x = centre (y right, z up)"))):
        ax = fig.add_subplot(2, 4, 5 + si)
        segs = []
        for p in geom_rest.pieces:
            try:
                sec = p.section(plane_origin=ctr / 1000.0, plane_normal=nrm)
            except Exception:  # noqa: BLE001
                sec = None
            if sec is None:
                continue
            for e in sec.entities:
                pts = sec.vertices[e.points] * 1000.0
                segs += [pts[i:i + 2][:, [a, 2]] for i in range(len(pts) - 1)]
        ax.add_collection(LineCollection(segs, colors="navy", linewidths=1.2))
        ax.axhline(lo[2], color="saddlebrown", lw=1.5)
        _draw_markers(ax, markers, a, 2, cols)
        _frame(ax, lo, hi, a, 2, title)
    ax = fig.add_subplot(2, 4, 4, projection="3d")
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    shade = 0.35 + 0.6 * np.clip(np.abs(N @ light), 0, 1)
    ax.add_collection3d(Poly3DCollection(V[F], facecolors=np.c_[np.repeat(shade[:, None], 3, 1) * [0.55, 0.65, 0.8], np.ones(len(F))], linewidths=0.0))
    for i, m in enumerate(markers):
        p = np.asarray(m["contact_mm"], dtype=float)
        ax.scatter(*p, color=cols[i % 10], s=30)
        ax.text(*p, m.get("marker", "?"), color=cols[i % 10], fontsize=11, weight="bold")
    span = float((hi - lo).max()) / 2 + 5
    for k, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        setter(ctr[k] - span, ctr[k] + span)
    ax.view_init(elev=30, azim=-60)
    ax.set_title("oblique (elev 30, azim -60)", fontsize=9)
    ax = fig.add_subplot(2, 4, 7)
    ax.axis("off")
    lines = []
    for i, m in enumerate(markers):
        ms = m.get("measures") or {}
        keep = {k: ms[k] for k in list(ms)[:6]}
        lines.append(f"{m.get('marker')}: {m['pool']} / {m['gate']} ({m.get('why')}) c={m['contact_mm']} w={m.get('width_mm')} {json.dumps(keep)[:110]}")
    ax.text(0, 1, "\n".join(lines) or "no markers (no near miss recorded)", va="top", fontsize=7, family="monospace", wrap=True)
    ax = fig.add_subplot(2, 4, 8)
    ax.axis("off")
    ax.text(0, 1, "markers: coloured dot = pinch centre (contact), segment = closing axis (+-width/2),\narrow = approach (from 30 mm behind the "
            "contact toward it).\nBrown line = the support plane. Axes in mm, rest frame.", va="top", fontsize=8)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path


def _draw_markers(ax, markers, a, b, cols):
    for i, m in enumerate(markers):
        p = np.asarray(m["contact_mm"], dtype=float)
        col = cols[i % 10]
        ax.plot(p[a], p[b], "o", color=col, ms=6, mec="k", mew=0.5, zorder=5)
        ax.annotate(m.get("marker", "?"), (p[a], p[b]), xytext=(4, 4), textcoords="offset points", color=col, fontsize=11, weight="bold", zorder=6)
        if m.get("closing_axis") and m.get("width_mm"):
            y = np.asarray(m["closing_axis"], dtype=float) * float(m["width_mm"]) / 2
            ax.plot([p[a] - y[a], p[a] + y[a]], [p[b] - y[b], p[b] + y[b]], "-", color=col, lw=2, zorder=5)
        if m.get("approach"):
            z = np.asarray(m["approach"], dtype=float) * 30.0
            ax.annotate("", (p[a], p[b]), xytext=(p[a] - z[a], p[b] - z[b]), arrowprops={"arrowstyle": "->", "color": col, "lw": 1.2}, zorder=4)


def _frame(ax, lo, hi, a, b, title):
    pad = 0.12 * float((hi - lo).max()) + 5
    ax.set_xlim(lo[a] - pad, hi[a] + pad)
    ax.set_ylim(lo[b] - pad, hi[b] + pad)
    ax.set_aspect("equal")
    ax.grid(True, lw=0.3)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("xyz"[a] + " (mm)", fontsize=8)
    ax.set_ylabel("xyz"[b] + " (mm)", fontsize=8)
    ax.tick_params(labelsize=7)


# ------------------------------------------------------------------------------------------------ the contract: checks and conversion


def check_proposals(doc: Dict[str, Any], entry_id: Optional[str] = None) -> List[str]:
    """Errors of a proposal document against :data:`CONTRACT` (empty = valid). Shape and value checks only; whether a pinch works is
    for the gates and physics to decide."""
    from ispatialgym.grasping import candidates as CA

    err = []
    need = {"contract", "id", "feasible", "proposals", "object_note"}
    if not isinstance(doc, dict):
        return ["not a JSON object"]
    if set(doc) != need:
        err.append(f"keys {sorted(doc)} != {sorted(need)}")
    if doc.get("contract") != CONTRACT:
        err.append(f"contract {doc.get('contract')!r} != {CONTRACT!r}")
    if entry_id is not None and doc.get("id") != entry_id:
        err.append(f"id {doc.get('id')!r} != {entry_id!r}")
    props = doc.get("proposals")
    if not isinstance(props, list):
        return err + ["proposals is not a list"]
    if doc.get("feasible") is True and not props:
        err.append("feasible but no proposal")
    if len(props) > MAX_PROPOSALS:
        err.append(f"{len(props)} proposals > {MAX_PROPOSALS}")
    pkeys = {"from_marker", "contact_mm", "closing_axis", "approach", "pad_offset_mm", "expand", "rationale", "confidence"}
    for i, p in enumerate(props):
        if not isinstance(p, dict) or set(p) != pkeys:
            err.append(f"proposal {i}: keys {sorted(p) if isinstance(p, dict) else p} != {sorted(pkeys)}")
            continue
        for k in ("contact_mm", "closing_axis", "approach"):
            v = p[k]
            if not (isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)):
                err.append(f"proposal {i}: {k} is not 3 finite numbers")
        po = p["pad_offset_mm"]
        if not (isinstance(po, list) and 1 <= len(po) <= 5 and all(isinstance(x, (int, float)) and 0 <= x <= 28.8 for x in po)):
            err.append(f"proposal {i}: pad_offset_mm must be 1-5 numbers in [0, 28.8]")
        if p["from_marker"] is not None and p["from_marker"] not in MARKERS:
            err.append(f"proposal {i}: from_marker {p['from_marker']!r}")
        if p["confidence"] not in CONFIDENCE:
            err.append(f"proposal {i}: confidence {p['confidence']!r}")
        ex = p["expand"]
        if ex is not None:
            if not isinstance(ex, dict) or not set(ex) <= set(CA.AGENT_EXPAND_DEFAULT):
                err.append(f"proposal {i}: expand keys {sorted(ex) if isinstance(ex, dict) else ex}")
            else:
                n = 1
                for kk in CA.AGENT_EXPAND_DEFAULT:
                    vv = ex.get(kk, CA.AGENT_EXPAND_DEFAULT[kk])
                    if not (isinstance(vv, (list, tuple)) and vv and all(isinstance(x, (int, float)) and abs(x) <= 45 for x in vv)):
                        err.append(f"proposal {i}: expand.{kk} must be 1+ numbers within +-45")
                    n *= len(vv) if isinstance(vv, (list, tuple)) else 1
                if n * len(po if isinstance(po, list) else [po]) > CA.AGENT_EXPAND_MAX:
                    err.append(f"proposal {i}: the expansion makes {n * len(po)} poses > {CA.AGENT_EXPAND_MAX}")
        if not err or not any(e.startswith(f"proposal {i}: ") for e in err):
            try:
                CA.proposal_frame(p["approach"], p["closing_axis"])
            except ValueError as e:
                err.append(f"proposal {i}: {e}")
    return err


def agent_overrides(doc: Dict[str, Any]) -> Dict[str, Any]:
    """The cand_overrides of an entry's step-3 build: only the proposals (no mode, no sampled family), the gate dump on."""
    props = []
    for p in doc["proposals"]:
        q = {k: p[k] for k in ("contact_mm", "closing_axis", "approach", "pad_offset_mm")}
        if p.get("expand"):
            q["expand"] = p["expand"]
        q["from_marker"] = p.get("from_marker")
        props.append(q)
    return {"modes": [], "families": ["agent"], "agent_proposals": props, "gate_dump": True}


def agent_job(job: Dict[str, Any], doc: Dict[str, Any], proposals_path: str) -> Dict[str, Any]:
    """The step-3 manifest entry of ``job`` (same id, category, model, scale, constraint, rest: the same entry identity)."""
    j = {k: v for k, v in copy.deepcopy(job).items() if k not in ("cand_overrides", "ladder", "retry_status")}
    j["cand_overrides"] = agent_overrides(doc)
    j["ladder"] = {"step": 3, "source": "agent", "contract": CONTRACT, "proposals": proposals_path, "n_proposals": len(doc["proposals"])}
    return j


# ------------------------------------------------------------------------------------------------ step 4: the legacy grasps


def legacy_grasps(key: str, root: str = LEGACY_ROOT) -> List[Dict[str, Any]]:
    """A legacy key's grasps (``cands/<key>.yaml``, isaac_grasp: ``right_eef_link`` in ``base_link``) as ``{"name", "position",
    "quat_xyzw", "legacy_physics"}``; ``legacy_physics`` = the legacy validation's verdict (``results/<key>.json``: AG attached and held
    after gravity) - recorded, never used as a gate. A floating (GraspGen) key returns its grasps too (the caller decides)."""
    import yaml

    p = os.path.join(root, "cands", f"{key}.yaml")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        doc = yaml.safe_load(f) or {}
    res = S.load_json(os.path.join(root, "results", f"{key}.json")) or {}
    tested = {t.get("grasp"): bool(t.get("ag_attached") and t.get("held_after_gravity")) for t in res.get("tested") or []}
    out = []
    for name, g in (doc.get("grasps") or {}).items():
        o = g.get("orientation") or {}
        xyz = list(o.get("xyz") or [0.0, 0.0, 0.0])
        out.append({"name": str(name), "position": [float(v) for v in g["position"]], "quat_xyzw": [float(v) for v in xyz] + [float(o.get("w", 1.0))],
                    "legacy_physics": tested.get(name)})
    return out


def legacy_job(job: Dict[str, Any], grasps: Sequence[Dict[str, Any]], key: str) -> Dict[str, Any]:
    """The step-4 manifest entry: the legacy grasps as given poses through the same gates and physics (owner approval required to run
    and to use - scripts/grasp_ladder/legacy_step.py refuses without it)."""
    j = {k: v for k, v in copy.deepcopy(job).items() if k not in ("cand_overrides", "ladder", "retry_status")}
    j["cand_overrides"] = {"modes": [], "families": ["legacy"], "gate_dump": True,
                           "legacy_grasps": [{k: g[k] for k in ("name", "position", "quat_xyzw")} for g in grasps]}
    j["ladder"] = {"step": 4, "source": "legacy", "legacy_key": key, "legacy_root": os.path.relpath(LEGACY_ROOT, REPO_ROOT), "n_grasps": len(grasps)}
    return j


# ------------------------------------------------------------------------------------------------ reports


def entry_report(model_dir: str) -> Dict[str, Any]:
    """Per pool (proposal / legacy grasp / mode) of a built entry: poses -> each gate's survivors -> selected -> geometry ok ->
    representatives -> physics-valid; when a pool lost everything, the gate that did it with its drop reasons and measured values
    (gate_dump.json's last pass)."""
    cand = S.load_json(os.path.join(model_dir, "candidates.json")) or {}
    dump = S.load_json(os.path.join(model_dir, "gate_dump.json")) or {}
    val = S.load_json(os.path.join(model_dir, "validation.json")) or {}
    lib = S.load_json(os.path.join(model_dir, "library.json")) or {}
    reps = set(cand.get("representatives") or [])
    by_name = {c["name"]: c for c in cand.get("candidates") or []}
    trials = {t.get("candidate"): t for t in val.get("trials") or [] if t.get("candidate")}
    valid = {g["name"] for g in lib.get("grasps") or []}
    pools_dump = ((dump.get("passes") or [{}])[-1].get("pools") or {})
    out = {"dir": model_dir, "n_valid": len(valid), "pools": {}}
    for pool, L in (cand.get("log") or {}).items():
        if not isinstance(L, dict):
            continue
        cs = [c for c in by_name.values() if (c.get("family") or c["mode"]) == pool]
        r = {"counts": {k: v for k, v in L.items() if k.startswith("n_")}, "source": L.get("source"), "error": L.get("error"),
             "selected": len(cs), "geometry_ok": sum(1 for c in cs if c["geometry"]["ok"]),
             "geometry_reasons": dict(Counter(c["geometry"]["reason"] for c in cs)),
             "representatives": sum(1 for c in cs if c["name"] in reps), "physics_valid": sum(1 for c in cs if c["name"] in valid),
             "physics_reasons": dict(Counter(r_ for c in cs if c["name"] in trials and trials[c["name"]].get("verdict") != "valid"
                                             for r_ in (trials[c["name"]].get("reasons") or ["?"])))}
        stages = pools_dump.get(pool) or []
        last = None
        for st in stages:
            if st["n_in"] > 0 and st["n_out"] == 0:
                last = st
                break
        if last is not None:
            r["killed_by"] = {"gate": last["gate"], "n_in": last["n_in"], "dropped_by": last.get("dropped_by") or last.get("summary"),
                              "stats": last.get("stats"), "examples": (last.get("dropped") or last.get("records") or [])[:3]}
        out["pools"][pool] = r
    return out


def summarize(rep: Dict[str, Any]) -> str:
    lines = [f"{os.path.basename(rep['dir'])}: physics-valid {rep['n_valid']}"]
    for pool, r in rep["pools"].items():
        c = r["counts"]
        chain = " -> ".join(f"{k[2:]} {v}" for k, v in c.items())
        s = f"  {pool}: {chain} | selected {r['selected']} geo_ok {r['geometry_ok']} reps {r['representatives']} valid {r['physics_valid']}"
        if r.get("error"):
            s += f" | error {r['error']}"
        if r.get("killed_by"):
            k = r["killed_by"]
            s += f" | killed at {k['gate']} ({k['dropped_by']})"
        if r["physics_reasons"]:
            s += f" | physics {r['physics_reasons']}"
        lines.append(s)
    return "\n".join(lines)
