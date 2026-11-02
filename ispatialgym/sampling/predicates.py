"""Robust BDDL predicate evaluation shared by the sampler's goal check and the task evaluator (D-007, option A, owner
2026-09-16 21:17).

``under`` / ``ontop`` through OmniGibson alone are fragile on a physically correct placement: ``Under`` / ``OnTop`` shoot a
single vertical ray from the supporting object's own surface point into the resting object, and whether that ray
registers the object depends on sub-millimetre contact / penetration (teacup on the saucer well, 0916-172359 and
0916-192655: the cup 1.4 mm from its goal, 2.7 deg tilt, ``binary_success_ever`` False; the same geometry passed at 18:41).

``evaluate_predicate(predicate_cls, *entities)`` is a drop-in for ``omnigibson.utils.bddl_utils.evaluate_bddl_predicate``:

1. **omnigibson** — OmniGibson's verdict; True ends the evaluation.
2. **rays** (``under`` / ``ontop`` only) — several rays across the contact patch: the resting object's footprint centre and a
   ring at half its footprint radius; at each point the supporting object's surface under the resting object is found with
   a downward ray (the resting object's bodies ignored), the probe ray starts ``RAY_BELOW_SURFACE_M`` below that surface and
   goes up (the supporting object's bodies ignored); any hit on the resting object counts.
3. **geometry** — the resting object's AABB bottom within ``BOTTOM_GAP_MAX_M`` of the supporting surface under its footprint
   centre, the footprint centre inside the supporting object's top footprint, the resting object's up axis within
   ``UP_AXIS_MAX_DEG`` of vertical (a concave partner whose collision mesh lets the rays through).

``touching`` (owner 2026-09-19 00:49): OmniGibson's ``Touching`` is a PhysX contact report, which at rest between two
free-standing objects with zero normal force is a 0.1 mm lottery (touching 0918-200452 vs 0918-203047 / 224622: the same
final geometry to 0.1 mm, one read True for 260 steps, the others False). After an OmniGibson False the **geometry** branch
measures the separation of the two objects' convex hulls (the collision boundary points against the other hull's
half-spaces, exact for a face-to-face contact, a lower bound otherwise) and accepts a gap up to ``TOUCHING_GAP_MAX_M``
(3 mm; ``ISG_TOUCHING_GAP_MAX_M`` overrides).

``inside`` of a container without a fillable meta link (owner 2026-10-12, IORS large-scale inside: a kitchen sink - togglebutton /
fluidsink / fluidsource links -, a fridge - lights): OmniGibson's ``Inside`` tests the inner object's AABB centre against the convex
hulls of the container's ``fillable`` / ``openfillable`` meta links only, so it reads False for such a container whatever the geometry.
After that False the **rays** branch (:func:`inside_rays`) applies BEHAVIOR's earlier kinematic rule: the inner object's AABB centre
inside the container's AABB and the container on both sides of it along at least ``INSIDE_MIN_AXES`` of the three axes (vertical,
the container's own horizontal axes; rays from the centre, the inner object's bodies ignored, a hit on the container anywhere along
the ray counts). A container with such a meta link keeps OmniGibson's verdict.

``attached`` (owner 2026-10-10, replacing the 2026-09-19 5 cm / 15 deg join): the pair's collision stays ON until the parts
are joined, and only the task's rule joins them - OmniGibson's contact-triggered ``AttachedTo._update`` is blocked for the
registered pairs (once the pair can touch it joins on contact within 5 cm / 15 deg, checked without the part's symmetry,
snaps the part - twisting a round cap up to 15 deg - and ignores the hand: 0919-174551 snapped the lid 4 cm out of the hand
mid-descent). After an OmniGibson False the **alignment** branch joins a registered child only when (a) no robot holds it,
(b) PhysX reports it in contact with its partner and (c) its male meta link is aligned with the partner's female link
**modulo the part's recorded symmetry** (the group / pose target's ``orientation_equivalences``, registered by the task
through :func:`set_attachment_context`; ``zinf`` = yaw free) within ``JOIN_LATERAL_MAX_M`` across the female axis,
``JOIN_ANGLE_MAX_RAD`` and ``JOIN_VERTICAL_MIN_M`` .. ``JOIN_VERTICAL_MAX_M`` along it (the window covers the gap at which
the part comes to rest on its partner with collision on). No join in mid-air or at the release instant of a part that does
not touch its partner yet. The join (:func:`join_mode`, ``ISG_ATTACH_JOIN_MODE``) is ``snap`` (default, owner 2026-10-10) -
the part first moved onto its seat (the fingers are open by then, nothing is yanked) by :func:`symmetric_snap`, anchored on
the attachment frames: a part whose symmetry is continuous about one axis (``zinf``: 71 turns of 5° about object z) keeps its
spin about that axis - only its tilt is removed, its male origin on the female origin and its axis on the seat's (OmniGibson's
snap forces one spin: a cap arriving 90° off would be turned 90°) - any other part is carried onto the female frame at its
nearest symmetric equivalent (no equivalences: the raw frame snap) - or ``in_place`` - the BEHAVIOR fixed joint at the links'
current relative pose (:func:`attach_in_place`); a cap or a cork comes to rest on its partner's rim beyond the layout tolerance,
so only ``snap`` seats them - or ``set_value`` (the stock path). It filters the pair's collision in the same instant
(:func:`set_pair_collision_filter`; the stock ``AttachedTo._attach`` did that, this fork's does not), so a joined part does not
fight its partner's hull. ``JOINS`` records every join (trigger, alignment, contact, the snap's mode / move / tilt / spin).

``pose_settled`` of an object joined by an attachment joint (owner 2026-09-20 13:57; the attached child and its parent):
OmniGibson's ``PoseSettled`` needs the root velocity below 1 mm/s on every step of its window AND the pose drift within
1 cm / 0.05 rad over it. PhysX enforces the fixed attachment joint iteratively, so both bodies carry a residual velocity of
1-3 mm/s while their pose drifts by hundredths of a millimetre (pwgbsb / jlalfc 0919-231624: lid 0.9 mm/s median, 2.7 peak,
63 / 144 window steps above the threshold, total drift 0.015 mm / 0.000 deg) - the velocity gate can never pass, so the
settle metric was unmeasurable for every attachment. After an OmniGibson False the **drift** branch decides from the state's
own window (``n_steps`` samples, the configured tolerances): the buffer full and every sample's root pose (and joints)
within tolerance of the window anchor - the velocity gate alone is dropped.

``pose_settled`` of the **robot** (owner 2026-09-24): the same drift rule. The R1Pro chassis stands on its three wheel spheres
(gravity on the base-fixed links only, the z / rx / ry virtual joints free) and the base-footprint velocity PoseSettled reads
is the contact solver's per-substep correction residual, not motion: 0.96 mm/s median with the stock 47.6 kg chassis, 1.43
with the evaluator's 250 kg, on every step of a window in which the base moves 2 um and no joint 0.05 mrad (pod probe
``scripts/experiments/probe_base_velocity_artifact.py``, house_single_floor; more velocity iterations or softer base drives
make it worse, only 64 position iterations or no chassis weight remove it). The 1 mm/s gate therefore never holds for the
agent (30 of the 153 successful first-demo runs, every base-only and held-object task among them) while the robot is at rest.

``pose_settled`` of **every object** (owner 2026-09-24): the same residual makes 86 of the 228 settled-object atoms of those
runs fail while the object moves <= 0.24 mm / 2.5 mrad over the window (25 / 26 ontop-small, 10 / 10 under-small and inside:
objects resting on or in other objects, the supports themselves, articulated lids, held objects), so the drift rule applies to
all entities (``DRIFT_SETTLED_ENTITIES`` = ``all``; ``ISG_DRIFT_SETTLED`` narrows it). The rest criterion then lives in the
PoseSettled window the task configures - 45 steps (1.5 s), 2 mm / 5 mrad / 5 mrad joints, ``tasks.ispatialgym_task.
DEFAULT_POSE_SETTLED_CONFIG`` - chosen as the tightest round values every settled entity of the 28 latest first-demo
successes clears at its last step (their 1.5 s drift: <= 0.074 mm, <= 0.98 mrad = the float32 quaternion resolution,
<= 0.036 mrad of joint), while a hinge creeping 0.25 deg/s (6.5 mrad over the window) or an object rolling 3 mm/s (4.5 mm) fails.

Every evaluation appends ``{"predicate", "entities", "branch", "ok", "detail"}`` to the branch log (``reset_branches`` /
``branches``); the callers copy it into ``goal_check.binary_branch`` (sampler) and ``meta.binary_branch`` (evaluator) so the
ledger shows which branch decided. Other predicates pass straight through.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch as th

RAY_BELOW_SURFACE_M = 0.003
RAY_LENGTH_M = 0.40
BOTTOM_GAP_MAX_M = 0.008
UP_AXIS_MAX_DEG = 15.0
RING_POINTS = 8
TOUCHING_GAP_MAX_M = float(os.environ.get("ISG_TOUCHING_GAP_MAX_M", "0.003"))
#: ``inside`` rays branch: the axes with the container on both sides of the inner object's centre, and the rays' length beyond the
#: container's AABB diagonal
INSIDE_MIN_AXES = 2
INSIDE_RAY_MARGIN_M = 0.05
CONTAINER_META_LINK_TYPES = ("fillable", "openfillable")
#: attachment alignment record's coarse ``ok`` (OmniGibson ``AttachedTo``'s own thresholds: m.DEFAULT_POSITION_THRESHOLD /
#: m.DEFAULT_ORIENTATION_THRESHOLD); the join itself needs the tight bounds below
ATTACH_POS_MAX_M = 0.05
ATTACH_ORN_MAX_RAD = math.radians(15.0)
#: the join's alignment (owner 2026-10-10): the male link within this of the seat axis, within this angle of the female frame
#: modulo the part's symmetry, and this far out of the seat along the axis. Attach step 0 (2026-10-10, the r5 instances, the pair
#: colliding, the part dropped aligned from 3 / 6 mm above first contact) found each part at rest ON its partner's hull, its male
#: link above the seat by: cap / lemonade bottle 10.2 mm, cork / wine bottle 17.9 mm, lid / teapot 4.85 mm (lateral <= 0.63 mm,
#: angle <= 0.018 rad, tilt <= 1.2 deg) - the vertical window covers them with 5 mm to spare
JOIN_LATERAL_MAX_M = 0.004
JOIN_ANGLE_MAX_RAD = 0.10
JOIN_VERTICAL_MIN_M = -0.004
JOIN_VERTICAL_MAX_M = 0.023
JOIN_MODES = ("in_place", "snap", "set_value")
#: the snap of a part whose recorded symmetry is continuous about one axis (``zinf``: the 5° group about object z, 71 rotations)
#: removes only its tilt and keeps its spin (:func:`symmetric_snap`): at least this many equivalences, every non-identity one
#: about one axis (|dot| >= ``SNAP_CONTINUOUS_AXIS_DOT``); the oracle's ``_reaim_attach_targets`` uses the same count
SNAP_CONTINUOUS_MIN_EQUIVALENCES = 12
SNAP_CONTINUOUS_AXIS_DOT = 0.999
#: ``pose_settled`` entities judged by configuration drift after an OmniGibson False (module docstring): ``attached`` (the
#: members of an attachment joint), ``robot`` (the agent), ``all`` (every object - the default since 2026-09-24, with the
#: tightened drift window ``tasks.ispatialgym_task.DEFAULT_POSE_SETTLED_CONFIG``). ``ISG_DRIFT_SETTLED`` overrides (comma
#: list; ``attached,robot`` = the 2026-09-20 / 09-24 stages, ``attached`` = the 09-20 rule alone).
DRIFT_SETTLED_ENTITIES = frozenset(s.strip() for s in os.environ.get("ISG_DRIFT_SETTLED", "all").split(",") if s.strip())

_BRANCHES: List[Dict[str, Any]] = []
#: (child object name, parent object name) -> {"child_link", "parent_link", "equivalences": [[x, y, z, w], ...]} — the task's
#: ``attachment_targets`` with the child part's recorded symmetry (``set_attachment_context``)
#: keyed by (scene index or None, child name, parent name): a vectorized task registers every clone's pairs under its scene index,
#: a single-env task under None; the lookup tries the entity's own scene first (``attachment_context``)
_ATTACHMENTS: Dict[Tuple[Optional[int], str, str], Dict[str, Any]] = {}


def reset_branches() -> None:
    _BRANCHES.clear()


def branches() -> List[Dict[str, Any]]:
    return list(_BRANCHES)


MAX_LOG = 200  # the evaluator calls every step: keep the latest entries only
#: tasks.ispatialgym_task's shared goal evaluation (ISPATIALGYM_SHARED_GOAL_EVAL): the entries _log appends while a list
#: is set here, and the number of attachment joints the alignment branch has created
_RECORD: Optional[List[Dict[str, Any]]] = None
_JOINTS = [0]


def _log(predicate: str, entities: Sequence[Any], branch: str, ok: bool, detail: Optional[Dict[str, Any]] = None) -> None:
    entry = {"predicate": predicate, "entities": [getattr(e, "name", str(e)) for e in entities], "branch": branch, "ok": bool(ok),
             "detail": detail or {}}
    _BRANCHES.append(entry)
    if _RECORD is not None:
        _RECORD.append(entry)
    if len(_BRANCHES) > MAX_LOG:
        del _BRANCHES[: len(_BRANCHES) - MAX_LOG]


def replay_log(entries: Sequence[Dict[str, Any]]) -> None:
    """Append ``entries`` (recorded by an evaluation of the current state) again, as a second evaluation of the same state
    would log them."""
    for e in entries:
        _BRANCHES.append({**e, "entities": list(e["entities"])})
        if len(_BRANCHES) > MAX_LOG:
            del _BRANCHES[: len(_BRANCHES) - MAX_LOG]


def decided_by() -> Dict[str, str]:
    """``{"<predicate>(<a>, <b>)": branch}`` of the latest evaluation of every under / ontop atom in the log — what a caller
    writes as ``binary_branch``."""
    out: Dict[str, str] = {}
    for e in _BRANCHES:
        if e["predicate"] in ("under", "ontop", "touching", "attached") or (e["predicate"] == "pose_settled" and e["branch"] != "omnigibson"):
            out[f"{e['predicate']}({', '.join(e['entities'])})"] = e["branch"]
    return out


# ----------------------------------------------------------------------------------------------------------------
# attached: alignment modulo the part's symmetry, the BEHAVIOR joint created when aligned
# ----------------------------------------------------------------------------------------------------------------


def set_attachment_context(pairs: Sequence[Dict[str, Any]], scene_idx: Optional[int] = None) -> None:
    """Register the task's attachment pairs: ``{"child": <object name>, "parent": <object name>, "child_link", "parent_link",
    "equivalences": [[x, y, z, w], ...]}`` (object-frame rotations of the child that leave its shape unchanged; the identity
    is implied). Called by the task when the TRO targets are configured; ``[]`` clears. ``scene_idx``: the clone the pairs
    belong to (a vectorized environment registers each clone's pairs separately; only that clone's entries are replaced).
    Registering a pair installs the ``AttachedTo`` overrides (:func:`install_attach_overrides`) that leave its joining to this
    module's rule."""
    for key in [k for k in _ATTACHMENTS if k[0] == scene_idx]:
        del _ATTACHMENTS[key]
    for p in pairs or []:
        if p.get("child") and p.get("parent"):
            _ATTACHMENTS[(scene_idx, str(p["child"]), str(p["parent"]))] = {"child_link": p.get("child_link"), "parent_link": p.get("parent_link"),
                                                                            "equivalences": [[float(v) for v in q] for q in (p.get("equivalences") or [])]}
    if _ATTACHMENTS:
        install_attach_overrides()


def attachment_context(child: Any, parent: Any) -> Optional[Dict[str, Any]]:
    """The pair's record: the one registered for the child's own scene (``child.scene.idx``) when there is one, else the
    single-env registration (scene None)."""
    names = (getattr(child, "name", str(child)), getattr(parent, "name", str(parent)))
    idx = getattr(getattr(child, "scene", None), "idx", None)
    if idx is not None and (idx, *names) in _ATTACHMENTS:
        return _ATTACHMENTS[(idx, *names)]
    return _ATTACHMENTS.get((None, *names))


def attachment_partners(child: Any) -> List[str]:
    """Names of the parents registered for ``child``, with :func:`attachment_context`'s precedence: the pairs of the child's
    own scene first, then the single-env registration (scene None), each in registration order."""
    name = getattr(child, "name", str(child))
    idx = getattr(getattr(child, "scene", None), "idx", None)
    out: List[str] = []
    for scene_idx in (idx, None) if idx is not None else (None,):
        for s, c, p in _ATTACHMENTS:
            if s == scene_idx and c == name and p not in out:
                out.append(p)
    return out


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return th.tensor([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw,
                      aw * bw - ax * bx - ay * by - az * bz], dtype=th.float64)


def _qinv(q):
    x, y, z, w = q
    return th.tensor([-x, -y, -z, w], dtype=th.float64)


def _qangle(a, b) -> float:
    d = abs(float(th.dot(a / th.linalg.norm(a), b / th.linalg.norm(b))))
    return 2.0 * math.acos(min(1.0, d))


def _rotate(q, v):
    """``R(q) v`` for a xyzw quaternion and a 3-vector (float64)."""
    x, y, z, w = (float(c) for c in q)
    v = th.as_tensor(v, dtype=th.float64)
    R = th.tensor([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                   [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                   [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=th.float64)
    return R @ v


def _qunit(q) -> th.Tensor:
    q = th.as_tensor(q, dtype=th.float64)
    return q / th.linalg.norm(q)


def _shared_axis(equivalences: Sequence[Sequence[float]]) -> Optional[th.Tensor]:
    """The one object-frame axis every non-identity rotation of ``equivalences`` turns about (|dot| >= ``SNAP_CONTINUOUS_AXIS_DOT``),
    or None (none, or several axes)."""
    axis = None
    for q in equivalences or []:
        v = th.as_tensor([float(c) for c in q[:3]], dtype=th.float64)
        n = float(th.linalg.norm(v))
        if n < 1e-9:
            continue
        if axis is None:
            axis = v / n
        elif abs(float(th.dot(v / n, axis))) < SNAP_CONTINUOUS_AXIS_DOT:
            return None
    return axis


def continuous_symmetry_axis(equivalences: Optional[Sequence[Sequence[float]]]) -> Optional[th.Tensor]:
    """The object-frame axis of a continuous symmetry, or None: ``equivalences`` (xyzw, the identity implied or listed) hold at
    least ``SNAP_CONTINUOUS_MIN_EQUIVALENCES`` rotations, all about one axis - a ``zinf`` part's 5° group about object z. Eleven
    steps of 30° (the geometric fallback's most) are a discrete symmetry."""
    eqs = list(equivalences or [])
    return _shared_axis(eqs) if len(eqs) >= SNAP_CONTINUOUS_MIN_EQUIVALENCES else None


def swing_twist(q, axis) -> Tuple[Optional[th.Tensor], Optional[th.Tensor]]:
    """``q = swing ⊗ twist`` (xyzw, the Hamilton product of ``_qmul``): ``twist`` turns about the unit ``axis``, ``swing`` about an
    axis perpendicular to it. ``(None, None)`` for a half turn about a perpendicular axis, which has no twist."""
    q = _qunit(q)
    a = th.as_tensor(axis, dtype=th.float64)
    a = a / th.linalg.norm(a)
    t = th.cat([float(th.dot(q[:3], a)) * a, q[3:]])
    n = float(th.linalg.norm(t))
    if n < 1e-9:
        return None, None
    t = t / n
    return _qunit(_qmul(q, _qinv(t))), t


def apply_fix(p_fix, q_fix, p, q) -> Tuple[th.Tensor, th.Tensor]:
    """The pose ``(p, q)`` moved by the world transform ``(p_fix, q_fix)`` (x -> R(q_fix) x + p_fix): ``fix ∘ pose`` (float64)."""
    return _rotate(q_fix, p) + th.as_tensor(p_fix, dtype=th.float64), _qunit(_qmul(q_fix, _qunit(q)))


def symmetric_snap(p_male, q_male, p_female, q_female, q_obj, equivalences: Optional[Sequence[Sequence[float]]]
                   ) -> Tuple[th.Tensor, th.Tensor, Dict[str, Any]]:
    """The world transform ``fix = (p_fix, q_fix)`` (x -> R(q_fix) x + p_fix) that seats a symmetric part, anchored on the
    attachment frames: the male frame's origin goes onto the female frame's (a turn about the object origin would shift a part
    whose attachment point sits off it - lid/pbmsji, lid/ppkjuy: 1.4-2.7 mm).

    A continuous symmetry about ``a_obj`` (:func:`continuous_symmetry_axis`, object frame) keeps the part's spin: ``q_err =
    q_female ⊗ q_male^-1`` is split into swing ⊗ twist about the part's axis in the world ``a = R(q_obj) a_obj`` and only the
    swing - the tilt correction, its axis perpendicular to ``a`` - is applied (``continuous``). Otherwise, or for a half-turn flip
    (no twist), the male frame of the nearest equivalent (the smallest angle to the female frame, as ``attachment_alignment``
    picks it) is carried onto the female frame - ``attachment_alignment``'s ``fix`` (``nearest_equivalence``; with no
    equivalences ``raw``: OmniGibson's frame snap). Returns ``(p_fix, q_fix, info)``; ``info`` holds JSON-safe floats: ``mode``,
    ``moved_m`` (the male origin's move), ``tilt_rad`` / ``spin_rad`` (the applied rotation's swing / signed twist about the
    part's axis - its equivalences' common axis, else object z)."""
    p_m, q_m = th.as_tensor(p_male, dtype=th.float64), _qunit(q_male)
    p_f, q_f = th.as_tensor(p_female, dtype=th.float64), _qunit(q_female)
    q_o = _qunit(q_obj)
    eqs = [[float(v) for v in e] for e in (equivalences or [])]
    a_obj = continuous_symmetry_axis(eqs)
    q_fix, mode = None, None
    if a_obj is not None:
        swing, _ = swing_twist(_qmul(q_f, _qinv(q_m)), _rotate(q_o, a_obj))
        if swing is not None:
            q_fix, mode = swing, "continuous"
    if q_fix is None:
        q_best, best = q_m, _qangle(q_m, q_f)
        for e in eqs:
            q_alt = _qmul(_qmul(_qmul(q_o, th.as_tensor(e, dtype=th.float64)), _qinv(q_o)), q_m)
            ang = _qangle(q_alt, q_f)
            if ang < best:
                q_best, best = q_alt, ang
        q_fix, mode = _qunit(_qmul(q_f, _qinv(_qunit(q_best)))), ("nearest_equivalence" if eqs else "raw")
    p_fix = p_f - _rotate(q_fix, p_m)
    axis = _shared_axis(eqs)
    a = _rotate(q_o, axis if axis is not None else th.tensor([0.0, 0.0, 1.0], dtype=th.float64))
    swing, twist = swing_twist(q_fix, a)
    if twist is None:
        tilt, spin = math.pi, 0.0
    else:
        tilt = 2.0 * math.acos(min(1.0, abs(float(swing[3]))))
        spin = 2.0 * math.atan2(float(th.dot(twist[:3], a)), float(twist[3]))
        spin = math.atan2(math.sin(spin), math.cos(spin))
    info = {"mode": mode, "moved_m": round(float(th.linalg.norm(p_f - p_m)), 6), "tilt_rad": round(tilt, 6), "spin_rad": round(spin, 6)}
    return p_fix, q_fix, info


def snap_joint_frame(p_fix, q_fix, p_male, q_male, p_female, q_female) -> Tuple[th.Tensor, th.Tensor]:
    """The male link's pose in the female link's frame after the snap, computed from the transform (``fix ∘ male``) instead of
    links re-read right after the teleport (their poses may show it only after the next physics step; OmniGibson's ``_attach``
    never re-reads them either) - the joint frame of :func:`attach_in_place`."""
    p_new, q_new = apply_fix(p_fix, q_fix, p_male, q_male)
    q_f = _qunit(q_female)
    return _rotate(_qinv(q_f), p_new - th.as_tensor(p_female, dtype=th.float64)), _qunit(_qmul(_qinv(q_f), q_new))


def attachment_alignment(child: Any, parent: Any, equivalences: Optional[Sequence[Sequence[float]]] = None, with_fix: bool = False) -> Dict[str, Any]:
    """Meta-link alignment of ``child``'s male link with ``parent``'s free female link, modulo the child's symmetry: for every
    object-frame equivalence ``q_e`` (identity included) the male frame of the equivalently rotated child is
    ``q_obj q_e q_obj^-1 q_male``; the smallest orientation difference to the female frame decides. Returns ``{"ok", "pos_m",
    "rot_rad", "rot_raw_rad", "equivalence", "child_link", "parent_link", "first_free_pair"}``; ``ok`` False with ``reason``
    when the pair has no attachment meta links or is already attached elsewhere."""
    from omnigibson.object_states import AttachedTo

    if AttachedTo not in getattr(child, "states", {}) or AttachedTo not in getattr(parent, "states", {}):
        return {"ok": False, "reason": "no AttachedTo state"}
    st_c, st_p = child.states[AttachedTo], parent.states[AttachedTo]
    if st_c.parent is not None:
        return {"ok": st_c.parent == parent, "reason": "already attached"}
    cands = st_c._get_parent_candidates(parent) or {}
    ctx = attachment_context(child, parent) or {}
    if equivalences is None:
        equivalences = ctx.get("equivalences") or []
    eqs = [th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float64)] + [th.tensor([float(v) for v in q], dtype=th.float64) for q in equivalences]
    op, oq = (x.double() for x in child.get_position_orientation())
    best: Optional[Dict[str, Any]] = None
    first_free: Optional[Tuple[str, str]] = None
    for child_name, parent_names in cands.items():
        for parent_name in sorted(parent_names):
            if st_p.children.get(parent_name) is not None:
                continue
            if first_free is None:
                first_free = (child_name, parent_name)
            if ctx.get("child_link") and ctx.get("parent_link") and (child_name, parent_name) != (ctx["child_link"], ctx["parent_link"]):
                continue
            cl, pl = st_c.links[child_name], st_p.links[parent_name]
            cp, cq = (x.double() for x in cl.get_position_orientation())
            pp, pq = (x.double() for x in pl.get_position_orientation())
            pos = float(th.linalg.norm(cp - pp))
            raw = _qangle(cq, pq)
            rot, which, q_best = raw, 0, cq
            for k, qe in enumerate(eqs[1:], start=1):
                q_alt = _qmul(_qmul(_qmul(oq, qe), _qinv(oq)), cq)
                ang = _qangle(q_alt, pq)
                if ang < rot:
                    rot, which, q_best = ang, k, q_alt
            rec = {"ok": pos <= ATTACH_POS_MAX_M and rot <= ATTACH_ORN_MAX_RAD, "pos_m": round(pos, 5), "rot_rad": round(rot, 5), "rot_raw_rad": round(raw, 5),
                   "equivalence": which, "n_equivalences": len(eqs) - 1, "child_link": child_name, "parent_link": parent_name}
            if with_fix:
                # the world transform carrying the (best equivalent) male frame onto the female frame: applied to the hand it is the
                # closed-loop correction of an attach place that leaves a symmetric part's free yaw alone
                q_fix = _qmul(pq, _qinv(q_best / th.linalg.norm(q_best)))
                p_fix = pp - th.as_tensor(_rotate(q_fix, cp), dtype=th.float64)
                rec["fix"] = (p_fix.float(), q_fix.float())
            if best is None or (rec["ok"] and not best["ok"]) or (rec["ok"] == best["ok"] and rec["pos_m"] + rec["rot_rad"] < best["pos_m"] + best["rot_rad"]):
                best = rec
    if best is None:
        return {"ok": False, "reason": "no free attachment meta-link pair"}
    best["first_free_pair"] = list(first_free) if first_free else None
    return best


def joined_by_attachment(obj: Any) -> bool:
    """``obj`` is a member of a BEHAVIOR attachment joint right now: the child (``AttachedTo.parent`` set) or a parent
    (one of its female links has a child)."""
    from omnigibson.object_states import AttachedTo

    st = getattr(obj, "states", {}).get(AttachedTo) if hasattr(getattr(obj, "states", None), "get") else None
    if st is None:
        return False
    if getattr(st, "parent", None) is not None:
        return True
    return any(c is not None for c in (getattr(st, "children", None) or {}).values())


def is_robot(obj: Any) -> bool:
    """``obj`` is a robot (an OmniGibson ``Robot``; a stand-in with ``is_robot`` or a ``controller_action_idx`` in the tests)."""
    try:
        from omnigibson.robots import Robot

        if isinstance(obj, Robot):
            return True
    except Exception:  # noqa: BLE001 - no simulator module (the test stubs)
        pass
    return bool(getattr(obj, "is_robot", False)) or hasattr(obj, "controller_action_idx")


def drift_settled_kind(obj: Any) -> Optional[str]:
    """The ``DRIFT_SETTLED_ENTITIES`` class of ``obj`` when its ``pose_settled`` is judged by drift after an OmniGibson False:
    ``"robot"``, ``"attached"`` or ``"all"`` (any other object, when enabled); None = the stock velocity-gated rule stands."""
    kind = "robot" if is_robot(obj) else ("attached" if joined_by_attachment(obj) else "all")
    if kind in DRIFT_SETTLED_ENTITIES or "all" in DRIFT_SETTLED_ENTITIES:
        return kind
    return None


def settled_by_drift(obj: Any) -> Dict[str, Any]:
    """``pose_settled`` by configuration drift alone, read from the object's ``PoseSettled`` window: active, the buffer
    full (``n_steps`` distinct simulator steps) and every sample's root pose (and joint positions) within the state's own
    tolerances of the window anchor — OmniGibson's rule minus the per-step velocity gate that a fixed attachment joint's
    solver residual (1-3 mm/s at 0.0x mm of motion) or the robot chassis' wheel-contact residual (1-2.5 mm/s at 2 um of
    motion) never lets pass. For the robot the joints include the six virtual base joints, so the base's planar drift is
    checked twice (root pose and joints). Returns ``{"ok", "n", "n_steps", "max_pos_drift_m", "max_ori_drift_rad",
    "max_joint_drift", "velocity_still_steps"}``."""
    from omnigibson.object_states import PoseSettled

    st = obj.states.get(PoseSettled) if hasattr(obj.states, "get") else None
    if st is None:
        return {"ok": False, "reason": "no PoseSettled state"}
    if not getattr(st, "is_active", False):
        return {"ok": False, "reason": "PoseSettled inactive"}
    hist = list(getattr(st, "_history", []) or [])
    n_steps = int(getattr(st, "n_steps", len(hist)))
    out: Dict[str, Any] = {"n": len(hist), "n_steps": n_steps, "velocity_still_steps": sum(1 for e in hist if e[4])}
    if len(hist) < n_steps or not hist:
        out.update({"ok": False, "reason": "window not full"})
        return out
    _, anchor_pos, anchor_quat, anchor_jpos, _ = hist[0]
    max_pos, max_ori, max_joint = 0.0, 0.0, 0.0
    for _, pos, quat, jpos, _ in hist:
        max_pos = max(max_pos, float(th.linalg.norm(pos - anchor_pos)))
        max_ori = max(max_ori, _qangle(quat.double(), anchor_quat.double()))
        if anchor_jpos is not None and jpos is not None and jpos.numel() == anchor_jpos.numel() and jpos.numel() > 0:
            max_joint = max(max_joint, float(th.max(th.abs(jpos - anchor_jpos))))
    pos_tol = float(getattr(st, "_pos_tolerance", 0.01))
    ori_tol = float(getattr(st, "_ori_tolerance", 0.05))
    joint_tol = float(getattr(st, "_joint_tolerance", 0.05))
    out.update({"max_pos_drift_m": round(max_pos, 6), "max_ori_drift_rad": round(max_ori, 6), "max_joint_drift": round(max_joint, 6),
                "tolerances": {"pos_m": pos_tol, "ori_rad": ori_tol, "joint": joint_tol},
                "ok": max_pos <= pos_tol and max_ori <= ori_tol and max_joint <= joint_tol})
    return out


def held_by_robot(obj: Any) -> bool:
    """``obj`` is in a robot's assisted grasp (``_ag_obj_in_hand``) in its scene."""
    scene = getattr(obj, "scene", None)
    for robot in getattr(scene, "robots", None) or []:
        ag = getattr(robot, "_ag_obj_in_hand", None) or {}
        if any(o is obj or (o is not None and getattr(o, "name", None) == getattr(obj, "name", None)) for o in ag.values()):
            return True
    return False


def attach_in_place(child: Any, parent: Any, child_link: str, parent_link: str, can_joint_break: bool = True,
                    rel_pose: Optional[Tuple[th.Tensor, th.Tensor]] = None) -> bool:
    """``AttachedTo._attach`` without its teleport: the BEHAVIOR fixed joint between ``child``'s male link and ``parent``'s
    female link is created at the links' **current** relative pose, so nothing moves when the joint appears - or at
    ``rel_pose`` (the male link's pose in the female link's frame) when the caller has just moved the child and computed it
    (:func:`snap_joint_frame`).

    OmniGibson's ``_attach`` first re-poses the child so the two meta-link frames coincide exactly (raw frames, 6-DoF) and
    only then creates the joint. For a symmetric part whose alignment was accepted modulo its recorded equivalences (a
    ``zinf`` cap: ~72 yaw steps of 5°) the raw residual can be a full equivalence step: cap/xxqyrt 0920-203052 was corrected
    to 2.6 mm / 0.011 rad modulo the symmetry, the raw yaw residual was 0.090 rad, and ``set_value`` twisted the cap 5.1°
    inside the still-closed fingers - a 7.4 mm yank when they opened, i.e. the visible snap the owner rejects. The joint
    frame here is the child link's pose in the parent link's frame (identity in the child frame); the bookkeeping is the
    stock one (``parent`` / ``parent_link`` / the parent's ``children`` map, the stock joint prim path, so ``get_value``,
    ``_detach`` and the joint-break subscription behave as after ``set_value``). Returns True when the joint was created."""
    import omnigibson.utils.transform_utils as T
    from omnigibson.object_states import AttachedTo
    from omnigibson.object_states import attached_to as _AT
    from omnigibson.utils.constants import JointType
    from omnigibson.utils.usd_utils import create_joint

    st = child.states[AttachedTo]
    if st.parent is not None:
        return st.parent == parent
    c_link, p_link = child.links.get(child_link), parent.links.get(parent_link)
    if c_link is None or p_link is None:
        return False
    if rel_pose is not None:
        rel_pos, rel_quat = (th.as_tensor(x, dtype=th.float32) for x in rel_pose)
    else:
        parent_pos, parent_quat = p_link.get_position_orientation()
        child_pos, child_quat = c_link.get_position_orientation()
        # the child link (= the joint frame) expressed in the parent link's frame: pose_child = pose_parent @ rel
        rel_pos, rel_quat = T.relative_pose_transform(child_pos, child_quat, parent_pos, parent_quat)
    child.keep_still()
    parent.keep_still()
    st.parent = parent
    st.parent_link = p_link
    parent.states[AttachedTo].children[p_link.body_name] = child
    kwargs = {"break_force": _AT.m.DEFAULT_BREAK_FORCE, "break_torque": _AT.m.DEFAULT_BREAK_TORQUE} if can_joint_break else {}
    create_joint(
        prim_path=st.attachment_joint_prim_path,
        joint_type=JointType.JOINT_FIXED,
        body0=f"{p_link.prim_path}",
        body1=f"{c_link.prim_path}",
        joint_frame_in_parent_frame_pos=rel_pos,
        joint_frame_in_parent_frame_quat=rel_quat,
        joint_frame_in_child_frame_pos=th.zeros(3),
        joint_frame_in_child_frame_quat=th.tensor([0.0, 0.0, 0.0, 1.0]),
        exclude_from_articulation=True,
        **kwargs,
    )
    return True


def snap_join(child: Any, parent: Any, child_link: str, parent_link: str,
              equivalences: Optional[Sequence[Sequence[float]]] = None) -> Tuple[bool, Dict[str, Any]]:
    """The ``snap`` join of :func:`attach_if_aligned` (the pair's collision already filtered): the child's root moved by
    :func:`symmetric_snap` (its equivalences: ``equivalences``, else the registered pair's), both objects kept still, and the
    fixed joint created at the relative pose that transform gives (:func:`snap_joint_frame`). Returns ``(joined, info)``."""
    p_m, q_m = (x.double() for x in child.links[child_link].get_position_orientation())
    p_f, q_f = (x.double() for x in parent.links[parent_link].get_position_orientation())
    p_o, q_o = (x.double() for x in child.get_position_orientation())
    eqs = equivalences if equivalences is not None else (attachment_context(child, parent) or {}).get("equivalences") or []
    p_fix, q_fix, info = symmetric_snap(p_m, q_m, p_f, q_f, q_o, eqs)
    p_new, q_new = apply_fix(p_fix, q_fix, p_o, q_o)
    child.set_position_orientation(position=p_new.float(), orientation=q_new.float())
    child.keep_still()
    parent.keep_still()
    rel = snap_joint_frame(p_fix, q_fix, p_m, q_m, p_f, q_f)
    return bool(attach_in_place(child, parent, child_link, parent_link, rel_pose=rel)), info


def join_mode() -> str:
    """The attach rule's join (``ISG_ATTACH_JOIN_MODE``): ``snap`` (the part moved onto its seat by :func:`symmetric_snap`, then
    joined; default), ``in_place`` (the joint at the resting pose) or ``set_value`` (the stock ``AttachedTo.set_value``; for a
    registered pair its ``_attach`` override joins in place)."""
    mode = (os.environ.get("ISG_ATTACH_JOIN_MODE") or "snap").strip()
    if mode not in JOIN_MODES:
        raise ValueError(f"ISG_ATTACH_JOIN_MODE={mode!r}: one of {JOIN_MODES}")
    return mode


#: (child, parent, female link) -> the pair's insertion axis in the female link's frame (seat_axis)
_SEAT_AXIS_LOCAL: Dict[Tuple[str, str, str], th.Tensor] = {}


def seat_axis(child: Any, parent: Any, parent_link: str) -> th.Tensor:
    """The pair's insertion axis in the world (unit, pointing out of the seat): the female link's +z, unless it deviates more than
    45 deg from the direction from the partner's collision-hull centroid to the female link - then that direction (the sampler's
    rule, ``sampling.v2.tasks.attachment``: BEHAVIOR's attachment frames need only match between male and female; the cork /
    wine bottle frames have +z along the bottle's -y). Fixed in the female link's frame at the first call for the pair."""
    from omnigibson.object_states import AttachedTo

    pf, qf = (x.double() for x in parent.states[AttachedTo].links[parent_link].get_position_orientation())
    key = (getattr(child, "name", str(child)), getattr(parent, "name", str(parent)), parent_link)
    loc = _SEAT_AXIS_LOCAL.get(key)
    if loc is None:
        z = _rotate(qf, th.tensor([0.0, 0.0, 1.0], dtype=th.float64))
        axis = z
        pts = _boundary_points(parent).double()
        if len(pts):
            geo = pf - pts.mean(dim=0)
            if float(th.linalg.norm(geo)) > 0.02:
                g = geo / th.linalg.norm(geo)
                if math.degrees(math.acos(max(-1.0, min(1.0, float(th.dot(g, z)))))) > 45.0:
                    axis = g
        loc = _rotate(_qinv(qf), axis)
        _SEAT_AXIS_LOCAL[key] = loc
    a = _rotate(qf, loc)
    return a / th.linalg.norm(a)


def seat_offset(child: Any, parent: Any, child_link: str, parent_link: str) -> Dict[str, float]:
    """The male link against the female link along the pair's insertion axis (:func:`seat_axis`): ``lateral_m`` across it and
    ``vertical_m`` along it (+ = the male frame out of the seat, i.e. above it for a part put on from above)."""
    from omnigibson.object_states import AttachedTo

    cp, _ = (x.double() for x in child.states[AttachedTo].links[child_link].get_position_orientation())
    pp, _ = (x.double() for x in parent.states[AttachedTo].links[parent_link].get_position_orientation())
    a = seat_axis(child, parent, parent_link)
    d = cp - pp
    v = float(th.dot(d, a))
    return {"lateral_m": round(float(th.linalg.norm(d - v * a)), 5), "vertical_m": round(v, 5)}


def pair_in_contact(child: Any, parent: Any) -> bool:
    """PhysX reported a contact between a link of ``child`` and a link of ``parent`` in the latest physics step."""
    from omnigibson.utils.usd_utils import RigidContactAPI

    return bool(RigidContactAPI.is_in_contact(scene_idx=child.scene.idx, query_set=[child], with_set=[parent], ignore_set=None, current_only=True))


#: (scene index, child name, parent name) of the pairs whose collision is filtered right now (set_pair_collision_filter). The
#: scene index (None for a single-scene world) keys every per-pair record below: a vectorized world's clones share their object
#: names, and a key without it made one clone's filter / pending record stand for every clone's (2026-10-23).
_FILTERED: set = set()


def _pair_key(child: Any, parent: Any):
    """``(scene index, child name, parent name)`` of a pair - the clone's own record in a vectorized world."""
    return (getattr(getattr(child, "scene", None), "idx", None), getattr(child, "name", None), getattr(parent, "name", None))


def reset_pair_records(scene: Any = None) -> Dict[str, int]:
    """Forget the per-pair records of ``scene`` (its index; None = every scene): the pairs still filtered are restored first
    (``set_pair_collision_filter(..., False)`` through the scene's object registry), then the pending / released sets are cleared.
    The vectorized loader calls it for a clone before the clone goes back to its template state (the USD filtered-pairs
    relationships are not part of the scene state, so a pair left filtered would stay filtered into the next episode)."""
    idx = None if scene is None else getattr(scene, "idx", None)
    match = (lambda k: True) if scene is None else (lambda k: k[0] == idx)
    out = {"unfiltered": 0, "pending_cleared": 0}
    for key in [k for k in list(_FILTERED) if match(k)]:
        child = parent = None
        if scene is not None:
            child, parent = scene.object_registry("name", key[1]), scene.object_registry("name", key[2])
        if child is not None and parent is not None:
            try:
                set_pair_collision_filter(child, parent, False)
                out["unfiltered"] += 1
                continue
            except Exception as e:  # noqa: BLE001
                import logging

                logging.getLogger(__name__).warning("reset_pair_records: %s/%s not unfiltered: %r", key[1], key[2], e)
        _FILTERED.discard(key)
    for pending in (_PENDING_UNFILTER, _PENDING_FILTER, _RELEASED_TOUCHING):
        for key in [k for k in list(pending) if match(k)]:
            pending.discard(key)
            out["pending_cleared"] += 1
    return out


def _filtered_pairs_rel(link: Any):
    """The link's USD filtered-pairs relationship, its schema applied when missing - call inside an editing block (a link's own
    ``_collision_filter_api`` applies a missing schema in a block of its own, which cannot nest)."""
    import omnigibson.lazy as lazy

    api = lazy.pxr.UsdPhysics.FilteredPairsAPI
    prim = link._prim
    return (api(prim) if prim.HasAPI(api) else api.Apply(prim)).GetFilteredPairsRel()


def set_pair_collision_filter(child: Any, parent: Any, filtered: bool) -> bool:
    """Filter (``filtered``) or restore the collision between every link of ``child`` and every link of ``parent``: the USD
    filtered-pairs relationship of both. While the simulation plays this is the stock route (``AttachedTo.
    _disable_collision_between_child_and_parent``): dump the state, stop, edit, play, load the state - PhysX re-creates the pair's
    actors on the edit and an edit while playing leaves its tensor views dead (attach step 0, 2026-10-10: the next pose read failed
    and ``og.sim.update_handles()`` segfaulted); the simulation clock (``og.sim.current_time``) restarts at the play. Stopped (or
    inside an editing block): the edit alone. Never call it inside a physics step or a state load (see the deferrals of
    :func:`install_attach_overrides`). Idempotent; returns True when it changed something."""
    import omnigibson as og

    key = _pair_key(child, parent)
    if bool(filtered) == (key in _FILTERED):
        return False

    def edit():
        for cl in child.links.values():
            for pl in parent.links.values():
                rc, rp = _filtered_pairs_rel(cl), _filtered_pairs_rel(pl)
                if filtered:
                    rc.AddTarget(pl.prim_path)
                    rp.AddTarget(cl.prim_path)
                else:
                    rc.RemoveTarget(pl.prim_path)
                    rp.RemoveTarget(cl.prim_path)

    if getattr(og.sim, "_editing_usd", False):
        edit()
    elif og.sim.is_playing():
        state = og.sim.dump_state()
        og.sim.stop()
        with og.sim.editing_usd():
            edit()
        og.sim.play()
        og.sim.load_state(state)
    else:
        with og.sim.editing_usd():
            edit()
    if filtered:
        _FILTERED.add(key)
    else:
        _FILTERED.discard(key)
    return True


#: deferred work for registered pairs, done by the next attach_if_aligned (the per-step check, after the step's physics): the pair's
#: collision restored after a detach / filtered after a re-attach that happened inside a state load (no physics restart there), and
#: a part released while already touching its partner (the release hook runs while the step's actions are applied)
_PENDING_UNFILTER: set = set()
_PENDING_FILTER: set = set()
_RELEASED_TOUCHING: set = set()


#: every join this process made for a registered pair: {"child", "parent", "trigger", "mode", "lateral_m", "vertical_m",
#: "rot_rad", "pos_m", "contact", "released", "snap"? ({"mode", "moved_m", "tilt_rad", "spin_rad", ...}), "sim_time"?} (run_isg
#: copies it into meta.json)
JOINS: List[Dict[str, Any]] = []


def attach_if_aligned(child: Any, parent: Any, equivalences: Optional[Sequence[Sequence[float]]] = None, in_place: bool = True,
                      released: bool = False, trigger: str = "per_step") -> Dict[str, Any]:
    """The BEHAVIOR attachment of ``child`` to ``parent`` by the attach rule (module docstring): no robot holds the child, it is
    in PhysX contact with the parent, and its male link is aligned with the parent's free female link modulo its symmetry within
    ``JOIN_LATERAL_MAX_M`` / ``JOIN_ANGLE_MAX_RAD`` / ``JOIN_VERTICAL_MIN_M``..``JOIN_VERTICAL_MAX_M``. Then the pair's collision
    is filtered and the fixed joint created (:func:`join_mode`) - at the current relative pose (``in_place``), or after moving
    the child onto its seat (``snap``): by :func:`symmetric_snap`, which keeps a continuously symmetric part's spin about its
    axis (only the tilt goes) and otherwise carries the nearest equivalent onto the female frame, both ``keep_still``; the
    joint frame comes from that transform (:func:`snap_joint_frame`), not from the links re-read after the teleport.
    ``set_value`` (or ``in_place`` False) = the stock ``AttachedTo.set_value(True, bypass_alignment_checking=True)``.
    ``released``: called at the instant a robot's assisted grasp
    let go of ``child`` (:func:`attach_on_release`), while its in-hand registry still names it; the contact rule still applies
    (no join at the release instant of a part that does not touch its partner), and a part that does touch is joined by the
    step's own check after its physics (the filter restarts the physics, which must not happen while the step's actions are
    applied) - the join keeps the trigger ``release_hook``. The deferred filter / unfilter of the pair runs first. Returns the
    alignment record (JSON-safe; no transform tensors) with ``attached``, ``joint``, ``snap`` (``mode``, ``moved_m``,
    ``tilt_rad``, ``spin_rad``, and the alignment at the join) for a snap, and ``reason`` when not joined."""
    from omnigibson.object_states import AttachedTo

    key = _pair_key(child, parent)
    hook_call = trigger == "release_hook"
    a = attachment_alignment(child, parent, equivalences)
    a.pop("fix", None)
    if a.get("reason") == "already attached":
        a["attached"] = bool(a["ok"])
        if a["attached"] and key in _PENDING_FILTER:
            set_pair_collision_filter(child, parent, True)
            _PENDING_FILTER.discard(key)
        return a
    if key in _PENDING_UNFILTER:
        set_pair_collision_filter(child, parent, False)
        _PENDING_UNFILTER.discard(key)
    if trigger == "per_step" and key in _RELEASED_TOUCHING:
        released, trigger = True, "release_hook"
    _RELEASED_TOUCHING.discard(key)
    a["trigger"] = trigger
    if "child_link" not in a:
        a["attached"] = False
        return a
    a.update(seat_offset(child, parent, a["child_link"], a["parent_link"]))
    a["aligned"] = bool(a["lateral_m"] <= JOIN_LATERAL_MAX_M and a["rot_rad"] <= JOIN_ANGLE_MAX_RAD and JOIN_VERTICAL_MIN_M <= a["vertical_m"] <= JOIN_VERTICAL_MAX_M)
    if not a["aligned"]:
        a.update({"attached": False, "reason": "not aligned"})
        return a
    if held_by_robot(child) and not released:
        a.update({"attached": False, "reason": "held by a robot"})
        return a
    a["contact"] = pair_in_contact(child, parent)
    if not a["contact"]:
        a.update({"attached": False, "reason": "not in contact with the partner"})
        return a
    pair = (a["child_link"], a["parent_link"])
    if a.get("first_free_pair") is not None and tuple(a["first_free_pair"]) != pair:
        a.update({"attached": False, "reason": "the aligned pair is not OmniGibson's first free candidate pair"})
        return a
    if hook_call:
        _RELEASED_TOUCHING.add(key)
        a.update({"attached": False, "reason": "touching at the release: joined by the step's own check after its physics"})
        return a
    try:
        import omnigibson as og

        sim_time = round(float(og.sim.current_time), 4)  # before the filter restarts the clock: the step whose physics just ran
    except Exception:  # noqa: BLE001 - no simulator (the test stubs)
        sim_time = None
    mode = join_mode() if in_place else "set_value"
    if mode == "set_value":
        a["attached"] = bool(child.states[AttachedTo].set_value(parent, True, bypass_alignment_checking=True))
    else:
        set_pair_collision_filter(child, parent, True)
        if mode == "snap":
            a["attached"], info = snap_join(child, parent, a["child_link"], a["parent_link"], equivalences)
            a["snap"] = {**info, **{k: a[k] for k in ("pos_m", "rot_rad", "lateral_m", "vertical_m")}}
        else:
            a["attached"] = bool(attach_in_place(child, parent, a["child_link"], a["parent_link"]))
    a["joint"] = mode
    if a["attached"]:
        _JOINTS[0] += 1
        _PENDING_FILTER.discard(key)
        JOINS.append({"child": child.name, "parent": parent.name, "trigger": trigger, "mode": mode, "released": bool(released), "contact": True,
                      **{k: a[k] for k in ("lateral_m", "vertical_m", "rot_rad", "pos_m")}, **({"snap": a["snap"]} if "snap" in a else {}),
                      **({"sim_time": sim_time} if sim_time is not None else {})})
    return a


#: the record of the latest attach at a release (attach_on_release), for the oracle's audit
LAST_RELEASE_ATTACH: Dict[str, Any] = {}


def attach_on_release(robot: Any, arm: str, obj: Any) -> Optional[Dict[str, Any]]:
    """The release hook (a robot's ``release_callbacks``, called by our Behavior-1K fork's ``_release_grasp``): a registered
    attachment child let go is judged by the attach rule in that instant - in particular it must already touch its partner -
    and, when it holds, joined by that step's own check after its physics (``trigger`` ``release_hook``); a part released above
    its partner is joined by the per-step check once it lands. The same rule acts for an evaluated policy's release. Returns the
    alignment record (also kept in ``LAST_RELEASE_ATTACH``), or None when ``obj`` is not an attachment child."""
    scene = getattr(obj, "scene", None)
    for parent_name in attachment_partners(obj):
        if scene is None:
            continue
        parent = scene.object_registry("name", parent_name)
        if parent is None:
            continue
        rec = attach_if_aligned(obj, parent, released=True, trigger="release_hook")
        rec.update(hook="release", arm=arm, parent=parent_name)
        LAST_RELEASE_ATTACH.clear()
        LAST_RELEASE_ATTACH.update({k: v for k, v in rec.items() if k != "fix"})
        return rec
    return None


def _registered_member(obj: Any) -> bool:
    """``obj`` is a child or parent of a pair registered for its own scene or for a single env (scene None)."""
    name = getattr(obj, "name", None)
    idx = getattr(getattr(obj, "scene", None), "idx", None)
    return any(name in (c, p) for s, c, p in _ATTACHMENTS if s is None or s == idx)


def _link_name(obj: Any, link: Any) -> Optional[str]:
    return next((n for n, l in getattr(obj, "links", {}).items() if l is link), None)


_OVERRIDES: Dict[str, Any] = {}


def install_attach_overrides() -> bool:
    """``AttachedTo`` for the registered pairs (once per process; the overrides look the registry up at call time): ``_update``
    (OmniGibson's contact-triggered join) does nothing for a registered child or parent - only :func:`attach_if_aligned` joins
    them; ``_attach`` (``set_value(True)``, e.g. a state load restoring an attached checkpoint) joins a registered pair at the
    current relative pose instead of teleporting the child onto the raw female frame; ``_detach`` (e.g. a state load rolling back
    to before the join) frees it. Both run inside state loads, where the physics must not restart: the pair's collision is
    filtered / restored by the next per-step check (``_PENDING_FILTER`` / ``_PENDING_UNFILTER``). Other objects keep the stock
    behaviour. Returns True when installed by this call."""
    if _OVERRIDES:
        return False
    try:
        from omnigibson.object_states import AttachedTo
    except Exception:  # noqa: BLE001 - no simulator (the test stubs)
        return False
    stock = {"_update": AttachedTo._update, "_attach": AttachedTo._attach, "_detach": AttachedTo._detach}

    def _update(self):
        if _registered_member(self.obj):
            return None
        return stock["_update"](self)

    def _attach(self, other, child_link, parent_link, joint_type=None, can_joint_break=True):
        key = _pair_key(self.obj, other)
        if attachment_context(self.obj, other) is not None:
            attach_in_place(self.obj, other, _link_name(self.obj, child_link), _link_name(other, parent_link), can_joint_break=can_joint_break)
            _PENDING_UNFILTER.discard(key)
            if key not in _FILTERED:
                _PENDING_FILTER.add(key)
            return None
        return stock["_attach"](self, other, child_link, parent_link, joint_type=joint_type, can_joint_break=can_joint_break)

    def _detach(self):
        parent = self.parent
        out = stock["_detach"](self)
        key = _pair_key(self.obj, parent)
        if parent is not None and attachment_context(self.obj, parent) is not None:
            _PENDING_FILTER.discard(key)
            if key in _FILTERED:
                _PENDING_UNFILTER.add(key)
        return out

    AttachedTo._update, AttachedTo._attach, AttachedTo._detach = _update, _attach, _detach
    _OVERRIDES.update(stock)
    return True


def install_release_hook(robots: Any) -> int:
    """Registers :func:`attach_on_release` in the ``release_callbacks`` of every robot in ``robots`` (``env.robots``: one
    list per scene), once each; returns the number of robots that carry it."""
    n, stack = 0, list(robots or [])
    while stack:
        robot = stack.pop()
        if isinstance(robot, (list, tuple)):
            stack.extend(robot)
            continue
        callbacks = getattr(robot, "release_callbacks", None)
        if callbacks is None:
            robot.release_callbacks = callbacks = []
        if attach_on_release not in callbacks:
            callbacks.append(attach_on_release)
        n += 1
    return n


def evaluate_predicate(predicate_cls, *entities) -> bool:
    """Drop-in for ``evaluate_bddl_predicate`` with the robust ``under`` / ``ontop`` / ``touching`` re-checks (module docstring)."""
    from omnigibson.utils.bddl_utils import evaluate_bddl_predicate

    name = str(getattr(predicate_cls, "__name__", predicate_cls)).lower()
    name = {"posesettled": "pose_settled", "attachedto": "attached"}.get(name, name)  # the BDDL name (the callers pass it) or the state class
    ok = bool(evaluate_bddl_predicate(predicate_cls, *entities))
    if name == "pose_settled" and len(entities) == 1 and entities[0] is not None and _is_object(entities[0]):
        if ok:
            _log(name, entities, "omnigibson", True)
            return True
        kind = drift_settled_kind(entities[0])
        if kind is None:
            _log(name, entities, "omnigibson", False)
            return False
        try:
            s = settled_by_drift(entities[0])
        except Exception as e:  # noqa: BLE001 - a read failure must not turn into a verdict
            s = {"ok": False, "error": repr(e)}
        s["entity"] = kind
        _log(name, entities, "drift" if s.get("ok") else "none", bool(s.get("ok")), s)
        return bool(s.get("ok"))
    if name not in ("under", "ontop", "touching", "attached", "inside") or len(entities) != 2 or any(e is None or not _is_object(e) for e in entities):
        _log(name, entities, "omnigibson", ok)
        return ok
    if ok:
        _log(name, entities, "omnigibson", True)
        return True
    if name == "inside":
        if has_container_link(entities[1]):
            _log(name, entities, "omnigibson", False)
            return False
        try:
            r = inside_rays(entities[0], entities[1])
        except Exception as e:  # noqa: BLE001 - a raycast failure must not turn into a verdict
            r = {"ok": False, "error": repr(e)}
        _log(name, entities, "rays" if r.get("ok") else "none", bool(r.get("ok")), r)
        return bool(r.get("ok"))
    if name == "attached":
        # attached(child, parent): the BEHAVIOR joint does not exist yet — align modulo the child's symmetry and create it
        try:
            a = attach_if_aligned(entities[0], entities[1])
        except Exception as e:  # noqa: BLE001 - an alignment failure must not turn into a verdict
            a = {"ok": False, "attached": False, "error": repr(e)}
        got = bool(a.get("attached"))
        _log(name, entities, "alignment" if got else "none", got, a)
        return got
    if name == "touching":
        try:
            t = touching_geometry(entities[0], entities[1])
        except Exception as e:  # noqa: BLE001 - a geometry failure must not turn into a verdict
            t = {"ok": False, "error": repr(e)}
        _log(name, entities, "geometry" if t.get("ok") else "none", bool(t.get("ok")), t)
        return bool(t.get("ok"))
    # under(a, b): a is under b -> a supports b; ontop(a, b): a rests on b -> b supports a
    lower, upper = (entities[0], entities[1]) if name == "under" else (entities[1], entities[0])
    try:
        r = support_rays(lower, upper)
    except Exception as e:  # noqa: BLE001 - a raycast failure must not turn into a verdict
        r = {"ok": False, "error": repr(e)}
    if r.get("ok"):
        _log(name, entities, "rays", True, r)
        return True
    try:
        g = support_geometry(lower, upper)
    except Exception as e:  # noqa: BLE001
        g = {"ok": False, "error": repr(e)}
    if g.get("ok"):
        _log(name, entities, "geometry", True, g)
        return True
    _log(name, entities, "none", False, {"rays": r, "geometry": g})
    return False


# ----------------------------------------------------------------------------------------------------------------
# branches
# ----------------------------------------------------------------------------------------------------------------


def _is_object(e: Any) -> bool:
    return hasattr(e, "links") and hasattr(e, "get_position_orientation")


def _boundary_points(obj: Any) -> th.Tensor:
    """World collision boundary points (the convex-hull vertices of every link's collision geometry)."""
    try:
        import omnigibson as og

        og.sim.sync_physx_to_fabric()  # the boundary points read the render-side transforms (sim_utils.local_hull)
    except Exception:  # noqa: BLE001
        pass
    pts = [p for p in (link.collision_boundary_points_world for link in getattr(obj, "links", {}).values()) if p is not None and len(p)]
    return th.cat([th.as_tensor(p).float() for p in pts], dim=0) if pts else th.zeros(0, 3)


def hull_gap(points_a: th.Tensor, points_b: th.Tensor) -> float:
    """Separation of the convex hulls of two point sets: max over the hull-of-b half-spaces of the signed distance of a's
    points, minimised over a's points, and vice versa; the smaller of the two. Exact for a face-to-face contact (the
    contact face's vertices against the other face's plane), a lower bound for edge / corner contacts; negative when the
    hulls interpenetrate."""
    import numpy as np
    from scipy.spatial import ConvexHull

    hulls = {id(points_a): ConvexHull(points_a.numpy().astype(np.float64)), id(points_b): ConvexHull(points_b.numpy().astype(np.float64))}

    def surface_points(pts: th.Tensor) -> th.Tensor:
        # the hull vertices plus every hull triangle's centroid and edge midpoints: two faces overlapping only partially
        # (each overhanging the other) have no vertex inside the other's extent, and vertices alone would overestimate
        h = hulls[id(pts)]
        tri = pts[th.as_tensor(h.simplices, dtype=th.long)]  # (n_tri, 3, 3)
        extra = th.cat([tri.mean(dim=1), (tri[:, 0] + tri[:, 1]) / 2, (tri[:, 1] + tri[:, 2]) / 2, (tri[:, 0] + tri[:, 2]) / 2], dim=0)
        return th.cat([pts, extra], dim=0)

    def one_way(pts: th.Tensor, hull_pts: th.Tensor) -> float:
        eq = th.as_tensor(hulls[id(hull_pts)].equations).float()  # n·x + d <= 0 inside
        signed = surface_points(pts) @ eq[:, :3].T + eq[:, 3]  # (n_pts, n_faces)
        return float(signed.max(dim=1).values.min())

    return min(one_way(points_a, points_b), one_way(points_b, points_a))


def touching_geometry(a: Any, b: Any) -> Dict[str, Any]:
    """Geometric touching: the two objects' convex hulls within ``TOUCHING_GAP_MAX_M`` of each other (or interpenetrating)."""
    pa, pb = _boundary_points(a), _boundary_points(b)
    if len(pa) < 4 or len(pb) < 4:
        return {"ok": False, "why": "no collision boundary points"}
    gap = hull_gap(pa, pb)
    return {"ok": gap <= TOUCHING_GAP_MAX_M, "gap_m": round(gap, 5), "gap_max_m": TOUCHING_GAP_MAX_M}


def _link_paths(obj: Any) -> List[str]:
    return [link.prim_path for link in getattr(obj, "links", {}).values()]


def has_container_link(obj: Any) -> bool:
    """Whether ``obj`` has a fillable / openfillable meta link - the volume OmniGibson's ``Inside`` tests (it judges such a container)."""
    try:
        from omnigibson.object_states.contains import m as contains_m

        types = set(contains_m.CONTAINER_META_LINK_TYPES)
    except Exception:  # noqa: BLE001 - the macro moved: the types of this OmniGibson version
        types = set(CONTAINER_META_LINK_TYPES)
    return any(getattr(link, "is_meta_link", False) and getattr(link, "meta_link_type", None) in types for link in getattr(obj, "links", {}).values())


def inside_both_sides(sides: Dict[str, Sequence[bool]], min_axes: int = INSIDE_MIN_AXES) -> Tuple[bool, List[str]]:
    """BEHAVIOR's kinematic ``inside``: the axes (``{"z": (+, -), "x": ..., "y": ...}``) with the container on both sides of the inner
    object; True when there are at least ``min_axes`` of them."""
    axes = [k for k, v in sides.items() if len(v) == 2 and bool(v[0]) and bool(v[1])]
    return len(axes) >= int(min_axes), axes


def inside_rays(inner: Any, container: Any) -> Dict[str, Any]:
    """The ``inside`` rays branch (module docstring): the inner object's AABB centre in the container's AABB, then six rays from it -
    up / down and both ways along the container's own x / y axes, ``INSIDE_RAY_MARGIN_M`` beyond the container's AABB diagonal, the inner
    object's bodies ignored - and :func:`inside_both_sides` on the sides where a ray meets the container."""
    from omnigibson.utils.sampling_utils import raytest_batch

    lo_i, hi_i = _aabb(inner)
    lo_c, hi_c = _aabb(container)
    c = (lo_i + hi_i) / 2
    centre = [round(float(v), 4) for v in c]
    if not bool(((c >= lo_c) & (c <= hi_c)).all()):
        return {"ok": False, "why": "the inner object's AABB centre is outside the container's AABB", "centre": centre}
    _, q = container.get_position_orientation()
    x, y, z, w = [float(v) for v in th.as_tensor(q).float()]
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    ca, sa = math.cos(yaw), math.sin(yaw)
    reach = float(th.linalg.norm(hi_c - lo_c)) + INSIDE_RAY_MARGIN_M
    keys, starts, ends = [], [], []
    for k, a in (("z", (0.0, 0.0, 1.0)), ("x", (ca, sa, 0.0)), ("y", (-sa, ca, 0.0))):
        for sgn in (1.0, -1.0):
            keys.append((k, 0 if sgn > 0 else 1))
            starts.append([float(c[0]), float(c[1]), float(c[2])])
            ends.append([float(c[i]) + sgn * reach * a[i] for i in range(3)])
    hits = raytest_batch(starts, ends, only_closest=False, ignore_bodies=_link_paths(inner) or None)
    paths = tuple(_link_paths(container))
    sides: Dict[str, List[bool]] = {"z": [False, False], "x": [False, False], "y": [False, False]}
    for (k, i), hs in zip(keys, hits):
        hs = hs if isinstance(hs, list) else [hs]
        sides[k][i] = any(h.get("hit") and str(h.get("rigidBody", "")).startswith(paths) for h in hs)
    ok, axes = inside_both_sides(sides)
    return {"ok": ok, "axes_both_sides": axes, "sides": sides, "centre": centre}


def _aabb(obj: Any) -> Tuple[th.Tensor, th.Tensor]:
    lo, hi = obj.aabb
    return th.as_tensor(lo).float(), th.as_tensor(hi).float()


def _footprint_points(upper: Any) -> List[Tuple[float, float]]:
    """The resting object's footprint centre and a ring at half its footprint radius (world xy)."""
    lo, hi = _aabb(upper)
    cx, cy = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)
    r = 0.25 * float(min(hi[0] - lo[0], hi[1] - lo[1]))  # half the footprint radius
    pts = [(cx, cy)]
    for k in range(RING_POINTS):
        a = 2 * math.pi * k / RING_POINTS
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _surface_under(lower: Any, upper: Any, x: float, y: float) -> Optional[float]:
    """z of the supporting object's surface under world (x, y): a downward ray from above its AABB top with the resting
    object's bodies ignored; None when the first hit is not the supporting object."""
    from omnigibson.utils.sampling_utils import raytest_batch

    lo_l, hi_l = _aabb(lower)
    start = [x, y, float(hi_l[2]) + 0.05]
    end = [x, y, float(lo_l[2]) - 0.02]
    hits = raytest_batch([start], [end], only_closest=True, ignore_bodies=_link_paths(upper) or None)
    h = hits[0] if hits else {}
    if not h.get("hit") or not str(h.get("rigidBody", "")).startswith(tuple(_link_paths(lower))):
        return None
    return float(h["position"][2])


def support_rays(lower: Any, upper: Any) -> Dict[str, Any]:
    """Branch 2: probe rays from just below the supporting surface up into the resting object across its contact patch."""
    from omnigibson.utils.sampling_utils import raytest_batch

    upper_paths = tuple(_link_paths(upper))
    lower_paths = _link_paths(lower)
    starts, ends = [], []
    n_surface = 0
    for x, y in _footprint_points(upper):
        z = _surface_under(lower, upper, x, y)
        if z is None:
            continue
        n_surface += 1
        starts.append([x, y, z - RAY_BELOW_SURFACE_M])
        ends.append([x, y, z - RAY_BELOW_SURFACE_M + RAY_LENGTH_M])
    if not starts:
        return {"ok": False, "n_rays": 0, "n_surface": 0, "why": "no supporting surface under the footprint"}
    hits = raytest_batch(starts, ends, only_closest=True, ignore_bodies=lower_paths or None)
    n_hit = sum(1 for h in hits if h.get("hit") and str(h.get("rigidBody", "")).startswith(upper_paths))
    return {"ok": n_hit > 0, "n_rays": len(starts), "n_surface": n_surface, "n_hits_on_upper": n_hit}


def support_geometry(lower: Any, upper: Any) -> Dict[str, Any]:
    """Branch 3: bottom-to-surface gap under the footprint centre, footprint centre inside the supporting top, up axis."""
    lo_u, hi_u = _aabb(upper)
    lo_l, hi_l = _aabb(lower)
    cx, cy = float((lo_u[0] + hi_u[0]) / 2), float((lo_u[1] + hi_u[1]) / 2)
    z_surface = _surface_under(lower, upper, cx, cy)
    if z_surface is None:
        return {"ok": False, "why": "no supporting surface under the footprint centre"}
    gap = float(lo_u[2]) - z_surface
    inside = bool(lo_l[0] <= cx <= hi_l[0] and lo_l[1] <= cy <= hi_l[1])
    _, q = upper.get_position_orientation()
    x, y, z, w = [float(v) for v in th.as_tensor(q).float()]
    up_z = 1.0 - 2.0 * (x * x + y * y)  # z component of the rotated local z axis
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, up_z))))
    ok = abs(gap) <= BOTTOM_GAP_MAX_M and inside and tilt <= UP_AXIS_MAX_DEG
    return {"ok": ok, "bottom_gap_mm": round(gap * 1000, 1), "centre_inside_support": inside, "tilt_deg": round(tilt, 1)}


if os.environ.get("ISPATIALGYM_FAST_SUPPORT") in ("1", "check"):
    import sys as _sys

    from ispatialgym.sampling import predicates_fast as _PF

    _PF.install(_sys.modules[__name__], check=os.environ.get("ISPATIALGYM_FAST_SUPPORT") == "check")
