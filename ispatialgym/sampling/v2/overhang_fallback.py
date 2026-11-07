"""The overhang FALLBACK of a failed side grasp in the IORS small-scale draw (i7-plateqtf2, owner decision 2026-11-08 02:35 item 2,
approved with the oracle program change).

When the exact IK of a drawn SIDE grasp (library ``mode == "side"``) fails in ``tabletop._phase3_draw`` for a reason the sampler cannot
fix - the shared support policy (``support_policy: ...`` at the grasp pose, ``support policy on the approach tail``): the grasp pose is
fixed relative to the object on its support, so no other stance or position changes it - the draw may take the OVERHANG pick instead:
push the object along the support toward the edge the manipulation stance stands at until the grasp's fingers stand in the free space
beyond it, then take the same library grasp at the overhang. Fallback only: the direct grasp is always tried first.

The mechanism is IORSLarge's ``pick_overhang_push`` (:mod:`ispatialgym.sampling.v2.overhang`, ``IORSLarge._overhang_pick``): its
methods are bound here UNCHANGED (:class:`SmallOverhangPick`, the attribute contract of ``tests/test_iors_large_overhang._Sampler``) on
an adapter of the IORSSmall sampler, so the push and the overhang pick take exactly IORSLarge's checks - the push geometry from the
support edge (raycast), the fingers' sweep beyond the edge and the centre of mass inside, the grasp's exact IK at the pushed pose with the
support present (in the stance's image, the program check's fingertip gate), the push chain's exact IK (open / closed hand, pitches,
light support sweep), the physics settle of the pushed object - and the same records (``contacts.push_pick``, kind ``pick_overhang``;
``grasps[0].pick_overhang``), which ``programs._pick_actions`` turns into push -> retreat -> grasp -> lift (``program_iors_small``).
Options: ``generation.pick_overhang_push`` when the alias sets it (mode ``fallback``), else :data:`overhang.DEFAULTS`; the small program
has no second grasp stance, so ``grasp_back_offs_m`` is always ``[]``.

Switch ``ISPATIALGYM_OVERHANG_FALLBACK`` (default on, ``0`` off). Kit counters ``overhang_fallback.trigger.<reason class>``,
``.tried`` / ``.found`` / ``.failed`` / ``.memo_hit`` (+ IORSLarge's own ``pick_overhang.*``); the attempt summary's ``pick_overhang``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Sequence, Tuple

from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.v2 import overhang as OH
from ispatialgym.sampling.v2.stances import GraspRecord

log = logging.getLogger(__name__)

ENV = "ISPATIALGYM_OVERHANG_FALLBACK"
#: the side-grasp exact-IK failure classes the sampler cannot fix (the support policy at the pose / along the approach tail)
POLICY_CLASSES = ("non_finger_link", "palm_contact", "wrist_camera_contact", "wrist_contact", "finger_penetration", "policy_other")


def on() -> bool:
    return os.environ.get(ENV, "1").strip().lower() not in ("0", "off", "false", "no")


def reason_class(reason: str) -> Optional[str]:
    """The unfixable class of an exact-IK failure reason (:data:`POLICY_CLASSES`), or None (IK / reach / joint limit / collisions with
    other objects / tracking: another stance or position can fix those)."""
    w = str(reason or "").lower()
    if "support_policy" not in w and "support policy" not in w:
        return None
    if "non-finger link" in w:
        return "non_finger_link"
    for k in ("palm_contact", "wrist_camera_contact", "wrist_contact", "finger_penetration"):
        if k in w:
            return k
    return "policy_other"


def is_side_grasp(g: Any) -> bool:
    """A library grasp of mode ``side`` (v2 library ``mode``; its wrist-flipped twin keeps the row's meta)."""
    return str((getattr(g, "meta", None) or {}).get("mode") or "") == "side"


def _count(s, key: str) -> None:
    st = getattr(getattr(s, "kit", None), "stats", None)
    if isinstance(st, dict):
        st[key] = st.get(key, 0) + 1


def options(s) -> Dict[str, Any]:
    """``generation.pick_overhang_push`` of the alias (mode forced to ``fallback``) or the defaults; no second grasp stance."""
    gen = getattr(s, "gen", None)
    v = gen("pick_overhang_push") if callable(gen) else None
    opts = OH.options(v if v not in (None, False, "off") else True)
    opts = dict(opts or OH.options(True))
    opts.update(mode="fallback", grasp_back_offs_m=[])
    return opts


class SmallOverhangPick:
    """IORSLarge's overhang pick bound onto an IORSSmall sampler ``s`` for one drawn object pose ``t_pose`` (module doc). Every attribute
    IORSLarge's methods read comes from ``s`` (``__getattr__``), but the ones whose IORSSmall form differs: ``target`` / ``hull`` /
    ``rest`` (IORSSmall keeps them per role), ``surface`` (the support top), ``tname``, ``pose_on`` (the drawn pose itself), ``num``."""

    def __init__(self, s, t_pose: G.Pose, opts: Dict[str, Any]):
        from ispatialgym.sampling.v2.tasks.iors_large import IORSLarge as IL

        object.__setattr__(self, "_s", s)
        object.__setattr__(self, "_IL", IL)
        object.__setattr__(self, "_t_pose", (G.t(t_pose[0]).float(), G.t(t_pose[1]).float()))
        object.__setattr__(self, "_opts", dict(opts))
        sup = list(s.support_objects())
        self.target = s.obj("target")
        self.hull = s.hull["target"]
        self.rest = s.rest["target"]
        self.surface = {"kind": "support", "z": float(s.top.z), "top": s.top, "obj": sup[0] if sup else None}
        self.tname = s.bindings.one("target").name
        self.v2_overhang = None
        self.robot = getattr(s, "robot", None) or getattr(getattr(s, "world", None), "robot", None)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_s"), name)

    # IORSLarge's methods, unchanged (the same exact checks)
    def _bound(name):  # noqa: N805 - class-body helper
        def call(self, *a, **k):
            return getattr(self._IL, name)(self, *a, **k)
        call.__name__ = name
        return call

    _overhang_edges = _bound("_overhang_edges")
    _overhang_geometry = _bound("_overhang_geometry")
    _overhang_touch_ok = _bound("_overhang_touch_ok")
    _overhang_push_chain = _bound("_overhang_push_chain")
    _overhang_push_branch = _bound("_overhang_push_branch")
    _overhang_drifted = _bound("_overhang_drifted")
    _overhang_settle = _bound("_overhang_settle")
    _overhang_pick = _bound("_overhang_pick")
    _overhang_grasp = _bound("_overhang_grasp")
    overhang_records = _bound("overhang_records")
    target_com_local = _bound("target_com_local")
    num = _bound("num")
    del _bound

    def pick_overhang_opts(self) -> Dict[str, Any]:
        return dict(self._opts)

    def pose_on(self, xy, yaw: float, z: float) -> G.Pose:
        """The object pose the direct grasp failed at (the draw's pose: its rest, yaw and support height)."""
        return self._t_pose

    def _overhang_grasp_stances(self, st, arm, opts):
        return []  # the small program grasps from the manipulation stance (no recorded travel to a second grasp stance)

    def _check_deadline(self, stage: str) -> None:
        f = getattr(self._s, "_check_deadline", None)
        if callable(f):
            f(stage)


def _memo_key(st, g, pose: G.Pose) -> Tuple[Any, ...]:
    p = G.t(pose[0]).float()
    return (tuple(round(float(v), 3) for v in st.base_pose2d), round(float(st.translate), 3), round(float(st.tilt), 3), str(getattr(g, "name", g)),
            round(float(G.yaw_of(G.t(pose[1]).float())), 2), round(float(p[0]), 2), round(float(p[1]), 2))


def try_fallback(s, st, arm: str, g, pose: G.Pose, reason: str, twins: bool, name: str, eef_of) -> Optional[Tuple[G.Pose, GraspRecord]]:
    """The overhang fallback of the side grasp ``g`` whose direct exact IK failed with ``reason`` at the drawn object ``pose`` from stance
    ``st`` (``arm``): ``(the object's INIT pose, the GraspRecord at the overhang)`` or None (not a side grasp, a fixable reason, the switch
    off, no support edge / push / grasp / chain / settle - IORSLarge's reasons, logged). The record carries the pick (``_overhang``) for
    :func:`record` and ``extras["overhang_fallback"]``."""
    if not on() or not is_side_grasp(g) or getattr(s, "overhang_fallback_enabled", None) is False:
        return None
    cls = reason_class(reason)
    if cls is None:
        return None
    _count(s, f"overhang_fallback.trigger.{cls}")
    memo = s.__dict__.setdefault("_overhang_fallback_memo", {})
    key = _memo_key(st, g, pose)
    if key in memo:
        _count(s, "overhang_fallback.memo_hit")
        return None
    _count(s, "overhang_fallback.tried")
    opts = options(s)
    ad = SmallOverhangPick(s, pose, opts)
    p = G.t(pose[0]).float()
    yaw = float(G.yaw_of(G.t(pose[1]).float()))
    ov, why = ad._overhang_pick(st, {arm: st}, [arm], g, yaw, p[:2], float(s.top.z), bool(twins), opts)
    if ov is None:
        memo[key] = why
        _count(s, "overhang_fallback.failed")
        log.info("%s: overhang fallback of %s (%s arm, after '%s') fails: %s", getattr(getattr(s, "gp", None), "variant", "?"), getattr(g, "name", g), arm,
                 str(reason)[:160], str(why)[:900])
        return None
    _count(s, "overhang_fallback.found")
    pick = ov.get("pick")
    g_rec = getattr(pick, "grasp", None) or g
    eio = getattr(pick, "eef_in_object", None) or eef_of(g)
    finger_q = list(getattr(g_rec, "finger_q", []) or []) or None
    rec = GraspRecord(name, ov["arm"], getattr(g_rec, "key", None), getattr(g_rec, "name", None), yaw, ov["pre"], ov["eef"], finger_q=finger_q, eef_in_object=eio)
    rec.extras["overhang_fallback"] = {"trigger": cls, "why": str(reason)[:300], "push_m": round(float(ov["push_m"]), 4)}
    rec._overhang = ov  # in memory: the records are written when the grasp is recorded (:func:`record`)
    log.info("%s: overhang fallback of %s (%s arm, after '%s'): push %.3f m, then the grasp at the overhang", getattr(getattr(s, "gp", None), "variant", "?"),
             getattr(g, "name", g), ov["arm"], str(reason)[:160], float(ov["push_m"]))
    return ov["t_pose"], rec


def record(s, grasp: GraspRecord) -> bool:
    """Write the overhang pick's records of ``grasp`` (``contacts.push_pick`` + ``grasps[0].pick_overhang``: IORSLarge's
    ``overhang_records``) when it is one; True when written."""
    ov = getattr(grasp, "_overhang", None)
    if ov is None:
        return False
    ad = SmallOverhangPick(s, ov["t_pose"], options(s))
    ad.overhang_records(ov, grasp)
    return True


def drop_records(s) -> None:
    """Undo :func:`record` for a rejected candidate (the grasps are cleared by the caller)."""
    c = getattr(getattr(s, "sidecar", None), "contacts", None)
    if isinstance(c, dict):
        c.pop(OH.CONTACT_SLOT, None)


def summary(grasp: GraspRecord) -> Optional[Dict[str, Any]]:
    ov = getattr(grasp, "_overhang", None)
    return None if ov is None else ov.get("facts")


def side_grasps(grasps: Sequence[Any]) -> int:
    return sum(1 for g in grasps if is_side_grasp(g))
