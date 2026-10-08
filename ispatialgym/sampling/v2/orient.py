"""Wrist orientation of a contact (owner 2026-09-27, first-demo v2.1).

The R1Pro parallel-jaw hand is C2-symmetric about its approach axis except for the wrist camera, and its wrist is spherical: a
contact pose and the same pose turned pi about the approach axis (the fingers swapped) are both proposals, and every IK solution
of one gives a solution of the other with the same shoulder and elbow (``conventions.twin_arm_q``: q5 -/+ pi, -q6, -q7). Which
proposals exist is the generator's business (its hard rules: a camera-up-only fist, a camera-up-only flip); :func:`choose`
decides the order among them and the arm configuration:

1. success - the caller's checks plus :func:`camera_clear`; proposals are tried in the order below and the first that passes wins;
2. joint-limit safe - every configuration of the action's chain at least ``IKChecker.JOINT_LIMIT_MARGIN`` from every arm joint
   limit before an unsafe one; neither safe: more clearance first;
3. easier to reach - the smaller largest joint change from where the arm starts; within :data:`REACH_TOL` the generator's own
   order wins (its default: the camera-away / camera-up turn, the library grasp).

Besides the proposals the caller solves, each solution contributes the wrist-flipped configuration of its twin proposal
(source ``map``): the same shoulder / elbow branch, scored from the mapped configurations and checked by the caller only when it
is the one to try (0927 estimate: bar soap / sprouts reached the library pose in the rotated grasp's branch with its clearance
and 1.2-1.3 rad less travel). A proposal that failed contributes the witnesses it left from its chain's start (B2's rescues:
the library grasp stalled at joint 5 at the end of its approach, its twin left from the mapped pre-grasp witness). Every
proposal and decision record carries the arm making the contact (by default the arm of the call).

The arm is one more dimension of the choice (owner 2026-09-27, first-demo v2.2): the proposals of a contact may span both arms
(:func:`with_arms`: every proposal of the default arm ranked before the other arm's, each arm's own order kept, so within
:data:`REACH_TOL` the default arm wins), each solved with its arm's IK and checked against its arm's wrist camera; the reach of
an arm is measured from that arm's own starting posture (``q_start`` per arm). The decision then records ``arm_choice``: the
arms offered, the arm chosen, why it came before the best viable option of the other arm (:func:`arm_reason`) and whether it is
not the default arm."""
from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch as th

from ispatialgym.robot import conventions as C

log = logging.getLogger("ispatialgym.sampling.v2")

#: rad: a reach difference up to this keeps the generator's own order (the continuity tolerance of the first twin-grasp rule, B2)
REACH_TOL = 0.30
#: m: the wrist camera's spheres must keep this from the world with the contacted objects present (= ``tabletop.TWIN_CAMERA_CLEARANCE_M``
#: and the margin of ``iors_small.hand_overlap``)
CAMERA_MARGIN = 0.005
#: rad: joint-limit clearances closer than this count as equal (a flip leaves every binding joint but joint 5 where it is)
CLEARANCE_EQ = 1e-3

Pose = Tuple[th.Tensor, th.Tensor]


@dataclass
class Proposal:
    """One orientation of a contact: ``label`` (the generator's name), ``rank`` (its order in the generator, 0 = its default),
    ``payload`` (what the caller records), ``chain`` (the poses the arm goes through, the pre-pose first), ``q_first`` (the
    configuration the chain starts from: a wrist-flipped solution of the twin proposal), ``source`` ``ik`` | ``map``, ``arm``
    (the arm making the contact; None = the arm of the :func:`choose` call - there is no arm choice yet, the field carries
    the arm through the scoring and the records)."""

    label: str
    rank: int
    payload: Any
    chain: List[Pose]
    q_first: Optional[th.Tensor] = None
    source: str = "ik"
    arm: Optional[str] = None


@dataclass
class Scored:
    proposal: Proposal
    ok: bool
    result: Any = None
    qs: Optional[List[th.Tensor]] = None
    margin: Optional[float] = None
    nearest: Optional[str] = None
    reach: Optional[float] = None
    why: str = ""
    predicted: bool = False

    def facts(self) -> Dict[str, Any]:
        return {"label": self.proposal.label, "arm": self.proposal.arm, "rank": self.proposal.rank, "source": self.proposal.source, "ok": bool(self.ok),
                "margin_rad": None if self.margin is None else round(float(self.margin), 4), "nearest_joint": self.nearest,
                "reach_rad": None if self.reach is None else round(float(self.reach), 3), **({"why": str(self.why)[:120]} if self.why else {}),
                **({"predicted": True} if self.predicted else {})}


def is_twin(a: Pose, b: Pose, pos_tol: float = 1e-3, ang_tol: float = 1e-2) -> bool:
    """``b`` is ``a`` turned pi about its approach axis (the same pad centre, the fingers swapped)."""
    t = C.twin_pose((C.as_tensor(a[0]).float(), C.as_tensor(a[1]).float()))
    pb, qb = C.as_tensor(b[0]).float(), C.as_tensor(b[1]).float()
    return float(th.linalg.norm(C.as_tensor(t[0]).float() - pb)) <= pos_tol and C.quat_angle(C.as_tensor(t[1]).float(), qb) <= ang_tol


def chain_qs(ikc, arm: str, chain: Sequence[Pose]) -> Optional[List[th.Tensor]]:
    """The configurations the last check left for ``chain`` (``IKChecker.solutions``), None when one is missing."""
    out = witnesses(ikc, arm, chain)
    return out if len(out) == len(chain) else None


def witnesses(ikc, arm: str, chain: Sequence[Pose]) -> List[th.Tensor]:
    """The remembered configurations of ``chain``'s poses from its start up to the first pose without one."""
    out = []
    for p in chain:
        q = ikc.solutions.get((arm, ikc._pose_key(p)))
        if q is None:
            break
        out.append(q)
    return out


def chain_margin(ikc, qs: Sequence[th.Tensor], arm: str) -> Tuple[float, str]:
    worst, joint = float("inf"), ""
    for q in qs:
        m, jn = ikc.joint_limit_margin(q, arm)
        if m < worst:
            worst, joint = m, jn
    return worst, joint


def reach(ikc, q: th.Tensor, q_start, arm: str) -> Optional[float]:
    """Largest arm-joint change from ``q_start`` to ``q`` (rad): what sets the length of the transit to the pre-pose. ``q_start``
    may be a dict ``{arm: configuration}`` (proposals of both arms: each arm from its own starting posture)."""
    if isinstance(q_start, dict):
        q_start = q_start.get(arm)
    if q_start is None or q is None:
        return None
    idx = ikc.kit.robot.arm_control_idx[arm]
    return float((C.as_tensor(q).float()[idx] - C.as_tensor(q_start).float()[idx]).abs().max())


def camera_clear(ikc, q: th.Tensor, arm: str, margin: float = CAMERA_MARGIN) -> bool:
    """The wrist camera (``<arm>_realsense_link``, the one part of the hand the turn moves) keeps ``margin`` from the world as it
    stands - the contacted object and the support included, which the IK removes at the contact pose."""
    fn = getattr(ikc.kit, "links_clear", None)
    if fn is None:
        return True
    try:
        return bool(fn(C.as_tensor(q).float(), [f"{arm}_realsense_link"], float(margin))[0])
    except Exception as e:  # noqa: BLE001 - a kit without the camera link in its sphere model checks nothing
        log.debug("camera_clear: %r", e)
        return True


def prefer(a: Scored, b: Scored) -> int:
    """-1 when ``a`` is tried before ``b`` (the owner's order), 1 when after."""
    sa, sb = a.margin is not None and a.margin >= _safe(), b.margin is not None and b.margin >= _safe()
    if sa != sb:
        return -1 if sa else 1
    if not sa and a.margin is not None and b.margin is not None and abs(a.margin - b.margin) > CLEARANCE_EQ:
        return -1 if a.margin > b.margin else 1
    if a.reach is not None and b.reach is not None and abs(a.reach - b.reach) > REACH_TOL:
        return -1 if a.reach < b.reach else 1
    if a.proposal.rank != b.proposal.rank:
        return -1 if a.proposal.rank < b.proposal.rank else 1
    if a.predicted != b.predicted:  # the configuration the caller solved before a mapped one of the same proposal
        return 1 if a.predicted else -1
    return 0


def _safe() -> float:
    from ispatialgym.sampling.v2.ik import IKChecker

    return float(IKChecker.JOINT_LIMIT_MARGIN)


def reason(chosen: Scored, rest: Sequence[Scored]) -> str:
    """Why ``chosen`` came before the best viable option of another orientation in ``rest`` (the options sorted after it): the
    first of the owner's criteria that separates the two."""
    o = next((r for r in rest if r.proposal.label != chosen.proposal.label), None)
    if o is None:
        return "only feasible"
    safe = _safe()
    c_safe, o_safe = chosen.margin is not None and chosen.margin >= safe, o.margin is not None and o.margin >= safe
    if c_safe and not o_safe:
        return "safety"
    if not c_safe and not o_safe and chosen.margin is not None and (o.margin is None or chosen.margin - o.margin > CLEARANCE_EQ):
        return "clearance"
    if chosen.reach is not None and o.reach is not None and o.reach - chosen.reach > REACH_TOL:
        return "reach"
    return "default"


def arm_reason(chosen: Scored, rest: Sequence[Scored]) -> str:
    """Why ``chosen``'s arm came before the other arm: :func:`reason` against the best viable option of another arm in ``rest``."""
    return reason(chosen, [r for r in rest if r.proposal.arm != chosen.proposal.arm])


def with_arms(proposals_by_arm: Sequence[Tuple[str, Sequence[Proposal]]]) -> List[Proposal]:
    """One contact's proposals for several arms, the default arm first: every proposal carries its arm, each arm's own order is
    kept and ranked after every proposal of the arms before it (within :data:`REACH_TOL` the default arm wins, then the
    generator's order); with more than one arm the labels are prefixed with the arm (``left:grasp_3~r180``)."""
    out: List[Proposal] = []
    base = 0
    many = len(proposals_by_arm) > 1
    for a, props in proposals_by_arm:
        for p in props:
            out.append(Proposal(f"{a}:{p.label}" if many else p.label, base + int(p.rank), p.payload, p.chain, p.q_first, p.source, a))
        base += max((int(p.rank) for p in props), default=-1) + 1
    return out


def viable_arms(decision: Dict[str, Any]) -> List[str]:
    """The arms with a passing option in a decision record, the chosen arm first (the order of a segment's fallback)."""
    arms: List[str] = []
    first = decision.get("arm")
    if decision.get("chosen") is not None and first:
        arms.append(first)
    for o in decision.get("options", []):
        if o.get("ok") and not o.get("predicted") and o.get("arm") and o["arm"] not in arms:
            arms.append(o["arm"])
    return arms


def choose(ikc, arm: str, proposals: Sequence[Proposal], evaluate: Callable[[Proposal], Tuple[bool, Any, Optional[List[th.Tensor]], str]],
           q_start: Optional[th.Tensor], camera_idx: Optional[Sequence[int]] = None, stage: str = "", check_camera: bool = True,
           downstream: Optional[Callable[["Scored"], Tuple[bool, str]]] = None, map_twins: bool = True) -> Tuple[Optional[Scored], Dict[str, Any]]:
    """The orientation of one contact among ``proposals`` (the generator's own order = ``rank``). ``evaluate(proposal)`` runs the
    caller's checks - with ``proposal.q_first`` as the chain's start when set - and returns ``(ok, result, qs, why)`` (``qs``:
    every configuration of the action's chain, None to read them off the IK checker for ``proposal.chain``). ``camera_idx``:
    the chain indices where :func:`camera_clear` applies (default: the last pose, where the IK removed the contacted objects).
    ``downstream(scored)``: the caller's checks that need the chosen chain (the pushes with a brace arm locked at it); an option
    failing them is passed over for the next in the order ("if that one fails, try the other one"). ``map_twins`` False: no
    wrist-flipped options (a downstream that solves the contact again from its own seeds would not keep the flipped branch).
    Each proposal is scored on its own ``arm`` (default: ``arm``, the arm of the call, which is also the default arm of
    ``arm_choice``); wrist twins are paired and mapped within one arm; ``q_start`` is one configuration or ``{arm: configuration}``
    (each arm's reach from its own starting posture). Returns ``(the chosen Scored or None, the decision record)``. A single proposal (the generator allows one turn: a
    camera-up-only fist) is only evaluated: its orientation is the generator's hard rule and the camera check guards the
    reordering only."""
    check_camera = check_camera and len(proposals) > 1
    for p in proposals:
        if p.arm is None:
            p.arm = arm
    scored: List[Scored] = []

    def score(p: Proposal, ok: bool, result: Any, qs: Optional[List[th.Tensor]], why: str, predicted: bool = False) -> Scored:
        if ok and qs is None:
            qs = chain_qs(ikc, p.arm, p.chain)
        if ok and check_camera and qs:
            idx = list(camera_idx) if camera_idx is not None else [len(p.chain) - 1]
            bad = [i for i in idx if 0 <= i < len(qs) and not camera_clear(ikc, qs[i], p.arm)]
            if bad:
                ok, why = False, f"wrist camera within {CAMERA_MARGIN * 1000:.0f} mm of the world at chain pose {bad[0] + 1}/{len(p.chain)}"
        s = Scored(p, bool(ok), result=result, qs=qs, why=why, predicted=predicted)
        if qs:
            s.margin, s.nearest = chain_margin(ikc, qs, p.arm)
            s.reach = reach(ikc, qs[0], q_start, p.arm)
        return s

    for p in proposals:
        ok, result, qs, why = evaluate(p)
        scored.append(score(p, ok, result, qs, why))
    passing = [s for s in scored if s.ok]
    mapped: List[Scored] = []
    for s in (scored if map_twins else []):
        a = s.proposal.arm
        src = s.qs if s.ok else witnesses(ikc, a, s.proposal.chain)
        if not src:
            continue
        for p in proposals:
            if p is s.proposal or p.arm != a or not is_twin(s.proposal.chain[0], p.chain[0]):
                continue
            mq = [ikc.twin_q(q, a) for q in src]
            if any(q is None for q in mq):
                continue
            m = Scored(Proposal(p.label, p.rank, p.payload, p.chain, q_first=mq[0], source="map", arm=a), True, qs=mq, predicted=True)
            m.margin, m.nearest = chain_margin(ikc, mq, a)
            m.reach = reach(ikc, mq[0], q_start, a)
            mapped.append(m)
    key = functools.cmp_to_key(prefer)
    pool = sorted(passing + mapped, key=key)
    chosen: Optional[Scored] = None
    tried: List[Scored] = []
    while pool:
        s = pool.pop(0)
        if s.predicted:
            ok, result, qs, why = evaluate(s.proposal)
            v = score(s.proposal, ok, result, qs, why)
            tried.append(v)
            if not v.ok:
                continue
            if pool and prefer(pool[0], v) < 0:  # verified worse than predicted: back into the order
                pool = sorted(pool + [v], key=key)
                continue
            s = v
        if downstream is not None:
            ok_d, why_d = downstream(s)
            if not ok_d:
                s.ok, s.why = False, f"downstream: {why_d}"
                continue
        chosen = s
        break
    rest = [o for o in pool if o.ok]
    decision: Dict[str, Any] = {"stage": stage, "arm": arm if chosen is None else chosen.proposal.arm, "chosen": None if chosen is None else chosen.proposal.label,
                                "source": None if chosen is None else chosen.proposal.source,
                                "reason": None if chosen is None else reason(chosen, rest),
                                "against_default": bool(chosen is not None and chosen.proposal.rank > 0),
                                "options": [s.facts() for s in scored] + [m.facts() for m in mapped] + [t.facts() for t in tried]}
    arms = sorted({p.arm for p in proposals}, key=lambda a: (a != arm, a))
    if len(arms) > 1:
        decision["arm_choice"] = {"default": arm, "offered": arms, "chosen": None if chosen is None else chosen.proposal.arm,
                                  "reason": None if chosen is None else arm_reason(chosen, rest),
                                  "against_default": bool(chosen is not None and chosen.proposal.arm != arm), "viable": viable_arms(decision)}
    if len(proposals) < 2:
        return chosen, decision
    _count(ikc, "orient_choices")
    ac = decision.get("arm_choice")
    if ac is not None:
        _count(ikc, "orient_arm_choices")
    if chosen is not None:
        _count(ikc, f"orient_reason.{decision['reason'].replace(' ', '_')}")
        if chosen.proposal.source == "map":
            _count(ikc, "orient_source.map")
        if decision["against_default"]:
            _count(ikc, "orient_against_default")
        if ac is not None:
            _count(ikc, f"orient_arm.{ac['chosen']}")
            _count(ikc, f"orient_arm_reason.{ac['reason'].replace(' ', '_')}")
        log.info("orient [%s] %s (%s%s)%s%s | %s", f"{stage}, {chosen.proposal.arm} arm", chosen.proposal.label, decision["reason"], ", wrist-flipped branch" if chosen.proposal.source == "map" else "",
                 " AGAINST the generator's order" if decision["against_default"] else "",
                 f" [arm {ac['chosen']} of {'/'.join(ac['offered'])}: {ac['reason']}]" if ac is not None else "",
                 " | ".join(_side(o) for o in decision["options"]))
    return chosen, decision


def _side(o: Dict[str, Any]) -> str:
    if not o.get("ok"):
        return f"{o['label']}[{o['source']}] fail: {o.get('why', '')[:60]}"
    return f"{o['label']}[{o['source']}{', predicted' if o.get('predicted') else ''}] margin {o['margin_rad']} ({o.get('nearest_joint')}) reach {o['reach_rad']}"


def _count(ikc, key: str) -> None:
    fn = getattr(ikc, "_count", None)
    if fn is not None:
        fn(key)


def choose_first(ikc, arm: str, stream, *, key: Callable[[Any], Pose], label: Callable[[Any], str], chain: Callable[[Any], List[Pose]],
                 evaluate: Callable[[Proposal], Tuple[bool, Any, Optional[List[th.Tensor]], str]], q_start: Optional[th.Tensor],
                 camera_idx: Optional[Sequence[int]] = None, stage: str = "",
                 downstream: Optional[Callable[["Scored"], Tuple[bool, str]]] = None,
                 arm_of: Optional[Callable[[Any], str]] = None) -> Tuple[Optional[Scored], Optional[Dict[str, Any]], List[str]]:
    """A generator's candidates in its own order, its wrist twins grouped (:func:`pairs` by ``key``, within one arm): the first
    group with a passing member, the member :func:`choose` picks. ``arm_of(candidate)``: the candidate's arm (default ``arm``).
    Returns ``(chosen, decision, the failures of the groups before it)``."""
    fails: List[str] = []
    for group in pairs(stream, key, arm_of):
        props = [Proposal(label(c), i, c, chain(c), arm=arm_of(c) if arm_of is not None else None) for i, c in enumerate(group)]
        chosen, decision = choose(ikc, arm, props, evaluate, q_start, camera_idx=camera_idx, stage=stage, downstream=downstream)
        if chosen is not None:
            return chosen, decision, fails
        fails.extend(f"{o['label']}: {o.get('why', '')}" for o in decision["options"] if not o.get("ok"))
    return None, None, fails


def pairs(stream, key: Callable[[Any], Pose], arm_of: Optional[Callable[[Any], str]] = None):
    """Group a candidate stream into orientation groups: consecutive candidates of one arm (``arm_of``, default: all one arm)
    whose ``key`` poses are wrist twins form one group (a generator yields a contact's two turns back to back); every other
    candidate is a group of its own. Order kept."""
    group: List[Any] = []
    same_arm = (lambda a, b: arm_of(a) == arm_of(b)) if arm_of is not None else (lambda a, b: True)
    for item in stream:
        if group and len(group) == 1 and same_arm(group[0], item) and is_twin(key(group[0]), key(item)):
            group.append(item)
            yield group
            group = []
            continue
        if group:
            yield group
        group = [item]
    if group:
        yield group
