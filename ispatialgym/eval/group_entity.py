"""Nonphysical scope entity for BDDL ``:groups`` symbols.

A :class:`GroupEntity` is what ``task.object_scope[<group symbol>]`` holds. Pointwise predicates on a group are
expanded per member at parse time, so the only atoms that reach the entity are the aggregates
``(pose_reached ?g)`` and ``(layout_reached ?g)``. Both are evaluated from one shared member-to-slot assignment
computed once per simulator step (see :meth:`GroupEntity.evaluate`).

The entity exposes ``states`` keyed by the OmniGibson state classes so that the stock
``omnigibson.utils.bddl_utils.evaluate_bddl_predicate`` dispatch works unchanged.
"""

from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch as th

from omnigibson.object_states import LayoutReached, PoseReached

from ispatialgym.eval.geometry import (
    assign_by_cost_fn,
    assign_by_slot_position,
    orientation_error,
    relation_error,
)
from ispatialgym.eval.targets import GroupTarget

if os.environ.get("ISPATIALGYM_FAST_GEOMETRY") in ("1", "check"):
    from ispatialgym.eval.geometry_fast import orientation_error, relation_error  # noqa: F811

#: ISPATIALGYM_FAST_ASSIGN=1: one relation_error per (from member, to member, relation) within an evaluation, shared by
#: the assignment's permutations and the final relation errors (their inputs are the same there); =check also recomputes
#: every reused value and prints FAST_ASSIGN_CHECK counts
FAST_ASSIGN = os.environ.get("ISPATIALGYM_FAST_ASSIGN", "")
_ASSIGN_COUNTS = {"reused": 0, "mismatches": 0, "fresh_s": 0.0}


@dataclass
class GroupEvaluation:
    """Result of one group evaluation (one shared assignment)."""

    assignment: Dict[str, str]  # member instance -> slot id
    member_errors: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    relation_errors: List[Dict[str, object]] = field(default_factory=list)
    pose_reached: Optional[bool] = None  # None when target_slots / pose_tolerances are not configured
    layout_reached: Optional[bool] = None  # None when layout_relations / layout_tolerances are not configured

    @property
    def max_pose_position_error(self) -> Optional[float]:
        vals = [e["position_error"] for e in self.member_errors.values() if e.get("position_error") is not None]
        return max(vals) if vals else None

    @property
    def max_pose_orientation_error(self) -> Optional[float]:
        vals = [e["orientation_error"] for e in self.member_errors.values() if e.get("orientation_error") is not None]
        return max(vals) if vals else None

    @property
    def max_layout_position_error(self) -> Optional[float]:
        vals = [r["position_error"] for r in self.relation_errors]
        return max(vals) if vals else None

    @property
    def max_layout_orientation_error(self) -> Optional[float]:
        vals = [r["orientation_error"] for r in self.relation_errors]
        return max(vals) if vals else None

    def as_dict(self) -> Dict[str, object]:
        return {
            "assignment": dict(self.assignment),
            "pose_reached": self.pose_reached,
            "layout_reached": self.layout_reached,
            "member_errors": {m: dict(e) for m, e in self.member_errors.items()},
            "relation_errors": [dict(r) for r in self.relation_errors],
            "max_pose_position_error": self.max_pose_position_error,
            "max_pose_orientation_error": self.max_pose_orientation_error,
            "max_layout_position_error": self.max_layout_position_error,
            "max_layout_orientation_error": self.max_layout_orientation_error,
        }


class _GroupState:
    """Minimal object-state look-alike (``get_value`` / ``set_value``) bound to a :class:`GroupEntity`."""

    def __init__(self, entity: "GroupEntity"):
        self.entity = entity
        self.obj = entity

    def get_value(self):
        raise NotImplementedError

    def set_value(self, *args, **kwargs):
        return False

    def get_error_report(self):
        return None


class GroupPoseReached(_GroupState):
    def get_value(self):
        ev = self.entity.evaluate()
        return bool(ev.pose_reached) if ev is not None else False

    def get_error_report(self):
        ev = self.entity.evaluate()
        if ev is None or ev.pose_reached is None:
            return None
        return {
            "position_error": ev.max_pose_position_error,
            "orientation_error": ev.max_pose_orientation_error,
            "within_tolerance": bool(ev.pose_reached),
            "members": {m: dict(e) for m, e in ev.member_errors.items()},
            "assignment": dict(ev.assignment),
        }


class GroupLayoutReached(_GroupState):
    def get_value(self):
        ev = self.entity.evaluate()
        return bool(ev.layout_reached) if ev is not None else False

    def get_error_report(self):
        ev = self.entity.evaluate()
        if ev is None or ev.layout_reached is None:
            return None
        return {
            "position_error": ev.max_layout_position_error,
            "orientation_error": ev.max_layout_orientation_error,
            "within_tolerance": bool(ev.layout_reached),
            "relations": [dict(r) for r in ev.relation_errors],
            "assignment": dict(ev.assignment),
        }


class GroupEntity:
    """Scope entity for a declared group.

    Args:
        name: group symbol (e.g. ``blocks_group.n.01_1``).
        members: member instance names in declaration order.
        resolve_member: callable mapping a member instance name to its simulator object (or ``None``).
    """

    def __init__(self, name: str, members: List[str], resolve_member: Callable[[str], object]):
        self.name = name
        self.members = list(members)
        self._resolve_member = resolve_member
        self.target: Optional[GroupTarget] = None
        self.initialized = True
        self.category = "group"
        self.states = {PoseReached: GroupPoseReached(self), LayoutReached: GroupLayoutReached(self)}
        self._cache_key = None
        self._cache: Optional[GroupEvaluation] = None

    def __repr__(self):
        return f"GroupEntity({self.name!r}, members={self.members})"

    # ------------------------------------------------------------------ configuration

    def set_target(self, target: Optional[GroupTarget]) -> None:
        if target is not None and set(target.members) != set(self.members):
            raise ValueError(
                f"Group target {target.name!r} members {sorted(target.members)} do not match BDDL group "
                f"{self.name!r} members {sorted(self.members)}"
            )
        self.target = target
        self.invalidate()

    @property
    def has_target(self) -> bool:
        return self.target is not None

    def invalidate(self) -> None:
        self._cache_key = None
        self._cache = None

    # ------------------------------------------------------------------ evaluation

    def member_poses(self) -> Optional[Dict[str, tuple]]:
        poses = {}
        for m in self.members:
            obj = self._resolve_member(m)
            if obj is None or not getattr(obj, "initialized", False):
                return None
            pos, quat = obj.get_position_orientation()
            poses[m] = (pos.detach().float(), quat.detach().float())
        return poses

    def evaluate(self) -> Optional[GroupEvaluation]:
        """Evaluate the group with one shared assignment. Cached while the member poses are unchanged."""
        if self.target is None:
            return None
        poses = self.member_poses()
        if poses is None:
            return None
        key = tuple((m, tuple(p.tolist()), tuple(q.tolist())) for m, (p, q) in poses.items())
        if key == self._cache_key and self._cache is not None:
            return self._cache
        ev = self._evaluate(poses)
        self._cache_key, self._cache = key, ev
        return ev

    @staticmethod
    def _relation_error(memo, poses, m_from, m_to, i, rel, eqs, pos_tol, ori_tol):
        if memo is None:
            return relation_error(poses[m_from], poses[m_to], rel.pose, eqs, eqs, pos_tol, ori_tol)
        key = (m_from, m_to, i)
        hit = memo.get(key)
        if hit is None:
            hit = memo[key] = relation_error(poses[m_from], poses[m_to], rel.pose, eqs, eqs, pos_tol, ori_tol)
            return hit
        _ASSIGN_COUNTS["reused"] += 1
        if FAST_ASSIGN == "check":
            t0 = time.perf_counter()
            fresh = relation_error(poses[m_from], poses[m_to], rel.pose, eqs, eqs, pos_tol, ori_tol)
            _ASSIGN_COUNTS["fresh_s"] += time.perf_counter() - t0
            if [struct.pack("<d", float(v)) for v in fresh] != [struct.pack("<d", float(v)) for v in hit]:
                _ASSIGN_COUNTS["mismatches"] += 1
                print(f"FAST_ASSIGN_CHECK mismatch {key}: memo {hit!r} fresh {fresh!r}", flush=True)
            if _ASSIGN_COUNTS["reused"] % 1000 == 0:
                print(f"FAST_ASSIGN_CHECK reused {_ASSIGN_COUNTS['reused']} mismatches {_ASSIGN_COUNTS['mismatches']}", flush=True)
            return fresh
        return hit

    def _evaluate(self, poses: Dict[str, tuple]) -> GroupEvaluation:
        target = self.target
        eqs = target.orientation_equivalences or None
        slots = target.target_slots
        relations = target.layout_relations
        memo = {} if FAST_ASSIGN in ("1", "check") else None
        assignment = self._assign(poses, memo)
        ev = GroupEvaluation(assignment=assignment)

        # Absolute placement: every member at its assigned slot.
        if slots:
            pos_tol, ori_tol = target.pose_tolerances if target.pose_tolerances else (None, None)
            all_ok = True
            for m, slot_id in assignment.items():
                slot = slots[slot_id]
                p, q = poses[m]
                pos_err = float(th.norm(p - slot.position))
                ori_err = orientation_error(q, slot.orientation, eqs)
                ok = (pos_tol is None or pos_err <= pos_tol) and (ori_tol is None or ori_err <= ori_tol)
                all_ok = all_ok and ok
                ev.member_errors[m] = {
                    "slot": slot_id,
                    "position_error": pos_err,
                    "orientation_error": ori_err,
                    "within_tolerance": ok,
                }
            ev.pose_reached = all_ok if target.pose_tolerances else None

        # Relative arrangement: every relation between the assigned members.
        if relations:
            pos_tol, ori_tol = target.layout_tolerances if target.layout_tolerances else (None, None)
            slot_to_member = {s: m for m, s in assignment.items()}
            all_ok = True
            for i, rel in enumerate(relations):
                m_from, m_to = slot_to_member[rel.from_slot], slot_to_member[rel.to_slot]
                pos_err, ori_err = self._relation_error(memo, poses, m_from, m_to, i, rel, eqs, pos_tol, ori_tol)
                ok = (pos_tol is None or pos_err <= pos_tol) and (ori_tol is None or ori_err <= ori_tol)
                all_ok = all_ok and ok
                ev.relation_errors.append(
                    {
                        "from": rel.from_slot,
                        "to": rel.to_slot,
                        "from_member": m_from,
                        "to_member": m_to,
                        "position_error": pos_err,
                        "orientation_error": ori_err,
                        "within_tolerance": ok,
                    }
                )
            ev.layout_reached = all_ok if target.layout_tolerances else None
        return ev

    def _assign(self, poses: Dict[str, tuple], memo: Optional[dict] = None) -> Dict[str, str]:
        target = self.target
        if target.assignment_mode == "fixed":
            return dict(target.member_to_slot)
        slots = target.target_slots
        if target.assignment_cost == "slot_position":
            return assign_by_slot_position({m: p for m, (p, _) in poses.items()}, {s: sl.position for s, sl in slots.items()})

        # Layout cost: the permutation whose relations best reproduce the stored relative poses (invariant to a
        # global rigid motion of the whole arrangement). Ties broken by orientation error, then lexicographically.
        members, slot_ids = sorted(poses), sorted(slots)
        eqs = target.orientation_equivalences or None
        pos_tol, ori_tol = target.layout_tolerances if target.layout_tolerances else (None, None)

        def perm_cost(perm):
            slot_to_member = {slot_ids[j]: members[i] for i, j in enumerate(perm)}
            total_pos, total_ori = 0.0, 0.0
            for k, rel in enumerate(target.layout_relations):
                pos_err, ori_err = self._relation_error(
                    memo, poses, slot_to_member[rel.from_slot], slot_to_member[rel.to_slot], k, rel, eqs, pos_tol, ori_tol
                )
                total_pos += pos_err
                total_ori += ori_err
            return (round(total_pos, 6), round(total_ori, 6))

        return assign_by_cost_fn(members, slot_ids, perm_cost)
