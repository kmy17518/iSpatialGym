"""Nonphysical scope entity of a BDDL ``:groups`` symbol.

Pointwise predicates on a group are expanded per member at parse time, so the only atoms that reach a
:class:`GroupEntity` are the aggregates ``(pose_reached ?g)`` and ``(layout_reached ?g)``. Both are evaluated from one
shared member-to-slot assignment, recomputed only when a member moves. The entity exposes ``states`` keyed by the
OmniGibson state classes, so the stock ``evaluate_bddl_predicate`` dispatch works unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch as th
from omnigibson.object_states import LayoutReached, PoseReached

from ispatialgym.eval.geometry import assign_by_cost_fn, assign_by_slot_position, orientation_error, relation_error
from ispatialgym.eval.targets import GroupTarget


@dataclass
class GroupEvaluation:
    """One group evaluation (one shared assignment)."""

    assignment: dict[str, str]  # member instance -> slot id
    member_errors: dict[str, dict[str, float | None]] = field(default_factory=dict)
    relation_errors: list[dict[str, object]] = field(default_factory=list)
    pose_reached: bool | None = None  # None without target_slots / pose_tolerances
    layout_reached: bool | None = None  # None without layout_relations / layout_tolerances

    @staticmethod
    def _max(values) -> float | None:
        values = [v for v in values if v is not None]
        return max(values) if values else None

    @property
    def max_pose_position_error(self) -> float | None:
        return self._max(e.get("position_error") for e in self.member_errors.values())

    @property
    def max_pose_orientation_error(self) -> float | None:
        return self._max(e.get("orientation_error") for e in self.member_errors.values())

    @property
    def max_layout_position_error(self) -> float | None:
        return self._max(r["position_error"] for r in self.relation_errors)

    @property
    def max_layout_orientation_error(self) -> float | None:
        return self._max(r["orientation_error"] for r in self.relation_errors)


class _GroupState:
    """Minimal object-state look-alike bound to a :class:`GroupEntity`."""

    def __init__(self, entity: GroupEntity):
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
    """Scope entity of a declared group.

    Args:
        name: group symbol (e.g. ``targets_group.n.01_1``).
        members: member instance names in declaration order.
        resolve_member: maps a member instance name to its simulator object (or ``None``).
    """

    def __init__(self, name: str, members: list[str], resolve_member: Callable[[str], object]):
        self.name = name
        self.members = list(members)
        self._resolve_member = resolve_member
        self.target: GroupTarget | None = None
        self.initialized = True
        self.category = "group"
        self.states = {PoseReached: GroupPoseReached(self), LayoutReached: GroupLayoutReached(self)}
        self._cache_key = None
        self._cache: GroupEvaluation | None = None

    def __repr__(self):
        return f"GroupEntity({self.name!r}, members={self.members})"

    def set_target(self, target: GroupTarget | None) -> None:
        if target is not None and set(target.members) != set(self.members):
            raise ValueError(
                f"Group target {target.name!r} members {sorted(target.members)} do not match BDDL group "
                f"{self.name!r} members {sorted(self.members)}"
            )
        self.target = target
        self.invalidate()

    def invalidate(self) -> None:
        self._cache_key = None
        self._cache = None

    def member_poses(self) -> dict[str, tuple] | None:
        poses = {}
        for m in self.members:
            obj = self._resolve_member(m)
            if obj is None or not getattr(obj, "initialized", False):
                return None
            pos, quat = obj.get_position_orientation()
            poses[m] = (pos.detach().float(), quat.detach().float())
        return poses

    def evaluate(self) -> GroupEvaluation | None:
        """Evaluate the group with one shared assignment, cached while the member poses are unchanged."""
        if self.target is None:
            return None
        poses = self.member_poses()
        if poses is None:
            return None
        key = tuple((m, tuple(p.tolist()), tuple(q.tolist())) for m, (p, q) in poses.items())
        if key != self._cache_key or self._cache is None:
            self._cache_key, self._cache = key, self._evaluate(poses)
        return self._cache

    def _evaluate(self, poses: dict[str, tuple]) -> GroupEvaluation:
        target = self.target
        eqs = target.orientation_equivalences or None
        ev = GroupEvaluation(assignment=self._assign(poses))

        if target.target_slots:
            pos_tol, ori_tol = target.pose_tolerances or (None, None)
            all_ok = True
            for m, slot_id in ev.assignment.items():
                slot = target.target_slots[slot_id]
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

        if target.layout_relations:
            pos_tol, ori_tol = target.layout_tolerances or (None, None)
            slot_to_member = {s: m for m, s in ev.assignment.items()}
            all_ok = True
            for rel in target.layout_relations:
                m_from, m_to = slot_to_member[rel.from_slot], slot_to_member[rel.to_slot]
                pos_err, ori_err = relation_error(poses[m_from], poses[m_to], rel.pose, eqs, eqs, pos_tol, ori_tol)
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

    def _assign(self, poses: dict[str, tuple]) -> dict[str, str]:
        target = self.target
        if target.assignment_mode == "fixed":
            return dict(target.member_to_slot)
        pinned = target.pinned
        slots = {s: sl for s, sl in target.target_slots.items() if s not in pinned.values()}
        if target.assignment_cost == "slot_position":
            free = assign_by_slot_position(
                {m: p for m, (p, _) in poses.items() if m not in pinned}, {s: sl.position for s, sl in slots.items()}
            )
            return dict(sorted({**free, **pinned}.items()))

        # Layout cost: the permutation whose relations best reproduce the stored relative poses (invariant to a rigid
        # motion of the whole arrangement); ties go to the smaller orientation error, then lexicographic order.
        members, slot_ids = sorted(m for m in poses if m not in pinned), sorted(slots)
        eqs = target.orientation_equivalences or None
        pos_tol, ori_tol = target.layout_tolerances or (None, None)

        def perm_cost(perm):
            slot_to_member = {slot_ids[j]: members[i] for i, j in enumerate(perm)}
            slot_to_member.update({s: m for m, s in pinned.items()})
            total_pos, total_ori = 0.0, 0.0
            for rel in target.layout_relations:
                pos_err, ori_err = relation_error(
                    poses[slot_to_member[rel.from_slot]],
                    poses[slot_to_member[rel.to_slot]],
                    rel.pose,
                    eqs,
                    eqs,
                    pos_tol,
                    ori_tol,
                )
                total_pos += pos_err
                total_ori += ori_err
            return (round(total_pos, 6), round(total_ori, 6))

        return dict(sorted({**assign_by_cost_fn(members, slot_ids, perm_cost), **pinned}.items()))
