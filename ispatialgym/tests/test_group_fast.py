"""Group evaluation with the per-evaluation relation_error memo (ISPATIALGYM_FAST_ASSIGN) against without it, bit for
bit (no simulator; members are stand-ins with fixed poses)."""

import math
import struct

import pytest
import torch as th

GE = pytest.importorskip("ispatialgym.eval.group_entity")
from ispatialgym.eval import geometry as G  # noqa: E402
from ispatialgym.eval.targets import GroupTarget  # noqa: E402

YAW4 = [[0.0, 0.0, math.sin(k * math.pi / 4), math.cos(k * math.pi / 4)] for k in range(1, 4)]


class _Member:
    def __init__(self, p, q):
        self._p, self._q = p, q
        self.initialized = True

    def get_position_orientation(self):
        return self._p.clone(), self._q.clone()


def _target(eqs):
    members = [f"toy_dice.n.01_{i}" for i in range(1, 5)]
    slots = {f"slot_{i}": {"position": [0.1 * i, 0.0, 0.0], "orientation": [0, 0, 0, 1]} for i in range(4)}
    relations = [{"from": f"slot_{i}", "to": f"slot_{j}", "position": [0.1 * (j - i), 0.0, 0.0], "orientation": [0, 0, 0, 1]}
                 for i in range(4) for j in range(i + 1, 4)]
    return GroupTarget.from_dict("blocks_group.n.01_1", {
        "members": members, "target_slots": slots, "layout_relations": relations, "orientation_equivalences": eqs,
        "layout_tolerances": {"position": 0.02, "orientation": 0.1}, "pose_tolerances": {"position": 0.02, "orientation": 0.1}})


def _members(seed):
    gen = th.Generator().manual_seed(seed)
    order = th.randperm(4, generator=gen).tolist()
    out = {}
    for i, k in enumerate(order):
        p = th.tensor([0.1 * k, 0.0, 0.0]) + th.randn(3, generator=gen) * 0.01
        yaw = (int(th.randint(0, 4, (1,), generator=gen)) * math.pi / 2) + float(th.randn(1, generator=gen)) * 0.05
        q = G.quat_normalize(th.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]) + th.randn(4, generator=gen) * 0.01)
        out[f"toy_dice.n.01_{i + 1}"] = _Member(p.float(), q.float())
    return out


def _key(ev):
    def f(x):
        if isinstance(x, float):
            return struct.pack("<d", x).hex()
        if isinstance(x, dict):
            return {k: f(v) for k, v in sorted(x.items())}
        if isinstance(x, (list, tuple)):
            return [f(v) for v in x]
        return x
    return f({"assignment": ev.assignment, "members": ev.member_errors, "relations": ev.relation_errors,
              "pose": ev.pose_reached, "layout": ev.layout_reached})


@pytest.mark.parametrize("eqs", [[], YAW4])
def test_assignment_memo_gives_the_same_evaluation(monkeypatch, eqs):
    for seed in range(6):
        objs = _members(seed)
        evs = {}
        for mode in ("", "1", "check"):
            monkeypatch.setattr(GE, "FAST_ASSIGN", mode)
            ent = GE.GroupEntity("blocks_group.n.01_1", list(objs), resolve_member=lambda n: objs[n])
            ent.set_target(_target(eqs))
            evs[mode] = _key(ent.evaluate())
        assert evs["1"] == evs[""] and evs["check"] == evs[""]
