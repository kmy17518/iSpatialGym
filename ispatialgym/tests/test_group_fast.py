"""Group evaluation with the per-evaluation relation_error memo (ISPATIALGYM_FAST_ASSIGN) against without it, bit for
bit, and a pinned member's assignment (no simulator; members are stand-ins with fixed poses)."""

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


def _board_target(pinned: bool):
    """Four blocks in a row and a board behind them (``slot_board``), every pair related; the board pinned or exchangeable."""
    members = [f"toy_dice.n.01_{i}" for i in range(1, 5)] + ["tray.n.01_1"]
    slots = {f"slot_{i}": [0.1 * i, 0.0, 0.0] for i in range(4)}
    slots["slot_board"] = [0.15, 0.1, 0.0]
    ids = list(slots)
    relations = [{"from": a, "to": b, "position": [slots[b][k] - slots[a][k] for k in range(3)], "orientation": [0, 0, 0, 1]}
                 for i, a in enumerate(ids) for b in ids[i + 1:]]
    assignment = {"mode": "exchangeable", "cost": "layout", **({"pinned": {"tray.n.01_1": "slot_board"}} if pinned else {})}
    return GroupTarget.from_dict("targets_group.n.01_1", {
        "members": members, "assignment": assignment, "orientation_equivalences": [],
        "target_slots": {s: {"position": p, "orientation": [0, 0, 0, 1]} for s, p in slots.items()},
        "layout_relations": relations, "layout_tolerances": {"position": 0.01, "orientation": 0.05},
        "pose_tolerances": {"position": 0.03, "orientation": 0.1}})


def test_pinned_member_keeps_its_slot(monkeypatch):
    unit = th.tensor([0.0, 0.0, 0.0, 1.0])
    at = {"toy_dice.n.01_1": [0.2, 0.0, 0.0], "toy_dice.n.01_2": [0.0, 0.0, 0.0], "toy_dice.n.01_3": [0.3, 0.0, 0.0],
          "toy_dice.n.01_4": [0.1, 0.0, 0.0], "tray.n.01_1": [0.15, 0.1, 0.0]}
    # the board and the block of slot_1 trade places: only an exchangeable board takes the swapped arrangement for the layout
    swapped = {**at, "toy_dice.n.01_4": [0.15, 0.1, 0.0], "tray.n.01_1": [0.1, 0.0, 0.0]}
    results = {}
    for name, poses in (("placed", at), ("swapped", swapped)):
        objs = {m: _Member(th.tensor(p), unit) for m, p in poses.items()}
        for pinned in (True, False):
            for mode in ("", "1"):
                monkeypatch.setattr(GE, "FAST_ASSIGN", mode)
                ent = GE.GroupEntity("targets_group.n.01_1", list(objs), resolve_member=lambda n: objs[n])
                ent.set_target(_board_target(pinned))
                results[name, pinned, mode] = ent.evaluate()
    for key in results:
        assert _key(results[key]) == _key(results[key[:2] + ("",)])
    placed = results["placed", True, ""]
    assert placed.layout_reached and placed.pose_reached
    assert placed.assignment == {"toy_dice.n.01_1": "slot_2", "toy_dice.n.01_2": "slot_0", "toy_dice.n.01_3": "slot_3",
                                 "toy_dice.n.01_4": "slot_1", "tray.n.01_1": "slot_board"}
    assert list(placed.assignment) == sorted(placed.assignment)
    assert results["swapped", True, ""].assignment["tray.n.01_1"] == "slot_board" and not results["swapped", True, ""].layout_reached
    assert results["swapped", False, ""].assignment["tray.n.01_1"] == "slot_1" and results["swapped", False, ""].layout_reached


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
