"""ISPATIALGYM_GROUP_ASSIGN_POSE_TIEBREAK (i7-blockdiag 2026-11-09): block_design_01's symmetric 2 x 2 target (bdv_grid 320101:
every die seated within 1.3 mm, the layout-cost permutation flipped by pi -> pose_reached 4 cm / pi off), and no change where
there is no symmetric tie (no simulator; members are stand-ins with fixed poses)."""

import math

import pytest
import torch as th

GE = pytest.importorskip("ispatialgym.eval.group_entity")
from ispatialgym.eval.targets import GroupTarget  # noqa: E402

from ispatialgym.tests.test_group_fast import _key, _members, _target  # noqa: E402

SW = "ISPATIALGYM_GROUP_ASSIGN_POSE_TIEBREAK"

# bdv_grid configuration_matching-block_design_01 320101 (trunk_f1fc4ac1): the TRO's group target and the final step's die poses
SLOTS = {"slot_0": {"position": [5.915757, 3.834063, 0.820419], "orientation": [-0.00153, 0.000354, 0.999998, 0.001275]},
         "slot_1": {"position": [5.915601, 3.802683, 0.845841], "orientation": [0.001276, 0.999998, -0.000359, 0.001542]},
         "slot_2": {"position": [5.884282, 3.834162, 0.845722], "orientation": [-0.999998, 0.001275, -0.001535, -0.000354]},
         "slot_3": {"position": [5.884278, 3.802744, 0.820344], "orientation": [-0.000353, -0.00153, -0.001276, 0.999998]}}
RELATIONS = [{"from": "slot_0", "to": "slot_1", "position": [-1e-06, 0.031399, 0.0254], "orientation": [1.0, -1e-06, -1.2e-05, -4e-06]},
             {"from": "slot_0", "to": "slot_2", "position": [0.031398, -0.0, 0.025399], "orientation": [0.0, 1.0, 0.0, -5e-06]},
             {"from": "slot_0", "to": "slot_3", "position": [0.0314, 0.0314, -1e-06], "orientation": [0.0, -1e-06, -1.0, -1e-06]},
             {"from": "slot_1", "to": "slot_2", "position": [0.0314, 0.031399, -0.0], "orientation": [-7e-06, -4e-06, -1.0, -0.0]},
             {"from": "slot_1", "to": "slot_3", "position": [0.031402, -1e-06, 0.0254], "orientation": [0.0, -1.0, 5e-06, 1.2e-05]},
             {"from": "slot_2", "to": "slot_3", "position": [-1e-06, 0.0314, 0.0254], "orientation": [1.0, 1e-06, 5e-06, -1e-06]}]
FINAL = {"toy_dice.n.01_1": [5.91542, 3.834615, 0.820418, -0.001549, 0.000316, 0.999991, 0.003823],
         "toy_dice.n.01_2": [5.914842, 3.802705, 0.845839, 0.008313, 0.999964, -0.00031, 0.00155],
         "toy_dice.n.01_3": [5.883898, 3.83317, 0.845722, -0.999988, -0.004672, -0.00154, -0.000354],
         "toy_dice.n.01_4": [5.884946, 3.802386, 0.820347, -0.000308, -0.00155, -0.009179, 0.999957]}


class _M:
    def __init__(self, v):
        self._p, self._q = th.tensor(v[:3]), th.tensor(v[3:])
        self.initialized = True

    def get_position_orientation(self):
        return self._p.clone(), self._q.clone()


def _bd_entity(final=FINAL):
    target = GroupTarget.from_dict("targets_group.n.01_1", {
        "members": sorted(final), "assignment": {"mode": "exchangeable", "cost": "layout"}, "target_slots": SLOTS,
        "layout_relations": RELATIONS, "orientation_equivalences": [],
        "pose_tolerances": {"position": 0.03, "orientation": 0.1}, "layout_tolerances": {"position": 0.01, "orientation": 0.05}})
    objs = {m: _M(v) for m, v in final.items()}
    ent = GE.GroupEntity("targets_group.n.01_1", sorted(final), objs.get)
    ent.set_target(target)
    return ent


@pytest.mark.parametrize("fast", ["", "1"])
def test_block_design_320101_flipped_assignment(monkeypatch, fast):
    monkeypatch.setattr(GE, "FAST_ASSIGN", fast)
    monkeypatch.setenv(SW, "0")
    ev = _bd_entity().evaluate()
    assert ev.layout_reached is True and ev.pose_reached is False  # the recorded symbolic_only verdict
    assert ev.assignment == {"toy_dice.n.01_1": "slot_2", "toy_dice.n.01_2": "slot_3", "toy_dice.n.01_3": "slot_0", "toy_dice.n.01_4": "slot_1"}
    assert ev.max_pose_position_error > 0.039 and ev.max_pose_orientation_error > 3.1
    monkeypatch.setenv(SW, "1")
    GE.pose_tiebreak_reset()
    ev = _bd_entity().evaluate()
    assert ev.layout_reached is True and ev.pose_reached is True
    assert ev.assignment == {f"toy_dice.n.01_{i + 1}": f"slot_{i}" for i in range(4)}
    assert ev.max_pose_position_error < 0.002 and ev.max_pose_orientation_error < 0.03
    assert ev.max_layout_orientation_error <= 0.05
    rep = GE.pose_tiebreak_report()
    assert rep["evaluations"] == 1 and rep["acted"] == 1
    assert rep["last_acted"]["chosen"]["toy_dice.n.01_1"] == "slot_0"


def test_layout_outside_keeps_the_layout_choice(monkeypatch):
    # die 1 turned 0.08 rad about the vertical: no permutation satisfies every relation -> the switch keeps the layout choice
    final = dict(FINAL)
    p, q = final["toy_dice.n.01_1"][:3], th.tensor(final["toy_dice.n.01_1"][3:])
    a = 0.08
    dz = th.tensor([0.0, 0.0, math.sin(a / 2), math.cos(a / 2)])
    from ispatialgym.eval.geometry import quat_multiply
    final["toy_dice.n.01_1"] = p + quat_multiply(dz, q).tolist()
    monkeypatch.setenv(SW, "0")
    off = _bd_entity(final).evaluate()
    monkeypatch.setenv(SW, "1")
    GE.pose_tiebreak_reset()
    on = _bd_entity(final).evaluate()
    assert off.layout_reached is False and _key(on) == _key(off)
    assert GE.pose_tiebreak_report()["acted"] == 0


@pytest.mark.parametrize("seed", range(12))
def test_no_symmetric_tie_no_change(monkeypatch, seed):
    # the line target of test_group_fast (no global motion maps it onto itself within tolerance): bit-identical on / off
    members = _members(seed)
    ent_off = GE.GroupEntity("blocks_group.n.01_1", sorted(members), members.get)
    ent_off.set_target(_target([]))
    monkeypatch.setenv(SW, "0")
    off = ent_off.evaluate()
    monkeypatch.setenv(SW, "1")
    ent_on = GE.GroupEntity("blocks_group.n.01_1", sorted(members), members.get)
    ent_on.set_target(_target([]))
    on = ent_on.evaluate()
    assert _key(on) == _key(off)


def test_report_absent_when_off(monkeypatch):
    monkeypatch.setenv(SW, "0")
    assert GE.pose_tiebreak_report() is None


def test_switch_default_on(monkeypatch):
    monkeypatch.delenv(SW, raising=False)
    assert GE.pose_tiebreak_on()  # default on since lead #100
    monkeypatch.setenv(SW, "0")
    assert not GE.pose_tiebreak_on()
