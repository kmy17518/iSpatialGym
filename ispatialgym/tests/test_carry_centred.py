"""i7-goalmaps 2026-11-08 (lead decision): the opt-in centred carry poses (ISPATIALGYM_CARRY_CENTRED) - the switch, and the hand lateral that
puts the carried object's centre on the base's centre line."""
import math

import torch as th

from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.v2.tasks import iors_large as IL


def test_the_switch_is_off_by_default(monkeypatch):
    monkeypatch.delenv(IL.CARRY_CENTRED_ENV, raising=False)
    assert not IL.carry_centred_on()
    monkeypatch.setenv(IL.CARRY_CENTRED_ENV, "1")
    assert IL.carry_centred_on()
    monkeypatch.setenv(IL.CARRY_CENTRED_ENV, "0")
    assert not IL.carry_centred_on()


def _rod(x0, x1):
    """A rod along the object's x from x0 to x1 (hull points, object frame)."""
    return th.tensor([[x0, -0.01, -0.01], [x1, 0.01, 0.01], [x0, 0.01, -0.01], [x1, -0.01, 0.01]])


def test_a_rod_held_at_its_centre_needs_no_offset():
    eef = (th.tensor([1.0, 2.0, 0.3]), G.quat_from_yaw(0.0))
    grasp = (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0]))  # eef at the object origin, frames aligned
    left = th.tensor([0.0, 1.0])
    assert abs(IL.centred_hand_lateral(eef, grasp, _rod(-0.4, 0.4), left)) < 1e-6


def test_a_rod_across_the_heading_held_off_centre_is_shifted_back():
    # the base faces +x (left = +y); the rod lies along world y (object yaw 90 deg), held 0.2 m from its centre toward -y:
    # its centre sits 0.2 m to the hand's left, so the hand goes 0.2 m to the right of the centre line
    q = G.quat_from_yaw(math.pi / 2)
    eef = (th.tensor([0.5, 0.0, 0.3]), q)
    grasp = (th.tensor([-0.2, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))  # the eef 0.2 m along -x of the object (= -y in the world)
    lat = IL.centred_hand_lateral(eef, grasp, _rod(-0.4, 0.4), th.tensor([0.0, 1.0]))
    assert abs(lat - (-0.2)) < 1e-5, lat


def test_without_a_hull_the_root_is_centred():
    eef = (th.tensor([0.5, 0.1, 0.3]), G.quat_from_yaw(0.0))
    grasp = (th.tensor([0.0, 0.1, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))  # the root 0.1 m to the eef's right
    lat = IL.centred_hand_lateral(eef, grasp, None, th.tensor([0.0, 1.0]))
    assert abs(lat - 0.1) < 1e-5, lat


# ---------------------------------------------------------------------------------------------- the sideways carried drive (CARRY_SIDEWAYS)


def test_the_sideways_switch_is_off_by_default(monkeypatch):
    monkeypatch.delenv(IL.CARRY_SIDEWAYS_ENV, raising=False)
    assert not IL.carry_sideways_on()
    monkeypatch.setenv(IL.CARRY_SIDEWAYS_ENV, "1")
    assert IL.carry_sideways_on()


def test_fill_yaws_heads_along_the_segments_and_the_offset_turns_them():
    from ispatialgym.sampling.v2 import navigation as NAV

    nan = float("nan")
    wps = [(0.0, 0.0, 0.3), (0.0, 1.0, nan), (1.0, 1.0, nan), (1.0, 1.0, math.pi)]
    plain = NAV.fill_yaws(wps)
    assert plain[0][2] == 0.3 and abs(plain[1][2] - 0.0) < 1e-9 and plain[2][2] == plain[1][2] and plain[3][2] == math.pi
    side = NAV.fill_yaws(wps, math.pi / 2)
    assert abs(side[1][2] - math.pi / 2) < 1e-9 and side[0][2] == 0.3 and side[3][2] == math.pi  # the end poses keep their yaws
    side_r = NAV.fill_yaws([(0.0, 0.0, 0.0), (0.0, 0.0, nan), (0.0, 1.0, nan), (0.0, 2.0, 0.0)], -math.pi / 2)
    assert abs(side_r[2][2] - 0.0) < 1e-9  # heading +pi/2 (north) - pi/2


def test_the_sideways_offset_nearer_the_place_yaw_comes_first():
    # driving north (heading +pi/2) to a stance facing -x (yaw pi): +pi/2 gives yaw pi (no turn at the end), -pi/2 gives 0
    assert IL.sideways_offsets((0.0, 0.0), (0.0, 2.0), math.pi) == [math.pi / 2, -math.pi / 2]
    assert IL.sideways_offsets((0.0, 0.0), (0.0, 2.0), 0.0) == [-math.pi / 2, math.pi / 2]
