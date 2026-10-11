"""The attachment snap keeps a symmetric part's spin (owner 2026-10-10): ``predicates.symmetric_snap`` seats a part on its
partner's attachment frame - the male origin on the female origin, its symmetry axis on the seat's - removing only the tilt
when its recorded symmetry is continuous about one axis (``zinf``: the 5° group), else carrying the nearest equivalent onto the
female frame (no equivalences: OmniGibson's raw frame snap); the joint frame comes from the transform (``snap_joint_frame``);
the attach rule's record stays JSON-safe and the paths that do not join move nothing."""
import json
import math
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.sampling import predicates as PRED


def _q(axis, deg):
    a = th.tensor(axis, dtype=th.float64)
    a = a / th.linalg.norm(a)
    h = math.radians(deg) / 2.0
    return th.cat([math.sin(h) * a, th.tensor([math.cos(h)], dtype=th.float64)])


ZINF = [_q([0, 0, 1], 5 * k).tolist() for k in range(1, 72)]  # 71 turns of 5° about object z (the identity implied)
STEP30 = [_q([0, 0, 1], 30 * k).tolist() for k in range(1, 12)]  # 11 turns of 30°: a discrete symmetry
# the male attachment frame in the object frame: off the object origin (lid/pbmsji-like) and turned
P_MO, Q_MO = th.tensor([0.0014, -0.0021, 0.012], dtype=th.float64), _q([1.0, 2.0, 3.0], 25.0)
# the female attachment frame in the world
P_F, Q_F = th.tensor([0.31, -0.22, 0.93], dtype=th.float64), _q([0.2, -0.5, 1.0], 40.0)
A_MALE = PRED._rotate(PRED._qinv(Q_MO), th.tensor([0.0, 0.0, 1.0], dtype=th.float64))  # the symmetry axis in the male frame


def _seated(phi):
    """the object pose seated exactly (male origin on the female origin, axes aligned) at spin ``phi`` about its axis."""
    q_m = PRED._qmul(Q_F, _q(A_MALE.tolist(), phi))
    q_o = PRED._qunit(PRED._qmul(q_m, PRED._qinv(Q_MO)))
    return P_F - PRED._rotate(q_o, P_MO), q_o


def _male(p_o, q_o):
    return p_o + PRED._rotate(q_o, P_MO), PRED._qunit(PRED._qmul(q_o, Q_MO))


def _perturbed(phi, tilt=10.0, offset=(0.012, -0.011, 0.0117)):
    """the seated pose tilted ``tilt`` deg about an axis perpendicular to the symmetry axis and moved by ``offset`` (2 cm)."""
    p_o, q_o = _seated(phi)
    a = PRED._rotate(q_o, th.tensor([0.0, 0.0, 1.0], dtype=th.float64))
    u = th.linalg.cross(a, th.tensor([1.0, 0.0, 0.0], dtype=th.float64))
    q_t = _q((u / th.linalg.norm(u)).tolist(), tilt)
    return p_o + th.tensor(offset, dtype=th.float64), PRED._qunit(PRED._qmul(q_t, q_o))


def _snap(p_o, q_o, eqs):
    p_m, q_m = _male(p_o, q_o)
    p_fix, q_fix, info = PRED.symmetric_snap(p_m, q_m, P_F, Q_F, q_o, eqs)
    return PRED.apply_fix(p_fix, q_fix, p_o, q_o), (p_fix, q_fix, p_m, q_m), info


@pytest.mark.parametrize("phi", [0.0, 37.0, 90.0, 180.0])
def test_a_continuous_symmetry_is_seated_at_its_own_spin(phi):
    assert th.linalg.norm(th.tensor([0.012, -0.011, 0.0117])) == pytest.approx(0.02, abs=2e-4)
    p_o, q_o = _perturbed(phi)
    (p_new, q_new), (p_fix, q_fix, p_m, q_m), info = _snap(p_o, q_o, ZINF)
    p_ref, q_ref = _seated(phi)
    assert float(th.linalg.norm(p_new - p_ref)) < 1e-6 and PRED._qangle(q_new, q_ref) < 1e-6
    assert info["mode"] == "continuous" and info["spin_rad"] == pytest.approx(0.0, abs=1e-6)
    assert info["tilt_rad"] == pytest.approx(math.radians(10.0), abs=1e-6) and info["moved_m"] > 0.015
    # the joint frame: the male link at the female origin, turned about the seat's axis by the kept spin
    rel_p, rel_q = PRED.snap_joint_frame(p_fix, q_fix, p_m, q_m, P_F, Q_F)
    assert float(th.linalg.norm(rel_p)) < 1e-9 and PRED._qangle(rel_q, _q(A_MALE.tolist(), phi)) < 1e-6


def test_the_zinf_list_is_continuous_thirty_degree_steps_are_not():
    z = th.tensor([0.0, 0.0, 1.0], dtype=th.float64)
    for eqs in (ZINF, [[0.0, 0.0, 0.0, 1.0]] + ZINF):
        axis = PRED.continuous_symmetry_axis(eqs)
        assert axis is not None and abs(float(th.dot(axis, z))) > 0.999
    assert PRED.continuous_symmetry_axis(STEP30) is None and PRED.continuous_symmetry_axis([]) is None
    assert PRED.continuous_symmetry_axis(ZINF[:-1] + [_q([1, 0, 0], 180).tolist()]) is None  # a second axis


def test_thirty_degree_steps_seat_at_the_nearest_equivalent():
    p_o, q_o = _perturbed(37.0)
    (p_new, q_new), _, info = _snap(p_o, q_o, STEP30)
    p_ref, q_ref = _seated(30.0)
    assert info["mode"] == "nearest_equivalence" and info["spin_rad"] == pytest.approx(math.radians(-7.0), abs=1e-6)
    assert float(th.linalg.norm(p_new - p_ref)) < 1e-6 and PRED._qangle(q_new, q_ref) < 1e-6


def test_no_equivalences_is_the_raw_frame_snap():
    p_o, q_o = _perturbed(37.0)
    (p_new, q_new), _, info = _snap(p_o, q_o, [])
    p_m, q_m = _male(p_new, q_new)
    assert info["mode"] == "raw" and float(th.linalg.norm(p_m - P_F)) < 1e-9 and PRED._qangle(q_m, Q_F) < 1e-9
    assert info["spin_rad"] == pytest.approx(math.radians(-37.0), abs=1e-6)


def test_a_half_turn_flip_falls_back_to_the_nearest_equivalent():
    p_o, q_o = _seated(0.0)
    a = PRED._rotate(q_o, th.tensor([0.0, 0.0, 1.0], dtype=th.float64))
    u = th.linalg.cross(a, th.tensor([1.0, 0.0, 0.0], dtype=th.float64))
    q_flip = PRED._qunit(PRED._qmul(_q((u / th.linalg.norm(u)).tolist(), 180.0), q_o))  # upside down: no twist about the axis
    (p_new, q_new), _, info = _snap(p_o, q_flip, ZINF)
    p_m, q_m = _male(p_new, q_new)
    assert info["mode"] == "nearest_equivalence" and float(th.linalg.norm(p_m - P_F)) < 1e-9
    assert PRED._qangle(q_m, Q_F) < math.radians(2.5) + 1e-9  # the nearest 5° step


class _Body:
    """a part (or partner) with a root pose and attachment links that ride on it."""

    def __init__(self, name, p, q, links):
        self.name, self.p, self.q, self.moves = name, p.clone(), q.clone(), []
        self.links = {k: SimpleNamespace(get_position_orientation=self._link(lp, lq)) for k, (lp, lq) in links.items()}

    def _link(self, lp, lq):
        return lambda: (self.p + PRED._rotate(self.q, lp), PRED._qunit(PRED._qmul(self.q, lq)))

    def get_position_orientation(self):
        return self.p.float(), self.q.float()

    def set_position_orientation(self, position, orientation):
        self.p, self.q = th.as_tensor(position, dtype=th.float64), PRED._qunit(th.as_tensor(orientation, dtype=th.float64))
        self.moves.append((self.p.clone(), self.q.clone()))

    def keep_still(self):
        pass


def _rule(monkeypatch, *, aligned=True, held=False):
    p_o, q_o = _perturbed(37.0, tilt=3.0, offset=(0.001, 0.0, 0.004))
    child = _Body("cap_0", p_o, q_o, {"M": (P_MO, Q_MO)})
    parent = _Body("bottle_0", P_F, Q_F, {"F": (th.zeros(3, dtype=th.float64), th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float64))})
    rec = {"ok": True, "pos_m": 0.0041, "rot_rad": 0.052, "rot_raw_rad": 0.7, "equivalence": 7, "child_link": "M", "parent_link": "F",
           "first_free_pair": ["M", "F"]}
    joints = []
    monkeypatch.setattr(PRED, "attachment_alignment", lambda c, p, eq=None, with_fix=False: dict(rec))
    monkeypatch.setattr(PRED, "seat_offset", lambda c, p, cl, pl: {"lateral_m": 0.001 if aligned else 0.006, "vertical_m": 0.004})
    monkeypatch.setattr(PRED, "held_by_robot", lambda o: held)
    monkeypatch.setattr(PRED, "pair_in_contact", lambda c, p: True)
    monkeypatch.setattr(PRED, "set_pair_collision_filter", lambda c, p, f: True)
    monkeypatch.setattr(PRED, "attach_in_place", lambda c, p, cl, pl, can_joint_break=True, rel_pose=None: joints.append(rel_pose) or True)
    monkeypatch.setattr(PRED, "JOINS", [])
    for name in ("_RELEASED_TOUCHING", "_PENDING_FILTER", "_PENDING_UNFILTER"):
        monkeypatch.setattr(PRED, name, set())
    monkeypatch.setenv("ISG_ATTACH_JOIN_MODE", "snap")
    return child, parent, joints


def test_the_attach_rule_snaps_symmetrically_with_a_json_safe_record(monkeypatch):
    child, parent, joints = _rule(monkeypatch)
    a = PRED.attach_if_aligned(child, parent, equivalences=ZINF)
    assert a["attached"] and a["joint"] == "snap" and "fix" not in a
    json.dumps(a)
    json.dumps(PRED.JOINS)
    assert a["snap"]["mode"] == "continuous" and a["snap"]["spin_rad"] == pytest.approx(0.0, abs=1e-6)
    # moved once (float32 through the object API): seated at its own 37° spin; the joint frame given, not re-read
    p_ref, q_ref = _seated(37.0)
    assert len(child.moves) == 1 and float(th.linalg.norm(child.p - p_ref)) < 1e-6 and PRED._qangle(child.q, q_ref) < 1e-5
    rel_p, rel_q = joints[0]
    assert float(th.linalg.norm(rel_p)) < 1e-9 and PRED._qangle(rel_q, _q(A_MALE.tolist(), 37.0)) < 1e-5


def test_the_paths_that_do_not_join_move_nothing(monkeypatch):
    for kw, why in (({"aligned": False}, "not aligned"), ({"held": True}, "held by a robot")):
        child, parent, joints = _rule(monkeypatch, **kw)
        a = PRED.attach_if_aligned(child, parent, equivalences=ZINF)
        assert not a["attached"] and a["reason"] == why and not child.moves and not joints and "fix" not in a
        json.dumps(a)
