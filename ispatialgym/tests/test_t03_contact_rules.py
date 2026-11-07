"""i7-t03 (owner request 2026-11-08): item 2 - the pieces a support split keeps in the world are FINGER-ONLY contact (the finger hulls may
touch them, clearance >= 0, never penetrate; every other part keeps the obstacle rule); item 3 - the side_9 hull fallback (i7-plateqtf2)
extended to the split-off pieces and the exact-IK witness: a NARROW sphere failure of the hand's rigid links (within the model's 2 mm
buffer) is judged by their exact hulls. Both behind switches, default off (ik.SPLIT_FINGER_TOUCH_ENV / ik.WITNESS_HULL_FALLBACK_ENV).
Simulator-free: a fake kit whose per-link clearances to the split-off side board / the plane pieces / other obstacles are given."""

import contextlib
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.grasping import contact_policy as CP
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.sampling.v2 import ik as IK

ARM = "left"
F1, F2 = M.FINGER_LINKS[ARM]
PALM, CAM, WRIST = "left_gripper_link", "left_realsense_link", "left_arm_link7"
ARM5 = "left_arm_link5"
LINKS = ["torso_link1", ARM5, WRIST, CAM, PALM, F1, F2]
BOOKCASE = SimpleNamespace(name="bookcase_kmuacd_0")
SIDE = "side_board"
PLAN = SimpleNamespace(kept=1, clips=[], off=frozenset({"plane_board"}))


def _checker(monkeypatch, side=None, hull=None, other=None, split=True, registered=()):
    """``side`` = per link the clearance (m) of its padded spheres to the split-off side board (negative: into it), ``hull`` = per hand role
    the exact hull's clearance to it, ``other`` = per link to any other obstacle. ``split=False``: the bookcase has no split."""
    state = {"whole": set(), "ignored": set()}
    side, hull, other = dict(side or {}), dict(hull or {}), dict(other or {})

    def cl_of(link):
        c = (other or {}).get(link, 0.05)
        if BOOKCASE.name not in state["whole"]:
            c = min(c, side.get(link, 0.05))
        return c

    def links_clear(qs, links, margin):
        n = int(th.as_tensor(qs).reshape(-1, 28).shape[0])
        return th.tensor([min(cl_of(l) for l in links) >= margin - 1e-12] * n)

    def collision_free(qs, margin=0.0):
        return links_clear(qs, LINKS, margin)

    @contextlib.contextmanager
    def ignoring(objs, ground=True, ground_meshes=True, split=True):
        prev = (set(state["whole"]), set(state["ignored"]))
        for o in objs:
            state["ignored"].add(o.name)
            if not split:
                state["whole"].add(o.name)
        try:
            yield
        finally:
            state["whole"], state["ignored"] = prev

    monkeypatch.setattr(IK.SS, "split_on", lambda: True)
    monkeypatch.setattr(IK.SS, "plan_for", lambda ikc, o, count=True: PLAN if (split and o is BOOKCASE) else None)
    monkeypatch.setattr(IK.SS, "_pieces", lambda o: [("plane_board", None), (SIDE, None)])
    monkeypatch.setattr(IK.SS, "registered_names", lambda ikc: {o.name for o in registered})
    chk = IK.IKChecker.__new__(IK.IKChecker)
    backend = SimpleNamespace(kin_entry=lambda n: None, _collision_costs=lambda e: {"link_of_sphere": LINKS})
    chk.kit = SimpleNamespace(backend=backend, links_clear=links_clear, collision_free=collision_free, stats={},
                              robot=SimpleNamespace(gripper_control_idx={ARM: [0, 1]}), world=SimpleNamespace(spawned={}, scene=SimpleNamespace(objects=list(registered))))
    chk.ignoring = ignoring
    chk._pieces_geometry = lambda sp, only=None: ("pieces", sp.name, None if only is None else tuple(sorted(only)))
    chk._eef_matrix = lambda q, arm: th.eye(4)
    chk._touch_hand_checker = lambda arm: SimpleNamespace(clearance=lambda T, g, fq: {r: th.tensor([hull.get(r, 0.05)]) for r in CP.H.ROLES})
    return chk, state


@pytest.fixture
def item2(monkeypatch):
    monkeypatch.setenv(IK.SPLIT_FINGER_TOUCH_ENV, "1")
    monkeypatch.delenv(IK.WITNESS_HULL_FALLBACK_ENV, raising=False)


@pytest.fixture
def item3(monkeypatch):
    monkeypatch.setenv(IK.WITNESS_HULL_FALLBACK_ENV, "1")
    monkeypatch.delenv(IK.SPLIT_FINGER_TOUCH_ENV, raising=False)
    monkeypatch.delenv("ISPATIALGYM_SUPPORT_HULL_FALLBACK_MIN_M", raising=False)


Q = th.zeros(1, 28)


def test_defaults_off(monkeypatch):
    monkeypatch.delenv(IK.SPLIT_FINGER_TOUCH_ENV, raising=False)
    monkeypatch.delenv(IK.WITNESS_HULL_FALLBACK_ENV, raising=False)
    assert IK.split_finger_touch_mode() == IK.witness_hull_fallback_mode() == "off" and not IK.contact_rejudge_on()
    chk, _ = _checker(monkeypatch, side={F1: -0.001}, hull={"finger": 0.0})
    assert not bool(chk.contact_rejudge(Q, ARM, [BOOKCASE])[0]) and chk.kit.stats == {}


def test_finger_touching_a_split_off_piece_passes(monkeypatch, item2):
    # the padded finger spheres 1 mm into the side board, the finger hull exactly touching it
    chk, st = _checker(monkeypatch, side={F1: -0.001}, hull={"finger": 0.0})
    assert bool(chk.contact_rejudge(Q, ARM, [BOOKCASE], where="track")[0])
    assert chk.kit.stats["support_split.finger_touch_allowed"] == 1 and chk.kit.stats["contact_rejudge.track.passed"] == 1
    assert st["whole"] == set()  # every context restored


def test_finger_penetrating_a_split_off_piece_fails(monkeypatch, item2):
    chk, _ = _checker(monkeypatch, side={F1: -0.003}, hull={"finger": -0.0005})
    assert not bool(chk.contact_rejudge(Q, ARM, [BOOKCASE])[0])
    assert chk.kit.stats["support_split.finger_touch_failed.penetration"] == 1


@pytest.mark.parametrize("link,role", [(PALM, "palm"), (CAM, "wrist_camera")])
def test_palm_or_camera_at_the_same_clearance_fails(monkeypatch, item2, link, role):
    chk, _ = _checker(monkeypatch, side={link: -0.001}, hull={role: 0.0})
    assert not bool(chk.contact_rejudge(Q, ARM, [BOOKCASE])[0])
    assert chk.kit.stats.get("contact_rejudge.witness.rejected.hand_link") == 1 and "support_split.finger_touch_allowed" not in chk.kit.stats


def test_an_arm_link_keeps_the_obstacle_rule(monkeypatch, item2):
    chk, _ = _checker(monkeypatch, side={F1: -0.001, ARM5: -0.0005}, hull={"finger": 0.001})
    assert not bool(chk.contact_rejudge(Q, ARM, [BOOKCASE])[0])
    assert chk.kit.stats["contact_rejudge.witness.rejected.non_hand_link"] == 1


def test_a_support_without_a_split_is_unchanged(monkeypatch, item2):
    chk, _ = _checker(monkeypatch, side={F1: -0.001}, hull={"finger": 0.0}, split=False)
    assert not bool(chk.contact_rejudge(Q, ARM, [BOOKCASE])[0])
    assert chk.kit.stats["contact_rejudge.witness.rejected.no_obstacle"] == 1


def test_finger_touch_needs_the_rest_of_the_world_clear(monkeypatch, item2):
    chk, _ = _checker(monkeypatch, side={F1: -0.001}, hull={"finger": 0.0}, other={F2: -0.0005})
    assert not bool(chk.contact_rejudge(Q, ARM, [BOOKCASE])[0])


def test_item3_narrow_hand_link_with_a_clear_hull_passes(monkeypatch, item3):
    # side_9's numbers against the side board: the palm sphere 0.9 mm in with the buffer, the palm hull 1 mm clear
    chk, _ = _checker(monkeypatch, side={PALM: -0.0009}, hull={"palm": 0.001})
    assert bool(chk.contact_rejudge(Q, ARM, [BOOKCASE])[0]) and chk.kit.stats["witness.hull_fallback_passed"] == 1


@pytest.mark.parametrize("side,hull,why", [
    ({PALM: -0.0025}, {"palm": 0.001}, "deeper_than_buffer"),
    ({CAM: -0.0009}, {"wrist_camera": 0.0003}, "hull_clearance"),   # below the side_9 threshold (0.5 mm)
])
def test_item3_rejections(monkeypatch, item3, side, hull, why):
    chk, _ = _checker(monkeypatch, side=side, hull=hull)
    assert not bool(chk.contact_rejudge(Q, ARM, [BOOKCASE])[0]) and chk.kit.stats[f"witness.hull_fallback_failed.{why}"] == 1


def test_item3_alone_does_not_let_fingers_touch(monkeypatch, item3):
    chk, _ = _checker(monkeypatch, side={F1: -0.001}, hull={"finger": 0.0})
    assert not bool(chk.contact_rejudge(Q, ARM, [BOOKCASE])[0]) and chk.kit.stats["support_split.finger_touch_failed.off"] == 1


def test_item3_witness_against_a_registered_support_in_the_world(monkeypatch, item3):
    """(b): at a pre-grasp nothing is ignored - the registered bookcase stands whole in the world; the narrow palm is judged by its hull."""
    chk, _ = _checker(monkeypatch, side={PALM: -0.0009}, hull={"palm": 0.001}, split=False, registered=(BOOKCASE,))
    assert bool(chk.contact_rejudge(Q, ARM, None)[0]) and chk.kit.stats["witness.hull_fallback_passed"] == 1


def test_shadow_counts_without_changing_the_verdict(monkeypatch):
    monkeypatch.setenv(IK.SPLIT_FINGER_TOUCH_ENV, "shadow")
    monkeypatch.setenv(IK.WITNESS_HULL_FALLBACK_ENV, "shadow")
    chk, _ = _checker(monkeypatch, side={F1: -0.001, PALM: -0.0009}, hull={"finger": 0.0, "palm": 0.001})
    assert not bool(chk.contact_rejudge(Q, ARM, [BOOKCASE])[0])
    assert chk.kit.stats["support_split.finger_touch_shadow_pass"] == 1 and chk.kit.stats["witness.hull_fallback_shadow_pass"] == 1


def test_cache_scope_only_when_on(monkeypatch):
    monkeypatch.delenv(IK.SPLIT_FINGER_TOUCH_ENV, raising=False)
    monkeypatch.setenv(IK.WITNESS_HULL_FALLBACK_ENV, "shadow")
    kit = SimpleNamespace()
    assert "contact_rejudge" not in IK.IKChecker(kit).cache_scope
    monkeypatch.setenv(IK.SPLIT_FINGER_TOUCH_ENV, "1")
    assert IK.IKChecker(kit).cache_scope["contact_rejudge"][:2] == ["on", "shadow"]
