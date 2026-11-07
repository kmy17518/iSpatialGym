"""i7-plateqtf2 (owner decision 2026-11-08 02:35 item 1): the support policy's sphere stage stays a conservative screen; a NARROW failure of
the hand's rigid links (each sphere less than the model's 2 mm sphere buffer into the policed support, nothing else touching) is judged
by their exact hulls against the support plane with a small real clearance (ik.HULL_FALLBACK_MIN_CLEARANCE_M). Simulator-free."""

import contextlib
import inspect
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.grasping import contact_policy as CP
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.sampling.v2 import ik as IK

PALM, CAM, WRIST = "right_gripper_link", "right_realsense_link", "right_arm_link7"
LINKS = ["torso_link1", "right_arm_link5", WRIST, CAM, PALM, "right_gripper_finger_link1", "right_gripper_finger_link2"]
TABLE = SimpleNamespace(name="breakfast_table_0", aabb=([0, 0, 0], [1, 1, 0.8126]))


def _checker(sphere_cl, hull_cl, other_cl=None):
    """An IKChecker over a fake kit: ``sphere_cl`` = per link the clearance (m) of its buffered spheres to the support (negative: into it),
    ``other_cl`` = per link to any other obstacle, ``hull_cl`` = per hand role the exact hull's clearance to the support plane."""
    state = {"ignored": set(), "hull_calls": 0}

    def links_clear(q, links, margin):
        cl = []
        for l in links:
            c = float("inf") if TABLE.name in state["ignored"] else sphere_cl.get(l, 0.05)
            cl.append(min(c, (other_cl or {}).get(l, 0.05)))
        return th.tensor([min(cl) >= margin - 1e-12])

    @contextlib.contextmanager
    def ignoring(objs, **kw):
        prev = set(state["ignored"])
        state["ignored"] |= {getattr(o, "name", "") for o in objs}
        try:
            yield
        finally:
            state["ignored"] = prev

    def clearance(T, plane, fq):
        state["hull_calls"] += 1
        return {r: th.tensor([hull_cl.get(r, 0.05)]) for r in CP.H.ROLES}

    def check(T, plane, fq, floor=False):
        cl = clearance(T, plane, fq)
        ok = all(float(cl[r][0]) >= 0.0 for r in ("palm", "wrist_camera", "wrist"))
        return th.tensor([ok]), ["ok" if ok else "palm_contact"], cl

    backend = SimpleNamespace(ground_contact_links=set(), kin_entry=lambda n: None, _collision_costs=lambda e: {"link_of_sphere": LINKS},
                              fk=lambda ls, q: {M.EEF_LINK["right"]: (th.tensor([0.5, 0.0, 0.85]), th.tensor([0.0, 0.0, 0.0, 1.0]))})
    chk = IK.IKChecker.__new__(IK.IKChecker)
    chk.kit = SimpleNamespace(backend=backend, links_clear=links_clear, stats={}, robot=SimpleNamespace(gripper_control_idx={"right": [0, 1]}))
    chk.ignoring = ignoring
    chk._hand_checker = SimpleNamespace(arm="right", check=check, clearance=clearance)
    chk.support_plane_z = {TABLE.name: 0.8126}
    return chk, state


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv(IK.SUPPORT_HULL_FALLBACK_ENV, "1")
    monkeypatch.delenv("ISPATIALGYM_SUPPORT_HULL_FALLBACK_MIN_M", raising=False)


def test_buffer_is_the_models_global_sphere_buffer():
    """The 2 mm is the robot model's cuRobo collision_sphere_buffer (every sphere of every link, every query), not a policy margin."""
    assert CP.MODEL_SPHERE_BUFFER_M == inspect.signature(M.build_curobo_robot_dict).parameters["collision_sphere_buffer"].default == 0.002
    assert CP.DEFAULT.sphere_support_margin_m == 0.0
    assert set(CP.DEFAULT.hand_rigid_links("right")) == {PALM, CAM, WRIST}


def test_narrow_sphere_miss_with_a_clear_hull_passes(on):
    # side_9: the lowest palm sphere (with the buffer) 0.9 mm into the table, the palm hull 0.99 mm above it
    chk, st = _checker({PALM: -0.0009}, {"palm": 0.00099})
    ev = chk.support_policy_eval(th.zeros(28), "right", [TABLE], [])
    assert ev["ok"] and ev["stage"] == "hulls" and ev["hull_fallback"]["ok"] and ev["hull_fallback"]["hull_min_m"] == pytest.approx(0.00099)
    assert chk.kit.stats == {"support_policy.hull_fallback_passed": 1}


def test_a_real_hull_contact_fails(on):
    chk, _ = _checker({PALM: -0.0009}, {"palm": -0.0002})
    ev = chk.support_policy_eval(th.zeros(28), "right", [TABLE], [])
    assert not ev["ok"] and ev["reason"] == "non-finger link touches the support" and ev["stage"] == "spheres"
    assert ev["hull_fallback"]["why"] == "hull_clearance" and chk.kit.stats == {"support_policy.hull_fallback_failed.hull_clearance": 1}
    # a clearance below the threshold (0.5 mm) is not enough either
    chk, _ = _checker({PALM: -0.0009}, {"palm": 0.0003})
    assert not chk.support_policy_eval(th.zeros(28), "right", [TABLE], [])["ok"]


def test_a_clear_sphere_pass_is_unchanged(on):
    chk, st = _checker({PALM: 0.004}, {"palm": 0.005})
    ev = chk.support_policy_eval(th.zeros(28), "right", [TABLE], [])
    assert ev["ok"] and "hull_fallback" not in ev and chk.kit.stats == {} and st["hull_calls"] == 1  # the hull stage only, as before


@pytest.mark.parametrize("sphere,other,why", [
    ({PALM: -0.0025}, None, "deeper_than_buffer"),          # deeper than the buffer: the spheres decide
    ({PALM: -0.0009, "right_arm_link5": -0.0005}, None, "non_hand_link"),  # an arm link has no exact hull
    ({PALM: 0.01}, {CAM: -0.0005}, "other_obstacle"),       # the touch is another obstacle's, not the support's
])
def test_not_narrow_or_not_the_support_keeps_the_rejection(on, sphere, other, why):
    chk, st = _checker(sphere, {"palm": 0.002}, other)
    ev = chk.support_policy_eval(th.zeros(28), "right", [TABLE], [])
    assert not ev["ok"] and ev["hull_fallback"]["why"] == why and st["hull_calls"] == 0


def test_on_by_default(monkeypatch):
    monkeypatch.delenv(IK.SUPPORT_HULL_FALLBACK_ENV, raising=False)
    assert IK.support_hull_fallback_mode() == "on" and IK.support_hull_fallback_min_m() == 0.0005


def test_switch_off_and_shadow(monkeypatch):
    monkeypatch.setenv(IK.SUPPORT_HULL_FALLBACK_ENV, "0")
    chk, _ = _checker({PALM: -0.0009}, {"palm": 0.00099})
    ev = chk.support_policy_eval(th.zeros(28), "right", [TABLE], [])
    assert not ev["ok"] and "hull_fallback" not in ev and chk.kit.stats == {}
    monkeypatch.setenv(IK.SUPPORT_HULL_FALLBACK_ENV, "shadow")  # judged and counted, the verdict unchanged
    chk, _ = _checker({PALM: -0.0009}, {"palm": 0.00099})
    ev = chk.support_policy_eval(th.zeros(28), "right", [TABLE], [])
    assert not ev["ok"] and ev["hull_fallback"]["ok"] and chk.kit.stats == {"support_policy.hull_fallback_shadow_pass": 1}
    monkeypatch.setenv("ISPATIALGYM_SUPPORT_HULL_FALLBACK_MIN_M", "0.0012")
    assert IK.support_hull_fallback_min_m() == 0.0012


def test_a_floor_support_never_takes_the_fallback(on):
    floor = SimpleNamespace(name="floors_0", aabb=([0, 0, -0.1], [5, 5, 0.0]))
    chk, _ = _checker({PALM: -0.0009}, {"palm": 0.002})
    chk.support_plane_z = {"floors_0": 0.0}
    chk.support_floor_names = {"floors_0"}
    TABLE_NAME = TABLE.name
    try:
        TABLE.name = "floors_0"
        ev = chk.support_policy_eval(th.zeros(28), "right", [floor], [])
    finally:
        TABLE.name = TABLE_NAME
    assert not ev["ok"] and ev["hull_fallback"]["why"] == "floor"


def test_the_reusable_check_takes_any_failed_links_geometry_and_threshold():
    """contact_policy.hull_fallback (reused by i7-t03 for split-off support pieces and the exact-IK witness): the links that failed, the
    geometry (a box piece here), the threshold - independent of the support policy."""
    piece = CP.Support.box((0.5, 0.0, 0.80), (0.1, 0.1, 0.01), name="piece_3")
    sph = {PALM: -0.001}
    lc = lambda links, m: all(sph.get(l, 0.05) >= m - 1e-12 for l in links)  # noqa: E731
    seen = []

    def clearance(T, g, fq):
        seen.append(g.name)
        return {r: th.tensor([0.0008]) for r in CP.H.ROLES}

    chk = SimpleNamespace(clearance=clearance)
    hand = CP.DEFAULT.hand_rigid_links("right")
    out = CP.hull_fallback(lc, [PALM], hand, chk, th.eye(4), [piece], threshold=0.0005)
    assert out["ok"] and out["hull_min_m"] == pytest.approx(0.0008) and seen == ["piece_3"]
    assert not CP.hull_fallback(lc, [PALM], hand, chk, th.eye(4), [piece], threshold=0.001)["ok"]  # the caller's threshold
    assert CP.hull_fallback(lc, ["right_arm_link5"], hand, chk, th.eye(4), [piece])["why"] == "no_hand_link"
    assert CP.hull_fallback(lc, [PALM], hand, chk, th.eye(4), [])["why"] == "no_geometry"
    sph[PALM] = -0.003
    assert CP.narrow_sphere_failure(lc, [PALM], hand) == "deeper_than_buffer"
    assert IK.HULL_FALLBACK_MIN_CLEARANCE_M == CP.HULL_FALLBACK_MIN_CLEARANCE_M
