"""i7-camhull (owner_20261108g item 1, an approved shared-gate change behind ISPATIALGYM_CAMERA_HULL_FALLBACK, default off): the wrist
camera's 5 mm rule (orient.camera_clear) against the POLICED support takes i7-plateqtf2's two-step - the padded spheres stay the screen,
a NARROW failure (the camera's spheres within the model's 2 mm buffer of the margin, nothing but the policed support close) is judged by
the camera's exact hull with at least 0.5 mm of real clearance (contact_policy.hull_fallback). Simulator-free."""

import contextlib
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.grasping import contact_policy as CP
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2 import orient as O
from ispatialgym.sampling.v2 import tabletop as TT

PALM, CAM, WRIST = "right_gripper_link", "right_realsense_link", "right_arm_link7"
LINKS = ["torso_link1", "right_arm_link5", WRIST, CAM, PALM, "right_gripper_finger_link1", "right_gripper_finger_link2"]
BAR = SimpleNamespace(name="bar_egwapq_0", aabb=([0, 0, 0], [1, 1, 0.8874]))
PLATE = SimpleNamespace(name="plate_luhkiz_0", aabb=([0.4, 0.4, 0.8874], [0.6, 0.6, 0.90]))
Q = th.zeros(28)


def _checker(sphere_cl, hull_cl, other_cl=None):
    """An IKChecker over a fake kit: ``sphere_cl`` = per link the clearance (m) of its buffered spheres to the bar, ``other_cl`` = per link to
    any other object, ``hull_cl`` = per hand role the exact hull's clearance to the bar's plane."""
    state = {"ignored": set(), "hull_calls": 0, "roles": None}

    def links_clear(q, links, margin):
        cl = []
        for l in links:
            c = float("inf") if BAR.name in state["ignored"] else sphere_cl.get(l, 0.05)
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
                              fk=lambda ls, q: {M.EEF_LINK["right"]: (th.tensor([0.5, 0.5, 0.92]), th.tensor([0.0, 0.0, 0.0, 1.0]))})
    chk = IK.IKChecker.__new__(IK.IKChecker)
    chk.kit = SimpleNamespace(backend=backend, links_clear=links_clear, stats={}, robot=SimpleNamespace(gripper_control_idx={"right": [0, 1]}))
    chk.ignoring = ignoring
    chk._hand_checker = SimpleNamespace(arm="right", check=check, clearance=clearance)
    chk.support_plane_z = {BAR.name: 0.8874}
    chk.approached_objects = lambda pose: [PLATE]
    return chk, state


def _picked(chk):
    """The exact IK of a grasp of the plate on the bar recorded its policed support (the bar; the plate is approached)."""
    chk._note_policed(Q, "right", [BAR, PLATE], [PLATE], None)


@pytest.fixture
def policy(monkeypatch):
    monkeypatch.setenv("ISPATIALGYM_SUPPORT_POLICY", "1")
    monkeypatch.delenv("ISPATIALGYM_SUPPORT_HULL_FALLBACK_MIN_M", raising=False)
    monkeypatch.delenv(O.CAMERA_BLAME_ENV, raising=False)


@pytest.fixture
def on(policy, monkeypatch):
    monkeypatch.setenv(IK.CAMERA_HULL_FALLBACK_ENV, "1")


def test_off_by_default_is_today(policy, monkeypatch):
    monkeypatch.delenv(IK.CAMERA_HULL_FALLBACK_ENV, raising=False)
    assert IK.camera_hull_fallback_mode() == "off" and IK.CAMERA_HULL_FALLBACK_DEFAULT == "0"
    chk, st = _checker({CAM: 0.0035}, {"wrist_camera": 0.003})
    _picked(chk)
    assert chk.__dict__.get("_policed_at", {}) == {}  # nothing recorded with the switch off
    assert O.camera_clear(chk, Q, "right") is False and chk.kit.stats == {} and st["hull_calls"] == 0
    for v in ("0", "off", "false"):
        monkeypatch.setenv(IK.CAMERA_HULL_FALLBACK_ENV, v)
        assert IK.camera_hull_fallback_mode() == "off"


def test_narrow_camera_failure_with_a_clear_hull_passes(on):
    # the sink: the camera's spheres 3.5 mm from the bar (< 5 mm, >= 5 - 2 mm), the camera hull 0.8 mm above its plane
    chk, st = _checker({CAM: 0.0035}, {"wrist_camera": 0.0008})
    _picked(chk)
    assert chk._policed_at == {("right", IK._q_key(Q)): [BAR]}
    assert O.camera_clear(chk, Q, "right") is True
    assert chk.kit.stats == {"camera_policy.hull_fallback_checked": 1, "camera_policy.hull_fallback_passed": 1}
    fb = chk.camera_hull_fallback(Q, "right", O.CAMERA_MARGIN)
    assert fb["ok"] and fb["hull_min_m"] == pytest.approx(0.0008) and fb["min_m"] == 0.0005


@pytest.mark.parametrize("sphere,hull,other,why", [
    ({CAM: 0.0035}, 0.0003, None, "hull_clearance"),         # under 0.5 mm of real clearance
    ({CAM: 0.0035}, -0.0002, None, "hull_clearance"),        # the hull in contact
    ({CAM: 0.0025}, 0.004, None, "deeper_than_buffer"),      # the spheres more than the buffer inside the margin
    ({CAM: 0.0100}, 0.004, {CAM: 0.004}, "other_obstacle"),  # the camera close to another object (the plate), not the support
])
def test_not_narrow_or_not_clear_keeps_the_refusal(on, sphere, hull, other, why):
    chk, st = _checker(sphere, {"wrist_camera": hull}, other)
    _picked(chk)
    assert O.camera_clear(chk, Q, "right") is False
    assert chk.kit.stats.get(f"camera_policy.hull_fallback_failed.{why}") == 1 and "camera_policy.hull_fallback_passed" not in chk.kit.stats
    assert st["hull_calls"] == (1 if why == "hull_clearance" else 0)


def test_only_the_camera_hull_is_judged(on):
    """The camera rule judges the camera: a palm hull into the plane (the support policy's business) does not decide it."""
    chk, _ = _checker({CAM: 0.0035}, {"wrist_camera": 0.0008, "palm": -0.001, "wrist": -0.001})
    _picked(chk)
    assert O.camera_clear(chk, Q, "right") is True


def test_without_a_policed_support_the_verdict_is_today(on):
    chk, st = _checker({CAM: 0.0035}, {"wrist_camera": 0.0008})
    assert O.camera_clear(chk, Q, "right") is False  # no exact-IK result recorded a policed support for this configuration
    assert chk.kit.stats == {"camera_policy.hull_fallback_checked": 1, "camera_policy.hull_fallback_failed.no_policed_support": 1}
    chk._note_policed(Q, "right", [BAR], [BAR], None)  # the bar approached (a pushed box): not a support of the approach
    assert chk.__dict__.get("_policed_at", {}) == {}
    chk.support_plane_z = {}  # an unregistered support is not policed
    chk._note_policed(Q, "right", [BAR, PLATE], [PLATE], None)
    assert chk.__dict__.get("_policed_at", {}) == {}


def test_a_floor_support_never_takes_it(on):
    chk, _ = _checker({CAM: 0.0035}, {"wrist_camera": 0.0008})
    chk.support_floor_names = {BAR.name}
    _picked(chk)
    assert O.camera_clear(chk, Q, "right") is False and chk.kit.stats.get("camera_policy.hull_fallback_failed.floor") == 1


def test_shadow_judges_and_counts_only(policy, monkeypatch):
    monkeypatch.setenv(IK.CAMERA_HULL_FALLBACK_ENV, "shadow")
    chk, _ = _checker({CAM: 0.0035}, {"wrist_camera": 0.0008})
    _picked(chk)
    assert O.camera_clear(chk, Q, "right") is False
    assert chk.kit.stats == {"camera_policy.hull_fallback_checked": 1, "camera_policy.hull_fallback_shadow_pass": 1}


def test_a_passing_camera_and_the_other_rules_are_unchanged(on, monkeypatch):
    chk, st = _checker({CAM: 0.006}, {"wrist_camera": 0.0001})
    _picked(chk)
    assert O.camera_clear(chk, Q, "right") is True and chk.kit.stats == {} and st["hull_calls"] == 0  # the spheres pass: nothing judged
    # the support policy's own sphere stage (ISPATIALGYM_SUPPORT_HULL_FALLBACK) keeps its counters, links and default margin
    monkeypatch.setenv(IK.SUPPORT_HULL_FALLBACK_ENV, "0")
    chk, _ = _checker({PALM: -0.0009}, {"palm": 0.00099})
    ev = chk.support_policy_eval(Q, "right", [BAR], [])
    assert not ev["ok"] and "hull_fallback" not in ev and chk.kit.stats == {}
    monkeypatch.setenv(IK.SUPPORT_HULL_FALLBACK_ENV, "1")
    chk, _ = _checker({PALM: -0.0009}, {"palm": 0.00099})
    ev = chk.support_policy_eval(Q, "right", [BAR], [])
    assert ev["ok"] and chk.kit.stats == {"support_policy.hull_fallback_passed": 1}


def test_the_support_top_screen_is_counted_never_changed(policy, monkeypatch):
    eef = (th.tensor([0.5, 0.5, 0.8874 + 0.10]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    cam = TT.wrist_camera_clearance(eef, 0.8874)
    plane = 0.8874 + 0.10 - 0.076 - 0.018 - 0.004  # the camera's spheres 4 mm above the top: refused by the 5 mm screen, narrow
    assert TT.wrist_camera_clearance(eef, plane) == pytest.approx(0.004, abs=1e-6) and cam > TT.TWIN_CAMERA_CLEARANCE_M
    for mode in ("0", "shadow", "1"):
        monkeypatch.setenv(IK.CAMERA_HULL_FALLBACK_ENV, mode)
        chk, _ = _checker({}, {"wrist_camera": 0.0008})
        s = SimpleNamespace(ikc=chk, kit=chk.kit)
        assert TT.camera_top_clear(s, eef, 0.8874, "right") is True
        assert TT.camera_top_clear(s, eef, plane, "right") is False  # the verdict in every mode
        want = {} if mode == "0" else {"camera_policy.top_screen.checked": 1, "camera_policy.top_screen.would_pass": 1}
        assert chk.kit.stats == want


def test_the_cache_identity_only_when_on(policy, monkeypatch):
    kit = SimpleNamespace(stats={})
    monkeypatch.setenv(IK.CAMERA_HULL_FALLBACK_ENV, "shadow")
    assert "camera_hull_fallback" not in IK.IKChecker(kit).cache_scope
    monkeypatch.setenv(IK.CAMERA_HULL_FALLBACK_ENV, "1")
    chk = IK.IKChecker(kit)
    assert chk.cache_scope["camera_hull_fallback"] == [0.0005, IK.CAMERA_HULL_FALLBACK_VERSION]
    assert "camera_hull_fallback" not in chk.exact_cache_scope()  # an exact-IK witness does not depend on the camera rule
