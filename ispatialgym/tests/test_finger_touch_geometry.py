"""i7-fingertouch 2026-11-07 (owner decision): ``approach_finger_support_touch`` under ``ISPATIALGYM_FINGER_TOUCH_RULE=geometry`` - the
spec's finger-GEOMETRY rule (P1 §2: the finger collision HULLS at most the contact policy's 2 mm into the support) instead of the customs'
5 mm on the padded finger spheres ("spheres"); "geometry" is the default since the A/B of 2026-11-07.

The toy kit is test_approach_finger_touch's (plate/luhkiz grasp_5 coming down 27.8 deg onto a counter); a fake ``HandChecker`` puts the
lower finger's hull ``hull_below`` m below its spheres (the real spheres sit inside the finger bodies) and measures it against the support
the rule hands it (the registered plane, or the support's convex pieces). One test runs the real ``HandChecker`` hulls against a
vertical board and a slab.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch as th

from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2.ik import ConvexPieces, FingerTouch, IKChecker
from ispatialgym.tests.test_approach_finger_touch import PRESHAPE, _approach, _finger_bottom, _Obstacle, _pose, _q0, _TouchKit

COUNTER = "bar_udatjt_0"
#: at the pre-shape the lower finger's SPHERES clear the counter by 16.66 mm 4 cm before the grasp (the deepest configuration that collides
#: fully open, test_approach_finger_touch); a hull this far below its spheres touches it by ~1.4 mm (passes 2 mm), ~2.9 mm (fails)
SHALLOW, DEEP = 0.0180, 0.0195


class _FakeHands:
    """``HandChecker`` stand-in: the lower finger's hull bottom = the eef z + the sphere bottom at the opening - ``hull_below``; its
    ``clearance`` = the support's ``sdf`` there (role "finger" only)."""

    arm = "right"

    def __init__(self, hull_below: float = 0.0, fixed: float = None):
        self.hull_below, self.fixed, self.calls = float(hull_below), fixed, []

    def clearance(self, T, support, finger_q=(0.05, 0.05)):
        T = th.as_tensor(T).reshape(-1, 4, 4)[0]
        self.calls.append((support, tuple(finger_q)))
        if self.fixed is not None:
            return {"finger": th.tensor([float(self.fixed)])}
        p = T[:3, 3].clone()
        p[2] += _finger_bottom(finger_q[0]) - self.hull_below
        return {"finger": support.sdf(p[None]).reshape(1)}


def _kit(*extra, palm_bottom=-0.0249):
    kit = _TouchKit([_Obstacle(COUNTER, 0.0), *extra], palm_bottom=palm_bottom)
    kit.backend.fk = lambda links, q: kit.fk(links, q)
    return kit


def _checker(kit, rule, hands=None, preshape=True, cap=0.005, plane=True):
    ikc = IKChecker(kit)
    ikc.set_finger_touch(FingerTouch(0.0, cap, preshape, supports=lambda: [kit.obstacles[0]], opening=lambda: PRESHAPE, rule=rule))
    if hands is not None:
        ikc._hand_checker = hands
    if plane:
        ikc.support_plane_z = {COUNTER: 0.0}
    return ikc


def _q_at(s: float, fingers: float = 0.05) -> th.Tensor:
    q = _q0(fingers)
    q[_TouchKit.RIGHT[:3]] = _pose(s)[0]
    return q


def _box(lo, hi) -> np.ndarray:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


# ------------------------------------------------------------------------------------------------ the switch


def test_switch_default_values_and_typos(monkeypatch):
    monkeypatch.delenv(IK.FINGER_TOUCH_RULE_ENV, raising=False)
    assert IK.finger_touch_rule() == "geometry" and FingerTouch().rule == "geometry"
    for v, want in (("", "geometry"), ("spheres", "spheres"), ("geometry", "geometry"), (" Geometry ", "geometry")):
        monkeypatch.setenv(IK.FINGER_TOUCH_RULE_ENV, v)
        assert IK.finger_touch_rule() == want
        # the iors_small call site (sample_template, touching_03) builds it positionally: the rule comes from the switch
        assert FingerTouch(0.0, 0.005, True, supports=lambda: [], opening=lambda: None).rule == want
    monkeypatch.setenv(IK.FINGER_TOUCH_RULE_ENV, "hulls")
    with pytest.raises(ValueError):
        IK.finger_touch_rule()
    with pytest.raises(ValueError):
        FingerTouch()
    with pytest.raises(ValueError):
        FingerTouch(rule="geometric")


def test_cache_scope_separates_the_rules():
    ikc = IKChecker(_kit())
    ikc.set_finger_touch(FingerTouch(0.0, 0.005, True, rule="spheres"))
    assert ikc.cache_scope["finger_touch"] == [0.0, 0.005, True], "the sphere rule's entries stay valid"
    ikc.set_finger_touch(FingerTouch(0.0, 0.005, True, rule="geometry"))
    assert ikc.cache_scope["finger_touch"] == ["geometry", 0.002, 0.0, True, IK.FINGER_TOUCH_GEOMETRY_VERSION]


def test_iors_large_setup_reads_the_switch(monkeypatch):
    from types import SimpleNamespace

    from ispatialgym.sampling.v2.tasks.iors_large import IORSLarge

    gen = {"approach_finger_support_touch": {"margin_m": 0.0, "max_penetration_m": 0.005, "preshape": True}}
    s = SimpleNamespace(gp=SimpleNamespace(variant="a", gen=lambda k, d=None: gen.get(k, d)), ikc=IKChecker(_kit()),
                        grasps=[SimpleNamespace(meta={"width": 0.00783})], surface={"kind": "furniture"})
    s.grasp_ignore_objects = lambda: ["bar"]
    s.preshape_opening = lambda: IORSLarge.preshape_opening(s)
    monkeypatch.setenv(IK.FINGER_TOUCH_RULE_ENV, "geometry")
    assert IORSLarge.setup_finger_touch(s) is None and s.ikc.finger_touch.rule == "geometry"
    assert s.ikc.cache_scope["finger_touch"][0] == "geometry"
    monkeypatch.setenv(IK.FINGER_TOUCH_RULE_ENV, "spheres")
    assert IORSLarge.setup_finger_touch(s) is None and s.ikc.finger_touch.rule == "spheres"
    monkeypatch.delenv(IK.FINGER_TOUCH_RULE_ENV)
    assert IORSLarge.setup_finger_touch(s) is None and s.ikc.finger_touch.rule == "geometry", "the default since the A/B"
    s.ikc.set_finger_touch(None)
    monkeypatch.setenv(IK.FINGER_TOUCH_RULE_ENV, "bogus")
    err = IORSLarge.setup_finger_touch(s)
    assert err and IK.FINGER_TOUCH_RULE_ENV in err and s.ikc.finger_touch is None, "a bad switch value: the template's spec rejection"


# ------------------------------------------------------------------------------------------------ decisions


def test_two_millimetre_threshold_on_the_hulls():
    """One configuration that collides with the counter (fully open, 4 cm before the grasp), every other check passing: the finger
    clearance the HandChecker reports decides - -1.99 mm passes, -2.01 mm fails; the passing one's penetration is returned (mm)."""
    kit = _kit()
    q = _q_at(0.04)[None]
    for cl, want in ((-0.00199, True), (-0.002, True), (-0.00201, False), (0.003, True), (-0.0049, False)):
        ikc = _checker(kit, "geometry", _FakeHands(fixed=cl))
        ok, pen = ikc._finger_touch_ok(q, "right", [], [kit.obstacles[0]])
        assert bool(ok[0]) is want, cl
        if want:
            assert pen == pytest.approx(max(0.0, -1000 * cl), abs=0.01)
    assert kit.stats.get("finger_touch.geometry.rejected_finger_penetration") == 2
    assert kit.stats.get("finger_touch.geometry.passed") == 3


def test_spheres_vs_geometry_decisions():
    """The same approach under each rule: (1) pre-shaped, the hull 18.0 / 19.5 mm below the spheres - the spheres clear (the sphere
    rule passes both), the hulls 1.36 / 2.86 mm in (geometry passes the first, refuses the second); (2) fully open with a 15 mm cap - the
    spheres 12.8 mm in pass the cap, the hulls refuse (the cap is not read under geometry); (3) fully open, 5 mm cap, a hull ABOVE its
    spheres (fake) - the sphere rule refuses, geometry judges the hull only."""
    kit = _kit()
    assert _approach(_checker(kit, "spheres", _FakeHands(DEEP)), kit).ok
    assert _approach(_checker(kit, "geometry", _FakeHands(SHALLOW)), kit).ok
    res = _approach(_checker(kit, "geometry", _FakeHands(DEEP)), kit)
    assert not res.ok and "an intermediate configuration of the approach collides" in res.reason
    assert _approach(_checker(kit, "spheres", _FakeHands(0.0), preshape=False, cap=0.015), kit).ok
    assert not _approach(_checker(kit, "geometry", _FakeHands(0.0), preshape=False, cap=0.015), kit).ok
    assert not _approach(_checker(kit, "spheres", _FakeHands(-0.012), preshape=False, cap=0.005), kit).ok
    assert _approach(_checker(kit, "geometry", _FakeHands(-0.012), preshape=False, cap=0.005), kit).ok


def test_geometry_measures_at_the_preshape_against_the_registered_plane():
    kit = _kit()
    hands = _FakeHands(SHALLOW)
    ikc = _checker(kit, "geometry", hands)
    assert _approach(ikc, kit).ok
    assert hands.calls and all(fq == (pytest.approx(PRESHAPE), pytest.approx(PRESHAPE)) for _, fq in hands.calls)
    sup = hands.calls[0][0]
    assert getattr(sup, "kind", None) == "plane" and sup.z == 0.0 and sup.name == COUNTER


def test_non_finger_links_still_block():
    """The palm's spheres 2 mm into the counter 4 cm before the grasp: refused under geometry whatever the hulls (the policy's sphere
    stage), the fingers are not measured for that configuration."""
    kit = _kit(palm_bottom=-0.0249 - 0.0208)
    res = _approach(_checker(kit, "geometry", _FakeHands(fixed=0.0)), kit)
    assert not res.ok and "an intermediate configuration of the approach collides" in res.reason
    assert kit.stats.get("finger_touch.geometry.rejected_non_finger_link", 0) >= 1
    ok, _ = _checker(kit, "geometry", _FakeHands(fixed=0.0))._finger_touch_ok(_q_at(0.04)[None], "right", [], [kit.obstacles[0]])
    assert not bool(ok[0])


def test_fingers_against_other_objects_still_block():
    kit = _kit(_Obstacle("dish_rack", 0.003, x_min=0.44))
    res = _approach(_checker(kit, "geometry", _FakeHands(fixed=0.0)), kit)
    assert not res.ok and "collides" in res.reason
    assert kit.stats.get("finger_touch.geometry.rejected_body", 0) >= 1


def test_program_check_chain_gets_the_geometry_rule():
    """The program check's 3-segment grasp approach: the same verdicts as the draw's one segment."""
    kit = _kit()
    chain = [_pose(0.10), _pose(0.10 * 2 / 3), _pose(0.10 / 3), _pose(0.0)]
    for hull, want in ((SHALLOW, True), (DEEP, False)):
        ikc = _checker(kit, "geometry", _FakeHands(hull))
        res = ikc.waypoints_ok(chain, None, "right", ignore_for_last=[kit.obstacles[0]], q_seed=_q0(), cache=False, approached=[])
        assert res.ok is want, res.reason


def test_log_lines_carry_the_measured_hull_penetration(caplog):
    kit = _kit()
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        assert _approach(_checker(kit, "geometry", _FakeHands(SHALLOW)), kit).ok
        assert not _approach(_checker(kit, "geometry", _FakeHands(DEEP)), kit).ok
    msgs = [r.getMessage() for r in caplog.records]
    ok = [m for m in msgs if m.startswith("approach finger-support touch allowed")]
    assert ok and "max penetration 1.36 mm (rule geometry: finger hulls, cap 2.0 mm" in ok[0], ok
    assert f"{COUNTER} (plane z 0.0000, registered)" in ok[0] and "0.0139 m (pre-shaped)" in ok[0]
    bad = [m for m in msgs if m.startswith("approach finger-support touch refused")]
    assert bad and "max penetration 2.86 mm" in bad[0] and "1 of 3 configurations" in bad[0]
    assert kit.stats.get("finger_touch_allowed") == 1


# ------------------------------------------------------------------------------------------------ the touched geometry


def test_convex_pieces_measure_a_vertical_board_through_its_face():
    board = ConvexPieces.from_vertices([_box([0.50, -0.5, 0.0], [0.52, 0.5, 0.4])], name="bookcase")
    pts = th.tensor([[0.501, 0.0, 0.2], [0.503, 0.0, 0.2], [0.51, 0.0, 0.2], [0.495, 0.0, 0.2], [0.3, 0.0, 0.2], [0.51, 0.0, 0.3995]])
    assert board.sdf(pts).tolist() == pytest.approx([-0.001, -0.003, -0.010, 0.005, ConvexPieces.CULL_M, -0.0005], abs=1e-6)
    shelf = _box([0.0, -0.5, 0.18], [0.5, 0.5, 0.2])
    flat = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    both = ConvexPieces.from_vertices([_box([0.50, -0.5, 0.0], [0.52, 0.5, 0.4]), shelf, flat], name="bookcase")
    assert len(both.halfspaces) == 2 and both.n_skipped == 1
    assert both.sdf(th.tensor([[0.25, 0.0, 0.199], [0.505, 0.0, 0.1]])).tolist() == pytest.approx([-0.001, -0.005], abs=1e-6), "the union: min over pieces"
    assert both.sdf(th.zeros(2, 3, 3)).shape == (2, 3)


def test_touch_geometry_plane_pieces_or_unmeasured():
    """Registered plane -> the plane; none -> the support's convex pieces (a vertical board refuses a hull 3 mm through its face); no
    readable pieces -> every touch refused (counted)."""
    kit = _kit()
    ikc = _checker(kit, "geometry", plane=False)
    ikc._support_pieces_world = lambda sp, only=None: [_box([0.50, -0.5, 0.0], [0.52, 0.5, 0.4])]
    g, why = ikc._touch_geometry(kit.obstacles[0])
    assert isinstance(g, ConvexPieces) and why == "1 convex pieces"
    ikc.support_plane_z = {COUNTER: 0.84}
    g, why = ikc._touch_geometry(kit.obstacles[0])
    assert g.kind == "plane" and g.z == pytest.approx(0.84) and why == "plane z 0.8400, registered"

    # the board's face at x = 0.5: a fake finger point 1.5 mm / 3 mm through it
    class _Pt(_FakeHands):
        def __init__(self, x):
            super().__init__()
            self.x = x

        def clearance(self, T, support, finger_q=(0.05, 0.05)):
            return {"finger": support.sdf(th.tensor([[self.x, 0.0, 0.2]])).reshape(1)}

    q = _q_at(0.04)[None]
    for x, want in ((0.5015, True), (0.503, False)):
        ikc = _checker(kit, "geometry", _Pt(x), plane=False)
        ikc._support_pieces_world = lambda sp, only=None: [_box([0.50, -0.5, 0.0], [0.52, 0.5, 0.4])]
        ok, pen = ikc._finger_touch_ok(q, "right", [], [kit.obstacles[0]])
        assert bool(ok[0]) is want
        if want:
            assert pen == pytest.approx(1.5, abs=0.01)
    ikc = _checker(kit, "geometry", _FakeHands(fixed=0.0), plane=False)  # the toy backend has no mesh geometry
    g, why = ikc._touch_geometry(kit.obstacles[0])
    assert g is None and "unmeasured" in why
    ok, _ = ikc._finger_touch_ok(q, "right", [], [kit.obstacles[0]])
    assert not bool(ok[0]) and kit.stats.get("finger_touch.geometry.rejected_unmeasured") == 1


def test_support_pieces_read_from_the_collision_meshes():
    """The pieces come from the support's collision meshes (``support_split.piece_geometry``: the backend's scaled mesh vertices at each
    piece prim's world pose); ``only`` (the split's ``Plan.off``) keeps those prims; one unreadable piece = the support is unmeasured."""
    from types import SimpleNamespace

    class _Cm:
        def __init__(self, path, verts, pos, bad=False):
            self.prim_path, self.verts, self.pos, self.bad = path, verts, pos, bad

        def get_position_orientation(self):
            return th.tensor(self.pos, dtype=th.float32), th.tensor([0.0, 0.0, 0.7071068, 0.7071068])  # 90 deg about z

    class _Backend:
        def _mesh_geometry(self, cm):
            if cm.bad:
                raise RuntimeError("no points")
            return cm.verts.astype(np.float32), np.zeros((0, 3), dtype=np.int64), np.array([2.0, 2.0, 2.0], dtype=np.float32)

    unit = _box([-0.005, -0.1, 0.0], [0.005, 0.1, 0.1])  # scaled x2 and turned 90 deg: y in [-0.01, 0.01], x in [-0.2, 0.2], z in [0, 0.2]
    cms = {"a": _Cm("/World/bookcase/base/a", unit, [1.0, 0.0, 0.5]), "b": _Cm("/World/bookcase/base/b", unit, [1.0, 0.5, 0.5])}
    sp = SimpleNamespace(name="bookcase", links={"base": SimpleNamespace(collision_meshes=cms)})
    kit = _kit()
    kit.backend._mesh_geometry = _Backend()._mesh_geometry
    ikc = _checker(kit, "geometry", plane=False)
    pieces = ikc._support_pieces_world(sp)
    assert len(pieces) == 2
    lo, hi = pieces[0].min(0), pieces[0].max(0)
    assert lo.tolist() == pytest.approx([0.8, -0.01, 0.5], abs=1e-6) and hi.tolist() == pytest.approx([1.2, 0.01, 0.7], abs=1e-6)
    g, why = ikc._touch_geometry(sp)
    assert why == "2 convex pieces" and g.sdf(th.tensor([[1.0, 0.009, 0.6], [1.0, 0.25, 0.6]])).tolist() == pytest.approx([-0.001, ConvexPieces.CULL_M], abs=1e-6)
    assert len(ikc._support_pieces_world(sp, only=frozenset({"/World/bookcase/base/b"}))) == 1
    cms["c"] = _Cm("/World/bookcase/base/c", unit, [0.0, 0.0, 0.0], bad=True)
    assert ikc._support_pieces_world(sp) == []


def test_real_hand_hulls_against_a_board_and_a_slab():
    """The contact policy's HandChecker hulls (USD export): a top-down pinch (closing along y) whose finger presses sideways into a vertical
    board is measured through the board's face (1.5 mm passes, 3 mm fails); against a slab the pieces measure equals the plane's (the
    policy's own measure) and ``finger_hull_clearance`` is ``HandChecker.clearance(...)["finger"]``."""
    from ispatialgym.grasping import contact_policy as CP

    hc = CP.HandChecker("right", CP.DEFAULT)
    T = th.eye(4)
    T[:3, :3] = th.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])  # eef +z down, +y along -y world
    T[:3, 3] = th.tensor([0.0, 0.0, 0.30])
    fq = (0.0139, 0.0139)
    W = th.cat(list(IK.finger_points_world(hc, T, fq).values()))
    y_max = float(W[:, 1].max())
    allow = CP.DEFAULT.finger_support_max_penetration_m
    for depth, want in ((0.0015, True), (0.003, False)):
        board = ConvexPieces.from_vertices([_box([-0.5, y_max - depth, 0.0], [0.5, y_max - depth + 0.02, 0.6])], name="board")
        cl = IK.finger_hull_clearance(hc, T, board, fq)
        assert cl == pytest.approx(-depth, abs=2e-4)
        assert (cl >= -allow - 1e-9) is want
    z_min = float(W[:, 2].min())
    plane = CP.Support.plane(z_min + 0.0015)
    slab = ConvexPieces.from_vertices([_box([-1.0, -1.0, z_min + 0.0015 - 0.05], [1.0, 1.0, z_min + 0.0015])], name="slab")
    c_plane = IK.finger_hull_clearance(hc, T, plane, fq)
    assert c_plane == pytest.approx(float(hc.clearance(T, plane, fq)["finger"][0])) and c_plane == pytest.approx(-0.0015, abs=1e-5)
    assert IK.finger_hull_clearance(hc, T, slab, fq) == pytest.approx(c_plane, abs=1e-5)


# ------------------------------------------------------------------------------------------------ the A/B's shadow verdict counter


def test_shadow_counts_flipped_verdicts_without_changing_them(monkeypatch, caplog):
    """ISPATIALGYM_FINGER_TOUCH_SHADOW=1: the other rule judges the same configurations; the active rule still decides. The DEEP hull
    (2.86 mm in, the spheres clear): the sphere rule passes, geometry refuses - the flipped group counted from either side."""
    monkeypatch.setenv(IK.FINGER_TOUCH_SHADOW_ENV, "1")
    for rule, want_ok in (("spheres", True), ("geometry", False)):
        kit = _kit()
        with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
            caplog.clear()
            assert _approach(_checker(kit, rule, _FakeHands(DEEP)), kit).ok is want_ok
        st = kit.stats
        assert st.get("finger_touch.decision_changed", 0) >= 1, (rule, st)  # the refusing geometry arm tries the approach twice
        assert st.get("finger_touch.verdict_changed.spheres_passes", 0) >= 1 and not st.get("finger_touch.verdict_changed.geometry_passes")
        assert st.get("finger_touch.shadow.checked", 0) >= st.get("finger_touch.verdict_changed")
        msg = [r.getMessage() for r in caplog.records if "decision changed (shadow)" in r.getMessage()]
        assert len(msg) == st["finger_touch.decision_changed"] and "finger hulls -2.86 mm" in msg[0] and "body free" in msg[0] and "non-finger links clear" in msg[0], msg
        if rule == "spheres":
            # the geometry shadow is quiet: none of its counters or log lines appear under the sphere arm
            assert not any(k.startswith("finger_touch.geometry.") for k in st), st
            assert not any(r.getMessage().startswith("approach finger-support touch refused") for r in caplog.records)


def test_shadow_agreeing_rules_and_off_by_default(monkeypatch):
    kit = _kit()
    monkeypatch.delenv(IK.FINGER_TOUCH_SHADOW_ENV, raising=False)
    assert _approach(_checker(kit, "spheres", _FakeHands(SHALLOW)), kit).ok
    assert not any(k.startswith("finger_touch.shadow") or k.startswith("finger_touch.verdict") for k in kit.stats)
    monkeypatch.setenv(IK.FINGER_TOUCH_SHADOW_ENV, "0")
    assert not IK.finger_touch_shadow_on()
    monkeypatch.setenv(IK.FINGER_TOUCH_SHADOW_ENV, "1")
    kit = _kit()
    assert _approach(_checker(kit, "geometry", _FakeHands(SHALLOW)), kit).ok
    assert kit.stats.get("finger_touch.shadow.checked", 0) >= 1 and "finger_touch.verdict_changed" not in kit.stats


def test_shadow_failure_never_changes_the_verdict(monkeypatch):
    monkeypatch.setenv(IK.FINGER_TOUCH_SHADOW_ENV, "1")
    kit = _kit()
    ikc = _checker(kit, "spheres", _FakeHands(DEEP))

    def boom(*a, **k):
        raise RuntimeError("shadow")

    ikc._finger_touch_geometry_ok = boom
    assert _approach(ikc, kit).ok
    assert kit.stats.get("finger_touch.shadow.error", 0) >= 1
