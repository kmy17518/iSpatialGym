"""``generation.approach_finger_support_touch`` (owner 2026-10-13 ~03:15 item 3, inside_large#03): along a tracked grasp approach the
acting arm's finger links may touch the grasp's support lightly; the palm, wrist and arm, and the fingers against every other object,
keep the approach tracking's margin.

The toy kit reproduces plate/luhkiz grasp_5 on a counter (FK of the gripper spheres from the R1Pro URDF + cuRobo sphere file, 2026-10-13):
the approach comes in 27.8 deg down, the eef 24.8 mm above the counter at the grasp; with the fingers fully open (0.05, the sampler's
model) the lower finger's spheres are 3.5 / 8.2 / 12.8 mm inside the counter 6 / 5 / 4 cm before the grasp - the three configurations
'an intermediate configuration of the approach collides (3/3, [0.04, 0.05, 0.06] m from the start; collides with ... bar_udatjt_0)'
reports - and clear from 7 cm on; at the oracle's pre-shaped opening (pinch 7.8 mm / 2 + 10 mm = 13.9 mm) they clear by 16.7-26 mm.
"""

from __future__ import annotations

import contextlib
import logging
import math
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2.ik import FingerTouch, IKChecker, finger_touch_params

SIN = math.sin(math.radians(27.8))
COS = math.cos(math.radians(27.8))
Z_GRASP = 0.0248  # eef above the counter top at the grasp
PRESHAPE = 0.5 * 0.00783 + 0.010  # grasp_5: pinch width / 2 + the oracle's pre-shape margin


def _finger_bottom(fingers: float) -> float:
    """Lowest point of the lower finger's spheres below the eef (m): -56.4 mm fully open, -26.8 mm at the pre-shape (URDF FK)."""
    return -0.0268 - 0.82 * (float(fingers) - 0.0139)


class _Obstacle:
    def __init__(self, name: str, top: float, x_min: float = -10.0):
        self.name, self.top, self.x_min = name, float(top), float(x_min)


class _TouchKit:
    """eef position = arm joints 0..2 (identity orientation), finger opening = the gripper joints; links' sphere bottoms at fixed offsets
    below the eef (the finger's depends on the opening); obstacles = horizontal half-spaces (their top, from ``x_min`` on)."""

    RIGHT, LEFT, GRIP = th.arange(7, 14), th.arange(14, 21), th.tensor([21, 22])

    def __init__(self, obstacles, palm_bottom: float = -0.0249, arm_bottom: float = 0.09):
        self.robot = SimpleNamespace(arm_control_idx={"right": self.RIGHT, "left": self.LEFT}, gripper_control_idx={"right": self.GRIP},
                                     joint_lower_limits=th.full((28,), -3.0), joint_upper_limits=th.full((28,), 3.0))
        self.obstacles = list(obstacles)
        self.disabled = set()
        self.stats = {}
        self.bottoms = {"right_gripper_finger_link1": None, "right_gripper_finger_link2": 0.03, "right_gripper_link": float(palm_bottom),
                        "right_arm_link6": float(arm_bottom), "torso_link4": 0.4}
        kit = self

        class _Backend:
            def solve_ik_path(self, name, targets, q_start, **kw):
                traj = [q_start.clone()]
                for t in targets:
                    q = traj[-1].clone()
                    q[kit.RIGHT[:3]] = list(t.values())[0][0]
                    traj.append(q)
                return SimpleNamespace(success=True, q_traj=th.stack(traj), status="ok", diagnostics={})

            def kin_entry(self, name):
                return name

            def _collision_costs(self, entry):
                return {"link_of_sphere": [l for l in kit.bottoms for _ in range(2)] + [""]}

        self.backend = _Backend()

    def _ensure_synced(self):
        pass

    @contextlib.contextmanager
    def ignoring(self, objs):
        added = [o for o in objs if o not in self.disabled]
        self.disabled.update(added)
        try:
            yield
        finally:
            self.disabled.difference_update(added)

    def _clearance(self, q, link):
        """Smallest clearance (m, < 0 = penetration) of ``link``'s spheres to the enabled obstacles at ``q``."""
        p = q[self.RIGHT[:3]]
        off = _finger_bottom(float(q[self.GRIP[0]])) if self.bottoms[link] is None else self.bottoms[link]
        out = math.inf
        for o in self.obstacles:
            if o in self.disabled or float(p[0]) < o.x_min:
                continue
            out = min(out, float(p[2]) + off - o.top)
        return out

    def links_clear(self, qs, links, margin):
        qs = qs.unsqueeze(0) if qs.dim() == 1 else qs
        return th.tensor([all(self._clearance(q, l) >= float(margin) - 1e-12 for l in links) for q in qs], dtype=th.bool)

    def collision_free(self, qs, margin=0.0):
        return self.links_clear(qs, list(self.bottoms), margin)

    def fk(self, links, q):
        return {links[0]: (q[self.RIGHT[:3]].clone(), th.tensor([0.0, 0.0, 0.0, 1.0]))}

    def ik(self, pose, q_seed, arm="right"):
        q = q_seed.clone()
        q[self.robot.arm_control_idx[arm][:3]] = pose[0]
        return q


def _pose(s: float):
    """The eef ``s`` (m) back along the approach from the grasp (the approach comes in along +x, 27.8 deg down)."""
    return th.tensor([0.5 - s * COS, 0.0, Z_GRASP + s * SIN]), th.tensor([0.0, 0.0, 0.0, 1.0])


def _q0(fingers: float = 0.05) -> th.Tensor:
    q = th.zeros(28)
    q[_TouchKit.GRIP] = fingers
    return q


def _checker(kit, touch=None, preshape=True, cap=0.005, margin=0.0, supports=None):
    ikc = IKChecker(kit)
    if touch:
        ikc.set_finger_touch(FingerTouch(margin, cap, preshape, supports=lambda: supports if supports is not None else [kit.obstacles[0]],
                                         opening=lambda: PRESHAPE))
    return ikc


def _approach(ikc, kit, chain=None, ignore_last=None):
    poses = chain or [_pose(0.10), _pose(0.0)]
    return ikc.exact(poses, None, "right", ignore_for_last=[kit.obstacles[0]] if ignore_last is None else ignore_last, q_seed=_q0(), cache=False,
                     approached=[])


def test_the_counter_geometry_is_inside03s():
    """The toy kit reproduces the measured clearances: fully open 3.5 / 8.2 / 12.8 mm inside the counter at 6 / 5 / 4 cm, clear from 7 cm;
    pre-shaped >= 16.7 mm clear over the same configurations."""
    kit = _TouchKit([_Obstacle("bar_udatjt_0", 0.0)])
    for s, want in ((0.06, -3.5), (0.05, -8.2), (0.04, -12.8), (0.07, 1.2)):
        q = _q0()
        q[_TouchKit.RIGHT[:3]] = _pose(s)[0]
        assert abs(kit._clearance(q, "right_gripper_finger_link1") * 1000 - want) < 0.3
        q[_TouchKit.GRIP] = PRESHAPE
        assert kit._clearance(q, "right_gripper_finger_link1") > 0.016


def test_default_off_is_todays_approach_check():
    """No key: the configurations 4-6 cm before the grasp collide with the counter, as in the inside#03 logs."""
    kit = _TouchKit([_Obstacle("bar_udatjt_0", 0.0)])
    ikc = _checker(kit)
    assert ikc.finger_touch is None and ikc._touch_objects([kit.obstacles[0]]) == []
    res = _approach(ikc, kit)
    assert not res.ok and res.code == "approach_untrackable"
    assert "an intermediate configuration of the approach collides (3/3, [0.04, 0.05, 0.06] m from the start" in res.reason


def test_preshaped_fingers_let_the_approach_through_with_the_log_line(caplog):
    kit = _TouchKit([_Obstacle("bar_udatjt_0", 0.0)])
    ikc = _checker(kit, touch=True)
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        res = _approach(ikc, kit)
    assert res.ok, res.reason
    assert abs(float(res.q[7]) - 0.5) < 1e-6 and float(res.q[21]) == pytest.approx(0.05), "the witness keeps the configuration's own opening"
    lines = [r.getMessage() for r in caplog.records if "approach finger-support touch allowed" in r.getMessage()]
    assert lines and "max penetration 0 mm" in lines[0] and "0.0139 m (pre-shaped)" in lines[0] and "bar_udatjt_0" in lines[0]
    assert kit.stats.get("finger_touch_allowed") == 1


def test_the_cap_alone_does_not_free_the_fully_open_finger():
    """``preshape: false`` (the sampler's fully open fingers): 12.8 mm at 4 cm is over a 5 mm cap; a 15 mm cap lets it through."""
    kit = _TouchKit([_Obstacle("bar_udatjt_0", 0.0)])
    res = _approach(_checker(kit, touch=True, preshape=False, cap=0.005), kit)
    assert not res.ok and "an intermediate configuration of the approach collides" in res.reason
    res = _approach(_checker(kit, touch=True, preshape=False, cap=0.015), kit)
    assert res.ok, res.reason


def test_palm_wrist_and_arm_keep_the_normal_margin():
    """The palm 2 mm / an arm link 6 mm into the counter at 4 cm: rejected whatever the finger rule."""
    kit = _TouchKit([_Obstacle("bar_udatjt_0", 0.0)], palm_bottom=-0.0249 - 0.0208)
    res = _approach(_checker(kit, touch=True, cap=0.03), kit)
    assert not res.ok and "an intermediate configuration of the approach collides" in res.reason
    kit = _TouchKit([_Obstacle("bar_udatjt_0", 0.0)], arm_bottom=-0.05)
    assert not _approach(_checker(kit, touch=True, cap=0.03), kit).ok


def test_the_fingers_keep_the_normal_margin_against_other_objects():
    """A second object (a dish rack under the approach's middle, x >= 0.44) the open fingers sink into: not the support -> rejected."""
    kit = _TouchKit([_Obstacle("bar_udatjt_0", 0.0), _Obstacle("dish_rack", 0.003, x_min=0.44)])
    res = _approach(_checker(kit, touch=True), kit)
    assert not res.ok and "collides" in res.reason


def test_program_check_chain_and_scope():
    """The program check tracks the grasp approach as pre -> 1/3 -> 2/3 -> grasp with the support in the last pose's ignore set only:
    the rule covers every segment (same verdict as the draw's one segment). A chain whose last ignore set is not the grasp's support
    (a place: the partner) gets no rule."""
    kit = _TouchKit([_Obstacle("bar_udatjt_0", 0.0), _Obstacle("drop_in_sink", -1.0)])
    chain = [_pose(0.10), _pose(0.10 * 2 / 3), _pose(0.10 / 3), _pose(0.0)]
    assert not _approach(_checker(kit), kit, chain=chain).ok
    ikc = _checker(kit, touch=True)
    res = ikc.waypoints_ok(chain, None, "right", ignore_for_last=[kit.obstacles[0]], q_seed=_q0(), cache=False, approached=[])
    assert res.ok, res.reason
    assert ikc._touch_objects([kit.obstacles[1]]) == [] and ikc._touch_objects([kit.obstacles[0], kit.obstacles[1]]) == [kit.obstacles[0]]
    assert not _approach(ikc, kit, ignore_last=[kit.obstacles[1]]).ok


def test_backward_tracking_gets_the_rule():
    """The end's branch is tracked back to the pre pose (``track_approach``'s second try): the support near the pre pose gets the rule."""
    kit = _TouchKit([_Obstacle("bar_udatjt_0", 0.0)])
    ikc = _checker(kit, touch=True)
    q_end = ikc.kit.ik(_pose(0.0), _q0())
    q_pre, why = ikc.track_segment(q_end, _pose(0.0), _pose(0.10), "right", ignore_end=(), ignore_start=[kit.obstacles[0]], touch=[kit.obstacles[0]])
    assert q_pre is not None, why
    q_pre, why = IKChecker(kit).track_segment(q_end, _pose(0.0), _pose(0.10), "right", ignore_end=(), ignore_start=[kit.obstacles[0]])
    assert q_pre is None and "an intermediate configuration of the approach collides" in why


def test_key_parsing():
    assert finger_touch_params(None) is None and finger_touch_params(False) is None
    assert finger_touch_params(True) == {"margin_m": 0.0, "max_penetration_m": 0.005, "preshape": True}
    assert finger_touch_params({"margin_m": 0.0, "max_penetration_m": 0.005}) == {"margin_m": 0.0, "max_penetration_m": 0.005, "preshape": True}
    assert finger_touch_params({"max_penetration_m": 0.01, "preshape": False}) == {"margin_m": 0.0, "max_penetration_m": 0.01, "preshape": False}
    for bad in ({"max_penetration_m": -0.001}, {"max_penetration_m": 0.05}, {"margin_m": -0.01}, {"cap": 0.005}, {"preshape": 1},
                {"margin_m": "0"}, {"max_penetration_m": True}, 0.005, "on", [0.005]):
        with pytest.raises(ValueError):
            finger_touch_params(bad)


def test_cache_scope_and_preshape_margin_match_the_oracle():
    from ispatialgym.oracle.motions import MotionsConfig

    assert IK.ORACLE_PRESHAPE_MARGIN_M == MotionsConfig().gripper_preshape_margin
    ikc = IKChecker(_TouchKit([_Obstacle("bar", 0.0)]))
    ikc.set_finger_touch(FingerTouch(0.0, 0.005, True))
    assert ikc.cache_scope["finger_touch"] == [0.0, 0.005, True]
    ikc.set_finger_touch(None)
    assert "finger_touch" not in ikc.cache_scope and ikc.finger_touch is None


def test_iors_large_setup_and_preshape_opening():
    from ispatialgym.sampling.sim_utils import Reject
    from ispatialgym.sampling.v2.tasks.iors_large import IORSLarge

    def stub(gen, widths):
        s = SimpleNamespace(gp=SimpleNamespace(variant="a", gen=lambda k, d=None: gen.get(k, d)), ikc=IKChecker(_TouchKit([_Obstacle("bar", 0.0)])),
                            grasps=[SimpleNamespace(meta={} if w is None else {"width": w}) for w in widths], surface={"kind": "furniture"})
        s.grasp_ignore_objects = lambda: ["bar"]
        s.preshape_opening = lambda: IORSLarge.preshape_opening(s)
        return s

    s = stub({}, [0.00783])
    assert IORSLarge.setup_finger_touch(s) is None and s.ikc.finger_touch is None
    s = stub({"approach_finger_support_touch": {"margin_m": 0.0, "max_penetration_m": 0.005}}, [0.00783])
    assert IORSLarge.setup_finger_touch(s) is None
    ft = s.ikc.finger_touch
    assert ft is not None and ft.max_penetration_m == 0.005 and ft.preshape and ft.supports() == ["bar"]
    assert ft.opening() == pytest.approx(PRESHAPE)
    assert IORSLarge.preshape_opening(stub({}, [0.00783, 0.04])) == pytest.approx(0.03), "the widest row's pre-shape"
    assert IORSLarge.preshape_opening(stub({}, [0.00783, None])) is None and IORSLarge.preshape_opening(stub({}, [0.2])) == 0.05
    s = stub({"approach_finger_support_touch": {"max_penetration_m": 0.1}}, [0.00783])
    s.finger_touch_error = IORSLarge.setup_finger_touch(s)
    assert "max_penetration_m" in s.finger_touch_error and s.ikc.finger_touch is None
    with pytest.raises(Reject) as e:
        IORSLarge.bind_scene_roles(s)  # the template's setup rejects it (sample_isg builds the sampler outside its try)
    assert e.value.reason == "spec"


def test_backend_shrinks_only_the_asked_links_spheres():
    from ispatialgym.oracle.curobo_planner import shrink_link_spheres

    sp = th.tensor([[[[0.0, 0.0, 0.0, 0.010], [0.0, 0.0, 0.0, 0.004], [0.0, 0.0, 0.0, 0.02], [0.0, 0.0, 0.0, -1.0]]]])
    mask = th.tensor([[True, True, False, True]])
    out, act = shrink_link_spheres(sp, mask, 0.003)
    assert act == 0.003 and out is sp
    out, act = shrink_link_spheres(sp, mask, -0.005)
    assert act == 0.0
    assert out[0, 0, :, 3].tolist() == pytest.approx([0.005, 1e-4, 0.02, -1.0]), "masked enabled spheres shrink (>= 0.1 mm); others and disabled ones stay"
    assert sp[0, 0, 0, 3] == pytest.approx(0.010), "the input is not modified"
