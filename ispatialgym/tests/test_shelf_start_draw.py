"""oracle-fix 2026-11-06 (owner feedback on the touching#03 demo: "touching 3 - it does need more varied start - why don't we try also
initializing in different compartments' walls, and at different lean angles?"): ``generation.shelf_start_draw`` - each attempt of the shelf
layout draws the compartment, the board the target leans against (the arm on that side works) and both books' leans. The draws (layouts,
boards, leans within bounds, deterministic per seed), the records, the spec checks, the eval / no-start-per-attempt restrictions, the goal
drafted parallel to the partner, default off = today, and the measured train-repeat rule against an eval-like instance. ``IORSSmall``
methods run on a bare instance with a ray-cast stand-in for a three-cube bookcase row; no simulator is launched."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling.sim_utils import Reject, SupportTop
from ispatialgym.sampling.v2.stances import EdgeFrame

#: hardback/acbrnv (target) and hardback/agbqsy (partner): height x width x thickness
ACBRNV = th.tensor([0.22274, 0.17153, 0.03120]) / 2
AGBQSY = th.tensor([0.21317, 0.16850, 0.03298]) / 2
#: the kmuacd_0 row at 0.84 m along its edge tangent (x here): solid pieces (outer sides and the two dividers); the inner faces of the centre
#: cube at +0.166 / -0.176 (template_meta: the rays' hits), the right cube's at -0.20 / -0.54
PIECES = [(-0.60, -0.54), (-0.20, -0.176), (0.166, 0.19), (0.54, 0.60)]
BOARD = "/World/bookcase_x_0/base_link/collisions/x__base_link__collision__3"
LAYOUTS = [[0.0, "right"], [-0.37, "left"], [-0.37, "right"]]
#: the eval instance's layout (the template's): the centre cube, the target on the left board, both leaning the key's 15 deg
EVAL_LAYOUT = (0.0, 1.0)


def _iors():
    from ispatialgym.sampling.v2.tasks import iors_small as IS

    return IS


def _box(h) -> th.Tensor:
    return th.tensor([[sx * h[0], sy * h[1], sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=th.float32)


def _hull_at(hull, pose):
    return C.transform_points((th.as_tensor(pose[0]).float(), th.as_tensor(pose[1]).float()), hull)


def _raycast(starts, ends):
    """The first solid piece along the ray (x = the edge tangent); distance 0 when the ray starts inside one."""
    x0, x1 = float(starts[0, 0]), float(ends[0, 0])
    sg = 1.0 if x1 > x0 else -1.0
    for lo, hi in PIECES:
        if lo <= x0 <= hi:
            return [{"hit": True, "rigidBody": BOARD, "distance": 0.0}]
    faces = [lo - x0 for lo, _ in PIECES if lo > x0] if sg > 0 else [x0 - hi for _, hi in PIECES if hi < x0]
    if not faces or min(faces) > abs(x1 - x0):
        return [{"hit": False}]
    return [{"hit": True, "rigidBody": BOARD, "distance": min(faces)}]


def _sampler(monkeypatch, gen=None, spa=True, layouts=LAYOUTS, lean=(11.0, 20.0)):
    """A bare ``IORSSmall`` after ``sample_template`` of the shelf layout: the key leaning 15 deg left, the centre cube, the two books, the
    board raster of the row (x in [-0.53, 0.53], y in [-0.29, -0.01]: the robot faces -y, its left is +x), ``shelf_start_draw`` on."""
    from ispatialgym.sampling.v2 import tabletop as TT
    from ispatialgym.tests.test_v2_tabletop import BU_LEAN_DEG, BU_REST_XYZW

    IS = _iors()
    monkeypatch.setenv("ISPATIALGYM_START_PER_ATTEMPT", "1" if spa else "0")
    monkeypatch.setattr(TT, "arm_bands_for", lambda s, arms, **kw: {a: f"band-{a}" for a in arms})
    generation = {"shelf_spine_behind_edge_m": 0.06, "shelf_free_span_min_m": 0.12, "shelf_working_arm_default": True}
    if layouts is not None:
        generation["shelf_start_draw"] = {"layouts": {"bookcase_x_0": layouts}, "lean_deg": None if lean is None else list(lean)}
    generation.update(gen or {})
    s = object.__new__(IS.IORSSmall)
    support = SimpleNamespace(name="bookcase_x_0", links={"base_link": SimpleNamespace(prim_path="/World/bookcase_x_0/base_link")})
    xs, ys = th.arange(-0.53, 0.54, 0.02), th.arange(-0.29, 0.0, 0.02)
    s.__dict__.update(
        gen=lambda key, default=None: generation.get(key, default), gp=SimpleNamespace(variant="touching_shelf_stub", generation=generation),
        obj=lambda role: support, relation="touching", hull={"target": _box(ACBRNV), "partner": _box(AGBQSY)},
        rest={"target": th.tensor(BU_REST_XYZW)}, segment_arms=["left"], bands=None, ikc=SimpleNamespace(default_working_arms=["right"]),
        top=SupportTop("bookcase_x_0", xs, ys, th.ones(len(xs), len(ys), dtype=th.bool), th.zeros(len(xs), len(ys), dtype=th.bool), 0.8354, 0.0),
        edge=EdgeFrame("+u", th.tensor([0.0, -0.01]), th.tensor([0.0, -1.0]), 1.08), shelf_pose=None, shelf_partner_pose=None,
        world=SimpleNamespace(raycast=_raycast, hull_at=_hull_at, bottom_offset=lambda h, q: -float(C.transform_points((th.zeros(3), q), h)[:, 2].min())),
        template_facts={})
    s.shelf_start = {"key": "hardback__acbrnv__bu", "lean_side": 1.0, "lean_deg": BU_LEAN_DEG, "context": "bookcase_floor_wall"}
    s.rest["partner"] = TT.upright_lean_quat(s.hull["partner"], -1.0, BU_LEAN_DEG)
    if s.shelf_start_draw() is not None:  # sample_template's capture
        s._shelf_template = {k: s.shelf_start.get(k) for k in ("key", "lean_side", "lean_deg", "context")}
        s._shelf_rest0 = {"target": s.rest["target"].clone(), "partner": s.rest["partner"].clone()}
    return s


def _draw(s, seed):
    rng = np.random.default_rng(seed)
    s.draw_shelf_side(rng)
    s.draw_shelf_layout(rng)
    return dict(s._shelf_draw)


def _lean(quat) -> float:
    """Signed lean (deg) of a book's height axis toward world +x (the edge tangent: the robot's left) at ``quat``."""
    R = C.transform_points((th.zeros(3), th.as_tensor(quat).float()), th.eye(3))
    up = R[int(th.argmax(R[:, 2].abs()))]
    up = up * (1.0 if float(up[2]) > 0 else -1.0)
    return math.degrees(math.atan2(float(up[0]), float(up[2])))


# ----------------------------------------------------------------------------------------------------------------
# the rest orientation turned to another board / lean
# ----------------------------------------------------------------------------------------------------------------


def test_releaned_quat_turns_the_key_rest_to_the_drawn_board_and_lean_and_keeps_the_spine():
    from ispatialgym.sampling.v2 import tabletop as TT
    from ispatialgym.tests.test_v2_tabletop import BU_LEAN_DEG, BU_REST_XYZW

    rest = th.tensor(BU_REST_XYZW)
    assert th.equal(TT.releaned_quat(rest, 1.0, BU_LEAN_DEG, 1.0, BU_LEAN_DEG), rest.float()), "the key's own lean: the key's rest, bit for bit"
    hull = _box(ACBRNV)
    for side, lean in ((-1.0, 15.0), (1.0, 11.0), (-1.0, 20.0), (1.0, 20.0)):
        q = TT.releaned_quat(rest, 1.0, BU_LEAN_DEG, side, lean)
        R0, R1 = (C.transform_points((th.zeros(3), qq), th.eye(3)) for qq in (rest, q))
        up = R1[int(th.argmax(R1[:, 2].abs()))]
        up = up * (1.0 if float(up[2]) > 0 else -1.0)
        assert math.copysign(1.0, float(up[1])) == side and math.degrees(math.asin(float(th.linalg.norm(up[:2])))) == pytest.approx(lean, abs=0.05)
        # every local axis keeps its rest-frame x component: the turn is about +x (spine -> into the shelf)
        assert th.allclose(R0[:, 0], R1[:, 0], atol=1e-5)
        # a mirrored book's partner rest is the same shape (extents equal): the template's Phase-2 inputs do not change with the board
        assert th.allclose(TT.hull_extents(hull, q), TT.hull_extents(hull, TT.releaned_quat(rest, 1.0, BU_LEAN_DEG, -side, lean)), atol=1e-5)


# ----------------------------------------------------------------------------------------------------------------
# the spec
# ----------------------------------------------------------------------------------------------------------------


def test_the_spec_is_off_by_default_normalised_when_on_and_checked(monkeypatch):
    off = _sampler(monkeypatch, layouts=None)
    assert off.shelf_start_draw() is None and not _iors().shelf_draw_on(off)
    tabletop = _sampler(monkeypatch)
    tabletop.shelf_start = None
    assert tabletop.shelf_start_draw() is None, "outside the shelf layout the knob does nothing"
    s = _sampler(monkeypatch)
    spec = s.shelf_start_draw()
    assert spec["layouts"] == [(0.0, -1.0), (-0.37, 1.0), (-0.37, -1.0)] and spec["lean_deg"] == (11.0, 20.0)
    assert spec["spine_behind_edge_m"] is None and spec["tries"] == 8
    assert s.shelf_draw_facts() == {"layouts": [[0.0, "right"], [-0.37, "left"], [-0.37, "right"]], "lean_deg": (11.0, 20.0), "spine_behind_edge_m": None,
                                    "tries": 8, "template_layout": [0.0, "left"], "mirror": True}
    # a support without a list keeps the template's layout (its bay, the key's board)
    other = _sampler(monkeypatch, gen={"shelf_bay_lateral_m": {"bookcase_x_0": 0.19}})
    other.gp.generation["shelf_start_draw"] = {"layouts": {"bookcase_y_1": LAYOUTS}, "lean_deg": [11.0, 20.0]}
    other._shelf_draw_spec = None
    assert other.shelf_start_draw()["layouts"] == [(0.19, 1.0)]
    for bad in ({"layouts": {"bookcase_x_0": [[0.0, "up"]]}}, {"layouts": {"bookcase_x_0": []}}, {"layouts": [[0.0, "left"]]},
                {"lean_deg": [20.0, 11.0]}, {"lean_deg": [11.0]}, {"tries": 0}, {"spine_behind_edge_m": [0.1, 0.05]}, [1, 2]):
        s = _sampler(monkeypatch)
        s.gp.generation["shelf_start_draw"] = bad
        s._shelf_draw_spec = None
        with pytest.raises(Reject) as e:
            s.shelf_start_draw()
        assert e.value.reason == "spec", bad


# ----------------------------------------------------------------------------------------------------------------
# the draws
# ----------------------------------------------------------------------------------------------------------------


def test_each_attempt_draws_a_listed_layout_and_leans_in_range_deterministically(monkeypatch):
    s = _sampler(monkeypatch)
    seen, leans = set(), []
    for seed in range(60):
        f = _draw(s, seed)
        layout = (f["bay_m"], f["target_board"])
        assert [layout[0], layout[1]] in LAYOUTS and f["layout"] == [layout[0], layout[1]]
        seen.add(layout)
        assert 11.0 <= f["target_lean_deg"] <= 20.0 and 11.0 <= f["partner_lean_deg"] <= 20.0
        leans += [f["target_lean_deg"], f["partner_lean_deg"]]
        # the arm on the target's side works; the start's record follows the draw
        arm = "left" if f["target_board"] == "left" else "right"
        assert s.segment_arms == [arm] and s.ikc.default_working_arms == [arm] and arm in s.arm_offer_why and s.arm_bands == {arm: f"band-{arm}"}
        assert s.shelf_start["lean_side"] == (1.0 if arm == "left" else -1.0) and s.shelf_start["bay_m"] == f["bay_m"]
        assert s.shelf_start["lean_deg"] == f["target_lean_deg"] and s.shelf_start["partner_lean_deg"] == f["partner_lean_deg"]
        assert f["tries"] == 1 + len(f["rejected"]) and f["free_span_m"] >= 0.12
        # the same seed draws the same start
        g = _draw(s, seed)
        assert g == f and th.equal(s.shelf_pose[0], s.shelf_pose[0])
    assert seen == {(0.0, "right"), (-0.37, "left"), (-0.37, "right")}
    assert min(leans) < 12.5 and max(leans) > 18.5, "the leans cover the range"


def test_the_books_lean_against_the_drawn_compartments_boards(monkeypatch):
    from ispatialgym.sampling.v2 import tabletop as TT

    s = _sampler(monkeypatch)
    # the inner board faces of each compartment: (left board, right board) along +x
    faces = {0.0: (0.166, -0.176), -0.37: (-0.20, -0.54)}
    for seed in range(40):
        f = _draw(s, seed)
        side = 1.0 if f["target_board"] == "left" else -1.0
        left, right = faces[f["bay_m"]]
        t_w, p_w = _hull_at(s.hull["target"], s.shelf_pose), _hull_at(s.hull["partner"], s.shelf_partner_pose)
        t_wall, p_wall = (left, right) if side > 0 else (right, left)
        # each book's leaning top edge 2 mm short of its board, the target on the drawn board, the partner on the other
        assert (float(t_w[:, 0].max()) if side > 0 else float(t_w[:, 0].min())) == pytest.approx(t_wall - side * TT.SHELF_WALL_GAP_M, abs=1e-4)
        assert (float(p_w[:, 0].min()) if side > 0 else float(p_w[:, 0].max())) == pytest.approx(p_wall + side * TT.SHELF_WALL_GAP_M, abs=1e-4)
        assert f["wall_lateral_m"] == pytest.approx([t_wall, p_wall], abs=1e-4)
        # the drawn leans, toward each book's own board; both stand on the board, spine 6 cm behind the front edge
        assert _lean(s.shelf_pose[1]) == pytest.approx(side * f["target_lean_deg"], abs=0.1)
        assert _lean(s.shelf_partner_pose[1]) == pytest.approx(-side * f["partner_lean_deg"], abs=0.1)
        for w in (t_w, p_w):
            assert float(w[:, 2].min()) == pytest.approx(0.8354 + TT.Z_OFFSET, abs=1e-4) and float(w[:, 1].max()) == pytest.approx(-0.06, abs=1e-4)
        # the free span between the two books is what the record says
        assert f["free_span_m"] == pytest.approx(_iors().shelf_free_span(t_w, p_w, s.edge, side), abs=1e-4)


def test_a_failing_layout_is_drawn_again_on_the_same_board_then_rejected(monkeypatch):
    # a bay inside a divider (the ray starts in it) and the wide cube: the divider bay is redrawn among the same board's layouts
    s = _sampler(monkeypatch, layouts=[[-0.19, "left"], [-0.37, "left"], [0.0, "right"]])
    hits = 0
    for seed in range(40):
        f = _draw(s, seed)
        if f["layout"] == [-0.19, "left"]:
            hits += 1
            assert f["bay_m"] == -0.37 and f["target_board"] == "left" and f["rejected"] and "[shelf_wall]" in f["rejected"][0]
    assert hits > 0
    # only layouts that fail (the divider bay; a span the books cannot leave at 20 deg): the attempt is rejected with that reason
    for layouts, lean, reason in (([[-0.19, "left"]], (11.0, 20.0), "shelf_wall"), ([[0.0, "left"]], (30.0, 30.0), "shelf_width")):
        s = _sampler(monkeypatch, layouts=layouts, lean=lean)
        with pytest.raises(Reject) as e:
            _draw(s, 0)
        assert e.value.reason == reason and "in 8 tries" in str(e.value)


def test_eval_and_no_start_per_attempt_keep_the_keys_board(monkeypatch):
    # eval (generation.partner_side right: the target on the left board) draws among the left-board layouts only
    s = _sampler(monkeypatch, gen={"partner_side": "right"})
    for seed in range(20):
        f = _draw(s, seed)
        assert f["target_board"] == "left" and f["bay_m"] == -0.37 and "partner_side right" in f["restricted"] and s.segment_arms == ["left"]
    # without start_per_attempt the template's Phase-2 candidates are its arm's: no mirror; a list without the key's board -> the template's bay
    s = _sampler(monkeypatch, spa=False, layouts=[[0.0, "right"]])
    f = _draw(s, 3)
    assert f["layout"] == [0.0, "left"] and f["target_board"] == "left" and "no start_per_attempt" in f["restricted"] and s.segment_arms == ["left"]
    assert s.shelf_draw_facts()["mirror"] is False


def test_without_a_lean_range_both_books_keep_the_keys_lean(monkeypatch):
    from ispatialgym.tests.test_v2_tabletop import BU_LEAN_DEG

    s = _sampler(monkeypatch, layouts=[[-0.37, "right"]], lean=None)
    f = _draw(s, 5)
    assert f["target_lean_deg"] == BU_LEAN_DEG and f["partner_lean_deg"] == BU_LEAN_DEG
    assert _lean(s.shelf_pose[1]) == pytest.approx(-BU_LEAN_DEG, abs=0.1) and _lean(s.shelf_partner_pose[1]) == pytest.approx(BU_LEAN_DEG, abs=0.1)


def test_the_side_draw_resets_the_nominal_rests_before_the_start_draw(monkeypatch):
    """The start's Phase 2 and its cache key read the rests and ``shelf_start``: before every start draw they are the key's lean mirrored to
    the drawn board (2 cache variants), never the previous attempt's drawn leans."""
    from ispatialgym.sampling.v2 import tabletop as TT
    from ispatialgym.tests.test_v2_tabletop import BU_LEAN_DEG, BU_REST_XYZW

    s = _sampler(monkeypatch)
    _draw(s, 1)
    for seed in range(10):
        s.draw_shelf_side(np.random.default_rng(seed))
        side = s.shelf_start["lean_side"]
        assert s.shelf_start["lean_deg"] == BU_LEAN_DEG and "partner_lean_deg" not in s.shelf_start and s.shelf_pose is None
        assert th.equal(s.rest["target"], TT.releaned_quat(th.tensor(BU_REST_XYZW), 1.0, BU_LEAN_DEG, side, BU_LEAN_DEG))
        assert th.equal(s.rest["partner"], TT.upright_lean_quat(s.hull["partner"], -side, BU_LEAN_DEG))
        s.draw_shelf_layout(np.random.default_rng(seed))


# ----------------------------------------------------------------------------------------------------------------
# the goal, the order in ``sample`` and default off
# ----------------------------------------------------------------------------------------------------------------


def test_the_goal_is_drafted_parallel_to_the_partners_drawn_lean_and_keeps_the_bottom_height(monkeypatch):
    from ispatialgym.sampling.v2 import tabletop as TT

    IS = _iors()
    s = _sampler(monkeypatch, layouts=[[0.0, "left"]])
    manip = SimpleNamespace()
    for seed in range(10):
        f = _draw(s, seed)
        q = s.target_goal_quat(s.shelf_pose, manip)
        assert _lean(q) == pytest.approx(-f["partner_lean_deg"], abs=0.1), "the target ends leaning the partner's way at its lean"
        dz = IS.shelf_goal_dz(s.shelf_start, s.hull["target"], s.shelf_pose, q)
        z0 = float(_hull_at(s.hull["target"], s.shelf_pose)[:, 2].min())
        z1 = float(_hull_at(s.hull["target"], (s.shelf_pose[0] + th.tensor([0.0, 0.0, dz]), q))[:, 2].min())
        assert z1 == pytest.approx(z0, abs=1e-5)
    # without a drawn partner lean (the knob off): the mirrored start's own lean and no height change, as before
    s.shelf_start = {"key": "k", "lean_side": 1.0, "lean_deg": 15.0}
    pose = s.shelf_pose
    assert th.equal(s.target_goal_quat(pose, manip), TT.mirrored_lean_quat(pose[1], s.edge, 1.0, 15.0))
    assert IS.shelf_goal_dz(s.shelf_start, s.hull["target"], pose, th.tensor([0.0, 0.0, 0.0, 1.0])) == 0.0
    assert IS.shelf_goal_dz(None, s.hull["target"], pose, th.tensor([0.0, 0.0, 0.0, 1.0])) == 0.0


def test_sample_draws_the_board_before_the_start_and_the_compartment_after_it(monkeypatch):
    from ispatialgym.sampling.v2 import core as CORE

    calls = []
    monkeypatch.setattr(CORE.SamplerV2, "sample", lambda self, rng: calls.append("core") or "draft")
    for on in (True, False):
        calls.clear()
        s = _sampler(monkeypatch, layouts=LAYOUTS if on else None)
        s.draw_shelf_side = lambda rng: calls.append("side")
        s.draw_shelf_layout = lambda rng: calls.append("layout")
        s.spa_redraw_start = lambda rng: calls.append("start")
        assert s.sample(np.random.default_rng(0)) == "draft" and s.region_side is None
        assert calls == (["side", "start", "layout", "core"] if on else ["start", "core"])


def test_default_off_keeps_the_cache_and_memo_keys(monkeypatch):
    s = _sampler(monkeypatch, layouts=None)
    c = SimpleNamespace(base_pose2d=(1.0, 2.0, 0.5), translate=0.25, tilt=0.0, focus=th.tensor([1.0, 2.0, 0.9]))
    assert s._spa_drop_key(c) == s._spa_cand_key(c)
    on = _sampler(monkeypatch)
    _draw(on, 0)
    assert on._spa_drop_key(c) == on._spa_cand_key(c) + (tuple(on.segment_arms),)
    for x in (s, on):
        x.ikc.q_locked = lambda st: th.zeros(4)
        x.shelf_backoff_m = lambda: 0.0
    st = SimpleNamespace(base_pose2d=(1.0, 2.0, 0.5), translate=0.25, tilt=0.0)
    s.shelf_pose = on.shelf_pose
    base = s._fixed_start_memo_key(st, "left", None, False)
    assert len(base) == 7, "off: the key as before"
    assert on._fixed_start_memo_key(st, "left", None, False)[:7] == base and on._fixed_start_memo_key(st, "left", None, False)[7] == "start"


# ----------------------------------------------------------------------------------------------------------------
# the measured train-repeat rule (instance_separation.find_repeat, REPEAT_MEASURED)
# ----------------------------------------------------------------------------------------------------------------


def _record(s, base=(-0.065, 0.55, -math.pi / 2)):
    """An instance record of the drawn start as the instance filter builds it: the start base, both books' start (TRO root_link) and goal
    targets (the partner stays; the target's goal 6 cm from the partner toward its start, drafted parallel to it) at the task tolerances."""
    from ispatialgym.sampling import instance_separation as SEP

    tro = {"ispatialgym": {"activity": "inter_object_relation_satisfaction", "task_variant": "touching_shelf_stub", "scene": {"model": "house"},
                           "pose_targets": {}}, "robot_poses": {"robot": [{"position": [base[0], base[1], 0.0],
                                                                         "orientation": [0.0, 0.0, math.sin(base[2] / 2), math.cos(base[2] / 2)]}]}}
    side = s.shelf_start["lean_side"]
    goal_t = (s.shelf_partner_pose[0] + th.tensor([side * 0.06, 0.0, 0.0]), s.target_goal_quat(s.shelf_pose, SimpleNamespace()))
    for name, start, goal in (("hardback.n.01_1", s.shelf_pose, goal_t), ("hardback.n.01_2", s.shelf_partner_pose, s.shelf_partner_pose)):
        tro[name] = {"root_link": {"pos": [float(v) for v in start[0]], "ori": [float(v) for v in start[1]]}}
        tro["ispatialgym"]["pose_targets"][name] = {"position": [float(v) for v in goal[0]], "orientation": [float(v) for v in goal[1]], "frame": "world",
                                                    "position_metric": "full", "orientation_metric": "full", "position_tolerance": 0.03,
                                                    "orientation_tolerance": 0.2}
    return SEP.record_from_dicts(tro)


def test_a_drawn_layout_no_longer_repeats_the_eval_instance_while_its_own_layout_still_would(monkeypatch):
    from ispatialgym.sampling import instance_separation as SEP

    monkeypatch.delenv("ISPATIALGYM_FAST", raising=False)
    monkeypatch.delenv("ISPATIALGYM_FAST_REPEAT_MEASURED", raising=False)
    ev = _sampler(monkeypatch, layouts=[[EVAL_LAYOUT[0], "left"]], lean=(15.0, 15.0))
    _draw(ev, 0)
    eval_rec = _record(ev, base=(-0.19, 0.55, -math.pi / 2))  # the eval instance's base: 0.125 m from the train cell's
    same = _record(ev)
    assert SEP.find_repeat(same, [eval_rec]) is not None, "today's start: a repeat (fix1: 265 / 265 completed attempts)"
    # the eval's own layout at any lean in range still repeats it (<= 0.9 cm, <= 5 deg): why the customs leave it out
    own = _sampler(monkeypatch, layouts=[[EVAL_LAYOUT[0], "left"]])
    for seed in range(20):
        _draw(own, seed)
        assert SEP.find_repeat(_record(own), [eval_rec]) is not None, seed
    # every layout of the customs moves both books' starts by >= 0.2 m: never a repeat, at any lean
    s = _sampler(monkeypatch)
    for seed in range(30):
        f = _draw(s, seed)
        rec = _record(s)
        assert SEP.find_repeat(rec, [eval_rec]) is None, f
        gap = math.dist(rec.measured_start[0].p, eval_rec.measured_start[0].p)
        assert gap > 0.2, (f, gap)


def test_the_start_cells_are_memoised_per_working_arm_when_the_draw_is_on(monkeypatch):
    """A start cell holds the working arm's Phase-2 candidates (its band, the other arm idling): with the draw on the in-process memo keys it by
    the arm as the disk key does (``segment_arms`` in the payload); off, per (edge, cell) as before."""
    IS = _iors()
    monkeypatch.setattr(IS, "cache_get", lambda key: None)
    monkeypatch.setattr(IS, "cache_put", lambda key, value, round_floats=False: None)
    monkeypatch.setattr(IS, "cache_key", lambda kind, payload: "k")
    for on in (True, False):
        s = _sampler(monkeypatch, layouts=LAYOUTS if on else None)
        computed = []
        s._spa_cell_payload = lambda *a: {}
        s._spa_cell_compute = lambda edge, trunk, lateral, *a: computed.append(tuple(s.segment_arms)) or {"ok": False, "reject": ["stance_collision", "x"]}
        for arms in (["left"], ["right"], ["left"]):
            s.segment_arms = arms
            s._spa_cell(s.edge, None, 3, 22, -0.54, 0.54, 0.05, {}, 0.22, None)
        assert computed == ([("left",), ("right",)] if on else [("left",)])
